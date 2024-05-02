# https://github.com/jaxony/unet-pytorch/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import os
from models.pytorch_losses import Nadam
import torch.nn.utils.weight_norm as weight_norm

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return weight_norm(nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups))

def upconv2x2(in_channels, out_channels, mode='upsample'):
    if mode == 'transpose':
        return weight_norm(nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2))
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return weight_norm(nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1))


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, 
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=4, 
                 start_filts=16, up_mode='transpose', 
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
        encoder_outs = []
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x


def unet(args):
    return UNet(args.num_classes)


def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class PyTorchModelKerasStyle:
    def __init__(self, args):
        self.model_path = args.load_model_path
        width = height = args.crop_size
        self.model_type = model_type = args.model_type
        num_classes = args.num_classes
        self.loss_fn = torch.nn.MSELoss(size_average=False)
        self.use_cuda = args.use_cuda

        if os.path.exists(args.load_model_path):
            # TODO Load pytorch model.
            if args.resume:
                if os.path.isfile(args.resume):
                    print("=> loading checkpoint '{}'".format(args.resume))
                    checkpoint = torch.load(args.resume)
                    args.start_epoch = checkpoint['epoch']
                    best_prec1 = checkpoint['best_prec1']
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                          .format(args.resume, checkpoint['epoch']))
                else:
                    print("=> no checkpoint found at '{}'".format(args.resume))
            print("Loading existing model...", args.load_model_path, "With inferred number of classes", args.num_classes)
            # TODO: Get model loading without matching code working!
            try:
                model = keras.models.load_model(args.load_model_path)
                args.num_classes = model.layers[-1].output_shape[-1]
            except Exception as ex:
                print(ex)
                print("BUILDING model from args...")

                import sys
                current_module = sys.modules[__name__]
                model = getattr(current_module, model_type)
                print("Importing model type named", model_type)
                model = model(args)
                model.load_weights(args.load_model_path, by_name=True)
        else:
            import sys
            current_module = sys.modules[__name__]
            model = getattr(current_module, model_type)
            print("Importing model type named", model_type)
            model = model(args)
            print("No model by this name exists; creating new model instead...", args.load_model_path)
        #model.save('model_checkpoint.h5')
        self.model = model
        self.optimizer = Nadam(self.model.parameters(), lr=args.lr)
        if args.use_cuda:
            self.model.cuda()

    def get_weights(self):
        return self.model.weights

    def predict(self, batch, batch_size=1):
        print("TRYING TO DO PREDICTION WITH IMAGE OF SIZE", batch.shape)
        # TODO: TF is awful! I can't just make things the shape I want!
        #return np.zeros((batch.shape[0], batch.shape[1], batch.shape[2], self.num_classes))

        #batch = batch[:, :self.height, :self.width, :]
        #if batch.shape[-1] == 3:
        #    print("Predict Batch shape", batch.shape)
        #    batch = np.reshape(np.mean(batch, axis=-1), (batch_size, batch.shape[1], batch.shape[2], 1))
        #print("Batch stats:", np.mean(batch), np.std(batch), np.min(batch), np.max(batch))
        #logits = self.sess.run(self.y_conv, feed_dict={self.x:batch, self.keep_prob: 1.0})
        #print(logits.shape)
        #print("Pred stats:", np.mean(logits), np.std(logits), np.min(logits), np.max(logits))
        #return logits
        batch = np.transpose(batch, (0, 3, 1, 2))
        batch = torch.autograd.Variable(torch.Tensor(batch).cuda())
        if self.use_cuda:
            batch = batch.cuda()
        return np.transpose(self.model(batch).data.cpu().numpy(), (0, 2, 3, 1))

    def train_step(self):
        batch, targets = next(self.generator)
        batch = np.transpose(batch, (0, 3, 1, 2))
        batch = torch.autograd.Variable(torch.Tensor(batch).cuda())
        targets = torch.autograd.Variable(torch.Tensor(targets).cuda())
        if self.use_cuda:
            batch.cuda()
            targets.cuda()
        y_pred = self.model(batch)
        targets = targets.permute(0,3,1,2)
        #print("y_pred, targets.shape", y_pred.shape, targets.shape)
        loss = self.loss_fn(y_pred, targets)
        #model.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        
        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its data and gradients like we did before.
        '''with torch.no_grad():
            for param in model.parameters():
                param.data -= learning_rate * param.grad
        '''
        return np.transpose(y_pred.data.cpu().numpy(), (0, 2, 3, 1)), loss.data.cpu().numpy()

    def fit_generator(self, generator=None, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0):
        self.generator = generator
        if steps_per_epoch is None:
            steps_per_epoch = 5000
        self.epoch = initial_epoch
        self.iteration = initial_epoch * steps_per_epoch

        while self.epoch < epochs:
            print("Epoch:", self.epoch)
            for step in range(steps_per_epoch):
                if verbose > 1:
                    print("Training iteration", self.iteration)
                y_pred, loss = self.train_step()
                if self.iteration % 20 == 0:
                  print("Average prediction:", np.mean(y_pred), "Loss:", loss)
                # Call all of the callbacks!
                logs = {"loss":loss}
                for callback in callbacks:
                    callback.on_batch_end(batch=self.iteration, logs=logs)
                self.iteration += 1
            # Call all of the callbacks!
            for callback in callbacks:
                callback.on_epoch_end(self.epoch, logs=logs)
            self.epoch += 1
        pass

    def count_params(self):
        return 0

    def summary(self):
        return "Model Summary: Not Implemented"

    def save(self, filepath):
        #save_model(self.sess, "./" + filepath + str(self.iteration))
        save_checkpoint({
            'epoch': self.epoch + 1,
            'arch': self.model_type,
            'state_dict': self.model.state_dict(),
            #'best_prec1': best_prec1,
            'optimizer' : self.optimizer.state_dict()})

def build_model(args):
    print("")
    print("")
    print("")
    print("Building PyTorch model of type", args.model_type)

    return PyTorchModelKerasStyle(args)
