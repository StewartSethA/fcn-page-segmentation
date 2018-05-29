from __future__ import print_function
import numpy as np

def mask_by_channel(image, mask):
    gt_mask = np.clip(mask*255,0,1)
    gt_mask = np.expand_dims(gt_mask, -1)
    print(gt_mask.shape)
    gt_mask = np.tile(gt_mask, (1, 1, image.shape[-1]))
    image = np.multiply(image, gt_mask)
    return image

# https://gist.github.com/frnsys/91a69f9f552cbeee7b565b3149f29e3e
# Expands an indexed enumeration of class labels to a one-hot representation.
def enum_to_onehot(array2d, num_labels=None, ignore_first=False):
    a = array2d
    
    # the 3d array that will be the one-hot representation
    # a.max() + 1 is the number of labels we have
    if num_labels is None:
        num_labels = a.max() + 1
    if ignore_first: # Trim the first "zeroeth" channel; it's background.
        num_labels = num_labels - 1
        a = np.clip(a - 1, 0, num_labels) 
        # WARNING: This will conflate 'empty' with class 0 since there is not enough representational capacity for it.
        # It is suggested to use the "bitmask to multihot" encoding method instead since it allows blank labels on both inputs and outputs.
    
    b = np.zeros((a.shape[0], a.shape[1], num_labels), dtype=a.dtype)
    # if you visualize this as a stack of layers,
    # where each layer is a sample,
    # this first index selects each layer separately
    layer_idx = np.arange(a.shape[0]).reshape(a.shape[0], 1)
    # this index selects each component separately
    component_idx = np.tile(np.arange(a.shape[1]), (a.shape[0], 1))
    # then we use `a` to select indices according to category label
    b[layer_idx, component_idx, a] = 1
    return b

# Convert one-hot to enum. [[1,0,0,...],[0,1,0,...],...] -> [0,1,2,3,4,...,N]
def onehot_to_enum(onehot_array, one_indexed=False):
    num_classes = onehot_array.shape[-1]
    if one_indexed:
        onehot_array = np.concatenate([np.ones((onehot_array.shape[0], onehot_array.shape[1], 1), dtype=onehot_array.dtype), onehot_array], axis=-1)
        onehot_array[:,:,0] -= np.sum(onehot_array[:,:,1:], axis=-1)
        onehot_array = np.clip(onehot_array, 0, 1)
    flattened = np.argmax(onehot_array, axis=-1)
    return flattened
    
# https://gist.github.com/frnsys/91a69f9f552cbeee7b565b3149f29e3e
def enum_to_multihot(array2d, num_labels=None, ignore_first=False, one_indexed=False):
    # Classes are one-indexed. Zero is the absence of any class.
    a = array2d
    
    # the 3d array that will be the one-hot representation
    # a.max() + 1 is the number of labels we have
    if num_labels is None:
        num_labels = a.max() + 1
    if ignore_first: # Ignore the first channel; it's background.
        num_labels = num_labels - 1
        a = a-1 #np.clip(a - 1, 0, num_labels)
    if one_indexed == True:
        num_labels += 1
    
    b = np.zeros((a.shape[0], a.shape[1], num_labels), dtype=a.dtype)
    for l in range(num_labels):
        b[:,:,l] += np.equal(l, a).astype(a.dtype)
    return b

# Convert multihot to bitmask. [1,2,4,8,16,...,2^N], where multiple bits can be on in the same spatial location (e.g. 255)
# Not working????
def multihot_to_bitmask(multihot_array):
    num_classes = multihot_array.shape[-1]
    flattened = np.zeros(multihot_array.shape[:-1], multihot_array.dtype)
    bitmask = 1
    for l in range(multihot_array.shape[-1]):
        flattened += bitmask * multihot_array[:,:,l]
        bitmask = bitmask << 1
    return flattened

# Convert bitmask to multihot. [1|2|4|8|...] -> [[1,0,0,...],[0,1,0,...],...]
def bitmask_to_multihot(bitmask, num_classes=None):
    if num_classes is None:
        #num_classes = int(round(math.log(np.max(bitmask), 2)))
        num_classes = np.max(bitmask).bit_length()
    a = bitmask
    b = np.zeros((a.shape[0], a.shape[1], num_classes), dtype=a.dtype)
    bit = 1
    for l in range(num_classes):
        #b[:,:,num_classes-l-1] = np.greater(a & bit, 0)
        b[:,:,l] = np.greater(a & bit, 0).astype(a.dtype)
        bit = bit << 1
    return b.astype(a.dtype)

# Transform an image with values stored as sequential ints [0,1,2,3,...] to an image with values stored as bits, where multiple bits (features) can be "on", e.g. features=[1,2,4,8,16,...], values=[1,255,7,4,3,0,1,6,...]
def enum_to_bitmask(enum_array, num_classes=None):
    if num_classes is None:
        #num_classes = int(round(math.log(np.max(bitmask), 2)))
        num_classes = np.max(enum_array) + 1
    flattened = np.zeros(enum_array.shape, enum_array.dtype)
    bitmask = 1
    for l in range(1, num_classes+1):
        flattened += bitmask * np.equal(enum_array, l).astype(enum_array.dtype)
        bitmask = bitmask << 1
    return flattened
    
# LOSSY compression of bitmasks back down to more compact enum representation.
def bitmask_to_enum(bitmask_array, num_classes=None):
    if num_classes is None:
        num_classes = np.max(bitmask_array) + 1
    flattened = np.zeros(bitmask_array.shape, bitmask_array.dtype)
    enum = np.log2(bitmask_array*2, where=np.not_equal(bitmask_array, 0))
    return enum.astype(bitmask_array.dtype)

def maskify(mask, num_copies=None):
    if len(mask.shape) == 3:
        if num_copies is None:
            num_copies = mask.shape[-1]
        mask = np.clip(np.sum(mask, axis=-1)*255.0, 0, 1)

    mask = np.expand_dims(mask, -1)
    mask = np.tile(mask, (1, 1, num_copies))
    return mask

import unittest
class TestUtils(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[[0,1,0],[1,0,0]],[[0,0,0],[0,0,1]]])
        self.a = np.array([[2,1],[0,3]])
        self.bm = np.array([[2,1],[0,4]]) # This WAS backwards!! (np.array([[2,4],[0,1]]) Remember, higher order bits in this case read from LEFT TO RIGHT!
        self.enum = np.array([[2,1],[0,3]])
        self.a1 = np.array([[2,1],[1,3]])
        self.A1 = np.array([[[0,1,0],[1,0,0]],[[1,0,0],[0,0,1]]])

    def test_enum_to_onehot_and_back(self):
        arr = np.random.randint(0,16,size=(25,25))
        np.testing.assert_array_equal(arr, onehot_to_enum(enum_to_onehot(arr)))

    def test_onehot_to_enum(self):
        np.testing.assert_array_equal(self.a, onehot_to_enum(self.A, one_indexed=True))

    def test_enum_to_onehot(self):
        #print(make_onehot(self.a1, ignore_first=True))
        np.testing.assert_array_equal(self.A1, enum_to_onehot(self.a1, ignore_first=True), verbose=True)

    def test_multihot_to_bitmask(self):
        np.testing.assert_array_equal(self.bm, multihot_to_bitmask(self.A))

    def test_bitmask_to_multihot(self):
        #print(bitmask_to_multihot(self.bm))
        np.testing.assert_array_equal(self.A, bitmask_to_multihot(self.bm, self.A.shape[-1]))
    
    def test_multihot_to_bitmask_and_back_and_back(self):
        bm1 = multihot_to_bitmask(self.A)
        np.testing.assert_array_equal(self.bm, bm1)
        Abm = bitmask_to_multihot(bm1, self.A.shape[-1])
        np.testing.assert_array_equal(self.A, Abm)
        Abm1 = multihot_to_bitmask(Abm)
        np.testing.assert_array_equal(bm1, Abm1)
    
    def test_enum_to_bitmask(self):
        #print(enum_to_bitmask(self.enum, num_classes=4))
        np.testing.assert_array_equal(self.bm, enum_to_bitmask(self.enum, num_classes=4))
    
    def test_bitmask_to_enum(self):
        #print(bitmask_to_enum(self.bm, num_classes=4))
        np.testing.assert_array_equal(self.enum, bitmask_to_enum(self.bm, num_classes=4))
    
    def test_onehot_enum_bitmask_and_back(self):
        enum = onehot_to_enum(self.A.astype('uint8'), one_indexed=False)
        #print(enum.dtype)
        bitmask = enum_to_bitmask(enum, num_classes=self.A.shape[-1])
        #print(bitmask.dtype)
        enum2 = bitmask_to_enum(bitmask, num_classes=self.A.shape[-1]).astype(enum.dtype)
        #print(enum2.dtype)
        #print(np.sum(np.abs(enum - enum2)))
        np.testing.assert_array_equal(enum, enum2)

def get_power_of_two_padding(image, power=5):
    align_boundary = 2**power
    leftover_h = image.shape[0] % align_boundary
    leftover_w = image.shape[1] % align_boundary
    pad_x = align_boundary-leftover_w if leftover_w != 0 else 0
    pad_y = align_boundary-leftover_h if leftover_h != 0 else 0
    return pad_x, pad_y


if __name__ == '__main__':
    unittest.main(verbosity=2)
