from __future__ import print_function
# Old HBA code. Blah. #TODO: GO through this and salvage anything useful.
import numpy as np
import os
import json
import cv2
from models.nn_utils import *

from visuals.visuals import *
from postprocessing import *

from multiprocessing import Pool
import multiprocessing

import time

import re
from joblib import Parallel, delayed
from utils import _mkdir

from visuals.visuals import vis_img as show

def get_lines_and_masks(linechunk):
    yxs = np.zeros((len(linechunk),2),dtype=np.uint16)
    for i, line in enumerate(linechunk):
        line = line.rstrip('\n')
        linesplit = line.split('\t')
        if len(linesplit) < 2:
            linesplit = line.split(' ')
        x = int(linesplit[0])
        y = int(linesplit[1])
        yxs[i,0] = y
        yxs[i,1] = x
        #predvalue = predmax[y,x] + 1
        #image[y,x,int(predvalue-1)] = 255 #1.0
        #gt_mask[y,x] = 255
        #max_x = max(x, max_x)
        #max_y = max(y, max_y)
        #newlines.append(str(x) + "\t" + str(y) + "\t" + str(int(predvalue)) +"\n")
        #line = line + '\t' + str(int(predvalue)) +'\n'
        #print((x,y,predvalue,line))
    return yxs #(x,y) #(x,y,line) #,predvalue,line)


def processInput(i):
    return i*i

def load_gt_mask(base_filename):
    gt_mask = None
    gt_mask_png_file = base_filename.replace(".txt", "_gtmask.png")
    gt_mask_npz_file = base_filename.replace(".txt", "_gtmask.npz")
    print("SEARCHING MOST DILIGENTLY FOR ", gt_mask_png_file, gt_mask_npz_file)
    if os.path.exists(gt_mask_npz_file):
        print("Loading npz gt mask from disk!", gt_mask_npz_file)
        gt_mask = np.load(gt_mask_npz_file)['arr_0']
    elif os.path.exists(gt_mask_png_file):
        print("Loading png from disk!", gt_mask_png_file)
        gt_mask = cv2.imread(gt_mask_png_file, 0)

    return gt_mask

def write_predictions_to_pnglayers(preds, output_path, basename, verbose=0):
    #TODO: Implement me!!!
    for channel in range(preds.shape[-1]):
        layer = preds[:,:,channel]
        cv2.imwrite(os.path.join(output_path, basename + "_" + str(channel) + ".jpg"), layer*255)
    vis = vis_img(preds, bgr=False)
    cv2.imwrite(os.path.join(output_path, basename + "_" + "SS_RGB" + ".jpg"), vis*255)

def write_predictions_to_png_indexed(predmax, im_path, gt_mask_file, verbose=True):
    splitext = os.path.splitext(im_path)
    dirname = os.path.dirname(im_path)
    basename = os.path.basename(splitext[0])
    # Now load in GT mask and output text to experimental output folder.
    '''
    gt_mask_file = os.path.join(os.path.join(dirname, "gt/"), basename + ".txt")
    if not os.path.exists(gt_mask_file):
        gt_mask_file = os.path.join(dirname, basename + ".txt")
    if not os.path.exists(gt_mask_file):
        gt_mask_file = os.path.join(os.path.dirname(dirname), basename + ".txt") # Look for gt in parent directory
    if not os.path.exists(gt_mask_file):
        print("Major error; aborting prediction write. File not found:", gt_mask_file)
    '''
    _mkdir(os.path.join(dirname, "result/"))
    pred_file = os.path.join(os.path.join(dirname, "result/"), basename + ".png")
    preds_enum = predmax
    print("Writing predictions as .PNG file...", pred_file, "Shape:", preds_enum.shape)
    cv2.imwrite(pred_file, preds_enum, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    return

def write_predictions_to_txt(predmax, im_path, gt_mask_file, verbose=True):
    splitext = os.path.splitext(im_path)
    dirname = os.path.dirname(im_path)
    basename = os.path.basename(splitext[0])
    # Now load in GT mask and output text to experimental output folder.
    '''
    gt_mask_file = os.path.join(os.path.join(dirname, "gt/"), basename + ".txt")
    if not os.path.exists(gt_mask_file):
        gt_mask_file = os.path.join(dirname, basename + ".txt")
    if not os.path.exists(gt_mask_file):
        gt_mask_file = os.path.join(os.path.dirname(dirname), basename + ".txt") # Look for gt in parent directory
    if not os.path.exists(gt_mask_file):
        print("Major error; aborting prediction write. File not found:", gt_mask_file)
    '''
    gt_mask = None
    gt_mask_png_file = gt_mask_file.replace(".txt", "_gtmask.png")
    gt_mask_npz_file = gt_mask_file.replace(".txt", "_gtmask.npz")
    print(gt_mask_file, gt_mask_png_file)
    if os.path.exists(gt_mask_file.replace(".txt", ".jpg_gt.npz")):
        print("Loading npz mask from disk!")
        gt = np.load(gt_mask_file.replace(".txt", ".jpg_gt.npz"))['arr_0']
    t = time.time()
    newlines = []
    if os.path.exists(gt_mask_png_file):
        gt_mask = cv2.imread(gt_mask_png_file, 0)
    elif os.path.exists(gt_mask_npz_file):
        gt_mask = np.load(gt_mask_npz_file)['arr_0']


    if os.path.exists(gt_mask_png_file) and False:
        # Option 3: Use a .PNG image and Numpy. About 2x faster than Option 2, but it outputs elements in the "wrong" order.
        if verbose:
            print("   Reading GT mask from PNG file:", gt_mask_png_file)
        gt_mask = cv2.imread(gt_mask_png_file, 0)
        (x,y) = np.where(gt_mask > 0)
        predslist = predmax[x,y] + 1
        if verbose:
            print("phase 1")
        #print("   ", predslist)
        #image = np.multiply(predmax, gt_mask)
        #print(x.shape, y.shape, predslist.shape)
        #print(x.dtype, y.dtype, predslist.dtype)
        #newarr = np.zeros((x.shape[0], 3), dtype=np.uint16)
        #newarr[:,0] = x
        #newarr[:,1] = y
        #newarr[:,2] = predslist

        xyp = np.array((x,y,predslist))
        print("UNSORTED", xyp[:,0])
        print("UNSORTED", xyp[:,1])
        print("UNSORTED", xyp[:,100])
        xyp = np.sort(xyp, axis=1)
        print("SORTED", xyp[:,0])
        print("SORTED", xyp[:,1])
        print("SORTED", xyp[:,100])
        x = xyp[0]
        y = xyp[1]
        predslist = xyp[2]

        #newarr = newarr.astype('|S5').astype(np.object_) #'|S20') #view(np.chararray)
        #newarr = newarr.astype(np.object_) #'|S20') #view(np.chararray)

        x = x.astype('|S5').astype(np.object_)
        y = y.astype('|S5').astype(np.object_)
        predslist = predslist.astype('|S1').astype(np.object_)

        #xyp = np.sort(xyp, 0)

        k = time.time()
        #newarr = newarr[:,0] + '\t' + newarr[:,1] + '\t' + newarr[:,2] + '\n'
        newarr = y + '\t' + x + '\t' + predslist + '\n'
        #newarr = xyp[0] + '\t' + xyp[1] + '\t' + xyp[2] + '\n'
        k = time.time() - k
        #print(newarr)

        #newarr = np.sort(newarr, 0) #, order=[])

        #newarr = np.sort(newarr, axis=0)

        newlines = newarr.tolist()

        #for i in range(newarr.shape[0]): # SLOW! 17s on just this tiny loop.
        #    newlines.append("" + str(newarr[i,0]) + "\t" + str(newarr[i,1]) + "\t" + str(newarr[i,2]))
        #strs = newarr[:,0] + "\t" + newarr[:,1] + "\t" + newarr[:,2]
        #newlines = strs.tolist()

        #print("1---")
        #newlines = [str(x[i]) + "\t" + str(y[i]) + "\t" + str(int(predslist[i])) for i in range(predslist.shape[0])]
        if verbose:
            print("   Got", len(newlines), "newlines.")

        print(newlines[0])
    else:
        k = 0
        if verbose:
            print("   Reading GT mask from TXT file", gt_mask_file)
        oldlines = newlines
        newlines = []
        max_x = 0
        max_y = 0
        #image = np.zeros((8000,6000,6), dtype=np.float32) #,dtype=np.uint8)
        if gt_mask is None:
            gt_mask = np.zeros((predmax.shape[0], predmax.shape[1]), dtype=np.uint8)

        with open(gt_mask_file, 'r') as f:
            lines = f.readlines()
            '''
            nls = np.zeros((len(lines),), dtype='|S12')
            ps  = np.zeros((len(lines),), dtype='|S1')
            for i,line in enumerate(lines):
                line = line.strip('\n')
                al = nls[i] #gt_mask_file[i]
                nls[i] = line
                strs = line.split('\t', 2)[:2]
                #x, y = al[0], al[1] = int(strs[0]), int(strs[1])
                x, y = int(strs[0]), int(strs[1])
                predvalue = predmax[y,x] + 1
                ps[i] = predvalue
                #gt_mask[y,x] = 1
                #max_x = max(x, max_x)
                #max_y = max(y, max_y)
                #newlines.append(line + "\t" + str(int(predvalue)) +"\n")
                #image = image[:max_y+1,:max_x+1,:]
            nls = nls.astype(np.object_)
            ps = ps.astype(np.object_)
            nls = nls + '\t' + ps + '\n'
            newlines = nls.tolist()
            '''
            multiproc = True
            if not multiproc:
                # Option 1: SLOW
                import tqdm
                gt_mask_old = gt_mask
                num_correct = 0
                num_incorrect = 0
                for line in tqdm.tqdm(lines):
                    line = line.rstrip('\n')
                    linesplit = line.split('\t')
                    if len(linesplit) < 2:
                        linesplit = line.split(' ')
                    x = int(linesplit[0])
                    y = int(linesplit[1])
                    predvalue = predmax[y,x] + 1

                    if gt is not None:
                        if predvalue == np.argmax(gt[y,x], axis=-1)+1:
                            num_correct += 1
                        else:
                            num_incorrect += 1

                    #image[y,x,int(predvalue-1)] = 255 #1.0
                    #gt_mask[y,x] = 255
                    #max_x = max(x, max_x)
                    #max_y = max(y, max_y)
                    newlines.append(str(x) + "\t" + str(y) + "\t" + str(int(predvalue)) +"\n")
                    #newlines.append(line + '\t' + str(int(predvalue)) +'\n')
                    #image = image[:max_y+1,:max_x+1,:]
                print("SUMDIFF:", np.count_nonzero(gt_mask_old-gt_mask)) #, np.count_nonzero(gt_mask), np.count_nonzero(gt_mask_old))
                #from itertools import groupby
                #print([len(list(group)) for key, group in groupby(a)])
                print("SYMDIFF of lines:", len(set(newlines).symmetric_difference(set(oldlines))))

                print("Correct, incorrect:", num_correct, num_incorrect, "Accuracy:", float(num_correct) / float(num_correct + num_incorrect))

                gt_mask_slow = gt_mask
                yonlines = newlines
                print(newlines[0])
            else:
                # Option 2: Parallel
                # This is about 6X faster on 12 cores than the non-parallel version above.
                cores = multiprocessing.cpu_count()
                print("Cores:", cores, "Lines:", len(lines))
                t = time.time()
                #lines = lines[:10000]
                def chunks(l, n):
                    for i in xrange(0, len(l), n):
                        yield l[i:i+n]
                yxchunks = Parallel(n_jobs=cores)(delayed(get_lines_and_masks)(linechunk) for linechunk in chunks(lines, 10000))
                yxs = [item for sublist in yxchunks for item in sublist]
                print("Parallel Chunks:", len(yxchunks))
                print("Total entries:", len(yxs))
                if len(yxs) == 0:
                    newlines = []
                else:
                    yxs = np.array(yxs)
                    print("yxs.shape", yxs.shape)
                    x = yxs[:,1]
                    y = yxs[:,0] # Do away with need to index like this!!!
                    predslist = predmax[y,x] + 1 #yxs] #predmax[x,y]
                    if verbose:
                        print("phase 1")

                    x = x.astype('|S5').astype(np.object_)
                    y = y.astype('|S5').astype(np.object_)
                    predslist = predslist.astype('|S1').astype(np.object_)

                    k = time.time()
                    newarr = x + '\t' + y + '\t' + predslist + '\n'
                    k = time.time() - k

                    newlines = newarr.tolist()
                    print(newlines[0])

                    #gt_mask[yxs] = 255 # TODO: This is incorrect!!!!!!
                    t = time.time()-t
                    print("Time elapsed in parallel loop:", t)
                    #inputs = range(100)
                    #results = Parallel(n_jobs=cores)(delayed(processInput)(i) for i in inputs)
                    #print(inputs)
                    #print("SUMDIFF 2:", np.count_nonzero(gt_mask_slow-gt_mask) #, np.count_nonzero(gt_mask), np.count_nonzero(gt_mask_slow))
                    #print("SYMDIFF of lines:", len(set(newlines).symmetric_difference(set(yonlines))))
            print("*** GOT >>> ", len(newlines), "newlines.") #, len(oldlines), newlines[0])
    max_x = predmax.shape[1]
    max_y = predmax.shape[0]
    t = time.time() - t
    if verbose:
        print("   Reading mask time:", t, "Inner time:", k) #, "vs", t1, t/t1)
    _mkdir(os.path.join(dirname, "result/"))
    pred_file = os.path.join(os.path.join(dirname, "result/"), basename + ".txt")
    if verbose:
        print("   Writing predictions to " + pred_file)
    t = time.time()
    with open(pred_file, 'w') as f:
        f.writelines(newlines)
    t = time.time() - t
    if verbose:
        print("   Writing mask time:", t)
    print("   DONE Writing predictions to " + pred_file)

    '''
    gtm = np.pad(gt_mask, [(0, predmax.shape[0]-gt_mask.shape[0]), (0, predmax.shape[1]-gt_mask.shape[1])], mode='constant')
    print("Enum to onehot")
    image = vis_img(enum_to_multihot(predmax, num_labels=6), predmax.shape[0], predmax.shape[1])
    print("Write")
    cv2.imwrite(im_path + "_final_fullres_pred.jpg", image*255)
    print("Maskify")
    image = np.multiply(maskify(gtm, 3), image)
    cv2.imwrite(im_path + "_final_fullres_pred_gtmasked.jpg", image*255)
    #image = vis_img(image, max_y+1, max_x+1)
    #cv2.imwrite(im_path + "_final_fullres_pred.jpg", image)
    '''
    return gt_mask

def write_preds(queue):
    try:
        #print(os.getpid(), "working")
        while True:
            item = queue.get(True)
            if item == 'DONE':
                #print(os.getpid(), "Exiting...")
                return
            #print(os.getpid(), "got", item[1])
            write_predictions_to_pnglayers(*item)
            time.sleep(1)
    except Exception as e:
        print(e)

#def write_txt_and_image(predmaxes, im_path, gt_mask_filenames[ip]):
#    gt_mask = np.clip(cv2.resize(gt_mask, (0,0), fx=1.0/testscale, fy=1.0/testscale)[:orig_image_shape[0],:orig_image_shape[1]]*255, 0, 1)
#    if pad_x != 0 or pad_y != 0:
#        gt_mask = np.pad(gt_mask, [(0,pad_y),(0, pad_x)], mode='constant')
# TODO: Make initial or large-template kernels zero-mean! This will help gradients go where they need to go.
def TestModel(model_basepath=None, model=None, testfolder="./", output_folder="./", testscale=1.0, do_median=False, pixel_counts_byclass=None, multiprocess=False, verbose_level=0, suffix_to_class_map=None):
    from data_loaders.data_loaders import WholeImageOnlyBatcher
    from data_loaders.utils import get_power_of_two_padding
    import sys
    sys.path.append("../")
    from utils import mkdir_p
    mkdir_p(output_folder)

    if multiprocess:
        work_queue = multiprocessing.Queue()
        worker_pool = multiprocessing.Pool(8, write_preds, (work_queue,))

    #testscale=8.0
    #bag_of_models = []
    #if model is not None:
    #    bag_of_models.append(model)

    #print("Testing " + str(len(bag_of_models)) + " models.")

    # Load GT txts and re-populate as a new snapshot, and .tar.gz it!
    img_filenames = sorted(os.listdir(testfolder))
    test_filenames = []
    for img_filename in img_filenames:
        im_path = os.path.join(testfolder, img_filename)
        splitext = os.path.splitext(im_path.lower())
        ext = splitext[-1]
        if not ext in ['.jpg']:
            continue
        if '.jpg_' in os.path.basename(splitext[0]):
            continue # It's a GT layer file or an output; not an input; IGNORE.
        splitext = os.path.splitext(im_path)
        dirname = os.path.dirname(im_path)
        basename = os.path.basename(splitext[0])
        test_filenames.append(im_path)

    # Automatically infer the scale of the images.
    image_scale = re.match(".*scale_([\d]+)[^\d]*", testfolder)
    if image_scale is not None and len(image_scale.groups()) > 0:
        #print(image_scale.groups())
        image_scale = image_scale.groups()[0]
        print("Using inferred Image scale:", image_scale, "/100")
        testscale = 100.0/float(image_scale)
    else:
        testscale=1.0#1.0/.13#8.0

    pred = np.ones((1,1,1,1), dtype=np.float32)
    print("Testing", len(test_filenames), "Test images.")
    ip = 0
    ok = False
    for im_path in test_filenames:
        splitext = os.path.splitext(im_path)
        dirname = os.path.dirname(im_path)
        basename = os.path.basename(splitext[0])

        #if "Book4" in im_path and not (ok or "459" in im_path):
        #     continue
        #ok = True
        print("Inferencing on", im_path)

        #if ip >= 2:
        #    return
        image = cv2.imread(im_path).astype('float32')/255.0
        if len(image.shape) < 3:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        print("Image size:", image.shape)

        if verbose_level > 0:
            cv2.imshow('Inferencing on image', image)
            cv2.waitKey(5)

        # DISABLED dynamic resize for test images.
        #if testscale != 1.0:
        #    #image = cv2.resize(image, (0,0), fx=1.0/testscale, fy=1.0/testscale)
        #    #image = cv2.resize(image, (0,0), fx=1.0/testscale, fy=1.0/testscale)
        pad_x, pad_y = get_power_of_two_padding(image, 8)
        orig_image_shape = image.shape
        if pad_x != 0 or pad_y != 0:
            image = np.pad(image, [(0,pad_y),(0, pad_x),(0,0)], mode='constant')
        print("Image size padded:", image.shape)

        image = np.reshape(image, [-1, image.shape[0], image.shape[1], image.shape[2]])
        print("Image size: reshaped", image.shape)

        predmaxes = []
        #for model in bag_of_models:
        if do_median:
            if model_basepath is not None:
                model_num = 0
                model_path = model_basepath + "_" + str(model_num) + ".h5"
                while os.path.exists(model_path):
                    model.load_weights(model_path)
                    #bag_of_models.append(model)
                    print("Appended model " + model_path + " for testing.")
                    model_num += 1
                    model_path = model_basepath + "_" + str(model_num) + ".h5"
                    pred = model.predict(image, batch_size=1)[0]
                    # Now scale up the predictions
                    predmax = np.argmax(pred, axis=-1)
                    predmaxes.append(predmax)

            # NOW do median filtering on the predmaxes.
            predmaxes = np.array(predmaxes)
            #print(predmaxes.shape)
            predmaxes = np.transpose(predmaxes, [1,2,0])
            predmaxes = np.median(predmaxes, axis=-1)
        else:
            print("Performing inference...")
            try:
                pred = model.predict(image, batch_size=1)#[0]
                a = 1/0
            except Exception as ex: #tf.errors.ResourceExhaustedError as ex:
                print("Validation threw Out of Memory Error", ex)
                # Now try chunking and stitching the image for better inference.
                # Send image chunks in instead.
                size = 224*4 #max(image.shape[1], image.shape[2]) / 2 + 2
                stride = size / 2
                batch_size = 1
                pred = None
                while True:
                    print("Performing chunked inference at size", size, "with stride", stride)
                    try:
                        for x in range(0, image.shape[2]-size+stride/2, stride):
                            for y in range(0, image.shape[1]-size+stride/2, stride):
                                imchunk = image[:, y:y+size, x:x+size, :]
                                yx,xs = imchunk.shape[1:3]
                                xd = size-imchunk.shape[2]
                                yd = size-imchunk.shape[1]
                                if xd > 0 or yd > 0:
                                    imchunk = np.pad(imchunk, ((0,0), (0,yd), (0,xd), (0,0)), mode='constant', constant_values=0)
                                #cv2.imshow('imchunk', imchunk[0])
                                #cv2.waitKey()
                                p = model.predict(imchunk, batch_size=batch_size)
                                #cv2.imshow('p', show(p[0], bgr=True))
                                if pred is None:
                                    pred = np.zeros((image.shape[0], image.shape[1], image.shape[2], p.shape[-1]))
                                u = stride/2 if y > 0 else 0
                                l = stride/2 if x > 0 else 0
                                b = size-stride/2 if y+size < image.shape[1] else image.shape[1]-y
                                r = size-stride/2 if x+size < image.shape[2] else image.shape[2]-x
                                pred[0, y+u:y+b, x+l:x+r, :] = p[0, u:b, l:r, :]

                                #cv2.imshow('pred', show(pred[0], bgr=True))
                                #cv2.waitKey()
                        break
                    except tf.errors.ResourceExhaustedError as ex:
                        print("Still getting OOM Error on chunked inference. Halving chunk dimensions...")
                        size = size / 2 + 1
                        stride = size / 2
                        if size < 16:
                            print("Context too small for realistic inference. Aborting...")
                            exit(-1)


            print("Image shape:", image.shape)
            print("Pred shape:", pred.shape)
            # TODO retrieve GT mask.
            print("Pred minmax BEFORE postprocess", np.min(pred), np.max(pred))
            #pred = postprocess_preds(image, pred, gt_mask=None, pixel_counts_byclass=pixel_counts_byclass)
            print("Pred minmax after postprocess", np.min(pred), np.max(pred))
            #show(image[0], None, pred[0], 1024, 1024, im_path)

            #predmaxes = np.argmax(pred[0], axis=-1)

        # Now scale back up to original size.
        #print(testfolder)

        preds = pred[0]
        if testscale != 1.0:
            print("")
            print("RESCALING...", testscale)
            #predmaxes = cv2.resize(predmaxes, (0,0), fx=testscale, fy=testscale, interpolation=cv2.INTER_LINEAR)
            preds = cv2.resize(preds, (0,0), fx=testscale, fy=testscale, interpolation=cv2.INTER_LINEAR)
            print(preds.shape)

        if False:
            predsrgbchannels = preds[:int(orig_image_shape[0]*testscale), :int(orig_image_shape[1]*testscale)]
            predsrgbchannels = multihot_to_multiindexed_rgb(predsrgbchannels)
            cv2.imwrite(im_path.replace(test_folder, output_folder) + "_predsrgbchannels.png", predsrgbchannels)

        predshow = show(preds, bgr=False)
        if output_folder is not None:
            cv2.imwrite(os.path.join(output_folder, basename + "-Pred.png"), predshow*255)

        write_predictions_to_pnglayers(preds, output_folder, basename)

        ip += 1
    print("FINISHED inference. Waiting for workers to complete...")
    if multiprocess:
        for i in range(12):
            work_queue.put('DONE')
        worker_pool.close()
        print("Close instruction sent.")
        worker_pool.join()
        print("All workers finished.")

    # Recursive recognition starting at the whole page level.
    def recognize(self, page):
        # Slice up image into regions and make dense predictions for each region!!!
        img = cv2.imread(page['image_path'], 0)
        display = False
        if display:
            output = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        if self.recognition_scale != 1:
            img = cv2.resize(img, (int(img.shape[1]*self.recognition_scale), int(img.shape[0]*self.recognition_scale)))
        confident_matches = []
        import random
        # Now for the fun part. Display the image, show bounding boxes, and show what is recognized in each of the boxes!!!!
        width = self.width
        height = self.height
        num_classes = self.num_classes

        #for randpc in range(max_attempts):
        # Start with zoomed out version.
        # Slice into overlapping quadrants. Find out what each one is.
        # Zoom in as appropriate.
        # VERY FAST!!! DO These in batches!!!

        num_dwn = 1 if self.height < 100 else 0
        num_up = 0
        orig_img = img
        img_pyramid = []
        if num_up == 1:
            img_pyramid.append(cv2.pyrUp(orig_img))
        img_pyramid.append(img)
        for im in range(num_dwn):
            img_pyramid.append(cv2.pyrDown(img_pyramid[-1]))

        do_offsets = True
        densepreds = np.zeros((img.shape[0], img.shape[1], num_classes))
        tot_regions = 0
        regions_drawn = 0
        scale = 2.0 ** (num_up+1)
        tot_masks = len(img_pyramid) + 1

        for img in img_pyramid:
            scale /= 2.0
            output = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            div=1 # 4
            num_in_batch = 0
            gt_label = np.zeros((self.batch_size, height, width, num_classes))
            im_batch = np.zeros((self.batch_size, height, width, 1))
            xys = []
            offx = 0
            offy = 0
            offsets = [(offx, offy)]

            if scale == 1 and do_offsets:
                offx = width/8
                offy = height/8
                offsets.append((offx, offy))
                offx = width/4
                offy = height/4
                offsets.append((offx, offy))
                offx = 3*width/8
                offy = 3*height/8
                offsets.append((offx, offy))
                #offx = -width/4
                #offy = -height/4
                #offsets.append((offx, offy))
            wait=50
            for offx,offy in offsets:
                for randy in np.arange(offx,img.shape[0],height/div):
                    if display:
                        overlay = output.copy() #cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                    for randx in np.arange(offy,img.shape[1],width/div):

                        #print(randpc, "/", max_attempts)
                        #randx = random.randint(0,img.shape[1]-28)
                        #randy = random.randint(0,img.shape[0]-28)
                        #overlay.fill(0)
                        #cv2.rectangle(overlay, (randx, randy), (randx+width, randy+height), (255, 0, 0), 1)


                        crop = img[randy:randy+height,randx:randx+width]
                        if crop.shape[:2] != (self.height, self.width):
                            crop = scale_pad_and_center(crop, height, width, do_center=False)
                        crop = crop.astype('float32')
                        #clasnum = random.randint(0,3)
                        #clas = self.class_num_to_symbol[clasnum]
                        #crop, gtcrop = self.class_training_exemplar_generators[clas](self.img, self.mask, maskval=0, numclasses=4, label=clasnum, minsize=self.height/4, maxsize=self.height*4, height=self.height, width=self.width, maskchannel=clasnum)

                        im_batch[num_in_batch] = np.reshape(crop, [height, width, 1])/255.0
                        xys.append((randx, randy))
                        #print(randx, randy)
                        num_in_batch += 1
                        tot_regions += 1

                        if num_in_batch == self.batch_size or (randy >= img.shape[0]-height-1 and randx >= img.shape[1]-width-1):
                            #print(im_batch.shape, gt_label.shape)
                            # RUN RUN RUN RUN RUN RUN RUN !!!!!!!
                            recs = self.sess.run(self.y_conv, feed_dict={self.x:1.0-im_batch, self.y_:gt_label, self.keep_prob:1.0})
                            #print("Ran recognition batch:", recs.shape)
                            regions_drawn += len(xys)
                            for b in range(len(xys)): #range(recs.shape[0]):
                                #rec = recs[b][13][13]
                                #print(rec.shape)
                                #recInd = np.argmax(rec[0:12])
                                #angleInd = np.argmax(rec[12:16])
                                #angle = angleInd * 90
                                #recChar = str(recInd)
                                r = recs[b]
                                #cv2.imshow('recognized_patch', r)
                                #cv2.waitKey(10)
                                #cv2.imshow('patch_gt', gtcrop)
                                #cv2.waitKey(10)

                                #cv2.imshow('original_patch', im_batch[b])
                                #cv2.waitKey(wait)
                                rx, ry = xys[b]
                                #print(scale, ry, ry+int(height/scale), rx, rx+int(width/scale))
                                h,w,_ = densepreds[int(ry/scale):int((ry+height)/scale),int(rx/scale):int((rx+width)/scale),:].shape
                                #try:
                                densepreds[int(ry/scale):int((ry+height)/scale),int(rx/scale):int((rx+width)/scale),:] += cv2.resize(r, (int(height/scale), int(width/scale)), interpolation=cv2.INTER_LINEAR)[0:h,0:w,:] # Add into image sum.
                                #except:
                                #    pass
                                if display:
                                    for px in range(r.shape[1]):
                                        for py in range(r.shape[0]):
                                            rec = r[py][px]
                                            recInd = np.argmax(rec)
                                            if rec[recInd] > 0.0:
                                                confColor = int(255.0*rec[recInd])
                                                if self.class_num_to_symbol[recInd] == 'blank':
                                                    cv2.rectangle(overlay, (rx+px, ry+py), (rx+px, ry+py), (confColor, confColor, confColor), -1) # Blank is white
                                                elif self.class_num_to_symbol[recInd] == 'hw':
                                                    cv2.rectangle(overlay, (rx+px, ry+py), (rx+px, ry+py), (0, confColor, 0), -1) # Handwriting is green
                                                elif self.class_num_to_symbol[recInd] == 'machprint':
                                                    cv2.rectangle(overlay, (rx+px, ry+py), (rx+px, ry+py), (0, 0, confColor), -1) # Machine Print is red
                                                elif self.class_num_to_symbol[recInd] == 'lines':
                                                    cv2.rectangle(overlay, (rx+px, ry+py), (rx+px, ry+py), (confColor, 0, 0), -1) # Lines are blue
                                            #else:
                                            #    cv2.rectangle(overlay, (rx+px, ry+py), (rx+px, ry+py), (0, 0, 0), -1) # Unk is black

                            #print(rec)
                            if display:
                                alpha = 0.5
                                cv2.addWeighted(overlay, alpha, output, 1.0-alpha, 0, output)
                                cv2.imshow('ImageRec', output)
                                cv2.waitKey(10)

                            num_in_batch = 0
                            gt_label = np.zeros((self.batch_size, height, width, num_classes))
                            im_batch = np.zeros((self.batch_size, height, width, 1))
                            xys = []
        print(tot_regions, "Total regions recognized", regions_drawn, "drawn")
        #blurred = cv2.blur(output, (2*height/div, 2*width/div))
        #cv2.imshow('Blurred', blurred)
        densepreds = densepreds / tot_masks
        ext = ".jpg"
        densepredscopy = np.copy(densepreds)
        densepredscopy[:,:,0] += densepredscopy[:,:,self.class_to_num['stamp']]/2.0# Blue+Red channel also gets stamps.
        densepredscopy[:,:,2] += densepredscopy[:,:,self.class_to_num['stamp']]/2.0
        # Red+Green (Yellow) channel also gets underlines.
        densepredscopy[:,:,1] += densepredscopy[:,:,self.class_to_num['lines']]/2.0
        densepredscopy[:,:,2] += densepredscopy[:,:,self.class_to_num['lines']]/2.0
        densepredscopy = (np.clip(densepredscopy[:,:,:3], 0, 1)*255.0).astype('uint8')
        for i in range(10):
            print("")
        print(page["image_path"])
        oimp = page["image_path"].replace(test_folder, output_folder)
        cv2.imwrite(oimp + "_SS_RGB"+ext, densepredscopy)
        densepreds = (np.clip(densepreds[:,:,:], 0, 1)*255.0).astype('uint8')
        cv2.imwrite(oimp + "_LN"+ext, densepreds[:,:,self.class_to_num['lines']])
        cv2.imwrite(oimp + "_ST"+ext, densepreds[:,:,self.class_to_num['stamp']])
        cv2.imwrite(oimp + "_MP"+ext, densepreds[:,:,self.class_to_num['machprint']])
        cv2.imwrite(oimp + "_HW"+ext, densepreds[:,:,self.class_to_num['hw']])
        cv2.imwrite(oimp + "_DL"+ext, densepreds[:,:,self.class_to_num['dotted_lines']])

        write_masked_channels(img, densepreds, oimp)

        if display:
            cv2.imshow("Dense Averaged Multi-scale predictions", densepreds[:,:,0:3])
            cv2.waitKey(10)
        print("Done recognizing.")
        return densepreds
