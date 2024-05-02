from __future__ import print_function
import random
import cv2
import numpy as np

# Count overlapping bounding boxes containing characters.
def get_overlap_amount(region, bbox):
    bbox_in_region = [[max(bbox[0][0], region[0][0]), max(bbox[0][1], region[0][1])], [min(bbox[1][0], region[1][0]), min(bbox[1][1], region[1][1])]]
    w,h = bbox_in_region[1][0]-bbox_in_region[0][0],bbox_in_region[1][1]-bbox_in_region[0][1]
    w = max(0,w)
    h = max(0,h)
    return float(w*h)/float((bbox[1][0]-bbox[0][0])*(bbox[1][1]-bbox[0][1]))

def bbox_overlaps(region, bbox):
    if bbox[1][0] < region[0][0] or bbox[0][0] > region[1][0] or bbox[0][1] > region[1][1] or bbox[1][1] < region[0][1]:
        return False
    return True

def get_masked_regionsampler_semanticseg_multichannel_withtextimages(img, maskpixels, charmasks=None, maskval=0, numclasses=4, label=0, minsize=28, maxsize=400, height=28, width=28, blank_label=-1, maskchannel=0, page=None, char_to_idx={}, random_scale=False, prenormed=True):
    random_scale = random.random() > 0.5
    if random_scale:
        invalid = True
        tries = 0
        criterion = 0.03 #.03 # 1 is center pixel, <1 is proportion of "on" pixels
        if criterion == 1:
            while invalid:
                x = random.randint(0,img.shape[1]-1)
                y = random.randint(0,img.shape[0]-1)
                if maskpixels[y][x][maskchannel] != maskval:
                    tries += 1
                else:
                    invalid = False
                    break
            size = random.randint(minsize,maxsize)
            l = max(0, x-size//2)
            r = min(img.shape[1], x+size//2)
            u = max(0, y-size//2)
            b = min(img.shape[0], y+size//2)
        else:
            if random.random() > 0.25:
                wordnum = get_word_with_charsegs(page)
                gt = page['gt_as_wordlist'][wordnum]
                bbox = page['bboxes'][wordnum]
                charsegs = page['bbox_charsegs'][wordnum]
                charnum = random.randint(0, len(charsegs)-1) if len(charsegs) > 1 else 0
                l = left = charsegs[charnum-1] + bbox[0][0] if charnum > 0 else bbox[0][0]
                r = right = charsegs[charnum] + bbox[0][0] if charnum < len(charsegs) else bbox[1][0]
                u = bbox[0][1]
                b = bbox[1][1]
                # Expand aspect ratio in either direction.
                hwdiff = (b-u)-(r-l)
                if hwdiff < 0:
                    split = random.random()
                    b += int(split*abs(hwdiff)+0.5)
                    u = b - (r-l)
                elif hwdiff > 0:
                    split = random.random()
                    r += int(split*abs(hwdiff)+0.5)
                    l = r - (b-u)
                size = b-u
                x = (l + r) / 2
                y = (u + b) / 2
                #sample = img[bbox[0][1]:bbox[1][1],left:right]
                invalid = False
            else:
                while invalid:
                    x = random.randint(0,img.shape[1]-1)
                    y = random.randint(0,img.shape[0]-1)
                    #print(minsize, maxsize)
                    size = random.randint(minsize,maxsize)
                    l = max(0, x-size//2)
                    r = min(img.shape[1], x+size//2)
                    u = max(0, y-size//2)
                    b = min(img.shape[0], y+size//2)
                    if maskchannel >= maskpixels.shape[2]: # Skip testing for blank; it's virtually always there.
                        invalid = False
                        break
                    #if np.mean((255.0-maskpixels)[u:b,l:r])/255.0 < criterion: # TODO: Assumes maskval is zero
                    if np.mean(maskpixels[u:b,l:r,maskchannel]) < criterion:
                        tries += 1
                    else:
                        invalid = False
                        break
        print(tries, maskval, maskchannel, label)

        #print(x,y,size,img.shape)
        pad_l = max(0, size//2-x)
        pad_r = max(0, size//2-(img.shape[1]-1-x))
        pad_u = max(0, size//2-y)
        pad_b = max(0, size//2-(img.shape[0]-1-y))
        #print(l,r,u,b, ":", pad_l, pad_r, pad_u, pad_b)


        region = np.zeros((size,size))
        region.fill(1.0)
        region[pad_u:pad_u+b-u,pad_l:pad_l+r-l] = img[u:b,l:r]
        region = cv2.resize(region, (width, height))
        #encoded_gt = np.zeros(numclasses)
        #encoded_gt[label] = 1.0
        gt_region = np.zeros((size,size,numclasses))
        #gt_region[pad_u:pad_u+b-u,pad_l:pad_l+r-l,label] = (255.0-maskpixels[u:b,l:r])/255.0
        mp = maskpixels[u:b,l:r]
        #print("mp shape", mp.shape)
        #gt_region[pad_u:pad_u+b-u,pad_l:pad_l+r-l,:maskpixels.shape[2]] = mp
        if charmasks is not None:
            #cm = charmasks[u:b,l:r,:]/255.0
            #print(cm.shape, gt_region.shape)
            gt_region[pad_u:pad_u+b-u,pad_l:pad_l+r-l,:] = mp # np.concatenate((mp, cm), axis=2)
        encoded_gt = cv2.resize(gt_region, (width, height))

        #encoded_gt = np.concatenate((encoded_gt, np.zeros((height, width, 1))), axis=2)
        #encoded_gt[:,:,numclasses-1] = np.zeros((height, width))
        #encoded_gt = np.zeros([height, width, numclasses])
        #gt_resized = cv2.resize(maskpixels[u:b,l:r], (width, height))
        #encoded_gt[:,:,label] = (255.0 - gt_resized)/255.0 # Dark pixels become 1 in this plane
        #encoded_gt[:,:,blank_label] = (gt_resized)/255.0 # Label background as blank_label (otherwise, it is zero--a good label!)


        #print(char_presence_histogram)

        return region, encoded_gt
    maxtries = 3
    for i in range(0, maxtries):
        x = random.randint(0,img.shape[1]-1-width)
        y = random.randint(0,img.shape[0]-1-height)
        region = img[y:y+height,x:x+width]
        if np.std(region) == 0 and i < maxtries-1:
            continue
        encoded_gt = maskpixels[y:y+height,x:x+width,:]
        break
    return region, encoded_gt

def get_masked_regionsampler_semanticseg_multichannel_withtextboxes(img, maskpixels, maskval=0, numclasses=4, label=0, minsize=28, maxsize=400, height=28, width=28, blank_label=-1, maskchannel=0, page=None, char_to_idx={}):
    invalid = True
    tries = 0
    criterion = 0.05 #.03 # 1 is center pixel, <1 is proportion of "on" pixels
    if criterion == 1:
        while invalid:
            x = random.randint(0,img.shape[1]-1)
            y = random.randint(0,img.shape[0]-1)
            if maskpixels[y][x][maskchannel] != maskval:
                tries += 1
            else:
                invalid = False
                break
        if random.random() > 0.75:
            size = random.randint(minsize,maxsize)
        else:
            size = random.randint(minsize,((maxsize)/8+minsize))
        l = max(0, x-size//2)
        r = min(img.shape[1], x+size//2)
        u = max(0, y-size//2)
        b = min(img.shape[0], y+size//2)
    else:
        if random.random() > 0.25:
            wordnum = get_word_with_charsegs(page)
            gt = page['gt_as_wordlist'][wordnum]
            bbox = page['bboxes'][wordnum]
            charsegs = page['bbox_charsegs'][wordnum]
            charnum = random.randint(0, len(charsegs)-1) if len(charsegs) > 1 else 0
            l = left = charsegs[charnum-1] + bbox[0][0] if charnum > 0 else bbox[0][0]
            r = right = charsegs[charnum] + bbox[0][0] if charnum < len(charsegs) else bbox[1][0]
            u = bbox[0][1]
            b = bbox[1][1]
            # Expand aspect ratio in either direction.
            hwdiff = (b-u)-(r-l)
            if hwdiff < 0:
                split = random.random()
                b += int(split*abs(hwdiff)+0.5)
                u = b - (r-l)
            elif hwdiff > 0:
                split = random.random()
                r += int(split*abs(hwdiff)+0.5)
                l = r - (b-u)
            size = b-u
            x = (l + r) / 2
            y = (u + b) / 2
            #sample = img[bbox[0][1]:bbox[1][1],left:right]
            invalid = False
        while invalid:
            x = random.randint(0,img.shape[1]-1)
            y = random.randint(0,img.shape[0]-1)
            #print(minsize, maxsize)
            if random.random() > 0.5:
                size = random.randint(minsize,maxsize)
            else:
                size = random.randint(minsize,(maxsize+minsize)//2)
            l = max(0, x-size//2)
            r = min(img.shape[1], x+size//2)
            u = max(0, y-size//2)
            b = min(img.shape[0], y+size//2)
            if maskchannel >= maskpixels.shape[2]: # Skip testing for blank; it's virtually always there.
                invalid = False
                break
            #if np.mean((255.0-maskpixels)[u:b,l:r])/255.0 < criterion: # TODO: Assumes maskval is zero
            if np.mean(maskpixels[u:b,l:r,maskchannel])/255.0 < criterion:
                tries += 1
            else:
                invalid = False
                break
    #print(tries)

    #print(x,y,size,img.shape)
    pad_l = max(0, size//2-x)
    pad_r = max(0, size//2-(img.shape[1]-1-x))
    pad_u = max(0, size//2-y)
    pad_b = max(0, size//2-(img.shape[0]-1-y))
    #print(l,r,u,b, ":", pad_l, pad_r, pad_u, pad_b)
    region = np.zeros((size,size))
    region.fill(255)
    region[pad_u:pad_u+b-u,pad_l:pad_l+r-l] = img[u:b,l:r]
    region = cv2.resize(region, (width, height))
    #encoded_gt = np.zeros(numclasses)
    #encoded_gt[label] = 1.0
    gt_region = np.zeros((size,size,numclasses))
    #gt_region[pad_u:pad_u+b-u,pad_l:pad_l+r-l,label] = (255.0-maskpixels[u:b,l:r])/255.0
    gt_region[pad_u:pad_u+b-u,pad_l:pad_l+r-l,:maskpixels.shape[2]] = maskpixels[u:b,l:r]/255.0
    encoded_gt = cv2.resize(gt_region, (width, height))
    #encoded_gt = np.concatenate((encoded_gt, np.zeros((height, width, 1))), axis=2)
    #encoded_gt[:,:,numclasses-1] = np.zeros((height, width))
    #encoded_gt = np.zeros([height, width, numclasses])
    #gt_resized = cv2.resize(maskpixels[u:b,l:r], (width, height))
    #encoded_gt[:,:,label] = (255.0 - gt_resized)/255.0 # Dark pixels become 1 in this plane
    #encoded_gt[:,:,blank_label] = (gt_resized)/255.0 # Label background as blank_label (otherwise, it is zero--a good label!)

    char_presence_histogram = np.zeros((len(char_to_idx)))
    region_bb = [[l, u], [r, b]]
    overlap_inclusion_threshold = 0.75
    char_count = 0
    if page is not None:
        for b, bbox in enumerate(page['bboxes']):
            if not bbox_overlaps(region_bb, bbox):
                continue
            if len(page['gt_as_wordlist']) <= b:
                continue
            word = page['gt_as_wordlist'][b]
            if len(page['bbox_charsegs'][b]) != len(word)-1:
                print("WARNING: Inconsistent charsegs and word length for bbox id", b, word)
                continue
            if len(word) == 1:
                pass
            startx = bbox[0][0]
            endx = bbox[1][0]
            lastx = startx
            for wi in range(len(word)):
                c = word[wi]
                # Replace "_" with " "
                c = c.replace("_", " ")
                nextx = startx + page['bbox_charsegs'][b][wi] if wi < len(page['bbox_charsegs'][b]) else endx
                cbox = [[lastx, bbox[0][1]], [nextx, bbox[1][1]]]
                if get_overlap_amount(region_bb,cbox) >= overlap_inclusion_threshold:
                    char_presence_histogram[char_to_idx[c]] = 1.0
                    char_count += 1

    #print(char_presence_histogram)

    return region, encoded_gt, char_presence_histogram

# Retrieves a training sample from an image with the given ground truth label and multichannel region mask
def get_masked_regionsampler_semanticseg_multichannel_withpairedregions(img, maskpixels, maskval=0, numclasses=4, label=0, minsize=28, maxsize=400, height=28, width=28, blank_label=-1, maskchannel=0, page=None):
    invalid = True
    tries = 0
    criterion = 0.01 #.03 # 1 is center pixel, <1 is proportion of "on" pixels
    if criterion == 1:
        while invalid:
            x = random.randint(0,img.shape[1]-1)
            y = random.randint(0,img.shape[0]-1)
            if maskpixels[y][x][maskchannel] != maskval:
                tries += 1
            else:
                invalid = False
                break
        size = random.randint(minsize,maxsize)
        l = max(0, x-size//2)
        r = min(img.shape[1], x+size//2)
        u = max(0, y-size//2)
        b = min(img.shape[0], y+size//2)
    else:
        while invalid:
            x = random.randint(0,img.shape[1]-1)
            y = random.randint(0,img.shape[0]-1)
            #print(minsize, maxsize)
            size = random.randint(minsize,maxsize)
            l = max(0, x-size//2)
            r = min(img.shape[1], x+size//2)
            u = max(0, y-size//2)
            b = min(img.shape[0], y+size//2)
            if maskchannel >= maskpixels.shape[2]: # Skip testing for blank; it's virtually always there.
                invalid = False
                break
            if np.mean(maskpixels[u:b,l:r,maskchannel]) < criterion:
                tries += 1
            else:
                invalid = False
                break
    #print(tries)

    #print(x,y,size,img.shape)
    pad_l = max(0, size//2-x)
    pad_r = max(0, size//2-(img.shape[1]-1-x))
    pad_u = max(0, size//2-y)
    pad_b = max(0, size//2-(img.shape[0]-1-y))
    #print(l,r,u,b, ":", pad_l, pad_r, pad_u, pad_b)
    region = np.zeros((size,size))
    region.fill(1.0)
    region[pad_u:pad_u+b-u,pad_l:pad_l+r-l] = img[u:b,l:r]
    region = cv2.resize(region, (width, height))
    gt_region = np.zeros((size,size,numclasses))
    #gt_region[pad_u:pad_u+b-u,pad_l:pad_l+r-l,label] = (255.0-maskpixels[u:b,l:r])/255.0
    gt_region[pad_u:pad_u+b-u,pad_l:pad_l+r-l,:maskpixels.shape[2]] = maskpixels[u:b,l:r]
    scc = maskpixels.shape[2]
    # Insert page channels here?
    encoded_gt = cv2.resize(gt_region, (width, height), interpolation=cv2.INTER_LINEAR)
    return region, encoded_gt

# Retrieves a training sample from an image with the given ground truth label and multichannel region mask
def get_masked_regionsampler_semanticseg_multichannel(img, maskpixels, maskval=0, numclasses=4, label=0, minsize=28, maxsize=400, height=28, width=28, blank_label=-1, maskchannel=0, page=None, background=0):
    invalid = True
    tries = 0
    criterion = 0.00005 #.03 # 1 is center pixel, <1 is proportion of "on" pixels
    if criterion == 1:
        while invalid:
            x = random.randint(0,img.shape[1]-1)
            y = random.randint(0,img.shape[0]-1)
            if maskpixels[y][x][maskchannel] != maskval:
                tries += 1
            else:
                invalid = False
                break
        size = random.randint(minsize,maxsize)
        l = max(0, x-size//2)
        r = min(img.shape[1], x+size//2)
        u = max(0, y-size//2)
        b = min(img.shape[0], y+size//2)
    else:
        if np.sum(maskpixels[:,:,maskchannel]) == 0:
            pass
            x = random.randint(0,img.shape[1]-1)
            y = random.randint(0,img.shape[0]-1)
            size = random.randint(minsize,maxsize)
            l = max(0, x-size//2)
            r = min(img.shape[1], x+size//2)
            u = max(0, y-size//2)
            b = min(img.shape[0], y+size//2)
        else:
            while invalid and tries < 50:
                x = random.randint(0,img.shape[1]-1)
                y = random.randint(0,img.shape[0]-1)
                if maskchannel == 4:
                    x = random.randint(int(img.shape[1]*.35),img.shape[1]-1)
                    y = random.randint(0,img.shape[0]//4-1)

                #print(minsize, maxsize)
                size = random.randint(minsize,maxsize)
                l = max(0, x-size//2)
                r = min(img.shape[1], x+size//2)
                u = max(0, y-size//2)
                b = min(img.shape[0], y+size//2)
                if maskchannel >= maskpixels.shape[2]: # Skip testing for blank; it's virtually always there.
                    invalid = False
                    break
                #if np.mean((255.0-maskpixels)[u:b,l:r])/255.0 < criterion: # TODO: Assumes maskval is zero
                if np.mean(maskpixels[u:b,l:r,maskchannel]) < criterion:
                    tries += 1
                else:
                    invalid = False
                    break
                #minsize = int(minsize*1.1)
                #maxsize = int(maxsize*1.1)
                #minsize += 20
                #maxsize += 20
                #minsize = maxsize -20
    #print("Tries:", tries)

    #print(x,y,size,img.shape)
    pad_l = max(0, size//2-x)
    pad_r = max(0, size//2-(img.shape[1]-1-x))
    pad_u = max(0, size//2-y)
    pad_b = max(0, size//2-(img.shape[0]-1-y))
    #print(l,r,u,b, ":", pad_l, pad_r, pad_u, pad_b)
    if len(img.shape) == 2:
        region = np.zeros((size,size))
    elif len(img.shape) == 3:
        region = np.zeros((size,size,img.shape[-1]))

    #region.fill(1.0)
    region.fill(np.random.random())
    if len(img.shape) == 2:
        region[pad_u:pad_u+b-u,pad_l:pad_l+r-l] = img[u:b,l:r]
    elif len(img.shape) == 3:
        region[pad_u:pad_u+b-u,pad_l:pad_l+r-l,:] = img[u:b,l:r,:]
    region = cv2.resize(region, (width, height))
    #encoded_gt = np.zeros(numclasses)
    #encoded_gt[label] = 1.0
    gt_region = np.zeros((size,size,numclasses))
    #gt_region[pad_u:pad_u+b-u,pad_l:pad_l+r-l,label] = (255.0-maskpixels[u:b,l:r])/255.0
    #print(maskpixels.shape, u, b, l, r, maskpixels)
    #import sys
    #sys.stdout.flush()
    gt_region[pad_u:pad_u+b-u,pad_l:pad_l+r-l,:maskpixels.shape[2]] = maskpixels[u:b,l:r]
    encoded_gt = cv2.resize(gt_region, (width, height), interpolation=cv2.INTER_LINEAR)
    #encoded_gt = np.concatenate((encoded_gt, np.zeros((height, width, 1))), axis=2)
    #encoded_gt[:,:,numclasses-1] = np.zeros((height, width))
    #encoded_gt = np.zeros([height, width, numclasses])
    #gt_resized = cv2.resize(maskpixels[u:b,l:r], (width, height))
    #encoded_gt[:,:,label] = (255.0 - gt_resized)/255.0 # Dark pixels become 1 in this plane
    #encoded_gt[:,:,blank_label] = (gt_resized)/255.0 # Label background as blank_label (otherwise, it is zero--a good label!)
    return region, encoded_gt

# Retrieves a training sample from an image with the given ground truth label and binary region mask
def get_masked_regionsampler_semanticseg(img, maskpixels, maskval=0, numclasses=4, label=0, minsize=8, maxsize=400, height=28, width=28, blank_label=3):
    invalid = True
    tries = 0
    criterion = 0.03 # 1 is center pixel, <1 is proportion of "on" pixels
    if criterion == 1:
        while invalid:
            x = random.randint(0,img.shape[1]-1)
            y = random.randint(0,img.shape[0]-1)
            if maskpixels[y][x] != maskval:
                tries += 1
            else:
                invalid = False
                break
        size = random.randint(minsize,maxsize)
        l = max(0, x-size//2)
        r = min(img.shape[1], x+size//2)
        u = max(0, y-size//2)
        b = min(img.shape[0], y+size//2)
    else:
        while invalid:
            x = random.randint(0,img.shape[1]-1)
            y = random.randint(0,img.shape[0]-1)
            size = random.randint(minsize,maxsize)
            l = max(0, x-size//2)
            r = min(img.shape[1], x+size//2)
            u = max(0, y-size//2)
            b = min(img.shape[0], y+size//2)
            if np.mean((255.0-maskpixels)[u:b,l:r])/255.0 < criterion: # TODO: Assumes maskval is zero
                tries += 1
            else:
                invalid = False
                break
    #print(tries)

    #print(x,y,size,img.shape)
    pad_l = max(0, size//2-x)
    pad_r = max(0, size//2-(img.shape[1]-1-x))
    pad_u = max(0, size//2-y)
    pad_b = max(0, size//2-(img.shape[0]-1-y))
    #print(l,r,u,b, ":", pad_l, pad_r, pad_u, pad_b)
    region = np.zeros((size,size))
    region.fill(255)
    region[pad_u:pad_u+b-u,pad_l:pad_l+r-l] = img[u:b,l:r]
    region = cv2.resize(region, (width, height))
    #encoded_gt = np.zeros(numclasses)
    #encoded_gt[label] = 1.0
    gt_region = np.zeros((size,size,numclasses))
    gt_region[pad_u:pad_u+b-u,pad_l:pad_l+r-l,label] = (255.0-maskpixels[u:b,l:r])/255.0
    encoded_gt = cv2.resize(gt_region, (width, height))
    #encoded_gt = np.zeros([height, width, numclasses])
    #gt_resized = cv2.resize(maskpixels[u:b,l:r], (width, height))
    #encoded_gt[:,:,label] = (255.0 - gt_resized)/255.0 # Dark pixels become 1 in this plane
    #encoded_gt[:,:,blank_label] = (gt_resized)/255.0 # Label background as blank_label (otherwise, it is zero--a good label!)
    return region, encoded_gt


# Retrieves a training sample from an image with the given ground truth label and binary region mask
def get_masked_regionsampler(img, maskpixels, maskval=0, numclasses=2, label=0, minsize=8, maxsize=400, height=28, width=28):
    invalid = True
    tries = 0
    # TODO: Multi-class region sampler will sum pixels in region to get total proposal score.
    criterion = 0.1 # 1 is center pixel, <1 is proportion of "on" pixels
    if criterion == 1:
        while invalid:
            x = random.randint(0,img.shape[1]-1)
            y = random.randint(0,img.shape[0]-1)
            if maskpixels[y][x] != maskval:
                tries += 1
            else:
                invalid = False
                break
        size = random.randint(minsize,maxsize)
        l = max(0, x-size//2)
        r = min(img.shape[1], x+size//2)
        u = max(0, y-size//2)
        b = min(img.shape[0], y+size//2)
    else:
        while invalid:
            x = random.randint(0,img.shape[1]-1)
            y = random.randint(0,img.shape[0]-1)
            size = random.randint(minsize,maxsize)
            l = max(0, x-size//2)
            r = min(img.shape[1], x+size//2)
            u = max(0, y-size//2)
            b = min(img.shape[0], y+size//2)
            if np.mean((255.0-maskpixels)[u:b,l:r])/255.0 < criterion: # TODO: Assumes maskval is zero
                tries += 1
            else:
                invalid = False
                break
    #print(tries)
    size = random.randint(minsize,maxsize)
    l = max(0, x-size//2)
    r = min(img.shape[1], x+size//2)
    u = max(0, y-size//2)
    b = min(img.shape[0], y+size//2)
    #print(x,y,size,img.shape)
    pad_l = max(0, size//2-x)
    pad_r = max(0, size//2-(img.shape[1]-1-x))
    pad_u = max(0, size//2-y)
    pad_b = max(0, size//2-(img.shape[0]-1-y))
    #print(l,r,u,b, ":", pad_l, pad_r, pad_u, pad_b)
    region = np.zeros((size,size))
    region.fill(255)
    region[pad_u:pad_u+b-u,pad_l:pad_l+r-l] = img[u:b,l:r]
    region = cv2.resize(region, (width, height))
    encoded_gt = np.zeros(numclasses)
    encoded_gt[label] = 1.0
    return region, encoded_gt

def get_word_with_charsegs(page, min_segs=0):
    invalid = True
    max_tries = 10000
    tries = 0
    # TODO: Cache words with character segmentations in page data (dictionary)!
    while invalid and tries < max_tries:
        wordnum = random.randint(0, len(page['gt_as_wordlist']))
        if len(page['bbox_charsegs']) <= wordnum or len(page['bbox_charsegs'][wordnum])+1 < len(page['gt_as_wordlist'][wordnum]) or len(page['bbox_charsegs'][wordnum]) < min_segs or len(page['gt_as_wordlist'][wordnum]) < min_segs:
            #print(tries, wordnum, len(page['bbox_charsegs']))
            if len(page['bbox_charsegs']) > wordnum:
                pass
                #print(len(page['bbox_charsegs'][wordnum]), len(page['gt_as_wordlist'][wordnum]), min_segs)
            invalid = True
            tries += 1
        else:
            #print(tries, wordnum, len(page['bbox_charsegs']))
            if len(page['bbox_charsegs']) > wordnum:
                pass
                #print(len(page['bbox_charsegs'][wordnum]), len(page['gt_as_wordlist'][wordnum]), min_segs)
            gt = page['gt_as_wordlist'][wordnum]
            invalid = False
            break
    return wordnum

def scale_pad_and_center(sample, height=28, width=28, pad_value=0, do_center=True):
    #print(sample.shape)
    scalefactor = min(float(height)/float(sample.shape[0]), float(width)/float(sample.shape[1]))
    sample = cv2.resize(sample, (int(sample.shape[1] * scalefactor), int(sample.shape[0] * scalefactor)))
    #print(sample.shape)
    sample = pad_and_center(sample, height, width, do_center=do_center)
    return sample

def pad_and_center(sample, height=28, width=28, pad_value=0, mode='constant', do_center=True): #mean'): #'linear_ramp'): #'edge'
    pad_w = width-sample.shape[1]
    pad_h = height-sample.shape[0]
    pad_t = pad_h//2
    pad_b = pad_h - pad_t
    pad_l = pad_w / 2
    pad_r = pad_w - pad_l
    #print("Min/Max", np.min(sample), np.max(sample))
    mean_value = np.mean(sample)
    if not do_center:
        pad_r = pad_l + pad_r
        pad_b = pad_t + pad_b
        pad_l = 0
        pad_t = 0
    sample = np.pad(sample, ((pad_t,pad_b), (pad_l, pad_r)), mode=mode, constant_values=mean_value) #, end_values=mean_value) # TODO: pad_value is not used
    #print(sample.shape, pad_w, pad_h)
    #print("Min/Max", np.min(sample), np.max(sample))
    #sample = np.roll(sample, pad_h/2, axis=0)
    #sample = np.roll(sample, pad_w/2, axis=1)
    #cv2.imshow('Sample Orig', sample)
    #cv2.waitKey(1000)
    #print(sample.shape)
    return sample

if False:
    import string_utils
    def char_sample_generator(img, page, height=28, width=28, char_to_idx={}, extra_classes=7):
        wordnum = get_word_with_charsegs(page)
        gt = page['gt_as_wordlist'][wordnum]
        bbox = page['bboxes'][wordnum]
        charsegs = page['bbox_charsegs'][wordnum]
        charnum = random.randint(0, len(charsegs)-1) if len(charsegs) > 1 else 0
        left = charsegs[charnum-1] + bbox[0][0] if charnum > 0 else bbox[0][0]
        right = charsegs[charnum] + bbox[0][0] if charnum < len(charsegs) else bbox[1][0]
        sample = img[bbox[0][1]:bbox[1][1],left:right]
        #cv2.imshow('Sample Orig', sample)
        #cv2.waitKey(1000)
        sample = scale_pad_and_center(sample, height, width)
        gt = gt[charnum]

        #encoded_gt = string_utils.str2label_single(gt, char_to_idx)
        #print("Seth", len(char_to_idx), gt, encoded_gt.shape)
        #encoded_gt = np.pad(encoded_gt, (0,extra_classes), mode='constant', constant_values=0)
        #print("Seth", len(char_to_idx), gt, encoded_gt.shape)
        # Append character density map here!
        encoded_gt = np.zeros(len(char_to_idx)+extra_classes)
        encoded_gt[char_to_idx[gt]] = 1.0

        return sample, encoded_gt

    def interchar_sample_generator(img, page, height=28, width=28, char_to_idx={}, extra_classes=7):
        wordnum = get_word_with_charsegs(page, 1)
        gt = page['gt_as_wordlist'][wordnum]
        bbox = page['bboxes'][wordnum]
        charsegs = page['bbox_charsegs'][wordnum]
        segnum = random.randint(0, len(charsegs)-1) if len(charsegs) > 1 else 0
        #print(segnum, len(charsegs))
        left = max(0, charsegs[segnum] + bbox[0][0] - width//2)
        right = min(img.shape[1]-1, charsegs[segnum] + bbox[0][0] + width//2)
        sample = img[bbox[0][1]:bbox[1][1],left:right]
        sample = scale_pad_and_center(sample, height, width)
        gt = ""
        encoded_gt = np.zeros(len(char_to_idx)+extra_classes)
        encoded_gt[len(char_to_idx)] = 1.0 # Set class to interchar
        # Append character density map here!
        return sample, encoded_gt

    def interword_sample_generator(img, page, height=28, width=28, char_to_idx={}, extra_classes=7):
        wordnum = get_word_with_charsegs(page, 1)
        gt = page['gt_as_wordlist'][wordnum]
        bbox = page['bboxes'][wordnum]
        charsegs = page['bbox_charsegs'][wordnum]
        segnum = random.randint(0, len(charsegs)-1) if len(charsegs) > 1 else 0
        left = max(0, charsegs[segnum] + bbox[0][0] - width//2)
        right = min(img.shape[1]-1, charsegs[segnum] + bbox[0][0] + width//2)
        sample = img[bbox[0][1]:bbox[1][1],left:right]
        sample = scale_pad_and_center(sample, height, width)
        gt = ""
        encoded_gt = np.zeros(len(char_to_idx)+extra_classes)
        encoded_gt[len(char_to_idx)] = 1.0 # Set class to interchar
        # Append character density map here!
        return sample, encoded_gt

    def multichar_sample_generator(img, page, height=28, width=28, char_to_idx={}, extra_classes=7):
        wordnum = get_word_with_charsegs(page, 1)
        gt = page['gt_as_wordlist'][wordnum]
        bbox = page['bboxes'][wordnum]
        charsegs = page['bbox_charsegs'][wordnum]
        segnum = random.randint(0, len(charsegs)-1) if len(charsegs) > 1 else 0
        left = bbox[0][0] #max(0, charsegs[segnum] + bbox[0][0] - width//2)
        right = bbox[1][0]#min(img.shape[1]-1, charsegs[segnum] + bbox[0][0] - width//2)
        sample = img[bbox[0][1]:bbox[1][1],left:right]
        sample = scale_pad_and_center(sample, height, width)
        encoded_gt = np.zeros(len(char_to_idx)+extra_classes)
        encoded_gt[len(char_to_idx)+1] = 1.0 # Set class to multichar
        #encoded_gt = string_utils.str2label_single(gt, char_to_idx)
        # Append character density map here!
        return sample, encoded_gt

    def interline_sample_generator(img, page, height=28, width=28, char_to_idx={}, extra_classes=7):
        wordnum = get_word_with_charsegs(page, 0)
        gt = page['gt_as_wordlist'][wordnum]
        bbox = page['bboxes'][wordnum]
        charsegs = page['bbox_charsegs'][wordnum]
        segnum = random.randint(0, len(charsegs)-1) if len(charsegs) > 1 else 0
        left = bbox[0][0] #max(0, charsegs[segnum] + bbox[0][0] - width//2)
        right = bbox[1][0]#min(img.shape[1]-1, charsegs[segnum] + bbox[0][0] - width//2)
        offset = random.randint(height//4,height//2+1)
        direct = -1 if random.random() < 0.5 else 1
        top = max(0, bbox[0][1]+offset*direct)
        bottom = max(1, bbox[1][1]+offset*direct)
        sample = img[top:bottom,left:right]
        #print(sample.shape)
        sample = scale_pad_and_center(sample, height, width)
        #print(sample.shape)
        encoded_gt = np.zeros(len(char_to_idx)+extra_classes)
        encoded_gt[len(char_to_idx)+2] = 1.0 # Set class to interline
        #encoded_gt = string_utils.str2label_single(gt, char_to_idx)
        # Append character density map here!
        return sample, encoded_gt

    def multiline_sample_generator(img, page, height=28, width=28, char_to_idx={}, extra_classes=7):
        # Sample lots of bboxes until one spanning multiple lines is found
        captured = False
        tries = 0
        max_tries = 100
        avg_height = 0
        num_bboxes = 10
        for i in range(num_bboxes):
            bbox = random.choice(page['bboxes'])
            avg_height += (bbox[1][1] - bbox[0][1])
        avg_height /= num_bboxes
        minheight = int(avg_height * 2)
        maxheight = img.shape[1]
        ht = random.randint(minheight, maxheight)
        while not captured and tries < max_tries:
            left = random.randint(0, max(1, img.shape[1]-ht))
            top = random.randint(0, max(1, img.shape[0]-ht))
            bottom = min(top + ht, img.shape[0]-1)
            right = min(left + ht, img.shape[1]-1)
            bottomRow = -1
            for bbox in page['bboxes']:
                brow = bbox[1][1]
                if bottomRow == -1:
                    bottomRow = brow
                if bottomRow != -1 and brow - bottomRow > avg_height:
                    captured = True # Multiline criterion satisfied!!!
                    break
            tries += 1
        sample = img[top:bottom,left:right]
        sample = scale_pad_and_center(sample, height, width)
        encoded_gt = np.zeros(len(char_to_idx)+extra_classes)
        encoded_gt[len(char_to_idx)+3] = 1.0 # Set class to multiline
        #encoded_gt = string_utils.str2label_single(gt, char_to_idx)
        # Append character density map here!
        return sample, encoded_gt

    def blank_sample_generator(img, page, height=28, width=28, char_to_idx={}, extra_classes=7):
        # Sample lots of bboxes until one spanning multiple lines is found
        captured = False
        tries = 0
        max_tries = 100
        avg_height = 0
        num_bboxes = 10
        for i in range(num_bboxes):
            bbox = random.choice(page['bboxes'])
            avg_height += (bbox[1][1] - bbox[0][1])
        avg_height /= num_bboxes
        minheight = 2
        maxheight = int(avg_height)
        ht = random.randint(minheight, maxheight)
        while not captured and tries < max_tries:
            left = random.randint(0, max(1, img.shape[1]-ht))
            top = random.randint(0, max(1, img.shape[0]-ht))
            bottom = min(top + ht, img.shape[0]-1)
            right = min(left + ht, img.shape[1]-1)
            overbox = [[left,top],[right,bottom]]
            bottomRow = -1
            inboxes = False
            for bbox in page['bboxes']:
                if boxes_overlap(overbox, bbox):
                    inboxes = True
                    break
            if not inboxes:
                captured = True
            tries += 1
        sample = img[top:bottom,left:right]
        sample = scale_pad_and_center(sample, height, width)
        encoded_gt = np.zeros(len(char_to_idx)+extra_classes)
        encoded_gt[len(char_to_idx)+5] = 1.0 # Set class to blank
        #encoded_gt = string_utils.str2label_single(gt, char_to_idx)
        # Append character density map here!
        return sample, encoded_gt

    def subchar_sample_generator(img, page, height=28, width=28, char_to_idx={}, extra_classes=7):
        wordnum = get_word_with_charsegs(page)
        gt = page['gt_as_wordlist'][wordnum]
        bbox = page['bboxes'][wordnum]
        charsegs = page['bbox_charsegs'][wordnum]
        charnum = random.randint(0, len(charsegs)-1) if len(charsegs) > 1 else 0
        #print(charsegs)
        left = charsegs[charnum-1] + bbox[0][0] if charnum > 0 else bbox[0][0]
        right = charsegs[charnum] + bbox[0][0] if charnum < len(charsegs) else bbox[1][0]
        w = random.randint(5, max(6, right-left))
        offx = random.randint(0, max(1,bbox[1][0]-bbox[0][0]-w))
        offy = random.randint(0, max(1,bbox[1][1]-bbox[0][1]-w))
        left = min(left + offx, img.shape[1]-2)
        right = min(left + offx + w, img.shape[1]-1)
        top = min(bbox[0][1] + offy, img.shape[0]-2)
        bottom = min(bbox[0][1] + offy+w, img.shape[0]-1)
        sample = img[top:bottom,left:right]
        #print(sample.shape)
        sample = scale_pad_and_center(sample, height, width)
        gt = gt[charnum]
        encoded_gt = np.zeros(len(char_to_idx)+extra_classes)
        encoded_gt[len(char_to_idx)+4] = 1.0 # Set class to blank
        #encoded_gt = string_utils.str2label_single(gt, char_to_idx)
        # Append character density map here!
        return sample, encoded_gt

def inbox(bbox, pt):
    return pt[0] >= bbox[0][0] and pt[0] <= bbox[1][0] and pt[1] >= bbox[0][1] and pt[1] <= bbox[1][1]

def boxcorners_inbox(box1, box2):
    return inbox(box1, box2[0]) or inbox(box1, box2[1]) or inbox(box1, [box2[0][0],box2[1][0]]) or inbox(box1, [box2[0][0],box2[1][0]])

def boxes_overlap(box1, box2):
    return boxcorners_inbox(box1, box2) or boxcorners_inbox(box2, box1)

def image_sample_generator(img, page, height=28, width=28, char_to_idx={}, extra_classes=7):
    pass

def get_batch(self, batch_size=64, input_height=28, input_width=28):


      # Create an image, GT batch from available transcriptions.
      # Automatically use random-length subwords
      batch_imgs = []
      batch_gts = []
      # TODO: Split into train/test!
      # TODO: Allow to use from all pages..
      page_img = cv2.imread(self.page['image_path'], 0)
      maxlengt = 0
      for b in range(batch_size):
          invalid = True
          max_tries = 10
          tries = 0
          while invalid and tries < max_tries:
              wordnum = random.randint(0, len(self.page['gt_as_wordlist']))
              if len(self.page['bbox_charsegs']) >= wordnum or len(self.page['bbox_charsegs'][wordnum]) < len(self.page['gt_as_wordlist'][wordnum]):
                  invalid = True
                  tries += 1
              else:
                  gt = self.page['gt_as_wordlist'][wordnum]
                  invalid = False
                  break

          # Pick a random subword from the image to train on
          gt_start_ind = random.randint(0, len(gt))
          gt_end_ind = random.randint(gt_start_ind+1, len(gt)+1)
          bbox = self.page['bbox'][wordnum]
          charsegs = self.page['bbox_charsegs'][wordnum]
          startx = charsegs[gt_start_ind-1] if gt_start_ind > 0 else 0
          endx = charsegs[gt_end_ind-1] if gt_end_ind < len(gt) else bbox[1][0]-bbox[0][0]
          train_img = page_img[bbox[0][1]:bbox[1][1],bbox[0][0]+startx:bbox[1][0]+endx]
          gt = gt[gt_start_ind:gt_end_ind]
          if len(gt) > maxlengt:
              maxlengt = len(gt)
          batch_imgs.append(train_img)
          batch_gts.append(gt)

      # PREPROCESS the batch here too!
      np_batch_imgs = np.zeros((batch_size, input_height, input_width))
      np_batch_gts = np.zeros((batch_size, maxlengt))
      for b, img in batch_imgs:
          #
          newsize = (img.shape[1]*input_height/img.shape[0], input_height)
          img = cv2.resize(img, newsize)
          if img.shape[1] > input_width:
              img = img[:,:input_width]
          img = np.pad(img, (input_height,input_width), axis=1)
          gt = batch_gts[b]
          encoded_gt = string_utils.str2label_single(gt, self.char_set)
          #print(encoded_gt.shape)
          np_batch_imgs[b] = img
          np_batch_gts[b] = np.pad(encoded_gt, (batch_size, maxlengt, len(self.char_set)))
      return np_batch_imgs, np_batch_gts
