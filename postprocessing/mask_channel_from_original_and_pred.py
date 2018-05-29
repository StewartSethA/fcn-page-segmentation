from __future__ import print_function
import numpy as np
import cv2

# TODO: Merge!
def write_masked_channels(img, preds, image_path):
    predmax = np.max(preds)
    imgmax = np.max(img)
    print("Max pred:", predmax, "Max img:", imgmax)
    predflt = preds.astype('float32')
    imgflt = img.astype('float32')
    if predmax > 1:
        predflt /= predmax
    if imgmax > 1:
        imgflt /= imgmax
    for l in range(preds.shape[-1]):
        layer = predflt[:,:,l]
        masked = 255*(np.multiply(layer, imgflt) + 1.0-layer)
        mip = image_path + "_" + str(l) + "_masked.jpg"
        cv2.imwrite(mip, masked)
        print("Wrote masked image to", mip)


def mask_image(originalfilepath, channelfilepath):
    original = cv2.imread(originalfilepath, 0).astype('float32')/255.0
    channel = cv2.imread(channelfilepath, 0).astype('float32')/255.0

    maskedimage = np.ones(original.shape).astype('float32')

    maskedimage = (1.0-channel) * maskedimage + (channel) * original
    return maskedimage
    
if __name__ == "__main__":
    import os,sys
    originalfilepath = sys.argv[1]
    channelfilepath = sys.argv[2]
    outputfilepath = sys.argv[3]
    maskedimage = mask_image(originalfilepath, channelfilepath)
    cv2.imwrite(outputfilepath, (maskedimage*255.0).astype('uint8'))
