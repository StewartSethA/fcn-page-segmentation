import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import operator
import scipy.signal
import math
import imutils
from sklearn.mixture import GMM

def template_match(img, template):
    template = template.astype('float')
    template -= np.mean(template)
    template /= 255.0
    template /= (template.shape[0]*template.shape[1])
    template = np.fliplr(template)
    template = np.flipud(template)
    convolved = scipy.signal.fftconvolve(img, template, mode='same')

    disp = cv2.resize(convolved, (640, convolved.shape[0]*640/convolved.shape[1]))
    disp = ((disp - (np.min(disp))) / (np.max(disp) - np.min(disp))) * 1.0
    cv2.imshow('conv', disp)
    cv2.waitKey()

    #disp = cv2.resize(convolved, (640, convolved.shape[0]*convolved.shape[1]/640))
    #disp = ((disp - (np.min(convolved))) / (np.max(convolved) - np.min(convolved))) * 255
    #disp = disp.astype('uint8')
    #cv2.imshow('conv', disp)
    #cv2.waitKey()

    #print convolved
    maxpt = np.argmax(convolved, 0)
    maxpty = np.argmax(convolved, 1)
    pt = np.unravel_index(convolved.argmax(), convolved.shape)

    ul = (pt[1] - im2.shape[1]/2, pt[0] - im2.shape[0]/2)
    br = (pt[1] + im2.shape[1]/2, pt[0] + im2.shape[0]/2)

    print ul, br

# Returns a grayscale version of a numpy image.
def gray(img):
    if len(img.shape) > 2:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img
    return gray_image

# Returns a Sobel edge version of a numpy image.
def derivative(img):
    img = cv2.Sobel(img, ddepth=-1, dx=1, dy=1)
    return img

# Returns a profile of pixel variance across the given axis of a numpy image.
def variance_profile(img, axis=1):
    prof = np.var(img, axis=axis)
    return prof

# Returns a profile of pixel averages across the given axis of a numpy image.
def profile(img, axis=1):
    prof = np.mean(img, axis=axis)
    if len(prof.shape) == 2:
        prof = np.mean(prof, axis=1)
    return prof

def overlay_transparency(img, overlay, opacity):
    imgcopy = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(overlay, opacity, imgcopy, 1-opacity, 0, imgcopy)
    return img

# Performs frequency domain filtering on a signal using an FT filter.
def fftfilt(array, ftfilt):
    ft = np.fft.fft(array)
    ft *= ftfilt
    ift = np.fft.ifft(ft)
    return ift

# Returns a numpy image rotated by the given number of degrees.
def rotate(img, angle, borderMode=cv2.BORDER_REPLICATE, borderValue=255, expand=False):
    a = angle
    p = (img.shape[1]/2, img.shape[0]/2)
    M = cv2.getRotationMatrix2D(p, a, 1)
    newH,newW = img.shape[0:2]
    if expand:
        h,w = img.shape[0:2]
        r = np.deg2rad(a)
        newW,newH = (abs(np.sin(r)*h) + abs(np.cos(r)*w),abs(np.sin(r)*w) + abs(np.cos(r)*h))
        (tx,ty) = ((newW-w)/2, (newH-h)/2)
        M[0,2] += tx
        M[1,2] += ty
    rot = cv2.warpAffine(img, M, (int(newW), int(newH)), borderMode=borderMode, borderValue=borderValue)
    return rot

# Constructs a numpy image pyramid with minimum width and height constraints.
# flat=True makes it a pass-through routine.
def build_pyramid(img, min_height=160, min_width=160, flat=False, debug=False):
    cursize=[img.shape[1], img.shape[0]]
    sizes = []
    if flat:
        sizes.append(tuple([img.shape[1], img.shape[0]]))
        yield img
    else:
        while cursize[0] > min_height or cursize[1] > min_width:
            cursize[0] /= 2
            cursize[1] /= 2
            sizes.insert(0, tuple(cursize))
        sizes.append(tuple([img.shape[1], img.shape[0]]))
        for size in sizes:
            if debug:
                print size
            yield cv2.resize(img, dsize=size)

# Shrinks a numpy image to the stricter of the width and height constraints.
def shrink(img, minWidth=2000, minHeight=1600):
    newshape = (min(minWidth, img.shape[1]), min(minHeight, img.shape[0]))
    newshape = (min(newshape[0], int(float(img.shape[1])*newshape[1]/img.shape[0])), min(newshape[1], int(float(img.shape[0])*newshape[0]/img.shape[1])))
    img = cv2.resize(img, newshape)
    return img

# Subtracts the background from a numpy image using the given kernel width.
def bg_subtract(img, radius=100, disp_wait=0):
    if radius % 2 == 0:
        radius += 1
    sub = cv2.medianBlur(img, radius)
    if disp_wait > 0:
        cv2.imshow('median', shrink(sub,2000))
        cv2.waitKey(disp_wait)
    return subtract(img, sub)

# Subtracts the background from a numpy image using bilateral filtering. WARNING: SLOW!
def bg_subtract_bilat(img, radius=100, colorDiff=50, disp_wait=0):
    bilat = cv2.bilateralFilter(img,colorDiff,radius,radius)
    if disp_wait > 0:
        cv2.imshow('bilat', shrink(bilat,2000))
        cv2.waitKey(disp_wait)
    return subtract(img, bilat)

# Subtracts a numpy image from another, optionally normalizing and trimming negatives
def subtract(img, sub, clipNegative=True, scale=True, asint=True):
    img = -img.astype('float')+sub.astype('float')
    if clipNegative:
        img[img<0] = 0
    if scale:
        img = ((img-img.min())/(img.max()-img.min())*255)
    if asint:
        img = img.astype(np.uint8)
    return img

# Computes the local window variance of a numpy image using convolution.
# Code from http://stackoverflow.com/questions/15361595/calculating-variance-image-python
def variance_image(img, radius):
    wmean, wsqrmean = (cv2.boxFilter(img, -1, (radius, radius), borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
    return wsqrmean - wmean*wmean

# TODO Helper function for bidirectional_variance_image
def bidirectional_variance(prof0, prof1, x, y, radius):
    var1 = np.var(prof1[max(0,x-radius):min(len(prof1)-1, x+radius)])
    var0 = np.var(prof0[max(0,y-radius):min(len(prof0)-1, y+radius)])
    return var0*var1

# TODO Bidirectional Variance Image
# WIP to compute local bidirectional variance, that is, a measure of variance similar to the
# variance_image method above but that is low or zero when either of the directional profile variances
# within the given window is low or zero.
# This should enable robustness to horizontal or vertical lines.
def bidirectional_variance_image(img, radius):
    prof1 = profile(img, 1)
    prof0 = profile(img, 0)
    #var_img = np.zeros(img.shape)
    '''
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            var1 = np.var(prof1[max(0,x-radius):min(img.shape[1]-1, x+radius)])
            var0 = np.var(prof0[max(0,y-radius):min(img.shape[0]-1, y+radius)])
            var_img[y][x] = var1*var0
    '''
    #vbv = np.vectorize(bidirectional_variance)
    #var_img = vbv(prof0, prof1, x, y, radius)
    return var_img

# TODO Computes the integral image from a numpy image.
def integral_image(img):
    pass

############################ CONVENIENCE COMMANDS #############################
# Parse the numbered command line argument with default
def parse_cmd(num, default):
    global cmd_num_internal_9999
    cmd_num_internal_9999 += 1
    return sys.argv[num] if len(sys.argv) > num else default

cmd_num_internal_9999 = 1
def parse_next(default):
    return parse_cmd(cmd_num_internal_9999, default)

def set_parse_num(num):
    global cmd_num_internal_9999
    cmd_num_internal_9999 = 0

# Helper function to color exception stack traces:
def myexcepthook(type, value, tb):
    import traceback
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import TerminalFormatter

    tbtext = ''.join(traceback.format_exception(type, value, tb))
    lexer = get_lexer_by_name("pytb", stripall=True)
    formatter = TerminalFormatter()
    sys.stderr.write(highlight(tbtext, lexer, formatter))

sys.excepthook = myexcepthook
