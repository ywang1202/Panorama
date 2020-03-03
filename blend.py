import math
import sys

import cv2
import numpy as np


class ImageInfo():
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         maxX: int for the maximum X value of a corner
         maxY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    height, width = img.shape[0], img.shape[1]

    p1 = M.dot(np.array([0, 0, 1]))
    p2 = M.dot(np.array([0, height - 1, 1]))
    p3 = M.dot(np.array([width - 1, 0, 1]))
    p4 = M.dot(np.array([width - 1, height - 1, 1]))

    x1 = p1[0] / p1[2]
    y1 = p1[1] / p1[2]
    x2 = p2[0] / p2[2]
    y2 = p2[1] / p2[2]
    x3 = p3[0] / p3[2]
    y3 = p3[1] / p3[2]
    x4 = p4[0] / p4[2]
    y4 = p4[1] / p4[2]

    minX = int(min(x1, x2, x3, x4))
    minY = int(min(y1, y2, y3, y4))
    maxX = int(max(x1, x2, x3, x4))
    maxY = int(max(y1, y2, y3, y4))
    #TODO-BLOCK-END
    return minX, minY, maxX, maxY


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN
    (img_height, img_width, img_channel) = img.shape
    (acc_height, acc_width, acc_channel) = acc.shape
    inv_M = np.linalg.inv(M)

    (minX, minY, maxX, maxY) = imageBoundingBox(img, M)

    for h in range(minY, maxY):
        for w in range(minX, maxX):
            pixel = np.array([w, h, 1]).reshape(3, 1)
            src_pixel = inv_M.dot(pixel)

            x = float(src_pixel[0]) / float(src_pixel[2])
            y = float(src_pixel[1]) / float(src_pixel[2])

            if (x < 0 or x > img_width - 1 or y < 0 or y > img_height - 1):
                continue

            x_floor = int(np.floor(x))
            x_ceil = int(np.ceil(x))
            y_floor = int(np.floor(y))
            y_ceil = int(np.ceil(y))

            if ((img[y_floor, x_floor, 0] == 0 and img[y_floor, x_floor, 1] == 0 and img[y_floor, x_floor, 2] == 0)
                    or (img[y_floor, x_ceil, 0] == 0 and img[y_floor, x_ceil, 1] == 0 and img[y_floor, x_ceil, 2] == 0)
                    or (img[y_ceil, x_ceil, 0] == 0 and img[y_ceil, x_ceil, 1] == 0 and img[y_ceil, x_ceil, 2] == 0)
                    or (img[y_ceil, x_floor, 0] == 0 and img[y_ceil, x_floor, 1] == 0 and img[
                        y_ceil, x_floor, 2] == 0)):
                continue

            if (blendWidth > img_width / 2.0):
                blendWidth = img_width / 2.0

            loc = min(w - minX, maxX - w)
            alpha = 1.0
            if (loc < blendWidth):
                alpha = float(loc) / float(blendWidth)

            if (x % 1.0 == 0.0 and y % 1.0 == 0.0):
                for c in range(3):
                    acc[h, w, c] += alpha * img[int(y), int(x), c]
            elif (x % 1.0 == 0.0 and y % 1.0 != 0.0):
                c1 = (1.0 - (y - float(y_floor)))
                c2 = (1.0 - (float(y_ceil) - y))
                for c in range(3):
                    acc[h, w, c] += alpha * (c1 * img[y_floor, int(x), c]
                                             + c2 * img[y_ceil, int(x), c])
            elif (x % 1.0 != 0.0 and y % 1.0 == 0.0):
                c1 = (1.0 - (x - float(x_floor)))
                c2 = (1.0 - (float(x_ceil) - x))
                for c in range(3):
                    acc[h, w, c] += alpha * (c1 * img[int(y), x_floor, c]
                                             + c2 * img[int(y), x_ceil, c])
            else:
                c1 = (1.0 - (x - float(x_floor))) * (1.0 - (y - float(y_floor)))
                c2 = (1.0 - (float(x_ceil) - x)) * (1.0 - (y - float(y_floor)))
                c3 = (1.0 - (float(x_ceil) - x)) * (1.0 - (float(y_ceil) - y))
                c4 = (1.0 - (x - float(x_floor))) * (1.0 - (float(y_ceil) - y))
                for c in range(3):
                    acc[h, w, c] += alpha * ((c1 * img[y_floor, x_floor, c] + c2 * img[y_floor, x_ceil, c]
                                              + c3 * img[y_ceil, x_ceil, c] + c4 * img[y_ceil, x_floor, c]))

            acc[h, w, 3] += alpha
    #TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    height, width = acc.shape[0], acc.shape[1]

    img = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            weights = acc[i, j, 3]
            if weights > 0:
                img[i, j, :] = acc[i, j, :3] / weights

    img = np.uint8(img)
    #TODO-BLOCK-END
    # END TODO
    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accHeight: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = sys.maxsize
    minY = sys.maxsize
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        new_minX, new_minY, new_maxX, new_maxY = imageBoundingBox(img, M)
        minX = min(minX, new_minX)
        minY = min(minY, new_minY)
        maxX = max(maxX, new_maxX)
        maxY = max(maxY, new_maxY)
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN
    if is360:
        A = computeDrift(x_init, y_init, x_final, y_final, width)
    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

