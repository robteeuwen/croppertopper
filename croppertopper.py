import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import json

# settings
with open('settings.json') as json_file:
    settings = json.load(json_file)
    xsize = settings['xsize']
    ysize = settings['ysize']
    topDistance = settings['topDistance']
    bottomDistance = settings['bottomDistance']
    rawloc = settings['rawloc']
    foldername = settings['foldername']
    imageQuality = settings['imageQuality']

printOutput = 0
plotSteps = 0

# get a list of photos
files = os.listdir(rawloc)

# make the writing directory
writeloc = rawloc + foldername + '/'
os.mkdir(writeloc)

def analyse_image(file,rawloc,xsize,ysize,topDistance,bottomDistance,printOutput = 0,plotSteps = 0):
    # load image
    img = cv2.imread(rawloc + file, -1)
    imgcp = img.copy()

    # resize image
    imgcp = cv2.resize(imgcp,None,fx=0.2,fy=0.2)
    if printOutput:
        print(f'new shape: {imgcp.shape}, original shape: {img.shape}')

    # detect edges
    edges = cv2.Canny(imgcp,20,100)
    if printOutput:
        print(f'shape of edges: {edges.shape}')

    # close
    kernelSize = 30
    kernel = np.ones((kernelSize,kernelSize),np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # open
    kernelSize = 30
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # find incides that are bigger than zero, and create a binary silhouette matrix
    sil = np.zeros_like(closed)
    sil[np.where(closed>0)] = 1

    # plot the image
    if plotSteps:
        plt.imshow(sil, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

    # ----------------------------------
    # find bounds of the subject as a ratio of the original picture's height
    # ----------------------------------
    imsize = edges.shape
    imheight = imsize[0]
    imwidth = imsize[1]

    hor = np.where(np.sum(sil,0)>0)
    ver = np.where(np.sum(sil,1)>0)
    left = hor[0][0] / imwidth
    right = hor[0][-1] / imwidth
    top = ver[0][0] / imheight
    bottom = ver[0][-1] / imheight
    if printOutput:
        print(f'left bound: {left} right bound: {right} top: {top} bottom {bottom}')

    # ----------------------------------
    # also find the center of the subject
    # ----------------------------------
    # loop rows of the image and find the center of the subject
    cent = []
    for row in range(sil.shape[0]):
        currentrow = sil[row,:]
        nonzero = np.where(currentrow>0)
        if len(nonzero[0]):
            cent.append(np.mean(np.where(currentrow>0)))
    center = np.mean(cent)
    centerRatio = center/imwidth

    # ----------------------------------
    # calculate the height in pixels of the model and the target height in percent
    modelheight = (bottom-top) * img.shape[0] # nr of pixels that the model is tall in the original photo
    modelTargetHeightRatio = 1-topDistance-bottomDistance # ratio that the model is tall in the final photo

    # calculate where the vertical crop should go
    cropHeight = round(modelheight/modelTargetHeightRatio) # nr of pixels that we need to cut out of the original

    # calculate how many pixels the model's head should start from the final photo
    modelTopTarget = round(topDistance * cropHeight)
    modelBottomTarget = round(bottomDistance * cropHeight)

    # calculate how many pixels from the top of the original photo the model starts
    modelTopOriginal = round(top * img.shape[0])
    modelBottomOriginal = round(bottom * img.shape[0])

    # calculate target top crop location
    cropTop = modelTopOriginal - modelTopTarget
    cropBottom = modelBottomOriginal + modelBottomTarget

    # cropTop and cropBottom should never extend beyond the range of the image, so let's make sure they don't
    cropTop = cropTop if cropTop >= 0 else 0
    cropBottom = cropBottom if cropBottom <= img.shape[0] else img.shape[0]

    # ----------------------------------
    # now figure out the sides
    if 0:
        # center the model by equalizing the whitespace
        leftspace = left * img.shape[1]
        rightspace = img.shape[1] - right * img.shape[1]

        # crop
        if leftspace > rightspace:
            horOffset = int(round(leftspace - rightspace))
            cropped = img[int(cropTop):int(cropBottom), horOffset:, :]
        else:
            horOffset = int(round(rightspace - leftspace))
            cropped = img[int(cropTop):int(cropBottom), :-horOffset, :]
    else:
        # center the model by the center of gravity
        centerPoint = img.shape[1] * centerRatio # center of the model
        centerOffset = img.shape[1]/2 - centerPoint

        if centerOffset < 0:
            cropped = img[int(cropTop):int(cropBottom), int(centerOffset*-1):, :]
        else:
            cropped = img[int(cropTop):int(cropBottom), :-int(centerOffset), :]

    if plotSteps:
        plt.imshow(cropped, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

    # ----------------------------------
    # now crop again, to get the ratio correct
    targetRatio = ysize/xsize
    currentRatio = cropped.shape[0]/cropped.shape[1]

    if printOutput:
        print(f'currentratio {currentRatio} targetrratio {targetRatio}' )

    # if the currentRatio is smaller, that means the current width is too large, which means we can just cut
    if currentRatio < targetRatio:
        targetWidth = cropped.shape[0]/targetRatio # y / ratio = x
        offset = int(round((cropped.shape[1]-targetWidth)/2)) # (x - newx) / 2 gives offset
        cropped2 = cropped[:,offset:-offset,:]
    else:
        # in this case we have to stretch
        print(f'stretching sides...')
        targetWidth = cropped.shape[0] / targetRatio
        offset = int(round((targetWidth - cropped.shape[1]) / 2))

        safeleft = int((0.9*left)*img.shape[1])
        saferight = int((1.1*right)*img.shape[1])

        # stretch left
        L = cropped[:,:safeleft,:]
        Ls = cv2.resize(L,(L.shape[1]+offset,img.shape[0]))
        R = cropped[:,saferight:,:]
        Rs = cv2.resize(R,(R.shape[1]+offset,img.shape[0]))

        subj = cropped[:,safeleft:saferight,:]

        # stitch together
        cropped2 = np.concatenate((Ls,subj,Rs),1)


    resized = cv2.resize(cropped2,(xsize,ysize))

    if plotSteps:
        plt.imshow(resized, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

    writeSuccess = cv2.imwrite(writeloc + file,resized,[int(cv2.IMWRITE_JPEG_QUALITY), imageQuality])

    return writeSuccess

failed_images = []
xx = ['hoepla','yay','tralala','hatsekidee','hatseflats','hopla','hoeplakee','omnomnom','lalala','jippie']
print(f'let the cropping commence.')
for f in range(len(files)):
    file = files[f]
    if file[-3:] != 'jpg':
        continue

    try:
        analyse_image(file,rawloc,xsize,ysize,topDistance,bottomDistance)
        rr = np.random.randint(len(xx) - 1)
        print(f'cropped image {f} of {len(files)} ({file}), {xx[rr]}')
    except:
        print(f'image {file} failed.')
        failed_images.append(file)

print(f'finished. failed images: {failed_images}')

