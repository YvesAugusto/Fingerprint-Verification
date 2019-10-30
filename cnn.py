import numpy as np
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
import cv2 as cv
def loadData(f):
    trainx=[]
    testx=[]
    trainy=[]
    testy=[]

    for i in range(1, 17):
        for j in range(1,int(f*8)):
            v = np.zeros(16)
            v[i - 1] = 1
            trainx.append(cv.imread("database/" + str(i)+"_"+str(j)+".png", cv.IMREAD_GRAYSCALE))
            trainy.append(i)
        for j in range(int(f*8), 8):
            testx.append(cv.imread("database/" + str(i)+"_"+str(j)+".png", cv.IMREAD_GRAYSCALE))
            v=np.zeros(16)
            v[i-1]=1
            testy.append(v)

    trainx=np.array(trainx)
    trainx=np.resize(trainx,(80,338,248,1))
    trainy=np.array(trainy)
    trainy=np.resize(trainy,(80,16))
    testx=np.array(testx)
    testx=np.resize(testx, (32,338,248,1))
    testy=np.array(testy)
    testy=np.resize(testy,(32,16))
    return trainx,testx,trainy,testy

import cv2 as cv
import numpy as np
from skimage.morphology import skeletonize, thin
from keras.models import model_from_json

def removedot(invertThin):
    temp0 = np.array(invertThin[:])
    temp0 = np.array(temp0)
    temp1 = temp0 / 255
    temp2 = np.array(temp1)
    temp3 = np.array(temp2)

    enhanced_img = np.array(temp0)
    filter0 = np.zeros((10, 10))
    W, H = temp0.shape[:2]
    filtersize = 6

    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize, j:j + filtersize]
            flag = 0
            if sum(filter0[:, 0]) == 0:
                flag += 1
            if sum(filter0[:, filtersize - 1]) == 0:
                flag += 1
            if sum(filter0[0, :]) == 0:
                flag += 1
            if sum(filter0[filtersize - 1, :]) == 0:
                flag += 1
            if flag > 3:
                temp2[i:i + filtersize, j:j + filtersize] = np.zeros((filtersize, filtersize))

    return temp2


def getFeature(img):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    #img = image_enhance.image_enhance(img)
    img = np.array(img, dtype=np.uint8)
    # Threshold
    ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    # Normalize to 0 and 1 range
    img[img == 255] = 1

    # Thinning
    skeleton = skeletonize(img)
    skeleton = np.array(skeleton, dtype=np.uint8)
    skeleton = removedot(skeleton)

    # Harris corners
    harris_corners = cv.cornerHarris(img, 3, 3, 0.04)
    harris_normalized = cv.normalize(harris_corners, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32FC1)
    threshold_harris = 125
    # Extract keypoints
    keypoints = []
    for x in range(0, harris_normalized.shape[0]):
        for y in range(0, harris_normalized.shape[1]):
            if harris_normalized[x][y] > threshold_harris:
                keypoints.append(cv.KeyPoint(y, x, 1))
    # Define deor
    orb = cv.ORB_create()
    # Compute deors
    _, des = orb.compute(img, keypoints)
    return (keypoints, des)

def load_model():
  json_file = open('net.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("net.h5")
  print("Loaded model from disk")

  return loaded_model

