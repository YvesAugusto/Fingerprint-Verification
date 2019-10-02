from flask import Flask, request, Response
import jsonpickle
from flask_migrate import Migrate, MigrateCommand
from flask_script import Manager, Server
import numpy as np
import cv2
from flask_sqlalchemy import SQLAlchemy

# from sqlalchemy.dialects.postgresql import ARRAY


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:senha098@127.0.0.1:5432/fingerprint'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['DEBUG'] = True

db = SQLAlchemy(app)


# db.create_all()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255))

    def __init__(self, username):
        self.username = username


class Feature(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    vector = db.Column(db.ARRAY(db.Float, dimensions=2))

    def __init__(self, vector):
        self.vector = vector


migrate = Migrate(app, db)
manager = Manager(app)
manager.add_command('db', MigrateCommand)
server = Server(host="0.0.0.0", port=5000)
manager.add_command('runserver', server)

import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt
from enhance import image_enhance
from skimage.morphology import skeletonize, thin
import time


# os.chdir("/home/raskolnikov/Área de Trabalho/python-fingerprint-recognition")

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


def get_deors(img):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = image_enhance.image_enhance(img)
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

def create():
    for i in range(1,17):
        print(i)
        img_name = str(i)+"_1.png"
        img1 = cv.imread("database/" + img_name, cv.IMREAD_GRAYSCALE)
        kp1, des1 = get_deors(img1)
        des1 = des1.astype(float)
        feature = Feature(list(des1))
        db.session.add(feature)
        db.session.commit()

def call(img_aux):
    # img_name = "1_1.png"
    # img1 = cv.imread("database/" + img_name, cv.IMREAD_GRAYSCALE)
    # kp1, des1l = get_deors(img1)
    # exit(1)
    # print(type(des1l[0][0]))
    start = time.time()
    kp2, des2 = get_deors(img_aux)
    des2 = np.array(des2, dtype=np.uint8)
    identification=np.zeros(17)
    for i in range(1, 17):
        des1=Feature.query.get(i).vector
        des1=np.asarray(des1)
        des1 = np.array(des1,dtype=np.uint8)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda match: match.distance)
        score = 0
        for match in matches:
            score += match.distance
        score_threshold = 34.1
        print(score / len(matches))
        if score / len(matches) < score_threshold:
            print("True")
            identification[i]=1
        else:
            print("False")
            identification[i]=0

    return identification, float(time.time()-start)


@app.route('/api/test', methods=['POST'])
def test():
    req = request
    nparr = np.fromstring(req.data, np.uint8)
    img = cv2.imdecode(nparr, cv.IMREAD_GRAYSCALE)
    #create()
    identification, tempo=call(img)

    ans = {'message': 'image received, size={}x{}'.format(img.shape[1], img.shape[0]), 'time': tempo}
    ans = jsonpickle.encode(ans)
    return Response(response=ans, status=200, mimetype="application/json")


manager.run()
