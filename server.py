from flask import Flask, request, Response
from flask_migrate import Migrate, MigrateCommand
from flask_script import Manager, Server
import jsonpickle
import getFeature
import cv2 as cv
from flask_sqlalchemy import SQLAlchemy
import cnn
import matplotlib.pyplot as plt
import numpy as np
import time

app = Flask(__name__)
app.config.from_pyfile('config.py')
db = SQLAlchemy(app)

migrate = Migrate(app, db)
manager = Manager(app)
manager.add_command('db', MigrateCommand)
server = Server(host="0.0.0.0", port=5000)
manager.add_command('runserver', server)

@app.route('/conv', methods=['POST'])
def conv():
    req=request.get_json()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    data = []
    feature = []
    for i in range(1, 17):
        for j in range(1, 49):
            img = cv.imread("database/" + str(i) + "_" + str(j) + ".png", cv.IMREAD_GRAYSCALE)
            data.append(img)

    for i in range(1,17):
            img = cv.imread("database/" + str(i) + "_1"+ ".png", cv.IMREAD_GRAYSCALE)
            kp, des = cnn.getFeature(img)
            feature.append(des)
            print(i)

    timeCount=[]
    net = cnn.load_model()
    img = np.reshape(cv.imread("database/"+str(req["img"]), cv.IMREAD_GRAYSCALE), (1, 338, 248, 1))
    """
    p = net.predict(img, verbose=0).max()
    p2 = net.predict(img, verbose=0).argmax()
    print(p)
    print(p2)
    exit(1)
    """
    f=[]
    for ft in feature:
        img=np.reshape(img,(1,338,248,1))
        s = time.time()
        p = net.predict(img, verbose=0).argmax()
        img = np.reshape(img, (338, 248))
        kp, des = cnn.getFeature(img)
        matches = sorted(bf.match(ft, des), key=lambda match: match.distance)
        score = 0
        for match in matches:
            score += match.distance
        score /= len(matches)
        print(score)
        f.append(score)
        e = time.time()
        timeCount.append(e-s)
    timeCount=np.array(timeCount)
    f=np.array(f)
    print(timeCount.mean())
    print(str(p+1) + ", calculated in " + str(e - s) + " seconds")
    print(str(f.argmin() + 1))
    return Response(jsonpickle.encode('Ok!'))


@app.route('/', methods=['POST'])
def index():
    req = request.get_json()
    # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    """
    desc=[]
    for i in range(1,17):
        for j in range(1,9):
            kp1, des1 = getFeature.getFeature(cv.imread("database/" + str(i)+"_"+str(j)+".png", cv.IMREAD_GRAYSCALE))
            print(des1)
            desc.append(np.array(des1))
            print(i,j)

    desc=np.array(desc)
    np.savetxt('desc', desc)
    exit(1)
    """
    # kp1,des1=getFeature.getFeature(cv.imread("database/" + str(req["img2"]), cv.IMREAD_GRAYSCALE))

    img = cv.imread(str(req["img"]), cv.IMREAD_GRAYSCALE)
    print(img)
    exit(1)
    s = time.time()
    kp, des = getFeature.getFeature(img)
    print(str(time.time() - s))
    matches = sorted(bf.match(des, des1), key=lambda match: match.distance)
    score = 0
    for match in matches:
        score += match.distance
    score /= len(matches)

    """
    for descp in desc:
        matches = sorted(bf.match(des, descp), key=lambda match: match.distance)
        score = 0
        for match in matches:
            score += match.distance
        score /= len(matches)
        print(score)

    """
    f = time.time()
    print(str(f - s))
    return Response(jsonpickle.encode("Ok!"))
