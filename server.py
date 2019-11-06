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

class ImprovedFeature(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    label = db.Column(db.Integer)
    vector = db.Column(db.ARRAY(db.Float, dimensions=2))

    def __init__(self, label, vector):
        self.label = label
        self.vector = vector


class Feature(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    label = db.Column(db.Integer)
    vector = db.Column(db.ARRAY(db.Float, dimensions=2))

    def __init__(self,label,vector):
        self.label=label
        self.vector = vector

class Scores(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    label = db.Column(db.Integer)
    vector = db.Column(db.ARRAY(db.Float, dimensions=2))

    def __init__(self,label,vector):
        self.label=label
        self.vector = vector

def matching(f1,f2,bf):
    #print(len(f1), len(f1[0]))
    #print(len(f2), len(f2[0]))
    matches = sorted(bf.match(f1, f2), key=lambda match: match.distance)
    score = 0
    for match in matches:
        score += match.distance
    score /= len(matches)
    return score

def get_feature_from_db():
    des1=[]
    des2=[]
    for j in range(1,769):
        des = Feature.query.get(j).vector
        des= np.asarray(des)
        des = np.array(des, dtype=np.uint8)
        des1.append(des)
        des = ImprovedFeature.query.get(j).vector
        des = np.asarray(des)
        des = np.array(des, dtype=np.uint8)
        des2.append(des)

    ft=np.array(des1)
    imp_ft=np.array(des2)
    print(ft.shape)
    print(imp_ft.shape)

    return ft, imp_ft

def get_label_score(feature,i,bf):
    feature=feature[i*48:48*(i+1)]
    original=feature[0:8]
    train=feature[8:38]
    test=feature[38:48]
    matches=[]
    for tr in train:
        v=[]
        for orig in original:
            v.append(matching(tr,orig,bf))
        matches.append(v)

    matches=np.array(matches)
    mean=matches.sum(axis=0)/len(matches)
    maxx=matches.max(axis=0)
    #print(mean)
    #print(maxx)
    #print((mean+maxx)/2)
    return mean

def testing(scores,feature,i,bf):
    global acc
    ft=feature[48*i:48*(i+1)]
    ft=ft[0:8]
    dec=[]
    for k in range(16):
        if(k==i):
            print("--------- Same Class ----------")
            continue
        else:
            x=feature[48*k:48*(k+1)]
            acc=0
            for j in range(8,48):
                c=0
                scs=[]
                for l in range(8):
                    aux=matching(x[j], ft[l], bf)
                    scs.append(aux)
                    if(aux < scores[l]):
                        c+=1


                if (c >= 6):
                    print("Yes", c, scs)
                else:
                    print("No", c, scs)
                    acc+=1
        acc=acc/40
        print(k)
        print("------------ " + "False Negative Rate for this class: " + str(1-acc) + " -----------------")
        dec.append(1-acc)
        time.sleep(1.5)
    dec = np.array(dec)
    print(dec.mean())
    return dec

@app.route('/create',methods=['POST'])
def create():

    for j in range(1,17):
        for i in range(1,49):
            img_name = str(j)+"_"+str(i)+".png"
            img1 = np.array(cv.imread("database/" + img_name, cv.IMREAD_GRAYSCALE))
            kp1, des1 = getFeature.getFeature(img1)
            kp2,des2=cnn.getFeature(img1)
            des1 = des1.astype(float)
            des2 = des2.astype(float)
            feature = ImprovedFeature(j,list(des1))
            impfeature=Feature(j,list(des2))
            db.session.add(feature)
            db.session.commit()
            db.session.add(impfeature)
            db.session.commit()
            print(j,i)

@app.route('/conv', methods=['POST'])
def conv():
    req=request.get_json()
    img = cv.imread("database/"+str(req["img"]), cv.IMREAD_GRAYSCALE)
    img = np.resize(img, (338, 248))
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    feature, imp_feature=get_feature_from_db()
    scores=get_label_score(feature,15,bf)
    scores+=1.5
    dec=[]
    decc=[]
    for i in range(16):
        print()
        print()
        print("----------------------- Testando falsos positivos com a classe " + str(i+1) + " ------------------------")
        print()
        print()
        x=testing(scores, feature, i, bf)
        decc.append(x)
        dec.append(x.mean())
        time.sleep(2.9)

    print(dec)
    dec=np.array(dec)
    print(dec.mean())
    print(decc)

    exit(1)
    timeCount=[]
    net = cnn.load_model()
    img=np.reshape(img,(1,338,248,1))
    p = net.predict(img, verbose=0)
    img = np.reshape(img, (338, 248))
    p2=p.max()
    p=p.argmax()
    print(p2, p)
    kp,desc=cnn.getFeature(img)
    #print(desc)
    #desc=np.asarray(desc)
    #print(desc)
    #kp,desc = np.array(desc, dtype=np.uint8)
    c=0
    scs=[]
    for i in range(8):
        v=feature[(48*p)+i]
        x=matching(desc, v, bf)
        scs.append(x)
        if(x < scores[i]):
            c+=1
    if(c>=5):
        print("Yes", c, scs)
    else:
        print("No", c, scs)
    """
    p = net.predict(img, verbose=0).max()
    p2 = net.predict(img, verbose=0).argmax()
    print(p)
    print(p2)
    exit(1)
    """
    """
    kp, des = getFeature.getFeature(img)
    for ft in feature:
        sc=[]
        #times = []
        for ftr in ft:
            #s=time.time()
            img=np.reshape(img,(1,338,248,1))
            p = net.predict(img, verbose=0).argmax()
            img = np.reshape(img, (338, 248))
            matches = sorted(bf.match(ftr, des), key=lambda match: match.distance)
            score = 0
            for match in matches:
                score += match.distance
            score /= len(matches)
            sc.append(score)
            #e=time.time()
            #times.append(e-s)
        #timeCount.append(times)
        f.append(sc)

    print(f)
    #print(timeCount)
    """
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
    img=np.reshape(img,(338,248))
    print(img)

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

