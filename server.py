from flask import Flask, request, Response
from flask_migrate import Migrate, MigrateCommand
from flask_script import Manager, Server
import jsonpickle
import cv2 as cv
from flask_sqlalchemy import SQLAlchemy
import cnn
import numpy as np
import time
from sklearn.svm import SVC
import pickle
import keras

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

class DetectedFeature(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    label = db.Column(db.Integer)
    vector = db.Column(db.ARRAY(db.Float, dimensions=2))

    def __init__(self,label,vector):
        self.label=label
        self.vector = vector

class SvmPesos(db.Model):
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

def get_detect_from_db():
    des1=[]
    des2=[]
    for j in range(1,769):
        des = DetectedFeature.query.get(j).vector
        des= np.asarray(des)
        des = np.array(des, dtype=np.uint8)
        des1.append(des)

    ft=np.array(des1)
    print(ft.shape)

    return ft

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
        if(k!=i):
            #print("--------- Same Class ----------")
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


                if (c >= 3):
                    print("Yes", c, scs)
                else:
                    print("No", c, scs)
                    acc+=1
        acc=acc/40
        print(k)
        print("------------ " + "True Positive Rate for this class: " + str(1-acc) + " -----------------")
        dec.append(1-acc)
        #time.sleep(1.5)
    dec = np.array(dec)
    print(dec.mean())
    return dec

import tensorflow as tf
graph = tf.get_default_graph()
net = cnn.load_model()
#net._make_predict_function()
feature=get_detect_from_db()
orb = cv.ORB_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
svms=[]
for i in range(16):
    file = open('./SVM/svm_' + str(i), 'rb')
    SVM = pickle.load(file)
    svms.append(SVM)
    file.close()

@app.route('/svm',methods=['POST'])
def svm():
    """
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    x=[]
    y=[]
    for i in range(16):
        v1=feature[48*i:(48*i)+8]
        for j in range(16):
            aux_x=[]
            aux_y=[]
            if i==j:
                v2=feature[48*i:48*(i+1)]
                aux_ft=[]
                aux_fty = []
                for ft2 in v2:
                    aux=[]
                    for ft1 in v1:
                        aux.append(matching(ft2,ft1,bf))
                    aux_ft.append(aux)
                    aux_fty.append(1)
                aux_x.append(aux_ft)
                aux_y.append(aux_fty)

            else:
                v2 = feature[48*j:48*(j+1)]
                aux_ft = []
                aux_fty = []
                for ft2 in v2:
                    aux = []
                    for ft1 in v1:
                        aux.append(matching(ft2, ft1, bf))
                    aux_ft.append(aux)
                    aux_fty.append(0)
                aux_x.append(aux_ft)
                aux_y.append(aux_fty)
            x.append(aux_x)
            y.append(aux_y)

    x=np.array(x)
    y=np.array(y)
    x=x.reshape((16,16,48,8))
    y=y.reshape((16,16,48))
    for k in x:
        k=np.concatenate(k)
    for k in y:
        k=np.concatenate(k)

    import pickle
    output = open('x.pkl', 'wb')
    pickle.dump(x, output)
    output.close()
    output = open('y.pkl', 'wb')
    pickle.dump(y, output)
    output.close()

    import pickle
    out=open('x.pkl', 'rb')
    x=pickle.load(out)
    out.close()
    out=open('y.pkl', 'rb')
    y=pickle.load(out)
    out.close()
    svms=[]
    X=[]
    Y=[]
    for i in range(16):
        X.append(np.concatenate(x[i]))
    for i in range(16):
        Y.append(np.concatenate(y[i]))

    X=np.array(X)
    Y=np.array(Y)
    print(X.shape)
    print(Y.shape)
    for i in range(16):
        clf=SVC(gamma='auto')
        clf.fit(X[i], Y[i])
        svms.append(clf)

    svms=np.array(svms)
    print(svms.shape)
    for i in range(16):
        output=open('./SVM/svm_'+str(i), 'wb')
        pickle.dump(svms[i], output)
        output.close()
    """
    print(net.summary())
    import keras
    keras.backend.clear_session()


    return Response(jsonpickle.encode('Ok!'))


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

@app.route('/create2',methods=['POST'])
def create2():
    orb = cv.ORB_create()
    times=[]
    for j in range(1,17):
        for i in range(1,49):
            img_name = str(j)+"_"+str(i)+".png"
            s=time.time()
            kp1, des1 = orb.detectAndCompute(cv.imread("database/" + img_name, cv.IMREAD_GRAYSCALE), None)
            des1 = des1.astype(float)
            feature=DetectedFeature(j,list(des1))
            db.session.add(feature)
            db.session.commit()

            print(j,i)

@app.route('/testTime', methods=['POST'])
def test_time():
    req=request.get_json()
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    img_name=req["img"]
    kp, des = orb.detectAndCompute(cv.imread("database/" + img_name, cv.IMREAD_GRAYSCALE), None)
    times = []
    for j in range(1, 17):
        v=[]
        for i in range(1, 49):
            img_name = str(j) + "_" + str(i) + ".png"
            s = time.time()
            kp1, des1 = orb.detectAndCompute(cv.imread("database/" + img_name, cv.IMREAD_GRAYSCALE), None)
            matching(des,des1,bf)
            e=time.time()
            v.append(e-s)
            print(j,i)
        print(v)
        times.append(v)

    print(np.array(times).mean())

    return Response(jsonpickle.encode('Ok!'))

@app.route('/conv', methods=['POST'])
def conv():
    times=[]
    req=request.get_json()
    score=[]
    start = time.time()
    img = cv.imread("database/"+str(req["img"]), cv.IMREAD_GRAYSCALE)
    s=time.time()
    kp, des = orb.detectAndCompute(img, None)
    times.append(time.time() - s)
    img=np.reshape(img,(1,338,248,1))
    s=time.time()
    with graph.as_default():
        label=net.predict(img).argmax()
    times.append(time.time()-s)
    s=time.time()
    ft=feature[48*label:48*label + 8]
    for f in ft:
        score.append(matching(f, des, bf))
    times.append(time.time() - s)
    end=time.time()
    s=time.time()
    print(svms[label].predict([score]))
    times.append(time.time() - s)
    print("Executado em " + str(end-start) + " segundos")
    print("Tempo da extração de caracteristicas : " + str(times[0]))
    print("Tempo do predict da CNN : " + str(times[1]))
    print("Tempo do matching : " + str(times[2]))
    print("Tempo do predict da SVM : " + str(times[3]))




    return Response(jsonpickle.encode('Ok!'))


@app.route('/', methods=['POST'])
def index():
    req = request.get_json()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    scores = []
    for i in range(16):
        scores.append(get_label_score(feature, i, bf) + 5)

    dec = []
    decc = []
    for i in range(16):
        print()
        print()
        print("----------------------- Testando falsos positivos com a classe " + str(
            i + 1) + " ------------------------")
        print()
        print()
        x = testing(scores[i], feature, i, bf)
        decc.append(x)
        dec.append(x.mean())
        #time.sleep(2.9)

    print(dec)
    dec = np.array(dec)
    print(dec.mean())
    print(decc)
    return Response(jsonpickle.encode("Ok!"))
    orb=cv.ORB_create()
    img_name=req["img1"]
    kp1,des1=orb.detectAndCompute(cv.imread("database/" + img_name, cv.IMREAD_GRAYSCALE),None)
    img_name = req["img2"]
    kp2,des2=orb.detectAndCompute(cv.imread("database/" + img_name, cv.IMREAD_GRAYSCALE),None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    score=matching(des1,des2,bf)
    print(score)

    return Response(jsonpickle.encode(str(score)))

