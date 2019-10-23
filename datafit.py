import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

from dataGenerator import dataGenerator
from dataEnhancer import dataEnhancer



class dataFit(object):
	"""docstring for dataFit"""
	def __init__(self, trainData, testData, trainLabel, testLabel, parameters, layers):

		self.trainData=trainData
		self.testData=testData
		self.trainLabel=trainLabel
		self.testLabel=testLabel
		self.parameters=parameters
		self.layers=layers

	def fit(self):
		trainData=self.trainData
		testLabel=self.testLabel
		trainLabel=self.trainLabel
		testData=self.testData
		p=self.parameters
		layers=self.layers

		net=Sequential()
		net.add( Conv2D(layers[0]['units'], layers[0]['kernel_size'], input_shape=(103,96,1) ))
		for i in range(1, len(layers)):
			net.add( Conv2D(layers[i]['units'], layers[i]['kernel_size']) )
			net.add( MaxPooling2D(pool_size=layers[i]['pool_size'], strides=layers[i]['strides']))
		net.add(Flatten())
		net.add(Dense(100))
		net.add( Dense(600, activation=p['activation']) )
		net.compile(optimizer=p['optimizer'], loss=p['loss'], metrics=p['metrics'])
		net.fit(x=trainData, y=trainLabel, epochs=3, batch_size=1, validation_data=(trainData, trainLabel))

		save=input("Salvar (s/n): ")
		if save=='s':
			net_json = net.to_json()
			with open("net.json", "w") as json_file:
			    json_file.write(net_json)
			# serialize weights to HDF5
			net.save_weights("net.h5")
			print("Saved net to disk !!")

		else:
			print("Net not saved to disk !!")











"""
def read_image(image_name):
# Load a subject's right hand, second finger image and risize to 352*352
    fingerprint = cv.imread(image_name, 0)
    fingerprint = cv.resize(fingerprint,(248,338))
    fpcopy = fingerprint[:]
    row, col = fingerprint.shape
    return row, col, np.array(fingerprint), np.array(fpcopy)
def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)
def generate_one_label(i):
	label=np.zeros((48, 16))
	for row in label:
		row[i]=1
	return np.array(label)
def generate_full_label():
	label=[]
	for i in range(16):
		label.append(generate_one_label(i))
	return np.array(label)
def read_one_label(label):
	images=[]
	for i in range(1, 9):
		row,col,fp,fpcopy=read_image(str(label)+"_"+str(i)+".png")
		images.append(fp)
	return images
def read_one_label_increased(label):
	images=[]
	for i in range(1, 49):
		row,col,fp,fpcopy=read_image(str(label)+"_"+str(i)+".png")
		images.append(fp)
	return images
def read_dataset_increased():
	images=[]
	for i in range(1, 17):
		images.append(read_one_label_increased(i))
	return np.array(images), concatenate(generate_full_label())
def generate_images_of_one_label(images, n):
	i=9
	for ind, img in enumerate(images):
		# ADDING MEDIAN NOISE:
		cv.imwrite(str(n)+"_"+str(i)+".png", cv.medianBlur(img, 5))
		i+=1
		# ADDING GAUSSIAN NOISE:
		cv.imwrite(str(n)+"_"+str(i)+".png", cv.GaussianBlur(img,(5,5), 0))
		i+=1
		# ADDING BILATERAL NOISE:
		cv.imwrite(str(n)+"_"+str(i)+".png", cv.bilateralFilter(img,9,75,75))
		i+=1
		# ADDING GAUSSIAN ROTATED NOISE
		cv.imwrite(str(n)+"_"+str(i)+".png", cv.GaussianBlur(rotate_image(cv.medianBlur(img, 3), 5),(3,3), 0))
		i+=1
		# ADDING GAUSSIAN DESROTATED NOISE
		cv.imwrite(str(n)+"_"+str(i)+".png", cv.GaussianBlur(rotate_image(cv.medianBlur(img, 3), -5),(3,3), 0))
		i+=1
def preprocess(fp):
	# Histogram Equalization
	fp=cv.equalizeHist(fp)
	# OTSU BINARIZATION WITH GAUSSIAN
	blur = cv.bilateralFilter(fp,9,10,10)
	ret3,th3 = cv.threshold(blur,0,1,cv.THRESH_BINARY+cv.THRESH_OTSU)
	return th3
def concatenate(v):
	return np.concatenate(v)
def preprocess_full_data(images):
	imgs=[]
	for user in images:
		aux=[]
		for img in user:
			aux.append(preprocess(img))
		imgs.append(aux)
	return np.array(np.reshape(concatenate(imgs), (768, 338, 248, 1)))
def rotate_image(img, theta):
	rows,cols=img.shape
	M = cv.getRotationMatrix2D((cols/2,rows/2),theta,1)
	dst = cv.warpAffine(img,M,(cols,rows))
	return dst
def increase_dataset():
	for i in range(1,17):
		generate_images_of_one_label(read_one_label(i), i)
	print("--- Dataset succesfully increased !! ---")
def shuffle_dataset(images, label):
	pair=np.array([[images[i], label[i]] for i in range(len(images))])
	pair=np.random.shuffle(pair)
	print(pair)
	a=np.array([pair[i][0] for i in range(len(images))])
	b=np.array([pair[i][1] for i in range(len(images))])
	return a,b
def fit(images,label):
	print("--------- Importando bibliotecas ---------")
	from keras.models import Sequential
	from keras.utils import to_categorical
	from keras.models import model_from_json
	from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
	print("--------- Iniciando treinamento ---------")
	# -------------------------------------------------------------------------------
	net=Sequential()
	net.add(Conv2D( 64, kernel_size=3, input_shape=(338,248,1) ))
	net.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	net.add(Conv2D( 16, kernel_size=3 ))
	net.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	net.add(Conv2D( 8, kernel_size=3 ))
	net.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	net.add(Flatten())
	net.add(Dense(50))
	net.add(Dense( 16, activation='softmax' ))
	net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	net.fit(x=images,y=label, epochs=3, batch_size=1,
	validation_data=(images, label))
	flag2=input("Deseja salvar a rede? (S/N): ")
	
	if(flag2=="S"):
		# SALVANDO A REDE
		net_json = net.to_json()
		with open("net.json", "w") as json_file:
		    json_file.write(net_json)
		# serialize weights to HDF5
		net.save_weights("net.h5")
		print("Saved net to disk !!")
	else:
		print("Net not saved to disk !!")
#increase_dataset()
images,label=read_dataset_increased()
images=preprocess_full_data(images)
fit(images,label)
"""
"""
data, label = create_data()
print(data.shape)
flag=input("Deseja treinar ou predizer? (T/P): ")
if(flag=="T"):
	net = Sequential()
	net.add( Conv2D( 64, kernel_size=5, activation='relu', input_shape=(338, 248, 3) ) )
	net.add( Conv2D( 16, kernel_size=5, activation='relu' ) )
	net.add( Conv2D( 8, kernel_size=5, activation='relu' ) )
	net.add( Flatten() )
	net.add( Dense(50, activation='relu') )
	net.add( Dense(16, activation='softmax') )
	net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	net.fit(x=data,y=label, epochs=3, batch_size=1,validation_data=(data, label))
	flag2=input("Deseja salvar a rede? (S/N): ")
	if(flag2=="S"):
		# SALVANDO A REDE
		net_json = net.to_json()
		with open("net.json", "w") as json_file:
		    json_file.write(net_json)
		# serialize weights to HDF5
		net.save_weights("net.h5")
		print("Saved net to disk")
	else:
		print("Rede não foi salva")
elif(flag=="P"):
	json_file = open('net.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("net.h5")
	print("Loaded model from disk")
	print("----------------------------- Testing phase -----------------------------")
	print("----- Digitar, por exemplo, os numeros 16 e 2, equivale a predizer a imagem 16_2.png da pasta onde este programa está salvo -----")
	while(True):
		print("Para sair disso basta digitar 0 no primeiro número ou no segundo número")
		print("-------")
		n1=input("Digite o primeiro número: ")
		n2=input("Digite o primeiro número: ")
		print()
		if(n1==0 and n2 == 0):
			break
		img=np.reshape( load_image(n1, n2), (1, 338,248, 3) )
		predict=loaded_model.predict(img, verbose=0)
		print("A digital é da classe: " + str(predict.argmax()))
		print("-------")
		print()
else:
	print("Comando não encontrado")
img=load_image(1,8)
img=np.reshape(img,(1, 248, 338, 1))
predict=net.predict(img, verbose=0)
argmax=predict.argmax()
predict=np.zeros(16)
predict[argmax]=1
print(predict)
"""
