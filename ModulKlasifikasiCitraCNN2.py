##################################################
# Progrm Ini dijadikan modul dengan nama
#   ModulKlasifikasiCItraCNN.py
#################################################

import os
from keras.models import load_model
import cv2
import numpy as np
from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import expand_dims
from keras.utils import load_img
from keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

# load the image
def LoadCitraTraining(sDir,LabelKelas):
  JumlahKelas=len(LabelKelas)
  TargetKelas = np.eye(JumlahKelas)
  # Menyiapkan variabel list untuk data menampung citra dan data target
  X=[]#Menampung Data Citra
  T=[]#Menampung Target
  for i in range(len(LabelKelas)):
    #Membaca file citra di setiap direktori data set
    DirKelas = os.path.join(sDir, LabelKelas[i])
    files = os.listdir(DirKelas)
    for f in files:
      ff=f.lower()
      print(f)
      #memilih citra dengan extensi jpg,jpeg,dan png
      if (ff.endswith('.jpg')|ff.endswith('.jpeg')|ff.endswith('.png')):
        NmFile = os.path.join(DirKelas,f)
        img= np.double(cv2.imread(NmFile,1))
        img=cv2.resize(img,(128,128))
        img= np.asarray(img)/255
        img=img.astype('float32')
        #Menambahkan citra dan target ke daftar
        X.append(img)
        T.append(TargetKelas[i])
    #--------akhir loop :Pfor f in files-----------------
  #-----akhir  loop :for i in range(len(LabelKelas))----

  #Mengubah List Menjadi numppy array
  X=np.array(X)
  T=np.array(T)
  X=X.astype('float32')
  T=T.astype('float32')
  return X,T

def ModelDeepLearningCNN(JumlahKelas):
  input_img = Input(shape=(128, 128, 3))
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
  x = MaxPooling2D((2, 2), padding='same')(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), padding='same')(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  x = Flatten()(x)
  x = Dense(100,activation='relu')(x)
  x=Dense(JumlahKelas,activation='softmax')(x)
  ModelCNN = Model(input_img, x)
  ModelCNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  #ModelCNN.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
  return ModelCNN

def TrainingCNN(JumlahEpoh,DirektoriDataSet,LabelKelas,NamaFileBobot):
  #Membaca Data training dan label Kelas
  X,D=LoadCitraTraining(DirektoriDataSet,LabelKelas)
  JumlahKelas = len(LabelKelas)
  #Membuat Model CNN
  ModelCNN =ModelDeepLearningCNN(JumlahKelas)
  #Trainng
  history=ModelCNN.fit(X, D,epochs=JumlahEpoh,shuffle=True)
  #Menyimpan hasil learning
  ModelCNN.save(NamaFileBobot)
  #Mengembalikan output
  return ModelCNN,history


def Klasifikasi(DirDataSet,DirKlasifikasi,LabelKelas,ModelCNN=[]):
#Menyiapkan Data input Yang akan di kasifikasikan
  X=[]
  ls = []
  DirKelas = DirDataSet+"/"+DirKlasifikasi
  print(DirKelas)
  files = os.listdir(DirKelas)
  n=0
  for f in files:
    ff=f.lower()
    print(f)
    if (ff.endswith('.jpg')|ff.endswith('.jpeg')|ff.endswith('.png')):
      ls.append(ff)
      NmFile = os.path.join(DirKelas,f)
      img= cv2.imread(NmFile,1)
      img=cv2.resize(img,(128,128))
      img= np.asarray(img)/255
      img=img.astype('float32')
      X.append(img)
    #----Akhir if-------------
  #---Akhir For
  X=np.array(X)
  X=X.astype('float32')
  #Melakukan prediksi Klasifikasi
  hs=ModelCNN.predict(X)

  LKlasifikasi=[]
  LKelasCitra =[]
  n = X.shape[0]
  for i in range(n):
    v=hs[i,:]
    if v.max()>0.5:
      idx = np.max(np.where( v == v.max()))
      LKelasCitra.append(LabelKelas[idx])
    else:
      idx=-1
      LKelasCitra.append("-")
    #------akhir if
    LKlasifikasi.append(idx)
  #----akhir for
  LKlasifikasi = np.array(LKlasifikasi)
  return ls, hs, LKelasCitra

def LoadModel(sf):
  ModelCNN=load_model(sf)
  return ModelCNN




def ImageAugmentation(SPath,Kelas):
  parent_dir = SPath
  directory = Kelas
  directoryExt =directory +"_ext"
  sdirExt = os.path.join(parent_dir, directoryExt)
  if not os.path.exists(sdirExt):
    os.mkdir(sdirExt)
  directory_image =directory
  sDir = os.path.join(parent_dir, directory_image)
  files = os.listdir(sDir)
  ii=0
  for f in files:
    ff=f.lower()
    if (ff.endswith('.jpg')|ff.endswith('.jpeg')|ff.endswith('.png')):
      print(ff)
      sfs= os.path.join(sDir,ff)
      img = load_img(sfs)
      img2 = np.array(img)
      sfn= os.path.join(sdirExt, ff)
      cv2.imwrite(sfn,img2)
      data = img_to_array(img)
      samples = expand_dims(data, 0)
      datagen = ImageDataGenerator(rotation_range=90,brightness_range=[0.2,2.0],zoom_range=[0.5,2.0],width_shift_range=0.2,height_shift_range=0.2)
      it = datagen.flow(samples, batch_size=1)
      for i in range(9):
        batch = it.next()
        image = batch[0].astype('uint8')
        now = datetime.now()
        ii=ii+1
        sf = now.strftime("%Y%m%d%H%M%S")+"_"+str(ii)+".jpg"
        sfn= os.path.join(sdirExt, sf)
        cv2.imwrite(sfn,image)