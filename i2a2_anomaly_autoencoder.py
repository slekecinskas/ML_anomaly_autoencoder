 %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"jupyter":{"outputs_hidden":true}}
%ls '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images'

# %% [code]
!pip install pydicom

# %% [code]
import pandas as pd
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
import cv2
import pydicom
import random
from tqdm import tqdm

%matplotlib inline
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical

# %% [code]
num_files = len([f for f in os.listdir('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images')if os.path.isfile(os.path.join('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images', f))])
print(num_files)

# %% [code]
#variaveis

desired_size = 28 #Tamanho da imagem para resize

batch_size = 32
epochs = 5
inChannel = 3
x, y = desired_size, desired_size
input_img = Input(shape = (x, y, inChannel))
num_classes = 1

# %% [code]
#pre processamento para imagem de treino(train)
def preprocess_image(image_path, desired_size=28):
    im = pydicom.dcmread(image_path)
    im = cv2.resize(np.array(im.pixel_array),(desired_size,desired_size))
    #im = crop_image_from_gray(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #im = np.expand_dims(im, axis=-1)
    return im

# %% [code]
#Resize das imagens dicom de treino/validacao, limitei para 3000 imagens devido a estouro de memoria

#N = train_df.shape[0]
x_train = np.empty((3000,desired_size, desired_size, 3), dtype=np.uint8)

#num_break = 3001

#random.shuffle(num_break)
#x_train = np.zeros((num_files, desired_size, desired_size, 1)).astype('float')

#for i,image_id in enumerate(os.listdir('./stage_2_train_images')):
for i,image_id in enumerate(tqdm(os.listdir('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images')[:3000])):
    x_train[i, :, :, :] = preprocess_image(
        f'/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/{image_id}'
    )
    #cv2.imwrite('/content/gdrive/My Drive/Desafio5_autoencoder_pulmao/rsna-pneumonia_files/' , {image_id})
    #if i >= num_break:
    #    break
    #i += 1

# %% [code]
# rescale com valor maximo do pixel np.max(trainX)=255.0
x_train.shape
trainX = x_train.astype("float32") / 255.0

# %% [code]
# após rescale o valor maximo do pixel deve ser 1.0
np.max(trainX)

# %% [code]
#split treino 80% e validacao 20%
from sklearn.model_selection import train_test_split
(train_X, val_X) = train_test_split(trainX, test_size=0.2,
	random_state=42)

# %% [code]
# Convolucao autoencoder

def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):    
    #decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 3 se for 1 channel(gray) alterar para Conv2D(1,(3,3))
    return decoded

# %% [code]
autoencoder = Model(input_img, decoder(encoder(input_img)))
#autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

from keras.optimizers import Adam
autoencoder.compile(optimizer=Adam(lr=0.001), 
              loss='mean_squared_error', 
              metrics=['accuracy'])

# %% [code]
autoencoder.summary()

# %% [code]
#Treinar o modelo autoencoder com as imagens de treino/validacao
#para após isso calcular o mse desse modelo e apartir dele colocar uma condicao nos imagens de teste outros passos abaixo

autoencoder_train = autoencoder.fit(train_X, train_X, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(val_X,val_X))

# %% [code]
# predict results
decoder_train = autoencoder.predict(train_X)

# %% [code]
#calcular o mse entre treino e decoder do treino (acima)
mse_train = np.mean(np.power(train_X - decoder_train, 2), axis=1)

# %% [code]
thresh = np.quantile(mse_train, 0.999) #threshold = 99% valores mse_train
idxs = np.where(np.array(mse_train) >= thresh)[0]
print("[INFO] mse threshold: {}".format(thresh))
print("[INFO] {} outliers found".format(len(idxs)))
#[INFO] mse threshold: 0.10662752984464183
#[INFO] 202 outliers found

# %% [code]
num_files_predict = len([f for f in os.listdir('/kaggle/input/chest-xray-anomaly-detection/images/')if os.path.isfile(os.path.join('/kaggle/input/chest-xray-anomaly-detection/images/', f))])
print(num_files_predict)

# %% [code]
# funcao preprocessamento imagens de teste (enviadas pelo professor)
def preprocess_image_predict(image_path, desired_size=28):
    im = cv2.imread(image_path)
    im = cv2.resize(im, (desired_size, desired_size))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

# %% [code]
# Preprocessando as imagens e salvando a comparacao do mse_train , maior/igual é anomalia, se nao, nao é

# pre processando dados teste
df_test = pd.read_csv("../input/chest-xray-anomaly-detection/sample_submission.csv")

depth = 3 #channel rgb

N = df_test.shape[0]
X_test = np.empty((N, desired_size, desired_size, depth), dtype=np.uint8)

y_teste = []
y_mse = []
thresh = 0.10662752984464183
#0.11931779579818275

for i, image_id in enumerate(tqdm(df_test['fileName'])):
    X_test[i, :, :, :] = preprocess_image_predict(
        f'../input/chest-xray-anomaly-detection/images/{image_id}'
    )
    result = autoencoder.predict(X_test)
    #X_test = X_test.astype("float32") / 255.0
    mse = np.mean((X_test - result) ** 2)
    #result = np.argmax(result, axis = 1)
    #y_teste.append(result)
    
    if np.array(mse) >= thresh:
       df_test["anomaly"][i] = 1
    else:
       df_test["anomaly"][i] = 0
    #Save images preprocessadas
    #cv2.imwrite('../test/preprocessada/' + image_id ,X_test[i])

# %% [code]
#salva .csv para submissao
df_test.to_csv('./submission2.csv', index = False)

# %% [code]
### Agradecimentos a todos do I2A2 
## Em especial nesse desafio ao Erique Souza, Thomas e o Vinicius
## Ao professor Fernando Camargo por toda a paciencia e explicacao em cada passo meu.

### :)
