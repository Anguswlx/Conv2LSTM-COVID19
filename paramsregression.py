import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
from keras.optimizers import SGD, Adam, Adamax
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.models import Sequential
from keras.layers import TimeDistributed, Conv2D, AveragePooling2D, Flatten, Dropout, Dense, BatchNormalization, CuDNNLSTM
from keras.layers.recurrent import LSTM

from tensorflow.keras.models import Model, load_model

x_dat = np.loadtxt('../COVIDMAPS_more.dat')

y_dat = x_dat[:,625:-2]
x_dat = x_dat[:,:625]

x_dat = x_dat.reshape((10000,30,-1))
x_dat1 = x_dat.reshape((10000,30,25,25,1))

y_dat = y_dat.reshape((10000,30,-1))
y_dat1 = y_dat[:,-1,:]

randomize = np.arange(len(x_dat1))
np.random.shuffle(randomize)
x_dat1 = x_dat1[randomize]
y_dat1 = y_dat1[randomize]

min0 = np.min(y_dat1[:,0])
min1 = np.min(y_dat1[:,1])
min2 = np.min(y_dat1[:,2])
max0 = np.max(y_dat1[:,0])
max1 = np.max(y_dat1[:,1])
max2 = np.max(y_dat1[:,2])
min3 = np.min(y_dat1[:,3])
max3 = np.max(y_dat1[:,3])
min4 = np.min(y_dat1[:,4])
max4 = np.max(y_dat1[:,4])
 
y_dat1[:,0] = (y_dat1[:,0]-min0)/(max0-min0)
y_dat1[:,1] = (y_dat1[:,1]-min1)/(max1-min1)
y_dat1[:,2] = (y_dat1[:,2]-min2)/(max2-min2)
y_dat1[:,3] = (y_dat1[:,3]-min3)/(max3-min3)
y_dat1[:,4] = (y_dat1[:,4]-min4)/(max4-min4)

print('min0,1,2,3,4,:',min0,min1,min2,min3,min4)
print('max0,1,2,3,4,:',max0,max1,max2,max3,max4)

xtrain=[]
ytrain=[]

lookback = 7

for i in range(800):
    xx = x_dat1[i]
    yy = y_dat1[i]
    for j in range(30-lookback):
        xxij = xx[j:j+lookback]
        ninitial = np.sum(xx[j+lookback])
        xtrain.append(xxij/ninitial)
        ytrain.append(yy)

xtest=[]
ytest=[]

for i in range(800,1000):
    xx = x_dat1[i]
    yy = y_dat1[i]
    for j in range(30-lookback):
        xxij = xx[j:j+lookback]
        ninitial = np.sum(xx[j+lookback])
        xtest.append(xxij/ninitial)
        ytest.append(yy)

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)
xtest = np.array(xtest)
ytest = np.array(ytest)


def regulize(x_dat, y_dat, shuffle=True):
    min0 = np.min(y_dat[:,0])
    min1 = np.min(y_dat[:,1])
    min2 = np.min(y_dat[:,2])
    max0 = np.max(y_dat[:,0])
    max1 = np.max(y_dat[:,1])
    max2 = np.max(y_dat[:,2])
    min3 = np.min(y_dat[:,3])
    max3 = np.max(y_dat[:,3])
    min4 = np.min(y_dat[:,4])
    max4 = np.max(y_dat[:,4])


    print('min0,1,2,3,4,:',min0,min1,min2,min3,min4)
    print('max0,1,2,3,4,:',max0,max1,max2,max3,max4)

    if shuffle == True:
        randomize = np.arange(len(x_dat))
        np.random.shuffle(randomize)
        x_dat = x_dat[randomize]
        y_dat = y_dat[randomize]

    return x_dat, y_dat

xtrain, ytrain = regulize(xtrain, ytrain)

xtest, ytest = regulize(xtest, ytest)

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

epochs = 50
epochs2 = 50

ytrain = ytrain[:,0] #transition rate
ytest = ytest[:,0]

print('x_dat.shape:',np.max(x_dat),np.min(x_dat))

from sklearn.model_selection import train_test_split


model = Sequential()

model.add(
    TimeDistributed(
        Conv2D(64, (3, 3), activation='relu', padding='same'), 
        input_shape=(lookback, 25, 25, 1)
    )
)

model.add(
    TimeDistributed(
        Conv2D(64, (3, 3), strides= (2, 2), activation='relu', padding='same'),
    )
)

model.add(
    TimeDistributed(
        Conv2D(64, (3, 3), activation='relu', padding='same'),
    )
)

model.add(
    TimeDistributed(
        Conv2D(64, (3, 3), activation='relu', padding='same'),
    )
)

model.add(TimeDistributed(Flatten()))

model.add(CuDNNLSTM(1024, return_sequences=False))

model.add(Dense(256, activation='relu'))

# normalized target in (0,1)
model.add(Dense(1, activation='sigmoid'))

model.summary()

adam = Adam(lr=0.00005)

checkpoint = ModelCheckpoint('params_regression.hdf5',monitor='val_loss',verbose=1, save_best_only=True, mode='auto', period=1)

model.compile(optimizer=adam,loss='mse',metrics=[r2_keras])

hist = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), batch_size=16, epochs=epochs, verbose=1, callbacks=[checkpoint])
#model1 = load_model('transmission_regression.hdf5',custom_objects={'r2_keras': r2_keras})

K.set_value(adam.lr,0.00001)
hist1 = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), batch_size=16, epochs=epochs, verbose=1, callbacks=[checkpoint])

K.set_value(adam.lr,0.000001)
hist2 = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), batch_size=16, epochs=epochs, verbose=1, callbacks=[checkpoint])

score = model.evaluate(xtest, ytest, verbose=0)
print('Test loss:', score[0])
print('Test R2 Correlation:', score[1])


y_pred = model.predict(xtest)
np.savetxt('prediction_test.txt',np.c_[y_pred,ytest])

losst_hist = hist.history['loss']
lossv_hist = hist.history['val_loss']

losst_hist = np.array(losst_hist)
lossv_hist = np.array(lossv_hist)

acct_hist = hist.history['r2_keras']
accv_hist = hist.history['val_r2_keras']

acct_hist = np.array(acct_hist)
accv_hist = np.array(accv_hist)

np.savetxt('learning_curve_stage0.txt',np.c_[losst_hist,lossv_hist,acct_hist,accv_hist])

losst_hist = hist1.history['loss']
lossv_hist = hist1.history['val_loss']

losst_hist = np.array(losst_hist)
lossv_hist = np.array(lossv_hist)

acct_hist = hist1.history['r2_keras']
accv_hist = hist1.history['val_r2_keras']

acct_hist = np.array(acct_hist)
accv_hist = np.array(accv_hist)

np.savetxt('learning_curve_stage1.txt',np.c_[losst_hist,lossv_hist,acct_hist,accv_hist])

losst_hist = hist2.history['loss']
lossv_hist = hist2.history['val_loss']

losst_hist = np.array(losst_hist)
lossv_hist = np.array(lossv_hist)

acct_hist = hist2.history['r2_keras']
accv_hist = hist2.history['val_r2_keras']

acct_hist = np.array(acct_hist)
accv_hist = np.array(accv_hist)

np.savetxt('learning_curve_stage2.txt',np.c_[losst_hist,lossv_hist,acct_hist,accv_hist])
