from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM


model = Sequential()
# model.add(Dense(10,input_dim=))
model.add(LSTM(10, input_length=5, input_dim=10))

#model.add(LSTM(10, input_shape=(5,10)))
#model.add(Dense(25, activation=('tanh')))
model.add(Dense(25, activation=('sigmoid')))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse',optimizer='rmsprop')


#data 
import scipy.io as sio
import numpy as np

matfn = 'mydata.mat'
data = sio.loadmat(matfn)
#data {'LinkU':,'__version__':,'__header__':,'__globals__':}
linku = data['LinkU']
length = len(linku)
list_9996 = []
for i in range(0,length-4):
    list_9996.append(linku[i:i+5])

myoutput = 'output.mat'
outputdata = sio.loadmat(myoutput)

datainput=list_9996
dataoutput=outputdata["Delay_output"]
dataoutput=np.array(dataoutput)
# train the model, iterating on the data in batches
# of 32 samples
datainput = np.array(datainput)
#print(datainput.shape)
#print(datainput.shape)
model.fit(datainput, dataoutput,validation_split=0.1, nb_epoch=10, batch_size=32)
result = model.predict(datainput)
print(result)
model.summary()


