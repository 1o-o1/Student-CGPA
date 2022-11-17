from tensorflow import keras
import tensorflow as tf # Imports tensorflow
import tensorflow_addons as tfa


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization,Embedding
from tensorflow.keras.layers import Conv2D, MaxPooling2D,LSTM
from tensorflow.keras import regularizers, optimizers,losses
from tensorflow.keras.metrics import Recall,Precision,AUC,TruePositives,TrueNegatives,FalseNegatives,FalsePositives
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd 
import matplotlib
import seaborn as sns
import sklearn
import imblearn
import matplotlib.pyplot as plt
import time
import sklearn.metrics as m

from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn import metrics

name ="RNN"
x= pd.read_csv("dataset.csv")

y = pd.read_csv("label.csv")

print(x)
print(y)
x=np.array(x).astype('float32')
y=np.array(y).astype('float32')
for i in range(y.size):
    if y[i]>=3.75:
         y[i]=5
    elif y[i]>=3.25:
         y[i]=4
    elif y[i]>=2.75:
         y[i]=3
    elif y[i]>=2.25:
         y[i]=2
    elif y[i]>=2:
         y[i]=1
    else:
        y[i]=0

y = to_categorical(y)
print(x.shape)
print(y.shape)
x=x.reshape(-1,1,19)
y=y.reshape(-1,1,6)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=24, test_size=0.2)

model = Sequential()
model.add(LSTM(32,input_shape=(1,19),dropout=0.1,return_sequences=True))
model.add(LSTM(64,dropout=0.1,return_sequences=True))
model.add(LSTM(128,dropout=0.1,return_sequences=True))
model.add(Dense(64,input_dim=19,activation="relu"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.1))
model.add(Dense(512,activation="relu"))
model.add(Dense(1024,activation="relu"))
model.add(Dense(512,activation="relu"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.1))
model.add(Dense(128,activation="relu"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.1))
model.add(Dense(6,activation="softmax"))
print(model.summary())
plot_model(model, to_file=name+'.png',show_shapes= True , show_layer_names=True)

model.compile(loss = keras.losses.BinaryCrossentropy(),
                optimizer=keras.optimizers.Adam(),
                metrics=['mae', 'msle','acc',Recall(),Precision(),AUC(),TruePositives(),TrueNegatives(),FalseNegatives(),FalsePositives()])

history = model.fit(X_train, y_train,
          batch_size=526,
          epochs=200,
          validation_data=(X_test, y_test), 
          verbose=1)

model.save(name+'.h5')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower left')
plt.savefig(name+'_acc.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(name+'_loss.png')
plt.show()

"""prd =model.predict(x)
a= np.asarray(prd)
print(prd)
pd.DataFrame(a).to_csv("prediction_"+name+"_.csv")
pd.DataFrame(y).to_csv("true_"+name+"_.csv")"""
pd.DataFrame.from_dict(history.history).to_csv(name+'_history.csv',index=False)

