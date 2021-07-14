#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
ks2018 = pd.read_csv("ks-projects-201801.csv")
ks2018= ks2018.drop(['ID', 'name', 'category', "goal", "pledged"], axis=1)
display(ks2018[:20])


# #preprocessing

# In[2]:


ks2018['deadline'] = pd.to_datetime(ks2018['deadline'])
ks2018['launched'] = pd.to_datetime(ks2018['launched'])
ks2018.dtypes
ks2018['deadline_launched'] = ks2018['deadline'] - ks2018['launched']
ks2018['deadline_launched']= ks2018["deadline_launched"].dt.days


# In[3]:


ks2018= ks2018.drop(['launched', 'deadline', "usd_pledged_real"], axis=1)
ks2018 =ks2018[ks2018.state != "live"]
ks2018 =ks2018[ks2018.state != "undefined"]
ks2018 =ks2018[ks2018.state != "suspended"]


# In[4]:


cleanup_nums = {"state":     {'canceled':1, 'failed':2, 'successful':3}}
ks2018.replace(cleanup_nums, inplace=True)


# In[5]:


cleanup_nums = {"main_category":     {'Art':1, 'Comics':2, 'Crafts':3, 'Dance':4, 'Design':5, 'Fashion':6, 
                                      'Film & Video':7,'Food':8, 'Games':9, 'Journalism':10, 'Music':11, 
                                      'Photography':12, 'Publishing':13,'Technology':14, 'Theater':15}}
ks2018.replace(cleanup_nums, inplace=True)


# In[6]:


cleanup_nums = {"currency":     {'AUD':1, 'CAD':2, 'CHF':3, 'DKK':4, 'EUR':5, 'GBP':6, 'HKD':7, 'JPY':8, 'MXN':9,
                                 'NOK':10, 'NZD':11, 'SEK':12, 'SGD':13, 'USD':14}}
ks2018.replace(cleanup_nums, inplace=True)


# In[7]:


cleanup_nums = {"country":     {'AT':1, 'AU':2, 'BE':3, 'CA':4, 'CH':5, 'DE':6, 'DK':7, 'ES':8, 'FR':9, 'GB':10, 
                                'HK':11,'IE':12, 'IT':13, 'JP':14, 'LU':15, 'MX':16, 'N,0"':17, 'NL':18, 'NO':19, 
                                'NZ':20,'SE':21, 'SG':22, 'US':23}}
ks2018.replace(cleanup_nums, inplace=True)


# In[8]:


print(ks2018.groupby(["state"]).count())
print(ks2018['state'].unique())


# In[9]:


import keras
keras.__version__


# In[10]:


pd.set_option('float_format', '{:f}'.format)
ks2018[["usd pledged", "usd_goal_real", "deadline_launched"]].describe().loc[['mean', "std", "min", "max"]]


# In[11]:


## x en y split met nummers niet one hot
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
import numpy
# Inputs
X = numpy.array(ks2018.loc[:, ks2018.columns != 'state'], dtype='int64')
y = numpy.array(ks2018['state'].values, dtype='int64')
class_weights = class_weight.compute_class_weight('balanced',
                                                 numpy.unique(y),
                                                 y)
print(class_weights)
print(np.unique(y))


# In[12]:


from imblearn.over_sampling import SMOTE

print("Before OverSampling, counts of label '1': {}".format(sum(y==1)))
print("Before OverSampling, counts of label '2': {} ".format(sum(y==2)))
print("Before OverSampling, counts of label '3': {} \n".format(sum(y==3)))

sm = SMOTE(random_state=4)
X_res, y_res = sm.fit_sample(X, y.ravel())

print('After OverSampling, the shape of X: {}'.format(X_res.shape))
print('After OverSampling, the shape of y: {} \n'.format(y_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_res==1)))
print("After OverSampling, counts of label '2': {}".format(sum(y_res==2)))
print("After OverSampling, counts of label '2': {}".format(sum(y_res==3)))

X = X_res
y = y_res


# In[13]:


from keras.utils import to_categorical
# one hot encoder
from sklearn.preprocessing import LabelBinarizer
onehot = LabelBinarizer()
y = onehot.fit_transform(y)

print(y)


# In[14]:


######  feature scaling robust
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
print(type(X))
X = scaler.fit_transform(X)
print(X.shape)


# In[15]:


## split
import numpy as np
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=999)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# #setting up Confusion matrix
# 

# In[16]:


print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class_names = ['canceled', 'successful', 'failed']



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_test_non_category, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax.grid(False)


np.set_printoptions(precision=2)


# #knn
# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train) 


# In[ ]:


print(knn.score(X_train, y_train))
print(knn.score(X_test, y_test))


# In[ ]:


from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
print(knn.score(X_test, y_test, sample_weight=None))
y_test_non_category = [ np.argmax(t) for t in y_test]
y_predict_non_category = [ np.argmax(t) for t in y_pred ]
cm = confusion_matrix(y_test_non_category, y_predict_non_category)


# In[ ]:


import matplotlib.pyplot as plt
print('Confusion matrix:\n', cm)
print()
labels = ['Cancelled', 'Succesfull', 'Failed']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.title("KNN")
plt.show()


# In[ ]:


y_test_non_category = [ np.argmax(t) for t in y_test]
y_predict_non_category = [ np.argmax(t) for t in y_pred ]


# In[ ]:


# Plot normalized confusion matrix
plot_confusion_matrix(y_test_non_category, y_predict_non_category, classes=class_names, normalize=False,
                      title='kNN')

plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test_non_category, y_predict_non_category, target_names=['Canceled', 'Succesfull', 'Failed']))


# #MLP
# 

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation
	# create model
model = Sequential()
model.add(Dense(34, input_dim=7, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(3, activation='softmax'))
keras.optimizers.Adam(lr=0.0001, beta_1=0.0, beta_2=0.0, epsilon=None, amsgrad=False)

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)]  

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=29, 
                    verbose=1, callbacks=callbacks_list)

print(history.history.keys())


# In[ ]:


print(model.summary())


# In[ ]:


from sklearn.metrics import accuracy_score
y_pred = model.predict_classes(X_test, verbose=1)
y = to_categorical(y_pred)
print(accuracy_score(y, y_test))


# In[ ]:



import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

y_test_non_category = [ np.argmax(t) for t in y_test]
y_predict_non_category = [ np.argmax(t) for t in y ]

#print(y_test_non_category)
#print(y_predict_non_category)
conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
#print(conf_mat)

#conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)
print()
labels = ['Canceled', 'Succesfull', 'Failed']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.title("Multilayer Perceptron")
plt.show()


# In[ ]:


# Plot normalized confusion matrix
plot_confusion_matrix(y_test_non_category, y_predict_non_category, classes=class_names, normalize=False,
                      title='Multilayer Perceptron')

plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test_non_category, y_predict_non_category, target_names=['Canceled', 'Succesfull', 'Failed']))


# In[ ]:





# #prep for conv and lstm
# 
# 

# In[ ]:


import pandas as pd

state_count = ks2018.state.value_counts()
print(state_count[0:3])
# Divide by class
df_class_1 = ks2018[ks2018['state'] == 1]
df_class_2 =  ks2018[ks2018['state'] == 2]  #{'canceled':1, 'failed':2, 'successful':3}
df_class_3 =  ks2018[ks2018['state'] == 3]#failed

#print('Proportion:', round(state_count[0] / state_count[1], 3), ': 1')
state_count.plot(kind='bar', title='Count (state)');


# In[ ]:


## x en y split met nummers niet one hot
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
import numpy

# Inputs
X = numpy.array(ks2018.loc[:, ks2018.columns != 'state'], dtype='int64')
y = numpy.array(ks2018['state'].values, dtype='int64')
class_weights = class_weight.compute_class_weight('balanced',
                                                 numpy.unique(y),
                                                 y)

print(class_weights)
print(numpy.unique(y))


# In[ ]:


from imblearn.over_sampling import SMOTE

print("Before OverSampling, counts of label '1': {}".format(sum(y==1)))
print("Before OverSampling, counts of label '2': {} ".format(sum(y==2)))
print("Before OverSampling, counts of label '3': {} \n".format(sum(y==3)))

sm = SMOTE(random_state=4)
X_res, y_res = sm.fit_sample(X, y.ravel())

print('After OverSampling, the shape of X: {}'.format(X_res.shape))
print('After OverSampling, the shape of y: {} \n'.format(y_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_res==1)))
print("After OverSampling, counts of label '2': {}".format(sum(y_res==2)))
print("After OverSampling, counts of label '2': {}".format(sum(y_res==3)))

X= numpy.array(X_res)
y= numpy.array(y_res)


# In[ ]:


from keras.utils import to_categorical
# one hot encoder
from sklearn.preprocessing import LabelBinarizer
onehot = LabelBinarizer()
y = onehot.fit_transform(y)

print(y)


# In[ ]:


######  feature scaling robust
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
print(type(X))
X = scaler.fit_transform(X)
print(X.shape)


# In[ ]:


## split
import numpy as np
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=63)
print(X_train.shape)
print(y_train)
print(X_test.shape)
print(y_test)
X_train = X_train.reshape(X_train.shape[0], 1, 7)
X_test = X_test.reshape(X_test.shape[0], 7, 1)


# #lstm
# 

# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np

data_dim = 1
timesteps =  7


# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))# returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(3, activation='softmax'))

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)]

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=200, epochs=50,
          validation_split=0.2, 
          callbacks=callbacks_list)


# In[ ]:


print(model.summary())


# In[ ]:


from sklearn.metrics import accuracy_score

y_pred = model.predict_classes(X_test, verbose=1)

print(y_test)
y = to_categorical(y_pred)
print(accuracy_score(y, y_test))


# In[ ]:


from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

y_test_non_category = [ np.argmax(t) for t in y_test]
y_predict_non_category = [ np.argmax(t) for t in y ]

#print(y_test_non_category)
#print(y_predict_non_category)
conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
#print(conf_mat)

#conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)
print()
labels = ['Canceled', 'Succesfull', 'Failed']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.title("Long Short-Term Memory")
plt.show()


# In[ ]:


# Plot normalized confusion matrix
plot_confusion_matrix(y_test_non_category, y_predict_non_category, classes=class_names, normalize=False,
                      title='Long Short-Term Memory')

plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test_non_category, y_predict_non_category, target_names=['Canceled', 'Succesfull', 'Failed']))


# #CONVNET

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.layers import Reshape, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import Dropout, GaussianNoise, LSTM

model_m = Sequential()
model_m.add(Conv1D(200, 5, activation='relu', input_shape=(7, 1))) #200 in begin 0.7522342314092222 mooi resultaat 0.752557923113105
model_m.add(Conv1D(200, 1, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(260, 1, activation='relu')) 
model_m.add(Conv1D(260, 1, activation='relu'))
model_m.add(GlobalAveragePooling1D())

model_m.add(Dropout(0.1))

model_m.add(Dense(3, activation='softmax'))

#keras.optimizers.Adam(lr=0.1,  # vandert voorheen 0.01
 #                     beta_1=0.9, # 0.9 0.834012228205855
  #                    beta_2=0.9, # 0,5 0.8294610681459287 meer stappen
  #                    epsilon=None, 
  #                    decay=0.0, 
   #                   amsgrad=False) 

print(model_m.summary())  


# In[ ]:


from sklearn.utils import class_weight
from keras import optimizers

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)]


model_m.compile(loss='categorical_crossentropy',
                optimizer="adam", metrics=['accuracy'])

BATCH_SIZE = 150
EPOCHS = 50

history = model_m.fit(X_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1) #zo laten


# In[ ]:


import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


from sklearn.metrics import accuracy_score

y_pred = model_m.predict_classes(X_test, verbose=1)

print(y_test)
y = to_categorical(y_pred)
print(accuracy_score(y, y_test))


# In[ ]:


from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

y_test_non_category = [ np.argmax(t) for t in y_test]
y_predict_non_category = [ np.argmax(t) for t in y ]

#print(y_test_non_category)
#print(y_predict_non_category)
conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
#print(conf_mat)

#conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)
print()
labels = ['Canceled', 'Succesfull', 'Failed']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.title("ConvNets")
plt.show()


# In[ ]:


# Plot normalized confusion matrix
plot_confusion_matrix(y_test_non_category, y_predict_non_category, classes=class_names, normalize=False,
                      title='ConvNets')

plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test_non_category, y_predict_non_category, target_names=['Canceled', 'Succesfull', 'Failed']))


# In[ ]:





# #examples prediction of new data
# 

# In[ ]:


Xnew =[[11,	14,	1.,	23,	1.00,		5000.00, 34],
      [8, 14,	16,	23,	1205.00,	1000.00,19],
      [9,	14,	0,	23,	0.00,	29000, 49],  
      [9,	6,	761,	10,	57763.78,	6469.73, 27], 
      [5,	6,	647,	23,	39693.00,		25000.00,	39], 
      [3,	4,	60,	7,	3059.73,	8442.45,	23]] 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

Xnew = np.asarray(Xnew)
print(Xnew.shape)
scaler = RobustScaler()
Xnew = scaler.fit_transform(Xnew)

Xnew = Xnew.reshape(6, 7, 1)
ynew = model_m.predict_classes(Xnew)
ycat = to_categorical(ynew, 3)


for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % ( i , ycat[i]))

