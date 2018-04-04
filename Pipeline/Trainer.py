
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd

classicML = True #Used in saving process 
preprocessing = True

print("\n\nStarting model training..")
print("--------------------------------------\n")
train_df = pd.read_csv("data.csv")
X = train_df.iloc[:,1:]
y = train_df.iloc[:,0]


# In[11]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler



if classicML:
    # XGBoost Code
    import xgboost as xgb
    from xgboost import XGBClassifier
    #Default settings - Now deprecated with optimized settings. 
    #pipeline = Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(0, 1), copy=True)),
    #                            ('XgbClassifier', XGBClassifier(objective = 'multi:softmax'))
    #                           ])
    #Based upon random search results.
    #Accuracy score was 91% on 20% of the training data.
    if preprocessing:
        pipeline = Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(0, 1), copy=True)),
                                   ('XgbClassifier', XGBClassifier(objective = 'multi:softmax',
                                                                   gamma = 0.9,
                                                                   learning_rate = 0.375,
                                                                   max_depth = 9,
                                                                   n_estimators = 9))])
    else:
         pipeline = Pipeline(steps=[('XgbClassifier', XGBClassifier(objective = 'multi:softmax',
                                                                   gamma = 0.9,
                                                                   learning_rate = 0.375,
                                                                   max_depth = 9,
                                                                   n_estimators = 9))])

else:
    pipeline = Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(0, 1), copy=True))  ])

    from keras.models import Model
    from keras.models import Sequential
    from keras.layers import Input, Dense
    from keras.losses import sparse_categorical_crossentropy
    from keras.optimizers import SGD
    def create_model():
        model = Sequential()
        #units = (Input+Output)/2
        model.add(Dense(units = (int(round(len(X.columns)+9)/2)), activation = 'relu', input_dim= len(X.columns))) 
        model.add(Dense(10, activation = 'softmax'))
        model.compile(loss=sparse_categorical_crossentropy,
                      optimizer='adam', metrics = ['accuracy'])
        #SGD(lr=0.01, momentum=0.9, nesterov=True)
        return model
    classifier = create_model()


# By creating a pipeline we can run one line of code to pre-process our data and train our model.
# Later on this pipeline will also enable us to only need 1 line of code to pre-process and make predictions on new data.
# Hence the code will be a lot cleaner.
# 
# Pipeline.steps can be called to view the all the components and parameters that make up the pipeline.

# In[12]:


# Random search code

# X = train_df.iloc[:,1:]
# y = train_df.iloc[:,0]

# from scipy.stats import randint
# import scipy
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.base import BaseEstimator, TransformerMixin
# import xgboost as xgb
# from xgboost import XGBClassifier
# from sklearn.model_selection import RandomizedSearchCV

# scaler = MinMaxScaler(feature_range=(0,1), copy=True)
# X = scaler.fit_transform(X)

# params = {'max_depth': randint(1, 10) ,
#           'learning_rate': scipy.stats.expon(scale=0.5),
#           'n_estimators': randint(1, 10),
#           'gamma': scipy.stats.expon(scale=1)
#            }


# import datetime
# print("\n\nStarting parameter search..")
# print("--------------------------------------\n")
# tstart = datetime.datetime.now()
# optimizer = RandomizedSearchCV(XGBClassifier(objective = 'multi:softmax'), params, n_iter = 25)
# optimizer.fit(X, y)
# tstop = datetime.datetime.now()
# tdelta = tstop - tstart
# print("Finished training.")
# print("Training duration in (Days/Hours/Seconds/Milliseconds): {0}".format(tdelta)) 
# print(optimizer.score()) #0.91130952380952379
# print(optimizer.best_params_) #{'gamma': 0.90390078036156596,
#                               #'learning_rate': 0.37483528867120858,
#                               #'max_depth': 9,
#                               #'n_estimators': 9}


# In[13]:


import datetime

tstart = datetime.datetime.now()
if classicML:
    pipeline.fit(X, y)
else:
    if preprocessing:
        X = pipeline.fit_transform(X, y)
    val_split = 0.1
    history = classifier.fit(X, y, batch_size = 128, epochs = 50, verbose = 1, validation_split=val_split)

tstop = datetime.datetime.now()
tdelta = tstop - tstart
print("Finished training.")
print("Training duration in (/Hours/Minutes/Seconds/Milliseconds): {0}".format(tdelta))


# Then we train our model

# In[14]:


if classicML:
    #If you want to visualize the xgboost decision tree..
    clf = pipeline.steps[1]
    graph = xgb.to_graphviz(clf[1])
    graph.render('pipelinetree')
else:
    #If you want to visualize the deep learning classifier
    history.history.keys()
    import matplotlib.pyplot as plt
    import seaborn
    history.history.keys()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title("Training accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Acccuracy")
    plt.legend(['Train accuracy', 'Validation accuracy'])
    plt.savefig("train_plot")


# In[15]:


if classicML:
    trainSize = len(train_df['pixel0'])
else:
    trainSize = len(train_df['pixel0']) * (1 - val_split)
    valSize = trainSize * val_split #Not used atm, but perhaps we want it later on.
    
Trainlog = {"TrainSize":trainSize, "Time":tdelta.total_seconds()}
def log_training():
    df = pd.DataFrame.from_dict(Trainlog, orient = 'Index')
    df.to_excel("train_log.xlsx")
log_training()


# We dont evaluate the model as this is up to the watcher, thus we save it now.

# In[11]:


from sklearn.externals import joblib

if classicML:
    joblib.dump(pipeline, "pipeline.pkl")
    print("Pipeline has been saved as pipeline.pkl")
else:
    joblib.dump(pipeline, "pipeline.pkl")
    print("Pipeline has been saved as pipeline.pkl")
    classifier.save('model.h5')
    print("Model has been saved as model.h5")


# In[16]:


trainSize

