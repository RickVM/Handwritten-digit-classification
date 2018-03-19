
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

print("\n\nStarting model training..")
print("--------------------------------------\n")
train_df = pd.read_csv("data.csv")


# # Random search code
# 
# X = train_df.iloc[:,1:]
# y = train_df.iloc[:,0]
# 
# from scipy.stats import randint
# import scipy
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.base import BaseEstimator, TransformerMixin
# import xgboost as xgb
# from xgboost import XGBClassifier
# from sklearn.model_selection import RandomizedSearchCV
# 
# scaler = MinMaxScaler(feature_range=(0,1), copy=True)
# X = scaler.fit_transform(X)
# 
# params = {'max_depth': randint(1, 10) ,
#           'learning_rate': scipy.stats.expon(scale=0.5),
#           'n_estimators': randint(1, 10),
#           'gamma': scipy.stats.expon(scale=1)
#            }
# 
# 
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

# By creating a pipeline we can run one line of code to pre-process our data and train our model.
# Later on this pipeline will also enable us to only need 1 line of code to pre-process and make predictions on new data.
# Hence the code will be a lot cleaner.
# 
# Pipeline.steps can be called to view the all the components and parameters that make up the pipeline.

# In[23]:


X = train_df.iloc[:,1:]
y = train_df.iloc[:,0]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


# XGBoost Code

import xgboost as xgb
from xgboost import XGBClassifier
#Default settings - Now deprecated with optimized settings. 
#pipeline = Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(0, 1), copy=True)),
#                            ('XgbClassifier', XGBClassifier(objective = 'multi:softmax'))
#                           ])

#Based upon random search results.
#Accuracy score was 91% on 20% of the training data.

pipeline = Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(0, 1), copy=True)),
                           ('XgbClassifier', XGBClassifier(objective = 'multi:softmax',
                                                           gamma = 0.9,
                                                           learning_rate = 0.375,
                                                           max_depth = 9,
                                                           n_estimators = 9))])


# # Deeplearning Code
# X = train_df.iloc[:,1:]
# y = train_df.iloc[:,0]
# 
# 
# from sklearn.base import BaseEstimator, TransformerMixin
# from keras.models import Sequentialquential
# from keras.layers import Dense
# 
# model = Sequential()
# model.add(Dense(units = X.columns), activation = 'relu' input_dim=((X.columns+9)/2) #(Input+Output)/2
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
# 
# pipeline = Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(0, 1), copy=True)),
#                             ('DeepNet', model))
#                            ])

# In[ ]:


import datetime

tstart = datetime.datetime.now()
pipeline = pipeline.fit(X, y)
tstop = datetime.datetime.now()
tdelta = tstop - tstart
print("Finished training.")
print("Training duration in (Days/Hours/Seconds/Milliseconds): {0}".format(tdelta))
pipeline.steps


# Then we train our model

# We dont evaluate the model as this is up to the watcher, thus we save it now.

# In[24]:


from sklearn.externals import joblib
joblib.dump(pipeline, "pipeline.pkl")
print("Pipeline has been saved as pipeline.pkl")

