
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

train_df = pd.read_csv("data.csv")


# By creating a pipeline we can run one line of code to pre-process our data and train our model.
# Later on this pipeline will also enable us to only need 1 line of code to pre-process and make predictions on new data.
# Hence the code will be a lot cleaner.
# 
# Pipeline.steps can be called to view the all the components and parameters that make up the pipeline.

# In[4]:


X = train_df.iloc[:,1:]
y = train_df.iloc[:,0]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
from xgboost import XGBClassifier

pipeline = Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(0, 1), copy=True)),
                            ('XgbClassifier', XGBClassifier(objective = 'multi:softmax'))
                           ])
print("\n\nStarting model training..")
print("--------------------------------------\n")
pipeline = pipeline.fit(X, y)
print("Finished training.")
pipeline.steps


# Then we train our model

# We dont evaluate the model as this is up to the watcher, thus we save it now.

# In[3]:


from sklearn.externals import joblib
joblib.dump(pipeline, "pipeline.pkl")
print("Pipeline has been saved as pipeline.pkl")

