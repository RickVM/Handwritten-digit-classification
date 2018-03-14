
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

train_df = pd.read_csv("data.csv")


# First we preprocess our data

# In[3]:


X = train_df.iloc[:,1:]
y = train_df.iloc[:,0]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
X = scaler.fit_transform(X)


# Then we train our model

# In[6]:


import xgboost as xgb

print("Starting model training..")
clf = xgb.XGBClassifier(objective = 'multi:softmax')
clf.fit(X, y)
print("Training finished")


# We dont evaluate the model as this is up to the watcher, thus we save it now.

# In[7]:


from sklearn.externals import joblib
joblib.dump(clf, "model.pkl")
print("Model has been saved as model.pkl")


# In[ ]:


import os
os.system("python Watcher.py")

