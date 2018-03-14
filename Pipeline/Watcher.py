
# coding: utf-8

# In[10]:


from sklearn.externals import joblib

print("Starting evaluation")
print("Loading model")
model = joblib.load("model.pkl")
print("Model loaded")


# Get evaluation data

# In[15]:


import pandas as pd

test_df = pd.read_csv("test_data.csv")

X = test_df.iloc[:,1:]
y = test_df.iloc[:,0]
#Temp step, this should be integrated in a seperate 'pre-processing' program
#Normalize test data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
X = scaler.fit_transform(X)


# In[16]:


y_pred = model.predict(X)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)

print("Test results\n-------------------------------------------------------------\n")
print("Accuracy: {0}%\n".format(accuracy))
print("Confusion matrix:")
print(cm)


# In[19]:


if accuracy > 85:
    print("Model passed the test.")
    print("Executing deployer")
    import os
    os.system("python Deployer")
    

