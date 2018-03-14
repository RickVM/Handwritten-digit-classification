
# coding: utf-8

# In[35]:


from sklearn.externals import joblib
import os
import pathlib
import datetime

version = str(datetime.datetime.now())
print(version)
print("Loading model..")
model = joblib.load("model.pkl")

print("Saving model in version control..")
joblib.dump(model, "./versions/model_"+ version +".pkl")
     
print("Saving model in production..")
joblib.dump(model, "Production/model.pkl")
print("Model deployed")

