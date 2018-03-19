
# coding: utf-8

# In[1]:


from sklearn.externals import joblib
import os
import pathlib
import datetime

version = str(datetime.datetime.now())

print("\nDeployer started")
print("--------------------------------------\n")
print(version)
print("Loading pipeline..")
model = joblib.load("pipeline.pkl")

print("Saving pipeline in version control..")
joblib.dump(model, "./versions/Pipeline_"+ version +".pkl")
     
print("Saving pipeline in production..")
joblib.dump(model, "Production/pipeline.pkl")
print("pipeline deployed")

