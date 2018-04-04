
# coding: utf-8

# In[26]:


from sklearn.externals import joblib
import dill

import os
import pathlib
import datetime

version = str(datetime.datetime.now())

print("\nDeployer started")
print("--------------------------------------\n")
print(version)
classicML = True
pipeline_name = 'pipeline'
model_name = 'model'


# In[27]:


def savePipeline():
    print("Loading pipeline")
    pipeline = joblib.load("pipeline.pkl")
    print("pipeline loaded")
    print("Saving pipeline in version control..")      
    joblib.dump(pipeline, "./versions/"+ pipeline_name + version +".pkl")
    print("Saving pipeline in production..")
    joblib.dump(pipeline, "Production/"+ pipeline_name + ".pkl")
    print("pipeline deployed")

def saveDeepModel():
    from keras.models import load_model
    print("Loading model")
    classifier = load_model("model.h5")
    print("Model loaded")
    print("Saving model in version control..")
    classifier.save('versions/'+ model_name + version + '.h5')
    print("Model saved in version control")
    print("Saving model in production..")
    classifier.save('Production/' + model_name + '.h5')
    print("Model saved in production.")


# In[28]:


savePipeline()
if classicML == False:
    saveDeepModel()

