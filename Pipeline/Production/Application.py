
# coding: utf-8

# In[4]:


from sklearn.externals import joblib

classicML = False #Deep learning or classic model

print("Starting application..")
print("--------------------------------------\n")
print("Loading pipeline..")
pipeline = joblib.load("pipeline.pkl")
print("pipeline loaded")

if classicML == False:
    from keras.models import load_model
    print("Loading model")
    model = load_model("model.h5")
    print("Model loaded")
    
import pandas as pd
test_df = pd.read_csv("./Production/Test.csv") #Relative path from ../ as this program is called from there, change if running local

X = test_df.iloc[:,:]

print("Running predictions..")
if classicML:
    predictions = pipeline.predict(X)
else:
    X = pipeline.transform(X)
    predictions = model.predict_classes(X)
print("Predictions made.")


# In[17]:


#predictions = model.predict()


# In[16]:


length = len(predictions)
ImageIds = []
for i in range(length):
    ImageIds.append(i+1)
    
data_to_submit = pd.DataFrame({'ImageId':ImageIds,
                               'Label':predictions})
data_to_submit.to_csv("Results.csv", index = False)
print("Predictions saved as Results.csv")

import subprocess

command = "kaggle competitions submit -c digit-recognizer -f Results.csv -m \"ANN_with_scaling_amount_of_training_data\""

process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
#output, error = process.communicate()
print(output)
print("Predictions submitted to kaggle")


# In[3]:


# #Use these commands after initiating docker to setup the tools required for kaggle api usage
#!pip install kaggle
#!mkdir /root/.kaggle
#!cp ./kaggle.json /root/.kaggle/kaggle.json

#If output file contains kaggle.json the operation was succesfull
#!ls -a /root/.kaggle


# In[2]:




