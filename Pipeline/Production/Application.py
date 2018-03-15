
# coding: utf-8

# In[19]:


from sklearn.externals import joblib

print("Starting application..")
print("Loading pipeline..")
pipeline = joblib.load("pipeline.pkl")
print("pipeline successfully loaded")

import pandas as pd
test_df = pd.read_csv("../../Test.csv")

X = test_df.iloc[:,:]

print("Running predictions")
predictions = pipeline.predict(X)


# In[17]:


#predictions = model.predict()


# In[18]:


length = len(predictions)
ImageIds = []
for i in range(length):
    ImageIds.append(i+1)
    
data_to_submit = pd.DataFrame({'ImageId':ImageIds,
                               'Label':predictions})
data_to_submit.to_csv("Results.csv", index = False)
print("Predictions saved as Results.csv")

