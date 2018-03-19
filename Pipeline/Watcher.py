
# coding: utf-8

# In[1]:


from sklearn.externals import joblib

print("\nStarting evaluation..")
print("--------------------------------------\n")
print("Loading pipeline")
pipeline = joblib.load("pipeline.pkl")
print("pipeline loaded")


# Get evaluation data

# In[2]:


import pandas as pd
test_df = pd.read_csv("test_data.csv")
X = test_df.iloc[:,1:]
y = test_df.iloc[:,0]

y_pred = pipeline.predict(X)
#X.describe()


# In[7]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)

print("Test results\n-------------------------------------------------------------\n")
results = {"Accuracy":accuracy, "Confusion matrix":cm.tolist()} #TODO: Also include used train and test size
print("Accuracy: {0}%\n".format(accuracy))
print("Confusion matrix:")
print(cm)


# In[4]:


import json

foundResults = False

from pathlib import Path
if Path('results.json').exists():
    with open('results.json', 'r') as f:
        previousResults = json.load(f)
    foundResults = True
else:
    print("Warning, did not find any previous results!!")


# In[5]:


def compareResults(Results, PreviousResults):
    print("Comparing results")
    if(Results['Accuracy'] >= PreviousResults['Accuracy']):
        print("\nModel passed the test.")
        return True
    else:
        print("Model did not pass the test.")
        return False
    
def modelPassed():
    #Save the results as the current model
    with open('results.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, sort_keys=True, indent = 4)
    #Save the results in version control
    with open('./Versions/Pipeline_' + version +'_results.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, sort_keys=True, indent = 4)

def modelFailed():  
    print("Saving results..")
    with open('./Versions/Failed/Pipeline_' + version +'_results.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, sort_keys=True, indent = 4)
    print("Results saved.")
    import sys
    sys.exit(1)


# In[6]:


import datetime
version = str(datetime.datetime.now())

if foundResults:
    if compareResults(results, previousResults):
        modelPassed()
    else:
        modelFailed()
else:
    print("Deploying model without previous results.")
    modelPassed()

