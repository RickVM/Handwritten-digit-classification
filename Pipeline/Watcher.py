
# coding: utf-8

# In[17]:


from sklearn.externals import joblib

print("Starting evaluation")
print("Loading pipeline")
pipeline = joblib.load("pipeline.pkl")
print("pipeline loaded")


# Get evaluation data

# In[18]:


import pandas as pd
test_df = pd.read_csv("test_data.csv")
X = test_df.iloc[:,1:]
y = test_df.iloc[:,0]

y_pred = pipeline.predict(X)



# In[65]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)

print("Test results\n-------------------------------------------------------------\n")
results = {"Accuracy":accuracy, "Confusion matrix":cm.tolist()}
print("Accuracy: {0}%\n".format(accuracy))
print("Confusion matrix:")
print(cm)


# In[83]:


import json

foundResults = False

from pathlib import Path
if Path('results.json').exists():
    with open('results.json', 'r') as f:
        previousResults = json.load(f)
    foundResults = True
else:
    print("Warning, did not find any previous results!!")


# In[76]:


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


# In[80]:


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

