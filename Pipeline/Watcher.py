
# coding: utf-8

# In[1]:


from sklearn.externals import joblib
import dill

classicML = True
preprocessing = True

print("\nStarting evaluation..")
print("--------------------------------------\n")
print("Loading pipeline")
pipeline = joblib.load("pipeline.pkl")
print("pipeline loaded")

if classicML == False:
    from keras.models import load_model
    print("Loading model")
    classifier = load_model("model.h5")
    print("Model loaded")


# Get evaluation data

# In[2]:


import pandas as pd
test_df = pd.read_csv("test_data.csv")
X = test_df.iloc[:,1:]
y = test_df.iloc[:,0]

print("Running predictions")
if classicML == False:
    if preprocessing:
        X = pipeline.transform(X)
    y_pred = classifier.predict_classes(X)
    y_proba = classifier.predict_proba(X)
else:
    y_pred = pipeline.predict(X)



# In[3]:


def getResults(y_true, y_pred, verbose = True):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
   
    #Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    #auc_global = roc_auc_score(y, y_pred, average = 'micro')
    #auc_per_class = roc_auc_score(y, y_pred)
    prfs_global = precision_recall_fscore_support(y_true, y_pred, average = 'micro')
    
    pr_score_global = precision_score(y_true, y_pred, average = 'micro')
    pr_score_per_class = precision_score(y_true, y_pred, average = None)
    recall_score_global = recall_score(y_true, y_pred, average = 'micro')
    recall_score_per_class = recall_score(y_true, y_pred, average = None)
    f1_score_global = f1_score(y_true, y_pred, average = 'micro')
    f1_score_per_class = f1_score(y_true, y_pred, average = None)
    
    prfs_per_class = precision_recall_fscore_support(y_true, y_pred)
    testSize = len(test_df.loc[:,'pixel0'])
    
    #Get trainlog-results
    logname = "train_log.xlsx"
    train_log_df = pd.read_excel(logname)
    train_time = train_log_df.loc['Time'].values
    trainSize = train_log_df.loc['TrainSize'].values
    
    if verbose:
        #Visualize Global(Averaged if mutliclass) results
        print("Test results\n-------------------------------------------------------------\n")
        print("Global statistics")
        print("Accuracy: {0}%".format(accuracy))
        print("Matthews correlation coefficient: {0}".format(mcc))
        print("Precision: {0}".format(pr_score_global))
        print("Recall: {0}".format(recall_score_global))
        print("F1-Score: {0}".format(f1_score_global))

        #Visualize and save per-class results
        print("\nPer-Class statistics")
        print("Precision: {0}".format(pr_score_per_class))
        print("Recall: {0}".format(recall_score_per_class))
        print("F1-Score: {0}".format(f1_score_per_class))
        print("\n")
        print("\nConfusion matrix:")
        print(cm)

    results = {"Train_size":trainSize[0], "Train_time":train_time[0], "Test_size":testSize, "Accuracy":accuracy, 
                   "MCC":mcc, "Precision_global:":pr_score_global, "Recall_global:":recall_score_global,
                   "F1_global:":f1_score_global
              } 
    
    #Now update with per-class scores.
    for classNr in range(0, cm.shape[0], 1):
        base = "_class_" + str(classNr)
        pr = "Precision" + base 
        re = "Recall" + base 
        f1 = "F1" + base
        results.update({pr:pr_score_per_class[classNr], re:recall_score_per_class[classNr], f1:f1_score_per_class[classNr]})
        
    return results

def results_to_excel(Results, Filename):
    from pathlib import Path
    #results_df = pd.DataFrame.from_dict(Results, orient = 'columns')
    results_df = pd.DataFrame([Results])
    
    file = Path(Filename)
    if file.exists():
        print("Found previous averaged metric results, appending new results.")
        oldresults = pd.read_excel(Filename)
        results_df = results_df.append(oldresults, ignore_index = True)
        results_df = results_df.sort_values("Train_size")
        print(results_df)
    else:
        print("Found no previous results, making a new file with results.")
    results_df.to_excel(Filename, index = False)


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
    
def modelPassed(Results, cm):
    #Save the results as the current model
    filename = "metric_results.xlsx"
    
    Results.update({"Confusion_matrix:":cm.tolist()})
    results_to_excel(Results, filename)
    print("----------------")
    print(Results)
    
    with open('results.json', 'w') as f:
        json.dump(Results, f, ensure_ascii=False, sort_keys=False, indent = 4) #Dump confusion_matrix
    #Save the results in version control
    with open('./Versions/Pipeline_' + version +'_results.json', 'w') as f:
        json.dump(Results, f, ensure_ascii=False, sort_keys=True, indent = 4) #Dump confusion_matrix


def modelFailed(Results, cm):  
    print("Saving results..")
    Results.update({"Confusion_matrix:":cm.tolist()})
    filename = "./Versions/Failed/metric_results_failed.xlsx"
    results_to_excel(Results, filename)

    
    with open('./Versions/Failed/Pipeline_' + version +'_results.json', 'w') as f:
        json.dump(Results, f, ensure_ascii=False, sort_keys=True, indent = 4)
    print("Results saved.")
    import sys
    sys.exit(1)


# In[6]:


import datetime
version = str(datetime.datetime.now())

results = getResults(y, y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

if foundResults:
    if compareResults(results, previousResults):
        modelPassed(results, cm)
    else:
        modelFailed(results, cm)
else:
    modelPassed(results, cm)

