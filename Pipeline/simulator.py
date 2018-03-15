
# coding: utf-8

# This simulator has 2 functions:
# 1. It splits the training and test data used by the programs.
# 2. It executes the pipeline and runs the production application once.

# Step 1. Split training and test data

# In[1]:


import pandas as pd

def splitData(Trainsize, Testsize):
    #Enter data
    count = 42000
    trainsize = Trainsize #As percentage. Example: 25% should be entered as 0.25
    testsize = Testsize #As percentage.

    trainsize = int(count*trainsize) 
    testsize = int(count*testsize)
    print("Train size is {0}".format(trainsize))
    print("Test size is {0}".format(testsize))

    df = pd.read_csv("../Train.csv")

    
    
    
    test_df = df.iloc[count-testsize:,:]
    train_df = df.iloc[:trainsize,:]

    train_df.to_csv("data.csv", index = False)
    test_df.to_csv("test_data.csv", index = False)
    print("Data has been split and saved as data.csv and test_data.csv")


# Step 2. Execute pipeline and run the production application.

# In[2]:


import os

def runPipeline():
    print("\nStarting trainer")
    os.system("python Trainer.py")

    print("\nExecuting watcher")
    if(os.system("python Watcher.py") == 0):
        print("\nExecuting deployer")
        if(os.system("python Deployer.py") == 0):
            print("Deployment succesfull.")
            print("Running application test")
            os.system("python ./Production/Application.py")
        else:
            print("Deployment failed.")
    else:
        print("Model did not pass the test, aborting.")


# In[3]:


def frange(start, stop, step):
    i = start
    while i <= stop:
        yield i
        i+=step
    

for i in frange(0.001, 0.02, 0.004):
    splitData(i, 0.2)
    runPipeline()
    
#Run once more, the accuracy should decrease with the result that the model should not be deployed.
splitData(0.005, 0.2)
runPipeline()

