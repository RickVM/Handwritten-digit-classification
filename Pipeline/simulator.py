
# coding: utf-8

# In[27]:


import pandas as pd
jupyter nbconvert --to script
#Enter data
count = 42000
trainsize = 0.02 #As percentage. Example: 25% should be entered as 0.25
testsize = 0.2 #As percentage.

trainsize = int(count*trainsize) 
testsize = int(count*testsize)
print("Train size is {0}".format(trainsize))
print("Test size is {0}".format(testsize))

df = pd.read_csv("../Train.csv")

test_df = df.iloc[count-testsize:,:]
train_df = df.iloc[:trainsize,:]


# In[28]:


train_df.describe()


# In[31]:


train_df.to_csv("data.csv", index = False)
test_df.to_csv("test_data.csv", index = False)
print("Data has been split and saved as data.csv and test_data.csv")


# In[ ]:


get_ipython().system('jupyter nbconvert --to script')

