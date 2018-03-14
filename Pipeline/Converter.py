
# coding: utf-8

# In[6]:


import os
import glob

for file in glob.glob("*.ipynb"):
    print(os.path.basename(file)
    #nbconvert --toscript file

