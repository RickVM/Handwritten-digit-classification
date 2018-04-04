
# coding: utf-8

# In[6]:


import os
import glob
from pathlib import Path

for file in (Path('.').glob("**/*.ipynb")):
    if(file.name.find("checkpoint") == -1):
        print("\nFound: {0}".format(file))
        get_ipython().system('jupyter nbconvert --to script $file')

