# Handwritten-digit-classification
Classification of digits with the MNIST dataset

# Purpose
The purpose of this project was to:
* Test deep learning performance vs XGBoost performance on MNIST
* Explore and visualize the variety of metrics that can be used for classification performance
* Learn how to save and deploy SKLearn and Keras/Tensorflow models.
* Create a machine learning pipeline that can automatically scale and improve with data whilst guaranteeing improved performance in order to gain insights on the subject.

# The pipeline consists of:
* Simulator      (Executes the full pipeline based upon input parameters.)  
* Model trainer  (Trains the model)  
* Model watcher  (Benchmarks the model performance against a test-set and saves metrics. Returns a pass or fail.)  
* Deployer  (If model passed the deployer saves the model in production)  
* Application (Runs model predictions against Kaggle test-set and submits these with the Kaggle API.)  

# Usage
The simulator can be used to run and adjust the full program aswell as the Train/test dataset size.   After making adjustments to any of the .ipynb files you need to convert the files to .py, this can be done with converter

When running the simulator an extensive classification performance metrics report is automatically created in /pipeline/metric_results  
Pre-recorded metrics are available in /  
After every training the model and its results are saved in /Pipeline/Versions  
If the model passed it is automatically saved in Production and the 'production app' is ran, this means that predictions will be done on the kaggle test-dataset
and submitted through the kaggle API.  
Note that for this to work you need to get a kaggle key, save it it in production and use the last codeblock in the application notebook.  
If it failed it is saved in /Pipeline/Versions/Failed/

#Findings
* Next time I'd rather generate performance reports through the use of matplotlib and seaborn rather than Excel, I feel this would better suit an automated environment.
* Even with a very small dataset deep learning starts with better performance than XGBoost
* Deep learning surpasses XGBoost performance overall
* Deep learning is actually faster on this dataset rather than XGBoost

