# Handwritten-digit-classification
Classification of digits with the MNIST dataset


# The pipeline consists of:
*Simulator      (Executes the full pipeline based upon input parameters.)  
*Model trainer  (Trains the model)  
*Model watcher  (Benchmarks the model performance against a test-set and saves metrics. Returns a pass or fail.)  
*Deployer  (If model passed the deployer saves the model in production)  
*Application (Runs model predictions against Kaggle test-set and submits these with the Kaggle API.)  

# Usage
The simulator can be used to run and adjust the full program aswell as the Train/set dataset size.
After making adjustments to any of the .ipynb files you need to convert the files to .py, this can be done with converter

When running the simulator an extensive classification performance metrics report is automatically created in /pipeline/metric_results  
Pre-recorded metrics are available in /  
After every training the model and its results are saved in /Pipeline/Versions  
If the model passed it is automatically saved in Production and the 'production app' is ran, this means that predictions will be done on the kaggle test-dataset
and submitted through the kaggle API.  
Note that for this to work you need to get a kaggle key, save it it in production and use the last codeblock in the application notebook.  
If it failed it is saved in /Pipeline/Versions/Failed/