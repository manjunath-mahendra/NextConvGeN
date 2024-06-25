import os
from nbconvert import NotebookExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import read, write
import warnings
warnings.filterwarnings("ignore")

# Get current working directory of a process
notebook_dir = os.getcwd() 

# Exporter class to export a notebook object
exporter = NotebookExporter()

for root, dirs, files in os.walk(notebook_dir):
    for file in files:
        if file.endswith('.ipynb') and not file.endswith('-checkpoint.ipynb'):
            notebook_filename = os.path.join(root, file)
            print(f"Executing {notebook_filename}...")
            with open(notebook_filename) as f:
                notebook = read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')  # You can adjust the timeout as needed
            executed_notebook, resources = ep.preprocess(notebook, {'metadata': {'path': root}})
            with open(notebook_filename, 'w', encoding='utf-8') as f:
                write(executed_notebook, f)
            print(f"{notebook_filename} executed successfully!")

print("All data preprocessing notebooks have been executed.")
