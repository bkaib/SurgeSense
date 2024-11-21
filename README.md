# Estimating Extreme Storm Surges in the Baltic Sea via Machine Learning
## Short Description
The aim of this project is to estimate extreme storm surges (>95%-percentiles) in the Baltic Sea via a Random Forest. For now only binary classification is applied. A future extension aims to also supply a full regression of extreme storm surge height using Random Regression Forests. The masterthesis is conducted together with the Helmholtz-Zentrum-Hereon and the University of Hamburg and is embedded in the CLimate INTelligence project.

The main.ipynb Jupyter Notebook contains a tutorial of the setup.

## Structure
data (package): Modules for preprocessing and manipulating data.

examples (folder): Containing sample folders for the predictor and predictand data

models (package): Modules for loading, fitting and evaluating a model.#

results (folder): Folder containing the subfolders of all experiments

gesla.py (package): Contains function provided by GESLA.org to handle the GESLA data.

main (ipynb): The main notebook explaining the model run.

requirements (txt): The python packages needed to run this model