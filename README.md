# Deepwater_DO_model
This repository contains the Python code used in the analysis for the paper titled 
"Translating climate-induced lake deoxygenation predictions into adaptive management strategies", 
which evaluates deepwater oxygen dynamics in during the summer stratified periods in Lake Erken, Sweden.

## Overview

The project involves:

- Using lake shape information and temperature profiles to assess deepwater stratification in Lake Erken.
- Running a dissolved oxygen (DO) model with varying values of:
  - Jv (water-column oxygen demand)
  - Ja (sediment oxygen demand)
- Selecting acceptable parameter sets that result in a mean DO profile error of less than 0.75 mg L⁻¹ for the years 2020 and 2021 using "simulate_deepwater_DO" function.
- Estimating a 90% confidence interval for modelled DO using Generalized Likelihood Uncertainty Estimation (GLUE), with Nash–Sutcliffe Efficiency (NSE)
   as the likelihood function, across the years 2019–2022.

## Dependencies
The code was written in Python 3.8+ and requires the following packages: 

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `scikit-learn`
- `hydroeval`  _(can be installed via pip: `pip install hydroeval`)_

*** These packages can be installed by running "install_packages.py" ***
## Setup
- input_dir: the path to your local folder containing the input data (e.i, Lake Erken bathymetry, temperature and DO profiles). (IMPORTANT: Update this path in the code to match your local setup.)
- output_dir: the folder where model outputs and results will be saved.
- output_data_checking_file: a local file used as a checkpoint to store a copy of the final results after code execution.
