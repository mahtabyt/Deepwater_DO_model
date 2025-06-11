# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 19:01:56 2025

@author: mahta
"""


from pathlib import Path
import pandas as pd
import numpy as np
import scipy.interpolate
import os
from sklearn.metrics import mean_squared_error
#!pip install hydroeval 
from hydroeval import evaluator, nse
import math
import matplotlib.pyplot as plt