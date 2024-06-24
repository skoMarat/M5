import pandas as pd
from importlib import reload
import numpy as np
import os

# import simulation
# reload(simulation)
# from simulation import *

import leaf
reload(leaf)
from leaf import *
import fdata
reload(fdata)
from  fdata import *
import tree
reload(tree)
from tree import *
import logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)


def main():
    """
    Requirements:
        Python 3.9.13 is a requirement for skicit-fda 
        Python 3.7 or higher is required for Prophet
    
    Code starts with importing data via import_data()
    It imports data from mondrian folder and the logic can be found in data_import.py file
    Currently the only reconciliation_method is bottom up
    Currently the only forecast_method is 'FPCR'
    
    Code initiliazes Tree object for a given KPI and date, where date is the 3 month ahead date
       Initialized tree, initializes 78 Leafs , for each hiearchy level
       Each leaf is populated with its own data and covid_data
       Each leaf is capable of forecasting itself.
            When forecasting a leaf, the data of the leaf is transformed into a function data object
            fda object stores the data, partially observed curve, principal component scores and vectors.
            fda object is capable of demeaned and remeaning itself. 
            fda object is capable of smoothing itself via smooth_data(n_basis:int) method
            fda object is capable of extracting PC scores and vectors via perform_FPCA() method
       Each leaf is equipped with methods that can plot the data and the forecasts
       It can plot forecasts of beta, forecasts of the curve, reconciled forecasts and plot the data of the leaf    
    A tree has a method to reconcile itself given a reconciliation method
    When reconciliation is started, a forecast command is passed to each tree
    After each tree is forecasted, Tree  matrix objects gets populated (mS, mW, mP, mYhat)
    Thereafter, reconciliation is performed. You can reconcile a tree only once. To try a different parameter for reconciliation, you need to reinitialize the tree
    After reconciliation, each tree gets populated with reconclied forecasts, completing the whole algorithm
    A tree is equipped with plot_errors() method that creates an interactive plot of percentage errors per leaf (reconciled forecasts and base forecasts) 
    """
    
    path=os.getcwd()+f"\\data\\M5\\fdata"
    
    weight_type =  "diag" # "mint_shrinkage"  "full"  "ols"   "None = bottom up"
    # forecast_method = 'pick-up' # "Prophet_FPCR"  "Prophet_FPCR_update"   "Prophet_TS" 
    forecast_method = "Prophet"
    iOoS=28
    
    tree=Tree( data_directory = path)
    tree.forecast(sForecMeth=forecast_method , iOoS=iOoS)
    tree.reconcile( sWeightType=weight_type)
    
    
    
    
if __name__ == "__main__":
    main()