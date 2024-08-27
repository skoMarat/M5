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

import forecast_prophet
reload(forecast_prophet)
from forecast_prophet import *

import logging
logging.getLogger('cmdstanpy').addHandler(logging.NullHandler())

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
    
    path=os.getcwd()+f"\\data\\M5\\sales_train_validation.csv"  # to data file 
    
    weight_type =  "diag"
    weight_type = "mint_shrink" 
    weight_type = "full"
    weight_type = "bottom_up"
    weight_type = "top_down_hp"
    weight_type = "top_down_ph"
    
    #forecast_method = 'pick-up'
    #forecast_method =  "Prophet_FPCR" 
    #forecast_method =  "Prophet_FPCR"
    #forecast_method =  "VAR" 
    forecast_method = "Prophet"
    
    iOoS=28  # at the bottom forecast frequency if temporal
    
    tree=Tree( data_directory = path , type='spatial')
    tree.forecast( sForecMeth = forecast_method , iOoS=iOoS )
    tree.reconcile( sWeightType = weight_type)
    
    
    evaluation_path = os.getcwd()+f"\\data\\M5\\sales_train_evaluation.csv"
    tree_eval=Tree(data_directory=evaluation_path, type='spatial')
    mYtrue=tree_eval.mY
    mYhat=tree.mYhat
    mYtilde=tree.mYtilde
    
    #evaluation   
    
    vMAEhat=np.sum(np.abs(mYhat-mYtrue),axis=1)/mYtrue.shape[1]
    vMAEtilde=np.sum(np.abs(mYtilde-mYtrue),axis=1)/mYtrue.shape[1]
    
    plt.plot(vMAEhat-vMAEtilde, label='hat-tilde')
    plt.legend()
    plt.axhline(y=0, color='r')
    plt.show()
    #if positive, then reconciliation made it beter , how can reconciliation make
    # it worse? How can we learn?

    
    
    
    
def compare():
    
    #bottom_up
    tree=Tree( data_directory = os.getcwd()+f"\\data\\M5\\sales_train_validation.csv" , type='spatial')
    tree.forecast( sForecMeth = "Prophet" , iOoS=28 )
    tree.reconcile( sWeightType = 'bottom_up')
    #true 
    evaluation_path = os.getcwd()+f"\\data\\M5\\sales_train_evaluation.csv"
    tree_eval=Tree(data_directory=evaluation_path, type='spatial')
    mYtrue=tree_eval.mY
    #get true,hat and tilde matrices
    mYtrue=tree_eval.mY
    mYhat=tree.mYhat
    mYtilde=tree.mYtilde
    
    #calculate MAE for bu
    vMAEhat=np.sum(np.abs(mYhat-mYtrue),axis=1)/mYtrue.shape[1]
    vMPEhat=np.sum((mYhat-mYtrue)/mYtrue*100,axis=1)/mYtrue.shape[1]
    vMAEtilde_bu=np.sum(np.abs(mYtilde-mYtrue),axis=1)/mYtrue.shape[1]
    # vMPEtilde_bu=np.sum((mYtilde-mYtrue)/mYtrue*100,axis=1)/mYtrue.shape[1]

    
    
    for sWeightType in ['bottom_up','top_down_hp','top_down_ph', 'diag','full','mint_shrink']:
        print("---------------"+ sWeightType+"-----------------")
        tree.reconcile( sWeightType = sWeightType)
        mYtilde=tree.mYtilde
        vMAEtilde=np.sum(np.abs(mYtilde-mYtrue),axis=1)/mYtrue.shape[1]
        vMPEtilde=np.sum((mYtilde-mYtrue)/mYtrue*100,axis=1)/mYtrue.shape[1]
        
        plt.plot(vMPEtilde, label=sWeightType)
        plt.title("MPE " + sWeightType)
        plt.legend()
        plt.axhline(y=0, color='r')
        plt.show()
        

        vRelMAE_hat=vMAEtilde/vMAEhat
        # vRelMPE_hat=vMPEtilde/vMPEhat
        AvgRelMAE_hat=np.sum(vRelMAE_hat)/len(vRelMAE_hat)
        # AvgRelMPE_hat=np.sum(vRelMPE_hat)/len(vRelMPE_hat)
        print("ARMAE_hat = " + str(AvgRelMAE_hat))
        # print("ARMPE_hat = " + str(AvgRelMPE_hat))
        
        plt.plot(vMPEhat-vMPEtilde, label=sWeightType)
        plt.title("MPE difference, base and reconciled forecasts ")
        plt.legend()
        plt.axhline(y=0, color='r')
        plt.show()
        
        vRelMAE=vMAEtilde/vMAEtilde_bu
        # vRelMPE=vMPEtilde/vMPEtilde_bu
        AvgRelMAE=np.sum(vRelMAE)/len(vRelMAE)
        # AvgRelMPE=np.sum(vRelMPE)/len(vRelMPE)
        print("ARMAE = " + str(AvgRelMAE))
        # print("ARMPE = " + str(AvgRelMPE))
         
        plt.title(sWeightType +' P matrix')
        tree.displayMatrix(tree.mP)
        plt.show()
       
        
        
        
        
        
    
    i=100
    plt.plot(mYtilde[i,:],label='tilde')
    plt.plot(mYhat[i,:],label='hat')
    plt.plot(mYtrue[i,:],label='True')
    plt.legend()    
                
    
    
if __name__ == "__main__":
    main()