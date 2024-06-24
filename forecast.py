from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
from numpy.linalg import inv
import os
import pandas as pd
import numpy as np
import os
from datetime import datetime,timedelta
from scipy.interpolate import splrep, splev
import re
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from pmdarima.arima import auto_arima
from pandas import to_datetime
from prophet import Prophet
import logging
import cmdstanpy
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
from dateutil.relativedelta import relativedelta
import  skfda
from skfda.preprocessing.dim_reduction import FPCA
import fdata
from fdata import FData


"""

This class contains methods for forecasting a single series
Forecast is an object, that must be an object because we might want to generate several forecasts in the future
using that same data and compare. We dont want to run the whole process again just to be able to do that

"""

class Forecast:
    def __init__(self, dfData: pd.DataFrame, iOoS:int , lX=None ):
        """
        dfData          : dataframe with column y to forecast
        iOoS            : integer number of oos observations to forecast
        lX              : list of strings, x columns as extra regressors, if None , then only Y is present
        
        """
        self.dfData=dfData    # data , [0] to be forecasted, index must be datetime index 
        self.iOoS=iOoS  # number of OoS forecast to generate in the same granularity as srY
        self.srYhat     # forecasted series TODO

        
    def Prophet(self, 
                dfX=None,
                scaling="absmax",
                holidays=None,
                changepoints=None,
                changepoint_prior_scale=0.05,
                holidays_prior_scale=0.9,
                yearly_seasonality=20):
        """
        Fit Prophet model for daily (or monthly data? TODO)
        regressor self.x must be known also for OoS forecast part. Use other method otherwise
        TODO add a predictor for X columns via Prophet or predict X out of this method via other methods
        and feed back to this method
        
        
        dfX (df)        : lX columns in self.dfData but extended with iOoS (predicted via Prophet or other method or available as data)
        holidays (list)        : of strings, can be days or days representing whole month
                                 if later, then whole month must be taken as a holiday
        changepoints (list)    : of strings, same as above, represents changes in slope of the trend
        
        
        returns dfModel
        dfModel.vYhat is the fitted and predicted vYhat of len(self.iOoS + len(dfData))
        
        """
        sFreq=self.dfData.index.inferred_freq
        if sFreq is None:
            print("frequency non inferrable, please provide regular frequency index")
            return None
        
        
        df=self.dfData['y'].copy().reset_index()
        df.rename(columns={'index' : 'ds'}, inplace=True)
 
        if holidays is not None: # then there are holidays to incorporate
            holiday_dfs=[]
            for i,holiday in enumerate(holidays): 
                h=pd.DataFrame({
                                'holiday': str(i+1), #holiday name , #TODO redundant?
                                'ds': pd.to_datetime([holidays[i]])
                                })
                holiday_dfs.append(h)
            df_holidays=pd.concat(holiday_dfs, ignore_index=True)
                   
        model=Prophet(holidays=df_holidays , 
                      scaling=scaling ,
                      changepoints=changepoints,
                      changepoint_prior_scale=changepoint_prior_scale,
                      holidays_prior_scale=holidays_prior_scale,
                      yearly_seasonality=yearly_seasonality)
        
        if dfX is not None: # add regressors
            for regressor in dfX.columns:
                #TODO prior scale might need to be adjusted based on regressor 
                model.add_regressor(regressor,prior_scale=0.5, standardize=True)
             
        model.fit(df)
        future=model.make_future_dataframe(periods=self.iOoS , freq=sFreq)
        
        if dfX is not None:
            future[dfX.columns]=dfX
        
        dfModel=model.predict(future)
                    
        return dfModel
    
    
    def AR():
        return    
        
        
    def FPCR():
        return
    
    def VAR():
        return   
        
            
    
    
