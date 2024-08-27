from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
from numpy.linalg import inv
import os
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime,timedelta
from scipy.interpolate import splrep, splev
import re
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
from dateutil.relativedelta import relativedelta
from leaf import *
from forecast_prophet import *


class Tree:
    def __init__(self, data_directory:str , type: str):
        """ 
        A tree object is a collection of leaf objects. 
        OnlyKPI and date is required for
        creation of whole tree according to listL4, listL3, listL2 specified below
        The order at which these lists are written is important. Stacking of reconciliation matrices is precise and is implied from dLeafs
                              
        Forecasts will be stored in matrix , shape of which is determined by the number of leafs in a given level.
        Inputs:
            sForecMeth (str)    Method to use in forecasting of leafs
            iPC (int) :          Number of principal components used in decomposing of the curve, defaults to 3
            sRecMeth (str)      Method to use when reconciling , 'multi' "bottom_up"
            sWeightType (str)   mW approximation method if sRecMeth is 'multi' choises are "diag" , "mint_shrinkage" ,  TODO "ols"
            data_directory(str) location of csv file of data at lowest granularity     
                                 
        Outputs:
           mS: summation matrix, see getMatrixS() for more detail
           mP: projection matrix, see getMatrixP() for more detail
           mW: weight matrix, see getMatrixW() for more detail 
        Methods for Outputs:
           There are several methods that allows user to quickly and efficiently conduct diagnostics of the algorithm results:
            (1) getLeaf() method allows user to specify strings for Level and get the leaf object. 
                          Thus no need to look up which leafs corresponds to which tree in dLeafs
            (2) plot_errors() methods plots interactive plotly graph of errors per leaf         
            (3) 
             
        """
        #reads data
        self.data=pd.read_csv(data_directory) 
        self.type=type
        
        #based on data, find all possible levels and datetime index
        if type=='spatial':
            self.levels=self.data.columns[pd.to_datetime(self.data.columns, errors='coerce').isna()]
            self.date_time_index=pd.to_datetime(self.data.drop(columns=self.levels).columns)
            
            #create a hierarchy list
            df=self.data[self.levels]
            list_of_leafs=df.values.tolist()  
            
            for level in self.levels[::-1]:
                df[level]=None
                df=df.drop_duplicates()
                list_of_leafs.extend(df.values.tolist())

            def sort_key(item):
                none_count = item.count(None)
                return (-none_count, item)
        
            self.list_of_leafs=sorted(list_of_leafs, key=sort_key)
            
            #create tree data matrix mY
            mY=np.zeros( (len(self.list_of_leafs), len(self.date_time_index)))
            
            for i,leaf_creds in enumerate(self.list_of_leafs):
                mY[i]=self.subset_data(leaf_creds).values
            
            # Get summattion matrix S                              
            mS=np.ones((1,self.data.shape[0])) # start with 1 row at the top of matrix s that is always a vector of ones of size equal to # of bottom level series
            
            for i,level in enumerate(self.levels.to_list()):
                groupByColumns=self.levels.to_list()[:i+1]
                vBtmLevelSeries=self.data.groupby(groupByColumns).count().iloc[:,0].values
                mS=np.vstack([mS,self.create_matrix_S(vBtmLevelSeries)])  
                     
        # elif type=='temporal':
        #     #data will be a series not dataframe
        #     levels=['T','H','D','W','M','Q','A']
        #     dLevels={'T': 60 , 'H' :24 , 'D': 7 , 'W':1, 'M': 3 , 'Q':4 , 'A': 1}
            
        #     start_index = levels.index(self.data.index.inferred_freq) #the bottom frequency, the freq of data
        #     end_index = 3 if start_index<3 else 6      
        #     self.levels=levels[start_index:end_index + 1]   
            
        #     #make sure series is summable :  full weeks, full years etc #TODO only works for D W now
        #     start_index = self.data[self.data.index.weekday == 6].index[0]
        #     end_index = self.data[self.data.index.weekday==6].index[-1]
        #     self.data=self.data[start_index:end_index]
            
        #     #create and populate mY
        #     levels=['T','H','D','W','M','Q','A']
        #     dLevels={'T': 60 , 'H' :24 , 'D': 7 , 'W':1, 'M': 3 , 'Q':4 , 'A': 1}

        #     start_index = levels.index(self.data.index.inferred_freq) #the bottom frequency, the freq of data
        #     end_index = 3 if start_index<3 else 6      
        #     levels=levels[start_index:end_index + 1] 

        #     if end_index==3: # works only for W and M data currently
        #         start_index = self.data[self.data.index.weekday == 0].index[0]
        #         end_index = self.data[self.data.index.weekday==6].index[-1]
        #     else:
        #         start_index = self.data[self.data.index.month == 1].index[0]
        #         end_index = self.data[self.data.index.month==12].index[-1]

        #     df=self.data[start_index:end_index]

        #     #create and populate mY
        #     aUnits=np.array([])
        #     n=1
        #     for sFreq in reversed(levels):
        #         n=dLevels[sFreq]*n
        #         aUnits=np.append(aUnits,int(n))
        #     self.aUnits=aUnits[::-1].astype(int)    
        #     n=int(aUnits.sum())
        #     m=df.resample(levels[-1]).sum().shape[0]


        #     mY=df.values.reshape( ( m , aUnits[0] )).T
        #     mS=self.create_matrix([aUnits[0]])
        #     for i,sFreq in enumerate(levels[1:]):
        #         df=df.resample(sFreq).sum()
        #         n = aUnits[i+1]
        #         mY_=df.values.reshape(( m ,n)).T 
        #         mY=np.vstack((mY_, mY))
  
        #         vBtmLevelSeries=np.full(aUnits[-i-2],int(aUnits[0]/aUnits[-i-2]))
        #         mS_=self.create_matrix(vBtmLevelSeries)
        #         mS=np.vstack(( mS, mS_ ))
                      
        self.mY = mY
        self.mS = mS          
        self.mP    = None
        self.mW    = None
        self.mYhat = None
        self.mYhatIS = np.zeros((self.mY.shape[0], self.mY.shape[1]))
        self.mYtilde = None 
        self.mRes = np.zeros((self.mY.shape[0], self.mY.shape[1]))    # matrix that stores in sample base forecast errors.



    def displayMatrix(self, matrix):
        plt.imshow(matrix,cmap='binary', interpolation='nearest')
        plt.show()  

    def subset_data(self,l:list):
        """
        subsets data to include only data of a certain leaf
        leaf_list (list)  size n, [0] is the level 0 while [-1] is the lowest level
        
        returns: serried of aggregated values for a given leaf_list
        """
        column_mask=(self.data[self.levels]==l).any(axis=0)  
        row_mask=(self.data[self.levels]==l).loc[:,column_mask].all(axis=1)
        
        srY=self.data[row_mask].drop(columns=self.levels).sum(axis=0)
        return srY 

    def create_matrix_S(self, list_of_lengths):
        """
        Creates a matrix with cascading binary entries. 
        list_of_lengths: number of 1's in each row
        number of rows in resulting matrix is equal to number of int in list_of_lengths
         
        """
        total_columns = sum(list_of_lengths)
        matrix = np.zeros((len(list_of_lengths), total_columns), dtype=int)
        
        start_index = 0
        for i, length in enumerate(list_of_lengths):
            matrix[i, start_index:start_index + length] = 1
            start_index += length
        
        return matrix     
                
    def getMatrixW(self , sWeightType:str):
        """
        Purpose:
        create the weights matrix
        
        Outputs:
        mW:            matrix of weights
        """

        mRes=self.mRes.T
        mW = np.eye(mRes.shape[1])
        vNonNanRows = np.setdiff1d(np.arange(0,mRes.shape[0]),  np.unique(np.argwhere(np.isnan(mRes))[:,0]))
        mRes = mRes[vNonNanRows,:]
        
        if sWeightType == 'diag':  #WLS
            for i in range(mRes.shape[1]):
                mW[i,i] = np.mean(mRes[:,i]**2)   
        
        elif sWeightType == 'full':  # full
            mW = mRes.T @ mRes / mRes.shape[0]
        
        elif sWeightType == 'mint_shrink':
            iRows = mRes.shape[0]
            iCols = mRes.shape[1]
            mWF = mRes.T @ mRes / iRows
            mWD = np.diag(np.diag(mWF)) # all non-diagonal entries of mWF set to 0
            #calculate numerator
            dBottom = 0 # lower side in the expression for tuning parameter lambda
            for i in range(iCols):
                for j in range(iCols):
                    if i>j:
                        dBottom = dBottom + 2*( mWF[i,j] / np.sqrt(mWF[i,i]*mWF[j,j]) )
            #Calculate denominator            
            mResScaled = mRes / np.sqrt(np.diag(mWF)) # elementwise division
            mResScaledSq = mResScaled**2
            mUp = (1/(iRows*(iRows-1))) * ( (mResScaledSq.T @ mResScaledSq)- (1/iRows)*((mResScaled.T @ mResScaled)**2) )
        
            dUp = 0 # lower side in the expression for tuning parameter lambda
            for i in range(iCols):
                for j in range(iCols):
                    if i>j:
                        dUp = dUp + 2*mUp[i,j]
            
            dLambda = np.max((np.min((dUp/dBottom, 1)), 0))
            mW = dLambda * mWD + (1-dLambda) * mWF
        
        self.mW=mW
    
    def getMatrixP(self , sWeigthType: str):  
        """
        Purpose:
            return projection matrix P for a given mS and mW, as per Wickramasuriya et al (2018) equation (9)
            aka matrix G in some litterature such as as Hyndman et al. (2019).
            If sMeth is bottom up then mW is not used.  
        
        Return value:
            self.mP     projection matrix
        """
        
        mS=self.mS
        mW=self.mW
        
        if sWeigthType == 'bottom_up':
            n=mS.shape[1]
            m=mS.shape[0]
            m0=np.full((n,m-n),0, dtype=int)
            mI=np.eye(n)
            mP=np.hstack((m0,mI))
            self.mP=mP
        elif 'top_down' in sWeigthType:
            n=mS.shape[1]
            m=mS.shape[0]
            m0=np.full((n,m-1),0, dtype=int)
            iB= len([sublist for sublist in self.list_of_leafs if sublist.count(None) == 0])# integer length of bottom level
            if sWeigthType=='top_down_hp': #historical proportions
                #TODO 70 dynamic
                vP = np.mean((self.mY[-iB:]/self.mY[0,:]),axis=1)
            elif sWeigthType=='top_down_ph': #proportions of the historical averages
                vP = np.mean(self.mY[-iB:],axis=1)/np.mean(self.mY[0,:],axis=0)
            # elif sWeigthType=='top_down_fp': #forecast proportions
            #     vP = 
            vP=vP.reshape((iB,1))
            mP=np.hstack((vP,m0))
            self.mP=mP    
        else:
            mWinv = np.linalg.inv(mW)
            mP= (np.linalg.inv(mS.T @ (mWinv @ mS)) @ (mS.T @ (mWinv)))
            self.mP=mP  
                
    def forecast(self, sForecMeth:str , iOoS:int):  #get mYhat
        """
        Performs the forecast algorithm at each leaf
        iOoS: of bottom level series
        
        """  
        # if self.mYhat is not None:
        #     return
        try:
            holidays=pd.read_csv(os.getcwd()+f"\\data\\M5\\holidays.csv")
            holidays=holidays.values.flatten()
        except: 
            holidays=None
            
        try:
            changepoints=pd.read_csv(os.getcwd()+f"\\data\\M5\\changepoints.csv")
            changepoints=changepoints.values.flatten()
        except: 
            changepoints=None    
        
        if self.type=='temporal': #then it is temporal reconciliation
            vYhat=np.array([])
            multiple=int(iOoS/self.aUnits[0])
            for i,sFreq in enumerate(self.levels):  # starts from bottom
                dfData=self.data.resample(sFreq).sum()
                if sForecMeth=="Prophet":
                    pht=Forecast_Prophet(dfData=dfData, iOoS=(self.aUnits*multiple)[i])
                    vYhat_ = pht.forecast(holidays=holidays, changepoints=changepoints).yhat.values
                vYhat=np.concatenate((vYhat_, vYhat)) 
            self.mYhat=vYhat    
                
        else:
            self.mYhat=np.zeros((len(self.list_of_leafs),iOoS))
            
            for i in range(self.mY.shape[0]):
                dfData = pd.DataFrame(data=self.mY[i] , index=self.date_time_index , columns=['y'])           
                
                if sForecMeth=='Prophet':      
                    pht = Forecast_Prophet(dfData=dfData, iOoS=iOoS)
                    vYhat = pht.forecast(holidays=holidays, changepoints=changepoints).yhat.values
                    self.mYhat[i] = vYhat[-iOoS:]
                    self.mYhatIS[i] = vYhat[:-iOoS]                                
                            
                # elif sForecMeth=="PCR":
                #     ### 7 daily curve
                #     # so that it is divisible by 7
                #     dfData=pd.DataFrame(data=self.mY[i][2:].reshape(int(self.mY[i][2:].shape[0]/7),7).T,
                #                         columns=self.time_index[2:][::7]) # so [2:] that it is divisible by 7
                #     #dfData=dfData.astype(int)
        
                        
        self.mRes=self.mYhatIS-self.mY    
                
                         
    def reconcile(self , sWeightType: str):
        """
        Performs whole reconciliation algorithm
        """                                  
        self.getMatrixW(sWeightType)      
        self.getMatrixP(sWeightType)            
        self.mYtilde=np.dot(np.dot(self.mS,self.mP),self.mYhat)
        
        print('Reconciliation is complete')
        
        