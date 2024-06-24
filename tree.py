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
from forecast import *


class Tree:
    def __init__(self, data_directory:str):
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
           mYhat0-mYhat3: matrix storing base forecasts
           mYrec0-mYrec3: matrix storing reconciled forecasts  
        Methods for Outputs:
           There are several methods that allows user to quickly and efficiently conduct diagnostics of the algorithm results:
            (1) getLeaf() method allows user to specify strings for Level and get the leaf object. 
                          Thus no need to look up which leafs corresponds to which tree in dLeafs
            (2) plot_errors() methods plots interactive plotly graph of errors per leaf         
            (3) 
             
        """
        #reads data
        self.data=pd.read_csv(data_directory)
        
        #based on data, find all possible levels and datetime index
        self.levels=self.data.columns[pd.to_datetime(self.data.columns, errors='coerce').isna()]
        self.time_index=pd.to_datetime(self.data.drop(columns=self.levels).columns)
        
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
        self.mY=np.zeros( (len(list_of_leafs), len(self.time_index)))
        
        def subset_data(leaf_list):
            """
            subsets data to include only data of a certain leaf
            leaf_list (list) list of size n, [0] is the level 0 while [-1] is the lowest level
            
            returns: serried of aggregated values for a given leaf_list
            """
            column_mask=(self.data[self.levels]==leaf_list).any(axis=0)  #should levels be self.levels?
            row_mask=(self.data[self.levels]==leaf_list[i]).loc[:,column_mask].all(axis=1)
            
            srY=self.data[row_mask].drop(columns=self.levels).sum(axis=0)
            # dfData=srY.to_frame().rename(columns={0: 'y'})
            # dfData.index=pd.to_datetime(dfData.index)
            return srY
        
        for i,leaf_creds in enumerate(list_of_leafs):
            # setattr(self, f"leaf_{i}", Leaf(date=date, 
            #                                 dfData=subset_data(leaf_creds),
            #                                 leaf=leaf_creds # list of leaf credentials
            #                              ))
            self.mY[i]=subset_data(leaf_creds).values
            
        # Get summattion matrix S                              
        m=self.data.shape[0]
        mS=np.ones((1,m))

        def create_matrix(lengths):
            total_columns = sum(lengths)
            matrix = np.zeros((len(lengths), total_columns), dtype=int)
            
            start_index = 0
            for i, length in enumerate(lengths):
                matrix[i, start_index:start_index + length] = 1
                start_index += length
            
            return matrix

        for i,level in enumerate(self.levels.to_list()):
            vNonZeroEntries=self.data.groupby(self.levels.to_list()[:i+1]).count().iloc[:,0].values
            mS=np.vstack([mS,create_matrix(vNonZeroEntries)])
 
        self.mS=mS
        
        self.mP    = None
        self.mW    = None
        self.mYhat = None
        self.mYhatIS = np.zeros((self.mY.shape[0], self.mY.shape[1]))
        self.mYrec = None 
        
        iIS = len(self.time_index)
        self.mRes = np.zeros((self.mY.shape[0], self.mY.shape[1]))    # matrix that stores in sample base forecast errors.

    def displayMatrix(self, matrix):
        plt.imshow(matrix,cmap='binary', interpolation='nearest')
        plt.show()
                
    def getMatrixW(self , sWeightType:str):
        """
        Purpose:
        Perform estimation using LS
        
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
        
        if sWeigthType is None:
            n=mS.shape[1]
            m=mS.shape[0]
            m0=np.full((n,m-n),0, dtype=int)
            mI=np.eye(n)
            mP=np.hstack((m0,mI))
            self.mP=mP
        else:
            mWinv = np.linalg.inv(mW)
            mP= (np.linalg.inv(mS.T @ (mWinv @ mS)) @ (mS.T @ (mWinv)))
            self.mP=mP  
        
            
    def forecast(self, sForecMeth:str , iOoS:int , iPC=None):  #get mYhat
        """
        Performs the forecast algorithm at each leaf
        """  
        self.mYhat=np.zeros((len(self.list_of_leafs),iOoS))
        
        if sForecMeth=='Prophet':
            for i in range(self.mY.shape[0]):
                dfData = pd.DataFrame(data=self.mY[i], index=self.time_index)
                fc = Forecast(dfData=dfData, iOoS=iOoS)
                vYhat = fc.Prophet().yhat.values
                self.mYhat[i] = vYhat[-iOoS:]
                self.mYhatIS[i] = vYhat[:iOoS]
            self.mRes=self.mYhatIS-self.mY
        
                

                         
    def reconcile(self , sRecMeth:str , sWeightType: str):
        """
        Performs whole reconciliation algorithm
        """                                  
        self.getMatrixW(sWeightType)      
        self.getMatrixP(sWeightType) 
           
        self.mYrec=np.dot(np.dot(self.mS,self.mP),self.mYhat)
        
        print('Reconciliation is complete')
        
        