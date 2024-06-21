import os
import pandas as pd
import numpy as np





path=os.getcwd() 
list_groupby=data.columns[:4]
for i in range(1,len(list_groupby)+1):
    
    df=data.groupby(list(list_groupby[:i])).sum(numeric_only=True)
    for index in df.index:
        df_leaf=pd.DataFrame(df.loc[index].values.reshape((-1,7)).T , 
                        columns=df.loc[index].index[::7])
        df_leaf.name = '_'.join(map(str, index))
        
        df_leaf.to_csv(path+f"\\data\\M5\\fdata\\{df_leaf.name}.csv", index=None)