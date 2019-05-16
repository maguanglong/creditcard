# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:39:35 2019

@author: guanglong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from numpy import NaN,Inf

def devide_coarse(df,name,y_name,nbins):
    if df[name].dtypes=='object':
        ts = df[df[name]!='-9999']
        indexmap={'-9999':-9999}
        i=0        
        for ele in ts[name].unique():
            indexmap[ele]=i
            i=i+1
        df.loc[:,'value']=df[name].map(indexmap)
        ts = df[df['value']!=-9999]
        t1 = ts['value']
        if (t1[t1==0].count())/t1.count() > 0.5:
            ts= ts[ts['value']!=0]
            nbins=min(len(ts['value'].unique()),nbins)
            t2,bins=pd.qcut(ts['value'],nbins,retbins='True',duplicates='drop')
            newbins = np.zeros(len(bins)+3)
            newbins[:3]=np.array([-Inf,-9999,0])
            newbins[3:]=bins
        else:
            nbins=min(len(ts['value'].unique()),nbins)
            t2,bins=pd.qcut(ts['value'],nbins,retbins='True',duplicates='drop')
            newbins = np.zeros(len(bins)+2)
            newbins[:2]=np.array([-Inf,-9999])
            newbins[2:]=bins
        t2=pd.cut(df['value'],newbins)
        dtab = pd.crosstab(t2,df[y_name])
    else:
        indexmap={}
        ts = df[df[name]!=-9999]
        temp, bins = pd.qcut(ts[name],nbins-1,retbins='True',duplicates='drop')
        a3 = np.array([-Inf,-9999])
        newbins = np.zeros(len(bins)+len(a3))
        newbins[:len(a3)] = a3
        newbins[len(a3):]=bins
        newbins[-1]=np.array([Inf])
        t2=pd.cut(df[name],newbins)
        dtab = pd.crosstab(t2,df[y_name])
        
    return dtab, newbins, indexmap
    

def woe_coarse(dtab):
    dout = dtab
    dout["total"]=dout.loc[:,1]+dout.loc[:,0]
    dout["L_good"]=dout.loc[:,0]/dout.loc[:,0].sum()
    dout["L_bad"]=dout.loc[:,1]/dout.loc[:,1].sum()
    dout["woe"]=np.log(dout["L_good"]/dout["L_bad"])
    dout = dout[(dout.L_good * dout.L_bad) !=0]
    dout.loc[:,"IV"]=(dout["L_good"]-dout["L_bad"])*np.log(dout["L_good"]/dout["L_bad"])
    iv = dout.IV.cumsum()[-1]
    return dout,iv,dout.woe



df = pd.read_csv('data.csv')
key = df.keys()
y_name = 'status'
i_iv,i_name=[],[]
for name in key:
    if name != y_name:
        nbins=6
        dtab, indexmap, newbins=devide_coarse(df,name,y_name,nbins)
        dout,iv_t,woe=woe_coarse(dtab)
        i_iv.append(iv_t)
        i_name.append(name)
        print(dout[['woe','total']])
        print("="*50)
IV=pd.DataFrame(data={'name':i_name,'IV':i_iv})
print(IV.sort_values(by='IV'))
ts = df[key[0]]
statistics_1 = pd.DataFrame()
statistics_2 = pd.DataFrame()
statistics_3 = pd.DataFrame()
total_amount = ts.count()
miss_amount = ts[ts==-9999].count()
