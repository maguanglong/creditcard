# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:06:15 2019

@author: guanglong
"""
import pandas as pd
import numpy as np

def missing_coarse(df):
    head = df.keys()
    dotype, domiss, dorate = [], [], []
    vartype = []
    for name in head:
        ds = df[name]
        dotype.append(ds.dtype)
        vartype.append(len(ds.unique()))
        if ds.dtype=='int64':
            mc, mn = ds[ds==-9999].count(), ds.count()
            domiss.append(mc)
            dorate.append(mc / mn)
        elif ds.dtype=='float64':
            mc, mn = ds[ds==-9999].count(), ds.count()
            domiss.append(mc)
            dorate.append(mc / mn)
        else:
            mc, mn = ds[ds=='-9999'].count(), ds.count()
            domiss.append(mc)
            dorate.append(mc / mn)
    out = pd.DataFrame(data={'name': head,'missing_count':domiss, 'missing_rate': dorate, 'vartype':vartype})
    out=out.sort_values(by="missing_rate")
    out.to_csv('missing_rate.csv')
    pre1=out[(out['missing_rate']<0.3)&(out['missing_rate']>0)]
    varlist = pre1.name
    return varlist, pre1

def woe_get(dout):
    dout["total"]=dout.loc[:,1]+dout.loc[:,0]
    dout["L_good"]=dout.loc[:,0]/dout.loc[:,0].sum()
    dout["L_bad"]=dout.loc[:,1]/dout.loc[:,1].sum()
    dout["woe"]=np.log(dout["L_good"]/dout["L_bad"])
    dout=dout[(dout.L_good * dout.L_bad) !=0]
    dout["IV"]=(dout["L_good"]-dout["L_bad"])*np.log(dout["L_good"]/dout["L_bad"])
    return dout
    

def iv_coarse(df,pre1,vname,nb):
    variv=[]
    varname=[]
    for name in pre1[vname]:
        num = len(df[name].unique())
        temp=df[(df[name]!=-9999)]
        if num<58:
            #print("名义变量%d"%(num))
            dout =  pd.crosstab(df[name],df.y)
        elif (num>58 and num<10000):
            #print("顺序变量%d"%(num))
            dcut=pd.cut(temp[name],nb)
            dout =  pd.crosstab(dcut,temp.y)
        else:
            #print("连续变量%d"%(num))
            dcut=pd.qcut(temp[name],nb)
            dout =  pd.crosstab(dcut,temp.y)
        dout = woe_get(dout)
        dout["total"]=dout.loc[:,1]+dout.loc[:,0]
        dout["L_good"]=dout.loc[:,0]/dout.loc[:,0].sum()
        dout["L_bad"]=dout.loc[:,1]/dout.loc[:,1].sum()
        dout["woe"]=np.log(dout["L_good"]/dout["L_bad"])
        dout=dout[(dout.L_good * dout.L_bad) !=0]
        dout["IV"]=(dout["L_good"]-dout["L_bad"])*np.log(dout["L_good"]/dout["L_bad"])
        variv.append(dout.IV.cumsum().max())
        varname.append(name)
    IV=pd.DataFrame({'name':varname,'info_value':variv}) 
    IV = IV.sort_values(by='info_value')
    return IV
