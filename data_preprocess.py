# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:04:54 2019

@author: guanglong ma
"""
from pandas import Series,DataFrame
import numpy as np
import pandas as pd


def describe_data(data,ifprint):
    out={'method':['count','min','max','argmin','argmax','idxmin','idxmax','quantile',
               'sum','mean','median','mad','var','std','skew','kurt']}
    dfout=DataFrame(out)
    varlist = data.keys()
    for var_name in varlist:
        vara=data[var_name]
        var_value=Series([vara.count(),
                          vara.min(),
                          vara.max(),
                          vara.argmin(),
                          vara.argmax(),
                          vara.idxmin(),
                          vara.idxmax(),
                          vara.quantile(),
                          vara.sum(),
                          vara.mean(),
                          vara.median(),
                          vara.mad(),
                          vara.var(),
                          vara.std(),
                          vara.skew(),
                          vara.kurt()])
        dfout[var_name]=var_value
    if ifprint==1:
        print(dfout)
    
    return dfout

def woe_var(data, var_name, status_name, nbins):
    #var_name="age"
    status=data[status_name]
    xvar=data[var_name]
    xmax=xvar.max()
    xmin=xvar.min()
    bad = status.sum()
    good = status.count() - bad
    if len(xvar.unique())<50:
        d1 = pd.DataFrame({"status": status, var_name: xvar, "Bucket": pd.cut(xvar, nbins)})
    else:
        d1 = pd.DataFrame({"status": status, var_name: xvar, "Bucket": pd.qcut(xvar, nbins)})
    d2 = d1.groupby('Bucket', as_index = True)
    woe_list = pd.DataFrame(d2[var_name].min(), columns = ['min'])
    woe_list['min']=d2[var_name].min()
    woe_list['max'] = d2[var_name].max()
    woe_list['total'] = d2.count().status
    woe_list['bad']=d2.sum().status
    woe_list['good']=woe_list['total']-woe_list['bad']
    woe_list['rate'] = d2.mean().status #=woe_list['bad']/woe_list['total']
    #woe_list['sum'] = d2.sum().status
    #woe_list['rate'] = d2.mean().status
    woe_list['woe']=np.log((woe_list['bad']/bad)/(woe_list['good']/good))
    woe_list['iv_i']=(woe_list['good']/good-woe_list['bad']/bad)*woe_list['woe']
    iv=woe_list['iv_i'].sum()
    woe_list['iv']=iv
    print(woe_list)
    print("="*66)
    mapping=[]
    mapping.append(xmin)
    for i in range(1,nbins+1):
        qua=xvar.quantile(i/(nbins+1))
        mapping.append(round(qua,4))
    mapping.append(xmax)
    return woe_list,mapping
