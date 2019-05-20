# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:57:41 2019

@author: guanglong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from numpy import NaN,Inf
from sklearn.metrics import roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib
import statsmodels.api as sm

##################################################################
##################################################################
##################################################################
##################################################################



df = pd.read_excel('model_p.xlsx')
key = df.keys()
##################################################################
##################################################################
##################################################################
##################################################################




#数据的基本统计
for x in df.keys():
    ds = df[x]
    statistics = ds.describe()
    if ds.dtype != 'object':
        statistics.loc['range'] = statistics.loc['max'] - statistics.loc['min'] # 极差
        statistics.loc['var'] = statistics.loc['std'] / statistics.loc['mean'] # 变异系数
        statistics.loc['dis'] = statistics.loc['75%'] - statistics.loc['25%']#四分位数间距
    print(statistics)
    print('='*50)
##################################################################
##################################################################
##################################################################
##################################################################
    
    
    
    

# 数据画图：名义变量条形图；顺序变量在去除缺失值后画箱型图    
for x in df.keys():
    ds = df[x]
    if ds.dtype == 'object':
        plt.hist(ds,len(ds.unique()))
    else:
        temp = ds[ds!=-9999]
        ds.plot(kind = 'box')
    plt.show()
##################################################################
##################################################################
##################################################################
##################################################################
    
    
    
    
#单变量箱型图
name = ''
ds = df[name]    
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False 
plt.figure()
p = ds.plot(kind = 'box')
x = p['fliers'][0].get_xdata()
y = p['fliers'][0].get_ydata()
y.sort()

for  i in range(len(x)):
    if i>0:
        plt.annotate(y[i],xy=(x[i],y[i]), xytext = (x[i]+0.05 -0.8/(y[i]-y[i-1]),y[i]))
    else:
        plt.annotate(y[i],xy=(x[i],y[i]), xytext = (x[i]+0.08,y[i]))
        
plt.show()#展示箱线图
##################################################################
##################################################################
##################################################################
##################################################################

# 缺失值统计
ntotal = df.count().max()
m1, m2 = [], []
for x in df.keys():
    ds = df[x]
    m1.append(ntotal-ds.count())
    if ds.dtype == 'object':
        m2.append(ds[ds=='-9999'].count())
    else:
        m2.append(ds[ds==-9999].count())
        
miss_count=pd.DataFrame({'key':df.keys(),'miss':m1, '-9999':m2})
miss_count.loc[:,'miss_total'] = miss_count.loc[:,'miss']+miss_count.loc[:,'-9999']
miss_count.loc[:,'miss_rate'] =  miss_count.loc[:,'miss_total']/ntotal
miss_count.sort_values(by='miss_rate',inplace = True)


varlist = miss_count[miss_count.miss_rate<0.5].key
xdata = df[varlist].drop(['status','preacustseg'],axis = 1,inplace = False)
ydata = df['status']    
##################################################################
##################################################################
##################################################################
##################################################################



#名义变量，连续变量
Nominal_variable,Continuous_variable=[],[]
for i in xdata.keys():
    if xdata[i].dtype == 'object':
        Nominal_variable.append(i)
    else:
        if len(xdata[i].unique())<13:
            Nominal_variable.append(i)
        else:
            Continuous_variable.append(i)
##################################################################
##################################################################
##################################################################
##################################################################
        
        
# IV值初步试探
iv_unique=[]
iv_value=[]
iv_name=[]
outliers=[]

#for i in xdata.keys():
for i in Continuous_variable:
    ds = xdata[i]
    nbins = 20
    ts = ds[ds!=-9999]
    ################## 分箱 ##################################
    dout, bins = pd.qcut(ts, nbins-1, retbins = 'True',duplicates='drop')
    newbins = np.zeros(len(bins)+1)
    newbins[1:] = bins
    newbins[:2]=np.array([-Inf,bins[0]-1])
    dout = pd.cut(ds, newbins)
    ###################### IV ##############################3
    total_good, total_bad  = ydata.value_counts()
    dtab = pd.crosstab(dout,ydata)
    dtab.loc[:,'P_good']=dtab.loc[:,0]/total_good
    dtab.loc[:,'P_bad']=dtab.loc[:,1]/total_bad
    dtab.loc[:,'woe'] = np.log(dtab["P_good"]/dtab["P_bad"])
    dtab.loc[:,'local_iv']=(dtab.loc[:,'P_good']-dtab.loc[:,'P_bad'])*dtab.loc[:,'woe']
    iv_temp = dtab.loc[:,'local_iv'].sum()
    if iv_temp == Inf:
        outliers.append(i)
    else:
        iv_name.append(i)
        iv_value.append(dtab.loc[:,'local_iv'].sum())
        iv_unique.append(len(newbins))

Info_value=pd.DataFrame({'name':iv_name,'iv_value':iv_value,'iv_unique()':iv_unique})
Info_value.sort_values(by='iv_value',inplace=True)    
        
for i in Nominal_variable:
    dout = pd.crosstab(xdata[i],ydata)
    dtab.loc[:,'P_good']=dtab.loc[:,0]/total_good
    dtab.loc[:,'P_bad']=dtab.loc[:,1]/total_bad
    dtab.loc[:,'woe'] = np.log(dtab["P_good"]/dtab["P_bad"])
    dtab.loc[:,'local_iv']=(dtab.loc[:,'P_good']-dtab.loc[:,'P_bad'])*dtab.loc[:,'woe']
    iv_temp = dtab.loc[:,'local_iv'].sum()
    print(dout)
