# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:57:41 2019

@author: guanglong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from numpy import NaN,Inf,null
from sklearn.metrics import roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib
import statsmodels.api as sm
from sklearn.cluster import KMeans
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
        Continuous_variable.append(i)
        #if len(xdata[i].unique())<13:
        #    Nominal_variable.append(i)
        #else:
        #    Continuous_variable.append(i)
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
#for i in Continuous_variable:
ds = xdata[i]
nbins = 20
ts = ds[ds!=-9999]
################## 分箱 ##################################
#dout, bins = pd.qcut(ts, nbins-1, retbins = 'True',duplicates='drop')
dout, bins = pd.cut(ts, nbins-1, retbins = 'True')
newbins = np.zeros(len(bins)+1)
newbins[1:] = bins
newbins[:2]=np.array([-Inf,bins[0]-1])

###################### IV ##############################3
dout = pd.cut(ds, newbins)
total_good, total_bad  = ydata.value_counts()
dtab = pd.crosstab(dout,ydata)
dtab.loc[:,'P'] = (dtab.loc[:,0]+dtab.loc[:,1])/(total_bad+total_good)
dtab.loc[:,'P_good']=dtab.loc[:,0]/total_good
dtab.loc[:,'P_bad']=dtab.loc[:,1]/total_bad
dtab.loc[:,'woe'] = np.log(dtab["P_good"]/dtab["P_bad"])
dtab.loc[:,'local_iv']=(dtab.loc[:,'P_good']-dtab.loc[:,'P_bad'])*dtab.loc[:,'woe']
iv_temp = dtab.loc[:,'local_iv'].sum()

iv_name.append(i)
iv_value.append(dtab.loc[:,'local_iv'].sum())
iv_unique.append(len(newbins))
    #if iv_temp == Inf:
    #    outliers.append(i)
    #else:
    #    iv_name.append(i)
    #    iv_value.append(dtab.loc[:,'local_iv'].sum())
    #    iv_unique.append(len(newbins))

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
        
        
def data_type_count(xdata):
    int_variable,float_variable,str_variable,error_variable=[],[],[],[]
    for i in xdata.keys():
        ditype = xdata[i].dtype
        if ditype == 'int64':
            int_variable.append(i)
        elif ditype=='float64':
            float_variable.append(i)
        elif ditype=='object':
            str_variable.append(i)
        else:
            error_variable.append(i)
    return int_variable,float_variable,str_variable,error_variable
        
    
    
def dtab_cal(dtab):
    total_good, total_bad  = ydata.value_counts()
    dtab.loc[:,'P'] = (dtab.loc[:,0]+dtab.loc[:,1])/(total_bad+total_good)
    dtab.loc[:,'P_good']=dtab.loc[:,0]/total_good
    dtab.loc[:,'P_bad']=dtab.loc[:,1]/total_bad
    dtab.loc[:,'woe'] = np.log(dtab["P_good"]/dtab["P_bad"])
    return dtab
    

def woe_judge_int(df,name,ydata,isprint):
    dtab1 = pd.crosstab(df[name],ydata)
    dtab1 = dtab_cal(dtab1)
    newbins=[]
    newbins.append(-Inf)
    newbins.append(-9999)
    ds = df[name]
    uniques = sorted(ds[ds!=-9999].unique())
    total=ds.count()
    t=0
    for i in uniques:
        t = t + ds[ds==i].count()
        if t > 0.05 * total:
            newbins.append(i)
            t=0
    newbins[-1]=Inf
    #newbins.append(Inf)
    newbins=sorted(newbins)
        
    dcut = pd.cut(ds, newbins)
    total_good, total_bad  = ydata.value_counts()
    dtab = pd.crosstab(dcut,ydata)
    dtab= dtab_cal(dtab)
    if isprint == True:
        print(dtab1)    
        print('-'*50)
        print(dtab)
        print('='*25+'self'+'='*25)
    else:
        dtab.loc[:,'local_iv']=(dtab.loc[:,'P_good']-dtab.loc[:,'P_bad'])*dtab.loc[:,'woe']
        iv_temp = dtab.loc[:,'local_iv'].sum()
        return dtab,iv_temp,newbins
        
    

def woe_judge_equl(df,name,ydata,isprint):
    ds = df[name]
    ts = ds[ds!=-9999]
    dcut,bins =pd.qcut(ts, nbins-1, retbins = 'True',duplicates='drop')
    newbins = []
    newbins.append(-Inf)
    newbins.append(-9999)
    for i in bins:
        newbins.append(i)
    #newbins = np.zeros(len(bins)+2)
    #newbins[0:2]=np.array([-Inf,-9999])
    #newbins[2:len(bins)+2]=bins
    #newbins[-1]=Inf
    dcut = pd.cut(ds, newbins)
    total_good, total_bad  = ydata.value_counts()
    dtab = pd.crosstab(dcut,ydata)
    dtab= dtab_cal(dtab)
    if isprint == True:
        print(dtab)
        print('='*25+'equal'+'='*25)
    else:
        dtab.loc[:,'local_iv']=(dtab.loc[:,'P_good']-dtab.loc[:,'P_bad'])*dtab.loc[:,'woe']
        iv_temp = dtab.loc[:,'local_iv'].sum()
        return dtab,iv_temp,newbins
    
def woe_judge(df,name,ydata,method):
    if method == 0:
        woe_judge_int(df,name,ydata,True)
        woe_judge_equl(df,name,ydata,True)
    elif method ==1:
        dtab,iv_temp,newbins = woe_judge_int(df,name,ydata,False)
        return dtab, iv_temp,newbins
    elif method ==2:
        dtab,iv_temp,newbins = woe_judge_int(df,name,ydata,False)
        return dtab, iv_temp,newbins
    else:
        print('please use correct method')
    

def woe_improve_cal(df,name,ydata,newbins,value):
    for v in value:
        if v in newbins:
            newbins.pop(newbins.index(v)) 
    dcut = pd.cut(df[name], newbins)
    dtab = pd.crosstab(dcut,ydata)
    dtab= dtab_cal(dtab)
    return dtab,newbins
    
    
    

def cluster_plot(d,k):
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize = (8,3))
    for j in range(0,k):
        plt.plot(ds[d==j],[j for i in d[d==j]],'o')
    plt.ylim(-0.5,k-0.5)
    return plt

def woe_kmeans(df,name,ydata,nbins):
    k=nbins
    ds = df[name]
    kmodel = KMeans(n_clusters = k, n_jobs = 1)
    kmodel.fit(ds.reshape((len(ds),1)))
    c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)
    #c.rolling(window=2,center=False).mean()
    w = pd.rolling_mean(c,2).iloc[1:]
    w = [0]+list(w[0])+[ds.max()]
    d3 = pd.cut(ds,w,labels= range(k))
    cluster_plot(d3,k).show()
    pd.DataFrame({'d3':d3,'ds':ds})
    

def woe_improve(df,name,ydata,newbins):
    #dtab, iv_temp,newbins=woe_judge(df,name,ydata,1)
    dcut = pd.cut(df[name], newbins)
    total_good, total_bad  = ydata.value_counts()
    dtab = pd.crosstab(dcut,ydata)
    dtab= dtab_cal(dtab)
    t1 = dtab['woe']
    t1_e = abs(t1-t1.shift(1))
    r = t1_e.idxmin().right
    newbins.pop(newbins.index(r))
    dcut = pd.cut(df[name], newbins)
    total_good, total_bad  = ydata.value_counts()
    dtab = pd.crosstab(dcut,ydata)
    dtab= dtab_cal(dtab)
    return dtab,newbins 

name = int_variable[12]
woe_judge(df,name,ydata,0)
dtab, iv_temp,newbins=woe_judge(df,name,ydata,1)
dtab,newbins=woe_improve_cal(df,name,ydata,newbins,value)
dout = pd.cut(df[name],newbins,labels=dtab.woe)
dtab = pd.crosstab(dout,ydata)
dtab = dtab_cal(dtab)
dtab.loc[:,'local_iv']=(dtab.loc[:,'P_good']-dtab.loc[:,'P_bad'])*dtab.loc[:,'woe']
iv_temp = dtab.loc[:,'local_iv'].sum()
