# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:39:35 2019

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


def data_missing_count(df, tol):
    df_missing_count = pd.DataFrame()
    name = []
    value = []
    for i in df.keys():
        name.append(i)
        if df[i].dtypes=='object':
            value.append(df[df==-9999].count())
        else:
            value.append(df[df=='-9999'].count())
    df_missing_count = pd.DataFrame({'name':name,'missing_amount':value})
    df_missing_count.sort_values(by='missing_amount', inplace = True)
    data_set = df_missing_count[df_missing_count['missing_amount']<tol]
    df_missing_count.to_csv('data_missing_count.csv')
    data_set.to_csv('data_no_missing.csv')
    return data_set


def devide_coarse(data_set,x_name,y_name,nbins):
    if data_set[x_name].dtypes=='object':
        ts = data_set[data_set[x_name]!='-9999']
        indexmap={'-9999':-9999}
        i=0        
        for ele in ts[x_name].unique():
            indexmap[ele]=i
            i=i+1
        data_set.loc[:,'value']=data_set[x_name].map(indexmap)
        ts = data_set[data_set['value']!=-9999]
        t1 = ts['value']
        if (t1[t1==0].count())/t1.count() > 0.5:
            ts= ts[ts['value']!=0]
            nbins=min(len(ts['value'].unique()),nbins)
            dout,bins=pd.qcut(ts['value'],nbins,retbins='True',duplicates='drop')
            newbins = np.zeros(len(bins)+3)
            newbins[:3]=np.array([-Inf,-9999,0])
            newbins[3:]=bins
        else:
            nbins=min(len(ts['value'].unique()),nbins)
            dout,bins=pd.qcut(ts['value'],nbins,retbins='True',duplicates='drop')
            newbins = np.zeros(len(bins)+2)
            newbins[:2]=np.array([-Inf,-9999])
            newbins[2:]=bins
        dout=pd.cut(data_set['value'],newbins)
    else:
        indexmap={}
        ts = data_set[data_set[x_name]!=-9999]
        #ts=data_set
        dout, bins = pd.qcut(ts[x_name],nbins-1,retbins='True',duplicates='drop')
        #newbins=bins
        a3 = np.array([-Inf])
        #newbins = np.zeros(len(bins)+len(a3))
        newbins[:len(a3)] = a3
        newbins[len(a3):] = bins
        #newbins[-1]=np.array([Inf])
        dout=pd.cut(data_set[x_name],newbins)
    # 返回粗分箱后的数据集，分箱，以及名义变量方式    
    return dout, newbins, indexmap
    
def woe_calculate(dout,dy):
    dtab = pd.crosstab(dout,dy)
    dtab["total"]=dtab.loc[:,1]+dtab.loc[:,0]
    dtab["L_good"]=dtab.loc[:,0]/dtab.loc[:,0].sum()
    dtab["L_bad"]=dtab.loc[:,1]/dtab.loc[:,1].sum()
    dtab["woe"]=np.log(dtab["L_good"]/dtab["L_bad"])
    dtab = dtab[(dtab.L_good * dtab.L_bad) !=0]
    dtab.loc[:,"IV"]=(dtab["L_good"]-dtab["L_bad"])*np.log(dtab["L_good"]/dtab["L_bad"])
    iv = dtab.IV.cumsum()[-1]
    return dtab,iv,dtab.woe

def woe_change_value(dout, x_name, woe_t):
    #将dout中的数据改为woe_t中的woe值
    a=[]
    woe_dout = dout
    for i in woe_dout.unique():
        a.append(i)
        a.sort()
    for m in range(len(a)):
        woe_dout.replace(a[m], woe_t.values[m], inplace = True)
    woe_name = 'woe_' + x_name
    return woe_dout, woe_name



def IV_calculate(data_set,y_name,tol):
    key = data_set.keys()
    i_iv,i_name=[],[]
    for x_name in key:
        if x_name != y_name:
            nbins=8
            dout, indexmap, newbins=devide_coarse(data_set,x_name,y_name,nbins)
            dtab,iv_t,woe=woe_calculate(dout, data_set['status'])
            i_iv.append(iv_t)
            i_name.append(x_name)
    Info_value=pd.DataFrame(data={'x_name':i_name,'IV':i_iv})
    out_name = Info_value[Info_value['IV']>tol].x_name
    xdata, ydata = data_set[out_name], data_set[y_name]
    return Info_value, xdata, ydata

# 评分卡制作
def get_score_1(dtab,coe,p):
    dtab.loc[:,'score'] = round(coe * p * dtab.loc[:,'woe'],0)
    return dtab.loc[:,'score']

def creat_creditcard(dtab,coe):
    p = 20/np.log(2)
    q = 600 - 20*np.log(20)/np.log(2)
    baseScore = round(q + p * coe[0], 0)
    dtab.loc[:,'score'] = get_score_1(coe[-1],dtab,p)



df = pd.read_csv('data.csv')
data_set = data_missing_count(df, 0.5)
y_name = 'status'
Info_value, xdata, ydata = IV_calculate(data_set,y_name,0.2)
data_woe = pd.DataFrame()
for x_name in xdata.keys():
    nbins = 5
    dout, indexmap, newbins=devide_coarse(data_set,x_name,y_name,nbins)
    dtab,iv_t,woe_t=woe_calculate(dout, data_set[y_name])
    woe_out, woe_name = woe_change_value(dout, x_name, woe_t)
    data_woe.loc[:,woe_name] = woe_out


## 模型验证

x_train, x_test, y_train, y_test = train_test_split(data_woe, data_set[y_name], test_size = 0.4, random_state = 0)
model = LogisticRegression()
clf = model.fit(x_train, y_train)
print('scorecard:{}'.format(clf.score(x_test, y_test)))
y_pred = clf.predict(x_test)
y_pred1 = clf.decision_function(x_test) 

x1=sm.add_constant(x_train)
logit=sm.Logit(y_train,x1)
result=logit.fit()
print(result.summary())

matplotlib.rcParams['font.sans-serif'] =['Microsoft YaHei']    # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False

#通过ROC曲线和AUC来评估模型的拟合能力。
X2=sm.add_constant(x_test)
resu=result.predict(X2)
fpr,tpr,threshold=roc_curve(y_test,resu) 
rocauc=auc(fpr,tpr)
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% rocauc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

for x_name in xdata.keys():
    nbins = 5
    dout, indexmap, newbins=devide_coarse(data_set,x_name,y_name,nbins)
    dtab,iv_t,woe_t=woe_calculate(dout, data_set[y_name])
    #dtab.to_excel('woe.xlsx',sheet_name=x_name)


