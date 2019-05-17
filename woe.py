import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from numpy import NaN,Inf
from sklearn.metrics import roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib
import statsmodels.api as sm

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
        dout=pd.cut(df['value'],newbins)
    else:
        indexmap={}
        #ts = df[df[name]!=-9999]
        ts=df
        dout, bins = pd.qcut(ts[name],nbins-1,retbins='True',duplicates='drop')
        newbins=bins
        #a3 = np.array([-Inf,-9999])
        #newbins = np.zeros(len(bins)+len(a3))
        #newbins[:len(a3)] = a3
        #newbins[len(a3):]=bins
        #newbins[-1]=np.array([Inf])
        #dout=pd.cut(df[name],newbins)
        
    return dout, newbins, indexmap
    

def woe_coarse(dout,dy):
    dtab = pd.crosstab(dout,dy)
    dtab["total"]=dtab.loc[:,1]+dtab.loc[:,0]
    dtab["L_good"]=dtab.loc[:,0]/dtab.loc[:,0].sum()
    dtab["L_bad"]=dtab.loc[:,1]/dtab.loc[:,1].sum()
    dtab["woe"]=np.log(dtab["L_good"]/dtab["L_bad"])
    dtab = dtab[(dtab.L_good * dtab.L_bad) !=0]
    dtab.loc[:,"IV"]=(dtab["L_good"]-dtab["L_bad"])*np.log(dtab["L_good"]/dtab["L_bad"])
    iv = dtab.IV.cumsum()[-1]
    return dtab,iv,dtab.woe

def woe_change_value(dout, name, woe_t):
    a=[]
    woe_dout = dout
    for i in woe_dout.unique():
        a.append(i)
        a.sort()
    for m in range(len(a)):
        woe_dout.replace(a[m], woe_t.values[m], inplace = True)
    woe_name = 'woe_' + name
    return woe_dout, woe_name

df = pd.read_csv('data.csv')
key = df.keys()
y_name = 'status'
i_iv,i_name=[],[]
for name in key:
    if name != y_name:
        nbins=8
        dout, indexmap, newbins=devide_coarse(df,name,y_name,nbins)
        dtab,iv_t,woe=woe_coarse(dout, df['status'])
        i_iv.append(iv_t)
        i_name.append(name)
IV=pd.DataFrame(data={'name':i_name,'IV':i_iv})

out_name = IV[IV['IV']>0.02].name
xdata, ydata = df[out_name], df[y_name]

head = xdata.keys()
xwoe = pd.DataFrame()
for name in head:
    nbins = 5
    dout, indexmap, newbins=devide_coarse(df,name,y_name,nbins)
    dtab,iv_t,woe_t=woe_coarse(dout, df[y_name])
    woe_out, woe_name = woe_change_value(dout, name, woe_t)
    xwoe.loc[:,woe_name] = woe_out


## 模型验证

x_train, x_test, y_train, y_test = train_test_split(xwoe, df[y_name], test_size = 0.4, random_state = 0)
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
