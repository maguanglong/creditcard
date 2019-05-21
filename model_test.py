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
import seaborn as sns

df = pd.read_excel('fenduan.xlsx')
for x in df.keys():
    ds = df[x]
    statistics = ds.describe()
    if ds.dtype != 'object':
        statistics.loc['range'] = statistics.loc['max'] - statistics.loc['min'] # 极差
        statistics.loc['var'] = statistics.loc['std'] / statistics.loc['mean'] # 变异系数
        statistics.loc['dis'] = statistics.loc['75%'] - statistics.loc['25%']#四分位数间距
    print(statistics)
    print('='*50)



xdata = df[['taxmonthlyincomesection', 'credit_usedrate',
      'zxrcy6mcrecardmaxdelqperiod', 'zxrcy12mcrecardmaxdelqperiod',
      'preacustseg', 'zxrcy24mcrecardmaxdelqperiod', 'cre_ovd_cntrate',
      'zxcreditcardrcyupdatedate1', 'zxcrecarddelqmaxamt', 'all_ovd_cnt',
      'zxcreditbal', 'all_ovd_month', 'zxfirstcrecarddate1', 'zxdelqcnt',
      'cred_3month_badstatus', 'zxcrecardnowshouldpayamt',
      'zxrcy6mcrecardquerycnt', 'zxissubankcnt', 'jigou_sixmonth',
      'zxmincreditamt', 'ave_loan_balance', 'zxcreditavglimit',
      'zxmaxupdated1', 'zxblanceshouldpayamt', 'age']]
ydata = df['status']


def dtab_cal(dtab):
    total_good, total_bad  = ydata.value_counts()
    dtab.loc[:,'P'] = (dtab.loc[:,0]+dtab.loc[:,1])/(total_bad+total_good)
    dtab.loc[:,'P_good']=dtab.loc[:,0]/total_good
    dtab.loc[:,'P_bad']=dtab.loc[:,1]/total_bad
    dtab.loc[:,'woe'] = np.log(dtab["P_good"]/dtab["P_bad"])
    return dtab

xwoe = pd.DataFrame()

for x in xdata.keys():
    ds = df[x]
    dtab = pd.crosstab(ds,ydata)
    dtab = dtab_cal(dtab)
    a = sorted(df[x].unique())
    b = dtab.woe.values
    for m in range(len(a)):
        ds.replace(a[m], b[m], inplace = True)
    xwoe.loc[:,x]=ds

#######逻辑回归#####################################
x_train, x_test, y_train, y_test = train_test_split(xwoe, ydata, test_size = 0.4, random_state = 0)
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


corr = xdata.corr()
xticks = xdata.keys()
yticks = list(corr.index)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 5,  'color': 'blue'})
ax1.set_xticklabels(xticks, rotation=0, fontsize=10)
ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
plt.show()




