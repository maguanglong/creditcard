# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:30:20 2019

@author: guanglong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from numpy import NaN,Inf
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
import matplotlib
import statsmodels.api as sm

from sklearn.linear_model import RandomizedLogisticRegression as RLR

from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model.logistic import LogisticRegression
#from sklearn.cross_validation import train_test_split

df = pd.read_csv('datawoe.csv')

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

dataset_woe = df[['taxmonthlyincomesection', 'credit_usedrate',
       'zxrcy6mcrecardmaxdelqperiod', 'zxrcy12mcrecardmaxdelqperiod',
       'preacustseg', 'zxrcy24mcrecardmaxdelqperiod', 'cre_ovd_cntrate',
       'zxcreditcardrcyupdatedate1', 'zxcrecarddelqmaxamt', 'all_ovd_cnt',
       'zxcreditbal', 'all_ovd_month', 'zxfirstcrecarddate1', 'zxdelqcnt',
       'cred_3month_badstatus', 'zxcrecardnowshouldpayamt',
       'zxrcy6mcrecardquerycnt', 'zxissubankcnt', 'jigou_sixmonth',
       'zxmincreditamt', 'ave_loan_balance', 'zxcreditavglimit',
       'zxmaxupdated1', 'zxblanceshouldpayamt', 'age', 'status']]

'''
model = LogisticRegression()
clf = model.fit(x_train, y_train)
print('scorecard:{}'.format(clf.score(x_test, y_test)))
y_pred = clf.predict(x_test)
y_pred1 = clf.decision_function(x_test) 
'''

def forward_selection(x_train,y_train,namein,nameset,sle):
    x1 = x_train[nameset]
    xmodel = sm.add_constant(x1)
    logit = sm.Logit(y_train,xmodel)
    result=logit.fit()
    print(result.summary())
    t_value = abs(result.tvalues)
    p_value = result.pvalues
    k1 = p_value[p_value<sle].keys()
    if 'const' in k1:
        k1 = k1[1:]
    if len(k1)!=0:
        in_name = t_value[k1].idxmax()
        print(in_name, p_value[in_name],t_value[in_name])
        namein.append(in_name)
        nameset.pop(nameset.index(in_name))
    else:
        in_name = ''
        print('no new variable to insert')
    return namein, nameset, in_name

def backward_selection(x_train,y_train,nameout,nameset,sls):
    x1 = x_train[nameset]
    xmodel = sm.add_constant(x1)
    logit = sm.Logit(y_train,xmodel)
    result=logit.fit()
    print(result.summary())
    t_value = abs(result.tvalues)
    p_value = result.pvalues
    k1 = p_value[p_value>sls].keys()
    if len(k1)!=0:
        out_name = t_value[k1].idxmax()
        print(out_name, p_value[out_name],t_value[out_name])
        nameout.append(out_name)
        nameset.pop(nameset.index(out_name))
    else:
        out_name = ''
        print('no new varibale to remove')
    return nameout, nameset, out_name
    
    


def model_selection(x_train,y_train,method,sle, sls):
    namein, nameout, nameset=[],[],list(x_train.keys())
    in_name, out_name = 'const', 'const'
    if method == 'forward':
        while in_name != '':
            namein, nameset, in_name = forward_selection(x_train,y_train,namein,nameset,sle)
        train_set = namein
    elif method == 'backward':
        while out_name !='':
            nameout, nameset, out_name = backward_selection(x_train,y_train,nameout,nameset,sls)
        train_set = nameset  
    elif method == 'all':
        train_set = nameset 
    elif method =='stepwise':
        while in_name !='' or out_name!='':
            namein, nameset, in_name = forward_selection(x_train,y_train,namein,nameset,sle)
            nameout, namein, out_name = backward_selection(x_train,y_train,nameout,namein,sls)
        train_set = namein
    else:
        print('Please use correct methods')
        return method
    x1 = x_train[train_set]
    xmodel = sm.add_constant(x1)
    logit = sm.Logit(y_train,xmodel)
    result=logit.fit()
    return result, train_set
    
           
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size = 0.4, random_state = 0)
sle, sls = 0.05, 0.05
method = 'stepwise'
result,train_set = model_selection(x_train,y_train,method,sle, sls)
print(result.summary())

x = x_train
y = y_train

lr = LR(class_weight='balanced')
lr.fit(x,y)
print('scorecard:',lr.score(x,y))

from sklearn import metrics
y_pred_on_train = lr.predict(x)
y_pred_on_test = lr.predict(x_test)
y_pred_proba = lr.predict_proba(x_test)[:,1]

cm = metrics.confusion_matrix(y_test,y_pred_on_test)
ps = metrics.precision_score(y_test,y_pred_on_test)
rs = metrics.recall_score(y_test,y_pred_on_test)
auc = metrics.roc_auc_score(y_test,y_pred_on_test)
fpr,tpr,thresholds = metrics.roc_curve(y_test,y_pred_proba)

print('accuracy:',metrics.accuracy_score(y_test,y_pred_on_test))
print('comfusion matrix:',cm)
print('recall score:',rs)
print('precision score:',ps)
print('roc-auc score:',auc)
print ('KS:',max(tpr-fpr))

yr = pd.DataFrame({'y_pred':y_pred_proba,'y_test':y_test}
                  
'''
# 特征筛选
x = x_train[train_set].as_matrix()
y = y_train.as_matrix()
rlr = RLR()
rlr.fit(x,y)
rlr.get_support()
t_set2 = x_train[train_set].columns[rlr.get_support()]
print(u'通过随机逻辑回归模型筛选特征结束')
print(u'有效特征为：',t_set2)


train_x = x_train[train_set]

lr=LR()
lr.fit(train_x,y_train)
print('scorecard:{}'.format(lr.score(train_x, y_test)))

from sklearn import metrics
y_pred_on_train = lr.predict(x_train[train_set])
y_pred_on_test = lr.predict(x_test[train_set])
print('accuracy:'.format(metrics.accuracy_score(y_test,y_pred_on_test)))

cm = metrics.confusion_matrix(y_test,y_pred_on_test)

from sklearn.metrics import roc_curve, roc_auc_score
logit_scores_proba = lr.predict_proba(x_train[train_set])
logit_scores = logit_scores_proba[:,1]
def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1], "k--") # 画直线做参考
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive rate")

fpr_logit, tpr_logit, thresh_logit = roc_curve(y_train, logit_scores)
plot_roc_curve(fpr_logit,tpr_logit)
print( 'AUC Score : ', (roc_auc_score(y_train,logit_scores)))
'''
