# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:30:20 2019

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
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size = 0.4, random_state = 0)

def stepwise_model(x_train,y_train):
    namein,nameout=[],[]
    nameset = list(x_train.keys())

    while len(nameset)!=0:
        # forward
        x1 = x_train[nameset]
        xmodel = sm.add_constant(x1)
        logit = sm.Logit(y_train,xmodel)
        result=logit.fit()
        t_value = abs(result.tvalues)
        p_value = result.pvalues
        sle = 0.05
        k1 = p_value[p_value<sle].keys()
        if 'const' in k1:
            k1 = k1[1:]
        if len(k1)!=0:
            in_name = t_value[k1].idxmax()
            print(in_name, p_value[in_name],t_value[in_name])
            namein.append(in_name)
            nameset.pop(nameset.index(in_name))
        
        #backward
        sls = 0.05
        x2 = x_train[namein]
        xmodel = sm.add_constant(x2)
        logit = sm.Logit(y_train,xmodel)
        result=logit.fit()
        t_value = abs(result.tvalues)
        p_value = result.pvalues
        
        k1 = p_value[p_value>=sls].keys()
        if len(k1)!=0:
            out_name = t_value[k1].idxmax()
            print(out_name, p_value[out_name],t_value[out_name])
            if out_name in namein:
                namein.pop(namein.index(out_name))
            if out_name != in_name:
                nameset.append(out_name)
            nameout.append(out_name)

    x2 = x_train[namein]
    xmodel = sm.add_constant(x2)
    logit = sm.Logit(y_train,xmodel)
    result=logit.fit()    
    print(result.summary())
    
    return result,namein
