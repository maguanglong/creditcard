import pandas as pd
import numpy as np
import math
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

'''
已完成：数据清洗；数据分箱；数据转化；模型拟合等
已得到：
1）入模变量；2）WOE数据；3）逻辑回归系数，包括intercept;
目标完成：
* 制作评分卡；
* 把WOE值转化成分值，并得到最终的评分卡分数；
* 对分数进行分段统计；
'''
def DataToScore(datawoe,coe):
# 读取WOE数据转化为评分卡分值
    B = 20 / math.log(2)     #p值（比例因子） 
    A = 600 - 20 * math.log(20) / math.log(2)    # 
    basescore = round(A - B *coe['intercept'] , 0)
    temp = datawoe.copy()
    datawoe = datawoe.multiply(coe)*B
    datascore = datawoe.sum(axis=1)
    datascore = basescore - datascore
    datawoe = temp
    return round(datascore)


def CrosstabSocreCard(dx,dy):
#制作透视表，观测分组坏占比
    dx = pd.qcut(dx,10,duplicates='drop')
    N = len(dy)
    #NG = dy[dy==0].count()
    NB = dy[dy==1].count()
    #if bins in input_list:
    #    dx = pd.cut(dx,bins)
    ytab = pd.crosstab(dx,dy)
    ytab.loc[:,'Bad_proportion_in_group'] = ytab.iloc[:,1]/(ytab.iloc[:,0]+ytab.iloc[:,1])
    ytab.loc[:,'Proportion'] = (ytab.iloc[:,0]+ytab.iloc[:,1])/N
    ytab.loc[:,'Bad_proportion_among_group'] = ytab.iloc[:,1]/NB
    ytab.loc[:,'Cumsum_proportion'] = ytab.loc[:,'Proportion'].cumsum()
    ytab.loc[:,'Cumsum_Bad_proportion'] = ytab.loc[:,'Bad_proportion_among_group'].cumsum()
    #ytab.plot(x=ytab.index,y=['Cumsum_Bad_proportion_among_group','Cumsum_proportion'],kind = 'line')
    #ytab.plot(x=ytab.index,y=['Cumsum_Bad_proportion_among_group','Cumsum_proportion'],kind = 'line')
    #ytab.plot(x=ytab.index,y=['Bad_proportion_in_group','Proportion','Bad_proportion_among_group'],kind = 'line')
    return ytab
 
# 读取woe表格 
datawoe = pd.read_csv('datawoe.csv')
# 分成自变量和因变量
X = datawoe.iloc[:,1:26]
y = datawoe['status']
# 分成训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)
# 逻辑回归
lr = LogisticRegression(solver = 'lbfgs')#指定solver只是为了不出现提示信息
lr.fit(X_train, y_train)#模型训练
y_pred = lr.predict(X_test)#输出结果，感觉没啥用

coe = pd.Series(lr.coef_[0],index = X_train.keys()) 
coe['intercept']=lr.intercept_[0]# 得到模型的系数，储存为pandas.Series格式

print('LR without weight:',metrics.confusion_matrix(y_test, y_pred))    
print('coef without weight:',lr.coef_)
print('socre without weight:',lr.score(X_test,y_test))


strain = DataToScore(X_train,coe)# 制作训练集评分卡
stest = DataToScore(X_test,coe)# 制作测试集评分卡

train_tab = CrosstabSocreCard(strain,y_train)#分组检测
test_tab = CrosstabSocreCard(stest,y_test)#分组检测

print('test_tab_b',test_tab_b.iloc[:,2:4])
print('test_tab',test_tab.iloc[:,2:4])
