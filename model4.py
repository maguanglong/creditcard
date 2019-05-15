# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:46:12 2019

@author: zhupi
"""
import pandas as pd


data=pd.read_excel('model_p.xlsx')
head=data.keys()

writer = pd.ExcelWriter('data_out.xlsx')
for name in head:
    if name != 'y':
        df=data[name]
        dout=pd.crosstab(df,data.y)
        dout["total"]=dout.loc[:,1]+dout.loc[:,0]
        dout["P_bad"]=dout.loc[:,1]/df.count()
        dout["G_bad"]=dout.loc[:,1]/dout["total"]
        #print(dout)
        if len(name)>30:
            name = str(len(name))
        dout.to_excel(writer,sheet_name=name)
        
        
#writer = pd.ExcelWriter('missing_out.xlsx')

do1=[]
do2=[]
for name in head:
    df=data[name]
    if df.dtype==('int64'):
        do1.append(df[(df==-9999)].count())
        do2.append(df[(df==-9999)].count()/df.count())
    elif df.dtype==('float64'):
        do1.append(df[(df==-9999)].count())
        do2.append(df[(df==-9999)].count()/df.count())
    else:
        do1.append(df[(df=="-9999")].count())
        do2.append(df[(df=="-9999")].count()/df.count())

dout2=pd.DataFrame({"name":head, "missing_coun":do1, "missing_rate":do2})

dout2.to_excel("missing_out.xlsx",sheet_name="missing_data")


        
date=dout2[dout2.missing_rate<0.25]
head=date.name

writer = pd.ExcelWriter('out2.xlsx')
"""
for name in head:
    if name != 'y':
        df=data[name]
        dout=pd.crosstab(df,data.y)
        dout["total"]=dout.loc[:,1]+dout.loc[:,0]
        dout["P_bad"]=dout.loc[:,1]/df.count()
        dout["G_bad"]=dout.loc[:,1]/dout["total"]
        print(dout)
        if len(name)>30:
            name = str(len(name))
        dout.to_excel(writer,sheet_name=name)
"""
writer = pd.ExcelWriter('out2.xlsx')
for name in head:
    if name != "y":
        if len(data[name].unique())<100:
            print("="*5+name+"="*5)
            df=data[name]
            #status_name="y"
            dp=pd.crosstab(df,data.y)
            dp.loc[:,'total']=dp.loc[:,1]+dp.loc[:,0]
            dp.loc[:,'local']=dp.loc[:,0]/(dp.loc[:,0]+dp.loc[:,1])
            dp.loc[:,'global']=dp.loc[:,0]/(dp.loc[:,0].sum())
            dp.loc[:,'global_cum']=dp.loc[:,'global'].cumsum()
            #print(dp)
            if len(name)>30:
                name = str(len(name))
            dp.to_excel(writer,sheet_name=name)
