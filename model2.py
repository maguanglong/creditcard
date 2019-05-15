import pandas as pd
import numpy as np
import woe_coarse as woe

df=pd.read_excel('model_p.xlsx')

varlist, pre1=woe.missing_coarse(df)

IV1=woe.iv_coarse(df,pre1,"name",6)
IV2=woe.iv_coarse(df,pre1,"name",7)
IV3=woe.iv_coarse(df,pre1,"name",8)
IV4=woe.iv_coarse(df,pre1,"name",10)

varlist = IV4[IV4['info_value']>0.03]['name']
data=df[varlist]
data.loc[:,'status']=df['y']

name='zxloanbalanceamt'

temp1=df[(df[name]!=-9999)]
temp2=df[(df[name]==-9999)]


temp=pd.crosstab(data[name],data['status'])
fig,axes = plt.subplots()
data[name].plot(kind='box',ax=axes)
axes.set_ylabel(name)

'''
对于连续变量，先派出缺失值，得到data1;
再对data2进行等频分割，得到data3 和范围;
对范围加上缺失值，再和status进行统计;
计算woe值；
'''
'''
varlist=55                    zxloanbalanceamt
24        zxcrecardacctnowtotaldelqamt
15              zxrcy3mcrecardquerycnt
34                       zxissubankcnt
47      zxcreditcarddelqstatus24months
31                        zxaccountcnt
20            zxcrecardnowshouldpayamt
49     zxcreditcarddelqstatus30dayyues
29              zxrcy6mavgusecreditamt
16       zxrcy6mcrecardquerycompanycnt
33                      zxusecreditamt
22                  zxfirstcrecarddate
36              zxcrecardusecreditrate
23                         zxcreditbal
46    zxcreditcarddelqstatus60to90days
50    zxcreditcarddelqstatus30to60days
17              zxrcy6mcrecardquerycnt
45    zxcreditcarddelqstatus6to12month
48    zxcreditcarddelqstatus90to180day
28             zxcreditcardaccudelqcnt
26        zxrcy24mcrecardmaxdelqperiod
25        zxrcy12mcrecardmaxdelqperiod
27         zxrcy6mcrecardmaxdelqperiod
4                       zxrcyquerydate
3                            dxmascore
'''
name='zxloanbalanceamt'
len(df[name].unique())
df[name].value_counts()
temp1=df[(df[name]!=-9999)]
dcut,bins=pd.qcut(temp1[name],10,retbins=True)
'''
newbins=np.array([-9999,  0.00000000e+00,   4.50000000e+03,   1.05768000e+04,
         2.00000000e+04,   3.46682000e+04,   5.66620000e+04,
         9.95254000e+04,   1.92364900e+05,   3.38271800e+05,
         5.82870200e+05,   2.77574880e+07])
'''
temp1=df[(df[name]>0)]
dcut,bins=pd.qcut(temp1[name],10,retbins=True)
dcut,bins=pd.cut(df[name],newbins,retbins=True)
dout=pd.crosstab(dcut,data['status'])

'''
画图
import seaborn as sns
corr=data.corr()
xticks = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9',
          'x10','x11','x12','x13','x14','x15','x16','x17','x18','x19',
          'x20','x21','x22','x23','x24','x25']
fig=plt.figure()
fig.set_size_inches(25,12)
ax1=fig.add_subplot(111)
# 默认是0到1的，vmin和vmax可自定义设置
sns.heatmap(corr,vmin=-1, vmax=1 ,cmap='hsv', annot=True, square=True)
ax1.set_xticklabels(xticks,rotation= 0)
plt.show()
'''
'''
vname='name'
nb=6
variv=[]
varname=[]
for name in pre1[vname]:
    num = len(df[name].unique())
    temp=df[(df[name]!=-9999)]
    if num<58:
        print("名义变量%d"%(num))
        dout =  pd.crosstab(temp[name],temp.y)
    elif (num>58 and num<10000):
        print("顺序变量%d"%(num))
        dcut=pd.cut(temp[name],nb)
        dout =  pd.crosstab(dcut,temp.y)
    else:
        print("连续变量%d"%(num))
        dcut=pd.qcut(temp[name],nb)
        dout =  pd.crosstab(dcut,temp.y)

    dout["total"]=dout.loc[:,1]+dout.loc[:,0]
    dout["L_good"]=dout.loc[:,0]/dout.loc[:,0].sum()
    dout["L_bad"]=dout.loc[:,1]/dout.loc[:,1].sum()
    dout["woe"]=np.log(dout["L_good"]/dout["L_bad"])
    dout=dout[(dout.L_good * dout.L_bad) !=0]
    dout["IV"]=(dout["L_good"]-dout["L_bad"])*np.log(dout["L_good"]/dout["L_bad"])
    variv.append(dout.IV.cumsum().max())
    varname.append(name)
IV=pd.DataFrame({'name':varname,'info_value':variv}) 
'''




'''
head = df.keys()
dotype, domiss, dorate = [], [], []
vartype = []
for name in head:
    ds = df[name]
    dotype.append(ds.dtype)
    vartype.append(len(ds.unique()))
    if ds.dtype=='int64':
        mc, mn = ds[ds==-9999].count(), ds.count()
        domiss.append(mc)
        dorate.append(mc / mn)
    elif ds.dtype=='float64':
        mc, mn = ds[ds==-9999].count(), ds.count()
        domiss.append(mc)
        dorate.append(mc / mn)
    else:
        mc, mn = ds[ds=='-9999'].count(), ds.count()
        domiss.append(mc)
        dorate.append(mc / mn)
out = pd.DataFrame(data={'name': head,'missing_count':domiss, 'missing_rate': dorate, 'vartype':vartype})
out=out.sort_values(by="missing_rate")
out.to_csv('missing_rate.csv')
pre1=out[(out['missing_rate']<0.3)&(out['missing_rate']>0)]
varlist = pre1.name
print(pre1)


var = pre1[pre1.vartype<58]
variv=[]
varname=[]
for name in var['name']:
#name = 'zxbadbebtsbalance'
    dout =  pd.crosstab(df[name],df.y)
    dout["total"]=dout.loc[:,1]+dout.loc[:,0]
    dout["L_good"]=dout.loc[:,0]/dout.loc[:,0].sum()
    dout["L_bad"]=dout.loc[:,1]/dout.loc[:,1].sum()
    dout["woe"]=np.log(dout["L_good"]/dout["L_bad"])
    dout=dout[(dout.L_good * dout.L_bad) !=0]
    dout["IV"]=(dout["L_good"]-dout["L_bad"])*np.log(dout["L_good"]/dout["L_bad"])
    variv.append(dout.IV.cumsum().max())
    varname.append(name)
    
    #print("IV value of %s is %f"%(name,variv))
    print('='*50)
    
IV = pd.DataFrame({'name':varname,'info_value':variv})


def box_png(df,name):
    temp=df[(df[name]!=-9999)]
    fig,axes = plt.subplots()
    temp[name].plot(kind='box',ax=axes)
    axes.set_ylabel(name)
    fig.savefig('%s.png'%(name))
    

#print(dout)
variv=[]
varname=[]
var2 = pre1[pre1.vartype>1000]
for name in var2['name']:
    temp=df[(df[name]!=-9999)]
    fig,axes = plt.subplots()
    temp[name].plot(kind='box',ax=axes)
    axes.set_ylabel(name)
    fig.savefig('%s.png'%(name))
    dcut=pd.qcut(temp[name],4)
    dout =  pd.crosstab(dcut,temp.y)
    dout["total"]=dout.loc[:,1]+dout.loc[:,0]
    dout["L_good"]=dout.loc[:,0]/dout.loc[:,0].sum()
    dout["L_bad"]=dout.loc[:,1]/dout.loc[:,1].sum()
    dout["woe"]=np.log(dout["L_good"]/dout["L_bad"])
    dout=dout[(dout.L_good * dout.L_bad) !=0]
    dout["IV"]=(dout["L_good"]-dout["L_bad"])*np.log(dout["L_good"]/dout["L_bad"])
    variv.append(dout.IV.cumsum().max())
    varname.append(name)
    print(name)
IV2 = pd.DataFrame({'name':varname,'info_value':variv})


variv=[]
varname=[]
var3 = pre1[(pre1.vartype<1000)&(pre1.vartype>=58)]
for name in var3['name']:
    temp=df[(df[name]!=-9999)]
    fig,axes = plt.subplots()
    temp[name].plot(kind='box',ax=axes)
    axes.set_ylabel(name)
    fig.savefig('%s.png'%(name))
    dcut=pd.cut(temp[name],4)
    dout =  pd.crosstab(dcut,temp.y)
    dout["total"]=dout.loc[:,1]+dout.loc[:,0]
    dout["L_good"]=dout.loc[:,0]/dout.loc[:,0].sum()
    dout["L_bad"]=dout.loc[:,1]/dout.loc[:,1].sum()
    dout["woe"]=np.log(dout["L_good"]/dout["L_bad"])
    dout=dout[(dout.L_good * dout.L_bad) !=0]
    dout["IV"]=(dout["L_good"]-dout["L_bad"])*np.log(dout["L_good"]/dout["L_bad"])
    variv.append(dout.IV.cumsum().max())
    varname.append(name)
    print(name)
IV3 = pd.DataFrame({'name':varname,'info_value':variv})    
    
    
variv=[]
varname=[]
for name in pre1['name']:
    num = len(df[name].unique())
    temp=df[(df[name]!=-9999)]
    #fig,axes = plt.subplots()
    #temp[name].plot(kind='box',ax=axes)
    #axes.set_ylabel(name)
    #fig.savefig('%s.png'%(name))
    if num<58:
        print("名义变量%d"%(num))
        dout =  pd.crosstab(df[name],df.y)
    elif (num>58 and num<1000):
        print("顺序变量%d"%(num))
        dcut=pd.cut(temp[name],4)
        dout =  pd.crosstab(dcut,temp.y)
    else:
        print("连续变量%d"%(num))
        dcut=pd.qcut(temp[name],4)
        dout =  pd.crosstab(dcut,temp.y)
    dout["total"]=dout.loc[:,1]+dout.loc[:,0]
    dout["L_good"]=dout.loc[:,0]/dout.loc[:,0].sum()
    dout["L_bad"]=dout.loc[:,1]/dout.loc[:,1].sum()
    dout["woe"]=np.log(dout["L_good"]/dout["L_bad"])
    dout=dout[(dout.L_good * dout.L_bad) !=0]
    dout["IV"]=(dout["L_good"]-dout["L_bad"])*np.log(dout["L_good"]/dout["L_bad"])
    variv.append(dout.IV.cumsum().max())
    varname.append(name)
    
IV=pd.DataFrame({'name':varname,'info_value':variv})    
'''   
