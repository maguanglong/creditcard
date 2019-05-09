from pandas import Series,DataFrame
def describe_data(data,ifprint):
    out={'method':['count','min','max','argmin','argmax','idxmin','idxmax','quantile',
               'sum','mean','median','mad','var','std','skew','kurt']}
    dfout=DataFrame(out)
    varlist = data.keys()
    for var_name in varlist:
        vara=data[var_name]
        var_value=Series([vara.count(),
                          vara.min(),
                          vara.max(),
                          vara.argmin(),
                          vara.argmax(),
                          vara.idxmin(),
                          vara.idxmax(),
                          vara.quantile(),
                          vara.sum(),
                          vara.mean(),
                          vara.median(),
                          vara.mad(),
                          vara.var(),
                          vara.std(),
                          vara.skew(),
                          vara.kurt()])
        dfout[var_name]=var_value
    if ifprint==1:
        print(dfout)
    
    return dfout
