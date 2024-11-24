import pandas as pd
from sklearn.preprocessing import MinMaxScaler
col_names = ["unit_no","time","os1","os2","os3"] + [f'sm{i}' for i in range(1,22)] 
df_train = pd.read_csv("./CMAPSSData/train_FD001.txt",sep=" ",header=None,names=col_names,index_col=False)
df_test = pd.read_csv("./CMAPSSData/test_FD001.txt",sep=" ",header=None,names=col_names,index_col=False)
# y_test = pd.read_csv('/home/lazyguy/Sem 5/ELL409/Project/CMAPSSData/RUL_FD001.txt' , header=None , names=['rul'] )
def add_operating_condition(df_op_cond):
    
    df_op_cond['os1'] = df_op_cond['os1'].round()
    df_op_cond['os2'] = df_op_cond['os2'].round(decimals=2)
    
    # converting settings to string and concatanating makes the operating condition into a categorical variable
    df_op_cond['op_cond'] = df_op_cond['os1'].astype(str) + '_' + df_op_cond['os2'].astype(str) + '_' + df_op_cond['os3'].astype(str)
    return df_op_cond

df_train = add_operating_condition(df_train)
df_test = add_operating_condition(df_test)
def condition_scaler(df_train, df_test, sensor_names):
    scaler = MinMaxScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond']==condition, sensor_names])
        df_train.loc[df_train['op_cond']==condition, sensor_names] = scaler.transform(df_train.loc[df_train['op_cond']==condition, sensor_names])
        df_test.loc[df_test['op_cond']==condition, sensor_names] = scaler.transform(df_test.loc[df_test['op_cond']==condition, sensor_names])
    df_train = df_train.drop(columns="op_cond")
    df_test = df_test.drop(columns= "op_cond")
    return df_train, df_test
col_names = ["unit_no","time","os1","os2","os3"] + [f'sm{i}' for i in range(1,22)] 
df_train,df_test =  condition_scaler(df_train,df_test,col_names[2:])
file_path = './CMAPSSData/train_FD001_normed.txt'
df_train.to_csv(file_path, sep=' ', index=False, header=False)
df_test.to_csv("./CMAPSSData/test_FD001_normed.txt",sep = " ",index=False,header=False)