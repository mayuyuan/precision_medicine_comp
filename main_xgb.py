#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 11:01:05 2018
@author: mayuyuan
"""
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import decomposition
from sklearn.model_selection import train_test_split
#读取数据
train=pd.read_csv('d_train_20180102.csv', encoding='gbk', parse_dates=['体检日期'], dtype={'血糖':np.float64})
train=train[train['性别'].isin(['男', '女'])]#性别不明者占比特别少，所以直接删掉。如果占比多，就另寻方法处理。
test=pd.read_csv('d_test_A_20180102.csv', encoding='gbk', parse_dates=['体检日期'])
test['血糖']='unknown'
data=pd.concat([train, test], axis=0)
data=data.sample(frac = 1) #乱序
del data['体检日期']
# onehot
data=pd.concat([pd.get_dummies(data['性别']), data], axis=1)
del data['性别']
# 去除缺失值太多的特征
data=data.loc[:,data.count()/data.shape[0]>0.5]
names_without_sex=data.columns.drop(['id', '血糖', '男', '女'])
# 缺失值填充
imputer=preprocessing.Imputer().fit(data.loc[data['血糖']!='unknown', names_without_sex])
data[names_without_sex]=imputer.transform(data[names_without_sex])
# outlier处理
scaler=preprocessing.RobustScaler(with_centering=False, with_scaling=False, 
                quantile_range=(1, 99)).fit(data.loc[data['血糖']!='unknown', names_without_sex])
data[names_without_sex]=scaler.transform(data[names_without_sex])
# 特征变换
for i in range(len(names_without_sex)):
    data[names_without_sex[i]+'_log1p']=data[names_without_sex[i]].map(np.log1p)
    data[names_without_sex[i]+'_sqrt']=data[names_without_sex[i]].map(np.sqrt)
    for j in range(i+1, len(names_without_sex)):
        data[names_without_sex[i]+'*'+names_without_sex[j]]=data[names_without_sex[i]]*data[names_without_sex[j]]
del i,j
# 特征选择
names=data.columns.drop(['id', '血糖'])
    #划分训练集和测试集
train_xs=data.loc[data['血糖'] != 'unknown', names]
train_ys=data.loc[data['血糖'] != 'unknown', '血糖']
test_x  =data.loc[data['血糖'] == 'unknown', names]
test_y  =data.loc[data['血糖'] == 'unknown', ['id', '血糖']]
    #特征选择:方差选择
VarianceThreshold=feature_selection.VarianceThreshold(threshold=0.1).fit(train_xs)
train_xs=VarianceThreshold.transform(train_xs)
test_x  =VarianceThreshold.transform(test_x)
    #特征选择:SelectPercentile
SelectPercentile=feature_selection.SelectPercentile(feature_selection.f_regression, percentile=50).fit(
                                        train_xs, train_ys.map(np.float64))
train_xs=SelectPercentile.transform(train_xs)
test_x  =SelectPercentile.transform(test_x)
# 降维
PCA=decomposition.PCA(n_components=50).fit(train_xs)
train_xs=PCA.transform(train_xs)
test_x  =PCA.transform(test_x)
#%%
# 划分训练集和测试集
X_train, X_test, y_train, y_test=train_test_split(train_xs, train_ys, test_size=0.1)
dtrain=xgb.DMatrix(X_train, label=y_train)
dtest=xgb.DMatrix(X_test, label=y_test)
evallist  = [(dtest,'eval'), (dtrain,'train')]
#设置参数  
num_trees=450  
params = {"booster": "gbtree",# gbtree使用树模型，gblinear使用线性模型
          "objective": "reg:linear",
          "eta": 0.15, # step size shrinkage 收缩步长
          "max_depth": 4,# maximum depth of a tree
          "subsample": 0.7,# 用于训练模型的子样本占整个样本集合的比例, 防过拟合
          "silent": 1# 取0时表示打印出运行时信息，取1时表示以缄默方式运行
          }
gbm = xgb.train(params, dtrain, num_trees, evals=evallist)


