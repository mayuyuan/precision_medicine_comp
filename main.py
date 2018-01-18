#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import decomposition
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
#标准化
scaler=preprocessing.StandardScaler().fit(train_xs)
train_xs=scaler.transform(train_xs)
test_x  =scaler.transform(test_x)
#%%
def layer(layername, x, output_size=1, keep_prob=1.0, lamb=0., activation_function=None):
    # add one more layer and return the output of this layer  
    in_size=x.shape[1].value
    with tf.name_scope(layername):
        with tf.name_scope('Weights'):
            W = tf.Variable(tf.truncated_normal([in_size, output_size],
                            stddev=0.1))
            tf.summary.histogram('value', W)
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamb)(W))
        with tf.name_scope('biases'):
            b = tf.Variable(tf.truncated_normal([1, output_size],
                            stddev=0.1))
            tf.summary.histogram('value', b)
        with tf.name_scope('Wx_plus_b'):
            output = tf.nn.dropout(tf.matmul(x, W)+b, keep_prob)
            if activation_function:
                output = activation_function(output)
    tf.summary.histogram(layername+'/output', output)
    return output

def a1():
    return
def a2():#看数据截面
    return

def a3():
    xt=data.loc[data['血糖'] != 'unknown']
    for name in names:
        fig = plt.figure(figsize=(12,5))
        plt.subplot(121)
        sns.distplot(xt[name], norm_hist=True)
        plt.subplot(122)
        res = stats.probplot(xt[name].map(np.float32), plot=plt)
        plt.savefig('./pic/'+name+'.png')
#%%
f_num=len(train_xs[0])
with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32, [None, f_num], name='x_input')
    ys=tf.placeholder(tf.float32, [None,], name='y_input')
    
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
lamb_L2=tf.placeholder(tf.float32, name='lamb_L2')
learning_rate=tf.placeholder(tf.float32, name='learning_rate')

outsize_m=int((f_num+1)*2/3)
middlelayer =layer('layer1', xs, output_size=outsize_m, activation_function=tf.nn.relu, lamb=lamb_L2)

y_pre =layer('layer_y', middlelayer, output_size=1, activation_function=None, lamb=lamb_L2)
mse_loss = tf.reduce_mean(tf.square(y_pre-ys))/2
tf.add_to_collection('losses', mse_loss)
# loss
with tf.name_scope('loss'):
#    cross_entropy = -tf.reduce_mean(ys * tf.log(y_pre))    
#    loss = cross_entropy
    loss=tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('loss_L2', loss)
    tf.summary.scalar('mse_loss', mse_loss)

# optimizer
with tf.name_scope('train'):
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss)

#evaluate
with tf.name_scope('evaluate'):
    with tf.name_scope('correct_prediction'):
#        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(ys, 1))
        correct_prediction=tf.equal(ys, y_pre)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('value', accuracy)
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()
#%%
sess=tf.Session()
sess.run(init)
writer=tf.summary.FileWriter("./log", sess.graph)
# 运行后，会在相应的目录里生成一个文件，执行：tensorboard --logdir='./log'
#%%
##########开始训练##########
#模型训练好多好多周期,用minibatch，一次数万个确实太大了
batch=3000
l_rate=0.0001
lamb=0.01
kp=1.
len_train = len(train_xs)
batch = min(batch, len_train)
n = 0
for i in range(1001):
#feed的是numpy.ndarray格式
    if n+batch < len_train:
        batch_xs = train_xs[n:n+batch]
        batch_ys = train_ys[n:n+batch]
        n = n+batch
    else:
        batch_xs = np.vstack((train_xs[n:], train_xs[:n+batch-len_train]))
        batch_ys = np.hstack((train_ys[n:], train_ys[:n+batch-len_train])) #hstack
        n = n+batch-len_train
    fd={xs:batch_xs, ys:batch_ys, keep_prob:kp, learning_rate:l_rate, lamb_L2:lamb}
    sess.run(train_step, feed_dict=fd)#
    if i%100 == 0:
        results = sess.run(merged, feed_dict=fd)
        writer.add_summary(results, i)
        print ('第{}步:\tloss:{}\tmse_loss:{}'.format
               (i, 
               sess.run(loss, feed_dict=fd),
               sess.run(mse_loss, feed_dict=fd)))
# 男女模型结果合体
test_y['血糖'] = sess.run(y_pre, feed_dict={xs:test_x})
test_y=test_y.sort_values(by='id')
result=test_y['血糖'].map(lambda x:round(x,3))
#%%
########################## 存储答案 ##########################
result.to_csv('results/predict.csv', encoding='utf-8', index=False)