#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
#%%
#读取数据
train=pd.read_csv('d_train_20180102.csv', encoding='gbk', parse_dates=['体检日期'])
train=train[train['性别'].isin(['男', '女'])]#性别不明者占比特别少，所以直接删掉。如果占比多，就另寻方法处理。
test=pd.read_csv('d_test_A_20180102.csv', encoding='gbk', parse_dates=['体检日期'])
test['血糖']='unknown'
data=pd.concat([train, test], axis=0)
data=pd.concat([pd.get_dummies(data['性别']), data], axis=1)
del data['性别']
del data['体检日期']
data=data.sample(frac = 1) #乱序
names=data.columns.drop(['id', '男', '女', '血糖'])
names_plus_id=data.columns.drop(['男', '女', '血糖'])
#分男女填充缺失值
data.loc[data['男']==1] = data.loc[data['男']==1].fillna(data.loc[data['男']==1].mean())
data.loc[data['女']==1] = data.loc[data['女']==1].fillna(data.loc[data['女']==1].mean())
#划分训练集和测试集
train_male_xs=data.loc[(data['男']==1)&(data['血糖'] != 'unknown'), names].values
train_male_ys=data.loc[(data['男']==1)&(data['血糖'] != 'unknown'), '血糖']
test_male_xy=data.loc[(data['男']==1)&(data['血糖'] == 'unknown')][names_plus_id]
scaler_male=preprocessing.StandardScaler().fit(train_male_xs)
train_male_xs=scaler_male.transform(train_male_xs)
test_male_xy[names]=scaler_male.transform(test_male_xy[names])
train_female_xs=data.loc[(data['女']==1)&(data['血糖'] != 'unknown'), names].values
train_female_ys=data.loc[(data['女']==1)&(data['血糖'] != 'unknown'), '血糖']
test_female_xy=data.loc[(data['女']==1)&(data['血糖'] == 'unknown')][names_plus_id]
scaler_female=preprocessing.StandardScaler().fit(train_female_xs)
train_female_xs=scaler_female.transform(train_female_xs)
test_female_xy[names]=scaler_female.transform(test_female_xy[names])
#%%
def layer(layername, x, output_size=int, keep_prob=1.0, lamb=0., activation_function=None):
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

def multilayer(n_layer, x, output_size, keep_prob, lamb, activation_function):
    for n in range(n_layer):
        layername = 'layer' + str(n+1)
        x = layer(layername, x, output_size=output_size[n], 
                  keep_prob=keep_prob[n], lamb=lamb[n], 
                  activation_function=activation_function[n])
    return x

def a1():#看各特征和血糖的关系，蓝点男人，红点女人
    c=names
    fig=plt.figure(figsize=(50,50))
    for i in range(len(c)):
        plt.subplot(7,7,i+1)
        f=train.loc[train['性别']=='女', [c[i], '血糖']].dropna(axis=0)
        plt.scatter(f[c[i]], f['血糖'], s=8, c='r', alpha=0.3)
        m=train.loc[train['性别']=='男', [c[i], '血糖']].dropna(axis=0)
        plt.scatter(m[c[i]], m['血糖'], s=8, c='b', alpha=0.3)
        plt.xlable="{}.{}".format(i+1, c[i])
        plt.ylable='血糖'
    plt.savefig('a1.png')
def a2():#看数据截面
    data_d=pd.DataFrame([],columns=data.columns)
    data['血糖']=data['血糖'].replace('unknown', np.nan)
    for d,ind in [[data,'男+女'], [data.loc[data['女']==1],'女'], [data.loc[data['男']==1],'女']]:
        data_dscr=pd.DataFrame([d.min(),d.mean(),d.max()],columns=d.columns,index=['min'+ind,'mean'+ind,'max'+ind])
        data_d=pd.concat([data_d, data_dscr])
    data['血糖']=data['血糖'].replace(np.nan,'unknown')
    return data_d
#%%
with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32, [None, len(names)], name='x_input')
    ys=tf.placeholder(tf.float32, [None,], name='y_input')
#with tf.name_scope('middlelayer'):
#    n_layer = 1
#    output_size = [int((len(names)+1)*2/3)]*n_layer
#    keep_prob = tf.placeholder(tf.float32, [n_layer,], name='keep_prob')
#    lamb=[0.]*n_layer
#    activation_function = [tf.nn.relu]*n_layer
#    middlelayer = multilayer(n_layer, xs, 
#                             output_size=output_size, 
#                             keep_prob=keep_prob, 
#                             lamb=lamb,
#                             activation_function=activation_function)
    
n_layer = 1
keep_prob = tf.placeholder(tf.float32, [n_layer,], name='keep_prob')

y_pre =layer('layer_y', xs, output_size=1, activation_function=None, lamb=0.1)
mse_loss = tf.reduce_mean(tf.square(y_pre-ys))
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
    train_step=tf.train.AdamOptimizer(0.0005).minimize(loss)

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
sess_male=tf.Session()
sess_male.run(init)
writer_male=tf.summary.FileWriter("./log", sess_male.graph)
#graph仍无法显示两个模型，算了，这个目前不重要
sess_female=tf.Session()
sess_female.run(init)
writer_female=tf.summary.FileWriter("./log", sess_female.graph)
# 运行后，会在相应的目录里生成一个文件，执行：tensorboard --logdir='./log'
#%%
##########开始训练##########
#模型训练好多好多周期,用minibatch，一次数万个确实太大了
batch=2000
len_train_male = len(train_male_xs)
len_train_female = len(train_female_xs)
batch = min(batch, len_train_male, len_train_female)
n = 0
for i in range(5001):
#feed的是numpy.ndarray格式
    if n+batch < len_train_male:
        batch_male_xs = train_male_xs[n:n+batch]
        batch_male_ys = train_male_ys[n:n+batch]
        n = n+batch
    else:
        batch_male_xs = np.vstack((train_male_xs[n:], train_male_xs[:n+batch-len_train_male]))
        batch_male_ys = np.hstack((train_male_ys[n:], train_male_ys[:n+batch-len_train_male])) #hstack
        n = n+batch-len_train_male
    sess_male.run(train_step, feed_dict={xs:batch_male_xs, ys:batch_male_ys, keep_prob:[1.]*n_layer})#
    
    if n+batch < len_train_female:
        batch_female_xs = train_female_xs[n:n+batch]
        batch_female_ys = train_female_ys[n:n+batch]
        n = n+batch
    else:
        batch_female_xs = np.vstack((train_female_xs[n:], train_female_xs[:n+batch-len_train_female]))
        batch_female_ys = np.hstack((train_female_ys[n:], train_female_ys[:n+batch-len_train_female])) #hstack
        n = n+batch-len_train_female
    sess_female.run(train_step, feed_dict={xs:batch_female_xs, ys:batch_female_ys, keep_prob:[1.]*n_layer})
    
    if i%200 == 0:
        results_male = sess_male.run(merged, feed_dict={xs:batch_male_xs, ys: batch_male_ys, keep_prob:[1.]*n_layer})
        writer_male.add_summary(results_male, i)
        print ('male第{}步:\tloss:{}\tmse_loss:{}'.format(i, 
               sess_male.run(loss, feed_dict={xs:batch_male_xs, ys: batch_male_ys, keep_prob:[1.]*n_layer}),
               sess_male.run(mse_loss, feed_dict={xs:batch_male_xs, ys: batch_male_ys, keep_prob:[1.]*n_layer})))
        results_female = sess_female.run(merged, feed_dict={xs:batch_female_xs, ys: batch_female_ys, keep_prob:[1.]*n_layer})
        writer_female.add_summary(results_female, i)
        print ('female第{}步:\tloss:{}\tmse_loss:{}'.format(i, 
               sess_female.run(loss, feed_dict={xs:batch_male_xs, ys: batch_male_ys, keep_prob:[1.]*n_layer}),
               sess_female.run(mse_loss, feed_dict={xs:batch_male_xs, ys: batch_male_ys, keep_prob:[1.]*n_layer})))
#%%
# 男女模型结果合体
test_male_xy['血糖'] = sess_male.run(y_pre, feed_dict={xs:test_male_xy[names].values})
test_female_xy['血糖'] = sess_female.run(y_pre, feed_dict={xs:test_female_xy[names].values})
test_xy=pd.concat([test_male_xy, test_female_xy], axis=0).sort_values(by='id')
result=test_xy['血糖']
#%%
########################## 存储答案 ##########################
result.to_csv('results/predict.csv', encoding='utf-8', index=False)
