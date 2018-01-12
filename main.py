#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:54:14 2018

@author: mayuyuan
"""
import pandas as pd
import numpy as np
import tensorflow as tf
#%%
train=pd.read_csv('d_train_20180102.csv', encoding='gbk', parse_dates=['体检日期'])
test=pd.read_csv('d_test_A_20180102.csv', encoding='gbk', parse_dates=['体检日期'])
test['血糖']='unknown'

data=pd.concat([train, test], axis=0)
data=pd.concat([pd.get_dummies(data['性别']), data], axis=1)
data.loc[data['??']==1, ['女', '男']]=[0.5,0.5]
del data['性别']
del data['体检日期']
del data['??']
data=data.sample(frac = 1) #乱序
data.fillna(data.quantile(0.5), inplace=True)
f_data=data.columns.drop(['血糖','id']) #41 columns
train_xs=data.loc[data['血糖'] != 'unknown', f_data].values
train_ys=data.loc[data['血糖'] != 'unknown', '血糖']
test_xy=data.loc[data['血糖'] == 'unknown']
test_xy.sort_values(by='id', inplace=True)
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

def multilayer(n_layer, x, output_size, keep_prob, activation_function):
    for n in range(n_layer):
        layername = 'layer' + str(n+1)
        x = layer(layername, x, output_size=output_size[n], 
                  keep_prob=keep_prob[n], 
                  activation_function=activation_function[n])
    return x
#%%
with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32, [None, len(f_data)], name='x_input')
    ys=tf.placeholder(tf.float32, [None,], name='y_input')
#with tf.name_scope('middlelayer'):
#    n_layer = 40
#    output_size = [len(f_data)*6]*6+[len(f_data)*3]*(n_layer-6)
#    keep_prob = tf.placeholder(tf.float32, [n_layer,], name='keep_prob')
#    activation_function = [tf.nn.relu]*n_layer
#    middlelayer = multilayer(n_layer, xs, output_size=output_size, 
#                    keep_prob=keep_prob, activation_function=activation_function)
y_pre =layer('layer_y', xs, output_size=1, activation_function=None, lamb=100.0)
mse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_pre-ys), reduction_indices=[1]))
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
    train_step=tf.train.AdamOptimizer(0.001).minimize(loss)

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
sess = tf.Session()
sess.run(init)
writer=tf.summary.FileWriter("./log", sess.graph)
# 运行后，会在相应的目录里生成一个文件，执行：tensorboard --logdir='./log'
#%%
##########开始训练##########
#模型训练好多好多周期,用minibatch，一次数万个确实太大了
batch = 2000
n = 0
len_train = len(train_xs)
for i in range(3001):
#feed的是numpy.ndarray格式
    if n+batch < len_train:
        batch_xs = train_xs[n:n+batch]
        batch_ys = train_ys[n:n+batch]
        n = n+batch
    else:
        batch_xs = np.vstack((train_xs[n:], train_xs[:n+batch-len_train]))
        batch_ys = np.hstack((train_ys[n:], train_ys[:n+batch-len_train])) #hstack
        n = n+batch-len_train
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})#, keep_prob:[0.6]*n_layer
    if i%200 == 0:
        results = sess.run(merged, feed_dict={xs:batch_xs, ys: batch_ys})#, keep_prob:[1]*n_layer
        writer.add_summary(results, i)
        print ('第{}步，loss为{}'.format(i, sess.run(loss, feed_dict={xs:batch_xs, ys: batch_ys})))
        print ('第{}步，mse_loss为{}'.format(i, sess.run(mse_loss, feed_dict={xs:batch_xs, ys: batch_ys})))
#%%
# 测试第30天
test_xy['血糖'] = sess.run(y_pre, feed_dict={xs:test_xy[f_data].values})#, keep_prob:[1]*n_layer
result=test_xy['血糖']
#%%
########################## 存储答案 ##########################
result.to_csv('results/predict.csv', encoding='utf-8', index=False)
