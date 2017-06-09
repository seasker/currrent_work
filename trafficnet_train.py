#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:00:22 2017

@author: seasker
"""
import time
import numpy as np
import tensorflow as tf
import os


BATCH_SIZE=300
LEARNING_RATE_BASE=0.01
LEARNING_RATE_DECAY=0.5
MAX_ITER=3000
MODEL_SAVE_DIR='.'
MODEL_SAVE_NAME='model.ckpt'
def variable(name, shape, initializer):
    return tf.get_variable(name=name,   
                               shape=shape,   
                               initializer=initializer)
    
def weight_variable_with_decay(name,shape,stddev,wd):
    weight=tf.get_variable(name=name,
                           shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(weight), wd, name='weight_loss')  
        tf.add_to_collection(name='losses', value=weight_decay) 
    return weight

def weight_variable_with_regularizer(name,shape,stddev,regularizer):
    weight=tf.get_variable(name=name,
                           shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=stddev))
    if regularizer is not None:
        tf.add_to_collection(name='losses', value=regularizer(weight)) 
    return weight

def bias_variable(name, shape, value):
    return tf.get_variable(name=name,   
                               shape=shape,   
                               initializer=tf.constant_initializer(value))

def forward(batch_data):
    with tf.variable_scope('conv1'):
        weights=weight_variable_with_decay([5,5,3,8],stddev=0.1,wd=0.0)
        bias=bias_variable('bias',[8],0.0)
        conv1=tf.nn.relu(tf.nn.conv2d(batch_data,weights,[1,1,1,1],padding='SAME')+bias)
        pool1=tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1])
    with tf.variable_scope('conv2'):
        weights=weight_variable_with_decay([3,3,8,16],0.1,0.0)
        bias=bias_variable('bias',[16],0.0)
        conv2=tf.nn.relu(tf.nn.conv2d(pool1,weights,[1,1,1,1],padding='SAME')+bias)
        pool2=tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1])
    with tf.variable_scope('conv3'):
        weights=weight_variable_with_decay([3,3,16,32],0.1,0.0)
        bias=bias_variable('bias',[32],0.0)
        conv2=tf.nn.relu(tf.nn.conv2d(pool2,weights,[1,1,1,1],padding='SAME')+bias)
        pool3=tf.nn.max_pool(conv2,[1,2,2,1],[1,2,2,1])
    with tf.variable_scope('fc1'):
        pool3_vec=tf.reshape(pool3,[BATCH_SIZE,-1])
        pool3_shape=pool3_vec.g.get_shape().as_list()
        weights=weight_variable_with_decay([pool3_shape[1],105],0.1)
        fc1=tf.nn.sigmoid(tf.matmul(pool3_vec,weights))
    return fc1
    

        
    
def train():
    global_step=tf.Variable(0,trainable=False)
    #
    #load data
    #

    prediction=forward(datum)
    mse=tf.reduce_mean(tf.square(prediction-target))   
    loss=mse+tf.add_n(tf.get_collection('losses'))
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,100,LEARNING_RATE_DECAY)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for step in range(MAX_ITER):
            #start_time = time.time()
            _,loss_value,step = sess.run([train_step,loss])
            
            #duration = time.time() - start_time

            if step % 500 == 0:
                print('After %d training step(s),loss on training '
                      'batch is %g.' % (step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_DIR,MODEL_SAVE_NAME),global_step=global_step)

def main():
    train()
if __name__=='__main__':
    tf.app.run()
    
