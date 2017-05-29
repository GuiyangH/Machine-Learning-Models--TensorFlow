# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 22:58:42 2016

@author: guiyang

This is taking a random sequence(from a TF tutorial), and use NN to train it.
"""
from __future__ import print_function
import tensorflow as tf


## How to add a neuron::
def add_layer(X, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    bias = tf.Variable(tf.zeros([1,out_size])+0.1)
    pre_cal = tf.matmul(X,Weights) + bias
    if activation_function is None:
        return pre_cal
    else:
        return activation_function(pre_cal)
        
import numpy as np

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise

x_data = tf.to_float(x_data)
y_data = tf.to_float(y_data)


    
def inference(X):
    #from 1 input to 10 neurons.
    input_layer = add_layer(X,1,10,activation_function=tf.nn.relu)
    #from 10 neurons to 1 output.
    prediction = add_layer(input_layer,10,1,activation_function=None)
    return prediction
    
    
def	loss(X, Y):
	Y_predicted = inference(X)
     # for logistic, we use cross-entropy to find loss(penalize harder for un like)
	return tf.reduce_mean(tf.reduce_sum(tf.square(Y-Y_predicted), reduction_indices=[1]))

def	train(total_loss):
	learning_rate	=	0.0001
	return	tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
 
def	evaluate(sess,	X,	Y):
    #is_p1_class,is_p2_class,is_p3_class,is_male,fare,n_sib,n_parent,age
	return inference(X)



with tf.Session() as sess:

    X,Y = x_data,y_data
    
    total_loss = loss(X,Y)
    train_op = train(total_loss)
    
    sess.run(tf.initialize_all_variables())
    
    training_steps = 1000    

    
    for step in range(training_steps):
        sess.run([train_op])
        #for debug
        if step % 100 == 0:
            #print "predict y",sess.run(inference(X))
            print("loss: ",sess.run([total_loss]))
            #print "weight:",sess.run(W)
    evaluate(sess, X, Y)
        
