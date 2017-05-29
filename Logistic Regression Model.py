# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 16:09:05 2016

@author: guiyang

Lets try Titianic

"""


import tensorflow as tf

# construct a graph
W = tf.Variable(tf.zeros([8,1]),name = 'weight')
b = tf.Variable(0.,name="bias")


def inference(X):
    e_x = tf.add(tf.matmul(X,W),b)
    return tf.to_float(tf.sigmoid(e_x))
 
def	loss(X, Y):
	Y_predicted = inference(X)
     # for logistic, we use cross-entropy to find loss(penalize harder for un like)
	return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(Y, Y_predicted))
 

# in contest training data
def	inputs():
    import csv
    with open('train.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        figure = []
        label = []
        for row in reader:
            is_p1_class = float(row['Pclass']=='1')
            is_p2_class = float(row['Pclass']=='2')
            is_p3_class = float(row['Pclass']=='3')
            is_male = float(row['Sex']=='male')
            fare = float(row['Fare'])
            n_sib = float(row['SibSp'])
            n_parent = float(row['Parch'])
            try:
                age = float(row['Age'])
            except:
                age = float(30)
            figure.append([is_p1_class,is_p2_class,is_p3_class,is_male,fare,n_sib,n_parent,age])
            label.append([float(row['Survived'])])
    return tf.to_float(figure), tf.to_float(label)

 
def	train(total_loss):
	learning_rate	=	0.00000001
	return	tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
 
def	evaluate(sess,	X,	Y):
    #is_p1_class,is_p2_class,is_p3_class,is_male,fare,n_sib,n_parent,age
	print	sess.run(inference([[0.,0.,1.,1.,8.,0.,0.,35.]]))	#	~	303




with tf.Session() as sess:
    tf.initialize_all_variables().run()
    X,Y = inputs()
    total_loss = loss(X,Y)
    train_op = train(total_loss)
    coord=tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    
    #actual train
    training_steps = 1000000
    for step in range(training_steps):
        sess.run([train_op])
        #for debug
        if step % 10000 == 0:
            #print "predict y",sess.run(inference(X))
            print "loss: ",sess.run([total_loss])
            #print "weight:",sess.run(W)
    evaluate(sess, X, Y)
        
    coord.request_stop()
    coord.join(threads)
    sess.close()
