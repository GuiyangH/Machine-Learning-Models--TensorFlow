"""
Created on Sat Nov 26 16:09:05 2016

@author: guiyang

Softmax the Data(Iris data from UCI)

"""


import tensorflow as tf

# construct a graph
W = tf.Variable(tf.to_float([[1,0,0],[0,1,0],[0,0,1],[0,0,1]]),name = 'weight')
b = tf.Variable(tf.zeros([3]),name="bias")


def inference(X):
    before = tf.matmul(X,W)+b
    return tf.nn.softmax(before)
 
def	loss(X, Y):
	Y_predicted = inference(X)
     # for logistic, we use cross-entropy to find loss(penalize harder for un like)
	return -tf.reduce_mean(Y*tf.log(Y_predicted))
 

# in contest training data
def	inputs():
    with open('bezdekIris.data','r') as dafile:
        figure = []
        label = []
        for line in dafile:
            try:
                lst =line.split(',')
                is_setosa = float('setosa' in lst[4])
                is_versicolor = float('versicolor' in lst[4])
                is_virginica = float('virginica' in lst[4])
                label.append([is_setosa,is_versicolor,is_virginica])
                figure.append([float(lst[0]),float(lst[1]),float(lst[2]),float(lst[3])])
            except:
                break
    return tf.to_float(figure), tf.to_float(label)

 
def	train(total_loss):
	learning_rate	=	0.001
	return	tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
 
def	evaluate(sess,	X,	Y):
    #is_p1_class,is_p2_class,is_p3_class,is_male,fare,n_sib,n_parent,age
	return inference(X)#	~	303




with tf.Session() as sess:
    tf.initialize_all_variables().run()
    X,Y = inputs()
    total_loss = loss(X,Y)
    train_op = train(total_loss)
    #actual train
    training_steps = 100000
    for step in range(training_steps):
        sess.run([train_op])
        #for debug
        if step % 1000 == 0:
            #print "predict y",sess.run(inference(X))
            print "loss: ",sess.run([total_loss])
            #print "weight:",sess.run(W)
    evaluate(sess, X, Y)
        
    sess.close()
