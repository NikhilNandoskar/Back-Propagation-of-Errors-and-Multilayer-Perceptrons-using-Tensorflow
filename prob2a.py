from __future__ import print_function
import os 
import numpy as np
import tensorflow as tf
import pandas as pd  
import matplotlib.pyplot as plt  
import sys
#You have freedom of using eager execution in tensorflow
#Instead of using With tf.Session() as sess you can use sess.run() whenever needed
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
Problem 2a: 1-Layer MLP for IRIS

@author - Alexander G. Ororbia II and ankur mali
'''
'''def Softmax(X,indices, probs_hot):
    N = tf.shape(X, out_type=tf.float64)
    exp = tf.exp(X)
    probs = tf.divide(exp, tf.reduce_sum(exp, axis = 1, keepdims = True))
    #print(probs.shape)
    # Correct probs:
    probs_1 = tf.gather_nd(probs, indices)
    loss_param = -(tf.divide(tf.reduce_sum(tf.log(probs_1)), N[0]))
    #print(correct_probs.shape)
    
    # Backward Pass:
    b_p = probs - probs_hot
    b_p = tf.divide(b_p, N[0])
    return(loss_param, b_p, probs)'''
    
def computeCost(X,y,theta,reg,indices):
	# WRITEME: write your code here to complete the routine
    indices = tf.concat([indices, tf.cast(tf.reshape(y, (-1, 1)), dtype=tf.int64)], 1)
    W = theta[0]
    b = theta[1]
    W2 = theta[2]
    b2 = theta[3]
    z = tf.add(tf.matmul(X, W), b)
    h = tf.maximum(tf.cast(0, dtype=tf.float64), z)
    f = tf.add(tf.matmul(h, W2), b2)
    N = tf.shape(f, out_type=tf.float64)
    exp = tf.exp(f)
    probs = tf.divide(exp, tf.reduce_sum(exp, axis = 1, keepdims = True))
    # Correct probs:
    probs_1 = tf.gather_nd(probs, indices)
    loss_param = -(tf.divide(tf.reduce_sum(tf.log(probs_1)), N[0]))
    #softmax_out, _, _ = Softmax(f, indices, probs_hot)
    reg_1 = tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(W2)))
    reg_1 = 0.5*tf.multiply(reg, reg_1)
    #regularise_param = 0.5*reg*tf.reduce_sum(tf.square(W)) + 0.5*reg*tf.reduce_sum(tf.square(W2))
    cost = tf.add(loss_param , reg_1)
    return cost		
				
			
def computeGrad(X,y,theta,reg,indices): # returns nabla
    indices = tf.concat([indices, tf.cast(tf.reshape(y, (-1, 1)), dtype=tf.int64)], 1)
    W = theta[0]
    b = theta[1]
    W2 = theta[2]
    b2 = theta[3]
    z = tf.add(tf.matmul(X, W), b)
    h = tf.maximum(tf.cast(0, dtype=tf.float64), z)
    f = tf.add(tf.matmul(h, W2), b2)
    N = tf.shape(f, out_type=tf.float64)
    exp = tf.exp(f)
    probs = tf.divide(exp, tf.reduce_sum(exp, axis = 1, keepdims = True))
   # _, back_softmax_out, _ = Softmax(f, indices, probs_hot)
    probs_hot = tf.contrib.layers.one_hot_encoding(indices[:,1], K)
    probs_hot = tf.cast(probs_hot, dtype=tf.float64)
    b_p = probs - probs_hot
    b_p = tf.divide(b_p, N[0])
    dh = tf.matmul(b_p, tf.transpose(W2))
    dh = tf.where(tf.greater(z,  0), dh, tf.zeros(tf.shape(dh), dtype=tf.float64))
	# WRITEME: write your code here to complete the routine
    dW = tf.add(tf.matmul(tf.transpose(X), dh), tf.multiply(reg, W))
    db = tf.reduce_sum(dh, axis=0)
    dW2 = tf.add(tf.matmul(tf.transpose(h), b_p), tf.multiply(reg, W2))
    db2 = tf.reduce_sum(b_p, axis=0)
    nabla1 = tf.tuple([dW, db, dW2, db2])
    return (nabla1)

def predict(X,theta):
	# WRITEME: write your code here to complete the routine
    W = theta[0]
    b = theta[1]
    W2 = theta[2]
    b2 = theta[3]
    z = tf.add(tf.matmul(X, W), b)
    h = tf.maximum(tf.cast(0, dtype=tf.float64), z)
    scores = tf.add(tf.matmul(h, W2), b2)
   # _, _, predicts = Softmax(scores, indices)
    exp = tf.exp(scores)
    probs = tf.divide(exp, tf.reduce_sum(exp, axis = 1, keepdims = True))
    return (scores,probs)
	
def create_mini_batch(X, y, start, end):
	# WRITEME: write your code here to complete the routine
	mb_x = X[start : end, :]
	mb_y = y[start: end]
	return (mb_x, mb_y)
		
def shuffle(X,y):
	ii = np.arange(X.shape[0])
	ii = np.random.shuffle(ii)
	X_rand = X[ii]
	y_rand = y[ii]
	X_rand = X_rand.reshape(X_rand.shape[1:])
	y_rand = y_rand.reshape(y_rand.shape[1:])
	return (X_rand,y_rand)
	
np.random.seed(1491189)
tf.set_random_seed(1491189)
# Load in the data from disk
path = os.getcwd() + '/data/iris_train.dat'  
data = pd.read_csv(path, header=None) 

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)
y = y.flatten()
X_tf = tf.constant(X, dtype=tf.float64)
Y_tf = tf.constant(y, dtype=tf.float64)
# load in validation-set
path = os.getcwd() + '/data/iris_test.dat'
data = pd.read_csv(path, header=None) 

cols = data.shape[1]  
X_v = data.iloc[:,0:cols-1]  
y_v = data.iloc[:,cols-1:cols] 

X_v = np.array(X_v.values)  
y_v = np.array(y_v.values)
y_v = y_v.flatten()



X_V_tf = tf.constant(X_v)
Y_V_tf = tf.constant(y_v)


# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1


# initialize parameters randomly
h = 100 # size of hidden layer
'''initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
W = tf.Variable(initializer([D, h]))
b = tf.Variable(tf.random_normal([h]))
W2 = tf.Variable(initializer([h, K]))
b2 = tf.Variable(tf.random_normal([K]))
theta = (W,b,W2,b2)'''

W = 0.01 * np.random.randn(D,h)
W = tf.Variable(W, dtype=tf.float64, name = "W")
b = tf.Variable(tf.zeros([h], dtype=tf.float64), dtype = tf.float64, name = "b")
W2 = 0.01 * np.random.randn(h,K)
W2 = tf.Variable(W2, dtype=tf.float64, name = "W2")
b2 = tf.Variable(tf.zeros([K], dtype=tf.float64), dtype = tf.float64, name = "b2")
theta = (W,b,W2,b2)

# some hyperparameters
n_e = 200
n_b = 10
check = 10
step_size = tf.constant(0.01, dtype = tf.float64)
reg = tf.constant(0.01, dtype = tf.float64) # regularization strength

# Indices for training, testng data and compute grad:
indices = {}
indices[0] = tf.constant(np.array([[i] for i in range(y.shape[0])]), dtype=tf.int64)
indices[1] = tf.constant(np.array([[i] for i in range(y_v.shape[0])]), dtype=tf.int64)
indices[2] = tf.constant(np.array([[i] for i in range(n_b)]), dtype=tf.int64) 

#Placeholders for Training 
X_p_t = tf.placeholder(tf.float64, shape = (X.shape[0], X.shape[1]))
Y_p_t = tf.placeholder(tf.float64, shape = (y.shape[0]))

#Placeholders for Testing
X_p_v = tf.placeholder(tf.float64, shape = (X_v.shape[0], X_v.shape[1]))
Y_p_v = tf.placeholder(tf.float64, shape = (y_v.shape[0]))

# Mini Batch size placeholders
X_p_b = tf.placeholder(tf.float64, shape = (n_b, X.shape[1]))
Y_p_b = tf.placeholder(tf.float64, shape = (n_b))


# Session parameters:
ComputeCost_Train_Loss = computeCost(X_p_t,Y_p_t,theta,reg, indices[0])
ComputeCost_Test_Loss = computeCost(X_p_v,Y_p_v,theta,reg, indices[1])

ComputeGrad_Grad = computeGrad(X_p_b,Y_p_b,theta,reg, indices[2])


train_cost = []
valid_cost = []
# gradient descent loop
num_examples = X.shape[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(n_e):
        #X, y = tf.random_shuffle([X_tf,Y_tf, indices, probs_hot]) # re-shuffle the data at epoch start to avoid correlations across mini-batches
        X, y = shuffle(X, y)
		# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
		#          you can use the "check" variable to decide when to calculate losses and record/print to screen (as in previous sub-problems)
        # Training Loss
        train_loss = sess.run([ComputeCost_Train_Loss], feed_dict={X_p_t: X, Y_p_t: y})
        train_cost.append(train_loss)
        # Testing Loss
        test_loss = sess.run([ComputeCost_Test_Loss], feed_dict={X_p_v: X_v, Y_p_v: y_v})
        valid_cost.append(test_loss)
        if i % check == 0:
            print("iteration: training loss: , validation loss:" , (i, train_loss, test_loss))
            #print(s)
		# WRITEME: write the inner training loop here (1 full pass, but via mini-batches instead of using the full batch to estimate the gradient)
        s = 0
        while (s < num_examples):
			# build mini-batch of samples
            X_mb, y_mb = create_mini_batch(X,y, s, s+n_b)
		
			# WRITEME: gradient calculations and update rules go here
            #ComputeGrad_Grad = computeGrad(X_p_b,Y_p_b,theta,reg, indices_g, probs_hot_g)
            grand_new1 = sess.run([ComputeGrad_Grad], feed_dict={X_p_b: X_mb , Y_p_b: y_mb })
            grand_new2 = [item for sublist in grand_new1 for item in sublist]
            W_t1 = tf.subtract(W, tf.multiply(step_size, grand_new2[0]))
        #W_t_1 = tf.convert_to_tensor(W_t1)
            b_t1 = tf.subtract(b, tf.multiply(step_size, grand_new2[1]))
        #b_t_1 = tf.convert_to_tensor(b_t1)
            W_t2 = tf.subtract(W2, tf.multiply(step_size, grand_new2[2]))
            #W_t_2 = tf.convert_to_tensor(W_t2)
            b_t2 = tf.subtract(b2, tf.multiply(step_size, grand_new2[3]))
        #b_t_2 = tf.convert_to_tensor(b_t2)
            sess.run(tf.assign(W, W_t1))
            sess.run(tf.assign(b, b_t1))
            sess.run(tf.assign(W2, W_t2))
            sess.run(tf.assign(b2, b_t2))
            
            s += n_b
            
        

    print(' > Training loop completed!')
# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
#sys.exit(0) 

    scores, probs = predict(X,theta)
    predicted_class = sess.run(tf.argmax(scores, axis=1))
    print('training accuracy: %.2f' % (sess.run(tf.reduce_mean(tf.to_float(predicted_class == y)))))

    scores, probs = predict(X_v,theta)
    predicted_class = sess.run(tf.argmax(scores, axis=1))
    print('validation accuracy: %.2f' % (sess.run(tf.reduce_mean(tf.to_float(predicted_class == y_v)))))

# NOTE: write your plot generation code here (for example, using the "train_cost" and "valid_cost" list variables)
    plt.plot(range(n_e), train_cost, range(n_e), valid_cost)
    plt.legend(["Training Loss", "Testing Loss"])
    plt.show()
