import os 
import numpy as np
import tensorflow as tf
import pandas as pd  
import matplotlib.pyplot as plt  
#import sys
#You have freedom of using eager execution in tensorflow
#Instead of using With tf.Session() as sess you can use sess.run() whenever needed

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
Problem 1d: MLPs \& the Spiral Problem

@author - Alexander G. Ororbia II and Ankur Mali
'''
def Softmax(X,indices):
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
    return(loss_param, b_p, probs)
    
def computeCost(X,y,theta,reg, indices):
	# WRITEME: write your code here to complete the routine
    W = theta[0]
    b = theta[1]
    W2 = theta[2]
    b2 = theta[3]
    z = tf.add(tf.matmul(X, W), b)
    h = tf.maximum(tf.cast(0, dtype=tf.float64), z)
    f = tf.add(tf.matmul(h, W2), b2)
    softmax_out, _, _ = Softmax(f, indices)
    reg_1 = tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(W2)))
    reg_1 = 0.5*tf.multiply(reg, reg_1)
    #regularise_param = 0.5*reg*tf.reduce_sum(tf.square(W)) + 0.5*reg*tf.reduce_sum(tf.square(W2))
    cost = tf.add(softmax_out , reg_1)
    return cost		
			
def computeGrad(X,y,theta,reg,indices): # returns nabla
    W = theta[0]
    b = theta[1]
    W2 = theta[2]
    b2 = theta[3]
    z = tf.add(tf.matmul(X, W), b)
    #print(tf.shape(z))
    #print('***')
    h = tf.maximum(tf.cast(0, dtype=tf.float64), z)
    f = tf.add(tf.matmul(h, W2), b2)
    _, back_softmax_out, _ = Softmax(f, indices)
    
    dh = tf.matmul(back_softmax_out, tf.transpose(W2))
    #print(tf.shape(dh))
    #dh = tf.where(tf.greater(z,  tf.cast(0, tf.float64)), dh, tf.zeros(tf.shape(dh), dtype=tf.float64))
    #print(tf.shape(dh))
    #dh = tf.maximum(dh, tf.cast(0, dtype=tf.float64))
    
    #print(tf.shape(dz))
    dh = tf.where(tf.greater(z,  0), dh, tf.zeros(tf.shape(dh), dtype=tf.float64))
    #dz = tf.to_float(tf.greater_equal(dh, tf.cast(0, dtype=tf.float64)))
    #dz = tf.cast(dz, dtype=tf.float64)
	# WRITEME: write your code here to complete the routine
    dW = tf.add(tf.matmul(tf.transpose(X), dh), tf.multiply(reg, W))
    db = tf.reduce_sum(dh, axis=0)
    dW2 = tf.add(tf.matmul(tf.transpose(h), back_softmax_out), tf.multiply(reg, W2))
    db2 = tf.reduce_sum(back_softmax_out, axis=0)
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
	
np.random.seed(1491189) #Provide your unique Random seed
tf.set_random_seed(1491189)
# Load in the data from disk
path = os.getcwd() + '/data/spiral_train.dat'  
data = pd.read_csv(path, header=None) 

#Save images
#makdir (os.getcwd() + '/Result_images' + '/prob_1b')
save_dir = os.getcwd() + '/Result_images' + '/prob_1d'

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)
y = y.flatten()

# Indices:
indices = []
indices = [[i, y[i]] for i in range(y.shape[0])]
indices = np.array(indices)
indices = tf.constant(indices, dtype=tf.int64)



X_tf = tf.constant(X)
Y_tf = tf.constant(y)



# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1

# initialize parameters randomly
h = 100 # size of hidden layer
#initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=0, dtype=tf.float64)
'''initializer = tf.contrib.layers.xavier_initializer(seed=1491189, dtype=tf.float64)
W = tf.Variable(initializer([D, h]), dtype = tf.float64, name = "W")
b = tf.Variable(tf.zeros([h], dtype=tf.float64), dtype = tf.float64, name = "b")
W2 = tf.Variable(initializer([h, K]), dtype = tf.float64, name = "W2")
b2 = tf.Variable(tf.zeros([K], dtype=tf.float64), dtype = tf.float64, name = "b2")'''

W = 0.01 * np.random.randn(D,h)
W = tf.Variable(W, dtype=tf.float64, name = "W")
b = tf.Variable(tf.zeros([h], dtype=tf.float64), dtype = tf.float64, name = "b")
W2 = 0.01 * np.random.randn(h,K)
W2 = tf.Variable(W2, dtype=tf.float64, name = "W2")
b2 = tf.Variable(tf.zeros([K], dtype=tf.float64), dtype = tf.float64, name = "b2")
theta = (W,b,W2,b2)

# One hot encoding of y:
probs_hot = tf.contrib.layers.one_hot_encoding(indices[:,1], K)
probs_hot = tf.cast(probs_hot, dtype=tf.float64)

# some hyperparameters
n_e = 2000
check = 10 # every so many pass/epochs, print loss/error to terminal
step_size = tf.constant(1, dtype = tf.float64)
reg = tf.constant(0.0001, dtype = tf.float64) # regularization strength
error_param = []

X_p = tf.placeholder(tf.float64, shape = (X.shape[0], X.shape[1]))
Y_p = tf.placeholder(tf.float64, shape = (y.shape[0]))

# Session parameters:
ComputeCost = computeCost(X_p,Y_p,theta,reg, indices)
ComputeGrad = computeGrad(X_p,Y_p,theta,reg, indices)

# gradient descent loop

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(n_e):
		# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
        loss = sess.run([ComputeCost], feed_dict={X_p: X, Y_p: y})
        error_param.append(loss)
        if i % check == 0:
            print("iteration %d: loss %f" % (i, loss[0]))

	# perform a parameter update
	# WRITEME: write your update rule(s) here
        grand_new1 = sess.run([ComputeGrad], feed_dict={X_p: X , Y_p: y})
        #print('*******')
        #print((grand_new1[0])) 
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
       
        
 
# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
#sys.exit(0) 
  

    scores, probs = predict(X,theta)
    predicted_class = sess.run(tf.argmax(scores, axis=1))
    print('training accuracy: %.2f' % (sess.run(tf.reduce_mean(tf.to_float(predicted_class == y)))))

# plot the resulting classifier
    H = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, H), np.arange(y_min, y_max, H))
    xx_1 = np.expand_dims(xx.ravel(), axis = 1)
    yy_1 = np.expand_dims(yy.ravel(), axis = 1)
    Z1, _ = sess.run(predict(tf.concat([xx_1, yy_1], 1), theta))
    Z1 = np.argmax(Z1, axis=1)
    Z1 = Z1.reshape(xx.shape)
    fig = plt.figure(1)
    plt.contourf(xx, yy, Z1, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
#fig.savefig('spiral_linear.png')
    reg1 = sess.run(reg)
    step_size1 = sess.run(step_size)
    plt.title("Classifier")
    plt.savefig(save_dir + '_step_size' + str(step_size1) + '_reg ' + str(reg1)+'.png')

    

fig = plt.figure()
plt.plot(error_param)
plt.title('Loss vs Epoch')
plt.xlabel('Number of epochs')
plt.ylabel('Cost')
plt.savefig(save_dir + '/Loss_vs_Epoch' + str(step_size) + '_reg ' + str(reg)+'.png')
plt.show()
