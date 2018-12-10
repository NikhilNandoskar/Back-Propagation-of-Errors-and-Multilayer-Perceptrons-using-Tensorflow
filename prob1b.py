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
Problem 1b: Softmax Regression \& the Spiral Problem

@author - Alexander G. Ororbia II and Ankur Mali
'''		

def Softmax(x,indices):
    N = tf.cast(tf.shape(x), tf.float64)
    exp = tf.exp(x)
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
    
   
def computeGrad(X,y,theta,reg, indices): # returns nabla
	# WRITEME: write your code here to complete the routine
    W = theta[0]
    b = theta[1]
    #N = tf.cast(tf.shape(X), tf.float64)
    f = tf.add(b, tf.matmul(X,W)) 
    _, back_softmax_out, _ = Softmax(f, indices)
    dW = tf.matmul(tf.transpose(X), back_softmax_out) + tf.multiply(reg, W)
    db = tf.reduce_sum(back_softmax_out)
    nabla1 = tf.tuple([dW, db])
    return (nabla1)

def computeCost(X,y,theta,reg,indices):
	# WRITEME: write your code here to complete the routine                                                                             
    #N = tf.cast(tf.shape(X), tf.float64)
    W = theta[0]
    b = theta[1]
    #f = b + tf.matmul(X,W)
    f = tf.add(b, tf.matmul(X,W))                                                            
    
    # Softmax Calculation:
    #exp = tf.exp(f)
    #softmax_out = tf.divide(exp, tf.reduce_sum(exp, axis = 1, keepdims = True))
    # Getting Softmax_out:
    softmax_out, _, _ = Softmax(f, indices)
    
    
      
    # Loss Calculation:
    #loss_param = -(tf.divide(tf.reduce_sum(softmax_out), N[0]))
    #loss_param = -(tf.reduce_mean(tf.log(softmax_out)))
    regularise_param = tf.divide(tf.multiply(reg, tf.reduce_sum(tf.square(W))), tf.cast(2, dtype = tf.float64))
    cost = softmax_out + regularise_param                                                                                                                    
    return cost

def predict(X,theta, indices):
	# WRITEME: write your code here to complete the routine
    W = theta[0]
    b = theta[1]
	# evaluate class scores
    scores = b + tf.matmul(X,W)
    _, _, predicts = Softmax(scores, indices)
	# compute the class probabilities
    #f_scores = tf.exp(scores)
    #probs = tf.divide(f_scores, tf.reduce_sum(f_scores, axis = 1, keepdims = True))
    return (scores,predicts)



np.random.seed(1491189) #Provide your unique Random seed
tf.set_random_seed(1491189)
# Load in the data from disk
path = os.getcwd() + '/data/spiral_train.dat'  
data = pd.read_csv(path, header=None) 

#Save images
#makdir (os.getcwd() + '/Result_images' + '/prob_1b')
save_dir = os.getcwd() + '/Result_images' + '/prob_1b'

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

# One hot encoding of y:
probs_hot = tf.contrib.layers.one_hot_encoding(indices[:,1], 3)
probs_hot = tf.cast(probs_hot, dtype=tf.float64)

X_tf = tf.constant(X)
Y_tf = tf.constant(y)

X_p = tf.placeholder(tf.float64, shape = (X.shape[0], X.shape[1]))
Y_p = tf.placeholder(tf.float64, shape = (y.shape[0]))

# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1

#Train a Linear Classifier
#You will be using X_tf and Y_tf within your session , numpy variables are provided to do sanity check
# initialize parameters randomly
#initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=0, dtype=tf.float64)
initializer = tf.contrib.layers.xavier_initializer(seed=1491189, dtype=tf.float64)
#W = tf.cast(tf.Variable(initializer([D, K])), dtype = tf.float64)
W = tf.Variable(initializer([D, K]), dtype = tf.float64, name = "W")
b = tf.Variable(tf.zeros([K], dtype=tf.float64), dtype = tf.float64, name = "b")
theta = (W,b)

# some hyperparameters
n_e = 500
check = 10 # every so many pass/epochs, print loss/error to terminal
#check = 10 # every so many pass/epochs, print loss/error to terminal
step_size = tf.cast(0.1, dtype = tf.float64)
reg = tf.cast(0.1, dtype = tf.float64) # regularization strength
xrange = range
error_param = []

# Session parameters:
ComputeCost = computeCost(X_p,Y_p,theta,reg, indices)
ComputeGrad = computeGrad(X_p,Y_p,theta,reg, indices)

# gradient descent loop
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in xrange(n_e):
		# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
        loss = sess.run([ComputeCost], feed_dict={X_p: X, Y_p: y})
        error_param.append(loss)
        if i % check == 0:
            print("iteration %d: loss %f" % (i, loss[0]))

        grand_new1 = sess.run([ComputeGrad], feed_dict={X_p: X , Y_p: y})
        #print('*******')
        #print((grand_new1[0])) 
        grand_new2 = [item for sublist in grand_new1 for item in sublist]
        #grand_new2 = tf.cast(grand_new2, dtype=tf.float64)
        #print(grand_new2[1])
        W_t1 = tf.subtract(W, tf.multiply(step_size, grand_new2[0]))
        W_t2 = tf.convert_to_tensor(W_t1)
        b_t1 = tf.subtract(b, tf.multiply(step_size, grand_new2[1]))
        b_t2 = tf.convert_to_tensor(b_t1)
        sess.run(tf.assign(W, W_t2))
        sess.run(tf.assign(b, b_t2))
        #print(W)
        #print(b)
        
 
# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
#sys.exit(0) 
  
# evaluate training set accuracy

    scores, probs = predict(X,theta, indices)
#scores = np.dot(X, W) + b
    predicted_class = sess.run(tf.argmax(scores, axis=1))
    #q = sess.run(tf.reduce_mean(tf.to_float(predicted_class == y)))
    print('training accuracy: %.2f' % sess.run(tf.reduce_mean(tf.to_float(predicted_class == y))))
    #print('training accuracy: %.2f' % (q))

# plot the resulting classifier

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx_1 = np.expand_dims(xx.ravel(), axis = 1)
    yy_1 = np.expand_dims(yy.ravel(), axis = 1)
    Z1 = sess.run(tf.add(tf.matmul(tf.concat([xx_1, yy_1], 1), W), b))
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

    

fig = plt.figure(2)
plt.plot(error_param)
plt.title('Loss vs Epoch')
plt.xlabel('Number of epochs')
plt.ylabel('Cost')
plt.savefig(save_dir + '/Loss_vs_Epoch' + str(step_size) + '_reg ' + str(reg)+'.png')
plt.show()