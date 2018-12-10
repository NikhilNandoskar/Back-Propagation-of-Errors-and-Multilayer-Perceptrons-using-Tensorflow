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
Problem 1a: Softmax Regression \& the XOR Problem

@author - Alexander G. Ororbia II and Ankur Mali
'''
    
def Softmax(x, indices):
    #Forward
    N = tf.shape(x, out_type = tf.float64)
    exp = tf.exp(x)
    probs = tf.divide(exp, tf.reduce_sum(exp, axis = 1, keepdims = True))
    probs_1 = tf.gather_nd(probs, indices)
    loss_param =- tf.divide(tf.reduce_sum(tf.log(probs_1)), N)

	# Correct probs:
    b_p = probs - probs_hot
    b_p = tf.divide(b_p, N[0])
    return loss_param,b_p, probs
    

def scatter_matrix(indice_scatter, updates, shape):
	indicess = tf.constant([indice_scatter])
	#updates = tf.constant([updates])
	scatterMatrix = tf.scatter_nd(indicess, updates, shape)
	scatterMatrix = tf.cast(scatterMatrix, dtype = tf.float64)
	return scatterMatrix

def computeNumGrad(X,y,theta,reg, indices): # returns approximate nabla
    # WRITEME: write your code here to complete the routine
    eps = 1e-5
    theta_list = list(theta)
    nabla_n = []
    # NOTE: you do not have to use any of the code here in your implementation...

    for q in range(len(theta_list)):
        #sess = tf.Session()
        #sess.run(tf.global_variables_initializer())
        
        m = [param_dim[2*q], param_dim[2*q + 1]]
        param_grad = (tf.zeros(tf.shape(theta_list[q]), dtype = tf.float64))
        #param_grad = tf.Variable(param_grad)
        #sess.run(tf.global_variables_initializer())

        #m = list(np.shape(param_grad))

        for i in range(m[0]):
            for j in range(m[1]):

                param = theta_list[q]
                #print(param.shape)
                
                # Scatter matrix for param
                indice_scatter = [i,j] if q==0 else [i]
                scatter = scatter_matrix(indice_scatter, tf.constant([eps]),tf.shape(param))

                #J+
                param += scatter

                if q==0:
                    theta_plus =(param, theta[q ^ 1])
                else:
                    theta_plus = (theta[q ^ 1], param)
                    
                J_plus = computeCost(X, y, theta_plus, reg, indices)


                #J-
                param -= 2 * scatter

                if q==0:
                    theta_minus =(param, theta[q ^ 1])
                else:
                    theta_minus = (theta[q ^ 1], param)

                J_minus = computeCost(X, y, theta_minus, reg, indices)

                
                #Compute Cost difference
                param_grad_updtae  = tf.divide(tf.subtract(J_plus, J_minus), (2 * eps))
                param_grad_updtae = param_grad_updtae.eval()
                
                # Param_grad scatter matrix
                param_scatter = scatter_matrix(indice_scatter, tf.constant([param_grad_updtae]),tf.shape(param))
                
                param_grad = param_grad + param_scatter

                #Reset Param
                #theta_list[k][i][j] =  theta_list[k][i][j] + eps
        
        nabla_n.append(param_grad)
    
    return tuple(nabla_n)
			
	
def computeGrad(X,y,theta,reg, indices): # returns nabla
	# WRITEME: write your code here to complete the routine
    #W = theta[0]
    #b = theta[1]
	#N = tf.shape(X)
    f = tf.add(tf.matmul(X,theta[0]), theta[1])
    _, back_softmax_out, _ = Softmax(f,indices)
    dW = tf.matmul(tf.transpose(X), back_softmax_out) + tf.multiply(reg, theta[0])
    db = tf.reduce_sum(back_softmax_out, axis = 0)	
    nabla = tf.tuple([dW, db])
    return (nabla)

def computeCost(X,y,theta,reg, indices):
	# WRITEME: write your code here to complete the routine
    #N = tf.shape(X)
    #W = theta[0]
    #b = theta[1]
    f = tf.add(tf.matmul(X,theta[0]), theta[1])
    # Loss Calculation:
    softmax_out, _, _ = Softmax(f, indices)
    #print(softmax_out.eval(session = sess))
    #loss_param = -(tf.divide(tf.reduce_sum(tf.log(softmax_out)), N[0]))
    #regularise_param = tf.divide(tf.multiply(reg, tf.reduce_sum(tf.square(W))), tf.cast(2, dtype = tf.float64))
    regularise_param = 0.5 * tf.multiply(tf.reduce_sum(tf.square(theta[0])), reg)
    cost = softmax_out + regularise_param
    return cost

def predict(X,theta, indices):
	# WRITEME: write your code here to complete the routine
    #W = theta[0]
    #b = theta[1]
	# evaluate class scores
    scores = tf.add(tf.matmul(X,theta[0]), theta[1])
	# compute the class probabilities
    _, _, predicts = Softmax(scores, indices)
	#probs = 0.0
    return (scores,predicts)



tf.set_random_seed(1491189)
np.random.seed(1491189) #Provide your unique Random seed
# Load in the data from disk
path = os.getcwd() + '/data/xor.dat'  
data = pd.read_csv(path, header=None) 


#Save images
#makdir (os.getcwd() + '/Result_images' + '/prob_1b')
save_dir = os.getcwd() + '/Result_images' + '/prob_1a'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)
y = y.flatten()


#Create indices matrix with row number and corresponding label value
indices = []
indices = [[i, y[i]] for i in range(y.shape[0])]
indices = np.array(indices)
#print(indices[:, 1])
indices = tf.constant(indices, dtype=tf.int64)

X_tf = tf.constant(X)
Y_tf = tf.constant(y)

#Placeholdeers
X_p= tf.placeholder(tf.float64, shape = (X.shape[0],X.shape[1]))
y_p= tf.placeholder(tf.float64, shape= (y.shape[0]))


#Train a Linear Classifier

# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1

# initialize parameters in such a way to play nicely with the gradient-check!
#W = 0.01 * np.random.randn(D,K)
#b = np.zeros((1,K)) + 1.0

#initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=20510119, dtype=tf.float64)
#initializer = tf.contrib.layers.xavier_initializer(seed = 20510119, dtype=tf.float64)
#W_t = tf.Variable(0.01 * initializer([D, K]), dtype = tf.float64, name = "w_t")
W = 0.01 * np.random.randn(D,K)
W_t = tf.Variable(W, dtype = tf.float64, name = "w_t")
#b_t = tf.Variable(tf.random_normal([K], dtype=tf.float64), dtype = tf.float64, name = "b_t")
#b_t = np.zeros((K,1))
b_t = tf.Variable(tf.zeros([K], dtype=tf.float64), dtype = tf.float64, name = "b_t")
theta = (W_t, b_t)

#Onehot of y
probs_hot = tf.contrib.layers.one_hot_encoding(indices[:, 1], K)
probs_hot = tf.cast(probs_hot , dtype = tf.float64)

#Number of examples
N = tf.shape(X, out_type = tf.float64)[0]

#Save dimensions of parameters
param_dim = [D, K, K, 1]

# some hyperparameters
reg = 1e-3 # regularization strength
reg = tf.constant(reg, dtype = tf.float64,  name = "reg") # regularization strength


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	nabla_n = computeNumGrad(X_tf, Y_tf,theta, reg, indices)
	#nabla_n = sess.run([],feed_dict = {X_p:X , y_p:y })
	nabla = computeGrad(X_tf, Y_tf,theta, reg, indices)
	nabla_n = list(nabla_n)
	nabla = list(nabla)
	
#Initialize your variables
#sess = tf.Session()
	for jj in range(0,len(nabla)):
		is_incorrect = 0 # set to false
		grad = nabla[jj]
		grad_n = nabla_n[jj]
		grad_sub = tf.subtract(grad_n,grad)
		grad_add = tf.add(grad_n,grad)
		err = tf.div(tf.norm(grad_sub, ord=2) , (tf.norm(grad_add, ord=2)))
		if(err.eval() > 1e-7):
			print("Param {0} is WRONG, error = {1}".format(jj, sess.run(err)))
		else:
			print("Param {0} is CORRECT, error = {1}".format(jj, sess.run(err)))



# Re-initialize parameters for generic training
#initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=1491189, dtype=tf.float64) #You can use Xavier or Ortho for weight init
#If using other init compare that with Guassian init and report your findings
W1 = 0.01 * np.random.randn(D,K)
W_t = tf.Variable(W1, dtype = tf.float64, name = "w_t")

b_t = tf.Variable(tf.zeros([K], dtype=tf.float64), dtype = tf.float64, name = "b_t")
theta = (W_t, b_t)

#play with hyperparameters for better performance 
n_e = 100 #number of epochssess.run(tf.global_variables_initializer())
check = 10 # every so many pass/epochs, print loss/error to terminal
step_size = tf.constant(1, dtype = tf.float64)
reg = tf.constant(0.001, dtype = tf.float64) # regularization strength
error_param = []

# Creating Placeholders:
X_p = tf.placeholder(tf.float64, shape = (X.shape[0], X.shape[1]))
Y_p = tf.placeholder(tf.float64, shape = (y.shape[0]))

# Session parameters:
ComputeCost = computeCost(X_p,Y_p,theta,reg, indices)
ComputeGrad = computeGrad(X_p,Y_p,theta,reg, indices)
# gradient descent loop
num_examples = X.shape[0]
with tf.Session() as sess: #You can exclude this ans use sess.run() whenever needed 
    sess.run(tf.global_variables_initializer())
    for i in range(n_e):
		# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
        loss = sess.run([ComputeCost], feed_dict={X_p: X, Y_p: y})
        error_param.append(loss)
        if i % check == 0:
            print ("iteration %d: loss %f" % (i, loss[0]))

		# perform a parameter update
		# WRITEME: write your update rule(s) here
        grand_new1 = sess.run([ComputeGrad], feed_dict={X_p: X , Y_p: y})
        #print('*******')
        #print((grand_new1[0])) 
        grand_new2 = [item for sublist in grand_new1 for item in sublist]
        #grand_new2 = tf.cast(grand_new2, dtype=tf.float64)
        #print(grand_new2[1])
        W_t1 = tf.subtract(W_t, tf.multiply(step_size, grand_new2[0]))
        #W_t2 = tf.convert_to_tensor(W_t1)
        b_t1 = tf.subtract(b_t, tf.multiply(step_size, grand_new2[1]))
        #b_t2 = tf.convert_to_tensor(b_t1)
        sess.run(tf.assign(W_t, W_t1))
        sess.run(tf.assign(b_t, b_t1))
 
# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
#sys.exit(0) 
   
    scores, probs = predict(X,theta, indices)
#scores = np.dot(X, W) + b
    predicted_class = sess.run(tf.argmax(scores, axis=1))
    #q = sess.run(tf.reduce_mean(tf.to_float(predicted_class == y)))
    print('training accuracy: %.2f' % sess.run(tf.reduce_mean(tf.to_float(predicted_class == y))))
    reg1 = sess.run(reg)
    step_size1 = sess.run(step_size)
    plt.title("Classifier")
    plt.savefig(save_dir  + '_step_size' + str(step_size1) + '_reg ' + str(reg1)+'.png')
    

    fig = plt.figure(1)
    plt.plot(error_param)
    plt.title('Loss vs Epoch')
    plt.xlabel('Number of epochs')
    plt.ylabel('Cost')
    plt.savefig(save_dir  + '_step_size' + str(step_size1) + '_reg ' + str(reg1)+'.png')
plt.show()
