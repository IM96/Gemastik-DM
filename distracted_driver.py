import tensorflow as tf
import pandas as pd
import numpy as np
PATH_TRAIN = r"/media/sf_Distracted_Driver_Dataset/trainset_distracted_driver.csv"
PATH_CLASS = r"/media/sf_Distracted_Driver_Dataset/kelas_train_distracted_driver.csv"
N_CLASS = 10
N_PIXELS = 7500 #100x75 lxT

sess = tf.InteractiveSession()

def read_data():
#    X = pd.read_csv("trainset_distracted_driver.csv", header=None)
#    Y = pd.read_csv("kelas_train_distracted_driver.csv", header=None)
#    pict_id = X.ix[:,0]
    dat = pd.read_csv('dataset_full.csv', header=None)
    return dat

def bagi_testing_train():
    
    print "Membaca data"
    dat = read_data()
    print "Data sudah terbaca"
    
    tmp = dat.values
    m = len(dat)
    del dat
    print "Shuffling data"
#    for i in range(10):
#        np.random.shuffle(dat)
#        print "loop ke-%d" %(i+1)
    np.random.shuffle(tmp)
    
    train = pd.DataFrame(tmp)
    del tmp
    print "Split train testing"
#    print "lewat2"
    testing = train.ix[int(0.8*m):,:]
    testing_id = testing.ix[:,0].values
    testing_class = testing.ix[:,7501:].values
    testing = testing.ix[:,1:7500].values
#    print "lewat3"
    train = train.ix[0:int(0.8*m),:]
    train_id = train.ix[:,0].values
    train_class = train.ix[:,7501:].values
    train = train.ix[:,1:7500].values
#    print "lewat4"
    return train, train_id, train_class, testing, testing_id, testing_class
    
#    data
#    testing


"""
Coba pakai convolutional neural network
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_5x5(x):
  return tf.nn.max_pool(x, ksize=[1, 5, 5, 1],
                        strides=[1, 5, 5, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def convolution(x):
#    layer 1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,75,100,1]) #reshape ulang gambar
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

#    layer 2 
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

#    Densly connected laye
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#    dropout untuk mencegah overfit
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#    output
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    
    return y_conv, keep_prob
  
def train_n_eval(x, y, train, train_class, testing, testing_class):
    max_iter = 10000
    prev_part=0
    curr_part=50
    m = len(train)
    prediction, keep_prob = convolution(x)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    for i in range(max_iter):
#        batch = mnist.train.next_batch(50)#the fuck is batch
        if curr_part>=m:
            curr_part = 50
            prev_part = 0
        """"""if i%10 == 0:
#            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0})
            train_accuracy = accuracy.eval(feed_dict={x:train[prev_part:curr_part,:], y: train_class[prev_part:curr_part,:], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))"""
#       
""" train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
        train_step.run(feed_dict={x: train[prev_part:curr_part,:], y: train_class[prev_part:curr_part,:], keep_prob: 0.5})
        prev_part = curr_part
        curr_part+=50
        print"Bagian ke-%d/%d" %(i+1,m%50)

#def predict():
    print("test accuracy %g"%accuracy.eval(feed_dict={x: testing, y: testing_class, keep_prob: 1.0})) #sek pelajari iki
    #sess.run(tf.initialize_all_variables())

x = tf.placeholder('float',[None, N_PIXELS])
y = tf.placeholder('float', [None, N_CLASS])  
  
train, train_id, train_class, testing, testing_id, testing_class = bagi_testing_train()

train_n_eval(x, y, train, train_class, testing, testing_class)

sess.close()   
"""

#coba pakai neural network 3 hidden layer

train, train_id, train_class, testing, testing_id, testing_class = bagi_testing_train()

n_nodes_hl1 = len(train[0,:])
n_nodes_hl2 = int(n_nodes_hl1/2)
n_nodes_hl3 = int(n_nodes_hl2/2)

n_classes = 10
#batch = bagian feature yg dikirimkan ke tensor (something like that)
batch_size = 1000

def batch(data, prev_batch, next_batch):
	return data[prev_batch:next_batch,:]

def neural_network_model(data):
	#(input_data * weights) + biases#
 	hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([N_PIXELS, n_nodes_hl1])),
 					  'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
 	
 	hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 
 					  'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}
 	
 	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
 					  'biases' : tf.Variable(tf.random_normal([n_nodes_hl3])) }
 	
 	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
 					'biases' : tf.Variable(tf.random_normal([n_classes]))}

 	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
 	#activation fuction
 	l1 = tf.nn.relu(l1)
 	


	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)
 	


 	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
 	l3 = tf.nn.relu(l3)

 	output = tf.matmul(l3, output_layer['weights'])+ output_layer['biases']

	return output

def train_neural_network(x, y, train, train_class, testing, testing_class):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	m = len(train)
	#banyak siklus prediction + backprop
	prev_part = 0
	curr_part = 1000
	hm_epochs = 100
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for i in range( int( m/batch_size) ):
				if curr_part <=m:
					curr_part=m
				epoch_x = batch(train, prev_part, curr_part)
				epoch_y = batch(train_class, prev_part, curr_part)
				i, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y} )
				epoch_loss +=c
				prev_part=curr_part
				curr_part+=1000
			print('Epoch', epoch,' completed out of', hm_epochs, 'loss: ', epoch_loss)
		
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float') )
		print('Accuracy:', accuracy.eval( {x:testing, y:testing_class}) )

x = tf.placeholder('float',[None, N_PIXELS])
y = tf.placeholder('float', [None, N_CLASS])

train_neural_network(x, y, train, train_class, testing, testing_class)

#hmm masih cacad akursasi cm 0.12
#need improvement