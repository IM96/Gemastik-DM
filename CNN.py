import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def read_image_list(image_list_file):
	f = open(image_list_file, 'r')
	people = []
	labels = []
	filename = []
	for line in f:
		line = line.rstrip()
		record = line.split(',')
		if record[0] == "subject":
			continue
		record[2] = "imgs/train/"+str(record[1])+"/"+record[2]
		people.append(record[0])
		labels.append(int(record[1][1]))
		filename.append(record[2])
	return people, labels, filename

def read_image_from_disk(input_queue):
	person = input_queue[1]
	labels = input_queue[2]
	file_contents = tf.read_file(input_queue[0])
	example = tf.image.decode_jpeg(file_contents, channels=3)
	compressed = tf.image.resize_images(example, 48, 64, 1)
	return compressed, person, labels

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(height, width, depth, output, x, stride, pad):
	W_conv = weight_variable([height, width, depth, output])
	b_conv = bias_variable([output])
	return tf.nn.relu(tf.nn.conv2d(x, W_conv, strides=stride, padding=pad) + b_conv)

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def int_to_one_hot(labels, num_classes=10):
	vec = np.zeros(num_classes)
	vec[labels] = 1
	return vec

def batch_normalization(x, units):
    beta = tf.Variable(tf.constant(0.0, shape=[units]))
    gamma = tf.Variable(tf.constant(1.0, shape=[units]))

    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])

    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    return tf.nn.batch_norm_with_global_normalization(x, ema_mean, ema_var, beta, gamma, 1e-3, True)

def dense_hidden(x, input_unit, output_unit):
	W_fc = weight_variable([input_unit, output_unit])
	b_fc = bias_variable([output_unit])
	return tf.sigmoid(tf.matmul(x, W_fc) + b_fc)

def dense_output(x, input_unit, output_unit):
	W_fc = weight_variable([input_unit, output_unit])
	b_fc = bias_variable([output_unit])
	return tf.nn.softmax(tf.matmul(h_fc_drop, W_fc) + b_fc)

imageList = read_image_list("driver_imgs_list.csv")
images = ops.convert_to_tensor(imageList[2], dtype=dtypes.string)
labels = ops.convert_to_tensor(imageList[1], dtype=dtypes.int32)
person = ops.convert_to_tensor(imageList[0], dtype=dtypes.string)

#pembagian training + testing data (manual untuk menjamin tidak ada driver testing yang masuk data training)
train_images = ops.convert_to_tensor(imageList[2][:18587], dtype=dtypes.string)
train_labels = ops.convert_to_tensor(imageList[1][:18587], dtype=dtypes.int32)
train_person = ops.convert_to_tensor(imageList[0][:18587], dtype=dtypes.string)

test_images = ops.convert_to_tensor(imageList[2][18587:], dtype=dtypes.string)
test_labels = ops.convert_to_tensor(imageList[1][18587:], dtype=dtypes.int32)
test_person = ops.convert_to_tensor(imageList[0][18587:], dtype=dtypes.string)

train_data = 18587

train_queue = tf.train.slice_input_producer([train_images,train_person,train_labels], shuffle=True)
img_train, ppl_train, label_train = read_image_from_disk(train_queue)

test_queue = tf.train.slice_input_producer([test_images,test_person,test_labels], shuffle=True)
img_test, ppl_test, label_test = read_image_from_disk(test_queue)

x = tf.placeholder(tf.float32, [None, 48, 64, 3])
y_ = tf.placeholder(tf.float32, [None, 10])


h_conv1 = conv2d(5,5,3,32,x,[1,1,1,1],'SAME')
h_norm1 = batch_normalization(h_conv1, 32)

h_conv2 = conv2d(5,5,32,32,h_norm1,[1,1,1,1],'SAME')
h_norm2 = batch_normalization(h_conv2, 32)

h_conv3 = conv2d(5,5,32,64,h_norm2,[1,2,2,1],'VALID')
h_norm3 = batch_normalization(h_conv3, 64)

h_conv4 =  conv2d(5,5,64,64,h_norm3,[1,1,1,1],'SAME')
h_norm4 = batch_normalization(h_conv4, 64)

h_conv5 = conv2d(5,5,64,128,h_norm4,[1,2,2,1],'VALID')
h_norm5 = batch_normalization(h_conv5, 128)

h_conv6 = conv2d(5,5,128,128,h_norm5,[1,1,1,1],'SAME')
h_norm6 = batch_normalization(h_conv6, 128)


h_flat = tf.reshape(h_norm6, [-1, 9*13*128])
h_fc = dense_hidden(h_flat, 9*13*128, 256)

keep_prob = tf.placeholder(tf.float32)
h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
y_conv = dense_output(h_fc_drop, 256, 10)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 100

with tf.Session() as sess:
	#init all variables
	sess.run(tf.initialize_all_variables())

	#Coordinate loading of images
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	for fold in range(1,8):
		validation_start = train_data/8
		validation_start = validation_start * (fold-1)
		validation_end = min(validation_start + (train_data/8), train_data) - 1

		valid_image = []
		valid_label = []

		data_id = 0

		for steps in range(224):
			train_image = []
			train_label = []

			for i in range(batch_size):
				image_tensor = sess.run([img_train, label_train, ppl_train])
				
				if data_id>=validation_start and data_id<=validation_end:
					valid_image.append(image_tensor[0])
					valid_label.append(int_to_one_hot(image_tensor[1]))
					i = i-1
				else:					
					train_image.append(image_tensor[0])
					train_label.append(int_to_one_hot(image_tensor[1]))

			train_step.run(feed_dict={x:train_image, y_:train_label, keep_prob: 0.8})

		print ("Training fold %g:" % (fold))
		print ("Cross Entropy Loss: %g" % (cross_entropy.eval(feed_dict={x:valid_image, y_:valid_label, keep_prob: 1.0})))
		print ("Akurasi: %g" % (accuracy.eval(feed_dict={x:valid_image, y_:valid_label, keep_prob: 1.0})))
		print ("")

	print ("Testing data:")
	print ("Cross Entropy Loss: %g" % (cross_entropy.eval(feed_dict={x:test_images, y_:test_labels, keep_prob: 1.0})))
	print ("Akurasi: %g" % (accuracy.eval(feed_dict={x:test_images, y_:test_labels, kepp_prob: 1.0})))

	coord.request_stop()
	coord.join(threads)
