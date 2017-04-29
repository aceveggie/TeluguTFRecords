import tensorflow as tf
import numpy as np
import input_data
import cv2
import os
import time

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

teluguDict= {}
teluguDict[7] = u'\u0C05'
teluguDict[6] = u'\u0C06'
teluguDict[5] = u'\u0C07'
teluguDict[4] = u'\u0C08'
teluguDict[3] = u'\u0C09'
teluguDict[2] = u'\u0C0A'
teluguDict[1] = u'\u0C60'
teluguDict[0] = u'\u0C61'

teluguDict[8] = u'\u0C14'
teluguDict[9] = u'\u0C13'
teluguDict[10] = u'\u0C12'
teluguDict[11] = u'\u0C10'
teluguDict[12] = u'\u0C0F'
teluguDict[13] = u'\u0C0E'
teluguDict[14] = u'\u0C0C'
teluguDict[15] = u'\u0C0B'

tfrecords_train_filename = 'Telugu_MNIST_Train.tfrecords'
tfrecords_test_filename = 'Telugu_MNIST_Test.tfrecords'

############################ TF RECORD WRITER ############################
def writeMyTFRecords():
	mnist = input_data.read_data_sets("./", one_hot=False)
	trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
	trX = trX.reshape(-1, 50, 50, 1)
	teX = teX.reshape(-1, 50, 50, 1)
	print 'now writing telugu tfrecords'
	time.sleep(3)

	trainWriter = tf.python_io.TFRecordWriter(tfrecords_train_filename)
	testWriter = tf.python_io.TFRecordWriter(tfrecords_test_filename)

	for eachImg in zip (trX, trY):
		img, label = eachImg[0], eachImg[1]
		img = (img * 255).astype(np.uint8)
		print label, teluguDict[label]
		# cv2.imshow("img", img)
		# cv2.waitKey(10)
		width, height = img.shape[0], img.shape[1]
		img = img.tostring()
		label = label.tostring()
		
		example = tf.train.Example(features=tf.train.Features(feature={
			'height': _int64_feature(height),
			'width': _int64_feature(width),
			'image_raw': _bytes_feature(img),
			'label_raw': _bytes_feature(label)}))
		trainWriter.write(example.SerializeToString())

	for eachImg in zip (teX, teY):
		img, label = eachImg[0], eachImg[1]
		img = (img * 255).astype(np.uint8)
		print label, teluguDict[label]
		# cv2.imshow("img", img)
		# cv2.waitKey(10)
		width, height = img.shape[0], img.shape[1]
		img = img.tostring()
		label = label.tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
			'height': _int64_feature(height),
			'width': _int64_feature(width),
			'image_raw': _bytes_feature(img),
			'label_raw': _bytes_feature(label)}))
		testWriter.write(example.SerializeToString())


	trainWriter.close()
	testWriter.close()
	print 'finished writing tf records'


############################ TF RECORD READER ############################
def readMyTFRecords():

	print 'now reading telugu tfrecords'
	time.sleep(3)

	trainRecordIterator = tf.python_io.tf_record_iterator(path=tfrecords_train_filename)
	testRecordIterator = tf.python_io.tf_record_iterator(path=tfrecords_test_filename)

	for eachRecord in trainRecordIterator:
		example = tf.train.Example()
		example.ParseFromString(eachRecord)

		height = int(example.features.feature['height']
			.int64_list
			.value[0])

		width = int(example.features.feature['width']
			.int64_list
			.value[0])

		img = (example.features.feature['image_raw']
			.bytes_list
			.value[0])

		label = (example.features.feature['label_raw']
			.bytes_list
			.value[0])

		img = np.fromstring(img, dtype=np.uint8)
		img = img.reshape(width, height)
		label = np.fromstring(label, dtype=np.uint8)[0]

		print label, teluguDict[label]
		cv2.imshow("img", img)
		cv2.waitKey(0)
		width, height = img.shape[0], img.shape[1]

	print 'finished reading tf records'


if(os.path.exists('Telugu_MNIST_Train.tfrecords') and os.path.exists('Telugu_MNIST_Test.tfrecords')):
	# if tf record file exists, just read the data
	readMyTFRecords()
	pass
else:
	# if tfrecrod file doesn't exist, create the data and then read the data
	# first lets read the data
	writeMyTFRecords()
	# then lets write the data
	readMyTFRecords()
	pass