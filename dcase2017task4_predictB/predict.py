import numpy as np
import os
import os.path
import cPickle as cP
import re
import csv

np.random.seed(0)

class Options(object):
	def __init__(self):
		self.test_size = 488 # 1103
		self.num_tags = 17
		self.gpu_use = 1
		self.hop = [1] # 1, 0.5, 0.25
		self.model_list = ['input893']
		'''
		input372,input557,input627,input743,input893,input1486,input2678,input3543
		'''
		self.threshold = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

options = Options()
theano_config = 'mode=FAST_RUN,device=gpu%d,floatX=float32,lib.cnmem=0.1' % options.gpu_use
os.environ['THEANO_FLAGS'] = theano_config

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import ZeroPadding1D,Convolution1D,MaxPooling1D,AveragePooling1D
from keras.models import model_from_json,Model
from keras.callbacks import Callback,ModelCheckpoint,EarlyStopping
from keras.layers import Input,merge
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import sys

feature_path = '/media/ssd2/dcase2017_task4/wav_to_npy/'
label_path = '/media/ssd2/dcase2017_task4/processed_label/'
model_path = '/home/richter/richter_chopin5_44452/dcase2017task4_models/'
predict_path = '/home/richter/richter_chopin5_44452/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/evaluation/predictiontest/'

def gen_path(file_name):
	dir_name = feature_path + file_name.replace('.wav','.npy')
	return dir_name

# load data
train_list = cP.load(open(label_path + 'training_set_train_list.cP','r'))
valid_list = cP.load(open(label_path + 'training_set_valid_list.cP','r'))
test_list = cP.load(open(label_path + 'testing_set_list.cP','r'))
print len(train_list),len(valid_list),len(test_list)
#evaluation_list = cP.load(open(label_path + 'evaluation_set_list.cP','r'))

#test_list = evaluation_list

# load model
model_list = options.model_list
models = []
input_shapes = []
sgd = SGD(lr=0.01)
for iter in range(len(model_list)):
	
	architecture_name = model_path + 'architecture_' + model_list[iter] + '.json'
	weight_name = model_path + 'weights_' + model_list[iter] + '.hdf5'
	
	json_file = open(architecture_name,'r')
	loaded_json = json_file.read()
	json_file.close()
	models.append(model_from_json(loaded_json))
	models[iter].load_weights(weight_name)

	models[iter].compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

	#models[iter].summary()
	input_shapes.append(models[iter].input_shape[1])
	
print input_shapes	
print 'all model loaded!!!'

# load ytid to tagid
tagid_to_index = cP.load(open(label_path + 'tagid_to_index.cP','r'))
tagid_to_name = cP.load(open(label_path + 'tagid_to_name.cP','r'))
eval_list_to_tagid = cP.load(open(label_path + 'testing_set_ytid_to_tagid.cP','r'))

index_to_tagid = dict(zip(tagid_to_index.values(),tagid_to_index.keys()))

# predicts & generate form
thresholds = options.threshold
hops = options.hop
hz = 44100

# hop iter
for hop_iter in range(len(hops)):
	hopping = int(hops[hop_iter] * hz)

	# model iter
	model_to_predict = {}
	for model_iter in range(len(model_list)):
		sample_length = input_shapes[model_iter]

		# file iter
		file_to_predict = {}
		for file_iter in range(0,len(test_list)):
			
			# load file
			file_path = gen_path(test_list[file_iter]) 
			tmp = np.load(file_path)
			y_length = len(tmp)
			
			num_segment_per_song = int(y_length/hopping) 

			# segmentation iter
			segmentized_song = np.zeros((num_segment_per_song,sample_length,1))
			predict_per_song = np.zeros((num_segment_per_song,options.num_tags))
			for seg_iter in range(num_segment_per_song):
				tmp_seg = np.zeros((sample_length))
				
				start = seg_iter*hopping
				
				if start+sample_length > y_length:
					# padding
					length = y_length - start
					tmp_seg[:length] = tmp[start:]
					#print start,start+length,hopping
				else:
					tmp_seg = tmp[start:start+sample_length]
					#print start,start+sample_length,hopping
				
				segmentized_song[seg_iter,:,0] = tmp_seg

			# predicts
			predict_per_song = models[model_iter].predict(segmentized_song)

			file_to_predict[test_list[file_iter]] = predict_per_song

		model_to_predict[model_list[model_iter]] = file_to_predict


		# result of each model
		# threshold iter
		hop_sec = round(hops[hop_iter],3)
		org_test_list =  model_to_predict[model_list[model_iter]].keys()
		org_predict = model_to_predict[model_list[model_iter]].values()
		for threshold_iter in range(len(thresholds)):
			threshold_this = thresholds[threshold_iter]
	
			file_to_predict_thresholded = {}
			for file_iter2 in range(len(org_predict)):
				predict = np.copy(org_predict[file_iter2])
				predict[org_predict[file_iter2] >= threshold_this] = 1
				predict[org_predict[file_iter2] < threshold_this] = 0
			
				file_to_predict_thresholded[org_test_list[file_iter2]] = predict

			predict_name = 'hop:' + str(hops[hop_iter]) + 'sec_' + model_list[model_iter] + '_' + 'threshold:' + str(threshold_this) + '.csv'
			print predict_name
	
			# each model file write
			test_list_thresholded = file_to_predict_thresholded.keys()
			predictio_thresholded = file_to_predict_thresholded.values()
			with open(predict_path + predict_name,'wb') as f:
				wr = csv.writer(f,quoting=csv.QUOTE_NONE,delimiter='\t')
				for file_iter2 in range(len(test_list_thresholded)):
				
					tmp_predictio = predictio_thresholded[file_iter2]
					for each_hop in range(len(tmp_predictio)):

						# tag iter
						for tag_iter in range(len(tagid_to_name)):
							if tmp_predictio[each_hop,tag_iter] == 0:
								continue
							else:
							
								prints = [test_list_thresholded[file_iter2][0:], str(round(each_hop*hop_sec,3)), str(round((each_hop+1)*hop_sec,3)), tagid_to_name[index_to_tagid[tag_iter]]]
								wr.writerow(prints)
								

	# avg before threshold
	hop_sec = round(hops[hop_iter],3)
	# threshold iter
	for threshold_iter in range(len(thresholds)):
		threshold_this = thresholds[threshold_iter]


		# aggregate multi-scale results
		file_to_predict_thresholded = {}
		test_list = model_to_predict[model_list[0]].keys()
		predictio = np.zeros_like(model_to_predict[model_list[0]].values())
		for model_iter2 in range(len(model_list)):
			predictio += model_to_predict[model_list[model_iter2]].values()
			print predictio
	
		# set aggregated value to average score
		predictio /= len(model_list)
		print predictio

		# thresholding
		for file_iter2 in range(len(test_list)):
			predictio[file_iter2][predictio[file_iter2] >= threshold_this] = 1
			predictio[file_iter2][predictio[file_iter2] < threshold_this] = 0
		print predictio

		# save name
		predict_name = 'hop:' + str(hops[hop_iter]) + 'sec_'
		for model_iter2 in range(len(model_list)):
			predict_name = predict_name + model_list[model_iter2]
		predict_name = predict_name + '_threshold:' + str(threshold_this) + '_AvgBeforeThreshold.csv'
		print predict_name

		# each model file write
		with open(predict_path + predict_name,'wb') as f:
			wr = csv.writer(f,quoting=csv.QUOTE_NONE,delimiter='\t')
			for file_iter2 in range(len(test_list)):
			
				tmp_predictio = predictio[file_iter2]
				for each_hop in range(len(tmp_predictio)):

					# tag iter
					for tag_iter in range(len(tagid_to_name)):
						if tmp_predictio[each_hop,tag_iter] == 0:
							continue
						else:
						
							prints = [test_list[file_iter2][0:], str(round(each_hop*hop_sec,3)), str(round((each_hop+1)*hop_sec,3)), tagid_to_name[index_to_tagid[tag_iter]]]
							wr.writerow(prints)







	'''
	# avg after threshold
	# result of multi-scale model
	hop_sec = round(hops[hop_iter],3)
	# threshold iter
	for threshold_iter in range(len(thresholds)):
		threshold_this = thresholds[threshold_iter]

		model_to_predict_thresholded = {}
		# model iter
		for model_iter2 in range(len(model_list)):
			org_test_list =  model_to_predict[model_list[model_iter2]].keys()
			org_predict = model_to_predict[model_list[model_iter2]].values()
			
			# file iter
			file_to_predict_thresholded = {}
			for file_iter2 in range(len(org_predict)):
				predict = np.copy(org_predict[file_iter2])
				predict[org_predict[file_iter2] >= threshold_this] = 1
				predict[org_predict[file_iter2] < threshold_this] = 0
			
				file_to_predict_thresholded[org_test_list[file_iter2]] = predict
			
			model_to_predict_thresholded[model_list[model_iter2]] = file_to_predict_thresholded


		# aggregate multi-scale results
		file_to_predict_thresholded = {}
		test_list_thresholded = model_to_predict_thresholded[model_list[0]].keys()
		predictio_thresholded = np.zeros_like(model_to_predict_thresholded[model_list[model_iter2]].values())
		for model_iter2 in range(len(model_list)):
			predictio_thresholded += model_to_predict_thresholded[model_list[model_iter2]].values()
			print predictio_thresholded
	
		# set aggregated value to binary
		predictio_thresholded /= len(model_list)
		print predictio_thresholded

		# save name
		predict_name = 'hop:' + str(hops[hop_iter]) + 'sec_'
		for model_iter2 in range(len(model_list)):
			predict_name = predict_name + model_list[model_iter2]
		predict_name = predict_name + '_threshold:' + str(threshold_this) + '_AvgAfterThreshold.csv'
		print predict_name

		# each model file write
		with open(predict_path + predict_name,'wb') as f:
			wr = csv.writer(f,quoting=csv.QUOTE_NONE,delimiter='\t')
			for file_iter2 in range(len(test_list_thresholded)):
			
				tmp_predictio = predictio_thresholded[file_iter2]
				for each_hop in range(len(tmp_predictio)):

					# tag iter
					for tag_iter in range(len(tagid_to_name)):
						if tmp_predictio[each_hop,tag_iter] == 0:
							continue
						else:
						
							prints = [test_list_thresholded[file_iter2], str(round(each_hop*hop_sec,3)), str(round((each_hop+1)*hop_sec,3)), tagid_to_name[index_to_tagid[tag_iter]]]
							wr.writerow(prints)
	'''









