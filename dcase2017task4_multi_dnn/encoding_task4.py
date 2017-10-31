import os
import numpy as np
import cPickle as cP
import re

gpu_use = 1

theano_config = "mode=FAST_RUN,device=gpu%d,floatX=float32" % gpu_use
os.environ['THEANO_FLAGS'] = theano_config

from keras.layers.core import Dropout,Activation,Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution1D,MaxPooling1D
from keras.models import model_from_json,Model
from keras.layers import Input,merge
from keras import backend as K
from theano import function

nst = 0

partition = 1

#'372'
model_list = ['372','557','627','743','893','1486','2678','3543']

model_path = '/home/richter/dcase2017task4_models/'
feature_path = '/media/ssd2/dcase2017_task4/wav_to_npy/'
save_path = '/media/ssd2/dcase2017_task4/multi_features/'
label_path = '/media/ssd2/dcase2017_task4/processed_label/'

def load_sample(file_name,sample_length):
	dir_name = feature_path + file_name.replace('.wav','.npy')
	tmp = np.load(dir_name)
	y_length = len(tmp)
	#print y_length

	num_segment = int(441000/sample_length)+1

	tmp_segmentized = np.zeros((num_segment,sample_length,1))
	# segmentation for tagging
	for iter2 in range(0,num_segment):
		
		hopping = (y_length-sample_length)/(num_segment-1)
		count_tmp = 0
		if hopping < 0:
			if count_tmp == 0:
				tmp_tmp = np.repeat(tmp,10)
				count_tmp += 1
			y_length_tmp = len(tmp_tmp)
			hopping = (y_length_tmp - sample_length)/(num_segment-1)
			tmp_segmentized[iter2,:,0] = tmp_tmp[iter2*hopping:iter2*hopping+sample_length]
		else:
			tmp_segmentized[iter2,:,0] = tmp[iter2*hopping:iter2*hopping+sample_length]
	
	# segmentation for sound event detection
	hopping = 1 * 44100

	if np.remainder(y_length,hopping) == 0:
		num_segment_per_song = int(y_length/hopping)
	else: 
		num_segment_per_song = int(y_length/hopping) + 1

	num_segment_within = int(44100/sample_length)+1
	print 'num_segment_within:',num_segment_within

	tmp_segmentized_sed = np.zeros((num_segment_per_song,num_segment_within,sample_length,1))
	for seg_iter in range(num_segment_per_song):
		tmp_seg = np.zeros((num_segment_within,sample_length))

		start = seg_iter*hopping

		if num_segment_within == 1:
			if start+sample_length > y_length:
				# padding
				length = y_length - start
				tmp_seg[0,:length] = tmp[start:]
				#print 'start:',start,'start+length',start+length,44100,'y_length',y_length

				'''
				# duplicate last segments so input is not empty
				length = y_length - (start)
				given_area = tmp[start:]
				print 'given_area:', start,y_length
				tmp_tmp = np.repeat(given_area,(10*sample_length/length))
				tmp_seg[0,:] = tmp_tmp[:sample_length]
				'''
			else:
				tmp_seg[0,:] = tmp[start:start+sample_length]
				#print 'start:',start,'start+sample_length',start+sample_length,44100,'y_length',y_length

		else:
			count_tmp = 0

			for iter2 in range(0,num_segment_within):
				
				hopping_within = (44100-sample_length)/(num_segment_within-1)
				#print 'hopping_within:',hopping_within
				
				if start+iter2*hopping_within+sample_length > y_length:

					# for last seconds
					if count_tmp == 0:
						given_area = tmp[start+iter2*hopping_within:]
						#print 'given_area:', start+iter2*hopping_within, y_length
						
						length = y_length - (start+iter2*hopping_within)
						#print length

						tmp_tmp = np.repeat(given_area,(10*sample_length/length))
						count_tmp += 1
					
					tmp_seg[iter2,:] = tmp_tmp[iter2*hopping_within:iter2*hopping_within+sample_length]
					#print iter2*hopping_within,iter2*hopping_within+sample_length,y_length

				else:
					tmp_seg[iter2,:] = tmp[start+iter2*hopping_within:start+iter2*hopping_within+sample_length]
					#print start+iter2*hopping_within,start+iter2*hopping_within+sample_length,y_length

		
		tmp_segmentized_sed[seg_iter,:,:,0] = tmp_seg

	#print tmp_segmentized.shape,tmp_segmentized_sed.shape

	return tmp_segmentized,tmp_segmentized_sed


# load data
'''
train_list = cP.load(open(label_path + 'training_set_train_list.cP','r'))
valid_list = cP.load(open(label_path + 'training_set_valid_list.cP','r'))
test_list = cP.load(open(label_path + 'testing_set_list.cP','r'))

all_list = train_list + valid_list + test_list
'''
all_list = cP.load(open(label_path + 'evaluation_set_list.cP','r'))
all_size = len(all_list)
print all_size

# model iter
for model_iter in range(len(model_list)):
	model_name = model_list[model_iter]

	# load models
	architecture_name = 'architecture_input' + model_name + '.json'
	weight_name = 'weights_input' + model_name + '.hdf5'

	json_file = open(model_path + architecture_name,'r')
	loaded_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_json)

	model.load_weights(model_path + weight_name)
	model.summary()
	print 'model loaded!!!'

	sample_length = model.input_shape[1]
	print sample_length

	# get the symbolic outputs of each 'key' layer
	layer_dict = dict([(layer.name,layer) for layer in model.layers[1:]])
	layer_num = (len(layer_dict)-1)/4
	print layer_num

	# compile
	sgd = SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)
	model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

	start_layer = layer_num - 2
	end_layer = layer_num + 1

	# level iter
	for level_iter in range(start_layer,end_layer):
		
		layer_name = 'activation_%d' % (level_iter)

		# set intermediate layer output function
		layer_output = layer_dict[layer_name].output
		get_level_output = K.function([model.layers[0].input, K.learning_phase()], [layer_output])

		# divide iter
		for divide_iter in range(int(nst*all_size/partition),int((nst+1)*all_size/partition)):
		
			'''	
			# encoding tagging
			# same segment avg features

			# each level save path
			save_path_specific_tagging = save_path + 'tagging/' + 'input' + model_name + '/' + layer_name + '/'
			save_name = save_path_specific_tagging + all_list[divide_iter].replace('.wav','.npy')
			print divide_iter,save_name
	
			# check existence
			if not os.path.exists(os.path.dirname(save_name)):
				os.makedirs(os.path.dirname(save_name))

			if os.path.isfile(save_name) == 1:
				print divide_iter, save_name + '_file_exist!!!!!!!!!!!'
				continue
			'''

			# load data
			x_sample_tagging,x_sample_sed = load_sample(all_list[divide_iter],sample_length)
			


			'''
			# prediction
			weight = get_level_output([x_sample_tagging,0])[0]
			print x_sample_tagging.shape,weight.shape

			# Frame-wise max pooling
			maxpooled = np.amax(weight,axis=1)

			# Time-wise average pooling
			averagepooled = np.average(maxpooled,axis=0)
			print averagepooled.shape

			# save each level in different directory
			np.save(save_name,averagepooled)
			'''
			
			
			# encoding sound event detection
			# 1 sec avg features for different input size with different hopping strategy

			# each level save path
			save_path_specific_sed = save_path + 'sed/' + 'input' + model_name + '/' + layer_name + '/'
			save_name = save_path_specific_sed + all_list[divide_iter].replace('.wav','.npy')
			print divide_iter,save_name

			print x_sample_sed.shape	

			# check existence
			if not os.path.exists(os.path.dirname(save_name)):
				os.makedirs(os.path.dirname(save_name))

			if os.path.isfile(save_name) == 1:
				print divide_iter, save_name + '_file_exist!!!!!!!!!!!'
				continue

			# TODO 
			# 10,2,32768,1 or 10,1,65536,1 treats => predict => FM => TM => 10,256

			flattened = x_sample_sed.reshape(x_sample_sed.shape[0]*x_sample_sed.shape[1],x_sample_sed.shape[2],x_sample_sed.shape[3])
			print flattened.shape

			# prediction			
			weight = get_level_output([flattened,0])[0]
			print weight.shape
			weight = np.reshape(weight,(x_sample_sed.shape[0],x_sample_sed.shape[1],weight.shape[1],weight.shape[2]))
			print flattened.shape,weight.shape

			# Frame-wise max pooling
			maxpooled = np.amax(weight,axis=2)

			# Time-wise average pooling
			averagepooled = np.average(maxpooled,axis=1)
			print averagepooled.shape

			# save each level in different directory
			np.save(save_name,averagepooled)
			
			
			


