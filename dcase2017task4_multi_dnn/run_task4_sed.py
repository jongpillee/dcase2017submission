import os
import numpy as np
import cPickle as cP
import re
import sed_eval
import nose.tools
import csv
import time

class Options(object):
	def __init__(self):
		self.train_size = 45313
		self.valid_size = 5859
		self.test_size = 488
		self.num_tags = 17
		self.batch_size = 23
		self.nb_epoch = 1000
		self.lr_list = [0.002]
		self.lrdecay = 1e-6
		self.gpu_use = 1
		self.trial = 3
		self.num_neurons = [4096,8192]
		self.build_model = [2]
		self.model_list = ['372','557','627','743','893','1486','2678','3543']#
		self.level_select = [[-2,-1,0]]
		self.activ = 'relu'
		self.init = 'he_uniform'
		self.patience = 5
		self.threshold = [0.5]
		
options = Options()
theano_config = "mode=FAST_RUN,device=gpu%d,floatX=float32" % options.gpu_use
os.environ['THEANO_FLAGS'] = theano_config

from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.optimizers import SGD
from keras.models import model_from_json,Model
from keras.callbacks import Callback,ModelCheckpoint,EarlyStopping
from keras.layers import Input,merge
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from sklearn.metrics import precision_recall_fscore_support

activ = options.activ
init = options.init

feature_path = '/media/ssd2/dcase2017_task4/multi_features/sed/'
model_path = '/home/richter/dcase2017task4_models/'
label_path = '/media/ssd2/dcase2017_task4/processed_label/'	
data_path = '/home/richter/dcase2017task4_multi_dnn/evaluation/'


def build_model0(feature_length,num_neurons):
	
	pool_input = Input(shape=(feature_length,))

	output = Dense(options.num_tags,activation='sigmoid')(pool_input)
	model = Model(input=pool_input,output=output)
	return model

def build_model1(feature_length,num_neurons):
	
	pool_input = Input(shape=(feature_length,))

	dense1 = Dense(num_neurons,activation=activ)(pool_input)

	output = Dense(options.num_tags,activation='sigmoid')(dense1)
	model = Model(input=pool_input,output=output)
	return model

def build_model2(feature_length,num_neurons):

	pool_input = Input(shape=(feature_length,))

	dense1 = Dense(num_neurons,activation=activ)(pool_input)
	dense2 = Dense(num_neurons,activation=activ)(dense1)

	output = Dense(options.num_tags,activation='sigmoid')(dense2)
	model = Model(input=pool_input,output=output)
	return model

def load_label():
	train_list = cP.load(open(label_path + 'training_set_train_list.cP','r'))
	valid_list = cP.load(open(label_path + 'training_set_valid_list.cP','r'))
	test_list = cP.load(open(label_path + 'testing_set_list.cP','r'))
	evaluation_list = cP.load(open(label_path + 'evaluation_set_list.cP','r'))
	print len(train_list),len(valid_list),len(test_list)

	balanced_list_to_tagid = cP.load(open(label_path + 'training_set_ytid_to_tagid.cP','r'))
	eval_list_to_tagid = cP.load(open(label_path + 'testing_set_ytid_to_tagid.cP','r'))
	tagid_to_index = cP.load(open(label_path + 'tagid_to_index.cP','r'))
	tagid_to_name = cP.load(open(label_path + 'tagid_to_name.cP','r'))

	return train_list,valid_list,test_list,balanced_list_to_tagid,eval_list_to_tagid,tagid_to_index,tagid_to_name,evaluation_list

def load_multi_features(file_name,num_models,num_levels,model_specific_list,activation_dict,level_select):
	
	for model_iter in range(num_models):
		for i,level_iter in enumerate(level_select):
			# each level save path
			model_specific = model_specific_list[model_iter]
			activation_layer = activation_dict[model_specific]
			feature_path_specific = feature_path + model_specific + '/' + activation_layer[level_iter] + '/'

			feature_name = feature_path_specific + file_name.replace('.wav','.npy')
			tmp = np.load(feature_name)
			#print model_specific, tmp.shape,file_name

			if (i == 0 and model_iter == 0):
				all_feature = tmp
			else:
				all_feature = np.hstack((all_feature,tmp))
				#print all_feature.shape
	return all_feature

def test_dcase_style(save_file_name):
	
	reference = data_path + 'groundtruth/groundtruth_strong_label_testing_set.csv'
	estimated = save_file_name
	
	reference_event_list = sed_eval.io.load_event_list(reference)
	estimated_event_list = sed_eval.io.load_event_list(estimated)

	evaluated_event_labels = reference_event_list.unique_event_labels
	evaluated_files = reference_event_list.unique_files

	segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
		event_label_list=evaluated_event_labels,
		time_resolution=1.0
	)

	for file in evaluated_files:
		reference_event_list_for_current_file = reference_event_list.filter(file=file)
		estimated_event_list_for_current_file = estimated_event_list.filter(file=file)
		segment_based_metrics.evaluate(
			reference_event_list = reference_event_list_for_current_file,
			estimated_event_list = estimated_event_list_for_current_file
		)
	results = segment_based_metrics.results()
	print 'ER:', results['overall']['error_rate'], 'F_score:', results['overall']['f_measure']
	return results


class SGDLearningRateTracker(Callback):
	def on_epoch_end(self,epoch,logs={}):
		optimizer = self.model.optimizer

		# lr printer
		lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
		print('\nEpoch %d lr: %.6f' % (epoch+1, lr))


def calculate_num_segs(file_list,num_models,num_levels,model_specific_list,activation_dict,level_select):
	accum_segs = 0
	for iter in range(len(file_list)):
		tmp = load_multi_features(file_list[iter],num_models,num_levels,model_specific_list,activation_dict,level_select)
		
		num_segs = tmp.shape[0]
		
		accum_segs += num_segs
		#print accum_segs,iter
	print accum_segs
	return accum_segs

def rerun(num_neurons,level_select,lr,lr_prev,trial,model_dense):

	model_list = options.model_list
	num_models = len(model_list)
	num_levels = len(level_select)

	# define model path
	model_specific_list = []
	activation_dict = {}
	# load models
	for model_iter in range(num_models):
		model_name = model_list[model_iter]

		architecture_name = 'architecture_input' + model_name + '.json'
		weight_name = 'weights_input' + model_name + '.hdf5'

		json_file = open(model_path + architecture_name,'r')
		loaded_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_json)

		#model.summary()

		layer_dict = dict([(layer.name,layer) for layer in model.layers[1:]])
		layer_num = (len(layer_dict)-1)/4
		print layer_num

		end_layer = layer_num 

		# define
		model_specific = 'input' + model_name
		model_specific_list.append(model_specific)

		activation_tmp_list = []
		for iter in range(end_layer-2, end_layer+1):
			prints = 'activation_%d' % iter
			print prints
			activation_tmp_list.append(prints)
		activation_dict[model_specific] = activation_tmp_list

		print model_specific_list

	# load labels
	train_list,valid_list,test_list,balanced_list_to_tagid,eval_list_to_tagid,tagid_to_index,tagid_to_name = load_label()

	# generate file indicator
	for model_iter in range(num_models):
		model_specific_nm = '[input%s]' % model_list[model_iter]
		if model_iter == 0:
			model_specific_list_nm = model_specific_nm
		else:
			model_specific_list_nm += model_specific_nm
	print model_specific_list_nm
	neuron_prints = '[num_neurons:%d]' % num_neurons
	lr_prints = '[lr:%.6f]' % lr
	lr_prev_prints = '[lr:%.6f]' % lr_prev
	trial_prints = '[trial:%d]' % trial
	model_prints = '[model:%d]' % model_dense
	indicator = model_prints + str(level_select) + neuron_prints + model_specific_list_nm + lr_prints + trial_prints
	indicator_prev = model_prints + str(level_select) + neuron_prints + model_specific_list_nm + lr_prev_prints + trial_prints

	print indicator

	# parameters
	batch_size = options.batch_size
	nb_epoch = options.nb_epoch
	lrdecay = options.lrdecay

	# loading 1 sample for measure feature_length(concatenated features)
	tmp_feature = load_multi_features(train_list[0],num_models,num_levels,model_specific_list,activation_dict,level_select)
	feature_length = tmp_feature.shape[1]
	print feature_length

	# load data
	train_size = len(train_list)
	valid_size = len(valid_list)
	test_size = len(test_list)

	# calculate total segments
	train_num_segs = calculate_num_segs(train_list,num_models,num_levels,model_specific_list,activation_dict,level_select)
	valid_num_segs = calculate_num_segs(valid_list,num_models,num_levels,model_specific_list,activation_dict,level_select)
	test_num_segs = calculate_num_segs(test_list,num_models,num_levels,model_specific_list,activation_dict,level_select)

	# initialize
	x_train = np.zeros((train_num_segs,feature_length)) 
	x_valid = np.zeros((valid_num_segs,feature_length))
	y_train = np.zeros((train_num_segs,options.num_tags))
	y_valid = np.zeros((valid_num_segs,options.num_tags))

	accum_segs = 0
	for iter in range(len(train_list)):
		tmp = load_multi_features(train_list[iter],num_models,num_levels,model_specific_list,activation_dict,level_select)
		
		num_segs = tmp.shape[0]

		x_train[accum_segs:accum_segs+num_segs,:] = tmp

		# label
		tagindex_list = [tagid_to_index[x] for x in balanced_list_to_tagid[train_list[iter]]]
		y_tmp = np.zeros((1,options.num_tags))
		for iter2 in range(len(tagindex_list)):
			y_tmp[0][tagindex_list[iter2]] = 1

		y_tvt = np.repeat(y_tmp,num_segs,axis=0)
		y_train[accum_segs:accum_segs+num_segs,:] = y_tvt

		accum_segs += num_segs

		if np.remainder(iter,1000) == 0:
			print iter
	print iter+1
	print x_train.shape,y_train.shape

	accum_segs = 0
	for iter in range(len(valid_list)):
		tmp = load_multi_features(valid_list[iter],num_models,num_levels,model_specific_list,activation_dict,level_select)
		
		num_segs = tmp.shape[0]

		x_valid[accum_segs:accum_segs+num_segs,:] = tmp

		# label
		tagindex_list = [tagid_to_index[x] for x in balanced_list_to_tagid[valid_list[iter]]]
		y_tmp = np.zeros((1,options.num_tags))
		for iter2 in range(len(tagindex_list)):
			y_tmp[0][tagindex_list[iter2]] = 1

		y_tvt = np.repeat(y_tmp,num_segs,axis=0)
		y_valid[accum_segs:accum_segs+num_segs,:] = y_tvt

		accum_segs += num_segs

		if np.remainder(iter,1000) == 0:
			print iter
	print iter+1
	print x_valid.shape,y_valid.shape

	# load model
	prev_weight_name = './models_sed/weights' + indicator_prev + '.hdf5'
	prev_architecture_name = './models_sed/architecture' + indicator_prev + '.json'

	json_file = open(prev_architecture_name,'r')
	loaded_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_json)
	model.load_weights(prev_weight_name)
	print 'model loaded'

	# compile & optimizer
	sgd = SGD(lr=lr,decay=lrdecay,momentum=0.9,nesterov=True)
	model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
	model.summary()

	# callbacks
	weight_name = './models_sed/weights' + indicator + '.hdf5'
	checkpointer = ModelCheckpoint(weight_name,monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
	earlyStopping = EarlyStopping(monitor='val_loss',patience=options.patience,verbose=0,mode='auto')
	lr_tracker = SGDLearningRateTracker()

	# train
	hist = model.fit(x_train,y_train,callbacks=[earlyStopping,checkpointer,lr_tracker],batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_valid,y_valid))

	# save model architecture	
	json_string = model.to_json()
	json_name = './models_sed/architecture' + indicator + '.json'
	open(json_name,'w').write(json_string)

	# load best model weights for testing
	model.load_weights(weight_name)


	# test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# load test sets & prediction
	index_to_tagid = dict(zip(tagid_to_index.values(),tagid_to_index.keys()))

	file_to_predict = {}
	for iter in range(len(test_list)):
		tmp = load_multi_features(test_list[iter],num_models,num_levels,model_specific_list,activation_dict,level_select)
		
		predict_per_song = model.predict(tmp)

		file_to_predict[test_list[iter]] = predict_per_song
	
	# threshold iter
	org_predict = file_to_predict.keys()

	threshold = options.threshold
	for iter in range(len(threshold)):
		threshold_this = threshold[iter]
		threshold_prints = '[threshold:%f]' % threshold_this
		indicator_tmp = indicator + threshold_prints

		file_to_predict_thresholded = {}
		for file_iter in range(len(org_predict)):
			predict = np.copy(org_predict[file_iter])
			predict[org_predict[file_iter] >= threshold_this] = 1
			predict[org_predict[file_iter] < threshold_this] = 0

			file_to_predict_thresholded[test_list[file_iter]] = predict

		print indicator_tmp

		# write csv file		
		predictio_thresholded = file_to_predict_thresholded.values()
		save_file_name = data_path + 'prediction_sed/' + indicator_tmp + '.csv'
		with open(save_file_name, 'wb') as f:
			wr = csv.writer(f,quoting=csv.QUOTE_NONE,delimiter='\t')
			for file_iter2 in range(len(test_list)):
				
				tmp_predictio = predictio_thresholded[file_iter2]
				for each_hop in range(tmp_predictio.shape[0]):

					#tag iter
					for tag_iter in range(len(tagid_to_name)):
						if tmp_predictio[each_hop,tag_iter] == 0:
							continue
						else:
							prints = [test_list[file_iter2], str(each_hop), str(each_hop+1), tagid_to_name[index_to_tagid[tag_iter]]]
							wr.writerow(prints)

		# read csv file, calculate ER & F-score
		results = test_dcase_style(save_file_name)

		ER = results['overall']['error_rate']
		F_score = results['overall']['f_measure']

		print('overall_test_ER:,F1-score: %.4f, %.4f' % (ER,F_score))
		save_prints = '[ER:%.6f,F-score:%.6f]' % (ER,F_score)

		save_dir = '/home/richter/dcase2017task4_multi_dnn/evaluation/output_sed/'
		with open(save_dir+indicator_tmp+save_prints,'w') as filepath:
			filepath.write(results)
		filepath.close()
		print 'result save done!!!'





def main(num_neurons,level_select,lr,trial,model_dense):

	model_list = options.model_list
	num_models = len(model_list)
	num_levels = len(level_select)

	# define model path
	model_specific_list = []
	activation_dict = {}
	# load models
	for model_iter in range(num_models):
		model_name = model_list[model_iter]

		architecture_name = 'architecture_input' + model_name + '.json'
		weight_name = 'weights_input' + model_name + '.hdf5'

		json_file = open(model_path + architecture_name,'r')
		loaded_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_json)

		#model.summary()

		layer_dict = dict([(layer.name,layer) for layer in model.layers[1:]])
		layer_num = (len(layer_dict)-1)/4
		print layer_num

		end_layer = layer_num 

		# define
		model_specific = 'input' + model_name
		model_specific_list.append(model_specific)

		activation_tmp_list = []
		for iter in range(end_layer-2, end_layer+1):
			prints = 'activation_%d' % iter
			print prints
			activation_tmp_list.append(prints)
		activation_dict[model_specific] = activation_tmp_list

		print model_specific_list

	# load labels
	train_list,valid_list,test_list,balanced_list_to_tagid,eval_list_to_tagid,tagid_to_index,tagid_to_name,evaluation_list = load_label()

	# parameters
	batch_size = options.batch_size
	nb_epoch = options.nb_epoch
	lrdecay = options.lrdecay

	# loading 1 sample for measure feature_length(concatenated features)
	tmp_feature = load_multi_features(train_list[0],num_models,num_levels,model_specific_list,activation_dict,level_select)

	# ex. tmp_feature = (10,1280)

	feature_length = tmp_feature.shape[1]
	print feature_length

	# load data
	train_size = len(train_list)
	valid_size = len(valid_list)
	test_size = len(test_list)

	# calculate total segments
	train_num_segs = calculate_num_segs(train_list,num_models,num_levels,model_specific_list,activation_dict,level_select)
	valid_num_segs = calculate_num_segs(valid_list,num_models,num_levels,model_specific_list,activation_dict,level_select)
	test_num_segs = calculate_num_segs(test_list,num_models,num_levels,model_specific_list,activation_dict,level_select)

	# initialize
	x_train = np.zeros((train_num_segs,feature_length)) 
	x_valid = np.zeros((valid_num_segs,feature_length))
	y_train = np.zeros((train_num_segs,options.num_tags))
	y_valid = np.zeros((valid_num_segs,options.num_tags))

	accum_segs = 0
	for iter in range(len(train_list)):
		tmp = load_multi_features(train_list[iter],num_models,num_levels,model_specific_list,activation_dict,level_select)
		
		num_segs = tmp.shape[0]

		x_train[accum_segs:accum_segs+num_segs,:] = tmp

		# label
		tagindex_list = [tagid_to_index[x] for x in balanced_list_to_tagid[train_list[iter]]]
		y_tmp = np.zeros((1,options.num_tags))
		for iter2 in range(len(tagindex_list)):
			y_tmp[0][tagindex_list[iter2]] = 1

		y_tvt = np.repeat(y_tmp,num_segs,axis=0)
		y_train[accum_segs:accum_segs+num_segs,:] = y_tvt

		accum_segs += num_segs

		if np.remainder(iter,1000) == 0:
			print iter
	print iter+1
	print x_train.shape,y_train.shape

	accum_segs = 0
	for iter in range(len(valid_list)):
		tmp = load_multi_features(valid_list[iter],num_models,num_levels,model_specific_list,activation_dict,level_select)
		
		num_segs = tmp.shape[0]

		x_valid[accum_segs:accum_segs+num_segs,:] = tmp

		# label
		tagindex_list = [tagid_to_index[x] for x in balanced_list_to_tagid[valid_list[iter]]]
		y_tmp = np.zeros((1,options.num_tags))
		for iter2 in range(len(tagindex_list)):
			y_tmp[0][tagindex_list[iter2]] = 1

		y_tvt = np.repeat(y_tmp,num_segs,axis=0)
		y_valid[accum_segs:accum_segs+num_segs,:] = y_tvt

		accum_segs += num_segs

		if np.remainder(iter,1000) == 0:
			print iter
	print iter+1
	print x_valid.shape,y_valid.shape



	# build model
	model_eval = 'build_model%d(feature_length,num_neurons)' % model_dense
	model = eval(model_eval)

	# compile & optimizer
	sgd = SGD(lr=lr,decay=lrdecay,momentum=0.9,nesterov=True)
	model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
	model.summary()

	# generate file indicator
	for model_iter in range(num_models):
		model_specific_nm = '[input%s]' % model_list[model_iter]
		if model_iter == 0:
			model_specific_list_nm = model_specific_nm
		else:
			model_specific_list_nm += model_specific_nm
	print model_specific_list_nm
	neuron_prints = '[num_neurons:%d]' % num_neurons
	lr_prints = '[lr:%.6f]' % lr
	trial_prints = '[trial:%d]' % trial
	model_prints = '[model:%d]' % model_dense
	indicator = model_prints + str(level_select) + neuron_prints + model_specific_list_nm + lr_prints + trial_prints
	print indicator

	# callbacks
	weight_name = './models_sed/weights' + indicator + '.hdf5'
	checkpointer = ModelCheckpoint(weight_name,monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
	earlyStopping = EarlyStopping(monitor='val_loss',patience=options.patience,verbose=0,mode='auto')
	lr_tracker = SGDLearningRateTracker()

	# train or not
	hist = model.fit(x_train,y_train,callbacks=[earlyStopping,checkpointer,lr_tracker],batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_valid,y_valid))
		

	# save model architecture	
	json_string = model.to_json()
	json_name = './models_sed/architecture' + indicator + '.json'
	open(json_name,'w').write(json_string)

	# load best model weights for testing
	model.load_weights(weight_name)


	# test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# load test sets & prediction
	index_to_tagid = dict(zip(tagid_to_index.values(),tagid_to_index.keys()))

	file_to_predict = {}
	for iter in range(len(test_list)):
		tmp = load_multi_features(test_list[iter],num_models,num_levels,model_specific_list,activation_dict,level_select)
		
		predict_per_song = model.predict(tmp)

		file_to_predict[test_list[iter]] = predict_per_song
	
	# threshold iter
	org_test_list = file_to_predict.keys()
	org_predict = file_to_predict.values()

	threshold = options.threshold
	for iter in range(len(threshold)):
		threshold_this = threshold[iter]
		threshold_prints = '[threshold:%f]' % threshold_this
		indicator_tmp = indicator + threshold_prints

		file_to_predict_thresholded = {}
		for file_iter in range(len(org_predict)):
			predict = np.copy(org_predict[file_iter])
			predict[org_predict[file_iter] >= threshold_this] = 1
			predict[org_predict[file_iter] < threshold_this] = 0

			file_to_predict_thresholded[org_test_list[file_iter]] = predict


		# write csv file		
		test_list_thresholded = file_to_predict_thresholded.keys()
		predictio_thresholded = file_to_predict_thresholded.values()
		save_file_name = data_path + 'prediction_sed/' + indicator_tmp + '.csv'
		with open(save_file_name, 'wb') as f:
			wr = csv.writer(f,quoting=csv.QUOTE_NONE,delimiter='\t')
			for file_iter2 in range(len(test_list)):
				
				tmp_predictio = predictio_thresholded[file_iter2]
				for each_hop in range(len(tmp_predictio)):

					#tag iter
					for tag_iter in range(len(tagid_to_name)):
						if tmp_predictio[each_hop,tag_iter] == 0:
							continue
						else:
							
							prints = [test_list_thresholded[file_iter2], str(round(each_hop*1,3)), str(round((each_hop+1)*1,3)), tagid_to_name[index_to_tagid[tag_iter]]]
							wr.writerow(prints)

		time.sleep(0.1)
	
		# read csv file, calculate ER & F-score
		results = test_dcase_style(save_file_name)

		ER = results['overall']['error_rate']['error_rate']
		F_score = results['overall']['f_measure']['f_measure']

		print('overall_test_ER:,F1-score: %.4f, %.4f' % (ER,F_score))
		save_prints = '[ER:%.6f,F-score:%.6f]' % (ER,F_score)

		save_dir = '/home/richter/dcase2017task4_multi_dnn/evaluation/output_sed/'

		np.save(save_dir+indicator_tmp+save_prints,results)
		print 'result save done!!!'

	# evaluation!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# load test sets & prediction

	file_to_predict_eval = {}
	for iter in range(len(evaluation_list)):
		tmp = load_multi_features(evaluation_list[iter],num_models,num_levels,model_specific_list,activation_dict,level_select)
		
		predict_per_song = model.predict(tmp)

		file_to_predict_eval[evaluation_list[iter]] = predict_per_song
	
	# threshold iter
	org_evaluation_list = file_to_predict_eval.keys()
	org_predict = file_to_predict_eval.values()

	threshold = options.threshold
	for iter in range(len(threshold)):
		threshold_this = threshold[iter]
		threshold_prints = '[threshold:%f]' % threshold_this
		indicator_tmp = '[evaluation]' + indicator + threshold_prints

		file_to_predict_thresholded = {}
		for file_iter in range(len(org_predict)):
			predict = np.copy(org_predict[file_iter])
			predict[org_predict[file_iter] >= threshold_this] = 1
			predict[org_predict[file_iter] < threshold_this] = 0

			file_to_predict_thresholded[org_evaluation_list[file_iter]] = predict


		# write csv file		
		evaluation_list_thresholded = file_to_predict_thresholded.keys()
		predictio_thresholded = file_to_predict_thresholded.values()
		save_file_name = data_path + 'prediction_evaluation_sed/' + indicator_tmp + '.csv'
		with open(save_file_name, 'wb') as f:
			wr = csv.writer(f,quoting=csv.QUOTE_NONE,delimiter='\t')
			for file_iter2 in range(len(evaluation_list)):
				
				tmp_predictio = predictio_thresholded[file_iter2]
				for each_hop in range(len(tmp_predictio)):

					#tag iter
					for tag_iter in range(len(tagid_to_name)):
						if tmp_predictio[each_hop,tag_iter] == 0:
							continue
						else:
							
							prints = [evaluation_list_thresholded[file_iter2][1:], str(round(each_hop*1,3)), str(round((each_hop+1)*1,3)), tagid_to_name[index_to_tagid[tag_iter]]]
							wr.writerow(prints)

		time.sleep(0.1)
	



if __name__ == '__main__':

	num_neurons = options.num_neurons
	level_selects = options.level_select
	lr_list = options.lr_list
	model_dense = options.build_model

	for model_iter in range(len(model_dense)):
		for neuron_iter in range(len(num_neurons)):
			for level_iter in range(len(level_selects)):
				for trial_iter in range(0,options.trial):
					main(num_neurons[neuron_iter],level_selects[level_iter],lr_list[0],trial_iter,model_dense[model_iter])

	for model_iter in range(len(model_dense)):
		for neuron_iter in range(len(num_neurons)):
			for level_iter in range(len(level_selects)):
				for trial_iter in range(0,options.trial):
					for lr_idx in range(1,len(lr_list)):
						rerun(num_neurons[neuron_iter],level_selects[level_iter],lr_list[lr_idx],lr_list[lr_idx-1],trial_iter,model_dense[model_iter])














