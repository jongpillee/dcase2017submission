import os
import sys
from Models import *
import csv
import numpy as np
import cPickle as cP
import re

from keras.models import model_from_json
from keras.optimizers import SGD


label_path = '/media/ssd2/dcase2017_task4/processed_label/'
model_path = '/home/richter/dcase2017task4_models/'
feature_path = '/media/ssd2/dcase2017_task4/multi_features/tagging/'

def evaluateMetrics(groundtruth_filepath, predicted_filepath):
	#Load GroundTruth to memory, indexed by 
	groundTruthDS = FileFormat(groundtruth_filepath)
	predictedDS = FileFormat(predicted_filepath)
	#output = groundTruthDS.computeMetrics(preditedDS,output_filepath)
	
	print len(groundTruthDS.labelsDict.keys())
	print len(predictedDS.labelsDict.keys())


	'''
	#simple audioFileCheck
	if len(groundTruthDS.labelsDict.keys()) != len(predictedDS.labelsDict.keys()):
		print "The prediction file submitted does not have prediction for all the audio files"
		sys.exit(1)

	#complex check for audioFile
	if not groundTruthDS.validatePredictedDS(predictedDS):
		print "The prediction file submitted does not have prediction for all the audio files"
		sys.exit(1)
	'''


	#the submission is valid. Compute Metrics and Push to File
	output = groundTruthDS.computeMetricsString(predictedDS)

	return output
	'''
	with open(output_filepath, "w") as filepath:
		filepath.write(output)
	filepath.close()
	'''

def matToCsv(matrix,test_list,save_path,index_to_id,id_to_name):
	
	prediction = []
	for i in range(0, matrix.shape[0]):
		row_predict = []
		row_event = matrix[i]
		nonzero = np.nonzero(row_event)
		nonzero = np.array(nonzero).reshape(-1,)
		if not nonzero.size:
			name = test_list[i]
			
			idx = [m.start() for m in re.finditer('_',name)]
			audio_id = name[:idx[-2]]
			audio_id = audio_id[-11:]
			
			onset = name.split('_')[-2]
			offset = name.split('_')[-1]
			offset = offset.split('.wav')[0]

			file_name = audio_id + '_' + onset + '_' + offset + '.wav'

			row_predict.append((file_name, onset, offset))
			prediction.append(row_predict)
		else:
			for j in range(0,nonzero.size):
				name = test_list[i]
			
				idx = [m.start() for m in re.finditer('_',name)]
				audio_id = name[:idx[-2]]
				audio_id = audio_id[-11:]
				
				onset = name.split('_')[-2]
				offset = name.split('_')[-1]
				offset = offset.split('.wav')[0]
				label = id_to_name[index_to_id[nonzero[j]]]
			
				file_name = audio_id + '_' + onset + '_' + offset + '.wav'
		
				row_predict.append((file_name, onset, offset, label))
			prediction.append(row_predict)

	with open(save_path, 'wb') as f:
		writer = csv.writer(f, delimiter='\t')
		for row in prediction:
			for item in row:
				writer.writerow(item)

def load_label():
	test_list = cP.load(open(label_path + 'testing_set_list.cP','r'))
	eval_list_to_tagid = cP.load(open(label_path + 'testing_set_ytid_to_tagid.cP','r'))
	evaluation_list = cP.load(open(label_path + 'evaluation_set_list.cP','r'))

	tagid_to_index = cP.load(open(label_path + 'tagid_to_index.cP','r'))

	index_to_tagid = dict(zip(tagid_to_index.values(),tagid_to_index.keys()))
	id_to_name = cP.load(open(label_path + 'tagid_to_name.cP','r'))
	return test_list,eval_list_to_tagid,index_to_tagid,id_to_name,tagid_to_index,evaluation_list

def load_multi_features(file_name,num_models,num_levels,model_specific_list,activation_dict,level_select):

	for model_iter in range(num_models):
		for i,level_iter in enumerate(level_select):
			# each level save path
			model_specific = model_specific_list[model_iter]
			activation_layer = activation_dict[model_specific]
			feature_path_specific = feature_path + model_specific + '/' + activation_layer[level_iter] + '/'

			feature_name = feature_path_specific + file_name.replace('.wav','.npy')
			tmp = np.load(feature_name)

			if (i == 0 and model_iter == 0):
				all_feature = tmp
			else:
				all_feature = np.append(all_feature,tmp)
	return all_feature


def main():

	# load label
	test_list,eval_list_to_tagid,index_to_tagid,id_to_name,tagid_to_index,evaluation_list = load_label()

	# read models folder
	dnn_model_path = '/home/richter/dcase2017task4_multi_dnn/models/'
	dnn_model_list = [f.replace('.json','').replace('.hdf5','').replace('weights','').replace('architecture','') for f in os.listdir(dnn_model_path) if not f.startswith('.')]

	dnn_model_list = list(set(dnn_model_list))
	print dnn_model_list

	# dnn model loop
	for dnn_model_iter in range(len(dnn_model_list)):
		model_list = re.findall(r"\[([A-Za-z0-9_]+)\]",dnn_model_list[dnn_model_iter])
		model_list = [re.findall(r"\d+",x)[0] for x in model_list]
		print model_list
		indexes = re.findall(r"[^[]*\[([^]]*)\]",dnn_model_list[dnn_model_iter])
		print indexes

		model_dense = [int(x) for x in re.findall(r"\d+",indexes[0])]
		level_select = [int(x) for x in re.findall(r"(-?\d+)",indexes[1])]
		print model_dense,level_select

		num_models = len(model_list)
		num_levels = len(level_select)

		# for loading data
		model_specific_list = []
		activation_dict = {}

		for model_iter in range(num_models):
			model_name = model_list[model_iter]

			architecture_name = 'architecture_input' + model_name + '.json'
			weight_name = 'weights_input' + model_name + '.hdf5'

			json_file = open(model_path + architecture_name,'r')
			loaded_json = json_file.read()
			json_file.close()
			model = model_from_json(loaded_json)

			layer_dict = dict([(layer.name,layer) for layer in model.layers[1:]])
			layer_num = (len(layer_dict)-1)/4

			end_layer = layer_num

			model_specific = 'input' + model_name
			model_specific_list.append(model_specific)

			activation_tmp_list = []
			for iter in range(end_layer-2, end_layer+1):
				prints = 'activation_%d' % iter
				print prints
				activation_tmp_list.append(prints)
			activation_dict[model_specific] = activation_tmp_list

		print model_specific_list
		
		# loading 1 sample for measure feature_length(concatenated features)
		tmp_feature = load_multi_features(test_list[0],num_models,num_levels,model_specific_list,activation_dict,level_select)
		feature_length = len(tmp_feature)

		# load test sets
		test_size = len(test_list)
		x_test = np.zeros((test_size,feature_length))
		y_test = np.zeros((test_size,17))

		for iter in range(test_size):
			x_test[iter] = load_multi_features(test_list[iter],num_models,num_levels,model_specific_list,activation_dict,level_select)
			tagindex_list = [tagid_to_index[x] for x in eval_list_to_tagid[test_list[iter]]]
			for iter2 in range(len(tagindex_list)):
				y_test[iter][tagindex_list[iter2]] = 1
		print iter+1



		# load evaluation sets
		evaluation_size = len(evaluation_list)
		x_evaluation = np.zeros((evaluation_size,feature_length))

		for iter in range(evaluation_size):
			x_evaluation[iter] = load_multi_features(evaluation_list[iter],num_models,num_levels,model_specific_list,activation_dict,level_select)
		print iter+1



		# load dnn model
		dnn_weight_name = dnn_model_path + 'weights' + dnn_model_list[dnn_model_iter] + '.hdf5'
		dnn_architecture_name = dnn_model_path + 'architecture' + dnn_model_list[dnn_model_iter] + '.json'

		json_file = open(dnn_architecture_name,'r')
		loaded_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_json)
		model.load_weights(dnn_weight_name)
		print 'DNN model loaded'
			
		# compile & optimizer
		sgd = SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)
		model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
		model.summary()

		# prediction!!!
		predx_test = model.predict(x_test)

		prediction_save_dir = './prediction/'
		output_save_dir = './output/'
		groundtruth_save_dir = './groundtruth/'

		threshold = [0.1,0.2,0.3]	
		org_predx_test = predx_test

		for iter in range(len(threshold)):

			# threshold
			threshold_this = threshold[iter]
			threshold_prints = '[threshold:%f]' % threshold_this
			indicator_tmp = dnn_model_list[dnn_model_iter] + threshold_prints

			predx_test = np.copy(org_predx_test)
			predx_test[org_predx_test >= threshold_this] = 1
			predx_test[org_predx_test < threshold_this] = 0

			predicted_filepath = prediction_save_dir + indicator_tmp + '.csv'
			groundtruth_filepath = groundtruth_save_dir + 'groundtruth_weak_label_testing_set.csv'

			# matToCsv
			matToCsv(predx_test,test_list,predicted_filepath,index_to_tagid,id_to_name)
	
			# evaluation
			result = evaluateMetrics(groundtruth_filepath, predicted_filepath)

			#print result
			indices = [m.start() for m in re.finditer('F1 Score', result)]
			f_score = result[indices[-1]:indices[-1]+17]

			output_filepath = output_save_dir + indicator_tmp + '[' + f_score + ']'
			print f_score

			with open(output_filepath, "w") as filepath:
				filepath.write(result)
			filepath.close()

		# prediction!!! evaluation!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		predx_eval = model.predict(x_evaluation)

		prediction_save_dir = './prediction_evaluation_tagging/'
		output_save_dir = './output/'
		groundtruth_save_dir = './groundtruth/'

		threshold = [0.1,0.2,0.3]	
		org_predx_eval = predx_eval

		for iter in range(len(threshold)):

			# threshold
			threshold_this = threshold[iter]
			threshold_prints = '[threshold:%f]' % threshold_this
			indicator_tmp = '[evaluation]' + dnn_model_list[dnn_model_iter] + threshold_prints

			predx_eval = np.copy(org_predx_eval)
			predx_eval[org_predx_eval >= threshold_this] = 1
			predx_eval[org_predx_eval < threshold_this] = 0

			predicted_filepath = prediction_save_dir + indicator_tmp + '.csv'
			groundtruth_filepath = groundtruth_save_dir + 'groundtruth_weak_label_testing_set.csv'

			# matToCsv
			matToCsv(predx_eval,evaluation_list,predicted_filepath,index_to_tagid,id_to_name)
	
	


if __name__ == "__main__":


	main()


