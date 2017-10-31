import os
import numpy as np
import cPickle as cP
import librosa
import re
import csv

from keras.models import model_from_json
from keras.optimizers import SGD
from keras import backend as K
from sklearn.metrics import precision_recall_fscore_support

import sys

feature_path = '/media/ssd2/dcase2017_task4/wav_to_npy/'
label_path = '/media/ssd2/dcase2017_task4/processed_label/'

# load data
test_list = cP.load(open(label_path + 'testing_set_list.cP','r'))
evaluation_list = cP.load(open(label_path + 'evaluation_set_list.cP','r'))

tagid_to_index = cP.load(open(label_path + 'tagid_to_index.cP','r'))
index_to_id = dict(zip(tagid_to_index.values(),tagid_to_index.keys()))
id_to_name = cP.load(open(label_path + 'tagid_to_name.cP','r'))

#test_list = evaluation_list

print len(test_list),len(evaluation_list)

def calculate_num_segment(hop_size,num_frame_input):
	return int((int(441000/hop_size))/num_frame_input/2)+1

def calculate_sample_length(hop_size,num_frame_input):
	return hop_size*(num_frame_input)*2

def gen_path(file_name):
	dir_name = feature_path + file_name.replace('.wav','.npy')
	return dir_name

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


model_name = '6561frames_power3_input893_ETE'
num_frame_input = 6561
hop_size = 3
conv_window_size = 3
lr = 0.000016
test_size = len(test_list)
evaluation_size = 1103
num_tags = 17

eval_list_to_tagid = cP.load(open(label_path + 'testing_set_ytid_to_tagid.cP','r'))
tagid_to_index = cP.load(open(label_path + 'tagid_to_index.cP','r'))

# load model
sample_length = calculate_sample_length(hop_size,num_frame_input)
architecture_name = 'model_architecture_%s_%d_%d_%.6f.json' % (model_name,conv_window_size,hop_size,lr)
json_file = open(architecture_name,'r')
loaded_json = json_file.read()
json_file.close()
model = model_from_json(loaded_json)

load_weight = 'best_weights_%s_%d_%d_%.6f.hdf5' % (model_name,conv_window_size,hop_size,lr)
model.load_weights(load_weight)
print 'model loaded!!!!'

model.summary()

num_segment = calculate_num_segment(hop_size,num_frame_input)
print 'Number of segments per song:' + str(num_segment)

# load test set
x_test_tmp = np.zeros((num_segment,sample_length,1))
predx_test = np.zeros((test_size,num_tags))
y_test = np.zeros((test_size,num_tags))
test_split = 1

for iter2 in range(0,test_split):
	for iter in range(iter2*int(test_size/test_split),(iter2+1)*int(test_size/test_split)):
		file_path = gen_path(test_list[iter])
		tmp = np.load(file_path)
		y_length = len(tmp)

		for iter3 in range(0,num_segment):
			hopping = (y_length-sample_length)/(num_segment-1)
			if hopping < 0:
				tmp = np.repeat(tmp,10)
				y_length = len(tmp)
				hopping = (y_length - sample_length)/(num_segment-1)
				x_test_tmp[iter3,:,0] = tmp[iter3*hopping:iter3*hopping+sample_length]
			else:
				x_test_tmp[iter3,:,0] = tmp[iter3*hopping:iter3*hopping+sample_length]


		
		tagindex_list = [tagid_to_index[x] for x in eval_list_to_tagid[test_list[iter]]]
		
		for iter4 in range(len(tagindex_list)):
			y_test[iter][tagindex_list[iter4]] = 1
		

		# prediction each segments & Average them
		predx_test[iter] = np.mean(model.predict(x_test_tmp),axis=0)
		print iter

org_predx_test = predx_test

threshold = [0.05,0.1,0.2,0.3,0.4,0.5]
for iter in range(len(threshold)):
	threshold_this = threshold[iter]

	predx_test = np.copy(org_predx_test)
	predx_test[org_predx_test >= threshold_this] = 1
	predx_test[org_predx_test < threshold_this] = 0

	
	precision,recall,f1,support = precision_recall_fscore_support(y_test,predx_test,average='macro')
	precision_mic,recall_mic,f1_mic,support_mic = precision_recall_fscore_support(y_test,predx_test,average='micro')
	


	save_dir = './predictAtest/'
	#save_dir = './predictAevaluation/'
	


	f_prints = '[f1mac:%.4f,Precmac:%.4f,Remac:%.4f][f1mic:%.4f,Precmic:%.4f,Remic:%.4f]' % (f1,precision,recall,f1_mic,precision_mic,recall_mic)

	
	save_name = model_name + '_' + 'threshold:' + str(threshold_this) + '_' + f_prints
	
	
	save_list = []
	cP.dump(save_list,open(save_dir+save_name,'w'))
	
	# matToCsv
	predicted_filepath = save_dir + save_name + '.csv'
	matToCsv(predx_test,test_list,predicted_filepath,index_to_id,id_to_name)






