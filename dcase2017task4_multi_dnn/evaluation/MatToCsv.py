### by jypark

import os
import sys
import csv
import numpy as np
import cPickle as cP

def matToCsv(matrix):
	
	label_path = '/media/ssd2/dcase2017_task4/processed_label/'
	id_to_index = cP.load(open(label_path + 'tagid_to_index.cP','r'))
	id_to_name = cP.load(open(label_path + 'tagid_to_name.cP','r'))

	test_list = cP.load(open(label_path + 'testing_set_list.cP','r'))

	index_to_id = {}
	for tagid in id_to_index:
		for i in range(0,17):
			if id_to_index[tagid] == i:
				index_to_id[i] = tagid

	
	prediction = []
	for i in range(0, matrix.shape[0]):
		row_predict = []
		row_event = matrix[i]
		nonzero = np.nonzero(row_event)
		nonzero = np.array(nonzero).reshape(-1,)
		if not nonzero.size:
			name = test_list[i]
			onset = name.split('_')[1]
			offset = name.split('_')[2]
			offset = offset.split('.wav')[0]
			row_predict.append((name, onset, offset))
			prediction.append(row_predict)
		else:	
			for j in range(0,nonzero.size):
				name = test_list[i]
				onset = name.split('_')[1]
				offset = name.split('_')[2]
				offset = offset.split('.wav')[0]
				label = id_to_name[index_to_id[nonzero[j]]]
				row_predict.append((name, onset, offset, label))
			prediction.append(row_predict)


	with open("./prediction/check_subtaskA.csv", 'wb') as f:
		writer = csv.writer(f, delimiter='\t')
		for row in prediction:
			for item in row:
				writer.writerow(item)

	









