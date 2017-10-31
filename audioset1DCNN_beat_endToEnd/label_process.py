import numpy as np
import cPickle as cP
import csv
import os
import re

label_path = '/media/ssd2/dcase2017_task4/'
label_save_path = '/media/ssd2/dcase2017_task4/processed_label/'

file_name = 'training_set.csv'
indices = [s.start() for s in re.finditer('_',file_name)]  

save_name = file_name.replace('.csv','') #file_name[indices[2]+1:indices[3]]
print save_name

with open(label_path + file_name,'rb') as csvfile:
	reader = csv.reader(csvfile)
	data = [[x.strip().split(',') for x in row] for row in reader]

# class_lables_indices read
ytid = [x[0][0]+'_'+x[1][0]+'_'+x[2][0]+'.wav' for x in data]
print len(ytid)
cP.dump(ytid,file(label_save_path+save_name+'_list.cP','w'))

# gen multi labeled label file
with open(label_path + 'sound_event_list_17_classes.txt') as f:
	file_list = f.read().splitlines()
	file_list = file_list[1:]
	file_list = [x.split('\t') for x in file_list]

#print file_list
tagid_to_name = {}
tagid_to_index = {}
for iter in range(len(file_list)):
	tagid_to_index[file_list[iter][0]] = iter
	tagid_to_name[file_list[iter][0]] = file_list[iter][2]
# [label_id, confidence, label_name, label_index]	
#print file_list
print tagid_to_index
print tagid_to_name

cP.dump(tagid_to_index,file(label_save_path+'tagid_to_index.cP','w'))
cP.dump(tagid_to_name,file(label_save_path+'tagid_to_name.cP','w'))

ytid_to_tagid = {}
for iter in range(len(data)):
	ytid_to_tagid[data[iter][0][0]+'_'+data[iter][1][0]+'_'+data[iter][2][0]+'.wav'] = data[iter][4]

cP.dump(ytid_to_tagid,file(label_save_path+save_name+'_ytid_to_tagid.cP','w'))



'''
with open(label_path + 'sound_event_list_17_classes.txt','rb') as csvfile:
	reader = csv.reader(csvfile)
	data_label = [[x.strip().replace('"','').replace("'",'') for x in row] for row in reader]

	data_label = data_label[1:]
	print(data_label[0:3])

	tag_index = [int(x[0]) for x in data_label]
	tag_id = [x[1] for x in data_label]
	tag_name = [x[2] for x in data_label]

	print(tag_index[0:3])
	print(tag_id[0:3])
	print(tag_name[0:3])
	print(len(tag_name))


tagid_to_tagindex = dict(zip(tag_id,tag_index))
tagindex_to_tagname = dict(zip(tag_index,tag_name))

ytid = [x[0] for x in data]
tagindex = [[tagid_to_tagindex[y] for y in x[3:]] for x in data]
print len(ytid),len(tagindex)

list_to_tagid = dict(zip(ytid,tagindex))

cP.dump(tagid_to_tagindex,file(label_save_path+'tagid_to_tagindex.cP','w'))
cP.dump(tagindex_to_tagname,file(label_save_path+'tagindex_to_tagname.cP','w'))
cP.dump(list_to_tagid,file(label_save_path+'_list_to_tagid.cP','w'))
'''










































