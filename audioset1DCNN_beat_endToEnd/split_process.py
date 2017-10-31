import numpy as np
import cPickle as cP
import os
import re
import itertools
import random

# for reproduction
np.random.seed(0)

label_path = '/media/ssd2/dcase2017_task4/'
label_save_path = '/media/ssd2/dcase2017_task4/processed_label/'

ytid = cP.load(open(label_save_path+'training_set_list.cP','r'))
ytid_to_tagid = cP.load(open(label_save_path+'training_set_ytid_to_tagid.cP','r'))
tagid_to_name = cP.load(open(label_save_path+'tagid_to_name.cP','r'))

# sample nums for each class
name_to_sample_num = {}
name_to_sample_num['Train horn'] = 441
name_to_sample_num['Air horn, truck horn'] = 407
name_to_sample_num['Car alarm'] = 273
name_to_sample_num['Reversing beeps'] = 337
name_to_sample_num['Ambulance (siren)'] = 624
name_to_sample_num['Police car (siren)'] = 2399
name_to_sample_num['Fire engine, fire truck (siren)'] = 2399
name_to_sample_num['Civil defense siren'] = 1506
name_to_sample_num['Screaming'] = 744
name_to_sample_num['Bicycle'] = 2020
name_to_sample_num['Skateboard'] = 1617
name_to_sample_num['Car'] = 25744
name_to_sample_num['Car passing by'] = 3724
name_to_sample_num['Bus'] = 3745
name_to_sample_num['Truck'] = 7090
name_to_sample_num['Motorcycle'] = 3291
name_to_sample_num['Train'] = 2301

tagid = tagid_to_name.keys()

tagid_to_sample_num = {}
for iter in range(len(tagid)):
	tagid_to_sample_num[tagid[iter]] = name_to_sample_num[tagid_to_name[tagid[iter]]]/10

# shuffling
random.shuffle(ytid)

tagid_to_ytid = list(zip(ytid_to_tagid.values(),ytid_to_tagid.keys()))
print tagid_to_ytid

# take valid set
valid_ytid = []
for tag_iter in range(len(tagid)):

	sample_num_for_thetag = tagid_to_sample_num[tagid[tag_iter]]

	count = 0
	for ytid_iter in range(len(ytid)):
		L1 = ytid_to_tagid[ytid[ytid_iter]]
		L2 = [tagid[tag_iter]]
		L3 = [ytid[ytid_iter]]
		L4 = valid_ytid

		if any(i for i in L1 if i in L2)  == True:
			# if already exists in valid set, find another one for given class
			if any(i for i in L3 if i in L4) == True:
				continue
			valid_ytid.append(ytid[ytid_iter])	
			count += 1

		print sample_num_for_thetag,count,len(valid_ytid)

		# if valid set is saturated, go to next tag
		if count == sample_num_for_thetag:
			break

print valid_ytid
print len(valid_ytid)

train_ytid = [x for x in ytid if x not in valid_ytid]
print len(train_ytid)

cP.dump(valid_ytid,file(label_save_path+'training_set_valid_list.cP','w'))
cP.dump(train_ytid,file(label_save_path+'training_set_train_list.cP','w'))


