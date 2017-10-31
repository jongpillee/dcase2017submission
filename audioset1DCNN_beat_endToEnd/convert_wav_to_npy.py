import numpy as np
import librosa
import os.path
import cPickle as cP
import sys
import glob

fs = 44100
nth = 0
total_nth = 1
max_length = 44100*10
mono = False

label_path = '/media/ssd2/dcase2017_task4/processed_label/'
feature_path = '/media/ssd2/dcase2017_task4/wav_to_npy/'
#data_path = '/media/ssd2/dcase2017_task4/unbalanced_train_segments_testing_set_audio_formatted_and_segmented_downloads/' # or train_list
#data_path = '/media/ssd2/dcase2017_task4/training_set_training_set_audio_formatted_and_segmented_downloads/'
data_path = '/media/ssd2/dcase2017_task4/evaluation_set_formatted_audio_segments/'

# load train valid test list
'''
train_list = cP.load(open(label_path + 'training_set_list.cP','r'))
test_list = cP.load(open(label_path + 'testing_set_list.cP','r'))
'''
train_list = cP.load(open(label_path + 'evaluation_set_list.cP','r'))

# merge all list
all_list = train_list # test_list or train_list
length_id = len(all_list)
print length_id

cantread = []
def main(nth):
	print int(nth*length_id/total_nth),int((nth+1)*length_id/total_nth)

	for iter in range(int(nth*length_id/total_nth),int((nth+1)*length_id/total_nth)):
		save_name = feature_path + all_list[iter].replace('.wav','.npy')
		#file_name = data_path + 'Y'+all_list[iter]
		file_name = data_path + all_list[iter]

		if not os.path.exists(os.path.dirname(save_name)):
			os.makedirs(os.path.dirname(save_name))

		if os.path.isfile(save_name) == 1:
			print iter, save_name + '_file_exist!!!!!!!!'
			continue

		'''
		try:
		'''
		print file_name

		y,sr = librosa.load(file_name,sr=fs,mono=mono)
		#y,sr = librosa.load(data_path+'Y_TLzbbay6Hw_0.000_2.000.wav',sr=fs,mono=mono)
		y = y.astype(np.float32)

		#y = y[0:44100*2]

		print iter,y.shape,save_name
		np.save(save_name,y)
		'''
		except Exception:
			cantread.append(save_name)		
	print len(cantread)
	cP.dump(cantread,file('cannot_read.cP','r'))
		'''

if __name__ == '__main__':

	main(nth)
