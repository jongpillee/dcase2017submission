import nose.tools
import sed_eval
import os

label = []
label.append('Train horn')
label.append('Air horn, truck horn')
label.append('Car alarm')
label.append('Reversing beeps')
label.append('Ambulance (siren)')
label.append('Police car (siren)')
label.append('Fire engine, fire truck (siren)')
label.append('Civil defense siren')
label.append('Screaming')
label.append('Bicycle')
label.append('Skateboard')
label.append('Car')
label.append('Car passing by')
label.append('Bus')
label.append('Truck')
label.append('Motorcycle')
label.append('Train')

def test_dcase_style():

	data_path = '/home/richter/richter_chopin5_44452/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/evaluation/'
	
	reference = data_path + 'groundtruth/' + 'groundtruth_strong_label_testing_set.csv'
	estimated = data_path + 'predictiontest/' + 'hop:1sec_input893_threshold:0.5_AvgBeforeThreshold.csv'
	#estimated = data_path + 'prediction/' + 'hop:1sec_input893_threshold:0.5.csv'
	#estimated = '/home/richter/richter_chopin5_44452/dcase2017task4_multi_dnn/evaluation/prediction_sed/target.csv'

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
	print results['overall']['error_rate']['error_rate']
	print results['overall']['f_measure']['f_measure']
	print results

	for iter2 in range(len(label)):
		print label[iter2], results['class_wise'][label[iter2]]['error_rate']['error_rate']


if __name__ == '__main__':

	test_dcase_style()

