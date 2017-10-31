import csv

test = []
with open('./test.csv','rb') as csvfile:
	reader = csv.reader(csvfile,delimiter='\t')
	for row in reader:
		test.append(row[0])

target = []
with open('./target.csv','rb') as csvfile:
	reader = csv.reader(csvfile,delimiter='\t')
	for row in reader:
		target.append(row[0])

org_test = list(set(test))
org_target = list(set(target))

print len(org_test),len(org_target)

missing_list = list(set(test)-set(target))
print missing_list
print len(missing_list)

with open ('./target.csv','a') as f:
	writer = csv.writer(f,delimiter='\n')
	writer.writerow(missing_list)

