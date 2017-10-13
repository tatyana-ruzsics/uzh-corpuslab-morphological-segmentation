#! /usr/bin/env python

import sys, codecs

systemfile = sys.argv[1]
reffile = sys.argv[2]

total = 0
correct = 0
i = 0
for system, ref in zip(codecs.open(systemfile, 'r', encoding='utf-8'), codecs.open(reffile, 'r', encoding='utf-8')):
	if system.strip() == ref.strip():
		correct += 1
	else:
		print(str(i)+"\t"+system.strip().encode('utf8')+"\t"+ref.strip().encode('utf8'))
	total += 1
	i+=1

#print "Total items:", total
#print "Correct items:", correct
#print "Accuracy:", 100*float(correct)/float(total), "%"

#file_results = open(sys.argv[3], 'w')
with open(sys.argv[3], 'a') as f:
    f.write("Total: "+str(total)+"\n")
    f.write("Correct items: "+str(correct)+"\n")
    f.write("Accuracy: "+str(100*float(correct)/float(total))+"%"+"\n")
#print("Total: "+total, file=file_results)
#print("Correct items: "+correct, file=file_results)
#print("Accuracy: "+100*float(correct)/float(total), file=file_results)

#print "Correct items:", correct
#print "Accuracy:", 100*float(correct)/float(total), "%"
