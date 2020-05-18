import csv
import json
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import random
from random import shuffle

# list_of_data_aggreed
lag = []
# list of data
ld = []
row_count = 0
# get our data
with open('Data_Brexit.csv', encoding='utf-8') as csv_file:
    csvReader = csv.DictReader(csv_file);
    ld = [(row['tweet'], row['opinion']) for row in csvReader if row['golden'] == 'FALSE' and row['relevant2brexit'] == 'yes']

# getting unique data where the aggregate > 2
for i in range(0, len(ld), 4):
    # print(ld[i])
    counter = 0
    if ld[i][1] == ld[i+1][1]:
        counter += 1
    if ld[i][1] == ld[i+2][1]:
        counter += 1
    if ld[i][1] == ld[i+3][1]:
        counter +=1
    if counter > 1:
        lag.append(ld[i])
print(len(lag))
# shuffle the list
# shuffle(lag)
# dimiourgia leksikou
pos_words = []
neu_words = []
neg_words = []
ironic_words = []

# get the words for each category and create also the dictionary (bag of words)
for d in lag:
    if d[1] == '1':
        pos_words += word_tokenize(d[0])
    elif d[1] == '0':
        neu_words += word_tokenize(d[0])
    elif d[1] == '-1':
        neg_words += word_tokenize(d[0])
    elif d[1] == '2':
        ironic_words += word_tokenize(d[0])

# removing any numbers
pos_words = [w for w in pos_words if w.isalpha()]
neu_words = [w for w in neu_words if w.isalpha()]
neg_words = [w for w in neg_words if w.isalpha()]
ironic_words = [w for w in ironic_words if w.isalpha()]

# removing any word smaller than 2 chars
pos_words = [w for w in pos_words if len(w) > 2]
neu_words = [w for w in neu_words if len(w) > 2]
neg_words = [w for w in neg_words if len(w) > 2]
ironic_words = [w for w in ironic_words if len(w) > 2]

# removing stopwords
stopwords = stopwords.words('english')
pos_words = [w for w in pos_words if w not in stopwords]
neg_words = [w for w in neg_words if w not in stopwords]
neg_words = [w for w in neg_words if w not in stopwords]
ironic_words = [w for w in ironic_words if w not in stopwords]

# get the words unique only for each category
neg_only = [w for w in neg_words if w not in pos_words and w not in neu_words and w not in ironic_words]
pos_only = [w for w in pos_words if w not in neg_words and w not in neu_words and w not in ironic_words]
neu_only = [w for w in neu_words if w not in neg_words and w not in pos_words and w not in ironic_words]
ironic_only = [w for w in ironic_words if w not in neg_words and w not in pos_words and w not in neu_words] # den eminan ironic opotan that is xrisimopoiisoume oles oses eine opws eine
print(len(neg_only))
print(len(pos_only))
print(len(neu_only))
print(len(ironic_only))
index = sorted(neg_only + pos_only + ironic_only + neu_only)

def document_features(document, index):
	document_words = set(word_tokenize(document))
	features = {}
	for word in index:
		features['%s' % word] = (word in document_words)
	return features

featuressets = [(document_features(d, index), c) for (d, c) in lag]
print(len(featuressets))

train_set = featuressets[:300]
test_set = featuressets[301:]

classifier = nltk.NaiveBayesClassifier.train(train_set);
# DTclassifier = nltk.classify.DecisionTreeClassifier.train(train_set)
#
classifier.show_most_informative_features()

print(classifier.prob_classify(test_set[2][0]).prob('0'))
# print(DTclassifier.prob_classify(test_set[2][0]))
print(test_set[2][1])
