
# Interactive SVM Text Classification
# By: Yu Zhang, 12/03/2017

import time
import pandas
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

train_data_set = 'train_data_set.csv'
test_data_set = 'test_data_set.csv'
stop_list = 'stop_word_list.csv'
non_interactive_output = 'non_interactive_output.csv'

col_names = ['content', 'label']
train_data = pandas.read_csv(train_data_set, names=col_names)
train_content = train_data.content.tolist()
train_label = train_data.label.tolist()

test_data = pandas.read_csv(test_data_set, names=col_names)
test_content = test_data.content.tolist()
test_label = test_data.label.tolist()

word = ['word']
stop_words = pandas.read_csv(stop_list, names=word)
stop_words_list = stop_words.word.tolist()
count_vect = CountVectorizer(stop_words=frozenset(stop_words_list))
clf = svm.SVC(kernel='linear', probability=True)
train_content_count = count_vect.fit_transform(train_content)
test_content_count = count_vect.transform(test_content)

with open(non_interactive_output, "wb") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for line in []:
        writer.writerow(line)

start = time.clock()

for i in range(10, 1330):
    clf.fit(train_content_count[:i], train_label[:i])
    predicted_svm = clf.predict(test_content_count)
    cancer_num = 0
    cancer_num_tp = 0
    for item in predicted_svm:
        if item == 'cancer':
            cancer_num += 1
    for item in predicted_svm[:100]:
        if item == 'cancer':
            cancer_num_tp += 1
    precision = float("{0:.2f}".format(cancer_num_tp / cancer_num))
    recall = cancer_num_tp / 100
    f_score_long = 2 * precision * recall / (precision + recall)
    f_score = float("{0:.2f}".format(f_score_long))
    print(i, precision, recall, f_score)
    with open(non_interactive_output, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([i, precision, recall, f_score])

end = time.clock()
print(str(end-start))
