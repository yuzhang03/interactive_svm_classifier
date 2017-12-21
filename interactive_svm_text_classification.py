
# Interactive SVM Text Classification
# By: Yu Zhang, 12/07/2017

import time
import pandas
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

train_data_set = 'train_data_set.csv'
test_data_set = 'test_data_set.csv'
stop_list = 'stop_word_list.csv'
interactive_output = 'interactive_output.csv'

col_names = ['content', 'label', 'proba']
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
train_input = list(zip(train_content[:10], train_label[:10]))
left_train = list(zip(train_content[10:], train_label[10:]))
iter_num = 10

with open(interactive_output, "wb") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for line in []:
        writer.writerow(line)

start = time.clock()

while 1:

    train_content_input, train_label_input = zip(*train_input)
    train_content_input_list = list(train_content_input)
    train_label_input_list = list(train_label_input)
    train_content_input_matrix = count_vect.fit_transform(train_content_input_list)
    test_content_matrix = count_vect.transform(test_content)
    clf.fit(train_content_input_matrix, train_label_input_list)

    predicted_svm = clf.predict(test_content_matrix)
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
    print(iter_num, precision, recall, f_score)
    with open(interactive_output, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([iter_num, precision, recall, f_score])

    left_content, left_label = zip(*left_train)
    left_content_list = list(left_content)
    left_label_list = list(left_label)
    left_content_matrix = count_vect.transform(left_content_list)
    left_proba = clf.predict_proba(left_content_matrix)
    left_proba_list = list()
    for line in left_proba:
        left_proba_list.append(line[0])
    three_column = list(zip(left_content_list, left_label_list, left_proba_list))
    min_proba = min(three_column, key=lambda x: abs(x[2]-0.5))
    train_input.append(min_proba)
    three_column.remove(min_proba)
    if three_column:
        left_content, left_label, useless_proba = zip(*three_column)
        left_content_list = list(left_content)
        left_label_list = list(left_label)
        left_train = list(zip(left_content, left_label))
    else:
        break
    iter_num = iter_num + 1

end = time.clock()
print(str(end-start))