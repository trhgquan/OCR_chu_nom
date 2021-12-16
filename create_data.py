import os
import random

DATA_DIR = 'data_real'
train_csv = 'train_1.csv'
test_csv = 'test_1.csv'

# train.csv would consists of files inside MTH1000 and TKH
# test.csv would consists of files inside MTH1200

with open('filename.txt', 'w+') as f:
    labels = dict()

for file in os.listdir(os.path.join(DATA_DIR, 'img')):
    filename = file.split('.')[0]
    labels[file] = []

    label_path = os.path.join(DATA_DIR, 'label_char\\') + filename + '.txt'

    with open(label_path, 'r+', encoding = 'utf-8') as fo:
        for line in fo:
            labels[file].append(line.rstrip())

unicode_translate = dict()

# Get random images to put to train.csv
train_picked = []
test_picked = []

limit_test = 1000

for i in range(0, len(labels)):
    if len(test_picked) < limit_test:
        chosen_one = random.randint(0, 1)

        if chosen_one == 1:
            test_picked.append(i)

with open(train_csv, 'w+', encoding = 'utf-8') as fw:
    print('image_id,labels', file = fw)


with open(test_csv, 'w+', encoding = 'utf-8') as fw:
    print('image_id,labels', file = fw)

with open(train_csv, 'w+', encoding = 'utf-8') as train_f:
    print('image_id,labels', file = train_f)
    with open(test_csv, 'w+', encoding = 'utf-8') as test_f:
        print('image_id,labels', file = test_f)
        for index, (filename, label_in_file) in enumerate(labels.items()):
            if index in test_picked:
                print(filename, file = test_f, end = ',')
                print('Printing', filename, 'to test.csv')
                for label in label_in_file:
                    splitted = label.split(' ')

                    if hex(ord(splitted[0])) not in unicode_translate:
                        unicode_translate[hex(ord(splitted[0]))] = splitted[0]

                    splitted[0] = hex(ord(splitted[0]))
                    splitted[1] = int(float(splitted[1]))
                    splitted[2] = int(float(splitted[2]))
                    splitted[3] = int(float(splitted[3]))
                    for s in splitted:
                        print(s, file = test_f, end = ' ')
                print(file = test_f)
            else:
                print(filename, file = train_f, end = ',')
                print('Printing', filename, 'to train.csv')
                for label in label_in_file:
                    splitted = label.split(' ')

                    if hex(ord(splitted[0])) not in unicode_translate:
                        unicode_translate[hex(ord(splitted[0]))] = splitted[0]

                    splitted[0] = hex(ord(splitted[0]))
                    splitted[1] = int(float(splitted[1]))
                    splitted[2] = int(float(splitted[2]))
                    splitted[3] = int(float(splitted[3]))
                    for s in splitted:
                        print(s, file = train_f, end = ' ')
                print(file = train_f)

with open('unicode_translate.csv', 'w+', encoding = 'utf-8') as fw:
    print('unicode,value', file = fw)
    for key, value in unicode_translate.items():
        print(key, value, sep = ',', file = fw)
