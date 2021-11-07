import os
import pandas as pd

DATA_DIR = 'data_real'

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

with open('train.csv', 'w+', encoding = 'utf-8') as fw:
    print('image_id,labels', file = fw)

    for filename, label_in_file in labels.items():
        print(filename, file = fw, end = ',')
        print('Doing', filename)
        for label in label_in_file:
            splitted = label.split(' ')

            if hex(ord(splitted[0])) not in unicode_translate:
                unicode_translate[hex(ord(splitted[0]))] = splitted[0]

            splitted[0] = hex(ord(splitted[0]))
            splitted[1] = int(float(splitted[1]))
            splitted[2] = int(float(splitted[2]))
            splitted[3] = int(float(splitted[3]))
            for s in splitted:
                print(s, file = fw, end = ' ')
        print(file = fw)

with open('unicode_translate.csv', 'w+', encoding = 'utf-8') as fw:
    print('unicode,value', file = fw)
    for key, value in unicode_translate.items():
        print(key, value, sep = ',', file = fw)
