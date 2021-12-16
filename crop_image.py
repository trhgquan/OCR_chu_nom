import os
from tqdm import tqdm
from PIL import Image
import json
import pandas as pd
import numpy as np
import pickle

train_path = 'train_1.csv'
train_images = 'data_real/img/'
crop_dir = 'crop_letter/'

def get_annotation_list_train():
    df_train = pd.read_csv(train_path)

    df_train = df_train.dropna(axis = 0, how = 'any')
    df_train = df_train.reset_index(drop = True)

    # Creating dictionary of characters
    annotation_list_train = []
    category_names = set()

    print('Creating dictionary:')
    for i in tqdm(range(len(df_train))):
        ann = np.array(df_train.loc[i, 'labels'][:-1].split(' ')).reshape(-1, 5)
        category_names = category_names.union({i for i in ann[:, 0]})

    category_names = sorted(category_names)
    dict_cat = {list(category_names)[j] : str(j) for j in range(len(category_names))}
    inv_dict_cat = {str(j) : list(category_names)[j] for j in range(len(category_names))}

    print('Extracting positions:')
    for i in tqdm(range(len(df_train))):
        ann = np.array(df_train.loc[i, 'labels'][:-1].split(' ')).reshape(-1, 5)

        for j, category_name in enumerate(ann[:, 0]):
            ann[j, 0] = int(dict_cat[category_name])
            ann[j, 1] = int(float(ann[j, 1]))
            ann[j, 2] = int(float(ann[j, 2]))
            ann[j, 3] = int(float(ann[j, 3]))
            ann[j, 4] = int(float(ann[j, 4]))

        ann = ann.astype('int32')

        ann[:, 3] -= ann[:, 1]
        ann[:, 4] -= ann[:, 2]

        ann[:, 1] += ann[:, 3] // 2
        ann[:, 2] += ann[:, 4] // 2

        # Finding the center of ann

        annotation_list_train.append(['{}{}'.format(
            train_images,
            df_train.loc[i, 'image_id']
        ), ann])

    return annotation_list_train, dict_cat, inv_dict_cat

def create_train_input():
    if os.path.exists(crop_dir) == False: os.mkdir(crop_dir)

    train_input = []

    pic_count, count, failed_images = 0, 0, 0

    for ann_pic in tqdm(annotation_list_train):
        pic_count += 1
        with Image.open(ann_pic[0]) as img:
            for ann in ann_pic[1]:
                cat, cx, cy, height, width = ann[0], ann[1], ann[2], ann[3], ann[4]
                save_dir = '{}.jpg'.format(str(count))

                save_dir = crop_dir + save_dir

                try:
                    img.crop((int(cx - width / 2), int(cy - height / 2), int(cx + width / 2), int(cy + height / 2))).save(save_dir)
                    train_input.append([save_dir, cat])
                    count += 1
                except:
                    failed_images += 1

    return train_input


annotation_list_train, dict_cat, inv_dict_cat = get_annotation_list_train()

print('Total train images:', len(annotation_list_train))

train_input = create_train_input()

with open('train_input.pkl', 'wb') as f:
    pickle.dump(train_input, f)

# with open('train_input.pkl', 'rb') as f:
#     train_input = pickle.load(f)

# print(len(train_input))

# print(train_input[0])
# print(inv_dict_cat[str(train_input[0][1])])