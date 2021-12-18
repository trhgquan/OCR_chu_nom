import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import cv2
import glob
import os

from matplotlib.axes import Axes
from PIL import Image, ImageDraw, ImageFont
from pandas.io.json import json_normalize
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Conv2D, Conv2DTranspose, BatchNormalization, Activation, AveragePooling2D, GlobalAveragePooling2D, Input, Concatenate, MaxPool2D, MaxPooling2D, Flatten, Add, UpSampling2D, LeakyReLU, ZeroPadding2D
from keras.models import Model, load_model
from sklearn.metrics import mean_squared_error
from keras import backend as K
from keras.losses import binary_crossentropy, mean_squared_error
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

class OCR_Engine:
    def __init__(self):
        self.__n_category = 1
        self.__output_layer_n = self.__n_category + 4

        self.__base_detect_num_h = 25
        self.__base_detect_num_w = 25

        self.__pred_in_h = 512
        self.__pred_in_w = 512

        self.__pred_out_h = int(self.__pred_in_h / 4)
        self.__pred_out_w = int(self.__pred_in_w / 4)

        self.__model_1 = self.create_model(
            input_shape = (512, 512, 3),
            size_detection_mode = True    
        )

        self.__model_1.load_weights('models/model_1/final_weights_step1.hdf5')

        self.__model_2 = self.create_model(
            input_shape = (512, 512, 3),
            size_detection_mode = False
        )

        self.__model_2.load_weights('models/model_2/final_weights_step2.h5')

        self.__model_3 = load_model('models/model_3/model_chu_nom.h5')

        self.__yy_ = np.load('models/model_3/yy_.npy')
        self.__lb = LabelEncoder()
        self.__y_integer = self.__lb.fit_transform(self.__yy_)
    
    @staticmethod
    def cbr(x, out_layer, kernel, stride):
        x = Conv2D(out_layer, kernel_size = kernel, strides = stride, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha = 0.1)(x)
        return x
    
    @staticmethod
    def aggregation_block(x_shallow, x_deep, deep_ch, out_ch):
        x_deep = Conv2DTranspose(
            deep_ch, 
            kernel_size = 2, strides = 2, 
            padding = 'same', use_bias = False)(x_deep)
        x_deep = BatchNormalization()(x_deep)
        x_deep = LeakyReLU(alpha = 0.1)(x_deep)
        x = Concatenate()([x_shallow, x_deep])
        x = Conv2D(out_ch, kernel_size = 1, strides = 1, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha = 0.1)(x)
        return x
    
    def create_model(self, input_shape, size_detection_mode = True, aggregation = True):
        input_layer = Input(input_shape)

        # Resized input
        input_layer_1 = AveragePooling2D(2)(input_layer)
        input_layer_2 = AveragePooling2D(2)(input_layer_1)

        # Encoder

        # Transform 512 to 256
        x0 = self.cbr(input_layer, 16, 3, 2)
        concat1 = Concatenate()([x0, input_layer_1])

        # Transform from 256 to 128
        x1 = self.cbr(concat1, 32, 3, 2)
        concat2 = Concatenate()([x1, input_layer_2])

        # Transfrom from 128 to 64
        x2 = self.cbr(concat2, 64, 3, 2)
        x = self.cbr(x2, 64, 3, 1)
        x = self.resblock(x, 64)
        x = self.resblock(x, 64)

        # Transform from 64 to 32
        x3 = self.cbr(x, 128, 3, 2)
        x = self.cbr(x3, 128, 3, 1)
        x = self.resblock(x, 128)
        x = self.resblock(x, 128)
        x = self.resblock(x, 128)

        # Transform from 32 to 16
        x4 = self.cbr(x, 256, 3, 2)
        x = self.cbr(x4, 256, 3, 1)
        x = self.resblock(x, 256)
        x = self.resblock(x, 256)
        x = self.resblock(x, 256)
        x = self.resblock(x, 256)
        x = self.resblock(x, 256)

        # Transform from 16 to 8
        x5 = self.cbr(x, 512, 3, 2)
        x = self.cbr(x5, 512, 3, 1)
        x = self.resblock(x, 512)
        x = self.resblock(x, 512)
        x = self.resblock(x, 512)

        if size_detection_mode:
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            out = Dense(1, activation = 'linear')(x)
        
        else:
            # Centernet mode
            x1 = self.cbr(x1, self.__output_layer_n, 1, 1)
            x1 = self.aggregation_block(x1, x2, self.__output_layer_n, self.__output_layer_n)
            
            x2 = self.cbr(x2, self.__output_layer_n, 1, 1)
            x2 = self.aggregation_block(x2, x3, self.__output_layer_n, self.__output_layer_n)
            x1 = self.aggregation_block(x1, x2, self.__output_layer_n, self.__output_layer_n)
            
            x3 = self.cbr(x3, self.__output_layer_n, 1, 1)
            x3 = self.aggregation_block(x3, x4, self.__output_layer_n, self.__output_layer_n)
            x2 = self.aggregation_block(x2, x3, self.__output_layer_n, self.__output_layer_n)
            x1 = self.aggregation_block(x1, x2, self.__output_layer_n, self.__output_layer_n)

            x4 = self.cbr(x4, self.__output_layer_n, 1, 1)
            x = self.cbr(x, self.__output_layer_n, 1, 1)
            x = UpSampling2D(size = (2, 2))(x)

            x = Concatenate()([x, x4])
            x = self.cbr(x, self.__output_layer_n, 3, 1)
            x = UpSampling2D(size = (2, 2))(x)

            x = Concatenate()([x, x3])
            x = self.cbr(x, self.__output_layer_n, 3, 1)
            x = UpSampling2D(size = (2, 2))(x)

            x = Concatenate()([x, x2])
            x = self.cbr(x, self.__output_layer_n, 3, 1)
            x = UpSampling2D(size = (2, 2))(x)

            x = Concatenate()([x, x1])
            x = Conv2D(self.__output_layer_n, kernel_size = 3, strides = 1, padding = 'same')(x)

            out = Activation('sigmoid')(x)

        model = Model(input_layer, out)

        return model