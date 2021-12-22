import numpy as np
import json
import cv2
import os

from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Conv2D, Conv2DTranspose, BatchNormalization, Activation, AveragePooling2D, GlobalAveragePooling2D, Input, Concatenate, Add, UpSampling2D, LeakyReLU
from keras.models import Model, load_model
from tqdm import tqdm

class OCR_Chu_Nom_Engine:
    def __init__(self):
        self.__n_category = 1
        self.__output_layer_n = self.__n_category + 4

        self.__base_detect_num_h = 25
        self.__base_detect_num_w = 25

        self.__pred_in_h = 512
        self.__pred_in_w = 512

        self.__pred_out_h = int(self.__pred_in_h / 4)
        self.__pred_out_w = int(self.__pred_in_w / 4)

        with open('models/train_input_for_size_estimate.txt', 'r+') as f:
            self.__train_input_for_size_estimate = json.load(f)
        
        with open('models/aspect_ratio_pic_all.txt', 'r+') as f:
            self.__aspect_ratio_pic_all = json.load(f)

        with open('models/average_letter_size_all.txt', 'r+') as f:
            self.__average_letter_size_all = json.load(f)

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

        self.__fontsize = 50
        self.__font = ImageFont.truetype('NotoSansCJKjp-Regular.otf', self.__fontsize, encoding = 'utf-8')

    @staticmethod
    def draw_rectangle(box_and_score, img, color):
        number_of_rect = len(box_and_score) # np.minimum(500, len(box_and_score))

        print('Drawing bounding boxes..')

        for i in tqdm(reversed(list(range(number_of_rect)))):
            top, left, bottom, right = box_and_score[i, :]

            top, left, bottom, right = np.floor(top + 0.5).astype('int32'), np.floor(left + 0.5).astype('int32'), np.floor(bottom + 0.5).astype('int32'), np.floor(right + 0.5).astype('int32')

            draw = ImageDraw.Draw(img)

            thickness = 4

            if color == 'red':
                rect_color = (255, 0, 0)
            elif color == 'blue':
                rect_color = (0, 0, 255)
            else:
                rect_color = (0, 0, 0)
            
            if i == 0:
                thickness = 4
            
            for j in range(2 * thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline = rect_color)

        del draw
        return img

    @staticmethod
    def cbr(x, out_layer, kernel, stride):
        x = Conv2D(out_layer, kernel_size = kernel, strides = stride, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha = 0.1)(x)
        return x

    def resblock(self, x_in, layer_n):
        x = self.cbr(x_in, layer_n, 3, 1)
        x = self.cbr(x, layer_n, 3, 1)
        x = Add()([x, x_in])
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
        '''Create CNN model.
        '''

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

    def split_and_detect(self, model, img, height_split_recommended, width_split_recommended, score_thresh = 0.3, iou_thresh = 0.4):
        '''Split image to many parts and detect boxes on them.
        '''

        width, height = img.size
        pred_in_w, pred_in_h = 512, 512
        pred_out_w, pred_out_h = 128, 128

        category_n = 1
        maxlap = 0.5

        height_split = int(-(-height_split_recommended // 1) + 1)
        width_split = int(-(-width_split_recommended // 1) + 1)
        
        height_lap = (height_split - height_split_recommended) / (height_split - 1)
        height_lap = np.minimum(maxlap, height_lap)
        width_lap = (width_split - width_split_recommended) / (width_split - 1)
        width_lap = np.minimum(maxlap, width_lap)

        if height > width:
            crop_size = int((height) / (height_split - (height_split - 1) * height_lap))
            
            if crop_size >= width:
                crop_size = width
                stride = int((crop_size * height_split - height) / (height_split - 1))
                
                top_list = [i * stride for i in range(height_split - 1)] + [height - crop_size]
                left_list = [0]
            else:
                stride = int((crop_size * height_split - height) / (height_split - 1))
                top_list = [i * stride for i in range(height_split - 1)] + [height - crop_size]
                
                width_split = -(-width // crop_size)
                stride = int((crop_size * width_split - width) / (width_split - 1))
                left_list = [i * stride for i in range(width_split - 1)] + [width - crop_size]
        else:
            crop_size = int((width) / (width_split - (width_split - 1) * width_lap))

            if crop_size >= height:
                crop_size = height
                stride = int((crop_size * width_split - width) / (width_split - 1))
                left_list = [i * stride for i in range(width_split - 1)] + [width - crop_size]
                top_list = [0]
            else:
                stride = int((crop_size * width_split - width) / (width_split - 1))
                left_list = [i * stride for i in range(width_split - 1)] + [width - crop_size]
                height_split = -(-height // crop_size)
                stride = int((crop_size * height_split - height) / (height_split - 1))
                top_list = [i * stride for i in range(height_split - 1)] + [height - crop_size]

        count = 0

        for top_offset in top_list:
            for left_offset in left_list:
                img_crop = img.crop((left_offset, top_offset, left_offset + crop_size, top_offset + crop_size))

                predict = model.predict((np.asarray(img_crop.resize((pred_in_w, pred_in_h))).reshape(1, pred_in_h, pred_in_w, 3)) / 255).reshape(pred_out_h, pred_out_w, (category_n + 4))

                box_and_score = self.NMS_all(predict, category_n, score_thresh, iou_thresh)

                if len(box_and_score) == 0:
                    continue
                
                box_and_score = box_and_score * [1, 1, crop_size / pred_out_h, crop_size / pred_out_w, crop_size / pred_out_h, crop_size / pred_out_w] + np.array([0, 0, top_offset, left_offset, top_offset, left_offset])
                
                box_and_score_all = box_and_score if count == 0 else np.concatenate((box_and_score_all, box_and_score), axis = 0)

                count += 1

        if count == 0:
            box_and_score_all = []
        else:
            score = box_and_score_all[:, 1]
            yc = (box_and_score_all[:, 2] + box_and_score_all[:, 4]) / 2
            xc = (box_and_score_all[:, 3] + box_and_score_all[:, 5]) / 2

            height = -box_and_score_all[:, 2] + box_and_score_all[:, 4]
            width = -box_and_score_all[:, 3] + box_and_score_all[:, 5]

            box_and_score_all = self.NMS(
                box_and_score_all[:, 1], box_and_score_all[:, 2], 
                box_and_score_all[:, 3], box_and_score_all[:, 4],
                box_and_score_all[:, 5], iou_thresh = 0.5, merge_mode = True)
        
        return box_and_score_all
      
    def NMS_all(self, predicts, category_n, score_thresh, iou_thresh):
        '''Non Maximum Suppression to find best boxes among some boxes.
        '''
        yc = predicts[..., category_n] + np.arange(self.__pred_out_h).reshape(-1, 1)
        xc = predicts[..., category_n + 1] + np.arange(self.__pred_out_w).reshape(1, -1)

        height = predicts[..., category_n + 2] * self.__pred_out_h
        width = predicts[..., category_n + 3] * self.__pred_out_w

        count = 0

        for category in range(category_n):
            predict = predicts[..., category]
            mask = (predict > score_thresh)

            if mask.all == False:
                continue
            
            box_and_score = self.NMS(
                predict[mask], 
                yc[mask], xc[mask], 
                height[mask], width[mask], iou_thresh)

            box_and_score = np.insert(box_and_score, 0, category, axis = 1)

            box_and_score_all = box_and_score if count == 0 else np.concatenate((box_and_score_all, box_and_score), axis = 0)

            count += 1

        score_sort = np.argsort(box_and_score_all[:, 1])[::-1]

        box_and_score_all = box_and_score_all[score_sort]

        _, unique_idx = np.unique(box_and_score_all[:, 2], return_index = True)

        return box_and_score_all[sorted(unique_idx)]

    def NMS(self, score, yc, xc, height, width, iou_thresh, merge_mode = False):
        '''Non Maximum Suppression to find best box.
        '''
        if merge_mode:
            score = score
            top, left = yc, xc
            bottom, right = height, width
        else:
            score = score.reshape(-1)
            yc = yc.reshape(-1)
            xc = xc.reshape(-1)
            height = height.reshape(-1)
            width = width.reshape(-1)
            size = height * width
        
            top = yc - height / 2
            left = xc - width / 2
            bottom = yc + height / 2
            right = xc + width / 2

            inside_pic = (top > 0) * (left > 0) * (bottom < self.__pred_out_h) * (right < self.__pred_out_w)
            outside_pic = len(inside_pic) - np.sum(inside_pic)

            normal_size = (size < (np.mean(size) * 10)) * (size > (np.mean(size) / 10))

            score, top, left, bottom, right = score[inside_pic * normal_size], top[inside_pic * normal_size], left[inside_pic * normal_size], bottom[inside_pic * normal_size], right[inside_pic * normal_size]
        
        # Sorting
        score_sort = np.argsort(score)[::-1]
        score, top, left, bottom, right = score[score_sort], top[score_sort], left[score_sort], bottom[score_sort], right[score_sort]

        area = ((bottom - top) * (right - left))

        boxes = np.concatenate((score.reshape(-1, 1), top.reshape(-1, 1), left.reshape(-1, 1), bottom.reshape(-1, 1), right.reshape(-1, 1)), axis = 1)

        box_idx = np.arange(len(top))
        alive_box = []

        while len(box_idx) > 0:
            alive_box.append(box_idx[0])

            y1, x1, y2, x2 = np.maximum(top[0], top), np.maximum(left[0], left), np.minimum(bottom[0], bottom), np.minimum(right[0], right)

            cross_h = np.maximum(0, y2 - y1)
            cross_w = np.maximum(0, x2 - x1)

            still_alive = (((cross_h * cross_w) / area[0]) < iou_thresh)

            if np.sum(still_alive) == len(box_idx):
                print('Error')
                print(np.max((cross_h * cross_w)), area[0])
            
            top, left, bottom, right, area, box_idx = top[still_alive], left[still_alive], bottom[still_alive], right[still_alive], area[still_alive], box_idx[still_alive]

        return boxes[alive_box]
    
    def pipeline(self, image_path, print_img = False):
        base = os.path.basename(image_path).split('.')
        filename, extension = base[0], base[1]

        labeled_image_file = 'result/image/{}_labeled.{}'.format(filename, extension)
        boxed_image_file = 'result/image/{}_boxed.{}'.format(filename, extension)
        ocred_text_file = 'result/char/{}.txt'.format(filename)

        # Model1: determine how to split image
        print('Model 1: Calculating..')
        
        with Image.open(image_path).convert('RGB') as img_handle:
            img_h, img_w = img_handle.size
        
            aspect_ratio_pic = img_h / img_w

            img = np.asarray(img_handle.resize((512, 512)).convert('RGB'))
            predicted_size = self.__model_1.predict(
                img.reshape(1, 512, 512, 3) / 255
            )

            detect_num_h = aspect_ratio_pic * np.exp(-predicted_size / 2)
            detect_num_w = detect_num_h / aspect_ratio_pic

            h_split_recommend = np.maximum(1, detect_num_h / self.__base_detect_num_h)
            w_split_recommend = np.maximum(1, detect_num_w / self.__base_detect_num_w)

            print('Recommended split_h: {}, split_w: {}'.format(
                h_split_recommend,
                w_split_recommend
            ))
        
        # Model2: detection with CenterNet
        print('Model 2: Detecting..')
        
        with Image.open(image_path).convert('RGB') as img:
            box_and_score_all = self.split_and_detect(
                self.__model_2,
                img,
                h_split_recommend, w_split_recommend,
                score_thresh = 0.3, iou_thresh = 0.4
            )

            print('Found {} boxes'.format(len(box_and_score_all)))
        
        # Model3: classification
        print('Model 3: Recognising..')

        ocred_data = []

        img = cv2.imread(image_path)
        im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, im_th = cv2.threshold(im_grey, 130, 255, cv2.THRESH_BINARY_INV)

        if (len(box_and_score_all) > 0):
            for box in tqdm(box_and_score_all[:,1:]):
                top, left, bottom, right = box

                roi = im_th[int(top):int(bottom), int(left):int(right)]
                roi = cv2.resize(roi, (300, 300))

                _, th1 = cv2.threshold(roi, 155, 255, cv2.THRESH_BINARY)

                ProcessImage = th1.reshape(1, 300, 300, 1)
                y_pred = self.__model_3.predict(ProcessImage)
                y_true = np.argmax(y_pred, axis=1)
                ChuNom = self.__lb.inverse_transform(y_true)
                
                code = str(ChuNom[0])

                current_box = (code, top, left, bottom, right)

                ocred_data.append(current_box)

        if print_img:
            # Draw label.
            with Image.open(image_path).convert('RGB') as img:
                char_draw = ImageDraw.Draw(img)
                for out in ocred_data:
                    char, top, left, bottom, right = out
                    y = int((top + bottom) // 2)
                    x = int((left + right) // 2)

                    char_draw.text(
                        (x, y), char, 
                        fill = (0, 22, 255, 0), font = self.__font
                    )            

                img.save(labeled_image_file)
            
            # Draw boxes.
            if len(box_and_score_all) > 0:
                with Image.open(image_path).convert('RGB') as img:
                    img = self.draw_rectangle(box_and_score_all[:, 1:], img, 'red')
                    img.save(boxed_image_file)
        
        # Write ocr-ed data to file.
        with open(ocred_text_file, 'w+', encoding = 'utf-8') as f:
            for out in ocred_data:
                char, top, left, bottom, right = out
                print(char, top, left, bottom, right, file = f)

        if print_img:
            print('Image with bounding boxes saved to', boxed_image_file)
            print('Image with label saved to', labeled_image_file)
        
        print('OCR-ed data saved to', ocred_text_file)