import os
import sys
import getopt

from pathlib import Path
from utils import *

'''
Load aspect ratio and letter stats
'''
with open('models/train_input_for_size_estimate.txt', 'r+') as f:
    train_input_for_size_estimate = json.load(f)

with open('models/aspect_ratio_pic_all.txt', 'r+') as f:
    aspect_ratio_pic_all = json.load(f)

with open('models/average_letter_size_all.txt', 'r+') as f:
    average_letter_size_all = json.load(f)

'''
Load models
'''
# K.clear_session()

model_1 = create_model(
    input_shape = (512, 512, 3),
    size_detection_mode = True    
)

model_1.load_weights('models/model_1/final_weights_step1.hdf5')

model_2 = create_model(
    input_shape = (512, 512, 3),
    size_detection_mode = False
)

model_2.load_weights('models/model_2/final_weights_step2.h5')

model_3 = load_model('models/model_3/model_chu_nom.h5')

yy_ = np.load('models/model_3/yy_.npy')
lb = LabelEncoder()
y_integer = lb.fit_transform(yy_)

fontsize = 50
font = ImageFont.truetype('NotoSansCJKjp-Regular.otf', fontsize, encoding = 'utf-8')

def pipeline(image_path, print_img = False):
    # Model1: determine how to split image
    if print_img:
        print('Model 1')

    img_handle = Image.open(image_path).convert('RGB')

    img_h, img_w = img_handle.size

    aspect_ratio_pic = img_h / img_w

    img = np.asarray(img_handle.resize((512, 512)).convert('RGB'))
    predicted_size = model_1.predict(img.reshape(1, 512, 512, 3) / 255)

    detect_num_h = aspect_ratio_pic * np.exp(-predicted_size / 2)
    detect_num_w = detect_num_h / aspect_ratio_pic
    h_split_recommend = np.maximum(1, detect_num_h / base_detect_num_h)
    w_split_recommend = np.maximum(1, detect_num_w / base_detect_num_w)

    if print_img:
        print('Recommended split_h: {}, split_h: {}'.format(h_split_recommend, w_split_recommend))

    # Model2: detection with CenterNet
    if print_img:
        print('Model 2')

    img = Image.open(image_path).convert('RGB')
    box_and_score_all = split_and_detect(
        model_2,
        img,
        h_split_recommend, w_split_recommend,
        score_thresh = 0.3, iou_thresh = 0.4)

    if print_img:
        print('Found {} boxes'.format(len(box_and_score_all)))

    print_w, print_h = img.size

    if (len(box_and_score_all) > 0) and print_img: 
        img = draw_rectangle(box_and_score_all[:, 1:], img, 'red')
        img.save('result/image/boxed.jpg')

    # Model3: classification
    print('Model 3:')

    ocred_data = []

    img = cv2.imread(image_path)
    im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, im_th = cv2.threshold(im_grey, 130, 255, cv2.THRESH_BINARY_INV)

    if (len(box_and_score_all) > 0):
        for box in tqdm(box_and_score_all[:,1:]):
            top, left, bottom, right = box

            roi = im_th[int(top):int(bottom), int(left):int(right)]
            roi = cv2.resize(roi, (300, 300))

            ret, th1 = cv2.threshold(roi, 155, 255, cv2.THRESH_BINARY)

            ProcessImage = th1.reshape(1, 300, 300, 1)
            y_pred = model_3.predict(ProcessImage)
            y_true = np.argmax(y_pred, axis=1)
            ChuNom = lb.inverse_transform(y_true)
            
            code = str(ChuNom[0])

            current_box = (code, top, left, bottom, right)

            ocred_data.append(current_box)
    
    return ocred_data

def main(argv):
    if os.path.exists('result') == False:
        os.mkdir('result')
        os.mkdir('result/image')
        os.mkdir('result/char')

    img_path = ''
    print_img = False
    try:
        opts, args = getopt.getopt(argv, 'hi:o', ['input=', 'print_image'])

        for opt, arg in opts:
            if opt == '--input':
                img_path = arg
            if opt == '--print_image':
                print_img = True

        if input == '':
            print('Please specify input path')
            sys.exit(2)

    except getopt.GetoptError:
        print('Command line arguments invalid, try again')
        sys.exit(2)

    print(img_path)

    ans = pipeline(img_path, print_img = print_img)

    with open('result/char/result.txt', 'w+', encoding = 'utf-8') as f:
        for out in ans:
            char, top, left, bottom, right = out
            print(char, top, left, bottom, right, file = f)

    if print_img:
        with Image.open(img_path).convert('RGB') as img:
            char_draw = ImageDraw.Draw(img)
            for out in ans:
                char, top, left, bottom, right = out
                y = int((top + bottom) // 2)
                x = int((left + right) // 2)

                char_draw.text((x, y), char, fill = (0, 22, 255, 0), font = font)            

            img.save('result/image/labeled.jpg')

if __name__ == '__main__':
    main(sys.argv[1:])