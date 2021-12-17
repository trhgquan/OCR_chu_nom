# OCR Chữ Nôm, nhưng nó lạ lắm!

Đồ án môn học Nhập môn xử lí ngôn ngữ tự nhiên - Introduction to Natural Language Processing (CSC15006).

## Tóm tắt:
Với bài toán OCR (Optical Character Recognition - Nhận dạng kí tự quang học) chữ Nôm, hướng giải quyết được đưa ra như sau:
- Dùng CenterNet dựng heatmap và vẽ bounding box cho mỗi chữ.
- Dùng một model classify để phân loại chữ nằm trong vùng được nhận dạng bên trên.

## Mã nguồn:
Trong repository này chứa:
- `create_data.py`: Tạo file .csv gồm tên file ảnh và nhãn từng chữ Nôm ban đầu.
- `crop_image.py`: Crop ảnh thành các file nhỏ để train model classification, dùng cho bước 3 link [2] (xem tài liệu tham khảo)
- `do_an_real.ipynb`: Train model classification, dùng 42 ảnh đầu của dataset (16311 samples, 1144 classes).
- `do_an_real_second_approach.ipynb`: Train model CenterNet để dựng heatmap cho ảnh. Có demo pipeline khi hoàn tất.

## Các tài liệu tham khảo:
- [1] basu369victor - [Kuzushiji Recognition just like Digit Recognition](https://www.kaggle.com/basu369victor/kuzushiji-recognition-just-like-digit-recognition/notebook)
- [2] kmat2019 - [CenterNet -Keypoint Detector-
](https://www.kaggle.com/kmat2019/centernet-keypoint-detector)

## Trân trọng cảm ơn:
- basu369victor - Kuzushiji Recognition just like Digit Recognition
- kmat2019 - CenterNet -Keypoint Detector-
- Anh [Trần Xuân Hoàng](https://github.com/hoangxtr) (Thị giác máy tính - VNUHCM-UT) đã giải đáp khúc mắc và hướng dẫn trong quá trình train model.
- Bạn [Nguyễn Thanh Sang](https://github.com/hoangxtr) (Kỹ thuật dữ liệu - HCMUTE) đã cho nhiều tips tuyệt vời trong quá trình train model.
- Chị [Phạm Ngọc Thắm](https://www.facebook.com/vam.p.pham) (Khoa Ngôn ngữ Trung Quốc - VNUHCM-USSH) đã tham gia test .
- *Một bạn chưa biết tên* - (không biết khoa gì - VNUHCM-USSH) đã confirm những chữ được OCR đúng với bản gốc.
- MiAI - vì đã viết tutorial flow_from_directory.

VNUHCM - University of Sicence, 2021.

