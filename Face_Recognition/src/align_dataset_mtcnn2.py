from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imutils.video import VideoStream
import imutils
import cv2
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import align.detect_face
import datetime
import requests
from time import sleep


def main(args):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

    MINSIZE = 20  # Kích thước tối thiểu (pixel) của khuôn mặt được phát hiện
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709  # Hệ số được sử dụng trong thuật toán MTCNN dùng tìm kiếm khuôn mặt ở các tỷ lệ khác nhau

    # Khởi tạo biến số ảnh đã cắt và căn chỉnh thành công
    nrof_successfully_aligned = 0

    cam = VideoStream(src=1).start()

    name = input('Nhập tài khoản sinh viên:')
    dem = 0
    try:
        respone = getDat(name)
        print("Bắt đầu chụp ảnh sinh viên, nhấn q để thoát!")
        if respone.status_code == 200:
            id = respone.json()['data']
            while (True):
                frame = cam.read()
                frame = imutils.resize(frame, width=600)
                frame = cv2.flip(frame, 1)
                if dem < 4:
                    if dem == 0:
                        sleep(3)
                        cv2.putText(frame, "CHUAN BI CHUP", (200, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), thickness=3, lineType=2)
                    else:
                        sleep(1)
                        cv2.putText(frame, str(dem), (300, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                    3, (0, 255, 0), thickness=3, lineType=2)
                    dem += 1
                else:
                    frame = frame[:, :, 0:3]

                    # Phát hiện khuôn mặt trong khung hình
                    bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD,
                                                                      FACTOR)

                    # Đếm số khuôn mặt trong khung hình
                    faces_found = bounding_boxes.shape[0]

                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    try:
                        if faces_found > 1:
                            cv2.putText(frame, "ONLY 1 FACE", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                        elif faces_found > 0:
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)

                            bb[0] = np.maximum(det[0] - args.margin / 2, 0)
                            bb[1] = np.maximum(det[1] - args.margin / 2, 0)
                            bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
                            bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])

                            # Tạo thư mục đầu ra cho lớp này
                            output_class_dir = os.path.join(output_dir, id)

                            if not os.path.exists(output_class_dir):
                                os.makedirs(output_class_dir)

                            output_filename = os.path.join(output_class_dir,
                                                           id + '_' + str(nrof_successfully_aligned) + '.png')

                            if not os.path.exists(output_filename):

                                # Cắt và căn chỉnh kích thước ảnh khuôn mặt
                                cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                                from PIL import Image
                                cropped = cropped.astype('uint8')
                                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                                scaled = cv2.resize(cropped, (160, 160))
                                
                                cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                                cv2.putText(frame, str(nrof_successfully_aligned), (bb[0], bb[3] + 20),
                                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (255, 255, 255), thickness=1, lineType=2)

                                if cv2.Laplacian(scaled, cv2.CV_64F).var() >= 50:
                                    scaled = Image.fromarray(scaled)
                                    # scaled = cropped.resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)

                                    filename_base, file_extension = os.path.splitext(output_filename)
                                    # Đặt tên cho file đầu ra (nếu có nhiều khuôn mặt trong ảnh)
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                    # print("output: ",output_filename_n)

                                    # Tăng biến đếm số ảnh đã cắt và căn chỉnh thành công
                                    nrof_successfully_aligned += 1
                                    # Lưu ảnh đã cắt
                                    scaled.save(output_filename_n)

                        elif faces_found == 0:
                            cv2.putText(frame, "PUT YOUR FACE", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                    except:
                        pass

                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if nrof_successfully_aligned == 50:
                    break
            print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    except:
        while (True):
            frame = cam.read()
            frame = imutils.resize(frame, width=600)
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "KHONG TON TAI NHAN VIEN", (100, 300), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), thickness=3, lineType=2)
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cam.stream.release()
    cv2.destroyAllWindows()


def getDat(name):
    current_date = datetime.date.today()
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    url = "http://192.168.100.195:8000/api/user/get_code_user?username=" + str(name)
    client = requests.session()
    login_data = dict(day=current_date, time=current_time)
    r = client.get(url, data=login_data, headers={"Referer": "foo"})
    return r


def postData(name):
    current_date = datetime.date.today()
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    url = "http://192.168.100.195:8000/api/user/get_code_user?username=" + str(name)
    client = requests.session()
    login_data = dict(day=current_date, time=current_time)
    r = client.post(url, data=login_data, headers={"Referer": "foo"})
    return r


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
