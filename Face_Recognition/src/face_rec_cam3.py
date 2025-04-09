from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from imutils.video import VideoStream

import time
import datetime
import requests
import argparse
import facenet
import imutils
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from Silent_Face_Anti_Spoofing_master.test import test

def main():
    check = False
    text_x = text_y = 0
    time1_str = datetime.datetime.now().strftime("%H:%M:%S")
    time1 = datetime.datetime.strptime(time1_str, "%H:%M:%S")
    time2_str = datetime.datetime.now().strftime("%H:%M:%S")
    time2 = datetime.datetime.strptime(time2_str, "%H:%M:%S")
    ten = ["no","no","no"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)

    MINSIZE = 20 # Kích thước tối thiểu (pixel) của khuôn mặt được phát hiện
    THRESHOLD = [0.6, 0.7, 0.7] 
    FACTOR = 0.709 # Hệ số được sử dụng trong thuật toán MTCNN dùng tìm kiếm khuôn mặt ở các tỷ lệ khác nhau
    INPUT_IMAGE_SIZE = 160 # Kích thước ảnh được đưa vò Facenet model
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file) 
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        # Cai dat GPU neu co
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors

            # Lấy tensor đầu vào của mô hình. Tên của tensor này là "input" và có chỉ số 0
            images_placeholder = tf.compat.v1.keras.backend.get_session().graph.get_tensor_by_name("input:0")

            # Lấy tensor đầu ra của mô hình. Tên của tensor này là "embeddings" và có chỉ số 0.
            embeddings = tf.compat.v1.keras.backend.get_session().graph.get_tensor_by_name("embeddings:0")

            # Lấy tensor đại diện cho biến boolean "phase_train", được sử dụng để chỉ định xem mô hình đang ở chế độ huấn luyện hay không
            phase_train_placeholder = tf.compat.v1.keras.backend.get_session().graph.get_tensor_by_name("phase_train:0")

            #Sử dụng hàm create_mtcnn từ module align.detect_face để tạo ra các mô-đun phát hiện khuôn mặt P-Net, R-Net và O-Net
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            person_detected = collections.Counter()

            cap  = VideoStream(src = 0).start()

            while (True):
                frame = cap.read()
                
                label = test(
                    image= frame,
                    model_dir='D:/companySS/Face_Recognition/src/Silent_Face_Anti_Spoofing_master/resources/anti_spoof_models',
                    device_id=0
                )

                if label == 1: #real
                    print("REAL")
                    frame = imutils.resize(frame, width=600) #,height=600
                    frame = cv2.flip(frame, 1)
                    # Phát hiện khuôn mặt trong khung hình
                    bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                    # Đếm số khuôn mặt trong khung hình thuaw
                    faces_found = bounding_boxes.shape[0]
                    try:
                        # if faces_found > 1:
                        #     cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        #                 2, (255, 255, 255), thickness=1, lineType=2)
                        if faces_found > 0:
                            time2_str = datetime.datetime.now().strftime("%H:%M:%S")
                            time2 = datetime.datetime.strptime(time2_str, "%H:%M:%S")
                                            
                            if check == True:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    cv2.putText(frame, "SUCCESS", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 255, 0), thickness=1, lineType=2)
                                    cv2.putText(frame, ten[2], (text_x, text_y + 17),
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        1, (255, 255, 255), thickness=1, lineType=2)
                                    print("time2: ",time2,", time1: ",time1)
                                    print((time2-time1).total_seconds())
                                    if (time2-time1).total_seconds() >= 5.0:
                                        check = False
                            
                            if check == False:
                                #Lấy tọa độ bounding box của các khuôn mặt được phát hiện.
                                det = bounding_boxes[:, 0:4]

                                #Lưu trữ các tọa độ bounding box.
                                bb = np.zeros((faces_found, 4), dtype=np.int32)

                                for i in range(faces_found):
                                    #Sao chép tọa độ bounding box vào mảng bb.
                                    bb[i][0] = det[i][0]
                                    bb[i][1] = det[i][1]
                                    bb[i][2] = det[i][2]
                                    bb[i][3] = det[i][3]
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    # print(bb[i][3]-bb[i][1]) # chiều cao của bounding box của khuôn mặt 
                                    # print(frame.shape[0]) # chiều cao khung hình
                                    # print((bb[i][3]-bb[i][1])/frame.shape[0]) # tỷ lệ chiều cao của bounding box so với chiều cao khung hình

                                    if (bb[i][3]-bb[i][1])/frame.shape[0] > 0.25:

                                        # Cắt khung hình (frame) để lấy khuôn mặt đã được phát hiện, với tọa độ x và y được xác định bởi bb[i][0], bb[i][1], bb[i][2], bb[i][3]
                                        cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]

                                        # Điều chỉnh kích thước của khuôn mặt đã cắt thành kích thước mong muốn (INPUT_IMAGE_SIZE)
                                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                            interpolation=cv2.INTER_CUBIC)
                                        
                                        scaled = facenet.prewhiten(scaled)
                                        scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                                        # Tạo một feed dictionary (feed_dict) để cung cấp dữ liệu đầu vào cho mô hình. Trong trường hợp này, images_placeholder là tensor đầu vào
                                        # chứa khuôn mặt đã được chuẩn hóa và phase_train_placeholder là một biến boolean để chỉ định xem mô hình đang ở chế độ huấn luyện hay không.
                                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}

                                        # Chạy mô hình để tính toán các nhúng (embeddings) của khuôn mặt đã được chuẩn hóa
                                        emb_array = sess.run(embeddings, feed_dict = feed_dict)

                                        # Dự đoán xác suất của các lớp cho khuôn mặt đã được nhúng bằng cách sử dụng mô hình
                                        predictions = model.predict_proba(emb_array)
                                        print(predictions)
                                        # Xác định chỉ số của lớp có xác suất cao nhất trong các dự đoán
                                        best_class_indices = np.argmax(predictions, axis=1)

                                        # Lấy xác suất cao nhất tương ứng với lớp có xác suất cao nhất cho mỗi khuôn mặt đã được dự đoán
                                        best_class_probabilities = predictions[
                                            np.arange(len(best_class_indices)), best_class_indices]
                                        
                                        # Xác định tên của lớp có xác suất cao nhất
                                        best_name = class_names[best_class_indices[0]]
                                        print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                                        if best_class_probabilities > 0.85:
                                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                            name = class_names[best_class_indices[0]]
                                            person_detected[best_name] += 1
                                            ten = name.split('-')
                                            
                                            cv2.putText(frame, "SUCCESS", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1, (0, 255, 0), thickness=1, lineType=2)
                                            cv2.putText(frame, ten[2], (text_x, text_y + 17),
                                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1, (255, 255, 255), thickness=1, lineType=2)
                                            
                                            result = getDat(ten[2])
                                            if result.status_code == 200:
                                                print("result: ",result)
                                                postData(ten[0])
                                                check = True
                                                time1_str = datetime.datetime.now().strftime("%H:%M:%S")
                                                time1 = datetime.datetime.strptime(time1_str, "%H:%M:%S")
                                                
                                        else:
                                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0,255), 2)
                                            text_x = bb[i][0]
                                            text_y = bb[i][3] + 20

                                            name = class_names[best_class_indices[0]]
                                            cv2.putText(frame, "Unknow", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                                        1, (255, 255, 0), thickness=1, lineType=2)
                    except:
                        pass
                else:
                    print("FAKE")
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, "FAKE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                        2, (0,0,255), thickness=1, lineType=2)
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.stop()
            cv2.destroyAllWindows()

def getDat(name):
    current_date = datetime.date.today()
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    url = "http://192.168.1.8:8000/api/user/get_code_user?username=" + str(name)
    client = requests.session()
    login_data = dict(day=current_date,time = current_time)
    r = client.get(url, data=login_data, headers={"Referer": "foo"})
    return r

def postData(id):
    current_date = datetime.date.today()
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    url = "http://192.168.1.12:8000/api/time/add_time/" +str(id)+"?day=" + str(current_date)+"&time="+str(current_time)
    print("url = ",url)
    client = requests.session()
    login_data = dict(day=current_date,time = current_time)
    r = client.post(url, data=login_data, headers={"Referer": "foo"})
    return r

main()