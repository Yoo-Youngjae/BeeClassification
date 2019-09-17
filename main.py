import tensorflow as tf
# 필요한 패키지들
import os
from glob import glob
# PIL는 이미지를 load 할 때, numpy는 array
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from process import getTestList, getTrainList, getTrainLabel, getTestLabel
from testOpenCV import transBG2GW

folder_name_mel = 'mellifera/*.jpg'
folder_name_cer = 'cerana/*.jpg'

#기본적으로 두 사진의 양이 같다는 전제로 짬. 두 사진의 양이 10개 이상 차이나면 문제생길것.
folder_name = folder_name_cer
data_list = glob(folder_name)
total_image_num = len(data_list) * 2
width = 2736
height = 1824

batch_size = 20
total_epoch = 30
dropout = 0.7

test_size = 15
RGBnum = 1
lr= 0.0001

report = open("report", 'w')



X = tf.placeholder(tf.float32, [None, height, width, RGBnum], name='X')
Y = tf.placeholder(tf.float32, [None, 2], name='Y')
rate = tf.placeholder(tf.float32, name='rate')

b1 = tf.Variable(tf.truncated_normal(stddev=1.0,shape = [1, (int)(height/3), (int)(width/3), 32]))
W1 = tf.Variable(tf.random_normal([3, 3, RGBnum, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 3, 3, 1], padding='SAME')
L1 = tf.nn.relu(L1+b1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, rate)

b2 = tf.Variable(tf.truncated_normal(stddev=1.0,shape = [1, (int)(height/(3*2*4)), (int)(width/(3*2*4)), 64]))
W2 = tf.Variable(tf.random_normal([4, 4, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 4, 4, 1], padding='SAME')
L2 = tf.nn.relu(L2 + b2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, rate)

W3 = tf.Variable(tf.random_normal([ (height//48) * (width//48) * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, (height//48) * (width//48)  * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, rate)

W4 = tf.Variable(tf.random_normal([256, 2], stddev=0.01))
model = tf.matmul(L3, W4)
softmax = tf.nn.softmax(model, name='softmax')
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

#정확도 확인을 위함
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32),name='accuracy')

## for saver ###########################
SAVER_DIR = "model"
saver = tf.train.Saver(save_relative_paths=True)
checkpoint_path = os.path.join(SAVER_DIR, "model")
checkpoint_path = './model'
##############################################


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print('=======================')
print('START PROGRAM')
print('total_image_num:', total_image_num)
print('width:', width)
print('height:', height)
print('batch_size:', batch_size)
print('total_epoch:', total_epoch)
print('dropout:', dropout)
print('test_size:', test_size)
print('=======================')



for epoch in range(total_epoch):
    total_cost = 0
    print("Epoch ",epoch)

    rand_end = total_image_num // batch_size
    start = random.randrange(0, rand_end - 1)
    train_data = getTrainList(start, batch_size)
    test_data = getTestList(test_size)
    train_acc = 0
    #start 기준으로 batch size 갯수 만큼 뽑음.
    for i in train_data:
        i, batch_ys = getTrainLabel(i)

        # image = Image.open(i).resize((width, height)).convert('L')

        # ORIGINAL OPEN
        # image = Image.open(i).resize((width, height))

        # OPENCV OPEND

        image = transBG2GW(i)
        image = np.array(image)
        # 이미지 데이터를 CNN 모델을 위한 자료형태인 [28 28 1] 의 형태로 재구성합니다.
        batch_xs = image.reshape(-1, height, width, RGBnum)
        # print(batch_xs.shape)

        batch_ys = batch_ys.reshape(-1, 2)
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          rate: dropout})
        train_acc += sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, rate: dropout})
        total_cost += cost_val

    print('훈련 Avg. cost =', '{:.3f}'.format(total_cost / len(train_data)))
    print('훈련 정확도 =', train_acc/batch_size)
    # 훈련 끝

    # 전체 image 수가 batch_size 나눴을때 나머지 생기면 버림.
    if True:
        total_acc = 0
        w_epoch = "Epoch ", str(epoch), '\n'
        report.write(''.join(w_epoch))
        for i in test_data:
            # ORIGINAL OPEN
            # image = Image.open(i).resize((width, height))

            # OPENCV OPEND
            image = transBG2GW(i)
            image = np.array(image)
            test_xs = image
            test_xs = test_xs.reshape(-1, height, width, RGBnum)
            test_ys = getTestLabel(i).reshape(-1, 2)
            # report 에 쓰기

            res_arr = sess.run(softmax, feed_dict={X: test_xs, Y: test_ys, rate: 1})
            print(res_arr)

            test_res = sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys, rate: 1})
            # 틀렸을때 틀린 사진 보여주기
            if test_res == 0:
                if test_ys[0][0] == 1:
                    my_res = "정답은 cerana\n"
                else:
                    my_res = "정답은 mellifera\n"
                com_res = "이번 문제는 ", i, "\n"
                report.write(''.join(com_res))
                report.write(my_res)
            #     plt.imshow(image, cmap='gray')
            #     plt.show()

            total_acc += test_res

        print('테스트 정확도:', total_acc/test_size)
        res = total_acc/test_size
        test_res_report = '테스트 정확도:'+str(res)+"\n"
        report.write(test_res_report)
        ## 정확도 0.9 이상이면 Save
        if total_acc/test_size >=0.9:
            print("SAVE")
            saver.save(sess, checkpoint_path, global_step=epoch)
        report.write('============================\n')

print('프로그램 종료')
report.close()