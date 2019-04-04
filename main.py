import tensorflow as tf
# 필요한 패키지들
import os
from glob import glob
# PIL는 이미지를 load 할 때, numpy는 array
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from process import getTestList, getTrainList, getTestLabel

folder_name_mel = 'mellifera/*.jpg'
folder_name_cer = 'cerana/*.jpg'

#기본적으로 두 사진의 양이 같다는 전제로 짬. 두 사진의 양이 10개 이상 차이나면 문제생길것.
folder_name = folder_name_cer
data_list = glob(folder_name)
total_image_num = len(data_list)
width = 1536
height = 1024

batch_size = 5
total_epoch = 14
total_batch = int(len(data_list)/ batch_size)
dropout = 0.3

test_size = 20

report = open("report", 'w')

b1 = tf.Variable(tf.constant(1.0, shape = [1, (int)(height/4), (int)(width/4), 32]))
X = tf.placeholder(tf.float32, [None, height, width, 1])
Y = tf.placeholder(tf.float32, [None, 2])
rate = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([4, 4, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 4, 4, 1], padding='SAME')
L1 = tf.nn.relu(L1+b1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, rate)

W2 = tf.Variable(tf.random_normal([4, 4, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 4, 4, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, rate)

W3 = tf.Variable(tf.random_normal([24*16 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 24*16 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, rate)

W4 = tf.Variable(tf.random_normal([256, 2], stddev=0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

#정확도 확인을 위함
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print('total_image_num:', total_image_num)
print('width:', width)
print('height:', height)
print('batch_size:', batch_size)
print('total_epoch:', total_epoch)
print('total_batch:', total_batch)
print('dropout:', dropout)
print('test_size:', test_size)



for epoch in range(total_epoch):
    total_cost = 0
    print("Epoch ",epoch)
    #epoch 짝수일때는 mel폴더에서, 홀수일땐 cer 폴더에서 train.
    if epoch % 2 == 0:
        print("train set : Cerana")
        folder_name = folder_name_cer
        # cerana가 [1,0].

        batch_ys = np.array([1, 0])
    else:
        print("train set : melifera")
        folder_name = folder_name_mel
        # mellifera 가 [0,1],
        batch_ys = np.array([0, 1])
    data_list = glob(folder_name)

    rand_end = total_image_num // batch_size
    start = random.randrange(0, rand_end - 1)
    train_data = getTrainList(data_list, start, batch_size)
    test_data = getTestList(test_size)

    #start 기준으로 batch size 갯수 만큼 뽑음.
    for i in train_data:

        image = Image.open(i).resize((width, height)).convert('L')
        image = np.array(image)
        plt.imshow(image, cmap='gray')
        # plt.show()
        # 이미지 데이터를 CNN 모델을 위한 자료형태인 [28 28 1] 의 형태로 재구성합니다.

        batch_xs = image
        batch_xs = batch_xs.reshape(-1, height, width, 1)
        # print(batch_xs.shape)

        batch_ys = batch_ys.reshape(-1, 2)
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          rate: dropout})
        total_cost += cost_val

    print('훈련 Avg. cost =', '{:.3f}'.format(total_cost / len(train_data)))
    # 훈련 끝

    # 전체 image 수가 batch_size 나눴을때 나머지 생기면 버림.
    if True:
        total_acc = 0
        w_epoch = "Epoch ", str(epoch), '\n'
        report.write(''.join(w_epoch))
        for i in test_data:

            image = Image.open(i).resize((width, height)).convert('L')
            image = np.array(image)
            test_xs = image
            test_xs = test_xs.reshape(-1, height, width, 1)
            test_ys = getTestLabel(i).reshape(-1, 2)
            # report 에 쓰기
            my_res = "답은 ", np.array2string(test_ys, precision=2, separator=',', suppress_small=True), "\n"
            report.write(''.join(my_res))
            res_arr = sess.run(model, feed_dict={X: test_xs, Y: test_ys, rate: 1})
            com_res = "컴퓨터 답은 ", np.array2string(res_arr, precision=2, separator=',', suppress_small=True), "\n"
            report.write(''.join(com_res))

            test_res = sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys, rate: 1})
            # 틀렸을때 틀린 사진 보여주기
            # if test_res == 0:
            #     plt.imshow(image, cmap='gray')
            #     plt.show()

            total_acc += test_res
        print('테스트 정확도:', total_acc/test_size)

print('프로그램 종료')
report.close()