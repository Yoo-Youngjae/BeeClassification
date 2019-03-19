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
width = 1440
height = 960

batch_size = 10
total_epoch = 10
total_batch = int(len(data_list)/ batch_size)
dropout = 0.7

test_size = 5

X = tf.placeholder(tf.float32, [None, height, width, 3])
Y = tf.placeholder(tf.float32, [None, 2])
rate = tf.placeholder(tf.float32)


W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))

L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)

L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, rate)

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, rate)

W3 = tf.Variable(tf.random_normal([width * height * 4, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, width * height * 4])
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
    print('Epoch:', epoch)

    #epoch 짝수일때는 mel폴더에서, 홀수일땐 cer 폴더에서 train.
    if epoch % 2 == 0:
        folder_name = folder_name_cer
        # cerana가 [1,0].

        batch_ys = np.array([1, 0])
    else:
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

        image = Image.open(i).resize((width, height))
        image = np.array(image)
        # print(image.shape)
        plt.imshow(image)
        plt.show()
        # 이미지 데이터를 CNN 모델을 위한 자료형태인 [28 28 1] 의 형태로 재구성합니다.

        batch_xs = image
        batch_xs = batch_xs.reshape(-1, height, width, 3)
        # print(batch_xs.shape)

        batch_ys = batch_ys.reshape(-1, 2)
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          rate: dropout})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / len(train_data)))
    # 훈련 끝

    # 전체 image 수가 batch_size 나눴을때 나머지 생기면 버림.
    if epoch % 10 == 0 and epoch != 0:
        total_acc = 0
        for i in test_data:
            image = Image.open(i).resize((width, height))
            image = np.array(image)
            test_xs = image.reshape(-1, height, width, 3)
            test_ys = getTestLabel(i)
            total_acc += sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys, rate: 1})
        print('정확도:', total_acc/test_size)

print('최적화 완료!')
