import tensorflow as tf
# 필요한 패키지들
import os
from glob import glob
# PIL는 이미지를 load 할 때, numpy는 array
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random


data_list = glob('cerana/*.jpg')

X = tf.placeholder(tf.float32, [None, 1000, 800, 3])
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

W3 = tf.Variable(tf.random_normal([200 * 250 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 200 * 250 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, rate)

W4 = tf.Variable(tf.random_normal([256, 2], stddev=0.01))
model = tf.matmul(L3, W4)
#
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 5
total_batch = int(len(data_list)/ batch_size)


for epoch in range(15):
    total_cost = 0
    print('Epoch:', epoch)
    start = random.randrange(0, 2)
    for i in data_list[start*batch_size: start*batch_size+batch_size]:
        image = Image.open(i).resize((800, 1000))
        image = np.array(image)
        # print(image.shape)
        # plt.imshow(image)
        # plt.show()
#         # 이미지 데이터를 CNN 모델을 위한 자료형태인 [28 28 1] 의 형태로 재구성합니다.

        batch_xs = image
        batch_xs = batch_xs.reshape(-1, 1000, 800, 3)
        # print(batch_xs.shape)

        batch_ys = np.array([1,0])
        batch_ys = batch_ys.reshape(-1,2)
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          rate: 0.7})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
# print('정확도:', sess.run(accuracy,
#                         feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
#                                    Y: mnist.test.labels,
#                                    rate: 1}))