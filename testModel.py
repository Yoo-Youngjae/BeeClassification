import tensorflow as tf
import os
import numpy as np
from testOpenCV import transBG2GW
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv


######## 파일 이름 과 정답 적기 #######
filename = "./testright/Project_Image013_2.jpg" #Project_Image013_2.jpg"
metafileName = '/Users/youngjae/PycharmProjects/projectFirst/model/model-18.meta'
metafile = '/Users/youngjae/PycharmProjects/projectFirst/model/model-18'
answer = 'cerana'
###################################

## 변수
width = 2736
height = 1824
RGBnum = 3
# origin 사진 보여주기
image = Image.open(filename)

# 흑백화 사진 보여주기
# image = transBG2GW(filename)
image = np.array(image)
plt.imshow(image, cmap='gray')
plt.show()

test_xs = image
print(test_xs.shape)
test_xs = test_xs.reshape(-1, height, width, RGBnum)


# cerana = 0 = [1,0]
# mellifera = 1 = [0,1]

if answer == 'mellifera':
    test_ys = np.array([0, 1]).reshape(-1, 2)
else:        # cerana 일때
    test_ys = np.array([1, 0]).reshape(-1, 2)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(metafileName)
    saver.restore(sess,metafile)

    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    rate = graph.get_tensor_by_name("rate:0")
    #
    feed_dict ={X: test_xs, Y: test_ys, rate: 1}
    #
    accuracy = graph.get_tensor_by_name("accuracy:0")
    softmax = graph.get_tensor_by_name("softmax:0")
    #
    # 최종 결과 출력
    correct_res = sess.run(accuracy,feed_dict ={X: test_xs, Y: test_ys, rate: 1})
    soft_res = sess.run(softmax,feed_dict ={X: test_xs, Y: test_ys, rate: 1})
    print(soft_res)
    if soft_res[0][0] > 0.5:
        print('cerena 일 확률',soft_res[0][0]);
        print("답은 cerana")
    else:
        print('melifera 일 확률', soft_res[0][1]);
        print("답은 mellifera")

    if correct_res == 1:
        print("정답")
    else:
        print("오답")







