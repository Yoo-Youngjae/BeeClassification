import tensorflow as tf
import os
import numpy as np
from testOpenCV import transBG2GW
## 변수
width = 2736
height = 1824
filename = "./test/Project_Image133_2.jpg"
RGBnum = 1

image = transBG2GW(filename)
image = np.array(image)
test_xs = image
test_xs = test_xs.reshape(-1, height, width, RGBnum)
test_ys = np.array([0, 1]).reshape(-1, 2)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model-0.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    rate = graph.get_tensor_by_name("rate:0")
    #
    feed_dict ={X: test_xs, Y: test_ys, rate: 1}
    #
    accuracy = graph.get_tensor_by_name("accuracy:0")
    #
    # 최종 결과 출력
    print (sess.run(accuracy,feed_dict ={X: test_xs, Y: test_ys, rate: 1}))







