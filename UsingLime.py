import tensorflow as tf
import numpy as np
from testOpenCV import transBG2GW
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

######## 파일 이름 과 정답 적기 #######
filename = "./testright/Project_Image013_2.jpg" #Project_Image013_2.jpg"
metafileName = '/Users/youngjae/PycharmProjects/projectFirst/model/model-18.meta'
metafile = '/Users/youngjae/PycharmProjects/projectFirst/model/model-18'
answer = 'cerana'

## 변수
width = 2736
height = 1824
RGBnum = 3


# image = transBG2GW(filename)
# image = Image.open(filename).resize((width, height))
image = Image.open(filename)
image = np.array(image)
test_xs = image
print(test_xs.shape)
# test_xs = test_xs.reshape(-1,height, width,3)

cnt = 0

if answer == 'mellifera':
    test_ys = np.array([0, 1]).reshape(-1, 2)
else:        # cerana 일때
    test_ys = np.array([1, 0]).reshape(-1, 2)

with tf.Session() as sess:
    def predict_fn(test_xs):
        test_xs = test_xs.reshape(-1, height, width, 3)
        global test_ys, cnt
        cnt += 1
        print(cnt)
        return sess.run(softmax, feed_dict={X: test_xs, Y: test_ys, rate: 1})
    saver = tf.train.import_meta_graph(metafileName)
    saver.restore(sess, metafile)
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    rate = graph.get_tensor_by_name("rate:0")
    #
    feed_dict = {X: test_xs, Y: test_ys, rate: 1}
    #
    accuracy = graph.get_tensor_by_name("accuracy:0")
    softmax = graph.get_tensor_by_name("softmax:0")
    # 최종 결과 출력

    # correct_res = sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys, rate: 1})
    # soft_res = sess.run(softmax, feed_dict={X: test_xs, Y: test_ys, rate: 1})


    explainer = LimeImageExplainer()
    explanation = explainer.explain_instance(image, predict_fn, hide_color=0, top_labels=2, num_samples=1000)

    temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=2, hide_rest=False)
    cv.imwrite('limeTest.jpg', mark_boundaries(temp / 2 + 0.5, mask).astype('uint8'))

    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask).astype('uint8'))
    plt.show()