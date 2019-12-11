
from google.cloud import automl_v1beta1
import tensorflow as tf
import numpy as np
from testOpenCV import transBG2GW
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2 as cv
cnt = 0

def get_prediction(image):

    image = image.reshape(1824, 2736, 3)

    im = Image.fromarray(image)
    im.save("output.jpg")
    time.sleep(1.5)
    filename2 = "/Users/youngjae/PycharmProjects/projectFirst/output.jpg"
    with open(filename2, 'rb') as ff:
        content = ff.read()
    prediction_client = automl_v1beta1.PredictionServiceClient()
    global cnt
    cnt += 1
    print(cnt)
    name = 'projects/{}/locations/us-central1/models/{}'.format(593934753829, 'ICN6214291365001473062')
    payload = {'image': {'image_bytes': content}}
    params = {}
    request = prediction_client.predict(name, payload, params)
    request = request.payload[0]

    if request.display_name == 'cerana':
        return [[request.classification.score, 1 - request.classification.score]]
    else:
        return [[1 - request.classification.score, request.classification.score]]


filename = "./testright/Project_Image013_2.jpg"  # Project_Image013_2.jpg"
image = Image.open(filename)
image = np.array(image)
print('first')
print(image.shape)




explainer = LimeImageExplainer()
explanation = explainer.explain_instance(image, get_prediction, hide_color=0, top_labels=2, num_samples=700,batch_size=1)

temp, mask = explanation.get_image_and_mask(1, positive_only=True, num_features=2, hide_rest=False)
cv.imwrite('GoogleLimeTestMellifera.jpg', mark_boundaries(temp/2 +0.5 , mask).astype('uint8'))
plt.imshow(mark_boundaries(temp/2 +0.5, mask).astype('uint8'))
plt.show()
# 'content' is base-64-encoded image data.


