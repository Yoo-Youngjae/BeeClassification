
from google.cloud import automl_v1beta1
import tensorflow as tf
import numpy as np
from testOpenCV import transBG2GW
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
cnt = 0

def get_prediction(content):


    prediction_client = automl_v1beta1.PredictionServiceClient()
    global cnt
    print(cnt)
    name = 'projects/{}/locations/us-central1/models/{}'.format(593934753829, 'ICN6214291365001473062')
    payload = {'image': {'image_bytes': content}}
    params = {}
    request = prediction_client.predict(name, payload, params)
    return request  # waits till request is returned

filename = "./testright/Project_Image013_2.jpg"  # Project_Image013_2.jpg"
image = Image.open(filename)
image = np.array(image)




explainer = LimeImageExplainer()
explanation = explainer.explain_instance(image, get_prediction, hide_color=0, top_labels=2, num_samples=1000)

temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=2, hide_rest=False)
cv.imwrite('GoogleLimeTest.jpg', mark_boundaries(temp / 2 + 0.5, mask).astype('uint8'))
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask).astype('uint8'))
plt.show()
# 'content' is base-64-encoded image data.


