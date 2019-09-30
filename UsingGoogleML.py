
from google.cloud import automl_v1beta1
from PIL import Image
import numpy as np


# 'content' is base-64-encoded image data.
def get_prediction(content, project_id, model_id):

  prediction_client = automl_v1beta1.PredictionServiceClient()

  name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  payload = {'image': {'image_bytes': content }}
  params = {}
  request = prediction_client.predict(name, payload, params)
  return request  # waits till request is returned

if __name__ == '__main__':

  file_path = "./testright/Project_Image013_2.jpg"
  project_id = 593934753829
  model_id = 'ICN6214291365001473062'

  with open(file_path, 'rb') as ff:
      content = ff.read()

  image = Image.open(file_path)
  image = np.array(image)
  content1 = image.tobytes()

  res = get_prediction(content, project_id, model_id).payload[0]
  print(res.display_name)
  if res.display_name =='cerana':
    print([[res.classification.score,1-res.classification.score]])
  else:
    print([[ 1 -res.classification.score, res.classification.score]])
