from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import matplotlib as plt

def prefict_fn(images):
    return session.run(probabilities, feed_dict={processed_images: images})

explainer = LimeImageExplainer()
explanation = explainer.explain_instance(image, predict_fn, hide_color=0, top_labels=5, num_samples=1000)

temp, mask = explanation.get_image_and_mask(240, positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))