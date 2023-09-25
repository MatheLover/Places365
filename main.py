from keras.models import Model
from vgg16_places_365 import VGG16_Places365
img_size = (224, 224)
base_model = VGG16_Places365(weights='places', include_top=True, input_shape=(224, 224, 3))
feature_model = Model(base_model.input, base_model.get_layer('fc1').output)
print(feature_model.summary())