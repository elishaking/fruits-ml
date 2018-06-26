import keras
import tensorflowjs as tfjs
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

with CustomObjectScope(
        {'relu6': keras.applications.mobilenet.relu6, 'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = load_model('saved_models/mobilenet/fruits_2_5_30.h5')

tfjs.converters.save_keras_model(model, 'saved_models/tfjs_model/mobilenet/fruits_2_5_30')
