import keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

from keras import Model
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.layers import Dense

from keras.optimizers import Adam

import time

# prepare data
train_path = 'data/train'
valid_path = 'data/valid'

train_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=64
)

valid_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    valid_path,
    target_size=(224, 224),
    batch_size=64
)


def get_model(name='mobilenet', n_classes=1000, n_layers_to_remove=0, fine_tune=False, model_path="./"):
    if fine_tune:
        if name == 'mobilenet':
            with CustomObjectScope(
                    {'relu6': keras.applications.mobilenet.relu6,
                     'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                model = load_model(model_path)
        else:
            model = load_model(model_path)
    else:
        if name == 'mobilenet':
            o_model = MobileNet(alpha=0.25)
        else:
            o_model = Xception()

        o_model.summary()

        # replace last n layers with 'predictions' layer
        x = o_model.layers[-n_layers_to_remove - 1].output
        predictions = Dense(n_classes, activation='softmax')(x)
        model = Model(inputs=o_model.input, outputs=predictions)

    return model


def freeze_model_layers(model, n_layers):
    # freeze all layers except the last n
    for layer in model.layers[: (-n_layers - 1)]:
        layer.trainable = False

    return model


model = get_model('mobilenet', n_classes=64, n_layers_to_remove=5)
# model = get_model('xception')
model.summary()
model = freeze_model_layers(model, 7)


# compile model
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

t0 = time.time()

model.fit_generator(train_batches, steps_per_epoch=420, validation_data=valid_batches,
                    validation_steps=140, epochs=10, verbose=2)

t = time.time() - t0
print('total time:', t)

model.save('saved_models/mobilenet/fruits_2_5_10.h5')
