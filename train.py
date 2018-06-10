# import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

from keras import Model
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense

from keras.optimizers import Adam

# prepare data
train_path = 'data/train'
valid_path = 'data/valid'

train_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=2
)

valid_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=2
)

mobilenet_model = MobileNet(alpha=0.25)
# mobilenet_model.summary()


# replace last 5 layers with 'predictions' layer
x = mobilenet_model.layers[-6].output
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=mobilenet_model.input, outputs=predictions)
# model.summary()

# freeze all layers except the last 4
for layer in model.layers[: -5]:
    layer.trainable = False

# compile model
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=19, validation_data=valid_batches,
                    validation_steps=19, epochs=20, verbose=2)

model.save('saved_models/fruits_2_5_20.h5')
