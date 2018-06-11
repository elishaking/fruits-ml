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
    batch_size=10
)

valid_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    valid_path,
    target_size=(224, 224),
    batch_size=10
)

mobilenet_model = MobileNet(alpha=0.25)
# mobilenet_model.summary()


# replace last 5 layers with 'predictions' layer
x = mobilenet_model.layers[-6].output
predictions = Dense(64, activation='softmax')(x)
model = Model(inputs=mobilenet_model.input, outputs=predictions)
# model.summary()

# freeze all layers except the last 22
for layer in model.layers[: -23]:
    layer.trainable = False

# compile model
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=42, validation_data=valid_batches,
                    validation_steps=14, epochs=40, verbose=2)

model.save('saved_models/fruits_2_5_40.h5')
