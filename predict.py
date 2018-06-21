import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

test_path = 'data/train'

with CustomObjectScope(
        {'relu6': keras.applications.mobilenet.relu6, 'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = load_model('saved_models/fruits_2_5_40_40_100_200.h5')
# model.summary()


def plots(ims, figsize=(12, 6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)

        if ims.shape[-1] != 3:
            ims = ims.transpose((0, 2, 3, 1))

    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1

    for j in range(len(ims)):
        sp = f.add_subplot(rows, cols, j + 1)
        sp.axis('Off')

        if titles is not None:
            sp.set_title(titles[j])

        plt.imshow(ims[j], interpolation=None if interp else 'none')


test_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=10,
    shuffle=True
)

test_imgs, test_labels = next(test_batches)
# plots(test_imgs, titles=test_labels)

# test_labels = test_batches.classes
# test_classes = [cls for cls in test_batches.class_indices]
# print(test_classes)

predictions = model.predict_generator(test_batches, steps=1, verbose=0)
cm = confusion_matrix(test_labels[:, 0], np.round(predictions[:, 0]))
print(cm)

# predictions = [[0.8665099, 0.1334901],
#                [0.8632469, 0.1367531],
#                [0.6221927, 0.3778073],
#                [0.28187686, 0.7181232],
#                [0.5965126, 0.40348735],
#                [0.59895986, 0.4010401],
#                [0.9000578, 0.09994221],
#                [0.8674595, 0.13254051],
#                [0.58947843, 0.4105216]]
#
# test_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
# prediction_classes = []
# prediction_confidence = []
# actual_label = []
# i = 0
# print(len(predictions))
# for prediction in predictions:
#     highest_conf = max(prediction)
#     prediction_classes.append(test_classes[prediction.tolist().index(highest_conf)])
#     prediction_confidence.append(highest_conf)
#     actual_label.append(test_classes[test_labels[i]])
#
# for i in range(len(prediction_classes)):
#     print('class:', prediction_classes[i], 'conf:', prediction_confidence[i], 'label:', actual_label[i], sep=' ')
