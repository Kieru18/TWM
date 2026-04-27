import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
from keras import optimizers
from sklearn.metrics import confusion_matrix
import itertools

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.layers import BatchNormalization

from keras.datasets import cifar10

os.chdir(os.path.dirname(os.path.abspath(__file__)))

TRAIN = True                # IDE Toggle
if len(sys.argv) > 1:       # CLI arg: train / eval
    TRAIN = sys.argv[1].lower() == 'train'

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

y_train = y_train.reshape((1,-1))[0]
y_test = y_test.reshape((1,-1))[0]

print("Training matrix shape", X_train.shape, y_train.shape)
print("Testing matrix shape", X_test.shape, y_test.shape)

nb_classes = 10
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

cifar_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(tf.__version__)

###=========================================================================================================
### GENERATE MODEL FUNCTIONS
###=========================================================================================================
from tensorflow.keras.regularizers import l2

def generate_model():
    from tensorflow.keras.regularizers import l2
    from keras import Input, Model
    from keras.layers import (Conv2D, BatchNormalization, Activation, MaxPooling2D,
                               Dropout, GlobalAveragePooling2D, Dense,
                               Multiply, Reshape)

    wd = 1e-4

    def conv_bn_relu(x, filters):
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def se_block(x, filters, ratio=16):
        se = GlobalAveragePooling2D()(x)
        se = Reshape((1, 1, filters))(se)
        se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', use_bias=False)(se)
        return Multiply()([x, se])

    inputs = Input(shape=(32, 32, 3))

    # Block 1: 32x32 -> 16x16
    x = conv_bn_relu(inputs, 64)
    x = conv_bn_relu(x, 64)
    x = se_block(x, 64)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.2)(x)

    # Block 2: 16x16 -> 8x8
    x = conv_bn_relu(x, 128)
    x = conv_bn_relu(x, 128)
    x = se_block(x, 128)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.3)(x)

    # Block 3: 8x8 -> 4x4
    x = conv_bn_relu(x, 256)
    x = conv_bn_relu(x, 256)
    x = conv_bn_relu(x, 256)
    x = se_block(x, 256)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.4)(x)

    # Classifier head
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.summary()

    total_steps = (40000 // 100) * 75
    warmup_steps = (40000 // 100) * 5

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=total_steps - warmup_steps,
        alpha=1e-6,
        warmup_steps=warmup_steps,
        warmup_target=0.001
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=wd
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model

###=========================================================================================================
### TRAIN OR LOAD
###=========================================================================================================
checkpoint_path = "training/cp.weights.h5"
os.makedirs("training", exist_ok=True)

model = generate_model()

if TRAIN:
    print("=== TRAIN ===")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

    gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                             height_shift_range=0.08, zoom_range=0.08, validation_split=0.2)

    train_generator = gen.flow(X_train, Y_train, batch_size=100, subset='training')
    valid_generator = gen.flow(X_train, Y_train, batch_size=100, subset='validation')

    model.fit(
        train_generator,
        steps_per_epoch=40000 // 100,
        epochs=75,
        validation_data=valid_generator,
        validation_steps=10000 // 100,
        verbose=1,
        callbacks=[cp_callback]
    )
else:
    print("=== EVAL ===")
    model.load_weights(checkpoint_path)

###=========================================================================================================
### EVALUATE
###=========================================================================================================
score = model.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

predicted = model.predict(X_test)
predicted_classes = np.argmax(predicted, axis=1)

correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

def show_samples_rgb(indices, preds, images, labels, count=3, names=[]):
    plt.figure()
    for i, sample in enumerate(indices[:count**2]):
        pred_id = int(np.argmax(preds[sample]))
        real_id = int(labels[sample])
        pred_score = preds[sample][pred_id]
        real_score = preds[sample][real_id]
        plt.subplot(count, count, i+1)
        plt.imshow(images[sample], interpolation='none')
        plt.axis('off')
        if len(names) > 0:
            plt.title("P: {} ({:.2f})\nE: {} ({:.2f})".format(names[pred_id], pred_score, names[real_id], real_score))
        else:
            plt.title("P: {} ({:.2f})\nE: {} ({:.2f})".format(pred_id, pred_score, real_id, real_score))
    plt.tight_layout()
    plt.show()

show_samples_rgb(correct_indices, predicted, X_test, y_test, 5, cifar_names)
show_samples_rgb(incorrect_indices, predicted, X_test, y_test, 5, cifar_names)
