from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

FILTERS = 64

model = Sequential([
    # Layer 1
    Conv2D(
        input_shape=(32,32,3),
        filters=FILTERS,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=FILTERS,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    MaxPool2D(
        pool_size=(2,2),
        strides=(2,2),
        padding='same'),
    # Layer 2
    Conv2D(
        filters=FILTERS*2,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=FILTERS*2,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    MaxPool2D(
        pool_size=(2,2),
        strides=(2,2),
        padding='same'),
    # Layer 3
    Conv2D(
        filters=FILTERS*4,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=FILTERS*4,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=FILTERS*4,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    MaxPool2D(
        pool_size=(2,2),
        strides=(2,2),
        padding='same'),
    # Layer 4
    Conv2D(
        filters=FILTERS*8,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=FILTERS*8,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=FILTERS*8,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    MaxPool2D(
        pool_size=(2,2),
        strides=(2,2),
        padding='same'),
    # Layer 6
    Conv2D(
        filters=FILTERS*16,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Flatten(),
    # Layer 7
    Dense(units=256, activation='relu'),
    Dropout(rate=0.25),
    # Layer 8
    Dense(units=256, activation='relu'),
    Dropout(rate=0.25),
    # Layer 9
    Dense(units=100, activation='relu'),
    # Output layer
    Dense(units=100, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.save('../models/CIFAR-100/empty.model')
model.summary()