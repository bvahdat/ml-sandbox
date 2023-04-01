import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomRotation

parent_dir = './data/chest_xray'
train_dir = parent_dir + '/train'
val_dir = parent_dir + '/val'
test_dir = parent_dir + '/test'

image_height = image_width = 224
image_shape = (image_height, image_width) + (3,)
batch_size = 32
epochs = 15
learning_rate = .0001


def load_images():

    def load_images_for(directory):
        return tf.keras.utils.image_dataset_from_directory(
            directory,
            shuffle=True,
            batch_size=batch_size,
            image_size=(image_height, image_width))

    return [load_images_for(dir) for dir in [train_dir, val_dir, test_dir]]


def print_train_dataset_details(dataset):
    num_classes = len(dataset.class_names)
    print(f'in the training set {num_classes} class names to classify the images for: {dataset.class_names}')
    for image_batch, labels_batch in dataset:
        print(f'image batch shape: {image_batch.shape}')
        print(f'image label shape: {labels_batch.shape}')
        break


def use_buffered_prefetching(train, val, test):

    def use_buffered_prefetching_for(dataset):
        return dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return [use_buffered_prefetching_for(dataset) for dataset in [train, val, test]]


def build_transfer_learning_model():
    # data augmentation on gpu would run into an error on macOS
    with tf.device('/CPU:0'):
        data_augmentation = Sequential([
            RandomFlip('horizontal'),
            RandomRotation(0.2),
        ])

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    inputs = Input(shape=image_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(.2)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Dense(8)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs, name='pneumonia')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def train_model(m, train, val, test):
    hist = m.fit(train, epochs=epochs, validation_data=val)
    loss, accuracy = model.evaluate(test)
    print(f'accuracy/loss on the test dataset: {accuracy:.3f}/{loss:.3f}')

    return hist


def plot_accuracy_loss_curves(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


train_ds, val_ds, test_ds = load_images()
print_train_dataset_details(train_ds)
train_ds, val_ds, test_ds = use_buffered_prefetching(train_ds, val_ds, test_ds)
model = build_transfer_learning_model()
history = train_model(model, train_ds, val_ds, test_ds)
plot_accuracy_loss_curves(history)
