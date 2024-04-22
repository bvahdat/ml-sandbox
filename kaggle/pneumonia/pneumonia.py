import os

import matplotlib
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # metal_plugin/src/kernels/stateless_random_op.cc:282] Note the GPU implementation does not produce the same series as CPU implementation.

print('matplotlib version: %s' % matplotlib.__version__)
print('tensorflow version: %s' % tf.__version__)
print('number of GPUs available: %d' % len(tf.config.list_physical_devices('GPU')))

def load_images():

    def load_images_for(directory):
        return tf.keras.utils.image_dataset_from_directory(
            directory,
            shuffle=True,
            batch_size=64,
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
    data_augmentation = Sequential([
        RandomFlip('horizontal'),
        RandomRotation(0.2)
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
    # WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=.00001),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])

    return model


def train_model(m, train, val, test):
    hist = m.fit(train, epochs=50, validation_data=val)
    loss, accuracy, precision, recall = m.evaluate(test)
    print(f'the accuracy/precision/recall scores on the test dataset: {accuracy:.3f}/{precision:.3f}/{recall:.3f} with a loss of: {loss:.3f}')

    return hist


def plot_learning_curves(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    precision = hist.history['precision']
    val_precision = hist.history['val_precision']

    recall = hist.history['recall']
    val_recall = hist.history['val_recall']

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')

    plt.plot(precision, label='Training Precision')
    plt.plot(val_precision, label='Validation Precision')

    plt.plot(recall, label='Training Recall')
    plt.plot(val_recall, label='Validation Recall')

    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy, Precision and Recall')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Binary Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


train_ds, val_ds, test_ds = load_images()
print_train_dataset_details(train_ds)
train_ds, val_ds, test_ds = use_buffered_prefetching(train_ds, val_ds, test_ds)
model = build_transfer_learning_model()
history = train_model(model, train_ds, val_ds, test_ds)
plot_learning_curves(history)
