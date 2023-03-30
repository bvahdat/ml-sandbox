import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation

data_folder = './data'
parent_dir = data_folder + '/chest_xray'
zip_path = data_folder + '/chest_xray.zip'

train_dir = parent_dir + '/train'
val_dir = parent_dir + '/val'
test_dir = parent_dir + '/test'

image_height = image_width = 224
image_shape = (image_height, image_width) + (3,)
batch_size = 32
epochs = 30
base_learning_rate = .0001


def load_images():

    def load_images_for(dir, height=image_height, width=image_width, batch=batch_size):
        return tf.keras.utils.image_dataset_from_directory(
            dir,
            shuffle=True,
            batch_size=batch,
            image_size=(height, width))

    return [load_images_for(dir) for dir in [train_dir, val_dir, test_dir]]


def print_train_dataset_details(dataset):
    num_classes = len(dataset.class_names)
    print(f'in the training set {num_classes} class names to classify the images for: {dataset.class_names}')
    for image_batch, labels_batch in dataset:
        print(f'image batch shape: {image_batch.shape}')
        print(f'image label shape: {labels_batch.shape}')
        break


def use_buffered_prefetching(train_ds, val_ds, test_ds):

    def use_buffered_prefetching_for(dataset):
        return dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return [use_buffered_prefetching_for(dataset) for dataset in [train_ds, val_ds, test_ds]]


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
    base_model.summary()

    inputs = tf.keras.Input(shape=image_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    return model


train_ds, val_ds, test_ds = load_images()
print_train_dataset_details(train_ds)
train_ds, val_ds, test_ds = use_buffered_prefetching(train_ds, val_ds, test_ds)
model = build_transfer_learning_model()
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

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

loss, accuracy = model.evaluate(test_ds)
print(f'accuracy / loss on the test dataset: {accuracy} / {loss}')
