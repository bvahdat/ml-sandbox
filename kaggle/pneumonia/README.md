# What is this codebase about?

See [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

# How to run (on GPU)

I use [miniconda](https://docs.conda.io/en/latest/miniconda.html) through [Homebrew](https://formulae.brew.sh/cask/miniconda) on my Mac to setup the required environment. See also the details [here](https://developer.apple.com/metal/tensorflow-plugin/) about how to setup `tensorflow-metal` PluggableDevice to accelerate training with Metal on Mac GPUs.

```
conda create -n pneumonia -c conda-forge matplotlib==3.8.3 python=3.10.13
conda activate pneumonia
python -m pip install tensorflow==2.15.0
python -m pip install tensorflow-metal==1.1.0
python pneumonia.py
```

Which yields an output similar to:

```
matplotlib version: 3.8.3
tensorflow version: 2.15.0
number of GPUs available: 1
Found 5216 files belonging to 2 classes.
Found 16 files belonging to 2 classes.
Found 624 files belonging to 2 classes.
in the training set 2 class names to classify the images for: ['NORMAL', 'PNEUMONIA']
image batch shape: (32, 224, 224, 3)
image label shape: (32,)
Epoch 1/40
163/163 [==============================] - 11s 56ms/step - loss: 0.3888 - accuracy: 0.7709 - precision: 0.9923 - recall: 0.6970 - val_loss: 0.1980 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 2/40
163/163 [==============================] - 8s 49ms/step - loss: 0.2894 - accuracy: 0.8416 - precision: 0.9932 - recall: 0.7923 - val_loss: 0.2684 - val_accuracy: 0.7500 - val_precision: 1.0000 - val_recall: 0.5000
Epoch 3/40
163/163 [==============================] - 8s 49ms/step - loss: 0.2557 - accuracy: 0.8727 - precision: 0.9914 - recall: 0.8359 - val_loss: 0.4089 - val_accuracy: 0.6875 - val_precision: 1.0000 - val_recall: 0.3750
Epoch 4/40
163/163 [==============================] - 8s 49ms/step - loss: 0.2200 - accuracy: 0.8907 - precision: 0.9931 - recall: 0.8588 - val_loss: 0.3334 - val_accuracy: 0.8125 - val_precision: 1.0000 - val_recall: 0.6250
Epoch 5/40
163/163 [==============================] - 8s 48ms/step - loss: 0.1963 - accuracy: 0.9066 - precision: 0.9933 - recall: 0.8803 - val_loss: 0.1364 - val_accuracy: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 6/40
163/163 [==============================] - 8s 50ms/step - loss: 0.1746 - accuracy: 0.9189 - precision: 0.9895 - recall: 0.9004 - val_loss: 0.2778 - val_accuracy: 0.8125 - val_precision: 1.0000 - val_recall: 0.6250
Epoch 7/40
163/163 [==============================] - 8s 50ms/step - loss: 0.1582 - accuracy: 0.9262 - precision: 0.9904 - recall: 0.9094 - val_loss: 0.1232 - val_accuracy: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 8/40
163/163 [==============================] - 8s 49ms/step - loss: 0.1517 - accuracy: 0.9362 - precision: 0.9892 - recall: 0.9241 - val_loss: 0.1315 - val_accuracy: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 9/40
163/163 [==============================] - 8s 47ms/step - loss: 0.1413 - accuracy: 0.9396 - precision: 0.9866 - recall: 0.9314 - val_loss: 0.1391 - val_accuracy: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 10/40
163/163 [==============================] - 8s 48ms/step - loss: 0.1295 - accuracy: 0.9454 - precision: 0.9888 - recall: 0.9370 - val_loss: 0.1373 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 11/40
163/163 [==============================] - 8s 48ms/step - loss: 0.1234 - accuracy: 0.9507 - precision: 0.9892 - recall: 0.9440 - val_loss: 0.1288 - val_accuracy: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 12/40
163/163 [==============================] - 8s 48ms/step - loss: 0.1164 - accuracy: 0.9525 - precision: 0.9868 - recall: 0.9486 - val_loss: 0.1290 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 13/40
163/163 [==============================] - 8s 47ms/step - loss: 0.1026 - accuracy: 0.9574 - precision: 0.9882 - recall: 0.9541 - val_loss: 0.1568 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 14/40
163/163 [==============================] - 8s 48ms/step - loss: 0.1045 - accuracy: 0.9563 - precision: 0.9869 - recall: 0.9538 - val_loss: 0.1728 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 15/40
163/163 [==============================] - 8s 48ms/step - loss: 0.1042 - accuracy: 0.9588 - precision: 0.9880 - recall: 0.9561 - val_loss: 0.1102 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 16/40
163/163 [==============================] - 8s 49ms/step - loss: 0.0973 - accuracy: 0.9615 - precision: 0.9850 - recall: 0.9628 - val_loss: 0.1364 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 17/40
163/163 [==============================] - 8s 48ms/step - loss: 0.0974 - accuracy: 0.9618 - precision: 0.9860 - recall: 0.9623 - val_loss: 0.1202 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 18/40
163/163 [==============================] - 8s 48ms/step - loss: 0.0941 - accuracy: 0.9638 - precision: 0.9865 - recall: 0.9644 - val_loss: 0.1161 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 19/40
163/163 [==============================] - 8s 48ms/step - loss: 0.0918 - accuracy: 0.9651 - precision: 0.9860 - recall: 0.9667 - val_loss: 0.1414 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 20/40
163/163 [==============================] - 8s 47ms/step - loss: 0.0905 - accuracy: 0.9636 - precision: 0.9850 - recall: 0.9657 - val_loss: 0.1395 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 21/40
163/163 [==============================] - 8s 47ms/step - loss: 0.0868 - accuracy: 0.9645 - precision: 0.9853 - recall: 0.9667 - val_loss: 0.0985 - val_accuracy: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 22/40
163/163 [==============================] - 8s 48ms/step - loss: 0.0818 - accuracy: 0.9664 - precision: 0.9856 - recall: 0.9690 - val_loss: 0.1422 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 23/40
163/163 [==============================] - 8s 47ms/step - loss: 0.0801 - accuracy: 0.9691 - precision: 0.9889 - recall: 0.9693 - val_loss: 0.1767 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 24/40
163/163 [==============================] - 8s 48ms/step - loss: 0.0758 - accuracy: 0.9720 - precision: 0.9895 - recall: 0.9726 - val_loss: 0.0814 - val_accuracy: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 25/40
163/163 [==============================] - 8s 49ms/step - loss: 0.0727 - accuracy: 0.9693 - precision: 0.9884 - recall: 0.9701 - val_loss: 0.0875 - val_accuracy: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 26/40
163/163 [==============================] - 8s 48ms/step - loss: 0.0829 - accuracy: 0.9670 - precision: 0.9853 - recall: 0.9701 - val_loss: 0.1058 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 27/40
163/163 [==============================] - 8s 49ms/step - loss: 0.0780 - accuracy: 0.9672 - precision: 0.9861 - recall: 0.9695 - val_loss: 0.1209 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 28/40
163/163 [==============================] - 8s 49ms/step - loss: 0.0771 - accuracy: 0.9711 - precision: 0.9867 - recall: 0.9742 - val_loss: 0.0627 - val_accuracy: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 29/40
163/163 [==============================] - 8s 48ms/step - loss: 0.0715 - accuracy: 0.9722 - precision: 0.9885 - recall: 0.9739 - val_loss: 0.1304 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 30/40
163/163 [==============================] - 8s 48ms/step - loss: 0.0774 - accuracy: 0.9697 - precision: 0.9872 - recall: 0.9719 - val_loss: 0.0860 - val_accuracy: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 31/40
163/163 [==============================] - 8s 48ms/step - loss: 0.0678 - accuracy: 0.9745 - precision: 0.9875 - recall: 0.9781 - val_loss: 0.3030 - val_accuracy: 0.8750 - val_precision: 0.8000 - val_recall: 1.0000
Epoch 32/40
163/163 [==============================] - 8s 48ms/step - loss: 0.0701 - accuracy: 0.9718 - precision: 0.9872 - recall: 0.9747 - val_loss: 0.1635 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 33/40
163/163 [==============================] - 8s 47ms/step - loss: 0.0691 - accuracy: 0.9720 - precision: 0.9872 - recall: 0.9750 - val_loss: 0.1619 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 34/40
163/163 [==============================] - 8s 48ms/step - loss: 0.0661 - accuracy: 0.9730 - precision: 0.9895 - recall: 0.9739 - val_loss: 0.1389 - val_accuracy: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 35/40
163/163 [==============================] - 8s 48ms/step - loss: 0.0637 - accuracy: 0.9760 - precision: 0.9908 - recall: 0.9768 - val_loss: 0.1494 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 36/40
163/163 [==============================] - 8s 49ms/step - loss: 0.0684 - accuracy: 0.9735 - precision: 0.9875 - recall: 0.9768 - val_loss: 0.1170 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 37/40
163/163 [==============================] - 8s 49ms/step - loss: 0.0653 - accuracy: 0.9758 - precision: 0.9890 - recall: 0.9783 - val_loss: 0.1338 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 38/40
163/163 [==============================] - 8s 48ms/step - loss: 0.0641 - accuracy: 0.9726 - precision: 0.9877 - recall: 0.9752 - val_loss: 0.1748 - val_accuracy: 0.9375 - val_precision: 0.8889 - val_recall: 1.0000
Epoch 39/40
163/163 [==============================] - 8s 49ms/step - loss: 0.0611 - accuracy: 0.9751 - precision: 0.9880 - recall: 0.9783 - val_loss: 0.0787 - val_accuracy: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 40/40
163/163 [==============================] - 8s 48ms/step - loss: 0.0581 - accuracy: 0.9789 - precision: 0.9901 - recall: 0.9814 - val_loss: 0.0859 - val_accuracy: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
20/20 [==============================] - 1s 28ms/step - loss: 0.4931 - accuracy: 0.8558 - precision: 0.8247 - recall: 0.9769
the accuracy/precision/recall scores on the test dataset: 0.856/0.825/0.977 with a loss of: 0.493
```

The learning curves of the accuracy/precision/recall metrics look as the following:

![plot](./accuracy-loss.png)

Finally the created environment above can be removed through:

```
conda remove --name pneumonia --all
```

The achieved accuracy/precision/recall scores on the test dataset is `0.856/0.825/0.977` as it can be spotted by the last line of the log output above.
