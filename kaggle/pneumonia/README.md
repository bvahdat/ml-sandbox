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
image batch shape: (64, 224, 224, 3)
image label shape: (64,)
Epoch 1/50
82/82 [==============================] - 9s 79ms/step - loss: 0.8230 - accuracy: 0.5911 - precision: 0.8938 - recall: 0.5102 - val_loss: 0.5395 - val_accuracy: 0.8125 - val_precision: 0.7778 - val_recall: 0.8750
Epoch 2/50
82/82 [==============================] - 5s 59ms/step - loss: 0.5588 - accuracy: 0.6863 - precision: 0.9840 - recall: 0.5874 - val_loss: 0.3628 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 3/50
82/82 [==============================] - 5s 58ms/step - loss: 0.4739 - accuracy: 0.7115 - precision: 0.9905 - recall: 0.6175 - val_loss: 0.3198 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 4/50
82/82 [==============================] - 5s 60ms/step - loss: 0.4356 - accuracy: 0.7383 - precision: 0.9933 - recall: 0.6521 - val_loss: 0.3030 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 5/50
82/82 [==============================] - 5s 58ms/step - loss: 0.3971 - accuracy: 0.7577 - precision: 0.9958 - recall: 0.6766 - val_loss: 0.2744 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 6/50
82/82 [==============================] - 5s 60ms/step - loss: 0.3788 - accuracy: 0.7690 - precision: 0.9948 - recall: 0.6926 - val_loss: 0.2708 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 7/50
82/82 [==============================] - 5s 59ms/step - loss: 0.3594 - accuracy: 0.7845 - precision: 0.9946 - recall: 0.7138 - val_loss: 0.2666 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 8/50
82/82 [==============================] - 5s 59ms/step - loss: 0.3475 - accuracy: 0.7912 - precision: 0.9933 - recall: 0.7239 - val_loss: 0.2520 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 9/50
82/82 [==============================] - 5s 58ms/step - loss: 0.3338 - accuracy: 0.8029 - precision: 0.9927 - recall: 0.7401 - val_loss: 0.2454 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 10/50
82/82 [==============================] - 5s 59ms/step - loss: 0.3226 - accuracy: 0.8133 - precision: 0.9949 - recall: 0.7525 - val_loss: 0.2958 - val_accuracy: 0.8750 - val_precision: 1.0000 - val_recall: 0.7500
Epoch 11/50
82/82 [==============================] - 5s 60ms/step - loss: 0.3166 - accuracy: 0.8257 - precision: 0.9950 - recall: 0.7693 - val_loss: 0.2800 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 12/50
82/82 [==============================] - 5s 58ms/step - loss: 0.3012 - accuracy: 0.8334 - precision: 0.9947 - recall: 0.7799 - val_loss: 0.2437 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 13/50
82/82 [==============================] - 5s 59ms/step - loss: 0.3003 - accuracy: 0.8305 - precision: 0.9957 - recall: 0.7752 - val_loss: 0.2608 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 14/50
82/82 [==============================] - 5s 60ms/step - loss: 0.2902 - accuracy: 0.8390 - precision: 0.9958 - recall: 0.7866 - val_loss: 0.2997 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 15/50
82/82 [==============================] - 5s 58ms/step - loss: 0.2886 - accuracy: 0.8422 - precision: 0.9932 - recall: 0.7930 - val_loss: 0.2510 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 16/50
82/82 [==============================] - 5s 58ms/step - loss: 0.2816 - accuracy: 0.8470 - precision: 0.9965 - recall: 0.7969 - val_loss: 0.2677 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 17/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2794 - accuracy: 0.8516 - precision: 0.9939 - recall: 0.8052 - val_loss: 0.2797 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 18/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2700 - accuracy: 0.8556 - precision: 0.9931 - recall: 0.8114 - val_loss: 0.2445 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 19/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2713 - accuracy: 0.8516 - precision: 0.9946 - recall: 0.8046 - val_loss: 0.2509 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 20/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2688 - accuracy: 0.8599 - precision: 0.9953 - recall: 0.8152 - val_loss: 0.2488 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 21/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2673 - accuracy: 0.8604 - precision: 0.9940 - recall: 0.8170 - val_loss: 0.2254 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 22/50
82/82 [==============================] - 5s 57ms/step - loss: 0.2525 - accuracy: 0.8689 - precision: 0.9944 - recall: 0.8281 - val_loss: 0.2159 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 23/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2557 - accuracy: 0.8694 - precision: 0.9926 - recall: 0.8305 - val_loss: 0.2424 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 24/50
82/82 [==============================] - 5s 58ms/step - loss: 0.2489 - accuracy: 0.8731 - precision: 0.9969 - recall: 0.8317 - val_loss: 0.2366 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 25/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2462 - accuracy: 0.8765 - precision: 0.9945 - recall: 0.8385 - val_loss: 0.2107 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 26/50
82/82 [==============================] - 5s 58ms/step - loss: 0.2406 - accuracy: 0.8808 - precision: 0.9948 - recall: 0.8439 - val_loss: 0.2180 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 27/50
82/82 [==============================] - 5s 57ms/step - loss: 0.2403 - accuracy: 0.8779 - precision: 0.9936 - recall: 0.8410 - val_loss: 0.2324 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 28/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2369 - accuracy: 0.8819 - precision: 0.9951 - recall: 0.8452 - val_loss: 0.2060 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 29/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2341 - accuracy: 0.8819 - precision: 0.9933 - recall: 0.8467 - val_loss: 0.2168 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 30/50
82/82 [==============================] - 5s 56ms/step - loss: 0.2327 - accuracy: 0.8817 - precision: 0.9963 - recall: 0.8439 - val_loss: 0.1966 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 31/50
82/82 [==============================] - 5s 58ms/step - loss: 0.2248 - accuracy: 0.8882 - precision: 0.9949 - recall: 0.8539 - val_loss: 0.2315 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 32/50
82/82 [==============================] - 5s 58ms/step - loss: 0.2266 - accuracy: 0.8869 - precision: 0.9961 - recall: 0.8511 - val_loss: 0.2098 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 33/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2228 - accuracy: 0.8873 - precision: 0.9940 - recall: 0.8534 - val_loss: 0.2346 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 34/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2247 - accuracy: 0.8875 - precision: 0.9949 - recall: 0.8529 - val_loss: 0.2043 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 35/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2164 - accuracy: 0.8942 - precision: 0.9946 - recall: 0.8622 - val_loss: 0.2123 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 36/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2152 - accuracy: 0.8921 - precision: 0.9961 - recall: 0.8581 - val_loss: 0.2164 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 37/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2115 - accuracy: 0.8974 - precision: 0.9953 - recall: 0.8661 - val_loss: 0.2075 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 38/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2165 - accuracy: 0.8940 - precision: 0.9929 - recall: 0.8635 - val_loss: 0.2266 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 39/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2130 - accuracy: 0.8938 - precision: 0.9938 - recall: 0.8625 - val_loss: 0.2397 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 40/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2089 - accuracy: 0.8997 - precision: 0.9938 - recall: 0.8705 - val_loss: 0.1844 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 41/50
82/82 [==============================] - 5s 58ms/step - loss: 0.2078 - accuracy: 0.9015 - precision: 0.9947 - recall: 0.8720 - val_loss: 0.1823 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 42/50
82/82 [==============================] - 5s 57ms/step - loss: 0.2032 - accuracy: 0.9018 - precision: 0.9935 - recall: 0.8735 - val_loss: 0.2127 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 43/50
82/82 [==============================] - 5s 58ms/step - loss: 0.2091 - accuracy: 0.8965 - precision: 0.9938 - recall: 0.8661 - val_loss: 0.1986 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 44/50
82/82 [==============================] - 5s 59ms/step - loss: 0.1971 - accuracy: 0.9064 - precision: 0.9947 - recall: 0.8787 - val_loss: 0.1900 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 45/50
82/82 [==============================] - 5s 59ms/step - loss: 0.2019 - accuracy: 0.9013 - precision: 0.9930 - recall: 0.8733 - val_loss: 0.2235 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 46/50
82/82 [==============================] - 5s 59ms/step - loss: 0.1992 - accuracy: 0.9061 - precision: 0.9944 - recall: 0.8785 - val_loss: 0.2026 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 47/50
82/82 [==============================] - 5s 60ms/step - loss: 0.1948 - accuracy: 0.9032 - precision: 0.9938 - recall: 0.8751 - val_loss: 0.1913 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 48/50
82/82 [==============================] - 5s 59ms/step - loss: 0.1933 - accuracy: 0.9087 - precision: 0.9950 - recall: 0.8815 - val_loss: 0.2068 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 49/50
82/82 [==============================] - 5s 59ms/step - loss: 0.1988 - accuracy: 0.9016 - precision: 0.9947 - recall: 0.8723 - val_loss: 0.1907 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
Epoch 50/50
82/82 [==============================] - 5s 58ms/step - loss: 0.1967 - accuracy: 0.9091 - precision: 0.9942 - recall: 0.8828 - val_loss: 0.1959 - val_accuracy: 0.9375 - val_precision: 1.0000 - val_recall: 0.8750
10/10 [==============================] - 1s 68ms/step - loss: 0.3102 - accuracy: 0.8670 - precision: 0.8866 - recall: 0.9026
the accuracy/precision/recall scores on the test dataset: 0.867/0.887/0.903 with a loss of: 0.310
```

The learning curves of the accuracy/precision/recall metrics look as the following:

![plot](./accuracy-loss.png)

Finally the created environment above can be removed through:

```
conda deactivate && conda remove --name pneumonia --all
```

The achieved accuracy/precision/recall scores on the test dataset is `.867/.887/.903` as it can be spotted by the last line of the log output above.
