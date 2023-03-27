# What is this codebase about?

See [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

# How to run

I use [miniconda](https://docs.conda.io/en/latest/miniconda.html) through [Homebrew](https://formulae.brew.sh/cask/miniconda) on my Mac to setup the required environment:

```
conda create -n pneumonia -c conda-forge tensorflow python=3.10 
conda activate pneumonia
python pneumonia.py
```

Which yields an output similar to:

```
```

Finally the created environment above can be removed through:

```
conda remove --name pneumonia --all
```

The achieved `...` score is `...`.
