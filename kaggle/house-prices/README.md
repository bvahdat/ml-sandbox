# What is this codebase about?

See [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

# How to run

I use [miniconda](https://docs.conda.io/en/latest/miniconda.html) through [Homebrew](https://formulae.brew.sh/cask/miniconda) on my Mac to setup the required environment:

```
conda create -n house-prices -c conda-forge catboost lightgbm pandas scikit-learn python=3.10 
conda activate house-prices
python house-prices.py
```

Which yields an output similar to:

```
...
```

Finally you can remove the created environment above if you want:

```
conda remove --name house-prices --all
```

The achieved MSE score is `...`.
