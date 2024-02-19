# What is this codebase about?

See the `returns.ipynb` notebook in this folder.

# How to run

I use [miniconda](https://docs.conda.io/en/latest/miniconda.html) through [Homebrew](https://formulae.brew.sh/cask/miniconda) on my Mac to setup the required environment:

```
conda create -n returns -c conda-forge matplotlib==3.8.3 notebook==7.1.0 pyarrow==15.0.0 statsmodels==0.14.1 python=3.10.13
conda activate returns
jupyter notebook returns.ipynb
```

Finally the created environment above can be removed through:

```
conda remove --name returns --all
```
