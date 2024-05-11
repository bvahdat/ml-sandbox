# What is this codebase about?

See the `rag_from_scratch_summarization.ipynb` notebook in this folder.

# How to run

I use [miniconda](https://docs.conda.io/en/latest/miniconda.html) through [Homebrew](https://formulae.brew.sh/cask/miniconda) on my Mac to setup the required environment:

```
conda create -n rag-from-scratch -c conda-forge notebook==7.1.0 langchain=0.1.20 langchain-community=0.0.38 langchain-openai=0.0.8 pydub=0.25.1 pytube=15.0.0 yt-dlp=2024.4.9 python=3.10.13
conda activate rag-from-scratch
pip install openai-whisper==20231117
jupyter notebook rag_from_scratch_summarization.ipynb
```

The corresponding LangSmith traces are [here](https://smith.langchain.com/public/7ac73fde-b775-47ca-be38-adcb3f7d1606/r).

Finally the created environment above can be removed through:

```
conda deactivate && conda remove --name rag-from-scratch --all
```
