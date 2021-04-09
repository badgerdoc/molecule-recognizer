# Project setup

[Install miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers)

```
conda install rdkit
conda install python-Levenshtein
```

Activate conda environment.

```
pip install gdown
pip install "torch==1.4" "torchvision==0.5.0"
pip install fastai==1.0.61 opencv-python
```


Download assets

```
./download_model.sh
./download_sample_dataset.sh
```

# Run

```python run.py```