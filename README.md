# Project setup

[Install miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers)

```
conda install rdkit
conda install python-Levenshtein
```

Activate conda environment.

```
pip install gdown
pip install fastai==1.0.61 opencv-python
chmod +x install_mmdet.sh
./install_mmdet.sh
```


Download assets

```
./download_models.sh
./download_sample_dataset.sh
```

# Run

```python run.py```