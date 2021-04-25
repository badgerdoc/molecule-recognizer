## Project setup

### Install latest miniconda from bash file 

[Download miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Install dependencies
```
conda install -c rdkit rdkit=2020.09.1.0
conda install -c conda-forge python-levenshtein 
poetry install
```

### Download assets
```
./download_models.sh
./download_sample_dataset.sh
```

### Run
```python run.py```
