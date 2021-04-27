## Project setup

### Install miniconda for Python 3.8 from bash file 

[Download miniconda](https://docs.conda.io/en/latest/miniconda.html)

1. Make bash file executable
```bash
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
2. Run installation
```bash
./Miniconda3-latest-Linux-x86_64.sh
```
### Create local conda environment
Only python 3.7 will work
```bash
conda create --prefix ./.mol_rec python=3.7 -y
conda activate ./.mol_rec
conda install -c pytorch pytorch torchvision -y
```

### Install dependencies
```bash
conda install -c rdkit rdkit=2020.09.1.0 -y
conda install -c conda-forge python-levenshtein -y
./download_mmcv.sh
poetry install
```

### Download assets
```bash
./download_models.sh
./download_sample_dataset.sh
```

### Run
```bash
python run.py
```
