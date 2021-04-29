## Project setup

### Install miniconda for Python 3.8 from bash file 

[Download miniconda](https://docs.conda.io/en/latest/miniconda.html)

Run installation
```bash
./Miniconda3-latest-Linux-x86_64.sh
```
### Create local conda environment
Only python 3.7 will work
```bash
conda env create -f environment.yml -p ./.mol_rec python=3.7
conda activate ./.mol_rec
```

### Install poetry dependencies
```bash
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

##FAQ

### How to run bash files  
#### Linux
```bash
chmod +x <filename>.sh
./<filename>.sh
```
