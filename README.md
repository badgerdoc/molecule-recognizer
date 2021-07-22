# Installation

```bash
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
./download_dataset
rm gpu_dataset.zip
mkdir "content"
```

Download `prep_train.pkl` from Colab notebook to `content` folder

### Optional

Install `pydantic` plugin for pycharm.
