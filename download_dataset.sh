gdown https://drive.google.com/uc?id=1-6L--pLuORDCKsYimi4xKcKUAIgGFcTU -O gpu_dataset.zip
unzip -q gpu_dataset.zip "bms_fold_0/train/**/*"
unzip -q gpu_dataset.zip "*.csv"

gdown https://drive.google.com/uc?id=1-Xz8IBl-dC5bxoIfn67bWXjJSFrm54SV -O tokenizer.pth