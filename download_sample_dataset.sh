export DS_DIR="datasets"
export DS_ARCH="sample_ds.zip"
mkdir -p $DS_DIR
gdown "https://drive.google.com/uc?id=1b-F-_Evony3lgP0FKUWQz1P0PmFQyX0R" -O $DS_ARCH
unzip  -d $DS_DIR $DS_ARCH
rm $DS_ARCH
