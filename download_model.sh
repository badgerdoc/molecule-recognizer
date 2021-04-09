export MODEL_FILE="gen.pkl"
export MODEL_DIR="models"
mkdir -p $MODEL_DIR
gdown "https://drive.google.com/uc?id=1qjRmpjeeacDc_NyZt80coUc7F67uCM6g" -O $MODEL_FILE
mv $MODEL_FILE $MODEL_DIR
