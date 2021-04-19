export GAN_MODEL_FILE="gen_8.pkl"
export MPRNET_MODEL_FILE="mprnet.pth"
export DET_MODEL_FILE="epoch_15.pth"
export MODEL_DIR="models"
mkdir -p $MODEL_DIR

gdown "https://drive.google.com/uc?id=1-BJnA3-hy6Xoe8HUD1j99HKbN9tRUZ2E" -O $GAN_MODEL_FILE
mv $GAN_MODEL_FILE $MODEL_DIR

gdown "https://drive.google.com/uc?id=1-kq3ycew6UaqoUqPaqM20sjbIwL1QPey" -O $DET_MODEL_FILE
mv $DET_MODEL_FILE $MODEL_DIR

gdown "https://drive.google.com/uc?id=1-jDaPmnduoP_fSnA967WjpAi0WWyQssv" -O $MPRNET_MODEL_FILE
mv $MPRNET_MODEL_FILE $MODEL_DIR
