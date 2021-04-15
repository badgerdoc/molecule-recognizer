git clone --branch v2.7.0 'https://github.com/open-mmlab/mmdetection.git'
cd "mmdetection"
python setup.py install
pip install -r "requirements.txt"
cd -
pip install pillow==7.2.0
pip uninstall pycocotools -y
pip install mmpycocotools
pip install mmcv-full==1.2.1+torch1.7.0+cpu -f https://download.openmmlab.com/mmcv/dist/index.html

# overrides torch 1.4
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
