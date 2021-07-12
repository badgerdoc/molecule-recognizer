from pathlib import Path

from constants import *


class EffnetCfg:
    def __init__(self, name, img_size, feature_dim, batch_size):
        self.name = name
        self.img_size = img_size
        self.feature_dim = feature_dim
        self.batch_size = batch_size


effnet_b2 = EffnetCfg('efficientnet-b2', 260, 1408, batch_size=1)
effnet_b4 = EffnetCfg('efficientnet-b4', 380, 1792, batch_size=4)
effnet_b7 = EffnetCfg('efficientnet-b7', 600, 2560, batch_size=1)

effnet_v2_l = EffnetCfg('efficientnet_v2_l', None, 1792, batch_size=2)
# effnet_v2_xl = EffnetCfg('efficientnet_v2_xl', None, 1792, batch_size=1)

effnet_cfg = effnet_v2_l
scale = 1


class CFG:  # TODO: store config for each run

    # ====== Encoder config ======
    encoder_model = effnet_cfg.name
    # encoder_model = 'resnet18'
    # size = (effnet_cfg.img_size * scale, effnet_cfg.img_size * scale)
    # size = (500, 1200)
    size = (300, 400)
    # size = (450, 600)
    encoder_dim = effnet_cfg.feature_dim

    # ====== Decoder config ======
    # decoder_dim = effnet_cfg.feature_dim
    decoder_dim = 512
    attention_dim = 256
    embed_dim = 256
    max_len = 275
    # dropout=0.5
    dropout = 0.2

    # ====== Dataset config ======

    # Full dataset (only for CPU instance in Colab)
    # train_labels = '/content/bms-molecular-translation/train_labels.csv'
    # train_path = Path('/content/bms-molecular-translation/train')

    # Dataset for training on GPU
    train_labels = 'bms_fold_0/train.csv'
    train_path = Path('bms_fold_0/train')

    # ====== Training loop config ======
    continue_from = None
    # continue_from = 'latest'
    checkpoint_freq = 500  # steps
    latest_checkpoint = CHECKPOINT_PATH / EXPERIMENT_TITLE / effnet_cfg.name
    print_freq = 1
    num_workers = 2
    scheduler = 'CosineAnnealingLR'  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epochs = 1  # not to exceed 9h
    # factor=0.2 # ReduceLROnPlateau
    # patience=4 # ReduceLROnPlateau
    # eps=1e-6 # ReduceLROnPlateau
    T_max = 4  # CosineAnnealingLR
    # T_0=4 # CosineAnnealingWarmRestarts
    encoder_lr = 1e-4
    decoder_lr = 4e-4
    min_lr = 1e-6
    batch_size = effnet_cfg.batch_size
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 5
    seed = 1  # 42
    n_fold = 5
    trn_fold = [0]  # [0, 1, 2, 3, 4]
    train = True
