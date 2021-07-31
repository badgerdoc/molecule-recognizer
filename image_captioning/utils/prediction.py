from typing import Union

import cv2
import torch
import numpy as np
from pathlib import Path

from image_captioning.utils.training import get_transforms
from image_captioning.tokenizer import Tokenizer
from image_captioning.utils.helpers import load_config, load_checkpoint
from image_captioning.constants import ENCODER_CONFIG_YML


def greedy_decode(decoder, feature, max_len, tokenizer, device) -> str:
    # TODO: avoid duplication of this function when evaluation pipeline is merged
    feature = feature.to(device)
    memory = decoder.encode(feature)
    ys = torch.ones(1, 1).fill_(tokenizer.sos_idx).type(torch.long).to(device)
    for i in range(max_len - 1):
        out = decoder.decode(ys, memory, None)
        out = out.transpose(0, 1)
        prob = decoder.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word)], dim=0)
        if next_word == tokenizer.eos_idx:
            break
    return "".join([tokenizer.predict_caption(ys[i].tolist()) for i in range(len(ys))]).replace(tokenizer.sos, "")


def predict_caption(
    img_path: Union[str, Path], encoder, decoder, image_transforms, tokenizer, device, num_tokens=300
) -> str:
    # Convert path to str, required for cv2 to read the file
    img_path = str(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    transformed_img = image_transforms(image=img)['image']
    shp = transformed_img.shape
    transformed_img = transformed_img.view(1, shp[0], shp[1], shp[2])
    feature = encoder(transformed_img)
    return greedy_decode(decoder, feature, max_len=num_tokens, tokenizer=tokenizer, device=device)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Absolute paths should be used
    project_dir = Path('/Users/Egor_Osinkin/projects/molecule-recognizer/')
    tokenizer_path = project_dir / 'tokenizer.pth'
    checkpoint_path = project_dir / 'workdir/checkpoints/effnetv2_l_300x400_transformer-encoder-decoder/latest/'
    sample_img_path = project_dir / 'bms_fold_0/train/0/0/0/000b73470c57.png'

    tokenizer: Tokenizer = torch.load(tokenizer_path)
    encoder, decoder = load_checkpoint(checkpoint_path, device)
    encoder_config = load_config(checkpoint_path / ENCODER_CONFIG_YML)
    image_transforms = get_transforms(encoder_config)

    print(predict_caption(img_path=sample_img_path, encoder=encoder, decoder=decoder,
                          image_transforms=image_transforms, tokenizer=tokenizer, device=device))


if __name__ == '__main__':
    main()
