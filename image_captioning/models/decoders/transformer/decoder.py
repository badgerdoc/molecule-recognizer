import torch.nn as nn
from torch import Tensor

from image_captioning import register_model, ConfigurableModel
from image_captioning.models.decoders.transformer.config import CaptionTransformerConfig
from image_captioning.models.decoders.transformer.model import PositionalEncoding, PositionEncode2D, Swish, TokenEmbedding


@register_model(name='transformer-encoder-decoder')
class CaptionTransformer(nn.Module, ConfigurableModel):

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        features_size: int,
        emb_size: int,
        nhead: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super(CaptionTransformer, self).__init__()
        self.emb_size = emb_size

        self.project = nn.Sequential(
            nn.Conv2d(features_size, emb_size, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(emb_size),
            Swish()
        )

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.tgt_token_embedding = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.image_pos_encoding = PositionEncode2D(emb_size, 13, 10)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor
    ):
        batch_size = src.shape[0]
        src = self.project(src)
        src = self.image_pos_encoding(src)
        src = src.permute(0, 2, 3, 1)
        src = src.view(batch_size, -1, self.emb_size)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0)
        tgt_padding_mask = tgt_padding_mask.permute(1, 0)
        tgt_emb = self.positional_encoding(self.tgt_token_embedding(tgt))
        outs = self.transformer(src, tgt_emb, None, tgt_mask, None, None, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor = None):
        batch_size = src.shape[0]
        src = self.project(src)
        src = self.image_pos_encoding(src)
        src = src.permute(0, 2, 3, 1)
        src = src.reshape(-1, batch_size, self.emb_size)
        return self.transformer.encoder(src)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor = None):
        batch_size = tgt.shape[0]
        memory = memory.view(batch_size, -1, self.emb_size)
        tgt = tgt.permute(1, 0)
        return self.transformer.decoder(self.positional_encoding(self.tgt_token_embedding(tgt)), memory)

    @classmethod
    def from_config(cls, config: CaptionTransformerConfig):
        return cls(
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            features_size=config.features_size,
            emb_size=config.emb_size,
            nhead=config.nhead,
            tgt_vocab_size=config.tgt_vocab_size,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )
