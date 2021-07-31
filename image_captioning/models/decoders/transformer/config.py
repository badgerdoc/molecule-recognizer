from image_captioning.base import DecoderBaseConfig


class CaptionTransformerConfig(DecoderBaseConfig):
    num_encoder_layers: int
    num_decoder_layers: int
    emb_size: int
    nhead: int
    tgt_vocab_size: int
    dim_feedforward: int = 512
    dropout: float = 0.1
