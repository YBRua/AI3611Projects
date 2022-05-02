from .ae import AE
from .vae import VAE
from .encoder import MLPEncoder, MLP3Encoder, ConvEncoder
from .decoder import MLPDecoder, MLP3Decoder, ConvDecoder


def get_encoder(encoder: str, z_dim: int):
    if encoder == 'MLP':
        return MLPEncoder(28 * 28, 512, z_dim)
    elif encoder == 'MLP3':
        return MLP3Encoder(28 * 28, 512, 256, z_dim)
    elif encoder == 'CONV':
        return ConvEncoder(z_dim)
    else:
        raise ValueError(f'Encoder {encoder} not found')


def get_decoder(decoder: str, z_dim: int):
    if decoder == 'MLP':
        return MLPDecoder(z_dim, 512, 28 * 28)
    elif decoder == 'MLP3':
        return MLP3Decoder(z_dim, 256, 512, 28 * 28)
    elif decoder == 'CONV':
        return ConvDecoder(z_dim)
    else:
        raise ValueError(f'Decoder {decoder} not found')


def get_vae(encoder_arch: str, decoder_arch: str, z_dim: int):
    encoder = get_encoder(encoder_arch, z_dim)
    decoder = get_decoder(decoder_arch, z_dim)
    return VAE(encoder, decoder)


def get_ae(encoder_arch: str, decoder_arch: str, z_dim: int):
    encoder = get_encoder(encoder_arch, z_dim)
    decoder = get_decoder(decoder_arch, z_dim)
    return AE(encoder, decoder)
