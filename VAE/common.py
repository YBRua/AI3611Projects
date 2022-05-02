import logging
from datetime import datetime


def setup_logger(args) -> logging.Logger:
    logger = logging.getLogger('LM')
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(message)s")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    IS_AE = "AE" if args.ae else "VAE"
    fname = (
        './logs/'
        f'{timestamp}-{IS_AE}-{args.encoder}-{args.decoder}-{args.z_dim}.log')
    file_handler = logging.FileHandler(
        filename=fname,
        mode='a',
        encoding='utf-8')
    file_formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s]: %(message)s")

    stream_handler.setFormatter(stream_formatter)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
