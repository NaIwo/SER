import json
import logging
import sys
from typing import List

import tensorflow as tf

from .decoder_loader import DecoderLoader
from .model_loader import ModelLoader


def speech_to_text(files: List[str], output_file: str):
    # load provided utils using torch.hub for brevity
    decoder_loader = DecoderLoader()
    logging.info('Getting decoder')
    decoder, utils = decoder_loader.get_decoder()

    read_batch, split_into_batches, read_audio, prepare_model_input = utils

    # load model from tf hub
    model_loader = ModelLoader()
    logging.info('Getting model')
    tf_model = model_loader.load_model()
    model = tf_model.signatures["serving_default"]

    batches = split_into_batches(files, batch_size=1)

    texts = []

    # process files
    for filename, batch in zip(batches, files):
        logging.info(f'Processing {filename}')
        data = prepare_model_input(read_batch(batch))
        res = model(tf.constant(data.numpy())[0])['output_0']
        text = decoder(torch.Tensor(res.numpy()))
        texts.append(dict(filename=filename, text=text))

    # save results to file
    logging.info('Saving results')
    with open(output_file, 'w') as f:
        f.write(json.dumps(texts))
    logging.info('Success!')


if __name__ == '__main__':
    filenames = sys.argv[1]
    output = sys.argv[2]
    speech_to_text(filenames, output)
