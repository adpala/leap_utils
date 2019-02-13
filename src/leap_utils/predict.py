from typing import Sequence, Union
import logging

import keras
import numpy as np


def load_network(model_path: str, weights_path: str = None, image_size: Sequence = None) -> keras.Model:
    """Load network from file.

    Args:
        model_path - save with keras.save_model
        weights_path - optional - load custom weights
        image_size - optional - re-compile network to accept different image sizes
    Returns:
        keras model
    """
    if image_size and len(image_size) == 3:
        logging.warning(f'image_size should be 2D (width x height) but as {len(image_size)} values ({image_size}). Removing last dimension assuming it corresponds to the number of channels, which cannot be changed.')
        image_size = image_size[:-1]

    logging.info(f"loading model architecture from {model_path}")
    from keras.models import load_model as keras_load_model
    from keras.models import Model
    from keras.layers import Input
    m = keras_load_model(model_path)
    input_size = m.layers[0].input_shape[1:-1]
    input_channels = m.layers[0].input_shape[-1]
    if weights_path:
        logging.info(f"loading model weights from {weights_path}")
        m.load_weights(weights_path)
    if image_size and not np.all(tuple(image_size) == tuple(input_size)):
        logging.info(f"changing input image size from {input_size} to {image_size}")
        newInput = Input(batch_shape=(None, *image_size, input_channels))
        newOutputs = m(newInput)
        m = Model(newInput, newOutputs)
    return m


class BoxSequence(keras.utils.Sequence):
    """Returns batches of boxes."""

    def __init__(self, boxes: np.ndarray, batch_size: int) -> None:
        """Initialize sequence.

        Args:
            boxes: np.ndarray [nb_box, width, height, channels]
            batch_size: int
        """
        self.x = boxes
        self.batch_size = batch_size

    def __len__(self) -> int:
        """Get number of batches."""
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get batch at idx in boz sequence."""
        return self.x[idx * self.batch_size:(idx + 1) * self.batch_size, ...]


def predict_confmaps(network: Union[str, keras.Model], boxes: np.ndarray, batch_size: int = 100) -> np.ndarray:
    """Predict confidence maps from images.

    Args:
        model: keras model
        images: image - [nbox, w, h, chans]
    Returns:
        conf maps

    """
    if boxes.ndim is not 4:
        raise ValueError(f'`boxes` needs to be 4D - (nboxes, width, height, channels) - but has shape {boxes.shape}.')

    if isinstance(network, str):
        network = load_network(network, image_size=boxes.shape[1:4])  # this should return a compiled network

    input_size = network.input_shape[-3:-1]
    input_channels = network.input_shape[-1]
    if boxes.shape[-1] != input_channels:
        raise ValueError(f'the network expects {input_channels} channels but boxes have {boxes.shape[-1]}.')
    if boxes.shape[-3:-1] != input_size:
        raise ValueError(f'the network expects images of size {input_size} but boxes are of size {boxes.shape[-3:-1]}.')

    data_gen = BoxSequence(boxes, batch_size)
    confmaps = network.predict_generator(data_gen)
    return confmaps
