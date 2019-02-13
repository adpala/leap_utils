"""Train LEAP networks."""
from typing import Sequence
import numpy as np
from scipy.ndimage import gaussian_filter

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import keras
from keras_preprocessing import image as kp
import leap_utils.models
from leap_utils.predict import predict_confmaps
from leap_utils.postprocessing import process_confmaps_simple


class BoxMaskSequence(keras.utils.Sequence):
    """Returns batches of boxes."""

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 batch_size: int = 32, shuffle: bool = False,
                 hflip: bool = False, vflip: bool = False,
                 rg: float = 0, wrg: float = 0, hrg: float = 0,
                 zrg=None, brg=None) -> None:
        """Initialize sequence.

        Args:
            boxes: np.ndarray [nb_box, width, height, channels]
            batch_size (32): int
            shuffle (False)
            hflip: horizontal flip (along axis 1)
            vflip: vertical flip (along axis 0)
            rg: Rotation range, in degrees.
            wrg: Width shift range, as a float fraction of the width.
            hrg: Height shift range, as a float fraction of the height.
            zrg: Tuple of floats; zoom range (multiplier, e.g (0.9, 1.1)).
            brg: Tuple of floats; brightness range (multiplier, e.g (0.9, 1.1)).
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError('boxes and maps must have same size along first dimension.')
        self.x = x
        self.y = y
        self.n = x.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.hflip = hflip
        self.vflip = vflip
        self.rg = rg
        self.wrg = wrg
        self.hrg = hrg
        self.zrg = zrg
        self.brg = brg
        self._set_index_array()

    def on_epoch_end(self):
        self._set_index_array()

    def _set_index_array(self):
        self._index_array = np.arange(self.n)
        if self.shuffle:
            self._index_array = np.random.permutation(self.n)

    def _augment(self, x, y):
        for idx, (xx, yy) in enumerate(zip(x, y)):
            if self.hflip and np.random.rand() > 0.5:
                x[idx, ...] = kp.flip_axis(xx, axis=1)
                y[idx, ...] = kp.flip_axis(yy, axis=1)
            if self.vflip and np.random.rand() > 0.5:
                x[idx, ...] = kp.flip_axis(xx, axis=0)
                y[idx, ...] = kp.flip_axis(yy, axis=0)
            if self.brg is not None:
                # random_brightness per channel
                for chn in range(xx.shape[-1]):
                    u = np.random.uniform(*self.brg)
                    x[idx, ..., chn:chn + 1] = kp.apply_brightness_shift(xx[..., chn:chn + 1], u)
            if self.rg > 0 or self.wrg > 0 or self.hrg > 0 or self.zrg is not None:
                params = {'theta': 0, 'tx': 0, 'ty': 0, 'shear': 0, 'zx': 1, 'zy': 1}
                if self.rg > 0:
                    params['theta'] = np.random.uniform(-self.rg, self.rg)
                if self.wrg > 0 or self.hrg > 0:
                    h, w = xx.shape[0], xx.shape[1]
                    params['tx'] = np.random.uniform(-self.wrg, self.wrg) * w
                    params['ty'] = np.random.uniform(-self.hrg, self.hrg) * h
                if self.zrg is not None:
                    params['zx'] = np.random.uniform(*self.zrg, 1)[0]
                    params['zy'] = params['zx']
                x[idx, ...] = kp.apply_affine_transform(xx, **params)
                y[idx, ...] = kp.apply_affine_transform(yy, **params)
        return x, y

    def __len__(self) -> int:
        """Get number of batches."""
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get batch at idx in box sequence."""
        batch_idx = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_idx = batch_idx % self.n   # wrap around
        x = self.x[self._index_array[batch_idx], ...]
        y = self.y[self._index_array[batch_idx], ...]
        x, y = self._augment(x, y)
        return x, y


def points2mask(points: np.ndarray, size: Sequence, sigma: float = 5,
                normalize: bool = True, merge_channels: bool = False) -> np.ndarray:
    """Get stack of 2d map from points.

    Args:
        pts: points (npoints, 2)
        size: size of map
        sigma=2: blur factor
        normalize=True: scale each map to have a max of 1.0
        merge_channels=False: sum across channels
    Returns:
        mask: (size[0], size[1], npoints) or if merge_channels (size[0], size[1])
    """
    mask = np.zeros((size[0], size[1], points.shape[0]))
    for idx, point in enumerate(points):
        mask[point[0], point[1], idx] = 1
        mask[:, :, idx] = gaussian_filter(mask[:, :, idx], sigma=sigma)
        if normalize:
            mask[:, :, idx] /= np.max(mask[:, :, idx])
    if merge_channels:
        mask = np.sum(mask, axis=-1)
    return mask


def make_masks(points, size, sigma: float = 5,
               normalize: bool = True, merge_channels: bool = False):
    """Make masks from point sets.

    Args:
        points - [n, npoints, 2]
        size - [width, height]
    Returns
        masks = [n, width, height, npoints]

    """
    maps = np.zeros((points.shape[0], size[0], size[1], points.shape[1]))
    for idx, point_set in enumerate(points):
        maps[idx, ...] = points2mask(point_set, size)
    return maps


def initialize_network(image_size, output_channels, nb_filters: int = 64, network_type=leap_utils.models.leap_cnn):
    """Initialize LEAP network model."""
    m = network_type(image_size, output_channels, filters=nb_filters,
                     upsampling_layers=True, amsgrad=True, summary=True)
    m.compile(optimizer=Adam(amsgrad=True), loss="mean_squared_error")
    return m


def train_val_split(N, val_size=0.10, shuffle=True, seed=None):
    """Split datasets into training and validation sets."""
    if val_size < 1:
        val_size = int(np.round(N * val_size))

    idx = np.arange(N)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(idx)

    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    return train_idx, val_idx


def train_network(boxes, positions, save_weights_path,
                  batch_size: int = 32, epochs: int = 100,
                  val_size: float = 0.10, verbose: int = 1, flipall: bool = False,
                  sigma: float = 5, hflip: bool = False, vflip: bool = False,
                  rg: float = 0, wrg: float = 0, hrg: float = 0,
                  zrg=None, brg=None,
                  network_type=leap_utils.models.leap_cnn, seed=None):
    """Train LEAP network on boxes and positions.

    Args:
        boxes: [nb_boxes, width, height, color-channels]
        positions: [nb_boxes, nb_parts, x/y]
        batch_size:32
        epochs:100
        val_size:0.10
        verbose:1
        sigma: float = 6
        augmentation params:
            hflip=False, vflip=False,
            rg: float = 0, wrg: float = 0, hrg: float = 0,
            zrg=None, brg=None,
    Returns:
        fit history
    """
    box_size = boxes.shape[1:3]
    img_size = boxes.shape[1:4]
    nb_boxes = boxes.shape[0]

    maps = make_masks(positions, size=box_size, sigma=sigma)

    train_idx, val_idx = train_val_split(nb_boxes, val_size, seed=seed)
    G = BoxMaskSequence(boxes[train_idx, ...], maps[train_idx, ...],
                        batch_size=batch_size, hflip=hflip, vflip=vflip)
    G_val = BoxMaskSequence(boxes[val_idx, ...], maps[val_idx, ...],
                            batch_size=batch_size)

    m = initialize_network(image_size=img_size, output_channels=maps.shape[-1], network_type=network_type)
    m.save(f"{save_weights_path}.model")

    step_num = len(G)
    step_num_val = len(G_val)

    fit_hist = m.fit_generator(G, epochs=epochs, steps_per_epoch=step_num,
                               validation_data=G_val, validation_steps=step_num_val,
                               callbacks=[ModelCheckpoint(f"{save_weights_path}.best", save_best_only=True, verbose=verbose),
                                          EarlyStopping(patience=5),
                                          ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, cooldown=0,
                                                            min_delta=0.00001, min_lr=0.0, verbose=verbose)],)
    m.save_weights(f"{save_weights_path}.final")
    return fit_hist, train_idx, val_idx


def evaluate_network(network, boxes, positions, batch_size: int = 100):
    """Evaluate LEAP network on boxes and positions.

    Args:
        network
        boxes: [nb_boxes, width, height, color-channels]
        positions: [nb_boxes, nb_parts, x/y]
        batch_size:32
    Returns:
        mean_map_error: MSE between predicted and actual confmaps
        mean_position_error: MSE between prediction and actual positions
        position_error: MSE between prediction and actual positions

    """
    box_size = boxes.shape[1:3]
    confmaps = make_masks(positions, size=box_size)
    confmaps_predicted = predict_confmaps(network, boxes, batch_size)
    mean_map_error = np.sum(np.square(confmaps - confmaps_predicted))

    positions_predicted, confidence = process_confmaps_simple(confmaps)
    position_error = np.square(positions - positions_predicted)
    mean_position_error = np.mean(position_error)

    return mean_map_error, mean_position_error, position_error
