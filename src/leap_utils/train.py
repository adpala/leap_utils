"""Train LEAP networks."""
from typing import Sequence
import numpy as np
from scipy.ndimage import gaussian_filter

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import keras
from keras_preprocessing import image as kp
import leap.models


class BoxMaskSequence(keras.utils.Sequence):
    """Returns batches of boxes."""

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool=False,
                 hflip: bool = False, vflip: bool = False,
                 rg: float = 0, wrg: float = 0, hrg: float = 0,
                 zrg=None, brg=None) -> None:
        """Initialize sequence.

        Args:
            boxes: np.ndarray [nb_box, width, height, channels]
            batch_size (32): int
            shuffle (False)
            hflip: flip along axis 0
            vflip: flip along axis 1
            rg: Rotation range, in degrees.
            wrg: Width shift range, as a float fraction of the width.
            hrg: Height shift range, as a float fraction of the height.
            zrg: Tuple of floats; zoom range for width and height.
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
        print(self._index_array)

    def on_epoch_end(self):
        self._set_index_array()

    def _set_index_array(self):
        self._index_array = np.arange(self.n)
        if self.shuffle:
            self._index_array = np.random.permutation(self.n)

    def _augment(self, x, y):
        for idx, (xx, yy) in enumerate(zip(x, y)):
            if self.hflip and np.random.rand() > 0.5:
                x[idx, ...] = kp.flip_axis(xx, axis=0)
                y[idx, ...] = kp.flip_axis(yy, axis=0)
            if self.vflip and np.random.rand() > 0.5:
                x[idx, ...] = kp.flip_axis(xx, axis=1)
                y[idx, ...] = kp.flip_axis(yy, axis=1)
            if self.brg is not None:
                # random_brightness only takes 3-channel inputs but maps have more poses
                # so we need to do this channelwise
                for chn in range(xx.shape[-1]):
                    u = np.random.uniform(*self.brg)
                    x[idx, ..., chn:chn+1] = kp.apply_brightness_shift(xx[..., chn:chn+1], u)
                    y[idx, ..., chn:chn+1] = kp.apply_brightness_shift(yy[..., chn:chn+1], u)
            # neutral rot, shift, zoom values
            if self.rg > 0 or self.wrg > 0 or self.hrg > 0 or self.zrg is not None:
                # TODO: make dict and unpack as kwarg
                theta = 0
                tx = 0
                ty = 0
                shear = 0
                zx = 1
                zy = 1
                if self.rg > 0:
                    theta = np.random.uniform(-self.rg, self.rg)
                if self.wrg > 0 or self.hrg > 0:
                    h, w = xx.shape[0], xx.shape[1]
                    tx = np.random.uniform(-self.hrg, self.hrg) * h
                    ty = np.random.uniform(-self.wrg, self.wrg) * w
                if self.zrg is not None:
                    zx, zy = np.random.uniform(*self.zrg, 2)
                x[idx, ...] = kp.apply_affine_transform(xx, theta=theta, tx=tx, ty=ty, shear=0, zx=zx, zy=zy)
                y[idx, ...] = kp.apply_affine_transform(yy, theta=theta, tx=tx, ty=ty, shear=0, zx=zx, zy=zy)
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


def points2mask(points: np.ndarray, size: Sequence, sigma: float = 2,
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


def make_masks(points, size, sigma: float = 2,
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


def initialize_network(image_size, output_channels, nb_filters: int = 32):
    """Initialize LEAP network model."""
    m = leap.models.leap_cnn(image_size, output_channels, filters=nb_filters,
                             upsampling_layers=True, amsgrad=True, summary=True)
    m.compile(optimizer=Adam(amsgrad=True), loss="mean_squared_error")
    return m


def train_val_split(N, val_size=0.10, shuffle=True):
    """Split datasets into training and validation sets."""
    if val_size < 1:
        val_size = int(np.round(N * val_size))

    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)

    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    return train_idx, val_idx


def train_network(boxes, positions, save_weights_path,
                  batch_size: int = 32, epochs: int = 50,
                  val_size: float = 0.10, verbose: int = 1):
    """Train LEAP network on boxes and positions.

    Args:
        boxes
        positions
        batch_size:32
        epochs:50
        val_size:0.10
        verbose:1
    Returns:
        fit history
    """
    box_size = boxes.shape[1:3]
    nb_boxes = boxes.shape[0]

    maps = make_masks(positions, size=box_size)

    train_idx, val_idx = train_val_split(nb_boxes, val_size)
    G = BoxMaskSequence(boxes[train_idx, ...], maps[train_idx, ...])
    G_val = BoxMaskSequence(boxes[val_idx, ...], maps[val_idx, ...])

    m = initialize_network(image_size=boxes.shape[1:3], output_channels=maps.shape[-1])
    m.save(f"{save_weights_path}.model")

    step_num = int(nb_boxes / batch_size)
    fit_hist = m.fit_generator(G, epochs=epochs, steps_per_epoch=step_num,
                               validation_data=G_val, validation_steps=step_num,
                               callbacks=[ModelCheckpoint(f"{save_weights_path}.best", save_best_only=True, verbose=verbose),
                                          EarlyStopping(patience=20),
                                          ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, cooldown=0,
                                                            min_delta=0.00001, min_lr=0.0, verbose=verbose)])
    m.save_weights(f"{save_weights_path}.final")
    return fit_hist


def evaluate_network(network, boxes, positions):
    pass
