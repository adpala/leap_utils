import numpy as np
import skimage.filters
import skimage.feature

from .utils import it, max2d

from leap_utils.preprocessing import normalize_matlab_boxes
import h5py
import logging

# TODO:
# - return max values
# - "overload" max2d via num_peaks argument?
# - reorganize and to utils


def max2d_multi(mask: np.ndarray, num_peaks: int, smooth: float = None,
                exclude_border: bool = True, min_distance: int = 4) -> (np.ndarray, np.ndarray):
    """Detect one or multiple peaks in each channel of an image."""
    maxima = np.ndarray((2, num_peaks, mask.shape[-1]))
    for idx, plane in enumerate(it(mask, axis=-1)):
        if smooth:
            plane = skimage.filters.gaussian(plane, smooth)
        tmp = skimage.feature.peak_local_max(plane, num_peaks=num_peaks, exclude_border=exclude_border, min_distance=min_distance)
        maxima[..., idx] = tmp
    return maxima


def max_simple(confmap: np.ndarray) -> (np.ndarray, np.ndarray):
    """Detect maxima in each layer of the confmap.

    prob is val at max pos, normalized by sum over all values in respective layer.
    Arguments:
        confmap = [width, height, layers]
    Returns:
        locs - [layers, 2]
        probs - [layers, 1]

    """
    if confmap.ndim != 3:
        raise ValueError(f'input should be 3D (width, height, layers) but is {confmap.shape}.')

    nb_parts = confmap.shape[-1]
    peak_loc = np.zeros((nb_parts, 2))
    peak_prb = np.zeros((nb_parts, 1))

    for part_idx in range(nb_parts):
        peak_loc[part_idx, ...], peak_amp = max2d(confmap[..., part_idx])
        peak_prb[part_idx, ...] = peak_amp / np.sum(confmap[..., part_idx])

    return peak_loc, peak_prb


def process_confmaps_simple(confmaps: np.ndarray) -> (np.ndarray, np.ndarray):
    """Simply take the max.

    Arguments:
        confmaps = [nbmaps x width x height x layers]
    Returns:
        positions = [nbmaps, layers, 2]
        confidence =  [nbmaps, layers, 1]

    """
    nb_maps, nb_parts = confmaps.shape[0], confmaps.shape[-1]
    positions = np.zeros((nb_maps, nb_parts, 2))
    confidence = np.zeros((nb_maps, nb_parts, 1))
    for nb, confmap in enumerate(confmaps):
        positions[nb, ...], confidence[nb, ...] = max_simple(confmap)
    return positions, confidence


def process_confmaps_bayesian(confmaps: np.ndarray, prior_information) -> (np.ndarray, np.ndarray):
    """Take all local maxime (using skimage), choose maximum based in prio info and confidence."""
    positions = None
    confidence = None
    return positions, confidence


def load_labels(labelsPath: str = '/#Common/chainingmic/leap/training_data/big_dataset_train.labels.mat'):
    """ Load labels data from the *.mat file created from LEAP interface, selecting only fully labeled boxes

    Arguments:
        labelsPath - path to the *.label.mat file

    Returns:
        positions - [nboxes, nbodyparts, 2]
        initialization - [nboxes, 1]
        boxes - [nboxes, width, height, channels] already 'normalized' from matlab to python format
    """

    logging.info(f'   loading labels from: {labelsPath}.')
    f = h5py.File(labelsPath, 'r')
    # contains: boxPath, config, createdOn, history, initialization, lastModified, positions, savePath, session, skeleton, skeletonPath
    boxPath = "".join([chr(item) for item in f['boxPath']])
    initialization = f['initialization']
    positions = f['positions']
    logging.info(f'   found {positions.shape[0]} sets of positions.')

    logging.info(f'   loading boxes from: {boxPath}.')
    boxf = h5py.File(boxPath, 'r')
    boxes = boxf['box']
    logging.info(f'   found {boxes.shape[0]} boxes.')

    if positions.shape[0] != boxes.shape[0]:
        logging.error(f'   data dimensions do not match.')

    status = np.all(~np.isnan(positions), (1, 2))  # True if box has been fully labeled

    if np.sum(status) != positions.shape[0]:
        logging.info(f'   not all positions in file have been labeled. Selecting labeled data from file.')
        initialization = initialization[status, ...]
        positions = positions[status, ...]
        boxes = boxes[status, ...]
    else:
        logging.info(f'   all data in file has been labeled.')

    boxes = normalize_matlab_boxes(boxes)
    f.close()
    boxf.close()
    return positions, initialization, boxes
