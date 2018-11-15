from typing import Sequence, Union, List
import numpy as np
from skimage.transform import rotate as sk_rotate

from videoreader import VideoReader


def crop_frame(frame: np.array, center: np.uintp, box_size: np.uintp, mode: str='clip') -> np.array:
    """

    frame: np.array, center: np.uintp, box_size: np.uintp, mode: str='clip'
    """
    box_hw_left = np.ceil(box_size/2)
    box_hw_right = np.floor(box_size/2)
    x_px = np.arange(center[0]-box_hw_left[0], center[0]+box_hw_right[0]).astype(np.intp)
    y_px = np.arange(center[1]-box_hw_left[1], center[1]+box_hw_right[1]).astype(np.intp)
    return frame.take(y_px, mode=mode, axis=1).take(x_px, mode=mode, axis=0)


def export_boxes(vr: VideoReader, box_centers: np.array, box_size: List[int],
                 frame_numbers: Sequence=None, box_angles: np.array=None) -> (np.array, np.array, np.array):
    """ Export boxes...

    Args:
        vr: VideoReader istance
        frame_numbers: list or range or frames - if omitted (or None) will read all frames
        box_size: [width, height]
        box_centers: [nframes in vid, flyid, 2]
        box_angles: [nframes in vid, flyid, 1], if not None, will rotate flies
    Returns:
         boxes
         fly_id: fly id for each box
         fly_frame: frame number for each box
    """
    if frame_numbers is None:
        frame_numbers = range(vr.number_of_frames)

    nb_frames = len(frame_numbers)
    nb_flies = box_centers.shape[1]
    nb_boxes = nb_frames*nb_flies
    # check input:

    # make this a dict?
    boxes = np.zeros((nb_boxes, *box_size, vr.frame_channels), dtype=np.uint8)
    fly_id = -np.ones((nb_boxes,), dtype=np.intp)
    fly_frame = -np.ones((nb_boxes,), dtype=np.intp)

    box_idx = -1
    for frame_number in frame_numbers:
        frame = vr[frame_number]
        for fly_number in range(nb_flies):
            box_idx += 1
            fly_id[box_idx] = fly_number
            fly_frame[box_idx] = frame_number
            box = crop_frame(frame, box_centers[frame_number, fly_number, :], box_size)
            if box_angles is not None:
                box = sk_rotate(box, box_angles[frame_number, fly_number, :],
                                resize=True, mode='edge', preserve_range=True)
            boxes[box_idx, ...] = box

    return boxes, fly_id, fly_frame