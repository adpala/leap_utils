import os
import defopt
import logging
import deepdish as dd
import numpy as np
from videoreader import VideoReader
from leap_utils.preprocessing import export_boxes, angles, normalize_boxes, detect_bad_boxes_by_angle, fix_orientations
from leap_utils.postprocessing import process_confmaps_simple
from leap_utils.predict import predict_confmaps, load_network
from leap_utils.utils import iswin, ismac, flatten, unflatten

# limit tf threads
from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

if iswin():
    root = 'Z:/#Common/'
elif ismac():
    root = '/Volumes/ukme04/#Common/'
else:
    root = '/scratch/clemens10/'

res_path = root+'chainingmic/res'
network_path = root+'chainingmic/leap/best_model.h5'
priors_path = root+'chainingmic/leap/priors.h5'


def main(trackfilename: str, *, frame_start: int = 0, frame_stop: int = None, frame_step: int = 1, batch_size: int = 100, save_interval: int = 100, start_over: bool = False):

    expID = os.path.basename(trackfilename).partition('_tracks.h5')[0]

    track_path = f"{res_path}/{expID}/{expID}_tracks.h5"

    data_path = root+'chainingmic/dat'
    video_path = f"{data_path}/{expID}/{expID}.mp4"
    if not os.path.exists(video_path):
        data_path = root+'chainingmic/dat.processed'
        video_path = f"{data_path}/{expID}/{expID}.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError('video file not found.')

    trackfixed_path = f"{res_path}/{expID}/{expID}_tracks_fixed.h5"
    pose_path = f"{res_path}/{expID}/{expID}_poses.h5"

    try:
        os.mkdir(os.path.dirname(trackfixed_path))
    except FileExistsError as e:
        logging.debug(e)

    if not os.path.exists(trackfixed_path):
        logging.info(f"   fixing tracks.")
        fix_tracks(track_path, trackfixed_path)
    else:
        logging.info(f"   fixed tracks exist")

    logging.info(f"   opening video {video_path}.")
    vr = VideoReader(video_path)

    logging.info(f"   loading tracks from {trackfixed_path}.")
    try:
        data = dd.io.load(trackfixed_path)
        centers = data['centers'][:]    # nframe, channel, fly id, coordinates
        tracks = data['lines']
        chbb = data['chambers_bounding_box'][:]
        heads = tracks[:, 0, :, 0, ::-1]   # nframe, fly id, coordinates
        tails = tracks[:, 0, :, 1, ::-1]   # nframe, fly id, coordinates
        heads = heads + chbb[1][0][:]   # nframe, fly id, coordinates
        tails = tails + chbb[1][0][:]   # nframe, fly id, coordinates
        box_angles = angles(heads, tails)
        box_centers = centers[:, 0, :, :]   # nframe, fly id, coordinates
        box_centers = box_centers + chbb[1][0][:]
        nb_flies = box_centers.shape[1]
        logging.info(f"   nflies: {nb_flies}.")
    except OSError as e:
        logging.error(f'   could not load tracks.')
        logging.debug(e)

    if frame_stop is None:
        frame_stop = data['frame_count']
        logging.info(f'   Setting frame_stop: {frame_stop}.')
    frame_numbers = range(frame_start, frame_stop, frame_step)
    batch_idx = list(range(0, len(frame_numbers), batch_size))
    batch_idx.append(frame_stop-frame_start-1)        # ADRIAN TESTING THINGS, SHOULD BE REMOVED IF COMMITED ####################################
    logging.info(f"   frame range: {frame_start}:{frame_stop}:{frame_step}.")
    logging.info(f"   processing frames in {len(batch_idx)-1} batches of size {batch_size}.")

    box_size = [120, 120]
    network = load_network(network_path, image_size=box_size)
    priors_data = dd.io.load(priors_path)
    priors = priors_data['priors']
    nb_frames = len(frame_numbers)
    nb_parts = network.output_shape[-1]
    nb_boxes = nb_frames * nb_flies
    logging.info(f"   loading network from {network_path}.")

    too_small = False
    if not start_over and os.path.exists(pose_path):
        logging.info(f'loading existing results from {pose_path}.')
        d0 = dd.io.load(pose_path)
        if 'last_saved_batch' not in d0:
            logging.info(f"   existing results are from an old version - starting over.")
            start_over = True
        elif d0['positions'].shape[0] < nb_boxes:
            logging.info(f"   results structure too small for nboxes: {d0['positions'].shape[0]} < {nb_boxes} - starting and copying over.")
            start_over = True
            too_small = True
        else:
            d = d0
            del(d0)
            logging.info(f"   continuing from batch {d['last_saved_batch']}.")

    if start_over or not os.path.exists(pose_path):
        logging.info(f'initializing new results dictionary.')
        d = {'positions': np.zeros((nb_boxes, nb_parts, 2), dtype=np.uint16),
             'confidence': np.zeros((nb_boxes, nb_parts, 1), dtype=np.float16),
             'expID': expID,
             'fixed_angles': np.zeros((nb_boxes, 1), dtype=np.float16),
             'frame_numbers': frame_numbers,
             'fly_id': np.zeros((nb_boxes,), dtype=np.uintp),
             'fly_frame': np.zeros((nb_boxes,), dtype=np.uintp),
             'bad_boxes': np.zeros((nb_boxes, 1), dtype=np.bool),
             'last_saved_batch': 0,
             'prior_idxs': np.zeros((nb_boxes,), dtype=np.bool)}

    if too_small:
        logging.info(f'   copying old results to new results dictionary.')
        last_box = batch_idx[d['last_saved_batch']] * nb_flies
        d['positions'][:last_box, ...] = d0['positions'][:last_box, ...]
        d['confidence'][:last_box, ...] = d0['confidence'][:last_box, ...]
        d['fixed_angles'][:last_box, ...] = d0['fixed_angles'][:last_box, ...]
        d['bad_boxes'][:last_box, ...] = d0['bad_boxes'][:last_box, ...]
        d['fly_id'][:last_box] = d0['fly_id'][:last_box]
        d['fly_frame'][:last_box] = d0['fly_frame'][:last_box]
        d['prior_idxs'][:last_box] = d0['prior_idxs'][:last_box]
        del(d0)

    for batch_num in range(d['last_saved_batch'], len(batch_idx)-1):
        logging.info(f"PROCESSING BATCH {batch_num}.")
        batch_frame_numbers = list(range(frame_numbers[batch_idx[batch_num]], frame_numbers[batch_idx[batch_num+1]]))
        batch_box_numbers = list(range(batch_idx[batch_num]*nb_flies, batch_idx[batch_num+1]*nb_flies))
        logging.info(f"   loading frames {batch_frame_numbers[0]}:{batch_frame_numbers[-1]}.")
        frames = [frame[:, :, :1] for frame in vr[batch_frame_numbers]]  # keep only one color channel
        d['positions'][batch_box_numbers, ...], d['confidence'][batch_box_numbers, ...], _, d['bad_boxes'][batch_box_numbers, ...], d['fly_id'][batch_box_numbers], d['fly_frame'][
            batch_box_numbers], _, d['fixed_angles'][batch_box_numbers, ...], d['prior_idxs'][batch_box_numbers] = process_batch(network, frames, box_centers[batch_frame_numbers, ...], box_angles[batch_frame_numbers, ...], box_size, priors)

        if batch_num % save_interval == 0:
            logging.info(f"   saving intermediate results after {batch_num} to {pose_path}.")
            dd.io.save(pose_path, d)
            d['last_saved_batch'] = batch_num

    logging.info(f"   saving poses to {pose_path}.")
    dd.io.save(pose_path, d)


def process_batch(network, frames, box_centers, box_angles, box_size, priors):
    logging.info(f"   exporting boxes.")
    boxes, fly_id, fly_frame = export_boxes(frames, box_centers, box_size=box_size, box_angles=box_angles)
    logging.info(f"   predicting confidence maps for {boxes.shape[0]} boxes.")
    confmaps = predict_confmaps(network, normalize_boxes(boxes))
    logging.info(f"   processing confidence maps.")
    positions, confidence = process_confmaps_simple(confmaps)

    logging.info(f"   recalculating box angles.")
    # TODO: make positions self-documenting (named_list), make these args
    head_idx = 0
    tail_idx = 11
    nb_flies = box_angles.shape[1]

    newbox_angles, bad_boxes = detect_bad_boxes_by_angle(unflatten(positions[:, head_idx, :], nb_flies),
                                                         unflatten(positions[:, tail_idx, :], nb_flies),
                                                         epsilon=5)
    bad_frame_idx = np.any(bad_boxes, axis=1)[:, 0]   # for addressing bad boxes by frame
    bad_box_idx = np.repeat(bad_frame_idx, nb_flies)   # for addressing bad boxes by box
    fixed_angles = box_angles
    if np.sum(bad_boxes) > 0:
        logging.info(f"   found {np.sum(bad_boxes)} cases of boxes with angles above threshold.")
        logging.info(f"      re-exporting the bad boxes.")

        fixed_angles[bad_frame_idx, ...] = box_angles[bad_frame_idx, ...] + newbox_angles[bad_frame_idx, ...]
        boxes[bad_box_idx, ...], *_ = export_boxes([frames[int(idx)] for idx in np.where(bad_frame_idx)[0]],
                                                   box_centers[bad_frame_idx, ...],
                                                   box_size=np.array([120, 120]),
                                                   box_angles=fixed_angles[bad_frame_idx, ...])
        logging.info(f"      re-doing predictions.")
        confmaps[bad_box_idx, ...] = predict_confmaps(network, normalize_boxes(boxes[bad_box_idx, ...]))
        logging.info(f"      re-processing confidence maps.")
        positions[bad_box_idx, ...], confidence[bad_box_idx, ...] = process_confmaps_simple(confmaps[bad_box_idx, ...])
    # all results should be in nb_boxes format
    fixed_angles = flatten(fixed_angles)
    bad_boxes = flatten(bad_boxes)

    # Detection of errors for prior, according to position thresholds
    error_matrix = detect_prior_cases(positions)
    priorcase_idx = np.where(np.any(error_matrix, axis=1))[0]
    logging.info(f"   number of boxes to process with priors: {priorcase_idx.shape[0]}")
    # Try to fix positions applying priors
    positions[priorcase_idx, ...] = priors_processing(positions[priorcase_idx, ...], error_matrix[priorcase_idx, ...], boxes[priorcase_idx, ...], confmaps[priorcase_idx, ...], priors)
    # Re-evaluate error matrix
    new_error_matrix = detect_prior_cases(positions)
    new_idxs = np.where(np.any(new_error_matrix, axis=1))[0]
    logging.info(f"   number of boxes still unfixed after priors: {new_idxs.shape[0]}")

    return positions, confidence, confmaps, bad_boxes, fly_id, fly_frame, boxes, fixed_angles, np.any(error_matrix, axis=1)


def fix_tracks(track_file_name: str, save_file_name: str):
    """Load data, call fix_orientations and save data."""
    logging.info(f"   processing tracks in {track_file_name}. will save to {save_file_name}")
    data = dd.io.load(track_file_name)
    data['lines'] = fix_orientations(data['lines'])
    logging.info(f"   saving chaining data to {save_file_name}")
    dd.io.save(save_file_name, data)


########################## Adrian functions for priors ###################################

def detect_prior_cases(positions, epsilon=0):

    # Thresholds for error detection, values from 29.12.2018:
    bp_thresholds = np.zeros((12, 2, 2))    # [bodypart, x/y, min/max]
    # x min
    bp_thresholds[:, 1, 0] = [50, 40, 20, 10, 20, 50, 60, 50, 50, 10, 50, 50]
    # x max
    bp_thresholds[:, 1, 1] = [80, 70, 70, 60, 70, 100, 110, 100, 80, 70, 110, 80]
    # y min
    bp_thresholds[:, 0, 0] = [10, 25, 5, 25, 50, 5, 25, 50, 40, 60, 60, 60]
    # y max
    bp_thresholds[:, 0, 1] = [60, 75, 60, 100, 100, 60, 100, 100, 90, 120, 120, 100]

    error_matrix = np.zeros(positions.shape[0:2], dtype=bool)
    for bp in range(12):
        for jj in range(2):
            cond1 = positions[:, bp, jj] > (bp_thresholds[bp, jj, 1] - epsilon)
            cond2 = positions[:, bp, jj] < (bp_thresholds[bp, jj, 0] + epsilon)
            error_matrix[:, bp] = np.any([error_matrix[:, bp], cond1, cond2], axis=0)

    return error_matrix


def max2d_4priors(mask: np.ndarray, num_peaks: int, axis: int = 0, smooth: float = None,
                  exclude_border: bool = True, min_distance: int = 10) -> (np.ndarray, np.ndarray):
    """Detect one or multiple peaks in each channel of an image."""

    import skimage.filters
    from skimage.feature import peak_local_max
    from leap_utils.utils import it

    maxima = np.ndarray((num_peaks, 2, mask.shape[axis]), dtype=int)
    for idx, plane in enumerate(it(mask, axis=axis)):
        if smooth:
            plane = skimage.filters.gaussian(plane, smooth)
        tmp = peak_local_max(plane, num_peaks=num_peaks, exclude_border=exclude_border, min_distance=min_distance)
        if tmp.shape[0] == maxima.shape[0]:
            maxima[..., idx] = tmp
        else:
            maxima[:tmp.shape[0], :, idx] = tmp
            maxima[tmp.shape[0]:, :, idx] = 0
    return maxima


def priors_processing(input_pos, input_errors, input_boxes, input_cm, priors):

    # Initialize fixed positions
    output_pos = np.copy(input_pos).astype(np.int)

    # Loop through boxes with detected errors
    for ii in range(input_pos.shape[0]):

        # Select only body parts with detected errors
        bp_to_prior = np.where(input_errors[ii, :])[0]

        # Get peaks from confidence map
        maxima = max2d_4priors(input_cm[ii, :, :, bp_to_prior], num_peaks=5)
        maxima_val = input_cm[ii, maxima[:, 0, :], maxima[:, 1, :], bp_to_prior]

        # Loop through each body part with detected error
        for jj, bp in enumerate(bp_to_prior):
            bayes_score = np.multiply(maxima_val[:, jj], priors[maxima[:, 0, jj], maxima[:, 1, jj], bp])
            output_pos[ii, bp, :] = maxima[np.argmax(bayes_score), :, jj]

    return output_pos


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    defopt.run(main)
