import os
import defopt
import logging
import deepdish as dd
import numpy as np
from videoreader import VideoReader
from leap_utils.preprocessing import export_boxes, get_angles, normalize_boxes, detect_bad_boxes_byAngle, adrian_fix_tracks
from leap_utils.postprocessing import process_confmaps_simple
from leap_utils.predict import predict_confmaps
from leap_utils.plot import vplay

"""
TODO: Paths should be in '/' logic instead of real-strings, to be fixed soon.
TODO: Currently having boxes flipped when using our own full train model,
with some mistakes on head-tail flip, compared to fast train model from matlab
Last updated: 22.11.2018 - 14:41 [Adrian]
"""

## Plot stuff
play_boxes = True
doprediction = True

## Paths
dataPath = r'Z:\#Common\chainingmic\dat.processed'
resPath = r'Z:\#Common\chainingmic\res'
path = r'Z:\#Common\adrian\Workspace\projects\preprocessing\src\preprocessing'
networkPath = r'Z:\#Common\adrian\Workspace\temp\181029_122243-n=1450\best_model.h5'  # Model from fast train
# networkPath = r'Z:\#Common\adrian\Workspace\temp\leapermodels\chainingmic_2018-11-20_17-33.best.h5' # Last full training model


def main(expID: str = 'localhost-20180720_182837', frame_start: int = 1000, frame_stop: int = 2000, frame_step: int = 100):

    ## Fix tracks
    # Paths
    trackPath = f"{resPath}\{expID}\{expID}_tracks.h5"
    videoPath = f"{dataPath}\{expID}\{expID}.mp4"
    trackfixedPath = f"{resPath}\{expID}\\{expID}_tracks_fixed.h5"
    savingPath = f"{resPath}\{expID}"
    posePath = f"{savingPath}\{expID}_pose.h5"
    fixedBoxesPath = f"{savingPath}\{expID}_fixedBoxes.h5"

    # Do not fix if they are already fixed
    if not os.path.exists(trackfixedPath):
        logging.info(f"   doing adrian_fix_tracks")
        adrian_fix_tracks(trackPath, trackfixedPath)
    else:
        logging.info(f"   fixed tracks already exist")

    ## Load video
    logging.info(f"   loading video from {videoPath}.")
    vr = VideoReader(videoPath)
    ## Load track
    logging.info(f"   loading tracks from {trackfixedPath}.")
    has_tracks = False
    try:
        data = dd.io.load(trackfixedPath)
        centers = data['centers'][:]    # nframe, channel, fly id, coordinates
        tracks = data['lines']
        chbb = data['chambers_bounding_box'][:]
        heads = tracks[:, 0, :, 0, ::-1]   # nframe, fly id, coordinates
        tails = tracks[:, 0, :, 1, ::-1]   # nframe, fly id, coordinates
        heads = heads+chbb[1][0][:]   # nframe, fly id, coordinates
        tails = tails+chbb[1][0][:]   # nframe, fly id, coordinates
        has_tracks = True
        box_centers = centers[:, 0, :, :]   # nframe, fly id, coordinates
        box_centers = box_centers + chbb[1][0][:]
        nb_flies = box_centers.shape[1]
        logging.info(f"   nflies: {nb_flies}.")
    except OSError as e:
        logging.error(f'   could not load tracks.')

    ## Specifications for boxes
    if frame_stop == 0:
        frame_stop = data['frame_count']
        logging.info(f'   Setting frame_stop: {0}.'.format(frame_stop))
    frame_range = range(frame_start, frame_stop, frame_step)
    logging.info(f"   frame range: {frame_start}:{frame_stop}:{frame_step}.")

    ## Create list of frames
    logging.info(f"   getting frames from video.")
    frames = list(vr[frame_start:frame_stop:frame_step])
    nb_frames = len(frames)

    ## Calculate angle for export boxes
    logging.info(f"   calculating box angles.")
    box_angles = get_angles(heads[frame_range, ...], tails[frame_range, ...])

    ## Export boxes function
    logging.info(f"   exporting boxes of {expID}.")
    boxes, fly_id, fly_frame = export_boxes(frames, box_centers[frame_range, ...], box_size=np.array([120, 120]), box_angles=box_angles)
    boxes = normalize_boxes(boxes)
    # boxes = np.rot90(boxes,2,(1,2))   # Flips boxes, in case network expected fly images in the other direction

    ## Predictions
    logging.info(f"   doing predictions.")
    confmaps = predict_confmaps(networkPath, boxes[:, :, :, :1])
    logging.info(f"   processing confidence maps.")
    positions, confidence = process_confmaps_simple(confmaps)

    ## Recalculation of angles for further orientation fix
    logging.info(f"   recalculating box angles.")
    newbox_angles, bad_boxes = detect_bad_boxes_byAngle(positions)
    logging.info(f"   found {np.sum(bad_boxes)} cases of boxes with angles above threshold.")
    box_angles = box_angles + newbox_angles
    logging.info(f"   re-exporting boxes of {expID}.")
    boxes, fly_id, fly_frame = export_boxes(frames, box_centers[frame_range, ...], box_size=np.array([120, 120]), box_angles=box_angles)
    boxes = normalize_boxes(boxes)

    ## Final predictions
    logging.info(f"   re-doing predictions.")
    confmaps = predict_confmaps(networkPath, boxes[:, :, :, :1])
    logging.info(f"   re-processing confidence maps.")
    positions, confidence = process_confmaps_simple(confmaps)

    ## Saving data
    logging.info(f"   saving pose predictions to: {posePath}.")
    posedata = {'positions': positions, 'confidence': confidence, 'confmaps': confmaps, 'expID': expID, 'frame_range': frame_range, 'fly_id': fly_id, 'fly_frame': fly_frame, 'bad_boxes': bad_boxes}
    dd.io.save(posePath, posedata)
    logging.info(f"   saving fixed boxes to: {fixedBoxesPath}.")
    fixedBoxesdata = {'boxes': boxes, 'fly_id': fly_id, 'fly_frame': fly_frame}
    dd.io.save(fixedBoxesPath, fixedBoxesdata)

    ## Play movie of the exported boxes
    if play_boxes:
        logging.info(f"   playing video of boxes.")
        vplay(boxes[fly_id == 0, ...])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    defopt.run(main)
