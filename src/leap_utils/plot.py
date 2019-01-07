from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np


def color_confmaps(confmaps, cmap='gist_rainbow', invert: bool = False) -> (np.ndarray, np.ndarray):
    """Color code different layers in a confidence maps.

    Usage:
        confmaps_merge, colors = color_confmaps(confmaps, cmap='Set3')
        plt.imshow(confmaps_merge)

    Args:
        confmaps
        cmap - str, cmap object, nparray
        invert: white background
    Returns:
        colored_maps
        colors
    """
    if isinstance(cmap, str):  # name of colormap
        cm = plt.get_cmap(cmap)
        cols = np.array(cm(np.linspace(0, 1, confmaps.shape[-1])))
    elif isinstance(cmap, Colormap):  # colormap object
        cm = cmap
        cols = np.array(cm(np.linspace(0, 1, confmaps.shape[-1])))
    elif isinstance(cmap, np.ndarray):  # col vals
        cols = cmap

    colors = cols[..., :3]  # remove alpha channel form colors
    color_confmaps = np.zeros((*confmaps.shape[:2], 3))

    for mp in range(confmaps.shape[-1]):
        color_confmaps += confmaps[..., mp:mp+1] * colors[mp:mp+1, :]

    if invert:
        color_confmaps = 1-color_confmaps
    return color_confmaps, colors


def confmaps(confmaps, cmap='gist_rainbow', invert: bool = False):
    confmaps_merge, colors = color_confmaps(confmaps, cmap, invert)
    plt.imshow(confmaps_merge)


def boxpos(box, pos, cols=None):
    """Plot box and overlay with positions.

    Args:
        box: image
        pos: positions of tracked points in box
        cols: color for each tracked point (defaults to 'gist_rainbow' color map)
    """
    if cols is None:
        cm = plt.get_cmap('gist_rainbow')
        cols = np.array(cm(np.linspace(0, 1, pos.shape[0])))
    plt.imshow(box, cmap='gray')
    plt.scatter(pos[:, 1], pos[:, 0], c=cols)


def joint_distributions(positions, type):
    # check seaborn gallery:
    # https://seaborn.pydata.org/examples/multiple_joint_kde.html
    # https://seaborn.pydata.org/examples/cubehelix_palette.html
    pass


def annotate(frame, positions):
    """Annotate frame.

    Args:
        frame: [width, height, channels]
        positions: [nb.pos, x/y]
    """
    import cv2
    nb_pos = positions.shape[0]

    colors = np.zeros((1, nb_pos, 3), np.uint8)
    colors[0, :] = 220
    colors[0, :, 0] = np.arange(0, 180, 180.0/nb_pos)
    colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)[0].astype(np.float32) / 255.0
    colors = [list(map(float, thisColor)) for thisColor in colors]

    cm = plt.get_cmap('gist_rainbow')
    cols = np.array(cm(np.linspace(0, 1, nb_pos)))*255
    colors = cols[..., :3].astype(np.float32).tolist()
    for idx, pos in enumerate(positions):
        cv2.circle(frame, (int(pos[1]), int(pos[0])), radius=4, color=colors[idx], thickness=1)
    return frame


def vplay(frames: np.array, idx: np.array = None, positions: np.array = None, moviemode: bool = False):
    """Plots boxes, either in a movie (moviemode = True) or frame by frame (moviemode = False)

    Args:
        frames: [nb_frames, widht, height, channels] (output from export_boxes).
        idx: ...
        positions: [nb_frames, x/y]
        moviemode: auto play or advance/rewind via 'd' and 's'
    """
    import cv2
    import time

    if idx is None:
        idx = range(len(frames))

    if len(idx) == len(frames)/2:
        ridx = np.zeros(len(frames), dtype=int)
        ridx[::2], ridx[1::2] = idx, idx
        idx = ridx

    nb_chans = frames.shape[3]

    if nb_chans == 1:
        frames = np.repeat(frames, 3, axis=3)

    ii = 0
    while True:
        frame = frames[ii, ...]
        if positions is not None:
            frame = annotate(frame, positions[ii, ...])
        cv2.putText(frame, str(idx[ii]), (12, 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 250), lineType=4)
        cv2.imshow('movie', frame)

        if moviemode:
            ii += 1
            time.sleep(0.01)    # to reproduce at 100 frames per second
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            if ii >= len(frames)-1:
                ii = 0

        else:
            wkey = cv2.waitKey(0)
            if wkey & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            elif wkey & 0xFF == ord('d'):
                ii += 1
            elif wkey & 0xFF == ord('a'):
                ii -= 1

            if ii > len(frames)-1:
                ii = 0
            elif ii < 0:
                ii = len(frames)-1
