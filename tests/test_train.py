import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from leap_utils.plot import color_confmaps
from leap_utils.train import initialize_network, points2mask, make_masks, BoxMaskSequence, train_network
plt.ion()

boxes = np.random.random((27, 40, 40, 1))
maps = np.random.random((27, 40, 40, 12))
positions = np.random.randint(10, 30, (27, 12, 2))


def test_initialize_network():
    m = initialize_network(boxes.shape[1:3], maps.shape[-1])
    print(m)
    assert m.input_shape == (None, *boxes.shape[1:3], 1)
    assert m.output_shape == (None, *boxes.shape[1:3], maps.shape[-1])


def test_points2mask():
    masks = points2mask(positions[0, ...].astype(np.intp), boxes.shape[1:3])
    assert masks.shape == (*boxes.shape[1:3], positions.shape[1])


def test_make_masks():
    masks = make_masks(positions.astype(np.intp), boxes.shape[1:3])
    assert masks.shape == (positions.shape[0], *boxes.shape[1:3], positions.shape[1])


def test_BoxMaskSequence():
    boxes = np.random.random((10, 120, 120, 1))
    maps = np.random.random((10, 120, 120, 12))

    bms = BoxMaskSequence(boxes, maps, batch_size=6, shuffle=False, hflip=True, vflip=True, rg=10,
                          hrg=0.1, wrg=0.1, zrg=(0.9, 1.1), brg=(0.9, 1.1))
    assert len(bms) == 2
    for i in range(3):
        x, y = bms[i]
        print(x.shape)
        assert x.shape == (6, 120, 120, 1), f"x.shape={x.shape} is wrong."
        assert y.shape == (6, 120, 120, 12), f"y.shape={y.shape} is wrong."


def test_train_network():
    now = str(datetime.datetime.now())
    date = now[:10] + '_' + now[11:13] + '-' + now[14:16]
    save_weights_path = f"./test_{date}"

    fit_hist = train_network(boxes, positions, save_weights_path, batch_size=4, epochs=3)
    for ext in ['model', 'best', 'final']:
        filename = f"{save_weights_path}.{ext}"
        assert os.path.exists(filename), f"{filename} does not exist."
        os.remove(filename)
    assert fit_hist


def augment_plot_repeat(bms, title='', repeats=4, pause=.2):
    for _ in range(repeats):
        x, y = bms[0]
        print(x.shape)

        plt.subplot(121)
        plt.imshow(x[0, ..., 0])
        plt.subplot(122)
        plt.imshow(color_confmaps(y[0, ...])[0])
        plt.title(title)
        plt.pause(pause)


def test_augment():
    boxes = np.zeros((1, 120, 120, 1))
    boxes[0, 50, 50, 0] = 127
    boxes[0, 30, 50, 0] = 64

    points = np.array([[[50, 50], [30, 50]]])
    maps = make_masks(points, boxes.shape[1:3])

    bms = BoxMaskSequence(boxes, maps, batch_size=1, shuffle=False, hflip=True)
    augment_plot_repeat(bms, 'hflip')
    bms = BoxMaskSequence(boxes, maps, batch_size=1, shuffle=False, vflip=True)
    augment_plot_repeat(bms, 'vflip')
    bms = BoxMaskSequence(boxes, maps, batch_size=1, shuffle=False, rg=100)
    augment_plot_repeat(bms, 'rotate')
    bms = BoxMaskSequence(boxes, maps, batch_size=1, shuffle=False, hrg=0.5)
    augment_plot_repeat(bms, 'hshift')
    bms = BoxMaskSequence(boxes, maps, batch_size=1, shuffle=False, wrg=0.5)
    augment_plot_repeat(bms, 'vshift')
    bms = BoxMaskSequence(boxes, maps, batch_size=1, shuffle=False, zrg=[.5, 2])
    augment_plot_repeat(bms, 'zoom')
    bms = BoxMaskSequence(boxes, maps, batch_size=1, shuffle=False, brg=[0.1, 2])
    augment_plot_repeat(bms, 'brightness')
