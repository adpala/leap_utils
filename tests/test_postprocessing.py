import numpy as np
from leap_utils.postprocessing import max_simple, process_confmaps_simple


def test_max_simple():
    X = np.zeros((10, 10, 4))
    X[2, 4, 0] = 2
    positions, confidence = max_simple(X)
    assert positions[0, 0] == 2 and positions[0, 1] == 4
    assert confidence[0] == 1


def test_process_confmaps_simple():
    X = np.zeros((4, 10, 10, 4))
    X[0, 2, 4, 0] = 2.0
    positions, confidence = process_confmaps_simple(X)
    assert positions.shape == (4, 4, 2)
    assert confidence.shape == (4, 4, 1)
    assert confidence[0, 0, 0] == 1


def test_process_confmaps_simple_data():
    # confmaps = predict_conf_maps(boxes) OR LOAD SAVED CONFMAPS
    # positions, confidence = process_confmaps_simple(confmaps)
    # PLOT
    pass
