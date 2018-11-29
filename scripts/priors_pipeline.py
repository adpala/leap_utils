import numpy as np
import h5py
import matplotlib.pyplot as plt
import logging
import defopt
import os
from leap_utils.postprocessing import load_labels
from leap_utils.utils import mykde


def create_priors(labelsPath: str = '/#Common/chainingmic/leap/training_data/big_dataset_train.labels.mat', frame_size: int = 120, savePath: str = '/#Common/chainingmic/leap/training_data/priors.h5', overwrite: bool = True, testit: bool = False):
    """ Creates priors for each body part based on the labeled positions from the mat file specified.

    :param str labelsPath: path to the mat file
    :param int frame_size: size of frame, assuming squared frame
    :param str savePath: path to save priors array
    :param bool overwrite: allow to overwrite files
    :param bool testit: test for best bandwidth in the kde calculation (takes A LOT more time, not sure if it really works)
    """

    # Load labels
    positions, _, _ = load_labels(labelsPath)

    # Calculate probability density map (priors)
    logging.info(f'   initializing priors.')
    priors = np.zeros((frame_size, frame_size, 12))

    # Test or Generate priors
    if testit:
        logging.info(f"   testing bandwidths for kde.")
        bandwidths = np.linspace(10, 100, 5)
        for count, bw in enumerate(bandwidths):
            print(count, bw)
            plt.figure(count, figsize=[16, 2])
            for bp in range(12):
                priors[:, :, bp] = mykde(positions[:, :, bp], bw=bw).reshape((frame_size, frame_size))
                plt.subplot(1, 12, bp+1)
                plt.imshow(priors[:, :, bp])
                plt.title(str(bp))
                plt.axis('off')

            plt.tight_layout()
        plt.show()
    else:
        bw = 10
        logging.info(f'   generating priors with bandwidth = {bw}.')
        plt.figure(1, figsize=[16, 2])
        for bp in range(12):
            priors[:, :, bp] = mykde(positions[:, :, bp], bw=bw).reshape((frame_size, frame_size))
            plt.subplot(1, 12, bp+1)
            plt.imshow(priors[:, :, bp])
            plt.title(str(bp))
            plt.axis('off')
        plt.tight_layout()

    # Save priors
    logging.info(f'   saving priors to : {savePath}.')
    if os.path.exists(savePath):
        if overwrite:
            logging.warning(f"   {savePath} exists - deleting to overwrite.")
            os.remove(savePath)
        else:
            raise OSError(f"   {savePath} exists - cannot overwrite.")
    new_f = h5py.File(savePath)
    new_f.create_dataset('priors', data=priors, compression='gzip')
    new_f.close()
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    defopt.run(create_priors)
