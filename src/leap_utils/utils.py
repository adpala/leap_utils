import math
import numpy as np
from sys import platform
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


def it(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Yield slices from a numpy array over an arbitrary axis.

    Args:
        x(np.ndarray): numpy array
        axis(int): axis over which to iterate (defaults to last axis)
    Returns:
        generator that yields slice along axis

    Example:
        This code will return 10 4x4 matrices ('planes')
        stack_of_planes = np.zeros((10, 4, 4))
        for plane in it(stack_of_planes, axis=0):
            print(plane)

    """
    for idx in range(x.shape[axis]):
        yield x.take(idx, axis=axis)


def flatten(X: np.ndarray) -> (np.ndarray):
    """Convert arrays from [dim1, dim2, ...] to [dim2*dim2, ...].

    Arguments:
        X: np.ndarray [dim1, dim2, ...]
    Returns:
        X: np.ndarray [dim1 * dim2, ...]
    """
    return X.reshape((X.shape[0]*X.shape[1], *X.shape[2:]), order='A')


def unflatten(X: np.ndarray, dim2_len: int) -> (np.ndarray):
    """Convert from [dim1*dim2, ...] to [dim1, dim_len...].

    Arguments:
        X: np.ndarray [dim1_len*dim2_len, ...]
        dim2_len: int=2
    Returns:
        X: np.ndarray [dim1_len, dim2_len, ...]
    """
    return X.reshape((int(X.shape[0]/dim2_len), dim2_len, *X.shape[1:]), order='A')


def smooth(x, N):
    """Smooth signal using box filter of length N samples."""
    return np.convolve(x, np.ones((N,)) / N, mode='full')[(N - 1):]


def rotate_points(x, y, degrees, origin=(0, 0)):
    """Rotate a point around a given point."""
    radians = degrees / 180 * np.pi
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy


# - to utils
def max2d(X: np.ndarray) -> (tuple, float):
    """Get position and value of array maximum."""
    row, col = np.unravel_index(X.argmax(), X.shape)
    return (row, col), X[row, col]


def islinux():
    return platform == "linux" or platform == "linux2"


def ismac():
    return platform == "darwin"


def iswin():
    return platform == "win32"


def mykde(X, *, grid_size: int = 120, bw: float = 4, bw_select: bool = False, plotnow: bool = False, train_percentage: float = 0.1):
    """ Calculates the probability density given a set of positions through the Kernel Density Estimation approach.

    Arguments:
        X - positions [nsamples, 2]
        grid_size - default = 120
        bw - bandwidth, optimal value may change depending on amount of data
        bw_select - toggle option to search for best bandwidth
        plotnow - toggle option to plot a figure of the probability density map
        train_percentage - (if bw_select = True) percentage of the data set to be used for finding optimal bandwidth
    Returns:
        probdens - probability density map [frame_size, frame_size]
    """

    if bw_select:
        # Selecting the bandwidth via cross-validation
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=len(X[:int(len(X)*train_percentage), :]))
        grid.fit(X[:int(len(X)*train_percentage), :])
        bw = grid.best_params_['bandwidth']

    # Kernel Density Estimation
    kde = KernelDensity(bandwidth=bw).fit(X)

    # Grid creation
    xx_d = np.linspace(0, grid_size, grid_size)
    yy_d = np.linspace(0, grid_size, grid_size)
    xx_dv, yy_dv = np.meshgrid(xx_d, yy_d)
    coor = np.array([xx_dv.flatten(), yy_dv.flatten()]).swapaxes(0, 1)

    # Evaluation of grid
    logprob = kde.score_samples(coor)      # Array of log(density) evaluations. Normalized to be probability densities.
    probdens = np.exp(logprob)

    # Plot
    if plotnow:
        im = probdens.reshape((int(probdens.shape[0]/grid_size), grid_size))
        plt.imshow(im)
        plt.colorbar()
        # plt.contourf(xx_dv, yy_dv, probdens.reshape((xx_d.shape[0], xx_d.shape[0])))
        plt.scatter(X[:, 0], X[:, 1], c='red')
        plt.show()

    return probdens
