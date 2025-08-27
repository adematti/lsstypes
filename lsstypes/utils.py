import os

import numpy as np


def mkdir(dirname):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname)  # MPI...
    except OSError:
        return


def savefig(filename: str, fig=None, bbox_inches='tight', pad_inches=0.1, dpi=200, **kwargs):
    """
    Save figure to ``filename``.

    Warning
    -------
    Take care to close figure at the end, ``plt.close(fig)``.

    Parameters
    ----------
    filename : str
        Path to save the figure.
    fig : matplotlib.figure.Figure, optional
        Figure to save. If None, uses current figure.
    bbox_inches : str, optional
        Bounding box for saving.
    pad_inches : float, optional
        Padding around the figure.
    dpi : int, optional
        Dots per inch.
    **kwargs
        Optional arguments for :meth:`matplotlib.figure.Figure.savefig`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from matplotlib import pyplot as plt
    mkdir(os.path.dirname(filename))
    if fig is None:
        fig = plt.gcf()
    fig.savefig(filename, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=dpi, **kwargs)
    return fig


class FakeFigure(object):
    """
    Fake figure class to wrap axes for plotting utilities.

    Parameters
    ----------
    axes : list or object
        Axes to wrap.
    """
    def __init__(self, axes):
        if not hasattr(axes, '__iter__'):
            axes = [axes]
        self.axes = list(axes)


def plotter(*args, **kwargs):
    """
    Return wrapper for plotting functions, that adds the following (optional) arguments to ``func``:

    Parameters
    ----------
    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.
    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.
    show : bool, default=False
        If ``True``, show figure.
    """
    from functools import wraps

    def get_wrapper(func):

        @wraps(func)
        def wrapper(*args, fn=None, kw_save=None, show=False, fig=None, **kwargs):

            from matplotlib import pyplot as plt

            if fig is not None:

                if not isinstance(fig, plt.Figure):  # create fake figure that has axes
                    fig = FakeFigure(fig)

                elif not fig.axes:
                    fig.add_subplot(111)

                kwargs['fig'] = fig

            fig = func(*args, **kwargs)
            if fn is not None:
                savefig(fn, **(kw_save or {}))
            if show: plt.show()
            return fig

        return wrapper

    if kwargs or not args:
        if args:
            raise ValueError('unexpected args: {}, {}'.format(args, kwargs))
        return get_wrapper

    if len(args) != 1:
        raise ValueError('unexpected args: {}'.format(args))

    return get_wrapper(args[0])



@plotter
def plot_matrix(matrix, x1=None, x2=None, xlabel1=None, xlabel2=None, barlabel=None, label1=None, label2=None,
                figsize=None, norm=None, labelsize=None, fig=None):

    """
    Plot matrix.

    Parameters
    ----------
    matrix : array, list of lists of arrays
        Matrix, organized per-block.

    x1 : array, list of arrays, default=None
        Optionally, coordinates corresponding to the first axis of the matrix, organized per-block.

    x2 : array, list of arrays, default=None
        Optionally, coordinates corresponding to the second axis of the matrix, organized per-block.

    xlabel1 : str, list of str, default=None
        Optionally, label(s) corresponding to the first axis of the matrix, organized per-block.

    xlabel2 : str, list of str, default=None
        Optionally, label(s) corresponding to the second axis of the matrix, organized per-block.

    barlabel : str, default=None
        Optionally, label for the color bar.

    label1 : str, list of str, default=None
        Optionally, label(s) for the first observable(s) in the matrix, organized per-block.

    label2 : str, list of str, default=None
        Optionally, label(s) for the second observable(s) in the matrix, organized per-block.

    figsize : int, tuple, default=None
        Optionally, figure size.

    norm : matplotlib.colors.Normalize, default=None
        Scales the matrix to the canonical colormap range [0, 1] for mapping to colors.
        By default, the matrix range is mapped to the color bar range using linear scaling.

    labelsize : int, default=None
        Optionally, size for labels.

    fig : matplotlib.figure.Figure, default=None
        Optionally, a figure with at least as many axes as blocks in ``covariance``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize

    def _is_sequence(item):
        return isinstance(item, (tuple, list))

    if not _is_sequence(matrix[0]) or not np.size(matrix[0][0]):
        matrix = [[matrix]]
    mat = matrix
    size1, size2 = [row[0].shape[0] for row in mat], [col.shape[1] for col in mat[0]]

    def _make_list(x, size):
        if not _is_sequence(x):
            x = [x] * size
        return list(x)

    if x2 is None: x2 = x1
    x1, x2 = [_make_list(x, len(size)) for x, size in zip([x1, x2], [size1, size2])]
    if xlabel2 is None: xlabel2 = xlabel1
    xlabel1, xlabel2 = [_make_list(x, len(size)) for x, size in zip([xlabel1, xlabel2], [size1, size2])]
    if label2 is None: label2 = label1
    label1, label2 = [_make_list(x, len(size)) for x, size in zip([label1, label2], [size1, size2])]

    vmin, vmax = min(item.min() for row in mat for item in row), max(item.max() for row in mat for item in row)
    norm = norm or Normalize(vmin=vmin, vmax=vmax)
    nrows, ncols = [len(x) for x in [size2, size1]]
    if fig is None:
        figsize = figsize or tuple(max(n * 3, 6) for n in [ncols, nrows])
        if np.ndim(figsize) == 0: figsize = (figsize,) * 2
        xextend = 0.8
        fig, lax = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False,
                                figsize=(figsize[0] / xextend, figsize[1]),
                                gridspec_kw={'height_ratios': size2[::-1], 'width_ratios': size1},
                                squeeze=False)
        lax = lax.ravel()
        wspace = hspace = 0.18
        fig.subplots_adjust(wspace=wspace, hspace=hspace)
    else:
        lax = fig.axes
    lax = np.array(lax).reshape((nrows, ncols))
    cmap = plt.get_cmap('jet_r')
    for i in range(ncols):
        for j in range(nrows):
            ax = lax[nrows - j - 1][i]
            xx1, xx2 = x1[i], x2[j]
            if x1[i] is None: xx1 = 1 + np.arange(mat[i][j].shape[0])
            if x2[j] is None: xx2 = 1 + np.arange(mat[i][j].shape[1])
            xx1 = np.append(xx1, xx1[-1] + (1. if xx1.size == 1 else xx1[-1] - xx1[-2]))
            xx2 = np.append(xx2, xx2[-1] + (1. if xx2.size == 1 else xx2[-1] - xx2[-2]))
            mesh = ax.pcolormesh(xx1, xx2, mat[i][j].T, norm=norm, cmap=cmap)
            if i > 0 or x1[i] is None: ax.yaxis.set_visible(False)
            if j == 0 and xlabel1[i]: ax.set_xlabel(xlabel1[i], fontsize=labelsize)
            if j > 0 or x2[j] is None: ax.xaxis.set_visible(False)
            if i == 0 and xlabel2[j]: ax.set_ylabel(xlabel2[j], fontsize=labelsize)
            ax.tick_params()
            if label1[i] is not None or label2[j] is not None:
                text = '{}\nx {}'.format(label1[i], label2[j])
                ax.text(0.05, 0.95, text, horizontalalignment='left', verticalalignment='top',\
                        transform=ax.transAxes, color='black')

    fig.subplots_adjust(right=xextend)
    cbar_ax = fig.add_axes([xextend + 0.05, 0.15, 0.03, 0.7])
    cbar_ax.tick_params()
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    if barlabel: cbar.set_label(barlabel, rotation=90)
    return fig