import os


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