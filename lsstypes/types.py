import numpy as np

from .base import ObservableLeaf, ObservableTree
from .utils import plotter


class Spectrum2Pole(ObservableLeaf):

    _name = 'spectrum2pole'
    _default_coords = ('k',)

    def __init__(self, k=None, num=None, num_shotnoise=None, norm=None):
        super().__init__(k=k, num=num)
        if num_shotnoise is None:
            num_shotnoise = np.zeros_like(self.num)
        if norm is None:
            norm = np.ones_like(self.num)
        self._data['num_shotnoise'] = num_shotnoise
        self._values.append('num_shotnoise')
        self._data['norm'] = norm
        self._values.append('norm')

    def value(self):
        return (self.num - self.num_shotnoise) / self.norm

    @plotter
    def plot(self, fig=None, **kwargs):
        r"""
        Plot a power spectrum multipole.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.
        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.
        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.
        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        ax.plot(self.k, self.k * self.value(), **kwargs)
        ax.set_xlabel(r'$k$ [$(\mathrm{Mpc}/h)$]')
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        return fig


class Spectrum2Poles(ObservableTree):

    _name = 'spectrum2poles'

    def __init__(self, poles, ells=(0,)):
        """Initialize power spectrum multipoles."""
        super().__init__(poles, ells=ells)

    @plotter
    def plot(self, fig=None):
        r"""
        Plot the power spectrum multipoles.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.
        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.
        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.
        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ell in self.ells:
            pole = self.get(ell)
            pole.plot(fig=ax, label=rf'$\ell = {ell:d}$')
        ax.legend(frameon=False)
        return fig


class Count2(ObservableLeaf):

    """Pair counts."""
    _name = 'count2'


class Correlation2(ObservableTree):

    """Correlation function."""
    _name = 'correlation2'
    _is_leaf = True

    def __init__(self, DD=None, DR=None, RD=None, RR=None):
        super().__init__([DD, DR, RD, RR], key=['DD', 'DR', 'RD', 'RR'])

    def value(self):
        corr = self.get('DD').value() - self.get('DR').value() - self.get('RD').value() + self.get('RR').value()
        corr /= self.get('RR').value()
        return corr