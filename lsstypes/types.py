import numpy as np

from .base import ObservableLeaf, ObservableTree, LeafLikeObservableTree, WindowMatrix, CovarianceMatrix, register_type, _edges_names, _check_data_names
from .utils import plotter, my_ones_like, my_zeros_like


@register_type
class Mesh2SpectrumPole(ObservableLeaf):
    """
    Container for a power spectrum multipole :math:`P_\ell(k)`.

    Stores the binned power spectrum for a given multipole order :math:`\ell`, including shot noise, normalization, and mode counts.

    Parameters
    ----------
    k : array-like
        Bin centers for wavenumber :math:`k`.
    k_edges : array-like
        Bin edges for wavenumber :math:`k`.
    num_raw : array-like
        Raw power spectrum measurements.
    num_shotnoise : array-like, optional
        Shot noise contribution (default: zeros).
    norm : array-like, optional
        Normalization factor (default: ones).
    nmodes : array-like, optional
        Number of modes per bin (default: ones).
    ell : int, optional
        Multipole order :math:`\ell`.
    attrs : dict, optional
        Additional attributes.
    """
    _name = 'mesh2spectrumpole'

    def __init__(self, k=None, k_edges=None, num_raw=None, num_shotnoise=None, norm=None, nmodes=None, ell=None, attrs=None, **kwargs):
        kw = dict(k=k, k_edges=k_edges)
        if k_edges is None: kw.pop('k_edges')
        self.__pre_init__(**kw, coords=['k'], attrs=attrs)
        if num_shotnoise is None: num_shotnoise = my_zeros_like(num_raw)
        if norm is None: norm = my_ones_like(num_raw)
        if nmodes is None: nmodes = my_ones_like(num_raw, dtype='i4')
        self._values_names = ['value', 'num_shotnoise', 'norm', 'nmodes']
        for name in list(kwargs):
            if name in self._values_names: pass
            elif name in ['volume']: self._values_names.append(name)
            else: raise ValueError('{name} not unknown')
        self._update(num_raw=num_raw, num_shotnoise=num_shotnoise, norm=norm, nmodes=nmodes, **kwargs)
        if ell is not None:
            self._meta['ell'] = ell

    def _update(self, **kwargs):
        require_recompute = ['num_raw', 'num_shotnoise', 'norm']
        if set(kwargs) & set(require_recompute):
            if 'value' in self._data:
                self._data['num_raw'] = self._data.pop('value') * self._data['norm'] + self._data['num_shotnoise']
            self._data.update(kwargs)
            if 'value' not in self._data:
                self._data['value'] = (self._data.pop('num_raw') - self._data['num_shotnoise']) / self._data['norm']
        else:
            self._data.update(kwargs)
        _check_data_names(self)
        if 'norm' in kwargs:
            self._data['norm'] =  self._data['norm'] * my_ones_like(self._data['value'])

    def _plabel(self, name):
        if name == 'k':
            return r'$k$ [$h/\mathrm{Mpc}$]'
        if name == 'value':
            return r'$P(k)$ [$(\mathrm{Mpc}/h)^3$]'
        return None

    def _binweight(self, name=None):
        # weight, normalized
        if name == 'nmodes':
            return False, False
        return self.nmodes, True

    @classmethod
    def _sumweight(cls, observables, name=None):
        if name is None or name in ['value']:
            s = sum(observable.norm for observable in observables)
            return [observable.norm / s for observable in observables]
        if name in ['nmodes']:
            return None  # keep the first nmodes
        return [1] * len(observables)  # just sum

    def values(self, name=None):
        """
        Get value array(s).

        Parameters
        ----------
        name : str, optional
            Name of value.

        Returns
        -------
        values : array or dict
        """
        if name is not None and name == 'shotnoise':
            return self.num_shotnoise / self.norm
        return super().values(name=name)

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
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        return fig


@register_type
class Mesh2SpectrumPoles(ObservableTree):
    """
    Container for multiple power spectrum multipoles :math:`P_\ell(k)`.

    Stores a collection of `Mesh2SpectrumPole` objects for different multipole orders :math:`\ell`, allowing joint analysis and plotting.

    Parameters
    ----------
    poles : list of Mesh2SpectrumPole
        List of power spectrum multipole objects.

    ells : list of int, optional
        Multipole orders :math:`\ell` for each pole (default: inferred from `poles`).

    attrs : dict, optional
        Additional attributes.
    """
    _name = 'mesh2spectrumpoles'

    def __init__(self, poles, ells=None, attrs=None):
        """Initialize power spectrum multipoles."""
        if ells is None: ells = [pole.ell for pole in poles]
        super().__init__(poles, ells=ells, attrs=attrs)

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



@register_type
class Mesh2CorrelationPole(ObservableLeaf):
    r"""
    Container for a correlation function multipole :math:`\xi_\ell(s)`.

    Stores the binned correlation function for a given multipole order :math:`\ell`, including normalization and number of modes.

    Parameters
    ----------
    s : array-like
        Bin centers for separation :math:`s`.
    s_edges : array-like
        Bin edges for separation :math:`s`.
    num_raw : array-like
        Raw power spectrum measurements.
    num_shotnoise : array-like, optional
        Shot noise contribution (default: zeros).
    norm : array-like, optional
        Normalization factor (default: ones).
    nmodes : array-like, optional
        Number of modes per bin (default: ones).
    ell : int, optional
        Multipole order :math:`\ell`.
    attrs : dict, optional
        Additional attributes.
    """
    _name = 'mesh2correlationpole'

    def __init__(self, s=None, s_edges=None, num_raw=None, num_shotnoise=None, norm=None, nmodes=None, ell=None, attrs=None, **kwargs):
        kw = dict(s=s, s_edges=s_edges)
        if s_edges is None: kw.pop('s_edges')
        self.__pre_init__(**kw, coords=['s'], attrs=attrs)
        if num_shotnoise is None: num_shotnoise = my_zeros_like(num_raw)
        if norm is None: norm = my_ones_like(num_raw)
        if nmodes is None: nmodes = my_ones_like(num_raw, dtype='i4')
        self._values_names = ['value', 'num_shotnoise', 'norm', 'nmodes']
        for name in list(kwargs):
            if name in self._values_names: pass
            elif name in ['volume']: self._values_names.append(name)
            else: raise ValueError('{name} not unknown')
        self._update(num_raw=num_raw, num_shotnoise=num_shotnoise, norm=norm, nmodes=nmodes, **kwargs)
        if ell is not None:
            self._meta['ell'] = ell

    def _update(self, **kwargs):
        require_recompute = ['num_raw', 'num_shotnoise', 'norm']
        if set(kwargs) & set(require_recompute):
            if 'value' in self._data:
                self._data['num_raw'] = self._data.pop('value') * self._data['norm'] + self._data['num_shotnoise']
            self._data.update(kwargs)
            if 'value' not in self._data:
                self._data['value'] = (self._data.pop('num_raw') - self._data['num_shotnoise']) / self._data['norm']
        else:
            self._data.update(kwargs)
        _check_data_names(self)
        if 'norm' in kwargs:
            self._data['norm'] =  self._data['norm'] * my_ones_like(self._data['value'])

    def _binweight(self, name=None):
        # weight, normalized
        if name == 'nmodes':
            return False, False
        return self.nmodes, True

    @classmethod
    def _sumweight(cls, observables, name=None):
        if name is None or name in ['value']:
            s = sum(observable.norm for observable in observables)
            return [observable.norm / s for observable in observables]
        if name in ['nmodes']:
            return [1. / len(observables)] * len(observables)
        return [1] * len(observables)  # just sum

    def _plabel(self, name):
        if name == 's':
            return r'$s$ [$\mathrm{Mpc}/h$]'
        if name == 'value':
            return r'$\xi_\ell(s)$'
        return None

    def values(self, name=None):
        """
        Get value array(s).

        Parameters
        ----------
        name : str, optional
            Name of value.

        Returns
        -------
        values : array or dict
        """
        if name is not None and name == 'shotnoise':
            return self.num_shotnoise / self.norm
        return super().values(name=name)

    @plotter
    def plot(self, fig=None, **kwargs):
        r"""
        Plot a correlation function multipole.

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
        ax.plot(self.s, self.s**2 * self.value(), **kwargs)
        ax.set_xlabel(self._plabel('s'))
        ax.set_ylabel(r'$s^2 \xi_\ell(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        return fig



@register_type
class Mesh2CorrelationPoles(ObservableTree):
    r"""
    Container for multiple correlation function multipoles :math:`\xi_\ell(s)`.

    Stores a collection of `Mesh2CorrelationPole` objects for different multipole orders :math:`\ell`.

    Parameters
    ----------
    poles : list of Mesh2CorrelationPole
        List of correlation function multipole objects.
    ells : list of int, optional
        Multipole orders :math:`\ell` for each pole (default: inferred from `poles`).
    attrs : dict, optional
        Additional attributes.
    """
    _name = 'mesh2correlationpoles'

    def __init__(self, poles, ells=None, attrs=None):
        """Initialize correlattion function multipoles."""
        if ells is None: ells = [pole.ell for pole in poles]
        super().__init__(poles, ells=ells, attrs=attrs)

    @plotter
    def plot(self, fig=None):
        r"""
        Plot the correlattion function multipoles.

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


@register_type
class Mesh3SpectrumPole(ObservableLeaf):
    """
    Container for a bispectrum multipole :math:`B_\ell(k)`.

    Stores the binned bispectrum for a given multipole order :math:`\ell`, including shot noise, normalization, and mode counts.

    Parameters
    ----------
    k : array-like
        Bin centers for wavenumber :math:`k`.
    k_edges : array-like
        Bin edges for wavenumber :math:`k`.
    num_raw : array-like
        Raw power spectrum measurements.
    num_shotnoise : array-like, optional
        Shot noise contribution (default: zeros).
    norm : array-like, optional
        Normalization factor (default: ones).
    nmodes : array-like, optional
        Number of modes per bin (default: ones).
    ell : tuple, int, optional
        Multipole order :math:`\ell`.
    attrs : dict, optional
        Additional attributes.
    """
    _name = 'mesh3spectrumpole'

    def __init__(self, k=None, k_edges=None, num_raw=None, num_shotnoise=None, norm=None, nmodes=None, ell=None, basis='', attrs=None):
        kw = dict(k=k, k_edges=k_edges)
        if k_edges is None: kw.pop('k_edges')
        self.__pre_init__(**kw, coords=['k'], attrs=attrs)
        if num_shotnoise is None: num_shotnoise = my_zeros_like(num_raw)
        if norm is None: norm = my_ones_like(num_raw)
        if nmodes is None: nmodes = my_ones_like(num_raw, dtype='i4')
        self._values_names = ['value', 'num_shotnoise', 'norm', 'nmodes']
        self._update(**kw, num_raw=num_raw, num_shotnoise=num_shotnoise, norm=norm, nmodes=nmodes)
        if ell is not None:
            self._meta['ell'] = ell
        self._meta['basis'] = basis

    def _update(self, **kwargs):
        require_recompute = ['num_raw', 'num_shotnoise', 'norm']
        if set(kwargs) & set(require_recompute):
            if 'value' in self._data:
                self._data['num_raw'] = self._data.pop('value') * self._data['norm'] + self._data['num_shotnoise']
            self._data.update(kwargs)
            if 'value' not in self._data:
                self._data['value'] = (self._data.pop('num_raw') - self._data['num_shotnoise']) / self._data['norm']
        else:
            self._data.update(kwargs)
        _check_data_names(self)
        if 'norm' in kwargs:
            self._data['norm'] =  self._data['norm'] * my_ones_like(self._data['value'])

    def _plabel(self, name):
        if name == 'k':
            if 'scoccimarro' in self.basis:
                return r'$k_1, k_2, k_3$ [$h/\mathrm{Mpc}$]'
            return r'$k_1, k_2$ [$h/\mathrm{Mpc}$]'
        if name == 'value':
            if 'scoccimarro' in self.basis:
                return r'$B_{\ell_3}(k_3)$ [$(\mathrm{Mpc}/h)^{6}$]'
            return r'$B_{\ell_1, \ell_2, \ell_3}(k_1, k_2)$ [$(\mathrm{Mpc}/h)^{6}$]'
        return None

    def _binweight(self, name=None):
        # weight, normalized
        if name == 'nmodes':
            return False, False
        return self.nmodes, True

    @classmethod
    def _sumweight(cls, observables, name=None):
        if name is None or name in ['value']:
            s = sum(observable.norm for observable in observables)
            return [observable.norm / s for observable in observables]
        if name in ['nmodes']:
            return None  # keep the first nmodes
        return [1] * len(observables)  # just sum

    def values(self, name=None):
        """
        Get value array(s).

        Parameters
        ----------
        name : str, optional
            Name of value.

        Returns
        -------
        values : array or dict
        """
        if name is not None and name == 'shotnoise':
            return self.num_shotnoise / self.norm
        return super().values(name=name)

    @plotter
    def plot(self, fig=None, **kwargs):
        r"""
        Plot bispectrum multipole.

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
        ax.plot(np.arange(len(self.k)), self.k.prod(axis=-1) * self.value(), **kwargs)
        ax.set_xlabel('bin index')
        if 'scoccimarro' in self.basis:
            ax.set_ylabel(r'$k_1 k_2 k_3 B_{\ell_1 \ell_2 \ell_3}(k_1, k_2, k_3)$ [$(\mathrm{Mpc}/h)^{6}$]')
        else:
            ax.set_ylabel(r'$k_1 k_2 B_{\ell_1 \ell_2 \ell_3}(k_1, k_2)$ [$(\mathrm{Mpc}/h)^{4}$]')
        return fig


@register_type
class Mesh3SpectrumPoles(ObservableTree):
    """
    Container for multiple bispectrum multipoles :math:`B_\ell(k)`.

    Stores a collection of `Mesh3SpectrumPole` objects for different multipole orders :math:`\ell`, allowing joint analysis and plotting.

    Parameters
    ----------
    poles : list of Mesh3SpectrumPole
        List of bispectrum multipole objects.

    ells : list of int or tuples, optional
        Multipole orders :math:`\ell` for each pole (default: inferred from `poles`).

    attrs : dict, optional
        Additional attributes.
    """
    _name = 'mesh3spectrumpoles'

    def __init__(self, poles, ells=None, attrs=None):
        """Initialize bispectrum multipoles."""
        if ells is None: ells = [pole.ell for pole in poles]
        super().__init__(poles, ells=ells, attrs=attrs)

    @plotter
    def plot(self, fig=None):
        r"""
        Plot the bispectrum multipoles.

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
            pole.plot(fig=ax, label=rf'$\ell = {ell}$')
        ax.legend(frameon=False)
        return fig


@register_type
class Count2(ObservableLeaf):
    """
    Container for two-point pair counts.

    Stores binned pair counts and normalization for correlation function estimation.

    Parameters
    ----------
    counts : array-like
        Raw pair counts for each bin.

    norm : array-like, optional
        Normalization factor (default: ones).

    attrs : dict, optional
        Additional attributes.

    kwargs : dict
        Additional keyword arguments for initialization.
    """
    _name = 'count2'

    def __init__(self, counts=None, norm=None, attrs=None, **kwargs):
        self.__pre_init__(attrs=attrs, **kwargs)
        if norm is None: norm = my_ones_like(counts)
        self._values_names = ['normalized_counts', 'norm']
        self._update(counts=counts, norm=norm)

    def _binweight(self, name=None):
        # weight, normalized
        if name is None or name == 'normalized_counts':
            return False, False
        if name == 'norm':
            return False, True  # not normalized to avoid cases where weight is 0
        return self.normalized_counts, True

    @classmethod
    def _sumweight(cls, observables, name):
        if name in ['normalized_counts']:
            s = sum(observable.norm for observable in observables)
            return [observable.norm / s for observable in observables]
        return [1] * len(observables)  # just sum norm

    def _update(self, **kwargs):
        if 'value' in kwargs:
            kwargs['normalized_counts'] = kwargs.pop('value')
        require_recompute = ['counts', 'norm']
        if set(kwargs) & set(require_recompute):
            if 'normalized_counts' in self._data:
                self._data['counts'] = self._data.pop('normalized_counts') * self._data['norm']
            self._data.update(kwargs)
            if 'normalized_counts' not in self._data:
                self._data['normalized_counts'] = self._data.pop('counts') / self._data['norm']
        else:
            self._data.update(kwargs)
        _check_data_names(self)
        if 'norm' in kwargs:
            self._data['norm'] = self._data['norm'] * my_ones_like(self._data['normalized_counts'])

    def values(self, name=0):
        if name == 'counts':
            return self.normalized_counts * self.norm
        return ObservableLeaf.values(self, name=name)


def _nan_to_zero(array):
    return np.where(np.isnan(array), 0., array)


@register_type
class Count2Jackknife(LeafLikeObservableTree):
    """
    Container for jackknife two-point pair counts.

    Stores pair counts for all jackknife realizations, including cross-pairs, and provides methods for jackknife corrections and covariance estimation.

    Parameters
    ----------
    ii_counts : dict
        Dictionary of pair counts for each jackknife realization (auto-pairs).

    ij_counts : dict
        Dictionary of pair counts for each jackknife realization (cross-pairs: i-j).

    ji_counts : dict
        Dictionary of pair counts for each jackknife realization (cross-pairs: j-i).

    realizations : list, optional
        List of realization labels (default: keys of `ii_counts`).

    attrs : dict, optional
        Additional attributes.
    """
    _name = 'count2jackknife'

    def __init__(self, ii_counts, ij_counts, ji_counts, realizations=None, attrs=None):
        if realizations is None:
            realizations = list(ii_counts)
        assert set(ji_counts) == set(ij_counts) == set(ii_counts)
        counts = [ii_counts[real] for real in realizations]
        counts += [ij_counts[real] for real in realizations]
        counts += [ji_counts[real] for real in realizations]
        cross = ['ii'] * len(ii_counts) + ['ij'] * len(ij_counts) + ['ji'] * len(ji_counts)
        realizations = realizations * 3
        super().__init__(counts, attrs=attrs, cross=cross, realizations=realizations)

    def value(self, return_type='nparray'):
        """
        Return the pair count value.

        Parameters
        ----------
        return_type : str, None, optional
            If 'nparray', return a numpy array.
            If ``None``, return a :class:`Count2` instance.

        Returns
        -------
        estimator : array, Count2
            Two-point counts.
        """
        value = Count2.value(self)
        if return_type is None:
            for count in self.get(cross='ii'): break
            count = count.clone(counts=self.values('normalized_counts') * self.values('norm'), norm=self.values('norm'), **self.coords(), meta=dict(self._meta))
            return count
        return value

    def values(self, *args, **kwargs):
        """Two-point count values: 'normalized_counts' and 'norm'."""
        return Count2.values(self, *args, **kwargs)

    def coords(self, *args, **kwargs):
        """Two-point count coordinates."""
        return Count2.coords(self, *args, **kwargs)

    def __getattr__(self, name):
        if name in ['_data']:
            self._set_data()
            return self._data
        if name in self._values_names + self._coords_names:
            return ObservableLeaf.__getattr__(self, name)
        return super().__getattr__(name)

    def _set_data(self):
        # Set global :attr:`counts`, :attr:`norm` based on all jackknife realizations
        # Deleted after each select
        self._data = {}
        for name in ['counts', 'norm']:
            self._data[name] = sum(count.values(name) for count in self.get(cross='ii'))\
                             + sum(count.values(name) for count in self.get(cross='ij'))
        dcounts = self._data.pop('counts')
        self._data['normalized_counts'] = dcounts / self._data['norm']

        for iaxis, axis in enumerate(self._coords_names):
            reduce_axis = tuple(iax for iax in range(len(self._coords_names)) if iax != iaxis)
            self._data[axis] = sum(_nan_to_zero(count.coords(axis)) * count.values('counts').sum(axis=reduce_axis) for count in self.get(cross='ii')) \
                             + sum(_nan_to_zero(count.coords(axis)) * count.values('counts').sum(axis=reduce_axis) for count in self.get(cross='ij'))
            with np.errstate(divide='ignore', invalid='ignore'):
                self._data[axis] /= dcounts.sum(axis=reduce_axis)
        for name in ['size1', 'size2']:
            if name in self._branches[0]._meta:
                self._meta[name] = sum(count._meta[name] for count in self.get(cross='ii'))

    @property
    def nrealizations(self):
        """Number of realizations."""
        return len(self._labels) // 3

    def realization(self, ii, correction='mohammad21'):
        """
        Return jackknife realization ``ii``.

        Parameters
        ----------
        ii : int
            Label of jackknife realization.

        correction : str, default='mohammad21'
            Correction to apply to computed counts.
            If ``None``, no correction is applied.
            Else, if "mohammad21", rescale cross-pairs by factor eq. 27 in arXiv:2109.07071.
            Else, rescale cross-pairs by provided correction factor.

        Returns
        -------
        counts : Count2
            Two-point counts for realization ``ii``.
        """
        alpha = 1.
        if isinstance(correction, str):
            if correction == 'mohammad21':
                # arXiv https://arxiv.org/pdf/2109.07071.pdf eq. 27
                alpha = self.nrealizations / (2. + np.sqrt(2) * (self.nrealizations - 1))
            else:
                raise ValueError('Unknown jackknife correction {}'.format(correction))
        elif correction is not None:
            alpha = float(correction)
        counts = self.get(realizations=ii, cross='ii').copy()
        for name in ['counts', 'norm']:
            counts._data[name] = self.values(name) - self.get(realizations=ii, cross='ii').values(name)\
                                 - alpha * (self.get(realizations=ii, cross='ij').values(name) + self.get(realizations=ii, cross='ji').values(name))
        dcounts = counts._data.pop('counts')
        counts._data['normalized_counts'] = dcounts / counts._data['norm']
        for iaxis, axis in enumerate(self._coords_names):
            reduce_axis = tuple(iax for iax in range(len(self._coords_names)) if iax != iaxis)
            counts._data[axis] = _nan_to_zero(self.coords(axis=axis)) * self.values('counts').sum(axis=reduce_axis) \
                                - _nan_to_zero(self.get(realizations=ii, cross='ii').coords(axis)) * self.get(realizations=ii, cross='ii').values('counts').sum(axis=reduce_axis)\
                                - alpha * (_nan_to_zero(self.get(realizations=ii, cross='ij').coords(axis)) * self.get(realizations=ii, cross='ij').values('counts').sum(axis=reduce_axis)\
                                         + _nan_to_zero(self.get(realizations=ii, cross='ji').coords(axis)) * self.get(realizations=ii, cross='ji').values('counts').sum(axis=reduce_axis))
            with np.errstate(divide='ignore', invalid='ignore'):
                counts._data[axis] /= dcounts.sum(axis=reduce_axis)
                # The above may lead to rounding errors
                # such that seps may be non-zero even if counts is zero.
                mask = np.any(dcounts != 0, axis=reduce_axis)  # if ncounts / counts computed, good indicator of whether pairs exist or not
                # For more robustness we restrict to those separations which lie in between the lower and upper edges
                mask &= (counts._data[axis] >= self.edges(axis=axis)[..., 0]) & (counts._data[axis] <= self.edges(axis=axis)[..., 1])
                counts._data[axis][~mask] = np.nan
        for name in ['size1', 'size2']:
            if name in self._meta:
                counts._meta[name] = self._meta[name] - self.get(realizations=ii, cross='ii')._meta[name]
        return counts

    def cov(self, return_type='nparray', **kwargs):
        """
        Return jackknife covariance (of flattened counts).

        Parameters
        ----------
        kwargs : dict
            Optional arguments for :meth:`realization`.

        Returns
        -------
        covariance : array
            Covariance matrix.
        """
        realizations = [self.realization(ii, **kwargs) for ii in self.realizations]
        cov = (len(realizations) - 1) * np.cov([realization.value().ravel() for realization in realizations], rowvar=False, ddof=0)
        cov = np.atleast_2d(cov)
        if return_type is None:
            mean = realizations[0].mean(realizations)
            return CovarianceMatrix(observable=mean, value=cov)
        return cov


def _get_project_mode(estimator, mode=None, **kwargs):
    # Return projection mode depending on provided arguments
    if 'ell' in kwargs:
        kwargs['ells'] = kwargs.pop('ell')
    if mode is None:
        if 'ells' in kwargs:
            mode = 'poles'
        elif 'wedges' in kwargs:
            mode = 'wedges'
        elif list(estimator.coords()) == ['rp', 'pi']:
            mode = 'wp'
        else:
            mode = None
    else:
        assert isinstance(mode, str)
        mode = mode.lower()
    return mode, kwargs



@register_type
class Count2Correlation(LeafLikeObservableTree):
    """
    Correlation function estimator for two-point statistics.

    Supports Landy-Szalay and natural estimators, with or without shifted/random pairs.

    Parameters
    ----------
    estimator : str, optional
        Estimator type ('landyszalay' or 'natural'). Default is 'landyszalay'.

    attrs : dict, optional
        Additional attributes.

    kwargs : dict
        Pair count observables, e.g. DD, RR, DR, RD, DS, SD, SS.
    """

    _name = 'count2correlation'

    def __init__(self, estimator='landyszalay', attrs=None, **kwargs):
        with_shifted = any('S' in key for key in kwargs)
        if estimator == 'landyszalay':
            count_names = ['DD', 'RR']
            if with_shifted: count_names += ['DS', 'SD', 'SS']
            else: count_names += ['DR', 'RD']
        elif estimator == 'natural':
            count_names = ['DD', 'RR']
            if with_shifted: count_names += ['SS']
        else:
            raise NotImplementedError(f'estimator {estimator} not implemented')
        super().__init__([kwargs[count_name] for count_name in count_names], count_names=count_names,
                         meta=dict(estimator=estimator, with_shifted=with_shifted), attrs=attrs)

    def value(self):
        """
        Return the correlation function value.

        Returns
        -------
        estimator : array
        """
        RR = self.get('RR').value()
        nonzero = RR != 0
        scount_name = 'S' if self.with_shifted else 'R'
        if self.estimator == 'landyszalay':
            corr = self.get('DD').value() - self.get('D'  + scount_name).value() - self.get(scount_name + 'D').value() + self.get(scount_name * 2).value()
            corr /= RR
        elif self.estimator == 'natural':
            corr = self.get('DD').value() - self.get(scount_name * 2).value()
            corr /= RR
        return np.where(nonzero, corr, np.nan)

    def coords(self, *args, **kwargs):
        """
        Get coordinate array(s).

        Parameters
        ----------
        axis : str or int, optional
            Name or index of coordinate.

        Returns
        -------
        coords : array or dict
        """
        return self.get('RR').coords(*args, **kwargs)

    def project(self, mode=None, **kwargs):
        """
        Project the correlation function onto multipoles, wedges, or projected bins.

        Parameters
        ----------
        mode : str, optional
            Projection mode ('poles', 'wedges', 'wp'). If None, inferred from kwargs.

        ells : list of int, optional
            Orders of Legendre polynomials to project onto (default is ``[0, 2, 4]``).

        ignore_nan : bool, optional
            If ``True``, ignore NaN values in the correlation function during integration (default is ``False``).

        kw_window : dict, optional
            Optional arguments for window matrix calculation:
            - 'RR': :class:`Count2` instance with finer `s`-binning, to override ``estimator.get('RR')``
            - 'resolution' (default=1): number of evaluation points per RR-bin.
            If provided, also returns the window matrix for convolving theory.

        kw_covariance : dict, optional
            Optional arguments for jackknife covariance estimation (if input :class:`Count2JackknifeCorrelation`).
            If provided, also returns the covariance matrix of the multipoles.

        Returns
        -------
        poles : Count2CorrelationPole, Count2CorrelationPoles, Count2CorrelationWedge, Count2CorrelationWedges, Count2CorrelationWp
            Correlation function multipoles.

        window : WindowMatrix, optional
            Window matrix for convolving theory (returned if ``kw_window`` is provided).

        covariance : CovarianceMatrix, optional
            Covariance matrix of the multipoles (returned if ``kw_covariance`` is provided).
            """
        mode, kwargs = _get_project_mode(self, mode=mode, **kwargs)
        if mode == 'poles':
            return _project_to_poles(self, **kwargs)
        if mode == 'wedges':
            return _project_to_wedges(self, **kwargs)
        if mode == 'wp':
            return _project_to_wp(self, **kwargs)

    @classmethod
    def _sumweight(cls, observables, name):
        if name in ['normalized_counts']:
            s = sum(observable.get('DD').norm for observable in observables)
            return [observable.get('DD').norm / s for observable in observables]
        return [1] * len(observables)  # just sum norm

    @classmethod
    def mean(cls, *args, **kwargs):
        return NotImplementedError('mean not defined')


@register_type
class Count2JackknifeCorrelation(Count2Correlation):
    """
    Correlation function estimator for two-point statistics, with jackknife realizations.

    Parameters
    ----------
    estimator : str, optional
        Estimator type ('landyszalay' or 'natural'). Default is 'landyszalay'.

    attrs : dict, optional
        Additional attributes.

    kwargs : dict
        Pair count observables, e.g. DD, RR, DR, RD, DS, SD, SS.
    """
    _name = 'count2jackknifecorrelation'

    @property
    def realizations(self):
        """List of jackknife realizations."""
        return self.get('RR').realizations

    @property
    def nrealizations(self):
        """Number of jackknife realizations."""
        return self.get('RR').nrealizations

    def realization(self, ii, **kwargs):
        """
        Return jackknife realization ``ii``.

        Parameters
        ----------
        ii : int
            Label of jackknife realization.

        kwargs : dict
            Optional arguments for :meth:`Count2JackknifeCorrelation.realization`.

        Returns
        -------
        estimator : Count2Correlation
            Two-point estimator for realization ``ii``.
        """
        kw = {name: self.get(name).realization(ii, **kwargs) for name in self.count_names}
        return Count2Correlation(**kw, estimator=self.estimator, attrs=self.attrs)

    def value(self, return_type='nparray'):
        """
        Return the correlation function value.

        Parameters
        ----------
        return_type : str, None, optional
            If 'nparray', return a numpy array.
            If ``None``, return a :class:`Count2Correlation` instance.

        Returns
        -------
        estimator : array, Count2Correlation
            Two-point estimator.
        """
        value = super().value()
        if return_type is None:
            kw = {name: self.get(name) for name in self.count_names}
            return Count2Correlation(**kw, estimator=self.estimator, attrs=self.attrs)
        return value

    def cov(self, return_type='nparray', **kwargs):
        """
        Return the correlation function covariance.

        Parameters
        ----------
        return_type : str, None, optional
            If 'nparray', return a numpy array.
            If ``None``, return a :class:`CovarianceMatrix` instance.

        kwargs : dict
            Optional arguments for :meth:`Count2JackknifeCorrelation.realization`.

        Returns
        -------
        covariance : array, CovarianceMatrix
        """
        realizations = [self.realization(ii, **kwargs) for ii in self.realizations]
        cov = (len(realizations) - 1) * np.cov([realization.value().ravel() for realization in realizations], rowvar=False, ddof=0)
        cov = np.atleast_2d(cov)
        if return_type is None:
            mean = realizations[0].mean(realizations)
            return CovarianceMatrix(observable=mean, value=cov)
        return cov


def _matrix_lininterp(xout, xin):
    # Matrix for linear interpolation
    toret = np.zeros((len(xout), len(xin)), dtype='f8')
    for iout, xout in enumerate(xout):
        iin = np.searchsorted(xin, xout, side='right') - 1
        if 0 <= iin < len(xin) - 1:
            frac = (xout - xin[iin]) / (xin[iin + 1] - xin[iin])
            toret[iout, iin] = 1. - frac
            toret[iout, iin + 1] = frac
        elif np.isclose(xout, xin[-1]):
            toret[iout, iin] = 1.
    return toret


def _matrix_bininteg(list_edges, resolution=1):
    r"""
    Build a window matrix for binning in the continuous limit.

    This implements the integral of the theory :math:`f`, :math:`\int dx\, x^2 f(x) / \int dx\, x^2`, over each bin.

    Parameters
    ----------
    list_edges : list of array-like
        List of bin edges for each axis (e.g., separation, k, etc.).
    resolution : int, default=1
        Number of evaluation points per bin for the integral (higher = more accurate).

    Returns
    -------
    xin : ndarray
        Input theory coordinates (e.g., bin centers).
    edgesin : ndarray
        Input theory bin edges.
    full_matrix : ndarray
        Window matrix mapping theory to binned observable.
    """
    resolution = int(resolution)
    if resolution <= 0:
        raise ValueError('resolution must be a strictly positive integer')
    if not isinstance(list_edges, (tuple, list)): list_edges = [list_edges]

    step = min(np.diff(edges, axis=-1).min() for edges in list_edges) / resolution
    start, stop = min(np.min(edges) for edges in list_edges), max(np.max(edges) for edges in list_edges)
    edgesin = np.arange(start, stop + step / 2., step)
    edgesin = np.column_stack([edgesin[:-1], edgesin[1:]])
    xin = 3. / 4. * (edgesin[..., 1]**4 - edgesin[..., 0]**4) / (edgesin[..., 1]**3 - edgesin[..., 0]**3)

    matrices = []
    for edges in list_edges:
        x, w = [], []
        for ibin, limit in enumerate(edgesin):
            edge = np.linspace(*limit, resolution + 1)
            #x.append((edge[1:] + edge[:-1]) / 2.)
            x.append(3. / 4. * (edge[1:]**4 - edge[:-1]**4) / (edge[1:]**3 - edge[:-1]**3))
            row = np.zeros(len(edgesin) * resolution, dtype='f8')
            tmp = edge[1:]**3 - edge[:-1]**3
            row[ibin * resolution:(ibin + 1) * resolution] = tmp / tmp.sum()
            w.append(row)
        matrices.append(np.column_stack(w).dot(_matrix_lininterp(np.concatenate(x), xin)))  # linear interpolation * integration weights
    full_matrix = []
    for i, mat in enumerate(matrices):
        row = []
        for iin, _ in enumerate(matrices):
            if i == iin:
                row.append(mat)
            else:
                row.append(np.zeros_like(mat))
        full_matrix.append(row)
    full_matrix = np.block(full_matrix)
    return xin, edgesin, full_matrix


def _window_matrix_RR(counts, sedges, muedges, out_sedges, ells=(0, 2, 4), out_ells=(0, 2, 4), resolution=1):
    r"""
    Construct the window matrix for binning a theoretical correlation function onto observed bins.

    Parameters
    ----------
    counts : ndarray
        2D array of RR (random-random) weighted pair counts, shape ``(len(sedges), len(muedges))``.
        Used to weight the window matrix according to the survey geometry.

    sedges : array-like
        Bin edges for separation :math:`s` corresponding to `counts`.

    muedges : array-like
        Bin edges for angle :math:`\mu` corresponding to `counts`.

    out_sedges : list
        List of output :math:`s`-edges (observable bins).

    ells : tuple of int, optional
        Input theory multipoles :math:`\ell` (default is ``(0, 2, 4)``).

    out_ells : tuple of int, optional
        Output observable multipoles :math:`\ell` (default is ``(0, 2, 4)``).

    resolution : int, optional
        Number of evaluation points per ``sedges`` bin for the integral (higher values yield more accurate integration).

    Returns
    -------
    sin : ndarray
        Array of input theory coordinates (typically bin centers for :math:`s`).

    edgesin : ndarray
        Array of input theory bin edges.

    full_matrix : ndarray
        Window matrix mapping theory multipoles to observed binned multipoles.
        Shape depends on the number of output bins and multipoles.
    """
    from scipy import special
    sin, edgesin, binmatrix = _matrix_bininteg(sedges, resolution=resolution)  # binmatrix shape (len(sin), len(sinedges) - 1)
    full_matrix, idxin = [], []
    if not isinstance(out_sedges, (tuple, list)): out_sedges = [out_sedges] * len(out_ells)
    for ellout, out_sedges in zip(out_ells, out_sedges):
        mask = (sedges[None, ..., 0] >= out_sedges[:, None, ..., 0]) & (sedges[None, ..., 1] <= out_sedges[:, None, ..., 1])
        row = []
        for ellin in ells:
            integ = (special.legendre(ellout) * special.legendre(ellin)).integ()
            matrix = np.zeros_like(mask, dtype='f8')
            #print(idx)
            for iout in range(matrix.shape[0]):
                idx = np.flatnonzero(mask[iout])
                count = counts[idx]
                count_mu = np.sum(count, axis=0)
                mask_nonzero = count_mu != 0.
                count_mu[~mask_nonzero] = 1.
                tmp = count / count_mu
                # Integration over mu
                tmp = (2. * ellout + 1.) * np.sum(tmp * mask_nonzero * (integ(muedges[..., 1]) - integ(muedges[..., 0])), axis=-1) / np.sum(mask_nonzero * (muedges[..., 1] - muedges[..., 0]))  # normalization of mu-integral over non-empty s-rebinned RR(s, mu) bins
                matrix[iout, idx] = tmp
            matrix = matrix.dot(binmatrix)
            row.append(matrix)
        full_matrix.append(row)
        idxin.append(np.any([np.any(matrix != 0, axis=0) for matrix in row], axis=0))  # all sin for which wmatrix != 0
    idxin = np.any(idxin, axis=0)
    # Let's remove useless sin
    sin, edgesin = sin[idxin], edgesin[idxin]
    full_matrix = [[matrix[:, idxin] for matrix in row] for row in full_matrix]
    full_matrix = np.block(full_matrix)
    return sin, edgesin, full_matrix


def _project_to_poles(estimator, ells=None, ignore_nan=False, kw_window=None, kw_covariance=None):
    r"""
    Project a two-dimensional correlation function :math:`\xi(s, \mu)` onto Legendre polynomial multipoles.

    This computes the multipole moments:
    :math:`\xi_\ell(s) = (2\ell + 1) / 2 \int_{-1}^{1} \xi(s, \mu) L_\ell(\mu) d\mu`
    for each separation bin :math:`s` and multipole order :math:`\ell`.

    Parameters
    ----------
    estimator : Count2Correlation, Count2JackknifeCorrelation
        Estimator for the :math:`(s, \mu)` correlation function.

    ells : list of int, optional
        Orders of Legendre polynomials to project onto (default is ``[0, 2, 4]``).

    ignore_nan : bool, optional
        If ``True``, ignore NaN values in the correlation function during integration (default is ``False``).

    kw_window : dict, optional
        Optional arguments for window matrix calculation:
        - 'RR': :class:`Count2` instance with finer `s`-binning, to override ``estimator.get('RR')``
        - 'resolution' (default=1): number of evaluation points per RR-bin.
        If provided, also returns the window matrix for convolving theory.

    kw_covariance : dict, optional
        Optional arguments for jackknife covariance estimation (if input :class:`Count2JackknifeCorrelation`).
        If provided, also returns the covariance matrix of the multipoles.

    Returns
    -------
    poles : Count2CorrelationPole or Count2CorrelationPoles
        Correlation function multipoles.

    covariance : CovarianceMatrix, optional
        Covariance matrix of the multipoles (returned if ``kw_covariance`` is provided).

    window : WindowMatrix, optional
        Window matrix for convolving theory (returned if ``kw_window`` is provided).

    """
    return_window = kw_window is not None
    kw_window = dict(kw_window or {})
    return_covariance = kw_covariance is not None
    kw_covariance = dict(kw_covariance or {})
    from scipy import special
    assert list(estimator.coords()) == ['s', 'mu']
    if ells is None: ells = [0, 2, 4]
    isscalar = not isinstance(ells, list)
    if isscalar: ells = [ells]
    ells = list(ells)
    sedges = estimator.edges('s')
    muedges = estimator.edges('mu')
    dmu = np.diff(muedges, axis=-1)[..., 0]
    values, mask = [], []
    estimator_value, estimator_RR, estimator_norm = estimator.value(), estimator.get('RR').value(), estimator.get('DD').norm
    for ell in ells:
        # \sum_{i} \xi_{i} \int_{\mu_{i}}^{\mu_{i+1}} L_{\ell}(\mu^{\prime}) d\mu^{\prime}
        poly = special.legendre(ell).integ()(muedges)
        legendre = (2 * ell + 1) * np.diff(poly, axis=-1)[..., 0]
        if ignore_nan:
            mask = []
            value, RR0, norm = (np.empty(estimator_value.shape[0], dtype=estimator_value.dtype) for i in range(3))
            for i_s, value_s in enumerate(estimator_value):
                mask_s = my_ones_like(value_s, dtype='?')
                mask_s &= ~np.isnan(value_s)
                mask.append(mask_s)
                value[i_s] = np.sum(value_s[mask_s] * legendre[mask_s], axis=-1) / np.sum(dmu[mask_s])
                RR0[i_s] = np.sum(estimator_RR[i_s, mask_s])
                norm[i_s] = np.sum(estimator_norm[i_s, mask_s])
        else:
            value = np.sum(estimator_value * legendre, axis=-1) / np.sum(dmu)
            RR0 = np.sum(estimator_RR.sum(axis=-1))
            norm = np.sum(estimator_norm.sum(axis=-1))

        value = Count2CorrelationPole(value=value, s=estimator.coords('s'), s_edges=sedges, ell=ell, RR0=RR0, norm=norm, attrs=estimator.attrs)
        values.append(value)
    if isscalar:
        values = values[0]
    else:
        values = Count2CorrelationPoles(values)

    toret = [values]
    if return_covariance:
        realizations = [_project_to_poles(estimator.realization(ii, **kw_covariance), ells=ells[0] if isscalar else ells, ignore_nan=ignore_nan) for ii in estimator.realizations]
        cov = (len(realizations) - 1) * np.cov([realization.value().ravel() for realization in realizations], rowvar=False, ddof=0)
        cov = np.atleast_2d(cov)
        mean = realizations[0].mean(realizations)
        toret.append(CovarianceMatrix(observable=mean, value=cov))
    if return_window:
        RR = kw_window.get('RR', None)
        if RR is None: RR = estimator.get('RR')
        ells = kw_window.get('ells', (0, 2, 4))
        s, s_edges, window = _window_matrix_RR(RR.value(), RR.edges('s'), RR.edges('mu'), estimator.edges('s'), ells=ells, out_ells=ells, resolution=kw_window.get('resolution', 1))
        theory = Count2CorrelationPoles([Count2CorrelationPole(s=s, s_edges=s_edges, value=np.zeros_like(s), ell=ell) for ell in ells])
        window = WindowMatrix(value=window, observable=values.clone(value=np.zeros_like(values.value())), theory=theory)
        toret.append(window)
    return toret if len(toret) > 1 else toret[0]


def _project_to_wedges(estimator, wedges=None, ignore_nan=False, kw_covariance=None):
    r"""
    Project a two-dimensional correlation function :math:`\xi(s, \mu)` onto wedges (integrating over :math:`\mu`).

    Parameters
    ----------
    estimator : Count2Correlation
        Estimator for the :math:`(s, \mu)` correlation function.

    wedges : list of tuples, optional
        :math:`mu`-edges (min, max) of each wedge, e.g. [(-1., -2. / 3), (-2. / 3, -1. / 3), (-1. / 3, 0.), (0., 1. / 3), (1. / 3, 2. / 3), (2. / 3, 1.)]
        or [-1., -2. / 3, -1. / 3, 0., 1. / 3, 2. / 3, 1.]

    ignore_nan : bool, optional
        If ``True``, ignore NaN values in the correlation function during integration (default is ``False``).

    kw_covariance : dict, optional
        Optional arguments for jackknife covariance estimation (if input :class:`Count2JackknifeCorrelation`).
        If provided, also returns the covariance matrix of the multipoles.

    Returns
    -------
    wedges : Count2CorrelationWedge or Count2CorrelationWedges
        Correlation function wedges.

    covariance : CovarianceMatrix, optional
        Covariance matrix of the wedges (returned if ``kw_covariance`` is provided).
    """
    return_covariance = kw_covariance is not None
    kw_covariance = dict(kw_covariance or {})
    assert list(estimator.coords()) == ['s', 'mu']
    if wedges is None: wedges = [-1., -2. / 3, -1. / 3, 0., 1. / 3, 2. / 3, 1.]
    isscalar = not isinstance(wedges, list)
    if isscalar: wedges = [wedges]
    if np.ndim(wedges[0]) == 0: wedges = [wedges]
    sedges = estimator.edges('s')
    muedges = estimator.edges('mu')
    mumid = np.mean(muedges, axis=-1)
    dmu = np.diff(muedges, axis=-1)[..., 0]
    values, mask = [], []
    estimator_value, estimator_RR, estimator_norm = estimator.value(), estimator.get('RR').value(), estimator.get('DD').norm
    for wedge in wedges:
        mask_w = (mumid >= wedge[0]) & (mumid < wedge[1])
        mask_ws = []
        if ignore_nan:
            value, RR0, norm = (np.empty(estimator_value.shape[0], dtype=estimator_value.dtype) for i in range(3))
            for i_s, value_s in enumerate(estimator_value):
                mask_s = mask_w.copy()
                if ignore_nan: mask_s &= ~np.isnan(value_s)
                mask_ws.append(mask_s)
                value[i_s] = np.sum(value_s[mask_s] * dmu[mask_s], axis=-1) / np.sum(dmu[mask_s])
                RR0[i_s] = np.sum(estimator_RR[i_s, mask_s])
                norm[i_s] = np.sum(estimator_norm[i_s, mask_s])
            mask.append(np.array(mask_ws, dtype='?'))
        else:
            value = np.sum(estimator_value[:, mask_w] * dmu[mask_w], axis=-1) / np.sum(dmu[mask_w])
            RR0 = np.sum(estimator_RR.sum(axis=-1))
            norm = np.sum(estimator_norm.sum(axis=-1))
        value = Count2CorrelationWedge(value=value, s=estimator.coords('s'), s_edges=sedges, mu_edges=wedge, RR0=RR0, norm=norm, attrs=estimator.attrs)
        values.append(value)
    if isscalar:
        values = values[0]
    else:
        values = Count2CorrelationWedges(values)
    toret = [values]
    if return_covariance:
        realizations = [_project_to_wedges(estimator.realization(ii, **kw_covariance), wedges=wedges[0] if isscalar else wedges, ignore_nan=ignore_nan) for ii in estimator.realizations]
        cov = (len(realizations) - 1) * np.cov([realization.value().ravel() for realization in realizations], rowvar=False, ddof=0)
        cov = np.atleast_2d(cov)
        mean = realizations[0].mean(realizations)
        toret.append(CovarianceMatrix(observable=mean, value=cov))
    return toret if len(toret) > 1 else toret[0]


def _project_to_wp(estimator, ignore_nan=False, kw_covariance=None):
    r"""
    Integrate :math:`(r_p, \pi)` correlation function over :math:`\pi` to obtain :math:`w_p(r_p)`.

    Parameters
    ----------
    estimator : Count2Correlation
        Estimator for the :math:`(r_p, \pi)` correlation function.

    ignore_nan : bool, optional
        If ``True``, ignore NaN values in the correlation function during integration (default is ``False``).

    kw_covariance : dict, optional
        Optional arguments for jackknife covariance estimation (if input :class:`Count2JackknifeCorrelation`).
        If provided, also returns the covariance matrix of the multipoles.

    Returns
    -------
    wedges : Count2CorrelationWp
        Projected correlation function.

    covariance : CovarianceMatrix, optional
        Covariance matrix of the projected correlation function (returned if ``kw_covariance`` is provided).
    """
    return_covariance = kw_covariance is not None
    kw_covariance = dict(kw_covariance or {})
    assert list(estimator.coords()) == ['rp', 'pi']
    piedges = estimator.edges('pi')
    dpi = np.diff(piedges, axis=-1)[..., 0]
    estimator_value, estimator_RR, estimator_norm = estimator.value(), estimator.get('RR').value(), estimator.get('DD').norm
    mask = []
    if ignore_nan:
        value, RR0, norm = (np.empty(estimator_value.shape[0], dtype=estimator_value.dtype) for i in range(3))
        for i_rp, value_rp in enumerate(estimator_value):
            mask_rp = ~np.isnan(value_rp)
            mask.append(mask_rp)
            value[i_rp] = np.sum(value_rp[mask_rp] * dpi[mask_rp], axis=-1) * np.sum(dpi) / np.sum(dpi[mask_rp])  # extra factor to correct for missing bins
            RR0[i_rp] = np.sum(estimator_RR[i_rp, mask_rp], axis=-1)
            norm[i_rp] = np.sum(estimator_norm[i_rp, mask_rp], axis=-1)
    else:
        value = np.sum(estimator_value * dpi, axis=-1)
        RR0 = np.sum(estimator_RR, axis=-1)
        norm = np.sum(estimator_norm, axis=-1)
    toret = []
    value = Count2CorrelationWp(value=value, rp=estimator.coords('rp'), rp_edges=estimator.edges('rp'), pi_edges=estimator.edges('pi')[[0, -1], [0, 1]], RR0=RR0, norm=norm, attrs=estimator.attrs)
    toret.append(value)
    if return_covariance:
        realizations = [_project_to_wp(estimator.realization(ii, **kw_covariance), ignore_nan=ignore_nan) for ii in estimator.realizations]
        cov = (len(realizations) - 1) * np.cov([realization.value().ravel() for realization in realizations], rowvar=False, ddof=0)
        cov = np.atleast_2d(cov)
        mean = realizations[0].mean(realizations)
        toret.append(CovarianceMatrix(observable=mean, value=cov))
    return toret if len(toret) > 1 else toret[0]


@register_type
class Count2CorrelationPole(ObservableLeaf):
    r"""
    Container for a correlation function multipole :math:`\xi_\ell(s)`.

    Stores the binned correlation function for a given multipole order :math:`\ell`, including normalization and RR pair counts.

    Parameters
    ----------
    s : array-like
        Bin centers for separation :math:`s`.
    s_edges : array-like
        Bin edges for separation :math:`s`.
    value : array-like
        Correlation function multipole values for each bin.
    RR0 : array-like, optional
        (Isotropic-average of) RR (random-random) pair counts for each bin (default: ones).
    norm : array-like, optional
        Normalization factor (default: ones).
    ell : int, optional
        Multipole order :math:`\ell`.
    attrs : dict, optional
        Additional attributes.
    """
    _name = 'count2correlationpole'

    def __init__(self, s=None, s_edges=None, value=None, RR0=None, norm=None, ell=None, attrs=None, **kwargs):
        kw = dict(s=s, s_edges=s_edges)
        if s_edges is None: kw.pop('s_edges')
        self.__pre_init__(**kw, coords=['s'], attrs=attrs)
        if RR0 is None: RR0 = my_ones_like(value)
        if norm is None: norm = my_ones_like(value)
        self._values_names = ['value', 'norm', 'RR0']
        for name in list(kwargs):
            if name in self._values_names: pass
            elif name in ['volume']: self._values_names.append(name)
            else: raise ValueError('{name} not unknown')
        self._update(value=value, norm=norm, RR0=RR0, **kwargs)
        if ell is not None:
            self._meta['ell'] = ell

    def _binweight(self, name=None):
        # weight, normalized
        if name == 'RR0':
            return False, False
        return self.RR0, True

    @classmethod
    def _sumweight(cls, observables, name=None):
        if name is None or name in ['value']:
            s = sum(observable.norm for observable in observables)
            return [observable.norm / s for observable in observables]
        if name in ['RR0']:
            return [1. / len(observables)] * len(observables)
        return [1] * len(observables)  # just sum

    def _plabel(self, name):
        if name == 's':
            return r'$s$ [$\mathrm{Mpc}/h$]'
        if name == 'value':
            return r'$\xi_\ell(s)$'
        return None

    @plotter
    def plot(self, fig=None, **kwargs):
        r"""
        Plot a correlation function multipole.

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
        ax.plot(self.s, self.s**2 * self.value(), **kwargs)
        ax.set_xlabel(self._plabel('s'))
        ax.set_ylabel(r'$s^2 \xi_\ell(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        return fig


@register_type
class Count2CorrelationPoles(ObservableTree):
    r"""
    Container for multiple correlation function multipoles :math:`\xi_\ell(s)`.

    Stores a collection of `Count2CorrelationPole` objects for different multipole orders :math:`\ell`.

    Parameters
    ----------
    poles : list of Count2CorrelationPole
        List of correlation function multipole objects.
    ells : list of int, optional
        Multipole orders :math:`\ell` for each pole (default: inferred from `poles`).
    attrs : dict, optional
        Additional attributes.
    """
    _name = 'count2correlationpoles'

    def __init__(self, poles, ells=None, attrs=None):
        """Initialize correlattion function multipoles."""
        if ells is None: ells = [pole.ell for pole in poles]
        super().__init__(poles, ells=ells, attrs=attrs)

    @plotter
    def plot(self, fig=None):
        r"""
        Plot the correlattion function multipoles.

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


@register_type
class Count2CorrelationWedge(ObservableLeaf):
    r"""
    Container for a correlation function wedge :math:`\xi_(s, \mu)`.

    Stores the binned correlation function for a given :math:`\mu`-wedge, including normalization and RR pair counts.

    Parameters
    ----------
    s : array-like
        Bin centers for separation :math:`s`.
    s_edges : array-like
        Bin edges for separation :math:`s`.
    value : array-like
        Correlation function values for each bin.
    RR0 : array-like, optional
        (Isotropic-average of) RR (random-random) pair counts for each bin (default: ones).
    norm : array-like, optional
        Normalization factor (default: ones).
    mu_edges : tuple, optional
        :math:`\mu`-edges for this wedge (array of size 2).
    attrs : dict, optional
        Additional attributes.
    """
    _name = 'count2correlationwedge'

    def __init__(self, s=None, s_edges=None, mu_edges=None, value=None, RR0=None, norm=None, attrs=None):
        if RR0 is None: RR0 = my_ones_like(value)
        if norm is None: norm = my_ones_like(value)
        super().__init__(s=s, s_edges=s_edges, value=value, RR0=RR0, norm=norm, coords=['s'], attrs=attrs)
        if mu_edges is not None:
            mu_edges = np.array(mu_edges)
            assert mu_edges.size == 2
            self._meta['mu_edges'] = mu_edges

    def _binweight(self, name=None):
        # weight, normalized
        return Count2CorrelationPole._binweight(self, name=name)

    @classmethod
    def _sumweight(cls, observables, name=None):
        return Count2CorrelationPole._sumweight(observables, name=name)

    def _plabel(self, name):
        if name == 's':
            return r'$s$ [$\mathrm{Mpc}/h$]'
        if name == 'value':
            return r'$\xi(s, \mu)$'
        return None

    @plotter
    def plot(self, fig=None, **kwargs):
        r"""
        Plot a correlation function wedge.

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
        ax.plot(self.s, self.s**2 * self.value(), **kwargs)
        ax.set_xlabel(self._plabel('s'))
        ax.set_ylabel(r'$s^2 \xi(s, \mu)$ [$(\mathrm{Mpc}/h)^{2}$]')
        return fig



@register_type
class Count2CorrelationWedges(ObservableTree):
    r"""
    Container for multiple correlation function wedges :math:`\xi_(s, \mu)`.

    Stores a collection of `Count2CorrelationWedge` objects for different wedges.

    Parameters
    ----------
    poles : list of Count2CorrelationWedge
        List of correlation function wedges.
    attrs : dict, optional
        Additional attributes.
    """
    _name = 'count2correlationwedges'

    def __init__(self, wedges, attrs=None):
        """Initialize correlattion function multipoles."""
        wedges_labels = [f'w{i + 1:d}' for i in range(len(wedges))]
        super().__init__(wedges, wedges=wedges_labels, attrs=attrs)

    def _eq_label(self, label1, label2):
        if not isinstance(label2, str):
            label2 = np.array(label2)
            branch = self._branches[self.wedges.index(label1)]
            return np.allclose(label2, branch.mu_edges)
        return label1 == label2

    @plotter
    def plot(self, fig=None):
        r"""
        Plot the correlattion function wedges.

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
        for wedge in self.wedges:
            wedge = self.get(wedge)
            wedge.plot(fig=ax, label=rf'${wedge.mu_edges[0]:.2f} < \mu < {wedge.mu_edges[1]:.2f}$')
        ax.legend(frameon=False)
        return fig


@register_type
class Count2CorrelationWp(ObservableLeaf):
    r"""
    Container for the projected correlation function :math:`w_p`.

    Stores the binned correlation function, including normalization and RR pair counts.

    Parameters
    ----------
    s : array-like
        Bin centers for separation :math:`s`.
    s_edges : array-like
        Bin edges for separation :math:`s`.
    value : array-like
        Correlation function values for each bin.
    RR0 : array-like, optional
        (Isotropic-average of) RR (random-random) pair counts for each bin (default: ones).
    norm : array-like, optional
        Normalization factor (default: ones).
    pi_edges : tuple, optional
        :math:`\pi`-edges (array of size 2).
    attrs : dict, optional
        Additional attributes.
    """
    _name = 'count2correlationwp'

    def __init__(self, rp=None, rp_edges=None, pi_edges=None, value=None, RR0=None, norm=None, attrs=None):
        if RR0 is None: RR0 = my_ones_like(value)
        if norm is None: norm = my_ones_like(value)
        super().__init__(rp=rp, rp_edges=rp_edges, value=value, RR0=RR0, norm=norm, coords=['rp'], attrs=attrs)
        if pi_edges is not None:
            pi_edges = np.array(pi_edges)
            assert pi_edges.size == 2
            self._meta['pi_edges'] = pi_edges

    def _binweight(self, name=None):
        # weight, normalized
        return Count2CorrelationPole._binweight(self, name=name)

    @classmethod
    def _sumweight(cls, observables, name=None):
        return Count2CorrelationPole._sumweight(observables, name=name)

    def _plabel(self, name):
        if name == 'rp':
            return r'$r_p$ [$\mathrm{Mpc}/h$]'
        if name == 'value':
            return r'$w_p(r_p)$'
        return None

    @plotter
    def plot(self, fig=None):
        r"""
        Plot the projected correlation function :math:`w(r_\mathrm{p})`.

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
        ax.plot(self.rp, self.rp * self.value())
        return fig