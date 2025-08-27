import numpy as np

from .base import ObservableLeaf, ObservableTree, LeafLikeObservableTree, register_type
from .utils import plotter


@register_type
class Mesh2SpectrumPole(ObservableLeaf):

    _name = 'mesh2spectrumpole'

    def __init__(self, k=None, k_edges=None, num_raw=None, num_shotnoise=None, norm=None, nmodes=None, attrs=None):
        self.__pre_init__(k=k, k_edges=k_edges, coords=['k'], attrs=attrs)
        if num_shotnoise is None: num_shotnoise = np.zeros_like(num_raw)
        if norm is None: norm = np.ones_like(num_raw)
        if nmodes is None: nmodes = np.ones_like(num_raw, dtype='i4')
        self._update(k=k, k_edges=k_edges, num_raw=num_raw, num_shotnoise=num_shotnoise, norm=norm, nmodes=nmodes)

    def _update(self, **kwargs):
        self._values_names = ['value', 'num_shotnoise', 'norm', 'nmodes']
        for name in list(kwargs):
            if name in ['k', 'k_edges'] + self._values_names:
                self._data[name] = kwargs.pop(name)
        for name in list(kwargs):
            if name in ['num_raw']:
                self._data['value'] = (kwargs.pop(name) - self.num_shotnoise) / self.norm
        if kwargs:
            raise ValueError(f'Could not interpret arguments {kwargs}')

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


@register_type
class Mesh2SpectrumPoles(ObservableTree):

    _name = 'mesh2spectrumpoles'

    def __init__(self, poles, ells=(0,), attrs=None):
        """Initialize power spectrum multipoles."""
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
class Count2(ObservableLeaf):

    """Pair counts."""
    _name = 'count2'

    def __init__(self, counts=None, norm=None, attrs=None, **kwargs):
        self.__pre_init__(attrs=attrs, **kwargs)
        if norm is None: norm = np.ones_like(counts)
        self._update(counts=counts, norm=norm)

    def _binweight(self, name=None):
        # weight, normalized
        if name is None or name == 'normalized_counts':
            return False, False
        if name == 'norm':
            return False, True  # not normalized to avoid cases where weight is 0
        return self.normalized_counts, True

    def _update(self, **kwargs):
        if 'value' in kwargs:
            kwargs['normalized_counts'] = kwargs.pop('value')
        self._values_names = ['normalized_counts', 'norm']
        for name in list(kwargs):
            if name in self._coords_names + self._values_names:
                self._data[name] = kwargs.pop(name)
        for name in list(kwargs):
            if name in ['counts']:
                self._data['normalized_counts'] = kwargs.pop(name) / self.norm
        if kwargs:
            raise ValueError(f'Could not interpret arguments {kwargs}')

    def values(self, name=0):
        if name == 'counts':
            return self.normalized_counts * self.norm
        return ObservableLeaf.values(self, name=name)

    @classmethod
    def _sumweight(cls, observables, name):
        if name in ['normalized_counts']:
            s = sum(observable.norm for observable in observables)
            return [observable.norm / s for observable in observables]
        return [1] * len(observables)  # just sum norm


def _nan_to_zero(array):
    return np.where(np.isnan(array), 0., array)


@register_type
class Count2Jackknife(LeafLikeObservableTree):

    """Jackknife pair counts."""

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

    def value(self, *args, **kwargs):
        return Count2.value(self, *args, **kwargs)

    def values(self, *args, **kwargs):
        return Count2.values(self, *args, **kwargs)

    def coords(self, *args, **kwargs):
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
            if name in self._leaves[0]._meta:
                self._meta[name] = sum(count._meta[name] for count in self.get(cross='ii'))

    @property
    def nrealizations(self):
        return len(self._labels) // 3

    def realization(self, ii, correction='mohammad21'):
        """
        Return jackknife realization ``ii``.

        Parameters
        ----------
        ii : int
            Label of jackknife realization.

        correction : string, default='mohammad'
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
        counts = self.get(realization=ii, cross='ii').copy()
        for name in ['counts', 'norm']:
            counts._data[name] = self.values(name) - self.get(realization=ii, cross='ii').values(name)\
                                 - alpha * (self.get(realization=ii, cross='ij').values(name) + self.get(realization=ii, cross='ji').values(name))
        dcounts = counts._data.pop('counts')
        counts._data['normalized_counts'] = dcounts / counts._data['norm']
        for iaxis, axis in enumerate(self._coords_names):
            reduce_axis = tuple(iax for iax in range(len(self._coords_names)) if iax != iaxis)
            counts._data[axis] = _nan_to_zero(self.coords(axis=axis)) * self.values('counts') \
                                - _nan_to_zero(self.get(realization=ii, cross='ii').coords(axis)) * self.get(realization=ii, cross='ii').values('counts').sum(axis=reduce_axis)\
                                - alpha * (_nan_to_zero(self.get(realization=ii, cross='ij').coords(axis)) * self.get(realization=ii, cross='ij').values('counts').sum(axis=reduce_axis)\
                                         + _nan_to_zero(self.get(realization=ii, cross='ji').coords(axis)) * self.get(realization=ii, cross='ji').values('counts').sum(axis=reduce_axis))
            with np.errstate(divide='ignore', invalid='ignore'):
                counts._data[axis] /= dcounts.sum(axis=reduce_axis)
                # The above may lead to rounding errors
                # such that seps may be non-zero even if wcounts is zero.
                mask = dcounts != 0  # if ncounts / wcounts computed, good indicator of whether pairs exist or not
                # For more robustness we restrict to those separations which lie in between the lower and upper edges
                mask &= (counts._data[axis] >= self.edges(axis=axis)[..., 0]) & (counts._data[axis] <= self.edges(axis=axis)[..., 1])
                counts._data[axis][~mask] = np.nan
        for name in ['size1', 'size2']:
            if name in self._meta:
                counts._meta[name] = self._meta[name] - self.get(realization=ii, cross='ii')._meta[name]
        return counts

    def cov(self, **kwargs):
        """
        Return jackknife covariance (of flattened counts).

        Parameters
        ----------
        kwargs : dict
            Optional arguments for :meth:`realization`.

        Returns
        -------
        cov : array
            Covariance matrix.
        """
        return (self.nrealizations - 1) * np.cov([self.realization(ii, **kwargs).value().ravel() for ii in self.realizations], rowvar=False, ddof=0)



def _get_project_mode(mode=None, **kwargs):
    # Return projection mode depending on provided arguments
    if 'ell' in kwargs:
        kwargs['ells'] = kwargs.pop('ell')
    if mode is None:
        if 'ells' in kwargs:
            mode = 'poles'
        elif 'wedges' in kwargs:
            mode = 'wedges'
        elif 'pimax' in kwargs:
            mode = 'wp'
        else:
            mode = None
    else:
        assert isinstance(mode, str)
        mode = mode.lower()
    return mode, kwargs



@register_type
class Count2Correlation(LeafLikeObservableTree):

    """Correlation function."""
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
        super().__init__([kwargs[count_name] for count_name in count_names], count_name=count_names,
                         meta=dict(estimator=estimator, with_shifted=with_shifted), attrs=attrs)

    def value(self):
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
        return self.get('RR').coords(*args, **kwargs)

    def project(self, mode=None, **kwargs):
        mode, kwargs = _get_project_mode(mode=mode, **kwargs)
        if mode == 'poles':
            return _project_to_poles(self, **kwargs)
        if mode == 'wedges':
            return _project_to_wedges(self, **kwargs)
        if mode == 'wp':
            return _project_to_wp(self, **kwargs)


@register_type
class Count2JackknifeCorrelation(Count2Correlation):

    """Correlation function."""
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
            Optional arguments for :meth:`JackknifeTwoPointCounter.realization`.

        Returns
        -------
        estimator : BaseTwoPointEstimator
            Two-point estimator for realization ``ii``.
        """
        kw = {}
        for name in self.count_names:
            counts = getattr(self, name)
            try:
                kw[name] = counts.realization(ii, **kwargs)
            except AttributeError:
                kw[name] = counts  # in case counts are not jackknife, e.g. analytic randoms (but that'd be wrong!)
        return Count2Correlation(**kw, estimator=self.estimator, attrs=self.attrs)

    def cov(self, **kwargs):
        cov = (self.nrealizations - 1) * np.cov([self.realization(ii, **kwargs).value().ravel() for ii in self.realizations], rowvar=False, ddof=0)
        return np.atleast_2d(cov)


def _project_to_poles(estimator, ells=(0, 2, 4), return_cov=None, ignore_nan=False, rp=None, return_mask=False, **kwargs):
    r"""
    Project :math:`(s, \mu)` correlation function estimation onto Legendre polynomials.

    Parameters
    ----------
    estimator : BaseTwoPointEstimator
        Estimator for :math:`(s, \mu)` correlation function.

    ells : tuple, int, default=(0, 2, 4)
        Order of Legendre polynomial.

    return_sep : bool, default=True
        Whether (``True``) to return separation.

    return_cov : bool, default=None
        If ``True`` or ``None`` and input ``estimator`` holds (jackknife) :meth:`realization`,
        return covariance matrix estimate (for all successive ``ells``).
        If ``True`` and input ``estimator`` does not have :meth:`realization`,
        raise :class:`TwoPointEstimatorError`.

    ignore_nan : bool, default=False
        If ``True``, ignore NaN values of the correlation function in the integration.

    rp : tuple, default=None
        Optionally, tuple of min and max values for a :math:`r_{p} = s \sqrt(1 - \mu^{2})` cut.

    return_mask : bool, default=False
        Return mask of :math:`\mu`-bins that are summed over, for each :math:`s`; of shape ``estimator.shape``.

    kwargs : dict
        Optional arguments for :meth:`JackknifeTwoPointEstimator.realization`, when relevant.

    Returns
    -------
    sep : array
        Optionally, array of separation values.

    poles : array
        Correlation function multipoles.

    cov : array
        Optionally, covariance estimate (for all successive ``ells``), see ``return_cov``.
    """
    from scipy import special
    assert list(estimator.coords()) == ['s', 'mu']
    isscalar = not isinstance(ells, list)
    if isscalar: ells = [ells]
    ells = list(ells)
    sedges = estimator.edges('s')
    muedges = estimator.edges('mu')
    dmu = np.diff(muedges, axis=-1)[..., 0]
    values, mask = [], []
    corr = estimator.value()
    for ell in ells:
        # \sum_{i} \xi_{i} \int_{\mu_{i}}^{\mu_{i+1}} L_{\ell}(\mu^{\prime}) d\mu^{\prime}
        poly = special.legendre(ell).integ()(muedges)
        legendre = (2 * ell + 1) * np.diff(poly, axis=-1)[..., 0]
        if ignore_nan or rp:
            mask = []
            value = np.empty(corr.shape[0], dtype=corr.dtype)
            for i_s, corr_s in enumerate(corr):
                mask_s = np.ones_like(corr_s, dtype='?')
                if ignore_nan:
                    mask_s &= ~np.isnan(corr_s)
                if rp:
                    se = sedges[..., 0]
                    mue = np.max(np.abs(muedges), axis=1)  # take the most conservative limit
                    rp_s = se[i_s] * np.sqrt(1. - mue**2)
                    mask_s &= (rp_s >= rp[0]) & (rp_s < rp[-1])
                mask.append(mask_s)
                value[i_s] = np.sum(corr_s[mask_s] * legendre[mask_s], axis=-1) / np.sum(dmu[mask_s])
        else:
            value = np.sum(corr * legendre, axis=-1) / np.sum(dmu)
        value = ObservableLeaf(value=value, s=estimator.coords('s'), s_edges=sedges, coords=['s'])
        values.append(value)
    if isscalar:
        values = values[0]
    else:
        values = ObservableTree(values, ells=ells)
    toret = [values]
    if return_mask:
        if mask: mask = np.array(mask, dtype='?')
        else: mask = np.ones(estimator.shape, dtype='?')
        toret.append(mask)
    return toret if len(toret) > 1 else toret[0]


def _project_to_wedges(estimator, wedges=None, return_cov=None, ignore_nan=False, rp=None, return_mask=False, **kwargs):
    r"""
    Project :math:`(s, \mu)` correlation function estimation onto wedges (integrating over :math:`\mu`).

    Parameters
    ----------
    estimator : BaseTwoPointEstimator
        Estimator for :math:`(s, \mu)` correlation function.

    wedges : tuple, default=[-1., -2. / 3, -1. / 3, 0., 1. / 3, 2. / 3, 1.]
        :math:`mu`-wedges.
        Single or list of tuples :math:`(\mu_{\mathrm{min}}, \mu_{\mathrm{max}})`,
        or :math:`\mu`-edges :math:`(\mu_{0}, ..., \mu_{n})`,

    return_sep : bool, default=True
        Whether (``True``) to return separation.

    return_cov : bool, default=None
        If ``True`` or ``None`` and input ``estimator`` holds (jackknife) :meth:`realization`,
        return covariance matrix estimate (for all successive ``ells``).
        If ``True`` and input ``estimator`` does not have :meth:`realization`,
        raise :class:`TwoPointEstimatorError`.

    ignore_nan : bool, default=False
        If ``True``, ignore NaN values of the correlation functions in the integration.

    rp : tuple, default=None
        Optionally, tuple of min and max values for a :math:`r_{p} = s \sqrt(1 - \mu^{2})` cut.

    return_mask : bool, default=False
        Return mask of :math:`\mu`-bins that are summed over, for each :math:`s`; of shape ``(n,) + estimator.shape``,
        with :math:`n` the number of wedges.

    kwargs : dict
        Optional arguments for :meth:`JackknifeTwoPointEstimator.realization`, when relevant.

    Returns
    -------
    sep : array
        Optionally, array of separation values.

    wedges : array
        Correlation function wedges.

    cov : array
        Optionally, covariance estimate (for all successive ``wedges``), see ``return_cov``.
    """
    assert list(estimator.coords()) == ['s', 'mu']
    if wedges is None: wedges = [-1., -2. / 3, -1. / 3, 0., 1. / 3, 2. / 3, 1.]
    isscalar = not isinstance(wedges, list)
    if isscalar: wedges = [wedges]
    if np.ndim(wedges[0]) == 0: wedges = [wedges]
    sedges = estimator.edges('s')
    muedges = estimator.edges('mu')
    mumid = np.mean(muedges, axis=-1)
    dmu = np.diff(muedges, axis=-1)[..., 0]
    corr = estimator.value()
    values, mask = [], []
    for wedge in wedges:
        mask_w = (mumid >= wedge[0]) & (mumid < wedge[1])
        mask_ws = []
        if ignore_nan or rp:
            value = np.empty(corr.shape[0], dtype=corr.dtype)
            for i_s, corr_s in enumerate(corr):
                mask_s = mask_w.copy()
                if ignore_nan:
                    mask_s &= ~np.isnan(corr_s)
                if rp:
                    se = sedges[..., 0]
                    mue = np.max(np.abs(muedges), axis=-1)  # take the most conservative limit
                    rp_s = se[i_s] * np.sqrt(1. - mue**2)
                    mask_s &= (rp_s >= rp[0]) & (rp_s < rp[-1])
                mask_ws.append(mask_s)
                value[i_s] = np.sum(corr_s[mask_s] * dmu[mask_s], axis=-1) / np.sum(dmu[mask_s])
            mask.append(np.array(mask_ws, dtype='?'))
        else:
            value = np.sum(corr[:, mask_w] * dmu[mask_w], axis=-1) / np.sum(dmu[mask_w])
        value = ObservableLeaf(value=value, s=estimator.coords('s'), s_edges=sedges, coords=['s'])
        values.append(value)
    if isscalar:
        values = values[0]
    else:
        values = ObservableTree(values, wedge=wedges)
    toret = [values]
    if return_mask:
        if mask: mask = np.array(mask, dtype='?')
        else: mask = np.ones(corr.shape[:-1] + estimator.shape, dtype='?')
        toret.append(mask)
    return toret if len(toret) > 1 else toret[0]


def _project_to_wp(estimator, return_cov=None, ignore_nan=False, return_mask=False, **kwargs):
    r"""
    Integrate :math:`(r_{p}, \pi)` correlation function over :math:`\pi` to obtain :math:`w_{p}(r_{p})`.

    Parameters
    ----------
    estimator : BaseTwoPointEstimator
        Estimator for :math:`(r_{p}, \pi)` correlation function.

    return_sep : bool, default=True
        Whether (``True``) to return separation.

    return_cov : bool, default=None
        If ``True`` or ``None`` and input ``estimator`` holds (jackknife) :meth:`realization`,
        return covariance matrix estimate (for all successive ``ells``).
        If ``True`` and input ``estimator`` does not have :meth:`realization`,
        raise :class:`TwoPointEstimatorError`.

    ignore_nan : bool, default=False
        If ``True``, ignore NaN values of the correlation functions in the integration.

    return_mask : bool, default=False
        Return mask of :math:`\pi`-bins that are summed over, for each :math:`r_{p}`; of shape ``estimator.shape``.

    kwargs : dict
        Optional arguments for :meth:`JackknifeTwoPointEstimator.realization`, when relevant.

    Returns
    -------
    sep : array
        Optionally, array of separation values.

    wp : array
        Estimated :math:`w_{p}(r_{p})`.

    cov : array
        Optionally, covariance estimate, see ``return_cov``.
    """
    assert list(estimator.coords()) == ['rp', 'pi']
    piedges = estimator.edges('pi')
    dpi = np.diff(piedges, axis=-1)[..., 0]
    corr = estimator.value()
    mask = []
    if ignore_nan:
        value = np.empty(corr.shape[0], dtype=corr.dtype)
        for i_rp, corr_rp in enumerate(corr):
            mask_rp = ~np.isnan(corr_rp)
            mask.append(mask_rp)
            value[i_rp] = np.sum(corr_rp[mask_rp] * dpi[mask_rp], axis=-1) * np.sum(dpi) / np.sum(dpi[mask_rp])  # extra factor to correct for missing bins
    else:
        value = np.sum(corr * dpi, axis=-1)
    toret = []
    toret.append(value)
    if return_mask:
        if mask: mask = np.array(mask, dtype='?')
        else: mask = np.ones(estimator.shape, dtype='?')
        toret.append(mask)
    return toret if len(toret) > 1 else toret[0]
