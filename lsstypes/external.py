import numpy as np

from .types import (Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh3SpectrumPole, Mesh3SpectrumPoles, Count2, Count2Jackknife,
Count2Correlation, Count2JackknifeCorrelation, Mesh2CorrelationPole, Mesh2CorrelationPoles)
from .utils import my_ones_like


def from_pypower(power, complex=True):
    r"""
    Convert a :mod:`pypower` power spectrum or correlation function object to :class:`Mesh2SpectrumPoles` or :class:`Mesh2CorrelationPoles` format.

    Parameters
    ----------
    power : object
        Input power spectrum or correlation function object.

    complex : bool, default=True
        If ``False``, take real part of even poles and imaginary part of odd poles.

    Returns
    -------
    Mesh2SpectrumPoles, Mesh2CorrelationPoles
        Poles object containing the converted power spectrum or correlation function data.
    """
    def to_real(value, ell=0):
        if complex:
            return value + 0j
        if ell % 2:
            return np.imag(value)
        return np.real(value)

    if hasattr(power, 's'):  # correlation
        corr = power
        ells = corr.ells
        poles = []
        for ill, ell in enumerate(ells):
            s_edges = np.column_stack([corr.edges[0][:-1], corr.edges[0][1:]])
            s = corr.s
            ones = (0. >= corr.edges[0][:-1]) & (0. < corr.edges[0][1:])
            num_raw = corr.corr[ill] * corr.wnorm + (ell == 0) * corr.shotnoise_nonorm
            poles.append(Mesh2CorrelationPole(s=s, s_edges=s_edges, num_raw=to_real(num_raw, ell=ell),
                                        num_shotnoise=to_real(corr.shotnoise_nonorm * ones * (ell == 0), ell=ell),
                                        norm=to_real(corr.wnorm * ones),
                                        nmodes=corr.nmodes, ell=ell))
        return Mesh2CorrelationPoles(poles)
    else:
        ells = power.ells
        poles = []
        for ill, ell in enumerate(ells):
            k_edges = np.column_stack([power.edges[0][:-1], power.edges[0][1:]])
            k = power.k
            ones = my_ones_like(power.power_nonorm[ill])
            num_raw = power.power[ill] * power.wnorm + (ell == 0) * power.shotnoise_nonorm
            poles.append(Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=to_real(num_raw, ell=ell),
                                        num_shotnoise=to_real(power.shotnoise_nonorm * ones * (ell == 0), ell=ell),
                                        norm=to_real(power.wnorm * ones),
                                        nmodes=power.nmodes, ell=ell))
        return Mesh2SpectrumPoles(poles)


def from_pycorr(correlation):
    r"""
    Convert a **pycorr** correlation object to :class:`Count2Correlation` or :class:`Count2JackknifeCorrelation` format.

    Parameters
    ----------
    correlation : object
        Input correlation object.

    Returns
    -------
    Count2Correlation or Count2JackknifeCorrelation
        Correlation object containing the converted pair counts, with jackknife support if applicable.
    """
    counts = {}
    is_jackknife = correlation.name.startswith('jackknife')
    estimator = correlation.name.replace('jackknife-', '')

    def get_count(count):
        if count.mode == 'smu':
            coord_names = ['s', 'mu']
        elif count.mode == 'rppi':
            coord_names = ['rp', 'pi']
        elif count.mode == 's':
            coord_names = ['s']
        elif count.mode == 'theta':
            coord_names = ['theta']
        # check for not None (analytical pair counts)
        attrs = {name: getattr(count, name) for name in ['size1', 'size2']}
        if attrs['size2'] is None: attrs['size2'] = attrs['size1']  # size2 set to None in pycorr analytical auto pair counts
        coords = {coord_names[axis]: count.sepavg(axis=axis) for axis in range(count.ndim)}
        edges = {f'{coord_names[axis]}_edges': np.column_stack([count.edges[axis][:-1], count.edges[axis][1:]]) for axis in range(count.ndim)}
        return Count2(counts=count.wcounts, norm=my_ones_like(count.wcounts) * count.wnorm, **coords, **edges, coords=coord_names, attrs=attrs)

    for count_name in correlation.count_names:
        count = getattr(correlation, count_name)
        if is_jackknife:
            ii_counts = {realization: get_count(count) for realization, count in count.auto.items()}
            ij_counts = {realization: get_count(count) for realization, count in count.cross12.items()}
            ji_counts = {realization: get_count(count) for realization, count in count.cross21.items()}
            count = Count2Jackknife(ii_counts, ij_counts, ji_counts)
        else:
            count = get_count(count)
        counts[count_name.replace('1', '').replace('2', '')] = count
    return (Count2JackknifeCorrelation if is_jackknife else Count2Correlation)(estimator=estimator, **counts)


def from_triumvirate(spectrum, ells=None):
    r"""
    Convert a **triumvirate** object to :class:`Mesh2SpectrumPole` or :class:`Mesh3SpectrumPole` format.

    Parameters
    ----------
    spectrum : dict, list
        Input spectrum pole, or list of poles.
    ells : list, optional
        If `spectrum` is a list of poles, corresponding :math:`\ell`.

    Returns
    -------
    Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh3SpectrumPole or Mesh3SpectrumPoles
    """
    def get_edges(kbin):
        edges = (kbin[:-1] + kbin[1:]) / 2.
        edges = np.concatenate([[2 * edges[0] - edges[1]], edges, [2 * edges[-1] - edges[-2]]], axis=0)
        return np.column_stack([edges[:-1], edges[1:]])

    if isinstance(spectrum, (tuple, list)):
        assert ells is not None
        poles = []
        for ill, ell in enumerate(ells):
            poles.append(from_triumvirate(spectrum[ill], ells=ell))
        if isinstance(poles[0], Mesh2SpectrumPole):
            return Mesh2SpectrumPoles(poles)
        if isinstance(poles[0], Mesh3SpectrumPole):
            return Mesh3SpectrumPoles(poles)

    kw = dict()
    if ells is not None: kw.update(ell=ells)

    if 'pk_raw' in spectrum:

        def get_real(num):
            if ells is None:
                return num
            ell = int(ells)
            return num.real if ell % 2 == 0 else num.imag

        k_edges = get_edges(spectrum[f'kbin'].ravel())
        k = spectrum['keff']
        nmodes = spectrum['nmodes'].ravel()
        num_raw = get_real(spectrum['pk_raw'].ravel())
        num_shotnoise = get_real(spectrum['pk_shot'].ravel())
        return Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=num_raw, num_shotnoise=num_shotnoise, nmodes=nmodes, **kw)

    if 'bk_raw' in spectrum:

        def get_real(num):
            if ells is None:
                return num
            ell1, ell2 = ells[:2]
            return num.real if (ell1 + ell2) % 2 == 0 else num.imag

        ndim = 2
        k_edges = np.concatenate([get_edges(spectrum[f'k{axis + 1:d}_bin'].ravel())[:, None, :] for axis in range(ndim)], axis=1)
        k = np.column_stack([spectrum[f'k{axis + 1:d}_eff'].ravel() for axis in range(ndim)])
        nmodes = np.column_stack([spectrum[f'nmodes_{axis + 1:d}'].ravel() for axis in range(ndim)]).prod(axis=-1)
        num_raw = get_real(spectrum['bk_raw'].ravel())
        num_shotnoise = get_real(spectrum['bk_shot'].ravel())
        basis = 'sugiyama'
        if all(np.allclose(k_edges[:, axis, :], k_edges[:, 0, :]) for axis in range(k_edges.shape[1])):
            basis = 'sugiyama-diagonal'
        if basis == 'sugiyama':
            # if triu, make full
            # Taken from https://github.com/MikeSWang/Triumvirate/blob/ea495d35e1330c8a5e1e52c22dea4ab51fdb5587/src/triumvirate/_arrayops.py#L789
            nbins = (np.sqrt(8 * num_raw.size + 1) - 1) / 2.
            if np.isclose(nbins, int(nbins)):
                nbins = int(nbins)
                # Reshape the flattened multipole array into a symmetric matrix.
                triu_indices = np.triu_indices(nbins)

                def get_full(value):
                    tmp = np.zeros((nbins, nbins), dtype=value.dtype)
                    tmp[triu_indices] = value
                    tmp.T[triu_indices] = value
                    return tmp.ravel()

                k_edges = np.concatenate([np.column_stack([get_full(k_edges[..., i, axis]) for axis in range(ndim)])[:, None, :] for i in range(2)], axis=1)
                k = np.column_stack([get_full(k[..., axis]) for axis in range(ndim)])
                nmodes, num_raw, num_shotnoise = (get_full(array) for array in (nmodes, num_raw, num_shotnoise))

        return Mesh3SpectrumPole(k=k, k_edges=k_edges, num_raw=num_raw, num_shotnoise=num_shotnoise, nmodes=nmodes, basis=basis, **kw)

    raise NotImplementedError('input triumvirate object is not recognized')
