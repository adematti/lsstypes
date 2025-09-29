import numpy as np

from .types import Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh3SpectrumPole, Mesh3SpectrumPoles, Count2, Count2Jackknife, Count2Correlation, Count2JackknifeCorrelation
from .utils import my_ones_like


def from_pypower(power):
    r"""
    Convert a :mod:`pypower` power spectrum object to :class:`Mesh2SpectrumPoles` format.

    Parameters
    ----------
    power : object
        Input power spectrum object.

    Returns
    -------
    Mesh2SpectrumPoles
        Poles object containing the converted power spectrum data.
    """
    ells = power.ells
    poles = []
    for ill, ell in enumerate(ells):
        k_edges = np.column_stack([power.edges[0][:-1], power.edges[0][1:]])
        k = power.k
        ones = my_ones_like(power.power_nonorm[ill])
        num_raw = power.power[ill] * power.wnorm + (ell == 0) * power.shotnoise_nonorm
        poles.append(Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=num_raw,
                                       num_shotnoise=power.shotnoise_nonorm * ones * (ell == 0),
                                       norm=power.wnorm * ones,
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
        meta = {name: getattr(count, name) for name in ['size1', 'size2']}
        coords = {coord_names[axis]: count.sepavg(axis=axis) for axis in range(count.ndim)}
        edges = {f'{coord_names[axis]}_edges': np.column_stack([count.edges[axis][:-1], count.edges[axis][1:]]) for axis in range(count.ndim)}
        return Count2(counts=count.wcounts, norm=my_ones_like(count.wcounts) * count.wnorm, **coords, **edges, coords=coord_names, meta=meta)

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
    spectrum : dict
        Input spectrum object.

    Returns
    -------
    Mesh2SpectrumPole or Mesh3SpectrumPole
    """
    def get_edges(kbin):
        edges = (kbin[:-1] + kbin[1:]) / 2.
        edges = np.concatenate([[2 * edges[0] - edges[1]], edges, [2 * edges[-1] - edges[-2]]], axis=0)
        return np.column_stack([edges[:-1], edges[1:]])

    if ells is not None and isinstance(ells, (tuple, list)):
        poles = []
        for ill, ell in enumerate(ells):
            poles.append(from_triumvirate(spectrum[ill], ell=ell))
        if isinstance(poles[0], Mesh2SpectrumPole):
            return Mesh2SpectrumPoles(poles)
        if isinstance(poles[0], Mesh3SpectrumPole):
            return Mesh3SpectrumPoles(poles)

    kw = dict()
    if ells is not None: kw.update(ell=ells)

    if 'pk_raw' in spectrum:
        k_edges = get_edges(spectrum[f'kbin'].ravel())
        k = spectrum['keff']
        nmodes = spectrum['nmodes'].ravel()
        num_raw = spectrum['pk_raw'].ravel()
        num_shotnoise = spectrum['pk_shot'].ravel()
        return Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=num_raw, num_shotnoise=num_shotnoise, nmodes=nmodes)

    if 'bk_raw' in spectrum:
        k_edges = np.concatenate([get_edges(spectrum[f'k{axis + 1:d}_bin'].ravel())[:, None, :] for axis in range(2)], axis=1)
        k = np.column_stack([spectrum[f'k{axis + 1:d}_eff'].ravel() for axis in range(2)])
        nmodes = np.column_stack([spectrum[f'nmodes_{axis + 1:d}'].ravel() for axis in range(2)]).prod(axis=-1)
        num_raw = spectrum['bk_raw'].ravel()
        num_shotnoise = spectrum['bk_shot'].ravel()
        basis = 'sugiyama'
        if all(np.allclose(k_edges[:, axis, :], k_edges[:, 0, :]) for axis in range(k_edges.shape[1])):
            basis = 'sugiyama-diagonal'
        return Mesh3SpectrumPole(k=k, k_edges=k_edges, num_raw=num_raw, num_shotnoise=num_shotnoise, nmodes=nmodes, basis=basis)

    raise NotImplementedError('input triumvirate object is not recognized')
