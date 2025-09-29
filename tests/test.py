import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import lsstypes as types
from lsstypes import ObservableLeaf, ObservableTree, read, write
from lsstypes import Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh3SpectrumPole, Mesh3SpectrumPoles, Count2, Count2Jackknife, Count2Correlation, Count2JackknifeCorrelation
from lsstypes import WindowMatrix, CovarianceMatrix, GaussianLikelihood


def test_tree():

    test_dir = Path('_tests')

    s = np.linspace(0., 100., 51)
    mu = np.linspace(-1., 1., 101)
    rng = np.random.RandomState(seed=42)
    labels = ['DD', 'DR', 'RR']
    leaves = []
    for label in labels:
        counts = 1. + rng.uniform(size=(s.size, mu.size))
        leaves.append(ObservableLeaf(counts=counts, s=s, mu=mu, coords=['s', 'mu'], attrs=dict(los='x')))

    leaf = leaves[0]
    fn = test_dir / 'leaf.h5'
    write(fn, leaf)
    leaf2 = read(fn)
    assert leaf2 == leaf

    fn = test_dir / 'leaf.txt'
    write(fn, leaf)
    leaf2 = read(fn)
    assert leaf2 == leaf

    leaf2 = leaf.select(s=(10., 80.), mu=(-0.8, 1.))
    assert np.all(leaf2.coords('s') <= 80)
    assert np.all(leaf2.coords('mu') >= -0.8)
    assert leaf2.values('counts').ndim == 2
    assert leaf2.attrs == dict(los='x')
    leaf3 = leaf.at[...].select(s=(10., 80.), mu=(-0.8, 1.))
    assert leaf3.shape == leaf2.shape
    leaf4 = leaf.at(s=(10., 80.)).select(s=(20., 70.), mu=(-0.8, 1.))
    assert len(leaf4.coords('s')) == 41

    tree = ObservableTree(leaves, keys=labels)
    assert tree.labels(return_type='keys') == ['keys']
    assert tree.labels(return_type='unflatten') == {'keys': ['DD', 'DR', 'RR']}
    assert tree.labels(return_type='flatten') == [{'keys': 'DD'}, {'keys': 'DR'}, {'keys': 'RR'}]
    assert len(tree.value()) == tree.size
    tree2 = tree.at(keys='DD').select(s=(10., 80.))
    assert tree2.get(keys='DD').shape != tree2.get(keys='DR').shape
    tree2 = tree.select(s=(10., 80.))
    assert tree2.get(keys='DD').shape == tree2.get(keys='DR').shape
    assert tree.get(['DD', 'RR']).size == tree.size * 2 // 3
    assert tree.get(['DD', 'RR']).get('DD') == tree.get('DD')
    assert isinstance(tree.get('DD'), ObservableLeaf)
    assert isinstance(tree.get(['DD']), ObservableTree)

    RR = tree.get('RR').select(mu=(-0.8, 0.7))
    DD = tree.get('DD').match(RR)
    assert DD.shape == RR.shape
    assert np.allclose(DD.mu, RR.mu)
    tree2 = tree.clone(value=np.zeros(tree.size))
    assert np.allclose(tree2.value(), 0.)

    DD.concatenate([DD] * 3)

    k = np.linspace(0., 0.2, 21)
    spectrum = rng.uniform(size=k.size)
    leaf = ObservableLeaf(spectrum=spectrum, k=k, coords=['k'], attrs=dict(los='x'))
    tree2 = ObservableTree([tree, leaf], observable=['correlation', 'spectrum'])
    assert tree2.labels(level=None, return_type='keys') == ['observable', 'keys']
    assert tree2.labels(level=1, return_type='flatten') == [{'observable': 'correlation'}, {'observable': 'spectrum'}]
    assert tree2.labels(level=None, return_type='flatten') == [{'observable': 'correlation', 'keys': 'DD'}, {'observable': 'correlation', 'keys': 'DR'}, {'observable': 'correlation', 'keys': 'RR'}, {'observable': 'spectrum'}]

    fn = test_dir / 'tree.h5'
    write(fn, tree2)
    #tree3 = read(fn)
    #assert tree3 == tree2

    fn = test_dir / 'tree.txt'
    write(fn, tree2)
    tree3 = read(fn)
    assert tree3 == tree2


def test_at():

    def get_poles(seed=None):
        ells = [0, 2, 4]
        rng = np.random.RandomState(seed=seed)
        poles = []
        for ell in ells:
            k_edges = np.linspace(0., 0.2, 41)
            k_edges = np.column_stack([k_edges[:-1], k_edges[1:]])
            k = np.mean(k_edges, axis=-1)
            poles.append(Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=rng.uniform(size=k.size)))
        return Mesh2SpectrumPoles(poles, ells=ells)

    poles = get_poles()

    print(poles)
    print(poles.get(2))

    at = poles.at(2)
    at._hook = lambda new, transform: new
    poles2 = at.select(k=slice(0, None, 2))
    poles2 = poles.at(0).select(k=(0., 0.1))
    assert poles2.get(0).k.size == poles.get(0).k.size // 2

    poles2 = poles.at(2).at(k=(0.04, 0.14)).select(k=slice(0, None, 2))
    assert len(poles2.get(2).k) < len(poles.get(2).k)

    poles2 = poles.at(2).at(k=(0.1, 0.3)).select(k=slice(0, None, 2))
    assert len(poles2.get(2).k) < len(poles.get(2).k)

    poles2 = poles.at(2).at(k=(0.04, 0.14)).select(k=(0.1, 0.12)) #slice(0, None, 2))
    assert len(poles2.get(2).k) == 24


    poles2 = poles.at(2).clone(value=np.zeros(poles.get(2).size))
    assert np.allclose(poles2.get(2).value(), 0.)
    assert not np.allclose(poles2.value(), 0.)

    def get_counts():
        s_edges = np.linspace(0., 100., 21)
        s_edges = np.column_stack([s_edges[:-1], s_edges[1:]])
        mu_edges = np.linspace(-1., 1., 11)
        mu_edges = np.column_stack([mu_edges[:-1], mu_edges[1:]])
        s, mu = np.mean(s_edges, axis=-1), np.mean(mu_edges, axis=-1)
        rng = np.random.RandomState(seed=42)
        counts = 1. + rng.uniform(size=(s.size, mu.size))
        counts = Count2(counts=counts, norm=np.ones_like(counts), s=s, mu=mu, s_edges=s_edges, mu_edges=mu_edges, coords=['s', 'mu'], attrs=dict(los='x'))
        return counts

    counts = get_counts()
    tree = ObservableTree([counts, counts], keys=['DD', 'RR'])
    at = tree.at('DD')
    at._hook = lambda new, transform: new
    tree2 = at.select(mu=slice(0, None, 2))
    assert tree2.get('DD').shape[1] == tree.get('DD').shape[1] // 2

    tree = ObservableTree([tree, poles], observables=['correlation', 'spectrum'])
    tree2 = tree.at(observables='spectrum').at(0).select(k=(0., 0.1))
    assert np.all(tree2.get(observables='spectrum', ells=0).k < 0.1)


def test_matrix(show=False):

    test_dir = Path('_tests')

    def get_spectrum(size=40, seed=None):
        ells = [0, 2, 4]
        rng = np.random.RandomState(seed=seed)
        spectrum = []
        for ell in ells:
            k_edges = np.linspace(0., 0.2, size + 1)
            k_edges = np.column_stack([k_edges[:-1], k_edges[1:]])
            k = np.mean(k_edges, axis=-1)
            spectrum.append(Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=rng.uniform(size=k.size)))
        return Mesh2SpectrumPoles(spectrum, ells=ells)

    observable = get_spectrum(size=40)
    theory = get_spectrum(size=60)
    rng = np.random.RandomState(seed=None)
    value = rng.uniform(0., 1., size=(40 * 3, 60 * 3))
    winmat = WindowMatrix(value=value, observable=observable, theory=theory)
    obs = winmat.observable.select(k=slice(0, None, 2))
    assert isinstance(obs, Mesh2SpectrumPoles)

    winmat2 = types.sum([winmat, winmat])
    assert np.allclose(winmat2.value(), winmat.value())
    assert np.allclose(winmat2.observable.value(), winmat.observable.value())

    matrix2 = winmat.at.theory.select(k=slice(0, None, 2))
    assert np.allclose(matrix2.value().sum(axis=-1), winmat.value().sum(axis=-1))
    assert matrix2.shape[1] == winmat.shape[1] // 2
    winmat.plot(show=show)

    assert winmat.dot(winmat.theory).shape == (winmat.shape[0],)
    assert winmat.dot(winmat.theory, return_type=None).labels(level=None, return_type='flatten') == winmat.observable.labels(level=None, return_type='flatten')

    def test(matrix):
        fn = test_dir / 'matrix.h5'
        matrix.write(fn)
        matrix = read(fn)

        matrix2 = matrix.at.observable.at(2).select(k=slice(0, None, 2))
        assert matrix2.shape[0] < matrix.shape[0]
        assert matrix2.observable.get(2).size == matrix.observable.get(2).size // 2

        matrix2 = matrix.at.observable.at(2).at[...].select(k=slice(0, None, 2))
        assert matrix2.shape[0] < matrix.shape[0]
        assert matrix2.observable.get(2).size == matrix.observable.get(2).size // 2

        matrix2 = matrix.at.observable.at(2).at(k=(0.05, 0.15)).select(k=slice(0, None, 2))
        assert matrix2.shape[0] < matrix.shape[0]

        matrix2 = matrix.at.observable.get([0, 2])
        assert matrix2.shape[0] == matrix.shape[0] * 2 // 3

    test(winmat)
    winmat.plot_slice(indices=2, show=show)

    value = rng.uniform(0., 1., size=(40 * 3, 40 * 3))
    covmat = CovarianceMatrix(value=value, observable=observable)

    assert covmat.std().size == covmat.shape[0]
    assert covmat.corrcoef().shape == covmat.shape
    covmat.plot(show=show)
    covmat.plot_diag(show=show)
    covmat.plot_slice(indices=2, show=show)
    test(covmat)

    covmat = types.cov([get_spectrum(size=40, seed=seed) for seed in range(100)])
    covmat.plot(show=show)

    likelihood = GaussianLikelihood(observable=observable, window=winmat, covariance=covmat)

    fn = test_dir / 'likelihood.h5'
    likelihood.write(fn)
    likelihood = read(fn)

    likelihood2 = likelihood.at.observable.select(k=(0.05, 0.15))
    assert likelihood2.window.shape[0] < likelihood.window.shape[0]

    likelihood2 = likelihood.at.observable.get([0])
    assert likelihood2.window.shape[0] < likelihood.window.shape[0]

    likelihood2 = likelihood.at.observable.at(2).at[...].select(k=slice(0, None, 2))
    assert likelihood2.observable.get(2).size == likelihood.observable.get(2).size // 2

    chi2 = likelihood2.chi2(winmat.theory)


def test_rebin():

    def get_counts():
        s_edges = np.linspace(0., 100., 21)
        s_edges = np.column_stack([s_edges[:-1], s_edges[1:]])
        mu_edges = np.linspace(-1., 1., 11)
        mu_edges = np.column_stack([mu_edges[:-1], mu_edges[1:]])
        s, mu = np.mean(s_edges, axis=-1), np.mean(mu_edges, axis=-1)
        rng = np.random.RandomState(seed=42)
        counts = 1. + rng.uniform(size=(s.size, mu.size))
        counts = Count2(counts=counts, norm=np.ones_like(counts), s=s, mu=mu, s_edges=s_edges, mu_edges=mu_edges, coords=['s', 'mu'], attrs=dict(los='x'))
        return counts

    counts = get_counts()
    matrix = counts._transform(slice(1, None, 2), axis=1, name='normalized_counts', full=True)
    assert matrix.shape[1] == counts.size
    tmp = matrix.dot(counts.normalized_counts.ravel())
    matrix = counts._transform(slice(1, None, 2), axis=1, name='normalized_counts')
    tmp2 = np.moveaxis(np.tensordot(matrix, counts.normalized_counts, axes=(1, 1)), 0, 1).ravel()
    assert np.allclose(tmp, tmp2)
    counts2 = counts.select(s=slice(0, None, 2))
    assert counts2.shape[0] == counts.shape[0] // 2
    assert np.allclose(np.mean(counts2.normalized_counts), 2 * np.mean(counts.normalized_counts))
    counts3 = counts2.select(mu=slice(0, None, 2))
    assert counts3.shape[1] == counts.shape[1] // 2
    assert np.allclose(np.mean(counts3.normalized_counts), 2 * np.mean(counts2.normalized_counts))


def test_types(show=False):

    test_dir = Path('_tests')

    def get_spectrum(seed=42):
        ells = [0, 2, 4]
        rng = np.random.RandomState(seed=seed)
        spectrum = []
        for ell in ells:
            k_edges = np.linspace(0., 0.2, 41)
            k_edges = np.column_stack([k_edges[:-1], k_edges[1:]])
            k = np.mean(k_edges, axis=-1)
            spectrum.append(Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=rng.uniform(size=k.size)))
        return Mesh2SpectrumPoles(spectrum, ells=ells)

    def get_bispectrum(seed=42, basis='sugiyama'):
        ells = [0, 2]
        rng = np.random.RandomState(seed=seed)

        assert basis in ['sugiyama', 'sugiyama-diagonal', 'scoccimarro', 'scoccimarro-equilateral']
        if 'scoccimarro' in basis: ndim = 3
        else: ndim = 2

        spectrum = []
        for ell in ells:
            uedges = np.linspace(0., 0.2, 41)
            uedges = [np.column_stack([uedges[:-1], uedges[1:]])] * ndim
            k = [np.mean(uedge, axis=-1) for uedge in uedges]
            nmodes1d = [np.ones(uedge.shape[0], dtype='i') for uedge in uedges]

            def _product(array):
                if not isinstance(array, (tuple, list)):
                    array = [array] * ndim
                if 'diagonal' in basis or 'equilateral' in basis:
                    grid = [np.array(array[0])] * ndim
                else:
                    grid = np.meshgrid(*array, sparse=False, indexing='ij')
                return np.column_stack([tmp.ravel() for tmp in grid])

            def get_order_mask(edges):
                xmid = _product([np.mean(edge, axis=-1) for edge in edges])
                mask = True
                for i in range(xmid.shape[1] - 1): mask &= xmid[:, i] <= xmid[:, i + 1]  # select k1 <= k2 <= k3...
                return mask

            mask = get_order_mask(uedges)
            # of shape (nbins, ndim, 2)
            k_edges = np.concatenate([_product([edge[..., 0] for edge in uedges])[..., None], _product([edge[..., 1] for edge in uedges])[..., None]], axis=-1)[mask]
            k = _product(k)[mask]
            nmodes = np.prod(_product(nmodes1d)[mask], axis=-1)
            k = np.mean(k_edges, axis=-1)
            spectrum.append(Mesh3SpectrumPole(k=k, k_edges=k_edges, num_raw=rng.uniform(size=k.shape[0])))
        return Mesh3SpectrumPoles(spectrum, ells=ells)

    def get_count(mode='smu', seed=42):
        rng = np.random.RandomState(seed=seed)
        if mode == 'smu':
            coords = ['s', 'mu']
            edges = [np.linspace(0., 200., 201), np.linspace(-1., 1., 101)]
        if mode == 'rppi':
            coords = ['rp', 'pi']
            edges = [np.linspace(0., 200., 51), np.linspace(-20., 20., 101)]

        edges = [np.column_stack([edge[:-1], edge[1:]]) for edge in edges]
        coords_values = [np.mean(edge, axis=-1) for edge in edges]

        counts = 1. + rng.uniform(size=tuple(v.size for v in coords_values))
        return Count2(counts=counts, norm=np.ones_like(counts), **{coord: value for coord, value in zip(coords, coords_values)},
                      **{f'{coord}_edges': value for coord, value in zip(coords, edges)}, coords=coords, attrs=dict(los='x'))

    def get_correlation(mode='smu', seed=42):
        counts = {label: get_count(mode=mode, seed=seed + i) for i, label in enumerate(['DD', 'DR', 'RD', 'RR'])}
        return Count2Correlation(**counts)

    def get_correlation_jackknife(mode='smu', seed=42):
        def get_count_jk(seed=42):
            realizations = list(range(24))
            ii_counts = {ireal: get_count(mode=mode, seed=seed + ireal) for ireal in realizations}
            ij_counts = {ireal: get_count(mode=mode, seed=seed + ireal + 1) for ireal in realizations}
            ji_counts = {ireal: get_count(mode=mode, seed=seed + ireal + 2) for ireal in realizations}
            return Count2Jackknife(ii_counts, ij_counts, ji_counts)

        counts = {label: get_count_jk(seed=seed + i) for i, label in enumerate(['DD', 'DR', 'RD', 'RR'])}
        return Count2JackknifeCorrelation(**counts)

    spectrum = get_spectrum()
    spectrum.plot(show=show)
    spectrum2 = spectrum.select(k=slice(0, None, 2))

    spectrum = types.sum([get_spectrum(seed=seed) for seed in range(2)])
    assert np.allclose(spectrum.get(0).norm, 2)
    spectrum = types.mean([get_spectrum(seed=seed) for seed in range(2)])
    spectrum2 = types.join([get_spectrum().get(ells=[0, 2]), get_spectrum().get(ells=[4])])
    assert spectrum2.labels(return_type='flatten') == [{'ells': 0}, {'ells': 2}, {'ells': 4}]

    fn = test_dir / 'spectrum.txt'
    spectrum.write(fn)
    spectrum2 = read(fn)
    assert spectrum2 == spectrum

    all_spectrum, all_labels = [], []
    z = [0.2, 0.4, 0.6]
    for iz, zz in enumerate(z):
        spectrum = get_spectrum()
        spectrum.attrs['zeff'] = zz
        all_spectrum.append(spectrum)
        all_labels.append(f'z{iz:d}')
    all_spectrum = ObservableTree(all_spectrum, z=all_labels)
    all_spectrum.write(fn)
    all_spectrum = read(fn)
    all_spectrum.get('z0').plot(show=show)

    for basis in ['sugiyama', 'sugiyama-diagonal', 'scoccimarro', 'scoccimarro-equilateral']:
        spectrum = get_bispectrum(basis=basis)
        spectrum.plot(show=show)
        spectrum2 = spectrum.select(k=slice(0, None, 2))
        spectrum2 = spectrum.select(k=(0., 0.15))
        spectrum2 = spectrum.select(k=[(0., 0.1), (0., 0.15)])
        fn = test_dir / 'spectrum.h5'
        spectrum.write(fn)
        spectrum2 = read(fn)
        assert spectrum2 == spectrum

    correlation = get_correlation()
    correlation2 = correlation.select(s=slice(0, None, 2))
    correlation3 = correlation2.at(s=(20., 100.)).select(s=slice(0, None, 2))
    #print(correlation3.edges('s'))
    assert correlation2.shape[0] < correlation.shape[0]
    assert correlation3.shape[0] < correlation2.shape[0]
    value, window = correlation3.project(ells=[0, 2, 4], kw_window=dict(RR=correlation.get('RR')))
    value.plot(show=show)
    window.plot(show=show)

    value = correlation.project(wedges=[(-1., -2. / 3.), (1. / 2., 2. / 3.)])
    value.plot(show=show)
    assert value.get((-1., -2. / 3.)) == value.get('w1')
    assert value.get((1. / 2., 2. / 3.)) == value.get('w2')

    correlation = get_correlation_jackknife(mode='smu')
    value, covariance, window = correlation.project(ells=[0, 2, 4], kw_covariance=dict(), kw_window=dict())
    value.plot(show=show)
    covariance.plot(show=show)
    window.plot(show=show)

    value, covariance = correlation.project(wedges=[(-1., -2. / 3.), (1. / 2., 2. / 3.)], kw_covariance=dict())
    value.plot(show=show)
    covariance.plot(show=show)

    correlation = get_correlation_jackknife(mode='rppi')
    value, covariance = correlation.project(kw_covariance=dict())
    value.plot(show=show)
    covariance.plot(show=show)


def test_sparse():

    from scipy.sparse import bsr_array
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
    array = bsr_array((data, indices, indptr), shape=(6, 6))
    vec = np.zeros(6)
    array.dot(vec)

    array = np.arange(12).reshape(3, 4)
    vec = np.arange(8).reshape(2, 4)
    print(np.tensordot(array, vec, axes=([1], [1])).shape)


def test_external():

    from lsstypes.external import from_pypower, from_pycorr, from_triumvirate

    test_dir = Path('_tests')

    def generate_catalogs(size=100000, boxsize=(500,) * 3, offset=(1000., 0., 0), seed=42):
        rng = np.random.RandomState(seed=seed)
        positions = np.column_stack([o + rng.uniform(0., 1., size) * b for o, b in zip(offset, boxsize)])
        weights = rng.uniform(0.5, 1., size)
        return positions, weights

    def generate_pypower():
        from pypower import CatalogFFTPower
        kedges = np.linspace(0., 0.2, 11)
        data_positions1, data_weights1 = generate_catalogs(seed=42)
        randoms_positions1, randoms_weights1 = generate_catalogs(seed=43)
        poles = CatalogFFTPower(data_positions1=data_positions1, data_weights1=data_weights1, randoms_positions1=randoms_positions1, randoms_weights1=randoms_weights1,
                                edges=kedges, ells=(0, 2, 4), nmesh=64, resampler='tsc', interlacing=2, los=None, position_type='pos', dtype='f8').poles
        return poles

    def generate_pycorr():
        from pycorr import TwoPointCorrelationFunction
        data_positions1, data_weights1 = generate_catalogs(seed=42)
        randoms_positions1, randoms_weights1 = generate_catalogs(seed=43)
        edges = (np.linspace(0., 101, 51), np.linspace(-1., 1., 101))
        return TwoPointCorrelationFunction('smu', edges, data_positions1=data_positions1, data_weights1=data_weights1,
                                            randoms_positions1=randoms_positions1, randoms_weights1=randoms_weights1,
                                            engine='corrfunc', position_type='pos', nthreads=4)

    def generate_pycorr_jackknife():
        from pycorr import TwoPointCorrelationFunction
        data_positions1, data_weights1 = generate_catalogs(seed=42)
        randoms_positions1, randoms_weights1 = generate_catalogs(seed=43)
        data_samples1 = np.rint(data_weights1 * 7).astype(int)
        randoms_samples1 = np.rint(randoms_weights1 * 7).astype(int)
        edges = (np.linspace(0., 101, 51), np.linspace(-1., 1., 101))
        #edges = (np.linspace(0., 101, 21), np.linspace(-1., 1., 11))
        return TwoPointCorrelationFunction('smu', edges, data_positions1=data_positions1, data_weights1=data_weights1, data_samples1=data_samples1,
                                            randoms_positions1=randoms_positions1, randoms_weights1=randoms_weights1, randoms_samples1=randoms_samples1,
                                            engine='corrfunc', position_type='pos', nthreads=4)

    def generate_triumvirate(return_type='bispctrum'):
        from triumvirate.catalogue import ParticleCatalogue
        from triumvirate.twopt import compute_powspec
        from triumvirate.threept import compute_bispec
        from triumvirate.parameters import ParameterSet
        from triumvirate.logger import setup_logger

        logger = setup_logger(20)
        boxsize = np.array([500.] * 3)
        meshsize = np.array([100] * 3)
        data_positions1, data_weights1 = generate_catalogs(seed=42, boxsize=boxsize)
        randoms_positions1, randoms_weights1 = generate_catalogs(seed=43, boxsize=boxsize)

        data = ParticleCatalogue(*data_positions1.T, ws=data_weights1, nz=data_positions1.shape[0] / boxsize.prod())
        randoms = ParticleCatalogue(*randoms_positions1.T, ws=randoms_weights1, nz=randoms_positions1.shape[0] / boxsize.prod())

        edges = np.arange(0., 0.1, 0.02)
        ell = (0, 0, 0)
        paramset = dict(norm_convention='particle', form='full', degrees=dict(zip(['ell1', 'ell2', 'ELL'], ell)), wa_orders=dict(i=None, j=None),
                        range=[edges[0], edges[-1]], num_bins=len(edges) - 1, binning='lin', assignment='cic', interlace=True, alignment='centre', padfactor=0.,
                        boxsize=dict(zip('xyz', boxsize)), ngrid=dict(zip('xyz', meshsize)), verbose=20)
        paramset = ParameterSet(param_dict=paramset)

        if return_type == 'spectrum':
            results = compute_powspec(data, randoms, paramset=paramset, logger=logger)
        elif return_type == 'bispectrum':
            results = compute_bispec(data, randoms, paramset=paramset, logger=logger)
        else:
            raise NotImplementedError
        return results

    pypoles = generate_pypower()
    poles = from_pypower(pypoles)
    assert np.allclose(poles.value(), pypoles.power.ravel())
    fn = test_dir / 'poles.h5'
    poles.write(fn)
    poles = read(fn)
    poles = poles.select(k=slice(0, None, 2))
    pypoles = pypoles[:(pypoles.shape[0] // 2) * 2:2]
    assert np.allclose(poles.get(0).coords('k'), pypoles.k, equal_nan=True)
    assert np.allclose(poles.value(), pypoles.power.ravel())

    pycorr = generate_pycorr()
    corr = from_pycorr(pycorr)
    fn = test_dir / 'corr.h5'
    corr.write(fn)
    corr = read(fn)
    assert np.allclose(corr.coords(axis='s'), pycorr.sepavg(axis=0), equal_nan=True)
    assert np.allclose(corr.value(), pycorr.corr, equal_nan=True)
    corr = corr.select(s=slice(0, None, 2))
    pycorr = pycorr[:(pycorr.shape[0] // 2) * 2:2]
    assert np.allclose(corr.coords(axis='s'), pycorr.sepavg(axis=0), equal_nan=True)
    assert np.allclose(corr.value(), pycorr.corr, equal_nan=True)
    xi = corr.project(ells=[0, 2, 4])
    assert np.allclose(xi.value(), np.ravel(pycorr(ells=[0, 2, 4])))

    pycorr = generate_pycorr_jackknife()
    corr = from_pycorr(pycorr)
    fn = test_dir / 'corr.h5'
    corr.write(fn)
    corr = read(fn)
    assert np.allclose(corr.coords(axis='s'), pycorr.sepavg(axis=0), equal_nan=True)
    assert np.allclose(corr.value(), pycorr.corr, equal_nan=True)
    corr = corr.select(s=slice(0, None, 2))
    pycorr = pycorr[:(pycorr.shape[0] // 2) * 2:2]
    assert np.allclose(corr.coords(axis='s'), pycorr.sepavg(axis=0), equal_nan=True)
    assert np.allclose(corr.value(), pycorr.corr, equal_nan=True)
    xi = corr.project(ells=[0, 2, 4])
    assert np.allclose(xi.value(), np.ravel(pycorr(ells=[0, 2, 4], return_std=False)))

    spec = generate_triumvirate(return_type='spectrum')
    pole = from_triumvirate(spec)
    fn = test_dir / 'pole.h5'
    pole.write(fn)
    read(fn)

    spec = generate_triumvirate(return_type='bispectrum')
    pole = from_triumvirate(spec)
    fn = test_dir / 'pole.h5'
    pole.write(fn)
    read(fn)


@contextmanager
def chdir(path):
    """Temporarily change working directory inside a context."""
    prev_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev_dir)


def test_readme():

    test_dir = Path('_tests')
    test_dir.mkdir(exist_ok=True)

    with chdir(test_dir):

        from lsstypes import ObservableLeaf, ObservableTree

        s = np.linspace(0., 200., 51)
        mu = np.linspace(-1., 1., 101)
        rng = np.random.RandomState(seed=42)
        xi = 1. + rng.uniform(size=(s.size, mu.size))
        # Specify all data entries, and the name of the coordinate axes
        # Optionally, some extra attributes
        correlation = ObservableLeaf(xi=xi, s=s, mu=mu, coords=['s', 'mu'], attrs=dict(los='x'))

        # Some predefined data types
        from lsstypes import Mesh2SpectrumPole, Mesh2SpectrumPoles

        def get_spectrum(seed=None):
            ells = [0, 2, 4]
            rng = np.random.RandomState(seed=seed)
            poles = []
            for ell in ells:
                k_edges = np.linspace(0., 0.2, 41)
                k_edges = np.column_stack([k_edges[:-1], k_edges[1:]])
                k = np.mean(k_edges, axis=-1)
                poles.append(Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=rng.uniform(size=k.size)))
            return Mesh2SpectrumPoles(poles, ells=ells)

        spectrum = get_spectrum()

        # Create a new data structure and load data
        tree = ObservableTree([correlation, spectrum], observables=['correlation', 'spectrum'])
        print(tree.get(observables='spectrum', ells=2))
        tree.write('data.hdf5')

        tree = read('data.hdf5')

        # Apply coordinate selection
        subset = tree.select(s=(0, 100))

        # Rebin data, optionally using sparse matrices
        rebinned = subset.select(k=slice(0, None, 2))

        # Save your processed data
        rebinned.write('rebinned.hdf5')


def test_savetxt():
    test_dir = Path('_tests')
    test_dir.mkdir(exist_ok=True)

    a = np.linspace(0., 1., 10)
    a = a - 1j * a
    fn = test_dir / 'test.txt'
    np.savetxt(fn, a, fmt='%.4f%+.4fj')
    a = np.loadtxt(fn, dtype=np.complex128)

    def get_spectrum(seed=None):
        ells = [0, 2, 4]
        rng = np.random.RandomState(seed=seed)
        poles = []
        for ell in ells:
            k_edges = np.linspace(0., 0.2, 41)
            k_edges = np.column_stack([k_edges[:-1], k_edges[1:]])
            k = np.mean(k_edges, axis=-1)
            poles.append(Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=rng.uniform(size=k.size) + 1j * rng.uniform(size=k.size)))
        return Mesh2SpectrumPoles(poles, ells=ells, attrs={'zeff': np.float64(0.8)})

    spectrum = get_spectrum()
    spectrum.write(fn)
    assert read(fn) == spectrum


if __name__ == '__main__':

    test_tree()
    test_types()
    test_sparse()
    test_rebin()
    test_at()
    test_matrix()
    test_external()
    test_readme()
    test_savetxt()