import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import lsstypes as types
from lsstypes import ObservableLeaf, ObservableTree, read, write
from lsstypes import Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh3SpectrumPole, Mesh3SpectrumPoles, Mesh2CorrelationPole, Mesh2CorrelationPoles, Mesh3CorrelationPole, Mesh3CorrelationPoles, Count2, Count2Jackknife, Count2Correlation, Count2JackknifeCorrelation
from lsstypes import WindowMatrix, CovarianceMatrix, GaussianLikelihood, ObservableLike


def _make_mesh3_spectrum_poles(seed=42):
    rng = np.random.RandomState(seed=seed)
    base_edges = np.linspace(0., 0.2, 51)
    base_edges = np.column_stack([base_edges[:-1], base_edges[1:]])
    grid = np.meshgrid(*([np.arange(base_edges.shape[0])] * 3), sparse=False, indexing='ij')
    index = np.column_stack([tmp.ravel() for tmp in grid])
    index = index[(index[:, 0] <= index[:, 1]) & (index[:, 1] <= index[:, 2])]
    k_edges = np.stack([base_edges[index[:, iaxis]] for iaxis in range(3)], axis=1)
    k = np.mean(k_edges, axis=-1)
    nmodes = np.arange(1, index.shape[0] + 1, dtype='f8')

    poles = []
    for ell in [0, 2]:
        poles.append(
            Mesh3SpectrumPole(
                k=k,
                k_edges=k_edges,
                nmodes=nmodes * (ell + 1),
                num_raw=rng.uniform(size=k.shape[0]),
                ell=ell,
                basis='scoccimarro',
            )
        )
    return Mesh3SpectrumPoles(poles, ells=[0, 2])


def test_tree():

    test_dir = Path('_tests')

    leaf = ObservableLeaf(value=np.ones(3))
    assert isinstance(leaf, ObservableLike)

    s_edges = np.linspace(0., 100., 51)
    mu_edges = np.linspace(-1., 1., 101)
    s_edges = np.column_stack([s_edges[:-1], s_edges[1:]])
    mu_edges = np.column_stack([mu_edges[:-1], mu_edges[1:]])
    rng = np.random.RandomState(seed=42)
    labels = ['DD', 'DR', 'RR']
    leaves = []
    for label in labels:
        s, mu = np.mean(s_edges, axis=-1), np.mean(mu_edges, axis=-1)
        counts = 1. + rng.uniform(size=(s.size, mu.size))
        leaves.append(ObservableLeaf(counts=counts, s=s, mu=mu, s_edges=s_edges, mu_edges=mu_edges, coords=['s', 'mu'], attrs=dict(los='x')))

    leaf = leaves[0]
    assert np.allclose(leaf.value_as_leaf().value(), leaf.values('counts'))
    fn = test_dir / 'leaf.h5'
    write(fn, leaf, locking=False)
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
    assert len(leaf4.coords('s')) == 40

    tree = ObservableTree(leaves, keys=labels)
    assert isinstance(tree, ObservableLike)

    assert tree.labels(return_type='keys') == ['keys']
    assert tree.labels(return_type='unflatten') == {'keys': ['DD', 'DR', 'RR']}
    assert tree.labels(return_type='flatten') == [{'keys': 'DD'}, {'keys': 'DR'}, {'keys': 'RR'}]
    assert tree.labels(return_type='flatten_values') == [('DD',), ('DR',), ('RR',)]
    assert len(tree.value()) == tree.size
    tree2 = tree.at(keys='DD').select(s=(10., 80.))
    assert tree2.get(keys='DD').shape != tree2.get(keys='DR').shape
    tree2 = tree.select(s=(10., 80.))
    assert tree2.get(keys='DD').shape == tree2.get(keys='DR').shape
    assert tree.get(keys=['DD', 'RR']).size == tree.size * 2 // 3
    assert tree.get(keys=['DD', 'RR']).get('DD') == tree.get('DD')
    assert isinstance(tree.get('DD'), ObservableLeaf)
    assert isinstance(tree.get(keys=['DD']), ObservableTree)
    tree2 = tree.clear()
    assert tree2.labels(return_type='keys') == []
    assert tree2.size == 0
    for label, branch in zip(tree.labels(level=1), tree.flatten(level=1)):
        tree2 = tree2.insert(branch, **label)
    assert tree2 == tree

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
    tree3 = tree2.get([{'observable': 'correlation', 'keys': 'DD'}, {'observable': 'spectrum'}])
    assert tree3.labels(level=None, return_type='flatten') == [{'observable': 'correlation', 'keys': 'DD'}, {'observable': 'spectrum'}]
    fn = test_dir / 'tree.h5'
    write(fn, tree2)
    #tree3 = read(fn)
    #assert tree3 == tree2

    fn = test_dir / 'tree.txt'
    write(fn, tree2)
    tree3 = read(fn)
    assert tree3 == tree2

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
    poles2 = poles.get(ells=[2])
    poles3 = poles.match(poles2)
    assert poles3.labels() == poles2.labels()
    assert np.all(poles3.value() == poles2.value())



def test_io():
    test_dir = Path('_tests')
    def test_write_read(tree):
        fn = test_dir / 'tree.h5'
        write(fn, tree)
        tree2 = read(fn)
        assert tree2 == tree
        fn = test_dir / 'tree.txt'
        write(fn, tree)
        tree2 = read(fn)
        assert tree2 == tree

    rng = np.random.RandomState(seed=42)
    k1 = np.linspace(0., 0.2, 21)
    k2 = np.linspace(0., 0.2, 21)
    spectrum = rng.uniform(size=(k1.size, k2.size))
    leaf = ObservableLeaf(spectrum=spectrum, k1=k1, k2=k2, coords=['k1', 'k2'], attrs=dict(los='x'))
    tree = ObservableTree([leaf, leaf], observable=['spectrum', 'spectrum_recon'])
    test_write_read(tree)
    tree = ObservableTree([leaf, leaf], observable=['spectrum', (3, '_recon')], wa_orders=[0, 2])
    test_write_read(tree)

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
    spectrum.write(fn, locking=False)
    assert read(fn) == spectrum


def test_at():

    def get_poles(seed=42):
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
    pole = poles.at().get(ells=[2])
    assert pole.ells == [2]
    assert np.allclose(pole.value(), poles.get(ells=2).value())
    poles.get()
    pole2 = poles.get(ells=2)
    pole2 = pole2.clone(value=pole2.value() + 100.)
    poles2 = poles.at(ells=2).map(lambda pole: pole2)
    assert np.allclose(poles2.get(ells=2).value(), pole2.value())
    poles2 = poles.at(ells=2).replace(pole2)
    assert np.allclose(poles2.get(ells=2).value(), pole2.value())

    value = poles.get(ells=[2, 4]).value()
    assert value.size == poles.get(2).size + poles.get(4).size
    poles3 = poles.at([{'ells': 2}, {'ells': 4}]).clone(value=value + 10.)
    assert np.allclose(poles3.get(ells=[2, 4]).value(), value + 10.)

    pole = poles.get(ells=0)
    pnew, transform = pole.at.hook(lambda new, transform: (new, transform))().match(pole)
    assert transform.ndim == 1
    pnew, transform = poles.at.hook(lambda new, transform: (new, transform))().match(poles)
    assert transform.ndim == 1
    spoles = poles.select(k=(0., 0.1))
    pnew, transform = poles.at.hook(lambda new, transform: (new, transform))().match(spoles)
    assert transform.ndim == 1
    assert transform.size == spoles.size
    assert all(p.coords('k').max() < 0.1 for p in pnew)

    bao = ObservableTree([ObservableLeaf(value=np.ones(1)) for i in [0, 1]], parameters=['qiso', 'qap'])
    tree = ObservableTree([poles, bao], observables=['spectrum', 'bao'])
    cov = CovarianceMatrix(value=np.eye(tree.size), observable=tree)
    observable = tree.select(k=slice(0, None, 5))
    cov2 = cov.at.observable.match(observable)
    observable = tree.at(observables='spectrum', ells=[0]).select(k=slice(0, None, 5))
    observable = tree.at(observables='spectrum').get(ells=[0])
    cov2 = cov.at.observable.match(observable)
    assert np.allclose(cov2.observable.value(), observable.value())

    at = poles.at.hook(lambda new, transform: new)(2)
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
    at = tree.at.hook(lambda new, transform: new)('DD')
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
        matrix.write(fn, locking=False)
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

    assert np.allclose(covmat.inv(level=1), covmat.inv(level=0))
    assert covmat.std().size == covmat.shape[0]
    assert covmat.corrcoef().shape == covmat.shape
    covmat.plot(show=show)
    covmat.plot_diag(show=show)
    covmat.plot_slice(indices=2, show=show)
    test(covmat)

    covmat = types.cov([get_spectrum(size=40, seed=seed) for seed in range(100)])
    covmat.plot(show=show)


def test_likelihood():

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

    def get_bao(seed=None):
        rng = np.random.RandomState(seed=seed)
        alpha = rng.normal(loc=1., scale=0.01, size=2)
        return ObservableLeaf(value=alpha)

    def get_observable(seed=None):
        return ObservableTree([get_spectrum(seed=seed), get_bao(seed=seed)], observables=['spectrum', 'bao'])

    observable = get_observable(seed=42)
    window = WindowMatrix(observable=observable, theory=observable.copy(), value=np.eye(observable.size))
    covariance = types.cov([get_observable(seed=seed) for seed in range(500)])

    likelihood = GaussianLikelihood(observable=observable, window=window, covariance=covariance)

    fn = test_dir / 'likelihood.h5'
    likelihood.write(fn, locking=False)
    likelihood = read(fn)

    likelihood = likelihood.at.observable.get('spectrum')

    likelihood2 = likelihood.at.observable.select(k=(0.05, 0.15))
    assert likelihood2.window.shape[0] < likelihood.window.shape[0]

    likelihood2 = likelihood.at.observable.get([0])
    assert likelihood2.window.shape[0] < likelihood.window.shape[0]

    likelihood2 = likelihood.at.observable.at(2).at[...].select(k=slice(0, None, 2))
    assert likelihood2.observable.get(2).size == likelihood.observable.get(2).size // 2

    chi2 = likelihood2.chi2(window.theory)


def test_dict():

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

    def get_bao(seed=None):
        rng = np.random.RandomState(seed=seed)
        alpha = rng.normal(loc=1., scale=0.01, size=2)
        return ObservableLeaf(value=alpha)

    def get_observable(seed=None):
        return ObservableTree([get_spectrum(seed=seed), get_bao(seed=seed)], observables=['spectrum', 'bao'])

    observable = get_observable(seed=42)
    window = WindowMatrix(observable=observable, theory=observable.copy(), value=np.eye(observable.size))
    covariance = types.cov([get_observable(seed=seed) for seed in range(100)])

    likelihood = dict(observable=observable, window=window, covariance=covariance)
    fn = test_dir / 'dict.h5'
    write(fn, likelihood)
    likelihood2 = read(fn)
    assert isinstance(likelihood2, dict)
    assert likelihood2 == likelihood


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

    def toarray(transform):
        if hasattr(transform, 'toarray'):
            return transform.toarray()
        return transform

    counts = get_counts()
    matrix = toarray(counts._transform(slice(1, None, 2), axis=1, name='normalized_counts', full=True))
    assert matrix.shape[1] == counts.size
    tmp = matrix.dot(counts.normalized_counts.ravel())
    matrix = toarray(counts._transform(slice(1, None, 2), axis=1, name='normalized_counts'))
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

    def get_mesh2_spectrum(seed=42):
        ells = [0, 2, 4]
        rng = np.random.RandomState(seed=seed)
        spectrum = []
        for ell in ells:
            k_edges = np.linspace(0., 0.2, 41)
            k_edges = np.column_stack([k_edges[:-1], k_edges[1:]])
            k = np.mean(k_edges, axis=-1)
            spectrum.append(Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=rng.uniform(size=k.size)))
        return Mesh2SpectrumPoles(spectrum, ells=ells)

    def get_mesh_spectrum3(seed=42, basis='sugiyama', full=False):
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
            if full: mask = Ellipsis
            # of shape (nbins, ndim, 2)
            k_edges = np.concatenate([_product([edge[..., 0] for edge in uedges])[..., None], _product([edge[..., 1] for edge in uedges])[..., None]], axis=-1)[mask]
            k = _product(k)[mask]
            nmodes = np.prod(_product(nmodes1d)[mask], axis=-1)
            k = np.mean(k_edges, axis=-1)
            spectrum.append(Mesh3SpectrumPole(k=k, k_edges=k_edges, nmodes=nmodes, num_raw=rng.uniform(size=k.shape[0])))
        return Mesh3SpectrumPoles(spectrum, ells=ells)

    def get_mesh2_correlation(seed=42):
        ells = [0, 2, 4]
        rng = np.random.RandomState(seed=seed)
        correlation = []
        for ell in ells:
            s_edges = np.linspace(0., 200, 41)
            s_edges = np.column_stack([s_edges[:-1], s_edges[1:]])
            s = np.mean(s_edges, axis=-1)
            correlation.append(Mesh2CorrelationPole(s=s, s_edges=s_edges, num_raw=rng.uniform(size=s.size)))
        return Mesh2CorrelationPoles(correlation, ells=ells)

    def get_mesh3_correlation(seed=42, basis='sugiyama', full=False):
        ells = [0, 2]
        rng = np.random.RandomState(seed=seed)

        assert basis in ['sugiyama', 'sugiyama-diagonal']
        if 'scoccimarro' in basis:
            ndim = 3
            ells = [0, 2]
        else:
            ndim = 2
            ells = [(0, 0, 0), (2, 0, 2)]

        correlation = []
        for ell in ells:
            uedges = np.linspace(0., 100, 41)
            uedges = [np.column_stack([uedges[:-1], uedges[1:]])] * ndim
            s = [np.mean(uedge, axis=-1) for uedge in uedges]
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
            if full: mask = Ellipsis
            # of shape (nbins, ndim, 2)
            s_edges = np.concatenate([_product([edge[..., 0] for edge in uedges])[..., None], _product([edge[..., 1] for edge in uedges])[..., None]], axis=-1)[mask]
            s = _product(s)[mask]
            nmodes = np.prod(_product(nmodes1d)[mask], axis=-1)
            k = np.mean(s_edges, axis=-1)
            correlation.append(Mesh3CorrelationPole(s=s, s_edges=s_edges, nmodes=nmodes, num_raw=rng.uniform(size=k.shape[0])))
        return Mesh3CorrelationPoles(correlation, ells=ells)

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

    spectrum = get_mesh2_spectrum()
    spectrum.plot(show=show)
    spectrum2 = spectrum.select(k=slice(0, None, 2))

    spectrum = types.sum([get_mesh2_spectrum(seed=seed) for seed in range(2)])
    assert np.allclose(spectrum.get(0).norm, 2)
    spectrum = types.mean([get_mesh2_spectrum(seed=seed) for seed in range(2)])
    spectrum2 = types.join([get_mesh2_spectrum().get(ells=[0, 2]), get_mesh2_spectrum().get(ells=[4])])
    assert spectrum2.labels(return_type='flatten') == [{'ells': 0}, {'ells': 2}, {'ells': 4}]

    fn = test_dir / 'spectrum.txt'
    spectrum.write(fn)
    spectrum2 = read(fn)
    assert spectrum2 == spectrum

    all_spectrum, all_labels = [], []
    z = [0.2, 0.4, 0.6]
    for iz, zz in enumerate(z):
        spectrum = get_mesh2_spectrum()
        spectrum.attrs['zeff'] = zz
        all_spectrum.append(spectrum)
        all_labels.append(f'z{iz:d}')
    all_spectrum = ObservableTree(all_spectrum, z=all_labels)
    all_spectrum.write(fn)
    all_spectrum = read(fn)
    all_spectrum.get('z0').plot(show=show)

    for basis in ['sugiyama', 'sugiyama-diagonal', 'scoccimarro', 'scoccimarro-equilateral']:
        if basis in ['sugiyama', 'scoccimarro']:
            spectrum = get_mesh_spectrum3(basis=basis, full=True)
            spectrum2 = spectrum.unravel()
            for pole in spectrum2:
                assert len(pole.shape) > 1
            if basis != 'scoccimarro':
                spectrum2.plot(show=show)
            spectrum2 = spectrum2.ravel()
            for pole in spectrum2:
                assert len(pole.shape) == 1
            assert spectrum2 == spectrum

        spectrum = get_mesh_spectrum3(basis=basis)
        spectrum.plot(show=show)
        spectrum2.plot(show=show)
        spectrum2 = spectrum.select(k=slice(0, None, 2))
        spectrum2 = spectrum.select(k=(0., 0.15))
        spectrum2 = spectrum.select(k=[(0., 0.1), (0., 0.15)])
        fn = test_dir / 'spectrum.h5'
        spectrum.write(fn)
        spectrum2 = read(fn)
        assert spectrum2 == spectrum

    correlation = get_mesh2_correlation()
    correlation.plot(show=show)
    correlation2 = correlation.select(s=slice(0, None, 2))
    correlation = types.sum([get_mesh2_correlation(seed=seed) for seed in range(2)])
    assert np.allclose(correlation.get(0).norm, 2)

    correlation = types.mean([get_mesh2_correlation(seed=seed) for seed in range(2)])
    correlation2 = types.join([get_mesh2_correlation().get(ells=[0, 2]), get_mesh2_correlation().get(ells=[4])])
    assert correlation2.labels(return_type='flatten') == [{'ells': 0}, {'ells': 2}, {'ells': 4}]

    fn = test_dir / 'correlation.txt'
    correlation.write(fn)
    correlation2 = read(fn)
    assert correlation2 == correlation

    for basis in ['sugiyama', 'sugiyama-diagonal']:
        if basis in ['sugiyama', 'scoccimarro']:
            correlation = get_mesh3_correlation(basis=basis, full=True)
            correlation2 = correlation.unravel()
            for pole in correlation2:
                assert len(pole.shape) > 1
            if basis != 'scoccimarro':
                correlation2.plot(show=show)
            correlation2 = correlation2.ravel()
            for pole in correlation2:
                assert len(pole.shape) == 1
            assert correlation2 == correlation

        correlation = get_mesh3_correlation(basis=basis)
        correlation.plot(show=show)
        correlation2.plot(show=show)
        correlation2 = correlation.select(s=slice(0, None, 2))
        correlation2 = correlation.select(s=(0., 0.15))
        correlation2 = correlation.select(s=[(0., 0.1), (0., 0.15)])
        fn = test_dir / 'correlation.h5'
        correlation.write(fn)
        correlation2 = read(fn)
        assert correlation2 == correlation

    correlation = get_correlation(mode='smu', seed=42)
    RR = correlation.get('RR')
    RR4 = RR.sum([RR] * 4)
    assert np.allclose(RR4.value(), RR.value())
    assert np.allclose(RR4.values('norm'), 4. * RR.values('norm'))
    correlations = [correlation, get_correlation(mode='smu', seed=84)]
    correlation2 = types.sum(correlations)
    assert np.allclose(correlation2.get('DD').values('norm'), sum(correlation.get('DD').values('norm') for correlation in correlations))
    assert np.allclose(correlation2.get('DD').values('counts'), sum(correlation.get('DD').values('counts') for correlation in correlations))
    assert np.allclose(correlation2.get('RR').values('norm'), sum(correlation.get('DD').values('norm') for correlation in correlations))
    RR2 = correlation2.get('RR').value()
    RRav = 0.
    for correlation in correlations:
        RRav += correlation.get('RR').values('normalized_counts') * correlation.get('DD').values('norm')
    RRav /= sum(correlation.get('DD').values('norm') for correlation in correlations)
    assert np.allclose(RR2, RRav)
    correlation2 = Count2Correlation(estimator='(DD - DR - RD + RR) / RR', **{name: correlation.get(name) for name in ['DD', 'DR', 'RD', 'RR']})
    assert np.allclose(correlation2.value(), correlation.value())
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
    assert len(correlation.realizations) == 24
    assert callable(correlation.realization)
    value, covariance, window = correlation.project(ells=[0, 2, 4], kw_covariance=dict(), kw_window=dict(resolution=2))
    value.plot(show=show)
    covariance.plot(show=show)
    window.plot(show=show)
    assert window.shape[1] == 2 * window.shape[0]
    correlation_no_jackknife = correlation.value(return_type=None)
    assert type(correlation_no_jackknife.get('DD')) is Count2

    value, covariance = correlation.project(wedges=[(-1., -2. / 3.), (1. / 2., 2. / 3.)], kw_covariance=dict())
    value.plot(show=show)
    covariance.plot(show=show)

    correlation = get_correlation_jackknife(mode='rppi')
    value, covariance = correlation.project(kw_covariance=dict())
    value.plot(show=show)
    covariance.plot(show=show)

    from lsstypes import Count2Pole, Count2Poles, Count2CorrelationPoles, Count2PolesJackknife

    def get_count2poles(seed=42):
        ells = [0, 2, 4]

        def get_count2pole(ell):
            rng = np.random.RandomState(seed=seed + int(ell))
            coords = ['s']
            edges = [np.linspace(0., 200., 201)]
            edges = [np.column_stack([edge[:-1], edge[1:]]) for edge in edges]
            coords_values = [np.mean(edge, axis=-1) for edge in edges]
            counts = 1. + rng.uniform(size=tuple(v.size for v in coords_values))
            return Count2Pole(counts=counts, norm=np.ones_like(counts), **{coord: value for coord, value in zip(coords, coords_values)},
                        **{f'{coord}_edges': value for coord, value in zip(coords, edges)}, coords=coords, ell=ell, attrs=dict(los='x'))
        return Count2Poles([get_count2pole(ell) for ell in ells])

    def get_correlation2poles(seed=42):
        counts = {label: get_count2poles(seed=seed + i) for i, label in enumerate(['DD', 'DR', 'RD', 'RR'])}
        return Count2Correlation(**counts)

    def get_correlation2poles_jackknife(seed=42):
        def get_count_jk(seed=42):
            realizations = list(range(24))
            ii_counts = {ireal: get_count2poles(seed=seed + ireal) for ireal in realizations}
            ij_counts = {ireal: get_count2poles(seed=seed + ireal + 1) for ireal in realizations}
            ji_counts = {ireal: get_count2poles(seed=seed + ireal + 2) for ireal in realizations}
            return Count2PolesJackknife(ii_counts, ij_counts, ji_counts)

        counts = {label: get_count_jk(seed=seed + i) for i, label in enumerate(['DD', 'DR', 'RD', 'RR'])}
        return Count2JackknifeCorrelation(**counts)

    correlation = get_correlation2poles()
    assert isinstance(correlation.project(), Count2CorrelationPoles)
    correlation2 = correlation + correlation
    assert np.allclose(correlation2.get(count_names='DD', ells=2).values('counts'), 2 * correlation.get(count_names='DD', ells=2).values('counts'))

    correlation = get_correlation2poles_jackknife()
    assert len(correlation.realizations) == 24
    assert callable(correlation.realization)
    value, covariance = correlation.project(kw_covariance=dict())
    value.plot(show=show)
    covariance.plot(show=show)
    correlation_no_jackknife = correlation.value(return_type=None)
    assert type(correlation_no_jackknife.get('DD')) is Count2Poles

    from lsstypes import Count3Pole, Count3Poles, Count3Correlation, Count3CorrelationPoles

    def get_ells(ellmax=2):
        return [
            (ell, ellp, m)
            for ell in range(ellmax + 1)
            for ellp in range(ellmax + 1)
            for m in range(min(ell, ellp) + 1)
        ]

    def get_count3poles(seed=42, ellmax=2):
        ells = get_ells(ellmax)

        def get_count3pole(ell):
            rng = np.random.RandomState(seed=seed + 100 * ell[0] + 10 * ell[1] + ell[2])

            coords = ['s1', 's2']
            edges = [
                np.linspace(0., 200., 21),
                np.linspace(0., 200., 21),
            ]
            edges = [np.column_stack([edge[:-1], edge[1:]]) for edge in edges]
            coords_values = [np.mean(edge, axis=-1) for edge in edges]

            shape = tuple(v.size for v in coords_values)
            counts = 1. + rng.uniform(size=shape)
            norm = np.ones_like(counts)

            return Count3Pole(counts=counts, norm=norm, s1=coords_values[0], s2=coords_values[1],
                              s1_edges=edges[0], s2_edges=edges[1], coords=coords, ell=ell, attrs=dict(los='x'))

        return Count3Poles([get_count3pole(ell) for ell in ells])

    def get_correlation3poles(seed=42):
        labels = ['DDD', 'DDR', 'DRD', 'RDD', 'DRR', 'RDR', 'RRD', 'RRR']
        counts = {label: get_count3poles(seed=seed + i) for i, label in enumerate(labels)}
        return Count3Correlation(**counts)

    correlation = get_correlation3poles()

    # basic tree arithmetic / label propagation
    correlation2 = correlation + correlation
    assert np.allclose(
        correlation2.get(count_names='DDD', ells=(1, 1, 0)).values('counts'),
        2. * correlation.get(count_names='DDD', ells=(1, 1, 0)).values('counts'),
    )

    # value should be flattened over multipoles, then spatial bins
    value = correlation.value()
    assert value.shape[0] == len(correlation.get('RRR').ells)
    assert value.shape[1:] == correlation.get('RRR').get((0, 0, 0)).value().shape

    # project should return Count3CorrelationPoles
    poles = correlation.project()
    assert isinstance(poles, Count3CorrelationPoles)
    assert poles.ells == correlation.get('RRR').ells

    # check one projected pole has expected coords and shape
    pole = poles.get((1, 1, 0))
    assert np.allclose(pole.coords('s1'), correlation.get('RRR').get((1, 1, 0)).coords('s1'))
    assert np.allclose(pole.coords('s2'), correlation.get('RRR').get((1, 1, 0)).coords('s2'))
    assert pole.value().shape == correlation.get('RRR').get((1, 1, 0)).value().shape

    # ravel / unravel round-trip on Count3Poles
    raveled = correlation.get('DDD').ravel()
    assert all(leaf.is_raveled for leaf in raveled)
    unraveled = raveled.unravel()
    assert np.allclose(
        unraveled.get((1, 1, 0)).values('counts'),
        correlation.get('DDD').get((1, 1, 0)).values('counts'),
    )

    poles.plot(show=show)


def test_select_reuses_mesh3_transform_signature():
    poles = _make_mesh3_spectrum_poles()
    rebinned = poles.select(k=slice(0, None, 3))

    for ell in [0, 2]:
        pole = poles.get(ell)
        selected = rebinned.get(ell)

        def toarray(transform):
            if hasattr(transform, 'toarray'):
                return transform.toarray()
            return transform

        coord_transform = toarray(pole._transform(slice(0, None, 3), axis='k', name='k', full=False))
        value_transform = toarray(pole._transform(slice(0, None, 3), axis='k', name='value'))
        nmodes_transform = toarray(pole._transform(slice(0, None, 3), axis='k', name='nmodes'))

        expected_k = np.tensordot(coord_transform, pole.k, axes=([1], [0]))
        expected_value = value_transform.dot(pole.value())
        expected_nmodes = nmodes_transform.dot(pole.nmodes)
        wrong_nmodes = value_transform.dot(pole.nmodes)

        assert np.allclose(coord_transform, value_transform)
        assert not np.allclose(value_transform, nmodes_transform)
        assert np.allclose(selected.k, expected_k)
        assert np.allclose(selected.value(), expected_value)
        assert np.allclose(selected.nmodes, expected_nmodes)
        assert not np.allclose(selected.nmodes, wrong_nmodes)


def test_select_keeps_multidim_coordinate_and_value_transforms_separate():
    s_edges = np.linspace(0., 80., 9)
    s_edges = np.column_stack([s_edges[:-1], s_edges[1:]])
    mu_edges = np.linspace(-1., 1., 7)
    mu_edges = np.column_stack([mu_edges[:-1], mu_edges[1:]])
    s = np.mean(s_edges, axis=-1)
    mu = np.mean(mu_edges, axis=-1)

    counts = np.arange(1, s.size * mu.size + 1, dtype='f8').reshape(s.size, mu.size)
    norm = 1. + counts[::-1]
    observable = Count2(
        counts=counts,
        norm=norm,
        s=s,
        mu=mu,
        s_edges=s_edges,
        mu_edges=mu_edges,
        coords=['s', 'mu'],
    )

    rebinned = observable.select(s=slice(0, None, 2))

    def toarray(transform):
        if hasattr(transform, 'toarray'):
            return transform.toarray()
        return transform

    coord_transform = toarray(observable._transform(slice(0, None, 2), axis='s', name='s', full=False))
    value_transform = toarray(observable._transform(slice(0, None, 2), axis='s', name='normalized_counts'))
    norm_transform = toarray(observable._transform(slice(0, None, 2), axis='s', name='norm'))

    expected_s = np.tensordot(coord_transform, observable.s, axes=([1], [0]))
    expected_counts = np.tensordot(value_transform, observable.normalized_counts, axes=([1], [0]))
    expected_norm = np.tensordot(norm_transform, observable.norm, axes=([1], [0]))
    wrong_counts = np.tensordot(coord_transform, observable.normalized_counts, axes=([1], [0]))

    assert not np.allclose(coord_transform, value_transform)
    assert np.allclose(rebinned.s, expected_s)
    assert np.allclose(rebinned.normalized_counts, expected_counts)
    assert np.allclose(rebinned.norm, expected_norm)
    assert not np.allclose(rebinned.normalized_counts, wrong_counts)


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

    def generate_triumvirate(return_type='bispectrum', ell=(0, 0, 0)):
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
    print(corr.get('DD').attrs)
    assert np.allclose(corr.value_as_leaf().value(), corr.value())
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
    pole = from_triumvirate(spec, ells=0)
    fn = test_dir / 'pole.h5'
    pole.write(fn)
    read(fn)

    ells = [(0, 0, 0), (2, 0, 2)]
    spec = [generate_triumvirate(return_type='bispectrum', ell=ell) for ell in ells]
    pole = from_triumvirate(spec, ells=ells)
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


def test_utils():
    from lsstypes import utils
    nobs, nbins, nparams = 1000, 50, 10
    factor = utils.get_hartlap2007_factor(nobs, nbins)
    assert np.allclose(factor, 0.948948948948949), factor
    factor = utils.get_percival2014_factor(nobs, nbins, nparams)
    assert np.allclose(factor, 1.0302691965504787), factor

    def get_theory(k, ell):
        from cosmoprimo.fiducial import DESI
        cosmo = DESI(engine='eisenstein_hu')
        pk1d = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
        return 1. / (1 + sum(ell)) * pk1d(k[..., 0]) * pk1d(k[..., 1])

    def get_spectrum3(seed=42):
        rng = np.random.RandomState(seed=seed)
        ndim = 2
        ells = [(0, 0, 0), (2, 0, 2)]
        spectrum = []
        for ell in ells:
            uedges = np.linspace(0., 0.2, 41)
            uedges = [np.column_stack([uedges[:-1], uedges[1:]])] * ndim
            k = [np.mean(uedge, axis=-1) for uedge in uedges]
            nmodes1d = [np.ones(uedge.shape[0], dtype='i') for uedge in uedges]

            def _product(array):
                if not isinstance(array, (tuple, list)):
                    array = [array] * ndim
                grid = np.meshgrid(*array, sparse=False, indexing='ij')
                return np.column_stack([tmp.ravel() for tmp in grid])

            # of shape (nbins, ndim, 2)
            k_edges = np.concatenate([_product([edge[..., 0] for edge in uedges])[..., None], _product([edge[..., 1] for edge in uedges])[..., None]], axis=-1)
            k = _product(k)
            nmodes = np.prod(_product(nmodes1d), axis=-1)
            k = np.mean(k_edges, axis=-1)
            num_raw = get_theory(k, ell)
            spectrum.append(Mesh3SpectrumPole(k=k, k_edges=k_edges, nmodes=nmodes, num_raw=num_raw))
        return Mesh3SpectrumPoles(spectrum, ells=ells)

    import numbers
    import scipy as sp

    def make_spectrum_rebinning_matrix(current_theory: types.ObservableTree,
                                       new_coords: int=None,
                                       interp_order: int=3,
                                       diag: str=None):
        assert diag in [None, 'separate']
        flattened_theory = current_theory.flatten(level=None)
        coord_name = list(flattened_theory[0].coords())
        assert len(coord_name) == 1
        coord_name = coord_name[0]
        current_coords = np.concatenate([pole.coords(coord_name) for pole in flattened_theory], axis=0)
        current_coords = [np.unique(current_coord) for current_coord in current_coords.T]
        if new_coords is None:
            new_coords = current_coords
        elif isinstance(new_coords, numbers.Number):
            n = new_coords
            new_coords = [np.linspace(coords.min(), coords.max(), n) for coords in current_coords]

        def flatten_coords(*coords):
            coords_flat = np.meshgrid(*coords, indexing='ij')
            return np.column_stack([coord.ravel() for coord in coords_flat])

        new_coords_flat = flatten_coords(*new_coords)

        value = []
        for label, pole in theory.items(level=None):
            _current_coords = next(iter(pole.coords().values()))
            current_coords = [np.unique(current_coord) for current_coord in _current_coords.T]
            assert np.allclose(flatten_coords(*current_coords), _current_coords)
            matrices1d = [utils.matrix_spline_interp(new_coord, current_coord, interp_order=interp_order) for new_coord, current_coord in zip(new_coords, current_coords)]
            matrixnd = matrices1d[0]
            for matrix1d in matrices1d[1:]:
                matrixnd = np.kron(matrixnd, matrix1d)
            if diag == 'separate':
                shape = tuple(len(coord) for coord in current_coords)
                assert all(ss == shape[0] for ss in shape[1:])
                multi_index = tuple(np.arange(s) for s in shape)
                diag_index = np.ravel_multi_index(multi_index, dims=shape)
                # zero-out diagonal contribution
                matrixnd[diag_index, :] = 0.
                # then add the diagonal contribution
                matrixdiag = np.zeros(matrixnd.shape[:1] + (len(diag_index),))
                matrixdiag[diag_index, np.arange(len(diag_index))] = 1.
                matrixnd = np.concatenate([matrixnd, matrixdiag], axis=-1)
            value.append(matrixnd)

        def f(pole):
            coord_values = new_coords_flat
            if diag == 'separate':
                coord_values = np.concatenate([new_coords_flat, np.column_stack(current_coords)], axis=0)
            return types.ObservableLeaf(value=np.zeros(len(coord_values)), **{coord_name: coord_values}, coords=[coord_name])

        new_theory = current_theory.map(lambda pole: f(pole), level=None)
        value = sp.linalg.block_diag(*value)
        return types.WindowMatrix(value=value, theory=new_theory, observable=current_theory)

    theory = get_spectrum3()
    matrix = make_spectrum_rebinning_matrix(theory, new_coords=20, interp_order=3, diag='separate')
    theory2 = matrix.theory.map(lambda pole, label: pole.clone(value=get_theory(pole.coords('k'), label['ells'])), input_label=True)
    interpolated = matrix.dot(theory2.value())
    interpolated = theory.clone(value=interpolated)

    if False:
        import matplotlib.pyplot as plt
        ax = plt.gca()
        for ill, (label, pole) in enumerate(interpolated.items()):
            color = f'C{ill:d}'
            pole2 = pole.unravel()
            ell = label['ells']
            k1 = pole2.coords('k1')
            noffsets = 5
            for offset in range(noffsets):
                k2 = k1[offset:]
                alpha = 1. - offset / noffsets
                ax.plot(k2, k2**2 * pole2.value().diagonal(offset=offset), color=color, linestyle='-', alpha=alpha)
                ax.plot(k2, k2**2 * get_theory(np.column_stack([k1[:len(k1) - offset], k2]), ell), color=color, linestyle='--', alpha=alpha)
        plt.show()


def test_wrap():

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

    def wrap(count: types.Count2) -> types.Count2:
        """
        Wrap a Count2 with symmetric edges along the 2nd dimension as a Count2 with half the number of bins along that dimension, and counts summed accordingly.
        This is useful for example to wrap a correlation function in (s, mu) coordinates as a correlation function in (s, |mu|) coordinates.
        """
        if count.shape[1] % 2:
            raise ValueError(f'input counts cannot be wrapped as 2nd dimension is odd = {count.shape[1]:d}')
        mid = count.shape[1] // 2
        sl_neg, sl_pos = slice(mid - 1, None, -1), slice(mid, None, 1)
        coord_name = count._coords_names[1]
        mu_edges = count.edges(coord_name)
        if not np.allclose(mu_edges[sl_neg], - mu_edges[sl_pos, ::-1]):
            raise ValueError(f'input counts cannot be wrapped as 2nd dimension edges are not symmetric; {mu_edges[sl_neg]} != {-mu_edges[sl_pos, ::-1]}')
        mu_edges = mu_edges[sl_pos]
        counts = count.values('counts')
        # Sum counts in the negative and positive halves along the 2nd dimension
        counts = counts[..., sl_neg] + counts[..., sl_pos]
        norm = count.values('norm')[..., sl_pos]
        # Prepare wrapped edges
        edges = dict(count.edges())
        edges[coord_name] = mu_edges
        edges = {'{}_edges'.format(coord): edge for coord, edge in edges.items()}
        # Prepare wrapped coordinates
        mu_coord = count.coords(coord_name)[sl_pos]
        coords = dict(count.coords())
        coords[coord_name] = mu_coord
        return Count2(counts=counts, norm=norm, **coords, **edges, coords=list(coords), attrs=dict(count.attrs))

    correlation = get_correlation()
    correlation = correlation.map(lambda count: wrap(count))
    assert correlation.get('DD').shape[1] == 50

    correlation = get_correlation_jackknife()
    correlation = correlation.map(lambda count: wrap(count), level=None, is_leaf=lambda branch: False)  #type(branch) is types.Count2)
    assert correlation.get('DD').realization(0).shape[1] == 50


def test_io_speed():
    import time

    test_dir = Path('_tests')

    def get_spectrum(seed=42):
        ells = [0, 2, 4]
        rng = np.random.RandomState(seed=seed)
        spectrum = []
        for ell in ells:
            k_edges = np.arange(0., 0.4, 0.001)
            k_edges = np.column_stack([k_edges[:-1], k_edges[1:]])
            k = np.mean(k_edges, axis=-1)
            spectrum.append(Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=rng.uniform(size=k.size)))
        return Mesh2SpectrumPoles(spectrum, ells=ells)

    spectrum = get_spectrum()
    fn = test_dir / 'spectrum.h5'
    spectrum.write(fn)

    import h5py
    from lsstypes.base import _h5py_recursively_read_dict, from_state

    n = 100
    t0 = time.time()
    for i in range(n):
        spectrum2 = read(fn)
    print(f'Readout time h5py: {(time.time() - t0) / n:.5f} s')

    state = spectrum.__getstate__(to_file=True)
    fn_npy = fn.with_suffix('.npy')
    np.save(fn_npy, state)

    t0 = time.time()
    for i in range(n):
        state = np.load(fn_npy, allow_pickle=True)[()]
        spectrum2 = from_state(state)
    print(f'Readout time npy: {(time.time() - t0) / n:.5f} s')

    spectrum_batch = {i: get_spectrum(seed=seed) for i, seed in enumerate(range(100))}
    fn = test_dir / 'spectrum.h5'
    types.write(fn, spectrum_batch)

    # Read-out time scales linearly with the number of hdf5 groups
    t0 = time.time()
    spectrum2 = types.read(fn)
    print(f'Readout time npy: {(time.time() - t0):.5f} s')



if __name__ == '__main__':

    test_utils()
    test_tree()
    test_types()
    test_sparse()
    test_rebin()
    test_at()
    test_matrix()
    test_likelihood()
    test_dict()
    test_readme()
    test_io()
    test_external()
    test_wrap()
    test_select_reuses_mesh3_transform_signature()
