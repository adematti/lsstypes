from pathlib import Path
import numpy as np

from lsstypes import ObservableLeaf, ObservableTree, read, write


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

    tree = ObservableTree(leaves, keys=labels)
    assert tree.labels(keys_only=True) == ['keys']
    assert tree.labels() == [{'keys': 'DD'}, {'keys': 'DR'}, {'keys': 'RR'}]
    assert len(tree.value()) == tree.size
    tree2 = tree.at(keys='DD').select(s=(10., 80.))
    assert tree2.get(keys='DD').shape != tree2.get(keys='DR').shape
    tree2 = tree.select(s=(10., 80.))
    assert tree2.get(keys='DD').shape == tree2.get(keys='DR').shape

    k = np.linspace(0., 0.2, 21)
    spectrum = rng.uniform(size=k.size)
    leaf = ObservableLeaf(spectrum=spectrum, k=k, coords=['k'], attrs=dict(los='x'))
    tree2 = ObservableTree([tree, leaf], observable=['correlation', 'spectrum'])
    assert tree2.labels(keys_only=True) == ['observable', 'keys']
    assert tree2.labels(level=0) == [{'observable': 'correlation'}, {'observable': 'spectrum'}]
    assert tree2.labels() == [{'observable': 'correlation', 'keys': 'DD'}, {'observable': 'correlation', 'keys': 'DR'}, {'observable': 'correlation', 'keys': 'RR'}, {'observable': 'spectrum'}]

    fn = test_dir / 'tree.h5'
    write(fn, tree2)
    #tree3 = read(fn)
    #assert tree3 == tree2

    fn = test_dir / 'tree.txt'
    write(fn, tree2)
    tree3 = read(fn)
    assert tree3 == tree2


def test_rebin():

    from lsstypes import Count2

    s_edges = np.linspace(0., 100., 21)
    s_edges = np.column_stack([s_edges[:-1], s_edges[1:]])
    mu_edges = np.linspace(-1., 1., 11)
    mu_edges = np.column_stack([mu_edges[:-1], mu_edges[1:]])
    s, mu = np.mean(s_edges, axis=-1), np.mean(mu_edges, axis=-1)
    rng = np.random.RandomState(seed=42)
    counts = 1. + rng.uniform(size=(s.size, mu.size))
    counts = Count2(counts=counts, s=s, mu=mu, s_edges=s_edges, mu_edges=mu_edges, coords=['s', 'mu'], attrs=dict(los='x'))
    counts2 = counts.rebin(s=slice(0, None, 2))
    assert counts2.shape[0] == counts.shape[0] // 2
    assert np.allclose(np.mean(counts2.counts), 2 * np.mean(counts.counts))
    counts3 = counts2.rebin(mu=slice(0, None, 2))
    assert counts3.shape[1] == counts.shape[1] // 2
    assert np.allclose(np.mean(counts3.counts), 2 * np.mean(counts2.counts))


def test_types():

    test_dir = Path('_tests')

    from lsstypes import Mesh2SpectrumPole, Mesh2SpectrumPoles

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
    poles.plot(show=True)
    poles2 = poles.rebin(k=slice(0, None, 2))

    fn = test_dir / 'spectrum.txt'
    poles.write(fn)
    poles2 = read(fn)
    assert poles2 == poles

    all_poles, all_labels = [], []
    z = [0.2, 0.4, 0.6]
    for iz, zz in enumerate(z):
        poles = get_poles()
        poles.attrs['zeff'] = zz
        all_poles.append(poles)
        all_labels.append(f'z{iz:d}')
    all_poles = ObservableTree(all_poles, z=all_labels)
    all_poles.write(fn)
    all_poles = read(fn)
    all_poles.get('z0').plot(show=True)


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

    from lsstypes.external import from_pypower, from_pycorr

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
        data_samples1 = np.rint(data_weights1 * 10).astype(int)
        randoms_samples1 = np.rint(randoms_weights1 * 10).astype(int)
        edges = (np.linspace(0., 101, 51), np.linspace(-1., 1., 101))
        return TwoPointCorrelationFunction('smu', edges, data_positions1=data_positions1, data_weights1=data_weights1, data_samples1=data_samples1,
                                            randoms_positions1=randoms_positions1, randoms_weights1=randoms_weights1, randoms_samples1=randoms_samples1,
                                            engine='corrfunc', position_type='pos', nthreads=4)

    pypoles = generate_pypower()
    poles = from_pypower(pypoles)
    assert np.allclose(poles.value(), pypoles.power.ravel())
    poles = poles.rebin(k=slice(0, None, 2))
    pypoles = pypoles[:(pypoles.shape[0] // 2) * 2:2]
    assert np.allclose(poles.get(0).coords('k'), pypoles.k, equal_nan=True)
    assert np.allclose(poles.value(), pypoles.power.ravel())

    pycorr = generate_pycorr()
    corr = from_pycorr(pycorr)
    assert np.allclose(corr.coords(axis='s'), pycorr.sepavg(axis=0), equal_nan=True)
    assert np.allclose(corr.value(), pycorr.corr, equal_nan=True)
    corr = corr.rebin(s=slice(0, None, 2))
    pycorr = pycorr[:(pycorr.shape[0] // 2) * 2:2]
    assert np.allclose(corr.coords(axis='s'), pycorr.sepavg(axis=0), equal_nan=True)
    assert np.allclose(corr.value(), pycorr.corr, equal_nan=True)

    pycorr = generate_pycorr_jackknife()
    corr = from_pycorr(pycorr)
    assert np.allclose(corr.coords(axis='s'), pycorr.sepavg(axis=0), equal_nan=True)
    assert np.allclose(corr.value(), pycorr.corr, equal_nan=True)
    corr = corr.rebin(s=slice(0, None, 2))
    pycorr = pycorr[:(pycorr.shape[0] // 2) * 2:2]
    assert np.allclose(corr.coords(axis='s'), pycorr.sepavg(axis=0), equal_nan=True)
    assert np.allclose(corr.value(), pycorr.corr, equal_nan=True)


if __name__ == '__main__':

    #test_tree()
    #test_types()
    #test_sparse()
    #test_rebin()
    test_external()