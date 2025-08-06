import numpy as np

from lsstypes import ObservableLeaf, ObservableTree


def test_counts():
    s = np.linspace(0., 100., 51)
    mu = np.linspace(-1., 1., 101)
    rng = np.random.RandomState(seed=42)
    labels = ['DD', 'DR', 'RR']
    leaves = []
    for label in labels:
        counts = 1. + rng.uniform(size=(s.size, mu.size))
        leaves.append(ObservableLeaf(values={'counts': counts}, coords={'s': s, 'mu': mu}, attrs=dict(los='x')))

    leaf = leaves[0]
    leaf2 = leaf.select(s=(10., 80.), mu=(-0.8, 1.))
    assert np.all(leaf2.coords('s') <= 80)
    assert np.all(leaf2.coords('mu') >= -0.8)
    assert leaf2.values('counts').ndim == 2
    assert leaf2.attrs == dict(los='x')
    leaf3 = leaf.at[...].select(s=(10., 80.), mu=(-0.8, 1.))
    assert leaf3.shape == leaf2.shape
    leaf4 = leaf.at(s=(10., 80.)).select(s=(20., 70.), mu=(-0.8, 1.))

    tree = ObservableTree(leaves, corr=labels)
    assert tree.labels(keys_only=True) == ['corr']
    assert tree.labels() == [{'corr': 'DD'}, {'corr': 'DR'}, {'corr': 'RR'}]
    assert len(tree.values()) == tree.size

    leaf = ObservableLeaf(values={'spectrum': counts}, coords={'k': s, 'mu': mu}, attrs=dict(los='x'))
    tree2 = ObservableTree([tree, leaf], observable=['correlation', 'spectrum'])
    assert tree2.labels(keys_only=True) == ['observable', 'corr']
    assert tree2.labels(level=0) == [{'observable': 'correlation'}, {'observable': 'spectrum'}]
    assert tree2.labels() == [{'observable': 'correlation', 'corr': 'DD'}, {'observable': 'correlation', 'corr': 'DR'}, {'observable': 'correlation', 'corr': 'RR'}, {'observable': 'spectrum'}]


if __name__ == '__main__':

    test_counts()