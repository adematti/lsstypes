from .base import ObservableLeaf, ObservableTree, read, write, register_type, tree_flatten, tree_map
from .base import WindowMatrix, CovarianceMatrix, GaussianLikelihood
from .types import (Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh2CorrelationPole, Mesh2CorrelationPoles, Mesh3SpectrumPole, Mesh3SpectrumPoles,
Mesh3CorrelationPole, Mesh3CorrelationPoles, Count2, Count2Jackknife, Count2Correlation, Count2JackknifeCorrelation, Count2CorrelationPole, Count2CorrelationPoles, Count2CorrelationWedge, Count2CorrelationWedges)

__version__ = '1.0.0'


def sum(observables, **kwargs):
    assert len(observables) >= 1
    return observables[0].__class__.sum(observables, **kwargs)


def mean(observables):
    assert len(observables) >= 1
    return observables[0].__class__.mean(observables)


def cov(observables):
    assert len(observables) >= 1
    return observables[0].__class__.cov(observables)


def join(observables):
    assert len(observables) >= 1
    return observables[0].__class__.join(observables)