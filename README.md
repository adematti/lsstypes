## lsstypes

Structured data types designed for large-scale structure (LSS) measurements, with built-in support for coordinate selection, rebinning, and flexible file formats.

---

## Features

- **Hierarchical data structure** — organize complex LSS data using tree-like structures.
- **Coordinate-based selection and rebinning** — easily filter and rebin data subsets based on spatial or other dimensions.
- **Flexible storage** — seamlessly read from and write to:
  - **HDF5** (via `h5py`) — for efficient binary data handling.
  - **Plain text** — for easy inspection and interoperability.


See notebooks in nb/.

---

## Installation

```bash
pip install git+https://github.com/adematti/lsstypes.git
```

**Dependencies**:

- Required: `numpy`
- Optional (enhanced functionality):
  - `h5py` — to enable HDF5 format
  - `scipy` — to support sparse-matrix‑based rebinning

---

## Basic usage

```python
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
```

See the `nb/` directory for interactive examples using Jupyter notebooks.


## Predefined types

- Mesh2SpectrumPoles: power spectrum multipoles
- Count2: pair counts
- Count2Correlation: correlation function from pair counts
- Count2Jackknife: Jackknife pair counts
- Count2JackknifeCorrelation: correlation function from jackknife pair counts