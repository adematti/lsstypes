# lsstypes

Data types for large scale structure measurements.

## Overview

Define data types in a tree-like structure, implementing:
- selection on coordinates
- rebinning

Two file formats supported:
- hdf5 (preferred; requires `h5py`)
- plain text


See notebooks in nb/.

---

## Installation

```bash
pip install git+https://github.com/adematti/lsstypes.git
```

Dependencies:
- `numpy`
- `h5py` (optional)

---

## TODO

- [ ] rebinning
- [ ] covariance matrix
- [ ] window matrix

---