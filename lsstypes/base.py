import os
import shutil

from . import utils

import numpy as np


def _h5py_recursively_write_dict(h5file, path, dic, with_attrs=True):
    """Save a nested dictionary of arrays to an HDF5 file using h5py."""
    import h5py
    for key, item in dic.items():
        if with_attrs and key == 'attrs':
            h5file[path].attrs.update(item)
            continue
        path_key = f'{path}/{key}'.rstrip('/')

        if isinstance(item, dict):
            # If dict has 'attrs' and other keys, it's a group with metadata
            grp = h5file.create_group(path_key, track_order=True)
            _h5py_recursively_write_dict(h5file, path_key, item, with_attrs=with_attrs)
        else:
            # Assume it's an array-like and write as dataset
            item = np.asarray(item)
            if isinstance(item.flat[0].item(), str):
                dset = h5file.create_dataset(path_key, shape=item.shape, dtype=h5py.string_dtype())
                dset[...] = item
            else:
                dset = h5file.create_dataset(path_key, data=item)


def _h5py_recursively_read_dict(h5file, path='/'):
    """
    Load a nested dictionary of arrays from an HDF5 file.
    Attributes are stored in a special 'attrs' key.
    """
    import h5py
    dic = {}
    for key, item in h5file[path].items():
        path_key = f'{path}/{key}'.rstrip('/')
        if isinstance(item, h5py.Group):
            dic[key] = _h5py_recursively_read_dict(h5file, path_key)
        elif isinstance(item, h5py.Dataset):
            dic[key] = item[...]
            if h5py.check_string_dtype(item.dtype):
                dic[key] = dic[key].astype('U')
            if not dic[key].shape:
                dic[key] = dic[key].item()
    # Load group-level attributes, if any
    if h5file[path].attrs:
        dic['attrs'] = {k: v for k, v in h5file[path].attrs.items()}

    return dic


def _npy_auto_format_specifier(array):
    """Return a format specifier string for numpy array dtype for text output."""
    if np.issubdtype(array.dtype, np.bool_):
        return '%d'
    elif np.issubdtype(array.dtype, np.integer):
        return '%d'
    elif np.issubdtype(array.dtype, np.floating):
        return '%.18e'
    elif np.issubdtype(array.dtype, np.str_):
        maxlen = array.dtype.itemsize // 4  # 4 bytes per unicode char
        return f'%{maxlen}s'
    elif np.issubdtype(array.dtype, np.bytes_):
        maxlen = array.dtype.itemsize
        return f'%{maxlen}s'
    else:
        raise TypeError(f"Unsupported dtype: {array.dtype}")


def _txt_recursively_write_dict(path, dic, with_attrs=True):
    """Save a nested dictionary of arrays to an HDF5 file using h5py."""
    utils.mkdir(path)
    for key, item in dic.items():
        path_key = os.path.join(path, key)
        if with_attrs and key == 'attrs':
            import json
            with open(path_key + '.json', 'w') as file:
                json.dump(item, file)
            continue  # handle attrs below

        if isinstance(item, dict):
            # If dict has 'attrs' and other keys, it's a group with metadata
            utils.mkdir(path_key)
            _txt_recursively_write_dict(path_key, item, with_attrs=with_attrs)
        else:
            # Assume it's an array-like and write as dataset
            item = np.asarray(item)
            np.savetxt(path_key + '.txt', np.ravel(item), fmt=_npy_auto_format_specifier(item),
                       header=f'dtype = {str(item.dtype)}\nshape = {str(item.shape)}')


def _txt_recursively_read_dict(path='/'):
    """
    Load a nested dictionary of arrays from an HDF5 file.
    Attributes are stored in a special 'attrs' key.
    """
    dic = {}
    for key in os.listdir(path):
        path_key = os.path.join(path, key)
        if os.path.isdir(path_key):
            dic[key] = _txt_recursively_read_dict(path_key)
        elif os.path.isfile(path_key):
            if path_key.endswith('.json'):
                import json
                with open(path_key, 'r') as file:
                    dic['attrs'] = json.load(file)
                continue
            with open(path_key, 'r') as file:
                dtype = file.readline().rstrip('\r\n').replace(' ', '').replace('#dtype=', '')
                dtype = np.dtype(dtype)
                shape = file.readline().rstrip('\r\n').replace(' ', '').replace('#shape=', '')[1:-1].split(',')
                shape = tuple(int(s) for s in shape if s)
            key = key[:-4] if key.endswith('.txt') else key
            dic[key] = np.loadtxt(path_key, dtype=dtype)
            if not shape: dic[key] = dic[key].item()
            else: dic[key] = dic[key].reshape(shape)
    return dic


def _write(filename, state, overwrite=True):
    """
    Write a state dictionary to disk in HDF5 or text format.

    Parameters
    ----------
    filename : str
        Output file name.
    state : dict
        State dictionary to write.
    overwrite : bool, optional
        If True, overwrite existing file.
    """
    filename = str(filename)
    utils.mkdir(os.path.dirname(filename))
    if any(filename.endswith(ext) for ext in ['.h5', '.hdf5']):
        import h5py
        with h5py.File(filename, 'w' if overwrite else 'a') as f:
            _h5py_recursively_write_dict(f, '/', state)
    elif any(filename.endswith(ext) for ext in ['txt']):
        if overwrite:
            shutil.rmtree(filename[:-4], ignore_errors=True)
        _txt_recursively_write_dict(filename[:-4], state)
    else:
        raise ValueError(f'unknown file format: {filename}')


def _read(filename):
    """
    Read a state dictionary from disk in HDF5 or text format.

    Parameters
    ----------
    filename : str
        Input file name.

    Returns
    -------
    dic : dict
        State dictionary read from file.
    """
    filename = str(filename)
    if any(filename.endswith(ext) for ext in ['.h5', '.hdf5']):
        import h5py
        with h5py.File(filename, 'r') as f:
            dic = _h5py_recursively_read_dict(f, '/')
    elif any(filename.endswith(ext) for ext in ['.txt']):
        dic = _txt_recursively_read_dict(filename[:-4])
    else:
        raise ValueError(f'Unknown file format: {filename}')
    return dic


def from_state(state):
    """
    Instantiate an observable from a state dictionary.

    Parameters
    ----------
    state : dict
        State dictionary.

    Returns
    -------
    ObservableLeaf or ObservableTree
        Instantiated observable object.
    """
    _name = str(state.pop('name'))
    try:
        cls = _registry[_name]
    except KeyError:
        raise ValueError(f'Cannot find {_name} in registered observables: {_registry}')
    new = cls.__new__(cls)
    new.__setstate__(state)
    return new


def write(filename, observable):
    """
    Write observable to disk.

    Parameters
    ----------
    filename : str
        Output file name.
    """
    _write(filename, observable.__getstate__(to_file=True))


def read(filename):
    """
    Read observable from disk.

    Parameters
    ----------
    filename : str
        Input file name.

    Returns
    -------
    ObservableLeaf or ObservableTree
    """
    return from_state(_read(filename))


def _format_masks(shape, masks):
    """
    Format masks or slices into indices for array selection.

    Parameters
    ----------
    shape : tuple
        Shape of the array.
    masks : tuple of array-like or slices
        Masks or slices for each axis.

    Returns
    -------
    tuple
        Indices for selection.
    """
    # Return indices
    if not isinstance(masks, tuple): masks = (masks,)
    alls = [np.arange(s) for s in shape]
    masks = masks + (Ellipsis,) * (len(shape) - len(masks))
    indices = tuple(a[m] for a, m in zip(alls, masks))
    return indices


def _tensor_product(*arrays):
    """
    Compute the tensor (outer) product of multiple arrays.

    Parameters
    ----------
    *arrays : array-like
        Arrays to compute the tensor product of.

    Returns
    -------
    np.ndarray
        Tensor product array.
    """
    reshaped = [array.reshape((1,)*i + (-1,) + (1,)*(len(arrays)-i-1))
                for i, array in enumerate(arrays)]
    out = reshaped[0]
    for r in reshaped[1:]:
        out = out * r
    return out


_registry = {}


def register_type(cls):
    """
    Register a class in the observable registry.

    Parameters
    ----------
    cls : type
        Class to register. Must have a _name attribute.

    Returns
    -------
    type
        The registered class.
    """
    _registry[cls._name] = cls
    return cls


def deep_eq(obj1, obj2, equal_nan=True, raise_error=False, label=None):
    """(Recursively) test equality between ``obj1`` and ``obj2``."""
    if raise_error:
        label_str = f' for object {label}' if label is not None else ''
    if type(obj2) is type(obj1):
        if isinstance(obj1, dict):
            if obj2.keys() == obj1.keys():
                return all(deep_eq(obj1[name], obj2[name], raise_error=raise_error, label=name) for name in obj1)
            elif raise_error:
                raise ValueError(f'Different keys: {obj2.keys()} vs {obj1.keys()}{label_str}')
        elif isinstance(obj1, (tuple, list)):
            if len(obj2) == len(obj1):
                return all(deep_eq(o1, o2, raise_error=raise_error, label=label) for o1, o2 in zip(obj1, obj2))
            elif raise_error:
                raise ValueError(f'Different lengths: {len(obj2)} vs {len(obj1)}{label_str}')
        else:
            try:
                toret = np.array_equal(obj1, obj2, equal_nan=equal_nan)
            except TypeError:  # nan not supported
                toret = np.array_equal(obj1, obj2)
            if not toret and raise_error:
                raise ValueError(f'Not equal: {obj2} vs {obj1}{label_str}')
            return toret
    if raise_error:
        raise ValueError(f'Not same type: {type(obj2)} vs {type(obj1)}{label_str}')
    return False



def _build_big_tensor(m, shape, axis):
    """
    Build sparse matrix M corresponding to applying m on `axis`.

    Parameters
    ----------
    m : ndarray or sparse, shape (I, Iprime)
        Transformation matrix on the chosen axis.
    shape : tuple of ints
        Shape of the input array (I1, I2, ..., Ip).
    axis : int
        Axis (0..p-1) to which `m` applies.

    Returns
    -------
    M : scipy.sparse matrix, shape (np.prod(out_shape), np.prod(shape))
        Sparse matrix implementing the action.
    out_shape : tuple
        Shape of the output array after transformation.
    """
    try:
        import scipy.sparse as sp
        def get_m(m): return sp.csr_matrix(m)
        def get_id(s): return sp.identity(s, format='csr')
        def kron(a, b): return sp.kron(a, b, format='csr')
    except ImportError:
        def get_m(m): return m
        def get_id(s): return np.identity(s)
        def kron(a, b): return np.kron(a, b)


    nd = len(shape)
    Iprime = shape[axis]
    I = m.shape[0]
    assert m.shape[1] == Iprime

    ops = []
    for ax in range(nd):
        if ax == axis:
            ops.append(get_m(m))
        else:
            ops.append(get_id(shape[ax]))

    # Kronecker product in correct order
    M = ops[0]
    for op in ops[1:]:
        M = kron(M, op)

    out_shape = shape[:axis] + (I,) + shape[axis+1:]
    return M, out_shape


def _build_matrix_from_mask2(indexout, indexin, shape, toarray=False):
    """Build matrix that transforms an array of shape `shape` by selecting `indexin` to `indexout`."""
    try:
        import scipy.sparse as sp
    except ImportError:
        sp = None

    if sp is not None:
        matrix = sp.csr_matrix((np.ones(indexout.size, dtype=int), (indexout, indexin)), shape=shape)
        if toarray:
            return matrix.toarray()
        return matrix

    matrix = np.zeros(shape, dtype=int)
    matrix[indexout, indexin] = 1
    return matrix


def _build_matrix_from_mask(mask, size=None, toarray=False):
    """Build matrix that transforms an array of shape `size` by selecting `mask`."""
    # Build sparse selection matrix
    if np.issubdtype(mask.dtype, np.integer):
        indexin = mask
        nin = size
    else:
        indexin = np.flatnonzero(mask)
        nin = len(mask)
    nout = len(indexin)
    indexout = np.arange(nout)
    return _build_matrix_from_mask2(indexout, indexin, (nout, nin))



def _nan_to_zero(array):
    """Turn NaNs in array to zeros."""
    return np.where(np.isnan(array), 0., array)


@register_type
class ObservableLeaf(object):
    """A compressed observable with named values and coordinates, supporting slicing, selection, and plotting."""

    _name = 'leaf_base'
    _forbidden_names = ('name', 'attrs', 'values_names', 'coords_names', 'meta')
    _is_leaf = True

    def __init__(self, coords=None, attrs=None, meta=None, **data):
        """
        Parameters
        ----------
        **data : dict
            Dictionary of named arrays (e.g. {"spectrum": ..., "nmodes": ..., "k": ..., "mu": ...}).
        coords : list
            List of coordinates (e.g. ["k", "mu"]).
            Should match arrays provided in ``data``.
        attrs : dict, optional
            Additional attributes.
        """
        self.__pre_init__(coords=coords, attrs=attrs, meta=meta, **data)
        self.__post_init__()

    def __pre_init__(self, coords=None, attrs=None, meta=None, **data):
        # Setup attrs, meta, data, coords_names
        self._attrs = dict(attrs or {})
        self._meta = dict(meta or {})
        self._data = dict(data)
        self._coords_names = list(coords)

    def __post_init__(self):
        # Check data consistency
        assert not any(k in self._forbidden_names for k in self._data), f'Cannot use {self._forbidden_names} as name for arrays'
        _edges_names = [f'{axis}_edges' for axis in self._coords_names]
        self._values_names = [name for name in self._data if name not in self._coords_names and name not in _edges_names]
        assert len(self._values_names), 'Provide at least one value array'

    def __getattr__(self, name):
        """Access values and coords by name."""
        if name in self._meta:
            return self._meta[name]
        if name in self._data:
            return self._data[name]
        raise AttributeError(name)

    def coords(self, axis=None, center=None):
        """
        Get coordinate array(s).

        Parameters
        ----------
        axis : str or int, optional
            Name or index of coordinate.

        Returns
        -------
        coords : array or dict
        """
        if axis is None:
            return {axis: self.coords(axis=axis, center=center) for axis in self._coords_names}
        if not isinstance(axis, str):
            axis = self._coords_names[axis]
        edges = self.edges(axis=axis, default=None)
        if center == 'mid_if_edges':
            if edges is None: center = None
            else: center = 'mid'
        if center is None:
            return self._data[axis]
        assert edges is not None, 'edges must be provided'
        return np.mean(edges, axis=-1)

    def edges(self, axis=None, default=None):
        """
        Get edge array(s).

        Parameters
        ----------
        axis : str or int, optional
            Name or index of coordinate.

        default : any, optional
            If `None`, estimate edges from coordinates if not provided.

        Returns
        -------
        edges : array or dict
        """
        if axis is None:
            return {axis: self.edges(axis=axis) for axis in self._coords_names}
        if not isinstance(axis, str):
            axis = self._coords_names[axis]
        axis_edges = f'{axis}_edges'
        if axis_edges in self._data:
            return self._data[axis_edges]
        if default is None:
            return default
        coord = self._data[axis]
        edges = (coord[:-1] + coord[1:]) / 2.
        edges = np.concatenate([np.array([coord[0] - (coord[1] - coord[0])]), edges, np.array([coord[-1] + (coord[-1] - coord[-2])])])
        edges = np.column_stack([edges[:-1], edges[1:]])
        return edges

    def values(self, name=None):
        """
        Get value array(s).

        Parameters
        ----------
        name : str, optional
            Name of value.

        Returns
        -------
        values : array or dict
        """
        if name is None:
            return {name: self.values(name=name) for name in self._values_names}
        if not isinstance(name, str):
            name = self._values_names[name]
        return self._data[name]

    def value(self):
        """Get the 'main' value array (the first one)."""
        return self._data[self._values_names[0]]

    def __array__(self):
        return self.value()

    @property
    def shape(self):
        """Return observable shape."""
        return self._data[self._values_names[0]].shape

    @property
    def size(self):
        """Return observable size."""
        return self._data[self._values_names[0]].size

    @property
    def ndim(self):
        """NUmber of dimensions (coordinates)."""
        return len(self._coords_names)

    @property
    def attrs(self):
        """Return attributes dictionary."""
        return self._attrs

    def __getitem__(self, masks):
        """
        Mask or slice the observable.

        Parameters
        ----------
        masks : tuple of array-like or slice
            Mask or slice to apply to all value and coordinate arrays.

        Returns
        -------
        ObservableLeaf
        """
        indices = _format_masks(self.shape, masks)
        # Those are indices
        index = np.ix_(*indices)
        new = self.copy()
        for name in self._values_names:
            new._data[name] = self._data[name][index]
        for axis, index in zip(self._coords_names, indices):
            new._data[axis] = new._data[axis][index]
            axis_edges = f'{axis}_edges'
            if axis_edges in new._data:
                new._data[axis_edges] = new._data[axis_edges][index]
        return new

    def _transform(self, limit, axis=0, name=None, full=None, return_edges=False, center='mid_if_edges'):
        # Return mask or matrix to transform the observable with input limit
        # limit: tuple (select range in coordinates), slice, ObservableLeaf, array-like (coordinates or edges)
        # axis: int or str, axis to rebin
        # name: str or (bool, bool), name of value to use for bin weighting
        if not isinstance(axis, str):
            axis = self._coords_names[axis]
        if limit is None:
            size = len(self._data[axis])
            index = np.arange(size)
            if return_edges:
                return index, self.edges(axis=axis)
            return index

        undefined_weight = False
        if isinstance(name, tuple):
            weight, normalized = name
        else:
            bw = getattr(self, '_binweight', None)
            if bw is None:
                undefined_weight = True
                weight, normalized = False, True
            else:
                weight, normalized = bw(name=name)

        def _format_slice(lim, coords):
            if isinstance(lim, tuple):
                mask = (coords >= lim[0]) & (coords <= lim[1])
                return mask
            if lim is None: lim = slice(None)
            size = coords.size
            start, stop, step = lim.start, lim.stop, lim.step
            # To handle slice(0, None, 1)
            if start is None: start = 0
            if step is None: step = 1
            if stop is None: stop = size
            if step < 0:
                raise IndexError('positive slicing step only supported')
            return slice(start, stop, step)

        def _mask_from_slice(sl, size):
            mask = np.zeros(size, dtype='?')
            mask[sl] = True
            return mask

        def _isin2d(array1, array2):
            if array1.ndim == array2.ndim == 1: return np.isin(array1, array2)
            assert len(array1) == len(array2)
            toret = True
            for a1, a2 in zip(array1, array2): toret &= np.isin(a1, a2)
            return toret

        _self_coords = self.coords(axis=axis, center=center)
        self_coords = _self_coords[(Ellipsis,) + (None,) * (2 - _self_coords.ndim)]
        ndim = self_coords.shape[1]
        if isinstance(limit, ObservableLeaf):
            _limit = limit.edges(axis=axis)
            if _limit is not None: limit = _limit
            else: limit = limit.coords(axis=axis)
        selection_only = False
        if isinstance(limit, (tuple, list, slice)):  # (), slice(...), (slice(...), slice(...), ...), ()
            if isinstance(limit, slice) or not isinstance(limit[0], tuple):
                limit = [limit] * ndim
            limit = list(limit)
            selection_only = True
            assert len(limit) <= ndim, f'Provide at most {ndim:d} limits'
            for iaxis, lim in enumerate(limit):
                assert isinstance(lim, (tuple, slice)), f'expect tuple/slice, got {lim}'
                lim = _format_slice(lim, self_coords[..., iaxis])
                selection_only &= (not isinstance(lim, slice)) or lim.step == 1
                limit[iaxis] = lim
            limit += [_format_slice(None, self_coords[..., iaxis]) for iaxis in range(len(limit), ndim)]
        else:
            selection_only = np.ndim(limit) == _self_coords.ndim  # coords and not edges

        if selection_only:
            if isinstance(limit, list):
                limit = [_mask_from_slice(lim, self_coords[..., iaxis].size) if isinstance(lim, slice) else lim for iaxis, lim in enumerate(limit)]
                mask = np.logical_and.reduce(limit)
                index = np.flatnonzero(mask)
            else:
                if _self_coords.ndim == 2:

                    def view(a):
                        return a.view([('', a.dtype)] * a.shape[1])

                    s_self_coords, s_limit = view(_self_coords), view(limit)
                else:
                    s_self_coords, s_limit = _self_coords, limit
                _, ind1, ind2 = np.intersect1d(s_self_coords, s_limit, return_indices=True)
                index = ind1[np.argsort(ind2)]
                assert np.allclose(_self_coords[index], limit), f'Cannot match coords {_self_coords} to input {limit}'
            if return_edges:
                edges = self.edges(axis=axis)
                if edges is not None: edges = edges[index]
                return index, edges
            return index

        def get_unique_edges(edges):
            return [np.unique(edges[:, iax], axis=0) for iax in range(edges.shape[1])]

        def get_1d_slice(edges, index):
            if isinstance(index, slice):
                edges1 = edges[index, 0]
                edges2 = edges[index.start + index.step - 1::index.step, 1]
                size = min(edges1.shape[0], edges2.shape[0])
                return np.column_stack([edges1[:size], edges2[:size]])
            return edges[index]

        self_edges = self.edges(axis=axis, default=None)
        assert self_edges is not None, 'edges must be provided to rebin the observable'

        if isinstance(limit, list):
            if len(limit) == 1:
                edges = get_1d_slice(self_edges, limit[0])
            else:
                edges1d = [get_1d_slice(e, s) for e, s in zip(get_unique_edges(self_edges), limit)]

                # This is to keep the same ordering
                upedges = self_edges[..., 1][_isin2d(self_edges[..., 1].T, [e[..., 1] for e in edges1d])]
                lowedges = np.column_stack([edges1d[iax][..., 0][np.searchsorted(edges1d[iax][..., 1], upedges[..., iax])] for iax in range(ndim)])
                edges = np.concatenate([lowedges[..., None], upedges[..., None]], axis=-1)
        else:
            edges = limit

        iaxis = self._coords_names.index(axis)
        # Broadcast iedges[:, None, :] against edges[None, :, :]
        mask = (self_edges[None, ..., 0] >= edges[:, None, ..., 0]) & (self_edges[None, ..., 1] <= edges[:, None, ..., 1])  # (new_size, old_size) or (new_size, old_size, ndim)
        if mask.ndim >= 3:
            mask = mask.all(axis=-1)  # collapse extra dims if needed
        shape = self.shape

        def multiply(m, a):
            if a is None: return m
            if hasattr(m, 'multiply'):  # scipy sparse
                return m.multiply(a)
            return m * a

        if undefined_weight:
            if mask.sum(axis=-1).max() > 1:
                import warnings
                warnings.warn('Non-trivial rebinning requires a _binweight function to be defined')

        if weight is not False:
            if len(shape) > 1:
                if full or full is None:
                    mask = _build_big_tensor(mask, shape, axis=iaxis)[0]
                    weight = np.ravel(weight) if weight is not None else 1
                else:
                    weight = np.sum(weight, axis=tuple(iax for iax in range(weight.ndim) if iax != iaxis))
            matrix = multiply(mask, weight)
        else:
            if full and len(shape) > 1:
                mask = _build_big_tensor(mask, shape, axis=iaxis)[0]
            matrix = mask * 1
        if normalized:
            # all isn't implemented for scipy sparse, just check the sum of the boolean array
            norm = 1 / np.ravel(np.where((matrix != 0).sum(axis=-1) == 0, 1, matrix.sum(axis=-1)))[:, None]
            matrix = multiply(matrix, norm)
        if return_edges:
            return matrix, edges
        return matrix

    def _update(self, **kwargs):
        if 'value' in kwargs:
            kwargs[self._values_names[0]] = kwargs.pop('value')
        for name, value in kwargs.items():
            if name not in self._data: raise ValueError('{name} not unknown')
            self._data[name] = value

    def clone(self, **kwargs):
        """Copy and update data."""
        new = self.copy()
        new._update(**kwargs)
        return new

    def select(self, center='mid_if_edges', **limits):
        """
        Select a range in one or more coordinates.

        Parameters
        ----------
        limits : dict
            Each key is a coordinate name, value is either:
            - (min, max) tuple for this coordinate
            - slice for this coordinate
            - array-like of coordinates or edges to select

        center : str, optional
            How to compute the coordinate values if edges are provided:
            - 'mid': mean of edges
            - 'mid_if_edges': 'mid' if edges are provided, else use coordinates as is
            - `None`: use coordinates as is

        Returns
        -------
        ObservableLeaf
        """
        new = self.copy()
        for iaxis, axis in enumerate(self._coords_names):
            limit = limits.pop(axis, None)
            if limit is None: continue
            axis_edges = f'{axis}_edges'
            transform, edges = new._transform(limit, axis=axis, return_edges=True, center=center, full=False, name=axis)
            if transform.ndim == 1:  # mask
                index = transform
                for name in new._values_names:
                    new._data[name] = np.take(new._data[name], index, axis=iaxis)
                new._data[axis] = new._data[axis][index]
                if axis_edges in new._data:
                    new._data[axis_edges] = new._data[axis_edges][index]
            else:  # matrix
                nwmatrix_reduced = transform
                tmp = _nan_to_zero(new._data[axis])
                _data = {}
                _data[axis] = np.tensordot(nwmatrix_reduced, tmp, axes=([1], [0]))
                shape = tuple(len(_data[ax]) if ax == axis else len(new._data[ax]) for ax in new._coords_names)
                if axis_edges in new._data:
                    _data[axis_edges] = edges
                _cache = {}
                for name in new._values_names:
                    tmp = _nan_to_zero(new._data[name])
                    if name not in _cache: _cache[name] = new._transform(limit, axis=axis, name=name)
                    matrix = _cache[name]
                    if matrix.shape[1] == tmp.shape[iaxis]:  # compressed version
                        _data[name] = np.moveaxis(np.tensordot(matrix, tmp, axes=([1], [iaxis])), 0, iaxis)
                    else:
                        _data[name] = matrix.dot(tmp.ravel()).reshape(shape)
                new._data.update(_data)
        return new

    def match(self, observable):
        """Match coordinates to those of input observable."""
        return self.select(**{axis: observable for axis in self._coords_names})

    @property
    def at(self):
        """Update values in place."""
        return _ObservableLeafUpdateHelper(self)

    def copy(self):
        """Return a copy of the observable (numpy arrays not copied)."""
        new = self.__class__.__new__(self.__class__)
        new.__setstate__(self.__getstate__())
        return new

    def __getstate__(self, to_file=False):
        state = dict(self._data)
        for name in ['values_names', 'coords_names']:
            state[name] = list(getattr(self, '_' + name))
        state['name'] = self._name
        if self._attrs: state['attrs'] = dict(self._attrs)
        if self._meta: state['meta'] = dict(self._meta)
        return state

    def __setstate__(self, state):
        for name in ['values_names', 'coords_names']:
            setattr(self, '_' + name, [str(n) for n in state[name]])
        self._attrs = state.get('attrs', {})  # because of hdf5 reader
        self._meta = state.get('meta', {})
        self._data = {name: state[name] for name in self._values_names + self._coords_names}
        _edges_names = [f'{axis}_edges' for axis in self._coords_names]
        for name in _edges_names:
            if name in state: self._data[name] = state[name]

    def __eq__(self, other):
        return deep_eq(self.__getstate__(), other.__getstate__())

    def write(self, filename):
        """
        Write observable to disk.

        Parameters
        ----------
        filename : str
            Output file name.
        """
        return write(filename, self)

    def __add__(self, other):
        return self.sum([self, other])

    def __radd__(self, other):
        if other == 0: return self.copy()
        return self.__add__(other)

    def __iadd__(self, other):
        if other == 0: return self.copy()
        return self.__add__(other)

    @classmethod
    def _average(cls, observables, weights=None):
        # Average multiple observables
        new = observables[0].copy()
        if weights is None: weights = [1] * len(observables)
        for name in new._values_names + new._coords_names:
            if callable(weights):
                _weights = weights(observables, name)
                if _weights is None: continue  # keep as is
                divide = False
            else:
                _weights = weights
                divide = True
            assert len(_weights) == len(observables)
            new._data[name] = sum(weight * observable._data[name] for observable, weight in zip(observables, _weights))
            if divide: new._data[name] = new._data[name] / sum(_weights)
        return new

    @classmethod
    def sum(cls, observables):
        """Sum multiple observables."""
        return cls._average(observables, weights=getattr(cls, '_sumweight', None))

    @classmethod
    def mean(cls, observables):
        """Mean of multiple observables."""
        return cls._average(observables, weights=getattr(cls, '_meanweight', None))

    @classmethod
    def cov(cls, observables):
        """Covariance matrix of multiple observables."""
        assert len(observables) >= 1, 'Provide at least 2 observables to compute the covariance'
        mean = cls.mean(observables)
        value = np.cov([observable.value() for observable in observables], rowvar=False, ddof=1)
        return CovarianceMatrix(value, observable=mean)

    @classmethod
    def concatenate(cls, observables, axis=0):
        """
        Concatenate multiple observables.
        No check performed.
        """
        assert len(observables) >= 1, 'Provide at least 1 observable to concatenate'
        new = observables[0].copy()
        if not isinstance(axis, str):
            axis = new._coords_names[axis]
        iaxis = new._coords_names.index(axis)
        new._data[axis] = np.concatenate([observable._data[axis] for observable in observables], axis=0)
        axis_edges = f'{axis}_edges'
        if axis_edges in new._data:
            new._data[axis_edges] = np.concatenate([observable._data[axis_edges] for observable in observables], axis=0)
        for name in new._values_names:
            new._data[name] = np.concatenate([observable._data[name] for observable in observables], axis=iaxis)
        return new

    def __repr__(self):
        return f'{self.__class__.__name__}(coords={tuple(self._coords_names)}, values={tuple(self._values_names)}, shape={self.shape})'


def find_single_true_slab_bounds(mask):
    """
    Find the start and stop indices of a contiguous `True` region in a mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean or integer mask array.

    Returns
    -------
    tuple
        (start, stop) indices of the contiguous region.
    """
    start, stop = 0, 0
    if np.issubdtype(mask.dtype, np.integer):
        start, stop = mask[0], mask[-1] + 1
        if not np.all(np.diff(mask) == 1):
            raise ValueError('Discontinuous indexing')
    elif mask.any():
        start, stop = find_single_true_slab_bounds(np.flatnonzero(mask))
    return start, stop


class _ObservableLeafUpdateHelper(object):

    _observable: ObservableLeaf
    _hook = None

    def __init__(self, observable, hook=None):
        self._observable = observable
        self._hook = hook

    def __getitem__(self, masks):
        select = ('__getitem__', masks)
        return _ObservableLeafUpdateRef(self._observable, select, self._hook)

    def __call__(self, **kwargs):
        select = ('__select__', kwargs)
        return _ObservableLeafUpdateRef(self._observable, select, self._hook)


def _pad_transform(transform, start, size=None):
    # Pad a 1d (index) or 2d (matrix) transform to full size
    if transform.ndim == 1:
        if np.issubdtype(transform.dtype, np.integer):
            return start + transform
        mask = np.ones(size, dtype='?')
        mask[start:start + transform.size] = transform.ravel()
        return mask

    try:
        import scipy.sparse as sp
    except ImportError:
        sp = None

    stop = start + transform.shape[1]
    if sp is None:
        matrix = np.zeros_like(transform, shape=(start + transform.shape[0] + (size - stop), size))
        matrix[np.arange(start), np.arange(start)] = 1
        matrix[np.ix_(np.arange(start, start + transform.shape[0]), np.arange(start, stop))] = transform
        matrix[np.arange(size - stop, size), np.arange(size - stop, size)] = 1
    else:
        matrix = sp.block_diag([sp.identity(start, dtype=transform.dtype, format='csr'),
                                sp.csr_matrix(transform),
                                sp.identity(size - stop, dtype=transform.dtype, format='csr')], format='csr')
    return matrix


def _join_transform(cum_transform, transform, size=None):
    # Join two transforms (1d or 2d)
    if cum_transform is None:
        return transform
    else:
        if cum_transform.ndim < transform.ndim:
            cum_transform = _build_matrix_from_mask(cum_transform, size=size)
        elif cum_transform.ndim > transform.ndim:
            transform = _build_matrix_from_mask(transform, size=size)
        if cum_transform.ndim == 2:
            cum_transform = transform.dot(cum_transform)
        else:
            if np.issubdtype(transform.dtype, np.integer):
                assert np.issubdtype(cum_transform.dtype, np.integer)
                cum_transform = cum_transform[transform]
            else:
                cum_transform = cum_transform.copy()
                cum_transform[cum_transform] = transform
        return cum_transform


def _concatenate_transforms(transforms, starts, size):
    # WARNING: transforms are assumed disjoint
    assert len(transforms) == len(starts)
    is2d = any(transform.ndim == 2 for transform in transforms)
    if is2d:
        transforms = [_build_matrix_from_mask(transform, size=stop - start) if transform.ndim < 2 else transform for transform, (start, stop) in zip(transforms, starts)]
        try:
            import scipy.sparse as sp
        except ImportError:
            sp = None
        if sp is None:
            def _pad(transform, start, size):
                toret = np.zeros_like(transform, shape=(transform.shape[0], size))
                toret[:, start:start + transform.shape[1]] = transform
                return toret

            transforms = [_pad(transform, start[0], size) for start, transform in zip(starts, transforms)]
            matrix = np.concatenate(transforms, axis=0)
        else:
            def _pad(transform, start, size):
                m = [sp.csr_matrix((transform.shape[0], start)),
                     sp.csr_matrix(transform),
                     sp.csr_matrix((transform.shape[0], size - start - transform.shape[1]))]
                return sp.hstack(m)

            transforms = [_pad(transform, start[0], size) for start, transform in zip(starts, transforms)]
            matrix = sp.vstack(transforms)

        return matrix
    else:
        transforms = [np.flatnonzero(transform) if not np.issubdtype(transform.dtype, np.integer) else transform for transform in transforms]
        transforms = [start + transform for start, transform in zip(starts, transforms)]
        return np.concatenate(transforms, axis=0)


class _ObservableLeafUpdateRef(object):

    def __init__(self, observable, select=None, hook=None):
        self._observable = observable
        if select is None:
            self._limits = tuple((0, s) for s in self._observable.shape)
        else:
            if select[0] == '__getitem__':
                indices = _format_masks(self._observable.shape, select[1])
            else:
                kwargs = dict(select[1])
                center = kwargs.pop('center', 'mid_if_edges')
                limits = kwargs
                indices = []
                for axis in self._observable._coords_names:
                    transform = self._observable._transform(limits.pop(axis, None), axis=axis, center=center)
                    assert transform.ndim == 1, 'Only limits (min, max) are supported'
                    indices.append(transform)
            self._limits = tuple(find_single_true_slab_bounds(index) for index in indices)
        self._hook = hook
        assert len(self._limits) == self._observable.ndim

    def __getitem__(self, masks):
        """Select a section of the observable."""
        indices = _format_masks(self._observable.shape, masks)
        assert len(indices) == len(self._limits)
        indices = [start + index for start, index in zip(self._limits, indices)]
        new = self._observable[indices]
        if self._hook is not None:
            transform = np.ravel_multi_index(np.meshgrid(*indices, indexing='ij'), dims=self._observable.shape).ravel()
            return self._hook(new, transform=transform)
        return new

    def select(self, center='mid_if_edges', **limits):
        """Select a range in one or more coordinates."""
        new = self._observable.copy()
        cum_transform = None
        for iaxis, axis in enumerate(self._observable._coords_names):
            if axis not in limits: continue

            def _ravel_index(index, shape=None):
                indices = [np.arange(s) for s in shape]
                if isinstance(index, slice):
                    indices[iaxis] = np.arange(index.start, index.stop, index.step)
                else:
                    indices[iaxis] = index
                return np.ravel_multi_index(np.meshgrid(*indices, indexing='ij'), dims=shape).ravel()

            start, stop = self._limits[iaxis]
            sub = new.select(**{axis: slice(start, stop)})

            name = None
            if self._hook is not None: name = getattr(self._hook, 'weight', None)
            sub_transform = sub._transform(limit=limits[axis], axis=axis, center=center, full=True if self._hook is not None else None, name=name)
            sub = sub.select(**{axis: limits[axis]}, center=center)
            sub._data[axis] = np.concatenate([new.coords(axis)[:start], sub.coords(axis), new.coords(axis)[stop:]], axis=0)
            axis_edges = f'{axis}_edges'
            if axis_edges in new._data:
                sub._data[axis_edges] = np.concatenate([self._observable.edges(axis)[:start], sub.edges(axis), self._observable.edges(axis)[stop:]], axis=0)
            shape = tuple(len(sub._data[axis]) for axis in sub._coords_names)
            size = 1
            for s in shape: size *= s

            if self._hook is not None:
                if sub_transform.ndim == 1:
                    index1d = np.concatenate([np.arange(start), start + sub_transform, np.arange(new.shape[iaxis] - stop, new.shape[iaxis])], axis=0)
                    transform = _ravel_index(index1d, shape=new.shape)
                else:
                    if len(shape) == 1:
                        # Disjoint
                        transform = _concatenate_transforms([np.arange(start), sub_transform, np.arange(new.size - stop)], [(0, start), (start, stop), (stop, new.size)], new.size)
                    else:
                        # with hook
                        m1 = _build_matrix_from_mask(_ravel_index(slice(start, shape[iaxis] - (new.shape[iaxis] - stop)), shape=shape).ravel(), size=size)
                        m2 = _build_matrix_from_mask(_ravel_index(slice(start, stop), shape=new.shape).ravel(), size=new.size)
                        transform = m1.T.dot(sub_transform.dot(m2))
                        i1 = np.concatenate([np.arange(start), np.arange(stop, new.shape[iaxis])], axis=0)
                        i2 = np.concatenate([np.arange(start), np.arange(shape[iaxis] - (new.shape[iaxis] - stop), shape[iaxis])], axis=0)
                        transform += _build_matrix_from_mask2(_ravel_index(i2, shape=new.shape), _ravel_index(i1, shape=shape), shape=transform.shape)

            def put(value, index, array, axis=iaxis):
                indices = [np.arange(s) for s in new.shape]
                indices[axis] = index
                value[np.ix_(*indices)] = array

            for name in sub._values_names:
                tmp = _nan_to_zero(new._data[name])
                value = np.zeros_like(sub._data[name], shape=shape)
                i1 = np.concatenate([np.arange(start), np.arange(shape[iaxis] - (new.shape[iaxis] - stop), shape[iaxis])], axis=0)
                i2 = np.concatenate([np.arange(start), np.arange(stop, new.shape[iaxis])], axis=0)
                put(value, i1, np.take(tmp, i2, axis=iaxis), axis=iaxis)
                i1 = np.arange(start, shape[iaxis] - (new.shape[iaxis] - stop))
                put(value, i1, sub._data[name], axis=iaxis)
                sub._data[name] = value

            if self._hook is not None:
                cum_transform = _join_transform(cum_transform, transform, size=new.size)
            new = sub

        if self._hook is not None:
            return self._hook(new, transform=cum_transform)
        return new

    def match(self, observable):
        """Match coordinates to those of input observable."""
        return self.select(**{axis: observable for axis in self._observable._coords_names})


def _iter_on_tree(f, tree, level=None):
    """
    Recursively apply a function to all leaves of a tree structure.

    Parameters
    ----------
    f : callable
        Function to apply to each leaf.
    tree : object or list
        Observable tree(s).
    level : int, optional
        Level to apply function at. If `None`, apply to all leaves.

    Returns
    -------
    list
        List of results from applying f.
    """
    if isinstance(tree, list):
        if level == 0 or any(t._is_leaf for t in tree):
            return [f(tree)]
        toret = []
        for tree in zip(*[t._leaves for t in tree]):
            toret += _iter_on_tree(f, list(tree), level=level - 1 if level is not None else None)

    else:

        if level == 0 or tree._is_leaf:
            return [f(tree)]
        toret = []
        for tree in tree._leaves:
            toret += _iter_on_tree(f, tree, level=level - 1 if level is not None else None)

    return toret


def _get_leaf(tree, index=None):
    """
    Retrieve a leaf from a tree structure by index.

    Parameters
    ----------
    tree : object
        Tree structure of observables.
    index : tuple, optional
        Index tuple to locate the leaf.

    Returns
    -------
    object
        The leaf at the specified index.
    """
    if index is None:
        return tree
    toret = tree._leaves[index[0]]
    if len(index) == 1:
        return toret
    return _get_leaf(toret, index[1:])


def _format_input_labels(self, *args, **labels):
    """
    Format input labels for tree selection.

    Parameters
    ----------
    self : ObservableTree
        The tree object.
    *args : tuple
        Positional label arguments.
    **labels : dict
        Keyword label arguments.

    Returns
    -------
    dict
        Formatted label dictionary.
    """
    if args:
        assert not labels, 'Cannot provide both list and dict of labels'
        assert len(args) == 1 and len(self._labels) == 1, 'Args mode available only for one label entry'
        labels = {next(iter(self._labels)): args[0]}
    return labels


def _flatten_index_labels(indices):
    """
    Flatten nested index label dictionary into a list of index tuples.

    Parameters
    ----------
    indices : dict
        Nested index dictionary.

    Returns
    -------
    list of tuple
        Flattened list of index tuples.
    """
    toret = []
    for index, value in indices.items():
        if value is None:
            toret.append((index,))
        else:
            for flat_index in _flatten_index_labels(value):
                toret += [(index,) + flat_index]
    return toret


@register_type
class ObservableTree(object):
    """
    A collection of Observable objects, supporting selection, slicing, and labeling.
    """
    _name = 'tree_base'
    _forbidden_label_values = ('name', 'attrs', 'labels_names', 'labels_values')
    _sep_strlabels = '-'
    _is_leaf = False

    def __init__(self, leaves, attrs=None, meta=None, **labels):
        """
        Parameters
        ----------
        leaves : list of ObservableLeaf
            The leaves in the collection.
        labels : dict
            Label arrays (e.g. ell=[0, 2], observable=['spectrum',...]).
        """
        self._leaves = list(leaves)
        self._meta = dict(meta or {})
        self._attrs = dict(attrs or {})
        leaves_labels = []
        for leaf in self._leaves:
            if not leaf._is_leaf:
                leaves_labels += leaf.labels(keys_only=True)
        self._labels, self._strlabels = {}, {}
        nleaves = len(leaves)
        assert nleaves, 'At least one leaf must be provided'
        assert len(labels), 'At least one label must be provided'
        for k, v in labels.items():
            if isinstance(v, list):
                assert len(v) == nleaves, f'The length of labels must match that of leaves {nleaves:d}'
                self._labels[k] = v
            else:
                self._labels[k] = [v] * nleaves
            if k in leaves_labels:
                raise ValueError(f'Cannot use labels with same name at different levels: {k}')
            self._strlabels[k] = list(map(self._label_to_str, self._labels[k]))
            assert not any(v in self._forbidden_label_values for v in self._strlabels[k]), 'Cannot use "labels" as a label value'
            convert = list(map(self._str_to_label, self._strlabels[k]))
            assert convert == self._labels[k], f'Labels must be mappable to str; found label -> str -> label != identity:\n{convert} != {self._labels[k]}'
        uniques = []
        for ileaf in range(nleaves):
            labels = tuple(self._strlabels[k][ileaf] for k in self._strlabels)
            if labels in uniques:
                raise ValueError(f'Label {labels} is duplicated')
            uniques.append(labels)

    def _label_to_str(self, label):
        import numbers
        if isinstance(label, numbers.Number):
            return str(label)
        if isinstance(label, str):
            for char in ['_', self._sep_strlabels]:
                if char in label:
                    raise ValueError(f'Label cannot contain "{char}"')
            return label
        if isinstance(label, tuple):
            if len(label) == 1: raise ValueError('Tuples must be of length > 1')
            return '_'.join([self._label_to_str(lbl) for lbl in label])
        raise NotImplementedError(f'Unable to safely cast {label} to string. Implement "_label_to_str" and "_str_to_label".')

    def _str_to_label(self, str, squeeze=True):
        splits = list(str.split('_'))
        for i, split in enumerate(splits):
            try:
                splits[i] = int(split)
            except ValueError:
                pass
        if squeeze and len(splits) == 1:
            return splits[0]
        return tuple(splits)

    def _eq_label(self, label1, label2):
        # Compare input label label2 to self label1
        return label1 == label2

    def __repr__(self):
        return f'{self.__class__.__name__}(labels={self.labels(level=1)}, size={self.size})'

    @property
    def attrs(self):
        """Dictionary of attributes associated with the tree."""
        return self._attrs

    def labels(self, level=None, keys_only=False, as_str=False):
        """
        Return a list of dicts with the labels for each leaf.

        Parameters
        ----------
        level : int, optional
            Level to retrieve labels from. If `None`, retrieve all levels.
        keys_only : bool, optional
            If `True`, return only the list of unique keys (i.e. not label values) at the current level.
        as_str : bool, optional
            If `True`, return labels as strings.

        Returns
        -------
        labels : list or list of dict
        """
        toret = []
        if level == 0: return toret
        if keys_only:
            toret += [label for label in self._labels if label not in toret]
            for ileaf, leaf in enumerate(self._leaves):
                if leaf._is_leaf:
                    pass
                else:
                    for label in leaf.labels(level=level - 1 if level is not None else None, keys_only=keys_only, as_str=as_str):
                        if label not in toret: toret.append(label)
        else:
            for ileaf, leaf in enumerate(self._leaves):
                self_labels = {k: v[ileaf] for k, v in (self._strlabels if as_str else self._labels).items()}
                if level == 1 or leaf._is_leaf:
                    toret.append(self_labels)
                else:
                    for labels in leaf.labels(level=level - 1 if level is not None else None, keys_only=keys_only, as_str=as_str):
                        toret.append(self_labels | labels)

        return toret

    def __getattr__(self, name):
        if name in self._meta:
            return self._meta[name]
        if name in self._labels:
            return list(self._labels[name])
        raise AttributeError(name)

    def _index_labels(self, labels, flatten=True):
        labels = dict(labels)
        # Follows the original order
        def find(vselect, k):
            if isinstance(vselect, list):
                return sum((find(vs, k) for vs in vselect), start=[])
            if isinstance(vselect, str):
                return [i for i, v in enumerate(self._strlabels[k]) if v == vselect]
            return [i for i, v in enumerate(self._labels[k]) if self._eq_label(v, vselect)]

        self_index = list(range(len(self._leaves)))
        # First find labels in current level, keeping original order
        for k in self._labels:
            if k in labels:
                vselect = labels.pop(k)
                _indices = find(vselect, k)
                self_index = [index for index in self_index if index in _indices]
        if labels:  # remaining labels
            toret = {}
            for index in self_index:
                sub_index_labels = self._leaves[index]._index_labels(labels, flatten=False)
                if not sub_index_labels:
                    continue
                toret[index] = sub_index_labels

        else:
            toret = {index: None for index in self_index}
        if flatten:
            toret = _flatten_index_labels(toret)
        return toret

    def get(self, *args, **labels):
        """
        Return subtree or leaf corresponding to input labels.

        Parameters
        ----------
        *args : tuple
            Positional label arguments (only if one label entry).
        **labels : dict
            Keyword label arguments.

        Returns
        -------
        ObservableLeaf or ObservableTree
            The matching leaf or subtree.
        """
        labels = _format_input_labels(self, *args, **labels)
        isscalar = not any(isinstance(v, list) for v in labels.values())
        indices = self._index_labels(labels, flatten=False)
        if len(indices) == 0:
            raise ValueError(f'{labels} not found')

        if isscalar:
            flatten_indices = _flatten_index_labels(indices)
            if len(flatten_indices) == 1:
                return _get_leaf(self, flatten_indices[0])

        def get_subtree(tree, indices):
            leaves, ileaves = [], []
            for ileaf, leaf in enumerate(tree._leaves):
                if ileaf in indices:
                    if indices[ileaf] is not None:
                        leaf = get_subtree(tree, indices[ileaf])
                    leaves.append(leaf)
                    ileaves.append(ileaf)
            new = tree.copy()
            new._leaves = leaves
            new._labels = {k: [v[idx] for idx in ileaves] for k, v in tree._labels.items()}
            new._strlabels = {k: [v[idx] for idx in ileaves] for k, v in tree._strlabels.items()}
            return new

        return get_subtree(self, indices)

    def match(self, observable):
        """
        Match the the tree to the input observable, recursively, matching labels and leaves.

        Parameters
        ----------
        observable : ObservableTree
            Observable to match to.

        Returns
        -------
        ObservableTree
            New tree with leaves matched to input observable.
        """
        leaves, ileaves = [], []
        for ileaf, leaf in enumerate(observable._leaves):
            _ileaf = self._index_labels({k: v[ileaf] for k, v in observable._labels.items()}, flatten=True)
            assert len(_ileaf) == 1 and len(_ileaf[0]) == 1
            _ileaf = _ileaf[0][0]
            leaves.append(self._leaves[_ileaf].match(leaf))
            ileaves.append(_ileaf)
        new = self.copy()
        new._leaves = leaves
        new._labels = {k: [v[idx] for idx in ileaves] for k, v in self._labels.items()}
        new._strlabels = {k: [v[idx] for idx in ileaves] for k, v in self._strlabels.items()}
        return new

    def sizes(self, level=None):
        """Size of each leaf at given level."""
        return _iter_on_tree(lambda leaf: leaf.size, self, level=level)

    @property
    def size(self):
        """Total size of the tree."""
        return sum(self.sizes(level=1))

    def __iter__(self):
        """Iterate over leaves."""
        return iter(self._leaves)

    def select(self, **limits):
        """
        Select a range in one or more coordinates, applied to leaves with matching coordinate names.

        Parameters
        ----------
        limits : dict
            Each key is a coordinate name, value is either:
            - (min, max) tuple for this coordinate
            - slice for this coordinate
            - array-like of coordinates or edges to select

        center : str, optional, default='mid_if_edges'
            How to compute the coordinate values if edges are provided:
            - 'mid': mean of edges
            - 'mid_if_edges': 'mid' if edges are provided, else use coordinates as is
            - `None`: use coordinates as is

        Returns
        -------
        ObservableTree
            New tree with selected leaves.
        """
        leaves = []
        notfound = set(limits)

        def get_coords(leaf):
            return sum(_iter_on_tree(lambda leaf: tuple(leaf._coords_names), leaf, level=None), start=tuple())

        for leaf in self._leaves:
            _all_coord_names = get_coords(leaf)
            _ranges = {k: v for k, v in limits.items() if k in _all_coord_names}
            notfound -= set(_ranges)
            leaves.append(leaf.select(**_ranges))

        new = self.copy()
        new._leaves = leaves
        return new

    def value(self, concatenate=True):
        """
        Get (flattened) value from all leaves.

        Parameters
        ----------
        concatenate : bool, optional
            If True, concatenate along first axis.

        Returns
        -------
        value : list or array
        """
        def get_value(leaf):
            return _iter_on_tree(lambda leaf: leaf.value(), leaf, level=None)

        values = get_value(self)
        values = [value.ravel() for value in values]
        if concatenate:
            return np.concatenate(values, axis=0)
        return values

    def __array__(self):
        return self.value(concatenate=True)

    def clone(self, value=None):
        """Return a copy of the tree, optionally with updated (main) `value`."""
        new = self.copy()
        if value is None:
            return new
        start = 0
        for ileaf, leaf in enumerate(new._leaves):
            stop = start + leaf.size
            new._leaves[ileaf] = leaf.clone(value=value[start:stop])
            start = stop
        return new

    def __add__(self, other):
        return self.sum([self, other])

    def __radd__(self, other):
        if other == 0: return self.copy()
        return self.__add__(other)

    def __iadd__(self, other):
        if other == 0: return self.copy()
        return self.__add__(other)

    @classmethod
    def sum(cls, observables):
        """Sum multiple observables."""
        new = observables[0].copy()
        for ileaf, leaf in enumerate(new._leaves):
            labels = {k: v[ileaf] for k, v in new._labels.items()}
            new._leaves[ileaf] = leaf.sum([observable.get(**labels) for observable in observables])
        return new

    @classmethod
    def mean(cls, observables):
        """Mean of multiple observables."""
        new = observables[0].copy()
        for ileaf, leaf in enumerate(new._leaves):
            labels = {k: v[ileaf] for k, v in new._labels.items()}
            new._leaves[ileaf] = leaf.mean([observable.get(**labels) for observable in observables])
        return new

    @classmethod
    def cov(cls, observables):
        """Covariance matrix of multiple observables."""
        mean = cls.mean(observables)
        value = np.cov([observable.value() for observable in observables], rowvar=False, ddof=1)
        return CovarianceMatrix(value, observable=mean)

    @classmethod
    def join(cls, observables):
        """Join multiple trees into a single one."""
        leaves, labels = [], {label: [] for label in observables[0]._labels}
        for observable in observables:
            assert isinstance(observable, ObservableTree)
            assert set(observable._labels) == set(labels), 'All collections must have same labels'
            leaves += observable._leaves
            for k in labels:
                labels[k] = labels[k] + observable._labels[k]
        new = observables[0].copy()
        ObservableTree.__init__(new, leaves, **labels)  # check labels
        return new

    @property
    def at(self):
        """Helper to select or slice the tree in-place."""
        return _ObservableTreeUpdateHelper(self)

    def copy(self):
        """Return a copy of the tree (arrays not copied)."""
        new = self.__class__.__new__(self.__class__)
        new.__setstate__(self.__getstate__())
        return new

    def __getstate__(self, to_file=False):
        state = {}
        if not to_file:
            state['leaves'] = [leaf.__getstate__() for leaf in self._leaves]
            state['labels'] = dict(self._labels)
            state['strlabels'] = dict(self._strlabels)
        else:
            state['labels_names'] = self._sep_strlabels.join(list(self._labels.keys()))
            state['labels_values'] = []
            for ileaf, leaf in enumerate(self._leaves):
                label = self._sep_strlabels.join([self._strlabels[k][ileaf] for k in self._labels])
                state['labels_values'].append(label)
                state[label] = leaf.__getstate__(to_file=to_file)
        if self._meta: state['meta'] = dict(self._meta)
        if self._attrs: state['attrs'] = dict(self._attrs)
        state['name'] = self._name
        return state

    def __setstate__(self, state):
        self._meta = state.get('meta', {})
        self._attrs = state.get('attrs', {})
        if 'leaves' in state:
            leaves = state['leaves']
            self._leaves = [from_state(leaf) for leaf in leaves]
            self._labels = state['labels']
            self._strlabels = state['strlabels']
        else:  # h5py format
            label_names = np.array(state['labels_names']).item().split(self._sep_strlabels)
            label_values = list(map(lambda x: x.split(self._sep_strlabels), np.array(state['labels_values'])))
            self._labels, self._strlabels = {}, {}
            for i, name in enumerate(label_names):
                self._strlabels[name] = [v[i] for v in label_values]
                self._labels[name] = [self._str_to_label(s, squeeze=True) for s in self._strlabels[name]]
            nleaves = len(state['labels_values'])
            self._leaves = []
            for ileaf in range(nleaves):
                label = state['labels_values'][ileaf]
                self._leaves.append(from_state(state[label]))

    def __eq__(self, other):
        return deep_eq(self.__getstate__(), other.__getstate__())

    def write(self, filename):
        """
        Write observable to disk.

        Parameters
        ----------
        filename : str
            Output file name.
        """
        return write(filename, self)


class _ObservableTreeUpdateHelper(object):

    _tree: ObservableTree

    def __init__(self, tree, hook=None):
        self._tree = tree
        self._hook = hook

    def __call__(self, *args, **labels):
        """Select subtree or leaf corresponding to input labels."""
        labels = _format_input_labels(self._tree, *args, **labels)
        indices = self._tree._index_labels(labels)
        assert len(indices), f'Nothing found with {labels}'
        # Sub-tree
        return _ObservableTreeUpdateRef(self._tree, indices, hook=self._hook)


def _get_range_in_tree(tree, index):
    # Get start/stop in the full tree for a given index (leaf)
    start = 0
    current_tree = tree
    for idx in index:
        start += sum(leaf.size for leaf in current_tree._leaves[:idx])
        current_tree = current_tree._leaves[idx]
    return start, start + current_tree.size


def _replace_in_tree(tree, index, sub):
    # Replace a leaf in the tree, return start/stop of replaced leaf
    start = 0
    current_tree = tree
    for idx in index[:-1]:
        start += sum(leaf.size for leaf in current_tree._leaves[:idx])
        current_tree = current_tree._leaves[idx]
    start += sum(leaf.size for leaf in current_tree._leaves[:index[-1]])
    stop = start + current_tree._leaves[index[-1]].size
    current_tree._leaves[index[-1]] = sub
    return start, stop


class _ObservableTreeUpdateRef(object):

    def __init__(self, tree, indices=None, select=None, hook=None):
        self._tree = tree
        self._select = select
        self._indices = indices
        self._hook = hook

    def get(self, *args, **labels):
        """Return subtree or leaf corresponding to input labels."""
        # Order is preserved
        new = self._tree.copy()
        if self._hook is not None:
            mask = np.ones(self._tree.size, dtype='?')
        for index in (self._indices if self._indices is not None else [None]):
            leaf = _get_leaf(self._tree, index)
            _labels = _format_input_labels(leaf, *args, **labels)
            sub = leaf.get(**_labels)
            if index is None:
                new = sub
                start, stop = 0, leaf.size
            else:
                start, stop = _replace_in_tree(new, index, sub)
            if self._hook is not None:
                _mask = np.zeros(leaf.size, dtype='?')
                for _index in leaf._index_labels(_labels):
                    _mask[slice(*_get_range_in_tree(leaf, _index))] = True
                mask[start:stop] = _mask
        if self._hook is not None:
            return self._hook(new, transform=np.flatnonzero(mask))
        return new

    def __getitem__(self, masks):
        """Select a section of the tree."""
        hook = None
        if self._hook:
            def hook(leaf, transform): return leaf, transform
            hook.weight = getattr(self._hook, 'weight', None)
        new = self.copy()
        transform = None
        for index in (self._indices if self._indices is not None else self._tree._index_labels({})):
            leaf = _get_leaf(self._tree, index)
            leaf = _get_update_ref(leaf)(leaf, select=self._select, hook=hook).__getitem__(masks)
            size = new.size
            if self._hook:
                leaf, _transform = leaf
            size = new.size
            start, stop = _replace_in_tree(new, index, leaf)
            if self._hook:
                _transform = _pad_transform(_transform, start, size=size)
                transform = _join_transform(transform, _transform, size=size)

        if self._hook:
            return self._hook(new, transform=transform)
        return new

    def select(self, **limits):
        """Select a range in one or more coordinates."""
        hook = None
        if self._hook:
            def hook(leaf, transform): return leaf, transform
            hook.weight = getattr(self._hook, 'weight', None)

        new = self._tree.copy()
        transform = None
        for index in (self._indices if self._indices is not None else self._tree._index_labels({})):
            leaf = _get_leaf(self._tree, index)
            leaf = _get_update_ref(leaf)(leaf, select=self._select, hook=hook).select(**limits)
            if self._hook:
                leaf, _transform = leaf
            size = new.size
            start, stop = _replace_in_tree(new, index, leaf)
            if self._hook:
                _transform = _pad_transform(_transform, start, size=size)
                transform = _join_transform(transform, _transform, size=size)

        if self._hook:
            return self._hook(new, transform=transform)
        return new

    def match(self, observable):
        """Match coordinates to those of input tree."""
        hook = None
        if self._hook:
            def hook(leaf, transform): return leaf, transform
            hook.weight = getattr(self._hook, 'weight', None)

        if self._indices is not None:
            assert len(self._indices) == 1, 'match can only be applied to a tree or leaf'
            index = self._indices[0]
        else:
            index = None
        tree = _get_leaf(self._tree, index)
        transforms, starts = [], []
        leaves, ileaves = [], []
        for ileaf, leaf in enumerate(observable._leaves):
            _ileaf = tree._index_labels({k: v[ileaf] for k, v in observable._labels.items()}, flatten=True)
            assert len(_ileaf) == 1 and len(_ileaf[0]) == 1
            _ileaf = _ileaf[0][0]
            _leaf = tree._leaves[_ileaf]
            leaf = _get_update_ref(_leaf)(_leaf, select=self._select, hook=hook).match(leaf)
            if self._hook:
                leaf, _transform = leaf
            leaves.append(leaf)
            ileaves.append(_ileaf)
            if self._hook:
                start, stop = _get_range_in_tree(tree, (_ileaf,))
                transforms.append(_transform)
                starts.append((start, stop))

        if self._hook:
            transform = _concatenate_transforms(transforms, starts, size=tree.size)
        tree = tree.copy()
        tree._leaves = leaves
        tree._labels = {k: [v[idx] for idx in ileaves] for k, v in tree._labels.items()}
        tree._strlabels = {k: [v[idx] for idx in ileaves] for k, v in tree._strlabels.items()}
        new = tree
        if index is not None:
            new = self._tree.copy()
            start, stop = _replace_in_tree(new, (index,), tree)
            if self._hook:
                transform = _pad_transform(transform, start, size=self._tree.size)
        if self._hook:
            return self._hook(new, transform=transform)
        return new

    @property
    def at(self):
        """Helper to select or slice the tree or leaf in-place."""
        if self._indices is not None:
            assert len(self._indices) == 1, 'at can only be applied to a tree or leaf'
            index = self._indices[0]
        else:
            index = None
        at = _get_leaf(self._tree, index).at

        def hook(leaf, transform=None):
            if index is not None:
                new = self._tree.copy()
                start, stop = _replace_in_tree(new, index, leaf)
                if self._hook is not None:
                    transform = _pad_transform(transform, start, size=self._tree.size)
            if self._hook is not None:
                return self._hook(new, transform=transform)
            return new

        if self._hook is not None:
            hook.weight = getattr(self._hook, 'weight', None)
        at._hook = hook
        at._select = self._select
        return at


@register_type
class LeafLikeObservableTree(ObservableTree):

    """A collection of homogeneous observables, supporting selection, slicing, and labeling."""

    _name = 'leaf_like_tree_base'
    _is_leaf = True

    @property
    def _coords_names(self):
        return self._leaves[0]._coords_names

    @property
    def _values_names(self):
        return self._leaves[0]._values_names

    @property
    def shape(self):
        """Observable shape."""
        return self._leaves[0].shape

    @property
    def size(self):
        """Observable size."""
        return self._leaves[0].size

    def edges(self, *args, **kwargs):
        """Observable edges."""
        return self._leaves[0].edges(*args, **kwargs)

    def coords(self, *args, **kwargs):
        """Observable coordinates."""
        return self._leaves[0].coords(*args, **kwargs)

    def __getitem__(self, masks):
        indices = _format_masks(self.shape, masks)
        new = self.copy()
        for ileaf, leaf in new._leaves:
            new._leaves[ileaf] = leaf.__getitem__(indices)
        return new

    @property
    def at(self):
        """Helper to select or slice the tree in-place."""
        return _LeafLikeObservableTreeHelper(self)

    def value(self):
        """Main value of the observable."""
        raise NotImplementedError

    @classmethod
    def _average(cls, observables, weights=None):
        # Average multiple observables
        new = observables[0].copy()
        for ileaf, leaf in enumerate(new._leaves):
            labels = {k: v[ileaf] for k, v in new._labels.items()}
            new._leaves[ileaf] = leaf._average([observable.get(**labels) for observable in observables], weights=weights)
        return new

    @classmethod
    def sum(cls, observables):
        """Sum multiple observables."""
        return cls._average(observables, weights=getattr(cls, '_sumweight', None))

    @classmethod
    def mean(cls, observables):
        """Mean of multiple observables."""
        return cls._average(observables, weights=getattr(cls, '_meanweight', None))


def _get_update_ref(observable):
    if isinstance(observable, ObservableLeaf):
        return _ObservableLeafUpdateRef
    if isinstance(observable, LeafLikeObservableTree):
        return _LeafLikeObservableTreeUpdateRef
    if isinstance(observable, ObservableTree):
        return _ObservableTreeUpdateRef


class _LeafLikeObservableTreeHelper(object):

    def __init__(self, tree, hook=None):
        self._tree = tree
        self._hook = hook

    def __getitem__(self, masks):
        """Select a section of the observable."""
        select = ('__getitem__', masks)
        return _LeafLikeObservableTreeUpdateRef(self._tree, select, self._hook)

    def __call__(self, **kwargs):
        """Select a range in one or more coordinates."""
        select = ('__select__', kwargs)
        return _LeafLikeObservableTreeUpdateRef(self._tree, select, self._hook)


class _LeafLikeObservableTreeUpdateRef(object):

    def __init__(self, tree, select, hook=None):
        self._tree = tree
        self._select = select
        self._hook = hook

    def __getitem__(self, masks):
        """Select a section of the observable."""
        self._indices = None
        return _ObservableTreeUpdateRef.__getitem__(self, masks)

    def select(self, **limits):
        """Select a range in one or more coordinates."""
        self._indices = None
        return _ObservableTreeUpdateRef.select(self, **limits)

    def match(self, observable):
        """Match coordinates to those of input observable."""
        self._indices = None
        return _ObservableTreeUpdateRef.match(self, observable)


class _WindowMatrixUpdateHelper(object):

    def __init__(self, matrix):
        self._matrix = matrix

    @property
    def observable(self):
        """Helper to select or slice the observable side of the matrix in-place."""
        return _ObservableWindowMatrixUpdateHelper(self._matrix, axis=0)

    @property
    def theory(self):
        """Helper to select or slice the theory side of the matrix in-place."""
        return _ObservableWindowMatrixUpdateHelper(self._matrix, axis=1)


class _ObservableWindowMatrixUpdateHelper(object):

    def __init__(self, matrix, axis=0):
        self._matrix = matrix
        self._axis = axis
        self._observable = [matrix._observable, matrix._theory][self._axis]
        self._weight = None if self._axis == 0 else (False, False)  # no weight, not normalized

    def _select(self, observable, transform):
        _observable_name = ['observable', 'theory'][self._axis]
        if transform.ndim == 1:  # mask
            value = np.take(self._matrix.value(), transform, axis=self._axis)
        else:
            if self._axis == 0: value = transform.dot(self._matrix.value())
            else: value = transform.dot(self._matrix.value().T).T  # because transform can be a sparse matrix; works in all cases
        kw = {_observable_name: observable}
        return self._matrix.clone(value=value, **kw)

    def match(self, observable):
        """Match matrix coordinates to those of input observable."""
        def hook(observable, transform):
            return observable, transform
        hook.weight = self._weight
        observable, transform =  _get_update_ref(self._observable)(self._observable, hook=hook).match(observable)
        return self._select(observable, transform=transform)

    def select(self, **limits):
        """Select a range in one or more coordinates."""
        def hook(observable, transform):
            return observable, transform
        hook.weight = self._weight
        observable, transform = _get_update_ref(self._observable)(self._observable, hook=hook).select(**limits)
        return self._select(observable, transform=transform)

    def get(self, *args, **labels):
        """Return a matrix with observable selected given input labels."""
        assert isinstance(self._observable, ObservableTree), 'get only applies to a tree'
        def hook(observable, transform):
            return observable, transform
        hook.weight = self._weight
        observable, transform = _get_update_ref(self._observable)(self._observable, hook=hook).get(*args, **labels)
        return self._select(observable, transform=transform)

    @property
    def at(self):
        """Helper to select or slice the matrix in-place."""
        def hook(sub, transform=None):
            return self._select(sub, transform=transform)
        hook.weight = self._weight
        return self._observable.at.__class__(self._observable, hook=hook)


@register_type
class WindowMatrix(object):

    """A window matrix, with associated observable and theory."""

    _name = 'windowmatrix'

    def __init__(self, value, observable, theory):
        self._value = value
        self._observable = observable
        self._theory = theory

    @property
    def shape(self):
        """Matrix shape."""
        return self._value.shape

    def value(self):
        """Value (numpy array) of the window matrix."""
        return self._value

    def __array__(self):
        return self.value()

    @property
    def observable(self):
        """Observable side of the window matrix."""
        return self._observable

    @property
    def theory(self):
        """Theory side of the window matrix."""
        return self._theory

    @property
    def at(self):
        """Helper to select or slice the matrix in-place."""
        return _WindowMatrixUpdateHelper(self)

    def copy(self):
        """Return a copy of the window matrix (arrays not copied)."""
        new = self.__class__.__new__(self.__class__)
        new.__setstate__(self.__getstate__())
        return new

    def clone(self, **kwargs):
        """
        Copy and update data in the :class:`WindowMatrix` instance.

        Parameters
        ----------
        **kwargs : dict
            Attributes to update in the cloned instance.

        Returns
        -------
        WindowMatrix
            Cloned and updated instance.
        """
        new = self.copy()
        for name, value in kwargs.items():
            if name in ['observable', 'theory', 'value']:
                setattr(new, '_' + name, value)
        return new

    def __getstate__(self, to_file=False):
        state = {}
        state['name'] = self._name
        state['value'] = self.value()
        for name in ['observable', 'theory']:
            state[name] = getattr(self, name).__getstate__(to_file=to_file)
        return state

    def __setstate__(self, state):
        self._value = state['value']
        for name in ['observable', 'theory']:
            setattr(self, '_' + name, from_state(state[name]))
        return state

    def __eq__(self, other):
        return deep_eq(self.__getstate__(), other.__getstate__())

    def __add__(self, other):
        return self.sum([self, other])

    def __radd__(self, other):
        if other == 0: return self.copy()
        return self.__add__(other)

    def __iadd__(self, other):
        if other == 0: return self.copy()
        return self.__add__(other)

    @classmethod
    def sum(cls, matrices):
        """Sum multiple window matrices."""
        new = matrices[0].copy()

        def get_sumweight(leaves):
            sw = getattr(leaves[0], '_sumweight', None)
            weight = None
            if sw is not None:
                weight = sw(leaves)
            if weight is None:
                weight = [np.ones_like(leaves[0].size) / len(leaves)] * len(leaves)
            return weight

        weights = _iter_on_tree(get_sumweight, [matrix.observable for matrix in matrices])
        weights = [np.concatenate(w, axis=0) for w in zip(*weights)]
        new._value = sum(weight[..., None] * matrix._value for matrix, weight in zip(matrices, weights))
        new._observable = matrices[0]._observable.sum([matrix.observable for matrix in matrices])

        return new

    def dot(self, theory, zpt=False, return_type='nparray'):
        """
        Apply window matrix to theory.

        Parameters
        ----------
        theory : array-like or ObservableLeaf or ObservableTree
            Theory to apply window matrix to.
        zpt : bool, default=False
            If True, remove the zero-point theory :attr:`theory` before applying window matrix,
            and then add back the zero-point :attr:`observable` value.
        return_type : {'nparray', None}, default='nparray'
            If 'nparray', return numpy array; if None, return observable.

        Returns
        -------
        array or ObservableLeaf or ObservableTree
            Result of applying window matrix to theory.
        """
        self._cache = _cache = getattr(self, '_cache', {})
        if 'observablev' not in _cache: _cache['observablev'] = self._observable.value()
        if 'theoryv' not in _cache: _cache['theoryv'] = self._theory.value()

        if isinstance(theory, (ObservableLeaf, ObservableTree)):
            theory = theory.value()
        if zpt:
            diff = theory - _cache['theoryv']
            toret = _cache['observablev'] + self._value.dot(diff)
        else:
            toret = self._value.dot(theory)
        if return_type is None:
            return self._observable.clone(value=toret)
        return toret

    def write(self, filename):
        """
        Write window matrix to disk.

        Parameters
        ----------
        filename : str
            Output file name.
        """
        return write(filename, self)

    @utils.plotter
    def plot(self, level=None, **kwargs):
        """
        Plot window matrix.

        Parameters
        ----------
        barlabel : str, default=None
            Optionally, label for the color bar.

        figsize : int, tuple, default=None
            Optionally, figure size.

        norm : matplotlib.colors.Normalize, default=None
            Scales the matrix to the canonical colormap range [0, 1] for mapping to colors.
            By default, the matrix range is mapped to the color bar range using linear scaling.

        labelsize : int, default=None
            Optionally, size for labels.

        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``len(self._observables) * len(self._observables)`` axes.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        def _get(axis):
            xlabels, labels, x, indices = [], [], [], []
            observable = self.observable if axis == 0 else self.theory

            def get_x(leaf):
                xx = None
                if len(leaf._coords_names) == 1:
                    xx = leaf.coords(axis=leaf._coords_names[0])
                    if xx.ndim > 1: xx = None
                if xx is None:
                    xx = np.arange(leaf.size)
                return xx

            if observable._is_leaf:
                x.append(get_x(observable))
                indices.append(np.arange(observable.size))

            for label in observable.labels(level=level):
                leaf = observable.get(**label)
                x.append(get_x(leaf))
                start, stop = _get_range_in_tree(observable, observable._index_labels(label)[0])
                indices.append(np.arange(start, stop))

            return xlabels, labels, x, indices

        xlabels, labels, x, indices = zip(*[_get(axis) for axis in [0, 1]])

        if level == 0:
            indices = [np.concatenate(index) for index in indices]
            x = [[np.concatenate(xx, axis=0) for xx in x]]
            xlabels = [label[0] for label in xlabels]
            labels = []
        for ilabel, label in enumerate(xlabels):
            if label: kwargs.setdefault(f'xlabel{ilabel + 1:d}', label)
        for ilabel, label in enumerate(labels):
            if label: kwargs.setdefault(f'label{ilabel + 1:d}', label)
        mat = [[self._value[np.ix_(index1, index2)] for index2 in indices[1]] for index1 in indices[0]]
        return utils.plot_matrix(mat, x1=x[0], x2=x[1], **kwargs)



class _CovarianceMatrixUpdateHelper(object):

    def __init__(self, matrix):
        self._matrix = matrix

    @property
    def observable(self):
        """Helper to select or slice the covariance matrix in-place."""
        return _ObservableCovarianceMatrixUpdateHelper(self._matrix)


class _ObservableCovarianceMatrixUpdateHelper(_ObservableWindowMatrixUpdateHelper):

    def __init__(self, matrix):
        self._matrix = matrix
        self._observable = matrix._observable
        self._weight = None

    def _select(self, observable, transform):
        if transform.ndim == 1:  # mask
            value = self._matrix.value()[np.ix_(transform, transform)]
        else:
            value = transform.dot(self._matrix.value())
            value = transform.dot(value.T).T  # because transform can be a sparse matrix; works in all cases
        return self._matrix.clone(value=value, observable=observable)


@register_type
class CovarianceMatrix(object):

    """A covariance matrix, with associated observable."""

    _name = 'covariancematrix'

    def __init__(self, value, observable):
        self._value = value
        self._observable = observable

    @property
    def shape(self):
        """Matrix shape."""
        return self._value.shape

    def value(self):
        """Value (numpy array) of the covariance matrix."""
        return self._value

    def __array__(self):
        return self.value()

    @property
    def observable(self):
        """Observable corresponding to the covariance matrix."""
        return self._observable

    @property
    def at(self):
        """Helper to select or slice the matrix in-place."""
        return _CovarianceMatrixUpdateHelper(self)

    def std(self):
        """Standard deviation."""
        std = np.sqrt(np.diag(self._value))
        return std

    def corrcoef(self):
        """Correlation coefficient matrix."""
        std = self.std()
        corrcoef = self._value / (std[..., None] * std)
        return corrcoef

    def inv(self):
        """Inverse of the covariance matrix."""
        # FIXME
        return np.linalg.inv(self._value)

    def copy(self):
        """Return a copy of the covariance matrix (arrays not copied)."""
        new = self.__class__.__new__(self.__class__)
        new.__setstate__(self.__getstate__())
        return new

    def clone(self, **kwargs):
        """Copy and update data."""
        new = self.copy()
        for name, value in kwargs.items():
            if name in ['observable', 'value']:
                setattr(new, '_' + name, value)
        return new

    def __getstate__(self, to_file=False):
        state = {}
        state['name'] = self._name
        state['value'] = self.value()
        for name in ['observable']:
            state[name] = getattr(self, name).__getstate__(to_file=to_file)
        return state

    def __setstate__(self, state):
        self._value = state['value']
        for name in ['observable']:
            setattr(self, '_' + name, from_state(state[name]))

    def __eq__(self, other):
        return deep_eq(self.__getstate__(), other.__getstate__())

    def write(self, filename):
        """
        Write covariance matrix to disk.

        Parameters
        ----------
        filename : str
            Output file name.
        """
        return write(filename, self)

    @utils.plotter
    def plot(self, level=None, **kwargs):
        """
        Plot covariance matrix.

        Parameters
        ----------
        barlabel : str, default=None
            Optionally, label for the color bar.

        figsize : int, tuple, default=None
            Optionally, figure size.

        norm : matplotlib.colors.Normalize, default=None
            Scales the matrix to the canonical colormap range [0, 1] for mapping to colors.
            By default, the matrix range is mapped to the color bar range using linear scaling.

        labelsize : int, default=None
            Optionally, size for labels.

        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``len(self._observables) * len(self._observables)`` axes.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        def _get():
            xlabels, labels, x, indices = [], [], [], []
            observable = self.observable

            def get_x(leaf):
                xx = None
                if len(leaf._coords_names) == 1:
                    xx = leaf.coords(axis=leaf._coords_names[0])
                    if xx.ndim > 1: xx = None
                if xx is None:
                    xx = np.arange(leaf.size)
                return xx

            if observable._is_leaf:
                x.append(get_x(observable))
                indices.append(np.arange(observable.size))
            else:
                for label in observable.labels(level=level):
                    leaf = observable.get(**label)
                    x.append(get_x(leaf))
                    start, stop = _get_range_in_tree(observable, observable._index_labels(label)[0])
                    indices.append(np.arange(start, stop))

            return xlabels, labels, x, indices

        xlabels, labels, x, indices = zip(*[_get()] * 2)

        if level == 0:
            indices = [np.concatenate(index) for index in indices]
            x = [[np.concatenate(xx, axis=0) for xx in x]]
            xlabels = [label[0] for label in xlabels]
            labels = []
        for ilabel, label in enumerate(xlabels):
            if label: kwargs.setdefault(f'xlabel{ilabel + 1:d}', label)
        for ilabel, label in enumerate(labels):
            if label: kwargs.setdefault(f'label{ilabel + 1:d}', label)
        mat = [[self._value[np.ix_(index1, index2)] for index2 in indices[1]] for index1 in indices[0]]
        return utils.plot_matrix(mat, x1=x[0], x2=x[1], **kwargs)


class _GaussianLikelihoodUpdateHelper(object):

    def __init__(self, likelihood):
        self._likelihood = likelihood

    @property
    def observable(self):
        """Helper to select or slice the observable side of the likelihood in-place."""
        return _ObservableGaussianLikelihoodUpdateHelper(self._likelihood, axis=0)

    @property
    def theory(self):
        """Helper to select or slice the theory side of the likelihood in-place."""
        return _ObservableGaussianLikelihoodUpdateHelper(self._likelihood, axis=1)


class _ObservableGaussianLikelihoodUpdateHelper(object):

    def __init__(self, likelihood, axis=0):
        self._likelihood = likelihood
        self._axis = axis
        self._observable_name = ['observable', 'theory'][self._axis]

    def _select(self, observable):
        if self._axis == 0:
            covariance = self._likelihood.covariance.at.observable.match(observable)
            window = self._likelihood.window.at.observable.match(observable)
            observable = self._likelihood.observable.match(observable)
            return self._likelihood.clone(observable=observable, window=window, covariance=covariance)
        window = self._likelihood.window.at.theory.match(observable)
        return self._likelihood.clone(window=window)

    def match(self, observable):
        """Match likelihood coordinates to those of input observable."""
        return self._select(observable)

    def select(self, **limits):
        """Select a range in one or more coordinates."""
        if self._axis == 0:
            observable = self._likelihood.observable.select(**limits)
        else:
            observable = self._likelihood.window.theory.select(**limits)
        return self._select(observable)

    def get(self, *args, **labels):
        """Return a likelihood with observable selected given input labels."""
        if self._axis == 0:
            observable = self._likelihood.observable.get(*args, **labels)
        else:
            observable = self._likelihood.window.theory.get(*args, **labels)
        return self._select(observable)

    @property
    def at(self):
        """Helper to select or slice the likelihood in-place."""
        def hook(sub, transform=None):
            return self._select(sub)
        if self._axis == 0:
            observable = self._likelihood.observable
        else:
            observable = self._likelihood.window.theory
        return observable.at.__class__(observable, hook=hook)


@register_type
class GaussianLikelihood(object):

    """A Gaussian likelihood, with associated observable, window matrix, and covariance matrix."""

    _name = 'gaussianlikelihood'

    def __init__(self, observable, window, covariance):
        self._observable = observable
        self._window = window
        self._covariance = covariance

    @property
    def observable(self):
        """Observable corresponding to the likelihood."""
        return self._observable

    @property
    def window(self):
        """Window matrix corresponding to the likelihood."""
        return self._window

    @property
    def covariance(self):
        """Covariance matrix corresponding to the likelihood."""
        return self._covariance

    @property
    def at(self):
        """Helper to select or slice the likelihood in-place."""
        return _GaussianLikelihoodUpdateHelper(self)

    def chi2(self, theory):
        r"""
        Compute :math:'\chi^2` for input theory.

        Parameters
        ----------
        theory : array-like or ObservableLeaf or ObservableTree
            (Unwindowed) theory to compare to the observable.

        Returns
        -------
        float
            :math:'\chi^2` value.
        """
        if isinstance(theory, (ObservableLeaf, ObservableTree)):
            theory = theory.value()
        self._cache = _cache = getattr(self, '_cache', {})
        if 'observablev' not in _cache: _cache['observablev'] = self._observable.value()
        if 'covinv' not in _cache: _cache['covinv'] = self._covariance.inv()
        diff = _cache['observablev'] - self._window.dot(theory)
        return diff.T.dot(_cache['covinv']).dot(diff)

    def copy(self):
        new = self.__class__.__new__(self.__class__)
        new.__setstate__(self.__getstate__())
        return new

    def clone(self, **kwargs):
        """Copy and update likelihood."""
        new = self.copy()
        for name, value in kwargs.items():
            if name in ['observable', 'window', 'covariance']:
                setattr(new, '_' + name, value)
        return new

    def __getstate__(self, to_file=False):
        state = {}
        state['name'] = self._name
        for name in ['observable', 'window', 'covariance']:
            state[name] = getattr(self, name).__getstate__(to_file=to_file)
        return state

    def __setstate__(self, state):
        for name in ['observable', 'window', 'covariance']:
            setattr(self, '_' + name, from_state(state[name]))
        return state

    def __eq__(self, other):
        return deep_eq(self.__getstate__(), other.__getstate__())

    def write(self, filename):
        """
        Write likelihood to disk.

        Parameters
        ----------
        filename : str
            Output file name.
        """
        return write(filename, self)