import os
import shutil
from functools import partial

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
    elif np.issubdtype(array.dtype, np.complexfloating):
        return '%.18e+%.18ej'
    elif np.issubdtype(array.dtype, np.str_):
        maxlen = array.dtype.itemsize // 4  # 4 bytes per unicode char
        return f'%{maxlen}s'
    elif np.issubdtype(array.dtype, np.bytes_):
        maxlen = array.dtype.itemsize
        return f'%{maxlen}s'
    else:
        raise TypeError(f"Unsupported dtype: {array.dtype}")


import json

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.complexfloating,)):
            return complex(obj)
        return super().default(obj)



def _txt_recursively_write_dict(path, dic, with_attrs=True):
    """Save a nested dictionary of arrays to an HDF5 file using h5py."""
    utils.mkdir(path)
    for key, item in dic.items():
        path_key = os.path.join(path, key)
        if with_attrs and key == 'attrs':
            with open(path_key + '.json', 'w') as file:
                json.dump(item, file, cls=NumpyEncoder)
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
        try:  # backward-compatibility
            __registry = {name.replace('_', ''): value for name, value in _registry.items()}
            cls = __registry[_name]
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
    from lsstypes import __version__
    _write(filename, dict(_version=__version__) | observable.__getstate__(to_file=True))


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
    state = _read(filename)
    state.pop('_version', None)
    return from_state(state)


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


def deep_eq(obj1, obj2, equal_nan=True, type_permissive=True, raise_error=False, label=None):
    """(Recursively) test equality between ``obj1`` and ``obj2``."""
    if raise_error:
        label_str = f' for object {label}' if label is not None else ''

    def array_equal(obj1, obj2):
        toret = False
        try:
            toret = np.array_equal(obj1, obj2, equal_nan=equal_nan)
        except TypeError:  # nan not supportedt
            try:
                toret = np.array_equal(obj1, obj2)
            except:
                pass
        if not toret and raise_error:
            raise ValueError(f'Not equal: {obj2} vs {obj1}{label_str}')
        return toret

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
            return array_equal(obj1, obj2)
    if type_permissive:
        return array_equal(obj1, obj2)
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



def _edges_name(axis):
    return f'{axis}_edges'


def _edges_names(axes):
    return list(map(_edges_name, axes))


def _check_data_names(self):
    edges_names = _edges_names(self._coords_names)
    known_names = set(self._coords_names + self._values_names + _edges_names(self._coords_names))
    data_names = set(self._data)
    unknown = data_names - known_names
    if unknown:
        raise ValueError(f'{unknown} not unknown, expected one of {known_names}')
    missing = known_names - data_names - set(edges_names)
    if missing:
        raise ValueError(f'{missing} missing, expected all of {known_names - set(edges_names)}')


def _check_data_shapes(self):
    edges_names = _edges_names(self._coords_names)
    for coord_name, edges_name in zip(self._coords_names, edges_names):
        if edges_name in self._data:
            eshape = self._data[edges_name].shape
            cshape = self._data[coord_name].shape + (2,)
            assert eshape == cshape, f'expected shape of {edges_name} is {cshape}, got {eshape}'
    cshape = tuple(self._data[coord_name].shape[0] for coord_name in self._coords_names)
    for name in self._values_names:
        try:
            vshape = self._data[name].shape
        except AttributeError as exc:
            raise AttributeError(f'{name} is not an array') from exc
        assert vshape == cshape, f'expected shape of {name} is {cshape}, got {vshape}'


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
        self._coords_names = list(coords or [])

    def __post_init__(self):
        # Check data consistency
        assert not any(k in self._forbidden_names for k in self._data), f'Cannot use {self._forbidden_names} as name for arrays'
        self._values_names = [name for name in self._data if name not in self._coords_names and name not in _edges_names(self._coords_names)]
        assert len(self._values_names), 'Provide at least one value array'
        _check_data_names(self)
        _check_data_shapes(self)

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
        if center == 'mid_if_edges_and_nan':
            mid = self.coords(axis=axis, center='mid_if_edges')
            coord = self._data[axis]
            return np.where(np.isnan(coord), mid, coord)
        if center == 'mid_if_edges':
            if edges is None: center = None
            else: center = 'mid'
        if center is None:
            return self._data[axis]
        assert edges is not None, 'edges must be provided'
        return np.mean(edges, axis=-1)

    def edges(self, axis=None, **kwargs):
        """
        Get edge array(s).

        Parameters
        ----------
        axis : str or int, optional
            Name or index of coordinate.

        default : any, optional
            When edges are not in the observable, and default is provided, return default.
            Else if default is not provided, estimate edges from coordinates.

        Returns
        -------
        edges : array or dict
        """
        if axis is None:
            return {axis: self.edges(axis=axis) for axis in self._coords_names}
        if not isinstance(axis, str):
            axis = self._coords_names[axis]
        axis_edges = _edges_name(axis)
        if axis_edges in self._data:
            return self._data[axis_edges]
        with_default = kwargs
        if with_default:
            default = kwargs['default']
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

    def value_as_leaf(self, **kwargs):
        """Return value as leaf."""
        value = self.value(**kwargs)
        axes_edges = [_edges_name(axis) for axis in self._coords_names]
        edges = {axis_edge: self.edges(axis_edge, default=None) for axis_edge in axes_edges}
        edges = {axis_edge: edge for axis_edge, edge in edges.items() if edge is not None}
        return ObservableLeaf(value=value, **self.coords(), **edges, coords=self._coords_names)
    
    def __array__(self):
        return np.asarray(self.value())

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
        """Number of dimensions (coordinates)."""
        return len(self._coords_names)

    @property
    def attrs(self):
        """Return attributes dictionary."""
        return self._attrs

    @property
    def meta(self):
        return self._meta

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
            axis_edges = _edges_name(axis)
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
            binweight = getattr(self, '_binweight', None)
            if binweight is None:
                undefined_weight = True
                weight, normalized = False, True
            else:
                weight, normalized = binweight(name=name)

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
        self._data.update(kwargs)

    def clone(self, **kwargs):
        """Copy and update data."""
        new = self.copy()
        for name in ['attrs', 'meta']:
            if name in kwargs:
                setattr(new, f'_{name}', dict(kwargs.pop(name) or {}))
        new._update(**kwargs)
        _check_data_names(self)
        _check_data_shapes(self)
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
            axis_edges = _edges_name(axis)
            transform, edges = new._transform(limit, axis=axis, return_edges=True, center=center, full=False, name=axis)
            if transform.ndim == 1:  # mask
                index = transform
                for name in new._values_names:
                    # Cast to numpy array, else error with JAX: "NotImplementedError: The 'raise' mode to jnp.take is not supported."
                    new._data[name] = np.take(np.asarray(new._data[name]), index, axis=iaxis)
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
            state[name] = list(getattr(self, f'_{name}'))
            if not state[name]: state.pop(name)
        state['name'] = self._name
        if self._attrs: state['attrs'] = dict(self._attrs)
        if self._meta: state['meta'] = dict(self._meta)
        return state

    def __setstate__(self, state):
        for name in ['values_names', 'coords_names']:
            setattr(self, '_' + name, [str(n) for n in state.get(name, [])])
        self._attrs = state.get('attrs', {})  # because of hdf5 reader
        self._meta = state.get('meta', {})
        self._data = {name: state[name] for name in self._values_names + self._coords_names}
        for name in _edges_names(self._coords_names):
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
            elif name in new._values_names:
                _weights = weights
                divide = True
            else:  # coords or edges
                _weights = [1] * len(observables)
            if name not in new._values_names:  # coords or edges
                divide = True
            assert len(_weights) == len(observables)
            new._data[name] = sum(weight * observable._data[name] for observable, weight in zip(observables, _weights))
            if divide: new._data[name] = new._data[name] / sum(_weights)
        return new

    @classmethod
    def sum(cls, observables, weights=None):
        """Sum multiple observables."""
        sumweight = getattr(cls, '_sumweight', None)
        if sumweight is not None: sumweight = partial(sumweight, weights=weights)
        else: sumweight = weights
        return cls._average(observables, weights=sumweight)

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
        axis_edges = _edges_name(axis)
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


def _pad_transform(transform, start=0, stop=None, size=None):
    # Pad a 1d (index) or 2d (matrix) transform to full size
    if transform.ndim == 1:
        if np.issubdtype(transform.dtype, np.integer):
            return np.concatenate([np.arange(start), start + transform, np.arange(stop, size)])
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
            axis_edges = _edges_name(axis)
            if axis_edges in new._data:
                sub._data[axis_edges] = np.concatenate([self._observable.edges(axis)[:start], sub.edges(axis), self._observable.edges(axis)[stop:]], axis=0)
            shape = tuple(len(sub._data[axis]) for axis in sub._coords_names)
            size = 1
            for s in shape: size *= s

            if self._hook is not None:
                if sub_transform.ndim == 1:
                    index1d = _pad_transform(sub_transform, start=start, stop=stop, size=new.shape[iaxis])
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


def _tree_iter(tree, callback, level=None, is_leaf=None, input_label=False, input_strlabel=False, **kwargs):
    input_not_leaf = False
    if isinstance(is_leaf, str) and is_leaf == 'input_not_leaf':
        input_not_leaf = True
        is_leaf = None

    if is_leaf is None:
        def is_leaf(branch):
            return branch._is_leaf

    def _is_leaf(branch, is_input):
        if input_not_leaf and is_input: return False
        return is_leaf(branch)

    def _stop(branch, level, is_input=False):
        return level is not None and level <= 0 or _is_leaf(branch, is_input)

    def __tree_iter(tree, level=0, label=None, strlabel=None, is_input=False):
        if _stop(tree, level, is_input=is_input):
            kw = {}
            if input_label:
                kw.update(label=label)
            if input_strlabel:
                kw.update(strlabel=strlabel)
            return callback(tree, **kw, **kwargs)
        level = level - 1 if level is not None else None
        for ibranch, branch in enumerate(tree._branches):
            kw = {}
            if input_label:
                kw.update(label=label | {key: value[ibranch] for key, value in tree._labels.items()})
            if input_strlabel:
                kw.update(strlabel=strlabel | {key: value[ibranch] for key, value in tree._strlabels.items()})
            __tree_iter(branch, level=level, **kw)

    return __tree_iter(tree, level=level, label={}, strlabel={}, is_input=True)


def tree_flatten(tree, level=1, is_leaf=None, return_labels=False, return_strlabels=False):
    """
    Flatten the tree into a list of branches or leaves.

    Parameters
    ----------
    tree : object
        Observable tree.
    level : int, optional
        Level up to which flatten the tree. If `None`, goes to maximum depth.
    is_leaf : callable, optional
        Function to apply to a branch which returns `True` if branch is to be considered a leaf
        else `False` (and iterated over).
    return_labels : bool, optional
        If `True`, also return labels.
    return_strlabels: bool, optional
        If `True`, also return labels with values as str.

    Returns
    -------
    list
        List of branches or leaves up to level.
    """
    branches, labels, strlabels = [], [], []

    def callback(branch, label=None, strlabel=None):
        branches.append(branch)
        labels.append(label)
        strlabels.append(strlabel)

    _tree_iter(tree, callback, level=level, is_leaf=is_leaf, input_label=return_labels, input_strlabel=return_strlabels)

    toret = [branches]
    if return_labels:
        toret.append(labels)
    if return_strlabels:
        toret.append(strlabels)
    if len(toret) == 1:
        return toret[0]
    return tuple(toret)


def tree_labels(tree, level=1, return_type='flatten', as_str=False, is_leaf=None):
    """
    Return a list of dicts with the labels for each branch or leaf.

    Parameters
    ----------
    tree : object
        Observable tree.
    level : int, optional
        Level to retrieve labels from. If `None`, retrieve all levels.
    return_type : str, optional
        If 'keys' or 'names', return only the list of unique keys (i.e. not label values) up to `level`.
        If 'flatten' (default), return the list of dictionaries {label key: label value} for each branch or leaf up to `level`.
        If 'flatten_values', return the list of values.
        If 'unflatten', return a dictionary of {label key, label values}. If a label key does not exit in a leaf, fill with `Ellipsis`.
    as_str : bool, optional
        If `True`, return labels as strings.
    is_leaf : callable, optional
        Function to apply to a branch which returns `True` if branch is to be considered a leaf
        else `False` (and iterated over).

    Returns
    -------
    labels : list or dict, or list of dict
    """
    _, labels = tree_flatten(tree, level=level, is_leaf=is_leaf, return_labels=not as_str, return_strlabels=as_str)

    def _find_all_keys(labels):
        all_keys = []
        for label in labels:
            for key in label:
                if key not in all_keys: all_keys.append(key)
        return all_keys

    if return_type in ['keys', 'names']:
        toret = _find_all_keys(labels)
    elif return_type == 'flatten':
        toret = labels
    elif return_type == 'flatten_values':
        toret = [tuple(label.values()) for label in labels]
    elif return_type == 'unflatten':
        all_keys = _find_all_keys(labels)
        toret = {key: [label.get(key, Ellipsis) for label in labels] for key in all_keys}
    return toret



def tree_map(f, tree, level=None, input_label=False, is_leaf=None):
    """
    Apply a function (that should return a branch) to branches of a tree structure.

    Parameters
    ----------
    f : callable
        Function to apply to each branch or leaf.
    tree : object or list
        Observable tree(s).
        The returned tree order is that of the first tree.
    level : int, optional
        Level to apply function at. If `None`, goes to maximum depth.
    input_label : bool, optional
        Also pass labels to `f`: `f(branch, label)`.
    is_leaf : callable, optional
        Function to apply to a branch which returns `True` if branch is to be considered a leaf
        (and passed to `f`), else `False` (and iterated over).

    Returns
    -------
    new
        New tree.
    """
    input_not_leaf = False
    if isinstance(is_leaf, str) and is_leaf == 'input_not_leaf':
        input_not_leaf = True
        is_leaf = None

    if is_leaf is None:
        def is_leaf(branch):
            return branch._is_leaf

    def _is_leaf(branch, is_input):
        if input_not_leaf and is_input: return False
        return is_leaf(branch)

    def _stop(branch, level, is_input=False):
        if isinstance(branch, list):
            return any(_stop(b, level) for b in branch)
        return level is not None and level <= 0 or _is_leaf(branch, is_input)

    def _copy(branch):
        # Shallow copy
        if isinstance(branch, list):
            return [_copy(b) for b in branch]
        if branch._is_leaf:
            return branch.copy()
        new = branch.__class__.__new__(branch.__class__)
        new._branches = list(branch._branches)
        for name in ['_labels', '_strlabels', '_attrs', '_meta']:
            setattr(new, name, getattr(branch, name).copy())
        return new

    def _tree_map(tree, label, level=0, is_input=False):

        if _stop(tree, level, is_input=is_input):
            args = (tree,)
            if input_label: args += (label,)
            toret = f(*args)
            assert isinstance(toret, (ObservableLeaf, ObservableTree, list))
            return toret

        new = None
        level = level - 1 if level is not None else None
        if isinstance(tree, list):  # f takes list
            tree0 = tree[0]
            branches = []
            for ibranch in range(len(tree0._branches)):
                _labels = {k: v[ibranch] for k, v in tree0._labels.items()}
                branches.append([t.get(**_labels) for t in tree])
        else:
            branches = tree._branches
            tree0 = tree

        for ibranch, branch in enumerate(branches):
            res = _tree_map(branch, label=label | {k: v[ibranch] for k, v in tree0._labels.items()}, level=level)
            if isinstance(res, list):  # f returns list
                if new is None:
                    new = [_copy(tree0) for i in range(len(res))]
                for i in range(len(res)):
                    new[i]._branches[ibranch] = res[i]
            else:
                if new is None:
                    new = _copy(tree0)
                new._branches[ibranch] = res
        return new

    return _tree_map(tree, label={}, level=level, is_input=True)


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
    toret = tree._branches[index[0]]
    if len(index) == 1:
        return toret
    return _get_leaf(toret, index[1:])


def _get_range_in_tree(tree, index):
    # Get start/stop in the full tree for a given index (branch)
    start = 0
    current_tree = tree
    for idx in index:
        start += sum(branch.size for branch in current_tree._branches[:idx])
        current_tree = current_tree._branches[idx]
    return start, start + current_tree.size


def _replace_in_tree(tree, index, sub):
    # Replace a branch in the tree, return start/stop index of replaced branch
    start = 0
    current_tree = tree
    for idx in index[:-1]:
        start += sum(branch.size for branch in current_tree._branches[:idx])
        current_tree = current_tree._branches[idx]
    start += sum(branch.size for branch in current_tree._branches[:index[-1]])
    stop = start + current_tree._branches[index[-1]].size
    current_tree._branches[index[-1]] = sub
    return start, stop


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


def _check_tree(self):
    branches_labels = []
    for branch in self._branches:
        if not branch._is_leaf:
            branches_labels += branch.labels(level=None, return_type='keys')
    nbranches = len(self._branches)
    for k, v in self._labels.items():
        if isinstance(v, list):
            assert len(v) == nbranches, f'The length of labels (found {len(v):d}) must match the number of branches (got {nbranches:d})'
            self._labels[k] = v
        else:
            self._labels[k] = [v] * nbranches
        if k in branches_labels:
            raise ValueError(f'Cannot use labels with same name at different levels: {k}')
        self._strlabels[k] = list(map(self._label_to_str, self._labels[k]))
        assert not any(v in self._forbidden_label_values for v in self._strlabels[k]), 'Cannot use "labels" as a label value'
        convert = list(map(self._str_to_label, self._strlabels[k]))
        assert convert == self._labels[k], f'Labels must be mappable to str; found label -> str -> label != identity:\n{convert} != {self._labels[k]}'
    uniques = []
    for ibranch in range(nbranches):
        labels = tuple(self._strlabels[k][ibranch] for k in self._strlabels)
        if labels in uniques:
            raise ValueError(f'Label {labels} is duplicated')
        uniques.append(labels)


@register_type
class ObservableTree(object):
    """
    A collection of Observable objects, supporting selection, slicing, and labeling.
    """
    _name = 'tree_base'
    _forbidden_label_values = ('name', 'attrs', 'labels_names', 'labels_values')
    _sep_strlabels = '-'
    _is_leaf = False

    def __init__(self, branches, attrs=None, meta=None, **labels):
        """
        Parameters
        ----------
        branches : list of ObservableLeaf
            The branches in the collection.
        labels : dict
            Label arrays (e.g. ell=[0, 2], observable=['spectrum',...]).
        """
        self._branches = []
        self._meta = dict(meta or {})
        self._attrs = dict(attrs or {})
        self._labels, self._strlabels = {key: [] for key in labels}, {key: [] for key in labels}
        new = self.insert(branches, **labels)
        self.__dict__.update(new.__dict__)

    def insert(self, *args, **labels):
        """
        Insert branches in tree.

        Parameters
        ----------
        index : optional
            Index or list of indices where to insert input branch(es) or leaf(ves).
            If not provided, appends branch(es) or leaf(ves) to the tree.
        branch : ObservableTree or ObservableLeaf
            Input branch(es) or leaf(ves) to insert.
        labels : dict
            Label arrays (e.g. ell=[0, 2], observable=['spectrum',...]).

        Returns
        -------
        new : ObservableTree
        """
        new = self.copy()
        if len(args) == 1:
            indices, branches = None, args[0]
        elif len(args) == 2:
            indices, branches = args
        else:
            raise ValueError(f'provide index, branch or branch; got {len(args):d} arguments')
        isscalar = isinstance(branches, (ObservableLeaf, ObservableTree))
        if isscalar:
            branches = [branches]
        if indices is None:
            indices = list(len(self._branches) + np.arange(len(branches)))
        if np.ndim(indices) == 0:
            indices = [indices]
        if isscalar:
            labels = {key: [value] for key, value in labels.items()}
        assert len(branches) == len(indices)
        assert all(len(values) == len(indices) for values in labels.values())
        for ii, index in enumerate(indices):
            new._branches.insert(index, branches[ii])
            for key in new._labels:
                new._labels[key].insert(index, labels[key][ii])
            for key in new._strlabels:
                new._strlabels[key].insert(index, self._label_to_str(labels[key][ii]))
        _check_tree(new)
        return new

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
        return f'{self.__class__.__name__}(labels={self.labels(level=1, return_type="flatten")}, size={self.size})'

    @property
    def attrs(self):
        """Dictionary of attributes associated with the tree."""
        return self._attrs

    @property
    def meta(self):
        return self._meta

    def labels(self, level=1, return_type='flatten', as_str=False, is_leaf=None):
        """
        Return a list of dicts with the labels for each branch or leaf.

        Parameters
        ----------
        level : int, optional
            Level to retrieve labels from. If `None`, retrieve all levels.
        return_type : str, optional
            If 'keys' or 'names', return only the list of unique keys (i.e. not label values) up to `level`.
            If 'flatten' (default), return the list of dictionaries {label key: label value} for each branch or leaf up to `level`.
            If 'flatten_values', return the list of values.
            If 'unflatten', return a dictionary of {label key, label values}. If a label key does not exit in a leaf, fill with `Ellipsis`.
        as_str : bool, optional
            If `True`, return labels as strings.

        Returns
        -------
        labels : list or dict, or list of dict
        """
        if is_leaf is None:
            is_leaf = 'input_not_leaf'
        return tree_labels(self, level=level, return_type=return_type, as_str=as_str, is_leaf=is_leaf)

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

        self_index = list(range(len(self._branches)))
        # First find labels in current level, keeping original order
        for k in self._labels:
            if k in labels:
                vselect = labels.pop(k)
                _indices = find(vselect, k)
                self_index = [index for index in self_index if index in _indices]
        if labels:  # remaining labels
            toret = {}
            for index in self_index:
                sub_index_labels = self._branches[index]._index_labels(labels, flatten=False)
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
            The matching subtree or leaf.
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
            branches, ibranches = [], []
            for ibranch, branch in enumerate(tree._branches):
                if ibranch in indices:
                    if indices[ibranch] is not None:
                        branch = get_subtree(tree, indices[ibranch])
                    branches.append(branch)
                    ibranches.append(ibranch)
            new = tree.copy()
            new._branches = branches
            new._labels = {k: [v[idx] for idx in ibranches] for k, v in tree._labels.items()}
            new._strlabels = {k: [v[idx] for idx in ibranches] for k, v in tree._strlabels.items()}
            return new

        return get_subtree(self, indices)

    def flatten(self, level=1, is_leaf=None, return_labels=False, return_strlabels=False):
        """
        Flatten the tree into a list of branches or leaves.

        Parameters
        ----------
        level : int, optional
            Level up to which flatten the tree. If `None`, goes to maximum depth.
        is_leaf : callable, optional
            Function to apply to a branch which returns `True` if branch is to be considered a leaf
            (and added to the output list), else `False` (and iterated over).
        return_labels : bool, optional
            If `True`, also return labels.
        return_strlabels: bool, optional
            If `True`, also return labels with values as str.

        Returns
        -------
        list
            List of branches or leaves up to level.
        """
        if is_leaf is None:
            is_leaf = 'input_not_leaf'
        return tree_flatten(self, level=level, is_leaf=is_leaf, return_labels=return_labels, return_strlabels=return_strlabels)

    def map(self, f, level=None, input_label=False, is_leaf=None):
        """
        Apply a function (that should return a branch) to branches of the tree.

        Parameters
        ----------
        f : callable
            Function to apply to each branch or leaf.
        level : int, optional
            Level to apply function at. If `None`, goes to maximum depth.
        input_label : bool, default=False
            Also pass labels to `f`: `f(branch, label)`.
        is_leaf : callable, optional
            Function to apply to a branch which returns `True` if branch is to be considered a leaf
            (and passed to `f`), else `False` (and iterated over).

        Returns
        -------
        new
            New tree.
        """
        if is_leaf is None:
            is_leaf = 'input_not_leaf'
        return tree_map(f, self, level=level, input_label=input_label, is_leaf=is_leaf)

    def match(self, observable):
        """
        Match the the tree to the input observable, recursively, matching labels and branches.

        Parameters
        ----------
        observable : ObservableTree
            Observable to match to.

        Returns
        -------
        ObservableTree
            New tree with branches matched to input observable.
        """
        new = tree_map(lambda observables: observables[1].match(observables[0]), [observable, self], is_leaf='input_not_leaf')
        for name in ['_attrs', '_meta']:
            setattr(new, name, getattr(self, name))
        return new

    def sizes(self, level=None):
        """Size of each branch at given level."""
        return list(map(lambda leaf: leaf.size, tree_flatten(self, level=level)))

    @property
    def size(self):
        """Total size of the tree."""
        return sum(self.sizes(level=1))

    def __iter__(self):
        """Iterate over branches."""
        return iter(self._branches)

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
        def f(leaf):
            _limits = limits
            if leaf._is_leaf:
                _limits = {k: v for k, v in limits.items() if k in leaf._coords_names}
            return leaf.select(**_limits)

        return tree_map(f, self, level=1, input_label=False, is_leaf='input_not_leaf')

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
        assert isinstance(concatenate, bool)
        leaves = tree_flatten(self, level=1)
        values = [leaf.value().ravel() for leaf in leaves]
        if concatenate:
            return np.concatenate(values, axis=0)
        return values

    def __array__(self):
        return self.value(concatenate=True)

    def clone(self, **kwargs):
        """
        Return a copy of the tree, with updated values.
        One can provide for each kwargs entry either a list of values,
        with ``None`` for the branches not to be updated, or a (concatenated) numpy array.
        """
        new = self.copy()
        if not kwargs:
            return new
        for name in ['attrs', 'meta']:
            if name in kwargs:
                setattr(new, f'_{name}', dict(kwargs.pop(name) or {}))

        def _get_values(kwargs, ibranch, start, stop, shape=None):
            kw = dict()
            for name, value in kwargs.items():
                if isinstance(value, (tuple, list)):
                    v = value[ibranch]
                else:
                    v = value[start:stop]
                    if shape is not None: v = v.reshape(shape)
                if v is not None: kw[name] = v
            return kw

        start = 0
        for ibranch, branch in enumerate(new._branches):
            stop = start + branch.size
            shape = branch.shape if branch._is_leaf else None
            new._branches[ibranch] = branch.clone(**_get_values(kwargs, ibranch, start, stop, shape=shape))
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
    def sum(cls, observables, weights=None):
        """Sum multiple observables."""
        return tree_map(lambda observables: observables[0].sum(observables, weights=weights), observables, level=1, is_leaf='input_not_leaf')

    @classmethod
    def mean(cls, observables):
        """Mean of multiple observables."""
        return tree_map(lambda observables: observables[0].mean(observables), observables, level=1, is_leaf='input_not_leaf')

    @classmethod
    def cov(cls, observables):
        """Covariance matrix of multiple observables."""
        mean = cls.mean(observables)
        value = np.cov([observable.value() for observable in observables], rowvar=False, ddof=1)
        return CovarianceMatrix(value, observable=mean)

    @classmethod
    def join(cls, observables):
        """Join multiple trees into a single one."""
        branches, labels = [], {label: [] for label in observables[0]._labels}
        for observable in observables:
            assert isinstance(observable, ObservableTree)
            assert set(observable._labels) == set(labels), 'All collections must have same labels'
            branches += observable._branches
            for k in labels:
                labels[k] = labels[k] + observable._labels[k]
        new = observables[0].copy()
        ObservableTree.__init__(new, branches, **labels)  # check labels
        return new

    @classmethod
    def concatenate(cls, observables, axis=0):
        """Concatenate multiple observables."""
        return tree_map(lambda observables: observables[0].concatenate(observables, axis=axis), observables, level=1, is_leaf='input_not_leaf')

    @property
    def at(self):
        """Helper to select or slice the tree in-place."""
        return _ObservableTreeUpdateHelper(self)

    def clear(self):
        """Return a tree with no branch."""
        new = self.__class__.__new__(self.__class__)
        labels = self.labels(return_type='keys')
        labels = {key: [] for key in labels}
        ObservableTree.__init__(new, [], attrs=self._attrs, meta=self._meta, **labels)
        return new

    def copy(self):
        """Return a copy of the tree (arrays not copied)."""
        new = self.__class__.__new__(self.__class__)
        new.__setstate__(self.__getstate__())
        return new

    def __getstate__(self, to_file=False):
        state = {}
        if not to_file:
            state['branches'] = [branch.__getstate__() for branch in self._branches]
            state['labels'] = {key: list(value) for key, value in self._labels.items()}
            state['strlabels'] = {key: list(value) for key, value in self._strlabels.items()}
        else:
            state['labels_names'] = self._sep_strlabels.join(list(self._labels.keys()))
            state['labels_values'] = []
            for ibranch, branch in enumerate(self._branches):
                label = self._sep_strlabels.join([self._strlabels[k][ibranch] for k in self._labels])
                state['labels_values'].append(label)
                state[label] = branch.__getstate__(to_file=to_file)
        if self._meta: state['meta'] = dict(self._meta)
        if self._attrs: state['attrs'] = dict(self._attrs)
        state['name'] = self._name
        return state

    def __setstate__(self, state):
        self._meta = state.get('meta', {})
        self._attrs = state.get('attrs', {})
        if 'branches' in state:
            branches = state['branches']
            self._branches = [from_state(branch) for branch in branches]
            self._labels = state['labels']
            self._strlabels = state['strlabels']
        else:  # h5py format
            label_names = np.array(state['labels_names']).item().split(self._sep_strlabels)
            label_values = list(map(lambda x: x.split(self._sep_strlabels), np.array(state['labels_values'])))
            self._labels, self._strlabels = {}, {}
            for i, name in enumerate(label_names):
                self._strlabels[name] = [v[i] for v in label_values]
                self._labels[name] = [self._str_to_label(s, squeeze=True) for s in self._strlabels[name]]
            nbranches = len(state['labels_values'])
            self._branches = []
            for ibranch in range(nbranches):
                label = state['labels_values'][ibranch]
                self._branches.append(from_state(state[label]))

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


class _ObservableTreeUpdateRef(object):

    def __init__(self, tree, indices=None, select=None, hook=None):
        self._tree = tree
        self._select = select
        self._indices = indices
        self._hook = hook

    def clone(self, **kwargs):
        """
        Return a copy of the tree, with updated values.
        One can provide for each kwargs entry either a list of values,
        with ``None`` for the branches not to be updated, or a (concatenated) numpy array.
        """
        if self._hook is not None:
            raise NotImplementedError('hook not implemented for clone')
        new = self._tree.copy()
        for index in (self._indices if self._indices is not None else [None]):
            branch = _get_leaf(self._tree, index)
            sub = branch.clone(**kwargs)
            if index is None:
                new = sub
            else:
                start, stop = _replace_in_tree(new, index, sub)
        return new

    def map(self, f, level=None, input_label=False):
        """Apply a function (that should return a branch) to branches of the tree."""
        if self._hook is not None:
            raise NotImplementedError('hook not implemented for clone')
        new = self._tree.copy()
        for index in (self._indices if self._indices is not None else [None]):
            branch = _get_leaf(self._tree, index)
            sub = branch.map(f, level=level, input_label=input_label)
            if index is None:
                new = sub
            else:
                start, stop = _replace_in_tree(new, index, sub)
        return new

    def get(self, *args, **labels):
        """Return subtree or leaf corresponding to input labels."""
        # Order is preserved
        new = self._tree.copy()
        if self._hook is not None:
            mask = np.ones(self._tree.size, dtype='?')
        for index in (self._indices if self._indices is not None else [None]):
            branch = _get_leaf(self._tree, index)
            _labels = _format_input_labels(branch, *args, **labels)
            sub = branch.get(**_labels)
            if index is None:
                new = sub
                start, stop = 0, branch.size
            else:
                start, stop = _replace_in_tree(new, index, sub)
            if self._hook is not None:
                _mask = np.zeros(branch.size, dtype='?')
                for _index in branch._index_labels(_labels):
                    _mask[slice(*_get_range_in_tree(branch, _index))] = True
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
            branch = _get_leaf(self._tree, index)
            branch = _get_update_ref(branch)(branch, select=self._select, hook=hook).__getitem__(masks)
            size = new.size
            if self._hook:
                branch, _transform = branch
            size = new.size
            start, stop = _replace_in_tree(new, index, branch)
            if self._hook:
                _transform = _pad_transform(_transform, start=start, stop=stop, size=size)
                transform = _join_transform(transform, _transform, size=size)

        if self._hook:
            return self._hook(new, transform=transform)
        return new

    def select(self, **limits):
        """Select a range in one or more coordinates."""
        hook = None
        if self._hook:
            def hook(branch, transform): return branch, transform
            hook.weight = getattr(self._hook, 'weight', None)

        new = self._tree.copy()
        transform = None
        for index in (self._indices if self._indices is not None else self._tree._index_labels({})):
            branch = _get_leaf(self._tree, index)
            branch = _get_update_ref(branch)(branch, select=self._select, hook=hook).select(**limits)
            if self._hook:
                branch, _transform = branch
            size = new.size
            start, stop = _replace_in_tree(new, index, branch)
            if self._hook:
                _transform = _pad_transform(_transform, start=start, stop=stop, size=size)
                transform = _join_transform(transform, _transform, size=size)

        if self._hook:
            return self._hook(new, transform=transform)
        return new

    def match(self, observable):
        """Match coordinates to those of input tree."""
        hook = None
        if self._hook:
            def hook(branch, transform): return branch, transform
            hook.weight = getattr(self._hook, 'weight', None)

        if self._indices is not None:
            assert len(self._indices) == 1, 'match can only be applied to a tree or leaf'
            index = self._indices[0]
        else:
            index = None
        tree = _get_leaf(self._tree, index)
        transforms, starts = [], []
        branches, ibranches = [], []
        for ibranch, branch in enumerate(observable._branches):
            _ibranch = tree._index_labels({k: v[ibranch] for k, v in observable._labels.items()}, flatten=True)
            assert len(_ibranch) == 1 and len(_ibranch[0]) == 1
            _ibranch = _ibranch[0][0]
            _branch = tree._branches[_ibranch]
            branch = _get_update_ref(_branch)(_branch, select=self._select, hook=hook).match(branch)
            if self._hook:
                branch, _transform = branch
            branches.append(branch)
            ibranches.append(_ibranch)
            if self._hook:
                start, stop = _get_range_in_tree(tree, (_ibranch,))
                transforms.append(_transform)
                starts.append((start, stop))

        if self._hook:
            transform = _concatenate_transforms(transforms, starts, size=tree.size)
        tree = tree.copy()
        tree._branches = branches
        tree._labels = {k: [v[idx] for idx in ibranches] for k, v in tree._labels.items()}
        tree._strlabels = {k: [v[idx] for idx in ibranches] for k, v in tree._strlabels.items()}
        new = tree
        if index is not None:
            new = self._tree.copy()
            start, stop = _replace_in_tree(new, (index,), tree)
            if self._hook:
                transform = _pad_transform(transform, start=start, stop=stop, size=self._tree.size)
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

        def hook(branch, transform=None):
            if index is not None:
                new = self._tree.copy()
                start, stop = _replace_in_tree(new, index, branch)
                if self._hook is not None:
                    transform = _pad_transform(transform, start=start, stop=stop, size=self._tree.size)
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
        return self._branches[0]._coords_names

    @property
    def _values_names(self):
        return self._branches[0]._values_names

    @property
    def shape(self):
        """Observable shape."""
        return self._branches[0].shape

    @property
    def size(self):
        """Observable size."""
        return self._branches[0].size

    def edges(self, *args, **kwargs):
        """Observable edges."""
        return self._branches[0].edges(*args, **kwargs)

    def coords(self, *args, **kwargs):
        """Observable coordinates."""
        return self._branches[0].coords(*args, **kwargs)

    def __getitem__(self, masks):
        indices = _format_masks(self.shape, masks)
        new = self.copy()
        for ibranch, branch in new._branches:
            new._branches[ibranch] = branch.__getitem__(indices)
        return new

    @property
    def at(self):
        """Helper to select or slice the tree in-place."""
        return _LeafLikeObservableTreeHelper(self)

    def value(self):
        """Main value of the observable."""
        raise NotImplementedError

    def value_as_leaf(self, **kwargs):
        """Return value as leaf."""
        return ObservableLeaf.value_as_leaf(self, **kwargs)
    
    @classmethod
    def _average(cls, observables, weights=None):
        # Average multiple observables
        return tree_map(lambda observables: observables[0]._average(observables, weights=weights), observables, level=1)

    @classmethod
    def sum(cls, observables, weights=None):
        """Sum multiple observables."""
        sumweight = getattr(cls, '_sumweight', None)
        if sumweight is not None: sumweight = partial(sumweight, weights=weights)
        else: sumweight = weights
        return cls._average(observables, weights=sumweight)

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
            value = np.take(np.asarray(self._matrix.value()), transform, axis=self._axis)
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

    _name = 'window_matrix'

    def __init__(self, value, observable, theory, attrs=None):
        self._value = value
        self._observable = observable
        self._theory = theory
        self._attrs = dict(attrs or {})

    @property
    def attrs(self):
        """Other attributes."""
        return self._attrs

    @property
    def shape(self):
        """Matrix shape."""
        return self._value.shape

    def value(self):
        """Value (numpy array) of the window matrix."""
        return self._value

    def __array__(self):
        return np.asarray(self.value())

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
            if name in ['attrs']:
                setattr(new, '_' + name, dict(value or {}))
            elif name in ['observable', 'theory', 'value']:
                setattr(new, '_' + name, value)
            else:
                raise ValueError(f'Unknown attribute {name}')
        return new

    def __getstate__(self, to_file=False):
        state = {}
        state['name'] = self._name
        state['value'] = self.value()
        state['attrs'] = dict(self.attrs)
        for name in ['observable', 'theory']:
            state[name] = getattr(self, name).__getstate__(to_file=to_file)
        return state

    def __setstate__(self, state):
        self._value = state['value']
        self._attrs = state.get('attrs', {})
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
    def sum(cls, matrices, weights=None):
        """Sum multiple window matrices."""
        new = matrices[0].copy()

        def get_sumweight(leaves):
            sumweight = getattr(leaves[0], '_sumweight', None)
            weight = None
            if sumweight is not None:
                weight = sumweight(leaves, weights=weights)
            if weight is None:
                weight = [np.ones_like(leaves[0].value()) / len(leaves)] * len(leaves)
            return weight

        assert all(matrix.observable.labels(level=None, return_type='flatten') == new.observable.labels(level=None, return_type='flatten') for matrix in matrices[1:])
        weights = list(map(get_sumweight, zip(*[tree_flatten(matrix.observable, level=None) for matrix in matrices])))
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
                    xx = leaf.coords(0)
                    if xx.ndim > 1: xx = None
                if xx is None:
                    xx = np.arange(leaf.size)
                return xx

            if observable._is_leaf:
                x.append(get_x(observable))
                indices.append(np.arange(observable.size))

            for label in observable.labels(level=level, return_type='flatten'):
                leaf = observable.get(**label)
                labels.append(','.join([observable._label_to_str(v) for v in label.values()]))
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

    @utils.plotter
    def plot_slice(self, indices, axis='observable', level=None, color='C0', label=None, xscale='linear', yscale='log', fig=None):
        """
        Plot a slice of the window matrix along the specified indices.

        Parameters
        ----------
        indices : int, array-like
            Indices (or values) along which to slice the window matrix. Can be a single index or a list.
        axis : str, optional
            Axis, "observable" (alias: "o") or "theory" (alias: "t") to slice along.
        level : int, optional
            Level in tree at which define different panels. Default to the maximum depth.
        color : str, default='C0'
            Color for the plotted lines.
        label : str, optional
            Label for the plotted lines.
        xscale : str, default='linear'
            X-axis scale ('linear', 'log', etc.).
        yscale : str, default='log'
            Y-axis scale ('linear', 'log', etc.).
        fig : matplotlib.figure.Figure, optional
            Figure to plot into. If None, a new figure is created.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        """
        from matplotlib import pyplot as plt
        if np.ndim(indices) == 0: indices = [indices]
        idxs = np.array(indices)
        alphas = np.linspace(1, 0.2, len(indices))

        def _get(observable):
            xlabels, labels, x, edges, indices = [], [], [], [], []

            def get_x(leaf):
                xx = None
                if len(leaf._coords_names) == 1:
                    xx = leaf.coords(0)
                    if xx.ndim > 1: xx = None
                if xx is None:
                    xx = np.arange(leaf.size)
                return xx

            def get_edges(leaf):
                edges = leaf.edges(0)
                return edges

            if observable._is_leaf:
                x.append(get_x(observable))
                edges.append(get_edges(observable))
                labels.append('')
                indices.append(np.arange(observable.size))
            else:
                for label in observable.labels(level=level, return_type='flatten'):
                    leaf = observable.get(**label)
                    labels.append(','.join([observable._label_to_str(v) for v in label.values()]))
                    x.append(get_x(leaf))
                    edges.append(get_edges(leaf))
                    start, stop = _get_range_in_tree(observable, observable._index_labels(label)[0])
                    indices.append(np.arange(start, stop))

            return xlabels, labels, x, edges, indices

        xlabels, labels, x, edges, indices = zip(*[_get(observable) for observable in [self.observable, self.theory]])
        fshape = tuple(map(len, x))
        if fig is None:
            fig, lax = plt.subplots(*fshape, sharex=False, sharey=False, figsize=(8, 6), squeeze=False)
        else:
            lax = np.array(fig.axes).reshape(fshape[::-1])

        axes = {'o': 0, 'observable': 0, 't': 1, 'theory': 1}
        assert axis in axes, f'axis must be one of {list(axes)}'
        iaxis = axes[axis]

        for it, xt in enumerate(x[1]):
            for io, xo in enumerate(x[0]):
                value = self._value[np.ix_(indices[0][io], indices[1][it])]
                for ix, idx in enumerate(idxs):
                    ii = [io, it][iaxis]
                    plotted_ii = [io, it][iaxis - 1]
                    iidx = idx
                    if np.issubdtype(idx.dtype, np.floating):
                        iidx = np.abs(x[iaxis][ii] - idx).argmin()
                    # Indices in approximate window matrix
                    xx = x[iaxis - 1][plotted_ii]
                    dx = 1.
                    if iaxis == 0:  # axis = 'o', showing theory, dividing by integration element dx
                        dx = edges[iaxis - 1][plotted_ii]
                        dx = dx[..., 1] - dx[..., 0]
                        if dx.ndim >= 1:  # e.g. bispectrum
                            dx = dx.prod(axis=-1)
                    v = np.take(value, iidx, axis=iaxis)
                    v = v / dx
                    if yscale == 'log': v = np.abs(v)
                    ax = lax[io][it]
                    ax.plot(xx, v, alpha=alphas[ix], color=color, label=label if ix == 0 else None)
                if labels[1][it] or labels[0][io]:
                    ax.set_title(r'${} \times {}$'.format(labels[1][it], labels[0][io]))
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
                ax.grid(True)
                if label and it == io == 0: lax[it][io].legend()

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.35, wspace=0.25)
        return fig


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

    _name = 'covariance_matrix'

    def __init__(self, value, observable, attrs=None):
        self._value = value
        self._observable = observable
        self._attrs = dict(attrs or {})

    @property
    def attrs(self):
        """Other attributes."""
        return self._attrs

    @property
    def shape(self):
        """Matrix shape."""
        return self._value.shape

    def value(self):
        """Value (numpy array) of the covariance matrix."""
        return self._value

    def __array__(self):
        return np.asarray(self.value())

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
            if name in ['attrs']:
                setattr(new, '_' + name, dict(value or {}))
            elif name in ['observable', 'value']:
                setattr(new, '_' + name, value)
            else:
                raise ValueError(f'Unknown attribute {name}')
        return new

    def __getstate__(self, to_file=False):
        state = {}
        state['name'] = self._name
        state['value'] = self.value()
        state['attrs'] = dict(self.attrs)
        for name in ['observable']:
            state[name] = getattr(self, name).__getstate__(to_file=to_file)
        return state

    def __setstate__(self, state):
        self._value = state['value']
        self._attrs = state.get('attrs', {})
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


    @classmethod
    def sum(cls, matrices, weights=None):
        """
        Sum multiple covariance matrices.
        The result would be the covariance on the sum of observables,
        if they were independent.

        WARNING
        -------
        Assumes observables are independent.
        """
        new = matrices[0].copy()

        def get_sumweight(leaves):
            sumweight = getattr(leaves[0], '_sumweight', None)
            weight = None
            if sumweight is not None:
                weight = sumweight(leaves, weights=weights)
            if weight is None:
                weight = [np.ones_like(leaves[0].value()) / len(leaves)] * len(leaves)
            return weight

        assert all(matrix.observable.labels(level=None, return_type='flatten') == new.observable.labels(level=None, return_type='flatten') for matrix in matrices[1:])
        weights = list(map(get_sumweight, zip(*[tree_flatten(matrix.observable, level=None) for matrix in matrices])))
        weights = [np.concatenate(w, axis=0) for w in zip(*weights)]
        weights2 = [weight[..., None] * weight for weight in weights]
        new._value = sum(weight2 * matrix._value for matrix, weight2 in zip(matrices, weights2))
        new._observable = matrices[0]._observable.sum([matrix.observable for matrix in matrices])
        return new
    
    @utils.plotter
    def plot(self, level=None, corrcoef=False, **kwargs):
        """
        Plot covariance matrix.

        Parameters
        ----------
        level : int, optional
            Level in tree at which define different panels. Default to the maximum depth.
        corrcoef : bool, option
            If ``True``, plot correlation matrix.
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
                    xx = leaf.coords(0)
                    if xx.ndim > 1: xx = None
                if xx is None:
                    xx = np.arange(leaf.size)
                return xx

            if observable._is_leaf:
                x.append(get_x(observable))
                indices.append(np.arange(observable.size))
            else:
                for label in observable.labels(level=level, return_type='flatten'):
                    leaf = observable.get(**label)
                    labels.append(','.join([observable._label_to_str(v) for v in label.values()]))
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

        value = self._value
        if corrcoef:
            std = self.std()
            value = value / (std[..., None] * std)
        mat = [[value[np.ix_(index1, index2)] for index2 in indices[1]] for index1 in indices[0]]
        return utils.plot_matrix(mat, x1=x[0], x2=x[1], **kwargs)

    @utils.plotter
    def plot_diag(self, offset=0, level=None, color='C0', xscale='linear', yscale='linear', ytransform=None, fig=None):
        """
        Plot diagonal (and optionally offset diagonals) of the covariance matrix.

        Parameters
        ----------
        offset : int or array-like, default=0
            Offset(s) from the main diagonal to plot. Can be a single integer or a list of offsets.
        level : int, optional
            Level in tree at which define different panels. Default to the maximum depth.
        color : str, default='C0'
            Color for the plotted lines.
        xscale : str, default='linear'
            X-axis scale ('linear', 'log', etc.).
        yscale : str, default='linear'
            Y-axis scale ('linear', 'log', etc.).
        ytransform : callable, optional
            Function to transform diagonal covariance values before plotting.
        fig : matplotlib.figure.Figure, optional
            Figure to plot into. If None, a new figure is created.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        """
        import itertools
        from matplotlib import pyplot as plt
        offsets = np.atleast_1d(offset)
        alphas = np.linspace(1, 0.2, len(offsets))

        def _get():
            xlabels, labels, x, indices = [], [], [], []
            observable = self.observable

            def get_x(leaf):
                xx = None
                if len(leaf._coords_names) == 1:
                    xx = leaf.coords(0)
                    if xx.ndim > 1: xx = None
                if xx is None:
                    xx = np.arange(leaf.size)
                return xx

            if observable._is_leaf:
                x.append(get_x(observable))
                labels.append('')
                indices.append(np.arange(observable.size))
            else:
                for label in observable.labels(level=level, return_type='flatten'):
                    leaf = observable.get(**label)
                    labels.append(','.join([observable._label_to_str(v) for v in label.values()]))
                    x.append(get_x(leaf))
                    start, stop = _get_range_in_tree(observable, observable._index_labels(label)[0])
                    indices.append(np.arange(start, stop))

            return xlabels, labels, x, indices

        xlabels, labels, x, indices = _get()
        fshape = (len(x),) * 2
        if fig is None:
            fig, lax = plt.subplots(*fshape, sharex=False, sharey=False, figsize=(8, 6), squeeze=False)
        else:
            lax = np.array(fig.axes).reshape(fshape[::-1])

        for i1, i2 in itertools.product(*[list(range(s)) for s in fshape]):
            value = self._value[np.ix_(indices[i1], indices[i2])]
            for offset, alpha in zip(offsets, alphas):
                index = np.arange(max(min(x[i].size - offset for i in [i1, i2]), 0))
                flag = int(i2 > i1)
                index1, index2 = index, index + offset
                diag = value[index1, index2]
                xx = x[i2 if flag else i1][index]
                if ytransform is not None: diag = ytransform(xx, diag)
                label = None
                if i1 == i2 == 0: label = r'$\mathrm{{offset}} = {:d}$'.format(offset)
                ax = lax[i2, i1]
                if labels[i1] or labels[i2]: ax.set_title(r'${} \times {}$'.format(labels[i1], labels[i2]))
                ax.plot(xx, diag, alpha=alpha, color=color, label=label)
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
                ax.grid(True)
                if i1 == i2 == 0 and len(offsets) > 1: ax.legend()

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.35, wspace=0.25)
        return fig

    @utils.plotter
    def plot_slice(self, indices, level=None, color='C0', label=None, xscale='linear', yscale='log', fig=None):
        """
        Plot a slice of the covariance matrix along the specified indices.

        Parameters
        ----------
        indices : int, array-like
            Indices (or values) along which to slice the covariance matrix. Can be a single index or a list.
        level : int, optional
            Level in tree at which define different panels. Default to the maximum depth.
        color : str, default='C0'
            Color for the plotted lines.
        label : str, optional
            Label for the plotted lines.
        xscale : str, default='linear'
            X-axis scale ('linear', 'log', etc.).
        yscale : str, default='log'
            Y-axis scale ('linear', 'log', etc.).
        fig : matplotlib.figure.Figure, optional
            Figure to plot into. If None, a new figure is created.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        """
        import itertools
        from matplotlib import pyplot as plt
        if np.ndim(indices) == 0: indices = [indices]
        idxs = np.array(indices)
        alphas = np.linspace(1, 0.2, len(indices))

        def _get():
            xlabels, labels, x, indices = [], [], [], []
            observable = self.observable

            def get_x(leaf):
                xx = None
                if len(leaf._coords_names) == 1:
                    xx = leaf.coords(0)
                    if xx.ndim > 1: xx = None
                if xx is None:
                    xx = np.arange(leaf.size)
                return xx

            if observable._is_leaf:
                x.append(get_x(observable))
                labels.append('')
                indices.append(np.arange(observable.size))
            else:
                for label in observable.labels(level=level, return_type='flatten'):
                    leaf = observable.get(**label)
                    labels.append(','.join([observable._label_to_str(v) for v in label.values()]))
                    x.append(get_x(leaf))
                    start, stop = _get_range_in_tree(observable, observable._index_labels(label)[0])
                    indices.append(np.arange(start, stop))

            return xlabels, labels, x, indices

        xlabels, labels, x, indices = _get()
        fshape = (len(x),) * 2
        if fig is None:
            fig, lax = plt.subplots(*fshape, sharex=False, sharey=False, figsize=(8, 6), squeeze=False)
        else:
            lax = np.array(fig.axes).reshape(fshape[::-1])

        for i1, i2 in itertools.product(*[list(range(s)) for s in fshape]):
            value = self._value[np.ix_(indices[i1], indices[i2])]
            ax = lax[i2, i1]
            for ix, idx in enumerate(idxs):
                iidx = idx
                if np.issubdtype(idx.dtype, np.floating):
                    iidx = np.abs(x[i1] - idx).argmin()
                v = np.take(value, iidx, axis=0)
                if yscale == 'log': value = np.abs(v)
                xx = x[i2]
                ax.plot(xx, v, alpha=alphas[ix], color=color, label=label if ix == 0 else None)
            if labels[i1] or labels[i2]: ax.set_title(r'${} \times {}$'.format(labels[i1], labels[i2]))
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.grid(True)
            if label and i1 == i2 == 0: lax[i1][i2].legend()

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.35, wspace=0.25)
        return fig


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

    _name = 'gaussian_likelihood'

    def __init__(self, observable, window, covariance, attrs=None):
        self._observable = observable
        self._window = window
        self._covariance = covariance
        self._attrs = dict(attrs or {})

    @property
    def attrs(self):
        """Other attributes."""
        return self._attrs

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
            if name in ['attrs']:
                setattr(new, '_' + name, dict(value or {}))
            elif name in ['observable', 'window', 'covariance']:
                setattr(new, '_' + name, value)
            else:
                raise ValueError(f'Unknown attribute {name}')
        return new

    def __getstate__(self, to_file=False):
        state = {}
        state['name'] = self._name
        state['attrs'] = dict(self._attrs)
        for name in ['observable', 'window', 'covariance']:
            state[name] = getattr(self, name).__getstate__(to_file=to_file)
        return state

    def __setstate__(self, state):
        self._attrs = state.get('attrs', {})
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