import os
import shutil

from .utils import mkdir

import numpy as np


def _h5py_recursively_write_dict(h5file, path, dic, with_attrs=True):
    """
    Save a nested dictionary of arrays to an HDF5 file using h5py.
    """
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
    """
    Save a nested dictionary of arrays to an HDF5 file using h5py.
    """
    mkdir(path)
    for key, item in dic.items():
        path_key = os.path.join(path, key)
        if with_attrs and key == 'attrs':
            import json
            with open(path_key + '.json', 'w') as file:
                json.dump(item, file)
            continue  # handle attrs below

        if isinstance(item, dict):
            # If dict has 'attrs' and other keys, it's a group with metadata
            mkdir(path_key)
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
    filename = str(filename)
    mkdir(os.path.dirname(filename))
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
    if not isinstance(masks, tuple): masks = (masks,)
    alls = [np.arange(s) for s in shape]
    masks = masks + (Ellipsis,) * (len(shape) - len(masks))
    masks = tuple(a[m] for a, m in zip(alls, masks))
    return masks


def _tensor_product(*arrays):
    reshaped = [array.reshape((1,)*i + (-1,) + (1,)*(len(arrays)-i-1))
                for i, array in enumerate(arrays)]
    out = reshaped[0]
    for r in reshaped[1:]:
        out = out * r
    return out


_registry = {}


def register_type(cls):
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


def _build_matrix_from_mask(mask, toarray=False):
    idx = np.flatnonzero(mask)
    # Build sparse selection matrix
    nout = len(idx)
    nin = len(mask)
    try:
        import scipy.sparse as sp

        def get():
            toret = sp.csr_matrix((np.ones(nout, dtype=int), (np.arange(nout), idx)), shape=(nout, nin))
            if toarray:
                return toret.toarray()
            return toret

    except ImportError:

        def get():
            matrix = np.zeros((nout, nin), dtype=int)
            matrix[np.arange(nout), idx] = 1
            return matrix

    return get()


def _nan_to_zero(array):
    return np.where(np.isnan(array), 0., array)


@register_type
class ObservableLeaf(object):
    """A compressed observable with named values and coordinates, supporting slicing, selection, and plotting."""

    _name = 'leaf_base'
    _forbidden_names = ('name', 'attrs', 'values_names', 'coords_names', 'meta')
    _is_leaf = True
    _binweight = None
    _rebin_weighted_normalized_names = None
    _rebin_weighted_names = None

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
        self._attrs = dict(attrs or {})
        self._meta = dict(meta or {})
        self._data = dict(data)
        self._coords_names = list(coords)
        assert not any(k in self._forbidden_names for k in self._data), f'Cannot use {self._forbidden_names} as name for arrays'
        _edges_names = [f'{axis}_edges' for axis in self._coords_names]
        self._values_names = [name for name in data if name not in self._coords_names and name not in _edges_names]
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
        Get edge arrays.

        Parameters
        ----------
        axis : str or int, optional
            Name or index of coordinate.

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
        return self._data[self._values_names[0]]

    def _update_value(self, value):
        self._data[self._values_names[0]] = value

    def __array__(self):
        return self.value()

    @property
    def shape(self):
        return self._data[self._values_names[0]].shape

    @property
    def size(self):
        return self._data[self._values_names[0]].size

    @property
    def ndim(self):
        return len(self._coords_names)

    @property
    def attrs(self):
        return self._attrs

    def __getitem__(self, masks):
        """
        Mask or slice the observable.

        Parameters
        ----------
        mask : array-like, bool or slice
            Mask or slice to apply to all value and coordinate arrays.

        Returns
        -------
        ObservableLeaf
        """
        masks = _format_masks(self.shape, masks)
        mask = np.ix_(*masks)
        new = self.copy()
        for name in self._values_names:
            new._data[name] = self._data[name][mask]
        for axis, mask in zip(self._coords_names, masks):
            new._data[axis] = new._data[axis][mask]
            axis_edges = f'{axis}_edges'
            if axis_edges in new._data:
                new._data[axis_edges] = new._data[axis_edges][mask]
        return new

    def _transform(self, limit, axis=0, weighted=True, normalize=True, full=None, return_edges=False, center='mid_if_edges'):
        # Return mask or matrix
        if not isinstance(axis, str):
            axis = self._coords_names[axis]
        if limit is None:
            size = len(self._data[axis])
            mask = np.ones(size, dtype='?')
            if return_edges:
                return mask, self.edges(axis=axis)
            return mask

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
            if stop < 0: stop = stop + size
            stop = min((size - start) // step * step, stop)
            #start, stop, step = sl.indices(len(self._x[iproj]))
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
            else:
                mask = _isin2d(_self_coords, limit)
                assert np.allclose(_self_coords[mask], limit), 'need to handle the fact that coords may not be order the same way, please file a github issue'
            if return_edges:
                edges = self.edges(axis=axis)
                if edges is not None: edges = edges[mask]
                return mask, edges
            return mask

        def get_unique_edges(edges):
            return [np.unique(edges[:, iax], axis=0) for iax in range(edges.shape[1])]

        def get_1d_slice(edges, index):
            if isinstance(index, slice):
                edges1 = edges[index, 0]
                edges2 = edges[index.start + index.step - 1:index.stop + index.step - 1:index.step, 1]
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

        if weighted:
            weight = self.binweight(default=None)
            if len(shape) > 1:
                if full or full is None:
                    mask = _build_big_tensor(mask, shape, axis=iaxis)[0]
                    weight = np.ravel(weight) if weight is not None else 1
                elif weight is None:
                    pass
                else:
                    weight = np.sum(weight, axis=tuple(iax for iax in range(weight.ndim) if iax != iaxis))
            matrix = multiply(mask, weight)
        else:
            if full and len(shape) > 1:
                mask = _build_big_tensor(mask, shape, axis=iaxis)[0]
            matrix = mask * 1
        if normalize:
            # all isn't implemented for scipy sparse, just check the sum of the boolean array
            norm = 1 / np.ravel(np.where((matrix != 0).sum(axis=-1) == 0, 1, matrix.sum(axis=-1)))[:, None]
            matrix = multiply(matrix, norm)
        if return_edges:
            return matrix, edges
        return matrix

    def binweight(self, default=None):
        """Get binning weights."""
        if self._binweight is None:
            if default is None:
                return None
            return np.ones_like(self.value(), dtype='i4')
        return self._data[self._binweight]

    def clone(self, **kwargs):
        """Copy and update data."""
        new = self.copy()
        for name, value in kwargs.items():
            if name == 'value':
                new._update_value(value)
                continue
            if name not in new._data: raise ValueError('{name} not unknown')
            new._data[name] = value
        new._data.update(**kwargs)
        return new

    def select(self, center='mid_if_edges', **limits):
        """
        Select a range in one or more coordinates.

        Parameters
        ----------
        limits : dict
            Each key is a coordinate name, value is (min, max) tuple.

        Returns
        -------
        ObservableLeaf
        """
        rebin_weighted_normalized_names = self._rebin_weighted_normalized_names
        rebin_weighted_names = self._rebin_weighted_names
        if rebin_weighted_normalized_names is None:
            rebin_weighted_normalized_names = set(self._values_names) - set([self._binweight] if self._binweight is not None else [])
        if rebin_weighted_names is None:
            rebin_weighted_names = list(set(self._values_names) - set(rebin_weighted_normalized_names) - set([self._binweight] if self._binweight is not None else []))

        new = self.copy()
        for iaxis, axis in enumerate(self._coords_names):
            limit = limits.pop(axis, None)
            if limit is None: continue
            axis_edges = f'{axis}_edges'
            transform, edges = new._transform(limit, axis=axis, weighted=False, normalize=False, return_edges=True, center=center)
            if transform.ndim == 1:  # mask
                mask = transform
                for name in new._values_names:
                    new._data[name] = np.take(new._data[name], np.flatnonzero(mask), axis=iaxis)
                new._data[axis] = new._data[axis][mask]
                if axis_edges in new._data:
                    new._data[axis_edges] = new._data[axis_edges][mask]
            else:  # matrix
                matrix = transform
                nwmatrix_reduced = new._transform(limit, axis=axis, weighted=True, normalize=True, full=False)
                if rebin_weighted_normalized_names:
                    nwmatrix = new._transform(limit, axis=axis, weighted=True, normalize=True)
                if rebin_weighted_names:
                    wmatrix = new._transform(limit, axis=axis, weighted=True, normalize=False)
                tmp = _nan_to_zero(new._data[axis])
                new._data[axis] = np.tensordot(nwmatrix_reduced, tmp, axes=([1], [0]))
                if axis_edges in new._data:
                    new._data[axis_edges] = edges
                for name in new._values_names:
                    tmp = _nan_to_zero(new._data[name])
                    if name in rebin_weighted_normalized_names: _matrix = nwmatrix
                    elif name in rebin_weighted_names: _matrix = wmatrix
                    else: _matrix = matrix
                    if _matrix.shape[1] == tmp.shape[iaxis]:  # compressed version
                        new._data[name] = np.moveaxis(np.tensordot(_matrix, tmp, axes=([1], [iaxis])), 0, iaxis)
                    else:
                        shape = tuple(len(new._data[axis]) for axis in new._coords_names)
                        new._data[name] = _matrix.dot(tmp.ravel()).reshape(shape)
        return new

    def match(self, observable):
        """Match coordinates to those of input observable."""
        return self.select(**{axis: observable for axis in self._coords_names})

    @property
    def at(self):
        return _ObservableLeafUpdateHelper(self)

    def copy(self):
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
        if to_file: state['value'] = self.value()
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
    def sum(cls, observables):
        new = observables[0].copy()
        for name in new._values_names:
            weights = [getattr(observable, '_sumweight', 1) for observable in observables]
            new._data[name] = sum(weight * observable._data[name] for observable, weight in zip(observables, weights)) / sum(weights)
        return new

    @classmethod
    def mean(cls, observables):
        new = observables[0].copy()
        for name in new._values_names:
            new._data[name] = sum(observable._data[name] for observable in observables) / len(observables)
        return new

    @classmethod
    def concatenate(cls, observables, axis=0):
        """
        Concatenate multiple observables.
        No check performed.
        """
        new = observables[0].copy()
        if not isinstance(axis, str):
            axis = new._coords_names[axis]
        iaxis = new._coords_names.index(axis)
        new._data[axis] = np.concatenate([observable._data[axis] for observable in observables], axis=0)
        for name in new._values_names:
            new._data[name] = np.concatenate([observable._data[name] for observable in observables], axis=iaxis)
        return new


def find_single_true_slab_bounds(mask):
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
        masks = _format_masks(self._observable.shape, masks)
        return _ObservableLeafUpdateRef(self._observable, masks, self._hook)

    def __call__(self, **limits):
        masks = []
        for axis in self._observable._coords_names:
            transform = self._observable._transform(limits.pop(axis, None), axis=axis)
            assert transform.ndim == 1, 'Only limits (min, max) are supported'
            masks.append(transform)
        return _ObservableLeafUpdateRef(self._observable, masks, self._hook)


def _pad_transform(transform, start, size):
    if transform.ndim == 1:
        mask = np.ones(size, dtype='?')
        mask[start:start + transform.size] = transform.ravel()
        return mask

    try:
        import scipy.sparse as sp
    except ImportError:
        sp = None

    stop = start + transform.shape[1]
    if sp is None:
        matrix = np.zeros((start + transform.shape[0] + (size - stop), size), dtype=transform.dtype)
        matrix[np.arange(start), np.arange(start)] = 1
        matrix[np.ix_(np.arange(start, start + transform.shape[0]), np.arange(start, stop))] = transform
        matrix[np.arange(size - stop, size), np.arange(size - stop, size)] = 1
    else:
        matrix = sp.block_diag([sp.identity(start, dtype=transform.dtype, format='csr'),
                                sp.csr_matrix(transform),
                                sp.identity(size - stop, dtype=transform.dtype, format='csr')], format='csr')
    return matrix


def _join_transform(cum_transform, transform):
    if cum_transform is None:
        return transform
    else:
        if cum_transform.ndim < transform.ndim:
            cum_transform = _build_matrix_from_mask(cum_transform)
        elif cum_transform.ndim > transform.ndim:
            transform = _build_matrix_from_mask(transform)
        if cum_transform.ndim == 2:
            cum_transform = transform.dot(cum_transform)
        else:
            cum_transform = cum_transform.copy()
            cum_transform[cum_transform] = transform
        return cum_transform


def _concatenate_transforms(transforms, starts, size):
    assert len(transforms) == len(starts)
    is2d = any(transform.ndim == 2 for transform in transforms)
    if is2d:
        transforms = [_build_matrix_from_mask(transform) if transform.ndim < 2 else transform for transform in transforms]
        try:
            import scipy.sparse as sp
        except ImportError:
            sp = None
        if sp is None:
            def _pad(transform, start, size):
                toret = np.zeros_like(transform, shape=(transform.shape[0], size))
                toret[:, start:start + transform.shape[1]] = transform
                return toret

            transforms = [_pad(transform, start, size) for start, transform in zip(starts, transforms)]
            matrix = np.concatenate(transforms, axis=0)
        else:
            def _pad(transform, start, size):
                m = [sp.csr_matrix((transform.shape[0], start)),
                     sp.csr_matrix(transform),
                     sp.csr_matrix((transform.shape[0], size - start - transform.shape[1]))]
                return sp.hstack(m)

            transforms = [_pad(transform, start, size) for start, transform in zip(starts, transforms)]
            matrix = sp.vstack(transforms)

        return matrix
    else:
        transforms = [np.flatnonzero(transform) if not np.issubdtype(transform.dtype, np.integer) else transform for transform in transforms]
        transforms = [start + transform for start, transform in zip(starts, transforms)]
        return np.concatenate(transforms, axis=0)


class _ObservableLeafUpdateRef(object):

    def __init__(self, observable, masks=None, hook=None):
        self._observable = observable
        if masks is None:
            self._limits = tuple((0, s) for s in self._observable.shape)
        else:
            self._limits = tuple(find_single_true_slab_bounds(mask) for mask in masks)
        self._hook = hook
        assert len(self._limits) == self._observable.ndim

    def select(self, center='mid_if_edges', **limits):
        new = self._observable.copy()
        cum_transform = None
        for iaxis, axis in enumerate(self._observable._coords_names):
            if axis not in limits: continue

            def _make_mask(mask):
                masks = [np.ones(s, dtype='?') for s in new.shape]
                if isinstance(mask, slice):
                    masks[iaxis][...] = False
                    masks[iaxis][mask] = True
                else:
                    masks[iaxis] = mask
                return _tensor_product(*masks)

            start, stop = self._limits[iaxis]
            sub = new.select(**{axis: slice(start, stop)})

            sub_transform = sub._transform(limit=limits[axis], axis=axis, center=center, full=True if self._hook is not None else None)
            sub = sub.select(**{axis: limits[axis]}, center=center)
            sub._data[axis] = np.concatenate([new.coords(axis)[:start], sub.coords(axis), new.coords(axis)[stop:]], axis=0)
            axis_edges = f'{axis}_edges'
            if axis_edges in new._data:
                sub._data[axis_edges] = np.concatenate([self._observable.edges(axis)[:start], sub.edges(axis), self._observable.edges(axis)[stop:]], axis=0)
            shape = tuple(len(sub._data[axis]) for axis in sub._coords_names)

            new_mask1d = np.zeros(new.shape[iaxis], dtype='?')
            new_mask1d[start:stop] = True
            sub_mask1d = np.zeros(shape[iaxis], dtype='?')
            sub_mask1d[start:shape[iaxis]-(new.shape[iaxis] - stop)] = True

            if self._hook is not None:
                if sub_transform.ndim == 1:
                    mask1d = np.ones(new.shape[iaxis], dtype='?')
                    mask1d[start:stop] = sub_transform
                    transform = _make_mask(mask1d).ravel()
                else:
                    if len(shape) == 1:
                        transform = sub_transform.dot(_build_matrix_from_mask(new_mask1d, toarray=True))
                    else:
                        # with hook
                        transform = sub_transform.dot(_build_matrix_from_mask(_make_mask(slice(start, stop)).ravel()))

            def put(value, mask, array, axis=iaxis):
                masks = [np.ones(s, dtype='?') for s in new.shape]
                masks[axis] = mask
                value[np.ix_(*masks)] = array

            for name in sub._values_names:
                tmp = _nan_to_zero(new._data[name])
                value = np.zeros_like(sub._data[name], shape=shape)
                put(value, np.flatnonzero(~sub_mask1d), np.take(tmp, np.flatnonzero(~new_mask1d), axis=iaxis), axis=iaxis)
                put(value, np.flatnonzero(sub_mask1d), sub._data[name], axis=iaxis)
                sub._data[name] = value

            if self._hook is not None:
                cum_transform = _join_transform(cum_transform, transform)
            new = sub

        if self._hook is not None:
            return self._hook(new, transform=cum_transform)
        return new

    def match(self, observable):
        """Match coordinates to those of input observable."""
        return self.select(**{axis: observable for axis in self._observable._coords_names})


def _iter_on_tree(f, tree, level=None):
    if level == 0 or tree._is_leaf:
        return [f(tree)]
    toret = []
    for tree in tree._leaves:
        toret += _iter_on_tree(f, tree, level=level - 1 if level is not None else None)
    return toret


def _get_leaf(tree, index=None):
    if index is None:
        return tree
    toret = tree._leaves[index[0]]
    if len(index) == 1:
        return toret
    return _get_leaf(toret, index[1:])


def _format_input_labels(self, *args, **labels):
    if args:
        assert not labels, 'Cannot provide both list and dict of labels'
        assert len(args) == 1 and len(self._labels) == 1, 'Args mode available only for one label entry'
        labels = {next(iter(self._labels)): args[0]}
    return labels


def _flatten_index_labels(indices):
    toret = []
    for index, value in indices.items():
        if value is None:
            toret.append((index,))
        else:
            for flat_index in _flatten_index_labels(index):
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

    @classmethod
    def _label_to_str(cls, label):
        import numbers
        if isinstance(label, numbers.Number):
            return str(label)
        if isinstance(label, str):
            for char in ['_', cls._sep_strlabels]:
                if char in label:
                    raise ValueError(f'Label cannot contain "{char}"')
            return label
        if isinstance(label, tuple):
            if len(label) == 1: raise ValueError('Tuples must be of length > 1')
            return '_'.join([cls._label_to_str(lbl) for lbl in label])
        raise NotImplementedError(f'Unable to safely cast {label} to string. Implement "_label_to_str" and "_str_to_label".')

    @classmethod
    def _str_to_label(cls, str, squeeze=True):
        splits = list(str.split('_'))
        for i, split in enumerate(splits):
            try:
                splits[i] = int(split)
            except ValueError:
                pass
        if squeeze and len(splits) == 1:
            return splits[0]
        return tuple(splits)

    @property
    def attrs(self):
        return self._attrs

    def labels(self, level=None, keys_only=False, as_str=False):
        """
        Return a list of dicts with the labels for each leaf.

        Returns
        -------
        labels : list of dict
        """
        toret = []
        if keys_only:
            for ileaf, leaf in enumerate(self._leaves):
                toret += [label for label in self._labels if label not in toret]
                if level == 0 or leaf._is_leaf:
                    pass
                else:
                    for label in leaf.labels(level=level - 1 if level is not None else None, keys_only=keys_only, as_str=as_str):
                        if label not in toret: toret.append(label)
        else:
            for ileaf, leaf in enumerate(self._leaves):
                self_labels = {k: v[ileaf] for k, v in (self._strlabels if as_str else self._labels).items()}
                if level == 0 or leaf._is_leaf:
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
        # Follows the original order
        def find(vselect, k):
            if isinstance(vselect, list):
                return sum((find(vs, k) for vs in vselect), start=[])
            if isinstance(vselect, str):
                return [i for i, v in enumerate(self._strlabels[k]) if v == vselect]
            return [i for i, v in enumerate(self._labels[k]) if v == vselect]

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
        """Return leave(s) corresponding to input labels."""
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
        return _iter_on_tree(lambda leaf: leaf.size, self, level=level)

    @property
    def size(self):
        return sum(self.sizes(level=1))

    def __iter__(self):
        return iter(self._leaves)

    def select(self, **limits):
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
        Get (flattened) values from all leaves.

        Parameters
        ----------
        name : str, optional
            Name of value.
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

    @classmethod
    def concatenate(cls, others):
        assert len(others) >= 1, f'Provide at least 1 {cls.__name__} to concatenate'
        leaves, labels = [], others[0]._labels
        for other in others:
            assert isinstance(other, ObservableTree)
            leaves += other._leaves
            assert set(other._labels) == set(labels), 'All collections must have same labels'
            for k in labels:
                labels[k] = labels[k] + other._labels[k]
        new = cls.__new__(cls)  # produce the correct type
        ObservableTree.__init__(new, leaves, **labels)
        return new

    @property
    def at(self):
        return _ObservableTreeUpdateHelper(self)

    def copy(self):
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
        labels = _format_input_labels(self._tree, *args, **labels)
        indices = self._tree._index_labels(labels)
        assert len(indices), f'Nothing found with {labels}'
        if len(indices) == 1 and _get_leaf(self._tree, indices[0])._is_leaf:
            # Single leaf
            return _ObservableTreeUpdateSingleRef(self._tree, indices[0], hook=self._hook)
        # Sub-tree
        return _ObservableTreeUpdateRef(self._tree, indices, hook=self._hook)


def _get_range_in_tree(tree, index):
    start = 0
    current_tree = tree
    for idx in index:
        start += sum(leaf.size for leaf in current_tree._leaves[:idx])
        current_tree = current_tree._leaves[idx]
    return start, start + current_tree.size


def _replace_in_tree(tree, index, sub):
    start = 0
    current_tree = tree
    for idx in index[:-1]:
        start += sum(leaf.size for leaf in current_tree._leaves[:idx])
        current_tree = current_tree._leaves[idx]
    start += sum(leaf.size for leaf in current_tree._leaves[:index[-1]])
    stop = start + current_tree._leaves[index[-1]].size
    current_tree._leaves[index[-1]] = sub
    return start, stop


class _ObservableTreeUpdateSingleRef(object):

    _tree: ObservableTree
    _index: tuple

    def __init__(self, tree, index, hook=None):
        self._tree = tree
        self._index = index
        self._hook = hook

    def __getitem__(self, masks):
        leaf = _get_leaf(self._tree, self._index)
        masks = _format_masks(leaf.shape, masks)
        sub = leaf.__getitem__(masks)
        new = self._tree.copy()
        start, stop = _replace_in_tree(new, self._index, sub)
        if self._hook:
            transform = np.logical_and.reduce(np.ix_(*masks)).ravel()
            return self._hook(new, transform=_pad_transform(transform, start, self._tree.size))
        return new

    def select(self, **limits):
        hook = None
        if self._hook:
            def hook(leaf, transform): return leaf, transform
        leaf = _get_leaf(self._tree, self._index)
        leaf = _ObservableLeafUpdateRef(leaf, masks=None, hook=hook).select(**limits)
        if self._hook:
            leaf, transform = leaf
        new = self._tree.copy()
        start, stop = _replace_in_tree(new, self._index, leaf)
        if self._hook:
            return self._hook(new, transform=_pad_transform(transform, start, self._tree.size))
        return new

    def match(self, observable):
        hook = None
        if self._hook:
            def hook(leaf, transform): return leaf, transform
        leaf = _get_leaf(self._tree, self._index)
        leaf = _ObservableLeafUpdateRef(leaf, masks=None, hook=hook).match(observable)
        if self._hook:
            leaf, transform = leaf
        new = self._tree.copy()
        start, stop = _replace_in_tree(new, self._index, leaf)
        if self._hook:
            return self._hook(new, transform=_pad_transform(transform, start, self._tree.size))
        return new

    @property
    def at(self):
        at = _get_leaf(self._tree, self._index).at

        def hook(leaf, transform=None):
            new = self._tree.copy()
            start, stop = _replace_in_tree(new, self._index, leaf)
            if self._hook is not None:
                return self._hook(new, transform=_pad_transform(transform, start, self._tree.size))
            return new

        at._hook = hook
        return at


class _ObservableTreeUpdateRef(object):

    def __init__(self, tree, indices=None, hook=None):
        self._tree = tree
        self._indices = indices
        self._hook = hook

    def get(self, *args, **labels):
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
            return self._hook(new, transform=mask)
        return new

    def select(self, **limits):
        hook = None
        if self._hook:
            def hook(leaf, transform): return leaf, transform

        new = self._tree.copy()
        transform = None
        for index in (self._indices if self._indices is not None else self._tree._index_labels({})):
            leaf = _get_leaf(self._tree, index)
            if isinstance(leaf, ObservableLeaf):
                leaf = _ObservableLeafUpdateRef(leaf, hook=hook).select(**limits)
            else:
                leaf = _ObservableTreeUpdateRef(leaf, hook=hook).select(**limits)
            if self._hook:
                leaf, _transform = leaf
            size = new.size
            start, stop = _replace_in_tree(new, index, leaf)
            if self._hook:
                _transform = _pad_transform(_transform, start, size)
                transform = _join_transform(transform, _transform)

        if self._hook:
            return self._hook(new, transform=transform)
        return new

    def match(self, observable):
        hook = None
        if self._hook:
            def hook(leaf, transform): return leaf, transform

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
            if isinstance(_leaf, ObservableLeaf):
                leaf = _ObservableLeafUpdateRef(_leaf, hook=hook).match(leaf)
            else:
                leaf = _ObservableTreeUpdateRef(_leaf, hook=hook).match(leaf)
            if self._hook:
                leaf, _transform = leaf
            leaves.append(leaf)
            ileaves.append(_ileaf)
            size = tree.size
            if self._hook:
                start, stop = _get_range_in_tree(tree, (_ileaf,))
                transforms.append(_transform)
                starts.append(start)

        if self._hook:
            transform = _concatenate_transforms(transforms, starts, tree.size)
        tree = tree.copy()
        tree._leaves = leaves
        tree._labels = {k: [v[idx] for idx in ileaves] for k, v in tree._labels.items()}
        tree._strlabels = {k: [v[idx] for idx in ileaves] for k, v in tree._strlabels.items()}
        new = tree
        if index is not None:
            new = self._tree.copy()
            start, stop = _replace_in_tree(new, (index,), tree)
            if self._hook:
                transform = _pad_transform(transform, start, self._tree.size)
        if self._hook:
            return self._hook(new, transform=transform)
        return new


class _WindowMatrixUpdateHelper(object):

    def __init__(self, matrix):
        self._matrix = matrix

    @property
    def observable(self):
        return _ObservableWindowMatrixUpdateHelper(self._matrix, axis=0)

    @property
    def theory(self):
        return _ObservableWindowMatrixUpdateHelper(self._matrix, axis=1)


class _ObservableWindowMatrixUpdateHelper(object):

    def __init__(self, matrix, axis=0):
        self._matrix = matrix
        self._axis = axis
        self._observable = [matrix._observable, matrix._theory][self._axis]

    def _select(self, observable, transform):
        _observable_name = ['observable', 'theory'][self._axis]
        if transform.ndim == 1:  # mask
            value = np.take(self._matrix.value(), np.flatnonzero(transform), axis=self._axis)
        else:
            if self._axis == 0: value = transform.dot(self._matrix.value())
            else: value = transform.dot(self._matrix.value().T).T  # because transform can be a sparse matrix; works in all cases
        kw = {_observable_name: observable}
        return self._matrix.clone(value=value, **kw)

    def match(self, observable):
        def hook(observable, transform):
            return observable, transform
        observable, transform = (_ObservableLeafUpdateRef if isinstance(self._observable, ObservableLeaf) else _ObservableTreeUpdateRef)(self._observable, hook=hook).match(observable)
        return self._select(observable, transform=transform)

    def select(self, **limits):
        def hook(observable, transform):
            return observable, transform
        observable, transform = (_ObservableLeafUpdateRef if isinstance(self._observable, ObservableLeaf) else _ObservableTreeUpdateRef)(self._observable, hook=hook).select(**limits)
        return self._select(observable, transform=transform)

    def get(self, *args, **labels):
        assert isinstance(self._observable, ObservableTree), 'get only applies to a tree'
        def hook(observable, transform):
            return observable, transform
        observable, transform = _ObservableTreeUpdateRef(self._observable, hook=hook).get(*args, **labels)
        return self._select(observable, transform=transform)

    @property
    def at(self):
        def hook(sub, transform=None):
            return self._select(sub, transform=transform)
        return (_ObservableLeafUpdateHelper if isinstance(self._observable, ObservableLeaf) else _ObservableTreeUpdateHelper)(self._observable, hook=hook)


@register_type
class WindowMatrix(object):

    _name = 'windowmatrix'

    def __init__(self, value, observable, theory):
        self._value = value
        self._observable = observable
        self._theory = theory

    @property
    def shape(self):
        return self._value.shape

    def value(self):
        return self._value

    def __array__(self):
        return self.value()

    @property
    def observable(self):
        return self._observable

    @property
    def theory(self):
        return self._theory

    @property
    def at(self):
        return _WindowMatrixUpdateHelper(self)

    def copy(self):
        new = self.__class__.__new__(self.__class__)
        new.__setstate__(self.__getstate__())
        return new

    def clone(self, **kwargs):
        """Copy and update data."""
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

    def write(self, filename):
        """
        Write covariance matrix to disk.

        Parameters
        ----------
        filename : str
            Output file name.
        """
        return write(filename, self)


class _CovarianceMatrixUpdateHelper(object):

    def __init__(self, matrix):
        self._matrix = matrix

    @property
    def observable(self):
        return _ObservableCovarianceMatrixUpdateHelper(self._matrix)


class _ObservableCovarianceMatrixUpdateHelper(_ObservableWindowMatrixUpdateHelper):

    def __init__(self, matrix):
        self._matrix = matrix
        self._observable = matrix._observable

    def _select(self, observable, transform):
        if transform.ndim == 1:  # mask
            value = self._matrix.value()[np.ix_(transform, transform)]
        else:
            value = transform.dot(self._matrix.value())
            value = transform.dot(value.T).T  # because transform can be a sparse matrix; works in all cases
        return self._matrix.clone(value=value, observable=observable)


@register_type
class CovarianceMatrix(object):

    _name = 'covariancematrix'

    def __init__(self, value, observable):
        self._value = value
        self._observable = observable

    @property
    def shape(self):
        return self._value.shape

    def value(self):
        return self._value

    def __array__(self):
        return self.value()

    @property
    def observable(self):
        return self._observable

    @property
    def at(self):
        return _CovarianceMatrixUpdateHelper(self)

    def copy(self):
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


class _GaussianLikelihoodUpdateHelper(object):

    def __init__(self, likelihood):
        self._likelihood = likelihood

    @property
    def observable(self):
        return _ObservableGaussianLikelihoodUpdateHelper(self._likelihood, axis=0)

    @property
    def theory(self):
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
        return self._select(observable)

    def select(self, **limits):
        if self._axis == 0:
            observable = self._likelihood.observable.select(**limits)
        else:
            observable = self._likelihood.window.theory.select(**limits)
        return self._select(observable)

    def get(self, *args, **labels):
        if self._axis == 0:
            observable = self._likelihood.observable.get(*args, **labels)
        else:
            observable = self._likelihood.window.theory.get(*args, **labels)
        return self._select(observable)

    @property
    def at(self):
        def hook(sub, transform=None):
            return self._select(sub)
        if self._axis == 0:
            observable = self._likelihood.observable
        else:
            observable = self._likelihood.window.theory
        return (_ObservableLeafUpdateHelper if isinstance(observable, ObservableLeaf) else _ObservableTreeUpdateHelper)(observable, hook=hook)


@register_type
class GaussianLikelihood(object):

    _name = 'gaussianlikelihood'

    def __init__(self, observable, window, covariance):
        self._observable = observable
        self._window = window
        self._covariance = covariance

    @property
    def observable(self):
        return self._observable

    @property
    def window(self):
        return self._window

    @property
    def covariance(self):
        return self._covariance

    @property
    def at(self):
        return _GaussianLikelihoodUpdateHelper(self)

    def copy(self):
        new = self.__class__.__new__(self.__class__)
        new.__setstate__(self.__getstate__())
        return new

    def clone(self, **kwargs):
        """Copy and update observable."""
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