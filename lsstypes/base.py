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
        raise ValueError(f'Cannot find {_name} in registered observables: {cls._registry}')
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


_registry = {}


def register_type(cls):
    _registry[cls._name] = cls
    return cls


def deep_eq(obj1, obj2, equal_nan=True, raise_error=True, label=None):
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

    def coords(self, axis=None):
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
            return {axis: self.coords(axis=axis) for axis in self._coords_names}
        if not isinstance(axis, str):
            axis = self._coords_names[axis]
        return self._data[axis]

    def edges(self, axis=None, default=None):
        """
        Get edge arrays.

        Parameters
        ----------
        axis : str or int, optional
            Name or index of coordinate.

        Returns
        -------
        edges : array, list
        """
        if axis is None:
            return [self.edges(axis=axis) for axis in self._coords_names]
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

    @property
    def shape(self):
        return tuple(len(self._data[coord]) for coord in self._coords_names)

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
        for name, mask in zip(self._coords_names, masks):
            new._data[name] = new._data[name][mask]
        return new

    def _index_select(self, **ranges):
        inverse = False
        masks = {name: np.ones(len(self._data[name]), dtype=bool) for name in self._coords_names}
        for k, v in ranges.items():
            array = self.coords(k)
            mask = (array >= v[0]) & (array <= v[1])
            if inverse: mask = ~mask
            masks[k] &= mask
        return tuple(masks.values())

    def clone(self, **kwargs):
        """Copy and update data."""
        new = self.copy()
        for name, value in kwargs.items():
            if name == 'value':
                new._update_value(value)
                continue
            if name not in new._coords_names + new._values_names: raise ValueError('{name} not unknown')
            new._data[name] = value
        new._data.update(**kwargs)
        return new

    def select(self, **ranges):
        """
        Select a range in one or more coordinates.

        Parameters
        ----------
        ranges : dict
            Each key is a coordinate name, value is (min, max) tuple.

        Returns
        -------
        ObservableLeaf
        """
        return self[self._index_select(**ranges)]

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


def _format_slice(sl, size):
    """Format a Python slice object for array indexing."""
    if sl is None: sl = slice(None)
    start, stop, step = sl.start, sl.stop, sl.step
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


class BinnedObservableLeaf(ObservableLeaf):

    _binweight = None
    _rebin_weighted_normalized_names = None
    _rebin_weighted_names = None

    def binweight(self, default=None):
        """Get binning weights."""
        if self._binweight is None:
            if default is None:
                return None
            return np.ones_like(self.value(), dtype='i4')
        return self._data[self._binweight]

    def _rebin_matrix(self, edges=None, axis=0, weighted=True, normalize=True, reduce=False, return_edges=False):
        # Return, for a given slice, the corresponding matrix to apply to the data arrays.
        if not isinstance(axis, str):
            axis = self._coords_names[axis]

        if edges is None:
            edges = slice(None)
        if isinstance(edges, ObservableLeaf):
            edges = edges.edges(axis=axis)

        def get_unique_edges(edges):
            return [np.unique(edges[:, iax], axis=0) for iax in range(edges.shape[1])]

        def get_1d_slice(edges, sl):
            sl1 = edges[sl, 0]
            sl2 = edges[sl.start + sl.step - 1:sl.stop + sl.step - 1:sl.step, 1]
            size = min(sl1.shape[0], sl2.shape[0])
            return np.column_stack([sl1[:size], sl2[:size]])

        self_edges = self.edges(axis=axis)
        assert self_edges is not None, 'edges must be provided to rebin the observable'
        iedges = edges
        ndim = (1 if self_edges.ndim < 3 else self_edges.shape[1])
        if isinstance(iedges, slice):
            iedges = (iedges,) * ndim
        if isinstance(iedges, tuple):
            assert all(isinstance(iedge, slice) for iedge in iedges)
            slices = [_format_slice(iedge, len(self_edges)) for iedge in iedges]
            assert len(slices) == ndim, f'Provided tuple of slices should be of size {ndim:d}, found {len(slices):d}'
            if self_edges.ndim == 2:
                iedges = get_1d_slice(self_edges, slices[0])
            else:
                iedges1d = [get_1d_slice(e, s) for e, s in zip(get_unique_edges(self_edges), slices)]

                def isin2d(array1, array2):
                    assert len(array1) == len(array2)
                    toret = True
                    for a1, a2 in zip(array1, array2): toret &= np.isin(a1, a2)
                    return toret

                # This is to keep the same ordering
                upedges = self_edges[..., 1][isin2d(self_edges[..., 1].T, [e[..., 1] for e in iedges1d])]
                lowedges = np.column_stack([iedges1d[iax][..., 0][np.searchsorted(iedges1d[iax][..., 1], upedges[..., iax])] for iax in range(ndim)])
                iedges = np.concatenate([lowedges[..., None], upedges[..., None]], axis=-1)

        iaxis = self._coords_names.index(axis)
        # Broadcast iedges[:, None, :] against edges[None, :, :]
        mask = (self_edges[None, ..., 0] >= iedges[:, None, ..., 0]) & (self_edges[None, ..., 1] <= iedges[:, None, ..., 1])  # (new_size, old_size) or (new_size, old_size, ndim)
        if mask.ndim >= 3:
            mask = mask.all(axis=-1)  # collapse extra dims if needed
        matrix = mask * 1
        if weighted:
            weight = self.binweight(default=None)
            if weight is not None:
                if reduce:
                    if weight.ndim > 1:
                        weight = np.sum(weight, axis=tuple(iax for iax in range(weight.ndim) if iax != iaxis))
                else:
                    if weight.ndim > 1:
                        raise NotImplementedError('Rebinning with non-trivial weights of dimension > 1 is not yet implemented. Open a PR.')
                matrix = mask * weight
        if normalize:
            norm = np.where(np.all(matrix == 0, axis=-1), 1, np.sum(matrix, axis=-1))[:, None]
            matrix = matrix / norm
        if return_edges:
            return matrix, iedges
        return matrix

    def rebin(self, **slices):
        """Rebin observable."""
        rebin_weighted_normalized_names = self._rebin_weighted_normalized_names
        rebin_weighted_names = self._rebin_weighted_names
        if rebin_weighted_normalized_names is None:
            rebin_weighted_normalized_names = set(self._values_names) - set([self._binweight] if self._binweight is not None else [])
        if rebin_weighted_names is None:
            rebin_weighted_names = list(set(self._values_names) - set(rebin_weighted_normalized_names) - set([self._binweight] if self._binweight is not None else []))

        new = self.copy()

        def _nan_to_zero(array):
            return np.where(np.isnan(array), 0., array)

        for axis, slice in slices.items():
            matrix, edges = new._rebin_matrix(slice, axis=axis, weighted=False, normalize=False, return_edges=True)
            nwmatrix_reduced = new._rebin_matrix(slice, axis=axis, weighted=True, normalize=True, reduce=True)
            if rebin_weighted_normalized_names:
                nwmatrix = new._rebin_matrix(slice, axis=axis, weighted=True, normalize=True, reduce=False)
            if rebin_weighted_names:
                wmatrix = new._rebin_matrix(slice, axis=axis, weighted=True, normalize=False)
            if not isinstance(axis, str):
                axis = new._coords_names[axis]
            iaxis = new._coords_names.index(axis)
            tmp = _nan_to_zero(new._data[axis])
            new._data[axis] = np.tensordot(nwmatrix_reduced, tmp, axes=([1], [0]))
            axis_edges = f'{axis}_edges'
            if axis_edges in new._data:
                new._data[axis_edges] = edges
            for name in new._values_names:
                tmp = _nan_to_zero(new._data[name])
                if name in rebin_weighted_normalized_names:
                    new._data[name] = np.tensordot(nwmatrix, tmp, axes=([1], [iaxis]))
                elif name in rebin_weighted_names:
                    new._data[name] = np.tensordot(wmatrix, tmp, axes=([1], [iaxis]))
                else:
                    new._data[name] = np.tensordot(matrix, tmp, axes=([1], [iaxis]))
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
        return (_BinnedObservableLeafUpdateRef if isinstance(self._observable, BinnedObservableLeaf) else _ObservableLeafUpdateRef)(self._observable, masks, self._hook)

    def __call__(self, **ranges):
        masks = self._observable._index_select(**ranges)
        return (_BinnedObservableLeafUpdateRef if isinstance(self._observable, BinnedObservableLeaf) else _ObservableLeafUpdateRef)(self._observable, masks, self._hook)


class _ObservableLeafUpdateRef(object):

    def __init__(self, observable, masks, hook=None):
        self._observable = observable
        self._limits = tuple(find_single_true_slab_bounds(mask) for mask in masks)
        self._hook = hook
        assert len(self._limits) == self._observable.ndim

    def select(self, **ranges):
        new = self._observable.copy()
        for iaxis, name in enumerate(self._observable._coords_names):
            if name not in ranges: continue
            _ranges = {name: ranges[name]}

            def create_index_tuple(sl):
                toret = [slice(None)] * self._observable.ndim
                toret[iaxis] = sl
                return tuple(toret)

            start, stop = self._limits[iaxis]
            sub = new[create_index_tuple(slice(start, stop))].select(**_ranges)
            self_coord = self._observable.coords(name)
            new_coord = [self_coord[:start], sub.coords(name), self_coord[stop:]]
            new_sizes = list(map(len, new_coord))
            sub_mask = np.arange(new_sizes[0], sum(new_sizes[:2]))
            new_coord = np.concatenate(new_coord, axis=0)
            new_shape = list(new.shape)
            new_shape[iaxis] = len(new_coord)
            self_mask = np.ones(len(self_coord), dtype=bool)
            self_mask[start:stop] = False
            new_mask = np.ones(len(new_coord), dtype=bool)
            new_mask[new_sizes[0]:sum(new_sizes[:2])] = False

            sub._data[name] = new_coord
            for vname in sub._values_names:
                value = np.zeros(new_shape, dtype=sub._data[vname].dtype)
                value[create_index_tuple(new_mask)] = new._data[vname][create_index_tuple(self_mask)]
                value[create_index_tuple(sub_mask)] = sub._data[vname]
                sub._data[vname] = value

            new = sub

        if self._hook is not None:
            return self._hook(new)
        return new


class _BinnedObservableLeafUpdateRef(_ObservableLeafUpdateRef):

    def rebin(self, **slices):
        new = self._observable.copy()
        matrices = []

        for iaxis, name in enumerate(self._observable._coords_names):
            if name not in slices: continue
            _slices = {name: slices[name]}

            def create_index_tuple(sl):
                toret = [slice(None)] * self._observable.ndim
                toret[iaxis] = sl
                return tuple(toret)

            start, stop = self._limits[iaxis]
            sub = new[create_index_tuple(slice(start, stop))].rebin(**_slices)
            self_coord = self._observable.coords(name)
            new_coord = [self_coord[:start], sub.coords(name), self_coord[stop:]]
            new_sizes = list(map(len, new_coord))
            sub_mask = np.arange(new_sizes[0], sum(new_sizes[:2]))
            new_coord = np.concatenate(new_coord, axis=0)
            new_shape = list(new.shape)
            new_shape[iaxis] = len(new_coord)
            self_mask = np.ones(len(self_coord), dtype=bool)
            self_mask[start:stop] = False
            new_mask = np.ones(len(new_coord), dtype=bool)
            new_mask[new_sizes[0]:sum(new_sizes[:2])] = False

            sub._data[name] = new_coord
            for vname in sub._values_names:
                value = np.zeros(new_shape, dtype=sub._data[vname].dtype)
                value[create_index_tuple(new_mask)] = new._data[vname][create_index_tuple(self_mask)]
                value[create_index_tuple(sub_mask)] = sub._data[vname]
                sub._data[vname] = value

            new = sub

        if self._hook is not None:
            return self._hook(new, matrices=matrices)
        return new


def _iter_on_tree(f, tree, level=None):
    if level == 0 or tree._is_leaf:
        return [f(tree)]
    toret = []
    for tree in tree._leaves:
        toret += _iter_on_tree(f, tree, level=level - 1 if level is not None else None)
    return toret


def _get_leaf(tree, index):
    toret = tree._leaves[index[0]]
    if len(index) == 1:
        return toret
    return _get_leaf(toret, index[1:])


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

    def _index_labels(self, **labels):
        def find(vselect, k):
            if isinstance(vselect, str):
                return [i for i, v in enumerate(self._strlabels[k]) if v == vselect]
            return [i for i, v in enumerate(self._labels[k]) if v == vselect]

        self_index = list(range(len(self._leaves)))
        # First find labels in current level
        for k in self._labels:
            if k in labels:
                vselect = labels.pop(k)
                self_index = [index for index in self_index if index in find(vselect, k)]
        if labels:  # remaining labels
            toret = []
            for index in self_index:
                sub_index_labels = self._leaves[index]._index_labels(**labels)
                if not sub_index_labels:
                    continue
                for sub_index in sub_index_labels:
                    toret.append((index,) + sub_index)
        else:
            toret = [(index,) for index in self_index]
        return toret

    def get(self, *args, **labels):
        """Return leave(s) corresponding to input labels."""
        if args:
            assert not labels, 'Cannot provide both list and dict of labels'
            assert len(args) == 1 and len(self._labels) == 1, 'Args mode available only for one label entry'
            labels = {next(iter(self._labels)): args[0]}
        indices = self._index_labels(**labels)
        toret = []
        for index in indices:
            toret.append(_get_leaf(self, index))
        if len(toret) == 0:
            raise ValueError(f'{labels} not found')
        if len(toret) == 1:
            return toret[0]
        return toret

    def sizes(self, level=None):
        return _iter_on_tree(lambda leaf: leaf.size, self, level=level)

    @property
    def size(self):
        return sum(self.sizes(level=1))

    def select(self, **ranges):
        leaves = []
        notfound = set(ranges)

        def get_coords(leaf):
            return sum(_iter_on_tree(lambda leaf: tuple(leaf._coords_names), leaf, level=None), start=tuple())

        for leaf in self._leaves:
            _all_coord_names = get_coords(leaf)
            _ranges = {k: v for k, v in ranges.items() if k in _all_coord_names}
            notfound -= set(_ranges)
            leaves.append(leaf.select(**_ranges))

        new = self.copy()
        new._leaves = leaves
        return new

    def rebin(self, **slices):
        leaves = []
        notfound = set(slices)

        def get_coords(leaf):
            return sum(_iter_on_tree(lambda leaf: tuple(leaf._coords_names), leaf, level=None), start=tuple())

        for leaf in self._leaves:
            _all_coord_names = get_coords(leaf)
            _slices = {k: v for k, v in slices.items() if k in _all_coord_names}
            notfound -= set(_slices)
            leaves.append(leaf.rebin(**_slices))

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

    def __init__(self, tree):
        self._tree = tree

    def __call__(self, **labels):
        indices = self._tree._index_labels(**labels)
        if len(indices) == 1:
            # Single leaf
            return _ObservableTreeUpdateSingleRef(self._tree, indices[0])
        # Sub-tree
        return _ObservableTreeUpdateRef(self._tree, indices)


def _replace_in_tree(tree, index, sub):
    current_tree = tree
    for idx in index[:-1]:
        current_tree = tree._leaves[idx]
    current_tree._leaves[index[-1]] = sub



class _ObservableTreeUpdateSingleRef(object):

    _tree: ObservableTree
    _index: tuple

    def __init__(self, tree, index):
        self._tree = tree
        self._index = index

    def __getitem__(self, masks):
        sub = _get_leaf(self._tree, self._index).__getitem__(masks)
        new = self._tree.copy()
        _replace_in_tree(new, self._index, sub)
        return new

    def select(self, **ranges):
        sub = _get_leaf(self._tree, self._index).select(**ranges)
        new = self._tree.copy()
        _replace_in_tree(new, self._index, sub)
        return new

    @property
    def at(self):
        at = self._get_leaf(self._tree, self._index).at

        def _replace(sub):
            new = self._tree.copy()
            _replace_in_tree(new, self._index, sub)

        at._hook = _replace
        return at


class _ObservableTreeUpdateRef(object):

    _tree: ObservableTree
    _indices: list

    def __init__(self, tree, indices):
        self._tree = tree
        self._indices = indices

    def get(self, **labels):
        new = self._tree.copy()
        for index in self._indices:
            sub = _get_leaf(self._tree, index).get(**labels)
            _replace_in_tree(new, index, sub)
        return new

    def select(self, **ranges):
        new = self._tree.copy()
        for index in self._indices:
            sub = _get_leaf(self._tree, index).select(**ranges)
            _replace_in_tree(new, index, sub)
        return new


class _WindowMatrixUpdateHelper(object):

    def __init__(self, ):


@register_type
class WindowMatrix(object):

    name = 'windowmatrix'

    def __init__(self, value, observable, theory):
        self._value = value
        self._observable = observable
        self._theory = theory

    @property
    def value(self):
        return self._value

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

    def __getstate__(self, to_file=False):
        state = {}
        state['name'] = self.name
        state['value'] = self.value
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
        Write window matrix to disk.

        Parameters
        ----------
        filename : str
            Output file name.
        """
        return write(filename, self)


class CovarianceMatrix(object):

    def __init__(self, value, observable):
        self._value = value
        self._observable = observable

    @property
    def value(self):
        return self._value

    @property
    def observable(self):
        return self._observable

    @property
    def at(self):
        pass

    def copy(self):
        new = self.__class__.__new__(self.__class__)
        new.__setstate__(self.__getstate__())
        return new

    def __getstate__(self, to_file=False):
        state = {}
        state['name'] = self.name
        state['value'] = self.value
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