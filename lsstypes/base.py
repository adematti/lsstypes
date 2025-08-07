import os
import shutil
from dataclasses import dataclass

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


def _format_masks(shape, masks):
    if not isinstance(masks, tuple): masks = (masks,)
    alls = [np.arange(s) for s in shape]
    masks = masks + (Ellipsis,) * (len(shape) - len(masks))
    masks = tuple(a[m] for a, m in zip(alls, masks))
    return masks


class RegisteredObservable(type):

    """Metaclass registering :class:`BinnedStatistic`-derived classes."""

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls._name] = cls
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



class ObservableLeaf(metaclass=RegisteredObservable):
    """A compressed observable with named values and coordinates, supporting slicing, selection, and plotting."""

    _name = 'leaf_base'
    _forbidden_names = ('name', 'attrs', 'values_names', 'coords_names')
    _default_coords = tuple()
    _is_leaf = True

    def __init__(self, coords=None, attrs=None, **data):
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
        self._data = dict(data)
        self._coords = list(coords or self._default_coords)
        assert not any(k in self._forbidden_names for k in self._data), f'Cannot use {self._forbidden_names} as name for arrays'
        self._values = [name for name in data if name not in self._coords]
        assert len(self._values), 'Provide at least one value array'
        self._attrs = dict(attrs) if attrs is not None else {}

    def __getattr__(self, name):
        """Access values and coords by name."""
        if name in self._coords + self._values:
            return self._data[name]
        raise AttributeError(name)

    def coords(self, name=None):
        """
        Get coordinate array(s).

        Parameters
        ----------
        name : str or int, optional
            Name or index of coordinate.
        sparse : bool, optional
            If ``True``, return meshgrid arrays for all coordinates.

        Returns
        -------
        coords : array or dict
        """
        coords = self._data
        if isinstance(name, str):
            coords = coords[name]
        elif name is not None:  # index
            coords = coords[self._coords[name]]
        return coords

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
        values = self._data
        if isinstance(name, str):
            values = values[name]
        elif name is not None:  # index
            values = values[self._values[name]]
        return values

    def value(self):
        return self._data[self._values[0]]

    @property
    def shape(self):
        return tuple(len(self._data[coord]) for coord in self._coords)

    @property
    def size(self):
        return self._data[self._values[0]].size

    @property
    def ndim(self):
        return len(self._coords)

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
        for name in self._values:
            new._data[name] = self._data[name][mask]
        for name, mask in zip(self._coords, masks):
            new._data[name] = new._data[name][mask]
        return new

    def _index_select(self, **ranges):
        inverse = False
        masks = {name: np.ones(len(self._data[name]), dtype=bool) for name in self._coords}
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
            if name not in new._coords + new._values: raise ValueError('{name} not unknown')
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

    def __getstate__(self, as_dict=False):
        state = dict(self._data)
        for name in ['values', 'coords']:
            state[name + '_names'] = list(getattr(self, '_' + name))
        state['attrs'] = dict(self._attrs)
        state['name'] = self._name
        return state

    def __setstate__(self, state):
        for name in ['values', 'coords']:
            setattr(self, '_' + name, [str(n) for n in state[name + '_names']])
        self._attrs = state['attrs']
        self._data = {name: state[name] for name in self._values + self._coords}

    @classmethod
    def from_state(cls, state):
        _name = str(state.pop('name'))
        try:
            cls = cls._registry[_name]
        except KeyError:
            raise ValueError(f'Cannot find {_name} in registered observables: {cls._registry}')
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def write(self, filename, overwrite=True):
        """
        Save observable to an HDF5 file.

        Parameters
        ----------
        filename : str
            Output file name.
        """
        _write(filename, self.__getstate__(as_dict=True), overwrite=overwrite)

    @classmethod
    def read(cls, filename):
        """
        Load observable from an HDF5 file.

        Parameters
        ----------
        filename : str
            Input file name.

        Returns
        -------
        ObservableLeaf
        """
        return cls.from_state(_read(filename))

    def __eq__(self, other):
        return deep_eq(self.__getstate__(), other.__getstate__())


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

    def __call__(self, **ranges):
        masks = self._observable._index_select(**ranges)
        return _ObservableLeafUpdateRef(self._observable, masks, self._hook)


class _ObservableLeafUpdateRef(object):

    def __init__(self, observable, masks, hook=None):
        self._observable = observable
        self._limits = tuple(find_single_true_slab_bounds(mask) for mask in masks)
        self._hook = hook
        assert len(self._limits) == self._observable.ndim

    def select(self, **ranges):
        new = self._observable.copy()
        for axis, name in enumerate(self._observable._coords):
            if name not in ranges: continue
            _ranges = {name: ranges[name]}

            def create_index_tuple(sl):
                toret = [slice(None)] * self._observable.ndim
                toret[axis] = sl
                return tuple(toret)

            start, stop = self._limits[axis]
            sub = new[create_index_tuple(slice(start, stop))].select(**_ranges)
            self_coord = self._observable.coords(name)
            new_coord = [self_coord[:start], sub.coords(name), self_coord[stop:]]
            new_sizes = list(map(len, new_coord))
            sub_mask = np.arange(new_sizes[0], sum(new_sizes[:2]))
            new_coord = np.concatenate(new_coord, axis=0)
            new_shape = list(new.shape)
            new_shape[axis] = len(new_coord)
            self_mask = np.ones(len(self_coord), dtype=bool)
            self_mask[start:stop] = False
            new_mask = np.ones(len(new_coord), dtype=bool)
            new_mask[new_sizes[0]:sum(new_sizes[:2])] = False

            sub._data[name] = new_coord
            for vname in sub._values:
                value = np.zeros(new_shape, dtype=sub._data[vname].dtype)
                value[create_index_tuple(new_mask)] = new._data[vname][create_index_tuple(self_mask)]
                value[create_index_tuple(sub_mask)] = sub._data[vname]
                sub._data[vname] = value

            new = sub

        if self._hook is not None:
            return self._hook(new)
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


class ObservableTree(metaclass=RegisteredObservable):
    """
    A collection of Observable objects, supporting selection, slicing, and labeling.
    """
    _name = 'tree_base'
    _forbidden_label_values = ('name', 'attrs', 'labels_names', 'labels_values')
    _sep_strlabels = '-'
    _is_leaf = False

    def __init__(self, leaves, attrs=None, **labels):
        """
        Parameters
        ----------
        leaves : list of ObservableLeaf
            The leaves in the collection.
        labels : dict
            Label arrays (e.g. ell=[0, 2], observable=['spectrum',...]).
        """
        self._leaves = list(leaves)
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
        if isinstance(label, int):
            return str(label)
        if isinstance(label, str):
            for char in ['_', cls._sep_strlabels]:
                if char in label:
                    raise ValueError(f'Label cannot contain "{char}"')
            return label
        if isinstance(label, tuple):
            if len(label) == 1: raise ValueError('Tuples must be of length > 1')
            return '_'.join([cls._label_to_str(lbl) for lbl in label])
        raise NotImplementedError('Unable to safely cast {} to string. Implement "_label_to_str" and "_str_to_label".')

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
            return _iter_on_tree(lambda leaf: tuple(leaf.coords()), leaf, level=None)

        for leaf in self._leaves:
            _all_coord_names = get_coords(leaf)
            _ranges = {k: v for k, v in ranges.items() if k in _all_coord_names}
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

    def __getstate__(self, as_dict=False):
        state = {}
        if not as_dict:
            state['leaves'] = [leaf.__getstate__() for leaf in self._leaves]
            state['labels'] = dict(self._labels)
            state['strlabels'] = dict(self._strlabels)
        else:
            state['labels_names'] = self._sep_strlabels.join(list(self._labels.keys()))
            state['labels_values'] = []
            for ileaf, leaf in enumerate(self._leaves):
                label = self._sep_strlabels.join([self._strlabels[k][ileaf] for k in self._labels])
                state['labels_values'].append(label)
                state[label] = leaf.__getstate__(as_dict=as_dict)
        state['attrs'] = dict(self._attrs)
        state['name'] = self._name
        return state

    def __setstate__(self, state):
        self._attrs = state['attrs']
        if 'leaves' in state:
            leaves = state['leaves']
            self._leaves = [ObservableTree.from_state(leaf) for leaf in leaves]
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
                self._leaves.append(ObservableTree.from_state(state[label]))

    @classmethod
    def from_state(cls, state):
        _name = state.pop('name')
        try:
            cls = cls._registry[_name]
        except KeyError:
            raise ValueError(f'Cannot find {_name} in registered observables: {cls._registry}')
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def write(self, filename):
        """
        Save observable to an HDF5 file.

        Parameters
        ----------
        filename : str
            Output file name.
        """
        _write(filename, self.__getstate__(as_dict=True))

    @classmethod
    def read(cls, filename):
        """
        Load observable from an HDF5 file.

        Parameters
        ----------
        filename : str
            Input file name.

        Returns
        -------
        ObservableLeaf
        """
        return cls.from_state(_read(filename))

    def __eq__(self, other):
        return deep_eq(self.__getstate__(), other.__getstate__())



class _ObservableTreeUpdateHelper(object):

    _tree: ObservableTree

    def __init__(self, tree):
        self._tree = tree

    def __call__(self, **labels):
        indices = self._tree._index_labels(**labels)
        if len(indices) == 1:
            return _ObservableTreeUpdateSingleRef(self._tree, indices[0])
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


@dataclass
class _ObservableTreeUpdateRef(object):

    _tree: ObservableTree
    _indices: list

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
