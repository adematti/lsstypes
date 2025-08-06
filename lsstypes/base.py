import os
from dataclasses import dataclass

"""
observable = Observable(value={"spectrum": ..., "nmodes": ...}, coords={"k": ..., "mu": ...}, attrs={})
mask = observable.coords("k") < 0.2
observable[mask, 0]
observable = observable.select(k=(0.0, 0.2), mu=(-0.8, 0.8))
observable = observable.select(mu=slice(0, None, 3))
observable.plot()

observable = ObservableTree([observable, observable], ell=[0, 2])
observable.labels()  # [{'ell': 0}, {'ell': 2}]
observable.ell  # [0, 2]
observable = observable.at(ell=0).select(k=(0.0, 0.2), mu=(-0.8, 0.8))
observable = observable.select(ell=2)
observable.at(ell=0)
observable['2']
observable.coords("k", concatenate=False)
observable.coords(0, concatenate=True)
observable.values()
observable.values("spectrum", concatenate=True)
observable.plot()

observable = ObservableTree([observable, observable], observable=['correlation', 'spectrum'])
observable.labels()  # [{'observable': 'correlation', 'ell': 0}, {'observable': 'correlation', 'ell': 2}, {'observable': 'spectrum', 'ell': 0}, {'observable': 'spectrum', 'ell': 2}]
# or ['correlation/0', ...]
# or ['correlation', 'spectrum']
observable.observable  # ['correlation', 'spectrum']
observable = observable.select(observable='correlation', ell=0) | observable.select(observable='correlation', ell=2)
observable['correlation/2', 'spectrum/0']
observable.at(observable='correlation', ell=2).select(k=(0.0, 0.2), mu=(-0.8, 0.8))
mask = observable.coords("k") < 0.2
observable.at(observable='correlation', ell=2)[mask, :]
observable.plot()

observable.save('observable.h5')
"""


import numpy as np


def _h5py_recursively_save_dict(h5file, path, dic, with_attrs=True):
    """
    Save a nested dictionary of arrays to an HDF5 file using h5py.
    """
    attrs = dic.get('attrs', {})
    for key, item in dic.items():
        if with_attrs and key == 'attrs':
            continue  # handle attrs below
        path_key = f'{path}/{key}'.rstrip('/')

        if isinstance(item, dict):
            # If dict has 'attrs' and other keys, it's a group with metadata
            grp = h5file.require_group(path_key, track_order=True)
            _h5py_recursively_save_dict(h5file, path_key, item, with_attrs=with_attrs)
        else:
            # Assume it's an array-like and write as dataset
            dset = h5file.create_dataset(path_key, data=np.asarray(item))

    if with_attrs:
        # Set attributes for the current group
        if isinstance(attrs, dict):
            h5file[path].attrs.update(attrs)


def _h5py_recursively_load_dict(h5file, path='/'):
    """
    Load a nested dictionary of arrays from an HDF5 file.
    Attributes are stored in a special 'attrs' key.
    """
    import h5py
    dic = {}
    for key, item in h5file[path].items():
        path_key = f'{path}/{key}'.rstrip('/')
        if isinstance(item, h5py.Group):
            dic[key] = _h5py_recursively_load_dict(h5file, path_key)
        elif isinstance(item, h5py.Dataset):
            dic[key] = item[()]
    # Load group-level attributes, if any
    if h5file[path].attrs:
        dic['attrs'] = {k: v for k, v in h5file[path].attrs.items()}

    return dic



def mkdir(dirname):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname)  # MPI...
    except OSError:
        return


def _txt_recursively_save_dict(path, dic, with_attrs=True):
    """
    Save a nested dictionary of arrays to an HDF5 file using h5py.
    """
    attrs = dic.get('attrs', {})
    for key, item in dic.items():
        if with_attrs and key == 'attrs':
            continue  # handle attrs below
        path_key = os.path.join(path, key)

        if isinstance(item, dict):
            # If dict has 'attrs' and other keys, it's a group with metadata
            mkdir(path_key)
            _txt_recursively_save_dict(path_key, item, with_attrs=with_attrs)
        else:
            # Assume it's an array-like and write as dataset
            item = np.asarray(item)
            dtype = item.dtype
            np.savetxt(path_key + '.txt', item, header=f'dtype = {str(dtype)}')

    if with_attrs:
        # Set attributes for the current group
        if isinstance(attrs, dict):
            path_key = os.path.join(path, 'attrs.txt')
            import json
            json.dump(attrs, path_key)


def _txt_recursively_load_dict(path='/'):
    """
    Load a nested dictionary of arrays from an HDF5 file.
    Attributes are stored in a special 'attrs' key.
    """
    dic = {}
    for key in os.listdir(path):
        path_key = os.path.join(path, key)
        if os.path.isdir(path_key):
            dic[key] = _txt_recursively_load_dict(path_key)
        elif os.path.isfile(path_key):
            with open(path_key, 'r') as file:
                dtype = file.readline().replace(' ', '').replace('dtype=', '')
                dtype = np.dtype(dtype)
            dic[key] = np.loadtxt(path_key, dtype=dtype)
    # Load group-level attributes, if any
    path_key = os.path.join(path, 'attrs.txt')
    if os.path.exists(path_key):
        import json
        dic['attrs'] = json.load(path_key)
    return dic


def _save(filename, state):
    filename = str(filename)
    if any(filename.endswith(ext) for ext in ['.h5', '.hdf5']):
        import h5py
        with h5py.File(filename, 'w') as f:
            _h5py_recursively_save_dict(f, '/', state)
    elif any(filename.endswith(ext) for ext in ['txt']):
         _txt_recursively_save_dict(filename[:-4], state)
    else:
        raise ValueError(f'unknown file format: {filename}')


def _load(filename):
    filename = str(filename)
    if any(filename.endswith(ext) for ext in ['.h5', '.hdf5']):
        import h5py
        with h5py.File(filename, 'w') as f:
            dic = _h5py_recursively_load_dict(f, '/')
    elif any(filename.endswith(ext) for ext in ['.txt']):
        dic = _txt_recursively_load_dict(filename[:-4])
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


class ObservableLeaf(metaclass=RegisteredObservable):
    """A compressed observable with named values and coordinates, supporting slicing, selection, and plotting."""

    _name = 'base'
    _forbidden_names = ('name',)

    def __init__(self, values, coords=None, attrs=None):
        """
        Parameters
        ----------
        value : dict
            Dictionary of named arrays (e.g. {"spectrum": ..., "nmodes": ...}).
        coords : dict
            Dictionary of named coordinate arrays (e.g. {"k": ..., "mu": ...}).
        attrs : dict, optional
            Additional attributes.
        """
        self._values = dict(values)
        if coords is None: coords = {}
        self._coords = dict(coords)
        for name in ['values', 'coords']:
            assert not any(k in self._forbidden_names for k in getattr(self, '_' + name)), f'Cannot use {self._forbidden_names} as name for arrays'
        self._attrs = dict(attrs) if attrs is not None else {}

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
        coords = self._coords
        if isinstance(name, str):
            coords = coords[name]
        elif name is not None:
            coords = coords[list(self._coords.keys())[name]]
        return coords

    def values(self, name=0, **kw):
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
        values = self._values
        if isinstance(name, str):
            values = values[name]
        elif name is not None:
            values = values[list(self._values.keys())[name]]
        return values

    @property
    def shape(self):
        return tuple(len(coord) for coord in self._coords.values())

    @property
    def size(self):
        return self.values().size

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
        values = {k: v[mask] for k, v in self._values.items()}
        coords = {k: v[mask] for mask, (k, v) in zip(masks, self._coords.items())}
        new = self.copy()
        new._values = values
        new._coords = coords
        return new

    def _index_select(self, inverse=False, **ranges):
        masks = {name: np.ones(len(self._coords[name]), dtype=bool) for name in self._coords}
        for k, v in ranges.items():
            array = self.coords(k)
            mask = (array >= v[0]) & (array <= v[1])
            if inverse: mask = ~mask
            masks[k] &= mask
        return tuple(masks.values())

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
        state = {}
        for name in ['values', 'coords', 'attrs']:
            state[name] = dict(getattr(self, '_' + name))
            if as_dict:
                state[name]['name'] = list(state[name].keys())
        state['name'] = self._name
        return state

    def __setstate__(self, state):
        for name in ['values', 'coords', 'attrs']:
            dic = dict(state[name])
            if 'name' in state[name]:
                names = state[name]['name']
                dic = {}
                for name in names: dic[name] = state[name]
            setattr(self, '_' + name, dic)

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

    def save(self, filename):
        """
        Save observable to an HDF5 file.

        Parameters
        ----------
        filename : str
            Output file name.
        """
        _save(filename, self.__getstate__(as_dict=True))

    @classmethod
    def load(cls, filename):
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
        return cls.from_state(_load(filename))


def find_single_true_slab_bounds(mask):
    start, stop = 0, 0
    if str(mask.dtype) != 'bool':
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
        for axis, name in enumerate(self._observable.coords()):
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

            sub._coords[name] = new_coord
            for vname in sub._values:
                value = np.zeros(new_shape, dtype=sub._values[vname].dtype)
                value[create_index_tuple(new_mask)] = new._values[vname][create_index_tuple(self_mask)]
                value[create_index_tuple(sub_mask)] = sub._values[vname]
                sub._values[vname] = value

            new = sub

        if self._hook is not None:
            return self._hook(new)
        return new


def _iter_on_tree(f, observable, level=None):
    if level == 0 or isinstance(observable, ObservableLeaf):
        return [f(observable)]
    toret = []
    for observable in observable._leaves:
        toret += _iter_on_tree(f, observable, level=level - 1 if level is not None else None)
    return toret


def _get_leaf(observable, index):
    toret = observable._leaves[index[0]]
    if len(index) == 1:
        return toret
    return _get_leaf(toret, index[1:])


class ObservableTree(metaclass=RegisteredObservable):
    """
    A collection of Observable objects, supporting selection, slicing, and labeling.
    """
    _name = 'base_tree'
    _forbidden_label_values = ('name', 'attrs', 'labels')
    _sep_strlabels = '-'

    def __init__(self, leaves, **labels):
        """
        Parameters
        ----------
        leaves : list of ObservableLeaf
            The leaves in the collection.
        labels : dict
            Label arrays (e.g. ell=[0, 2], observable=['spectrum',...]).
        """
        self._leaves = list(leaves)
        leaves_labels = []
        for leaf in self._leaves:
            if not isinstance(leaf, ObservableLeaf):
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
            convert = list(map(self._str_to_label, self._labels[k]))
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
            return str(int)
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
                splits[i] = split
            except ValueError:
                pass
        if squeeze and len(splits) == 1:
            return splits[0]
        return tuple(splits)

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
                if level == 0 or isinstance(leaf, ObservableLeaf):
                    pass
                else:
                    for label in leaf.labels(level=level - 1 if level is not None else None, keys_only=keys_only, as_str=as_str):
                        if label not in toret: toret.append(label)
        else:
            for ileaf, leaf in enumerate(self._leaves):
                self_labels = {k: v[ileaf] for k, v in (self._strlabels if as_str else self._labels).items()}
                if level == 0 or isinstance(leaf, ObservableLeaf):
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

    def get(self, **labels):
        """Return leave(s) corresponding to input labels."""
        indices = self._index_labels(**labels)
        toret = []
        for index in indices:
            toret.append(_get_leaf(self, index))
        if len(toret) == 1:
            return toret
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

    def values(self, concatenate=True):
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
        values : list or array
        """
        def get_values(leaf):
            return _iter_on_tree(lambda leaf: leaf.values(0), leaf, level=None)

        values = get_values(self)
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
            state['labels'] = {'names': self._sep_strlabels.join(list(self._labels.keys())),
                               'values': []}
            for ileaf, leaf in self._leaves:
                label = self._sep_strlabels.join([self._strlabels[k][ileaf] for k in self._labels])
                state['labels']['values'].append(label)
                state[label] = leaf.__getstate__(as_dict=as_dict)
        state['name'] = self._name
        return state

    def __setstate__(self, state):
        leaves = state['leaves']
        if isinstance(leaves, list):
            self._leaves = [ObservableTree.from_state(leaf) for leaf in leaves]
            self._labels = state['labels']
            self._strlabels = state['strlabels']
        else:  # h5py format
            label_names = np.array(state['labels']['names']).item().split(self._sep_strlabels)
            label_values = list(map(lambda x: x.split(self._sep_strlabels), np.array(state['labels']['values'])))
            self._labels, self._strlabels = {}, {}
            for i, name in enumerate(label_names):
                self._strlabels[name] = [v[i] for v in label_values]
                self._labels[name] = [self._str_to_label(s, squeeze=True) for s in self._strlabels[name]]
            nleaves = len(state['labels']['values'])
            self._leaves = []
            for ileaf in range(nleaves):
                label = state['labels']['values'][ileaf]
                self._leaves.append(state[label])

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

    def save(self, filename):
        """
        Save observable to an HDF5 file.

        Parameters
        ----------
        filename : str
            Output file name.
        """
        _save(filename, self.__getstate__(as_dict=True))

    @classmethod
    def load(cls, filename):
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
        return cls.from_state(_load(filename))



@dataclass
class _ObservableTreeUpdateHelper(object):

    observable: ObservableTree

    def __call__(self, **labels):
        indices = self.observable.get(**labels)
        if len(indices) == 1:
            return _ObservableTreeUpdateSingleRef(self.observable, indices[0])
        return _ObservableTreeUpdateRef(self.observable, indices)


def _replace_observable(observable, index, sub):
    current_observable = observable
    for idx in index[:-1]:
        current_observable = observable._leaves[idx]
    current_observable._leaves[index[-1]] = sub


@dataclass
class _ObservableTreeUpdateSingleRef(object):

    _observable: ObservableTree
    _index: tuple

    def __getitem__(self, masks):
        sub = _get_leaf(self._observable, self._index).__getitem__(masks)
        new = self._observable.copy()
        _replace_observable(new, self._index, sub)
        return new

    def select(self, **ranges):
        sub = _get_leaf(self._observable, self._index).select(**ranges)
        new = self._observable.copy()
        _replace_observable(new, self._index, sub)
        return new

    @property
    def at(self):
        at = self._get_leaf(self._observable, self._index).at

        def _replace(sub):
            new = self._observable.copy()
            _replace_observable(new, self._index, sub)

        at._hook = _replace
        return at


@dataclass
class _ObservableTreeUpdateRef(object):

    _observable: ObservableTree
    _indices: list

    def get(self, **labels):
        new = self._observable.copy()
        for index in self._indices:
            sub = _get_leaf(self._observable, index).get(**labels)
            _replace_observable(new, index, sub)
        return new

    def select(self, **ranges):
        new = self._observable.copy()
        for index in self._indices:
            sub = _get_leaf(self._observable, index).select(**ranges)
            _replace_observable(new, index, sub)
        return new
