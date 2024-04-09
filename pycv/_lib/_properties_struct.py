from types import DynamicClassAttribute
from typing import Iterable
from pycv._lib._inspect import is_privet_method

__all__ = [
    'Properties'
]

_NotAllowedNames_ = {
    '_use_cache_', '_cache_', '_properties_map_', '_sequence_'
}

_PropertiesTypeDir_ = {
    '__class__', '__doc__', '__name__', '__qualname__', '__module__',
    '__len__', '__repr__', '__getitem__', '__contains__'
}

_PropertiesDir_ = {
    '__class__', '__doc__', '__init__', '__iter__', '__getitem__', '__repr__',
}

_PropertiesWithSequence_ = {
    '__len__', 'push', 'extend', '__iadd__'
}

Properties = None


########################################################################################################################

class region_property(DynamicClassAttribute):
    def __init__(self, _get=None, _set=None, _del=None, _doc=None, _cache: bool = True):
        super().__init__(_get, _set, _del, _doc)
        self._cache = _cache

    def __call__(self, _get=None, _set=None, _del=None, _doc=None):
        super().__init__(_get, _set, _del, _doc)
        return self

    def __set_name__(self, owner, name):
        self._name = name
        self._clsname = owner.__name__
        self._instance = None
        owner._properties_map_[name] = self

    def __get__(self, instance, owner):
        if self.fget is None:
            raise AttributeError(f'{self._clsname} has no attribute {self._name}')
        if instance is None:
            return self
        if getattr(instance, '_sequence_', False):
            self._instance = instance
            return self
        if not self._cache:
            return self.fget(instance)
        _use_cache_ = getattr(instance, '_use_cache_', False)
        if not _use_cache_:
            return self.fget(instance)
        if self._name in instance._cache_:
            return instance._cache_[self._name]
        instance._cache_[self._name] = self.fget(instance)
        return instance._cache_[self._name]

    def __getitem__(self, _slc):
        if self._instance is None:
            return self.__get__(None, None)
        sub = self._instance[_slc]
        self._instance = None
        if isinstance(sub, list):
            return [self.__get__(s, None) for s in sub]
        return self.__get__(sub, None)

    def __set__(self, instance, value):
        if self.fset is None:
            raise AttributeError(f'{self._clsname} cannot set attribute to {self._name}')
        _use_cache_ = getattr(instance, '_use_cache_', False)
        if not _use_cache_ or self._name not in instance._cache_:
            return self.fset(instance, value)
        instance._cache_.pop(self._name)
        self.fset(instance, value)
        instance._cache_[self._name] = self.fget(instance)

    def __delete__(self, instance):
        if self.fdel is None:
            raise AttributeError(f'{self._clsname} cannot delete {self._name} attribute')
        _use_cache_ = getattr(instance, '_use_cache_', False)
        if not _use_cache_ or self._name not in instance._cache_:
            return self.fdel(instance)
        instance._cache_.pop(self._name)
        return self.fdel(instance)


########################################################################################################################

class properties(type):
    def __new__(mcs, cls, bases, clsdict, *, sequence: bool = False):
        _properties_map_ = dict()
        _sequence_ = sequence

        invalid_attrs = set(clsdict.keys()) & _NotAllowedNames_
        if invalid_attrs:
            raise ValueError(f'got invalid attributes in {cls}: {", ".join(invalid_attrs)}\n '
                             f'dont use any of {", ".join(_NotAllowedNames_)}')

        _base_type_ = mcs._find_property_root_(cls, bases)
        if _base_type_ not in (Properties, object):
            _properties_map_.update(getattr(_base_type_, '_properties_map_', {}))

        clsdict['_properties_map_'] = _properties_map_
        clsdict['_sequence_'] = _sequence_
        clsdict['_base_type_'] = _base_type_

        return super().__new__(mcs, cls, bases, clsdict)

    @classmethod
    def _find_property_root_(mcs, cls, bases):
        if not bases: return Properties
        base = bases[-1]
        if not isinstance(base, properties):
            raise TypeError(f'invalid creation of {cls} class')
        return base

    def __repr__(cls):
        return f'{cls.__name__}'

    def __bool__(cls):
        return True

    def __getitem__(cls, _p):
        _properties_map_ = cls.__dict__.get('_properties_map_', {})
        if _p not in _properties_map_:
            raise KeyError(f'{_p} is not a member of {cls}') from None
        return _properties_map_[_p]

    def __dir__(cls):
        _properties_dir = _PropertiesTypeDir_.copy()
        _properties_dir.update(cls._properties_map_.keys())
        if cls._base_type_:
            return sorted(set(dir(cls._base_type_)) | _properties_dir)
        return sorted(_properties_dir)

    def __contains__(cls, _p):
        return _p in cls._properties_map_


########################################################################################################################

class Properties(metaclass=properties):

    def __new__(cls, _use_cache: bool = False):
        obj = super().__new__(cls)
        _use_cache_ = _use_cache
        _cache_ = dict()
        _items_ = []
        if obj._sequence_:
            setattr(obj, '_items_', _items_)
        setattr(obj, '_cache_', _cache_)
        setattr(obj, '_use_cache_', _use_cache_)
        return obj

    def __getitem__(self, _slc):
        if not self._sequence_ or isinstance(_slc, str):
            if _slc not in self._properties_map_:
                raise KeyError(f'{_slc} is not a member of {self.__class__}') from None
            return getattr(self, _slc)
        return self._items_[_slc]

    def __len__(self):
        if not self._sequence_:
            return 0
        return len(self._items_)

    def __dir__(self):
        properties_dir = _PropertiesDir_.copy()
        inner_dir = getattr(self, '__dict__', {})

        for k in inner_dir:
            if not is_privet_method(k):
                properties_dir.add(k)

        if self._sequence_:
            properties_dir |= _PropertiesWithSequence_

        for base in self.__class__.__mro__:
            for k, i in base.__dict__.items():
                if is_privet_method(k):
                    continue
                if isinstance(i, (property, DynamicClassAttribute, region_property)):
                    if i.fget is None:
                        properties_dir.discard(k)
                    else:
                        properties_dir.add(k)
                else:
                    properties_dir.add(k)

        return sorted(properties_dir)

    def __iter__(self):
        if self._sequence_:
            return iter(self._items_)
        return ((p, self[p]) for p in self._properties_map_)

    def __contains__(self, _p):
        return _p in self._properties_map_

    def __repr__(self):
        if self._sequence_:
            return f'{self.__class__.__name__}: {self._items_}'
        return f'{self.__class__.__name__}'

    def _push_item(self, item):
        if not isinstance(item, self.__class__._base_type_):
            raise TypeError('self and item need to have the same base')
        self._items_ += [item]

    def push(self, item):
        self.__iadd__(item)

    def extend(self, items: Iterable):
        self.__iadd__(items)

    def __iadd__(self, other):
        if not self._sequence_:
            raise RuntimeError('cannot add item to non sequence object')
        if not isinstance(other, Iterable) or isinstance(other, Properties):
            other = [other]
        for o in other:
            self._push_item(o)
        return self

########################################################################################################################
