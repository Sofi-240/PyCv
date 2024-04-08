import warnings
from types import DynamicClassAttribute

__all__ = [

]

########################################################################################################################

_NotAllowedNames = {
    '_use_cache_', '_cache_', '_properties_names_',
}


########################################################################################################################

class region_property(DynamicClassAttribute):
    def __set_name__(self, owner, name):
        self._name = name
        self._clsname = owner.__name__

    def __get__(self, instance, owner):
        if self.fget is None:
            raise AttributeError(f'{self._clsname} has no attribute {self._name}')
        if instance is None:
            return self
        _use_cache_ = getattr(instance, '_use_cache_', False)
        if not _use_cache_:
            return self.fget(instance)
        if self._name in instance._cache_:
            return instance._cache_[self._name]
        instance._cache_[self._name] = self.fget(instance)
        return instance._cache_[self._name]

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

class properties_dict(dict):
    def __init__(self, clsname: str):
        super().__init__()
        self.clsname = clsname
        self.properties = {}

    def __setitem__(self, key, value):
        if isinstance(value, (region_property, property, DynamicClassAttribute)):
            self.properties[key] = None
        super().__setitem__(key, value)


class properties(type):
    @classmethod
    def __prepare__(mcs, cls, bases):
        clsdict = properties_dict(cls)
        return clsdict

    def __new__(mcs, cls, bases, clsdict):
        _properties_names_ = set(p for p in clsdict.properties)
        invalid_attrs = _properties_names_ & _NotAllowedNames
        if invalid_attrs:
            raise ValueError(f'got invalid attributes in {cls}: {", ".join(invalid_attrs)}\n '
                             f'dont use any of {", ".join(_NotAllowedNames)}')

        clsdict['_properties_names_'] = _properties_names_
        return super().__new__(mcs, cls, bases, clsdict)

    def __repr__(cls):
        return f'{cls.__name__}'

    def __contains__(cls, name):
        return name in cls._properties_names_

    def __bool__(self):
        return True

    def __getitem__(cls, name):
        _properties_names_ = cls.__dict__.get('_properties_names_', {})
        try:
            return getattr(cls, name)
        except KeyError:
            raise AttributeError(f'{name} is not a member of {cls}') from None


########################################################################################################################


class Properties(metaclass=properties):

    def __init__(self, *args, **kwargs):
        self._use_cache_ = False
        self._cache_ = dict()

    def __getitem__(self, name):
        if name in self._properties_names_:
            try:
                return getattr(self, name)
            except NotImplementedError as _exp:
                warnings.warn(f"{_exp} returning None")
                return None
        raise AttributeError(f'{name} is not a member of {self.__class__.__name__}') from None

    def __contains__(self, name):
        return name in self._properties_names_

    def __iter__(self):
        return ((p, self[p]) for p in self._properties_names_)


########################################################################################################################








