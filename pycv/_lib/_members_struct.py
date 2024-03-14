from .._lib._inspect import get_signature, isfunction, is_class_method, is_privet_method, is_magic_method
from types import DynamicClassAttribute, MappingProxyType
from typing import Iterable
from typing import final

__all__ = [
    'members',
    'Members',
    'extend_members',
    'function_members',
]

########################################################################################################################

Members = None

_NotAllowedMembers = {
    '_members_map', '_members_dtype', '_members_names', '_make_new_member',
    '_member_value', '_member_name', '__new_member__'
}

_MembersTypeDir = {
    '__class__', '__doc__', '__name__', '__qualname__', '__module__',
    '__len__', '__members__', '__repr__', '__iter__', '__reversed__',
    '__getitem__', '__contains__'
}

_MembersDir = {
    '__class__', '__doc__', '__eq__', '__hash__', '__module__', '__hash__', '__eq__',
    'name', 'value'
}


########################################################################################################################

class member_property(DynamicClassAttribute):
    def __get__(self, instance, owner):
        if instance is None or self.fget is None:
            try:
                return owner._members_map[self._name]
            except KeyError:
                raise AttributeError(f'{self._clsname} has no attribute {self._name}')
        return self.fget(instance)

    def __set__(self, instance, value):
        if self.fset is None:
            raise AttributeError(f'{self._clsname} cannot set attribute to {self._name}')
        return self.fset(instance, value)

    def __delete__(self, instance):
        if self.fdel is None:
            raise AttributeError(f'{self._clsname} cannot delete {self._name} attribute')
        return self.fdel(instance)

    def __set_name__(self, owner, name):
        self._name = name
        self._clsname = owner.__name__


class members_porto(object):
    def __init__(self, value):
        self.value = value

    def __set_name__(self, owner_cls, member_name):
        delattr(owner_cls, member_name)
        value = self.value
        init_args = (value,) if not isinstance(value, tuple) else value

        if owner_cls._make_new_member in (Members.__new__, object.__new__):
            new_member = owner_cls._make_new_member(owner_cls)
        else:
            new_member = owner_cls._make_new_member(owner_cls, *init_args)

        if not hasattr(new_member, '_member_value'):
            if owner_cls._members_dtype is object:
                new_member._member_value = value
            else:
                try:
                    new_member._member_value = owner_cls._members_dtype(*init_args)
                except Exception as exp:
                    out_exp = TypeError('_member_value was not created during __new__, unable to create value')
                    out_exp.__cause__ = exp
                    raise out_exp

        new_member._member_name = member_name
        new_member.__init__(*init_args)

        setattr(owner_cls, member_name, new_member)
        owner_cls._members_map[member_name] = new_member
        owner_cls._members_names.append(member_name)


########################################################################################################################

def _is_member_property(value) -> bool:
    return isinstance(value, (DynamicClassAttribute, member_property))


########################################################################################################################

class members_dict(dict):
    def __init__(self, clsname: str):
        super().__init__()
        self._clsname = clsname
        self.members_names = {}

    def __setitem__(self, key, value):
        if not (is_magic_method(key) or
                is_privet_method(key) or
                is_class_method(self._clsname, value) or
                _is_member_property(value)):
            if key in self:
                raise TypeError(f'{key} already defined as {self[key]}')
            self.members_names[key] = None
        super().__setitem__(key, value)


########################################################################################################################

def _get_members(_members: Iterable, clsdict: members_dict, _is_extension: bool = False):
    _msg = TypeError(
        '_members can be string (separated by comma or white space) or other iterable of tuples(name, value) pair'
    )
    _ex_msg = TypeError(
        'for extension members values must be given'
    )
    if isinstance(_members, str):
        if _is_extension:
            raise _ex_msg
        _members = _members.replace(',', ' ').split()
    elif isinstance(_members, dict):
        _members = list(_members.items())
    elif isinstance(_members, Iterable):
        _members = list(_members)
    else:
        raise _msg

    if _members and isinstance(_members[0], str):
        if _is_extension:
            raise _ex_msg
        for ii, member in enumerate(_members):
            clsdict[member] = ii + 1
    elif _members and isinstance(_members[0], tuple) and len(_members[0]) == 2 and isinstance(_members[0][0], str):
        for ii, member in enumerate(_members):
            clsdict[member[0]] = member[1]
    elif _members:
        raise _msg

    _msg = None


def _value_repr(value) -> str:
    if isfunction(value):
        return str(get_signature(value))
    return repr(value)


########################################################################################################################

class members(type):
    @classmethod
    def __prepare__(mcs, cls, bases):
        clsdict = members_dict(cls)
        return clsdict

    def __new__(mcs, cls, bases, clsdict):

        _members_names = clsdict.members_names
        clsdict = dict(**clsdict)  # will raise en error if not while assign porto member

        invalid_attrs = set(clsdict.keys()) & _NotAllowedMembers

        if invalid_attrs:
            raise ValueError(f'got invalid attributes in {cls}: {", ".join(invalid_attrs)}\n '
                             f'dont use any of {", ".join(_NotAllowedMembers)}')

        _root = mcs._find_member_root(cls, bases)
        _members_dtype = mcs._find_member_dtype(cls, bases)
        _make_new_member, _save_method = mcs._find_new_method_and_query_save(clsdict, _members_dtype, _root)

        clsdict['_make_new_member'] = _make_new_member

        for name in _members_names:
            un_init_value = clsdict[name]
            clsdict[name] = members_porto(un_init_value)

        _members_map = {}
        _members_names = []

        clsdict['_members_dtype'] = _members_dtype
        clsdict['_members_map'] = _members_map
        clsdict['_members_names'] = _members_names

        _new_cls = super().__new__(mcs, cls, bases, clsdict)
        clsdict.update(_new_cls.__dict__)

        if Members is not None:
            if _save_method:
                _new_cls.__new_member__ = _make_new_member
            _new_cls.__new__ = Members.__new__
        return _new_cls

    def __getitem__(cls, name):
        _members_map = cls.__dict__.get('_members_map', {})
        try:
            return _members_map[name]
        except KeyError:
            raise AttributeError(f'{name} is not a member of {cls}') from None

    def __getattr__(cls, name):
        if is_privet_method(name):
            raise AttributeError(f'{cls} has no attribute {name}')
        # if the attribute exist and this not privet method we don't get here but in case
        _members_map = cls.__dict__.get('_members_map', {})
        try:
            return _members_map[name]
        except KeyError:
            raise AttributeError(f'{name} is not a member of {cls}') from None

    def __setattr__(cls, key, value):
        _members_map = cls.__dict__.get('_members_map', {})
        if key in _members_map:
            raise AttributeError(f'cannot reassign member {key}')
        super().__setattr__(key, value)

    def __delattr__(cls, name):
        _members_map = cls.__dict__.get('_members_map', {})
        if name in _members_map:
            raise AttributeError(f'cannot delete member {name}')
        super().__delattr__(name)

    def __iter__(cls):
        return (cls._members_map[name] for name in cls._members_names)

    def __reversed__(cls):
        return (cls._members_map[name] for name in reversed(cls._members_names))

    def __len__(cls):
        return len(cls._members_names)

    @property
    def __members__(cls):
        return MappingProxyType(cls._members_map)

    def __repr__(cls):
        if not len(cls):
            return f'{cls.__name__}'
        return f'{cls.__name__}: {", ".join(cls._members_names)}'

    def __contains__(cls, name):
        return (isinstance(name, str) and name in cls._members_map) or \
            (isinstance(name, cls) and name._member_name in cls._members_map)

    def __dir__(cls):
        # we not include the privet methods
        _members_dir = _MembersTypeDir.copy()
        _members_dir.update(cls._members_names)
        if cls._make_new_member is not object.__new__:
            _members_dir.add('__new__')
        if cls.__init_subclass__ is not object.__init_subclass__:
            _members_dir.add('__init_subclass__')
        if cls._members_dtype is object:
            return sorted(_members_dir)
        return sorted(set(dir(cls._members_dtype)) | _members_dir)

    def __reduce_ex__(cls, proto):
        return cls.__class__, (cls._member_value,)

    def __call__(cls, cls_or_val, _members=None, *, dtype=None, module=None, prefix_qualname=None):
        if _members is None:
            return cls.__new__(cls, cls_or_val)
        return cls._build(
            cls_or_val, _members,
            dtype=dtype,
            module=module,
            prefix_qualname=prefix_qualname,
        )

    def __bool__(cls):
        return True

    def _build(cls, clsname, _members, *, dtype=None, module=None, prefix_qualname=None):
        mcs = cls.__class__

        bases = (cls,) if dtype is None else (dtype, cls)

        clsdict = members_dict(clsname)
        _get_members(_members, clsdict)

        if module is None:
            try:
                import inspect
                clsdict['__module__'] = inspect.currentframe().f_back.f_globals['__name__']
            except (AttributeError, ValueError, KeyError):
                def _no_module_reduce(self, proto):
                    raise TypeError(f'{self} cannot be pickled')

                clsdict['__reduce_ex__'] = _no_module_reduce
                clsdict['__module__'] = '<unknown>'
        else:
            clsdict['__module__'] = module

        clsdict['__qualname__'] = clsname if not prefix_qualname else f'{prefix_qualname}.{clsname}'

        return mcs.__new__(mcs, clsname, bases, clsdict)

    @classmethod
    def _valid_not_extension(mcs, cls, bases):
        for root in bases:
            for base in root.__mro__:
                if isinstance(base, members) and base._members_names:
                    raise TypeError(f'extension is currently not supported, {cls} cannot extend {base}')

    @classmethod
    def _find_member_root(mcs, cls, bases):
        if not bases: return Members
        mcs._valid_not_extension(cls, bases)
        root = bases[-1]
        if not isinstance(root, members):
            raise TypeError('invalid creation of members class')
        return root

    @classmethod
    def _find_member_dtype(mcs, cls, bases):
        dtypes = set()
        for root in bases:
            for base in root.__mro__:
                if base is object:
                    continue
                elif isinstance(base, members) and base._members_dtype is not object:
                    dtypes.add(base._members_dtype)
                elif not isinstance(base, members) and hasattr(base, '__new__'):
                    dtypes.add(base)

        if len(dtypes) > 1:
            raise TypeError(f'too many types for {cls}: {dtypes}')
        elif dtypes:
            return dtypes.pop()
        else:
            return object

    @classmethod
    def _find_new_method_and_query_save(mcs, clsdict, dtype, root):
        method = clsdict.get('__new__', None)
        save = method is not None and root is not None

        if method is None:
            methods_names = ('__new_member__', '__new__')
            ii = 0
            while ii < 2 and method is None:
                for base in (dtype, root):
                    _found = getattr(base, methods_names[ii], None)
                    if _found not in {None, None.__new__, object.__new__, Members.__new__}:
                        method = _found
                        break
                ii += 1

        if method is None:
            method = object.__new__

        return method, save


########################################################################################################################

class Members(metaclass=members):
    def __init__(self, *args, **kwargs):
        pass

    def __new__(cls, value):
        # we're not creating new objects
        if type(value) is cls:
            return value

        for member in cls._members_map.values():
            if member._member_value == value:
                return member

        raise AttributeError(f'{value} is not a member of {cls}')

    def __repr__(self):
        return f'{self.__class__.__name__}.{self._member_name}: {_value_repr(self._member_value)}'

    def __hash__(self):
        return hash(self._member_name)

    def __dir__(self):
        _members_dir = _MembersDir.copy()
        if self.__class__._members_dtype is not object:
            _members_dir.update(object.__dir__(self))

        _self_dict = getattr(self, '__dict__', {})
        for name in filter(lambda n: not is_privet_method(n), _self_dict.keys()):
            _members_dir.add(name)

        for base in self.__class__.__mro__:
            for name, obj in base.__dict__.items():
                if is_privet_method(name):
                    continue
                if _is_member_property(obj):
                    if obj.fget is not None or name not in self._members_map:
                        _members_dir.add(name)
                    else:
                        _members_dir.discard(name)
                else:
                    _members_dir.add(name)

        return sorted(_members_dir)

    def __eq__(self, other):
        if isinstance(other, str) and not isinstance(self._member_value, str):
            return self._member_name == other
        elif isinstance(other, members):
            return self == other
        return self._member_value == other

    @member_property
    def name(self):
        return self._member_name

    @member_property
    def value(self):
        return self._member_value


########################################################################################################################

def _members_to_dict(cls: members) -> dict:
    if not isinstance(cls, members):
        raise TypeError(f'expected cls to be members type got {type(cls)}')
    return dict((mem.name, mem.value) for mem in cls)


def _valid_extension(cls: members, _members: members | Iterable) -> dict:
    if isinstance(_members, members):
        _extension = _members_to_dict(_members)
    else:
        _extension = members_dict(cls.__name__)
        _get_members(_members, _extension, _is_extension=True)
        _extension = dict(**_extension)

    for name in _extension.keys():
        if name in cls:
            raise ValueError(f'{name} already defined as {cls[name]} in the {cls.__name__}')

    return _extension


def extend_members(cls: members, _members: members | Iterable) -> None:
    _extension = _valid_extension(cls, _members)
    for name, value in _extension.items():
        setattr(cls, name, value)
        _port = members_porto(value)
        try:
            _port.__set_name__(cls, name)
        except Exception as exp:
            out = TypeError(f'{cls.__name__} cannot cast {name}: {value} type of {type(value)} to {cls._members_dtype}')
            out.__cause__ = exp
            raise out


########################################################################################################################

@final
class _function_members_counts(object):
    _counts = dict()

    def __new__(cls):
        return cls

    @classmethod
    def add_method(cls, fcls) -> int:
        if not issubclass(fcls, function_members):
            raise TypeError(f'members_cls must be subclass of {function_members}')
        clsname = fcls.__name__
        if clsname not in cls._counts:
            cls._counts[clsname] = 0
        cls._counts[clsname] += 1
        return cls._counts[clsname]


class function_members(Members):
    def __init__(self, function):
        super().__init__()
        if not isfunction(function):
            raise TypeError(f'function members need to be type of function got {type(function)}')
        self._member_value = _function_members_counts.add_method(self.__class__)
        self.function = function

        self.__call__.__func__.__signature__ = get_signature(self.function)

    def __repr__(self):
        return f'{self.__class__.__name__}.{self._member_name}: {str(get_signature(self.function))}'

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


########################################################################################################################
