from pycv._lib._inspect import get_signature, get_params, isfunction, is_co_routine_function, SIGNATURE, \
    is_generator_function, ismethod, isclass, isCallable
from pycv._lib._members_struct import members, extend_members, function_members

__all__ = [
    'MetaDecorator',
    'dispatcher_decorator',
    'wrapper_decorator',
    'to_members'
]


########################################################################################################################

def _get_init(cls):
    if not isclass(cls):
        raise TypeError('cls is not class object')
    return cls.__init__


def _get_func_from_method(me):
    if not ismethod(me):
        raise TypeError('me is not a method')
    return me.__func__


def _get_func_for_decoration(obj):
    if isfunction(obj):
        return obj
    elif ismethod(obj):
        return _get_func_from_method(obj)
    elif isclass(obj):
        return _get_init(obj)
    elif isCallable(obj):
        return obj.__call__.__func__
    raise TypeError('obj is not callable')


def _bind(args: tuple, kwargs: dict, signature: SIGNATURE) -> tuple[tuple, dict]:
    if not isinstance(signature, SIGNATURE):
        raise ValueError('signature need to be type of inspect.Signature')

    ba = signature.bind(*args, **kwargs)
    ba.apply_defaults()
    return ba.args, ba.kwargs


def _decorate_function(func, _decorator):
    if not isfunction(func) or not isfunction(_decorator):
        raise TypeError('func and _decorator nee to be function type')

    func.__name__ = _decorator.__name__
    func.__doc__ = _decorator.__doc__
    func.__signature__ = get_signature(_decorator)
    func.__qualname__ = _decorator.__qualname__
    try:
        func.__defaults__ = _decorator.__defaults__
    except AttributeError:
        pass
    try:
        func.__kwdefaults__ = _decorator.__kwdefaults__
    except AttributeError:
        pass
    try:
        func.__annotations__ = _decorator.__annotations__
    except AttributeError:
        pass
    try:
        func.__module__ = _decorator.__module__
    except AttributeError:
        pass
    try:
        func.__dict__.update(_decorator.__dict__)
    except AttributeError:
        pass
    return func


def _low_level_decorator(
        func,
        wrapper=None,
        extra_args: tuple | None = None,
        extra_kw: dict | None = None,
        dispatcher=None
):
    if extra_args is None:
        extra_args = ()
    if extra_kw is None:
        extra_kw = {}

    if not isfunction(func):
        raise TypeError('func need to be type of function')

    if wrapper is None and dispatcher is None:
        return func
    elif wrapper is None:
        return _decorate_function(func, dispatcher)
    elif not isfunction(wrapper):
        raise TypeError('wrapper need to be type of function')

    func_sig = get_signature(func)

    if is_co_routine_function(wrapper):
        async def func_out(*f_args, **f_kwargs):
            f_args, f_kwargs = _bind(f_args, f_kwargs, func_sig)
            return await wrapper(func, *(extra_args + f_args), **{**f_kwargs, **extra_kw})

    elif is_generator_function(wrapper):
        def func_out(*f_args, **f_kwargs):
            f_args, f_kwargs = _bind(f_args, f_kwargs, func_sig)
            for res in wrapper(func, *(extra_args + f_args), **{**f_kwargs, **extra_kw}):
                yield res
    else:
        def func_out(*f_args, **f_kwargs):
            f_args, f_kwargs = _bind(f_args, f_kwargs, func_sig)
            return wrapper(func, *(extra_args + f_args), **{**f_kwargs, **extra_kw})

    if dispatcher is None:
        dispatcher = func

    func_out.__wrapped__ = wrapper

    return _decorate_function(func_out, dispatcher)


def _class_low_level_decorator(
        cls,
        wrapper=None,
        extra_args: tuple | None = None,
        extra_kw: dict | None = None,
        dispatcher=None
):
    if extra_args is None:
        extra_args = ()
    if extra_kw is None:
        extra_kw = {}

    if not isclass(cls):
        raise TypeError('cls need to be type of class')

    func = _get_init(cls)

    if wrapper is None and dispatcher is None:
        return cls
    elif wrapper is None:
        _decorate_function(func, dispatcher)
        return cls
    elif not isfunction(wrapper):
        raise TypeError('wrapper need to be type of function')

    func_sig = get_signature(func)

    if is_co_routine_function(wrapper):
        async def func_out(*f_args, **f_kwargs):
            f_args, f_kwargs = _bind((None, *f_args), f_kwargs, func_sig)
            f_args = f_args[1:]
            return await wrapper(cls, *(extra_args + f_args), **{**f_kwargs, **extra_kw})

    elif is_generator_function(wrapper):
        def func_out(*f_args, **f_kwargs):
            f_args, f_kwargs = _bind((None, *f_args), f_kwargs, func_sig)
            f_args = f_args[1:]
            for res in wrapper(cls, *(extra_args + f_args), **{**f_kwargs, **extra_kw}):
                yield res
    else:
        def func_out(*f_args, **f_kwargs):
            f_args, f_kwargs = _bind((None, *f_args), f_kwargs, func_sig)
            f_args = f_args[1:]
            return wrapper(cls, *(extra_args + f_args), **{**f_kwargs, **extra_kw})

    if dispatcher is None:
        dispatcher = func

    func_out.__wrapped__ = wrapper

    return _decorate_function(func_out, dispatcher)


########################################################################################################################

class MetaDecorator(object):
    def __init__(self, wrapper=None, dispatcher=None):
        if dispatcher is not None:
            dispatcher = _get_func_for_decoration(dispatcher)
        self._dispatcher = dispatcher
        self._wrapper = None
        self._signature = None
        self._sig_params = None
        self._dispatcher_pos = None
        self._call = None
        self._wrap(wrapper)

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def _decorate_call(self):
        if self._wrapper is None:
            return
        func = _get_func_from_method(self.__call__)
        func.__signature__ = self._signature.replace(parameters=self._sig_params)
        func.__name__ = self._wrapper.__name__
        func.__doc__ = self._wrapper.__doc__
        func.__wrapped__ = self._wrapper
        func.__qualname__ = self._wrapper.__qualname__
        func.__kwdefaults__ = getattr(self._wrapper, '__kwdefaults__', None)
        func.__dict__.update(self._wrapper.__dict__)

    def _wrap(self, wrapper=None):
        if wrapper is None:
            self._call = self._wrap
            return
        self._wrapper = _get_func_for_decoration(wrapper)
        self._signature = get_signature(self._wrapper)
        self._sig_params = get_params(self._signature)

        try:
            self._dispatcher_pos = next(
                i for i in range(len(self._sig_params)) if self._sig_params[i].name == 'dispatcher')
        except StopIteration:
            self._dispatcher_pos = None
            pass

        self._call = self._wrapper_decorated
        self._decorate_call()
        return self

    def _wrapper_decorated(self, func=None, *args, **kwargs):
        ex_args, ex_kw = _bind((self._wrapper, *args), kwargs, self._signature)

        if self._dispatcher_pos is not None:
            dispatcher = ex_args[self._dispatcher_pos]
        else:
            dispatcher = self._dispatcher

        ex_args = ex_args[1:]

        def func_out(_func):
            if isclass(_func):
                return _class_low_level_decorator(
                    _func, wrapper=self._wrapper, extra_args=ex_args, extra_kw=ex_kw, dispatcher=dispatcher
                )
            return _low_level_decorator(
                _func, wrapper=self._wrapper, extra_args=ex_args, extra_kw=ex_kw, dispatcher=dispatcher
            )

        return func_out(func) if func is not None else func_out


########################################################################################################################

def dispatcher_decorator(func=None, dispatcher=None):
    if func is not None:  # mean no dispatcher is given
        return func

    def func_out(_f):
        return _low_level_decorator(
            _f, wrapper=None, extra_args=None, extra_kw=None, dispatcher=dispatcher
        )

    return func_out


def wrapper_decorator(wrapper=None, dispatcher=None):
    def _wrapp(obj):
        return MetaDecorator(_get_func_for_decoration(obj), dispatcher=dispatcher)

    return _wrapp(wrapper) if wrapper is not None else _wrapp


########################################################################################################################

def to_members(func=None, members_cls: members = None, *args):
    if not members_cls:
        raise ValueError('members_cls must be given')
    if not isinstance(members_cls, members):
        raise TypeError(f'members_cls must be type of {type(members)} got {type(members_cls)}')
    if not issubclass(members_cls, function_members):
        raise TypeError(f'members_cls must be subclass of {function_members}')

    def _wrapp(f):
        extend_members(members_cls, [(f.__name__.upper(), (f, *args))])
        return f

    return _wrapp(func) if func is not None else _wrapp

########################################################################################################################
