import types
import typing
import collections
import inspect

__all__ = [
    'ArgSpec',
    'POS',
    'EMPTY',
    'PARAMETER',
    'SIGNATURE',
    'ismethod',
    'iscode',
    'isfunction',
    'isclass',
    'is_co_routine_function',
    'is_generator_function',
    'isCallable',
    'getargspec',
    'get_signature',
    'get_params',
    'get_doc',
    'fix_kw_syntax',
    'is_magic_method',
    'is_class_method',
    'is_privet_method'
]

ArgSpec = collections.namedtuple('ArgSpec', 'args varargs varkw defaults')
POS = inspect.Parameter.POSITIONAL_OR_KEYWORD
EMPTY = inspect.Parameter.empty
PARAMETER = inspect.Parameter
SIGNATURE = inspect.Signature


########################################################################################################################

def ismethod(me) -> bool:
    return isinstance(me, types.MethodType)


def isfunction(func) -> bool:
    return isinstance(func, types.FunctionType)


def iscode(co) -> bool:
    return isinstance(co, types.CodeType)


def isclass(cls) -> bool:
    return inspect.isclass(cls)


def is_co_routine_function(func) -> bool:
    return inspect.iscoroutinefunction(func)


def is_generator_function(func) -> bool:
    return inspect.isgeneratorfunction(func)


def isCallable(obj) -> bool:
    return isinstance(obj, typing.Callable)


########################################################################################################################

def getargspec(func) -> ArgSpec:
    """
    Return args spect of the function.

    Parameters
    ----------
    func : function

    Returns
    -------
    outputs : ArgSpec (namedtuple) with args, varargs, varkw and defaults fields

    Raises
    ------
    TypeError: if func is unsupported callable
    """
    spec = inspect.getfullargspec(func)
    return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)


########################################################################################################################

def get_signature(func) -> SIGNATURE:
    if not isfunction(func):
        raise TypeError('func need to be type of function')
    return inspect.signature(func)


def get_params(signature: SIGNATURE) -> list[PARAMETER]:
    if not isinstance(signature, SIGNATURE):
        raise ValueError('signature need to be type of inspect.Signature')
    return [p for p in signature.parameters.values() if p.kind is POS]


def get_doc(func) -> str:
    if not isfunction(func):
        raise TypeError('func need to be type of function')
    return inspect.getdoc(func)


def fix_kw_syntax(args: tuple, kwargs: dict, signature: SIGNATURE, kw_syntax: bool = False) -> tuple[tuple, dict]:
    """
    Fix args and kwargs to be consistent with the signature
    """
    if not isinstance(signature, SIGNATURE):
        raise ValueError('signature need to be type of inspect.Signature')

    if not kw_syntax:
        ba = signature.bind(*args, **kwargs)
        ba.apply_defaults()
        return ba.args, ba.kwargs

    params = get_params(signature)
    out_args = tuple()
    out_kwargs = dict()
    i = 0

    for par in params:
        if par.default is EMPTY:
            if i >= len(args):
                tmp = kwargs.get(par.name, None)
                if tmp is None:
                    raise ValueError(f'missing {par.name} parameter')
            else:
                tmp = args[i]
            out_args += (tmp,)
            i += 1
        else:
            out_kwargs[par.name] = kwargs.get(par.name, par.default)

    return out_args, out_kwargs


########################################################################################################################

def is_magic_method(name: str) -> bool:
    if len(name) > 4 and name[:2] == name[-2:] == '__':
        return True
    return False


def is_class_method(cls_name: str, func: types.FunctionType) -> bool:
    if not isfunction(func):
        return False
    qualname = func.__qualname__.split('.')
    if len(qualname) > 1 and qualname[-2] == cls_name:
        return True
    return False


def is_privet_method(name: str) -> bool:
    if len(name) > 1 and name[0] == "_":
        return True
    return False

########################################################################################################################
