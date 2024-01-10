from pycv._lib._inspect import get_signature, get_params, isfunction, EMPTY, fix_kw_syntax, is_co_routine_function, \
    is_generator_function

__all__ = [
    'registrate_decorator',
    'PUBLIC'
]

PUBLIC = []


########################################################################################################################

def registrate_decorator(caller_=None, dst_func=None, kw_syntax=False):
    """
    Meta-decorator for decorators that modify or validate function signatures.

    This decorator is designed to wrap another decorator and handle additional functionality
    related to caller functions, such as checking and modifying the function signature.

    Parameters:
    caller (callable): The caller function or decorator.
    dst_func (callable): The decorated function.
    kw_syntax (bool): If true the args of the dst func become as kw syntax

    Returns:
    - callable: A decorator function that can be applied to other functions.

    Usage:
    ```python
    @registrate_decorator
    def my_decorator(func, *args, **kwargs):
        # Your decorator implementation
        pass

    # Apply the decorator to a target function
    @my_decorator
    def my_function(arg1, arg2):
        # Your function implementation
        pass
    ```

    Note:
    - The provided decorator should accept the target function, additional arguments, and keyword arguments.
    - This meta-decorator ensures proper handling of function signatures and potential asynchronous or generator functions.
    - The resulting decorator maintains attributes and metadata from the original decorator.
    """

    def decorator(caller):
        if not isfunction(caller):
            raise TypeError('caller need to be type of function')

        caller_sig = get_signature(caller)
        caller_params = get_params(caller_sig)

        def caller_decorated(func_=None, *args, **kwargs):

            na = len(args) + 1
            extras = args
            for par in caller_params[na:]:
                if par.default is EMPTY:
                    arg = kwargs.get(par.name, None)
                    if arg is None:
                        raise ValueError(f'{par.name} argument is missing for the caller decorated function')
                else:
                    arg = kwargs.get(par.name, par.default)
                extras += (arg,)

            def dispatcher(func):
                if not isfunction(func):
                    raise TypeError('dispatcher need to get func type of function')

                func_sig = get_signature(func)

                if is_co_routine_function(caller):
                    async def func_out(*f_args, **f_kwargs):
                        f_args, f_kwargs = fix_kw_syntax(f_args, f_kwargs, func_sig, kw_syntax)
                        return await caller(func, *(extras + f_args), **f_kwargs)

                elif is_generator_function(caller):
                    def func_out(*f_args, **f_kwargs):
                        f_args, f_kwargs = fix_kw_syntax(f_args, f_kwargs, func_sig, kw_syntax)
                        for res in caller(func, *(extras + f_args), **f_kwargs):
                            yield res
                else:
                    def func_out(*f_args, **f_kwargs):
                        f_args, f_kwargs = fix_kw_syntax(f_args, f_kwargs, func_sig, kw_syntax)
                        return caller(func, *(extras + f_args), **f_kwargs)

                func_out.__name__ = func.__name__
                func_out.__doc__ = func.__doc__
                func_out.__wrapped__ = func
                func_out.__signature__ = func_sig
                func_out.__qualname__ = func.__qualname__

                try:
                    func_out.__defaults__ = func.__defaults__
                except AttributeError:
                    pass
                try:
                    func_out.__kwdefaults__ = func.__kwdefaults__
                except AttributeError:
                    pass
                try:
                    func_out.__annotations__ = func.__annotations__
                except AttributeError:
                    pass
                try:
                    func_out.__module__ = func.__module__
                except AttributeError:
                    pass
                try:
                    func_out.__dict__.update(func.__dict__)
                except AttributeError:
                    pass

                return func_out

            if func_ is not None and isfunction(func_):
                return dispatcher(func_)
            return dispatcher

        caller_decorated.__signature__ = caller_sig.replace(parameters=caller_params)
        caller_decorated.__name__ = caller.__name__
        caller_decorated.__doc__ = caller.__doc__
        caller_decorated.__wrapped__ = caller
        caller_decorated.__qualname__ = caller.__qualname__
        caller_decorated.__kwdefaults__ = getattr(caller, '__kwdefaults__', None)
        caller_decorated.__dict__.update(caller.__dict__)

        if dst_func is not None:
            return caller_decorated(dst_func)
        return caller_decorated

    return decorator if caller_ is None else decorator(caller_)

########################################################################################################################
