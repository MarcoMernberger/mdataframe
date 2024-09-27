import numpy as np


def func1():
    return 25


def func2():
    return 25


def func3():
    return 30


def func4(x):
    return x


def get_func5():
    y = 10

    def func5():
        return y

    return func5


def get_func6():
    y = 11

    def func6():
        return y

    return func6


def func7(x, y):
    return x


def func8(x):
    return x


def func9(x, y):
    return x + y


def get_func11():
    y = 11

    def func11():
        return y

    return func11


def get_func12():
    y = 11

    def func12():
        return 11

    return func12


def get_func13():
    y = 12

    def func13():
        return 11

    return func13


def get_func14():
    y = 12

    def func14():
        x = 12
        return 11

    return func14


def get_func15():
    y = 12

    def func15():
        x = 11
        return x

    return func15


def get_func16():
    y = 12

    def func16():
        x = 12
        return x

    return func16


def compare_function(f1, f2, expected):
    print("----------------------")
    print(f1.__name__, f2.__name__)
    print(f1.__closure__, f2.__closure__)
    print(f1.__code__.co_freevars, f2.__code__.co_freevars)
    print(f1.__code__.co_cellvars, f2.__code__.co_cellvars)
    print(f1.__code__.co_consts, f2.__code__.co_consts)
    print(f1.__code__.co_names, f2.__code__.co_names)
    print(f1.__code__.co_varnames, f2.__code__.co_varnames)
    eq_bytecode = f1.__code__.co_code == f2.__code__.co_code
    eq_closure = f1.__closure__ == f2.__closure__
    eq_constants = f1.__code__.co_consts == f2.__code__.co_consts
    eq_conames = f1.__code__.co_names == f2.__code__.co_names
    eq_varnames = f1.__code__.co_varnames == f2.__code__.co_varnames
    print(eq_bytecode, eq_closure, eq_constants, eq_conames)  # , eq_varnames)
    assert (
        eq_bytecode & eq_closure & eq_conames & eq_constants
    ) == expected  # & eq_varnames


compare_function(func1, func2, True)  # should be true
compare_function(func1, func3, False)  # should be false

compare_function(func4, func8, True)  # should be true
compare_function(func4, func7, True)  # should be true
compare_function(func4, func9, False)  # should be false

compare_function(get_func5(), get_func6(), False)  # should be true
compare_function(get_func6(), get_func11(), True)  # should be true
compare_function(get_func11(), get_func12(), False)  # should be False
compare_function(get_func12(), get_func13(), True)  # should be true
compare_function(get_func13(), get_func14(), False)  # should be false
compare_function(get_func14(), get_func15(), False)  # should be false
compare_function(get_func15(), get_func16(), False)  # should be false

compare_function(lambda x: np.log2(x), lambda x: np.log2(x), True)
compare_function(lambda x: np.log2(x), lambda x: np.log2(x + 1), False)
compare_function(lambda x: np.log2(x) + 1, lambda x: np.log2(x + 1), False)
