# Exercise 2.1.6

def foo():
    raise ArithmeticError

try:
    foo()
except ZeroDivisionError:
    print('ZeroDivisionError')
except ArithmeticError:
    print('ArithmeticError')
except AssertionError:
    print('AssertionError')
