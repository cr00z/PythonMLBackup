# Exercise 2.1.8
import sys


class NonPositiveError(Exception):
    pass


class PositiveList(list):
    def append(self, x) -> None:
        if x <= 0:
            raise NonPositiveError()
        super().append(x)


print(sys.modules)