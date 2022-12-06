# Exercise 2.3.5

import itertools


def primes():
    yield 2
    accum = [2]
    candidate = 1
    while True:
        candidate += 2
        for delim in accum:
            if candidate % delim == 0:
                break
        else:
            accum.append(candidate)
            yield candidate


print(list(itertools.takewhile(lambda x: x <= 31, primes())))
# [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

print([i + 1 for i in range(4)])

print([i for i in range(4)])

print([i for i in range(5)][1:])

print(list(i + 1 for i in range(4)))