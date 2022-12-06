# Exercise 2.1.7

parents = {}
catched = []


def is_parent(child, parent):
    return child == parent or \
           any(map(lambda p: is_parent(p, parent), parents[child]))


def intercepted_before(exept):
    for catch in catched:
        if is_parent(exept, catch):
            return True


n = int(input())
for _ in range(n):
    child, *excepts = input().split(' ')
    parents[child] = excepts[1:]

m = int(input())
for _ in range(m):
    exept = input()
    if intercepted_before(exept):
        print(exept)
    else:
        catched.append(exept)
