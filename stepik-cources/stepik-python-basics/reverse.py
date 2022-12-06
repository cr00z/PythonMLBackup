# Exercise 2.4.4

with open('test/dataset_24465_4.txt') as inf, open('test/reverse.txt', 'w') as ouf:
    lines = inf.readlines()
    for line in lines[::-1]:
        ouf.write(line)