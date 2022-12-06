# Exercise 2.4.6
import os

ouf = open('test/result.txt', 'w')
out = []
for dir in os.walk('test/main'):
    for file in dir[2]:
        if file.endswith('.py'):
            out.append(dir[0][5:])
            break
out.sort()
ouf.write('\n'.join(out))
ouf.close()
