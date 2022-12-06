# Exercise 3.1.6

s, a, b = [input() for _ in range(3)]
counter, s_old = 0, ''
if a == b and a in s:
    s, counter = '', 'Impossible'
while a in s:
    s_old = s.replace(a, b)
    if s_old == s or counter == 1000:
        break
    s, counter = s_old, counter + 1
if counter == 1000:
    counter = 'Impossible'
print(counter)
