# Exercise 3.1.7

s, t = input(), input()
idx, count = 0, 0
while True:
    idx = s.find(t, idx) + 1
    if idx == 0:
        break
    count += 1
print(count)
