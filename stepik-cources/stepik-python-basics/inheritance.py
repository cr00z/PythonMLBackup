# Exercise 1.6.7

inheritance = {}


n = int(input())
for _ in range(n):
    class_1 = input()
    parents = None
    if ':' in class_1:
        class_1, _, *parents = class_1.split(' ')
    inheritance[class_1] = set()
    if parents is not None:
        for class_p in parents:
            inheritance[class_1].add(class_p)
            if class_p in inheritance and inheritance[class_p]:
                inheritance[class_1] = inheritance[class_1].union(inheritance[class_p])

for _ in range(len(inheritance)):
    for key, value in inheritance.items():
        for cls in value:
            inheritance[key] = inheritance[key].union(inheritance[cls])

q = int(input())
for _ in range(q):
    class_1, class_2 = input().split(' ')
    if class_1 == class_2:
        print('Yes')
    elif class_2 not in inheritance:
        print('No')
    elif class_1 in inheritance[class_2]:
        print('Yes')
    else:
        print('No')
print(inheritance)
