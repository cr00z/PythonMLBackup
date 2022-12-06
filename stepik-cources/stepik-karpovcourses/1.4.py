worker = ['Olya', 'Silyutina', 350000, 6]
pos = ['junior', 'middle', 'senior'][(worker[3] >= 2) + (worker[3] > 5)]
status = '{0[0]} {0[1]} is {position}'.format(worker, position=pos)
print(status)

values = [12, 134, 10, 47, 100, 20, 50, 160, 210]
tens = [v for v in values if v % 10 == 0]
print(tens)

workers = [['Ivan', 'Ivanov', 100000, 2], ['Petr', 'Petrov', 150000, 2], ['Sidor', 'Sidorov', 200000, 3]]
for w in workers:
    print(f"{w[0]} {w[1]} is {['junior', 'middle', 'senior'][(w[3] >= 2) + (w[3] > 5)]}")