# Exercise 1.6.8

class ExtendedStack(list):
    def sum(self):
        """операция сложения"""
        top1 = self.pop()
        top2 = self.pop()
        self.append(top1 + top2)

    def sub(self):
        """операция вычитания"""
        top1 = self.pop()
        top2 = self.pop()
        self.append(top1 - top2)

    def mul(self):
        """операция умножения"""
        top1 = self.pop()
        top2 = self.pop()
        self.append(top1 * top2)

    def div(self):
        """операция целочисленного деления"""
        top1 = self.pop()
        top2 = self.pop()
        self.append(top1 // top2)
