# Exercise 1.5.8

class MoneyBox:
    def __init__(self, capacity):
        """конструктор с аргументом – вместимость копилки"""
        self.capacity = capacity
        self.money = 0

    def can_add(self, v):
        """True, если можно добавить v монет, False иначе"""
        if self.money + v > self.capacity:
            return False
        return True


    def add(self, v):
        """положить v монет в копилку"""
        if self.can_add(v):
            self.money += v
