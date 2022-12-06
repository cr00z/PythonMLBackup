# Exercise 1.6.8

class LoggableList(list, Loggable):
    def append(self, x):
        super().log(x)
        super().append(x)
