# Exercise 3.2.7
import re


pattern = r'abc'
string = 'abc'
match_obj = re.match(pattern, string)
print(match_obj.group())
