# Exercise 3.3.7


import re
import requests
url = input()
result = list({site
               for proto, site in re.findall(
                   r'<a.*href ?= ?[\'"](.*?://)?([^.][0-9A-Za-z-.]*)',
                   requests.get(url).text)})
print(*sorted(result), sep='\n')

#