# Exercise 3.3.6

import re, requests
url1, url2 = input(), input()
get_urls = lambda url: re.findall(r'<a.*href ?= ?[\'"](.*?)[\'"]', requests.get(url).text)
result = [1 for url in get_urls(url1) for link in get_urls(url) if link.startswith(url2)]
print('Yes' if result else 'No')
