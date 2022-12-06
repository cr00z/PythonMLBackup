import json
import requests
import sys


url = 'http://numbersapi.com/{}/math?json=true'


for line in sys.stdin:
    number = line.rstrip()
    content = requests.get(url.format(number))
    json_content = json.loads(content.text)
    if json_content['found']:
        print('Interesting')
    else:
        print('Boring')
