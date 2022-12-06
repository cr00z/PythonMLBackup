# Exercise 3.5.4

# maxanin
# maxanin244@porjoton.com
# Maxaning244
# Name	test
# Client Id	c2a5ac7db63518d2f719
# Client Secret	43e8f57a7d9dc83386724c0f6c9df48f
# {"type":"xapp_token","token":"eyJhbGciOiJIUzI1NiJ9.eyJyb2xlcyI6IiIsInN1YmplY3RfYXBwbGljYXRpb24iOiI2MjAxMGViYjZlNzMyMzAwMGQ2YTdhZDAiLCJleHAiOjE2NDQ4NDEyNzYsImlhdCI6MTY0NDIzNjQ3NiwiYXVkIjoiNjIwMTBlYmI2ZTczMjMwMDBkNmE3YWQwIiwiaXNzIjoiR3Jhdml0eSIsImp0aSI6IjYyMDEwZWJjZGNlMTMyMDAwZDg0Y2YzZSJ9.Gs4hDkHYC6jUyaSUUXiJw_l2mmHFyrPCloxDmS2jQ0U","expires_at":"2022-02-14T12:21:16+00:00","_links":{}}

import json
import requests
import sys


url = 'https://api.artsy.net/api/artists/{}'
headers = {
    'X-Xapp-Token':
    'eyJhbGciOiJIUzI1NiJ9.eyJyb2xlcyI6IiIsInN1YmplY3RfYXBwbGljYXRpb24iOiI2MjAxM'
    'GViYjZlNzMyMzAwMGQ2YTdhZDAiLCJleHAiOjE2NDQ4NDEyNzYsImlhdCI6MTY0NDIzNjQ3Niw'
    'iYXVkIjoiNjIwMTBlYmI2ZTczMjMwMDBkNmE3YWQwIiwiaXNzIjoiR3Jhdml0eSIsImp0aSI6I'
    'jYyMDEwZWJjZGNlMTMyMDAwZDg0Y2YzZSJ9.Gs4hDkHYC6jUyaSUUXiJw_l2mmHFyrPCloxDmS'
    '2jQ0U'
}
painters = []
for line in sys.stdin:
    painter_id = line.rstrip()
    content = requests.get(url.format(painter_id), headers=headers)
    json_content = json.loads(content.text)
    painters.append((json_content['sortable_name'], json_content['birthday']))
res = sorted(painters, key=lambda x: (x[1], x[0]))
print(res)
print(*[p[0] for p in res], sep='\n')
