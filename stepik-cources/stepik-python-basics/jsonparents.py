# Exercise 3.4.4

import json


def get_childs(parent, childs):
    for c, p in parent_list.items():
        if parent in p:
            childs.add(c)
            childs.union(get_childs(c, childs))
    return childs


temp_list = json.loads(input())
parent_list = {itm['name']: itm['parents'] for itm in temp_list}
out = []
for cl in parent_list.keys():
    childs = get_childs(cl, {cl})
    out.append('{class_} : {childs}'.format(class_=cl, childs=len(childs)))
print(*sorted(out), sep='\n')
