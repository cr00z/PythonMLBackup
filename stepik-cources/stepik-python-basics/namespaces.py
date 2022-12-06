# Exercise 1.4.10

namespaces = {
    'global': {
        'parent': None,
        'vars': []
    },
}


def find_namespace(namespace, name):
    for k in namespace.keys():
        if k in ['parent', 'vars']:
            continue
        if k == name:
            return namespace[name]
        ret_val = find_namespace(namespace[k], name)
        if ret_val is not None:
            return ret_val
    return None


n = int(input())
for _ in range(n):
    cmd, ns_name, name = input().split(' ')
    if cmd == 'create':
        work_namespace = find_namespace(namespaces, name)
        work_namespace[ns_name] = {
            'parent': name,
            'vars': [],
        }
    elif cmd == 'add':
        work_namespace = find_namespace(namespaces, ns_name)
        work_namespace['vars'].append(name)
    elif cmd == 'get':
        work_name = ns_name
        while True:
            work_namespace = find_namespace(namespaces, work_name)
            if work_namespace is None:
                print(None)
                break
            if name in work_namespace['vars']:
                print(work_name)
                break
            if work_namespace['parent'] is None:
                print(None)
                break
            work_name = work_namespace['parent']
    else:
        print('unknown command')

# import json
# print(json.dumps(namespaces, sort_keys=True, indent=4))
