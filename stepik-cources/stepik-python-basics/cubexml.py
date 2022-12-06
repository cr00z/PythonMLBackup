# Exercise 3.6.4

from xml.etree import ElementTree

def get_colors_value(root, colors, value):
    colors[root.attrib['color']] += value
    for child in root:
        colors = get_colors_value(child, colors, value + 1)
    return colors


colors = {
    'red': 0,
    'green': 0,
    'blue': 0,
}

root = ElementTree.fromstring(input())
colors = get_colors_value(root, colors, 1)
print(*colors.values())
