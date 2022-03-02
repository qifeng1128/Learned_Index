from objsize import get_deep_size
from btree import Item

the_size = get_deep_size(Item(1000000, 1000000))

print(the_size)