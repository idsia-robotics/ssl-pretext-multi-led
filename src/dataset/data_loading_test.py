from random import randint
from collections import namedtuple
import timeit

def dict_dataset():
    keys = ["a", "b", "c", "d", "e", "f"]
    return {k : randint(1, 100) for k in keys}

data_row = namedtuple('data_row', ['a', 'b', "c", "d", "e", "f"])
def named_tuple_dataset():
    return data_row(a = randint(1, 100), b = randint(1, 100),
                    c = randint(1, 100),
                    d = randint(1, 100),
                    e = randint(1, 100),
                    f = randint(1, 100))


if __name__ == "__main__":
    extime_dict = timeit.timeit(dict_dataset, number=10000000)
    extime_tuple = timeit.timeit(named_tuple_dataset, number=10000000)

    print(extime_dict)
    print(extime_tuple)
    