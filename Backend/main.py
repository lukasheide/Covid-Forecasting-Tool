import pandas as pd
from pandas import json_normalize
import json

def main():
    testDF = pd.DataFrame(
        {
            'abc': [1, 2, 33],
            'def': [4, 5, 6]
        }
    )

    list_a = [1,2,3]
    list_b = [4,5,6]

    lists_combined = add_two_lists(list_a,list_b)

    print(lists_combined)

    print('Hello world')


def add_two_lists(lst1: list, lst2: list) -> list:

    result_list = [lst1[x] + lst2[x] for x in range(0,len(lst1))]

    return result_list


if __name__ == '__main__':
    main()

    print(1+1)