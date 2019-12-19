from collections import defaultdict
import operator

def tuple_convert():
    list1 = [( 'John', ('Physics', 80)) , ('Daniel', ('Science', 90)),
    ('John', ('Science', 95)), ('Mark',('Maths', 100)),
    ('Daniel', ('History', 75)), ('Mark', ('Social', 95))]
    # intialize the dict
    d = defaultdict(list)
    # Iterate through the values and assign it to the same key
    for k , v in list1:
        d[k].append(v)
    # Sort the dictionary on basis of key
    sorted_dict = sorted(d.items(), key=operator.itemgetter(0))
    print(sorted_dict)



if __name__ == '__main__':
    tuple_convert()