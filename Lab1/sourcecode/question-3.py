def crawl():
    input_string = input("enter string : ")
    temp_string = ""
    dict = {}
    for charecter in range(len(input_string)):
        for i in range(charecter, len(input_string)):
            if not (input_string[i] in temp_string):
                temp_string += input_string[i]
            else:
                dict[temp_string] = len(temp_string)
                temp_string = ''
                break
    highest_val = max(dict.values())

    answer = []
    for key, val in dict.items():
        if highest_val == val:
            answer.append((key, val))
    print(answer)


if __name__ == '__main__':
    crawl()
