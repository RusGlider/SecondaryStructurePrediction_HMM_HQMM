def split_list_by_ratio(input,ratio=0.5):
    #splits the list by adjustable ratio
    length = len(input)
    middle = int(length * ratio)
    return input[:middle], input[middle:]


def val_in_range(val,min,max):
    return min < val < max


def get_sub_by_char(sequence, marking, char):
    # get list of subsequences from sequence from marking sequence with corresponding character
    subsequences = []
    sub = ''
    for i in range(len(sequence)):
        if marking[i] == char:
            sub += sequence[i]
        else:
            if sub == '':
                pass
            else:
                subsequences.append(sub)
                sub = ''
    return subsequences