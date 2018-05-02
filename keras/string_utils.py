# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
def str2label_single(value, characterToIndex={}, unknown_index=None):
    if unknown_index is None:
        unknown_index = len(characterToIndex)

    label = []
    for v in value:
        if v not in characterToIndex:
            continue
            # raise "Unknown Charactor to Label conversion"
        label.append(characterToIndex[v])
    return np.array(label, np.uint8)

def label2input_single(value, num_of_inputs, char_break_interval):
    idx1 = len(value) * (char_break_interval + 1) + char_break_interval
    idx2 = num_of_inputs + 1
    input_data = [[0 for i in range(idx2)] for j in range(idx1)]

    cnt = 0
    for i in range(char_break_interval):
        input_data[cnt][idx2-1] = 1
        cnt += 1

    for i in range(len(value)):
        if value[i] == 0:
            input_data[cnt][idx2-1] = 1
        else:
            input_data[cnt][value[i]-1] = 1
        cnt += 1

        for i in range(char_break_interval):
            input_data[cnt][idx2-1] = 1
            cnt += 1

    return np.array(input_data)


def str2label(strs, characterToIndex={}):
    longest_str = len(max(strs, key=lambda x: len(x)))
    labels = [[0 for i in range(longest_str)] for j in range(len(strs))]
    for i, s in enumerate(strs):
        for j, v in enumerate(s):
            labels[i][j] = characterToIndex[v]
    return np.array(labels, np.uint8)

def label2str_single(label, indexToCharacter, asRaw, spaceChar = "⋅"):
    string = ""
    #print(label)
    for i in range(len(label)):
        if label[i] == 0:
            if asRaw:
                string += spaceChar
            else:
                break
        else:
            val = label[i]
            string += indexToCharacter[val]
    return string

def naive_decode(output):
    rawPredData = np.argmax(output, axis=1)
    predData = []
    for i in range(len(output)):
        if rawPredData[i] != 0 and not ( i > 0 and rawPredData[i] == rawPredData[i-1] ):
            predData.append(rawPredData[i])
    return predData, list(rawPredData)
