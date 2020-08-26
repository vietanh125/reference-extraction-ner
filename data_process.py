import pandas as pd
import re
from collections import deque
import os
import math
from sklearn.model_selection import train_test_split
import numpy as np
s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
from feature_extraction import extract_bert, extract, sent_to_char


def remove_accents(input_str):
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s

def write_file(file,sentence):
    with open(file, 'a') as f:
        f.write(str(sentence)+'\n')
        f.close()

def is_tag(str):
    return (str[0] == '<' and str[-1] == '>') or (str[0:2] == '</')

def gen_label(tokenized_list):
    label = []
    stack = deque()
    is_capital = False

    for token in tokenized_list:
        # if (len(token) > 12):
        #     print('---------', token)
        if is_tag(token) == False:
            if len(stack) == 0:
                label.append('O')
            else:
                i = len(stack) - 1
                l = ""
                while(i >= 0):
                    l = stack[i] + '_' + l
                    if is_capital:
                        l = 'B-' + l
                        is_capital = False
                    else:
                        l = 'I-' + l
                    i -= 1
                l = l.strip('_')
                label.append(l)

        else:
            if token[0:2] == '</':
                stack.pop()
            else:
                stack.append(remove_accents(token[1:-1]))
                is_capital = True

    return tokenized_list, label


def remove_tags(sentence):
    new_sentence = []
    for x in sentence:
        if x[0] != '<' or x[-1] != '>':
            new_sentence.append(x)
    return new_sentence


def save(data, file_name):
    with open(file_name, 'w') as file:
        for sentence, label in data:
            for i in range(len(sentence)):
                file.writelines(sentence[i] + ' ' + label[i] + ' ' + label[i] + '\n')
            file.writelines('\n')


# def re_tokenize(path):
#     file = open('Reference_Extraction_Data_5_Folds/' + path, "r")
#     data = file.read().split('.')
#     res = ''
#     for sentence in data:
#         sentence = sentence.replace('_', ' ')
#         # To perform word segmentation only
#         sentence = rdrsegmenter.tokenize(sentence)
#         sentence = [x for sublist in sentence for x in sublist]
#         sentence = ' '.join(sentence)
#         sentence = sentence.replace('< / ', '</').replace('< ', '<').replace(' >', '>')
#         res = res + sentence + ' . '
#     f = open('new_data/' + path, 'w')
#     f.write(res)

count = 0
thresh = 0
SENT_LENGTH = 128
split_count = 0

def windowing(data_array):
    trim_len = int(len(data_array) / SENT_LENGTH) * SENT_LENGTH
    right_most = data_array[-min(SENT_LENGTH, len(data_array)):].tolist()
    trimmed = np.reshape(data_array[:trim_len], (-1, SENT_LENGTH)).tolist()
    trimmed.append(right_most)
    return trimmed

from tqdm import tqdm
def read(path):
    global count, thresh, split_count
    X = []
    y = []
    sents = []
    file = open(path, "r")
    data = file.read().split('.')
    for i in tqdm(range(len(data))):
        d = data[i]
        thresh -= 1
        if thresh > 0 or len(d) == 0:
            continue
        d = re.sub(".\s\)", "", d).replace('<', ' <').replace('>', '> ')
        d = re.sub(">", "> ", d)
        d = re.sub(r"(<[^><\s]+)", r"\1>", d)
        d = re.sub(">>", ">", d)
        d = re.sub('<<', '<', d)
        d = re.sub("</", " </", d).split()
#         d = d.split()
        sentence, label = gen_label(d)
        sentence = remove_tags(sentence)
        sentence = re.sub(r'<.*?>', '', ' '.join(sentence)).split()
        sentence = np.array(sentence)
        label = np.array(label)
#         print(np.unique(label))
        if label.shape != sentence.shape:
            print(len(label), len(sentence))
            print(label)
            print(sentence)
            print(data[i])
        assert label.shape == sentence.shape
        list_sent = windowing(sentence)
        list_label = windowing(label)
        assert len(list_label) == len(list_sent)
        assert len(list_sent[0]) == len(list_label[0])

        for i in range(len(list_sent)):
            sentence = ' '.join(list_sent[i])
            l = list_label[i]
            manual_feat = extract(sentence.split())
            char_list = sent_to_char(sentence.split())
            sentence = extract_bert(sentence)
            sentence = np.hstack((sentence, manual_feat, char_list))
            pad_len = SENT_LENGTH - len(l)
            l += ['pad'] * (pad_len)
            l = np.array(l, dtype='<U12')
            sentence = np.append(sentence, np.zeros((pad_len, sentence.shape[1])), axis=0)
            X.append(sentence)
            y.append(l)
        #print(sentence.shape, label.shape, '\n')
    return np.array(X), np.array(y, dtype='<U12')
path = "new_data/"
# FJoin = os.path.join
files = os.listdir(path)
# files = ['1.txt', '2.txtS']
for f in files:
    X, y = read(path + f)
    print(X.shape, y.shape, f)
    np.save("X_{}.npy".format(f.split('.')[0]), X)
    np.save("y_{}.npy".format(f.split('.')[0]), y)
    del X, y




