import pandas as pd
import re
from collections import deque
import os
import math
from sklearn.model_selection import train_test_split
import numpy as np
s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
from fairseq.models.roberta import RobertaModel
phobert = RobertaModel.from_pretrained('PhoBERT_large_fairseq', checkpoint_file='model.pt')
phobert.eval()  # disable dropout (or leave in train mode to finetune)
# from vncorenlp import VnCoreNLP
# rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
# Incorporate the BPE encoder into PhoBERT-base
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq import options
parser = options.get_preprocessing_parser()
parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE', default="PhoBERT_large_fairseq/bpe.codes")
args = parser.parse_args()
phobert.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT
from feature_extraction import extract_bert, extract

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
            if token[1] == '/':
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
from tqdm import tqdm
def read(path):
    global count, thresh, split_count
    X = []
    y = []
    file = open(path, "r")
    data = file.read().split('.')
    for i in tqdm(range(len(data))):
        d = data[i]
        count += 1
        if count < thresh:
            continue
        d = re.sub(".\s\)", "", d).replace('<', ' <').replace('>', '> ').split()
        # d = re.sub(">", "> ", d)
        # d = re.sub(r"(<[^><\s]+)", r"\1>", d)
        # d = re.sub(">>", ">", d)
        # d = re.sub("</", " </", d).split()

        sentence, label = gen_label(d)

        # dang bi loi o day
        sentence = remove_tags(sentence)
        list_sent = []
        list_label = []
        if len(sentence) == 0:
            continue
        if len(sentence) > SENT_LENGTH:
            split_count += 1
            temp = sentence[:int(len(sentence)/SENT_LENGTH)*SENT_LENGTH] + sentence[-SENT_LENGTH:]
            label_temp = label[:int(len(sentence)/SENT_LENGTH)*SENT_LENGTH] + label[-SENT_LENGTH:]
            temp = np.array(temp)
            label_temp = np.array(label_temp)
            list_sent += np.reshape(temp, (-1, SENT_LENGTH)).tolist()
            list_label += np.reshape(label_temp, (-1, SENT_LENGTH)).tolist()
        else:
            list_sent.append(sentence)
            list_label.append(label)
        for i in range(len(list_sent)):
            sentence = ' '.join(list_sent[i])
            l = list_label[i]
            # sentence = phobert.extract_features_aligned_to_words(sentence)
            # sentence = np.array([s.vector.detach().numpy() for s in sentence])[1:-1]
            manual_feat = extract(sentence.split())
            sentence = extract_bert(sentence)
            sentence = np.append(sentence, manual_feat, 1)
            pad_len = SENT_LENGTH - len(l)
            l += ['pad'] * (pad_len)
            l = np.array(l)
            sentence = np.append(sentence, np.zeros((pad_len, sentence.shape[1])), axis=0)
            X.append(sentence)
            y.append(l)
        #print(sentence.shape, label.shape, '\n')

    return np.array(X), np.array(y)
path = "new_data/"
FJoin = os.path.join
files = [FJoin(path, f) for f in os.listdir(path)]
empty = True
for f in files:
    if f == 'new_data/2.txt':
        X_test, y_test = read(f)
        print("test", X_test.shape)
        # np.save("X_test_128.npy", X_test)
        # np.save("y_test_128.npy", y_test)
        del X_test, y_test

    else:
        if empty:
            X, y = read(f)
            empty = False
        else:
            X_temp, y_temp = read(f)
            X = np.append(X, X_temp, axis=0)
            y = np.append(y, y_temp, axis=0)
        print(X.shape, y.shape)
    print(split_count)

print(X.shape, y.shape)
print(X[0])
print(y[0])


# np.save("X_128.npy", X)
# np.save("y_128.npy", y)




# Input
# sentence = sentence.replace('_', ' ')
# # To perform word segmentation only
# sentence = rdrsegmenter.tokenize(sentence)
# sentence = [x for sublist in sentence for x in sublist]
# sentence = ' '.join(sentence)
# sentence = sentence.replace('< / ', '</').replace('< ', '<').replace(' >', '>')
# print(sentence)
# sentence = phobert.extract_features_aligned_to_words(' '.join(sentence))
# sentence = np.array([s.vector.detach().numpy() for s in sentence])[1:-1]
# print(len(sentence))
import torch

# Load PhoBERT-base in fairseq

