import re
import numpy as np
import fasttext
model = fasttext.load_model("cc.vi.300.bin")
import torch
from collections import Counter
from typing import List
# Load PhoBERT-base in fairseq
from fairseq.models.roberta import RobertaModel
phobert = RobertaModel.from_pretrained('PhoBERT_base_fairseq', checkpoint_file='model.pt')
phobert.eval()  # disable dropout (or leave in train mode to finetune)

# Incorporate the BPE encoder into PhoBERT-base
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq import options


parser = options.get_preprocessing_parser()
parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE', default="PhoBERT_base_fairseq/bpe.codes")
args = parser.parse_args()
phobert.bpe = fastBPE(args)


def align_bpe_to_words(roberta, bpe_tokens: torch.LongTensor, other_tokens: List[str]):
    """
    Helper to align GPT-2 BPE to other tokenization formats (e.g., spaCy).
    Args:
        roberta (RobertaHubInterface): RoBERTa instance
        bpe_tokens (torch.LongTensor): GPT-2 BPE tokens of shape `(T_bpe)`
        other_tokens (List[str]): other tokens of shape `(T_words)`
    Returns:
        List[str]: mapping from *other_tokens* to corresponding *bpe_tokens*.
    """
    assert bpe_tokens.dim() == 1
    assert bpe_tokens[0] == 0
    def clean(text):
        return text.strip()
    # remove whitespaces to simplify alignment
    bpe_tokens = [roberta.task.source_dictionary.string([x]) for x in bpe_tokens]
    bpe_tokens = [clean(roberta.bpe.decode(x) if x not in {'<s>', ''} else x) for x in bpe_tokens]
    other_tokens = [clean(str(o)) for o in other_tokens]

    # strip leading <s>
    bpe_tokens = bpe_tokens[1:]
    # assert ''.join(bpe_tokens) == ''.join(other_tokens)

    # create alignment from every word to a list of BPE tokens
    alignment = []
    # print(bpe_tokens)
    # print(other_tokens, '\n')
    bpe_toks = filter(lambda item: item[1] != '', enumerate(bpe_tokens, start=1))
    j, bpe_tok = next(bpe_toks)
    for other_tok in other_tokens:
        if other_tok == '':
            print("empty")
        bpe_indices = []
        while True:
            if bpe_tok == '<unk>':
                unk_tok = roberta.bpe.encode(other_tok).split()[0].replace('@@', '')
                other_tok = other_tok[len(unk_tok):]
                try:
                    j, bpe_tok = next(bpe_toks)
                except StopIteration:
                    j, bpe_tok = None, None
            if other_tok.startswith(bpe_tok):
                bpe_indices.append(j)
                other_tok = other_tok[len(bpe_tok):]
                try:
                    j, bpe_tok = next(bpe_toks)
                        # break
                except StopIteration:
                    j, bpe_tok = None, None
            elif bpe_tok.startswith(other_tok):
                # other_tok spans multiple BPE tokens
                bpe_indices.append(j)
                bpe_tok = bpe_tok[len(other_tok):]
                other_tok = ''
            else:
                raise Exception('Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
            if other_tok == '':
                break
        assert len(bpe_indices) > 0
        alignment.append(bpe_indices)
    assert len(alignment) == len(other_tokens)

    return alignment

def align_features_to_words(roberta, features, alignment):
    """
    Align given features to words.

    Args:
        roberta (RobertaHubInterface): RoBERTa instance
        features (torch.Tensor): features to align of shape `(T_bpe x C)`
        alignment: alignment between BPE tokens and words returned by
            func:`align_bpe_to_words`.
    """
    assert features.dim() == 2

    bpe_counts = Counter(j for bpe_indices in alignment for j in bpe_indices)
    assert bpe_counts[0] == 0  # <s> shouldn't be aligned
    denom = features.new([bpe_counts.get(j, 1) for j in range(len(features))])
    weighted_features = features / denom.unsqueeze(-1)

    output = [weighted_features[0]]
    largest_j = -1
    for bpe_indices in alignment:
        output.append(weighted_features[bpe_indices].sum(dim=0))
        largest_j = max(largest_j, *bpe_indices)
    for j in range(largest_j + 1, len(features)):
        output.append(weighted_features[j])
    output = torch.stack(output)
    return output
char2Idx = {"PADDING":0, "UNKNOWN":1}
chars = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|–"
for c in chars:
    char2Idx[c] = len(char2Idx)
print(len(char2Idx))
def sent_to_char(tok_list):
    char_map = np.array([list(t) + ['PADDING'] * (50-len(t)) for t in tok_list])
#     for t in tok_list:
#         t = list(t)
#         for c in t:
#             print(c)
#             print(char2Idx[c])
        
    return np.vectorize(lambda x: char2Idx.get(x, 1))(char_map)

def contains_digit(s):
    return bool(re.search(r'\d', s))

def contains_special_characters(s):
    return bool(re.compile("[-@!#$%^&*()<>?/\|}{~:]").search(s))

def first_capital(s):
    return s[0].isupper()

def syllables_capital(s):
    s = s.strip('_').split('_')
    flag = True
    for x in s:
        flag *= first_capital(x)
    return bool(flag)

def has_keywords(s):
    s = s.lower()
    for key in ['nghị_định', 'quyết_định', 'luật', 'thông_tư', 'pháp_lệnh', 'nghị_quyết', 'hiến_pháp']:
        if re.search(key, s):
            return True
    return False
def extract(tok_list):
    feat = []
    for tok in tok_list:
        ftext = model[tok]
#         w2v = word2vec_model[tok]
        rules = np.array([contains_digit(tok), contains_special_characters(tok), first_capital(tok), syllables_capital(tok), has_keywords(tok)])
        feat.append(np.hstack((ftext, rules)))
    return np.array(feat)

def extract_bert(line):
    BERT_EMB_LEN = 768
    bpe_tokens = phobert.encode(line)
    alignment = align_bpe_to_words(phobert, bpe_tokens, line.split())
    if len(bpe_tokens) > 256:
        return np.zeros((len(line.split()), BERT_EMB_LEN))
    # last_layer_features = phobert.extract_features(bpe_tokens).squeeze(0)
    # feat = align_features_to_words(phobert, last_layer_features, alignment)
    # return feat.detach().numpy()[1:-1]
    list_feat = phobert.extract_features(bpe_tokens, return_all_hiddens=True)
    last_layer_features = list_feat[-1].squeeze(0)
    first_layer_features = list_feat[1].squeeze(0)
    last_feat = align_features_to_words(phobert, last_layer_features, alignment).detach().numpy()[1:-1]
    first_feat = align_features_to_words(phobert, first_layer_features, alignment).detach().numpy()[1:-1]
    return np.hstack((first_feat, last_feat))

