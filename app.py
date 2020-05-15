import torch
from fairseq.models.roberta import RobertaModel
phobert = RobertaModel.from_pretrained('PhoBERT_large_fairseq', checkpoint_file='model.pt')
phobert.eval()  # disable dropout (or leave in train mode to finetune)

# Incorporate the BPE encoder into PhoBERT-base
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq import options
parser = options.get_preprocessing_parser()
parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE', default="PhoBERT_large_fairseq/bpe.codes")
args = parser.parse_args()
phobert.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT

# Extract the last layer's features
line = "chuyển_nhượng , chuyển_giao quyền sở_hữu , quyền sử_dụng các đối_tượng được bảo_hộ theo quy_định của Luật_Sở_hữu_trí_tuệ , Luật Chuyển_giao công_nghệ mà đối_tượng chuyển_giao , chuyển quyền là đồng sở_hữu , đồng tác_giả của nhiều cá_nhân ( nhiều tác_giả ) thì người nộp thuế là từng cá_nhân có quyền sở_hữu , quyền tác_giả và hưởng thu_nhập từ việc chuyển_giao , chuyển quyền nêu trên . "  # INPUT TEXT IS WORD-SEGMENTED!

words = phobert.extract_features_aligned_to_words(line)
for word in words:
    print(str(word), word.vector.detach().numpy())