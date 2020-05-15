from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from keras_contrib.layers import CRF
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
import pickle
from tensorflow.keras.callbacks import Callback
from seqeval.metrics import classification_report, f1_score


def trim_padding(y_test, y_pred, tags2idx):
    y_pred_trim = []
    y_test_trim = []
    for i, y in enumerate(y_test):
        first_pad_list = np.where(y == tags2idx['pad'])[0]
        pred = y_pred[i]
        if len(first_pad_list) != 0:
            y = y[:first_pad_list[0]]
            pred = pred[:first_pad_list[0]]
        y_pred_trim.append(pred)
        y_test_trim.append(y)
    y_test = np.concatenate(y_test_trim).ravel()
    y_pred = np.concatenate(y_pred_trim).ravel()
    return y_test, y_pred


def spit_nested_labels(y):
    y_nested = []
    for i, row in enumerate(y):
        nested = []
        for j, x in enumerate(row):
            n = x.split('_')
            if len(n) > 1:
                nested.append(n[1])
                y[i][j] = n[0]
            else:
                nested.append('O')
        y_nested.append(nested)
    return np.append(y, np.array(y_nested), 1)


class F1Callback(Callback):
    def __init__(self, X_val, y_val, X_train, y_train):
        super().__init__()
        self.X_val = X_val
        self.X_train = X_train
        self.y_train = y_train
        self.y_val = y_val
        self.maps = []

    def eval_map(self, X, y):
        x_val, y_true = X, y
        y_pred = self.model.predict(x_val)
        y_pred = np.argmax(y_pred, axis=2)
        y_true = np.argmax(y_true, axis=2)
        y_pred = np.vectorize(idx2tags.get)(y_pred)
        y_true = np.vectorize(idx2tags.get)(y_true)

        y_true = spit_nested_labels(y_true)
        y_pred = spit_nested_labels(y_pred)
        return f1_score(y_true.tolist(), y_pred.tolist())

    def on_epoch_end(self, epoch, logs={}):
        val = self.eval_map(self.X_val, self.y_val)
        # train = self.eval_map(self.X_train, y_train)
        print(" - F1_val: %f" % (val))
        self.maps.append(val)

def make_residual_lstm_layers(input, rnn_width, rnn_depth, rnn_dropout):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """
    x = input
    for i in range(rnn_depth):
        return_sequences = i < rnn_depth
        x_rnn = Bidirectional(LSTM(rnn_width, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=return_sequences))(x)
        if return_sequences:
            # Intermediate layers return sequences, input is also a sequence.
            if i > 0 or input.shape[-1] == rnn_width:
                x = add([x, x_rnn])
            else:
                # Note that the input size and RNN output has to match, due to the sum operation.
                # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
                x = x_rnn
        else:
            # Last layer does not return sequences, just the last element
            # so we select only the last element of the previous output.
            def slice_last(x):
                return x[..., -1, :]
            x = add([Lambda(slice_last)(x), x_rnn])
    return x

folder = ""
# load train test and convert labels into indices
X = np.load(folder + "X.npy")
y = np.load(folder + "y.npy")
y[y == 'pad'] = '<PAD>'
print(X.shape)
tags2idx = {'B-HP': 0, 'B-TTLT': 2, 'I-PL_I-PL': 3, 'I-ND_I-TT': 4, 'I-NQ': 5, 'I-L_I-BL': 6, 'I-TTLT': 7, 'B-L': 8,
            'I-ND_I-L': 9, 'I-ND_I-ND': 10, 'I-ND_B-TT': 11, 'I-HP': 12, 'I-ND': 13, 'I-TT': 14, 'I-PL': 15, 'I-QD': 16,
            'I-L_I-L': 17, 'B-BL': 18, 'I-TT_B-ND': 19, 'I-L_B-ND': 20, 'I-ND_B-L': 21, 'B-ND': 22, 'B-PL': 23,
            'I-BL': 24, 'B-QD': 25, 'I-L_I-ND': 26, 'O': 27, 'I-TT_B-TT': 28, 'B-NQ': 29, 'I-L_B-L': 30, 'I-L_B-BL': 31,
            'I-TT_I-ND': 32, 'I-TT_I-TT': 33, 'I-L': 34, 'B-TT': 35, 'I-PL_B-PL': 36, 'I-ND_B-ND': 37, '<PAD>': 1}
y = np.vectorize(tags2idx.get)(y)
idx2tags = {v: k for k, v in tags2idx.items()}

# one-hot labels
y = np.array([to_categorical(l, num_classes=len(tags2idx.keys())) for l in y])

# split the train test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
callback = F1Callback(X_val, y_val, X_train, y_train)

# build model
input = Input(shape=(200, 1024))
x = Masking(mask_value=0., input_shape=(200, 1024))(input)
x = Bidirectional(LSTM(units=100, return_sequences=True))(x)
x = Dropout(0.1)(x)
x = Bidirectional(LSTM(units=100, return_sequences=True))(x)
x = Dropout(0.1)(x)
# x = make_residual_lstm_layers(x, rnn_width=40, rnn_depth=5, rnn_dropout=0.2)
output = TimeDistributed(Dense(len(tags2idx.keys()), activation='softmax'))(x)
model = Model(input, output)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=45, batch_size=64, shuffle=True,
#           callbacks=[callback])
# model.save('model.h5')
# model = load_model('model.h5', custom_objects={'CRF': CRF}, compile=False)
# # load test set
X_test = np.load(folder + "X.npy")
y_test = np.load(folder + "y.npy")
print(X_test.shape)

# convert labels into indices
# y_test = np.array([[tags2idx[i] for i in l] for l in y_test])

# predict the test set
y_pred = np.argmax(model.predict(X_test), axis=2)
y_pred = np.vectorize(idx2tags.get)(y_pred)

from seqeval.metrics import classification_report, f1_score

y_test = spit_nested_labels(y_test)
y_pred = spit_nested_labels(y_pred)

print(classification_report(y_test.tolist(), y_pred.tolist()))



# # Function checks if the input string(test)
# # contains any special character or not
# from facenet_pytorch import MTCNN, InceptionResnetV1
#
# # If required, create a face detection pipeline using MTCNN:
#
# # Create an inception resnet (in eval mode):
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
# import cv2
# import numpy as np
# import torch
# img = cv2.imread('uyen.jpg')
#
# # Get cropped and prewhitened image tensor
#
# # Calculate embedding (unsqueeze to add batch dimension)
# img_embedding = resnet(torch.Tensor(np.expand_dims(np.rollaxis(img, 2), 0)))
# print(img_embedding.detach().numpy().shape)
# from fairseq.models.roberta import RobertaModel
# from fairseq.data.encoders.fastbpe import fastBPE
# f = fastBPE()
# r = RobertaModel()

