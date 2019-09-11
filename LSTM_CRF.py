import tensorflow
import keras
import pandas
import numpy
import matplotlib.pyplot

BATCH_SIZE = 512
EPOCHS = 50
MAX_LEN = 40 
EMBEDDING = 40

data = pandas.read_csv('Testing.csv')

def make_sentences(data):
	sentences = []
	sentence = [(data['word'][0], data['pos'][0])]
	for i in range(1, len(data['word'])):
		if(data['index'][i] != data['index'][i-1]+1):
			sentences.append(sentence)
			sentence = [(data['word'][i], data['pos'][i])]
		else:
			sentence.append(tuple((data['word'][i], data['pos'][i])))
	sentences.append(sentence)
	return sentences

sentences = make_sentences(data)
print("Number of sentences in the dataset" , len(sentences))
#print(sentences[0])

words = list(set(data['word'].values))
n_words = len(words)
print("Number of words in the dataset ",  n_words)
pos_tags = list(set(data['pos'].values))
print("Tags : ", pos_tags)
n_tags = len(pos_tags)
print("Number of Labels: ", n_tags)


word2idx = {w: i+2 for i, w in enumerate(words)}
word2idx["UNK"] = 1 #Unknown Words
word2idx["PAD"] = 0 # Padding

idx2word = {i: w for w, i in word2idx.items()}

tag2idx = {t: i+1 for i, t in enumerate(pos_tags)}
tag2idx["PAD"] = 0
#print(tag2idx)

idx2tag = {i: w for w, i in tag2idx.items()}

#Convert each sentence from list of tokens to list of word_index
from keras.preprocessing.sequence import pad_sequences
X = [[word2idx[w[0]] for w in s] for s in sentences]
# Padding each sequence to have the same length
X = pad_sequences(maxlen = MAX_LEN, sequences = X, padding = "post", value = word2idx["PAD"])

# Convert Tag/Label to tag_index
y = [[tag2idx[w[1]] for w in s] for s in sentences]
# Padding each sentence to have the same lenght
y = pad_sequences(maxlen = MAX_LEN, sequences = y, padding = "post", value = tag2idx["PAD"])


# One-hot Encoding
from keras.utils import to_categorical
y = [to_categorical(i, num_classes = n_tags+1) for i in y]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Model Implementation
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF

# Model Definnition
input = Input(shape=(MAX_LEN, ))
model = Embedding(input_dim = n_words+2, output_dim=EMBEDDING, input_length = MAX_LEN, mask_zero = True)(input)
model = Bidirectional(LSTM(units = 50, return_sequences = True, recurrent_dropout = 0.1))(model)
model = TimeDistributed(Dense(50, activation = "relu"))(model)
crf = CRF(n_tags+1)
out = crf(model)

model = Model(input, out)
model.compile(optimizer = "rmsprop", loss=crf.loss_function, metrics = [crf.accuracy])
model.summary()


history = model.fit(X_train, numpy.array(y_train), batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = 0.2, verbose = 2)
pred_cat = model.predict(X_test)
pred = numpy.argmax(pred_cat, axis = -1)
y_test_true = numpy.argmax(y_test, -1)
from sklearn_crfsuite.metrics import flat_classification_report
pred_tag = [[idx2tag[i] for i in row] for row in pred]
y_test_true_tag = [[idx2tag[i] for i in row] for row in y_test_true]

report = flat_classification_report(y_pred = pred_tag, y_true = y_test_true_tag)
print(report)