import tensorflow
import keras
import pandas
import numpy
import matplotlib.pyplot

BATCH_SIZE = 512
EPOCHS = 50
MAX_LEN = 40 
EMBEDDING = 300

data = pandas.read_csv('TestingMix.csv')

def make_sentences_tokenized(data):
	sentences = []
	sentence = [(data['word'][0], data['event_tag1'][0])]
	for i in range(1, len(data['word'])):
		if(data['index'][i] != data['index'][i-1]+1):
			sentences.append(sentence)
			sentence = [(data['word'][i], data['event_tag1'][i])]
		else:
			sentence.append(tuple((data['word'][i], data['event_tag1'][i])))
	sentences.append(sentence)
	return sentences

def make_sentences(data):
	sentences = []
	sentence = data['word'][0]
	for i in range(1, len(data['word'])):
		if(data['index'][i] != data['index'][i-1]+1):
			sentences.append(sentence)
			sentence = str(data['word'][i])
		else:
			sentence = sentence + " " + str(data['word'][i])
	sentences.append(sentence)
	return sentences


# Load Embeddings
#import os, codecs
#from tqdm import tqdm
#print("Loading Word Emveddings")
#embedding_index = {}
#f = codecs.open('cc.hi.300.vec', encoding = 'utf-8')
#for line in tqdm(f):
#	values = line.rstrip().rsplit(' ')
#	word = values[0]
#	coefs = numpy.asarray(values[1:], dtype = 'float32')
#	embedding_index[word] = coefs
#f.close()
#print('Found word vectors %s' %len(embedding_index))


sentences = make_sentences_tokenized(data)
print("Number of sentences in the dataset" , len(sentences))
#print(sentences[0])

words = list(set(data['word'].values))
n_words = len(words)
print("Number of words in the dataset ",  n_words)
event_tags = list(set(data['event_tag1'].values))
print("Tags : ", event_tags)
n_tags = len(event_tags)
print("Number of Labels: ", n_tags)


word2idx = {w: i+1 for i, w in enumerate(words)}
#word2idx["UNK"] = 1 #Unknown Words
word2idx["PAD"] = 0 # Padding

idx2word = {i: w for w, i in word2idx.items()}

tag2idx = {t: i+1 for i, t in enumerate(event_tags)}
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



#print(X_train[0].shape, y_train[0].shape)


print("Preparing Embedding Matrix")
import fastText
embedding_model = fastText.load_model('cc.hi.300.bin')
#words_not_found = []
embedding_matrix = numpy.zeros((n_words, 300))
for word , i in word2idx.items():
	if i>=n_words:
		continue
	#embedding_vector = embedding_index.get(word)
	#if (embedding_vector is not None) and len(embedding_vector)>0:
	embedding_matrix[i] = embedding_model.get_word_vector(word)
	#else:
	#	words_not_found.append(word)
#print('number of null word embedding: %d' %numpy.sum(numpy.sum(embedding_matrix, axis = 1) == 0))




# Model Implementation
from keras.models import Model, Input, Sequential
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Flatten, SpatialDropout1D
#from keras_contrib.layers import CRF

# Model Definnition
#input = Input(shape=(MAX_LEN, ))
#model = Embedding(input_dim = n_words, output_dim=EMBEDDING, input_length = MAX_LEN, mask_zero = True, weights = [embedding_matrix])(input)
#model = Bidirectional(LSTM(units = 50, return_sequences = True, recurrent_dropout = 0.1))(model)
#model = TimeDistributed(Dense(50, activation = "relu"))(model)
#out = Dense(6, activation = 'softmax')(model)
#crf = CRF(n_tags)
#out = crf(model)

#model = Model(input, out)
#model.compile(optimizer = 'adam', loss = 'categorical_crossentropy' , metrics = ['accuracy'])
#model.summary()

input = Input(shape=(MAX_LEN, ))
model = Embedding(input_dim = n_words, output_dim=EMBEDDING, input_length = MAX_LEN, mask_zero = True, weights=[embedding_matrix])(input)
model = Bidirectional(LSTM(units = 50, return_sequences = True, recurrent_dropout = 0.1))(model)
model = TimeDistributed(Dense(50, activation = "relu"))(model)
out = Dense(6, activation = 'softmax')(model)
#crf = CRF(n_tags+1)
#out = crf(model)

model = Model(input, out)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()


history = model.fit(X_train, numpy.array(y_train), batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = 0.2, verbose = 2)

#history = model.fit(X_train, numpy.array(y_train), batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = 0.2, verbose = 2)
pred_cat = model.predict(X_test)
pred = numpy.argmax(pred_cat, axis = -1)
y_test_true = numpy.argmax(y_test, -1)
from sklearn_crfsuite.metrics import flat_classification_report
pred_tag = [[idx2tag[i] for i in row] for row in pred]
y_test_true_tag = [[idx2tag[i] for i in row] for row in y_test_true]

#from sklearn.metrics import f1_score
#report = f1_score(y_test, pred_cat)
report = flat_classification_report(y_pred = pred_tag, y_true = y_test_true_tag)
print(report)
