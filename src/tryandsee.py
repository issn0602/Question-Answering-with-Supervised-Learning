from __future__ import print_function
from functools import reduce
import re
import tarfile
import os
import numpy as np

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent,Dense,Merge,Dropout,Flatten
from keras.models import Model,Sequential
from keras.preprocessing.sequence import pad_sequences
from src import preprocess
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors


RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 300
SENT_HIDDEN_SIZE = 200
QUERY_HIDDEN_SIZE = 200
BATCH_SIZE = 1
EPOCHS = 10
WORD2VEC_EMBED_SIZE = 300
QA_EMBED_SIZE = 64

def get_stories(json_data, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    #data = parse_stories(f.readlines(), only_supporting=only_supporting)
    data = (preprocess.parse_data(json_data))
    #print(data)
    #print(data1)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    #data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    #data1 = [(flatten(story), q, answer) for story, q, answer in data1 if not max_length or len(flatten(story)) < max_length]
    #print(data)
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    #ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        #x=[]
        #xq=[]
        # let's not forget that index 0 is reserved
        #y = np.zeros(shape= (len(word_idx) + 1),dtype='uint8')
        #for w in answer:
        #    y[word_idx[w]] = 1
        xs.append(x)
        xqs.append(xq)
        #ys.append(y)
    #return np.array(xs),np.array(xqs), np.array(ys)
    return pad_sequences(xs, maxlen=story_maxlen), pad_sequences(xqs, maxlen=query_maxlen)

def vectorize_output(data, word_idx,story_maxlen):
    y = np.zeros(shape= (story_maxlen + 1),dtype='uint8')
    ys = []
    for story, query, answer in data:
        print(story)
        for w in answer:
            print(w)
            y[story.index(w)] = 1
        ys.append(y)
    return np.array(ys)



def batches(data,batch_idx,word_idx,story_maxlen,query_maxlen,batch_size = BATCH_SIZE):
    x, xq = vectorize_stories(train[batch_idx:batch_idx+batch_size], word_idx, story_maxlen, query_maxlen)
    y = vectorize_output(train[batch_idx:batch_idx+batch_size],word_idx,story_maxlen)
    return x,xq,y

print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                           EMBED_HIDDEN_SIZE,
                                                           SENT_HIDDEN_SIZE,
                                                           QUERY_HIDDEN_SIZE))
"""
try:
    path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
tar = tarfile.open(path)
# Default QA1 with 1000 samples
# challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
# QA1 with 10,000 samples
# challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
# QA2 with 1000 samples
challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
# QA2 with 10,000 samples
# challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'
"""

"""train = get_stories(open('C:\\Users\\Ash\\Documents\\GitHub\\CSC-522-ALDA-IDLE-Minds\\data\\small.txt'))
test = get_stories(open('C:\\Users\\Ash\\Documents\\GitHub\\CSC-522-ALDA-IDLE-Minds\\data\\small.txt'))
"""

json_file = "train-v1.1.json"
json_data = preprocess.open_json_file(json_file)
#print(json_data)
train = get_stories(json_data)
test = get_stories(json_data)

vocab = set()
for story, q, answer in train + test:
    #print(story)
    vocab |= set(story + q + answer)
vocab = sorted(vocab)
#print(vocab)
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
#print(vocab_size)
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
#print(len(word_idx))
#print(word_idx)
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
print(story_maxlen)
query_maxlen = max(map(len, (x for _, x, _ in train + test)))
print(query_maxlen)
answer_maxlen = max(map(len, (x for _, _, x in train + test)))
print(answer_maxlen)
#print('vocab = {}'.format(vocab))
#print('x.shape = {}'.format(x.shape))
#print('xq.shape = {}'.format(xq.shape))
#print('y.shape = {}'.format(y.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

print("Loading Word2Vec model and generating embedding matrix...")
word2vec = KeyedVectors.load_word2vec_format(
    "src/data/GoogleNews-vectors-negative300.bin.gz", binary=True)
embedding_weights = np.zeros((vocab_size, WORD2VEC_EMBED_SIZE))
for word, index in word_idx.items():
    try:
        embedding_weights[index, :] = word2vec[word.lower()]
    except KeyError:
        pass  # keep as zero (not ideal, but what else can we do?)

del word2vec
#del word_idx

print("Building model...")

# story encoder.
# output shape: (None, story_maxlen, QA_EMBED_SIZE)
senc = Sequential()
senc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=story_maxlen,
                   weights=[embedding_weights], mask_zero=True))
senc.add(recurrent.LSTM(QA_EMBED_SIZE, return_sequences=True))
senc.add(Dropout(0.3))

# question encoder
# output shape: (None, question_maxlen, QA_EMBED_SIZE)
qenc = Sequential()
qenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=query_maxlen,
                   weights=[embedding_weights], mask_zero=True))
qenc.add(recurrent.LSTM(QA_EMBED_SIZE, return_sequences=True))
qenc.add(Dropout(0.3))

# answer encoder
# output shape: (None, answer_maxlen, QA_EMBED_SIZE)
aenc = Sequential()
aenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=answer_maxlen,
                   weights=[embedding_weights], mask_zero=True))
aenc.add(recurrent.LSTM(QA_EMBED_SIZE, return_sequences=True))
aenc.add(Dropout(0.3))

# merge story and question => facts
# output shape: (None, story_maxlen, question_maxlen)
facts = Sequential()
facts.add(Merge([senc, qenc], mode="dot", dot_axes=[2, 2]))

# merge question and answer => attention
# output shape: (None, answer_maxlen, question_maxlen)
attn = Sequential()
attn.add(Merge([aenc, qenc], mode="dot", dot_axes=[2, 2]))

# merge facts and attention => model
# output shape: (None, story+answer_maxlen, question_maxlen)
model = Sequential()
model.add(Merge([facts, attn], mode="concat", concat_axis=1))
model.add(Flatten())
model.add(Dense(story_maxlen+1, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])


print('Training')
"""
model.fit([x, xq], y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.05)

          """

for e in range(EPOCHS):
    print("Parent Epoch:"+str(e))
    i = 0
    for i in range(int(len(train)/BATCH_SIZE)):

        batch_X, batch_XQ, batch_y = batches(train, i, word_idx, story_maxlen, query_maxlen)
        #model.train_on_batch([batch_X,batch_XQ], batch_y)
        model.fit([batch_X,batch_XQ], batch_y,batch_size=BATCH_SIZE,epochs=1,validation_split=0.05)
        i = i + 1
        print(i)

batch_X,batch_XQ, batch_y = batches(train,0,word_idx,story_maxlen,query_maxlen)

loss, acc = model.evaluate([batch_X,batch_XQ], batch_y,
                           batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))