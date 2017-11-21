from __future__ import print_function
from functools import reduce
import re
import tarfile

import numpy as np

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import preprocess
import preprocessTrain

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
     tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        #print(line)
        line = line.strip()
        #print(line)
        nid, line = line.split(' ', 1)

        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            print('!!!!!!!!!!!')
            q, a, supporting = line.split('\t')
            print(q)
            print(a)
            print(supporting)
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


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
        for w in answer:
            y[story.index(w)] = 1
        ys.append(y)
    return np.array(ys)


RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 500
SENT_HIDDEN_SIZE = 500
QUERY_HIDDEN_SIZE = 500
BATCH_SIZE = 256
EPOCHS = 10

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


#tx, txq = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

#print('vocab = {}'.format(vocab))
#print('x.shape = {}'.format(x.shape))
#print('xq.shape = {}'.format(xq.shape))
#print('y.shape = {}'.format(y.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

print('Build model...')

sentence = layers.Input(shape=(story_maxlen,), dtype='int32')
encoded_sentence = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
encoded_sentence = layers.Dropout(0.3)(encoded_sentence)

question = layers.Input(shape=(query_maxlen,), dtype='int32')
encoded_question = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
encoded_question = layers.Dropout(0.3)(encoded_question)
encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
encoded_question = layers.RepeatVector(story_maxlen)(encoded_question)

merged = layers.add([encoded_sentence, encoded_question])
merged = RNN(EMBED_HIDDEN_SIZE)(merged)
merged = layers.Dropout(0.3)(merged)
preds = layers.Dense(story_maxlen+1, activation='softmax')(merged)

model = Model([sentence, question], preds)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

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
        model.fit([batch_X,batch_XQ], batch_y,batch_size=BATCH_SIZE,epochs=50,validation_split=0.05)
        i = i + 1
        print(i)
	model.evaluate(preprocessTrain.getTrainQ(),preprocessTrain.getTrainA(),batch_size=BATCH_SIZE)

batch_X,batch_XQ, batch_y = batches(train,0,word_idx,story_maxlen,query_maxlen)

loss, acc = model.evaluate([batch_X,batch_XQ], batch_y,
                           batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
