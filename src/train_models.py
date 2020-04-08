"""
By : Aditya Jain
Last Update : 9 April 2020
Contact: https://adityajain.me
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.models import save_model
from models import getModels

from tensorflow.keras.applications.vgg19 import preprocess_input

import cv2
import matplotlib.pyplot as plt

import time
import pickle as pkl

#run fetch_data.sh
result = pd.read_csv("results.csv", sep="|")
print("Shape of input", result.shape)
result.columns=  [ col.strip() for col in result.columns]
print(result.head(2))

#remove non alpha characters
def preprocess_comment(x):
  to_replace = { "'s": " 's", "'ve": " have", "'re": " are", '-':' ' }
  x = str(x)
  for k,v in to_replace.items():
    x = x.replace( k, v )
  x = "".join([ c for c in x if c.isalpha() or c==" " ])
  return x.lower()
result.comment = result.comment.apply(preprocess_comment)
print(result.head(2))

words = { '<pad>', '<sos>', '<eos>' }
for comment in result.comment.values:
  for word in comment.split():
    words.add(word)
VOCAB_SIZE = len(words)
print('VOCAB_SIZE', VOCAB_SIZE)

EMBEDDING_DIM = 50
## Preparing GLove word embedding and embedding_matrix
word2idx = {'<pad>': 0, '<sos>':1, '<eos>':2}
idx2word = {0:'<pad>', 1: '<sos>', 2: '<eos>' }
i=3
for word in words:
  if word in ('<pad>', '<eos>', '<sos>'): continue
  word2idx[word] = i
  idx2word[i] = word
  i+=1
print(len(word2idx), len(idx2word))

glove_dict = { '<pad>': [0]*50, '<sos>':[0]*50, '<eos>':[0]*50 } 
with open('glove.6B.50d.txt', "r") as fp:
  for line in fp.readlines():
    a = line.split()
    glove_dict[a[0].lower()] = np.array(list(map(float, a[1:])))
len(glove_dict)

embedding_matrix = np.zeros( (VOCAB_SIZE, 50) )
unknowns=set()
for word,idx in word2idx.items():
  if word not in glove_dict:
    unknowns.add(word)
    embedding_matrix[idx] = glove_dict['unk']
  else:
    embedding_matrix[idx] = glove_dict[word]
print(embedding_matrix.shape)
print("Total Unknowns", len(unknowns))

#preparing decoder input and output
decoder_inp, decoder_out = [], []
for comment in result.comment.values:
  inp, out = [ 1 ],[]
  for word in comment.split():
    inp.append(word2idx[word])
    out.append(word2idx[word])
  out.append(2)
  decoder_inp.append(inp)
  decoder_out.append(out)

#taking 99 percentile of seq length
MAX_SEQ_LEN = int(np.percentile( [len(x) for x in decoder_inp], 99 ))
print( 'MAX_SEQ_LEN', MAX_SEQ_LEN )

#pad sequences
decoder_inp = pad_sequences( decoder_inp, maxlen=MAX_SEQ_LEN, padding='post', value=0 )
decoder_out = pad_sequences( decoder_out, maxlen=MAX_SEQ_LEN, padding="post", value=0 )
print( "decoder input shape", decoder_inp.shape, "decoder output shape", decoder_out.shape  )

images = result.image_name.apply( lambda x: 'flickr30k_images/flickr30k_images/'+x ).values
SAMPLE_SIZE = len(images)
print('SAMPLE_SIZE', SAMPLE_SIZE)

model, captionGenerator = getModels(embedding_matrix, MAX_SEQ_LEN, LATENT_DIM=100)

#custom acc because we dont want to consider padding
def acc(y_true, y_pred):
  # both are of shape ( _, Ty, VOCAB_SIZE )
  targ = K.argmax(y_true, axis=-1)
  pred = K.argmax(y_pred, axis=-1)
  correct = K.cast(  K.equal(targ,pred), dtype='float32') #cast bool tensor to float

  # 0 is padding, don't include those- mask is tensor representing non-pad value
  mask = K.cast(K.greater(targ, 0), dtype='float32') #cast bool-tensor to float 
  n_correct = K.sum(mask * correct) #
  n_total = K.sum(mask)
  return n_correct / n_total

#custom loss because we dont want to consider padding
def loss(y_true, y_pred):
   # both are of shape ( _, Ty, VOCAB_SIZE )
  mask = K.cast(y_true > 0, dtype='float32')
  out = mask * y_true * K.log(y_pred) #cross entopy loss
  return -K.sum(out) / K.sum(mask)

model.compile( optimizer="rmsprop", loss=loss, metrics=[acc])
print(model.summary())

#train_test_split
TRAIN_IDX, VAL_IDX = train_test_split( np.arange(0, SAMPLE_SIZE), test_size=0.2 )
print('TRAIN_SAMPLES', len(TRAIN_IDX), 'VAL_SAMPLE', len(VAL_IDX))

#to convert a batch of decoder outputs to one hot.
def ohe_decoder_output( x ):
  ohe = np.zeros( (*(x.shape), VOCAB_SIZE) )
  for i in range(len(x)):
    for j in range(MAX_SEQ_LEN):
      ohe[i,j,x[i][j]] = 1
  return ohe

#uses images, decoder_inp, decoder_out
def generator(indices, batch_size=256):
  n_batches = len(indices)//batch_size
  initial_h = np.zeros( (batch_size, LATENT_DIM) )
  initial_c = np.zeros( (batch_size, LATENT_DIM) )
  for i in range(n_batches):
    inp_images = []
    for image in images[ i*batch_size : (i+1)*batch_size ]:
      img = cv2.imread( image )
      img = cv2.resize( img, (224,224), interpolation=cv2.INTER_AREA )
      inp_images.append( preprocess_input(img) )
    inp_images =  np.array(inp_images)
    dec_inp = decoder_inp[ i*batch_size : (i+1)*batch_size ]
    dec_out = decoder_out[ i*batch_size : (i+1)*batch_size ]
    yield [ inp_images, dec_inp, initial_h, initial_c], ohe_decoder_output(dec_out)


#training loop
BATCH_SIZE=180
TRAIN_BATCHES, VAL_BATCHES = len(TRAIN_IDX)//BATCH_SIZE, len(VAL_IDX)//BATCH_SIZE

all_metrics = []
for epoch in range(1,20):

  tloss, tacc, ti, t0 = 0,0,0,time.time()
  for x,y in generator(TRAIN_IDX, BATCH_SIZE):
    metrics_train = model.train_on_batch(x, y)
    tloss, tacc, ti = tloss+metrics_train[0], tacc+metrics_train[1], ti+1
    print(f'\rTraining Progress: {ti}/{TRAIN_BATCHES} - loss: {tloss/ti:.4f} - acc: {tacc/ti:.4f}', end="")

  vloss, vacc, vi = 0, 0, 0
  for x,y in generator( VAL_IDX, BATCH_SIZE ):
    metrics_val = model.test_on_batch(x, y)
    vloss, vacc, vi = vloss+metrics_val[0], vacc+metrics_val[1], vi+1
    print(f'\rValidation Progress: {vi}/{VAL_BATCHES} - val_loss: {vloss/vi:.4f} - val_acc: {vacc/vi:.4f}', end="")

  all_metrics.append( ( tloss/ti, tacc/ti, vloss/vi, vacc/vi  ) )
  print(f"\rEpoch {epoch:2.0f} - {(time.time()-t0):.0f}s - loss: {tloss/ti:.4f} - acc: {tacc/ti:.4f} - val_loss: {vloss/vi:.4f} - val_acc: {vacc/vi:.4f}")


#printing some curves
epochs = len(all_metrics)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].plot( range(epochs), list(map(lambda x: x[0], all_metrics)), label="Train Loss" )
ax[0].plot( range(epochs), list(map(lambda x: x[2], all_metrics)), label="Val Loss" )
ax[0].legend();ax[0].set_xlabel('Epochs');ax[0].set_ylabel('Loss');ax[0].grid()
ax[0].set_title("Model training and Validation Loss curves")

ax[1].plot( range(epochs), list(map(lambda x: x[1], all_metrics)), label="Train Accuracy" )
ax[1].plot( range(epochs), list(map(lambda x: x[3], all_metrics)), label="Val Accuracy" )
ax[1].legend();ax[1].set_xlabel('Epochs');ax[1].set_ylabel('Accuracy');ax[1].grid()
ax[1].set_title("Model Training and Validation Accuracy Curves")
plt.show()

#saving Model
save_model( captionGenerator, "../saved_objects/captionGenerator.h5" )
pkl.dump( idx2word, open("../saved_objects/idx2word.pkl", "wb") )