"""
By : Aditya Jain
Last Update : 9 April 2020
Contact: https://adityajain.me
"""
import pandas as pd
import numpy as np

import cv2
import matplotlib.pyplot as plt

import pickle as pkl

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input


caption_generator = load_model( '../saved_objects/captionGenerator.h5' )
idx2word = pkl.load( open('../saved_objects/idx2word.pkl','rb') )

#run fetch_data.sh
result = pd.read_csv("results.csv", sep="|")
print("Shape of input", result.shape)
result.columns=  [ col.strip() for col in result.columns]
images = result.image_name.apply( lambda x: 'flickr30k_images/flickr30k_images/'+x ).values
SAMPLE_SIZE = len(images)

def getCaption( image ):
  img = cv2.resize( image, (224,224), interpolation=cv2.INTER_AREA )
  input_image = np.array( [preprocess_input(img)] )
  initial_h = np.zeros( (1, LATENT_DIM) )
  initial_c = np.zeros( (1, LATENT_DIM) )
  init_token = np.array( [[1]] )
  outputs  = caption_generator( [ input_image, init_token, initial_h, initial_c ] )
  outputs = np.argmax( outputs, axis=-1 )[0]
  print(outputs)
  outputs = [ idx2word[w] for w in outputs if w not in ('<pad>','<eos>') ]
  return  " ".join(outputs)

figure, ax = plt.subplots( nrows=3, ncols=3, figsize=(15,9) )
indexes = np.random.choice( np.arange(0, SAMPLE_SIZE), size=9 )
for i, index in enumerate(indexes):
  image = cv2.imread( images[index] )
  ax[i//3][i%3].imshow( np.flip(image, axis=-1) )
  ax[i//3][i%3].set_title( getCaption(image) )