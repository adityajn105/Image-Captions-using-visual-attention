"""
By: Aditya Jain
Date: 9 April 202
Contact: https://adityajain.me
"""

from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.layers import Reshape, Lambda, Input, Embedding, LSTM
from tensorflow.keras.layers import RepeatVector, Concatenate, Dense, Dropout, Dot

from tensorflow.keras import backend as K

from tensorflow.keras.models import Model

def getModels( embedding_matrix, MAX_SEQ_LEN, LATENT_DIM ):
	"""
	Args:
		embedding_matrix : pretrained glove word embeddings
		MAX_SEQ_LEN : maximum length of caption
		LATENT_DIM : size of decoder LSTM states

	Returns:
		model to train, model to generate captions
	"""
	Tx = 512
	Ty = MAX_SEQ_LEN
	VOCAB_SIZE = len(embedding_matrix)

	##########################Encoder Part##############################
	#We will use vgg19 to generate 512 feature maps of size 14 * 14
	#these feature maps will be flattened and combined to with previous decoder lstm state to calculate alphas
	vgg19 = VGG19( include_top=False, input_shape=(224,224,3) )
	for layer in vgg19.layers:
	  layer.trainable=False
	a = Reshape((14*14,Tx), name="feature_vecs")(vgg19.layers[-2].output) # ( None, 196, 512)
	a = Lambda( lambda x: K.permute_dimensions(x, pattern=(0,2,1)) )(a) #(None, 512, 196)


	#########################Attention Part##############################
	#Repear s(t-1) 512 times using repeat vector
	#concatenate s(t-1) with each feature map a_t
	#Pass it though a neural network with output of one neuron
	#apply softmax over time axis, other wise all alphas will be one
	#get weigher feature map (when we multiple feature map with respecitve alpha)
	#sum all weighted feature map, this will be context (hard attention)
	#last 2 steps can be achieved by dot product over axis=1
	def softmax_over_time(x): #(softmax on time axis instead of axis=-1)
	  e = K.exp( x - K.max(x, axis=1, keepdims=True) )
	  s = K.sum(e, axis=1, keepdims=True)
	  return e/s

	attn_repeatvector = RepeatVector(Tx) #to repeat previous decoder-state s(t-1) over Tx times
	attn_concatenate = Concatenate(axis=-1) #to concatenate s(t-1) with every encoder hidden_state
	attn_dense = Dense( 10, activation="tanh" ) #a dense layer
	attn_dropout = Dropout(0.1)
	attn_alpha = Dense( 1, activation=softmax_over_time ) #to get importance of each feature map
	attn_context = Dot(axes=1) # weighted(importnace) sum of each unit of all flattened feature map

	def one_step_attention( a, s_prev ):
	  x = attn_repeatvector( s_prev ) #( None, 512, M2)
	  x = attn_concatenate( [ a, x ] ) #( None, 512, 2*M1+M2)
	  x = attn_dense(x) #(None, Tx, 10)
	  x = attn_dropout(x)
	  alphas = attn_alpha(x) #(None, Tx, 1)
	  context = attn_context([alphas, a]) #(None, 1, 2*M1)
	  return context


	#########################Decoder Part##########################
	#take embedding of all decoder input which is actually output one step behind (for teacher forcing)
	#for each output time step t,
	#	generate context
	#	concat s(t-1) with Y(t-1)
	#	pass this through dense layer
	#	pass this thorugh decoder LSTM, get output and replace hidden state, and cell state with older one
	#	pass output to dense layer with softmax activation to get next word probabilities.
	# 	append this to a list
	#output list is of shape ( Ty, None, VOCAB_SIZE ), change it to (None, Ty, VOCAB_SIZE) using lambda layer
	decoder_input = Input( shape=(Ty,) ) #(None, Ty)
	initial_decoder_h = Input( shape=( LATENT_DIM, ) ) #(None, M2)
	initial_decoder_c = Input( shape=( LATENT_DIM, ) ) #(None, M2)

	embedding_layer = Embedding( VOCAB_SIZE, 50, weights=[embedding_matrix], trainable=False )
	decoder_inp_emb = embedding_layer(decoder_input) #None, Ty, 50

	concat_context_word_prev = Concatenate(axis=-1, name="word_context_concat")
	decoder_lstm = LSTM( LATENT_DIM, return_state=True, name="decoder_lstm", recurrent_dropout=0.1 )
	dense_context = Dense(100, activation='tanh', name="decoder_context")
	dropout_context = Dropout(0.1)
	dense_decoder = Dense( VOCAB_SIZE, activation='softmax', name="decoder_dense") 

	s = initial_decoder_h #(None, M2)
	c = initial_decoder_c #(None, M2)

	outputs = [] #to save each decoding timestep output
	for t in range(Ty):
	  context = one_step_attention( a, s ) #(None, 1, 2*M1)

	  selector = Lambda( lambda x: x[:,t:t+1], name=f"selector_{t}" ) #for teacher forcing

	  word_embedding = selector(decoder_inp_emb) #(None, 1, 50)
	  context_word = concat_context_word_prev([context,word_embedding]) #(None, 1, 2*M1+50)
	  context_word = dense_context( context_word ) #(None, 1, 100)
	  context_droput = dropout_context(context_word) #(None, 1, 100)

	  output, s, c = decoder_lstm( context_droput, initial_state=[s,c] ) #(None , 1, M2), (None, M2), (None, M2)
	  output = dense_decoder(output) #(None, 1, DECODER_VOCAB_SIZE)
	  outputs.append(output) # after loop it will be a list of length Ty (None, DEOCDER_VOCAB_SIZE)

	# to change outputs shape to (None, Ty, DEOCDER_VOCAB_SIZE)
	def stack_and_transpose(x):
	  x = K.stack(x) # it will convert list to a tensor of ( Ty, None, DEOCDER_VOCAB_SIZE )
	  x = K.permute_dimensions( x, pattern=(1,0,2) ) #(None, Ty, DEOCDER_VOCAB_SIZE)
	  return x

	stacker = Lambda(stack_and_transpose, name="stack_and_transpose")
	outputs = stacker(outputs) #(None, Ty, DEOCDER_VOCAB_SIZE)
	model1 =  Model( [ vgg19.input, decoder_input, initial_decoder_h, initial_decoder_c ], outputs )


	##########################Model to generate caption##############################
	# generate 512 flattened feature maps of size 196, this is already done in previous model.
	# instead of teacher forcing we will use previous ouptut, initialize that with <sos>
	# get word embedding of previous output/token
	# We will generate next word step by step Ty times 
	token = Input( shape=(1,) ) #( None, 1 ) #init_token - <sos>
	word_embedding = embedding_layer(token) #( None, 1, 50)
	s, c = initial_decoder_h, initial_decoder_c

	token_new = token
	next_token = Lambda( lambda x: K.expand_dims(K.argmax(x,axis=-1), axis=-1), name="next_token")
	outputs = []
	alphas = []
	for _ in range(Ty):
	  context = one_step_attention( a, s ) #(None ,1, 2*M1), (None, Tx, 1)
	  
	  context_word = concat_context_word_prev([ context, word_embedding ])
	  context_word = dense_context( context_word ) #(None, 1, 100)

	  output, s, c = decoder_lstm( context_word, initial_state=[s,c] ) #(None, 1, M2), (None, M2), (None, M2)
	  output = dense_decoder(output) #(None, 1, DECODER_VOCAB_SIZE)

	  token_new = next_token( output ) # (None, 1)
	  word_embedding = embedding_layer(token_new) # ( None, 1, 50 )

	  outputs.append(output) # after loop it will be a list of length Ty (None ,DEOCDER_VOCAB_SIZE) 

	outputs = stacker(outputs)
	caption_generator = Model( [ vgg19.input, token, initial_decoder_h, initial_decoder_c ], outputs )

	return model1, caption_generator