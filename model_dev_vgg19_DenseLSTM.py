from numpy import array
from pickle import load

#To use 0.5 % GPU
#import tensorflow
#from tensorflow.compat.v1.keras.backend import set_session
#config = tensorflow.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#set_session(tensorflow.compat.v1.Session(config=config))


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Average
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Lambda
import os
import tensorflow as tf
#from tensorflow.keras.utils import plot_model

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(tokens) > 1:
		# split id from description
		    image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		    if image_id in dataset:
			# create list
			    if image_id not in descriptions:
				    descriptions[image_id] = list()
			# wrap description in tokens
			    desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			    descriptions[image_id].append(desc)
		else:
		    pass
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features
	
	
# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
	
# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
	X1, X2, y = list(), list(), list()
	# walk through each description for the image
	for desc in desc_list:
		# encode the sequence
		seq = tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)

# Define the custom layer
class WeightedSum(Layer):
    def __init__(self, a, **kwargs):
        self.a = a
        super(WeightedSum, self).__init__(**kwargs)
    def call(self, model_outputs):
        return self.a * model_outputs[0] + (1 - self.a) * model_outputs[1]
    def compute_output_shape(self, input_shape):
        return input_shape[0]
        


        	
# define the captioning model
def define_model(vocab_size, max_length):   
  # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256,return_sequences=True)(se1)
    se3 = LSTM(256,return_sequences=True)(se2)
    se3_ = Lambda(lambda x : x*0.75)(se3)
    se2_ = Lambda(lambda x : x*0.25)(se2)
    temp1_ = Add()([se3_,se2_]) 
    se4 = LSTM(256,return_sequences=True)(temp1_)
    se4_ = Lambda(lambda x : x*0.8)(se4)
    se3_ = Lambda(lambda x : x*0.1)(se3)
    se2_ = Lambda(lambda x : x*0.1)(se2)
    temp2_ = Add()([se4_,se3_,se2_])
    se5 = LSTM(256)(temp2_)
    # decoder model
    decoder1 = add([fe2, se5]) # each element (element-wise add) in the resulting tensor is the sum of the corresponding elements in fe2 and se5
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics =[tf.keras.metrics.Accuracy()] )
    # summarize mode
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model
	
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
			yield [in_img, in_seq], out_word
	
	


# load training dataset (~25k)
filename = 'Flickr_30k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))

# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features_vgg19.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1 # Look at the bottom for explaination
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)


# define the model
model = define_model(vocab_size, max_length)
# train the model, run epochs manually and save after each epoch
epochs = 21
steps = len(train_descriptions)
os.chdir('model/VGG19_DenseLSTM_20')
for i in range(0,epochs,2):
	# create the data generator
	generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
	# fit for one epoch
	model.fit_generator(generator, epochs=i, steps_per_epoch=steps, verbose=1)
	# save model
	model.save('model_vgg19_Denselstm' + str(i) + '.h5')


"""
### READ - 1
'vocab_size' and 'max_length' are important parameters used in the process of preparing the data and defining the model for image captioning. 

1. vocab_size:
vocab_size represents the size of the vocabulary, i.e., the total number of unique words in the text data. In the context of image captioning, 
it is the number of unique words present in the descriptions of images in your training dataset. This value is determined by the tokenizer, 
which is responsible for mapping words in the text to numerical indices.

In the code, vocab_size is calculated as the length of the word index of the tokenizer plus 1. 
The +1 is added because the indexing of words usually starts from 1, and 0 is reserved for padding. The tokenizer is created using the create_tokenizer function, and it is fitted on the training descriptions.

2. max_length:
max_length represents the maximum length of a sequence (or sentence) in terms of the number of words. In the context of image captioning, 
it is the maximum number of words in any image description in your training dataset. This value is crucial for padding sequences to a fixed 
length so that they can be fed into the model for training or inference.

The max_length is calculated using the max_length function, which computes the length of the description with the most words. 
This ensures that all sequences are padded or truncated to this length.

In summary, vocab_size is the total number of unique words in your training data, and max_length is the maximum length of a sequence 
(in terms of words) in your training data. These values are important for creating the model architecture and preparing the input data for training.
"""


"""
### READ - 2
The Embedding layer is responsible for converting the integer-encoded words into dense vectors of fixed size (256 in this case). 
It essentially creates a dense representation of the words in the descriptions.

Before feeding the text descriptions into the LSTM training model, first the text embedding has been extracted and this is send to the LSTM block.
"""


"""
### READ - 3

The use of Lambda(lambda x: x**____) below is to introduce a weighting factor. The output of LTSM layer is multiplied element-wise by the scalar value.
The purpose of introducing these scalar weights is to control the contribution of each LSTM layer's output to the final combined output. 

    se3_ = Lambda(lambda x : x*0.75)(se3)
    se2_ = Lambda(lambda x : x*0.25)(se2)
    se4_ = Lambda(lambda x : x*0.8)(se4)
    se3_ = Lambda(lambda x : x*0.1)(se3)
    se2_ = Lambda(lambda x : x*0.1)(se2)

"""



