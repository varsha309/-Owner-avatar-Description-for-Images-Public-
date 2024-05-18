import tensorflow
from tensorflow import keras

#from tensorflow.compat.v1.keras.backend import set_session
#config = tensorflow.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#set_session(tensorflow.compat.v1.Session(config=config))

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from os import listdir
from pickle import dump
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	model = InceptionResNetV2()
	# re-structure the model
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# summarize
	print(model.summary())
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(299, 299))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % name)
	return features

# extract features from all images
directory = 'flickr30k_images'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features_InceptionResNetV2.pkl', 'wb'))
