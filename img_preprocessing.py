from scipy import misc
import pickle
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load image (subfunction of load_preprocess_pickle_data_from_initial_file())
def load_image(name):
	# Read image
	return misc.imread(name)

def crop_image(img):
	return img[56:120,:,:]

def select_color_channel(img,color_channel):
	if color_channel == "r":
		return img[:,:,0]
	if color_channel == "g":
		return img[:,:,1]
	if color_channel == "b":
		return img[:,:,2]


# Preprocess image (subfunction of load_preprocess_pickle_data_from_initial_file())
def preprocess_image(img, size):

	# Resize image
	img = misc.imresize(img,size)

	# Set image values to "float32"
	img = img.astype("float32")

	# Normalize image values between -0.5 and 0.5
	img = (img/255)-0.5

	return img


# If data is not already converted and stored in pickle file yet
# Load and convert data; store preprocessed data in pickle files
def load_preprocess_pickle_data_from_initial_file(image_rescale_size,color_channel):
	
	# Import csv-file to be used as basic data reference
	csv_data = pd.read_csv("driving_log.csv")

	# Load and preprocess images
	X = []
	for img in csv_data["center"]:
		loaded_image = load_image(img)
		#print(loaded_image.shape)
		cropped_image = crop_image(loaded_image)
		#print(cropped_image)
		color_channel_image = select_color_channel(cropped_image,color_channel)
		final_image = preprocess_image(color_channel_image,image_rescale_size)
		X.append(final_image[:,:,np.newaxis])
	X = np.array(X)
	# Output example images
	loaded_image = load_image(csv_data["center"][0])
	cropped_image = crop_image(loaded_image)
	color_channel_image = select_color_channel(cropped_image,color_channel)
	print(color_channel_image.shape)
	final_image = preprocess_image(color_channel_image,image_rescale_size)
	plt.imsave("example_image_1_before_preprocessing.jpg",loaded_image)
	plt.imsave("example_image_2_after_cropping.jpg",cropped_image)
	plt.imsave("example_image_3_after_color_channel.jpg",color_channel_image,cmap=plt.cm.gray)
	plt.imsave("example_image_4_after_preprocessing.jpg",final_image,cmap=plt.cm.gray)


	# Load image labels
	y = np.array(csv_data["steering"])

	# Dump preprocessed X in pickle file for future reuse
	with open("behavioral_cloning_X.p","wb") as f:
		pickle.dump(X,f)

	# Dump preprocessed y in pickle file for future reuse
	with open("behavioral_cloning_y.p","wb") as f:
		pickle.dump(y,f)

	print("Data was converted and stored in the pickle files 'behavioral_cloning_X.p' and 'behavioral_cloning_y.p' for future use")
	print()
	return X,y


# Load data from pickle file (used if pickle files are available)
def load_data_from_pickle():
		
	# Load X from pickle file
	with open("behavioral_cloning_X.p","rb") as f:
		X = pickle.load(f)

	# Load y from pickle file
	with open("behavioral_cloning_y.p","rb") as f:
		y = pickle.load(f)

	print("Data loaded from pickle files 'behavioral_cloning_X.p' and 'behavioral_cloning_y.p'")
	print()
	return X,y