#Preprocess_images_per_class

import pandas as pd 
import numpy as np 
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt 
from sklearn.utils.class_weight import compute_class_weight

def pull_image_path_from_labels(df, label):
	label_df = df[['Path', 'Sex', 'Age', 'Frontal/Lateral', label]]
	label_df = label_df.dropna()
	train_paths = []
	for index, row in label_df.iterrows():
		if row[label] == -1: train_paths.append([row['Path'], -1])
		elif row[label] == 0: train_paths.append([row['Path'], 0])
		elif row[label] == 1: train_paths.append([row['Path'], 1])
	return label_df, train_paths

def create_training_arrays(train_paths, label_df):
	X_train = []
	y_train = []
	metadata = []
	uncertain_data = []
	label_df.set_index('Path', inplace=True)
	for list_ in tqdm(train_paths):
		label_row = label_df.ix[list_[0]]
		img_path = "D:/" + list_[0]
		img = cv2.imread(img_path, 0)
		img = cv2.resize(img, (50, 50)).astype('uint8')

		# normalize pixel values
		mean = np.mean(img)
		std = np.std(img)
		img = img - mean
		img = img / std

		#retreive metadata
		meta = []
		if label_row['Sex'] == 'Female': meta.append(0)
		elif label_row['Sex'] == 'Male': meta.append(1)
		meta.append(label_row['Age'] / 100)
		if label_row['Frontal/Lateral'] == 'Frontal': meta.append(0)
		elif label_row['Frontal/Lateral'] == 'Lateral': meta.append(1)

		if list_[1] == -1: uncertain_data.append([img, meta])
		
		elif list_[1] == 0: 
			X_train.append(img)
			metadata.append(meta)
			y_train.append(0)
		elif list_[1] == 1:
			X_train.append(img)
			metadata.append(meta)
			y_train.append(1)

	return X_train, y_train, metadata, uncertain_data


train_df = pd.read_csv('D:/CheXpert-v1.0-small/train.csv')
test_df = pd.read_csv('D:/CheXpert-v1.0-small/valid.csv')
pathologies = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Consolidation', 'Pleural Effusion']


for pathology in pathologies[1:]:
	print('Preprocessing for ', pathology)

	label_df, train_paths = pull_image_path_from_labels(train_df, pathology)
	X_train, y_train, metadata, uncertain_data = create_training_arrays(train_paths, label_df)

	y_label, test_paths = pull_image_path_from_labels(test_df, pathology)
	X_test, y_test, y_metadata, y_uncertain = create_training_arrays(test_paths, y_label)

	print(len(X_test), len(y_test), len(y_metadata), len(y_uncertain))
	print('Saving data to numpy files...')

	np.save('X_train_%s.npy' % pathology, np.array(X_train))
	np.save('y_train_%s.npy' % pathology, np.array(y_train))
	np.save('metadata_%s.npy' % pathology, np.array(metadata))
	np.save('uncertain_data_%s.npy' % pathology, np.array(uncertain_data))
	np.save('X_test_%s.npy' % pathology, np.array(X_test))
	np.save('y_test_%s.npy' % pathology, np.array(y_test))
	np.save('meta_test_%s.npy' % pathology, np.array(y_metadata))