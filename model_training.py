# training models

import numpy as np 
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import tensorflow as tf 
from keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Flatten, concatenate, Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import time

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def GetFalseTruePositiveRate(y_true, y_prob, threshold):

    y_predict = np.fromiter([1 if x > threshold else 0 for x in y_prob ], int)
    n_positives = y_true.sum()
    n_negatives = y_true.shape[0] - n_positives
    
    # get n true positives
    n_true_pos = 0
    n_false_pos = 0
    for pred_value,true_value in zip(y_predict,y_true):
        # true positive
        if true_value == 1 and pred_value == 1:
            n_true_pos += 1
        # false positive
        elif true_value == 0 and pred_value == 1:
            n_false_pos += 1
    true_pos_rate = n_true_pos/n_positives
    false_pos_rate = n_false_pos/n_negatives
    return false_pos_rate,true_pos_rate

def plot_graphs(history):
    """
    Function used to plot accuracy and loss of model
    :param: history: from Sequential()
    """
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("acc.png")
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")

def get_model(X, meta):
	image_input = Input(shape=X.shape[1:])
	metadata_input = Input(shape=meta.shape[1:])

	conv1 = Conv2D(layer_size, kernel_size=3, activation='relu')(image_input)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(layer_size, kernel_size=3, activation='relu')(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(layer_size, kernel_size=3, activation='relu')(pool2)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	flat = Flatten()(pool3)
	merge = concatenate([flat, metadata_input])

	hidden1 = Dense(layer_size, activation='relu')(merge)
	hidden2 = Dense(layer_size, activation='relu')(hidden1)

	output = Dense(1, activation='sigmoid')(hidden2)
	model = Model(inputs=[image_input, metadata_input], output=output)
	# summarize layers
	print(model.summary())
	# plot graph
	plot_model(model, to_file=NAME + '.png')
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model


data_dir = "D:/CheXpert-v1.0-small/preprocessed_binary_data/50x50/"
pathologies = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Consolidation', 'Pleural Effusion']

num_classes = 2

for pathology in pathologies:
	print(pathology)
	pathology_dir = os.path.join(data_dir, pathology)
	classes = ['Negative', 'Positive']
	'''
	X_train = np.load(os.path.join(pathology_dir, "X_train_%s.npy" % pathology))
	X_train = np.reshape(X_train, (X_train.shape[0], 50, 50, 1))
	y_train = np.load(os.path.join(pathology_dir, "y_train_%s.npy" % pathology))
	meta_train = np.load(os.path.join(pathology_dir, "metadata_%s.npy" % pathology))
	'''
	X_test = np.load(os.path.join(pathology_dir, "X_test_%s.npy" % pathology))
	X_test = np.reshape(X_test, (X_test.shape[0], 50, 50, 1))
	y_test = np.load(os.path.join(pathology_dir, "y_test_%s.npy" % pathology))
	meta_test = np.load(os.path.join(pathology_dir, "meta_test_%s.npy" % pathology))


	'''
	sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
	for train_index, test_index in sss.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		meta_train, meta_test = meta[train_index], meta[test_index]
	'''
	#y_train = to_categorical(y_train, num_classes)
	#y_test = to_categorical(y_test, num_classes)

	new_meta_test = []
	for row, i in enumerate(meta_test):
		if len(i) == 3: new_meta_test.append(i)
		elif len(i) != 3:
			i.insert(0, 1)
			new_meta_test.append(i)

	meta_test = np.array(new_meta_test)			
	print(X_test.shape, meta_test.shape, y_test.shape)

	dense_layer = 2
	layer_size = 100
	conv_layer = 3

	NAME = "{}-{}-conv-{}-nodes-{}-dense".format(pathology, conv_layer, layer_size, dense_layer)
	print(NAME)

	model = load_model(os.path.join("D:/CheXpert-v1.0-small/models/50x50/", pathology, NAME + ".hdf5"))
	y_pred = model.predict([X_test, meta_test])
	y_predict = []
	for _ in y_pred:
		if _[0] > 0.6: y_predict.append(1)
		elif _[0] <= 0.6: y_predict.append(0)

	loss, accuracy = model.evaluate([X_test, meta_test], y_test)
	y_test = to_categorical(y_test, num_classes)
	y_predict = to_categorical(y_predict, num_classes)

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	acc = dict()
	loss = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(num_classes):
	    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predict[:, i])
	    roc_auc[i] = auc(fpr[i], tpr[i])


	for i in range(num_classes - 1):
		plt.figure()
		plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC-AUC curve: ' + pathology)
		#plt.text(s=text,x=0.1, y=0.8,fontsize=20)
		plt.legend(loc="lower right")
		plt.show()

	#lot_confusion_matrix(y_test, r_pred, classes=classes, title='Confusion matrix without normalization for %s' % pathology)
	#plt.show()



	'''
	model = get_model(X_train, meta_train)

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)

		class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
		model_checkpoint = ModelCheckpoint(NAME + '.hdf5', monitor='loss', save_best_only=True)
		tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
		
		history = model.fit([X_train, meta_train], y_train,
				batch_size=32,  
				epochs=10, 
				verbose=True,
				shuffle=True,
				validation_split=0.2,
				callbacks=[tensorboard, model_checkpoint],
				class_weight=class_weights)

		#model.save(NAME + '.model')
		plot_graphs(history)
	'''