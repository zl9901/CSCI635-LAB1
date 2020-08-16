from tensorflow.keras import layers, models, utils
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import sys

from process_data import get_data
from visualization import *


"""
This model is named train_naive_cnn which is not the best network we created first time
there are few convolution2D, maxpooling and dense layers in this naive CNN network
we also create learning curve in this function
"""
def train_naive_cnn(training_data ,training_label ,dev_data ,dev_label):
	training_label = utils.to_categorical(training_label)
	dev_label = utils.to_categorical(dev_label)

	model = models.Sequential()
	model.add(layers.Conv1D(filters=1, kernel_size=10, padding='same', activation='relu', input_shape=(640, 1)))
	model.add(layers.MaxPooling1D(pool_size=8))
	model.add(layers.Conv1D(filters=2, kernel_size=10, padding='same', activation='relu'))
	model.add(layers.MaxPooling1D(pool_size=8))
	model.add(layers.Flatten())
	model.add(layers.Dense(10, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

	history = model.fit(training_data, training_label, epochs=50, callbacks=[tensorboard], batch_size=1000,
						validation_data=(dev_data, dev_label))

	dev_loss, dev_acc = model.evaluate(dev_data, dev_label, verbose=2)
	print(dev_acc)

	plot_learning_curve(history, 'accuracy')

	return model


"""
This model is named train_optimized_cnn which is the best network we plan to use finally
we add more convolution, maxpooling and dense layers
we also implement the dropout in this model to avoid overfitting issue
besides learning we created, please use tensorboard to view more details
"""
def train_optimized_cnn(training_data ,training_label ,dev_data ,dev_label):
	model = models.Sequential()
	model.add(layers.Conv1D(filters=16, kernel_size=10, padding='same', activation='tanh', input_shape=(640, 1)))
	model.add(layers.MaxPooling1D(pool_size=2))
	model.add(layers.Dropout(rate=0.15))

	model.add(layers.Conv1D(filters=32, kernel_size=10, padding='same', activation='tanh'))
	model.add(layers.MaxPooling1D(pool_size=2))

	model.add(layers.Conv1D(filters=32, kernel_size=10, padding='same', activation='tanh'))
	model.add(layers.MaxPooling1D(pool_size=2))
	model.add(layers.Dropout(rate=0.2))


	model.add(layers.Conv1D(filters=64, kernel_size=10, padding='same', activation='tanh'))
	model.add(layers.MaxPooling1D(pool_size=2))

	model.add(layers.Conv1D(filters=128, kernel_size=10, padding='same', activation='tanh'))
	model.add(layers.MaxPooling1D(pool_size=4))
	model.add(layers.Dropout(rate=0.2))

	model.add(layers.Flatten())
	model.add(layers.Dense(256, activation='tanh'))
	model.add(layers.Dense(128, activation='tanh'))
	model.add(layers.Dense(64, activation='tanh'))

	model.add(layers.Dense(10, activation='softmax'))

	# to use tensorboard: tensorboard --logdir=logs/
	tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	history = model.fit(training_data, training_label, epochs=60, callbacks=[tensorboard], batch_size=1000,
						validation_data=(dev_data, dev_label))

	dev_loss, dev_acc = model.evaluate(dev_data, dev_label, verbose=2)
	print(dev_acc)

	plot_learning_curve(history, 'accuracy')

	return model


"""
This function is used to test the performance of our model
we generate confusion matrix and visualize the confusion matrix
so it will be more clear to compare the classificaiton rate for each class
"""
def test(model ,test_data ,test_label, model_type):

	"""
	check performance of test dataset
	"""
	yhat = model.predict(test_data)

	"""
	generate confusion matrix
	"""
	matrix = confusion_matrix(np.argmax(yhat, axis=1), test_label)
	visualize_confusion_matrix(matrix, model_type)

	visualize_waveform_samples(yhat, test_label, test_data)

"""
This function is designed to pick up the waveforms from test samples of each class

for each class, waveform for samples where the correct class probability is very high and very low
we generate both high probability and low probablity waveforms

for each class, waveforms for samples near the decision boundary
"""
def visualize_waveform_samples(pred, label, test_data):
	for label_class in range(10):
		curr_class_pred = []
		curr_index_list = []
		for i in range(label.shape[0]):
			if label[i] == label_class:
				curr_class_pred.append(pred[i])
				curr_index_list.append(i)
		curr_class_pred = np.array(curr_class_pred)
		curr_label_pred = curr_class_pred[:, label_class]
		max_index = curr_index_list[int(np.argmax(curr_label_pred))]
		min_index = curr_index_list[int(np.argmin(curr_label_pred))]

		visualize_waveform(test_data[min_index], 'class ' + str(label_class) + ' min prob waveform',
						   'figure/min_prob_waveform_sample/')
		visualize_waveform(test_data[max_index], 'class ' + str(label_class) + ' max prob waveform',
						   'figure/max_prob_waveform_sample/')

		if label_class == 0:
			curr_class_other_pred = curr_class_pred[:, label_class + 1:]
		elif label_class == 9:
			curr_class_other_pred = curr_class_pred[:, :label_class]
		else:
			curr_class_other_pred = np.concatenate((curr_class_pred[:, :label_class], curr_class_pred[:, label_class + 1:]), axis=1)
		max_class_other_pred = np.amax(curr_class_other_pred, axis=1)
		diff = np.absolute(max_class_other_pred - curr_label_pred)
		boundary_index = curr_index_list[int(np.argmin(diff))]

		visualize_waveform(test_data[boundary_index], 'class ' + str(label_class) + ' boundary waveform',
						   'figure/boundary_waveform_sample/')


def main():
	machine = 'l'  # use preprocessed data locally by default
	model_type = 'naive'  # run the naive model by default
	train_new = 'y'  # train new model by default

	if len(sys.argv) == 4:
		machine = sys.argv[1]
		model_type = sys.argv[2]
		train_new = sys.argv[3]

	if machine == 'l':
		tr_d = np.load('training_data.npy')
		tr_l = np.load('training_label.npy')
		v_d = np.load('validation_data.npy')
		v_l = np.load('validation_label.npy')
		te_d = np.load('testing_data.npy')
		te_l = np.load('testing_label.npy')
	else:
		tr_d, tr_l, v_d, v_l, te_d, te_l = get_data('/home/fac/cmh/tfrecord/', False) # use CS machine dataset

	# Train naive model and get test result
	if model_type == 'naive':
		# Train the model form very beginning
		if train_new == 'y':
			model = train_naive_cnn(tr_d, tr_l, v_d, v_l)
			model.reset_metrics()
			model.save('model/naive_model.h5')
		# Directly load the model without training
		else:
			model = models.load_model('model/naive_model.h5')
		test(model, te_d, te_l, 'naive')
	# Train optimized model and get test result
	else:
		if train_new == 'y':
			model = train_optimized_cnn(tr_d, tr_l, v_d, v_l)
			model.reset_metrics()
			model.save('model/optimized_model.h5')
		else:
			model = models.load_model('model/optimized_model.h5')
		test(model, te_d, te_l, 'optimized')

if __name__ == '__main__':
	main()














