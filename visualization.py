import matplotlib.pyplot as plt
import numpy as np


"""
Create the visualization of learning curve
"""
def plot_learning_curve(history, t):
	plt.plot(history.history[t])
	plt.plot(history.history['val_' + t])
	plt.title('model ' + t)
	plt.ylabel(t)
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig('figure/learning_curve/' + t + '_learning_curve.png')
	plt.clf()


"""
Visualiztion of the confusion matrix
we finally get a colored confusion matrix
it will be more clear to compare the classificaiton rate of each class
"""
def visualize_confusion_matrix(confusion_matrix, model_type):
	print('R/P   1   2   3   4   5   6   7   8   9   10')
	for i in range(len(confusion_matrix)):
		print(i + 1, end='   ')
		for j in range(len(confusion_matrix[i])):
			print(confusion_matrix[i][j], end='   ')
		print()

	# plot
	fig, ax = plt.subplots()

	tick_y = ["r_1", "r_2", "r_3", "r_4", "r_5", "r_6", "r_7", "r_8", "r_9", "r_10"]
	tick_x = ["p_1", "p_2", "p_3", "p_4", "p_5", "p_6", "p_7", "p_8", "p_9", "p_10"]

	ax.set_xticks(np.arange(len(tick_x)))
	ax.set_yticks(np.arange(len(tick_y)))

	ax.set_xticklabels(tick_x)
	ax.set_yticklabels(tick_y)

	for i in range(len(tick_x)):
		for j in range(len(tick_y)):
			text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="r")

	ax.set_title(model_type + " cnn confusion matrix")
	im = ax.imshow(confusion_matrix)
	fig.tight_layout()
	plt.savefig('figure/confusion_matrix/' + model_type + '_confusion_matrix.png')
	# plt.show()


"""
Using matlab library to generate the audio waveform of specific sample
"""
def visualize_waveform(record,save_name, save_path):
	record = record.reshape(-1)
	plt.figure()
	plt.title(save_name)
	plt.plot(record)
	plt.savefig(save_path + save_name+'.png')
	# plt.show()


"""
Our main function for visualization
"""
def main():
	sample_data = np.load('validation_data.npy')
	sample_label = np.load('validation_label.npy')
	visualize_list = []
	for i in range(sample_label.shape[0]):
		data = sample_data[i]
		label = sample_label[i]
		if label not in visualize_list:
			visualize_waveform(data, '1-D waveform with class ' + str(label), 'figure/1D_waveform_sample/')
			visualize_list.append(label)

if __name__ == '__main__':
	main()














