import tensorflow as tf
import numpy as np

def _parseme(raw_audio_record):
	feature_description = {
		'note': tf.io.FixedLenFeature([], tf.int64, default_value=0),
		'note_str': tf.io.FixedLenFeature([], tf.string, default_value=''),
		'instrument': tf.io.FixedLenFeature([], tf.int64, default_value=0),
		'instrument_str': tf.io.FixedLenFeature([], tf.string, default_value=''),
		'pitch': tf.io.FixedLenFeature([], tf.int64, default_value=0),
		'velocity': tf.io.FixedLenFeature([], tf.int64 ,default_value=0),
		'sample_rate': tf.io.FixedLenFeature([], tf.int64, default_value=0),
		'audio': tf.io.FixedLenSequenceFeature([], tf.float32,  allow_missing=True, default_value=0.0),
		'qualities': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
		'qualities_str': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value=''),
		'instrument_family': tf.io.FixedLenFeature([], tf.int64, default_value=0),
		'instrument_family_str': tf.io.FixedLenFeature([], tf.string, default_value=''),
		'instrument_source': tf.io.FixedLenFeature([], tf.int64, default_value=0),
		'instrument_source_str': tf.io.FixedLenFeature([], tf.string, default_value='')
	}
	return tf.io.parse_single_example(raw_audio_record, feature_description)


"""
Preprocess the data before our training process
because the test set does not include any samples for the 'synth lead' class (number 9)
we remove these data from both training and validation sets
"""
def processData(ar, prefix, save_data):
	data = []
	label = []
	for r in ar:
		if int(r['instrument_family'].numpy()) == 9:
			pass
		else:
			data.append(np.mean(r['audio'].numpy().reshape(-1, 100), axis=1)[:640])
			if int(r['instrument_family'].numpy()) == 10:
				label.append(9)
			else:
				label.append(int(r['instrument_family'].numpy()))

	data = np.array(data).reshape(len(data), 640, 1)
	data = np.array(data)
	label = np.array(label)

	if save_data:
		np.save(prefix + 'data.npy', data)
		np.save(prefix + 'label.npy', label)

	print(data.shape, label.shape)
	return data, label


"""
The first step to process the raw data
we need to load the data into memeory
"""
def get_data(data_dir, save_data=False):
	trainingData = tf.data.TFRecordDataset(data_dir + "nsynth-train.tfrecord")
	validationData = tf.data.TFRecordDataset(data_dir + "nsynth-valid.tfrecord")
	testData = tf.data.TFRecordDataset(data_dir + "nsynth-test.tfrecord")

	ar = trainingData.map(_parseme)
	tr_d, tr_l = processData(ar, 'training_', save_data)

	br = validationData.map(_parseme)
	v_d, v_l = processData(br, 'validation_', save_data)

	cr = testData.map(_parseme)
	te_d, te_l = processData(cr, 'testing_', save_data)

	return tr_d, tr_l, v_d, v_l, te_d, te_l

def main():
	get_data('/home/fac/cmh/tfrecord/', True)

if __name__ == '__main__':
	main()
















