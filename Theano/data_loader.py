import read_data_fast as rd
def load_data():
	rd.read_data_()
	datasets = [[], []]
	datasets[0] = rd.read_training_batch()
	datasets[1] = rd.read_test()
	return datasets
