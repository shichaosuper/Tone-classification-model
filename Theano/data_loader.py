import read_data_fast as rd
def load_data(iter_num, batch):
	rd.read_data_()
	datasets = [[], []]
	datasets[0] = rd.read_training_batch(int((iter_num + batch) / batch), batch)
	datasets[1] = rd.read_test()
	return datasets