from DataReader import DataReader
from RNN_1 import RNN

seq_length = 25
#read text from the "input.txt" file
data_reader = DataReader("input.txt", seq_length)
rnn = RNN(hidden_size=100, vocab_size=data_reader.vocab_size,seq_length=seq_length,learning_rate=1e-1)
rnn.train(data_reader)

rnn.predict(data_reader, 'get', 50)