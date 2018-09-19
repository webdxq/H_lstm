
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class ModelConfig(object):
	"""Wrapper class for model hyperparameters."""
	def __init__(self):
		self.num_layers = 2
		self.hidden_size = 128
		self.model = "lstm"
		

class DataConfig(object):
	"""Wrapper class for DataCofig hyperparameters."""
	def __init__(self, word_size =6937):
		self.minlen = 4
		self.maxlen = 15
		self.word_size = word_size
		self.cantoneses = "/home/pingan_ai/dxq/project/cantonese.txt"
		self.lyrics_file = '/media/pingan_ai/dxq/gen_blessing/dataset/hierarchical_line_lyrics.txt'
		self.id2word = "./line_lyrics2word_re.json"
		self.word2id = "./line_lyrics2id_re.json"
		
class TrainingConfig(object):
	"""Wrapper class for TrainingConfig hyperparameters."""
	def __init__(self):
		self.batch_size = 16
		self.epoch = 10
		self.num_gpus = 2
		self.checkpoint_path = "/media/pingan_ai/dxq/gen_blessing/H_lstm/model/"
		
class TestingConfig(object):
	"""Wrapper class for TestingConfig hyperparameters."""
	def __init__(self):
		self.bach_size = 1
		self.checkpoint_path = "/media/pingan_ai/dxq/gen_blessing/H_lstm/model/"
		# self.checkpoint_path = "/media/pingan_ai/dxq/gen_blessing/models_1/"

class Sentence_train(object):
	def __init__(self):
		self.minlen = 4
		self.maxlen = 15
		self.word_size = word_size
		self.cantoneses = "/home/pingan_ai/dxq/project/cantonese.txt"
		self.lyrics_file = '/home/pingan_ai/dxq/project/gen_blessing/dataset/data/line_lyrics.txt'
		self.id2word = "./line_lyrics2word_re.json"
		self.word2id = "./line_lyrics2id_re.json"