#!/usr/bin/python3
#-*- coding: UTF-8 -*-
import collections  
import numpy as np  
import os 
import re
import json
import sys
import configuration
import chardet
import codecs
import tensorflow as tf
# reload(sys)
# sys.setdefaultencoding('utf8')

def IsCantonese(line,cantonese):
	for i, patten in enumerate(cantonese):
		if patten.search(line):
			return True
	return False

def HasReapeatWord(string):
    flag = False
    for i,char in enumerate(string):
        # print i
        s = i
        m = i+1
        e = i+2 
        if flag:
            return True
        elif e >= (len(string)-1):
            return False
        else:
            if string[s] == string[m] and string[m] == string[e]:
                flag = True
            else:
                continue
def Remove_sentences(line,cantonese,minlen,maxlen):
	if u'_' in line or u'(' in line or u'（' in line or u'《' in line or u'[' in line:  
	    return True  
	if len(line) < minlen or len(line) > maxlen:  
	    return True    
	if IsCantonese(line,cantonese):
	    return True  
	if HasReapeatWord(line):
	    return True  
	return False

def process_data():
	data_config = configuration.DataConfig()
	stop = False
	minlen = data_config.minlen
	maxlen = data_config.maxlen
	id2word = data_config.id2word
	word2id = data_config.word2id
	filename = data_config.lyrics_file.split('.')[0]
	cantoneses = open(data_config.cantoneses,'r').readline().split(' ')
	cantonese = [re.compile(i.decode('utf-8')) for i in cantoneses]
	counts = 0
	blessings = []  
	all_words = [] 
	total_lyrics = []
	with codecs.open(data_config.lyrics_file, "r",'utf-8') as f:
		for i,lines in enumerate(f):
			lines = lines.strip(u'\n')
			lines_temp = lines.split('|')[:-1]
			lyric_flag = True
			if lines == '\n' or lines == "" or len(lines)<100:
				continue
			for line in lines_temp:
				line = line.replace(u' ',u'') 
				if Remove_sentences(line,cantonese,minlen,maxlen):
					lyric_flag = False
					break
				all_words += [word for word in line]
			if lyric_flag:
				# print lines
				lines = u'[' +lines[:-1] + u']' 
				total_lyrics.append(lines)
				counts = i
			if i%10000 == 0:
				print 'finish processed %d'%i
	print 'finish processed %d'%counts
	total_lyrics = sorted(total_lyrics,key=lambda line: len(line))  
	print len(total_lyrics[0]),total_lyrics[0]
	print len(total_lyrics[1]),total_lyrics[1]
	print len(total_lyrics[2]),total_lyrics[2]
	print len(total_lyrics[3]),total_lyrics[3]
	print len(total_lyrics[4]),total_lyrics[4]
	print len(total_lyrics[10]),total_lyrics[10]
	print len(total_lyrics[50]),total_lyrics[50]
	
def LoadDicts():
	data_config = configuration.DataConfig()
	word2id = data_config.word2id
	id2word = data_config.id2word
	# if not os.path.exists(word2id):
	# 	process_data()
	with open(word2id,'r') as ToIdf:
	    word_num_map = json.load(ToIdf)
	with open(id2word,'r') as ToWordf:
	    words = json.load(ToWordf)
	return word_num_map,words

def fill_np_matrix(vects, batch_size, value):
	max_len = max(len(vect) for vect in vects)
	res = np.full([batch_size, max_len], value, dtype=np.int32)
	for row, vect in enumerate(vects):
		res[row, :len(vect)] = vect
	# print res.shape
	return res,max_len

def batch_train_data(batch_size):
	"""Get training data in lyrics, batch major format"""
	
	data_config = configuration.DataConfig()
	stop = False
	minlen = data_config.minlen
	maxlen = data_config.maxlen
	id2word = data_config.id2word
	word2id = data_config.word2id
	filename = data_config.lyrics_file.split('.')[0]
	if not os.path.exists(filename+'.npy'): 
		cantoneses = open(data_config.cantoneses,'r').readline().split(' ')
		cantonese = [re.compile(i.decode('utf-8')) for i in cantoneses]
		counts = 0
		blessings = []  
		all_words = [] 
		total_lyrics = []
		with codecs.open(data_config.lyrics_file, "r",'utf-8') as f:
			for i,lines in enumerate(f):
				lines = lines.strip(u'\n')
				lines_temp = lines.split('|')[:-1]
				lyric_flag = True
				if lines == '\n' or lines == "" or len(lines)<100:
					continue
				for line in lines_temp:
					line = line.replace(u' ',u'') 
					if Remove_sentences(line,cantonese,minlen,maxlen):
						lyric_flag = False
						break
					all_words += [word for word in line]
				if lyric_flag:
					# print lines
					lines = u'[' +lines[:-1] + u']' 
					total_lyrics.append(lines)
					counts = i
				if i%10000 == 0:
					print 'finish processed %d'%i
		print 'finish processed %d'%counts
		total_lyrics = sorted(total_lyrics,key=lambda line: len(line))[:-40] 

		# print(u'lyrics numbers: %s'% len(total_lyrics))  
		counter = collections.Counter(all_words)  
		count_pairs = sorted(counter.items(), key=lambda x: -x[1])  
		# print total_lyrics
		print('*******************')
		words, _ = zip(*count_pairs)  
		# 取前多少个常用字  
		# print(len(words))
		words = words[:len(words)] + (u'[',)  
		words = words[:len(words)] + (u']',)  
		words = words[:len(words)] + (u'|',)  
		words = words[:len(words)] + (u'<unk>',)

		data_config.word_size = len(words) 
		word_num_map = dict(zip(words, range(len(words))))  
		print word_num_map.get(u'<unk>')
		print word_num_map.get('|')
		to_num = lambda word: word_num_map.get(word, len(words)-1) 
		lyrics_vector = [ list(map(to_num,lyric)) for lyric in total_lyrics]  
		print len(lyrics_vector[-1])
		print len(lyrics_vector[0])
		print len(lyrics_vector[0]),lyrics_vector[0]
		# print total_lyrics[0]
		del total_lyrics
		np.save(filename+'.npy', lyrics_vector)
		with codecs.open(word2id,'w','utf-8') as outfile:
		    json.dump(word_num_map,outfile,ensure_ascii=False)
		with codecs.open(id2word,'w','utf-8') as outfile2:
		    json.dump(words,outfile2,ensure_ascii=False)

	else:
		lyrics_vector = np.load(filename+'.npy')
		word_num_map,words = LoadDicts()
	print(len(words))
	print len(lyrics_vector)
	# sys.exit()
	index = 0
	stop = False
	while not stop:
		src = []
		target_lens = []
		
		for i in range(batch_size):
			# print total_lyrics[index]
			src.append(lyrics_vector[index])
			target_lens.append(len(lyrics_vector[index]))
			index += 1
			if index > len(lyrics_vector)-1:
				index = 0
				print 'finish one epoch'
				# stop = True
				# epoch -= 1
				# if epoch <0:
				# 	break
		# print src
		# print epoch,index
		src_padded, max_len = fill_np_matrix(src, batch_size, word_num_map[u']'])
		# src_lens = np.array(max_len)
		# print maxlen
		target = np.copy(src_padded)
		target[:,:-1] = src_padded[:,1:] 
		target_lens = np.array(target_lens) 
		# print target_lens
		yield src_padded, target, target_lens

# def Word2Vec():
def sentence_train(batch_size):
	blessings = []  
	all_words = [] 
	data_config = configuration.Sentence_train()
	stop = False
	minlen = data_config.minlen
	maxlen = data_config.maxlen
	id2word = data_config.id2word
	word2id = data_config.word2id
	filename = data_config.lyrics_file.split('.')[0]
	def IsCantonese(line):
	    for i, patten in enumerate(cantonese):
	        if patten.search(line):
	            return True
	    return False

	def HasReapeatWord(string):
	    flag = False
	    for i,char in enumerate(string):
	        # print i
	        s = i
	        m = i+1
	        e = i+2 
	        # print string[s],string[m],string[e]
	        if flag:
	            return True
	        elif e >= (len(string)-1):
	            return False
	        else:
	            if string[s] == string[m] and string[m] == string[e]:
	                flag = True
	            else:
	                continue
	with open(data_config.lyrics_file, "r") as f:  
	    for i,line in enumerate(f):  
	        if i == 0:
	            continue
	        line = line.decode('UTF-8')
	        line = line.strip(u'\n')
	        line = line.replace(u' ',u'')  
	        if u'_' in line or u'(' in line or u'（' in line or u'《' in line or u'[' in line:  
	            continue  
	        if len(line) < minlen or len(line) > maxlen:  
	            continue  
	        if IsCantonese(line):
	            continue
	        if HasReapeatWord(line):
	            continue
	        all_words += [word for word in line]
	        line = u'[' + unicode(chr(len(line)+61)) +line + u']'  
	        blessings.append(line)        
	        if i%50000== 0:
	            print(u'处理到%d'%i)

	blessings = sorted(blessings,key=lambda line: len(line))  
	print(u'歌词总行数: %s'% len(blessings))  
	counter = collections.Counter(all_words)  
	count_pairs = sorted(counter.items(), key=lambda x: -x[1])  

	print('*******************')
	words, _ = zip(*count_pairs)  
	# 取前多少个常用字  
	print(len(words))

	for i in range(65,66+maxlen-minlen):
	    # print(unicode(chr(i)))
	    words = words[:len(words)] + (unicode(chr(i)),)
	words = words[:len(words)] + (u'[',)  
	words = words[:len(words)] + (u']',)  
	words = words[:len(words)] + (u' ',)
	print(u'词表总数: %s'% len(words))  
	word_num_map = dict(zip(words, range(len(words))))  
	print(word_num_map[u'['])
	print(word_num_map[u']'])
	print(word_num_map[u' '])
	print(word_num_map[u'A'])
	print(word_num_map[u'L'])
	to_num = lambda word: word_num_map.get(word, len(words)-1) 
	blessings_vector = [ list(map(to_num,blessing)) for blessing in blessings]  
	np.save(filename+'.npy', lyrics_vector)
	with codecs.open(word2id,'w','utf-8') as outfile:
	    json.dump(word_num_map,outfile,ensure_ascii=False)
	with codecs.open(id2word,'w','utf-8') as outfile2:
	    json.dump(words,outfile2,ensure_ascii=False)

	index = 0
	stop = False
	while not stop:
		src = []
		target_lens = []
		
		for i in range(batch_size):
			# print total_lyrics[index]
			src.append(lyrics_vector[index])
			target_lens.append(len(lyrics_vector[index]))
			index += 1
			if index > len(lyrics_vector)-1:
				index = 0
				print 'finish one epoch'
				# stop = True
				# epoch -= 1
				# if epoch <0:
				# 	break
		# print src
		# print epoch,index
		src_padded, max_len = fill_np_matrix(src, batch_size, word_num_map[u']'])
		# src_lens = np.array(max_len)
		# print maxlen
		target = np.copy(src_padded)
		target[:,:-1] = src_padded[:,1:] 
		# print target_lens
		yield src_padded, target

def fill_dict(input_data,targets,input_length,fn):
	feed_dict = {}
	for i in range(2):
		x,y,z = fn.next()
		feed_dict[input_data[i]] = x
		feed_dict[targets[i]] = y
		feed_dict[input_length[i]] = z

        
	return feed_dict
	
def train_to_word(x):
    # print(u'x的长度',len(x))
    to_words = lambda num: words[num]
    x_words = map(to_words, x)
    # print(str(x_words).decode("unicode-escape"))
    outstr = ''.join(x_words)
    # token = outstr[1]
    # outstr = outstr[0:-1]
    # print u'[ '+outstr+u' ]'
    print outstr

def AlignSentence(sentence):
    sentence = sentence[:-2]
    sentence_re = ''
    print words[1]
    for i in range(len(sentence)):
        if not (sentence[i] >= u'\u4e00' and sentence[i]<=u'\u9fa5'):
            sentence_re += sentence[i]+u' '
        else:
            sentence_re += sentence[i]
    # return u'[ '+sentence[i] + u' ]'
    print u'[ '+ sentence_re + u' ]'
if __name__ == '__main__':
	# blessing_file = 
	# epoch = 2
	# while epoch > 0:
	# 	for i,j,k in batch_train_data(256):
	# 		print i[0],j[0]
	# 	epoch -= 1

	# process_data()
	# while epoch > 0:
	# 	for i,j,k in batch_train_data(256):
	# 		# print i[0],j[0]
	# 		print ' ',epoch
	# 	epoch -= 1
	# inputs = [[[] for i in range(3)] for i in range(2)] 
	# inputs = [[None for i in range(3)] for i in range(2)]  
	# print inputs[0][0]
	global word_num_map,words
	word_num_map,words = LoadDicts()

	fn = batch_train_data(256)
	# inputs = []
	input_data = []
	targets = []
	input_length = []
	for i in range(2):
		input_data.append(tf.placeholder(tf.int32, [256, None]))
		targets.append(tf.placeholder(tf.int32, [256, None]))
		# input_length = tf.placeholder(tf.int32, [None], name='input_length') 
		input_length.append(tf.placeholder(tf.int32, [None]))
	print input_data
	# inputs = [None for i in range(3)]
	# inputs[0] = tf.placeholder(tf.int32, [256, None])  
	# inputs[1] = tf.placeholder(tf.int32, [256, None]) 
	# # input_length = tf.placeholder(tf.int32, [None], name='input_length') 
	# inputs[2] = tf.placeholder(tf.int32, [None])
	config_proto = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )
	with tf.Session(config=config_proto) as sess:
		for i in range(100000):
			input_data1,targets1,input_length1 = sess.run([input_data,targets,input_length],feed_dict=fill_dict(input_data,targets,input_length,fn))
			# print input_data1[0][-1]
			print '************************'
			train_to_word(input_data1[0][-1])
		# print inputr[1]
			# print inputs_l[2]
	# print inputs[i][0]