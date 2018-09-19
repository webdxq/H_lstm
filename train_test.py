#!/usr/bin/python3
#-*- coding: UTF-8 -*-
import collections  
import numpy as np  
import tensorflow as tf  
import os
import sys
import chardet
import re
import model
import configuration
import data_utils
os.environ['CUDA_VISIBLE_DEVICES']='1'

def load_model(sess, saver,ckpt_path):
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
        print ('resume from', latest_ckpt)
        saver.restore(sess, latest_ckpt)
        return int(latest_ckpt[latest_ckpt.rindex('-') + 1:])
    else:
        print ('building model from scratch')
        sess.run(tf.global_variables_initializer())
        return -1
def neural_network(input_data, word_size, batch_size, model='lstm', rnn_size=128, num_layers=2):  
    if model == 'rnn':  
        cell_fun = tf.nn.rnn_cell.BasicRNNCell  
    elif model == 'gru':  
        cell_fun = tf.nn.rnn_cell.GRUCell  
    elif model == 'lstm':  
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell  
   
    cell = cell_fun(rnn_size, state_is_tuple=True)  
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)  
   
    initial_state = cell.zero_state(batch_size, tf.float32)  
    with tf.variable_scope('rnnlm',reuse=tf.AUTO_REUSE):  
        softmax_w = tf.get_variable("softmax_w", [rnn_size, word_size])  
        softmax_b = tf.get_variable("softmax_b", [word_size])  
        with tf.device("/cpu:0"):  
            embedding = tf.get_variable("embedding", [word_size, rnn_size])  
            inputs = tf.nn.embedding_lookup(embedding, input_data)  

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')  
    output = tf.reshape(outputs,[-1, rnn_size])  
   
    logits = tf.matmul(output, softmax_w) + softmax_b  
    probs = tf.nn.softmax(logits)  
    return logits, last_state, probs, cell, initial_state 
def train_neural_network():  
    with tf.Graph().as_default():
        train_config = configuration.TrainingConfig()
        batch_size = train_config.batch_size
        epoch = train_config.epoch
        word_size = configuration.DataConfig().word_size
        input_data = tf.placeholder(tf.int32, [batch_size, None], name='input_pl')  
        targets = tf.placeholder(tf.int32, [batch_size, None], name='target_pl') 
        input_length = tf.placeholder(tf.int32, [None], name='input_length') 
        targets_length = tf.placeholder(tf.int32, [None], name='target_length') 
        print targets_length
        maxlen = tf.reduce_max(targets_length)
        masks = tf.sequence_mask(
                lengths=targets_length,
                maxlen=maxlen,
                name='masks',
                dtype=tf.float32
            )
        print masks
        masks = tf.reshape(masks, [-1])
        logits, last_state, probs, _, _ = neural_network(input_data, word_size, batch_size)  
        print logits
        targets_reshape = tf.reshape(targets, [-1])  
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        	[logits], 
        	[targets_reshape], 
        	[masks], 
            )  
        
        # tf.summary.scalar('loss', loss)
        cost = tf.reduce_mean(loss)  
        tf.summary.scalar('cost', cost)
        learning_rate = tf.Variable(0.0, trainable=False)  
        tf.summary.scalar('learning_rate', learning_rate)
        tvars = tf.trainable_variables()  
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)    
        optimizer = tf.train.AdamOptimizer(learning_rate)   
        train_op = optimizer.apply_gradients(zip(grads, tvars))  
        config_proto = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                gpu_options=tf.GPUOptions(allow_growth=True)
            )
        merged = tf.summary.merge_all()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session(config=config_proto) as sess:
            train_writer = tf.summary.FileWriter('../' + 'logdir', sess.graph)
            sess.run(init_op)  
            saver = tf.train.Saver(tf.global_variables())
            # last_epoch = load_model(sess, saver,'/media/pingan_ai/dxq/gen_blessing/models/') 
            step = 0
            # sess.graph.finalize() 
            # for epoch in range(last_epoch + 1, 200):
            for epoch in range(0, 200):
                # if epoch > 50:
                print epoch
                # sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** (epoch-20))) )  
                #sess.run(tf.assign(learning_rate, 0.01))  
                all_loss = 0.0 
                for x,y,z in data_utils.batch_train_data(batch_size):
                    print x.shape
                    summary,train_loss, _ , _ ,probs_= sess.run(
                        [merged, cost, last_state, train_op, probs], 
                        feed_dict={input_data: x, targets: y, targets_length:z})  
                    step += 1
                    all_loss = all_loss + train_loss 
                    train_writer.add_summary(summary,step)
                    if step % 200 == 1:
                        print(epoch, step, 0.002 * (0.97 ** (epoch-20)),train_loss) 
                saver.save(sess, '/media/pingan_ai/dxq/gen_blessing/models/', global_step=epoch) 
                print (epoch,' Loss: ', all_loss * 1.0 / n_chunk) 
                # print(type(probs_))
                probs_= probs_.reshape(batch_size, -1, len(words))
                y_words = [ list(map(to_word, prob_)) for prob_ in probs_] 
                train_to_word(x[-1])
                AlignSentence(''.join(y_words[-1]))
                print('********************')
                train_to_word(x[-2])
                AlignSentence(''.join(y_words[-2]))
                del probs_
                del y_words
                del probs_


train_neural_network()  