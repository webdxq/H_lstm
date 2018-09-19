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
import time
from datetime import datetime
# os.environ['CUDA_VISIBLE_DEVICES']='1'

LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_DECAY = 0.99
TRAINING_STEPS = 50000

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

def to_word(weights):
    sample = np.argmax(weights)
    return words[sample] 

def to_word_second(weights):
    sort_weights = np.argsort(weights)
    return words[sort_weights[1]] 
    
def train_to_word(x):
    # print(u'x的长度',len(x))
    to_words = lambda num: words[num]
    x_words = map(to_words, x)
    # print(str(x_words).decode("unicode-escape"))
    outstr = ''.join(x_words)
    print outstr

def AlignSentence(sentence):
    sentence = sentence[:-2]
    sentence_re = ''
    # print words[1]
    for i in range(len(sentence)):
        if not (sentence[i] >= u'\u4e00' and sentence[i]<=u'\u9fa5'):
            sentence_re += sentence[i]+u' '
        else:
            sentence_re += sentence[i]
    # return u'[ '+sentence[i] + u' ]'
    print u'['+ sentence_re + u']'

def train_neural_network(batch_size, epoch, word_size):  
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    with tf.Graph().as_default():
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
        # print masks
        masks = tf.reshape(masks, [-1])
        logits, last_state, probs, _, _ = model.neural_network(input_data, word_size, batch_size)  
        # logits, last_state, probs, _, _, test_output = model.neural_network(input_data, word_size, batch_size)  
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
        new_learning_rate = tf.assign(learning_rate, 0.002 * (0.97 ** (epoch-20))) 
        with tf.Session(config=config_proto) as sess:
            train_writer = tf.summary.FileWriter('logdir', sess.graph)
            sess.run(init_op)  
            saver = tf.train.Saver(tf.global_variables())
            # last_epoch = load_model(sess, saver,'/media/pingan_ai/dxq/gen_blessing/models/') 
            step = 0
            sess.graph.finalize() 
            # for epoch in range(last_epoch + 1, 200):
            for step in xrange(TRAINING_STEPS):
                sess.run(new_learning_rate)  
                #sess.run(tf.assign(learning_rate, 0.01))  
                all_loss = 0.0 
                for x,y,z in data_utils.batch_train_data(batch_size):
                    # print x
                    summary,train_loss, _ , _ ,probs_= sess.run(
                        [merged, cost, last_state, train_op, probs], 
                        feed_dict={input_data: x, targets: y, targets_length:z})  
                    # print test.shape
                    step += 1
                    all_loss = all_loss + train_loss 
                    train_writer.add_summary(summary,step)
                    if step % 200 == 1:
                        print(epoch, step, train_loss) 
                        probs_= probs_.reshape(batch_size, -1, len(words))
                        y_words = [ list(map(to_word_second, prob_)) for prob_ in probs_] 
                        train_to_word(x[-1])
                        AlignSentence(''.join(y_words[-1]))
                        print('********************')
                        train_to_word(x[-2])
                        AlignSentence(''.join(y_words[-2]))
                            
                # saver.save(sess, '/media/pingan_ai/dxq/gen_blessing/models/', global_step=epoch) 
                # print (epoch,' Loss: ', all_loss * 1.0 / n_chunk) 
                # print(type(probs_))
               

def fill_dict(input_data,targets,input_length,fn):
    feed_dict = {}
    for i in range(2):
        x,y,z = fn.next()
        assert x.shape == y.shape
        feed_dict[input_data[i]] = x
        feed_dict[targets[i]] = y
        feed_dict[input_length[i]] = z       
    return feed_dict,x,y

def get_loss(input_data, targets, targets_length, word_size, batch_size, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        targets_reshape = tf.reshape(targets, [-1])      
        maxlen = tf.reduce_max(targets_length)
        masks = tf.sequence_mask(
                lengths=targets_length,
                maxlen=maxlen,
                name='masks',
                dtype=tf.float32
            )
        # print masks
        masks = tf.reshape(masks, [-1])
        logits, last_state, probs, _, _ = model.neural_network(input_data, word_size, batch_size)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits], 
            [targets_reshape], 
            [masks], 
            )
        cost = tf.reduce_mean(loss)  
    return cost,probs

# 计算每一个变量梯度的平均值。
def average_gradients(tower_grads):
    average_grads = []

    # 枚举所有的变量和变量在不同GPU上计算得出的梯度。
    for grad_and_vars in zip(*tower_grads):
        # 计算所有GPU上的梯度平均值。
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        # 将变量和它的平均梯度对应起来。
        average_grads.append(grad_and_var)
    # 返回所有变量的平均梯度，这个将被用于变量的更新。
    return average_grads

def Mul_Gpu_train(batch_size, epoch, word_size, num_gpus, checkpoint_path):
    # 将简单的运算放在CPU上，只有神经网络的训练过程放在GPU上。
    # TRAINING_STEPS = EPOCHS*n_chunk/N_GPU
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # inputs = [[None for i in range(3)] for i in range(num_gpus)]  
        input_data = []
        targets = []
        input_length = []
        for i in range(num_gpus):
            input_data.append(tf.placeholder(tf.int32, [batch_size, None]))
            targets.append(tf.placeholder(tf.int32, [batch_size, None]))
            # input_length = tf.placeholder(tf.int32, [None], name='input_length') 
            input_length.append(tf.placeholder(tf.int32, [None]))      
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE, global_step,97289 / batch_size, LEARNING_RATE_DECAY)
        optimizer = tf.train.AdamOptimizer(learning_rate) 
        
        tower_grads = []
        reuse_variables = False
        # 将神经网络的优化过程跑在不同的GPU上。
        
        for i in range(num_gpus):
            # 将优化过程指定在一个GPU上。
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    
                    cur_loss,probs= get_loss(
                        input_data[i],
                        targets[i],
                        input_length[i],
                        word_size,
                        batch_size,
                        reuse_variables)
                    # 在第一次声明变量之后，将控制变量重用的参数设置为True。这样可以
                    # 让不同的GPU更新同一组参数。
                    reuse_variables = True
                    grads = optimizer.compute_gradients(cur_loss)
                    tower_grads.append(grads)
        
        # 计算变量的平均梯度。
        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram('gradients_on_average/%s' % var.op.name, grad)

        # 使用平均梯度更新参数。
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # 计算变量的滑动平均值。
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_to_average = (tf.trainable_variables() +tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)
        # 每一轮迭代需要更新变量的取值并更新变量的滑动平均值。
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()        
        init = tf.global_variables_initializer()
        batch_data = data_utils.batch_train_data(batch_size)
        config_proto = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )
        with tf.Session(config=config_proto) as sess:
            # 初始化所有变量并启动队列。
            init.run()
            summary_writer = tf.summary.FileWriter(checkpoint_path, sess.graph)

            for step in xrange(TRAINING_STEPS):
                start_time = time.time()
                feed_dict, x, y= fill_dict(input_data,targets,input_length,batch_data)
                train_loss, _, summary,probs_ = sess.run(
                    [cur_loss, train_op,summary_op,probs], 
                    feed_dict=feed_dict
                    )  

                duration = time.time() - start_time
                # print (datetime.now(), step, train_loss)
                if step != 0 and step % 100 == 0:
                    num_examples_per_step = batch_size * num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / num_gpus
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print (format_str % (datetime.now(), step, train_loss, examples_per_sec, sec_per_batch))
                    # summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, step)
                    probs_= probs_.reshape(batch_size, -1, len(words))
                    y_words = [ list(map(to_word_second, prob_)) for prob_ in probs_] 
                    train_to_word(x[-1])
                    print (''.join(y_words[-1]))
                    # print('********************')
                    # train_to_word(x[-2])
                    # AlignSentence(''.join(y_words[-2]))
    
                # 每隔一段时间保存当前的模型。
                # if step % 1000 == 0 or (step + 1) == TRAINING_STEPS:
                #     saver.save(sess, checkpoint_path, global_step=step)
def run():
    data_config = configuration.DataConfig()
    train_config = configuration.TrainingConfig()
    batch_size = train_config.batch_size
    epoch = train_config.epoch
    num_gpus = train_config.num_gpus
    checkpoint_path = train_config.checkpoint_path
    word_size = configuration.DataConfig().word_size
    filename = data_config.lyrics_file.split('.')[0]
    global word_num_map,words
    # if os.path.exists(filename+'.npy'):       
    word_num_map,words = data_utils.LoadDicts()
    train_neural_network(batch_size, epoch, word_size)  
    # Mul_Gpu_train(batch_size, epoch, word_size, num_gpus, checkpoint_path)

if __name__ == '__main__':
    run()