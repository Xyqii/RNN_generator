import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq
import pickle

def get_ids(id_list, token_size, batch_size, seq_length):
    n_bat = ((token_size-1)//(batch_size*seq_length))
    assert n_bat!=0, 'Decrease batch_size or num_steps!'
    in_id_l = np.array(id_list[:n_bat*batch_size*seq_length])
    target_id_l = np.array(id_list[1:n_bat*batch_size*seq_length+1])
    in_id = np.split(in_id_l.reshape(batch_size, -1), n_bat, 1)
    target_id = np.split(target_id_l.reshape(batch_size, -1), n_bat, 1)
    return np.array(zip(in_id, target_id)), n_bat

class Config(object):
    f = open('tokdict.pickle','r')
    token_dict = pickle.load(f)
    dict_size = len(token_dict)
    f.close()
    batch_size = 128
    seq_len = 64
    num_layer = 2
    hidden_size = 512
    max_epoch = 10
    keep_prob = 0.8
    lr = 0.003
    gen_length = 100

class Model(object):
    def __init__(self, is_training, 
                is_new, config, 
                x_input, y_input, in_state):    
        inputs = tf.one_hot(x_input, config.dict_size, dtype=tf.float32)
        cell1 = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size)
        cell2 = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size)
        if is_training:
            cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, 
                                    output_keep_prob=config.keep_prob)
            cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, 
                                    output_keep_prob=config.keep_prob)
        layers = [cell1, cell2]
        
        cell = tf.nn.rnn_cell.MultiRNNCell(layers)
        n_s = tf.unstack(in_state, axis=0)
        state_tuple = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(n_s[idx][0], n_s[idx][1])
                    for idx in range(config.num_layer)]
                )
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, 
                        initial_state=state_tuple)
        logits = tf.contrib.layers.fully_connected(
                outputs, config.dict_size, activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                biases_initializer=tf.zeros_initializer())
        self._probs = tf.nn.softmax(logits, name='probs')
        cost = seq2seq.sequence_loss(logits, y_input, 
                tf.ones([config.batch_size, config.seq_len]))
        self._final_state = tf.identity(state, name='final_state')
        self._cost = tf.identity(cost, name='cost')

        optimizer = tf.train.AdamOptimizer(config.lr)
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self._train_op = optimizer.apply_gradients(capped_gradients)

    @property
    def final_state(self):
        return self._final_state

    @property
    def cost(self):
        return self._cost

    @property
    def probs(self):
        return self._probs

    @property
    def train_op(self):
        return self._train_op

if __name__ == "__main__":
    config = Config()

    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.int32, shape=[None, None], 
                                                name='input')
        y = tf.placeholder(tf.int32, shape=[None, None],
                                                name='target')
        state = tf.placeholder(tf.float32, 
                    shape=[config.num_layer, 2, None, None],
                                                name='state')
        with tf.name_scope('Train'):
            with tf.variable_scope('Model', reuse=None):
                m_train = Model(True, False, config, x, y, state)
                tf.add_to_collection("train_op", m_train.train_op)
        with tf.name_scope('Generate'):
            with tf.variable_scope('Model', reuse=True):
                m_gen = Model(False, True, config, x, y, state)

    f = open('train.pickle', 'r')
    train_id_list = pickle.load(f)
    f.close()
    token_size = len(train_id_list)
    train_batches, tn_bat = get_ids(train_id_list, token_size, 
                            config.batch_size, config.seq_len) 

    f = open('tokdict.pickle', 'r')
    tokdict = pickle.load(f)
    f.close()

    f = open('iddict.pickle', 'r')
    iddict = pickle.load(f)
    f.close()

    zero_state = np.zeros((config.num_layer, 2, 
                        config.batch_size, config.hidden_size))
    gen_state = np.zeros((config.num_layer, 2, 
                        1, config.hidden_size))
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())   
        for i in range(config.max_epoch):
            t_state = zero_state
            for j, (input_x, target_y) in enumerate(train_batches):
                fetch = [m_train.cost, m_train.final_state, m_train.train_op]
                feed_dict = {x:input_x, y:target_y, state:t_state}
                train_loss, t_state, _ = sess.run(fetch, feed_dict) 
                cur_step = i*tn_bat+j
                if cur_step%200==0 or cur_step==config.max_epoch*tn_bat:
                    print 'Step: %d Train Loss: %.3f' % (cur_step, train_loss)
                if cur_step%600==0:
                    gen_ids = []
                    pri_tok = '\n'
                    gen_ids.append(tokdict[pri_tok])
                    n_state = gen_state
                    for tok in range(config.gen_length):
                        input_seq = [[gen_ids[-1]]]
                        id_tensor = tf.argmax(m_gen.probs, axis=2)
                        feed_dict = {x:input_seq, state:n_state}
                        fetch = [id_tensor, m_gen.final_state]
                        new_id, n_state = sess.run(fetch, feed_dict)
                        gen_ids.append(new_id[-1][-1])
                    new_mol = ''
                    for tok in gen_ids:
                        new_mol += iddict[tok]
                    print 'Step: %d New Molecule: %s' % (i*tn_bat+j, new_mol)
        saver = tf.train.Saver()
        saver.save(sess, './model/save')

