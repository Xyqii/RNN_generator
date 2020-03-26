import tensorflow as tf
import numpy as np
import pickle
from model import get_ids
from model import Config

config = Config()
f = open('train_tl.pickle', 'r')
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

pri_tok = '\n'
cur_id = tokdict[pri_tok]
sta_t_loss = []
graph = tf.Graph()

with tf.Session(graph=graph) as sess:
    loader = tf.train.import_meta_graph('./model/save.meta')
    loader.restore(sess, './model/save')

    new_x = graph.get_tensor_by_name('input:0')
    new_y = graph.get_tensor_by_name('target:0')
    new_state = graph.get_tensor_by_name('state:0')
    final_train_state = graph.get_tensor_by_name('Train/Model/final_state:0')
    train_prob = graph.get_tensor_by_name('Train/Model/probs:0')
    cost = graph.get_tensor_by_name('Train/Model/cost:0')
    train_op = tf.get_collection('train_op')[0]
    final_gen_state = graph.get_tensor_by_name('Generate/Model/final_state:0')
    gen_prob = graph.get_tensor_by_name('Generate/Model/probs:0')
    zero_state = np.zeros((config.num_layer, 2, config.batch_size, config.hidden_size))
    gen_state = np.zeros((config.num_layer, 2, 1, config.hidden_size))
    for i in range(config.max_epoch):
        t_state = zero_state
        for j, (input_x, target_y) in enumerate(train_batches):
            fetch = [cost, final_train_state, train_op]
            feed_dict = {new_x:input_x, new_y:target_y, new_state:t_state}
            train_loss, t_state, _ = sess.run(fetch, feed_dict) 
            cur_step = i*tn_bat+j
            if cur_step%6==0 or cur_step==config.max_epoch*tn_bat:
                print 'Step: %d Train Loss: %.3f' % (cur_step, train_loss)
            if cur_step%50==0:
                gen_ids = []
                pri_tok = '\n'
                gen_ids.append(tokdict[pri_tok])
                n_state=gen_state
                for tok in range(config.gen_length):
                    input_seq = [[gen_ids[-1]]]
                    id_tensor = tf.argmax(gen_prob, axis=2)
                    feed_dict = {new_x:input_seq, new_state:n_state}
                    fetch = [id_tensor, final_gen_state]
                    new_id, n_state = sess.run(fetch, feed_dict)
                    gen_ids.append(new_id[-1][-1])
                new_mol = ''
                for tok in gen_ids:
                    new_mol += iddict[tok]
                print 'Step: %d New Molecule: %s' % (i*tn_bat+j, new_mol)
    saver = tf.train.Saver()
    saver.save(sess, './model_tl/save')
