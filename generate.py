import tensorflow as tf
import numpy as np
import pickle
from model import Config

config = Config()

f = open('tokdict.pickle', 'r')
tokdict = pickle.load(f)
f.close()

f = open('iddict.pickle', 'r')
iddict = pickle.load(f)
f.close()

pri_tok = '\n'
cur_id = tokdict[pri_tok]

length = 200000
graph = tf.Graph()

gen_state = np.zeros((config.num_layer, 2, 1, config.hidden_size))

with tf.Session(graph=graph) as sess:
    loader = tf.train.import_meta_graph('./model/save.meta')
    loader.restore(sess, './model/save')

    input_x = graph.get_tensor_by_name('input:0')
    new_state = graph.get_tensor_by_name('state:0')
    final_state = graph.get_tensor_by_name('Generate/Model/final_state:0')
    prob = graph.get_tensor_by_name('Generate/Model/probs:0')
    n_state = gen_state
    f = open('newmols', 'w')
    for i in range(length):
        input_tok = [[cur_id]]
        feed_dict = {input_x:input_tok, new_state:n_state}
        p, n_state = sess.run([prob, final_state], feed_dict)
        probability = np.array(p)[0,0,:]
        cur_id = np.random.choice(len(iddict), p=probability)
        f.write(iddict[cur_id])
    f.close()
