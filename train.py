"""
Training script.
"""

import os
import tensorflow as tf
from hyperparams import Hyperparams as Hp
from loader import Loader
from model import Model
from tqdm import tqdm


def train():
    # load & pre-process (if necessary) data
    data = Loader(Hp.batch_size, split=(0.8, 0.1, 0.1))

    # build tensorflow graph
    quacc = Model()
    saver = tf.train.Saver()
    best = 0

    # initialize graph
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter(os.path.join(Hp.log_dir, 'train'), sess.graph)
        # test_writer = tf.summary.FileWriter(os.path.join(Hp.log_dir, 'val'))
        sess.run(init)

        for ep in tqdm(range(Hp.n_epochs)):
            tr_loss, tr_acc = 0, 0

            for i in tqdm(range(data.batches_tr['n_batches'])):
                # get next batch
                p_batch, q_batch, p_lengths, q_length, ptr_batch = data.next_training_batch()
                # build a feed dict
                train_dict = {quacc.p_word_inputs: p_batch,
                              quacc.q_word_inputs: q_batch,
                              quacc.p_word_lengths: p_lengths,
                              quacc.q_word_lengths: q_length,
                              quacc.labels: ptr_batch}
                loss, acc, _ = sess.run([quacc.loss, quacc.exact_match, quacc.train_step], feed_dict=train_dict)
                tr_loss += loss
                tr_acc += acc

                if i % 100 == 0:
                    pass
                    # tensorboard
                    # summary = sess.run(merged, feed_dict=train_dict)
                    # train_writer.add_summary(summary, i)

            tr_loss /= data.batches_tr['n_batches']
            tr_acc /= data.batches_tr['n_batches']
            print('Training loss: {}, accuracy={:.2f}%'.format(tr_loss, 100 * tr_acc))

            if ep % 10 == 0:
                print('Validating...')
                va_acc = 0
                for _ in tqdm(range(data.batches_va['n_batches'])):
                    # get next batch
                    p_batch, q_batch, p_lengths, q_length, ptr_batch = data.next_validation_batch()
                    # build a feed dict
                    val_dict = {quacc.p_word_inputs: p_batch,
                                quacc.q_word_inputs: q_batch,
                                quacc.p_word_lengths: p_lengths,
                                quacc.q_word_lengths: q_length,
                                quacc.labels: ptr_batch}
                    acc, = sess.run(quacc.exact_match, feed_dict=val_dict)
                    va_acc += acc

                va_acc /= data.batches_va['n_batches']
                print('Validation accuracy={:.2f}%'.format(va_acc))
                if va_acc > best:
                    print('Validation accuracy increased. Saving model...')
                    best = va_acc
                    saver.save(sess, './models/quacc')
                else:
                    print('Validation accuracy decreased. Restoring model...')
                    saver.restore(sess, './models/quacc')


if __name__ == '__main__':
    train()
