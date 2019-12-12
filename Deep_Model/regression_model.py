import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import cleanse_feature, cleanse_sample, fill_nan, replace_invalid, \
                  process_name, process_class, process_course, process_going, \
                  standardize, slice_regression_data, slice_classification_data, fc_layer, bilinear_layer


class RacingPredictor:
    """
    Base class for building a horse racing prediction model.
    """
    def __init__(self,
                 file='',
                 batch_size=512,
                 num_epochs=None,
                 iterations=3e5,
                 learning_rate=5e-4):
        """
        Initializer of <class 'RacingPredictor'>.

        :param file: Relative directory of data in csv format.
        """
        self.file = os.path.join('./', file)
        self.data = pd.read_csv(self.file)

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.iterations = int(iterations)
        self.learning_rate = learning_rate

        with tf.variable_scope(name_or_scope='init'):
            self.training = tf.placeholder(tf.bool, name='training')
            self.weight = tf.placeholder(tf.float32, name='weight')

            self._input = tf.placeholder(tf.float32, [None, 13], name='input')
            self._velocity = tf.placeholder(tf.float32, [None, 1], name='velocity')
            self._alpha = tf.placeholder(tf.float32, [None, 1], name='alpha')

    def __str__(self):
        return str(self.data.shape)

    def pre_process(self, persistent=False):
        """
        To pre-process the data for further operation(s).

        :param persistent: A boolean variable indicating whether to make the pre-processed data persistent locally.
        """
        # create a duplicate of data
        print('start pre-processing...')
        duplicate = self.data.copy()

        # define keys for detecting duplicates
        keys = ['rdate', 'rid', 'hid']
        # define indices of rows to be removed
        indices = []
        # cleanse invalid sample(s)
        print('cleansing invalid sample...')
        duplicate = cleanse_sample(duplicate, keys=keys, indices=indices)

        # define rules for dropping feature
        rules = [  # useless features
                   'horsenum', 'rfinishm', 'runpos', 'windist', 'win', 'place', '(rm|p|m|d)\d+',
                   # features containing too many NANs
                   'ratechg', 'horseweightchg', 'besttime', 'age', 'priority', 'lastsix', 'runpos', 'datediff',
                   # features which are difficult to process
                   'gear', 'pricemoney'
                 ]
        # eliminate useless features
        print('eliminating useless features...')
        duplicate = cleanse_feature(duplicate, rules=rules)

        # specify columns to be filled
        columns = ['bardraw', 'finishm', 'horseweight', 'win_t5', 'place_t5']
        # specify corresponding methods
        methods = [('constant', 4), ('constant', np.inf), 'mean', 'mean', 'mean']
        # fill nan value(s)
        print('filling nans...')
        duplicate = fill_nan(duplicate, columns=columns, methods=methods)

        # specify columns to be replaced
        columns = ['bardraw', 'finishm', 'horseweight']
        # specify schema(s) of replacement
        values = [(0, 14), (0, np.inf), (0, duplicate['horseweight'].mean())]
        # replace invalid value(s)
        print('replacing invalid values...')
        duplicate = replace_invalid(duplicate, columns=columns, values=values)

        # convert 'finishm' into 'velocity'
        print('generating velocity...')
        duplicate['velocity'] = 1e4 * duplicate['distance'] / duplicate['finishm']

        # apply target encoding on 'class'
        print('processing class...')
        duplicate = process_class(duplicate)
        # apply target encoding on 'jname' and 'tname'
        print('processing jname and tname...')
        duplicate = process_name(duplicate)
        # apply target encoding on 'venue' and 'course'
        print('processing venue and course...')
        duplicate = process_course(duplicate)
        # apply target encoding on 'track' and 'going'
        print('processing track and going...')
        duplicate = process_going(duplicate)

        # conduct local persistence
        if persistent:
            # set index before saving
            duplicate.set_index('index', inplace=True)
            print('saving result...')
            duplicate.to_csv(self.file.replace('.csv', '_modified.csv'))

        return duplicate

    def model(self):
        """
        To generate a model.

        :return: The estimation of race finish time of a single horse in centi second
        """
        with tf.variable_scope(name_or_scope='race_predictor'):
            fc_0 = fc_layer(tf.layers.flatten(self._input), 512, training=self.training, name='fc_0')

            bi_0 = bilinear_layer(fc_0, 512, training=self.training, name='bi_0')
            bi_1 = bilinear_layer(bi_0, 512, training=self.training, name='bi_1')

            velocity_output = tf.layers.dense(bi_1, units=1, activation=None, use_bias=False, name='velocity_output')
            alpha_output = tf.layers.dense(bi_1, units=1, activation=None, use_bias=False, name='alpha_output')

            return velocity_output, alpha_output

    def train(self):
        # pre-process data
        try:
            modify = pd.read_csv(self.file.replace('.csv', '_modified.csv'))
        except FileNotFoundError:
            modify = self.pre_process()

        # perform standardization
        modify = standardize(modify)

        # slice data
        x_train, y_train = slice_regression_data(modify)

        # generate model
        velocity, alpha = self.model()
        velocity_summary = tf.summary.histogram('velocity_summary', velocity)
        alpha_summary = tf.summary.histogram('alpha_summary', alpha)

        with tf.variable_scope(name_or_scope='optimizer'):
            # loss function
            velocity_loss = tf.reduce_mean(tf.square(velocity - self._velocity), name='velocity_loss')
            alpha_loss = tf.reduce_mean(tf.square(alpha - self._alpha), name='alpha_loss')

            total_loss = velocity_loss + self.weight * alpha_loss
            loss_summary = tf.summary.scalar('loss_summary', total_loss)

            # optimizer
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_ops = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        # configuration
        if not os.path.isdir('save'):
            os.mkdir('save')
        config = tf.ConfigProto()

        print('Start training')
        with tf.Session(config=config) as sess:
            # initialization
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # saver
            saver = tf.train.Saver(max_to_keep=5)

            # store the network graph for tensorboard visualization
            writer = tf.summary.FileWriter('save/network_graph', sess.graph)
            merge_op = tf.summary.merge([velocity_summary, alpha_summary, loss_summary])

            # data set
            queue = tf.train.slice_input_producer([x_train, y_train],
                                                  num_epochs=self.num_epochs, shuffle=True)
            x_batch, y_batch = tf.train.batch(queue, batch_size=self.batch_size, num_threads=1,
                                              allow_smaller_final_batch=False)

            # enable coordinator
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            try:
                for i in range(self.iterations):
                    x, y = sess.run([x_batch, y_batch])

                    output, loss = sess.run([velocity, velocity_loss],
                                            feed_dict={self.training: False, self._input: x, self._velocity: y})

                    _, loss, sm = sess.run([train_ops, total_loss, merge_op],
                                           feed_dict={self.training: True, self._input: x, self._velocity: y,
                                                      self.weight: i / self.iterations, self._alpha: output - y})

                    if i % 10 == 0:
                        print('iteration %d: loss = %f' % (i, loss))
                        writer.add_summary(sm, i)
                        writer.flush()
                    if i % 500 == 0 and i != 0:
                        print('save at iteration %d' % i)
                        saver.save(sess, 'save/%s/model' %
                                   (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
                saver.save(sess, 'save/%s/model' %
                           (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))
                writer.close()
            finally:
                coord.request_stop()

            coord.join(threads)

    def predict(self):
        # pre-process data
        try:
            modify = pd.read_csv(self.file.replace('.csv', '_modified.csv'))
        except FileNotFoundError:
            modify = self.pre_process()

        # perform standardization
        modify = standardize(modify)

        # slice data
        x_test, y_test = slice_regression_data(modify)

        # get graph
        graph = tf.get_default_graph()
        # session
        with tf.Session(graph=graph) as sess:
            # restore the latest model
            file_list = os.listdir('save/')
            file_list.sort(key=lambda val: val)
            loader = tf.train.import_meta_graph('save/%s/model.meta' % file_list[-2])

            # get input tensor
            training_tensor = graph.get_tensor_by_name('init/training_1:0')
            input_tensor = graph.get_tensor_by_name('init/input_1:0')
            velocity_tensor = graph.get_tensor_by_name('init/velocity_1:0')

            # get output tensor
            output_tensor = graph.get_tensor_by_name('race_predictor/velocity_output/MatMul:0')
            alpha_tensor = graph.get_tensor_by_name('race_predictor/alpha_output/MatMul:0')

            # get loss tensor
            loss_tensor = graph.get_tensor_by_name('optimizer/velocity_loss:0')

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader.restore(sess, tf.train.latest_checkpoint('save/%s' % file_list[-2]))

            velocity, alpha, loss = sess.run([output_tensor, alpha_tensor, loss_tensor],
                                             feed_dict={training_tensor: False,
                                                        input_tensor: x_test,
                                                        velocity_tensor: y_test})

            self.data = cleanse_sample(self.data, ['rdate', 'rid', 'hid'], [])
            self.data = self.data.reset_index(drop=True)

            self.data['p_velocity'] = 0
            self.data['p_rank'] = 0

            output = np.reshape(velocity - alpha, newshape=(-1, ))

            i = 0
            groups = self.data.groupby(['rdate', 'rid'])
            for name, group in groups:
                if name[0] < '2019':
                    i += len(group)
                    continue

                match = output[i:i+len(group)]
                rank = np.argsort(match)[::-1]
                rank = np.array([np.where(rank == k)[0][0] + 1 for k in range(len(match))])

                j = 0
                for index, row in group.iterrows():
                    row['p_velocity'] = match[j]
                    row['p_rank'] = rank[j]
                    self.data.iloc[index] = row
                    j += 1
                i += len(group)

                print(group.get(['rdate', 'rid', 'hid', 'finishm', 'rank', 'velocity']))
                print(match)
                print(rank)

            self.data.to_csv('test_result.csv')


def main():
    # read data from disk
    model = RacingPredictor('Sample_test_modified.csv', iterations=1.5e5)

    # train
    # model.train()

    # predict
    # model.predict()

    # print the shape of data
    # print(model)


if __name__ == '__main__':
    main()
