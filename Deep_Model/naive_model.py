import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import cleanse_feature, cleanse_sample, fill_nan, replace_invalid, \
                  process_name, process_class, process_course, process_going, \
                  standardize, slice_naive_data, fc_layer, bilinear_layer, cross_entropy, rmse, normalize
from backtesting import backtest


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

            self._input = tf.placeholder(tf.float32, [None, 13], name='input')
            self._win = tf.placeholder(tf.float32, [None, 16], name='win')

    def __str__(self):
        return str(self.data.shape)

    @staticmethod
    def pre_process(file, persistent=False):
        """
        To pre-process the data for further operation(s).

        :param file: Path to a csv file.
        :param persistent: A boolean variable indicating whether to make the pre-processed data persistent locally.
        """
        # create a duplicate of data
        print('start pre-processing...')
        duplicate = pd.read_csv(file)

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
        columns = ['bardraw', 'finishm', 'exweight', 'horseweight', 'win_t5', 'place_t5']
        # specify corresponding methods
        methods = [('constant', 4), ('constant', 1e5), ('constant', 122.61638888121101),
                   ('constant', 1106.368874062333), ('constant', 26.101661368452852), ('constant', 6.14878956518161)]
        # fill nan value(s)
        print('filling nans...')
        duplicate = fill_nan(duplicate, columns=columns, methods=methods)

        # specify columns to be replaced
        columns = ['bardraw', 'finishm', 'exweight', 'horseweight']
        # specify schema(s) of replacement
        values = [(0, 14), (0, 1e5), (0, 122.61638888121101), (0, 1106.368874062333)]
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
            duplicate.to_csv(file.replace('.csv', '_modified.csv'))

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

            win_output = tf.nn.softmax(logits=tf.layers.dense(bi_1, units=16, activation=None, use_bias=False),
                                       name='win_output')

            return win_output

    def train(self):
        # pre-process data
        try:
            modify = pd.read_csv(self.file.replace('.csv', '_modified.csv'))
        except FileNotFoundError:
            modify = RacingPredictor.pre_process(self.file, persistent=True)

        # drop outdated data
        # modify = modify[:][[val > '2015' for val in modify['rdate']]]
        # perform standardization
        modify = standardize(modify)

        # slice data
        x_train, y_train = slice_naive_data(modify)

        # define validation set
        validation = None
        x_test, y_test = None, None

        # generate model
        win = self.model()
        win_summary = tf.summary.histogram('win_summary', win)

        with tf.variable_scope(name_or_scope='optimizer'):
            # loss function
            # total_loss = tf.reduce_mean(tf.reduce_sum(cross_entropy(self._win, win), axis=-1), name='total_loss')
            total_loss = tf.reduce_mean(rmse(self._win, win), name='total_loss')
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
            optimal = np.inf
            saver = tf.train.Saver(max_to_keep=5)

            # store the network graph for tensorboard visualization
            writer = tf.summary.FileWriter('save/network_graph', sess.graph)
            merge_op = tf.summary.merge([win_summary, loss_summary])

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

                    _, loss, sm = sess.run([train_ops, total_loss, merge_op],
                                           feed_dict={self.training: True, self._input: x, self._win: y})

                    if i % 1000 == 0 and i != 0:
                        for j in range(1, len(validation)):
                            _, loss = sess.run([train_ops, total_loss],
                                               feed_dict={self.training: True,
                                                          self._input: x_test[j], self._win: y_test[j]})
                    if i % 100 == 0:
                        print('iteration %d: loss = %f' % (i, loss))
                        writer.add_summary(sm, i)
                        writer.flush()
                    if i % 500 == 0:
                        if validation is None:
                            # read validation set
                            validation = [pd.read_csv('new_data/test_new.csv'),
                                          pd.read_csv('new_data/HR201911W2.csv'),
                                          pd.read_csv('new_data/HR201911W1.csv'),
                                          pd.read_csv('new_data/HR201910W4.csv'),
                                          pd.read_csv('new_data/HR201910W3.csv'),
                                          pd.read_csv('new_data/HR201910W2.csv')]
                            for j in range(len(validation)):
                                validation[j] = cleanse_sample(validation[j], keys=['rdate', 'rid', 'hid'], indices=[])
                            # slice testing data
                            path = ['new_data/test_new_modified.csv', 'new_data/HR201911W2_modified.csv',
                                    'new_data/HR201911W1_modified.csv', 'new_data/HR201910W4_modified.csv',
                                    'new_data/HR201910W3_modified.csv', 'new_data/HR201910W2_modified.csv']
                            x_test, y_test = [np.array([])] * len(validation), [np.array([])] * len(validation)
                            for j in range(len(path)):
                                x_test[j], y_test[j] = slice_naive_data(
                                    standardize(pd.read_csv(path[j])))

                        rmse_win, rmse_place = 0, 0
                        for j in range(len(validation)):
                            prob, loss = sess.run([win, total_loss],
                                                  feed_dict={self.training: False,
                                                             self._input: x_test[j], self._win: y_test[j]})

                            validation[j]['winprob'] = prob[:, 1]
                            validation[j]['2ndprob'] = prob[:, 2]
                            validation[j]['3rdprob'] = prob[:, 3]

                            validation[j]['winprob'] = validation[j].apply(normalize, axis=1, df=validation[j],
                                                                           key='winprob')
                            validation[j]['2ndprob'] = validation[j].apply(normalize, axis=1, df=validation[j],
                                                                           key='2ndprob')
                            validation[j]['3rdprob'] = validation[j].apply(normalize, axis=1, df=validation[j],
                                                                           key='3rdprob')
                            validation[j]['plaprob'] = validation[j]['winprob'] + validation[j]['2ndprob'] + \
                                                       validation[j]['3rdprob']

                            fixratio = 5e-4
                            mthresh = 1.75
                            print("Getting win stake...")
                            validation[j]['winstake'] = fixratio * \
                                                        (validation[j]['winprob'] * validation[j]['win_t5'] > mthresh)
                            print("Getting place stake...")
                            validation[j]['plastake'] = fixratio * \
                                                        (validation[j]['plaprob'] * validation[j]['place_t5'] > mthresh)

                            result = backtest(validation[j], 'winprob', 'plaprob', 'winstake', 'plastake')

                            rmse_win += result['AverageRMSEwin'] * (1 if j == 0 else 0.5)
                            rmse_place += result['AverageRMSEpalce'] * (1 if j == 0 else 0.5)

                        if 0.4 * rmse_win + 0.6 * rmse_place < optimal:
                            optimal = 0.4 * rmse_win + 0.6 * rmse_place
                            print('save at iteration %d with average loss of %f' %
                                  (i, 2 * optimal / (len(validation) + 1)))
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

    @staticmethod
    def predict(file):
        data = pd.read_csv(file)
        data = cleanse_sample(data, keys=['rdate', 'rid', 'hid'], indices=[])

        # pre-process data
        try:
            modify = pd.read_csv(file.replace('.csv', '_modified.csv'))
        except FileNotFoundError:
            modify = RacingPredictor.pre_process(file, persistent=True)

        # perform standardization
        modify = standardize(modify)

        # slice data
        x_test, y_test = slice_naive_data(modify)

        # get graph
        graph = tf.get_default_graph()
        # session
        with tf.Session(graph=graph) as sess:
            # restore the latest model
            file_list = os.listdir('save/')
            file_list.sort(key=lambda val: val)
            loader = tf.train.import_meta_graph('save/%s/model.meta' % file_list[-2])

            # get input tensor
            training_tensor = graph.get_tensor_by_name('init/training:0')
            input_tensor = graph.get_tensor_by_name('init/input:0')
            win_tensor = graph.get_tensor_by_name('init/win:0')

            # get output tensor
            output_tensor = graph.get_tensor_by_name('race_predictor/win_output:0')

            # get loss tensor
            loss_tensor = graph.get_tensor_by_name('optimizer/total_loss:0')

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader.restore(sess, tf.train.latest_checkpoint('save/%s' % file_list[-2]))

            prob, loss = sess.run([output_tensor, loss_tensor],
                                  feed_dict={training_tensor: False, input_tensor: x_test, win_tensor: y_test})

            data['winprob'] = prob[:, 1]
            data['2ndprob'] = prob[:, 2]
            data['3rdprob'] = prob[:, 3]

            data['winprob'] = data.apply(normalize, axis=1, df=data, key='winprob')
            data['2ndprob'] = data.apply(normalize, axis=1, df=data, key='2ndprob')
            data['3rdprob'] = data.apply(normalize, axis=1, df=data, key='3rdprob')
            data['plaprob'] = data['winprob'] + data['2ndprob'] + data['3rdprob']

            fixratio = 5e-4
            mthresh = 1.75

            print("Getting win stake...")
            data['winstake'] = fixratio * (data['winprob'] * data['win_t5'] > mthresh)
            print("Getting place stake...")
            data['plastake'] = fixratio * (data['plaprob'] * data['place_t5'] > mthresh)

            result = backtest(data, 'winprob', 'plaprob', 'winstake', 'plastake')

            return result


def main():
    # read data from disk
    # model = RacingPredictor('../Data/HR200709to201901.csv', iterations=1.5e5, learning_rate=1e-3, batch_size=256)

    # train
    # model.train()

    # predict
    RacingPredictor.predict('new_data/test_new.csv')
    RacingPredictor.predict('new_data/HR201911W2.csv')
    RacingPredictor.predict('new_data/HR201911W1.csv')
    RacingPredictor.predict('new_data/HR201910W4.csv')
    RacingPredictor.predict('new_data/HR201910W3.csv')
    RacingPredictor.predict('new_data/HR201910W2.csv')


if __name__ == '__main__':
    main()
