import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import cleanse_feature, cleanse_sample, fill_nan, replace_invalid, \
                  process_name, process_class, process_course, process_going, \
                  standardize, slice_regression_data, slice_classification_data, \
                  fc_layer, bilinear_layer, cross_entropy, rmse, WinP2PlaP


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

            self._input = tf.placeholder(tf.float32, [None, 182], name='input')
            self._win = tf.placeholder(tf.float32, [None, 14], name='win')

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
            duplicate.to_csv(self.file.replace('.csv', '_modified.csv'))

        return duplicate

    def model(self):
        """
        To generate a model.

        :return: The estimation of race finish time of a single horse in centi second
        """
        with tf.variable_scope(name_or_scope='race_predictor'):
            fc_0 = fc_layer(tf.layers.flatten(self._input), 256, training=self.training, name='fc_0')

            bi_0 = bilinear_layer(fc_0, 256, training=self.training, name='bi_0')
            bi_1 = bilinear_layer(bi_0, 256, training=self.training, name='bi_1')

            fc_1 = fc_layer(bi_1, 128, training=self.training, name='fc_1')

            win_output = tf.nn.softmax(
                tf.layers.dense(fc_1, units=14, activation=None), name='win_output')

            return win_output

    def train(self):
        # pre-process data
        try:
            modify = pd.read_csv(self.file.replace('.csv', '_modified.csv'))
            # shuffle among groups
            groups = [df.transform(np.random.permutation) for _, df in modify.groupby(['rdate', 'rid'])]
            modify = pd.concat(groups).reset_index(drop=True)
            # drop outdated data
            # modify = modify[:][[val > '2015' for val in modify['rdate']]]
        except FileNotFoundError:
            modify = self.pre_process()

        # perform standardization
        modify = standardize(modify)

        # slice data
        x_train, y_train = slice_classification_data(modify)

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

                    if i % 10 == 0:
                        print('iteration %d: loss = %f' % (i, loss))
                        writer.add_summary(sm, i)
                        writer.flush()
                    if i % 500 == 0 and i != 0:
                        output = sess.run(win, feed_dict={self.training: False, self._input: x, self._win: y})
                        print('ground truth: ', y[0:2])
                        print('prediction: ', output[0:2])

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
            modify = self.pre_process(persistent=True)

        # perform standardization
        modify = standardize(modify)

        # slice data
        x_test, y_test = slice_classification_data(modify)

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
            win_tensor = graph.get_tensor_by_name('init/win_1:0')

            # get output tensor
            output_tensor = graph.get_tensor_by_name('race_predictor/win_output:0')

            # get loss tensor
            loss_tensor = graph.get_tensor_by_name('optimizer/total_loss:0')

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader.restore(sess, tf.train.latest_checkpoint('save/%s' % file_list[-2]))

            win, loss = sess.run([output_tensor, loss_tensor],
                                 feed_dict={training_tensor: False,
                                            input_tensor: x_test,
                                            win_tensor: y_test})

            self.data = cleanse_sample(self.data, ['rdate', 'rid', 'hid'], [])
            self.data = self.data.reset_index(drop=True)

            self.data['winprob'] = 0

            i = 0
            groups = self.data.groupby(['rdate', 'rid'])
            for name, group in groups:
                total = np.sum(win[i, 0:len(group)])

                j = 0
                for index, row in group.iterrows():
                    row['winprob'] = win[i, j] / total
                    self.data.iloc[index] = row
                    j += 1
                i += 1

            self.data['plaprob'] = WinP2PlaP(self.data, wpcol='winprob')

            fixratio = 1 / 10000
            mthresh = 9
            print("Getting win stake...")
            self.data['winstake'] = fixratio * (self.data['winprob'] * self.data['win_t5'] > mthresh)
            print("Getting place stake...")
            self.data['plastake'] = fixratio * (self.data['plaprob'] * self.data['place_t5'] > mthresh)

            self.data.to_csv('test_result.csv')


def main():
    # read data from disk
    model = RacingPredictor('../Data/HR200709to201901.csv', batch_size=32, num_epochs=None,
                            iterations=5e4, learning_rate=3e-4)
    # model = RacingPredictor('Sample_test.csv')

    # train
    model.train()

    # predict
    # model.predict()

    # print the shape of data
    # print(model)


if __name__ == '__main__':
    main()
