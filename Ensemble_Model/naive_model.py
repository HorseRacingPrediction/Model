import os
import numpy as np
import pandas as pd

import lightgbm as lgb

from utils import cleanse_feature, cleanse_sample, fill_nan, replace_invalid, \
                  process_name, process_going, process_course, process_class, \
                  slice_naive_data, standardize, WinP2PlaP
from backtesting import backtest


class RacingPredictor:
    """
    Base class for building a horse racing prediction model.
    """
    def __init__(self, file='', debug=False):
        """
        Initializer of <class 'RacingPredictor'>.

        :param file: Relative directory of data in csv format.
        """
        self.file = os.path.join('./', file)
        self.data = pd.read_csv(self.file)
        self.debug = debug
        self.lgb_model = None

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

    def train(self):
        # pre-process data
        try:
            modify = pd.read_csv(self.file.replace('.csv', '_modified.csv'))
        except FileNotFoundError:
            modify = RacingPredictor.pre_process(self.file, persistent=True)

        # drop outdated data
        modify = modify[:][[val > '2010' for val in modify['rdate']]]
        # perform standardization
        modify = standardize(modify)

        # slice data
        x_train, y_train = slice_naive_data(modify)

        # convert training data into LightGBM dataset format
        d_train = lgb.Dataset(x_train, label=y_train)

        params = dict()
        params['learning_rate'] = 3e-4
        params['boosting_type'] = 'rf'
        params['objective'] = 'multiclass'
        params['metric'] = 'multi_logloss'
        params['num_class'] = 16

        params['bagging_freq'] = 1
        params['bagging_fraction'] = 0.8
        # params['lambda_l1'] = 10
        # params['lambda_l2'] = 1
        # params['max_depth'] = 10
        # params['cat_smooth'] = 10
        # params['feature_fraction'] = 0.8
        # params['num_leaves'] = 128
        # params['min_data_in_leaf'] = 32

        self.lgb_model = lgb.train(params, d_train, 400)

        self.lgb_model.save_model('lgb_classifier.txt', num_iteration=self.lgb_model.best_iteration)

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

        # prediction
        clf = lgb.Booster(model_file='lgb_classifier.txt')

        winprob = clf.predict(x_test)

        data['winprob'] = winprob[:, 1]
        data['plaprob'] = winprob[:, 1] + winprob[:, 2] + winprob[:, 3]

        fixratio = 5e-3
        mthresh = 1.6
        print("Getting win stake...")
        data['winstake'] = fixratio * (data['winprob'] * data['win_t5'] > mthresh)
        print("Getting place stake...")
        data['plastake'] = fixratio * (data['plaprob'] * data['place_t5'] > mthresh)

        data.to_csv('test_result.csv')

        return data


if __name__ == '__main__':
    # read data from disk
    # model = RacingPredictor('../Data/HR200709to201901.csv', debug=True)

    # model.train()

    backtest(RacingPredictor.predict('HR201911W1.csv'), 'winprob', 'plaprob', 'winstake', 'plastake')
