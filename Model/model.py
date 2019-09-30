import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


# import tensorflow as tf

from utils import cleanse_feature, cleanse_sample, fill_nan, replace_invalid, process_name, slice_data


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

    def pre_process(self, persistent=False):
        """
        To pre-process the data for further operation(s).

        :param persistent: A boolean variable indicating whether to make the pre-processed data persistent locally.
        """
        # define keys for detecting duplicates
        keys = ['rdate', 'rid', 'hid']
        # define indices of rows to be removed
        indices = [('rank', 0), ('finishm', 0)]
        # cleanse invalid sample(s)
        self.data = cleanse_sample(self.data, keys=keys, indices=indices)

        # define rules for dropping feature
        rules = [  # useless features
                   'index', 'horsenum', 'rfinishm', 'runpos', 'windist', 'win', 'place', '(rm|p|m|d)\d+',
                   # features containing too many NANs
                   'ratechg', 'horseweightchg', 'besttime', 'age', 'priority', 'lastsix', 'runpos', 'datediff',
                   # features which are difficult to process
                   'gear', 'class', 'pricemoney'
                 ]
        # eliminate useless features
        self.data = cleanse_feature(self.data, rules=rules)

        # specify columns to be filled
        columns = ['track', 'going', 'course', 'bardraw', 'finishm', 'horseweight', 'rating', 'win_t5', 'place_t5']
        # specify corresponding methods
        methods = ['ffill', 'ffill', ('constant', self.data['track'].fillna(method='ffill')), ('constant', 4),
                   'ffill', 'mean', 'mean', 'mean', 'mean']
        # fill nan value(s)
        self.data = fill_nan(self.data, columns=columns, methods=methods)

        # specify columns to be replaced
        columns = ['bardraw', 'horseweight']
        # specify schema(s) of replacement
        values = [(0, 4), (0, self.data['horseweight'].mean())]
        # replace invalid value(s)
        self.data = replace_invalid(self.data, columns=columns, values=values)

        # apply one-hot encoding on features
        self.data = pd.get_dummies(self.data, columns=['venue', 'track', 'going', 'course'])

        # apply target encoding on features
        self.data = process_name(self.data)

        # perform min-max standardization
        for key in self.data.keys():
            if key not in ['rdate', 'rid', 'hid', 'finishm', 'rank', 'ind_win', 'ind_pla']:
                self.data[key] = (self.data[key] - self.data[key].min()) / (self.data[key].max() - self.data[key].min())

        # conduct local persistence
        if persistent:
            self.data.to_csv(self.file.replace('.csv', '_modified.csv'))

    def tf_model(self):
        with tf.variable_scope(name_or_scope='race_predictor'):
            self.data = self.data
            pass

    def train(self, x_train, y_train):

        # convert training data into LightGBM dataset format
        d_train = lgb.Dataset(x_train, label=y_train)

        params = {}
        params['learning_rate'] = 0.003
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'multiclass'
        params['metric'] = 'multi_logloss'
        params['sub_feature'] = 0.5
        params['num_leaves'] = 10
        params['min_data'] = 50
        params['max_depth'] = 10
        params['num_class'] = 14

        self.lgb_model = lgb.train(params, d_train, 100)

    def predict(self, x_test):
        # prediction
        clf = self.lgb_model
        y_pred = clf.predict(x_test)
        return y_pred


def main():
    pass


if __name__ == '__main__':
    # read data from disk
    model = RacingPredictor('../Data/HR200709to201901.csv', debug=True)

    # pre-process data
    model.pre_process(persistent=False)

    # divide the data set into training set and testing set
    x, y = slice_data(model.data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    print(y_train.shape)
    print(y_train[0])
    model.train(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = (y_pred.argmax(1) == y_test).mean()
    print(accuracy)

    # print(x.shape, y.shape)
    # print the shape of data
    # print(model)
