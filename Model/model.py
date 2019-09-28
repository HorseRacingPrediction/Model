import os
from utils import *

import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier


class RacingPredictor:
    """
    Base class for building a horse racing prediction model.
    """
    def __init__(self, file=''):
        """
        Initializer of <class 'RacingPredictor'>.

        :param file: Relative directory of data in csv format.
        """
        self.file = os.path.join('./', file)
        self.data = pd.read_csv(self.file)

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

    def train(self):
        # not implemented yet
        pass

    def predict(self):
        # not implemented yet
        pass


def main():
    # read data from disk
    model = RacingPredictor('../Data/HR200709to201901.csv')
    # pre-process data
    model.pre_process(persistent=False)
    # print the shape of data
    print(model)


if __name__ == '__main__':
    main()
