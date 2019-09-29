import numpy as np
import pandas as pd
import tensorflow as tf

# Pandas's display settings
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm(x, axis=-1, training=True):
    return tf.layers.batch_normalization(
        inputs=x, axis=axis,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)


def fc_layer(x, units, training=True, dropout=True, name=''):
    with tf.variable_scope(name_or_scope=name):
        inputs = tf.layers.dense(x, units=units, activation=None, use_bias=False)
        inputs = tf.nn.relu(batch_norm(inputs, training=training))
        if dropout:
            return tf.layers.dropout(inputs, rate=0.25, training=training, name='output')
        else:
            return inputs


def slice_data(data):
    matches = data.groupby(['rdate', 'rid'])

    features = ['distance', 'jname', 'tname', 'exweight', 'bardraw', 'rating', 'horseweight', 'win_t5', 'place_t5',
                'venue_HV', 'venue_ST', 'track_ALL WEATHER TRACK', 'track_TURF', 'going_FAST', 'going_GOOD',
                'going_GOOD TO FIRM', 'going_GOOD TO YIELDING', 'going_SLOW', 'going_SOFT', 'going_WET FAST',
                'going_WET SLOW', 'going_YIELDING', 'going_YIELDING TO SOFT', 'course_A', 'course_A+3',
                'course_ALL WEATHER TRACK', 'course_B', 'course_B+2', 'course_C', 'course_C+3', 'course_TURF']
    ground_truth = ['rank']

    num_match = len(matches)
    num_horse = 14

    x = np.zeros(shape=[num_match, num_horse, len(features)])
    y = np.zeros(shape=[num_match, num_horse, 1])

    index = 0
    for (_, match) in matches:
        x_feature = match.get(features)
        y_feature = match.get(ground_truth)

        for row in range(len(x_feature)):
            x[index][row] = x_feature.iloc[row, :]
            y[index][row] = y_feature.iloc[row, :]

    return x, y


def count_frequency(data, key):
    """
    Calculate frequency of each non-nan unique value.

    :param data: Original data in format of Pandas DataFrame.
    :param key: A key representing a specific column of data.
    :return: A tuple containing frequency of each non-nan unique value and proportion of NANs over all data.
    """
    # group data by a given key
    group = data.groupby(key)[key]

    # count the frequency of each non-nan unique value
    frequency = group.count()
    # calculate the proportion of NANs over all data
    proportion = len(data[key][[type(val) == float and np.isnan(val) for val in data[key]]]) / len(data[key])

    # return
    return frequency, proportion


def cleanse_feature(data, rules):
    """
    To cleanse feature following given rules.

    :param data: Original data in format of Pandas DataFrame.
    :param rules: A Python list containing rules of cleansing.
    :return: Copy of data after cleansing.
    """
    # convert given rules into Python Regular Expression
    def rule2expression(rule):
        # return '^$' if no rule provided
        if len(rule) == 0:
            return '^$'
        # compute index of the rear element
        rear = len(rule) - 1
        # concat each rule
        expression = '^('
        for i in range(rear):
            expression += rule[i] + '|'
        expression += rule[rear] + ')$'
        # return a regular expression
        return expression

    # eliminate useless features
    return data.drop(data.filter(regex=rule2expression(rules)), axis=1)


def cleanse_sample(data, keys, indices):
    """
    To cleanse invalid observation(s).

    :param data: Original data in format of Pandas DataFrame.
    :param keys: Columns for identifying duplicates.
    :param indices: Identifier(s) of invalid observation(s).
    :return:
    """
    # create a duplicate of data
    duplicate = data.copy()

    # drop duplicates
    duplicate = duplicate.drop_duplicates(subset=keys, keep='first')

    # determine observations to be dropped
    for index in indices:
        # unpack the identifier
        column, value = index
        # drop invalid observations
        duplicate = duplicate.drop(duplicate[duplicate[column] == value].index)

    # return
    return duplicate


def fill_nan(data, columns, methods):
    """
    To fill values using the specified method.

    :param data: Original data in format of Pandas DataFrame.
    :param columns: A Python list of indices of columns.
    :param methods: Specified methods used in filling.
    :return: Copy of data after filling.
    """
    # create a duplicate of data
    duplicate = data.copy()

    # apply filling method to every given column
    for index in range(len(columns)):
        # retrieve a specific column
        col = duplicate[columns[index]]
        # unpack the corresponding method
        method, value = methods[index] if type(methods[index]) == tuple else (methods[index], 0)
        # fill nan with the given method
        if method == 'constant':
            col.fillna(value=value, inplace=True)
        elif method == 'mean':
            col.fillna(value=col.mean(), inplace=True)
        else:
            col.fillna(method=method, inplace=True)

    # return
    return duplicate


def replace_invalid(data, columns, values):
    """
    To replace invalid values following the specified schemata.

    :param data: Original data in format of Pandas DataFrame.
    :param columns: A Python list of indices of columns.
    :param values: Specified schemata used in replacement.
    :return: Copy of data after replacement.
    """
    # create a duplicate of data
    duplicate = data.copy()

    # apply filling method to every given column
    for index in range(len(columns)):
        # retrieve a specific column
        col = duplicate[columns[index]]
        # unpack the corresponding method
        before, after = values[index]
        # replace 'before' with 'after'
        col.replace(before, after, inplace=True)

    # return
    return duplicate


def process_lastsix(data):
    """
    To encode feature 'lastsix'.

    :param data: Original data in format of Pandas DataFrame.
    :return: Copy of data after encoding.
    """
    # create a duplicate of data
    duplicate = data.copy()

    # convert feature 'lastsix' into a number
    def lastsix2num(lastsix):
        if type(lastsix) != str:
            return np.nan
        else:
            accumulation, count = 0, 0
            for rank in lastsix.split('/'):
                if rank != '-':
                    accumulation, count = accumulation + int(rank), count + 1
            return np.nan if count == 0 else accumulation / count

    # encode feature 'lastsix'
    duplicate['lastsix'] = [lastsix2num(val) for val in duplicate['lastsix']]

    # replace NaN with algorithm average
    target = np.average(duplicate['lastsix'][np.isfinite(duplicate['lastsix'])])
    duplicate['lastsix'] = [target if np.isnan(val) else val for val in duplicate['lastsix']]

    # return
    return duplicate


def process_name(data):
    """
    To perform target encoding on feature 'jname' and 'tname' respectively.

    :param data: Original data in format of Pandas DataFrame.
    :return: Copy of data after encoding.
    """
    # create a duplicate of data
    duplicate = data.copy()

    # group data by 'jname'
    group = duplicate.groupby('jname')['rank']
    # calculate average rank for every unique jockey
    for (name, ranks) in group:
        # replace jockey name with average rank
        duplicate['jname'].replace(name, ranks.mean(), inplace=True)
    # process NANs
    duplicate['jname'].fillna(value=duplicate['jname'].mean(), inplace=True)

    # group data by 'tname'
    group = duplicate.groupby('tname')['rank']
    # calculate average rank for every unique trainer
    for (name, ranks) in group:
        # replace trainer name with average rank
        duplicate['tname'].replace(name, ranks.mean(), inplace=True)
    # process NANs
    duplicate['tname'].fillna(value=duplicate['tname'].mean(), inplace=True)

    # return
    return duplicate
