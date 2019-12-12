import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

# Pandas's display settings
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


# get probability of 2nd place given winner probability
def Place2nd(wp):
    '''
    Input:
        wp: an array of winning probabilities
    Output:
        p2s: a list of probabilities of 2nd place
    '''
    p2s = []
    for k, w in enumerate(wp):
        p2 = 0
        # due to Luce model, choose the 1st from the rest
        for ww in np.delete(wp, k):
            p2 += ww * w / (1 - ww)
        p2s.append(p2)
    return p2s


# get probability of 3rd place given winner probability
def Place3rd(wp):
    '''
    Input:
        wp: an array of winning probabilities
    Output:
        p3s: a list of probabilities of 3rd place
    '''
    p3s = []
    for k, w in enumerate(wp):
        p3 = 0
        wpx = np.delete(wp, k)
        # choose the 1st
        for i, x in enumerate(wpx):
            # then choose the 2nd
            for y in np.delete(wpx, i):
                p3 += x * y * w / ((1 - x) * (1 - x - y))
        p3s.append(p3)
    return p3s


# get probability of place (top3) given winner probability
def WinP2PlaP(datawp, wpcol):
    '''
    Input:
        datawp: the dataframe with column of winning probability
        wpcol: colunm name of the winning probability of datawp
    Output:
        top3: an array of probabilities of top 3
    '''
    p2nds = []
    p3rds = []
    for (rd, rid), group in datawp.groupby(['rdate', 'rid']):
        wp = group[wpcol].values
        p2nds += Place2nd(wp)
        p3rds += Place3rd(wp)

    top3 = datawp[wpcol].values + np.array(p2nds) + np.array(p3rds)
    return top3


def rmse(label, prop):
    return tf.sqrt(tf.reduce_mean(tf.square(label - prop + 1e-10), axis=-1))


def cross_entropy(label, prop):
    return -(label * tf.log(tf.clip_by_value(prop, 1e-10, 1.)) +
             (1 - label) * tf.log(tf.clip_by_value(1. - prop, 1e-10, 1.)))


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


def bilinear_layer(x, units, training=True, name=''):
    with tf.variable_scope(name_or_scope=name):
        shortcut = x
        inputs = fc_layer(x, units, training=training, name='fc_0')
        inputs = fc_layer(inputs, units, training=training, name='fc_1')
        return tf.add_n([inputs, shortcut], name='output')


def slice_regression_data(data):
    features = ['class', 'distance', 'jname', 'tname', 'exweight', 'bardraw', 'rating', 'horseweight', 'win_t5',
                'place_t5', 'straight', 'width', 'humidity']

    ground_truth = ['velocity']

    x = np.array(data.get(features))
    y = np.array(data.get(ground_truth))

    return x, y


def slice_classification_data(data):
    # data['rank'].replace(0, np.nan, inplace=True)

    matches = data.groupby(['rdate', 'rid'])

    features = ['class', 'distance', 'jname', 'tname', 'exweight', 'bardraw', 'rating', 'horseweight', 'win_t5',
                'place_t5', 'straight', 'width', 'humidity']
    ground_truth = ['rank']

    num_match = len(matches)
    num_horse = 14

    x = np.full(shape=(num_match, num_horse, len(features)), fill_value=np.nan)
    y = np.full(shape=(num_match, num_horse, 1), fill_value=np.nan)

    index = 0
    for (_, match) in matches:
        x_feature = match.get(features)
        y_feature = match.get(ground_truth)

        for row in range(len(x_feature)):
            x[index][row] = x_feature.iloc[row, :]
            y[index][row] = y_feature.iloc[row, :]
        index += 1

    x = np.reshape(x, (num_match, num_horse * len(features)))
    y = np.reshape(y, (num_match, num_horse))

    return x, np.nanargmin(y, axis=1)


def slice_naive_data(data, one_hot=True):
    features = ['class', 'distance', 'jname', 'tname', 'exweight', 'bardraw', 'rating', 'horseweight', 'win_t5',
                'place_t5', 'straight', 'width', 'humidity']
    ground_truth = ['rank']

    x = data.get(features).to_numpy()
    y = data.get(ground_truth).to_numpy()
    y = np.reshape(y, (-1, ))

    if one_hot:
        # perform one-hot encoding
        ind = np.zeros(shape=(y.shape[0], 16))
        ind[np.arange(y.shape[0]), y] = 1
        y = ind

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

    # reset index
    duplicate.reset_index(drop=True, inplace=True)

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
    jname = {'J Fortune': 1, 'N Pinna': 106, 'C Murray': 216, 'D Tudhope': 1, 'M Nunes': 435, 'F Durso': 11,
             'A de Vries': 1, 'G Gomez': 11, 'S K Sit': 546, 'C Williams': 154, 'H T Mo': 777, 'P Holmes': 1,
             'N Berry': 6, 'M F Poon': 759, 'M Smith': 6, 'C Demuro': 4, 'S Khumalo': 4, 'J Moreira': 3124,
             'A Atzeni': 80, 'C Velasquez': 4, 'E Wilson': 80, 'R Curatolo': 4, 'W L Ho': 17, 'J Lloyd': 1426,
             'G Cheyne': 892, 'T Durcan': 3, 'F Prat': 4, 'J Crowley': 2, 'S Dye': 153, 'O Murphy': 147,
             'E da Silva': 29, 'H Shii': 1, 'D Lane': 178, 'A Munro': 2, 'R Ffrench': 1, 'F Geroux': 4, 'A Hamelin': 1,
             'L Corrales': 15, 'T Mundry': 1, 'C Schofield': 1421, 'R Maia': 7, 'L Dettori': 17, 'R Woodworth': 1,
             'G van Niekerk': 166, 'D Bonilla': 2, 'F Coetzee': 658, 'J Talamo': 4, 'M L Yeung': 4154, 'L Salles': 8,
             'C Y Ho': 3742, "C O'Donoghue": 77, 'C Sutherland': 3, 'S Hamanaka': 4, 'K C Leung': 4809, 'A Badel': 427,
             'H W Lai': 4378, 'R Hughes': 16, 'T Queally': 78, 'Y Fujioka': 1, 'M Du Plessis': 1464, 'Y T Cheng': 3601,
             'M Barzalona': 57, 'K Tosaki': 12, 'M Rodd': 7, 'B Stanley': 3, 'Y Take': 6, 'C Y Lui': 760,
             'Y Fukunaga': 15, 'C Newitt': 2, 'N Rawiller': 1375, 'D McDonogh': 1, 'P H Lo': 361, 'M Kinane': 8,
             'H Bowman': 245, 'K Fallon': 13, 'D Whyte': 6167, 'O Bosson': 163, 'L Henrique': 7, 'K T Yeung': 583,
             'M Chadwick': 4169, 'N Juglall': 26, 'F Branca': 1, 'M Wepner': 2, 'A Crastus': 3, 'T Angland': 1536,
             'W Smith': 5, 'A Helfenbein': 1, 'B Vorster': 2, 'R Dominguez': 1, 'N Callow': 1, 'J Cassidy': 1,
             'K Manning': 2, 'R Fradd': 16, 'K M Chin': 32, 'C Soumillon': 287, 'G Schofield': 13, 'O Doleuze': 4272,
             'O Peslier': 38, 'I Ortiz Jr': 4, 'M Demuro': 310, 'R Migliore': 3, 'A Gryder': 98, 'M Cangas': 1,
             'N Hall': 1, 'R Thomas': 2, 'D Porcu': 1, 'C K Tong': 2873, 'K F Choi': 1, 'V Espinoza': 4, 'H Uchida': 5,
             'T Berry': 849, 'W Pike': 321, 'K W Leung': 269, 'T Jarnet': 3, 'W Lordan': 5, 'K K Chiong': 704,
             'D Beadman': 1667, 'F Blondel': 2, 'J Victoire': 269, 'A Marcus': 5, 'C W Wong': 1547, 'G Lerena': 158,
             'C Brown': 19, 'K C Ng': 2152, 'T Ono': 1, 'M Zahra': 89, 'C Wong': 324, 'A Starke': 17,
             'W C Marwing': 1904, 'A Delpech': 161, 'R Fourie': 709, 'K Teetan': 3089, 'H Goto': 1, 'Y Iwata': 19,
             'D Chong': 1, 'J Castellano': 4, 'Colin Keane': 4, 'A Sanna': 442, 'G Mosse': 2841, 'C Lemaire': 38,
             'L Nolen': 5, 'D Oliver': 47, 'K McEvoy': 9, 'S Arnold': 1, 'J Spencer': 9, 'W M Lai': 3396, 'F Berry': 2,
             'J Riddell': 1, 'H N Wong': 1393, 'Z Purton': 6021, 'C W Choi': 1, 'C Reith': 195, 'J Saimee': 1,
             'B Prebble': 5064, 'P Smullen': 16, 'G Boss': 438, 'K Shea': 91, 'D Browne': 1, 'T Clark': 913,
             'P Robinson': 1, 'P Strydom': 136, 'J Doyle': 2, 'P-C Boudot': 46, 'T H So': 3006, 'S Pasquier': 4,
             'K Tanaka': 2, 'G Benoist': 89, 'D Shibata': 2, 'V Cheminaud': 94, 'B Shinn': 5, 'M Ebina': 3,
             'M Wheeler': 1, 'S Fujita': 2, 'D Dunn': 22, 'K H Yu': 17, 'J Winks': 314, 'M W Leung': 1012, 'K Ando': 2,
             'C F Wong': 1, 'K Ikezoe': 4, 'G Gibbons': 1, 'T Thulliez': 3, 'A Suborics': 1522, 'J Leparoux': 4,
             'K Latham': 1, 'B Herd': 1, 'M Guyon': 405, 'W Buick': 13, 'C Perkins': 2, 'J McDonald': 67,
             'B Doyle': 1108, 'J Murtagh': 21, 'G Stevens': 4, 'T Hellier': 1, 'H Tanaka': 1, 'S de Sousa': 465,
             'R Silvera': 4, 'R Moore': 206, 'M Hills': 3, 'H Lam': 4, 'N Callan': 2941, 'D Nikolic': 3,
             "B Fayd'herbe": 9, 'L Duarte': 3, 'S Clipperton': 908, 'K L Chui': 843, 'U Rispoli': 1611, 'A Spiteri': 3,
             'E Saint-Martin': 513, 'I Mendizabal': 1, 'S Drowne': 3}

    # calculate appearance frequency for every unique jockey
    for name in duplicate['jname'].unique():
        if type(name) != str:
            continue
        elif jname.get(name) is None:
            # replace jockey name with 0
            duplicate['jname'].replace(name, 0, inplace=True)
        else:
            # replace jockey name with the logarithm of its appearance frequency
            duplicate['jname'].replace(name, np.log(jname[name]), inplace=True)
    # process NANs
    duplicate['jname'].fillna(value=0, inplace=True)

    # group data by 'tname'
    tname = {'K H Leong': 1, 'M J Freedman': 328, 'B Koriner': 1, 'M R Channon': 1, 'K W Lui': 4288, 'P Schiergen': 5,
             'F C Lor': 560, 'T W Leung': 1916, 'H Uehara': 1, 'A Fabre': 5, 'K Kato': 2, 'S Tarry': 1, 'Pat Lee': 1,
             'J E Hammond': 1, 'A T Millard': 4176, 'X T Demeaulte': 1, 'P C Kan': 3, 'K C Wong': 4, 'D Simcock': 1,
             'H Blume': 1, 'J S Moore': 1, 'R J Frankel': 1, 'Y Kato': 1, 'F Head': 4, 'J Lau': 4, 'A M Balding': 4,
             "D O'Meara": 1, 'I Sameshima': 2, 'M Delzangles': 4, 'S Kojima': 1, 'A Schutz': 2725, 'M L W Bell': 1,
             'P Chapple-Hyam': 1, 'N Meade': 1, 'A S Cruz': 6248, 'Y C Fung': 2, 'M Hawkes': 1, 'N Hori': 4,
             'J Stephens': 1, 'J Sadler': 1, 'K Sumii': 2, 'J M Moore': 2, 'L K Laxon': 2, 'C Marshall': 2,
             'S bin Suroor': 8, 'D Smaga': 1, 'B W Hills': 1, 'J Hawkes': 1, 'D J Hall': 4013, 'Y O Wong': 954,
             'P F Yiu': 4950, 'Y Yahagi': 4, 'K A Ryan': 1, "J O'Hara": 2, 'R Rohne': 1, 'C W Chang': 3940,
             'K Ikezoe': 1, 'C Hills': 2, 'R M H Cowell': 1, 'de Royer Dupre': 13, 'H-A Pantall': 1, 'N Takagi': 1,
             'R Heathcote': 1, 'T K Ng': 2940, 'D Cruz': 3459, 'M Ito': 1, 'L M Cumani': 6, 'K Prendergast': 1,
             'A Ngai': 1, 'J Moore': 5466, 'M Kon': 2, 'M Botti': 3, 'J-M Beguigne': 2, 'Rod Collet': 2,
             'B J Meehan': 1, 'J S Bolger': 2, "A P O'Brien": 4, 'A W Noonan': 1, 'R Collet': 2, 'M A Jarvis': 3,
             "J O'Shea": 2, "P O'Sullivan": 3875, 'R Bastiman': 1, 'Barande-Barbe': 5, 'A Lee': 4352, 'J Size': 5212,
             'M Figge': 2, 'J C Englehart': 1, 'A Wohler': 1, 'P Van de Poele': 2, 'W Hickst': 1, 'E Lellouche': 4,
             'S Chow': 4, 'H Fujiwara': 5, 'K Yamauchi': 1, 'S H Cheong': 1, 'F Doumen': 2, 'P Demercastel': 1,
             'P B Shaw': 2, 'J Bary': 1, 'W Figge': 1, 'D Morton': 1, 'T A Hogan': 5, 'E Botti': 1, 'B K Ng': 1546,
             'G A Nicholson': 1, 'J-P Gauvin': 1, 'R M Beckett': 1, 'N Sugai': 1, 'E Libaud': 2, 'J H M Gosden': 1,
             'D L Romans': 1, 'G M Begg': 1, 'Y Ikee': 2, 'C Clement': 1, 'J W Sadler': 1, 'T Yasuda': 4, 'G Eurell': 2,
             'W Y So': 2490, 'J Noseda': 1, 'D A Hayes': 3, 'B D A Cecil': 1, 'P Bary': 2, 'J R Fanshawe': 2,
             'S Woods': 2790, 'S L Leung': 1, 'B J Wallace': 1, 'S Kunieda': 3, 'D E Ferraris': 4696, 'M Johnston': 1,
             'M N Houdalakis': 1, "D O'Brien": 1, 'H J Brown': 2, 'M C Tam': 8, 'T Mundry': 1, 'R Varian': 1,
             'E M Lynam': 5, 'K H Ting': 119, 'M J Wallace': 1, 'C Fownes': 6219, 'L Ho': 4259, 'C H Yip': 6277,
             'P Rau': 2, 'T P Yung': 2056, 'M F de Kock': 17, 'K C Chong': 3, 'R Hannon Snr': 4, 'J-C Rouget': 1,
             'O Hirata': 1, 'R Charlton': 4, 'C S Shum': 4525, 'W H Tse': 1, 'E A L Dunlop': 5, 'R Gibson': 2668,
             'Y S Tsui': 6150, 'D L Freedman': 1, 'Sir M R Stoute': 8, 'P G Moody': 1, 'Laffon-Parias': 5,
             'G Allendorf': 7, 'J G Given': 1, 'W A Ward': 1, 'W P Mullins': 1, 'C Dollase': 1, 'N D Drysdale': 1,
             'K L Man': 4875, 'P Leyshan': 3, 'B Smart': 1, 'W J Haggas': 2, 'H Shimizu': 1, 'J M P Eustace': 1,
             'G W Moore': 11, 'M Nishizono': 1, 'R E Dutrow Jr': 1}

    # calculate appearance frequency for every unique trainer
    for name in duplicate['tname'].unique():
        if type(name) != str:
            continue
        elif tname.get(name) is None:
            # replace trainer name with 0
            duplicate['tname'].replace(name, 0, inplace=True)
        else:
            # replace trainer name with average rank
            duplicate['tname'].replace(name, np.log(tname[name]), inplace=True)
    # process NANs
    duplicate['tname'].fillna(value=0, inplace=True)

    # return
    return duplicate


def process_class(data):
    # define dictionary for mapping a 'class' to a numerical value
    class2rating = {'Class 5': 31.560311284046694, 'Class 4': 51.20929941722318, 'Class 3': 69.93396499768197,
                    'Class 2': 87.77822148631537, 'Class 1': 97.3286734086853, 'GROUP-3': 107.50331125827815,
                    '3YO+': 115.25714285714285, '4YO': 95.4054054054054, 'GRIFFIN': 57.391304347826086,
                    '23YO': 58.333333333333336, 'Hong Kong Group Three': 107.56483126110125, 'R': 60.94117647058823,
                    'Hong Kong Group Two': 114.21024258760107, 'G1': 119.22099447513813, '4S': 51.122950819672134,
                    'Hong Kong Group One': 107.70664739884393, '4B': 51.96153846153846, '2B': 87.91666666666667,
                    '3B': 71.52941176470588, '3S': 69.50413223140495, 'G2': 112.90196078431373,
                    '4R': 52.610169491525426, 'Group One': 117.72463768115942, 'Group Two': 113.51470588235294}

    # create a duplicate of data
    duplicate = data.copy()

    # traverse every class
    for key in class2rating.keys():
        # replace class name with a value
        duplicate['class'].replace(key, class2rating[key], inplace=True)

    # filter out row(s) having rating value of nan
    mask = [np.isnan(val) for val in duplicate['rating']]

    # fill nan value for 'rating'
    duplicate['rating'] = np.where(mask, duplicate['class'], duplicate['rating'])

    # return
    return duplicate


def process_course(data):
    # create a duplicate of data
    duplicate = data.copy()

    # define dictionary for mapping 'venue' and 'course' to 'straight' and 'width'
    rail = {'HV_A': (312, 30.5), 'HV_B': (338, 26.5), 'HV_C': (334, 22.5), 'HV_C+3': (335, 19.5),
            'ST_A': (430, 30.5), 'ST_A+3': (430, 28.5), 'ST_ALL WEATHER TRACK': (365, 22.8),
            'ST_B': (430, 26), 'ST_B+2': (430, 24), 'ST_C': (430, 21.3), 'ST_C+3': (430, 18.3)}

    # create new columns
    duplicate['straight'] = 0
    duplicate['width'] = 0

    # group data by 'venue' and 'course'
    for key in rail.keys():
        venue, course = key.split('_')
        # filter out corresponding row(s)
        mask = [row['venue'] == venue and row['course'] == course for index, row in duplicate.iterrows()]
        # get the corresponding straight and width
        straight, width = rail[key]
        # replace values for 'straight'
        duplicate.loc[mask, 'straight'] = straight
        # replace values for 'width'
        duplicate.loc[mask, 'width'] = width

    # drop 'venue' and 'course'
    duplicate.drop(columns=['venue', 'course'], inplace=True)

    # return
    return duplicate


def process_going(data):
    # create a duplicate of data
    duplicate = data.copy()

    # define dictionary for mapping 'venue' and 'course' to 'straight' and 'width'
    humidity = {'TURF_FIRM': 2.5, 'TURF_GOOD': 3, 'TURF_GOOD TO FIRM': 2.75,
                'TURF_GOOD TO YIELDING': 3.25, 'TURF_SOFT': 4, 'TURF_YIELDING': 3.5,
                'TURF_YIELDING TO SOFT': 3.75, 'ALL WEATHER TRACK_FAST': 2.5,
                'ALL WEATHER TRACK_GOOD': 2.75, 'ALL WEATHER TRACK_SLOW': 3,
                'ALL WEATHER TRACK_WET FAST': 3.25, 'ALL WEATHER TRACK_WET SLOW': 3.5}

    # create new columns
    duplicate['humidity'] = 0

    # group data by 'track' and 'going'
    for key in humidity.keys():
        track, going = key.split('_')
        # filter out corresponding row(s)
        mask = [row['track'] == track and row['going'] == going for index, row in duplicate.iterrows()]
        # replace values for 'straight'
        duplicate.loc[mask, 'humidity'] = humidity[key]

    # drop 'venue' and 'course'
    duplicate.drop(columns=['track', 'going'], inplace=True)

    # return
    return duplicate


def standardize(data, method='minmax'):
    # create a duplicate of data
    duplicate = data.copy()

    if method == 'minmax':
        # define references for standardization
        minmax = {'class': (31.560311284046694, 119.22099447513813), 'distance': (1000, 2400),
                  'jname': (0.0, 8.726967774991492), 'tname': (0.0, 8.744647438317532), 'exweight': (103, 133),
                  'bardraw': (1.0, 15.0), 'rating': (3.0, 134.0), 'horseweight': (693.0, 1369.0),
                  'win_t5': (1.05, 99.0), 'place_t5': (0.0, 69.35), 'straight': (312, 430), 'width': (18.3, 30.5),
                  'humidity': (2.5, 4.0)}

        # perform min-max standardization
        print('performing standardization...')
        for key in minmax.keys():
            duplicate[key] = np.clip((duplicate[key] - minmax[key][0]) / (minmax[key][1] - minmax[key][0]), 0.0, 1.0)

    elif method == 'zscore':
        # define references for standardization
        zscore = {'class': (61.08817482505135, 19.261043641533654),
                  'distance': (1418.3108064530988, 275.74050984376544),
                  'jname': (7.695839524459074, 1.050175960791586), 'tname': (8.30521355341357, 0.6505238871785868),
                  'exweight': (122.61638888121101, 6.326115323601956), 'bardraw': (6.858205036070649, 3.741593086895),
                  'rating': (61.08817482510817, 20.36383765927511),
                  'horseweight': (1106.6339043351936, 62.674589253614705),
                  'win_t5': (26.10166136845214, 26.507945247752932),
                  'place_t5': (6.148789565181681, 5.6954779814777945),
                  'straight': (387.1050701604061, 47.78145923294249), 'width': (24.99404903396278, 4.130792541957933),
                  'humidity': (2.9349185070528945, 0.18444070251568562)}

        # perform z-score standardization
        print('performing standardization...')
        for key in zscore.keys():
            duplicate[key] = (duplicate[key] - zscore[key][0]) / zscore[key][1]

    return duplicate


def normalize(data, df, key):
    rdate = data['rdate']
    rid = data['rid']

    accum = df.loc[(df['rdate'] == rdate) & (df['rid'] == rid), key].sum()

    out = data[key] / accum

    return out



"""
from collections import defaultdict


def j_winrate(df):

    df.rdate = pd.to_datetime(df.rdate)
    jname_date_winrate = defaultdict(lambda: {})
    jnames = df.jname.unique()
    for name in jnames:
        one_jackey = df[df.jname == name]
        if len(one_jackey) < 10:
            # too few entry
            continue

        win_sum = 0
        win_arr = []
        date_arr = []
        one_jackey.sort_values(by='rdate')
        for index, row in one_jackey.iterrows():
            while len(date_arr) and (row.rdate - date_arr[0]).days > 365:
                win_sum -= win_arr[0]
                date_arr = date_arr[1:]
                win_arr = win_arr[1:]

            if len(win_arr) >= 5:
                if row.rdate != date_arr[-1]:
                    jname_date_winrate[name][row.rdate] = win_sum / len(win_arr)

            win_sum += row.ind_pla
            win_arr.append(row.ind_pla)
            date_arr.append(row.rdate)

    df_winrate = pd.DataFrame(dict(jname_date_winrate))
    ser_winrate = df_winrate.stack()
    ser_winrate.name = 'past_win_rate'
    _df = df.join(ser_winrate, on=['rdate', 'jname'])
    
    return _df


def horse_weichg(df):

    horseweight_chg = np.zeros(df.shape[0])
    horsenums = df.horsenum.unique()
    for name in horsenums:
        one_horse = df[df.horsenum == name].copy()
        if len(one_horse) < 2:
            continue

        one_horse.sort_values(by='rdate', inplace=True)
        idxs = one_horse.index
        for i in range(1, len(one_horse)):
            val = one_horse['horseweight'].iloc[i] - one_horse['horseweight'].iloc[i - 1]
            horseweight_chg[idxs[i]] = val
            
    df['horseweight_chg'] = horseweight_chg
"""