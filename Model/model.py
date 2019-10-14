import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


# import tensorflow as tf

from utils import cleanse_feature, cleanse_sample, fill_nan, replace_invalid, process_name, slice_data


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

    def train(self, x_train, y_train):

        # convert training data into LightGBM dataset format
        d_train = lgb.Dataset(x_train, label=y_train)

        params = dict()
        params['learning_rate'] = 0.003
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'multiclass'
        params['metric'] = 'multi_logloss'
        params['sub_feature'] = 0.5
        params['num_leaves'] = 50
        params['min_data_in_leaf'] = 10
        params['max_depth'] = 10
        params['num_class'] = 14

        self.lgb_model = lgb.train(params, d_train, 100)

        self.lgb_model.save_model('lgb_classifier.txt', num_iteration=self.lgb_model.best_iteration)

    @staticmethod
    def predict(file):
        # read data
        data = pd.read_csv(file)
        # define keys for detecting duplicates
        keys = ['rdate', 'rid', 'hid']
        # define indices of rows to be removed
        indices = [('rank', 0), ('finishm', 0)]
        # cleanse invalid sample(s)
        data = cleanse_sample(data, keys=keys, indices=indices)

        data = data.reset_index(drop=True)

        duplicate = data.copy()

        # define rules for dropping feature
        rules = [  # useless features
                   'index', 'horsenum', 'rfinishm', 'runpos', 'windist', 'win', 'place', '(rm|p|m|d)\d+',
                   # features containing too many NANs
                   'ratechg', 'horseweightchg', 'besttime', 'age', 'priority', 'lastsix', 'runpos', 'datediff',
                   # features which are difficult to process
                   'gear', 'class', 'pricemoney'
                 ]
        # eliminate useless features
        duplicate = cleanse_feature(duplicate, rules=rules)

        # apply one-hot encoding on features
        columns = ['venue_HV', 'venue_ST', 'track_ALL WEATHER TRACK', 'track_TURF', 'going_FAST', 'going_GOOD',
                   'going_GOOD TO FIRM', 'going_GOOD TO YIELDING', 'going_SLOW', 'going_SOFT', 'going_WET FAST',
                   'going_WET SLOW', 'going_YIELDING', 'going_YIELDING TO SOFT', 'course_A', 'course_A+3',
                   'course_ALL WEATHER TRACK', 'course_B', 'course_B+2', 'course_C', 'course_C+3', 'course_TURF']
        for col in columns:
            duplicate[col] = 0

        for index, row in duplicate.iterrows():
            row['venue_%s' % row['venue']] = 1
            row['track_%s' % row['track']] = 1
            row['going_%s' % row['going']] = 1
            row['course_%s' % row['course']] = 1
            duplicate.iloc[index] = row

        duplicate = duplicate.drop(['venue', 'track', 'going', 'course'], axis=1)

        # apply target encoding on features
        jname = {'': 6.835154464623548, 'N Berry': 5.5, 'T Clark': 6.957283680175246, 'D Porcu': 13.0, 'R Thomas': 5.0, 'F Branca': 7.0, 'G Schofield': 6.923076923076923, 'M W Leung': 8.435770750988143, 'C Y Lui': 7.294736842105263, 'R Silvera': 9.5, 'F Blondel': 6.0, 'G Cheyne': 7.881165919282512, 'G Benoist': 7.044943820224719, 'A de Vries': 8.0, 'I Ortiz Jr': 9.0, 'J Fortune': 8.0, 'M Du Plessis': 7.116803278688525, 'C Reith': 7.958974358974359, 'E Saint-Martin': 6.639376218323587, 'J Leparoux': 7.5, 'M F Poon': 6.0855614973262036, 'C F Wong': 10.0, 'A Atzeni': 7.9, 'K Tosaki': 6.25, 'T Queally': 8.397435897435898, 'H Tanaka': 10.0, 'M Nunes': 7.314942528735632, 'Y Fujioka': 5.0, 'D Bonilla': 6.0, 'Y Fukunaga': 5.866666666666666, 'H Uchida': 5.6, 'G Mosse': 6.584008453680873, 'H W Lai': 7.704118993135012, 'A Starke': 6.470588235294118, 'J Doyle': 10.0, 'R Fradd': 6.75, 'I Mendizabal': 10.0, 'S Clipperton': 6.832959641255606, 'W Buick': 6.6923076923076925, 'T Thulliez': 7.0, 'W C Marwing': 6.3975840336134455, 'K Latham': 9.0, 'S K Sit': 8.333333333333334, 'J Victoire': 7.178438661710037, 'R Curatolo': 11.75, 'M L Yeung': 7.284816247582205, 'H Goto': 4.0, 'B Doyle': 7.624548736462094, "C O'Donoghue": 8.402597402597403, 'S Hamanaka': 6.25, 'K Teetan': 6.427687296416938, 'W L Ho': 8.470588235294118, 'V Espinoza': 11.5, 'R Maia': 7.0, 'K W Leung': 7.814126394052044, 'N Pinna': 8.679245283018869, 'C Newitt': 6.0, 'O Bosson': 7.84472049689441, 'K H Yu': 9.647058823529411, 'T Mundry': 6.0, 'N Callan': 6.643882433356118, 'A Munro': 5.5, 'D Dunn': 7.7272727272727275, 'N Hall': 8.0, 'H N Wong': 7.6976744186046515, 'B Shinn': 9.0, 'P Smullen': 8.25, 'M Wheeler': 8.0, 'M Demuro': 6.609677419354838, 'A Helfenbein': 9.0, 'M Wepner': 11.0, 'J Talamo': 7.75, 'K F Choi': 13.0, 'G Stevens': 10.75, 'J Murtagh': 6.523809523809524, 'L Duarte': 10.0, 'J Riddell': 9.0, 'K T Yeung': 6.927958833619211, 'L Henrique': 9.571428571428571, 'H Bowman': 6.134693877551021, 'P Robinson': 11.0, 'R Woodworth': 8.0, 'C Lemaire': 6.605263157894737, 'R Hughes': 8.625, 'F Prat': 6.0, 'B Herd': 11.0, 'A Delpech': 6.527950310559007, 'W Pike': 7.623052959501558, 'G Gibbons': 10.0, 'K Ando': 10.0, 'R Fourie': 6.73974540311174, 'L Corrales': 10.0, 'T H So': 7.895368210596468, 'L Nolen': 8.6, 'J Winks': 8.073248407643312, 'H T Mo': 6.974059662775616, 'L Salles': 8.125, 'R Moore': 6.088235294117647, 'O Doleuze': 6.610876699484295, 'B Stanley': 9.0, 'P H Lo': 8.20775623268698, 'D Beadman': 5.3827234553089385, 'C Perkins': 11.0, 'D Browne': 6.0, 'S Drowne': 8.0, 'T Hellier': 10.0, 'K M Chin': 8.96875, 'M Cangas': 14.0, 'D Oliver': 6.659574468085107, 'D Nikolic': 8.666666666666666, 'Y Take': 5.666666666666667, 'J Moreira': 4.465078854200193, 'B Prebble': 5.8738863591368045, 'T Jarnet': 8.333333333333334, 'T Durcan': 8.333333333333334, 'G Lerena': 7.335483870967742, 'J Saimee': 13.0, 'A Suborics': 7.72715318869165, 'K C Ng': 8.202979515828678, 'Y T Cheng': 7.193162868260145, 'M Zahra': 7.955056179775281, 'E da Silva': 7.25, 'D Tudhope': 7.0, 'S Khumalo': 8.25, 'D Lane': 7.955056179775281, 'O Murphy': 8.034246575342467, 'R Ffrench': 13.0, 'C Williams': 6.714285714285714, 'Colin Keane': 6.25, 'N Juglall': 9.192307692307692, 'Y Iwata': 5.315789473684211, 'G van Niekerk': 6.266666666666667, 'M Ebina': 9.666666666666666, 'P-C Boudot': 6.130434782608695, 'F Geroux': 6.25, 'K McEvoy': 5.777777777777778, 'D McDonogh': 12.0, 'O Peslier': 7.394736842105263, 'D Shibata': 11.0, 'A Sanna': 7.6077981651376145, 'K K Chiong': 6.757879656160458, 'W Smith': 9.2, 'A Crastus': 8.333333333333334, 'M Rodd': 8.857142857142858, 'A Hamelin': 10.0, 'L Dettori': 5.235294117647059, 'P Strydom': 8.433823529411764, 'H Lam': 9.25, 'C Demuro': 8.0, 'F Coetzee': 6.457446808510638, 'E Wilson': 8.8625, 'V Cheminaud': 8.065934065934066, 'R Dominguez': 10.0, 'M Kinane': 6.125, 'C Schofield': 6.5721393034825875, 'K L Chui': 7.34282325029656, 'S Arnold': 14.0, 'M Hills': 7.0, 'K Fallon': 8.692307692307692, 'C W Wong': 8.306399482870072, 'C Wong': 6.5607476635514015, 'J Spencer': 6.111111111111111, 'Z Purton': 5.698901464713715, 'S Pasquier': 6.0, 'M Chadwick': 6.699807877041306, 'S Fujita': 11.0, 'C Brown': 6.894736842105263, 'R Migliore': 10.0, 'K Manning': 12.0, 'A Badel': 6.817966903073286, 'T Ono': 8.0, 'C Soumillon': 5.334494773519164, 'C W Choi': 9.0, 'F Berry': 8.0, 'T Berry': 6.466351829988193, 'C K Tong': 8.63401882188916, 'M Guyon': 6.266666666666667, 'J Castellano': 7.0, 'H Shii': 8.0, 'C Velasquez': 6.75, 'C Y Ho': 7.134692782398712, 'D Whyte': 5.353180413209696, 'W M Lai': 7.874705188679245, 'U Rispoli': 6.735955056179775, 'M Smith': 7.166666666666667, 'N Rawiller': 6.256240822320118, 'B Vorster': 11.5, 'K Shea': 7.087912087912088, "B Fayd'herbe": 7.111111111111111, 'C Murray': 8.223809523809523, 'N Callow': 11.0, 'G Boss': 6.908675799086758, 'D Chong': 14.0, 'G Gomez': 8.181818181818182, 'J Crowley': 9.5, 'M Barzalona': 7.368421052631579, 'A Gryder': 8.66326530612245, 'T Angland': 6.360912052117264, 'K Ikezoe': 6.5, 'S Dye': 7.5816993464052285, 'C Sutherland': 8.0, 'A Spiteri': 10.333333333333334, 'F Durso': 8.090909090909092, 'W Lordan': 7.8, 'K Tanaka': 7.5, 'J McDonald': 7.015151515151516, 'J Cassidy': 9.0, 'K C Leung': 7.2411421425594, 'J Lloyd': 6.767180925666199, 'A Marcus': 7.4, 'P Holmes': 9.0, 'S de Sousa': 5.943844492440605}
        tname = {'': 6.835154464621644, 'P Bary': 10.5, 'L Ho': 7.274053163961421, 'F Head': 6.25, 'R Gibson': 6.694015807301468, 'N Sugai': 7.0, 'T A Hogan': 5.6, 'M R Channon': 10.0, 'K Prendergast': 12.0, 'D Morton': 14.0, 'J Bary': 9.0, 'M J Wallace': 13.0, 'W Hickst': 3.0, 'T W Leung': 7.41910229645094, "A P O'Brien": 9.0, 'D Smaga': 9.0, 'R Heathcote': 6.0, 'M Figge': 9.5, 'J M Moore': 11.0, 'K H Ting': 5.8474576271186445, 'J S Moore': 6.0, 'P Demercastel': 10.0, 'M Hawkes': 12.0, 'O Hirata': 5.0, 'A Ngai': 3.0, 'C Hills': 9.5, 'P Chapple-Hyam': 12.0, 'H Shimizu': 2.0, 'Pat Lee': 10.0, 'C S Shum': 6.919644839067702, 'Barande-Barbe': 4.8, 'E Lellouche': 7.5, 'R M Beckett': 7.0, 'Y C Fung': 10.5, 'L K Laxon': 12.0, 'M Delzangles': 4.25, 'B W Hills': 2.0, 'Y Yahagi': 7.5, 'W J Haggas': 9.0, 'A Lee': 7.44659300184162, 'J H M Gosden': 2.0, 'J G Given': 9.0, 'H-A Pantall': 11.0, 'D A Hayes': 4.666666666666667, 'A Schutz': 7.273796398382947, 'J W Sadler': 13.0, 'H Uehara': 10.0, 'E Libaud': 2.5, 'Sir M R Stoute': 6.5, 'M Nishizono': 6.0, 'K Sumii': 2.5, 'S bin Suroor': 4.125, 'A S Cruz': 6.474646983311938, 'J Stephens': 3.0, 'F C Lor': 5.678765880217786, "D O'Meara": 7.0, 'F Doumen': 7.0, 'W Y So': 7.055802668823292, 'Y S Tsui': 7.114738044720092, 'B K Ng': 6.888098318240621, 'R Varian': 6.0, 'T Yasuda': 3.5, 'K Yamauchi': 9.0, 'G A Nicholson': 11.0, 'T Mundry': 10.0, 'N Hori': 3.25, 'D L Romans': 9.0, 'S Tarry': 7.0, 'P Van de Poele': 5.0, 'H Fujiwara': 4.2, 'C Marshall': 13.0, 'C Dollase': 12.0, 'P C Kan': 12.333333333333334, 'W P Mullins': 5.0, 'J Moore': 5.952110091743119, 'D J Hall': 6.718906720160481, 'B D A Cecil': 5.0, 'K C Chong': 10.333333333333334, 'R E Dutrow Jr': 9.0, 'M Kon': 10.5, 'C H Yip': 6.938029068838843, 'R Rohne': 13.0, 'N Meade': 4.0, 'X T Demeaulte': 11.0, 'S L Leung': 8.0, 'M Botti': 9.666666666666666, 'J-M Beguigne': 5.0, 'W Figge': 10.0, 'H Blume': 3.0, 'A Wohler': 11.0, 'N D Drysdale': 11.0, 'G M Begg': 8.0, 'S Kunieda': 9.666666666666666, 'B Smart': 12.0, 'J Noseda': 7.0, 'R Collet': 6.5, 'J-C Rouget': 8.0, 'M Johnston': 13.0, 'J E Hammond': 13.0, 'W A Ward': 10.0, 'N Takagi': 8.0, 'A M Balding': 7.0, 'M C Tam': 10.75, 'J-P Gauvin': 10.0, 'G W Moore': 9.727272727272727, 'K W Lui': 7.080074923905409, 'Y Kato': 5.0, 'E Botti': 5.0, 'M F de Kock': 5.705882352941177, 'R Bastiman': 11.0, 'Y Ikee': 5.0, 'P F Yiu': 6.7780028357302005, 'R Charlton': 7.5, 'P Leyshan': 10.333333333333334, 'A T Millard': 6.52387808975282, 'A Fabre': 5.2, 'T K Ng': 7.647038801906058, 'K A Ryan': 9.0, 'E M Lynam': 8.4, 'G Eurell': 5.0, 'P B Shaw': 7.0, 'K Kato': 8.5, 'R M H Cowell': 14.0, 'M A Jarvis': 11.666666666666666, 'B Koriner': 5.0, 'P Rau': 8.0, 'T P Yung': 6.864481409001957, 'C W Chang': 7.267752608806312, 'K C Wong': 12.25, 'J Lau': 6.25, 'C Fownes': 6.387122801355495, 'C Clement': 11.0, 'A W Noonan': 5.0, 'R J Frankel': 7.0, 'M Ito': 4.0, 'de Royer Dupre': 5.769230769230769, "P O'Sullivan": 6.749095607235142, 'I Sameshima': 14.0, 'E A L Dunlop': 4.6, 'J R Fanshawe': 10.0, 'L M Cumani': 4.333333333333333, 'Rod Collet': 5.5, 'B J Wallace': 9.0, 'J Hawkes': 3.0, 'S Chow': 10.0, 'D Cruz': 7.304486251808973, 'J C Englehart': 12.0, 'J Sadler': 13.0, 'W H Tse': 12.0, 'S Woods': 7.227761836441894, 'S Kojima': 10.0, 'M N Houdalakis': 1.0, 'D L Freedman': 10.0, 'K L Man': 7.062692702980473, 'G Allendorf': 10.285714285714286, 'S H Cheong': 14.0, "J O'Hara": 12.5, 'J S Bolger': 12.0, 'P Schiergen': 7.0, 'R Hannon Snr': 9.75, "J O'Shea": 11.0, 'M J Freedman': 7.074303405572755, 'J Size': 5.533564146951337, 'D E Ferraris': 6.753846153846154, 'B J Meehan': 9.0, 'Y O Wong': 7.788259958071279, 'H J Brown': 9.5, 'K H Leong': 11.0, 'K Ikezoe': 9.0, 'P G Moody': 11.0, "D O'Brien": 9.0, 'J M P Eustace': 12.0, 'Laffon-Parias': 5.6, 'M L W Bell': 11.0, 'D Simcock': 7.0}

        for name in duplicate['jname'].unique():
            if type(name) != str:
                pass
            elif jname.get(name) is None:
                duplicate['jname'].replace(name, jname[''], inplace=True)
            else:
                duplicate['jname'].replace(name, jname[name], inplace=True)
        duplicate['jname'].fillna(value=jname[''], inplace=True)

        for name in duplicate['tname'].unique():
            if type(name) != str:
                pass
            elif tname.get(name) is None:
                duplicate['tname'].replace(name, tname[''], inplace=True)
            else:
                duplicate['tname'].replace(name, tname[name], inplace=True)
        duplicate['tname'].fillna(value=tname[''], inplace=True)

        # perform min-max standardization
        minmax = {'distance': (1000, 2400), 'jname': (4.0, 14.0), 'tname': (1.0, 14.0), 'exweight': (103, 133),
                  'bardraw': (1.0, 15.0), 'rating': (3.0, 134.0), 'horseweight': (693.0, 1369.0),
                  'win_t5': (1.05, 99.0), 'place_t5': (0.0, 69.35), 'venue_HV': (0, 1), 'venue_ST': (0, 1),
                  'track_ALL WEATHER TRACK': (0, 1), 'track_TURF': (0 ,1), 'going_FAST': (0, 1),
                  'going_GOOD': (0, 1), 'going_GOOD TO FIRM': (0, 1), 'going_GOOD TO YIELDING': (0, 1),
                  'going_SLOW': (0, 1), 'going_SOFT': (0, 1), 'going_WET FAST': (0, 1), 'going_WET SLOW': (0, 1),
                  'going_YIELDING': (0, 1), 'going_YIELDING TO SOFT': (0, 1), 'course_A': (0, 1),
                  'course_A+3': (0, 1), 'course_ALL WEATHER TRACK': (0, 1), 'course_B': (0, 1),
                  'course_B+2': (0, 1), 'course_C': (0, 1), 'course_C+3': (0, 1), 'course_TURF': (0, 1)}

        for key in duplicate.keys():
            if key not in ['rdate', 'rid', 'hid', 'finishm', 'rank', 'ind_win', 'ind_pla']:
                if duplicate[key].max() != duplicate[key].min():
                    duplicate[key] = np.clip((duplicate[key] - minmax[key][0]) / (minmax[key][1] - minmax[key][0]),
                                             0, 1)

        x, y = slice_data(duplicate)

        # prediction
        clf = lgb.Booster(model_file='lgb_classifier.txt')

        winprob = clf.predict(x)
        data['winprob'] = 0
        # data['p_rank'] = 0

        i = 0
        groups = data.groupby(['rdate', 'rid'])
        for name, group in groups:
            total = np.sum(winprob[i, 0:len(group)])

            # print(total)
            # rank = np.argsort(winprob[i, 0:len(group)])[::-1]
            # print(rank)
            # rank = np.array([np.where(rank == k)[0][0] + 1 for k in range(len(rank))])

            j = 0
            for index, row in group.iterrows():
                row['winprob'] = winprob[i, j] / total
                # row['p_rank'] = rank[j]
                data.iloc[index] = row
                j += 1
            i += 1

            # print(group)
            # print(rank)

        data['plaprob'] = WinP2PlaP(data, wpcol='winprob')

        ### choose a fixed ratio of bankroll and merit threshold to get betting stake vectors of win and place
        ## you should control the sum of betting ratios per week is less than 1, otherwise you may end up bankrupting!
        ## Higher ratio means bigger risk
        fixratio = 1 / 10000
        mthresh = 9
        print("Getting win stake...")
        data['winstake'] = fixratio * (data['winprob'] * data['win_t5'] > mthresh)
        print("Getting place stake...")
        data['plastake'] = fixratio * (data['plaprob'] * data['place_t5'] > mthresh)

        data.to_csv('test_result.csv')


if __name__ == '__main__':
    # read data from disk
    model = RacingPredictor('../Data/HR200709to201901.csv', debug=True)

    # pre-process data
    # model.pre_process(persistent=False)

    # # divide the data set into training set and testing set
    # x, y = slice_data(model.data)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # model.train(x, y)
    # model.train(x_train, y_train)

    model.predict('Sample_test.csv')
