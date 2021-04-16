# -----------------------------------------------------------------------------
# This is a stub file that provides necessary funtionaloity from a custom Python
# package created for working with CBS data. Rather than providing all the 
# source code for that package, which is a bit overkill for using a few utility
# functions, I stripped out the bits required for this study and posted them 
# in here.
# -----------------------------------------------------------------------------
# cwild 2021-04-15

import numpy as np
import pandas as pd
import re
import json

class Test(object):
    
    _NUM_RAW_FIELDS = 8
    _FEATURE_FIELD_NAMES = ['single_score'] + ['raw_field_%d'%(i+1) for i in range(_NUM_RAW_FIELDS)]

    _id = None
    _name = ''
    _abbrev = ''
    _features = ['final_score']
    _hc_feature = 'final_score'
    _domain_feature = 'final_score'
    _chance = 0
    _r_hc = 0

    def __init__(self):
        score_features = {self.features()[i]: self._FEATURE_FIELD_NAMES[i] for i in range(len(self.features()))}
        common_features = {x: x for x in self.common_session_features()}
        self._feature_map = {**score_features, **common_features}

    @property
    def name(self):
        """ The name of the test. """
        return self._name
    
    @property
    def id(self):
        """ The ID number for this tests. Corresponds to the MySQL ID. """
        return self._id
    
    @property
    def abbrev(self):
        """ Abbreviated version of the test name. """
        return self._abbrev
    
    @property
    def chance_level(self):
        """ Every test has an associated "chance" level of performance. """
        return self._chance

    @property
    def reliability(self, app='hc'):
        """ Reliability, based on CBS app, of the HC feature """
        if app == 'hc':
            return self._r_hc

    def features(self, include_common=False):
        """ A list of all features (names) for scores from this test. """
        if include_common:
            return self._features + self.common_session_features()
        else:
            return self._features

    def num_features(self, include_common=False):
        """ The number of features for scores from this Test. """
        if include_common:
            return len(self._features) + len(self.common_session_features())
        else:
            return len(self._features)

    @property
    def feature_fields(self):
        """ A list of all score fields that contain features. """
        return self._FEATURE_FIELD_NAMES[0:self.num_features(include_common=False)]

    @property
    def hc_feature(self):
        """ The name of the feature used for HC score calculations. """
        return self._hc_feature
    
    @property
    def hc_field(self):
        """ The score field that contains the feature used for HC score calculations. """
        return self.field_for_feature(self.hc_feature)

    @property
    def domain_feature(self):
        """ The name of the feature used for domain score calculations. """
        return self._domain_feature
    
    @property
    def domain_field(self):
        """ The score field that contains the feature used for domain score calculations. """
        return self.field_for_feature(self.domain_feature)
    
    def field_for_feature(self, feature_name):
        """ Given the name of a score feature, returns the score field that contains that data.
        Args:
            feature_name: the name of the feature (as a string)
        Returns:
            The corresponding field name, as a string.
        """
        return self._feature_map[feature_name]

    def parse_raw(self, raw_score_str):
        return {}

    @classmethod
    def common_session_feature_calcs(self):
        """ This dict maps scores features (names) to function calls.
            Keys are the names of features, values are the names of the function
            that uses the session data to calculate the feature value.
        """
        return {
            'num_errors': 'num_errors_from_session',
            'num_correct': 'num_correct_from_session',
            'num_attempts': 'num_attempts_from_session',
            'accuracy': 'accuracy_from_session',
            'better_than_chance': 'better_than_chance_from_session',
            'duration_ms': 'duration_from_session',
            'avg_ms_correct': 'avg_ms_correct_from_session',
        }
    
    @classmethod
    def common_session_features(self):
        """ Returns a list of features (names) that are derived from session
            data, and that are common to all tests. This is a classmethod,
            because it is like a property of the Test class, rather than an 
            instance of a specific Test.
        """
        return list(self.common_session_feature_calcs().keys())

    @property
    def test_specific_session_feature_calcs(self):
        return {}

    @property
    def all_session_feature_calcs(self):
        return { **self.common_session_feature_calcs(),
                 **self.test_specific_session_feature_calcs }

    @property
    def all_session_features(self):
        return list(self.all_session_feature_calcs.keys())

    def parse_session(self, session_data_str, extract_features=True):
        try:
            session_data = json.loads(session_data_str)
        except:
            print(f"WARNING: Cannot parse JSON session {session_data_str}")
            return {}

        if session_data['code'] != self.abbrev.lower():
            print(f"WARNING: Invalid Task Code, got {session_data['code']} expected {self.abbrev.lower()}")
            return {}
                
        if extract_features:
            return { feature: getattr(self, function)(session_data) for 
                        feature, function in self.all_session_feature_calcs.items() }

        return session_data
    
    def correct_trial_data(self, session_data):
        """ Returns a list of the correct trials from the parsed session_data
        """
        return [
            q for q in session_data['questions'] if \
                ('isCorrect' in q) and \
                ('endTimeSpan' in q) and \
                (q['isCorrect'])
        ]
    
    def question_difficulty(self, session_data):
        return np.nan

    def extract_trial_data(self, session_data):
        trials = []
        if (session_data is not None) and ('questions' in session_data):
            for q in session_data['questions']:
                if 'endTimeSpan' in q:
                    trials.append([
                        bool(q['isCorrect']),
                        self.question_difficulty(q), 
                        q['durationTimeSpan'],
                    ])
        return (pd
            .DataFrame(trials, columns=['correct', 'difficulty', 'duration'])
            .rename_axis('trial'))

    def num_errors_from_session(self, session_data):
        return session_data['errorsMade']

    def num_correct_from_session(self, session_data):
        return session_data['correctAnswers']

    def num_attempts_from_session(self, session_data):
        return session_data['errorsMade']+session_data['correctAnswers']

    def accuracy_from_session(self, session_data):
        total = self.num_attempts_from_session(session_data)
        correct = self.num_correct_from_session(session_data)
        return correct/total if total != 0 else np.nan

    def better_than_chance_from_session(self, session_data):
        accuracy = self.accuracy_from_session(session_data)
        return accuracy > self.chance_level

    def duration_from_session(self, session_data):
        return session_data['task']['durationTimeSpan']

    def avg_ms_correct_from_session(self, session_data):
        durations = np.array([q['durationTimeSpan'] 
            for q in self.correct_trial_data(session_data)])
        return durations.mean() if durations.shape[0] > 0 else np.nan


class MemoryTest(Test):

    _features = ['max_score','avg_score','avg_ms_per_item']
    _hc_feature = 'avg_score'
    _domain_feature = 'max_score'

    """ Memory Tests have a consistent raw score format """
    def parse_raw(self, raw_score_str):
        """ For memory tests, the raw score is a floating point number, which is
            just the average score feature. However, cases where the score is
            actually zero (0), the raw field contains an empty string rather
            than zero; check for this possibility. """
        if raw_score_str == '':
            avg_score = 0
        else:
            avg_score = float(raw_score_str)
        return {'avg_score': avg_score}

    @property
    def test_specific_session_feature_calcs(self):
        return {'avg_ms_per_item': 'calc_avg_ms_per_item'}

    def calc_avg_ms_per_item(self, session_data):
        item_times = [q['durationTimeSpan']/self.question_difficulty(q)
            for q in self.correct_trial_data(session_data)]
        return np.array(item_times).mean()

class SpatialSpanTest(MemoryTest):
    _id = 20
    _name = 'spatial_span'
    _abbrev = 'SS'
    _r_hc = 0.603409

    def question_difficulty(self, q):
        return len(q['question'])

class DigitSpanTest(MemoryTest):
    _id = 17
    _name = 'digit_span'
    _abbrev = 'DS'
    _r_hc = 0.608394

    def question_difficulty(self, q):
        return q['attributes']['score']

class MonkeyLadderTest(SpatialSpanTest):
    _id = 24
    _name = 'monkey_ladder'
    _abbrev = 'ML'
    _r_hc = 0.584249

    def question_difficulty(self, q):
        return len(q['question'])

class PairedAssociatesTest(SpatialSpanTest):
    _id = 19
    _name = 'paired_associates'
    _abbrev = 'PA'
    _r_hc = 0.488574

    def question_difficulty(self, q):
        return len(q['question'])

class TokenSearchTest(MemoryTest):
    _id = 18
    _name = 'token_search'
    _abbrev = 'TS'
    _r_hc = 0.596558

    def question_difficulty(self, q):
        return q['attributes']['0']['difficultyLevel']

class ReasoningTest(Test):
    _features = ['final_score','attempted','errors','max','correct_score']
    _hc_feature = 'final_score'
    _domain_feature = 'final_score'
    _chance = 0.5

    """ Reasoning Tests have a consistent raw score format """
    _raw_str_regexp = re.compile(r"Attempted\s+(?P<attempted>-?\d+)\s+Errors\s+(?P<errors>-?\d+)\s+Max\s+(?P<max>-?\d+)\s+Score\s+(?P<score>-?\d+)\s+CorrectScore\s+(?P<correct_score>-?\d+)")
    
    def parse_raw(self, raw_score_str):
        """ Use a regexp to parse out keyword separated feature values """
        m = re.search(self._raw_str_regexp, raw_score_str)
        return {feature:float(value) for (feature,value) in m.groupdict().items()} 

class RotationsTest(ReasoningTest):
    _id = 22
    _name = 'rotations'
    _abbrev = 'RT'
    _r_hc = 0.600405

    def question_difficulty(self, q):
        return q['difficulty']

class FeatureMatchTest(ReasoningTest):
    _id = 21
    _name = 'feature_match'
    _abbrev = 'FM'
    _r_hc = 0.56137

    def question_difficulty(self, q):
        return q['difficulty']

class OddOneOutTest(ReasoningTest):
    _id = 14
    _name = 'odd_one_out'
    _abbrev = 'OOO'
    _features = ['final_score','attempted','errors','max']
    _domain_feature = 'max'
    _chance = 1.0/9
    _r_hc = 0.454132

    """ OddOneOut is like other reasoning tests, but with slightly different raw format """
    _raw_str_regexp = re.compile(r"Attempted\s+(?P<attempted>-?\d+)\s+Errors\s+(?P<errors>-?\d+)\s+Max\s+(?P<max>-?\d+)")

    def question_difficulty(self, q):
        return q['attributes']['realDifficulty']

class DoubleTroubleTest(ReasoningTest):
    _id = 13
    _name = 'double_trouble'
    _abbrev = 'DT'
    _features = ['final_score','pct_CC','pct_CI','pct_IC','pct_II','RT_CC','RT_CI','RT_IC','RT_II']
    _r_hc = 0.822693

    """ DT has a very different raw format """
    _raw_str_regexp = re.compile(r"\A(?P<pct_CC>\S+)\s+(?P<pct_CI>\S+)\s+(?P<pct_IC>\S+)\s+(?P<pct_II>\S+)\s+(?P<RT_CC>\S+)\s+(?P<RT_CI>\S+)\s+(?P<RT_IC>\S+)\s+(?P<RT_II>\S+)")

    def question_difficulty(self, q):
        return {'CC': 0, 'CI': 1, 'IC': 1, 'II': 2}[q['problemCode']]

class PolygonsTest(Test):
    _id = 23
    _name = 'polygons'
    _abbrev = 'PO'
    _chance = 0.5
    _r_hc = 0.523616

    def question_difficulty(self, q):
        return q['difficulty']

class SpatialPlanningTest(Test):
    _id = 15
    _name = 'spatial_planning'
    _abbrev = 'SP'
    _r_hc = 0.731601

    def question_difficulty(self, q):
        return q['attributes']['difficultyLevel']

class GrammaticalReasoningTest(Test):
    _id = 16
    _name = 'grammatical_reasoning'
    _abbrev = 'GR'
    _chance = 0.5
    _r_hc = 0.758347


TESTS = { 
          'spatial_span': SpatialSpanTest(),
          'grammatical_reasoning': GrammaticalReasoningTest(),
          'double_trouble': DoubleTroubleTest(),
          'odd_one_out': OddOneOutTest(),
          'monkey_ladder': MonkeyLadderTest(),
          'rotations': RotationsTest(),
          'feature_match': FeatureMatchTest(),
          'digit_span': DigitSpanTest(),
          'spatial_planning': SpatialPlanningTest(),
          'paired_associates': PairedAssociatesTest(),
          'polygons': PolygonsTest(),
          'token_search': TokenSearchTest()
        }

def test_names():
    return list(TESTS.keys())

def all_feature_fields(include_common=False):
    """ Returns a list of tuples, where the first tuple element is the test 
        name, and the second tuple  value is a feature field name. Useful for 
        subselecting the partition of a full score data frame that contains
        data.
    """
    all_fields = []
    for test_name, test in TESTS.items():
        test_fields = test.feature_fields
        if include_common: test_fields += test.common_session_features()
        all_fields += [(test_name, field) for field in test_fields]
    return all_fields
    
def all_features(include_common=False, exclude=[], only=None):
    """ Returns a list of tuples, where the first tuple element is the test 
        name, and the second tuple value is a feature name. Useful for
        subselecting the partition of a full score data frame that contains
        data.
    Args:
        include_common: (default True) should we include all features that are
            common for all tests? e.g., num attempts, errors, etc.
        exclude: (list-like, default empty list) remove any features that match
            anything in the list. That is, filter out some features by name.
        only: (list-like, default empty list) retain only features that match
            anything in this list. Kind of the opposite of 'exclude'.
    """
    features = [(test_name, feature) for (test_name, test) in TESTS.items() for 
            feature in test.features(include_common)]
    filtered = []
    for feature in features:
        if (feature[1] not in exclude) and (only is None or feature[1] in only):
            filtered.append(feature)
    return filtered
    
def timing_features(exclude=[], abbrev=False):
    """ Returns all reaction-time based measures for each test.
    """
    def timing_feat(test):
        return 'avg_ms_per_item' if test in memory_tests() else 'avg_ms_correct'

    feats = [(test, timing_feat(test)) for test, _ in TESTS.items() if test not in exclude]
    if abbrev:
        feats = abbrev_features(feats)
    return feats

def abbrev_features(feature_list):
    """ Given a list of test score features as a tuple (used for multiindex),
        translate into a single level list using abbreviated test names.
    """
    return [f"{TESTS[t[0]].abbrev}_{t[1]}" for t in feature_list]

def test_features(app='domain', type_='feature'):
    """ Generate a list of test score features used to calculate score for a 
        given application. This is useful for indexing complete score dataframe 
        to select only features used in that application.
    Arguments:
        app: (string-like, default 'domain') can be 'domain' or 'hc'
        type_: (string-like, default 'feature') can be 'feature' or 'field'
    
    Returns:
        [list of tuples] - Each is of the form (test, feature)
    """
    if app not in ['domain', 'hc']:
        raise ValueError(f"Invalid application {app}")

    if type_ not in ['feature', 'field']:
        raise ValueError(f"Invalid argument {type_}")

    return [(test_name, getattr(test, f"{app}_{type_}")) for
            (test_name, test) in TESTS.items()]

def domain_feature_list():
    return test_features('domain', 'feature')

def hc_feature_list():
    return test_features('hc', 'feature')

def tests_of_type(type_, exclude=[]):
    parent_class = {'memory': MemoryTest,
                    'reasoning': ReasoningTest}.get(type_, None)

    if parent_class is None:
        raise AttributeError(f"Invalid test type {type_}")

    return [name for (name, test) in TESTS.items() if
            issubclass(test.__class__, parent_class) and
            name not in exclude]


def memory_tests(exclude=[]):
    return tests_of_type('memory', exclude)


def reasoning_tests(exclude=[]):
    return tests_of_type('reasoning', exclude)

DOMAIN_NAMES = ['STM', 'reasoning', 'verbal']

