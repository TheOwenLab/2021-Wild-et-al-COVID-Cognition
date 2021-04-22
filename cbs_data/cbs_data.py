# the CBSData class is a private package for querying the lab's Cambridge Brain 
# Sciences database, extracting data, etc. for our studies. However, we cannot
# share the existing package because it contains details about database access
# and other (non public) studies, etc. Therefore this a "stub" version of that 
# package that directly loads the data from .CSVs that should be placed along
# beside these scripts.

import pandas as pd
from os import path
from enum import Enum

class MetaCBSData(type):
    """ A meta class for data sources, basically all we need for this stubbed
        version is a reference to the working data directory. In another world,
        this has things like that database details, access settings, etc.
    """

    @property
    def data_directory(cls):
        """ The root data directory. By default this is a directory named 
            "data" that is in the root of this repository.
        """
        return path.join('.', 'cbs_data', 'data')


class Questionnaire(object):
    """ This class is intantiated to create a Questionnaire object, that 
        contains data in a Pandas DataFrame. Again, there are usually other 
        functions that go in here like the ability to access and pull from a
        database, parse data, etc.
    """

    @property
    def data_file(self):
        return path.join(self._study.data_directory, self._data_filename)

    @property
    def map_(self):
        """ Overriden in subclasses
        """
        return {}

    def __init__(self, study_class, data_file):
        self._study = study_class
        self._data_filename = data_file
        self.data = (pd
            .read_csv(self.data_file)
            .set_index('user')
        )
        self.data = self._convert_columns(self.data)


    def _convert_columns(self, df):
        """ Helper function to converty all columns of a questionnaire dataframe
            into their datatype, primarily used when pre-processing the raw
            questionnaire CSV.
        """
        from pandas.api.types import CategoricalDtype

        for q in list(self.categorical_items) + list(self.yesno_items):
            df[q] = df[q].astype('category')
            df[q] = df[q].cat.add_categories("NA")
            q_info = self._details_for_item(q)
            if len(q_info) > 2:
                cat_type = CategoricalDtype(categories=q_info[2], ordered=True)
                df[q] = df[q].astype(cat_type)

        for q in self.numeric_items:
            df[q] = pd.to_numeric(df[q], errors='coerce')

        for q in self.datetime_items:
            df[q] = pd.to_datetime(df[q], errors='coerce')

        for q in self.selectable_items:
            df[q] = ~df[q].isna()

        return df

    def _details_for_item(self, item):
        return self.map_[item]
    
    def _items_of_type(self, q_type):
        return {q_name for q_name, q_info in self.map_.items() if q_info[1] == q_type}

    @property
    def categorical_items(self):
        return self._items_of_type(QuestionType.CATEGORICAL)

    @property
    def yesno_items(self):
        return self._items_of_type(QuestionType.YESNO)

    @property
    def numeric_items(self):
        return self._items_of_type(QuestionType.NUMERIC)

    @property
    def selectable_items(self):
        return self._items_of_type(QuestionType.SELECTABLE)

    @property
    def datetime_items(self):
        return self._items_of_type(QuestionType.DATETIME)

    @property
    def multiselect_items(self):
        return self._items_of_type(QuestionType.MULTISELECT)

    @property
    def string_items(self):
        return self._items_of_type(QuestionType.STRING)


class ScoreData(Questionnaire):
    def __init__(self, study_class, data_file):
        self._study = study_class
        self._data_filename = data_file
        self.data = pd.read_csv(self.data_file, index_col=[0,1], header=[0,1])
        self.data.index.set_names(['user', 'device_type'], inplace=True)
        for col in self.data.columns:
            self.data[col] = pd.to_numeric(self.data[col])


class QuestionType(Enum):
    NUMERIC = 0
    CATEGORICAL = 1
    MULTISELECT = 2
    DATETIME = 3
    STRING = 4
    YESNO = 5
    SELECTABLE = 6