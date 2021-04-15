# the CBSData class is a private package for querying the lab's Cambridge Brain 
# Sciences database, extracting data, etc. for our studies. However, we cannot
# share the existing package because it contains details about database access
# and other (non public) studies, etc. Therefore this a "stub" version of that 
# package that directly loads the data from .CSVs that should be placed along
# beside these scripts.

import pandas as pd
from os import path

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
        return path.join('..', 'data')


class Questionnaire(object):
    """ This class is intantiated to create a Questionnaire object, that 
        contains data in a Pandas DataFrame. Again, there are usually other 
        functions that go in here like the ability to access and pull from a
        database, parse data, etc.
    """

    @property
    def data_file(self):
        return path.join(self._study.data_directory, self._data_filename)

    def __init__(self, study_class, data_file):
        self._study = study_class
        self._data_filename = data_file
        self.data = pd.read_csv(self.data_file)

class ScoreData(Questionnaire):
    def __init__(self, study_class, data_file):
        self._study = study_class
        self._data_filename = data_file
        self.data = pd.read_csv(self.data_file, index_col=[0,1], header=[0,1])
        self.data.index.set_names(['user', 'device_type'], inplace=True)