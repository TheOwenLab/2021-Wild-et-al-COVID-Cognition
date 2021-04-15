# Here is the stub for loading data from the COVID Brain Study (aka COVID 
# Cognition). Again, most things have been stripped out. Intead, data are
# directly loaded from CSV files.

from .cbs_data import Questionnaire, ScoreData, MetaCBSData

class MetaCovidCognition(MetaCBSData):
    def __init__(self, name, base, ns):
        self._questionnaire = Questionnaire(self, 'COVID_Cognition_Wild_etal_2021_questionnaire_data_2021-04-15.csv')
        self._score_data = ScoreData(self, 'COVID_Cognition_Wild_etal_2021_CBS_task_data_2021-04-15.csv')

    @property
    def questionnaire(self):
        return self._questionnaire

    @property
    def score_data(self):
        return self._score_data.data

class CovidCognition(object, metaclass=MetaCovidCognition):
    pass
