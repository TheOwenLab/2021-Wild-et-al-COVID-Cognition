# Here is the stub for loading the control dataset from another study.
# Again, most things have been stripped out and the data are directly loaded 
# from CSV files.

from .cbs_data import Questionnaire, ScoreData, MetaCBSData, QuestionType

class ControlStudyQuestionnaire(Questionnaire):

    @property
    def freq_categories(self):
        return [
            'Not during the past month',
            'Less than once a week',
            'Once or twice a week',
            'Three or more times a week',
            'Every day'
        ]

    @property
    def map_(self):
        return {
            'sex': ('sex', QuestionType.CATEGORICAL),
            'education': ('education', QuestionType.CATEGORICAL, ["None", "High School Diploma", "Bachelor's Degree", "Master's Degree", "Doctoral or Professional Degree"]),
            'age': ('age', QuestionType.NUMERIC),
            'SES_growing_up': ('SES_growing_up', QuestionType.CATEGORICAL),
            'exercise_freq': ('exercise_freq', QuestionType.CATEGORICAL, ['Not during the past month', 'Less than once a week', 'Once or twice a week', 'Three or more times a week', 'Every day']),
            'cannabis': ('cannabis', QuestionType.BOOLEAN),
            'stimulants': ('stimulants', QuestionType.BOOLEAN),
            'depressants': ('depressants', QuestionType.BOOLEAN),
            'fluent_english': ('fluent_english',  QuestionType.BOOLEAN),
            'hospital_stay': ('hospital_stay',  QuestionType.BOOLEAN)
        }

    def __init__(self, study_class, data_file):
        super().__init__(study_class, data_file)
        for col in self.categorical_items:
            if len(self.map_[col]) > 2:
                self.data[col] = (
                    self.data[col]
                    .cat.reorder_categories(new_categories=self.map_[col][2], ordered=True)
                )


class MetaControlStudy(MetaCBSData):
    def __init__(self, name, base, ns):
        self._questionnaire = ControlStudyQuestionnaire(self,
            self.most_recent_file('Control_Study_Wild_etal_2021_questionnaire_data')
        )

        self._score_data = ScoreData(self,
            self.most_recent_file('Control_Study_Wild_etal_2021_CBS_task_data')
        )

    @property
    def questionnaire(self):
        return self._questionnaire

    @property
    def score_data(self):
        return self._score_data.data.copy()

class ControlStudy(object, metaclass=MetaControlStudy):
    pass


