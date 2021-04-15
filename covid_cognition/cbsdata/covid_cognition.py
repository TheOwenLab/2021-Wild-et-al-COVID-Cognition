# Here is the stub for loading data from the COVID Brain Study (aka COVID 
# Cognition). Again, most things have been stripped out. Intead, data are
# directly loaded from CSV files.

from .cbs_data import Questionnaire, ScoreData, MetaCBSData, QuestionType

import pandas as pd
import numpy as np

def extract_date(r):
    try:
        return pd.to_datetime(f"{r.Q4_2} {r.Q4_1}")
    except:
        return pd.NaT

class CovidCogQuestionnaire(Questionnaire):

    @property
    def map_(self):
        return {
            'duration': ('Duration (in seconds)', QuestionType.NUMERIC),
            'start_date': ('StartDate', QuestionType.DATETIME),
            'progress': ('Progress', QuestionType.NUMERIC),
            'user': ('qrid', QuestionType.STRING),
            'Q2_cough': ('Q2_1', QuestionType.SELECTABLE),
            'Q2_fever': ('Q2_2', QuestionType.SELECTABLE),
            'Q2_difficulty_breathing': ('Q2_3', QuestionType.SELECTABLE),
            'Q2_pneumonia': ('Q2_4', QuestionType.SELECTABLE),
            'Q2_loss_of_smell': ('Q2_5', QuestionType.SELECTABLE),
            'Q2_travel': ('Q2_6', QuestionType.SELECTABLE),
            'Q2_close_contact': ('Q2_7', QuestionType.SELECTABLE),
            'Q2_no_symptoms': ('Q2_8', QuestionType.SELECTABLE),
            'positive_test': ('Q3', QuestionType.YESNO),
            'most_recent_test': ('Q4', QuestionType.DATETIME),
            'hospital_stay': ('Q5', QuestionType.YESNO),
            'days_in_hospital': ('Q6', QuestionType.NUMERIC),
            'supplemental_O2_hospital': ('Q7', QuestionType.YESNO),
            'ICU_stay': ('Q8', QuestionType.YESNO),
            'days_in_ICU': ('Q9', QuestionType.NUMERIC),
            'supplemental_O2_ICU': ('Q10', QuestionType.YESNO),
            'ventilator': ('Q11', QuestionType.YESNO),
            'days_on_ventilator': ('Q12', QuestionType.NUMERIC),
            'days_of_symptoms': ('Q13', QuestionType.NUMERIC),
            'daily_routine': ('Q14', QuestionType.YESNO),
            'age': ('Q15', QuestionType.NUMERIC),
            'sex': ('Q16', QuestionType.CATEGORICAL),
            'gender': ('Q17', QuestionType.CATEGORICAL),
            'handedness': ('Q18', QuestionType.CATEGORICAL),
            'country': ('Q19', QuestionType.CATEGORICAL),
            'speak_english': ('Q20_1', QuestionType.SELECTABLE),
            'speak_french': ('Q20_2', QuestionType.SELECTABLE),
            'speak_spanish': ('Q20_3', QuestionType.SELECTABLE),
            'ses': ('Q21', QuestionType.CATEGORICAL),
            'education': ('Q22', QuestionType.CATEGORICAL, ["No certificate, diploma or degree",  "High school diploma or equivalent", "Some university or college, no diploma", "Undergraduate degree or college diploma", "Graduate degree"]),
            'employment': ('Q23', QuestionType.CATEGORICAL),
            'unemployed_due_to_covid': ('Q24', QuestionType.YESNO),
            'nicotine': ('Q26', QuestionType.NUMERIC),
            'caffeine': ('Q27', QuestionType.NUMERIC),
            'alcohol': ('Q28', QuestionType.NUMERIC),
            'cannabis': ('Q29', QuestionType.NUMERIC),
            'drugs_none': ('Q30_1', QuestionType.SELECTABLE),
            'drugs_stimulants': ('Q30_2', QuestionType.SELECTABLE),
            'drugs_depressants': ('Q30_3', QuestionType.SELECTABLE),
            'drugs_hallucinogenics': ('Q30_4', QuestionType.SELECTABLE),
            'drugs_other': ('Q30_5', QuestionType.SELECTABLE),
            'PHQ2-1': ('Q31_1', QuestionType.CATEGORICAL, ['Not at all', 'Several days', 'More than half the days', 'Nearly every day']),
            'PHQ2-2': ('Q31_2', QuestionType.CATEGORICAL, ['Not at all', 'Several days', 'More than half the days', 'Nearly every day']),
            'GAD2-1': ('Q31_3', QuestionType.CATEGORICAL, ['Not at all', 'Several days', 'More than half the days', 'Nearly every day']),
            'GAD2-2': ('Q31_4', QuestionType.CATEGORICAL, ['Not at all', 'Several days', 'More than half the days', 'Nearly every day']),
            'exercise_freq': ('Q33', QuestionType.CATEGORICAL),
            'Q34_diabetes': ('Q34_1', QuestionType.SELECTABLE),
            'Q34_obesity': ('Q34_2', QuestionType.SELECTABLE),
            'Q34_hypertension': ('Q34_3', QuestionType.SELECTABLE),
            'Q34_stroke': ('Q34_4', QuestionType.SELECTABLE),
            'Q34_heart_attack': ('Q34_5', QuestionType.SELECTABLE),
            'Q34_memory_problem': ('Q34_6', QuestionType.SELECTABLE),
            'Q34_concussion': ('Q34_7', QuestionType.SELECTABLE),
            'Q34_none': ('Q34_8', QuestionType.SELECTABLE),
            'colourblind': ('Q35_1', QuestionType.YESNO),
            'colourblind_type': ('Q35_2', QuestionType.CATEGORICAL),
            'SF36-03': ('Q36_1', QuestionType.CATEGORICAL, ['Yes, limited a lot', 'Yes, limited a little', 'No, not limited at all']),
            'SF36-04': ('Q36_2', QuestionType.CATEGORICAL, ['Yes, limited a lot', 'Yes, limited a little', 'No, not limited at all']),
            'SF36-05': ('Q36_3', QuestionType.CATEGORICAL, ['Yes, limited a lot', 'Yes, limited a little', 'No, not limited at all']),
            'SF36-06': ('Q36_4', QuestionType.CATEGORICAL, ['Yes, limited a lot', 'Yes, limited a little', 'No, not limited at all']),
            'SF36-07': ('Q36_5', QuestionType.CATEGORICAL, ['Yes, limited a lot', 'Yes, limited a little', 'No, not limited at all']),
            'SF36-08': ('Q36_6', QuestionType.CATEGORICAL, ['Yes, limited a lot', 'Yes, limited a little', 'No, not limited at all']),
            'SF36-09': ('Q36_7', QuestionType.CATEGORICAL, ['Yes, limited a lot', 'Yes, limited a little', 'No, not limited at all']),
            'SF36-10': ('Q36_8', QuestionType.CATEGORICAL, ['Yes, limited a lot', 'Yes, limited a little', 'No, not limited at all']),
            'SF36-11': ('Q36_9', QuestionType.CATEGORICAL, ['Yes, limited a lot', 'Yes, limited a little', 'No, not limited at all']),
            'SF36-12': ('Q36_10', QuestionType.CATEGORICAL, ['Yes, limited a lot', 'Yes, limited a little', 'No, not limited at all']),
            'SF36-13': ('Q37_1', QuestionType.CATEGORICAL, ['Yes', 'No']),
            'SF36-14': ('Q37_2', QuestionType.CATEGORICAL, ['Yes', 'No']),
            'SF36-15': ('Q37_3', QuestionType.CATEGORICAL, ['Yes', 'No']),
            'SF36-16': ('Q37_4', QuestionType.CATEGORICAL, ['Yes', 'No']),
            'SF36-17': ('Q38_1', QuestionType.CATEGORICAL, ['Yes', 'No']),
            'SF36-18': ('Q38_2', QuestionType.CATEGORICAL, ['Yes', 'No']),
            'SF36-19': ('Q38_3', QuestionType.CATEGORICAL, ['Yes', 'No']),
            'SF36-23': ('Q40_1', QuestionType.CATEGORICAL, ['None of the time', 'A little of the time', 'Some of the time', 'A good bit of the time', 'Most of the time', 'All of the time']),
            'SF36-27': ('Q40_2', QuestionType.CATEGORICAL, ['None of the time', 'A little of the time', 'Some of the time', 'A good bit of the time', 'Most of the time', 'All of the time']),
            'SF36-29': ('Q40_3', QuestionType.CATEGORICAL, ['All of the time', 'Most of the time', 'A good bit of the time', 'Some of the time', 'A little of the time', 'None of the time']), 
            'SF36-31': ('Q40_4', QuestionType.CATEGORICAL, ['All of the time', 'Most of the time', 'A good bit of the time', 'Some of the time', 'A little of the time', 'None of the time']),
            'SF36-21': ('Q41', QuestionType.CATEGORICAL, ['Very severe', 'Severe', 'Moderate', 'Very mild', 'None']),
            'SF36-22': ('Q42', QuestionType.CATEGORICAL, ['Extremely', 'Quite a bit', 'Moderately', 'A little bit', 'Not at all']),
            'baseline_functioning': ('Q43', QuestionType.CATEGORICAL, ['No', 'Yes']),
            'subjective_memory': ('Q44', QuestionType.CATEGORICAL, ['Miserable', 'Poor', 'Less poor', 'Good', 'Excellent'])
        }

    @property
    def symptom_cols(self):
        return [
            'Q2_cough', 'Q2_fever', 'Q2_difficulty_breathing',
            'Q2_pneumonia', 'Q2_loss_of_smell'
        ]

    @property
    def n_WHO_cats(self):
        return 7

    @property
    def WHO_cats(self):
        return [f"WHO_{i}" for i in range(self.n_WHO_cats)]
    
    @property
    def SF36_map(self):
        return {
            'SF36_physical_functioning': np.arange(3, 13),
            'SF36_role_limitations_physical': [13, 14, 15, 16],
            'SF36_role_limitations_emotional': [17, 18, 19],
            'SF36_energy_fatigue': [23, 27, 29, 31],
            'SF36_pain': [21, 22]
        }

    @property
    def SF36_scales(self):
        return list(self.SF36_map.keys())

    def preprocess(self):
        q_fn = self.most_recent_csv
        q_df = pd.read_csv(q_fn)
        q_df = q_df.iloc[2:, :]

        q_df['Q4'] = q_df[['Q4_1', 'Q4_2']].apply(extract_date, axis=1)
        q_df = q_df[self.original_items]

        q_df = q_df.rename(columns=self.initial_rename_map)
        q_df = q_df.set_index('user')
        
        q_df = self._convert_columns(q_df)
        
        # Score the PHQ-2 and GAD-2
        q_df['PHQ2'] = q_df['PHQ2-1'].cat.codes + q_df['PHQ2-2'].cat.codes
        q_df['GAD2'] = q_df['GAD2-1'].cat.codes + q_df['GAD2-2'].cat.codes

        q_df = q_df[q_df.start_date >= self._study.start_date(tz_aware=False)]
        q_df = q_df[q_df.duration > self._DURATION_THRESHOLD]
        q_df = q_df[q_df.progress > self._PROGRESS_THRESHOLD]

        # FOR EACH DUPLICATED QUESIONNAIRE, KEEP THE ONE THAT HAS HIGHEST PROGRESS,
        # BREAKING TIES BY KEEPING THE 1ST ONE (default numpy behaviour).
        q_df['keep'] = True
        for qi, qu in q_df[q_df.index.duplicated(keep=False)].groupby(['user']):
            qu['keep'] = False
            qu.iloc[qu['progress'].argmax(), -1] = True
            q_df.loc[qi, :] = qu
        q_df = q_df[q_df.keep]
        print(f"Drop Duplicates, N={q_df.shape}")

        # CALCULATE WHO COVID SEVERITY
        q_df['symptoms'] = q_df[self.symptom_cols].any(axis=1)

        q_df['WHO_0'] = (q_df['symptoms'] == False) & \
                        (q_df['daily_routine'] == 'Yes') & \
                        (q_df['hospital_stay'].isin(['No', np.NaN]))
        q_df['WHO_1'] = (q_df['symptoms'] == True) & \
                        (q_df['hospital_stay'].isin(['No', np.NaN])) & \
                        (q_df['daily_routine'] == 'Yes')
        q_df['WHO_2'] = (q_df['hospital_stay'].isin(['No', np.NaN])) & \
                        (q_df['daily_routine'] == 'No')
        q_df['WHO_3'] = (q_df['hospital_stay'] == "Yes") & \
                        (q_df['supplemental_O2_hospital'] == 'No') & \
                        (q_df['ICU_stay'] == 'No')
        q_df['WHO_4'] = (q_df['hospital_stay'] == "Yes") & \
                        (q_df['supplemental_O2_hospital'] == 'Yes') & \
                        (q_df['ICU_stay'] == 'No')
        q_df['WHO_5'] = (q_df['hospital_stay'] == "Yes") & \
                        (q_df['ICU_stay'] == 'Yes') & \
                        (q_df['ventilator'] == 'No')
        q_df['WHO_6'] = (q_df['hospital_stay'] == "Yes") & \
                        (q_df['ICU_stay'] == 'Yes') & \
                        (q_df['ventilator'] == 'Yes')

        q_df['WHOi'] = np.dot(q_df[self.WHO_cats].astype('int32'), np.arange(0,self.n_WHO_cats))
        q_df['WHO']  = q_df['WHOi'].astype('category')
        q_df['WHO']  = q_df['WHO'].cat.rename_categories(self.WHO_cats)
        q_df['WHOc'] = pd.cut(q_df.WHOi, [-1, 0, 1, 2, 7], labels=['0', '1', '2', '3+'])

        print(q_df[self.WHO_cats].sum())

        # Recode function for SF36 columns
        recode = lambda x: x.cat.codes * (100/(len(x.cat.categories)-1))
        for scale, qs in self.SF36_map.items():
            cols = [f"SF36-{i:02d}" for i in qs]
            vals = q_df[cols].apply(recode)
            vals[vals<0] = np.nan
            q_df[scale] = vals.mean(axis=1)

        q_df['days_since_test'] = (q_df.start_date - q_df.most_recent_test).apply(lambda x: x.days)

        return q_df

class MetaCovidCognition(MetaCBSData):
    def __init__(self, name, base, ns):
        self._questionnaire = CovidCogQuestionnaire(self,
            'COVID_Cognition_Wild_etal_2021_questionnaire_data_2021-04-15.csv')

        self._score_data = ScoreData(self,
            'COVID_Cognition_Wild_etal_2021_CBS_task_data_2021-04-15.csv')

    @property
    def questionnaire(self):
        return self._questionnaire

    @property
    def score_data(self):
        return self._score_data.data

class CovidCognition(object, metaclass=MetaCovidCognition):
    pass