#%% [markdown]
# # Seeing through brain fog: disentangling the cognitive, physical, and mental health sequalae of COVID-19. 
# ### Conor J. Wild*, Loretta Norton, David K. Menon, David A. Ripsman, Richard H. Swartz, & Adrian M. Owen
# #### Corresponding Author: cwild@uwo.ca, conorwild@gmail.com

#%%
# Standard libraries
from IPython.core.display import display_svg
import pandas as pd
import numpy as np

from scipy.linalg import pinv
from scipy.stats import probplot # for QQ-plots
import statsmodels.formula.api as smf

# For data transformations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import \
	StandardScaler, \
	OneHotEncoder, \
	MinMaxScaler, \
	PowerTransformer
from sklearn.compose import ColumnTransformer

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import \
	calculate_bartlett_sphericity, \
	calculate_kmo

# Custom tools for working with CBS data.
# Non-vital stuff like names of tests, etc.
import covid_cognition.lib_cbs as cbs		

# Custom functions and helpers for doing stats, plots, etc.
# These are located alongside this script.
import covid_cognition.lib_stats as ws
import covid_cognition.lib_plots as wp
import covid_cognition.lib_colours as wc
from covid_cognition.lib_chord import chord_plot
from covid_cognition.lib_utils import \
	report_N, remove_unused_categories, set_column_names, \
	save_and_display_figure, save_and_display_table, tuckersCC

# Helper packages that load the study data.
from cbs_data.covid_cognition import CovidCognition as CC
from cbs_data.control_study import ControlStudy as CS # this is the CTRL dataset

# Plotly for plotting
import plotly
import plotly.express as px

from IPython.display import SVG, display

# Display options for in this notebook
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
np.set_printoptions(precision=3)

idx = pd.IndexSlice

#%% [markdown]
# ## Control (Pre-Pandemic) Dataset
# ### Load & Preprocess Data
#%% 
# Loads the control dataset from the (private) SS library
Yctrl = CS.score_data

# List columns corresponding to "timing" (RT) features
tf  = cbs.timing_features(exclude=['spatial_planning']) # SP does not have one
tf_ = cbs.abbrev_features(tf)		# Abbreviated name

# List of columns corresponding to score features used in domain score calcs.
df  = cbs.domain_feature_list()
df_ = cbs.abbrev_features(df)

# A list of "all" available score features
af  = list(Yctrl.columns)
af_ = cbs.abbrev_features(af)
				
# From Hampshire et al. 2012, and Wild et al. These are the variables that are
# related to performance on these cognitive tasks.
Xcovar = ['age', 'sex', 'post_secondary', 'SES', 'exercise',
	'nicotine', 'alcohol', 'cannabis', 'stimulants', 'depressants']

# Loads the control dataset (Sleep Study, 2017)
print("\nControl Scores:")
Yctrl = (Yctrl
	.pipe(set_column_names, af_)
	.reset_index('device_type')
	.pipe(report_N, "initial dataset", reset_count=True)
	.query('~(device_type in ["BOT", "CONSOLE", "MOBILE"])')
	.pipe(report_N, "drop unsupported devices")
	.reset_index().astype({'user': str})
	.set_index('user')
)

# Loads and organises the control questionnaire dataset
# Have to rename a few columns to match them up to the new study data
print("\nControl Questionnaires:")
Qctrl = (CS
	.questionnaire.data
	.reset_index().astype({'user': str})
	.set_index('user')
	.rename(columns={'SES_growing_up': 'SES'})
	.assign(post_secondary = lambda x: x.education >= "Bachelor's Degree")
	.assign(nicotine = lambda x: x.cigarettes_per_day > 0)
	.assign(alcohol = lambda x: x.alcohol_per_week > 14)
	.assign(exercise = lambda x: x.exercise_freq >= 'Once or twice a week')
	.pipe(report_N, "initial dataset", reset_count=True)
)

# Join the test scores (Yctrl) and the questionnaire data (Qctrl), then
# filter score columns to remove outliers (6 then 4 stdevs)
print("\nControl Dataset:")
Zctrl = (Yctrl
	.join(Qctrl[Xcovar], how='inner')
	.pipe(report_N, 'join datasets', reset_count=True)
	.query('(age >= 18) & (age <= 100)')
	.query('sex in ["Male", "Female"]')
	.pipe(report_N, 'filter age')
	.pipe(report_N, 'English only')
	.dropna(subset=Xcovar+af_)
	.pipe(report_N, 'drop missing data')
	.pipe(ws.filter_df, subset=af_, sds=[6], drop=True)
	.pipe(report_N, '6 SD filter')
	.pipe(ws.filter_df, subset=af_, sds=[4], drop=True)
	.pipe(report_N, '4 SD filter')
	.dropna()
	.pipe(remove_unused_categories)
	.pipe(report_N, 'final')
)

# We'll use these in a table.
Ystats_ctrl  = Zctrl[af_].agg(['mean', 'std'])

# Calculate mean and stdev for the score features, and remember them so we can 
# later apply these transformations to the test (CC) dataset.
Ytfm = Pipeline(steps=[
	('center', StandardScaler(with_mean=True, with_std=True)),
	('yeo', PowerTransformer(method='yeo-johnson'))
]).fit(Zctrl[af_].values)

# Z-scores the control test scores (all features)
Zctrl[af_] = Ytfm.transform(Zctrl[af_].values)

#%% [markdown]
# ### Factor Analysis (PCA) of CBS Test Scores (Control Group)

#%%
# Perform a PCA of the 12 primary CBS measures specifying 3 components and a
# varimax rotation. These choices are based on previous work with these tests:
# Hampshire et al (2012), Wild et al (2018).
pca_dmn_ctrl = FactorAnalyzer(
	method='principal',
	n_factors=3, 
	rotation='varimax').fit(Zctrl[df_])

# I know the scores turn out in this order....
domains = ['STM', 'reasoning', 'verbal']

# Build and collect dataframes that will be used for figures and table
# generation. First, the loadings.
loadings_ctrl = pd.DataFrame(
	pca_dmn_ctrl.loadings_, index=cbs.test_names(), columns=domains)

# Pairwise correlations between test scores
var_corrs = pd.DataFrame(
	pca_dmn_ctrl.corr_, 
	index=cbs.test_names(), columns=cbs.test_names())

# Eigenvalues of the components
eigen_values = pd.DataFrame(
	pca_dmn_ctrl.get_eigenvalues()[0][0:3], 
	index=domains, columns=['eigenvalues']).T

# Percentage variabnce explained by each component
pct_variance = pd.DataFrame(
	pca_dmn_ctrl.get_factor_variance()[1]*100, 
	index=domains, columns=['% variance']).T

# Generates and displays the chord plot to visualize the factors
fig_1a = chord_plot(
	loadings_ctrl.copy(), var_corrs.copy(), 
	cscale_name='Picnic', width=700, height=350, threshold=0.0)

save_and_display_figure(fig_1a, 'Figure_1A')

# Generate a table of task to composite score loadings
pca_table_ctrl = (pd
	.concat([loadings_ctrl, eigen_values, pct_variance], axis=0)
	.join(Ystats_ctrl[df_]
		.T.rename(index={r[0]: r[1] for r in zip(df_, cbs.test_names())}))
	.loc[:, ['mean', 'std']+domains]
)

pca_table_ctrl.to_csv('./outputs/tables/Table_S3.csv')

pca_table_ctrl

#%% [markdown]
# ### Control Sample: Calculate Composite Cognitive Scores
#%% 
# Calculates the 3 cognitive domain scores from the fitted PCA model
Zctrl[domains] = pca_dmn_ctrl.transform(Zctrl[df_])

# Measure of processing speed: take the 1st Principal Component across 
# timing-related features (the list of tf_), derived from ctrl group data.
pca_spd_ctrl = FactorAnalyzer(
	method='principal', 
	n_factors=1,
	rotation=None).fit(Zctrl[tf_])

# Force it to go the right away around, so higher scores are better
if pca_spd_ctrl.loadings_.mean() > 0:
	pca_spd_ctrl.loadings_ *= -1

Zctrl['processing_speed'] = pca_spd_ctrl.transform(Zctrl[tf_])

# Overall measure across CBS battery: the average of all 12 task z-scores,
# then rescale to have M = 0.0, SD = 1.0
Zctrl['overall'] = Zctrl[df_].mean(axis=1)
overall_tfm = StandardScaler(with_mean=True, with_std=True).fit(Zctrl[['overall']])
Zctrl['overall'] = overall_tfm.transform(Zctrl[['overall']])

comp_scores = cbs.DOMAIN_NAMES+['processing_speed', 'overall']
test_scores = df_

Yvar = test_scores+comp_scores

#%% [markdown]
# ## Relationship between covariate (nuisnace variables) and cognitive scores.
# Here, we'll do an an exploratory analysis to see which of our common 
# covariates have a relationship with cognitive scores -- in the control group.
# Simply, we will run a linear regression model for each test and composite
# score that includes all of these variables as predictors.
#%%
# Temporary duplication of the control dataset, because we modify the age 
# mean-centre the age variable for this exploratory analysis.
Zctrl_ = Zctrl.copy()
Zctrl_.age -= Zctrl_.age.mean()

covar_model = ws.build_model_expression(Xcovar)
print(f"Regression formula: {covar_model}")

# Run the linear regression models - this does one for each variable specified
# in Yvar, which is a list of all test and composite scores.
r, _ = ws.regression_analyses(covar_model, Yvar, Zctrl_)

# Generate a summary figure to show significant (p < 0.05, uncorrected) 
# relationships between covariates and cognitive scores (just exploratory)
f = wp.create_stats_figure(
	r.drop('Intercept', level='contrast', axis=0),
	'tstat', 'p', stat_range=[-10, 10], diverging=5, vertline=5)

#%% 
# Estimate and save the betas so we can correct the COVID+ data for the same
# variables. Note that we model age using linear+quadratic terms.
from sklearn.preprocessing import PolynomialFeatures

# Transform covariates (X) variables specifying:
#	- age, as a linear and quadratic term (see age_tfm)
#	- level of education, as a binary variable with 1 = False
#	- SES growing up, as binary variable with 1 indicating  "Below poverty level"
# 	- nicotine, as binary with 1 indicating smokes > 0 cigarettes per day
# 	- exercise, as binary with 1 indicating at least once per week
# 	- cannabis, as binary with 1 indicating > 1 unit per day
# 	- stimulants, as binary with 1 indicating at least once in the last 4 wks
# 	- depressants, as binary with 1 indicating at least once in the last 4 wks

age_q_tfm = Pipeline(steps=[
	('center', StandardScaler(with_mean=True, with_std=False)),
	('poly', PolynomialFeatures(degree=2, include_bias=False)),
])

x_tfms = [
	('cont',  age_q_tfm, ['age']),
	('c_sex', OneHotEncoder(drop=['Female']), ['sex']),
	('c_edu', OneHotEncoder(drop=[False]), ['post_secondary']),
	('c_ses', OneHotEncoder(drop=['At or above poverty level']), ['SES']),
	('c_oth', OneHotEncoder(drop=[False, False, True, False, False, False]),
		['nicotine', 'alcohol', 'exercise', 'cannabis', 'stimulants', 'depressants']),
]

Xtfm = ColumnTransformer(x_tfms, sparse_threshold=0).fit(Zctrl[Xcovar])

# Alternate transformer that just mean-centres age.
age_l_tfm = StandardScaler(with_mean=True, with_std=False)
x_tfms_alt = x_tfms.copy()
x_tfms_alt[0] = ('cont',  age_l_tfm, ['age'])

# Create and fit the alternate covariate transformer
Xtfm_ = ColumnTransformer(x_tfms_alt, sparse_threshold=0).fit(Zctrl[Xcovar])

# Transform the covariate columns into a numeric matrix
# "X" denotes a design matrix, "Y" denotes the dependent variables (scores)
Xctrl = Xtfm.transform(Zctrl[Xcovar])

# Add an intercept
Xctrl = np.c_[Xctrl, np.ones(Xctrl.shape[0])]

# Solve for the the linear regression parameters. 
# We'll use these to correct the scores from both groups by subtracting the 
# expected effects (i.e., regressing them out)
Bctrl = np.dot(pinv(Xctrl), Zctrl[Yvar])

#%% [markdown]
# ## COVID+ Sample
# ### Load & Process COVID Cognition Data
#%%
# Placeholder variables instead of typing these every time
mhvars   = ['GAD2', 'PHQ2']
subjvars = ['subjective_memory', 'baseline_functioning']
sf36vars = list(CC.questionnaire.SF36_map.keys())
severity = ['WHOc']

# Existing conditions related to COVID severity / outcomes, and / or performance
# on the CBS cognitive test battery
pre_exising_conditions = [
	'Q34_diabetes', 'Q34_obesity', 'Q34_hypertension', 'Q34_stroke', 
	'Q34_heart_attack', 'Q34_memory_problem'
]

# Load and process questionnaire data
print("\nCC Questionnaire:")
Qcc = (CC.questionnaire.data
	.rename(columns={'ses': 'SES'})
	.pipe(report_N, 'initial questionnaires', reset_count=True)
	.query('~index.duplicated(keep="first")')
	.pipe(report_N, 'remove duplicates')
	.assign(post_secondary = lambda x: x.education > 'Some university or college, no diploma')
	.assign(subjective_memory = lambda x: x.subjective_memory.cat.codes)
	.assign(pre_existing_condition = lambda x: x[pre_exising_conditions].any(axis=1))
	.assign(nicotine = lambda df: df.nicotine > 0)
	.assign(alcohol = lambda df: df.alcohol > 14)
	.assign(exercise = lambda df: df.exercise_freq >= 'Weekly')
	.assign(cannabis = lambda df: df.cannabis > 0)
	.rename(columns={'drugs_stimulants': 'stimulants', 'drugs_depressants': 'depressants'})
)

# Create an alternative WHO COVID severity variable that is: 0-1, 2, or 3+
Qcc['WHOc'] = (Qcc.WHOc
	.map({'0': '0-1', '1': '0-1', '2': '2', '3+': '3+'})
	.astype("category")
)

# Load and process the COVID+ CBS test scores, dropping any that were
# completed with an unsupported device type.
print("\nCC Scores:")
Ycc = (CC.score_data
	.pipe(set_column_names, af_)
	.reset_index('device_type')
	.groupby('user')
	.nth(0)
	.pipe(report_N, 'initial scores', reset_count=True)
	.query('~(device_type in ["BOT", "CONSOLE", "MOBILE"])')
	.pipe(report_N, 'drop unsupported devices')
)

# Combine score and questionnaire data, then chain them through a pipeline
# to remove datasets based on outliers, etc.
print("\nCC Dataset:")
Zcc = (Ycc
	.join(Qcc, how='left')
	.pipe(report_N, 'join data', reset_count=True)
	.query('(age >= 18)')
	.pipe(report_N, '>= 18 years')
	.query('(age <= 100)')
	.pipe(report_N, '<= 100 years')
	.query('sex in ["Male", "Female"]')
	.pipe(report_N, 'filter age')
	.pipe(ws.filter_df, subset=df_, sds=[6], drop=True)
	.pipe(report_N, '6 SD filter')
	.pipe(ws.filter_df, subset=df_, sds=[4], drop=True)
	.pipe(report_N, '4 SD filter')
	.dropna(subset=af_+Xcovar+mhvars+subjvars+sf36vars+severity)
	.pipe(report_N, 'drop cases with any missing values')
	.query('positive_test == "Yes"')
	.pipe(report_N, 'drop NO to positive test')
	.drop(columns=['positive_test'])
	.dropna(subset=['days_since_test'])
	.pipe(report_N, 'without a test date')
	.pipe(report_N, 'final')
	.pipe(remove_unused_categories)
)

#%% [markdown]
# ## COVID+ Sample
# ### Compare factor structure of cognitives scores to CTRL data/
#%%

# Copy of the original dataframe, because some figures, etc. have different
# codings for variables.
Zcc_ = Zcc.copy()

# Original score means and SDs, for use in a summary table.
Ystats_cc = Zcc_[af_].agg(['mean', 'std'])

# Z-score the test score features w/r/t to this sample (COVID+)
Zcc_[af_] = Pipeline(
	steps=[('center', StandardScaler(with_mean=True, with_std=True)),
		   ('yeo', PowerTransformer(method='yeo-johnson'))]
).fit_transform(Zcc_[af_].values)

# Perform rotated PCA analysis, specifying 3 components like in the CTRL group.
pca_dmn_cc = FactorAnalyzer(
	method='principal',
	n_factors=3, 
	rotation='varimax').fit(Zcc_[df_])

# Build and collect dataframes that will be used for figures and tables...

# Component (factor) loadings
loadings_cc = pd.DataFrame(
	pca_dmn_cc.loadings_, 
	index=cbs.test_names(), columns=domains)

# Pair-wise correlations between variables
var_corrs_cc = pd.DataFrame(
	pca_dmn_cc.corr_, 
	index=cbs.test_names(), columns=cbs.test_names())

# Eigenvalues of the components
eigen_values = pd.DataFrame(
	pca_dmn_cc.get_eigenvalues()[0][0:3], 
	index=domains, columns=['eigenvalues']).T

# % Variance explained by each component
pct_variance = pd.DataFrame(
	pca_dmn_cc.get_factor_variance()[1]*100, 
	index=domains, columns=['% variance']).T

# Create a table of loadings, test means/SDs, eigenvalues, % variance, etc.
pca_table_cc = (pd
	.concat([loadings_cc, eigen_values, pct_variance], axis=0)
	.join(Ystats_cc[df_]
		.T.rename(index={r[0]: r[1] for r in zip(df_, cbs.test_names())}))
	.loc[:, ['mean', 'std']+domains]
)

# Generates and displays the chord plot to visualize the factors
fig_1x = chord_plot(
	loadings_cc.copy(), var_corrs.copy(), 
	cscale_name='Picnic', width=700, height=350, threshold=0.0)

print("COVID+ Factor Structure")
save_and_display_figure(fig_1x, 'Figure_S5b')

print("CTRL Factor Structure")
save_and_display_figure(fig_1a, 'Figure_1A')

pca_table_cc.to_csv('./outputs/tables/Table_S4.csv')
display(pca_table_cc)

# Replicate the "processing speed" score?
pca_spd_cc = FactorAnalyzer(
	method='principal', 
	n_factors=1,
	rotation=None).fit(Zcc[tf_])

# Force it to go the right away around, so higher scores are better
if pca_spd_cc.loadings_.mean() > 0:
	pca_spd_cc.loadings_ *= -1

# Next, compare the similarity of the factor (domain) structures produced from
# each group (COVID+ and CTRL).
trg = pca_dmn_ctrl.loadings_
src = pca_dmn_cc.loadings_

# Calculates the congruency coeffient for each factor twice, 1st without doing
# a procustes transformation, 2nd time with the transformation.
domain_coefs = [tuckersCC(trg, src, do_procrustes = x) for x in [False, True]]

trg = pca_spd_ctrl.loadings_
src = pca_spd_cc.loadings_
speed_coefs = [tuckersCC(trg, src, do_procrustes = x) for x in [False, True]]

factor_similarities = (pd
	.DataFrame(
		np.vstack(domain_coefs),
		index = [False, True], columns = domains)
	.rename_axis('procrustes', axis = 0)
	.assign(processing_speed = np.squeeze(speed_coefs))
)

factor_similarities.to_csv('./outputs/tables/Table_S5.csv')
display(factor_similarities)
#%% TEST CFA

from factor_analyzer import ModelSpecification, ModelSpecificationParser, ConfirmatoryFactorAnalyzer

mm = ModelSpecificationParser.parse_model_specification_from_array(
	Zcc_[df_], loadings_ctrl.values
)

cfa = ConfirmatoryFactorAnalyzer(mm, max_iter=1000).fit(Zcc_[df_].values)
#%%

# Standardizes all score features using the transformer that was fitted on the 
# control sample data (i.e., using the control sample means and SDs, etc.)
# Now, cognitive scores will be relative to the CTRL group.
Zcc[af_] = Ytfm.transform(Zcc[af_])

# Calculate the composite scores
Zcc[domains] = pca_dmn_ctrl.transform(Zcc[df_])
Zcc['processing_speed'] = pca_spd_ctrl.transform(Zcc[tf_])
Zcc['overall'] = overall_tfm.transform(Zcc[df_].mean(axis=1).values.reshape(-1,1))

#%%

Zcc_ = Zcc.copy()
Zcc_[Xcovar] = Xtfm_.transform(Zcc_[Xcovar])
Zcc_['GAD_flag'] = Zcc_['GAD2'] >= 3
Zcc_['PHQ_flag'] = Zcc_['PHQ2'] >= 3

r, _ = ws.regression_analyses("%s ~ age + sex + post_secondary + SES", comp_scores, Zcc_)
r = ws.adjust_pvals(r, adj_across='all', adj_type='fdr_bh')

# Generate a summary figure to show significant (p < 0.05, uncorrected) 
# relationships between covariates and cognitive scores (just exploratory)
f = wp.create_stats_figure(
	r.drop('Intercept', level='contrast', axis=0), 'tstat', 'p_adj', 
	correction='fdr_bh', stat_range=[-10, 10], diverging=True, vertline=2)


Zcc = Zcc.rename(columns={'WHOi': 'WHO_COVID_severity'})

#%%
# Counts of responses for COVID-19 related questions
covid_vars = ['symptoms', 'hospital_stay', 'daily_routine', 'supplemental_O2_hospital', 'ICU_stay', 'ventilator']
q_counts = Zcc[covid_vars].copy()
q_counts.loc[:, 'symptoms'] = q_counts['symptoms'].map({True: "Yes", False: "No"})
q_counts = [q_counts[c].value_counts() for c in covid_vars]
q_counts = pd.concat(q_counts, axis=1).T
q_counts

#%%
# Distributions and frequencies of variables pertaining to socio-
# demographics, other covariates, etc.

# These need to be coded in the control dataset
lang_vars = {l: f"speak_{l.lower()}" for l in ['English', 'French', 'Spanish']}
for l, v in lang_vars.items():
	Qctrl[v] = Qctrl['languages_spoken'].map(
		lambda x: l in x if isinstance(x, str) else False)
lang_vars = list(lang_vars.values())

# Join the two datasets together so we can do .groupby etc.
covar_data = pd.concat(
	[Zctrl[Xcovar].join(Qctrl[['country']+lang_vars]), 
	 Zcc[Xcovar+lang_vars+['country']]], axis=0, 
	names=['dataset'], keys=['Control', 'COVID+'])

grp_counts = (covar_data
	.groupby('dataset')
	.agg('count')
	.iloc[:, 0]
)

catg_vars = pd.concat(
	{ v: (covar_data
			.groupby('dataset')[v]
			.value_counts()
			.to_frame()
			.rename(columns={v: 'count'})
			.assign(pct = lambda x: x['count'] / grp_counts * 100)
			.unstack('dataset')
			.swaplevel(axis=1).sort_index(axis=1)
		) for v in Xcovar[1:] + lang_vars
	}
)
display(catg_vars)

cont_vars = pd.concat(
	{ v: (covar_data
			.groupby('dataset')
			.agg(['mean', 'std'])[[v]]
			.T
			.unstack(1)
		) for v in ['age']
	}
)
display(cont_vars)

top5_countries = (pd.concat(
		{ dataset:
			(covar_data
				.xs(dataset)
				.loc[:, 'country']
				.value_counts(ascending=False)
				.iloc[0:5]
			) for dataset in ['COVID+', 'Control']
		}, names=['dataset'])
	.to_frame()
	.rename(columns={'country': 'count'})
	.assign(pct = lambda x: x['count'] / grp_counts * 100)
)
display(top5_countries)


#%% [markdown]
# ### COVID+ Sample: GAD-2 & PHQ-2 Summary
#%% 
n_GAD_flag = (Zcc.GAD2>=3).sum()
print(f"GAD2 >= 3, N = {n_GAD_flag} ({n_GAD_flag/Zcc.shape[0]*100:.01f}%)")

n_PHQ_flag = (Zcc.PHQ2>=3).sum()
print(f"PHQ2 >= 3, N = {n_PHQ_flag} ({n_PHQ_flag/Zcc.shape[0]*100:.01f}%)")

#%% [markdown]
# ## COVID+ Sample: Health Measures
#%%
# The questionnaire variables we are incorporating into the factor analysis:
fvars = mhvars + subjvars + sf36vars + ['WHO_COVID_severity', 'days_since_test']
fdata = Zcc[fvars].copy().dropna()	# copy it so we don't mess up the original df

var_rename = {
	'baseline_functioning_Yes': 'subj_baseline',
	'baseline_functioning': 'subj_baseline',
	'subjective_memory': 'subj_memory',
	'days_since_test': 'days_since_+test'
}

var_order = sf36vars + mhvars + [
	'subj_memory',
	'subj_baseline',
	'WHO_COVID_severity',
	'days_since_+test']

fdata.WHO_COVID_severity = fdata.WHO_COVID_severity >= 2
fdata_tfm = ColumnTransformer([
	('z', StandardScaler(), sf36vars+['days_since_test']),
	('r', MinMaxScaler(feature_range=(-1,1)), mhvars+['subjective_memory', 'WHO_COVID_severity']),
	('c', OneHotEncoder(drop='first'), ['baseline_functioning']),
]).fit(fdata)

fdata_desc = (fdata
	.assign(baseline_functioning=fdata['baseline_functioning'].cat.codes)
	.agg(['mean', 'std']).T
	.rename(var_rename)
	.loc[var_order, :]
)

fdata0 = (pd
	.DataFrame(
		fdata_tfm.transform(fdata),
		index=fdata.index,
		columns=ws.get_ct_feature_names(fdata_tfm))
	.rename(columns=var_rename)
	.loc[:, var_order]
)

# Reverse GAD-2 and PHQ-2 so they go in the same direction as other measures
# for these plots, and calculations of factor scores.
fdata0[mhvars] *= -1

# Fit the factor model, no rotation at first.
fa1 = FactorAnalyzer(rotation=None).fit(fdata0)

# Determine number of factors, given eigenvalues > 1.0
nf = (fa1.get_eigenvalues()[1]>1).sum()
fnames = [f"F{i+1}" for i in range(nf)] # name them F1 ... FN for now

# Bartlett; p-value should be 0 (statistically sig.)
chi_square_value,p_value=calculate_bartlett_sphericity(fdata0)
print(f"chi2 = {chi_square_value:.02f}, p = {p_value:.04f}")

# KMO; Value should be > 0.6
kmo_all,kmo_model=calculate_kmo(fdata0)
print(f"KMO = {kmo_model:.02f}")

# Now do the factor analysis retaining only the the number of factors,
# and do a rotation (Varimax)
rotation = "varimax"
fa = FactorAnalyzer(nf, rotation=rotation)
fa = fa.fit(fdata0)

# Calculate Factor scores using the fitted factor model
# - one new column per factor score (named F1 ... FN)
# - rescale to have SD = 1.0
Zcc[fnames] = np.nan
Zcc.loc[fdata0.index, fnames] = (
	StandardScaler()
		.fit_transform(
			fa.transform(fdata0)
		)
)

# Loadings table
loadings = pd.DataFrame(
	fa.loadings_, index=fdata0.columns, columns=fnames)

# Eigenvalues of each factor (for table)
eigen_values = pd.DataFrame(
	fa.get_eigenvalues()[0][0:nf], index=fnames, columns=['eigenvalues']).T

# Percentage variance explained by each factor (for table)
pct_variance = pd.DataFrame(
	fa.get_factor_variance()[1]*100, index=fnames, columns=['% variance']).T

# Combine to generate the 
table = (pd
	.concat([loadings, eigen_values, pct_variance], axis=0)
	.join(fdata_desc)
	.loc[:, ['mean', 'std']+fnames]
)

table.to_csv(f"./outputs/tables/Table_S2.csv")

# Visualize the factor structure with a chord plot
f = chord_plot(
		loadings, fdata0[var_order].corr(), 
		width=800, height=350, cscale_name='Picnic', threshold=0.0)

save_and_display_figure(f, 'Figure_1B')

# Correlations between health measures (same as shown in the above chord plot)
f = wp.correlogram(fdata0, subset=var_order, mask_diag=True)
save_and_display_figure(f, 'Figure_S2')

#%%
# Concatenate the COVID+ and Control datasets for subsequent comparisons 
# between groups
Zall = pd.concat([Zcc, Zctrl], axis=0, keys=['CC', 'CTRL'], names=['study'])

#%% 
# Defines columns that will be used in various stats tables
comp_columns = ['dR2', 'f2', 'BF10']					# model comparison stats
regr_columns = ['B', 'tstat', 'df', 'p_adj', 'CI']		# regression parameters
ttest_columns = ['diff', 'tstat', 'df', 'p_adj', 'CI', 'BF10'] # t-test stats
column_order = regr_columns + comp_columns

#%% [markdown]
# ## Relationships Between Factor Scores and Covariates
#%%
Zcc_ = Zcc.copy()
Zcc[Xcovar] = Xtfm_.transform(Zcc[Xcovar])

# Build and estimate regression models for each health factor score
# (fnames) using the expression: '~ age + sex + post_secondary + SES'.
r0_regressions, _ = ws.regression_analyses(
	'%s ~ age + sex + post_secondary + SES', fnames, Zcc, n_comparisons=8)

# Then, we perform model comparisons where the null or reduced model (h0) does
# not contain the variable of interest, whereas the alternative model (h1) DOES.
# This function returns Bayes factors, effect sizes, likelihood ratios, etc.
r0_comparisons = ws.compare_models([
	{ 'name': 'age',
		'h0': '%s ~ sex + post_secondary + SES', 
		'h1': '%s ~ age + sex + post_secondary + SES' }, 
	{ 'name': 'sex',
		'h0': '%s ~ age + post_secondary + SES', 
		'h1': '%s ~ age + sex + post_secondary + SES' },
	{ 'name': 'post_secondary',
		'h0': '%s ~ age + sex + SES', 
		'h1': '%s ~ age + sex + post_secondary + SES' },
	{ 'name': 'SES',
		'h0': '%s ~ age + sex + post_secondary', 
		'h1': '%s ~ age + sex + post_secondary + SES' }],

	Zcc, fnames, smf.ols, n_comparisons=8)

# Combine and reorganize the results tables
r0_table = (r0_regressions
	.drop('Intercept', level='contrast')
	.join(r0_comparisons.loc[:, comp_columns])
	.rename(columns={'value': 'B'})
	.loc[:, column_order]
)

# Display stats figure (here)
f1_ts = wp.create_stats_figure(
	r0_table, 'tstat', 'p_adj', diverging=True, vertline=None, 
	correction='bonferroni')

f1_bf = wp.create_bayes_factors_figure(r0_table, cell_scale=0.6)

# Save / show the stats table
save_and_display_table(r0_table, 'Table_S6')

#%% [markdown]
# ### Plots of Factors Scores vs. Covariates
#%% 
import plotly.graph_objects as go
bwmap = wc.create_mpl_cmap(plotly.colors.sequential.gray, alpha=1.0)

age_edges = np.array([18, 30, 60, 100])
age_bins = ['18-30', '30-60', '60+']
Zcc_['age_bin'] = pd.cut(Zcc_['age'], age_edges, labels=age_bins)
Zcc_['post_secondary'] = Zcc_['post_secondary'].map({True: 'Yes', False: 'No'})

_, age_bins = pd.qcut(Zcc_['age'], 3, labels=False, retbins=True, precision=0)
age_bins = [f"{int(x)}-{int(age_bins[i+1])}" for i, x in enumerate(age_bins[:-1])]
Zcc_['age_bin'] = pd.qcut(Zcc_['age'], 3, labels=age_bins)

wp.rc_title.update(text = 'A) Age (years)')
age_plot = wp.raincloud_plot(
	Zcc_, ['F1', 'F2'], 'age_bin', grp_order=age_bins,
	vio_args = {'spanmode': 'soft'},
)
age_plot

wp.rc_layout.update(boxgroupgap = 0.4, width = 280)
wp.rc_yaxis.update(showticklabels = False, title = None)
wp.rc_layout['margin'].update(l=30)
wp.rc_title.update(text = 'B) Post Secondary')

edu = wp.raincloud_plot(
	Zcc_, ['F1', 'F2'], 'post_secondary',
	vio_args = {'spanmode': 'soft'}, sym_offset = 2
)
edu

wp.rc_title.update(text = 'C) Sex')
sex = wp.raincloud_plot(
	Zcc_, ['F1', 'F2'], 'sex', 
	vio_args = {'spanmode': 'soft'}, sym_offset = 5
)
sex


Zcc2 = Zcc_.copy()
Zcc_['SES'] = Zcc_['SES'].cat.rename_categories(
	{'At or above poverty level': 'At / Above', 'Below poverty level': 'Below'}
)

wp.rc_title.update(text = 'D) SES (poverty level)')
ses = wp.raincloud_plot(
	Zcc_, ['F1', 'F2'], 'SES',
	vio_args = {'spanmode': 'soft'}, sym_offset = 7,
)
ses
#%%

r, _ = ws.regression_analyses("%s ~ F1 + F2", comp_scores, Zcc_)
# r = ws.adjust_pvals(r, adj_across='all', adj_type='fdr_bh')

# Generate a summary figure to show significant (p < 0.05, uncorrected) 
# relationships between covariates and cognitive scores (just exploratory)
f = wp.create_stats_figure(
	r.drop('Intercept', level='contrast', axis=0), 'tstat', 'p_adj', 
	correction='fdr_bh', stat_range=[-10, 10], diverging=True, vertline=2)

# Add an intercept
Xfcc = np.c_[Zcc[fnames], np.ones(Zcc.shape[0])]

# Solve for the the linear regression parameters. 
# We'll use these to correct the scores from both groups by subtracting the 
# expected effects (i.e., regressing them out)
Bfcc = np.dot(pinv(Xfcc), Zcc[fnames])

Zcc2 = Zcc.copy()
Zcc2[fnames] -= Xfcc.dot(Bfcc)

# Let's run a linear regression that predicts each composite cognitive score
# from each variable in the list. Then, adjust all the coefficient p-values
# (5 x # of vars) using false discovery rate.
r = [ws.regression_analyses(f"%s ~ {v}", comp_scores, Zcc) for v in fnames]
r = [r[0].drop('Intercept', level='contrast') for r in r]
r = pd.concat(r, axis=0)
r = ws.adjust_pvals(r, adj_across='all', adj_type='fdr_bh')
r
# # Here we'll do the Bayesian model comparisons, comparing a model with the 
# # single variable (~ v) to an intercept-only model (~ 1).
# models = [{'name': v, 'h0': f"%s ~ 1", 'h1': f"%s ~ {v}"} for v in vars_]
# rEXPL_comparisons = ws.compare_models(models, Zcc, comp_scores, smf.ols)

#%% [markdown]
# ## COVID+ vs. Control - Comparisons of Cognitive Performance

#%%

# Before comparing the groups, we will correct for all the estimated
# effects of those confounding variables.

# Adjust the CTRL group for the covariates...
Yctrl_hat = Xctrl.dot(Bctrl)
Zctrl[Yvar] -= Yctrl_hat

# Adjust the COVID+ group for the covariates...
Xcc = Xtfm.transform(Zcc[Xcovar])
Xcc = np.c_[Xcc, np.ones(Xcc.shape[0])]
Ycc_hat = Xcc.dot(Bctrl)
Zcc[Yvar] -= Ycc_hat

#%% 
# First, perform the t-tests comparing each composite score between groups,
# where group is specified by the 'study' column. Manually specifying 15
# comparisons because there are 5 DVs, but we are including the tests of 
# regression parameters from following analyses.
r1_ttests = (ws
	.two_sample_ttests('study', comp_scores, Zall, n_comparisons=15)
	.droplevel('contrast')
)

save_and_display_table(r1_ttests[ttest_columns], 'Table_3')

#%% [markdown]
# ## COVID+ vs. Control - Plots of Mean Scores
# - Divide the COVID+ sample into tercile bins (each has 33%) on each factor
# - Remember, Y=0.0 is the control sample mean
#%%
f1_edges = np.array([-np.Inf, *Zcc['F1'].quantile(q=np.array([1,2])/3).values, np.Inf])
f2_edges = np.array([-np.Inf, *Zcc['F2'].quantile(q=np.array([1,2])/3).values, np.Inf])
labels = ["worse", "average", "better"]
Zcc['F1_bin'] = pd.cut(Zcc['F1'], bins=f1_edges, labels=labels)
Zcc['F2_bin'] = pd.cut(Zcc['F2'], bins=f2_edges, labels=labels)

bar_args = {
	'range_y': [-0.7, 0.35],
	'labels': {'mean': 'SD Units'},
}

layout_args = {
	'xaxis': {
		'title': None, 
	},
	'yaxis': {
		'tickmode': 'array',
		'tickvals': np.arange(-0.6, 0.7, 0.2).round(1),
	},
	'width': 400, 'height': 275,
	'margin': {'l': 70, 'b': 75, 't': 20, 'r': 20},
	'legend_traceorder': 'reversed',
	'legend': {
		'title': "F1 (physical health)", 'orientation': 'v',
		'yanchor': 'top', 'xanchor': 'left', 'x': 1.02, 'y': 1.0,
	},
}
grayscale_map = wc.to_RGB_255(bwmap(np.linspace(0.3, 1.0, 3)))

f3a, _ = wp.means_plot(
	Zcc, comp_scores, 'score', group='F1_bin',
	group_color_sequence=grayscale_map,
	bar_args=bar_args, layout_args=layout_args 
)
save_and_display_figure(f3a, 'Figure_3A')

layout_args['legend']['title'] = 'F2 (mental health)'
f3b, _ = wp.means_plot(
	Zcc, comp_scores, 'score', group='F2_bin',
	group_color_sequence=grayscale_map,
	bar_args=bar_args, layout_args=layout_args 
)
save_and_display_figure(f3b, 'Figure_3B')

#%% [markdown]
# ## COVID+ Binned Groups vs. Controls

#%% 
# Compare groups in F1 bins to controls
r3a_ttests = {}
for grp, grp_data in Zcc.groupby('F1_bin'):
	Zall = pd.concat([grp_data, Zctrl], keys=['CC', 'SS'], names=['study'])
	r3a_ttests[grp] = (ws
		.two_sample_ttests('study', comp_scores, Zall, n_comparisons=15)
		.droplevel('contrast')
	)
r3a_ttests = pd.concat(r3a_ttests.values(), axis=0, keys=r3a_ttests.keys(), names=['F1_bin'])

save_and_display_table(r3a_ttests[ttest_columns], 'Table_S7')

# Compare groups in F2 bins to controls
r3b_ttests = {}
for grp, grp_data in Zcc.groupby('F2_bin'):
	Zall = pd.concat([grp_data, Zctrl], keys=['CC', 'SS'], names=['study'])
	r3b_ttests[grp] = (ws
		.two_sample_ttests('study', comp_scores, Zall, n_comparisons=15)
		.droplevel('contrast')
	)
r3b_ttests = pd.concat(r3b_ttests.values(), axis=0, keys=r3b_ttests.keys(), names=['F2_bin'])

save_and_display_table(r3b_ttests[ttest_columns], 'Table_S7b')

#%% [markdown]
# ## COVID+ Sample - Predict Cognitive Performance From Health Factors
#%%
# Build and estimate regression models for each composite cognitive score

full_model = ws.build_model_expression(fnames)
print(f"Model expression: {full_model}")

# This function constructs and estimates a linear regression model for each of
# the DVs (the 5 composite scores) using the "full model" formula, which
# in this case is just ~ F1 + F2.
r1_regressions, r1_models = ws.regression_analyses(
		full_model, comp_scores, Zcc, n_comparisons=15)

# Then, we perform model comparisons where the null or reduced model (h0) does
# not contain the variable of interest, whereas the alternative model (h1) DOES.
# This function returns Bayes factors, effect sizes, likelihood ratios, etc.
r1_comparisons = ws.compare_models([
	{'name': 'F1',
		'h0': ws.build_model_expression("F2"), # Null hypothesis doesn't include F1
		'h1': full_model}, 
	{'name': 'F2',
		'h0': ws.build_model_expression("F1"), # Null hypothesis doesn't include F2
		'h1': full_model}
	], Zcc, comp_scores, smf.ols, n_comparisons=15)

# Extract and combine the various statistics from the above regression
# analyses and build the table of results. We are only keeping rows for the 
# regressors of interest (not going to keep parameters etc. for nuisance vars)
res1_ = (r1_regressions
	.join(r1_comparisons.loc[:, comp_columns])
	.rename(columns={'value': 'B'})
	.loc[r1_comparisons.index, column_order]
)

save_and_display_table(res1_, 'Table_4')

# Generate figures to summarize these results...
plot_cols = ['tstat', 'p_adj', 'BF10']
renamer = {'group': 'COVID+ vs CTRL'}
res1_fig = (
	pd.concat(
		[pd.concat([r1_ttests[plot_cols]], keys=['group'], names=['contrast']).swaplevel(),
		 res1_[plot_cols]], axis=0
	)
	.loc[idx[comp_scores, :], :]
	.rename(renamer)
)

# Matrix of t-tstatistics
f1_ts = wp.create_stats_figure(
	res1_fig, 'tstat', 'p_adj', diverging=True, vertline=2,
	correction='bonferroni', stat_range=[-6.3, 6.3])

# Matrix of BFs
f1_bf = wp.create_bayes_factors_figure(res1_fig, suppress_h0=True, vertline=2)

f1_ts.savefig('./outputs/images/Figure_4A.svg')
f1_bf.savefig('./outputs/images/Figure_5B.svg')

#%% [markdown]
# ### QQ-Plots
# - Check that residuals from the regression models are normally distributed!
#%%
qq_nms, qq_res = np.empty([3, 2], dtype='object'), np.empty([3,2], dtype='object')
for im, m in enumerate(r1_models):
	ir, ic = np.unravel_index(im, [3,2])
	qq_res[ir, ic] = probplot(m.resid, dist="norm")
	qq_nms[ir, ic] = m.model.endog_names

qq_f = wp.qq_plots(qq_res, qq_nms,
	layout_args = {'width': 400, 'height': 400}
)

save_and_display_figure(qq_f, 'Figure_S3')

#%% [markdown]
# ## Supplementary Analysis
#  
# Let's include a bnuch of covariates in the regression models to see if they 
# explain the relationship(s) between F1 and cognitive scores. We'll include all
# the coviariates of no interest (Xcovar) that are common to both groups 
# (age, sex, level of education, SES, exercise, smoking, alcohol, other 
# stimulants, and other depressants). These have already been regressed out
# using parameters estimated from the control group, but there might be 
# residual effects.
#
# Also, we'll include a binary predictor that indicates the presence of a pre-
# exisiting medical condition.

#%% 
# Include covariates in the model expression(s)

full_model_w_covar = ws.build_model_expression(fnames, Xcovar, 'pre_existing_condition')
print(f"Full model: {full_model_w_covar}")
r1b_regressions, _ = ws.regression_analyses(
		full_model_w_covar, comp_scores, Zcc, n_comparisons=15)

r1b_comparisons = ws.compare_models([
	{'name': 'F1',
		'h0': ws.build_model_expression('F2', Xcovar, 'pre_existing_condition'),
		'h1': full_model_w_covar}, 
	{'name': 'F2',
		'h0': ws.build_model_expression('F1', Xcovar, 'pre_existing_condition'),
		'h1': full_model_w_covar}
	], Zcc, comp_scores, smf.ols, n_comparisons=15)

res1b_ = (r1b_regressions
	.join(r1b_comparisons.loc[:, comp_columns])
	.rename(columns={'value': 'B'})
	.loc[r1b_comparisons.index, column_order]
)

save_and_display_table(res1b_, 'Table_S8')

#%% [markdown]
# ## COVID+ Subgroup Analyses: Hospitalised vs Non-Hospitalised Cases

#%% 
# Compare means of measures that make up F1
hms = sf36vars+subjvars+mhvars+['WHO_COVID_severity']
Zcc['hospital_stay'] = Zcc['hospital_stay'].cat.remove_unused_categories()
Xhm = Zcc.loc[:, hms+['hospital_stay']]
Xhm['WHO_COVID_severity'] = Xhm['WHO_COVID_severity'].apply(lambda x: 6-x)
Xhm['baseline_functioning'] = Xhm['baseline_functioning'].cat.codes
Xhm = Xhm[(Xhm[sf36vars+mhvars+['subjective_memory', 'WHO_COVID_severity']] >= 0).all(axis=1)]

HMtfm = ColumnTransformer([
	('sf36vars', MinMaxScaler(feature_range=[0,1]), sf36vars),
	('subjmem',  MinMaxScaler(feature_range=[0,1]), ['subjective_memory']),
	('mhvars', MinMaxScaler(feature_range=[0,1]), mhvars),
	('bf', 'passthrough', ['baseline_functioning']),
	('WHO', MinMaxScaler(feature_range=[0,1]), ['WHO_COVID_severity'])
]).fit(Xhm[hms])

Xhm = Xhm[['hospital_stay']].join(
	pd.DataFrame(
		HMtfm.transform(Xhm[hms]), 
		index=Xhm.index, columns=ws.get_ct_feature_names(HMtfm)))

f, m = wp.means_plot(
	Xhm, hms, 'measure', group='hospital_stay', group_order=['Yes', 'No'],
	group_color_sequence=wc.to_RGB_255(bwmap([0.33, 1.0])),
	bar_args={'range_y': [0, 1]}, 
	layout_args={
		'xaxis': {'title': None},
		'yaxis': {'title': 'Normalized Units'},
		'width': 650, 'height': 350},
	group_tests=True)

save_and_display_figure(f, 'Figure_X1')

bar_args = {
	'range_y': [-0.8, 0.4],
	'labels': {'mean': 'SD Units'},
}

layout_args = {
	'xaxis': {
		'title': None, 
	},
	'yaxis': {
		'tickmode': 'array',
		'tickvals': np.arange(-0.6, 1.3, 0.2).round(1)
	},
	'width': 400, 'height': 275,
	'margin': {'l': 70, 'b': 75, 't': 20, 'r': 20},
	'legend': {
		'title': "Hospital", 'orientation': 'v',
		'yanchor': 'top', 'xanchor': 'left', 'x': 1.02, 'y': 1.0,
	},
}

grayscale_map = wc.to_RGB_255(bwmap([0.33, 1.0]))

f5a, m = wp.means_plot(
	Zcc, comp_scores, 'measure', group='hospital_stay', group_order=['Yes', 'No'],
	group_color_sequence=grayscale_map,
	bar_args=bar_args, layout_args=layout_args,
)

save_and_display_figure(f5a, 'Figure_5A')

layout_args['showlegend'] = False
layout_args['width'] = 200
layout_args['xaxis']['tickangle'] = 30
f5b, m = wp.means_plot(
	Zcc, fnames, 'measure', group='hospital_stay', group_order=['Yes', 'No'],
	group_color_sequence=grayscale_map,
	bar_args=bar_args, layout_args=layout_args,
)
save_and_display_figure(f5b, 'Figure_5B')

#%% [markdown]
# ### Directly Compare Non-/Hospitalised Groups
# Description here...
#
#%% 
# Two-sample t-tests and stats
r7_ttests = (ws
	.two_sample_ttests('hospital_stay', comp_scores+fnames, Zcc, n_comparisons=7)
	.droplevel('contrast')
	.rename(columns={'cohen-d': 'd'})
)

save_and_display_table(r7_ttests[ttest_columns], 'Table_S9')

#%% [markdown]
# ### Compare Each Non-/Hospitalised Group to Controls
#%% 
# Two-sample t-tests and stats, etc.
r8_ttests = {}
for grp, grp_data in Zcc.groupby('hospital_stay'):
	Zall = pd.concat([grp_data, Zctrl], keys=['CC', 'SS'], names=['study'])
	r8_ttests[grp] = (ws
		.two_sample_ttests('study', comp_scores, Zall, n_comparisons=10)
		.droplevel('contrast')
		.rename(columns={'cohen-d': 'd'})
	)
r8_ttests = pd.concat(r8_ttests.values(), axis=0, keys=r8_ttests.keys(), names=['hospital_stay'])

save_and_display_table(r8_ttests[ttest_columns], 'Table_S10')

#%% [markdown]
# ### Include Hospitalisation Status in Regression Models
# Description
#
# %% 
# Transform it into a binary variable, where 1 = "Yes" (hospitalised)
if Zcc.hospital_stay.dtype == 'category':
	Zcc['hospital_stay'] = (
		OneHotEncoder(drop=['No'])
		.fit_transform(Zcc[['hospital_stay']])
		.todense()
	)

full_model_w_hosp = ws.build_model_expression(fnames, 'hospital_stay')
print(f"Model Expression: {full_model_w_hosp}")

rH_regressions, _ = ws.regression_analyses(
	full_model_w_hosp, comp_scores, Zcc, n_comparisons=15)

rH_comparisons = ws.compare_models([
	{'name': 'F1',
		'h0': ws.build_model_expression('F2', 'hospital_stay'), # without F1
		'h1': full_model_w_hosp}, 
	{'name': 'F2',
		'h0': ws.build_model_expression('F1', 'hospital_stay'), # without F1
		'h1': full_model_w_hosp},
	{'name': 'hospital_stay',
		'h0': ws.build_model_expression(fnames), # without hospital_stay
		'h1': full_model_w_hosp}
	], Zcc, comp_scores, smf.ols, n_comparisons=15)

resH_ = (rH_regressions
	.join(rH_comparisons.loc[:, comp_columns])
	.rename(columns={'value': 'B'})
	.loc[rH_comparisons.index, column_order]
	.rename({'hospital_stay': 'Hospital'})
)

save_and_display_table(resH_, 'Table_S11')

rH_t_fig = wp.create_stats_figure(
	resH_, 'tstat', 'p_adj', stat_range=[-6.3, 6.3],
	vertline=2, diverging=True, correction='bonferroni')
rH_b_fig = wp.create_bayes_factors_figure(
	resH_, suppress_h0=True, vertline=2)

rH_t_fig.savefig('./outputs/images/Figure_6a.svg')
rH_b_fig.savefig('./outputs/images/Figure_6b.svg')

#%% [markdown]
# ## Other Exploratory Analyses
# Next, we'll just do simple pairwise correlations between a bunch of variables
# and cognitive scores to see if there any other associations that might be 
# of interest. Given the large number of comparisons, we'll just focus on the 
# Bayes Factor (10 - in support of the alternative hypothesis).

#%%
# This is the list of all the variables we are going to use as predictors
# of the five composite cognitive scores.
vars_ = fnames + mhvars + Xcovar + ['pre_existing_condition']

# Let's run a linear regression that predicts each composite cognitive score
# from each variable in the list. Then, adjust all the coefficient p-values
# (5 x # of vars) using false discovery rate.
rEXPL_regressions = [ws.regression_analyses(f"%s ~ {v}", comp_scores, Zcc) for v in vars_]
rEXPL_regressions = [r[0].drop('Intercept', level='contrast') for r in rEXPL_regressions]
rEXPL_regressions = pd.concat(rEXPL_regressions, axis=0)
rEXPL_regressions = ws.adjust_pvals(
	rEXPL_regressions, adj_across='all', adj_type='fdr_bh')

# Here we'll do the Bayesian model comparisons, comparing a model with the 
# single variable (~ v) to an intercept-only model (~ 1).
models = [{'name': v, 'h0': f"%s ~ 1", 'h1': f"%s ~ {v}"} for v in vars_]
rEXPL_comparisons = ws.compare_models(models, Zcc, comp_scores, smf.ols)

# Make the figures...
rEXPL_ts = wp.create_stats_figure(
	rEXPL_regressions, 'tstat', 'p_adj', diverging=True, vertline=2, 
	correction='FDR', stat_range=[-6, 6])

rEXPL_bf = wp.create_bayes_factors_figure(rEXPL_comparisons, 
	vertline=2, cell_scale=0.6)

rEXPL_ts.savefig('./outputs/images/Figure_S5A.svg')
rEXPL_bf.savefig('./outputs/images/Figure_S5B.svg')

# Make the table of BFs...
rEXPL_bf_table = (rEXPL_comparisons['BF10']
	.unstack('score')
	.loc[vars_, comp_scores]
)

rEXPL_bf_table.to_csv('./outputs/tables/Table_S12.csv')

# %% [markdown]
# QQ-plots of scores, by hospitalisation

#%%
qq = np.empty([len(comp_scores), 2], dtype='object')
qt = np.empty([len(comp_scores), 2], dtype='object')
for ir, score in enumerate(comp_scores):
	for ic, grp in enumerate(["Yes", "No"]):
		xx = Zcc_.loc[Zcc_.hospital_stay==grp, score]
		qq[ir, ic] = probplot(xx, dist="norm")
		qt[ir, ic] = f"{score} (Hospital = {grp})"

qq_hosp = wp.qq_plots(qq, qt, layout_args={
	'height': 800, 'width': 500
})

save_and_display_figure(qq_hosp, 'Figure_QQhosp')

#%% [markdown]
# ### Sankey Diagram
# - This needs to be seriously cleaned up
# - Shows the flow of participants though questionnaire
# - Groups by assigned WHO COVID severity score

# %%

import itertools

groupers = ['symptoms',
			'hospital_stay', 'supplemental_O2_hospital',
			'ICU_stay',
			'ventilator', 'daily_routine', 'WHO']

q2 = Qcc.loc[Zcc.index, groupers].copy()
q2['WHO'] = q2['WHO'].astype('category')
q2['WHO'] = q2['WHO'].cat.add_categories('NA')
q2 = q2.fillna('NA')
q2.symptoms = q2.symptoms.astype('category')

q2 = remove_unused_categories(q2)

opts = [list(q2[g].cat.categories) for g in groupers]

all_paths = list(itertools.product(*opts))

src = []
trg = []
cnt = []
lab = []
for i, p in enumerate(all_paths):
	cnt.append((q2[groupers] == p).all(axis=1).sum())

good_i = [i for i, count in enumerate(cnt) if count > 0]
good_paths = [all_paths[i] for i in good_i]

p_ = []
for p in good_paths:
	lab.append(q2[(q2[groupers] == p).all(axis=1)]['WHO'].describe().top)
	p_.append(q2[(q2[groupers] == p).all(axis=1)])

good_cnts = [cnt[i] for i in good_i]

cats = CC.questionnaire.WHO_cats
ncat = len(cats)


links = {'source': [], 'target': [], 'value': []}
j = 0
for j, p in enumerate(good_paths):
	for i,v in enumerate(p[:-1]):
		if p[i] != "NA":
			ti = i+1
			while ti < (len(p)-1) and p[ti] == "NA":
				ti += 1
			if p[ti] != "NA":
				src = f"{groupers[i]}_{p[i]}"
				trg = f"{groupers[ti]}_{p[ti]}"
				links['source'].append(src)
				links['target'].append(trg)
				links['value'].append(good_cnts[j])

from floweaver import *

cats = CC.questionnaire.WHO_cats
ncat = len(cats)

import matplotlib
colormap = matplotlib.colors.ListedColormap(plotly.colors.sequential.Plasma)
colors = colormap(np.linspace(0, 1, ncat))
colors[:, -1] = 0.1

colors = [matplotlib.colors.to_hex(c) for c in colors]
colormap = dict(zip(cats, colors[0:ncat]))

links = pd.DataFrame(good_paths, columns=groupers)
links['value'] = good_cnts
links = links.rename(columns={'symptoms': 'source', 'WHO': 'target'})
links[links=='NA'] = np.nan

target = Partition.Simple('target', CC.questionnaire.WHO_cats)

nodes = {
	'start': ProcessGroup([False, True], Partition.Simple('source', [False, True]), title='Symptoms'),
	'end': ProcessGroup(CC.questionnaire.WHO_cats, target, title='COVID Severity'),
	'hospital': Waypoint(Partition.Simple('hospital_stay', ['No', 'Yes']), title='Hospitalised'),
	'daily': Waypoint(Partition.Simple('daily_routine', ['Yes', 'No']), title='Functional'),
	'o2': Waypoint(Partition.Simple('supplemental_O2_hospital', ['No', 'Yes', "Don't Know"]), title='Supplemental O2'),
	'ICU': Waypoint(Partition.Simple('ICU_stay', ['No', 'Yes']), title='ICU'),
	'ventilator': Waypoint(Partition.Simple('ventilator', ['No', 'Yes']), title='Ventilator'),
}

order = [
	['start'],
	['hospital'],
	['o2'],
	['daily', 'ICU'],
	['ventilator'],
	['end']
]

bundles = [
	# Bundle('start', 'end', waypoints=['hospital', 'ICU'], flow_selection="(hospital_stay == 'Yes')"),
	Bundle('start', 'end', waypoints=['hospital', 'daily'], flow_selection="(hospital_stay == 'No')"),
	Bundle('start', 'end', waypoints=['hospital', 'o2', 'ICU', 'ventilator'], flow_selection="(ICU_stay == 'Yes')"),
	Bundle('start', 'end', waypoints=['hospital', 'o2', 'ICU'], flow_selection="(ICU_stay == 'No')"),
]

sdd = SankeyDefinition(nodes, bundles, order, flow_partition=target)
(weave(sdd, links, palette=colormap)
	.to_widget(width=1000, align_link_types=True)
	.auto_save_svg('./outputs/images/Figure_S1.svg'))

#%%
SVG('./outputs/images/Figure_S1.svg')
#%%

