# -----------------------------------------------------------------------------
# This lib file contains helper functions for doing many statistics. Rather
# than having all that code within a notebook, resuable functions are placed in 
# here.
#
# See:
#  - https://plotly.com/python/
#  - https://matplotlib.org/
#
# -----------------------------------------------------------------------------
# cwild 2021-04-15

import pandas as pd
import numpy as np
import itertools
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.formula.api import logit, mnlogit, ols

idx = pd.IndexSlice

def flatten(l):
    """ Recursively flattens a list of lists that might have strings, without
        unpacking all the characters in the strings.
        See:
        https://stackoverflow.com/questions/17864466/flatten-a-list-of-strings-and-lists-of-strings-and-lists-in-python
    """
    from collections.abc import Iterable
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def build_model_expression(*regressors):
    """ Given a list (or set) of strings that are variables in a dataframe, builds an expression
        (i.e., a string) that specifies a model for multiple regression. The dependent variable (y)
        is filled with '%s' so that it can replaced as in a format string.
    
    Args:
        regressors (list or set of strings): names of independent variables that will be used 
            to model the dependent variable (score).
        
    Returns:
        string: the model expression
                
    Example:
        >>> build_model_expression(['age', 'gender', 'other'])
            '%s ~ age+gender+other'
    """
    if len(regressors) == 0:
        regressors = ['1']
    else:
        regressors = list(flatten(regressors))

    return '%s ~ '+'+'.join(regressors)


def build_interaction_terms(*regressors):
    """ Given multiple lists (or sets) of regressors, returns a set of all interaction terms.
        The set of interaction terms can then be used to build a model expression.
    
    Args:
        *regressors (mutiple list or sets of strings)
            
    Returns:
        the set of all interaction terms.
            
    Examples:
        >>> build_interaction_terms(['age'], ['sleep_1', 'sleep_2'])
            {'age:sleep_1', 'age:sleep_2'}
        >>> build_interaction_terms(['age'], ['sleep_1', 'sleep_2'], ['gender_male', 'gender_other'])
            {'age:sleep_1:gender_male',
             'age:sleep_1:gender_other',
             'age:sleep_2:gender_male',
             'age:sleep_2:gender_other'}
    """
    return set([':'.join(pair) for pair in itertools.product(*regressors)])



def tstat(alpha, df):
    assert((alpha < 1.0) & (alpha > 0.0))
    from scipy.stats import t
    return t.ppf((1+alpha)/2, df)


def cohens_f_squared(full_model, restricted_model, type_='logit'):
    """ Calculate Cohen's f squared effect size statistic. See this reference:
    
        Selya, A. S., Rose, J. S., Dierker, L. C., Hedeker, D., & Mermelstein, R. J. (2012). 
            A practical guide to calculating Cohen’s f 2, a measure of local effect size, from PROC 
            MIXED. Frontiers in Psychology, 3, 1–6.
    
    """
    return (r2_from(full_model)-r2_from(restricted_model))/(1-r2_from(full_model))


def bayes_factor_01_approximation(full_model, restricted_model, min_value=0.001, max_value=1000):
    """ Estimate Bayes Factor using the BIC approximation outlined here:
        Wagenmakers, E.-J. (2007). A practical solution to the pervasive problems of p values.
            Psychonomic Bulletin & Review, 14, 779–804.
        
    Args:
        full_model (statsmodels.regression.linear_model.RegressionResultsWrapper):
            The estimated multiple regression model that represents H1 - the alternative hypothesis
        restricted_model (statsmodels.regression.linear_model.RegressionResultsWrapper):
            The estimated multiple regression model that represents H0 - the null hypothesis
        min_value (float): a cutoff to prevent values from getting too small.
        max_value (float): a cutoff to prevent values from getting too big
    
    Returns:
        A float - the approximate Bayes Factor in support of the null hypothesis
    """
    assert(full_model.nobs == restricted_model.nobs)
    bf = np.exp((full_model.bic - restricted_model.bic)/2)
    return np.clip(bf, min_value, max_value)


def likelihood_ratio_test(df, h1, h0, lr):
    h0 = lr(h0, df).fit(disp=False)
    h1 = lr(h1, df).fit(disp=False)
    return likelihood_ratio_test_calc(h0, h1)


def likelihood_ratio_test_calc(h0, h1):
    llf_full = h1.llf
    llf_restr = h0.llf
    df_full = h1.df_resid
    df_restr = h0.df_resid
    lrdf = (df_restr - df_full)
    lrstat = -2*(llf_restr - llf_full)
    lr_pvalue = stats.chi2.sf(lrstat, lrdf)
    return (lrstat, lr_pvalue, lrdf)

def r2_from(estimated_model):
    """ Get the R-squared statistic from a fitted model. In most cases this is
        simply the 'rsquared' parameters, but in some cases like with a logit
        model we need the pseudo-r-squared.
    """
    r2 = getattr(estimated_model, 'rsquared', None)
    if r2 is None:
        r2 = getattr(estimated_model, 'prsquared', None)
    if r2 is None:
        raise AttributeError("Can't find R2 function for model")
    return r2


def compare_models(model_comparisons, 
        data, score_columns, model_type, **correction_args):
    #UPDATE THIS    
    """ Performs statistical analyses to compare fully specified regression models to (nested)
        restricted models. Uses a likelihood ratio test to compare the models, and also a 
        Bayesian model comparison method.
        
    References:
        http://www.statsmodels.org/dev/regression.html
        http://www.statsmodels.org/dev/generated/statsmodels.sandbox.stats.multicomp.multipletests.html
        http://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.compare_lr_test.html
        
    Args:
        model_comparisons (list of dicts): each dict in the list represents a contrast
            between two models. The dict must have: a 'name' field; a 'h0' field that 
            is the model expression for the restricted (or null) model; and a 'h1' field
            that is the model expression string for the fully specified model.
        data (panads data frame): the full data frame with all independent variables
            (predictors) and dependent variables (scores). Model expression strings in
            the previous arg are built from names of columns in this data frame. 
        score_columns (list of strings): which columns of the data frame are dependent
            variables to be modelled using expressions in model_comparisons?
        alpha (float): What considered a significant p-value after correction.
        n_comparisons (float): If specified, this is used to correct the p-values returned
            by the LR tests (i.e., multiply each p-value by the number of comparisons).
            The resulting adjusted p-value is what is compared to the alpha argument.
            Note: this take precendence over the multiple correction type argument.
        correction (string): method for correcting for multiple comparisons. Can be
            None, or any of the options listed in the above documentation for multipletests.
                
    Returns:
        A pandas dataframe with one row per score, and a multindex column structure that 
        follows a (contrast_name, statistic_name) convention.
    
    """
    score_names = [score for score in score_columns]  # no prproc for now
    contrasts = [comparison['name'] for comparison in model_comparisons]
    statistics = ['LR', 'p', 'p_adj', 'df',
                  'dR2', 'f2', 'BF01', 'BF10']
    results_df = pd.DataFrame(
                    index=pd.MultiIndex.from_product(
                        [contrasts, score_names], names=['contrast', 'score']),
                              columns=statistics)
                              
    for contrast in model_comparisons:
        for score_index, score in enumerate(score_columns):
            score_name = score_names[score_index]

            # Fit the fully specified model (h1) and the nested restricted model (h0) using OLS
            h1 = model_type(contrast['h1'] % score, data=data).fit(disp=False)

            h0data = data.loc[h1.model.data.orig_exog.index, :]
            h0 = model_type(contrast['h0'] % score, data=h0data).fit(disp=False)

            # Perform Likelihood Ratio test to compare h1 (full) model to h0 (restricted) one
            lr_test_result = likelihood_ratio_test_calc(h0, h1)
            bayesfactor_01 = bayes_factor_01_approximation(
                h1, h0, min_value=0.0000001, max_value=10000000)
            all_statistics = [lr_test_result[0],
                              lr_test_result[1],
                              np.nan,
                              lr_test_result[2],
                              r2_from(h1) - r2_from(h0),
                              cohens_f_squared(h1, h0),
                              bayesfactor_01,
                              1/bayesfactor_01]
            results_df.loc[(contrast['name'], score_name), :] = all_statistics

    # Correct p-values for multiple comparisons across all tests of this contrast?
    results_df = (results_df
        .pipe(adjust_pvals, **correction_args)
        .swaplevel('contrast', 'score')
        .loc[idx[score_names,:], :]
    )

    return results_df

def corrected_alpha_from(**correction_args):
    if 'alpha' in correction_args:
        return correction_args['alpha']
    elif 'n_comparisons' in correction_args:
        return 0.05 / correction_args['n_comparisons']
    else:
        return 0.05

def adjust_pvals(results, 
        n_comparisons=None, adj_across=None, adj_type=None, alpha=None):
    """

    Args:
        results (Pandas dataframe): a results dataframe that has a column of 
            p-values, and a multi-index with two levels: 'contrast' and
            'score'.

    Optional Args:
        alpha (float): the alpha (e.g., 0.05) for determining statistical
            significance. You can etiher specify this here, or leave it and
            specify a method for correcting formultiple comparisons (see
            next two parameters). If set, this option takes priority.
        n_comparisons: like alpha, but instead it's the number of comparisons.
        adj_across (string): when doing an automatic correction for multiple
            comparisons, do you correct across DVs ("scores"), IVs ("contrasts")
            or all of them ("all")? 
        adj_type (string): the method for correcting for multiple comparisons,
            should be one of the options available in <insert function here>.
            For example 'fdr_bh', 'sidak', 'bonferroni', etc.

    Returns:
        a Pandas dataframe that  is like original results dataframe, but with an
        additional column 'p_adj' that has the corrected p-values.
    """

    if n_comparisons is not None:
        p_adj = results['p'] * n_comparisons

    elif alpha is not None:
        p_adj = results['p'] * 0.05/alpha

    elif adj_across is not None and adj_type is not None:
        if adj_across == 'all':
            p_vals = results[['p']]
        elif adj_across in ['scores', 'score', 's']:
            p_vals = results['p'].unstack('contrast')
        elif adj_across in ['contrasts', 'contrast', 'con', 'cons', 'c']:
            p_vals = results['p'].unstack('score')
        else:
            raise ValueError(f"Invalid adjust across = {adj_across}")
          
        p_adj = [multipletests(p_vals[col], method=adj_type)[1] for col in p_vals]
        p_adj = pd.DataFrame(np.column_stack(p_adj), index=p_vals.index, columns=p_vals.columns)
        if p_adj.shape[1] > 1:
            p_adj = p_adj.stack()
        p_adj = p_adj.reorder_levels(results.index.names).reindex(results.index)

    else:
        p_adj = results['p']

    results['p_adj'] = np.clip(p_adj.values, 0, 1)
    return results


def regression_analyses(formula, DVs, data, IVs=None, **correction_args):
    """ Insert description here.

    Args:
        Formula (string): the regression formula used for building models.
            Must begind with '%s' because the DV gets interpolated when looping
            over all the variables to be evaluated.
        DVs (list-like): the names of dependent variables. One regression model
            will be estimated for each DV.
        IVs (list-like): the independent variables to do statistics on. If 
            None then all variables in the formula are tested. (default: None)
        **correction args: keyword arguments that get passed to adust_pvals()

    """
    results = []
    models = []

    for dv in DVs:
        model = ols(formula % dv, data).fit()
        r = pd.concat(
                [model.params, 
                 model.tvalues,
                 model.pvalues, 
                 model.conf_int(corrected_alpha_from(**correction_args))],
                axis=1)
        r.columns = ['value', 'tstat', 'p', 'CI_lower', 'CI_upper']
        r['df'] = model.df_resid
        r.index.name = 'contrast'
        results.append(r)
        models.append(model)

    results = pd.concat(results, names=['score'], keys=DVs)
    results['CI'] = results.apply(lambda x: [x['CI_lower'], x['CI_upper']], axis=1)

    if IVs is not None:
        results = results.loc[idx[:, IVs], :]

    results = adjust_pvals(results, **correction_args)

    return results, models

def two_sample_ttests(
        group_var, DVs, data, paired=False, tails='two-sided',
        test_name='Mean Difference', **correction_args):

    """ Performs a series of two-sample t-tests, collecting all the statistics
        and returning a nicely formatted dataframe of results.

    Args:
        group_var (string): The variables that will be used to group/split
            the dataset. Bust have only two levels.
        DVs (list-like): the names of dependent variables. One t-test will be 
            done for each element of this list.
        data (Pandas dataframe): the raw data frame with variables as columns
            and rows as observations.
    
    Optional Args:
        paired (boolean): if True, performs paired-sample t-tests.
        tails (string): specifies "two-sided" or "one-sided" t-tests.
        test_name (string): the label for the test difference column.
        **correction args: keyword arguments that get passed to adust_pvals()

    """
    
    from pingouin import ttest
    grp_data = [d for _, d in data.groupby(group_var)]
    assert(len(grp_data)==2)

    results = []
    for dv in DVs:
        t = ttest(
            grp_data[0][dv], grp_data[1][dv], 
            confidence=(1-corrected_alpha_from(**correction_args)),
            paired=paired, tail=tails)
        t.index.names = ['contrast']
        t['value'] = grp_data[0][dv].mean() - grp_data[1][dv].mean()
        results.append(t)

    results = (pd
        .concat(results, names=['score'], keys=DVs)
        .rename(columns={'p-val': 'p', 'T': 'tstat', 'dof': 'df', 'value': 'diff'},
                index={'T-test': test_name})
    )

    # unpack the CIs provided by pingouin
    ci_col = [c for c in results.columns if c[0:2]=='CI'][0]
    results = (results
        .assign(CI_lower=results[ci_col].apply(lambda x: x[0]))
        .assign(CI_upper=results[ci_col].apply(lambda x: x[1]))
        .rename(columns={ci_col: 'CI'})
    )

    results = adjust_pvals(results, **correction_args)

    return results



def filter_df(df, sds = [6,4], subset=None, drop=False):
    """ Filters a dataframe (column-wise) by setting values more than X standard
        deviations (SDs) away from the mean to np.nan. Each column is considered
        in isolation.

    Args:
        df (Pandas dataframe): a dataframe with columns as variables and rows 
            as observations. By default all columns of the dataframe are
            filtered.
    
    Optional Args:
        sds (array/list-like): Threshold of X # of SDs away from the mean. If
            there are multiple values (like the default [6,4]) then filtering is
            done in multiple sequential passes where the mean and SD are
            re-calculated on each pass.
        subset (array/list-like): a subset of columns in the dataframe to filter.
        drop (boolean): if True, rows with outliers are dropped from the 
            dataframe before being returning. if False, outliers are just 
            masked as np.nan in the original dataframe. 
    """

    if subset is None:
        subset = df.columns
    
    df_ = df[subset].copy()

    outliers = np.full(df_.shape, False)
    for sd in sds:
        stats = df_.agg(['count', 'mean', 'std'])
        oorange = (abs(df_ - stats.loc['mean', :]) > sd*stats.loc['std', :])
        df_[oorange] = np.nan
        outliers = (outliers | oorange.values)
    df[subset] = df_

    if drop:
        df = df[~outliers.any(axis=1)]

    return df

from scipy import stats
## Helper functions for running ch2, 1-way ANOVA, or t-tests on a Pandas datafame.
def chi2_pval(df, grouper, var):
    """ Computes Chi2 stat and pvalue for a variable, given a grouping variable.
    """
    tabs = df[[grouper, var]].groupby([grouper])[var].value_counts()
    chi2 = stats.chi2_contingency(tabs.unstack(grouper))
    return chi2[1]

def f_1way_pval(df, grouper, var):
    """
    """
    g = [d[var].dropna() for i, d in df[[grouper, var]].groupby(grouper)]
    f = stats.f_oneway(*g)
    return f[1]

def t_pval(df, grouper, var):
    """ Assumes only two groups present! Otherwise use f_1way_pval
    """
    g = [d[var].dropna() for i, d in df[[grouper, var]].groupby(grouper)]
    t = stats.ttest_ind(g[0], g[1], equal_var=False)
    return t[1]


# Below here are helpers for working with some SciKit-Learn models and components

from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.pipeline import Pipeline

def get_feature_out(estimator, feature_in):
    if hasattr(estimator,'get_feature_names'):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f'vec_{f}' \
                for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(feature_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in


def get_ct_feature_names(ct):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers_:
        if name!='remainder':
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator=='passthrough':
            output_features.extend(ct._feature_names_in[features])
                
    return output_features

