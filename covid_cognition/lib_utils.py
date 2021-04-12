# This file contains miscellaneous helper functions and constants for the 
# covid_cogntition.py analysis script. I've palced these items in here to 
# increase readiability of the main script.
#
def remove_unused_categories(df):
	""" Helper function to remove unused categories from all categorical columns
		in a dataframe. For use in chained pipelines.
	"""
	for col in df.select_dtypes('category').columns:
		df[col] = df[col].cat.remove_unused_categories()
	return df

# This a global variable that will track the data sample N though a
# preprocessing pipeline.
nprev = 0

def report_N(df, label='', reset_count=False):
	""" Helper function to report the size of a dataframe in a pipeline chain.
		Optionally, will print a label. Useful for logging/debugging. If 
		reset_count is true, then the global counter (used to calculate change
		in sample size) is reset to zero.
	"""
	global nprev
	if reset_count:
		nprev = 0
	ncurrent = df.shape[0]
	delta = ncurrent - nprev
	print(f"N = {ncurrent:5d}, {delta:+6d} ({label})")
	nprev = df.shape[0]
	return df

def set_column_names(df, new_names):
	""" Another helper function that sets the columns of a dataframe to the 
		supplied new names (a list-like of names).
	"""
	df.columns = new_names
	return df

from IPython.display import SVG, display
from os import path
def save_and_display_figure(figure, file_name):
	""" Save images in SVG format (for editing/manuscript) then display inside
		the notebook. Avoids having to have multiple versions of the image, or 
		multiple ways of displaying images from plotly, etc.
	"""
	full_file_name = path.join('.', 'images', f"{file_name}.svg")
	figure.write_image(full_file_name)
	display(SVG(full_file_name))

def pval_format(p):
	""" Formatter for p-values in tables.
	"""
	if p < 0.001:
		return "< 0.001"
	else:
		return f"{p:.03f}"

def bf_format(bf):
	""" Formatter for Bayes Factors in tables.
	"""
	if isinstance(bf, str):
		bf = float(bf)
	if bf > 1000:
		return "> 1000"
	else:
		return f"{bf:.02f}"

def ci_format(ci, precision=3):
	""" Formatter for confidence intervals, where ci is a 2-element array.
	"""
	return f"({ci[0]:.{precision}f}, {ci[1]:.{precision}f})"

def styled_df(df, return_it=False):
	""" Styles a dataframe (df) for display.
	"""
	styled_table = df.style.format(table_style)
	display(styled_table)
	if return_it:
		return styled_table

def save_and_display_table(df, fn):
	""" Styles a dataframe according to the rules below, and saves it as a 
		.html file. Saving as .html so it's as close as possible to the final
		manuscript format. Still haven't found a better way to do this...
	"""
	with open(f"./tables/{fn}.html", 'w') as wf:
		wf.write(styled_df(df, return_it=True).render())

table_style = {
	'B': '{:.2f}',
	'tstat': '{:.2f}',
	'df': '{:.2f}',
	'p_adj': pval_format,
	'CI': ci_format,
	'dR2': '{:.3f}',
	'f2': '{:.3f}',
	'd': '{:.2f}',
	'BF10': bf_format,
}