# This file contains miscellaneous helper functions and constants for the 
# covid_cogntition.py analysis script. I've palced these items in here to 
# increase readiability of the main script.
#

import numpy as np
import pandas as pd

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

def save_and_display_figure(figure, file_name, update_repo=True):
	""" Save images in SVG format (for editing/manuscript) then display inside
		the notebook. Avoids having to have multiple versions of the image, or 
		multiple ways of displaying images from plotly, etc.
	"""
	if update_repo:
		img_path = path.join('.', 'images')
	else:
		img_path = path.join('.', 'tmp')
	full_file_name = path.join(img_path, f"{file_name}.svg")
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

def procrustes(X, Y, scaling=True, reflection='best'):
	""" A port of MATLAB's `procrustes` function to Numpy.

	https://stackoverflow.com/a/18927641

	Procrustes analysis determines a linear transformation (translation,
	reflection, orthogonal rotation and scaling) of the points in Y to best
	conform them to the points in matrix X, using the sum of squared errors
	as the goodness of fit criterion.

		d, Z, [tform] = procrustes(X, Y)

	Args:
		X, Y (matrices) - target and input coordinates, must have equal
			numbers of  points (rows), but Y may have fewer dimensions
			(columns) than X.

		scaling (boolean) - if False, the scaling component of the 
			transformation is forced to 1 (default = True)

		reflection (string) - if 'best' (default), the transformation solution 
			may or may not include a reflection component, depending on which 
			fits the data best. setting reflection to True or False forces a 
			solution wit reflection or no reflection respectively.

	Returns:
		d - the residual sum of squared errors, normalized according to a
			measure of the scale of X, ((X - X.mean(0))**2).sum()

		Z - the matrix of transformed Y-values

		tform - a dict specifying the rotation, translation and scaling that
			maps X --> Y

	"""

	n,m = X.shape
	ny,my = Y.shape

	muX = X.mean(0)
	muY = Y.mean(0)

	X0 = X - muX
	Y0 = Y - muY

	ssX = (X0**2.).sum()
	ssY = (Y0**2.).sum()

	# centred Frobenius norm
	normX = np.sqrt(ssX)
	normY = np.sqrt(ssY)

	# scale to equal (unit) norm
	X0 /= normX
	Y0 /= normY

	if my < m:
		Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

	# optimum rotation matrix of Y
	A = np.dot(X0.T, Y0)
	U,s2,Vt = np.linalg.svd(A,full_matrices=False)
	V = Vt.T
	T = np.dot(V, U.T)

	if reflection != 'best':

		# does the current solution use a reflection?
		have_reflection = np.linalg.det(T) < 0

		# if that's not what was specified, force another reflection
		if reflection != have_reflection:
			V[:,-1] *= -1
			s[-1] *= -1
			T = np.dot(V, U.T)

	traceTA = s2.sum()

	if scaling:

		# optimum scaling of Y
		b = traceTA * normX / normY

		# standarised distance between X and b*Y*T + c
		d = 1 - traceTA**2

		# transformed coords
		Z = normX*traceTA*np.dot(Y0, T) + muX

	else:
		b = 1
		d = 1 + ssY/ssX - 2 * traceTA * normY / normX
		Z = normY*np.dot(Y0, T) + muX

	# transformation matrix
	if my < m:
		T = T[:my,:]
	c = muX - b*np.dot(muY, T)
	
	#transformation values 
	tform = {'rotation':T, 'scale':b, 'translation':c}
   
	return d, Z, tform


def tuckersCC(df1, df2, do_procrustes=False, modified=False):
	""" Tucker's Congruence Coefficient for Factor Matching

	References:

		Lorenzo-Seva, U., & ten Berge, J. M. (2006). Tucker’s congruence 
			coefficient as a meaningful index of factor similarity. Methodology,
			2(2), 57-64. https://doi.org/10.1027/1614-2241.2.2.57

		Lovik, A., Nassiri, V., Verbeke, G. & Molenberghs, G. A modified 
			Tucker’s congruence coefficient for factor matching. Methodology 
			16, 59–74 (2020). https://meth.psychopen.eu/index.php/meth/article/view/2813/2813.html

		Tucker, L. R. (1951). A method for synthesis of factor analysis studies
			(No. PRS-984). Princeton, NJ, USA: Educational Testing Service.

	See also:
		https://meth.psychopen.eu/index.php/meth/article/download/2813/2813.html?inline=1#r15

	Args:
		df1, df2 (matrices) - target and input matrices, must have equal
			numbers of observed variables (M - rows) and factors (N - columns)

		do_procrustes (boolean) - if True, performs a procrustes transformation
			to align df2 to the target (df1) matrix before calculating the 
			congruence coefficient (default = False)

		modified (boolean) - if True, calculates the "modified" coefficient 
			proposed by Lobik et al. (2020)

	Returns:
		(array-like) with N elements, where each is the congruence coefficient
			between each column of the two matrics; i.e., the congruence 
			between each factor loading vector between the two matrices.

	"""

	assert df1.shape == df2.shape
	assert len(df1.shape) == 2
	np.seterr(invalid='ignore') 

	if isinstance(df1, pd.DataFrame):
		df1 = df1.to_numpy()
	if isinstance(df2, pd.DataFrame):
		df2 = df2.to_numpy()

	if do_procrustes:
		r = procrustes(df1, df2)
		df2 = r[1]  # the transformed matrix of df2

	ss11 = df1.T @ df1
	ss22 = df2.T @ df2
	ss12 = df1.T @ df2

	if modified:
		ss12 = np.abs(ss12)

	return np.diag(ss12 / np.sqrt(ss11 * ss22))