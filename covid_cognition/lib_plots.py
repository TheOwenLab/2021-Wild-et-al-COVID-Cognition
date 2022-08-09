# -----------------------------------------------------------------------------
# This lib file contains helper functions for creating plots and figures. Rather
# than having all that code within a notebook, resuable functions are placed in 
# here. It's a mix of Plotly and Matplotlib functions.
#
# See:
#  - https://plotly.com/python/
#  - https://matplotlib.org/
#
# -----------------------------------------------------------------------------
# cwild 2021-04-15

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from plotly.colors import colorbrewer as cb

from .lib_stats import f_1way_pval
from .lib_colours import D1_CMAP, D2_CMAP, D3_CMAP, D4_CMAP

idx = pd.IndexSlice

# Default settings for matplotlib
from matplotlib import rc
plt.rcParams['figure.dpi'] = 100
plt.rcParams.update({'font.size': 10})
rc('font',**{'family':'sans-serif','sans-serif':['Avenir']})
plt.rcParams['svg.fonttype'] = 'none'

# Default layout for any plots created with Plotly
_LINE_COLOUR = 'rgb(16, 16, 16)'
_GRID_COLOUR =  'rgb(220, 220, 220)'
def plotly_template():
    return {
        'layout': go.Layout(
            # plot_bgcolor = 'rgba(1,1,1,0.1)',
            plot_bgcolor = 'white',
            font_family = 'avenir',
            font = {'size': 8},
            xaxis = {
                'zeroline': True,
                'zerolinecolor': _GRID_COLOUR,
                'zerolinewidth': 1.5,
                'gridcolor': 'white',
                'gridwidth': 1
            },
            yaxis = {
                'zeroline': True, 
                'zerolinewidth': 1.5,
                'gridwidth': 1,
                'zerolinecolor': _GRID_COLOUR,
                'gridcolor': _GRID_COLOUR,
            },
        ),
        'data': {
            'bar': [go.Bar(
                marker_line_color = _LINE_COLOUR, 
                marker_line_width = 1.5,
                error_y = {
                    'color': _LINE_COLOUR,
                    'thickness': 1.5,
                }
            )],
            'scatter': [go.Scatter(
                marker_line_color = _LINE_COLOUR, 
                marker_line_width = 1.5,
                error_y = {
                    'color': _LINE_COLOUR,
                    'thickness': 1.5,
                }
            )]
        }
    }

def mm_to_pts(mm, dpi=300):
    """ Converts a length in mm to points, given the dots per inch (DPI)

    Args:
        mm (numeric): the length in millimetres
        dpi (numeric): the dots per inch (DPI) of the figure (default: 300)
    
    Returns:
        (numeric): the dimsension in points
    """
    return np.ceil(mm / 100 / 2.54 * dpi)

def create_stats_figure(
        results, stat_name, p_name, alpha=0.05, log_stats=True, 
        diverging=False, stat_range=None, correction=None, vertline=4, 
        marker_color=None
    ):
    """ Creates a matrix figure to summarize multple tests/scores. Each cell 
        represents a contrast (or model comparison) for a specific effect (rows)
        for a given score (columns). Also draws asterisks on cells for which 
        there is a statistically significant effect.
        
    Args:
        results (Pandas dataframe): a dataframe that contains the statistics to 
            display. Should be a rectangular dataframe with tests as rows and 
            effects as columns (i.e., the  transpose of the resulting image). 
            The dataframe index and column labels are used as labels for the 
            resulting figure.
        stat_name (string): Which statistic to plot. There might be multiple 
            columns for each effect (e.g., Likelihood Ratio, BFs, F-stats, etc.)
        p_name (string): The name of the column to use for p-values.
        alpha (float): what is the alpha for significant effects?
        log_stats (boolean): Should we take the logarithm of statistic values 
            before creating the image? Probably yes, if there is a large 
            variance in value across tests and effects.
        diverging (boolean): Are the stat values diverging, like a t-test, or 
            not (like F-stat). Determines the colour scales, etc.
        stat_range (list-like): upper and lower limits of the statistic range.
        correction (string): indicates how the alpha was corrected (e.g., FDR 
            or bonferroni) so the legend can be labelled appropriately.
        vertline (integer): Where to place the vertical white line that
            separates columns.
        marker_color (any kind of colour type): the colour for the markers.
            
    Returns:
        A matplotlib figure.
        
    """

    score_index = results.index.unique('score')
    contrast_index = results.index.unique('contrast')
    stat_values = (results
        .loc[:, stat_name]
        .unstack('contrast')
        .loc[score_index, contrast_index]
    )
    p_values = (results
        .loc[:, p_name]
        .unstack('contrast')
        .loc[score_index, contrast_index]
    )
    num_scores = stat_values.shape[0]
    num_contrasts = stat_values.shape[1]
    image_values = stat_values.values.astype('float32')

    # If it's a diverging scale, it's probably a t-stat or something. Don't
    # know why I have this here. There is a better solution.
    if diverging:
        log_stats = False

    image_values = np.log10(image_values) if log_stats else image_values

    imax = np.max(np.abs(image_values))
    if diverging:
        irange = [-1*imax, imax] if stat_range is None else stat_range
        cmap = D2_CMAP
    else:
        irange = [0, np.min([3, imax])] if stat_range is None else stat_range
        cmap = 'viridis'
        image_values = np.clip(image_values, 0, 100)

    figure = plt.figure(figsize=[num_scores*0.6, num_contrasts*0.6])
    plt_axis = figure.add_subplot(1, 1, 1)
    imgplot = plt_axis.imshow(
                image_values.T, aspect='auto', clim=irange, cmap=cmap)

    if vertline is not None:
        plt_axis.plot([num_scores-(vertline+.5), num_scores-(vertline+.5)],
                    [-0.5, num_contrasts-0.5], c='w')

    if marker_color is None:
        marker_color = 'whitesmoke' 

    plt_axis.set_yticks(np.arange(0, num_contrasts))
    plt_axis.set_yticklabels(list(contrast_index))
    plt_axis.set_xticks(np.arange(0, num_scores))
    plt_axis.set_xticklabels(list(score_index), rotation=45, ha='right')
    cbar = figure.colorbar(imgplot, ax=plt_axis, pad=0.2/num_scores)
    if log_stats:
        cbar.ax.set_ylabel('$Log_{10}$'+stat_name)
    else:
        cbar.ax.set_ylabel(f"{stat_name}")

    reject_h0 = (p_values.values.T < alpha).nonzero()
    legend_label = "p < %.02f" % alpha
    legend_label += f" ({'unc' if correction is None else correction})"
    plt_axis.plot(reject_h0[1], reject_h0[0], marker_color, linestyle='none',
                  marker='$\u2217$', label=legend_label, markersize=10)
    # plt_axis.plot(reject_h0[1], reject_h0[0], '*',
    #               markersize=10, label=legend_label)

    plt.legend(bbox_to_anchor=(1, 1.1), loc=4, borderaxespad=0.,
        facecolor='lightgray', edgecolor='lightgray')

    return figure


def create_bayes_factors_figure(results, log_stats=True, 
        vertline=None, cmap=None, cell_scale=0.6, suppress_h0=False):
    """ Creates a matrix figure to summarize Bayesian stats for multiple scores
        & tests. Each cell indicates the Bayes Factor (BF associated with a
        model comparison) for  a specific effect (rows) for a given score 
        (columns). Also draws symbols on cells to indicate the interpretation of
        that BF.
        
    Args:
        results (Pandas dataframe): a dataframe that contains the statistics to
            display. Should be a rectangular dataframe with tests as rows and 
            effects as columns (i.e., the  transpose of the resulting image). 
            The dataframe index and column labels are used as labels for the 
            resulting figure.
        log_stats (boolean): Should we take the logarithm of BF values before
            creating the image? Probably yes, if there is a large variance in 
            value across scores and effects.
        vert_line (integer): where to place the vertical white line that 
            separates subset of columns in the matrix figure.
        cmap (colourmap): You can override the default colour map.
        cell_scale (float): for sizing the cells.
        suppress_h0 (boolean): if True, markers will not be shown on cells that
            have a Bayes Factor in support of H0. This is just intended to 
            reduce visual clutter.
            
    Returns:
        A matplotlib figure
    
    """
    
    score_index = results.index.unique('score')
    contrast_index = results.index.unique('contrast')
    num_scores = len(score_index)
    num_contrasts = len(contrast_index)
    bf_values = results.loc[:, 'BF10'].unstack('contrast').reindex(
        index=score_index, columns=contrast_index).values.astype('float32')
    # Too small values cause problems for the image scaling

    np.place(bf_values, bf_values < 0.00001, 0.00001)

    if cmap is None:
        cmap = D2_CMAP

    figure = plt.figure(figsize=[num_scores*cell_scale, num_contrasts*cell_scale])
    plt_axis = figure.add_subplot(1, 1, 1)
    imgplot = plt_axis.imshow(np.log10(bf_values.T),
                              aspect='auto', cmap=cmap, clim=[-6.0, 6.0])

    if vertline is not None:
        plt_axis.plot([num_scores-(vertline+.5), num_scores-(vertline+.5)],
                    [-0.5, num_contrasts-0.5], c='w')

    plt_axis.set_yticks(np.arange(0, num_contrasts))
    plt_axis.set_yticklabels(list(contrast_index))
    plt_axis.set_xticks(np.arange(0, num_scores))
    plt_axis.set_xticklabels(list(score_index), rotation=45, ha='right')

    # Add a colour bar
    cbar = figure.colorbar(imgplot, ax=plt_axis, pad=0.2/num_scores)
    cbar.ax.set_ylabel('$H_0$   '+'$Log(BF_{10})$'+'   $H_1$')
    # cbar.ax.text(75,  4, "$H_1$")
    # cbar.ax.text(75, -5, "$H_0$")

    # Use absolute BFs for determining weight of evidence
    abs_bfs = bf_values
    abs_bfs[abs_bfs == 0] = 0.000001
    if not suppress_h0:
        abs_bfs[abs_bfs < 1] = 1/abs_bfs[abs_bfs < 1]

    # Custom markers for the grid
    # markers = [(2+i, 1+i % 2, i/4*90.0) for i in range(1, 3)]
    markers = [(3, 2, 22.5), '$\u2727$', '$\u2736$']
    markersize = 10 * cell_scale *2

    # Positive evidence BF 3 - 20
    positive = (abs_bfs >= 3) & (abs_bfs < 20)
    xy = positive.nonzero()
    plt_axis.plot(xy[0], xy[1], 'whitesmoke', linestyle='none',
                  marker=markers[0], label='positive', markersize=markersize)

    # Strong Evidence BF 20 - 150
    strong = (abs_bfs >= 20) & (abs_bfs < 150)
    xy = strong.nonzero()
    plt_axis.plot(xy[0], xy[1], 'whitesmoke', linestyle='none',
                  marker=markers[1], label='strong', markersize=markersize)

    # Very strong evidence BF > 150
    very_strong = (abs_bfs >= 150)
    xy = very_strong.nonzero()
    plt_axis.plot(xy[0], xy[1], 'whitesmoke', linestyle='none',
                  marker=markers[2], label='v. strong', markersize=markersize)

    plt.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center',
               borderaxespad=0., ncol=4, title='Bayesian Evidence',
               facecolor='lightgray', edgecolor='lightgray')

    return figure

# The following dicts (prefixed with rc_) are default options for the raincloud
# plots. Putting them here cleans up the main notebook, but we can still update
# values in order to change the plots when needed.

rc_title = {
	'pad': {'b': 10, 'l': 0},
	'yanchor': 'bottom', 'xanchor': 'left',
	'yref': 'paper', 'xref': 'paper',
	'x': 0, 'y': 1
}

rc_yaxis = {
	'title': {'text': 'Score (SDs)', 'standoff': 2},
    'automargin': False,
	'range': [-4.2, 4.1],
	'tickmode': 'array',
	'tickvals': np.arange(-4, 4+1),
	'ticktext': [f"{y}   " for y in np.arange(-4, 4+1)]
}

rc_layout = {
	'width': 400, 'height': 275,
	'margin': {'b': 15, 't': 30, 'l': 35, 'r': 10},
	'title': rc_title,
	'yaxis': rc_yaxis,
}

rc_legend = {
	'orientation': 'h', 'yanchor': 'top', 'xanchor': 'left', 
	'x': 0.01, 'y': 0.99,
	'title': {'side': 'left', 'text': ""},
}

def raincloud_plot(
        df, plt_vars, grp_var, grp_order=None, grp_colours=cb.Dark2,
        do_box=True, do_pts=True, do_vio=True,
        box_args={}, pts_args={}, mrk_args={}, vio_args={},
        pts_jitter=None, vio_jitter=False, 
        sym_offset=0, sym_constant=False, colour_offset=0,
        layout_args=None, legend_args=None
    ):
    """ This is a custom implementation of a "raincloud" plot, that displays
        adjacent jittered stripe plots, boxplots, and distribution curves
        (i.e., half violins) for a given dataset. I rewrote this using plotly / 
        python to make plots consistent for this study. 
        
        Sorry, horizontal orientation NYI. 
        
        This function Will adjust the width of the figure elements depending on 
        which plot elements (points, box, violing) that you want to display. By
        default, all three are shown.

        https://neuroconscience.wordpress.com/2018/03/15/introducing-raincloud-plots/
        https://github.com/RainCloudPlots/RainCloudPlots

        Reference: Allen, M., Poggiali, D., Whitaker, K., Marshall, T. R. & 
        Kievit, R. A. Raincloud plots: A multi-platform tool for robust data 
        visualization. Wellcome Open Res. 4, 1–51 (2019).

        Plotly details for the three subplots:
            https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Box.html
            https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Violin.html

    Required Args:
        df (Pandas DataFrame) - contains the input data.
        plt_vars (list-like) - the name of the columns that contain data to be
            plotted. Each element of the list will be plotted on a unique
            x-coordinate.
        grp_var (string) - the name of a column that contains a grouping
            variable. Groups are colour-coded, and appear grouped together
            over each x-coordinate.
    
    Optional Args:
        grp_order (list-like) - the order of groups (left to right). If not 
            Upplied (value of None) then the order is determined by order of
            appearance in the data. (default: None)
        do_box (boolean) - display the boxplots? (default: True)
        do_pts (boolean) - display the jittered stripe plot of data points?
            (default: True)
        do_vio (boolean) -  display the distribution (half-violin) plot?
            (default: True)
        box_args (dict) - additional plotly boxplot options to be passed along
            to the boxplot subtraces (e.g., to overwrite defaults options)
        pts_args (dict) - additional options to be passed along to the stripe
            (scatter/pt) traces (plotly Scatter graphobject)
        mrk_args (dict) - additional options to be passed along to the marker
            options of the stripe (scatter/pt) traces.
        vio_args (dict) - additional options to be passed along to the 
            violin traces.
        pts_jitter (float)- the amount of jitter of the stripe plots, ranging from 0.0
            to 1.0 (the width allocated for that subplot). 
            (default: None - .75/n_grps)
        vio_jitter (boolean) - if True, staggers the half violin plots like the
            box and stripe plots. if False, they overlap.
        sym_offset (integer) - offests the symbols ID (default: 0)
        sym_constant (boolean) - should the symbol be the same for all groups?
            (default: False)
        layout_args (dict) - extra layout options passed to plotly.
        legend_args (dict) - extra legend options passed to plotly.

    Returns:
        fig - the figure.
    """

    n_x = len(plt_vars)
    grps = df[grp_var].unique()
    n_grps = len(grps)

    if grp_order is None:
        grp_order = list(grps)

    # Widths of the gap, points, box, and violin sections
    w_x = 0.25
    w_pt = 1. if do_pts else 0
    w_bx = 2. if do_box else 0
    w_vi = 1. if do_vio else 0

    w_panel = w_x + w_pt + w_bx + w_vi

    # Centres for each of the points, box, and violin sections
    c_pt = [w_panel*x + (w_x + w_pt/2.) for x in range(n_x)]
    c_bx = [w_panel*x + (w_x + w_pt + w_bx/2.) for x in range(n_x)]
    c_vi = [w_panel*x + (w_x + w_pt + w_bx) for x in range(n_x)]

    # Offsets for each group within each section
    o_pt = w_pt*0.5*(np.arange(0, n_grps)/(n_grps-1) - 0.5) if n_grps > 0 else [0]
    o_bx = w_bx*0.46*(np.arange(0, n_grps)/(n_grps-1) - 0.5) if n_grps > 0 else [0]
    o_vi = w_vi*0.5*(np.arange(0, n_grps)/(n_grps-1)) if n_grps > 0 else [0]

    jitter = .75/n_grps if pts_jitter is None else pts_jitter

    fig = go.Figure()
    w_plt = 0

    grp_colours = grp_colours[colour_offset:]

    # This loop simply adds invisible traces so we have nice legend items
    for ig, g in enumerate(grp_order):
        fig.add_trace(
            go.Scatter(
                x = [-np.inf], y = [-np.inf],
                visible=True,
                showlegend=True,
                name = g,
                mode = 'markers',
                marker = dict(
                    symbol = sym_offset if sym_constant else ig+sym_offset,
                    color = grp_colours[ig],
                    opacity = 0.5,
                    line = dict(
                        width = 2.,
                        color = grp_colours[ig]
                    ) 
                )
            )
        )

    for iv, v in enumerate(plt_vars):
        for ig, g in enumerate(grp_order):
            y_pt = df.loc[df[grp_var] == g, v]
            nd = y_pt.shape[0]

            # Create the stripe (scatter/pts) trace
            if do_pts:
                x_pt = np.repeat(c_pt[iv]+o_pt[ig], nd)
                x_pt += (np.random.rand(nd) - 0.5) * jitter
                mrk_opts = dict(
                    symbol = ig+sym_offset,
                    color = grp_colours[ig],
                    opacity = 0.5,
                    size = 5,
                )
                pts_opts = dict(
                    x = x_pt,
                    y = y_pt,
                    name = g,
                    mode = 'markers',
                    marker = {**mrk_opts, **mrk_args},
                    showlegend=False,
                )
                fig.add_trace(go.Scatter({**pts_opts, **pts_args}))

            # Create the box trace
            if do_box:
                x_bx = np.repeat(c_bx[iv]+o_bx[ig], nd)
                box_opts = dict(
                    x = x_bx,
                    y = y_pt,
                    name = g,
                    marker_color = grp_colours[ig],
                    boxpoints = False,
                    boxmean = False,
                    notched = False,
                    line = dict(
                        width=1,
                    ),
                    showlegend=False,
                )
                fig.add_trace(go.Box({**box_opts, **box_args}))
                w_plt = x_bx[0]

            # Create the distribution (half-violin) trace
            if do_vio:
                x_vi = np.repeat(c_vi[iv], nd).astype('float32')
                x_vi += o_vi[ig] if vio_jitter else 0
                vio_opts = dict(
                        x = x_vi,
                        y = y_pt,
                        name = g,
                        scalegroup = iv,
                        scalemode = 'count',
                        side = 'positive',
                        width = w_vi*2,
                        points = False,
                        spanmode = 'hard',
                        line = dict(
                            color = grp_colours[ig],
                            width = 1
                        ),
                        showlegend=False,
                )
                fig.add_trace(go.Violin({**vio_opts, **vio_args}))
                w_plt = x_vi[0]

    template = plotly_template()
    template['data']['scatter'] = []

    if legend_args is None:
        legend_args = rc_legend

    if layout_args is None:
        layout_args = rc_layout

    legend_opts = dict(
        title = dict(
            text = grp_var,
            side = 'top',
        ),
        itemsizing = 'constant',
        bgcolor = 'rgba(0,0,0,0)',
    )

    layout_opts = dict(
        template = template,
        margin = {'b': 75, 't': 20, 'r': 30},
        boxgap = 0.,
        boxgroupgap = 0.2,
        xaxis = dict(
            tickmode = 'array',
            tickvals = c_bx,
            ticktext = plt_vars,
            range = [0, w_plt+w_x*3]
        ),
        yaxis = dict(
            title = 'Score (SDs)',
            range = [-3.7, 3.6],
            tickmode = 'array',
            tickvals = np.arange(-3, 3+1),
            ticktext = [f"{y}  " for y in np.arange(-3, 3+1)]
        ),
        legend = {**legend_opts, **legend_args}
    )

    fig.update_layout({**layout_opts, **layout_args})

    return fig

def means_plot(
        df, vars, vars_name,
        bar_args={}, layout_args={}, trace_args={},
        group=None, group_order=None, 
        group_color_sequence=px.colors.sequential.Plasma,
        group_tests=False
    ):
    """ Plots the means of columns of a dataframe, split by a grouping variable.
        The x-axis will have collections (groups) of bars, where each group 
        corresponds to a single measure (IV), and each bar in the group is
        one value of the grouping variable.

        This plotting functionality builds upon Plotly's Express "Bar" plot
        https://plotly.com/python/bar-charts/
        
    Required Args:
        df (Pandas dataframe): a dataframe that contains the data to be plotted.
            columns are dependent variables (e.g., scores) and independent
            variables (e.g., group)
        vars (list-like): The variables to be plotted, e.g., a list of DVs. 
            Each element of this list will correspond to a group of bars in the
            final figure.
        vars_name (string): What do you call the collection of variables to
            be plotted? For example "Test Scores". I think this ends up as the
            x-axis title.

    Optional Args:
        bar_args (dict): formatting options for bars that get passed along to
            Plotly express.
        layout_args (dict): formatting options related to layout that get 
            passed along to Plotly. Anything specified in this dict overrides
            the default layout options specified at the top of this file.
        trace_args (dict):  formatting arguments applied to the Plotly traces.
        group (string): The name of the grouping variable.
        group_order (list): The order, left to right, that bars within a group
            should be ordered (and the legend, too).
        group_color_sequence (colour-like object): the colour sequence for
            levels of the he grouping variable.
        group_tests (boolean): if True, peforms F-tests (uncorrected) to test
            for differences between the levels of the grouping variable, for 
            each variable. THese are uncorrected stats.
            
    Returns:
        A Plotly figure
    
    """
    stats = ['mean', 'std', 'count']
    order = {vars_name: vars}
    if group is None:
        means = df[vars].agg(stats).T
        means.index.name = vars_name
        means.columns.name = 'stat'
    else:
        if group_order is not None:
            order = {**order, group: group_order}
            group_nms = group_order
        else:
            group_nms = list(df[group].unique())

        ngrps = len(group_nms)
        means = df[vars+[group]].groupby(group).agg(stats)
        means.columns.names = [vars_name, 'stat']
        means = means.stack(vars_name)
    
    means['mean_se'] = means['std']/np.sqrt(means['count'])

    f = px.bar(
            means.reset_index(),
            x=vars_name, y='mean', error_y='mean_se', 
            color=group,
            category_orders=order,
            color_discrete_sequence=group_color_sequence,
            barmode='group',
            **bar_args)
            
    if group_tests:
        for v in vars:
            p = f_1way_pval(df, group, v)
            if  p < 0.001:
                txt = "***"
            elif p < 0.01:
                txt = "**"
            elif p < 0.05:
                txt = "*"
            else:
                txt = ""
            f.add_annotation(
                x=v, text=txt, showarrow=False,
                y=means.loc[idx[:, v], :][['mean', 'mean_se']].sum(axis=1).max(),
                xanchor='center', yanchor='bottom')


    f.update_traces(**trace_args)
    f.update_layout(
        template=plotly_template(),
        **layout_args)
    
    return f, means

def qq_plots(qq_results, titles, marker_size=5, lims=[-4,4], layout_args={}):
    """ Creates QQ plots from the results of a SciPy probplot:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html

        Basically, creates a scatter plot for each QQ result, and arranges them
        in a MxN matrix of subplots.
            https://plotly.com/python/line-and-scatter/
            https://plotly.com/python-api-reference/generated/plotly.subplots.make_subplots.html

        
    Required Args:
        qq_results (ND-array): an 2D MxN array of results from probplot, where
            each element corresponds to the analysis of a single variable. The
            arrangement of these results maps on to the layout of the subplots
            in the figure.
        title (ND-array): Names for each subplot. This should have the same
            dimensions as qq_results.

    Optional Args:
        marker_size (numeric): Size of the datapoint markers in the scatter
            plot.
        lims (list-like, 2 elements): threshold range (-x to +x).
        layout_args (dict): layout options to override the defaults.
            
    Returns:
        A Plotly figure
    
    """
    from plotly.subplots import make_subplots
    assert(isinstance(qq_results, np.ndarray))
    assert(qq_results.shape == titles.shape)

    nrows, ncols = qq_results.shape

    fig = make_subplots(rows=nrows, cols=ncols, 
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing = 0.04, vertical_spacing = 0.04)

    ii = 1
    for ir in range(nrows):
        for ic in range(ncols):
            qq = qq_results[ir, ic]
            if qq is not None:
                fig.add_trace(
                    go.Scatter(
                        x=qq[0][0], y=qq[0][1],
                        mode='markers', 
                        marker={'size': marker_size, 'opacity': 0.5, 'color': 'black'},
                        showlegend=False,
                    ),
                    row=ir+1, col=ic+1,
                )

                xx = np.array(lims)
                yy = xx*qq[1][0] + qq[1][1]
                fig.add_trace(
                    go.Scatter(
                        x=xx, y=yy, 
                        mode='lines', line={'color': 'white', 'width': 2},
                        showlegend=False,
                    ),
                    row=ir+1, col=ic+1,
                )
                fig.update_xaxes(
                    range=lims, 
                    zerolinecolor='white', 
                    row=ir+1, col=ic+1
                )
                fig.update_yaxes(
                    range=lims, row=ir+1, col=ic+1
                )

                fig.add_annotation(
                    yanchor='top', xanchor='left', 
                    x=lims[0]+0.5, y=lims[1]-0.5,
                    xref=f"x{ii}", yref=f"y{ii}",
                    text=titles[ir, ic], showarrow=False
                )
            ii += 1

            if ic == 0:
                fig.update_yaxes(
                    title={'text': 'observed'},
                    row=ir+1, col=ic+1,
                )
            if ir == nrows-1:
                fig.update_xaxes(
                    title={'text': 'theoretical'},
                    row=ir+1, col=ic+1,
                )

    fig.update_layout(
        template=plotly_template(),
        margin={'b': 75, 't': 20, 'r': 30},
        **layout_args)

    return fig

def correlogram(
        df, subset=None, mask_diag=True, thresh=None, 
        width=475, height=350, colormap='Picnic', layout_args={}):
    """ Description here
    Args: 
        df (dataframe): The dataframe with rows as observations and columns
            as variables.
        subset (list-like): Which variables (columns) to subselect. If None, all
            columns are used. (default: None)
        
    """
    if subset is None:
        subset = df.columns

    r = df[subset].corr()

    if thresh is not None:
        df[np.abs(df)<thresh] = 0

    if mask_diag:
        np.fill_diagonal(r.values, 0)

    r = (r
        .stack()
        .reset_index()
        .rename(columns={'level_0': 'x', 'level_1': 'y', 0: 'r'})
    )

    f = px.scatter(r, x='x', y='y', size=np.abs(r['r']), 
        color='r', range_color=[-1,1], opacity = 1,
        color_continuous_scale=getattr(px.colors.diverging, colormap))

    f.update_layout(
        xaxis={'title': None},
        yaxis={'title': None},
        width=width, height=height,
        coloraxis={'colorbar': 
            {'thickness': 10, 'tickmode': 'array', 'tickvals': [-1, 0, 1],
            'title': 
                {'text': 'correlation (r)', 'side': 'right'}
            },
        },
        font={'size': 8},
        margin={'t': 20, 'r': 10, 'l': 80, 'b': 20},
        **layout_args)
    return f
