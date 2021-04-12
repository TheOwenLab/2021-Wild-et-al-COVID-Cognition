import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from lib_stats import f_1way_pval
from lib_colours import D1_CMAP, D2_CMAP, D3_CMAP, D4_CMAP

idx = pd.IndexSlice

# Default settings for matplotlib
from matplotlib import rc
plt.rcParams['figure.dpi'] = 100
plt.rcParams.update({'font.size': 10})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rcParams['svg.fonttype'] = 'none'

_LINE_COLOUR = 'rgb(16, 16, 16)'
def plotly_template():
    return {
        'layout': go.Layout(
            plot_bgcolor = 'rgba(1,1,1,0.1)',
            font_family = 'sans-serif',
            font = {'size': 10},
            xaxis = {
                'zeroline': True,
                'zerolinecolor': _LINE_COLOUR,
                'zerolinewidth': 1,
                'gridcolor': 'white',
                'gridwidth': 1
            },
            yaxis = {
                'zeroline': True, 
                'zerolinewidth': 1,
                'gridwidth': 1,
                'zerolinecolor': 'white',
                'gridcolor': 'white',
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

def create_stats_figure(
        results, stat_name, p_name, alpha=0.05, log_stats=True, 
        diverging=False, stat_range=None, correction=None, vertline=4, 
        marker_color=None, reverse=False
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
        correction (string): indicates how the alpha was corrected (e.g., FDR 
            or bonferroni) so the legend can be labelled appropriately.
            
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
    """ Creates a matrix figure to summarize Bayesian stats for multiple scores & tests.
        Each cell indicates the Bayes Factor (BF associated with a model comparison) for 
        a specific effect (rows) for a given score (columns). Also draws symbols on cells
        to indicate the interpretation of that BF.
        
    Args:
        results (Pandas dataframe): a dataframe that contains the statistics to display. Should
            be a rectangular dataframe with tests as rows and effects as columns (i.e., the 
            transpose of the resulting image). The dataframe index and column labels are used
            as labels for the resulting figure.
        log_stats (boolean): Should we take the logarithm of BF values before creating 
            the image? Probably yes, if there is a large variance in value across scores and
            effects.
            
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

def means_plot(
        df, vars, vars_name,
        bar_args={}, layout_args={}, trace_args={},
        group=None, group_order=None, 
        group_color_sequence=px.colors.sequential.Plasma,
        group_tests=False, bar_tests=False, bar_correction='group',
    ):

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
    """ Assumes a A x B matrix of results from a statsmodels probplot function.
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
