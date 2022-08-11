# -----------------------------------------------------------------------------
# This lib file contains custom code for generating the "chord" plots that
# make up Figures 1A & 1B in the manuscript. The code in here should be cleaned
# up and better documented, but I haven't had time for that yet. For example,
# there are colour-related functions in here that exist elsewhere. Note, this
# script has been hacked together using bits and pieces from:
# - https://plotly.com/python/v3/filled-chord-diagram/
# - https://plotly.com/python/v3/chord-diagram/
# -----------------------------------------------------------------------------
# cwild 2021-04-15

import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.colors

def hex_to_rgb(h, as_str=False):
    h = h[1:]
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return str(rgb) if as_str else rgb

def rbga_str(rgb, a):
    if isinstance(rgb, str):
        from ast import literal_eval as make_tuple
        rgb = make_tuple(rgb[4:])
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{a})"

def chord_plot(
        loadings, 
        corrs, 
        height=400, width=400,
        cscale_name='Tropic',
        threshold=0.0,
        do_labels=True):
        
    n_arcs = loadings.shape[0]
    n_ring = loadings.shape[1]

    arc_colours = plotly.colors.qualitative.Alphabet[0:n_arcs]

    ring_colours = [hex_to_rgb(arc_colours[i]) for i in range(n_ring)]
    #%%
    PI = np.pi
    gap = 2 * PI * 0.005
    arc_len = (2 * PI - gap * n_arcs)/ n_arcs

    arc_ends = [(arc_len*i+gap*i, arc_len*(i+1)+gap*i) for i in range(n_arcs)]
    arc_mids = [np.mean(np.array(ends)) for ends in arc_ends]

    def moduloAB(x, a, b): #maps a real number onto the unit circle identified with 
                        #the interval [a,b), b-a=2*PI
            if a>=b:
                raise ValueError('Incorrect interval ends')
            y=(x-a)%(b-a)
            return y+b if y<0 else y+a

    def test_2PI(x):
        return 0<= x <2*PI

    def make_arc(R, phi, a=50):
        # R is the circle radius
        # phi is the list of ends angle coordinates of an arc
        # a is a parameter that controls the number of points to be evaluated on an arc
        if not test_2PI(phi[0]) or not test_2PI(phi[1]):
            phi=[moduloAB(t, 0, 2*PI) for t in phi]
        length=(phi[1]-phi[0])% 2*PI
        nr=5 if length<=PI/4 else int(a*length/PI)

        if phi[0] < phi[1]:
            theta=np.linspace(phi[0], phi[1], nr)
        else:
            phi=[moduloAB(t, -PI, PI) for t in phi]
            theta=np.linspace(phi[0], phi[1], nr)
        return R*np.exp(1j*theta)

    def make_arc_shape(path, line_color, fill_color):
        #line_color is the color of the shape boundary
        #fill_collor is the color assigned to an ideogram
        return  dict(
                    line=dict(
                    color=line_color,
                    width=0.45
                    ),

                path=  path,
                type='path',
                fillcolor=fill_color,
                layer='below'
            )

    axis=dict(
            showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

    layout = go.Layout(
                title=None,
                xaxis=dict(axis),
                yaxis=dict(axis),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                width=width,
                height=height,
                margin=dict(t=10, b=10, l=10, r=10),
                hovermode='closest',
                scene=dict(aspectmode="data"),
                shapes=[])
                

    arcs = []
    shapes = []
    ring_gap = 0.03
    ring_wid = 0.15

    cscale = getattr(plotly.colors.diverging, cscale_name)
    nscale = np.linspace(0, 1, len(cscale))
    cscale = [[nscale[i], cscale[i]] for i in range(len(cscale))]

    xpts = []
    ypts = []
    for r in range(n_ring):
        for k in range(n_arcs):
            load = loadings.iloc[k, r]
            arc_c = get_continuous_color(cscale, (load+1)/2) #rbga_str(ring_colours[r], np.abs(loadings.iloc[k, r]))
            if np.abs(load) < threshold:
                arc_c = 'rgba(0,0,0,0)'

            if r == 0:
                pt = 1 * np.exp(1j*arc_mids[k])
                xpts.append(pt.real)
                ypts.append(pt.imag)

            z= make_arc(1.0 + (r+1)*ring_wid - ring_gap, arc_ends[k])
            zi=make_arc(1.0 + (r*ring_wid), arc_ends[k])
            m=len(z)
            n=len(zi)
            arcs.append(
                go.Scatter(
                    x=z.real,
                    y=z.imag,
                    mode='lines',
                    line=dict(color=arc_c, shape='spline', width=0.25),
                    text=loadings.index[k]+f"<br>load: {load:.3f}",
                    hoverinfo='text'))

            path='M '
            for s in range(m):
                path+=str(z.real[s])+', '+str(z.imag[s])+' L '

            Zi=np.array(zi.tolist()[::-1])

            for s in range(m):
                path+=str(Zi.real[s])+', '+str(Zi.imag[s])+' L '
            path+=str(z.real[0])+' ,'+str(z.imag[0])

            shapes.append(make_arc_shape(path,'rgb(150,150,150)' , arc_c))

    pts = pd.DataFrame({'x': xpts, 'y':ypts}, index=loadings.index)
    from scipy.spatial.distance import squareform, pdist
    dist = pdist(pts)

    corrs_ = corrs.copy() #UGH
    np.fill_diagonal(corrs_.values, 0)
    corr = squareform(corrs_)

    ix, iy = np.triu_indices(pts.shape[0], k=1)

    def distd (A,B):
        return np.linalg.norm(np.array(A)-np.array(B))
    dist_bins = [0, distd([1,0], 2*[np.sqrt(2)/2]), np.sqrt(2),
                    distd([1,0],  [-np.sqrt(2)/2, np.sqrt(2)/2]), 2.0]
    # params = [2.0, 2.5, 3.0, 4.0]
    params=[p+0.2 for p in [1.2, 1.5, 1.8, 2.1]]

    class InvalidInputError(Exception):
        pass

    def deCasteljau(b,t):
        N=len(b)
        if(N<2):
            raise InvalidInputError("The  control polygon must have at least two points")
        a=np.copy(b) #shallow copy of the list of control points 
        for r in range(1, N):
            a[:N-r,:]=(1-t)*a[:N-r,:]+t*a[1:N-r+1,:]
        return a[0,:]

    def BezierCv(b, nr=5):
        t=np.linspace(0, 1, nr)
        return np.array([deCasteljau(b, t[k]) for k in range(nr)])

    def get_idx_interv(d, D):
        k=0
        while(d>D[k]):
            k+=1
        return  k-1

    max_r  = np.max(np.abs(corr))
    for ii in range(ix.shape[0]):
        K=get_idx_interv(dist[ii], dist_bins)
        A = pts.iloc[ix[ii], :]
        B = pts.iloc[iy[ii], :]
        b=[A, A/params[K], B/params[K], B]
        # color=edge_colors[K]
        crv = BezierCv(b, nr=20)
        abs_r = np.abs(corr[ii])
        clr = get_continuous_color(cscale, (corr[ii]/max_r+1)/2)
        clr = "rgba("+clr[4:-1]+f",{abs_r})"
        if abs_r < threshold:
            clr = "rgba(0,0,0,0)"
        arcs.append(
            go.Scatter(
                x=crv[:, 0], y=crv[:, 1], 
                mode='lines',
                hoverinfo='text',
                hovertext=f"{pts.index[ix[ii]]}, {pts.index[iy[ii]]}<br>r = {corr[ii]:.3f}",
                line=dict(
                    shape='spline',
                    width=20*abs_r,
                    color=clr)))

    layout = layout.update(shapes=shapes)
    layout = layout.update(xaxis=dict(constrain='domain'))
    layout = layout.update(yaxis=dict(scaleanchor='x'))
    fig = go.Figure(data=arcs, layout=layout)

    if do_labels:
        angle_bins = [0, 9*PI/20, 11*PI/20, PI, 29*PI/20, 31*PI/20, 2*PI]
        text_align = [
            ('left', 'middle'),
            ('center', 'bottom'),
            ('right', 'middle'),
            ('right', 'middle'),
            ('center', 'top'),
            ('left', 'middle')]

        for k in range(n_arcs):    
            pt = (1+ring_gap+(ring_wid*n_ring)) * np.exp(1j*arc_mids[k])
            x, y = pt.real, pt.imag
            xanch, yanch = text_align[get_idx_interv(arc_mids[k], angle_bins)]
            fig.add_annotation(
                x=pt.real, y=pt.imag,
                text=loadings.index[k],
                # textangle=-1*arc_mids[k] * 180 / PI,
                xanchor=xanch,
                yanchor=yanch,
                showarrow=False
            )
    
    colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=getattr(plotly.colors.diverging, cscale_name), 
            showscale=True,
            cmin=-1, cmax=1,
            colorbar=dict(
                title=dict(
                    text='correlation (r)',
                    side='right'),
                xanchor='left', x=0.08,
                yanchor='middle', y=0.5,
                lenmode='fraction', len=0.6,
                thickness=15,
                tickvals=[-1, 0, 1],
                outlinewidth=0)),
        hoverinfo='none'
    )

    fig.add_trace(colorbar_trace)
    fig.update_layout(
        font_family = 'Avenir',
        font = {'size': 10}
    )

    return fig

def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
    
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]
    if intermed >= 1:
        return colorscale[-1][1]

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    low_color = hex_to_rgb(low_color, as_str=True) if low_color[0]=="#" else low_color
    high_color = hex_to_rgb(high_color, as_str=True) if high_color[0]=="#" else high_color

    # noinspection PyUnboundLocalVariable
    return plotly.colors.find_intermediate_color(
        lowcolor=low_color, highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb")

# %%
