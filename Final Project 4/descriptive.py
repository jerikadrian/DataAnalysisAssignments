import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display,Markdown

def corrcoeff(x, y):
    r = np.corrcoef(x, y)[0,1]
    return r

def plot_regression_line(ax, x, y, **kwargs):
    a,b   = np.polyfit(x, y, deg=1)
    x0,x1 = min(x), max(x)
    y0,y1 = a*x0 + b, a*x1 + b
    ax.plot([x0,x1], [y0,y1], **kwargs)

def final_descriptive_plot(df):

    y     = df['total_energy']
    t_in  = df['t_in']
    t_out = df['t_out']
    h_in  = df['h_in']
    h_out = df['h_out']
    wind  = df['windspeed']
    vis   = df['visibility']

    fig,axs = plt.subplots( 2, 3, figsize=(10,6), tight_layout=True )
    ivs     = [t_in, t_out, h_in, h_out, wind, vis]
    colors  = 'b', 'r', 'g', 'y', 'orange', 'violet'
    for ax,x,c in zip(axs.flat, ivs, colors):
        ax.scatter( x, y, alpha=0.5, color=c )
        plot_regression_line(ax, x, y, color='k', ls='-', lw=2)
        r   = corrcoeff(x, y)
        ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))

    xlabels = 'Indoor Temperature', 'Outdoor Temperature', 'Indoor Humidity', 'Outdoor Humidity', 'Windspeed', 'Visibility'
    [ax.set_xlabel(s) for ax,s in zip(axs.flat,xlabels)]
    axs.flat[0].set_ylabel('Energy')
    [ax.set_yticklabels([])  for ax in axs.flat[1:]]

    fig.text(0.5, -0.05,
             "Figure 1. Correlations amongst main variables.",
             ha='center', va='center', fontsize=12)

def descriptive( df ):
    
    final_descriptive_plot(df)