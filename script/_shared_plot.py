#!/usr/local/bin/python
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import ticker
import os

def rem_axLine(rem_list=['top','right'],axlw=0.4):
    ax=plt.gca()
    for loc, spine in ax.spines.items():
        if loc in rem_list:
            spine.set_position(('outward',0)) # outward by 10 points
            spine.set_linewidth(0.)
        else:
            spine.set_linewidth(axlw)
    return

def set_ax_font(axfs=11):
    locs,labels = plt.gca().get_xticks()
    plt.setp(labels,fontsize=axfs)
    locs,labels = plt.gca().get_yticks()
    plt.setp(labels,fontsize=axfs)
    return
def rotate_labels(which_ax='both',rot=0,axfs=6):
    ax=plt.gca()
    if which_ax == 'x' or which_ax=='both':
        locs,labels = plt.xticks()
        plt.setp(labels,rotation=rot,fontsize=axfs)
    if which_ax == 'y' or which_ax=='both':
        locs,labels = plt.yticks()
        plt.setp(labels,rotation=rot,fontsize=axfs)
    return

def rem_ticks(which_ax='both'):
    ax=plt.gca()
    if which_ax == 'x' or which_ax=='both':
        ax.set_xticklabels([])
        ax.xaxis.set_ticks_position("none")
    if which_ax == 'y' or which_ax=='both':
        ax.set_yticklabels([])
        ax.yaxis.set_ticks_position("none")
    return
def rem_ticklabels(which_ax='both'):
    ax=plt.gca()
    if which_ax == 'x' or which_ax=='both':
        ax.set_xticklabels([])
    if which_ax == 'y' or which_ax=='both':
        ax.set_yticklabels([])
    return
def put_ticks(nticks=5,which_ax='both',axlw=0.3):
    ticksfmt=plt.FormatStrFormatter('%.1f')
    ax=plt.gca()
    if which_ax == 'x':
        ax.xaxis.set_ticks_position('bottom')
        lines = ax.get_xticklines()
        labels = ax.get_xticklabels()
        for line in lines:
            line.set_marker(mpl.lines.TICKDOWN)
        # for label in labels:
        #     label.set_y(-0.02)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=nticks,min_n_ticks=nticks))
    #........................................
    if which_ax == 'y':
        ax.yaxis.set_ticks_position('left')
        lines = ax.get_yticklines()
        labels = ax.get_yticklabels()
        for line in lines:
            line.set_marker(mpl.lines.TICKLEFT)
            line.set_linewidth(axlw)
        '''
        for label in labels:
            label.set_x(-0.02)
        '''
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=nticks,min_n_ticks=nticks))
    if which_ax=='both':
        ax.yaxis.set_ticks_position('left')
        lines = ax.get_yticklines()
        labels = ax.get_yticklabels()
        for line in lines:
            line.set_marker(mpl.lines.TICKLEFT)
            line.set_linewidth(axlw)
        '''
        for label in labels:
            label.set_x(-0.02)
        '''
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=nticks,min_n_ticks=nticks))

        ax.xaxis.set_ticks_position('bottom')
        lines = ax.get_xticklines()
        labels = ax.get_xticklabels()
        for line in lines:
            line.set_marker(mpl.lines.TICKDOWN)
        '''
        for label in labels:
            label.set_y(-0.02)
        '''
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=nticks,min_n_ticks=nticks))

def ax_clr(axfs=7):
    ax=plt.gca()
    rem_axLine(rem_list=['top','right','left','bottom'])
    rem_ticks(which_ax='both')
def ax_clrX(axfs=7,axlw=0.3,nticks=3):
    ax=plt.gca()
    rem_axLine(rem_list=['top','right','bottom'])
    rem_ticks(which_ax='x')
    ax.tick_params(axis='y', labelsize=axfs)
    put_ticks(which_ax='y',axlw=0.3,nticks=nticks)
def ax_clrY(axfs=7,axlw=0.3,nticks=3):
    ax=plt.gca()
    rem_axLine(rem_list=['top','right','left'])
    rem_ticks(which_ax='y')
    put_ticks(which_ax='x',axlw=0.3,nticks=nticks)
    ax.tick_params(axis='x', labelsize=axfs)

def ax_clrXY(axfs=7,axlw=0.3,nticks=3):
    ax=plt.gca()
    rem_axLine(rem_list=['top','right'])
#    rem_ticks(which_ax='y')
    put_ticks(which_ax='y',axlw=0.3,nticks=nticks)
    ax.tick_params(axis='both', labelsize=axfs)

def ax_orig(axfs=7,axlw=0.3,nticks=3):
    ax=plt.gca()
    rem_axLine(rem_list=['top','right'],axlw=axlw)
    put_ticks(which_ax='both',axlw=0.3,nticks=nticks)
    ax.tick_params(axis='both',labelsize=axfs)



def get_colomap(cmap_nm,bounds__,lowp=0.05,hip=0.95):
    '''
    Get the list of colors from any official colormaps in mpl. It returns the number of colors based on the number of items in the bounds. Bounds is a list of boundary for each color.
    '''
    cmap__ = mpl.cm.get_cmap(cmap_nm)
    color_listv=np.linspace(lowp,hip,len(bounds__)-1)
    rgba_ = [cmap__(_cv) for _cv in color_listv]
    return(rgba_)
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def get_ticks(_bounds, nticks=10):
    from math import ceil
    length = float(len(_bounds))
    co_ticks=[]
    for i in range(nticks):
        co_ticks=np.append(co_ticks,_bounds[int(ceil(i * length / nticks))])
    return(co_ticks[1:])
def mk_colo_cont(axcol_,bounds__,cm2,cblw=0.1,cbrt=0,cbfs=9,nticks=10,cbtitle='',col_scale='linear',tick_locs=[],ex_tend='both',cb_or='horizontal',spacing= 'uniform'):
    '''
    Plots the colorbar to the axis given by axcol_. Uses arrows on two sides.
    '''

    axco1=plt.axes(axcol_)
    if col_scale == 'linear':
        cb=mpl.colorbar.ColorbarBase(axco1,cmap=cm2, norm=mpl.colors.BoundaryNorm(bounds__, cm2.N),boundaries=bounds__,orientation = cb_or,drawedges=False,extend=ex_tend,ticks=bounds__[1:-1],spacing=spacing)
        if len(tick_locs) == 0:
            tick_locator = ticker.MaxNLocator(nbins=nticks,min_n_ticks=nticks)
        else:
            tick_locator = ticker.FixedLocator(tick_locs)
        cb.locator = tick_locator
        cb.update_ticks()
    if col_scale == 'log':

        cb=mpl.colorbar.ColorbarBase(axco1,cmap=cm2, norm=mpl.colors.BoundaryNorm(bounds__, cm2.N), boundaries=bounds__,orientation = cb_or,extend=ex_tend,drawedges=False,ticks=bounds__[1:-1],spacing=spacing)
        if cb_or == 'horizontal':
            tick_locs_ori=cb.ax.xaxis.get_ticklocs()
        else:
            tick_locs_ori=cb.ax.yaxis.get_ticklocs()
        tick_locs_bn=[]
        for _tl in tick_locs:
            tlInd=np.argmin(np.abs(bounds__[1:-1]-_tl))
            tick_locs_bn=np.append(tick_locs_bn,tick_locs_ori[tlInd])
        if cb_or == 'horizontal':
            cb.ax.xaxis.set_ticks(tick_locs_bn)
            cb.ax.xaxis.set_ticklabels(tick_locs)
        else:
            cb.ax.yaxis.set_ticks(tick_locs_bn)
            cb.ax.yaxis.set_ticklabels(tick_locs)

    cb.ax.tick_params(labelsize=cbfs,size=2,width=0.3)
    ##hack the lines of the colorbar to make them white, the same color of background so that the colorbar looks broken.
    cb.outline.set_alpha(0.)
    cb.outline.set_color('white')
    cb.outline.set_linewidth(0*cblw)
    '''
    cb.dividers.set_linewidth(0*cblw)
    cb.dividers.set_alpha(1.0)
    cb.dividers.set_color('white')
    for ll in cb.ax.xaxis.get_ticklines():
        ll.set_alpha(0.)
    '''
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(cbfs)
        t.set_rotation(cbrt)

    for t in cb.ax.get_xticklabels():
        t.set_fontsize(cbfs)
        t.set_rotation(cbrt)
        t.set_y(-0.02)
    if cbtitle != '':
        cb.ax.set_title(cbtitle,fontsize=1.3*cbfs)
    # cb.update_ticks()
    return(cb)
def mk_colo_tau_c(axcol_,bounds__,cm2,cblw=0.1,cbrt=0,cbfs=9,nticks=10,cbtitle='',col_scale='linear',tick_locs=[],ex_tend='both',cb_or='horizontal',spacing= 'uniform'):
    '''
    Plots the colorbar to the axis given by axcol_. Uses arrows on two sides.
    '''

    axco1=plt.axes(axcol_)
    tick_locator = ticker.FixedLocator(tick_locs)
    cb=mpl.colorbar.ColorbarBase(axco1,cmap=cm2, norm=mpl.colors.BoundaryNorm(bounds__, cm2.N),boundaries=bounds__,orientation = cb_or,drawedges=False,extend=ex_tend,ticks=tick_locs,spacing=spacing)
    cb.ax.tick_params(labelsize=cbfs,size=2,width=0.3)
    ##hack the lines of the colorbar to make them white, the same color of background so that the colorbar looks broken.
    cb.outline.set_alpha(0.)
    cb.outline.set_color('white')
    cb.outline.set_linewidth(0*cblw)
    '''
    cb.dividers.set_linewidth(0*cblw)
    cb.dividers.set_alpha(1.0)
    cb.dividers.set_color('white')
    for ll in cb.ax.xaxis.get_ticklines():
        ll.set_alpha(0.)
    '''
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(cbfs)
        t.set_rotation(cbrt)

    for t in cb.ax.get_xticklabels():
        t.set_fontsize(cbfs)
        t.set_rotation(cbrt)
        t.set_y(-0.02)
    if cbtitle != '':
        cb.ax.set_title(cbtitle,fontsize=1.3*cbfs)
    cb.update_ticks()
    return(cb)
