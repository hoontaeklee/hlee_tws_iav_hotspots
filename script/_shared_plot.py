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