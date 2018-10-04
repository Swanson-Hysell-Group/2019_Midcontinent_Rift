from __future__ import print_function
import itertools, os, sys
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

import cartopy.crs as ccrs

sys.path.append(os.path.abspath('../mcplates'))
import mcplates
from pymc.utils import hpd

plt.style.use('../bpr.mplstyle')
from mcplates.plot import cmap_red, cmap_green, cmap_blue

# Shift all longitudes by 180 degrees to get around some plotting
# issues. This is error prone, so it should be fixed eventually
lon_shift = 180.

# List of colors to use
dist_colors_short = ['darkblue', 'darkred', 'darkgreen']

# Used for making a custom legend for the plots
class LegendHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        c = orig_handle
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        x1, y1 = handlebox.width, handlebox.height
        x = (x0+x1)/2.
        y = (y0+y1)/2.
        r = min((x1-x0)/2., (y1-y0)/2.)
        patch = mpatches.Circle((x, y), 2.*r, facecolor=c,
                                alpha=0.5, lw=0,
                                transform=handlebox.get_transform())
        point = mpatches.Circle((x, y), r/2., facecolor=c,
                                alpha=1.0, lw=0,
                                transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        handlebox.add_artist(point)
        return patch,point

def plot_synthetic_paths(path, poles, pole_colors, ax, title=''):
    ax.gridlines()
    ax.set_global()

    direction_samples = []
    if path.n_euler_rotations > 0:
        direction_samples = path.euler_directions()
    if path.include_tpw:
        tpw_samples = path.tpw_poles()

    if path.n_euler_rotations == 1:
        dist_colors = itertools.cycle([cmap_red])
    elif path.n_euler_rotations == 2:
        dist_colors = itertools.cycle([cmap_green, cmap_red])

    for directions in direction_samples[::-1]:
        mcplates.plot.plot_distribution(ax, directions[:, 0], directions[:, 1], cmap=next(dist_colors), resolution=60)

    if path.include_tpw:
        mcplates.plot.plot_distribution(ax, tpw_samples[:, 0], tpw_samples[:, 1], cmap=cmap_blue, resolution=60)

    pathlons, pathlats = path.compute_synthetic_paths(n=20)
    for pathlon, pathlat in zip(pathlons, pathlats):
        ax.plot(pathlon, pathlat, transform=ccrs.PlateCarree(),
                  color='b', alpha=0.05, lw=1)

    colorcycle = itertools.cycle(pole_colors)
    for p in poles:
        p.plot(ax, color=next(colorcycle))

    if title != '':
        ax.set_title(title)


def plot_age_samples(path, poles, pole_colors, ax1, ax2, title1='', title2=''):

    colorcycle = itertools.cycle(pole_colors)
    for p, age_samples in zip(poles, path.ages()):
        c = next(colorcycle)
        age = np.linspace(1070, 1115, 1000)
        if p.age_type == 'gaussian':
            dist = st.norm.pdf(age, loc=p.age, scale=p.sigma_age)
        else:
            dist = st.uniform.pdf(age, loc=p.sigma_age[
                                  0], scale=p.sigma_age[1] - p.sigma_age[0])
        ax1.fill_between(age, 0, dist, color=c, alpha=0.6)
        ax2.hist(age_samples, color=c, normed=True, alpha=0.6)
    ax1.set_ylim(0., 1.)
    ax2.set_ylim(0., 1.)
    ax1.set_xlim(1070, 1115)
    ax2.set_xlim(1070, 1115)
    ax2.set_xlabel('Age (Ma)')
    ax1.set_ylabel('Prior probability')
    ax2.set_ylabel('Posterior probability')
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'))
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'))


    if title1 != '':
        ax1.set_title(title1)
    if title2 != '':
        ax2.set_title(title2)

def plot_synthetic_poles(path, poles, pole_colors, ax, title=''):

    ax.gridlines()

    colorcycle = itertools.cycle(pole_colors)
    lons, lats, ages = path.compute_synthetic_poles(n=50)
    for i in range(len(poles)):
        c = next(colorcycle)
        poles[i].plot(ax, color=c)
        ax.scatter(lons[:, i], lats[:, i], color=c,
                     transform=ccrs.PlateCarree(), s=2)

    if title != '':
        ax.set_title(title)

def plot_changepoints(path, ax, title=''):

    changepoints = path.changepoints()

    ax.set_xlabel('Changepoint (Ma)')
    ax.set_ylabel('Probability density')

    xmin= 1.e10
    xmax=0.0
    colorcycle = itertools.cycle( dist_colors_short )
    for i, change in enumerate(changepoints):

        c = next(colorcycle)

        #plot histogram
        ax.hist(change, bins=30, normed=True, alpha=0.5, color=c, label='Changepoint %i'%(i))

        # plot median, credible interval
        credible_interval = hpd(change, 0.05)
        median = np.median(change)
        print("Changepoint %i: median %f, credible interval "%(i, median), credible_interval)
        ax.axvline( median, lw=2, color=c )
        ax.axvline( credible_interval[0], lw=2, color=c, linestyle='dashed')
        ax.axvline( credible_interval[1], lw=2, color=c, linestyle='dashed')

        xmin = max(0., min( xmin, median - 2.*(median-credible_interval[0])))
        xmax = max( xmax, median + 2.*(credible_interval[1]-median))

    if path.n_euler_rotations > 2:
        ax.legend(loc='upper right')
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'))

    if title != '':
        ax.set_title(title)

def plot_plate_speeds(path, poles, ax, title = ''):

    # Load a series of points for Laurentia
    laurentia_data = np.loadtxt('Laurentia_lon_lat.csv', skiprows=1, delimiter=',')
    laurentia_lon = laurentia_data[:,2] - lon_shift
    laurentia_lat = laurentia_data[:,1]

    direction_samples = []
    rate_samples = []
    if path.n_euler_rotations > 0:
        direction_samples = path.euler_directions()
        rate_samples = path.euler_rates()
    if path.include_tpw:
        direction_samples.insert(0, path.tpw_poles())
        rate_samples.insert(0, path.tpw_rates())

    # Get a list of intervals for the rotations
    if path.n_euler_rotations > 1:
        changepoints = [ np.median(c) for c in path.changepoints() ]
    else:
        changepoints = []
    age_list = [p.age for p in poles]
    changepoints.insert( 0, max(age_list) )
    changepoints.append( min(age_list) )

    ax.set_xlabel('Plate speed (cm/yr)')
    ax.set_ylabel('Probability density')

    xmin = 1000.
    xmax = 0.
    colorcycle = itertools.cycle( dist_colors_short )
    if path.include_tpw == False:
        next(colorcycle)

    for i, (directions, rates) in enumerate(zip(direction_samples, rate_samples)):

        #comptute plate speeds
        speed_samples = np.empty_like(rates)
        for j in range(len(rates)):
            euler = mcplates.EulerPole(
                directions[j, 0], directions[j, 1], rates[j])
            speed_sample = 0.
            for slon, slat in zip(laurentia_lon, laurentia_lat):
                loc = mcplates.PlateCentroid(slon, slat)
                speed = euler.speed_at_point(loc)
                speed_sample += speed*speed/len(laurentia_lon)

            speed_samples[j] = np.sqrt(speed_sample)

        c = next(colorcycle)

        #plot histogram
        if path.include_tpw and i == 0:
            hist_label = 'TPW'
        elif path.include_tpw and i != 0:
            hist_label = '%i - %i Ma'%(changepoints[i-1], changepoints[i])
        else:
            hist_label = '%i - %i Ma'%(changepoints[i], changepoints[i+1])
        ax.hist(speed_samples, bins=30, normed=True, alpha=0.5, color=c, label=hist_label)

        # plot median, credible interval
        credible_interval = hpd(speed_samples, 0.05)
        median = np.median(speed_samples)
        print("Rotation %i: median %f, credible interval "%(i, median), credible_interval)
        ax.axvline( median, lw=2, color=c )
        ax.axvline( credible_interval[0], lw=2, color=c, linestyle='dashed')
        ax.axvline( credible_interval[1], lw=2, color=c, linestyle='dashed')

        xmin = max(0., min( xmin, median - 2.*(median-credible_interval[0])))
        xmax = max( xmax, median + 2.*(credible_interval[1]-median))

    if len(rate_samples) > 1:
        ax.legend(loc='upper right')
    ax.set_xlim(0, 40)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'))
    tick_interval = 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))

    if title != '':
        ax.set_title(title)

def make_legend(pole_names, pole_colors, ax, title):
    # Make a custom legend
    import textwrap
    colorcycle = itertools.cycle(pole_colors)
    color_list = [ next(colorcycle) for p in pole_names]
    legend_names = [ '\n'.join(textwrap.wrap(name, 35)) for name in pole_names]
    legend = ax.legend(color_list, legend_names, fontsize=7.5, loc='center', ncol=2,
                frameon=False, framealpha=1.0, handler_map={str: LegendHandler()})
    legend.get_frame().set_facecolor('white')

    ax.axis('off')
    if title != '':
        ax.set_title(title)
