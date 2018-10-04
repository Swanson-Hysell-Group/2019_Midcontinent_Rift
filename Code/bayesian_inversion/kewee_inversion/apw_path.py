from __future__ import print_function
import os, sys
import numpy as np
import scipy.stats as st
import pandas as pd

sys.path.append(os.path.abspath('../mcplates'))
import mcplates

# Shift all longitudes by 180 degrees to get around some plotting
# issues. This is error prone, so it should be fixed eventually
lon_shift = 180.

slat = 46.8  # Duluth lat
slon = 360. - 92.1 - lon_shift  # Duluth lon
duluth = mcplates.PlateCentroid(slon, slat)


def create_model(n_euler_rotations, use_tpw):
    if n_euler_rotations < 0:
        raise Exception("Number of plate Euler rotations must be greater than or equal to zero")
    if use_tpw != False and use_tpw != True:
        raise Exception("Must enter 'true' or 'false' for whether to use TPW")
    if n_euler_rotations == 0 and use_tpw == False:
        raise Exception("Must use either TPW or plate Euler rotations, or both")

    print("Fitting Keweenawan APW track with"\
            +("out TPW and " if use_tpw == False else " TPW and ")\
            +str(n_euler_rotations)+" Euler rotation"\
            + ("" if n_euler_rotations == 1 else "s") )


    data = pd.read_csv("pole_means.csv")
    # Give unnamed column an appropriate name
    data.rename(columns={'Unnamed: 0': 'Name',\
                         'Unnamed: 14': 'GaussianOrUniform'},\
                inplace=True)
    data = data[data.Name != 'Osler_N'] #Huge error, does not contribute much to the model
    data = data[data.PoleName != 'Abitibi'] # Standstill at the beginning, not realistic to fit
    data = data[data.PoleName != 'Haliburton'] #Much younger, far away pole, difficutlt to fit
    data.sort_values('AgeNominal', ascending=False, inplace=True)

    poles = []
    pole_names = []
    pole_colors = []
    for i, row in data.iterrows():
        pole_lat = row['PLat']
        pole_lon = row['PLon'] - lon_shift
        a95 = row['A95']
        age = row['AgeNominal']

        if row['GaussianOrUniform'] == 'gaussian':
            sigma_age = row['Gaussian_2sigma'] / 2.
        elif row['GaussianOrUniform'] == 'uniform':
            sigma_age = (row['AgeLower'], row['AgeUpper'])
        else:
            raise Exception("Unrecognized age error type")

        pole = mcplates.PaleomagneticPole(
            pole_lon, pole_lat, angular_error=a95, age=age, sigma_age=sigma_age)
        poles.append(pole)
        pole_names.append(row['PoleName'])
        pole_colors.append(row['color'])

    tpw_str = 'true' if use_tpw else 'false'
    prefix = 'keweenawan_'+str(n_euler_rotations)+'_'+tpw_str
    path = mcplates.APWPath(prefix, poles, n_euler_rotations)
    tpw_rate_scale = 2.5 if use_tpw else None
    path.create_model(site_lon_lat=(slon, slat), watson_concentration=0.0,\
            rate_scale=2.5, tpw_rate_scale=tpw_rate_scale)
    return path, poles, pole_names, pole_colors

def load_or_sample_model(path):
    if os.path.isfile(path.dbname):
        print("Loading MCMC results from disk...")
        path.load_mcmc()
        print("Done")
    else:
        path.sample_mcmc(2000000)

if __name__ == "__main__":
    # Parse input
    #Get number of euler rotations
    if len(sys.argv) < 3:
        raise Exception("Please enter the number of Euler rotations to fit, and 'true' or 'false' for whether to include TPW")
    n_euler_rotations = int(sys.argv[1])
    use_tpw = False if sys.argv[2] == 'false' else True;
    path, poles, pole_names, pole_colors = create_model(n_euler_rotations, use_tpw)
    load_or_sample_model(path)
