#%%
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import camb
from cosmology.compat.camb import Cosmology
import glass
import glass.ext.camb
from tqdm import tqdm



def lensing_cls_sim(cosmo_params, angular_modes=256, shell_spacing=200.0, random_seed=42): 
    '''
    Configs and Cosmological Params
    '''
    # creating a numpy random number generator for sampling
    rng = np.random.default_rng(seed=random_seed)

    # basic parameters of the simulation
    nside = lmax = angular_modes  # HEALPix resolution and max angular mode number

    # unpack cosmological parameters
    assert isinstance(cosmo_params, dict), "cosmo_params must be a dictionary"
    h = cosmo_params['h']
    Oc = cosmo_params['Oc']
    Ob = cosmo_params['Ob']

    # set up CAMB parameters for matter angular power spectrum
    pars = camb.set_params(
        H0=100 * h,
        omch2=Oc * h**2,
        ombh2=Ob * h**2,
        NonLinear=camb.model.NonLinear_both,
    )
    results = camb.get_background(pars)

    # get the cosmology from CAMB
    cosmo = Cosmology(results)

    # shells of 200 Mpc in comoving distance spacing
    zb = glass.distance_grid(cosmo, 0.0, 1.0, dx=shell_spacing)

    # linear radial window functions
    shells = glass.linear_windows(zb)

    # compute the angular matter power spectra of the shells with CAMB
    cls = glass.ext.camb.matter_cls(pars, lmax, shells)

    # apply discretisation to the full set of spectra:
    # - HEALPix pixel window function (`nside=nside`)
    # - maximum angular mode number (`lmax=lmax`)
    # - number of correlated shells (`ncorr=3`)
    cls = glass.discretized_cls(cls, nside=nside, lmax=lmax, ncorr=3)


    '''
    Matter sector
    '''
    # set up lognormal fields for simulation
    fields = glass.lognormal_fields(shells)

    # compute Gaussian spectra for lognormal fields from discretised spectra
    gls = glass.solve_gaussian_spectra(fields, cls)

    # generator for lognormal matter fields
    matter = glass.generate(fields, gls, nside, ncorr=3, rng=rng)


    '''
    Lensing sector
    '''
    # this will compute the convergence field iteratively
    convergence = glass.MultiPlaneConvergence(cosmo)


    '''
    Galaxies sector
    '''
    #localised redshift distribution
    # the actual density per arcmin2 does not matter here, it is never used
    z = np.linspace(0.0, 1.0, 101)
    dndz = np.exp(-((z - 0.5) ** 2) / (0.1) ** 2)

    # distribute dN/dz over the radial window functions
    ngal = glass.partition(z, dndz, shells)


    '''
    Simulation
    '''
    # the integrated convergence and shear field over the redshift distribution
    kappa_bar = np.zeros(12 * nside**2)
    gamm1_bar = np.zeros(12 * nside**2)
    gamm2_bar = np.zeros(12 * nside**2)

    # main loop to simulate the matter fields iterative
    for i, delta_i in tqdm(enumerate(matter)):
        # add lensing plane from the window function of this shell
        convergence.add_window(delta_i, shells[i])

        # get convergence field
        kappa_i = convergence.kappa

        # compute shear field
        gamm1_i, gamm2_i = glass.shear_from_convergence(kappa_i)

        # add to mean fields using the galaxy number density as weight
        kappa_bar += ngal[i] * kappa_i
        gamm1_bar += ngal[i] * gamm1_i
        gamm2_bar += ngal[i] * gamm2_i

    # normalise mean fields by the total galaxy number density
    kappa_bar /= ngal.sum()
    gamm1_bar /= ngal.sum()
    gamm2_bar /= ngal.sum()

    '''
    # Sample galaxies from the lensing fields for angular power spectra estimation
    '''
    # get the angular power spectra of the lensing maps
    sim_cls = hp.anafast(
        [kappa_bar, gamm1_bar, gamm2_bar],
        pol=True,
        lmax=lmax,
        use_pixel_weights=True,
    )

    return ((h, Oc, Ob), sim_cls[0])

#%%
c_params = {'h':0.7, 'Oc':0.3, 'Ob':0.04}
test = lensing_cls_sim(c_params)
print(test[0])
print(type(test[1]))
print(test[1])
print(test[1].shape)




























