#%%
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import camb
from cosmology import Cosmology
import glass
import glass.ext.camb
from tqdm import tqdm

#%%
def glxy_simulation(h, Oc, Ob, 
                    shell_spacing=200.0, 
                    random_seed=42): 
    '''
    Configs and Params
    '''
    # creating a numpy random number generator for sampling
    rng = np.random.default_rng(random_seed)
    # basic parameters of the simulation
    nside = lmax = 256
    # set up CAMB parameters for matter angular power spectrum
    pars = camb.set_params(
    H0=100 * h,
    omch2=Oc * h**2,
    ombh2=Ob * h**2,
    NonLinear=camb.model.NonLinear_both,
    )
    # get the cosmology from CAMB
    cosmo = Cosmology.from_camb(pars)

    '''
    Matter sector
    '''
    # shells of 200 Mpc in comoving distance spacing
    zb = glass.distance_grid(cosmo, 0.0, 3.0, dx=shell_spacing)
    # linear window functions for shells
    shells = glass.linear_windows(zb)
    # compute the angular matter power spectra of the shells with CAMB
    cls = glass.ext.camb.matter_cls(pars, lmax, shells)
    # apply discretisation to the full set of spectra:
    # - HEALPix pixel window function (`nside=nside`)
    # - maximum angular mode number (`lmax=lmax`)
    # - number of correlated shells (`ncorr=3`)
    cls = glass.discretized_cls(cls, nside=nside, lmax=lmax, ncorr=3)
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
    # galaxy density (using 1/100 of the expected galaxy number density for Stage-IV)
    n_arcmin2 = 0.3
    # true redshift distribution following a Smail distribution
    z = np.arange(0.0, 3.0, 0.01)
    dndz = glass.smail_nz(z, z_mode=0.9, alpha=2.0, beta=1.5)
    dndz *= n_arcmin2
    # distribute dN/dz over the radial window functions
    ngal = glass.partition(z, dndz, shells)
    # compute tomographic redshift bin edges with equal density
    nbins = 10
    zbins = glass.equal_dens_zbins(z, dndz, nbins=nbins)
    # photometric redshift error
    sigma_z0 = 0.03
    # constant bias parameter for all shells
    bias = 1.2
    # ellipticity standard deviation as expected for a Stage-IV survey
    sigma_e = 0.27

    '''
    Survey visibility mask
    '''
    vis = glass.vmap_galactic_ecliptic(nside)
    # checking the mask:
    # hp.mollview(vis, title="Stage IV Space Survey-like Mask", unit="Visibility")
    # plt.show()

    '''
    Simulation
    '''
    # we will store the catalogue as a structured numpy array, initially empty
    catalogue = np.empty(
    0,
    dtype=[
        ("RA", float),
        ("DEC", float),
        ("Z_TRUE", float),
        ("PHZ", float),
        ("ZBIN", int),
        ("G1", float),
        ("G2", float),
    ],
    )
    # simulate the matter fields in the main loop, and build up the catalogue
    for i, delta_i in tqdm(enumerate(matter)):
        # compute the lensing maps for this shell
        convergence.add_window(delta_i, shells[i])
        kappa_i = convergence.kappa
        gamm1_i, gamm2_i = glass.shear_from_convergence(kappa_i)
        # generate galaxy positions from the matter density contrast
        for gal_lon, gal_lat, gal_count in glass.positions_from_delta(
            ngal[i],
            delta_i,
            bias,
            vis,
            rng=rng,
        ):
            # generate random redshifts over the given shell
            gal_z = glass.redshifts(gal_count, shells[i], rng=rng)
            # generator photometric redshifts using a Gaussian model
            gal_phz = glass.gaussian_phz(gal_z, sigma_z0, rng=rng)
            # attach tomographic bin IDs to galaxies, based on photometric redshifts
            gal_zbin = np.digitize(gal_phz, np.unique(zbins)) - 1
            # generate galaxy ellipticities from the chosen distribution
            gal_eps = glass.ellipticity_intnorm(gal_count, sigma_e, rng=rng)
            # apply the shear fields to the ellipticities
            gal_she = glass.galaxy_shear(
                gal_lon,
                gal_lat,
                gal_eps,
                kappa_i,
                gamm1_i,
                gamm2_i,
            )
            # make a mini-catalogue for the new rows
            rows = np.empty(gal_count, dtype=catalogue.dtype)
            rows["RA"] = gal_lon
            rows["DEC"] = gal_lat
            rows["Z_TRUE"] = gal_z
            rows["PHZ"] = gal_phz
            rows["ZBIN"] = gal_zbin
            rows["G1"] = gal_she.real
            rows["G2"] = gal_she.imag
            # add the new rows to the catalogue
            catalogue = np.append(catalogue, rows)

    print(f"Total number of galaxies sampled: {len(catalogue):,}")
    #print(catalogue)
    return ([h, Oc, Ob], catalogue, [z, dndz, sigma_z0, zbins, nbins, n_arcmin2])


#%%
def plot_redshift_catalogue(catalogue, redshift_params):
    '''
    Check
    '''
    # extract redshift parameters from simulation instance
    z = redshift_params[0]
    dndz = redshift_params[1]
    sigma_z0 = redshift_params[2]
    zbins = redshift_params[3]
    nbins = redshift_params[4]
    n_arcmin2 = redshift_params[5]
    # split dndz using the same Gaussian error model assumed in the sampling
    tomo_nz = glass.tomo_nz_gausserr(z, 
                                    dndz, 
                                    sigma_z0, 
                                    zbins)

    # redshift distribution of tomographic bins & input distributions
    plt.figure()
    plt.title("redshifts in catalogue")
    plt.ylabel("dN/dz - normalised")
    plt.xlabel("z")
    for i in range(nbins):
        in_bin = catalogue["ZBIN"] == i
        plt.hist(
            catalogue["Z_TRUE"][in_bin],
            histtype="stepfilled",
            edgecolor="none",
            alpha=0.5,
            bins=50,
            density=1,
            label=f"cat. bin {i}",
        )
    for i in range(nbins):
        plt.plot(z, (tomo_nz[i] / n_arcmin2) * nbins, alpha=0.5, label=f"inp. bin {i}")
    plt.plot(z, dndz / n_arcmin2 * nbins, ls="--", c="k")
    plt.legend(ncol=2)
    plt.show()


























