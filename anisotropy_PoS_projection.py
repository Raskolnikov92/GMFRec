#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:59:03 2022
@author: atsouros
"""

from sys import exit
import numpy as np
import nifty7 as ift
import itertools
import random as rn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import fftpack
import numpy.ma as ma
import time
import numpy.linalg as la
from scipy.ndimage import gaussian_filter
from os.path import exists

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

# cosmic ray/domain parameters
c = 2.9979258 #* in units of 10^8 m/s
pc = 3.085677581 #* in units of 10^16 m
side_length = 3000 #pc
N_pixels = 50
cdt = side_length/N_pixels #* in units where cdt = 1pc
ell = cdt
energy = 0.5 #* in units of 10^20 eV
charge = 1 #* in units where q = Ze
# position = [-8300, 0, 0]#[x,y,z] in pc in Galactocentric cartesian system
position = [side_length/2, side_length/2, 0]
velocity = [0,0,-1] #[vx,vy,vz] such that norm is 1 (equal to c)
energy = 0.5
charge = 1

# N_data_points = 125
# frac = N_data_points/N_pixels**3
# anisotropy = True

class Cosmic_ray:
    def __init__(self, position, velocity, energy, charge): #its morphin' time!
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.energy = energy
        self.charge = charge
        self.X_position = [self.position[0]]
        self.Y_position = [self.position[1]]
        self.Z_position = [self.position[2]]

    def update_velocity(self, new_velocity):
        self.velocity = np.array(new_velocity)

    def update_position(self, new_position):
        self.position = np.array(new_position)

    def backtracking_cubic_grid(self, mat):
        while abs(self.position[0]) < side_length and abs(self.position[1]) < side_length and abs(self.position[2]) < side_length:
            magnetic_field, magnetic_field_strength = Bfield(mat, self.position)
            D = (cdt * pc * c * self.charge * 1e-6) / self.energy
            A =  magnetic_field_strength * D
            previous_velocity = self.velocity - A * np.cross(self.velocity, magnetic_field)
            previous_position = self.position - self.velocity * cdt
            self.X_position.append(previous_position[0])
            self.Y_position.append(previous_position[1])
            self.Z_position.append(previous_position[2])

            self.update_position(previous_position)
            self.update_velocity(previous_velocity)
            self.velocity /= np.linalg.norm(self.velocity)

class div_cleaning(ift.LinearOperator):
    """
    Div cleaning operator operator
    ----------
    domain : real space domain on which the input field lives
    """
    def __init__(self, domain):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

# call it in main body as O = Operator(correlated_field.target)
    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            mat = x.val
            dft_mat = np.zeros([3, N_pixels, N_pixels, N_pixels], dtype=np.complex_)
            mat_div_cleaned = np.zeros([3, N_pixels, N_pixels, N_pixels])


            for index in range(0,3):
                dft_mat[index] = fftpack.fftn(mat[index])
                dft_mat[index] = fftpack.fftshift(dft_mat[index])

            k = np.mgrid[:N_pixels,:N_pixels,:N_pixels] - N_pixels/2
            k_norm_sq = np.multiply(k[0],k[0]) + np.multiply(k[1],k[1]) + np.multiply(k[2],k[2])
            inner_product = np.multiply(dft_mat[0],k[0]) + np.multiply(dft_mat[1],k[1]) + np.multiply(dft_mat[2],k[2])
            dft_mat_div_cleaned = np.where(k_norm_sq >= 1e-10, 3/2*(dft_mat - k*inner_product/k_norm_sq), dft_mat)

            for index in range(0,3):
                dft_mat_div_cleaned[index] = fftpack.ifftshift(dft_mat_div_cleaned[index])
                mat_div_cleaned[index] = np.real(fftpack.ifftn(dft_mat_div_cleaned[index]))

            return ift.sugar.makeField(self.target, mat_div_cleaned)
        if mode == self.ADJOINT_TIMES:
            mat = x.val
            dft_mat = np.zeros([3, N_pixels, N_pixels, N_pixels], dtype=np.complex_)
            mat_div_cleaned = np.zeros([3, N_pixels, N_pixels, N_pixels])


            for index in range(0,3):
                dft_mat[index] = fftpack.fftn(mat[index])
                dft_mat[index] = fftpack.fftshift(dft_mat[index])

            k = np.mgrid[:N_pixels,:N_pixels,:N_pixels] - N_pixels/2
            k_norm_sq = np.multiply(k[0],k[0]) + np.multiply(k[1],k[1]) + np.multiply(k[2],k[2])
            inner_product = np.multiply(dft_mat[0],k[0]) + np.multiply(dft_mat[1],k[1]) + np.multiply(dft_mat[2],k[2])
            dft_mat_div_cleaned = np.where(k_norm_sq >= 1e-10, 3/2*(dft_mat - k*inner_product/k_norm_sq), dft_mat)

            for index in range(0,3):
                dft_mat_div_cleaned[index] = fftpack.ifftshift(dft_mat_div_cleaned[index])
                mat_div_cleaned[index] = np.real(fftpack.ifftn(dft_mat_div_cleaned[index]))
            return ift.sugar.makeField(self.target, mat_div_cleaned)

def deflection(mat, velocityCompare):
    cr_cubic_grid = Cosmic_ray(position, velocity, energy, charge)
    cr_cubic_grid.backtracking_cubic_grid(mat)
    v_rec = cr_cubic_grid.velocity
    deltaTheta = np.arccos(np.dot(v_rec,velocityCompare))
    return deltaTheta*180/np.pi

def rms(arr):
    return np.sqrt(np.mean(arr**2))

def Bfield(mat, position): #measured in μG
    """
    Returns the unit vector and strength at position.
    """
    Bx = mat[0][int(position[0]/ell), int(position[1]/ell), int(position[2]/ell)]
    By = mat[1][int(position[0]/ell), int(position[1]/ell), int(position[2]/ell)]
    Bz = mat[2][int(position[0]/ell), int(position[1]/ell), int(position[2]/ell)]
    B = np.sqrt(Bx**2 + By**2 + Bz**2)
    unit_vector = np.array([Bx/B, By/B, Bz/B])
    strength = B
    return unit_vector, strength

def make_perc_mask(N, frac):
    # this function does the same as make_random_mask, but you get to chose the
    # percentage of pixels you want to keep.
    arr = np.asarray([0,1])
    mat = np.zeros(shape = (N,N,N))
    for i,j,k in itertools.product(range(N),range(N),range(N)):
        mat[i][j][k] = int(np.asarray(rn.choices(arr, weights=[frac, 1 - frac])))
    return mat

def main():
    N_data_points = 125
    frac = N_data_points/N_pixels**3
    anisotropy = True

    position_space = ift.RGSpace([N_pixels, N_pixels, N_pixels])

    #  For a detailed showcase of the effects the parameters
    #  of the CorrelatedField model have on the generated fields,
    #  see 'getting_started_4_CorrelatedFields.ipynb'.
    args = {
        # Amplitude of field fluctuations
        'fluctuations': (1e0, 1e0),

        # Exponent of power law power spectrum component. Choose a Kolmogorov slope
        # as a mean, and allow for a ± 1 std
        'loglogavgslope': (-11/3, 1e0),

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (1.5e0,1e0),

        # How ragged the integrated Wiener process component is
        'asperity': None
    }

    cfmaker = ift.CorrelatedFieldMaker(prefix='', total_N=3)
    cfmaker.set_amplitude_total_offset(0., (1e-16, 1e-16))
    cfmaker.add_fluctuations(position_space, **args)
    pspec = cfmaker.power_spectrum
    signal = cfmaker.finalize()

    # Change model parameters for each component using dofdex

    #Define div cleaning operator.
    DC = div_cleaning(signal.target)

    # Define response operator
    mask = make_perc_mask(N_pixels, frac)
    # mask = np.array([mask, mask, mask])
    mask = np.array([mask, mask, np.ones(mask.shape)])
    mask = ift.Field.from_raw(signal.target, mask)
    # mask = ift.Field.from_raw(position_space, mask)
    R = ift.MaskOperator(mask)

    noiseArray = np.array([np.random.normal(loc=0.0, scale=5.0, size=(N_pixels,N_pixels,N_pixels)),
       np.random.normal(loc=0.0, scale=5.0, size=(N_pixels,N_pixels,N_pixels)),
np.zeros((N_pixels,N_pixels,N_pixels))])

    signal_response = R(DC(signal))

    mock_position = ift.from_random(signal_response.domain, 'normal')


    # This is in case you want to input the signal from_raw
    # mat = signal(mock_position).val
    # mock_signal = ift.Field.from_raw(signal.target, mat)

    #Create mock field and data
    # why not DCmock_signal = DC(signal(mock_position))
    DCmock_signal = DC(signal(mock_position))
    # DCmock_signal = DC(mock_signal)
    Barr = DCmock_signal.val #array of MF values on each voxel
    B_norm_arr = np.sqrt(np.multiply(Barr[0],Barr[0]) + np.multiply(Barr[1],Barr[1]) + np.multiply(Barr[2],Barr[2]))
    Brms = rms(B_norm_arr)
    if anisotropy == True:
        lam = 1 #turbulent-to-uniform ratio
        BrmsTrue = 1 #desired RMS value of random component (in μG)
        Barr = (BrmsTrue/Brms)*Barr #rescale random component so that it has RMS of BrmsTrue
        theta = rn.uniform(0,2*np.pi)
        phi = rn.uniform(0,np.pi)
        Barr[0] += 1/lam*np.cos(theta)*np.sin(phi) #magnitude of uniform component is set to 1/lam
        Barr[1] += 1/lam*np.sin(theta)*np.sin(phi)
        Barr[2] += 1/lam*np.cos(phi)
        B_norm_arr = np.sqrt(np.multiply(Barr[0],Barr[0]) + np.multiply(Barr[1],Barr[1]) + np.multiply(Barr[2],Barr[2]))
        BrmsTrue = 5 #desired RMS value of TOTAL field (in μG)
        Brms = rms(B_norm_arr)
        Barr = (BrmsTrue/Brms)*Barr #rescale true field so that it has RMS of (last) BrmsTrue
    elif anisotropy == False:
        BrmsTrue = 5 #desired RMS value of random component (in μG)
        Barr = (BrmsTrue/Brms)*Barr #rescale true field so that it has RMS of 5μG

    Barr += noiseArray #add the noise to the
    DCmock_signal = ift.Field.from_raw(signal.target, Barr) #the true field is now the rescaled one

    # Specify noise
    data_space = R.target
    # B_norm_arr = np.sqrt(np.multiply(Barr[0],Barr[0]) + np.multiply(Barr[1],Barr[1]) + np.multiply(Barr[2],Barr[2]))
    # Brms = rms(B_norm_arr)
    noiseCov = 5**2/4 #covariance of noise
    N = ift.ScalingOperator(data_space, noiseCov)

    data = R(DCmock_signal)


    # if anisotropy == True:
            # cfmaker = ift.CorrelatedFieldMaker(prefix='', total_N=3)
            # cfmaker.add_fluctuations(position_space, **args)
    cfmaker.set_amplitude_total_offset(0., (3e0, 1e0))
    signal = cfmaker.finalize()

    # Notice that R(signal) & DC(signal) are of _OpChain type
    # While R(mock_signal) & DC(mock_signal) are of Field type

    # Minimization parameters
    # ic_sampling = ift.AbsDeltaEnergyController(name="Sampling (linear)",
    #         deltaE=0.05, iteration_limit=100)
    # ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5,
    #         convergence_level=2, iteration_limit=35)
    # minimizer = ift.NewtonCG(ic_newton)
    # ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',
    #         deltaE=0.5, iteration_limit=15, convergence_level=2)
    # minimizer_sampling = ift.NewtonCG(ic_sampling_nl)
    ic_sampling = ift.AbsDeltaEnergyController(deltaE=0.05, iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(deltaE=0.5, convergence_level=2, iteration_limit=35)
    minimizer = ift.NewtonCG(ic_newton)
    ic_sampling_nl = ift.AbsDeltaEnergyController(deltaE=0.5, iteration_limit=15, convergence_level=2)
    minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

    # Set up likelihood energy and information Hamiltonian
    likelihood_energy = (ift.GaussianEnergy(mean=data, inverse_covariance=N.inverse) @
                         R(DC(signal)))
    H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)

    initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = initial_mean

    # number of samples used to estimate the KL
    N_samples = 6

    # Draw new samples to approximate the KL six times
    for i in range(6):
        if i==5:
            # Double the number of samples in the last step for better statistics
            N_samples = 2*N_samples
        # Draw new samples and minimize KL
        KL = ift.GeoMetricKL(mean, H, N_samples, minimizer_sampling, True)
        KL, convergence = minimizer(KL)
        mean = KL.position
        ift.extra.minisanity(data, lambda x: N.inverse, signal_response,
                             KL.position, KL.samples)

    sc = ift.StatCalculator()
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))

# Plotting results
################################################################################
    cr_cubic_grid = Cosmic_ray(position, velocity, energy, charge)
    cr_cubic_grid.backtracking_cubic_grid(Barr)
    v_true = cr_cubic_grid.velocity

    theta_arrays = np.array([])
    for sample in KL.samples:
        field = DC(signal(sample + KL.position)) #divergence clean
        theta_arrays = np.append(theta_arrays, deflection(field.val,v_true))


    dir = "CR_backtracking_results_(projected_POS)/"
    if anisotropy == True:
        path = dir + "N_data=%d_(anisotropic_BxBy)_lambda=%.2f.txt"%(N_data_points,lam)
    elif anisotropy == False:
        path = dir + "N_data=%d_(isotropic_BxBy)_test.txt"%N_data_points

    deltaTheta1 = np.mean(theta_arrays) #angle between the initial velocities obtained by backtracking through a) reconstructed and b) actual field

    deltaTheta2 = np.arccos(np.dot(v_true,velocity))*180/np.pi #angle between initial and final  velocities propagated through actual field

    if not exists(path):
        open(path,"x")
    if 'f' not in locals():
        f = open(path,"a")

    f.write("%.2f %.2f\n"%(deltaTheta1,deltaTheta2))

if __name__ == '__main__':
    t = time.time()
    for i in range(0,100):
        main()
    t = time.time() - t
    t *= 1/60 #time in minutes
    print('Total calculation time: %.2f minutes'%t)
