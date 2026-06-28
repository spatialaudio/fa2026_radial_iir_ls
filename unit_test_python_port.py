# https://github.com/spatialaudio/fa2026_radial_iir_ls
# Frank Schultz @ github: fs446
#
# Ground truth filter design for comparing Matlab reference vs. our Python port
#
# Used hard-/software:
# Apple M1 Pro, Mac OS 26.5.1, clang 15.0.0, fortran gcc 13.4, cython 3.2.4
# numpy / scipy -> LAPACK & BLAS: Apple's Accelerate
# MATLAB 23.2.0.3097123 (R2023b) Update 11, Optimization Toolbox 23.2
# -> 'OpenBLAS 0.3.21' and
# 'NAG Performance Components (NPC) Release 1.2.0,
# supporting Linear Algebra PACKage (LAPACK 3.9.1)'


import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, Rectangle  # for zplane plot
from scipy.io import loadmat
from scipy.signal import dimpulse, freqz, freqz_zpk, tf2zpk  # filter analysis
from util import mpiir_l2, get_package_versions  # see code details in util.py


get_package_versions()

# to match Matlab's 'format longE' terminal output
custom_formatters = {
    'float_kind': lambda x: "{:.15e}".format(x),
    'complexfloat': lambda x: f"{x.real:.15e} + {x.imag:.15e}j"
}
np.set_printoptions(formatter=custom_formatters)

write_npz = False
read_npz = True
log_file_str = 'unit_test_python_port.txt'

# some nice-numbered zeros and poles
z_ref = np.array([-1.25, +0.25, +0.5,
                  -0.5-1j, -0.5+1j])
p_ref = np.array([np.sqrt(2)/4*np.exp(+1j*3*np.pi/4),
                  np.sqrt(2)/4*np.exp(-1j*3*np.pi/4),
                  1/np.sqrt(2)*np.exp(+1j*np.pi/4),
                  1/np.sqrt(2)*np.exp(-1j*np.pi/4),
                  -0.5])
Nz = len(z_ref)  # number of (non-origin) zeros that can be optimised
Np = len(p_ref)  # number of (non-origin) poles that can be optimised

# the filter design with
max_pole_radius = 0.6
# is poor, because max pole radius is too small

# the filter design with
max_pole_radius = np.max(np.abs(p_ref))
# should exactly yield the intended target filter


print('max_pole_radius too restrictive:',
      np.max(np.abs(p_ref)) > max_pole_radius)

alpha = 0.5  # learning rate>0, connected to pole alignment, hence |alpha|<1!

om = np.logspace(np.log10(np.pi*2**-15), np.log10(np.pi), 2**9, endpoint=True)
_, D = freqz_zpk(z_ref, p_ref, 1, om)
D /= np.abs(D[0])  # make DC unity gain
W = np.ones_like(D, dtype='complex128')  # frequency weighting

# filter design by optimisation / solving a quadratic program
b, a, _ = mpiir_l2(Nz, Np, om, D, W, max_pole_radius,
                   alpha=alpha,            # Matlab default: 0.5
                   tol_mpiir_l2=1e-4,      # Matlab default: 1e-4
                   tol_update=1e-6,        # Matlab default: 1e-6
                   itmax=10,               # Matlab default: 10
                   nfft=2**10,             # Matlab default: 2**10
                   cnt_max=50,             # Matlab default: 50
                   solver="quadprog",      # chosen default: "quadprog"
                   verbose_solver=False,   # chosen default: False
                   verbose_descent=True,   # chosen default: False
                   log_file_str=log_file_str,
                   )
print('coeff b:\n', b, '\n', b.shape)
print('coeff a:\n', a, '\n', a.shape)

if write_npz:  # save all in-/output of mpiir_l2()
    np.savez('unit_test_python_port.npz',
             Nz_ref=Nz,
             Np_ref=Np,
             om_ref=om,
             D_ref=D,
             W_ref=W,
             max_pole_radius_ref=max_pole_radius,
             alpha=alpha,
             b_ref=b,
             a_ref=a)
if read_npz:
    print('\ncoeff b, a match our Python ground truth?')
    with np.load('unit_test_python_port.npz') as data:
        print(np.allclose(b, data['b_ref']))
        print(np.allclose(a, data['a_ref']))
if False:  # if Matlab results are available
    # note: it's probably good to compare the radius' and angles
    # of poles / zeros for Matlab vs. Python
    # however the potential different ordering makes this a
    # tedious task...in this example this works accidentally
    # see below
    coeff_b_matlab = loadmat('../lang_iir/coeff_b.mat')
    coeff_b_matlab = np.squeeze(coeff_b_matlab['b'])
    e_atol = coeff_b_matlab - b
    e_rtol = 1 - coeff_b_matlab / b
    print('# Matlab vs. Python')
    print('# coef b min/max e_atol:', np.min(e_atol), np.max(e_atol))
    print('# coef b min/max e_rtol:', np.min(e_rtol), np.max(e_rtol))
    coeff_a_matlab = loadmat('../lang_iir/coeff_a.mat')
    coeff_a_matlab = np.squeeze(coeff_a_matlab['a'])
    e_atol = coeff_a_matlab - a
    e_rtol = 1 - coeff_a_matlab / a
    print('# coef a min/max e_atol:', np.min(e_atol), np.max(e_atol))
    print('# coef a min/max e_rtol:', np.min(e_rtol), np.max(e_rtol))
    # Matlab vs. Python
    # coef b min/max e_atol: -0.00012891507170194327 0.00023327970727649028
    # coef b min/max e_rtol: -0.0005598713081051976 0.01154640791351802
    # coef a min/max e_atol: -5.0803116686343186e-05 0.00031660084860284565
    # coef a min/max e_rtol: -77904578115.12595 0.0003938025694902869
    #
    # !!! the following code relies on the same roots ordering!!!
    print('b roots rtol\n', 1 - np.roots(coeff_b_matlab) / np.roots(b))
    print('b abs roots rtol\n', 1 - np.abs(np.roots(coeff_b_matlab)) / np.abs(np.roots(b)))
    print('b angle roots rtol\n', 1 - np.angle(np.roots(coeff_b_matlab)) / np.angle(np.roots(b)))
    print('a roots rtol\n', 1 - np.roots(coeff_a_matlab) / np.roots(a))
    print('a abs roots rtol\n', 1 - np.abs(np.roots(coeff_a_matlab)) / np.abs(np.roots(a)))
    print('a angle roots rtol\n', 1 - np.angle(np.roots(coeff_a_matlab)) / np.angle(np.roots(a)))
    # b roots rtol
    #  [2.381655708205699e-07 + 0.000000000000000e+00j
    #  -2.538686026287706e-06 + -4.450486004614609e-07j
    #  -2.538686026287706e-06 + 4.450486004614609e-07j
    #  -2.427044700794578e-04 + 0.000000000000000e+00j
    #  1.733527296309267e-03 + 0.000000000000000e+00j]
    # b abs roots rtol
    #  [2.381655708205699e-07 -2.538686125763689e-06 -2.538686125763689e-06
    #  -2.427044700794578e-04 1.733527296309378e-03]
    # b angle roots rtol
    #  [0.000000000000000e+00 -2.187563210576116e-07 -2.187563210576116e-07 nan
    #  nan]
    # a roots rtol
    #  [5.582273709281438e-06 + -2.812102990684684e-05j
    #  5.582273709281438e-06 + 2.812102990684684e-05j
    #  -3.741034129789433e-04 + 0.000000000000000e+00j
    #  -6.497847256323741e-04 + 4.580930486351911e-04j
    #  -6.497847256323741e-04 + -4.580930486351911e-04j]
    # a abs roots rtol
    #  [5.581878310789712e-06 5.581878310789712e-06 -3.741034129789433e-04
    #  -6.498895821134276e-04 -6.498895821134276e-04]
    # a angle roots rtol
    #  [-3.580500718003421e-05 -3.580500718003421e-05 0.000000000000000e+00
    #  1.942944650723533e-04 1.942944650723533e-04]

# make a and b arrays the same length for proper zplane plot
if len(b) > len(a):
    a_tmp = np.zeros_like(b)
    a_tmp[:len(a)] = a
    a = np.copy(a_tmp)
elif len(a) > len(b):
    b_tmp = np.zeros_like(a)
    b_tmp[:len(b)] = b
    b = np.copy(b_tmp)
else:  # nothing to do, as length(a)=length(b)
    tmp = 0

# frequency response
om_plt = np.linspace(0, 1, 2**12)  # use high frequency resolution
_, H = freqz(b, a, om_plt * np.pi)

# plot filter characteristics
plt.figure(figsize=(16, 10))
plt.subplot(2, 2, 1)
plt.plot(om_plt, 20*np.log10(np.abs(H)),
         'C0-', lw=3, label='numerically solved')
plt.plot(om / np.pi, 20*np.log10(np.abs(D)),
         'C6-.', lw=3, label='desired / target')
plt.grid(True)
plt.axis([0, 1, -21, 9])
plt.yticks(np.arange(-21, 9+3, 3))
plt.xlabel(r'$\Omega / \pi$')
plt.ylabel('level in dB')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(om_plt, np.angle(H),
         'C0-', lw=3, label='numerically solved')
plt.plot(om / np.pi, np.angle(D),
         'C6-.', lw=3, label='desired / target')
plt.grid(True)
plt.axis([0, 1, -np.pi, +np.pi])
plt.xlabel(r'$\Omega / \pi$')
plt.ylabel('phase in rad')

plt.subplot(2, 2, 4)
k, h = dimpulse((b, a, 1))
h = np.squeeze(h)
markerline, stemlines, _ = plt.stem(k, h,
                                    linefmt='k-',
                                    basefmt=' ',
                                    markerfmt='ko', label='impz')
markerline.set_markersize(6)
stemlines.set_linewidth(1)
plt.plot(k, h, 'k:', lw=1)
plt.grid(True)
plt.axis([-1, 20, -0.75, +0.75])
plt.xlabel('sample index k')
plt.ylabel('signal value')
plt.legend()

plt.subplot(2, 2, 2)
z, p, k = tf2zpk(b, a)
# zplane plot taken from our own code of plot_zplane(z, p, k) at
# https://github.com/spatialaudio/signals-and-systems-exercises/blob/master/sig_sys_tools.py
Nf = 2**10
Om = np.arange(Nf) * 2*np.pi/Nf
plt.plot(np.cos(Om), np.sin(Om), 'k')

rect_box = Rectangle((-3, -3), 6, 6, color='gray', alpha=0.33)
plt.gcf().gca().add_artist(rect_box)
try:  # TBD: check if this pole is compensated by a zero
    circle = Circle((0, 0), radius=np.max(np.abs(p)),
                    color='white', alpha=1)
    plt.gcf().gca().add_artist(circle)
except ValueError:
    print('no pole at all, ROC is whole z-plane')

for pi in zip(p_ref):  # plot reference poles individually
    plt.plot(np.real(pi), np.imag(pi), ms=7, mew=3,
             color='C6', marker='x')
for zi in zip(z_ref):  # plot reference zeros individually
    plt.plot(np.real(zi), np.imag(zi), ms=5, mew=2,
             color='C6', marker='o', fillstyle='none')

zu, zc = np.unique(z, return_counts=True)  # find and count unique zeros
for zui, zci in zip(zu, zc):  # plot them individually
    plt.plot(np.real(zui), np.imag(zui), ms=10,
             color='C0', marker='o', fillstyle='none')
    if zci > 1:  # if multiple zeros exist then indicate the count
        plt.text(np.real(zui), np.imag(zui), zci)

pu, pc = np.unique(p, return_counts=True)  # find and count unique poles
for pui, pci in zip(pu, pc):  # plot them individually
    plt.plot(np.real(pui), np.imag(pui), ms=10,
             color='C0', marker='x')
    if pci > 1:  # if multiple poles exist then indicate the count
        plt.text(np.real(pui), np.imag(pui), pci)

plt.xticks(np.linspace(-1.5, 1.5, 13))
plt.yticks(np.linspace(-1.5, 1.5, 13))
plt.text(-1.5, 1.25, 'pink: ground truth poles/zeros')
plt.text(-1.5, -1.25, 'k=%f' % k)
plt.text(-1.5, -1.5, 'ROC for causal: gray')
plt.axis('square')
plt.xlabel(r'$\Re\{z\}$')
plt.ylabel(r'$\Im\{z\}$')
plt.grid(True)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('unit_test_python_port.png', dpi=600)
