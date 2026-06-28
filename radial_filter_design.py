# https://github.com/spatialaudio/fa2026_radial_iir_ls
# Examples of the paper [Sch2026]
# Frank Schultz, Nara Hahn, Sascha Spors (2026):
# "Discrete-time IIR filter design for radial filters
# by numerically optimised pole/zero placement"
# Proc. of Forum Acusticum, Graz, Austria, September 8-12, 2026
#
# Author: Frank Schultz @ github: fs446
#
# Note: the graphics shown in the paper were obtained with
# the Matlab reference code and Matlab scripts, the latter
# not being included in this repo for a streamlined
# Python code basis.
# The Matlab and Python filters are highly comparable.
#
# Used hard-/software:
# Apple M1 Pro, Mac OS 26.5.1, clang 15.0.0, fortran gcc 13.4, cython 3.2.4
# numpy / scipy -> LAPACK & BLAS: Apple's Accelerate
# MATLAB 23.2.0.3097123 (R2023b) Update 11, Optimization Toolbox 23.2
# -> 'OpenBLAS 0.3.21' and
# 'NAG Performance Components (NPC) Release 1.2.0,
# supporting Linear Algebra PACKage (LAPACK 3.9.1)'


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, Rectangle  # for zplane plot
from scipy.io import loadmat
from scipy.signal import dimpulse, freqz, tf2zpk  # for filter analysis
from scipy.special import spherical_jn, spherical_yn
from util import mpiir_l2, get_package_versions


def spherical_hn2(n, z, derivative):
    return spherical_jn(n, z, derivative) - 1j * spherical_yn(n, z, derivative)


get_package_versions()

# to match Matlab's 'format longE' terminal output
custom_formatters = {
    'float_kind': lambda x: "{:.15e}".format(x),
    'complexfloat': lambda x: f"{x.real:.15e}+{x.imag:.15e}j"
}
np.set_printoptions(formatter=custom_formatters)

# [Sch2026] examples 1: I, 2: II, 3: III, 4: IV, 5: V
example = 4

write_npz = False  # to write a Python ground truth
read_npz = True  # to read a Python ground truth, included in the repo

# constant variables:
R = 0.2  # m
fs = 32000  # Hz
c = 343  # m/s
alpha = 0.5  # this is the chosen default in Matlab reference code

match example:
    case 1:  # cf. paper figure 1
        r = 150 / fs * c + R
        n = 24
        Preringing_Delay = 0
        Np = n + 1
        Nz = n + 1 + Preringing_Delay
        max_pole_radius = 0.9
        wf = 0
        ww = 1
        log_file_str = 'radial_filter_design_example_1.txt'
        # Matlab vs. Python
        # coef b min/max e_atol: -0.007073700286667706 0.0109812282327163
        # coef b min/max e_rtol: -0.10795889547310722 0.2747854910576115
        # coef a min/max e_atol: -0.013883443345623636 0.019878916492630433
        # coef a min/max e_rtol: -0.897871474036924 2.533047610673138
    case 2:  # cf. paper figure 2
        r = 150 / fs * c + R
        n = 24
        Preringing_Delay = 30
        Np = 5
        Nz = n+1+Preringing_Delay
        max_pole_radius = 0.9
        wf = 6000
        ww = 1000
        log_file_str = 'radial_filter_design_example_2.txt'
        # Matlab vs. Python
        # coef b min/max e_atol: -1.848291830697235e-08 1.449074875370382e-08
        # coef b min/max e_rtol: -5.989676070683458e-07 2.7587475974755193e-06
        # coef a min/max e_atol: -4.499617300979253e-08 0.0
        # coef a min/max e_rtol: -6.696441623432747e-08 7.241925137346783e-08
    case 3:  # cf. paper figure 3
        r = 100 / fs * c + R
        n = 38
        Preringing_Delay = 10
        Np = 5
        Nz = 15 + 1 + Preringing_Delay
        max_pole_radius = 0.6
        wf = 5000
        ww = 1
        log_file_str = 'radial_filter_design_example_3.txt'
        # Matlab vs. Python
        # coef b min/max e_atol: -6.667172392282872e-06 5.766906218479528e-06
        # coef b min/max e_rtol: -8.861544579152891e-05 0.00015519344949410652
        # coef a min/max e_atol: 0.0 5.696965683299471e-05
        # coef a min/max e_rtol: -0.00140209382186951 0.0001388648876795573
    case 4:  # cf. paper figure 4
        r = 100 / fs * c + R
        n = 38
        Preringing_Delay = 10
        Np = 2
        Nz = n - 14 + 1 + Preringing_Delay
        max_pole_radius = 0.9
        wf = 8000
        ww = 5e3
        log_file_str = 'radial_filter_design_example_4.txt'
        # Matlab vs. Python
        # coef b min/max e_atol: -1.7913509564593255e-10 1.7150947329014343e-10
        # coef b min/max e_rtol: -1.8402969814701464e-08 9.399681477617605e-09
        # coef a min/max e_atol: 0.0 7.622049658095875e-10
        # coef a min/max e_rtol: -1.0984089193755153e-09 0.0
    case 5:  # cf. paper figure 5
        r = 200 / fs * c + R
        n = 24
        Preringing_Delay = 10
        Np = 4
        Nz = n + 1 + Preringing_Delay
        max_pole_radius = 0.9
        wf = 7000
        ww = 10**(100/20)
        alpha = 0.25  # note: different from the otherwise used Matlab default
        log_file_str = 'radial_filter_design_example_5.txt'
        # Matlab vs. Python
        # coef b min/max e_atol: -9.676608958564259e-09 7.695360570991738e-09
        # coef b min/max e_rtol: -8.66443872205025e-08 5.689962379040736e-08
        # coef a min/max e_atol: -1.110528091707863e-08 9.862465955023936e-09
        # coef a min/max e_rtol: 0.0 1.2040087815634593e-08

f_ctrl = np.logspace(np.log10(1), np.log10(fs/2), 2**9, endpoint=True)  # Hz
w = 2 * np.pi * f_ctrl  # rad/s
w_c = w / c  # rad/m
kr = w_c * r  # rad
kR = w_c * R  # rad
om = w / fs  # rad
# target frequency response, cf. Eq. (1)
D = -1j * spherical_hn2(n, kr, False) / spherical_hn2(n, kR, True)
D *= np.exp(+1j * w_c * (r-R)) * r/R  # get min phase and amplitude normalised
# cf. Eq. (4)
if f_ctrl[0] == 0:
    # highpass DC: -oo (modelled with -200dB), +90deg phase
    D[0] = 10**(-200/20) * np.exp(+1j * np.pi/2)
else:
    # use abs value of second entry, but the ideal +90deg phase
    D[0] = np.abs(D[1]) * np.exp(+1j * np.pi/2)
D *= np.exp(-1j * om * Preringing_Delay)
W = np.ones_like(D, dtype='complex128')  # frequency dependent weighting
# cf. Eq. (6)
W[f_ctrl < wf] = ww

# filter design by optimisation / solving a quadratic program
# cf. Eq. (6,7,8) in paper
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
    np.savez('radial_filter_design_example_'+str(example)+'.npz',
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
    with np.load('radial_filter_design_example_'+str(example)+'.npz') as data:
        print(np.allclose(b, data['b_ref']))
        print(np.allclose(a, data['a_ref']))
if False:  # if Matlab results are available
    # note: it's probably good to compare the radius' and angles
    # of poles / zeros for Matlab vs. Python
    # however the potential different ordering makes this a
    # tedious task...TBD
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
f_plot = np.logspace(np.log10(1), np.log10(fs/2), 2**14, endpoint=True)
om_plot = 2*np.pi * f_plot / fs
_, H = freqz(b, a, om_plot)

# plot filter characteristics
plt.figure(figsize=(16, 10))
plt.subplot(3, 2, 1)
plt.semilogx(f_plot, 20*np.log10(np.abs(H)),
             'C0-', lw=3, label='numerically solved H_HP')
plt.semilogx(f_ctrl, 20*np.log10(np.abs(D)),
             'C6-.', lw=3, label='desired / target H_HP')
plt.grid(True)
plt.axis([1e2, 20000, -120, 20])
plt.ylabel('level in dB')
plt.legend()

plt.subplot(3, 2, 2)
plt.semilogx(f_plot, 20*np.log10(np.abs(H)),
             'C0-', lw=3, label='obtained')
plt.semilogx(f_ctrl, 20*np.log10(np.abs(D)),
             'C6-.', lw=3, label='desired/target')
plt.grid(True)
plt.axis([5000, 20000, -10, 10])

plt.subplot(3, 2, 3)
# cf. caption Fig. 1(b): "All phase frequency responses are compensated
# for the pre-ringing delay m, hence the more pleasing quasi-minimum-phase
# is depicted."
plt.semilogx(f_plot, np.angle(H*np.exp(+1j * om_plot * Preringing_Delay)),
             'C0-', lw=3, label='obtained')
plt.semilogx(f_ctrl, np.angle(D*np.exp(+1j * om * Preringing_Delay)),
             'C6-.', lw=3, label='desired/target')
plt.grid(True)
plt.axis([1e2, 20000, -np.pi, +np.pi])
plt.xlabel('frequency in Hz')
plt.ylabel('minphase in rad')

plt.subplot(3, 2, 4)
plt.semilogx(f_plot, np.angle(H*np.exp(+1j * om_plot * Preringing_Delay)),
             'C0-', lw=3, label='obtained')
plt.semilogx(f_ctrl, np.angle(D*np.exp(+1j * om * Preringing_Delay)),
             'C6-.', lw=3, label='desired/target')
plt.grid(True)
plt.axis([5000, 20000, -np.pi, +np.pi])
plt.xlabel('frequency in Hz')

plt.subplot(3, 2, 5)
k, h = dimpulse((b, a, 1))
h = np.squeeze(h)
markerline, stemlines, _ = plt.stem(k, h,
                                    linefmt='k-',
                                    basefmt=' ',
                                    markerfmt='ko',
                                    label='impulse response h_HP')
markerline.set_markersize(6)
stemlines.set_linewidth(1)
plt.plot(k, h, 'k:', lw=1)
plt.grid(True)
plt.axis([-1, k[-1], -0.5, +0.5])
plt.xlabel('sample index k')
plt.ylabel('signal value')
plt.legend()

plt.subplot(3, 2, 6)
z, p, k = tf2zpk(b, a)
# zplane plot taken from our own code of plot_zplane(z, p, k) at
# https://github.com/spatialaudio/signals-and-systems-exercises/blob/master/sig_sys_tools.py
Nf = 2**7
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

zu, zc = np.unique(z, return_counts=True)  # find and count unique zeros
for zui, zci in zip(zu, zc):  # plot them individually
    plt.plot(np.real(zui), np.imag(zui), ms=5,
             color='C0', marker='o', fillstyle='none')
    if zci > 1:  # if multiple zeros exist then indicate the count
        plt.text(np.real(zui), np.imag(zui), zci)

pu, pc = np.unique(p, return_counts=True)  # find and count unique poles
for pui, pci in zip(pu, pc):  # plot them individually
    plt.plot(np.real(pui), np.imag(pui), ms=8,
             color='C3', marker='x')
    if pci > 1:  # if multiple poles exist then indicate the count
        plt.text(np.real(pui), np.imag(pui), pci)

plt.plot(np.max(np.abs(zu))*np.cos(Om),
         np.max(np.abs(zu))*np.sin(Om),
         'C0-', lw=1)
plt.plot(np.max(np.abs(pu))*np.cos(Om),
         np.max(np.abs(pu))*np.sin(Om),
         'C3-', lw=1)
plt.plot(max_pole_radius*np.cos(Om),
         max_pole_radius*np.sin(Om),
         'C1--', lw=1)

plt.text(-1.5, +1.25, 'k=%f' % k)
plt.text(-1.5, -1.5, 'ROC for causal IR: gray')
plt.axis('square')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.xlabel(r'$\Re\{z\}$')
plt.ylabel(r'$\Im\{z\}$')
plt.grid(True)

plt.tight_layout()
plt.savefig('radial_filter_design_example_'+str(example)+'.png', dpi=600)
