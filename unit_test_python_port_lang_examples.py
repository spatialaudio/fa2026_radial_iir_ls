# https://github.com/spatialaudio/fa2026_radial_iir_ls
# Frank Schultz @ github: fs446
#
# We check some of the examples from [Lan1999] Mathias Lang (1999):
# "Algorithms for the Constrained Design of Digital Filters with
#  Arbitrary Magnitude and Phase Responses"
# doctoral thesis, TU Wien (University of Technology in Vienna, Austria)
# thesis downloadable at https://mattsdsp.blogspot.com/p/phd-thesis.html
# and the example 1 of [Lan2000] Mathias C. Lang (2000):
# "Least-Squares Design of IIR Filters with
#  Prescribed Magnitude and Phase Responses
#  and a Pole Radius Constraint." In:
# IEEE Transactions on Signal Processing, Volume 48, Issue 11, Pages 3109-3121
# November 2000, https://doi.org/10.1109/78.875468
# with our Python port of Lang's Matlab reference code given in [Lan1999]
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
from scipy.signal import dimpulse, freqz, tf2zpk  # for filter analysis
from util import mpiir_l2, get_package_versions  # see code details in util.py


get_package_versions()

# to match Matlab's 'format longE' terminal output
custom_formatters = {
    'float_kind': lambda x: "{:.15e}".format(x),
    'complexfloat': lambda x: f"{x.real:.15e} + {x.imag:.15e}j"
}
np.set_printoptions(formatter=custom_formatters)

example = 8  # 1,2,3,4,6,7...[Lan1999] examples, 8... [Lan2000] example 1
# for example 2 we used "cvxopt" (which early stops in an inner loop,
# but the solution is useful), rather than "quadprog" (which runs forever)

write_npz = False  # to write a Python ground truth
read_npz = True  # to read a Python ground truth, included in the repo

match example:
    case 1:
        # Example 1: Fig. 5.2 / 5.3
        # Matlab vs. Python
        # coef b min/max e_atol: -5.929063420784253e-10 5.504839986625321e-10
        # coef b min/max e_rtol: -2.6319968871035826e-08 1.782791703153208e-08
        # coef a min/max e_atol: -6.568753097013769e-09 6.317854683857149e-09
        # coef a min/max e_rtol: 0.0 5.867415220350836e-09
        max_pole_radius = .98
        Nz = 4
        Np = 4
        om = np.concatenate((np.linspace(0, 0.2, 20),
                             np.linspace(0.4, 1., 60))) * np.pi
        D = np.concatenate((np.exp(-1j*om[0:20]*5),
                            np.zeros(60)))
        W = np.ones(80)
        log_file_str = 'unit_test_python_port_lang_example_1.txt'
    case 2:
        # Example 2: Fig. 5.4 / 5.5
        # quadprog solver does not converge with the chosen defaults
        # try to increase itmax and/or decrease alpha, another solver
        max_pole_radius = .8263
        Nz = 15
        Np = 15
        om = np.concatenate((np.linspace(0, 0.4, 40),
                             np.linspace(0.56, 1., 44))) * np.pi
        D = np.concatenate((np.exp(-1j*om[0:40]*15),
                            np.zeros(44)))
        W = np.concatenate((np.ones(40),
                            np.ones(44)*100))
        log_file_str = 'unit_test_python_port_lang_example_2.txt'
    case 3:
        # Example 3: Fig. 5.6 / 5.7
        # Matlab vs. Python
        # coef b min/max e_atol: -4.665774817669277e-06 4.458620202041352e-06
        # coef b min/max e_rtol: -4.473300487073395e-05 0.0005147716590924167
        # coef a min/max e_atol: -2.8377427059300686e-05 0.0
        # coef a min/max e_rtol: 0.0 2.4260588397440053e-05
        max_pole_radius = .9276
        Nz = 14
        Np = 6
        om = np.concatenate((np.linspace(0, 0.475, 50),
                             np.linspace(0.525, 1., 50))) * np.pi
        D = np.concatenate((np.zeros(50),
                            np.exp(-1j*om[50:]*12)))
        W = np.ones(100)
        log_file_str = 'unit_test_python_port_lang_example_3.txt'
    case 4:
        # Example 4: Fig. 5.9
        # Matlab vs. Python
        # coef b min/max e_atol: -1.5655300780841042e-06 2.5752000664997117e-06
        # coef b min/max e_rtol: -0.038726119360503075 0.011050804147822557
        # coef a min/max e_atol: -0.0004791316512067212 0.0008017669204103406
        # coef a min/max e_rtol: 0.0 0.00028101567344196443
        max_pole_radius = .98
        Nz = 20
        Np = 8
        om = np.concatenate((np.linspace(0, 0.36, 36),
                             np.linspace(0.4, 0.5, 10),
                             np.linspace(0.54, 1, 46))) * np.pi
        D = np.concatenate((np.zeros(36),
                            np.exp(-1j*om[36:46]*20),
                            np.zeros(46)))
        W = np.concatenate((np.ones(36)*100,
                            np.ones(10)*1,
                            np.ones(46)*100))
        log_file_str = 'unit_test_python_port_lang_example_4.txt'
    # case 5:
        # this thesis example is not working
        # neither in Matlab nor in Python
        # we don't yet know why this happens
    case 6:
        # Example 6: Fig. 5.12 / 5.13
        # Matlab vs. Python
        # coef b min/max e_atol: -3.541869730921965e-06 8.812844347269588e-07
        # coef b min/max e_rtol: -3.0343007664246358e-05 0.0001224322747481743
        # coef a min/max e_atol: -1.621960609488049e-05 1.5790351154176818e-05
        # coef a min/max e_rtol: -1.2113517319001232e-05 0.0
        max_pole_radius = .92
        Nz = 10
        Np = 6
        om = np.concatenate((np.linspace(0, 0.5, 50),
                             np.linspace(0.6, 1., 40))) * np.pi
        D = np.concatenate((np.exp(-1j*om[0:50]*8.5),
                            np.zeros(40))) * 0.5 * (1+10**(-0.5/20))
        W = np.concatenate((np.ones(50),
                            np.ones(40)*10))
        log_file_str = 'unit_test_python_port_lang_example_6.txt'
    case 7:
        # Example 7: Fig. 5.14 / 5.15
        # Matlab vs. Python
        # coef b min/max e_atol: -7.0492369260805e-07 1.2738862871614857e-06
        # coef b min/max e_rtol: -0.0053970019945035475 0.029359908462407658
        # coef a min/max e_atol: -0.00038089056672152566 0.00048804979197925036
        # coef a min/max e_rtol: 0.0 7.028159795285926e-05
        max_pole_radius = .98
        Nz = 40
        Np = 6
        om = np.concatenate((np.linspace(0, 0.2, 20),
                             np.linspace(0.23, 1., 77))) * np.pi
        D = np.concatenate((np.exp(-1j*om[0:20]*37),
                            np.zeros(77)))
        W = np.concatenate((np.ones(20),
                            np.ones(77)*1000))
        log_file_str = 'unit_test_python_port_lang_example_7.txt'
    case 8:
        # Lang's IEEE paper, Example 1: Fig. 2 & 3
        # Matlab vs. Python
        # coef b min/max e_atol: -2.9289610762761598e-05 5.9198825315270875e-06
        # coef b min/max e_rtol: -0.007199618149146936 0.0006737132763374731
        # coef a min/max e_atol: -8.493430160083637e-05 3.875315730894613e-05
        # coef a min/max e_rtol: -0.00012837849822555647 0.0
        max_pole_radius = .8263
        Nz = 15
        Np = 4
        om = np.concatenate((np.linspace(0, 0.4, 40),
                             np.linspace(0.56, 1., 44))) * np.pi
        D = np.concatenate((np.exp(-1j*om[0:40]*12),
                            np.zeros(44)))
        W = np.concatenate((np.ones(40),
                            np.ones(44)*10))
        log_file_str = 'unit_test_python_port_lang_example_8.txt'

# filter design by optimisation / solving a quadratic program
b, a, _ = mpiir_l2(Nz, Np, om, D, W, max_pole_radius,
                   alpha=0.5,              # Matlab default: 0.5
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
    np.savez('unit_test_python_port_lang_example_'+str(example)+'.npz',
             Nz_ref=Nz,
             Np_ref=Np,
             om_ref=om,
             D_ref=D,
             W_ref=W,
             max_pole_radius_ref=max_pole_radius,
             b_ref=b,
             a_ref=a)
if read_npz:
    print('\ncoeff b, a match our Python ground truth?')
    with np.load('unit_test_python_port_lang_example_'+str(example)+'.npz') as data:
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
om_plt = np.linspace(0, 1, 2**12)  # use high frequency resolution
_, H = freqz(b, a, om_plt * np.pi)

# plot filter characteristics
plt.figure(figsize=(16, 10))
plt.subplot(3, 2, 1)
plt.plot(om_plt, 20*np.log10(np.abs(H)),
         'C0-', lw=3, label='numerically solved')
plt.plot(om / np.pi, 20*np.log10(np.abs(D)),
         'C6-.', lw=3, label='desired / target')
plt.grid(True)
plt.axis([0, 1, -80, 1])
plt.ylabel('level in dB')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(om_plt, 20*np.log10(np.abs(H)), 'C0-', lw=3, label='obtained')
plt.plot(om / np.pi, 20*np.log10(np.abs(D)), 'C6-.', lw=3, label='desired/target')
plt.grid(True)
plt.ylim(-1, 1)
plt.xlabel(r'$\Omega / \pi$')

plt.subplot(3, 2, 3)
plt.plot(om_plt, np.angle(H), 'C0-', lw=3, label='obtained')
plt.plot(om / np.pi, np.angle(D), 'C6-.', lw=3, label='desired/target')
plt.grid(True)
plt.axis([0, 1, -np.pi, +np.pi])
plt.xlabel(r'$\Omega / \pi$')
plt.ylabel('phase in rad')

plt.subplot(3, 2, 4)
plt.semilogx(om_plt, np.angle(H), 'C0-', lw=3, label='obtained')
plt.semilogx(om / np.pi, np.angle(D), 'C6-.', lw=3, label='desired/target')
plt.grid(True)
plt.xlabel(r'$\Omega / \pi$')

plt.subplot(3, 2, 5)
k, h = dimpulse((b, a, 1))
h = np.squeeze(h)
markerline, stemlines, _ = plt.stem(k, h,
                                    linefmt='k-',
                                    basefmt=' ',
                                    markerfmt='ko',
                                    label='impulse response')
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

plt.text(-2, -1.75, 'k=%f' % k)
plt.text(-2, -2, 'ROC for causal IR: gray')
if (np.abs(z) > 2).any():
    # print(z[np.abs(z) > 2])
    plt.text(-2, 1.75, 'some zeros out of plot range!')
plt.axis('square')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel(r'$\Re\{z\}$')
plt.ylabel(r'$\Im\{z\}$')
plt.grid(True)

plt.tight_layout()
plt.savefig('unit_test_python_port_lang_example_'+str(example)+'.png', dpi=600)
