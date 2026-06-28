# https://github.com/spatialaudio/fa2026_radial_iir_ls
#
# Code for paper:
# Frank Schultz, Nara Hahn, Sascha Spors (2026):
# "Discrete-time IIR filter design for radial filters
# by numerically optimised pole/zero placement"
# Proc. of Forum Acusticum, Graz, Austria, September 8-12, 2026
#
# Code by:
# Frank Schultz, University of Rostock, Germany
# https://github.com/fs446
# https://orcid.org/0000-0002-3010-0294
#
# This is a Python port of the filter design proposed in
# [Lan2000] Mathias C. Lang (2000):
# "Least-Squares Design of IIR Filters with
#  Prescribed Magnitude and Phase Responses
#  and a Pole Radius Constraint." In:
# IEEE Transactions on Signal Processing, Volume 48, Issue 11, Pages 3109-3121
# November 2000, https://doi.org/10.1109/78.875468
#
# The original Matlab reference code is completely included in
# [Lan1999] Mathias Lang (1999):
# "Algorithms for the Constrained Design of Digital Filters with
#  Arbitrary Magnitude and Phase Responses."
# doctoral thesis, TU Wien (University of Technology in Vienna, Austria)
# thesis is downloadable at https://mattsdsp.blogspot.com/p/phd-thesis.html
#
# The referred Matlab reference code comprises the five functions
# mpiir_l2(), update(), locmax(), lslevin(), levin()
# that were back then developed with Matlab 5.0 / 5.1.
#
# These five functions have Python equivalents below.
#
# Note that Matlab's matrix operations a/b and a\b require
# e.g. scipy.linalg.solve() or scipy.linalg.lstsq() ports,
# which might behave slightly different than the Matlab operators.
# Matlab's a\b has a decision tree to choose the optimum
# solver algorithm for the given matrix characteristics.
# For rank deficient matrices Matlab's a\b finds the sparsest result.
# We however assume here, that all our involved matrices for a/b and a\b
# are square, symmetric, full rank. In Matlab, for this case
# very likely LU or Cholesky based solvers are utilised.
# For the Python port we handle symmetric information explicitly to
# scipy.linalg.solve(, , assume_a='sym')
#
# Note that a quadratic program needs to be solved in update(),
# for which we are not interested to fully replicate
# the originally employed Optimization Toolbox function qp()
# of Matlab 5.0 and/or 5.1.
# We however ensure that deterministic algorithmic parts
# match with the Matlab reference code, being debugged
# with MATLAB Version: 23.2.0.3097123 (R2023b) Update 11
# version -blas -> 'OpenBLAS 0.3.21'
# version -lapack -> 'NAG Performance Components (NPC) Release 1.2.0,
# supporting Linear Algebra PACKage (LAPACK 3.9.1)'
# on Operating System: macOS  Version: 26.5.1 Build: 25F80
# The Matlab code was slightly adapted to work
# with the recent quadprog() instead of the obsolete qp() solver.
#
# It appears tempting to utilise the "qpsolvers" package for the
# Python port. This allows to use e.g. solvers like "quadprog",
# "cvxopt" within the same framework. Please check
# "qpsolvers: Quadratic Programming Solvers in Python"
# https://github.com/qpsolvers/qpsolvers, version = 4.12.0, 2026
# Many of those solvers can be explicitly tuned with respect to
# tolerance limits and checks. We go for the defaults here,
# but feel free to use the tuning if needed, an approriate kwargs
# handling is then TBD.
#
# Besides citing [Lan1999] and [Lan2000] in the code,
# we also use references to the highly recommended textbooks
# [Ant2021] Andreas Antoniou, Wu-Sheng Lu (2021):
# "Practical Optimization---Algorithms and Engineering Applications"
# Springer Science+Business Media, LLC, 2nd edition
# [Ant1993] Andreas Antoniou (1993):
# "Digital Filters---Analysis, Design, And Applications"
# McGraw-Hill, 2nd edition

import numpy as np
import sys
import warnings

# these full imports are only needed for package version check:
import scipy as sp
import qpsolvers  # uv pip install "qpsolvers[open_source_solvers]"
import cvxopt

# the actually used (few) functions from qpsolvers and scipy are:
from qpsolvers import Problem, solve_problem
from scipy.fft import rfft
from scipy.linalg import LinAlgWarning, solve, solve_toeplitz
from scipy.signal import convolve, freqz, tf2zpk
# Note that scipy.linalg.solve is favoured to numpy.linalg.solve
# due to the optimised handling of symmetric matrix properties.


def get_package_versions():
    """ Print package versions.

    init dev / test as of July 2026 with Python 3.13.7 and
    - numpy ver: 2.4.6
    - scipy ver: 1.17.1
    - qpsolvers ver: 4.12.0
    - quadprog ver: 0.1.13
    - cvxopt ver: 1.3.3
    """
    print('Package Versions:')
    print('\tPython:', sys.version)
    print('\tnumpy:', np.__version__)
    print('\tscipy:', sp.__version__)
    print('\tqpsolvers:', qpsolvers.__version__)
    print('\tcvxopt:', cvxopt.__version__)
    print(qpsolvers.available_solvers)
    # get open source available_solvers with
    # uv pip install "qpsolvers[open_source_solvers]"
    print(np.show_config())  # checking BLAS and LAPACK versions
    print(sp.show_config())  # might be reasonable for np and sp
    print('######')


def mpiir_l2(Nz, Np, om, D, W, max_pole_radius,
             **kwargs):
    """ M.C. Lang's Least Squares Digital IIR Filter Design.

    Target filter frequency response can have arbitrary magnitude and phase.
    Poles of digital filter are constrained by maximum pole radius.

    - author of Matlab reference code:
    Mathias C. Lang, Vienna University of Technology, 1998
    - author of this Python port:
    Frank Schultz, University of Rostock, 2026

    cf. [Lan1999] Chapter5, code @ pg.171ff

    cf. [Lan2000]: https://doi.org/10.1109/78.875468

    Parameters
    ----------
    Nz : order of numerator polynomial (Nz=M non-origin zeros)
    Np : order of denominator polynomial (Np=N non-origin poles)
    om : digital frequency vector (0 <= om <= pi)
        np.shape: (len(om), )
    D : complex desired/target frequency response along om
        np.shape: (len(om), )
    W : positive weighting function along om
        np.shape: (len(om), )
    max_pole_radius : maximum pole radius for digital filter,
        user definable constraint, 0 < max_pole_radius < 1
    a0 : initial denominator guess, optional, default: empty
        np.shape: (Np+1, )
    tol_mpiir_l2 : tolerance outer loop, optional, default: 1e-4
    alpha : learning rate, >0, optional, default: 0.5
        connected to pole alignment, hence |alpha|<1!
    verbose_descent : verbose print True / False flag, optional,
        default: False
    tol_update : tolerance inner loop, optional, default: 1e-6
    cnt_max : maximum number of inner loop runs, optional, default: 50
    nfft : frequency resolution for pole placement, optional, default: 1024
        must be even, ideally a power of two
    itmax : iterations for pole refinement with Newton's method, optional,
        default: 10
    solver : solver algorithm, string, optional,
        default: "quadprog"
    verbose_solver : verbose print True / False flag, optional
        default: False
    log_file_str: name for log file, string, optional,
        default: "mpiir_l2_default_log.txt"

    Returns
    -------
    b : digital filter numerator polynomial coefficients
        np shape: (Nz+1, )
    a : digital filter denominator polynomial coefficients
        np shape: (Np+1, )
    l2error : least squares approximation error
    """
    M, N, r = Nz, Np, max_pole_radius

    srW = np.sqrt(W)
    EM = np.exp(-1j * om[:, None] @ np.arange(np.max([M, N])+1)[None, :])
    tol = kwargs.get("tol_mpiir_l2", 1e-4)
    alpha = kwargs.get("alpha", 0.5)
    log_file_str = kwargs.get("log_file_str", 'mpiir_l2_default_log.txt')

    descent_counter = 0  # additional counter
    verbose_descent = kwargs.get("verbose_descent", False)
    if verbose_descent:
        print('# ITER\t L2 ERROR\t MAX.RADIUS\t STEP SZ\t SLOPE')

    with open(log_file_str, "w") as f:
        f.write('ITER   L2 ERROR         MAX.RADIUS       STEP SZ          SLOPE\n')

    ini = False  # because we might have some useful a0 init
    a0 = kwargs.get("a0", np.empty(0))
    if a0.size > 0:
        a = np.copy(a0).astype('float64')  # ensure a float64 copy
        if len(a) != N+1:  # polynomial order does not match
            ini = True
        elif np.isclose(a[0], 0.):  # filter has no output signal
            ini = True
        elif np.max(np.abs(np.roots(a))) > r:  # pole radius is too large
            ini = True
        else:
            if not np.isclose(a[0], 1.):  # we rely on a[0]=1
                a /= a[0]
            _, A = freqz(a, 1, om, fs=2*np.pi)
            b = lslevin(M+1, om, A*D, W / np.real(np.abs(A)**2))
    else:
        ini = True
    if ini:
        # FIR as initial solution, i.e. all poles in origin
        # and some nice zero locations as starting point
        a = np.zeros(N+1)
        a[0] = 1.
        b = lslevin(M+1, om, D, W)
    x = np.concatenate(([a[1:], b]))  # fixed coeff a0=1 is not part of it
    delta = np.zeros_like(x)

    outer_loop = True
    while outer_loop:  # Gauss-Newton iteration
        # "compute complex error, Jacobian, and objective function value"
        A = np.squeeze(EM[:, :N+1] @ a[:, None])
        B = np.squeeze(EM[:, :M+1] @ b[:, None])
        H = B / A  # manual freqz()

        # |E|^2=W*|D-H|^2, [Lan2000, Eq. (2)], [Lan1999, Eq. (5.2)]:
        E = srW * (D-H)

        l2error = np.real(E.conj().T @ E)  # |E|^2 = E^H@E is real
        if N < 1:  # what do to with no poles at all?
            break  # we go for the FIR solution
        vec1 = srW / A  # grad(H) wrt coef b (incl. sqrt(W) for convenience)
        vec2 = -vec1 * H  # grad(H) wrt coef a (incl. sqrt(W) for convenience)
        # Jacobian, cf. [Ant2021, Ch. 5.4]:
        J = np.hstack((EM[:, 1:N+1] * vec2[:, None],  # [Lan1999, Ch. 5.4.1]
                       EM[:, 0:M+1] * vec1[:, None]))  # [Lan2000, Eq. (8)]

        # "compute search direction"
        delta0 = np.copy(delta)  # ensure a copy as delta gets changed
        delta, how = update(J, E, a, r, M, N, **kwargs)
        if not how:
            delta = np.copy(delta0)  # ensure a copy to avoid an alias
            break

        # "update solution" [Ant2021, Algorithm 5.6 Step 6]
        # [Ant1993, Algorithm 1 Step 5, pg.495]
        x += alpha * delta
        # extract filter coeff
        a = np.insert(x[:N], 0, 1.)  # denominator polynomial for N poles
        b = x[N:]  # numerator polynomial for M zeros

        # "display results" -> print in terminal
        step = np.linalg.norm(delta) / np.linalg.norm(x)
        pr = np.max(np.abs(np.roots(a)))
        slope = np.squeeze(
            # note: -2 real(E'J) delta = -2 real(J'E)^T delta
            # cf. [Lan1999, Eq. (5.10), last line, middle term]
            -2 * np.real(np.conj(E[None, :]) @ J) @ delta[:, None] / l2error
            )
        if verbose_descent:
            print(f'# {descent_counter:d}\t {l2error:+5.6e}\t {pr:+5.6e}\t {step:+5.6e}\t {slope:+5.6e}')

        with open(log_file_str, "a") as f:
            f.write(f'{descent_counter:03d}    {l2error:+5.6e}    {pr:+5.6e}    {step:+5.6e}    {slope:+5.6e}\n')

        descent_counter += 1

        # "check stopping criterion"
        if (step < tol and np.abs(slope) < tol) or (pr > r) or (descent_counter > 1000):
            z, p, k = tf2zpk(b, a)
            with open(log_file_str, "a") as f:
                f.write('\ncoef b:\n\t'+str(b)+'\n')
                f.write('\ncoef a:\n\t'+str(a)+'\n')
                f.write('\npoles:\n\t'+str(p)+'\n')
                f.write('\nzeros:\n\t'+str(z)+'\n')
                f.write('\ngain k:\n\t'+str(k)+'\n')
                f.write('\nmax. desired pole radius:\n\t'+str(r)+'\n')
                f.write('\nmax. obtained pole radius:\n\t'+str(np.max(np.abs(p)))+'\n')
                f.write('\nradius poles:\n\t'+str(np.abs(p))+'\n')
                f.write('\nangle poles:\n\t'+str(np.angle(p))+'\n')
                f.write('\nradius zeros:\n\t'+str(np.abs(z))+'\n')
                f.write('\nangle zeros:\n\t'+str(np.angle(z))+'\n')
            break  # terminate outer_loop due to nice solution?!

    return b, a, l2error


def update(J, E, a, r, M, N,
           **kwargs):
    """ Subroutine for mpiir_l2() (IIR filter design).

    Computes solution update subject to constraint on pole radii.
    Applies Rouche's Theorem.

    - author of Matlab reference code:
    Mathias C. Lang, Vienna University of Technology, 1998
    - author of this Python port:
    Frank Schultz, University of Rostock, 2026

    cf. [Lan1999] Chapter 5, code @ pg.171ff

    cf. [Lan2000] Sec. III (B) / (C)

    Note: we actually don't use M in this Python port as it can be deduced
    by dim of J and N

    Parameters
    ----------
    J : Jacobian matrix of actual frequency response
        np shape: (len(om), M+N+1)
    E : complex-valued error
        np shape: (len(om), )
    a : actual denominator coefficients of digital filter
        np shape: (N+1, )
    r : maximum pole radius for digital filter (i.e. user definable constraint,
        0 < r < 1)
    M : order of numerator polynomial (for M non-origin zeros)
    N : order of denominator polynomial (for N non-origin poles)
    tol_update : tolerance inner loop, optional, default: 1e-6
    cnt_max : maximum number of inner loop runs, optional, default: 50
    nfft : frequency resolution for pole placement, optional, default: 1024
        must be even, ideally power of two
    itmax : iterations for pole refinement with Newton's method, optional,
        default: 10
    solver : solver algorithm, string, optional,
        default: "quadprog"
    verbose_solver : verbose print True / False flag, optional,
        default: False

    Returns
    -------
    delta : filter coefficient updates for [a1 a2...aN b0 b1 b2...bM]
        np shape: (M+N+1, )
    how :  True / False as flag for 'quadratic program solution found'
    """
    # get kwarg or set kwarg default:
    tol = kwargs.get("tol_update", 1e-6)
    cnt_max = kwargs.get("cnt_max", 50)
    nfft = kwargs.get("nfft", 2**10)  # even!, a power of two is meaningful
    itmax = kwargs.get("itmax", 10)
    solver = kwargs.get("solver", "quadprog")  # "quadprog", "cvxopt"...
    verbose = kwargs.get("verbose_solver", False)

    se = np.sqrt(np.finfo(float).eps)
    # reset to initial:
    cnt = 0
    how = True

    om = 2*np.pi * np.arange(nfft//2+1) / nfft
    # avoid mods of array 'a' due to pass-by-object reference:
    a_tmp = np.copy(a / r**np.arange(0, N+1))
    A = np.abs(rfft(a_tmp, nfft))  # denominator frequency response at radius r

    # auto-corr, cf. [Lan1999, Eq. (5.21)]
    ra = convolve(a_tmp, a_tmp[::-1], mode="full")
    ra = ra[N:]

    tmp = np.arange(1, N+1)  # prep for last two equations in [Lan1999, pg.146]
    nra = tmp * ra[1:]
    n2ra = tmp * nra

    R = 1 / (r**tmp)
    Bact = []
    cact = []

    # "solve unconstrained problem"
    E = E[:, None]  # make E a column for matrix ops
    J = np.vstack((J.real, J.imag))
    E = np.vstack((E.real, E.imag))
    # approximate the Hessian for Gauss-Newton
    # cf. [Ant2021, Ch. 5.4 & Algorithm 5.6]
    H = J.T @ J  # [Ant2021, Eq.(5.35)],[Lan1999, Eq.(5.13)],[Lan2000,Eq.(8)]
    H += se * np.eye(H.shape[0])  # avoid singularity
    f = J.T @ E  # [Ant2021, below Eq. (5.35)],
    # [Lan1999, Eq. (5.13)],[Lan2000,Eq.(8)]

    # print(f'cond(H) = {np.linalg.cond(H):+5.6e}')

    # solving H delta = f for unknown delta, cf.
    # [Ant2021, below Eq. (5.35)], [Lan1999, Eq.(5.14)], [Lan2000,Eq.(9)]
    # [Ant1993, Ch. 14.3]:
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=LinAlgWarning)
        try:
            # Matlab's delta=H\f solves this very problem
            # probably via LU or Cholesky. We go for:
            delta = solve(H, f, assume_a='sym')  # H is square & symmetric
            # so provide this information via assume_a, which really helps
            # to get solutions that match with that of the Matlab world
            # [Lan2000, p.3111]: "H is real and positive definite"
            # we here assume full rank for H, but if not, we have the except:
        except LinAlgWarning:
            # we better stop if this warning occurs
            print('LinAlgWarning for solve(H, f, assume_a="sym")')
            sys.exit("H is ill-conditioned")

    # "compute matrices and vectors for qp-subproblems"
    # only the coeff a are optimised with the pole constraint
    # hence use only these relevant H matrix entries
    # by partioning H as follows
    # cf. the Appendix in [Lan2000], Ch. 5.4.3 in [Lan1999]
    H11 = H[:N, :N]
    H12 = H[:N, N:]
    H22 = H[N:, N:]
    # Matlab's operator handling for Hh=H12/H22 needs this in Python:
    # https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    # we again assume full rank for square & symmetric H22
    # setup H22^T Hh^T = H12^T -> solve for Hh^T ->
    # then Hh^T = H22^T^-1 H12^T -> Hh = H12 H22^-1
    # cf. [Lan1999, Eq. (5.25)], [Lan2000, Eq. (23)]
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=LinAlgWarning)
        try:
            Hh = solve(H22.T, H12.T, assume_a='sym').T
        except LinAlgWarning:
            # we better stop if this warning occurs
            print('LinAlgWarning for solve(H22.T, H12.T, assume_a="sym").T')
            sys.exit("H22.T is ill-conditioned")

    # [Lan2000, Eq. (23)], [Lan1999, Eq. (5.25)]:
    H = H11 - Hh @ H12.T  # H is not longer needed, thus overwrite it
    # H should be real, square, symmetric in theory
    # ensure this also numerically by averaging:
    H = (H + H.T) / 2

    # do coeff a vs. coeff b partioning also for the f vector
    f1, f2 = f[:N], f[N:]
    # f can be overwritten towards the f that is then used for coef a optimise
    f = Hh @ f2 - f1  # cf. [Lan2000, Eq. (23)], [Lan1999, Eq. (5.25)]
    # f2 is used for the coef b Gauss-Newton update, cf. last lines of this
    # function

    # "iterate" -> inner_loop using the set up H and f
    # of the quadratic program [Lan2000, Eq. (23)], [Lan1999, Eq. (5.25)]
    while cnt < cnt_max:

        # "compute error maxima on a grid"
        if len(delta) > 1:  # squeeze only if more than one value
            delta = np.squeeze(delta)  # then...
        d = np.copy(delta[:N])  # this works for all delta, copy needed (why?)!
        # Matlab: "d = d.*(r.^-(1:N).');"
        # 1 / (r**np.arange(1, N+1))
        # we have it already in R from above, hence
        d *= R
        D = np.abs(rfft(d, nfft))  # delta  frequency response at radius r
        # |D|^2-|A|^2 is better for finding the local maxima,
        # i.e. prep for finding the active constraints:
        Imax = locmax(D**2-A**2)  # [Lan2000, Eq. (16)], [Lan1999, Eq. (5.19)]
        # the constraint is however implemented as non-squared below, cf.
        # [Lan2000, Eq. (13)], [Lan1999, Eq. (5.18)]
        ommax = om[Imax]
        if Imax.size == 0:
            break  # no maxima found -> no resonances

        # ommax is theoretically continuous in [0,2pi), hence we
        # "refine maxima using Newton's method"
        # Newton method to refine pole positions from dense FFT-grid
        # cf. [Lan2000, p.3113, mid of right column]
        if N > 1:
            rd = convolve(d, d[::-1])  # [Lan1999, Eq. (5.21)]
            rd = rd[N-1:]
            tmp = np.arange(1, N)
            nrd = tmp * rd[1:]
            n2rd = tmp * nrd

        # the theory for this loop is discussed in [Lan2000, Sec. IIIC],
        # [Lan1999, pg.145ff], it is processed in matrix form
        # for multiple theta = ommax at the same time -> multiple exchange
        # the factor 2 included in the theory is omitted as it cancels
        for i in range(itmax):
            # print(i+1, '/', itmax)
            Mc = np.cos(np.outer(ommax, np.arange(1, N+1)))
            Ms = np.sin(np.outer(ommax, np.arange(1, N+1)))
            gp = -Ms @ nra[:, None]  # this is for c'(theta), cf. p.146
            gpp = -Mc @ n2ra[:, None]  # this is for c''(theta)
            if N > 1:
                gp += Ms[:, :N-1] @ nrd[:, None]  # delta stuff to c'(theta)
                gpp += Mc[:, :N-1] @ n2rd[:, None]  # delta stuff to c''(theta)
            Ipp = np.flatnonzero(gpp)
            if Ipp.size == 0:
                break
            # Newton update -> theta[new] = theta[old] - c'/c''
            # [Lan1999, Eq. (5.20)]:
            ommax[Ipp] -= np.squeeze(gp[Ipp] / gpp[Ipp])

        # "find violating maxima"
        # cf. [Lan2000, pg.3113 right column]
        Dmax = np.exp(-1j * np.outer(ommax, np.arange(1, N+1))) @ d[:, None]
        Amax = np.exp(-1j * np.outer(ommax, np.arange(0, N+1))) @ a_tmp[:, None]
        Iviol = np.where(np.abs(Dmax) > np.abs(Amax))[0]  # [Lan1999, Eq. (5.18)]
        omviol = ommax[Iviol]
        nviol = len(Iviol)
        Dviol = Dmax[Iviol]
        Aviol = Amax[Iviol]

        # "check stopping criterion"
        tmp1 = np.abs(Dmax) - np.abs(Amax)
        tmp2 = np.fmax(np.abs(Amax)*tol,
                       np.zeros_like(Amax)+se)
        if (nviol == 0) or (tmp1 <= tmp2).all():
            break
        cnt += 1

        # "formulate new constraints", cf. [Lan1999, p.147], [Lan2000, p.3113]
        PDviol = np.angle(Dviol)  # phase for Dviol
        # cf. equation between (5.22) and (5.23) in [Lan1999]:
        B = R * np.cos(np.outer(omviol, np.arange(1, N+1)) +
                       np.outer(PDviol, np.ones(N)))  # C in [Lan1999]
        c = np.abs(Aviol)  # d in [Lan1999]
        # these are the new constraints to be added to the
        # current active constraints:

        # "solve subproblem"
        Bact.append(B)  # append newest stuff to current list
        cact.append(c)
        B = np.vstack(Bact)  # as matrix to work with solve_qp()
        c = np.vstack(cact)

        # quadratic problem with linear inequality constraints
        # [Lan2000, Eq. (17)], [Lan1999, Eq. (5.22)]
        problem = Problem(H, f, B, c,
                          None, None, None, None)
        solution = solve_problem(problem,
                                 solver,
                                 verbose=verbose)
        how = solution.found
        delta = solution.x
        lam = solution.z  # get Lagrange for inequality constraints
        if not how:
            print('Warning: inner loop stopped !')
            print('QP solver found solution:', how, 'at inner loop cnt:', cnt)  # this verbose print might be helpful
            # for the cases where the solver cannot find a solution
            break  # its probably better to stop then

        # init new lists to prep for next iter
        Bact, cact = [], []

        # "find active constraints"
        act = lam > 0
        Bact.append(B[act, :])  # append lists by current matrix stuff
        cact.append(c[act, :])

    # "add numerator coefficient update"
    # optimisation for coeff b is not constrained, hence
    # from the partioned H matrices we just solve() with H22
    # H22 is square, symmmetric, thus
    # from above Hh = H12 H22^-1 -> [Lan2000, Eq. (22)]: Hh^T = H22^-1 H12^T
    # cf. also [Lan1999, Eq. (5.24)]
    if len(delta) == N:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=LinAlgWarning)
            try:
                tmp = solve(H22, f2, assume_a='sym') - Hh.T @ delta[:, None]
            except LinAlgWarning:
                # we better stop if this warning occurs
                print('LinAlgWarning for solve(H22, f2, assume_a="sym")')
                sys.exit("H22 is ill-conditioned")
        if len(tmp) == 1:  # make this if/else nicer?!
            delta = np.concatenate((delta, tmp[0]))
        else:
            delta = np.concatenate((delta, np.squeeze(tmp)))

    return delta, how


def locmax(x):
    """ Finds indices of local maxima of a signal/spectrum in array x.

    - author of Matlab reference code:
    Mathias C. Lang, Vienna University of Technology, 1998
    - author of this Python port:
    Frank Schultz, University of Rostock, 2026

    cf. [Lan1999]: p. 43, Chapter 2.2

    Parameters
    ----------
    x:  numpy array with data samples along a single axis

    Returns
    -------
    idx: numpy rank1 array with indices of local maxima, might be empty
    """
    x = np.squeeze(x)  # we assume that x at least contains two entries
    n = len(x)  # such that len(x) works after squeezing
    eps = np.finfo(float).eps
    if n:
        tmp1 = np.concatenate(([x[0]*(1-eps)-1], x[:n-1]))
        tmp2 = np.concatenate((x[1:], [x[-1]*(1-eps)-1]))
        idx = np.where((x > tmp1) & (x > tmp2))[0]
    else:
        idx = np.empty(0)
    return idx


def lslevin(N, om, D, W):
    """ Complex least-squares FIR filter design with Levinson's algorithm.

    - author of Matlab reference code:
    Mathias C. Lang, Vienna University of Technology, 1998
    - author of this Python port:
    Frank Schultz, University of Rostock, 2026

    cf. [Lan1999]: Ch. 2.1, Eq. (2.9), Eq. (2.10), text below Eq. (2.10),
    code on p. 20

    cf. [Ant2021]: Ch. 9.4.2.3, Eq. 9.47ff

    Parameters
    ----------
    N : filter length = number of FIR coefficients (N-1 zeros)
    om : digital frequency vector (0 <= om <= pi)
        np shape: (len(om), 1)
    D : complex desired/target frequency response along om
        np shape: (len(om), 1)
    W : positive weighting function along om
        np shape: (len(om), 1)

    Returns
    -------
    h : impulse response of FIR filter
        np shape: (N, )
    """
    L = len(om)
    # DR = D.real  # not needed
    # DI = D.imag  # not needed
    a = np.zeros(N)
    b = np.zeros(N)

    # "Set up vectors for quadratic objective function
    # (avoid building matrices)"
    dvec = np.copy(D)  # make a copy, as dvec gets iterated
    # and we need to avoid pass-by-object-reference mods
    evec = np.ones(L, dtype='complex128')
    e1 = np.exp(+1j * om)

    for i in range(N):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # to ignore the np.ComplexWarning
            # TBD: we should only ignore this specific warning, which is
            # currently not working somehow, so we ignore all
            # warnings for just these two lines:
            a[i] = np.squeeze(W[None, :] @ evec.real[:, None])
            b[i] = np.squeeze(W[None, :] @ dvec.real[:, None])
        evec *= e1
        dvec *= e1

    a /= L
    b /= L

    # "Compute weighted l2 solution"
    h = levin(a, b)
    return h


def levin(a, b):
    """ Solves system of complex linear equations toeplitz(a) * x = b.

    - author of Matlab reference code:
    Mathias C. Lang, Vienna University of Technology, 1998
    - author of Python port:
    Frank Schultz, University of Rostock, 2026

    cf. [Lan1999]: p. 201, Appendix B.1, Eq. (2.9)

    cf. [Ant2021]: Eq. 9.47ff, Ch. 9.4.2.3

    We go for scipy.linalg's solve_toeplitz() instead of porting
    the original Matlab code of levin.m

    Note that Matlab does not have an equivalent
    solve_toeplitz() to solve T x = b for unknown x and any(!) b

    Parameters
    ----------
    a : first column of positive definite Hermitian Toeplitz matrix
    b : right hand side vector, i.e. weighted target frequency response
        (do not confuse b and a here with filter coefficients,
         cf. documentation of solve_toeplitz() for further details)

    Returns
    -------
    impulse response h of the optimum FIR filter
    """
    return solve_toeplitz(a, b)
