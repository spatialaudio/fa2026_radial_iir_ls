# fa2026_radial_iir_ls

## Paper
Frank Schultz, Nara Hahn, Sascha Spors (2026): "Discrete-time IIR filter design for radial filters by numerically optimised pole/zero placement." In: *Proc. of Forum Acusticum*, Graz, Austria, September 8-12, 2026. https://forum-acusticum.org/fa2026/

- [paper (pdf)](Schultz_2026_Dode_Radial_With_IIR_FIR_LS.pdf)
- [slides? (pdf)]()
- [poster? (pdf)]()

## Abstract
For a certain mode in spherical wave field expansions, a radial filter
characterises the radially dependent diffraction phenomenon due to a spherical
obstacle.
Recently, a band-limited impulse invariance method (BLIIM) was proposed.
It allows a more precise design of discrete-time filters for improved numerical
simulations of such diffraction phenomena.
However, for frequencies close to the Nyquist frequency and/or
for high modal orders, BLIIM is numerically not straightforward
or not feasible.
Then, an optimisation-based infinite impulse response filter design with the
continuous-time frequency response as target function,
solving for the coefficients of the $z$-domain transfer
function is one obvious workaround.

An optimisation problem is exemplarily discussed for the
radial filter occurring in the velocity-to-pressure spherical exterior expansion.
This high-pass filter exhibits a steep slope and an extensively rippled pass band
for a high modal order.
The numerical experiments show that precise magnitude and phase responses of
such radial filters can be achieved, requiring only some informed heuristics
for the filter design parameters.
The proposed approach outperforms IDFT-based frequency sampling,
but not the BLIIM.
The approach can be used where BLIIM is not feasible.

## Theory in Short

For the $n$-th spherical mode, the highpass frequency response

$$H^{HP}_n(\omega,r,R) = -\mathrm{j} \cdot \frac{r}{R} \cdot \frac{h_n^{(2)}(\frac{\omega}{c}r)}{h_n^{'(2)}(\frac{\omega}{c}R)} \cdot \text{e}^{\text{+j} \frac{\omega}{c} (r-R)}$$

is the minimum-phase, amplitude-normalised version of the $n$-th order radial filter being involved in spherical, exterior wave field expansions. With rigid spherical radiator / source radius $R$ and freefield radius $r$, we hence assume $r>R$. Angular frequency $\omega$, speed of sound $c$, imaginary unit $\text{j}$, the $n$-th order spherical Hankel function $h^{(2)}_n(\cdot)$ of second kind and its derivative, and the time convention $\text{e}^{+\text{j} \omega t}$ are used. The latter implies that the `scipy.fft.fft()` is the transform from temporal domain to temporal-frequency domain, i.e. the quasi-standard engineering convention.

The highpass filter has $n+1$ poles and $n+1$ zeros in the Laplace-domain.

If $n$ is very high, the digital filter design based on mapping Laplace-domain to z-domain is not straightforward due to numerical issues in robustely finding the (many) roots of the numerator and denominator polynomials of the Laplace transfer function. See older papers by Hahn, Schultz, Spors.

Thus, we here motivate a discrete-time filter design by numerical finding optimum pole / zero positions directly in the z-domain
- using the above given frequency response directly as the target/desired frequency response, 
- employing a constraint for maximum allowed pole radius, and
- solving for the least-squares error.

Essentially, to match the magnitude and phase of the highpass' narrow passband for high $n$, we do not neccesarily need $n+1$ poles in the discrete-time domain. The steep slope of the highpass is realised by many zeros in discrete-time domain closely aligned around the unit circle in the stop band region. A short pre-delay (i.e. an FIR part) helps a lot.

## Filter Design Approach: Least-Squares IIR / FIR for Complex-Valued Frequency Response with User-Defineable Maximum-Allowed Pole Radius

We utilise the algorithm from M. Lang presented in:
- [Lan2000] Mathias C. Lang (2000): "Least-Squares Design of IIR Filters with Prescribed Magnitude and Phase Responses and a Pole Radius Constraint." *IEEE Transactions on Signal Processing*, Volume **48**, Issue **11**, Pages 3109-3121. November 2000, https://doi.org/10.1109/78.875468
- [Lan1999] Mathias Lang (1999): "Algorithms for the Constrained Design of Digital Filters with Arbitrary Magnitude and Phase Responses." doctoral thesis, TU Wien (University of Technology in Vienna, Austria), thesis is downloadable at https://mattsdsp.blogspot.com/p/phd-thesis.html

A Matlab reference code is given in Lang's thesis and comprises the dedicated functions
- `mpiir_l2()` (main solver function)
- `update()` (helper to solve quadratic program subproblem)
- `locmax()` (helper to find local maxima in discrete data)
- `lslevin()` (FIR filter designer for initial solution)
- `levin()` (helper for FIR filter, cf. `solve_toeplitz` of scipy.linalg)

We ported the functionality of these five functions to Python. The Python code can be found in `util.py` in this repository.

Some of the examples given in [Lan1999] are checked with `unit_test_python_port_lang_examples.py`.

A 2nd order shelving filter is designed and checked with `unit_test_python_port.py`. 

The filters for the actual paper are designed and checked with `radial_filter_design.py`.

The Matlab reference code and our Python port yield highly comparable results for these filters. Due to involved different numerical linear algebra and numerical optimisation in the Matlab vs. Python worlds, we should not expect precision down to bit-accuracy.

# Filter Example IV: Numerical IIR Highpass Filter Design with mpiir_l2()

## Filter Parameters

- `c = 343` m/s, speed of sound
- `fs = 32000` Hz, sampling frequency
- `R = 0.2` m, radius of rigid spherical radiator / source  
- `r = 100 / fs * c + R` = 1.271875 m, freefield radius
- `n = 38`, spherical mode -> 6(n+1) dB/octave highpass stopband slope, which merges to a 6dB/octave slope towards DC.  
- `Preringing_Delay = 10`, samples, note: the complementary lowpass has its impulse response peak exactly at this pre-ringing delay value, cf. Fig. 4(b) right, bottom, orange
- `Np = 2`, number of non-origin poles to be optimised, note: highpass passband is comparably small, hence only few poles are needed
- `Nz = 35`, number of non-origin zeros to be optimised, note: we need some to achieve the steep highpass slope, reasonable rule of thumb: Nz in the range of n...n + Preringing_Delay
- `max_pole_radius = 0.9`, user-defineable constraint, must be >0 and <1
- frequency dependent error weighting: f<8000 Hz: 5000, f>8000 Hz: 1
- log spaced frequency vector with 512 frequencies from 1 Hz to fs/2

## Graphical Results

### Python

![Numerical IIR-highpass filter design with mpiir_l2(): frequency response](python_port_example_4.png)
*Frequency & impulse response and z-plane for highpass of example IV, cf. Figure 4 in the paper. Results of the Python code, see `radial_filter_design.py` and `util.py`.*

### Matlab

![Numerical IIR-highpass filter design with mpiir_l2(): z-plane](zplane_r1.27188_n38_fdly0_pdly10_Np2_Nz35_rmax0.9_HLP0_wf8000_ww5000_alpha0.5.png)
*z-planes for example IV, cf. Figure 4(a) in the paper. Results of the Matlab reference code.*

![Numerical IIR-highpass filter design with mpiir_l2(): frequency response](freqz_r1.27188_n38_fdly0_pdly10_Np2_Nz35_rmax0.9_HLP0_wf8000_ww5000_alpha0.5.png)
*Frequency & impulse responses for example IV, cf. Figure 4(b) in the paper. Results of the Matlab reference code.*

## Numerical Results

### Python:
    - Python 3.13.7
    - numpy ver: 2.4.6
    - scipy ver: 1.17.1
    - qpsolvers ver: 4.12.0
    - cvxopt ver: 1.3.3
    - Blas & Lapack: Apple's Accelerate @ Mac OS 26.5.1, Apple M1 Pro
    - see script `radial_filter_design.py` in this repository

### Matlab
    - 23.2.0.3097123 (R2023b) Update 11
    - Operating System: macOS, Version: 26.5.1, Build: 25F80 
    - OpenBLAS 0.3.21
    - NAG Performance Components (NPC) Release 1.2.0, supporting Linear Algebra PACKage (LAPACK 3.9.1)

### Numerical Difference  

See calculation for `e_atol` and `e_rtol` in `radial_filter_design.py`.

    - filter coefficients b min/max atol error Python vs. Matlab:
        -2.413906852183345e-10 / +2.5781790591317133e-10
    - filter coefficients b min/max rtol error Python vs. Matlab:    
        -2.8731800716386147e-08 / +8.383148064616819e-09
    - filter coefficients a min/max atol error Python vs. Matlab:
        0.0 / +9.693343905325946e-10
    - filter coefficients a min/max rtol error Python vs. Matlab:    
        -1.3969028156424201e-09 / 0.0

Inherently to optimisation tasks, other filters might exhibit higher (or even lower) errors. Hence, and in general: We always should use the algorithms with great care, and we should always thoroughly check if the obtained results meet our demands.
