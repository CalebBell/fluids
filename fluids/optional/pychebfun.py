# -*- coding: utf-8 -*-
"""
Chebfun module
==============
Vendorized version from:
https://github.com/pychebfun/pychebfun/blob/master/pychebfun

The rational for not including this library as a strict dependency is that
it has not been released.

.. moduleauthor :: Chris Swierczewski <cswiercz@gmail.com>
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>
.. moduleauthor :: Gregory Potter <ghpotter@gmail.com>

The copyright notice (BSD-3 clause) is as follows:
    
Copyright 2017 Olivier Verdier

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

from __future__ import division
import operator
from functools import wraps
import numpy as np

import numpy.polynomial as poly
from numpy.polynomial.chebyshev import cheb2poly, Chebyshev
from numpy.polynomial.polynomial import Polynomial

import sys
emach = sys.float_info.epsilon # machine epsilon

global sp_fftpack_ifft
sp_fftpack_ifft = None
def fftpack_ifft(*args, **kwargs):
    global sp_fftpack_ifft
    if sp_fftpack_ifft is None:
        from scipy.fftpack import ifft as sp_fftpack_ifft
    return sp_fftpack_ifft(*args, **kwargs)

global sp_fftpack_fft
sp_fftpack_fft = None
def fftpack_fft(*args, **kwargs):
    global sp_fftpack_fft
    if sp_fftpack_fft is None:
        from scipy.fftpack import fft as sp_fftpack_fft
    return sp_fftpack_fft(*args, **kwargs)

global sp_eigvals
sp_eigvals = None
def eigvals(*args, **kwargs):
    global sp_eigvals
    if sp_eigvals is None:
        from scipy.linalg import eigvals as sp_eigvals
    return sp_eigvals(*args, **kwargs)

global sp_toeplitz
sp_toeplitz = None
def toeplitz(*args, **kwargs):
    global sp_toeplitz
    if sp_toeplitz is None:
        from scipy.linalg import toeplitz as sp_toeplitz
    return sp_toeplitz(*args, **kwargs)

def build_pychebfun(f, domain, N=15):
    fvec = lambda xs: [f(xi) for xi in xs]
    return chebfun(f=fvec, domain=domain, N=N)


def build_solve_pychebfun(f, goal, domain, N=15, N_max=100, find_roots=2):
    cache = {}
    def cached_fun(x):
        # Almost half the points are cached!
        if x in cache:
            return cache[x]
        val = f(x)
        cache[x] = val
        return val
    
    fun = build_pychebfun(cached_fun, domain, N=N)
    roots = (fun - goal).roots()
    
    while (len(roots) < find_roots and len(fun._values) < N_max):
        N *= 2
        fun = build_pychebfun(cached_fun, domain, N=N)
        roots = (fun - goal).roots()
        roots = [i for i in roots if domain[0] < i < domain[1]]
        
    return roots, fun

def chebfun_to_poly(coeffs_or_fun, domain=None, text=False):
    if isinstance(coeffs_or_fun, Chebfun):
        coeffs = coeffs_or_fun.coefficients()
        domain = coeffs_or_fun._domain
    elif hasattr(coeffs_or_fun, '__class__') and coeffs_or_fun.__class__.__name__ == 'ChebyshevExpansion':
        coeffs = coeffs_or_fun.coef()
        domain = coeffs_or_fun.xmin(), coeffs_or_fun.xmax()
    else:
        coeffs = coeffs_or_fun

    low, high = domain
    # Reverse the coefficients, and use cheb2poly to make it in the polynomial domain
    poly_coeffs = cheb2poly(coeffs)[::-1].tolist()
    if not text:
        return poly_coeffs
    s = 'coeffs = %s\n' %poly_coeffs
    delta = high - low
    delta_sum = high + low
    # Generate the expression
    s += 'horner(coeffs, %.18g*(x - %.18g))' %(2.0/delta, 0.5*delta_sum)
    # return the string
    return s

def cheb_to_poly(coeffs_or_fun, domain=None):
    """Just call horner on the outputs!"""
    from fluids.numerics import horner as horner_poly
    
    if isinstance(coeffs_or_fun, Chebfun):
        coeffs = coeffs_or_fun.coefficients()
        domain = coeffs_or_fun._domain
    elif hasattr(coeffs_or_fun, '__class__') and coeffs_or_fun.__class__.__name__ == 'ChebyshevExpansion':
        coeffs = coeffs_or_fun.coef()
        domain = coeffs_or_fun.xmin(), coeffs_or_fun.xmax()
    else:
        coeffs = coeffs_or_fun

    low, high = domain
    coeffs = cheb2poly(coeffs)[::-1].tolist() # Convert to polynomial basis
    # Mix in limits to make it a normal polynomial
    my_poly = Polynomial([-0.5*(high + low)*2.0/(high - low), 2.0/(high - low)])
    poly_coeffs = horner_poly(coeffs, my_poly).coef[::-1].tolist()
    return poly_coeffs


def cheb_range_simplifier(low, high, text=False):
    '''
    >>> low, high = 0.0023046250851646434, 4.7088985707840125
    >>> cheb_range_simplifier(low, high, text=True)
    'chebval(0.42493574399544564724*(x + -2.3556015979345885647), coeffs)'
    '''
    constant = 0.5*(-low-high)
    factor = 2.0/(high-low)
    if text:
        return 'chebval(%.20g*(x + %.20g), coeffs)' %(factor, constant)
    return constant, factor
    


def cast_scalar(method):
    """Cast scalars to constant interpolating objects."""
    @wraps(method)
    def new_method(self, other):
        if np.isscalar(other):
            other = type(self)([other],self.domain())
        return method(self, other)
    return new_method




class Polyfun(object):
    """Construct a Lagrange interpolating polynomial over arbitrary points.

    Polyfun objects consist in essence of two components:     1) An interpolant
    on [-1,1],     2) A domain attribute [a,b]. These two pieces of information
    are used to define and subsequently keep track of operations upon Chebyshev
    interpolants defined on an arbitrary real interval [a,b].
    """

    # ----------------------------------------------------------------
    # Initialisation methods
    # ----------------------------------------------------------------

    class NoConvergence(Exception):
        """Raised when dichotomy does not converge."""

    class DomainMismatch(Exception):
        """Raised when there is an interval mismatch."""

    @classmethod
    def from_data(self, data, domain=None):
        """Initialise from interpolation values."""
        return self(data,domain)

    @classmethod
    def from_fun(self, other):
        """Initialise from another instance."""
        return self(other.values(),other.domain())

    @classmethod
    def from_coeff(self, chebcoeff, domain=None, prune=True, vscale=1.):
        """
        Initialise from provided coefficients
        prune: Whether to prune the negligible coefficients
        vscale: the scale to use when pruning
        """
        coeffs = np.asarray(chebcoeff)
        if prune:
            N = self._cutoff(coeffs, vscale)
            pruned_coeffs = coeffs[:N]
        else:
            pruned_coeffs = coeffs
        values = self.polyval(pruned_coeffs)
        return self(values, domain, vscale)

    @classmethod
    def dichotomy(self, f, kmin=2, kmax=12, raise_no_convergence=True,):
        """Compute the coefficients for a function f by dichotomy.

        kmin, kmax: log2 of number of interpolation points to try
        raise_no_convergence: whether to raise an exception if the dichotomy does not converge
        """

        for k in range(kmin, kmax):
            N = pow(2, k)

            sampled = self.sample_function(f, N)
            coeffs = self.polyfit(sampled)

            # 3) Check for negligible coefficients
            #    If within bound: get negligible coeffs and bread
            bnd = self._threshold(np.max(np.abs(coeffs)))

            last = abs(coeffs[-2:])
            if np.all(last <= bnd):
                break
        else:
            if raise_no_convergence:
                raise self.NoConvergence(last, bnd)
        return coeffs

    @classmethod
    def from_function(self, f, domain=None, N=None):
        """Initialise from a function to sample.

        N: optional parameter which indicates the range of the dichotomy
        """
        # rescale f to the unit domain
        domain = self.get_default_domain(domain)
        a,b = domain[0], domain[-1]
        map_ui_ab = lambda t: 0.5*(b-a)*t + 0.5*(a+b)
        args = {'f': lambda t: f(map_ui_ab(t))}
        if N is not None: # N is provided
            nextpow2 = int(np.log2(N))+1
            args['kmin'] = nextpow2
            args['kmax'] = nextpow2+1
            args['raise_no_convergence'] = False
        else:
            args['raise_no_convergence'] = True

        # Find out the right number of coefficients to keep
        coeffs = self.dichotomy(**args)

        return self.from_coeff(coeffs, domain)

    @classmethod
    def _threshold(self, vscale):
        """Compute the threshold at which coefficients are trimmed."""
        bnd = 128*emach*vscale
        return bnd

    @classmethod
    def _cutoff(self, coeffs, vscale):
        """Compute cutoff index after which the coefficients are deemed
        negligible."""
        bnd = self._threshold(vscale)
        inds  = np.nonzero(abs(coeffs) >= bnd)
        if len(inds[0]):
            N = inds[0][-1]
        else:
            N = 0
        return N+1


    def __init__(self, values=0., domain=None, vscale=None):
        """Init an object from values at interpolation points.

        values: Interpolation values
        vscale: The actual vscale; computed automatically if not given
        """
        avalues = np.asarray(values,)
        avalues1 = np.atleast_1d(avalues)
        N = len(avalues1)
        points = self.interpolation_points(N)
        self._values = avalues1
        if vscale is not None:
            self._vscale = vscale
        else:
            self._vscale = np.max(np.abs(self._values))
        self.p = self.interpolator(points, avalues1)

        domain = self.get_default_domain(domain)
        self._domain = np.array(domain)
        a,b = domain[0], domain[-1]

        # maps from [-1,1] <-> [a,b]
        self._ab_to_ui = lambda x: (2.0*x-a-b)/(b-a)
        self._ui_to_ab = lambda t: 0.5*(b-a)*t + 0.5*(a+b)

    def same_domain(self, fun2):
        """Returns True if the domains of two objects are the same."""
        return np.allclose(self.domain(), fun2.domain(), rtol=1e-14, atol=1e-14)

    # ----------------------------------------------------------------
    # String representations
    # ----------------------------------------------------------------

    def __repr__(self):
        """Display method."""
        a, b = self.domain()
        vals = self.values()
        return (
            '%s \n '
            '    domain        length     endpoint values\n '
            ' [%5.1f, %5.1f]     %5d       %5.2f   %5.2f\n '
            'vscale = %1.2e') % (
                str(type(self)).split('.')[-1].split('>')[0][:-1],
                a,b,self.size(),vals[-1],vals[0],self._vscale,)

    def __str__(self):
        return "<{0}({1})>".format(
            str(type(self)).split('.')[-1].split('>')[0][:-1],self.size(),)

    # ----------------------------------------------------------------
    # Basic Operator Overloads
    # ----------------------------------------------------------------

    def __call__(self, x):
        return self.p(self._ab_to_ui(x))

    def __getitem__(self, s):
        """Components s of the fun."""
        return self.from_data(self.values().T[s].T)

    def __bool__(self):
        """Test for difference from zero (up to tolerance)"""
        return not np.allclose(self.values(), 0)

    __nonzero__ = __bool__

    def __eq__(self, other):
        return not(self - other)

    def __ne__(self, other):
        return not (self == other)

    @cast_scalar
    def __add__(self, other):
        """Addition."""
        if not self.same_domain(other):
            raise self.DomainMismatch(self.domain(),other.domain())

        ps = [self, other]
        # length difference
        diff = other.size() - self.size()
        # determine which of self/other is the smaller/bigger
        big = diff > 0
        small = not big
        # pad the coefficients of the small one with zeros
        small_coeffs = ps[small].coefficients()
        big_coeffs = ps[big].coefficients()
        padded = np.zeros_like(big_coeffs)
        padded[:len(small_coeffs)] = small_coeffs
        # add the values and create a new object with them
        chebsum = big_coeffs + padded
        new_vscale = np.max([self._vscale, other._vscale])
        return self.from_coeff(
            chebsum, domain=self.domain(), vscale=new_vscale
        )

    __radd__ = __add__


    @cast_scalar
    def __sub__(self, other):
        """Subtraction."""
        return self + (-other)

    def __rsub__(self, other):
        return -(self - other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __neg__(self):
        """Negation."""
        return self.from_data(-self.values(),domain=self.domain())


    def __abs__(self):
        return self.from_function(lambda x: abs(self(x)),domain=self.domain())

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------

    def size(self):
        return self.p.n

    def coefficients(self):
        return self.polyfit(self.values())

    def values(self):
        return self._values

    def domain(self):
        return self._domain

    # ----------------------------------------------------------------
    # Integration and differentiation
    # ----------------------------------------------------------------

    def integrate(self):
        raise NotImplementedError()

    def differentiate(self):
        raise NotImplementedError()

    def dot(self, other):
        r"""Return the Hilbert scalar product :math:`\int f.g`."""
        prod = self * other
        return prod.sum()

    def norm(self):
        """
        Return: square root of scalar product with itself.
        """
        norm = np.sqrt(self.dot(self))
        return norm


    # ----------------------------------------------------------------
    # Miscellaneous operations
    # ----------------------------------------------------------------
    def restrict(self,subinterval):
        """Return a Polyfun that matches self on subinterval."""
        if (subinterval[0] < self._domain[0]) or (subinterval[1] > self._domain[1]):
            raise ValueError("Can only restrict to subinterval")
        return self.from_function(self, subinterval)


    # ----------------------------------------------------------------
    # Class method aliases
    # ----------------------------------------------------------------
    diff = differentiate
    cumsum = integrate


class Chebfun(Polyfun):
    """Eventually set this up so that a Chebfun is a collection of Chebfuns.

    This will enable piecewise smooth representations al la Matlab Chebfun v2.0.
    """
    # ----------------------------------------------------------------
    # Standard construction class methods.
    # ----------------------------------------------------------------

    @classmethod
    def get_default_domain(self, domain=None):
        if domain is None:
            return [-1., 1.]
        else:
            return domain

    @classmethod
    def identity(self, domain=[-1., 1.]):
        """The identity function x -> x."""
        return self.from_data([domain[1],domain[0]], domain)

    @classmethod
    def basis(self, n):
        """Chebyshev basis functions T_n."""
        if n == 0:
            return self(np.array([1.]))
        vals = np.ones(n+1)
        vals[1::2] = -1
        return self(vals)

    # ----------------------------------------------------------------
    # Integration and differentiation
    # ----------------------------------------------------------------

    def sum(self):
        """Evaluate the integral over the given interval using Clenshaw-Curtis
        quadrature."""
        ak = self.coefficients()
        ak2 = ak[::2]
        n = len(ak2)
        Tints = 2/(1-(2*np.arange(n))**2)
        val = np.sum((Tints*ak2.T).T, axis=0)
        a_, b_ = self.domain()
        return 0.5*(b_-a_)*val

    def integrate(self):
        """Return the object representing the primitive of self over the domain.

        The output starts at zero on the left-hand side of the domain.
        """
        coeffs = self.coefficients()
        a,b = self.domain()
        int_coeffs = 0.5*(b-a)*poly.chebyshev.chebint(coeffs)
        antiderivative = self.from_coeff(int_coeffs, domain=self.domain())
        return antiderivative - antiderivative(a)

    def differentiate(self, n=1):
        """n-th derivative, default 1."""
        ak = self.coefficients()
        a_, b_ = self.domain()
        for _ in range(n):
            ak = self.differentiator(ak)
        return self.from_coeff((2./(b_-a_))**n*ak, domain=self.domain())

    # ----------------------------------------------------------------
    # Roots
    # ----------------------------------------------------------------
    def roots(self):
        """Utilises Boyd's O(n^2) recursive subdivision algorithm.

        The chebfun
        is recursively subsampled until it is successfully represented to
        machine precision by a sequence of piecewise interpolants of degree
        100 or less. A colleague matrix eigenvalue solve is then applied to
        each of these pieces and the results are concatenated.
        See:
        J. P. Boyd, Computing zeros on a real interval through Chebyshev
        expansion and polynomial rootfinding, SIAM J. Numer. Anal., 40
        (2002), pp. 1666–1682.
        """
        if self.size() == 1:
            return np.array([])

        elif self.size() <= 100:
            ak = self.coefficients()
            v = np.zeros_like(ak[:-1])
            v[1] = 0.5
            C1 = toeplitz(v)
            C2 = np.zeros_like(C1)
            C1[0,1] = 1.
            C2[-1,:] = ak[:-1]
            C = C1 - .5/ak[-1] * C2
            eigenvalues = eigvals(C)
            roots = [eig.real for eig in eigenvalues
                    if np.allclose(eig.imag,0,atol=1e-10)
                        and np.abs(eig.real) <=1]
            scaled_roots = self._ui_to_ab(np.array(roots))
            return scaled_roots
        else:
            try:
                # divide at a close-to-zero split-point
                split_point = self._ui_to_ab(0.0123456789)
                return np.concatenate(
                    (self.restrict([self._domain[0],split_point]).roots(),
                     self.restrict([split_point,self._domain[1]]).roots()))
            except:
                # Seems to have many fake roots for high degree fits
                coeffs = self.coefficients()
                domain = self._domain
                possibilities =  Chebyshev(coeffs, domain).roots()
                return np.array([float(i.real) for i in possibilities if i.imag == 0.0])

    # ----------------------------------------------------------------
    # Interpolation and evaluation (go from values to coefficients)
    # ----------------------------------------------------------------

    @classmethod
    def interpolation_points(self, N):
        """N Chebyshev points in [-1, 1], boundaries included."""
        if N == 1:
            return np.array([0.])
        return np.cos(np.arange(N)*np.pi/(N-1))

    @classmethod
    def sample_function(self, f, N):
        """Sample a function on N+1 Chebyshev points."""
        x = self.interpolation_points(N+1)
        return f(x)

    @classmethod
    def polyfit(self, sampled):
        """Compute Chebyshev coefficients for values located on Chebyshev
        points.

        sampled: array; first dimension is number of Chebyshev points
        """
        asampled = np.asarray(sampled)
        if len(asampled) == 1:
            return asampled
        evened = even_data(asampled)
        coeffs = dct(evened)
        return coeffs

    @classmethod
    def polyval(self, chebcoeff):
        """Compute the interpolation values at Chebyshev points.

        chebcoeff: Chebyshev coefficients
        """
        N = len(chebcoeff)
        if N == 1:
            return chebcoeff

        data = even_data(chebcoeff)/2
        data[0] *= 2
        data[N-1] *= 2

        fftdata = 2*(N-1)*fftpack_ifft(data, axis=0)
        complex_values = fftdata[:N]
        # convert to real if input was real
        if np.isrealobj(chebcoeff):
            values = np.real(complex_values)
        else:
            values = complex_values
        return values

    @classmethod
    def interpolator(self, x, values):
        """Returns a polynomial with vector coefficients which interpolates the
        values at the Chebyshev points x."""
        # hacking the barycentric interpolator by computing the weights in advance
        from scipy.interpolate import BarycentricInterpolator as Bary
        p = Bary([0.])
        N = len(values)
        weights = np.ones(N)
        weights[0] = .5
        weights[1::2] = -1
        weights[-1] *= .5
        p.wi = weights
        p.xi = x
        p.set_yi(values)
        return p

    # ----------------------------------------------------------------
    # Helper for differentiation.
    # ----------------------------------------------------------------

    @classmethod
    def differentiator(self, A):
        """Differentiate a set of Chebyshev polynomial expansion coefficients
        Originally from http://www.scientificpython.net/pyblog/chebyshev-
        differentiation.

        + (lots of) bug fixing + pythonisation
        """
        m = len(A)
        SA = (A.T* 2*np.arange(m)).T
        DA = np.zeros_like(A)
        if m == 1: # constant
            return np.zeros_like(A[0:1])
        if m == 2: # linear
            return A[1:2,]
        DA[m-3:m-1,] = SA[m-2:m,]
        for j in range(m//2 - 1):
            k = m-3-2*j
            DA[k] = SA[k+1] + DA[k+2]
            DA[k-1] = SA[k] + DA[k+1]
        DA[0] = (SA[1] + DA[2])*0.5
        return DA

# ----------------------------------------------------------------
# General utilities
# ----------------------------------------------------------------

def even_data(data):
    """
    Construct Extended Data Vector (equivalent to creating an
    even extension of the original function)
    Return: array of length 2(N-1)
    For instance, [0,1,2,3,4] --> [0,1,2,3,4,3,2,1]
    """
    return np.concatenate([data, data[-2:0:-1]],)

def dct(data):
    """Compute DCT using FFT."""
    N = len(data)//2
    fftdata     = fftpack_fft(data, axis=0)[:N+1]
    fftdata     /= N
    fftdata[0]  /= 2.
    fftdata[-1] /= 2.
    if np.isrealobj(data):
        data = np.real(fftdata)
    else:
        data = fftdata
    return data

# ----------------------------------------------------------------
# Add overloaded operators
# ----------------------------------------------------------------

def _add_operator(cls, op):
    def method(self, other):
        if not self.same_domain(other):
            raise self.DomainMismatch(self.domain(), other.domain())
        return self.from_function(
            lambda x: op(self(x).T, other(x).T).T, domain=self.domain(), )
    cast_method = cast_scalar(method)
    name = '__'+op.__name__+'__'
    cast_method.__name__ = name
    cast_method.__doc__ = "operator {}".format(name)
    setattr(cls, name, cast_method)

def rdiv(a, b):
    return b/a

for _op in [operator.mul, operator.truediv, operator.pow, rdiv]:
    _add_operator(Polyfun, _op)

# ----------------------------------------------------------------
# Add numpy ufunc delegates
# ----------------------------------------------------------------

def _add_delegate(ufunc, nonlinear=True):
    def method(self):
        return self.from_function(lambda x: ufunc(self(x)), domain=self.domain())
    name = ufunc.__name__
    method.__name__ = name
    method.__doc__ = "delegate for numpy's ufunc {}".format(name)
    setattr(Polyfun, name, method)

# Following list generated from:
# https://github.com/numpy/numpy/blob/master/numpy/core/code_generators/generate_umath.py
for func in [np.arccos, np.arccosh, np.arcsin, np.arcsinh, np.arctan, np.arctanh, np.cos, np.sin, np.tan, np.cosh, np.sinh, np.tanh, np.exp, np.exp2, np.expm1, np.log, np.log2, np.log1p, np.sqrt, np.ceil, np.trunc, np.fabs, np.floor, ]:
    _add_delegate(func)


# ----------------------------------------------------------------
# General Aliases
# ----------------------------------------------------------------
## chebpts = interpolation_points

# ----------------------------------------------------------------
# Constructor inspired by the Matlab version
# ----------------------------------------------------------------



def chebfun(f=None, domain=[-1,1], N=None, chebcoeff=None,):
    """Create a Chebyshev polynomial approximation of the function $f$ on the
    interval :math:`[-1, 1]`.

    :param callable f: Python, Numpy, or Sage function
    :param int N: (default = None)  specify number of interpolating points
    :param np.array chebcoeff: (default = np.array(0)) specify the coefficients
    """

    # Chebyshev coefficients
    if chebcoeff is not None:
        return Chebfun.from_coeff(chebcoeff, domain)

    # another instance
    if isinstance(f, Polyfun):
        return Chebfun.from_fun(f)

    # callable
    if hasattr(f, '__call__'):
        return Chebfun.from_function(f, domain, N)

    # from here on, assume that f is None, or iterable
    if np.isscalar(f):
        f = [f]

    try:
        iter(f) # interpolation values provided
    except TypeError:
        pass
    else:
        return Chebfun(f, domain)

    raise TypeError('Impossible to initialise the object from an object of type {}'.format(type(f)))