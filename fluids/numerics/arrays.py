# -*- coding: utf-8 -*-
# type: ignore
"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import division
from math import sin, exp, pi, fabs, copysign, log, isinf, acos, cos, sin, sqrt
import sys

REQUIRE_DEPENDENCIES = False
if not REQUIRE_DEPENDENCIES:
    IS_PYPY = True
else:
    try:
        # The right way imports the platform module which costs to ms to load!
        # implementation = platform.python_implementation()
        IS_PYPY = 'PyPy' in sys.version
    except AttributeError:
        IS_PYPY = False

#IS_PYPY = True # for testing
    
#if not IS_PYPY and not REQUIRE_DEPENDENCIES:
#    try:
#        import numpy as np
#    except ImportError:
#        np = None

__all__ = ['dot', 'inv', 'det', 'solve', 'norm2', 'inner_product',
           'eye', 'array_as_tridiagonals', 'solve_tridiagonal', 'subset_matrix']

def det(matrix):
    """Seem sto work fine.

    >> from sympy import *
    >> from sympy.abc import *
    >> Matrix([[a, b], [c, d]]).det()
    a*d - b*c
    >> Matrix([[a, b, c], [d, e, f], [g, h, i]]).det()
    a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g

    A few terms can be slightly factored out of the 3x dim.

    >> Matrix([[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]]).det()
    a*f*k*p - a*f*l*o - a*g*j*p + a*g*l*n + a*h*j*o - a*h*k*n - b*e*k*p + b*e*l*o + b*g*i*p - b*g*l*m - b*h*i*o + b*h*k*m + c*e*j*p - c*e*l*n - c*f*i*p + c*f*l*m + c*h*i*n - c*h*j*m - d*e*j*o + d*e*k*n + d*f*i*o - d*f*k*m - d*g*i*n + d*g*j*m

    72 mult vs ~48 in cse'd version'

    Commented out - takes a few seconds
    >> #Matrix([[a, b, c, d, e], [f, g, h, i, j], [k, l, m, n, o], [p, q, r, s, t], [u, v, w, x, y]]).det()

    260 multiplies with cse; 480 without it.
    """
    size = len(matrix)
    if size == 1:
        return matrix[0]
    elif size == 2:
        (a, b), (c, d) = matrix
        return a*d - c*b
    elif size == 3:
        (a, b, c), (d, e, f), (g, h, i) = matrix
        return a*(e*i - h*f) - d*(b*i - h*c) + g*(b*f - e*c)
    elif size == 4:
        (a, b, c, d), (e, f, g, h), (i, j, k, l), (m, n, o, p) = matrix
        return (a*f*k*p - a*f*l*o - a*g*j*p + a*g*l*n + a*h*j*o - a*h*k*n 
                - b*e*k*p + b*e*l*o + b*g*i*p - b*g*l*m - b*h*i*o + b*h*k*m 
                + c*e*j*p - c*e*l*n - c*f*i*p + c*f*l*m + c*h*i*n - c*h*j*m 
                - d*e*j*o + d*e*k*n + d*f*i*o - d*f*k*m - d*g*i*n + d*g*j*m)
    elif size == 5:
        (a, b, c, d, e), (f, g, h, i, j), (k, l, m, n, o), (p, q, r, s, t), (u, v, w, x, y) = matrix
        x0 = s*y
        x1 = a*g*m
        x2 = t*w
        x3 = a*g*n
        x4 = r*x
        x5 = a*g*o
        x6 = t*x
        x7 = a*h*l
        x8 = q*y
        x9 = a*h*n
        x10 = s*v
        x11 = a*h*o
        x12 = r*y
        x13 = a*i*l
        x14 = t*v
        x15 = a*i*m
        x16 = q*w
        x17 = a*i*o
        x18 = s*w
        x19 = a*j*l
        x20 = q*x
        x21 = a*j*m
        x22 = r*v
        x23 = a*j*n
        x24 = b*f*m
        x25 = b*f*n
        x26 = b*f*o
        x27 = b*h*k
        x28 = t*u
        x29 = b*h*n
        x30 = p*x
        x31 = b*h*o
        x32 = b*i*k
        x33 = p*y
        x34 = b*i*m
        x35 = r*u
        x36 = b*i*o
        x37 = b*j*k
        x38 = s*u
        x39 = b*j*m
        x40 = p*w
        x41 = b*j*n
        x42 = c*f*l
        x43 = c*f*n
        x44 = c*f*o
        x45 = c*g*k
        x46 = c*g*n
        x47 = c*g*o
        x48 = c*i*k
        x49 = c*i*l
        x50 = p*v
        x51 = c*i*o
        x52 = c*j*k
        x53 = c*j*l
        x54 = q*u
        x55 = c*j*n
        x56 = d*f*l
        x57 = d*f*m
        x58 = d*f*o
        x59 = d*g*k
        x60 = d*g*m
        x61 = d*g*o
        x62 = d*h*k
        x63 = d*h*l
        x64 = d*h*o
        x65 = d*j*k
        x66 = d*j*l
        x67 = d*j*m
        x68 = e*f*l
        x69 = e*f*m
        x70 = e*f*n
        x71 = e*g*k
        x72 = e*g*m
        x73 = e*g*n
        x74 = e*h*k
        x75 = e*h*l
        x76 = e*h*n
        x77 = e*i*k
        x78 = e*i*l
        x79 = e*i*m        
        return (x0*x1 - x0*x24 + x0*x27 + x0*x42 - x0*x45 - x0*x7 - x1*x6 
                + x10*x11 - x10*x21 - x10*x44 + x10*x52 + x10*x69 - x10*x74
                - x11*x20 + x12*x13 + x12*x25 - x12*x3 - x12*x32 - x12*x56
                + x12*x59 - x13*x2 + x14*x15 + x14*x43 - x14*x48 - x14*x57
                + x14*x62 - x14*x9 - x15*x8 + x16*x17 - x16*x23 - x16*x58 
                + x16*x65 + x16*x70 - x16*x77 - x17*x22 + x18*x19 + x18*x26
                - x18*x37 - x18*x5 - x18*x68 + x18*x71 - x19*x4 - x2*x25
                + x2*x3 + x2*x32 + x2*x56 - x2*x59 + x20*x21 + x20*x44 
                - x20*x52 - x20*x69 + x20*x74 + x22*x23 + x22*x58 - x22*x65
                - x22*x70 + x22*x77 + x24*x6 - x26*x4 - x27*x6 + x28*x29 
                - x28*x34 - x28*x46 + x28*x49 + x28*x60 - x28*x63 - x29*x33
                + x30*x31 - x30*x39 - x30*x47 + x30*x53 + x30*x72 - x30*x75
                - x31*x38 + x33*x34 + x33*x46 - x33*x49 - x33*x60 + x33*x63 
                + x35*x36 - x35*x41 - x35*x61 + x35*x66 + x35*x73 - x35*x78 
                - x36*x40 + x37*x4 + x38*x39 + x38*x47 - x38*x53 - x38*x72 
                + x38*x75 + x4*x5 + x4*x68 - x4*x71 + x40*x41 + x40*x61
                - x40*x66 - x40*x73 + x40*x78 - x42*x6 - x43*x8 + x45*x6 
                + x48*x8 + x50*x51 - x50*x55 - x50*x64 + x50*x67 + x50*x76 
                - x50*x79 - x51*x54 + x54*x55 + x54*x64 - x54*x67 - x54*x76
                + x54*x79 + x57*x8 + x6*x7 - x62*x8 + x8*x9)
    else:
        # TODO algorithm?
        import numpy as np
        return float(np.linalg.det(matrix))


def inv(matrix):
    """5 has way too many multiplies.

    >> from sympy import *
    >> from sympy.abc import *
    >> Matrix([a]).inv()
    Matrix([[1/a]])

    >> cse(Matrix([[a, b], [c, d]]).inv())
    Matrix([
    [1/a + b*c/(a**2*(d - b*c/a)), -b/(a*(d - b*c/a))],
    [          -c/(a*(d - b*c/a)),      1/(d - b*c/a)]])

    >> m_3 = Matrix([[a, b, c], [d, e, f], [g, h, i]])
    >> #cse(m_3.inv())

    >> m_4 = Matrix([[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]])
    >> cse(m_4.inv())

    # Note: for 3, 4 - forgot to generate code using optimizations='basic'
    """
    size = len(matrix)
    if size == 1:
        try:
            return [1.0/matrix[0]]
        except:
            return [1.0/matrix[0][0]]
    elif size == 2:
        try:
            (a, b), (c, d) = matrix
            x0 = 1.0/a
            x1 = b*x0
            x2 = 1.0/(d - c*x1)
            x3 = c*x2
            return [[x0 + b*x3*x0*x0, -x1*x2],
                    [-x0*x3, x2]]
        except:
            import numpy as np
            return np.linalg.inv(matrix).tolist()
    elif size == 3:
        (a, b, c), (d, e, f), (g, h, i) = matrix
        x0 = 1./a
        x1 = b*d
        x2 = e - x0*x1
        x3 = 1./x2
        x4 = b*g
        x5 = h - x0*x4
        x6 = x0*x3
        x7 = d*x6
        x8 = -g*x0 + x5*x7
        x9 = c*d
        x10 = f - x0*x9
        x11 = b*x6
        x12 = c*x0 - x10*x11
        x13 = a*e
        x14 = -x1 + x13
        x15 = 1./(-a*f*h - c*e*g + f*x4 + h*x9 - i*x1 + i*x13)
        x16 = x14*x15
        x17 = x12*x16
        x18 = x14*x15*x3
        x19 = x18*x5
        x20 = x10*x18
        return [[x0 - x17*x8 + x1*x3*x0*x0, -x11 + x12*x19, -x17],
                [-x20*x8 - x7, x10*x16*x5*x2**-2 + x3, -x20],
                [ x16*x8, -x19, x16]]
    elif size == 4:
        (a, b, c, d), (e, f, g, h), (i, j, k, l), (m, n, o, p) = matrix
        x0 = 1./a
        x1 = b*e
        x2 = f - x0*x1
        x3 = 1./x2
        x4 = i*x0
        x5 = -b*x4 + j
        x6 = x0*x3
        x7 = e*x6
        x8 = -x4 + x5*x7
        x9 = c*x0
        x10 = -e*x9 + g
        x11 = b*x6
        x12 = -x10*x11 + x9
        x13 = a*f
        x14 = -x1 + x13
        x15 = k*x13
        x16 = b*g*i
        x17 = c*e*j
        x18 = a*g*j
        x19 = k*x1
        x20 = c*f*i
        x21 = x15 + x16 + x17 - x18 - x19 - x20
        x22 = 1/x21
        x23 = x14*x22
        x24 = x12*x23
        x25 = m*x0
        x26 = -b*x25 + n
        x27 = x26*x3
        x28 = -m*x9 + o - x10*x27
        x29 = x23*x8
        x30 = -x25 + x26*x7 - x28*x29
        x31 = d*x0
        x32 = -e*x31 + h
        x33 = x3*x32
        x34 = -i*x31 + l - x33*x5
        x35 = -x11*x32 - x24*x34 + x31
        x36 = a*n
        x37 = g*l
        x38 = h*o
        x39 = l*o
        x40 = b*m
        x41 = h*k
        x42 = c*l
        x43 = f*m
        x44 = c*h
        x45 = i*n
        x46 = d*k
        x47 = e*n
        x48 = d*o
        x49 = d*g
        x50 = j*m
        x51 = 1.0/(a*j*x38 - b*i*x38 - e*j*x48 + f*i*x48 + p*x15 
                 + p*x16 + p*x17 - p*x18 - p*x19 - p*x20 + x1*x39 
                 - x13*x39 + x36*x37 - x36*x41 - x37*x40 + x40*x41
                 + x42*x43 - x42*x47 - x43*x46 + x44*x45 - x44*x50 
                 - x45*x49 + x46*x47 + x49*x50)
        x52 = x21*x51
        x53 = x35*x52
        x54 = x14*x22*x3
        x55 = x5*x54
        x56 = -x27 + x28*x55
        x57 = x52*x56
        x58 = x14*x51
        x59 = x28*x58
        x60 = x10*x54
        x61 = x33 - x34*x60
        x62 = x52*x61
        x63 = x34*x58
        return [[x0 - x24*x8 - x30*x53 + x1*x3*x0*x0, -x11 + x12*x55 - x35*x57, -x24 + x35*x59, -x53],
             [-x30*x62 - x60*x8 - x7, x10*x23*x5*x2**-2 + x3 - x56*x62, x59*x61 - x60, -x62],
             [x29 - x30*x63, -x55 - x56*x63, x14*x14*x22*x28*x34*x51 + x23, -x63],
             [x30*x52, x57, -x59, x52]]
    else:
        return inv_lu(matrix)
        # TODO algorithm?
#        import numpy as np
#        return np.linalg.inv(matrix).tolist()

def eye(N):
    mat = []
    for i in range(N):
        r = [0.0]*N
        r[i] = 1.0
        mat.append(r)
    return mat
   
def dot(a, b):
    try:
        ab = [sum([ri*bi for ri, bi in zip(row, b)]) for row in a]
    except:
        ab = [sum([ai*bi for ai, bi in zip(a, b)])]
    return ab

def inner_product(a, b):
    tot = 0.0
    for i in range(len(a)):
        tot += a[i]*b[i]
    return tot


def inplace_LU(A, ipivot, N):
    Np1 = N+1
    
    for j in range(1, Np1):
        for i in range(1, j):
            tot = A[i][j]
            for k in range(1, i):
                tot -= A[i][k]*A[k][j]
            A[i][j] = tot

        apiv = 0.0
        for i in range(j, Np1):
            tot = A[i][j]
            for k in range(1, j):
                tot -= A[i][k]*A[k][j]
            A[i][j] = tot
            
            if apiv < abs(A[i][j]):
                apiv, ipiv = abs(A[i][j]), i
        if apiv == 0:
            raise ValueError("Singular matrix")
        ipivot[j] = ipiv
        
        if ipiv != j:
            for k in range(1, Np1):
                t = A[ipiv][k]
                A[ipiv][k] = A[j][k]
                A[j][k] = t

        Ajjinv = 1.0/A[j][j]
        for i in range(j+1, Np1):
            A[i][j] *= Ajjinv
    return None
                

def solve_from_lu(A, pivots, b, N):
    Np1 = N + 1
        # Note- list call is very slow faster to replace with [i for i in row]
    b = [0.0] + [i for i in b] #list(b)
    for i in range(1, Np1):
        tot = b[pivots[i]]
        b[pivots[i]] = b[i]
        for j in range(1, i):
            tot -= A[i][j]*b[j]
        b[i] = tot
        
    for i in range(N, 0, -1):
        tot = b[i]
        for j in range(i+1, Np1):
            tot -= A[i][j]*b[j]
        b[i] = tot/A[i][i]
    return b

def solve_LU_decomposition(A, b):
    N = len(b)
    
    A_copy = [[0.0]*(N+1)]
    for row in A:
        # Note- list call is very slow faster to replace with [i for i in row]
        r = [0.0] + [i for i in row]
#        r = list(row)
#        r.insert(0, 0.0)
        A_copy.append(r)
    
    pivots = [0.0]*(N+1)
    inplace_LU(A_copy, pivots, N)
    return solve_from_lu(A_copy, pivots, b, N)[1:]


def inv_lu(a):
    N = len(a)
    Np1 = N + 1
    A_copy = [[0.0]*Np1]
    for row in a:
        # Note- list call is very slow faster to replace with [i for i in row]
        r = list(row)
        r.insert(0, 0.0)
        A_copy.append(r)
    a = A_copy
    
    ainv = [[0.0]*N for i in range(N)]
    pivots = [0]*Np1
    inplace_LU(a, pivots, N)
    
    for j in range(N):
        b = [0.0]*N
        b[j] = 1.0                          
        b = solve_from_lu(a, pivots, b, N)[1:]
        for i in range(N): 
            ainv[i][j] = b[i]
            
    return ainv


def solve(a, b):
    if len(a) > 4:
        if IS_PYPY or np is None:
            return solve_LU_decomposition(a, b)
        import numpy as np
        return np.linalg.solve(a, b).tolist()
    else:
        return dot(inv(a), b)
    


def norm2(arr):
    tot = 0.0
    for i in arr:
        tot += i*i
    return sqrt(tot)


def array_as_tridiagonals(arr):
    row_last = arr[0]
    a, b, c = [], [row_last[0]], []
    for i in range(1, len(row_last)):
        row = arr[i]
        b.append(row[i])
        c.append(row_last[i])
        a.append(row[i-1])
        row_last = row
    return a, b, c


def tridiagonals_as_array(a, b, c, zero=0.0):
    N = len(b)
    arr = [[zero]*N for _ in range(N)]
    row_last = arr[0]
    row_last[0] = b[0]
    for i in range(1, N):
        row = arr[i]
        row[i] = b[i] # set the middle row back
        row[i-1] = a[i-1]
        row_last[i] = c[i-1]
        row_last = row
    return arr


def solve_tridiagonal(a, b, c, d):
    b, d = [i for i in b], [i for i in d]
    N = len(d)
    for i in range(N - 1):
        m = a[i]/b[i]
        b[i+1] -= m*c[i]
        d[i+1] -= m*d[i]
    
    b[-1] = d[-1]/b[-1]
    for i in range(N-2, -1, -1):
        b[i] = (d[i] - c[i]*b[i+1])/b[i]
    return b

def subset_matrix(whole, subset):
    if type(subset) is slice:
        subset = range(subset.start, subset.stop, subset.step)
#    N = len(subset)
#    new = [[None]*N for i in range(N)]
#    for ni, i in enumerate(subset):
#        for nj,j in  enumerate(subset):
#            new[ni][nj] = whole[i][j]
    new = []
    for i in subset:
        whole_i = whole[i]
#        r = [whole_i[j] for j in subset]
#        new.append(r)
        new.append([whole_i[j] for j in subset])
#        r = []
#        for j in subset:
#            r.append(whole_i[j])
    return new
