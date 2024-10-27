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

import sys
from math import sqrt

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

__all__ = ['dot', 'inv', 'det', 'solve', 'norm2', 'inner_product', 'transpose',
           'eye', 'array_as_tridiagonals', 'solve_tridiagonal', 'subset_matrix',
           'argsort1d']
primitive_containers = frozenset([list, tuple])

def transpose(x):
    return [list(i) for i in zip(*x)]



def det(matrix):
    """Seems to work fine.

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

# The inverse function below is generated via the following script
'''
import sympy as sp
import re
from sympy import Matrix, Symbol, simplify, zeros, cse

def replace_power_with_multiplication(match):
    """Replace x**n with x*x*...*x n times"""
    var = match.group(1)
    power = int(match.group(2))
    if power <= 1:
        return var
    return '*'.join([var] * power)

def generate_symbolic_matrix(n):
    """Generate an nxn symbolic matrix with unique symbols"""
    syms = [[Symbol(f'm_{i}{j}') for j in range(n)] for i in range(n)]
    return Matrix(syms), syms

def analyze_matrix(n):
    """Generate symbolic expressions for determinant and inverse"""
    M, syms = generate_symbolic_matrix(n)
    det = M.det()
    inv = M.inv()
    return det, inv, syms

def post_process_code(code_str):
    """Apply optimizing transformations to the generated code"""
    # Replace x**n patterns with x*x*x... (n times)
    code_str = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\*\*(\d+)', replace_power_with_multiplication, code_str)
    # Replace **0.5 with sqrt()
    code_str = re.sub(r'\((.*?)\)\*\*0\.5', r'sqrt(\1)', code_str)
    return code_str

def generate_python_inv():
    """Generate a single unified matrix inversion function with optimized 1x1, 2x2, and 3x3 cases"""
    # Generate the specialized code for 2x2 and 3x3
    size_specific_code = {}
    for N in [2, 3, 4]:
        det, inv, _ = analyze_matrix(N)
        exprs = [det] + list(inv)
        replacements, reduced = cse(exprs, optimizations='basic')
        det_expr = reduced[0]
        inv_exprs = reduced[1:]
        
        # Build the size-specific code block
        code = []
        
        # Unpack matrix elements
        unpack_rows = []
        for i in range(N):
            row_vars = [f"m_{i}{j}" for j in range(N)]
            unpack_rows.append("(" + ", ".join(row_vars) + ")")
        code.append(f"        {', '.join(unpack_rows)} = matrix")
        
        # Common subexpressions
        code.append("\n        # Common subexpressions")
        for i, (temp, expr) in enumerate(replacements):
            code.append(f"        x{i} = {expr}")
        
        # Determinant check
        code.append("\n        # Calculate determinant and check if we need to use LU decomposition")
        code.append(f"        det = {det_expr}")
        code.append("        if abs(det) <= 1e-7:")
        code.append("            return inv_lu(matrix)")
        
        # Return matrix
        return_matrix = []
        for i in range(N):
            row = []
            for j in range(N):
                idx = i * N + j
                row.append(str(inv_exprs[idx]))
            return_matrix.append(f"            [{', '.join(row)}]")
        
        code.append("\n        return [")
        code.append(",\n".join(return_matrix))
        code.append("        ]")
        
        size_specific_code[N] = post_process_code("\n".join(code))
    
    # Generate the complete function
    complete_code = [
        "def inv(matrix):",
        "    size = len(matrix)",
        "    if size == 1:",
        "        return [[1.0/matrix[0][0]]]",
        "    elif size == 2:",
        size_specific_code[2],
        "    elif size == 3:",
        size_specific_code[3],
        "    elif size == 4:",
        size_specific_code[4],
        "    else:",
        "        return inv_lu(matrix)",
        ""
    ]
    
    return "\n".join(complete_code)

# Generate and print the complete function
print(generate_python_inv())
'''
def inv(matrix):
    size = len(matrix)
    if size == 1:
        return [[1.0/matrix[0][0]]]
    elif size == 2:
        (m_00, m_01), (m_10, m_11) = matrix

        # Common subexpressions
        x0 = m_00*m_11 - m_01*m_10

        # Calculate determinant and check if we need to use LU decomposition
        det = x0
        if abs(det) <= 1e-7:
            return inv_lu(matrix)

        x1 = 1.0/x0
        return [
            [m_11*x1, -m_01*x1],
            [-m_10*x1, m_00*x1]
        ]
    elif size == 3:
        (m_00, m_01, m_02), (m_10, m_11, m_12), (m_20, m_21, m_22) = matrix

        # Common subexpressions
        x0 = m_11*m_22
        x1 = m_01*m_12
        x2 = m_02*m_21
        x3 = m_12*m_21
        x4 = m_01*m_22
        x5 = m_02*m_11
        x6 = m_00*x0 - m_00*x3 + m_10*x2 - m_10*x4 + m_20*x1 - m_20*x5

        # Calculate determinant and check if we need to use LU decomposition
        det = x6
        if abs(det) <= 1e-7:
            return inv_lu(matrix)
        x7 = 1.0/x6

        return [
            [x7*(x0 - x3), -x7*(-x2 + x4), x7*(x1 - x5)],
            [-x7*(m_10*m_22 - m_12*m_20), x7*(m_00*m_22 - m_02*m_20), -x7*(m_00*m_12 - m_02*m_10)],
            [x7*(m_10*m_21 - m_11*m_20), -x7*(m_00*m_21 - m_01*m_20), x7*(m_00*m_11 - m_01*m_10)]
        ]
    else:
        return inv_lu(matrix)


def shape(value):
    '''Find and return the shape of an array, whether it is a numpy array or
    a list-of-lists or other combination of iterators.

    Parameters
    ----------
    value : various
        Input array, [-]

    Returns
    -------
    shape : tuple(int, dimension)
        Dimensions of array, [-]

    Notes
    -----
    It is assumed the shape is consistent - not something like [[1.1, 2.2], [2.4]]

    Examples
    --------
    >>> shape([])
    (0,)
    >>> shape([1.1, 2.2, 5.5])
    (3,)
    >>> shape([[1.1, 2.2, 5.5], [2.0, 1.1, 1.5]])
    (2, 3)
    >>> shape([[[1.1,], [2.0], [1.1]]])
    (1, 3, 1)
    >>> shape(['110-54-3'])
    (1,)
    '''
    try:
        return value.shape
    except:
        pass
    dims = [len(value)]
    try:
        # Except this block to handle the case of no value
        iter_value = value[0]
        for i in range(10):
            # try:
            if type(iter_value) in primitive_containers:
                dims.append(len(iter_value))
                iter_value = iter_value[0]
            else:
                break
            # except:
            #     break
    except:
        pass
    return tuple(dims)

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


def inplace_LU(A, ipivot):
    N = len(A)
    
    for j in range(N):
        for i in range(j):
            tot = A[i][j]
            for k in range(i):
                tot -= A[i][k] * A[k][j]
            A[i][j] = tot

        apiv = 0.0
        ipiv = j
        for i in range(j, N):
            tot = A[i][j]
            for k in range(j):
                tot -= A[i][k] * A[k][j]
            A[i][j] = tot

            if apiv < abs(A[i][j]):
                apiv = abs(A[i][j])
                ipiv = i
                
        if apiv == 0:
            raise ValueError("Singular matrix")
        ipivot[j] = ipiv

        if ipiv != j:
            for k in range(N):
                t = A[ipiv][k]
                A[ipiv][k] = A[j][k]
                A[j][k] = t

        Ajjinv = 1.0/A[j][j]
        for i in range(j + 1, N):
            A[i][j] *= Ajjinv


def solve_from_lu(A, pivots, b):
    N = len(b)
    b = b.copy()  # Create a copy to avoid modifying the input
    
    for i in range(N):
        tot = b[pivots[i]]
        b[pivots[i]] = b[i]
        for j in range(i):
            tot -= A[i][j] * b[j]
        b[i] = tot

    for i in range(N-1, -1, -1):
        tot = b[i]
        for j in range(i+1, N):
            tot -= A[i][j] * b[j]
        b[i] = tot/A[i][i]
    return b


def solve_LU_decomposition(A, b):
    N = len(b)
    A_copy = [row.copy() for row in A]  # Deep copy of A
    pivots = [0] * N
    inplace_LU(A_copy, pivots)
    return solve_from_lu(A_copy, pivots, b)


def inv_lu(a):
    N = len(a)
    A_copy = [row.copy() for row in a]  # Deep copy of a
    
    ainv = [[0.0] * N for i in range(N)]
    pivots = [0] * N
    inplace_LU(A_copy, pivots)

    for j in range(N):
        b = [0.0] * N
        b[j] = 1.0
        b = solve_from_lu(A_copy, pivots, b)
        for i in range(N):
            ainv[i][j] = b[i]

    return ainv

'''Script to generate solve function. Note that just like in inv the N = 4 case has too much numerical instability.
import sympy as sp
from sympy import Matrix, Symbol, simplify, solve_linear_system
import re

def generate_symbolic_system(n):
    """Generate an nxn symbolic matrix A and n-vector b"""
    A = Matrix([[Symbol(f'a_{i}{j}') for j in range(n)] for i in range(n)])
    b = Matrix([Symbol(f'b_{i}') for i in range(n)])
    return A, b

def generate_cramer_solution(n):
    """Generate symbolic solution using Cramer's rule for small matrices"""
    A, b = generate_symbolic_system(n)
    det_A = A.det()
    
    # Solve for each variable using Cramer's rule
    solutions = []
    for i in range(n):
        # Create matrix with i-th column replaced by b
        A_i = A.copy()
        A_i[:, i] = b
        det_i = A_i.det()
        # Store numerator only - we'll multiply by inv_det later
        solutions.append(det_i)
    
    return det_A, solutions

def generate_python_solve():
    """Generate a unified matrix solve function with optimized 1x1, 2x2, and 3x3 cases"""
    size_specific_code = {}
    
    # Special case for N=1
    size_specific_code[1] = """        # Direct solution for 1x1
        return [b[0]/matrix[0][0]]"""
    
    # Generate specialized code for sizes 2 and 3
    for N in [2, 3]:
        det, solutions = generate_cramer_solution(N)
        
        code = []
        
        # Unpack matrix elements
        unpack_rows = []
        for i in range(N):
            row_vars = [f"a_{i}{j}" for j in range(N)]
            unpack_rows.append("(" + ", ".join(row_vars) + ")")
        code.append(f"        {', '.join(unpack_rows)} = matrix")
        
        # Unpack b vector
        code.append(f"        {', '.join(f'b_{i}' for i in range(N))} = b")
        
        # Calculate determinant
        det_expr = str(det)
        code.append("\n        # Calculate determinant")
        code.append(f"        det = {det_expr}")
        
        # Check for singular matrix
        code.append("\n        # Check for singular matrix")
        code.append("        if abs(det) <= 1e-7:")
        code.append("            return solve_LU_decomposition(matrix, b)")
        
        # Calculate solution
        code.append("\n        # Calculate solution")
        code.append("        inv_det = 1.0/det")
        
        # Generate solution expressions (multiply by inv_det, don't divide by det)
        solution_lines = []
        for i, sol in enumerate(solutions):
            solution_lines.append(f"        x_{i} = ({sol}) * inv_det")
        code.append("\n".join(solution_lines))
        
        # Return solution
        code.append("\n        return [" + ", ".join(f"x_{i}" for i in range(N)) + "]")
        
        size_specific_code[N] = "\n".join(code)
    
    # Generate the complete function
    complete_code = [
        "def solve(matrix, b):",
        "    size = len(matrix)",
        "    if size == 1:",
        size_specific_code[1],
        "    elif size == 2:",
        size_specific_code[2],
        "    elif size == 3:",
        size_specific_code[3],
        "    else:",
        "        return solve_LU_decomposition(matrix, b)",
        ""
    ]
    
    return "\n".join(complete_code)

# Generate and print the optimized solve function
print(generate_python_solve())
'''

def solve(matrix, b):
    size = len(matrix)
    if size == 2:
        (a_00, a_01), (a_10, a_11) = matrix
        b_0, b_1 = b

        # Calculate determinant
        det = a_00*a_11 - a_01*a_10

        # Check for singular matrix
        if abs(det) <= 1e-7:
            return solve_LU_decomposition(matrix, b)

        # Calculate solution
        inv_det = 1.0/det
        x_0 = (a_11*b_0 - a_01*b_1) * inv_det
        x_1 = (-a_10*b_0 + a_00*b_1) * inv_det

        return [x_0, x_1]
    elif size == 3:
        (a_00, a_01, a_02), (a_10, a_11, a_12), (a_20, a_21, a_22) = matrix
        b_0, b_1, b_2 = b

        # Calculate determinant
        det = a_00*a_11*a_22 - a_00*a_12*a_21 - a_01*a_10*a_22 + a_01*a_12*a_20 + a_02*a_10*a_21 - a_02*a_11*a_20

        # Check for singular matrix
        if abs(det) <= 1e-7:
            return solve_LU_decomposition(matrix, b)

        # Calculate solution
        inv_det = 1.0/det
        x_0 = (b_0*(a_11*a_22 - a_12*a_21) + b_1*(-a_01*a_22 + a_02*a_21) + b_2*(a_01*a_12 - a_02*a_11)) * inv_det
        x_1 = (b_0*(-a_10*a_22 + a_12*a_20) + b_1*(a_00*a_22 - a_02*a_20) + b_2*(-a_00*a_12 + a_02*a_10)) * inv_det
        x_2 = (b_0*(a_10*a_21 - a_11*a_20) + b_1*(-a_00*a_21 + a_01*a_20) + b_2*(a_00*a_11 - a_01*a_10)) * inv_det

        return [x_0, x_1, x_2]
    else:
        return solve_LU_decomposition(matrix, b)



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
    '''
    Parameters
    ----------
    a : list[float]
        Lower diagonal, [-]
    b : list[float]
        Main diagonal along axis, [-]
    c : list[float]
        Upper diagonal, [-]
    d : list[float]
        Array being solved for, [-]

    Returns
    -------
    solve : list[float]
        result, [-]
    '''
    # the algorithm is in place
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



## argsort implementation

def argsort1d(arr):
    """
    Returns the indices that would sort a 1D list.

    Parameters
    ----------
    arr : list
        Input array [-]

    Returns
    -------
    indices : list[int]
        List of indices that sort the input array [-]

    Notes
    -----
    This function uses the built-in sorted function with a custom key to get the indices.
    Note this does not match numpy's sorting for nan and inf values.

    Examples
    --------
    >>> arr = [3, 1, 2]
    >>> argsort1d(arr)
    [1, 2, 0]
    """
    return [i[0] for i in sorted(enumerate(arr), key=lambda x: x[1])]
