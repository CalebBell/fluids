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

__all__ = ['dot_product', 'inv', 'det', 'solve', 'norm2', 'transpose', 'shape',
           'eye', 'array_as_tridiagonals', 'solve_tridiagonal', 'subset_matrix',
           'argsort1d', 'lu', 'gelsd', 'matrix_vector_dot', 'matrix_multiply',
           'sum_matrix_rows', 'sum_matrix_cols', 'sort_paired_lists',
           'scalar_divide_matrix', 'scalar_multiply_matrix', 'scalar_subtract_matrices', 'scalar_add_matrices',
           'stack_vectors', 'null_space']
primitive_containers = frozenset([list, tuple])

def transpose(matrix):
    """Convert a matrix into its transpose by switching rows and columns.

    Parameters
    ----------
    matrix : list[list[float]]
        Input matrix as a list of lists where each inner list represents a row.
        All rows must have the same length.

    Returns
    -------
    list[list[float]]
        The transposed matrix where element [i][j] in the input becomes [j][i]
        in the output.

    Raises
    ------
    ValueError
        If the input matrix has inconsistent row lengths.
    TypeError
        If the input is not a list of lists.

    Examples
    --------
    >>> transpose([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]

    >>> transpose([[1, 2], [3, 4]])  # Square matrix
    [[1, 3], [2, 4]]

    >>> transpose([[1, 2, 3]])  # Single row matrix
    [[1], [2], [3]]

    Notes
    -----
    - Empty matrices are preserved as empty lists
    - The function creates a new matrix rather than modifying in place
    - For an MxN matrix, the result will be an NxM matrix
    """
    # Handle empty matrix cases
    if not matrix:
        return []
    if not matrix[0]:
        return []
    
    # # Validate input
    # if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
    #     raise TypeError("Input must be a list of lists")
    
    # Check for consistent row lengths
    row_length = len(matrix[0])
    if not all(len(row) == row_length for row in matrix):
        raise ValueError("All rows must have the same length")
    
    return [list(i) for i in zip(*matrix)]

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

def eye(N, dtype=float):
    """
    Return a 2-D array with ones on the diagonal and zeros elsewhere.
    
    Parameters
    ----------
    N : int
        Number of rows and columns in the output matrix.
    dtype : type, optional
        The type of the array elements. Defaults to float.
        
    Returns
    -------
    list[list]
        A N x N matrix with ones on the diagonal and zeros elsewhere.
        
    Examples
    --------
    >>> eye(3)
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    
    >>> eye(2, dtype=int)
    [[1, 0], [0, 1]]
    
    Notes
    -----
    This function creates an identity matrix similar to numpy's eye function,
    but implemented in pure Python using nested lists.
    
    Raises
    ------
    ValueError
        If N is not a positive integer.
    TypeError
        If N is not an integer or dtype is not a valid type.
    """
    # Input validation
    if not isinstance(N, int):
        raise TypeError("N must be an integer")
    if N <= 0:
        raise ValueError("N must be a positive integer")
    
    # Create the matrix
    matrix = []
    zero, one = dtype(0), dtype(1)
    for i in range(N):
        row = [zero] * N  # Initialize row with zeros
        row[i] = one    # Set diagonal element to 1
        matrix.append(row)
    
    return matrix


def dot_product(a, b):
    """
    Compute the dot product (also known as scalar product or inner product) of two vectors.
    
    Calculates sum(a[i] * b[i]) for i in range(len(a)).
    
    Parameters
    ----------
    a : list[float]
        First vector
    b : list[float]
        Second vector of same length as a
        
    Returns
    -------
    float
        The dot product of vectors a and b
        
    Examples
    --------
    >>> dot_product([1, 2, 3], [4, 5, 6])
    32.0
    >>> dot_product([1, 0], [0, 1])
    0.0
    
    Notes
    -----
    
    Raises
    ------
    ValueError
        If vectors are not the same length
    TypeError 
        If inputs are not valid vector types
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have same length") 
    tot = 0.0
    for i in range(len(a)):
        tot += a[i]*b[i]
    return tot

def matrix_vector_dot(matrix, vector):
    """
    Compute the product of a matrix and a vector.

    Parameters
    ----------
    matrix : list[list[float]]
        Input matrix represented as a list of lists.
    vector : list[float]
        Input vector represented as a list of floats.

    Returns
    -------
    list[float]
        The result of the matrix-vector multiplication as a vector.

    Raises
    ------
    ValueError
        If the number of columns in the matrix does not match the length of the vector.
    TypeError
        If inputs are not valid matrix and vector types.

    Examples
    --------
    >>> matrix_vector_dot([[1, 2, 3], [4, 5, 6]], [1, 0, 1])
    [4, 10]
    >>> matrix_vector_dot([[1.0, 2.0], [3.0, 4.0]], [0, 1])
    [2.0, 4.0]
    """
    # Validate matrix dimensions
    N = len(vector)
    if not all(len(row) == N for row in matrix):
        raise ValueError("Matrix columns must match vector length")

    result = [sum(row[i] * vector[i] for i in range(N)) for row in matrix]
    return result

def matrix_multiply(A, B):
    r"""Multiply two matrices using pure Python.
    
    Computes the matrix product C = A·B where A is an m×p matrix and B is a p×n matrix,
    resulting in an m×n matrix C.
    
    Parameters
    ----------
    A : list[list[float]]
        First matrix as list of lists, with shape (m, p)
    B : list[list[float]]
        Second matrix as list of lists, with shape (p, n)
        
    Returns
    -------
    list[list[float]]
        Resulting matrix C with shape (m, n)
        
    Examples
    --------
    >>> A = [[1, 2], [3, 4]]
    >>> B = [[5, 6], [7, 8]]
    >>> matrix_multiply(A, B)
    [[19.0, 22.0], [43.0, 50.0]]
    
    Notes
    -----
    Uses a straightforward three-loop implementation optimized for pure Python:
    C[i,j] = sum(A[i,k] * B[k,j] for k in range(p))
    
    The implementation avoids repeated len() calls and list accesses by caching
    frequently used values.
    
    Raises
    ------
    ValueError
        If matrices have incompatible dimensions for multiplication
        If input matrices are empty or irregular (rows of different lengths)
    TypeError
        If A or B contains non-numeric values or is not a list of lists.
    """
    # Input validation
    if not A or not A[0] or not B or not B[0]:
        raise ValueError("Empty matrices cannot be multiplied")
    
    # Get dimensions
    m = len(A)  # rows in A
    p = len(A[0]) if m else 0 # cols in A = rows in B
    n = len(B[0]) if B else 0  # cols in B
    
    # Validate dimensions
    if not all(len(row) == p for row in A):
        raise ValueError("First matrix has irregular row lengths")
    if len(B) != p:
        raise ValueError(f"Incompatible dimensions: A is {m}x{p}, B is {len(B)}x{n}")
    if not all(len(row) == n for row in B):
        raise ValueError("Second matrix has irregular row lengths")
    
    # Pre-allocate result matrix with zeros
    C = [[0.0] * n for _ in range(m)]
    
    # Compute product using simple indexed loops
    for i in range(m):
        A_i = A[i]  # Cache current row of A
        C_i = C[i]  # Cache current row of C
        for j in range(n):
            tot = 0.0
            for k in range(p):
                tot += A_i[k] * B[k][j]
            C_i[j] = tot
            
    return C

def sum_matrix_rows(matrix):
    """Sum a 2D matrix along rows, equivalent to numpy.sum(matrix, axis=1).
    
    Parameters
    ----------
    matrix : list[list[float]]
        Input matrix as a list of lists where each inner list is a row
        
    Returns
    -------
    list[float]
        List containing the sum of each row
        
    Examples
    --------
    >>> sum_matrix_rows([[1, 2, 3], [4, 5, 6]])
    [6.0, 15.0]
    >>> sum_matrix_rows([[1], [2]])
    [1.0, 2.0]
    
    Notes
    -----
    For a matrix with shape (m, n), returns a list of length m
    where each element is the sum of the corresponding row.
    
    Raises
    ------
    ValueError
        If matrix is empty or has irregular row lengths
    TypeError
        If matrix is not a list of lists of numbers
    """
    if not matrix or not matrix[0]:
        raise ValueError("Empty matrix")
        
    n = len(matrix[0])
    if not all(len(row) == n for row in matrix):
        raise ValueError("Matrix has irregular row lengths")
    
    result = []
    for row in matrix:
        tot = 0.0
        for val in row:
            tot += val
        result.append(tot)
    return result

def sum_matrix_cols(matrix):
    """Sum a 2D matrix along columns, equivalent to numpy.sum(matrix, axis=0).
    
    Parameters
    ----------
    matrix : list[list[float]]
        Input matrix as a list of lists where each inner list is a row
        
    Returns
    -------
    list[float]
        List containing the sum of each column
        
    Examples
    --------
    >>> sum_matrix_cols([[1, 2, 3], [4, 5, 6]])
    [5.0, 7.0, 9.0]
    >>> sum_matrix_cols([[1], [2]])
    [3.0]
    
    Notes
    -----
    For a matrix with shape (m, n), returns a list of length n
    where each element is the sum of the corresponding column.
    
    Raises
    ------
    ValueError
        If matrix is empty or has irregular row lengths
    TypeError
        If matrix is not a list of lists of numbers
    """
    if not matrix or not matrix[0]:
        raise ValueError("Empty matrix")
        
    n = len(matrix[0])
    if not all(len(row) == n for row in matrix):
        raise ValueError("Matrix has irregular row lengths")
    
    result = [0.0] * n
    for row in matrix:
        for j, val in enumerate(row):
            result[j] += val
    return result

def scalar_add_matrices(A, B):
    """Add two matrices element-wise.
    
    Computes the element-wise sum of two matrices of the same dimensions.
    
    Parameters
    ----------
    A : list[list[float]]
        First matrix as a list of lists.
    B : list[list[float]]
        Second matrix as a list of lists.
        
    Returns
    -------
    list[list[float]]
        Resulting matrix after element-wise addition.
        
    Examples
    --------
    >>> A = [[1.0, 2.0], [3.0, 4.0]]
    >>> B = [[5.0, 6.0], [7.0, 8.0]]
    >>> scalar_add_matrices(A, B)
    [[6.0, 8.0], [10.0, 12.0]]
    
    Raises
    ------
    ValueError
        If matrices A and B have different shapes or if they are empty.
    TypeError
        If A or B contains non-numeric values or is not a list of lists.
    """
    if not A or not B or len(A) != len(B) or len(A[0]) != len(B[0]) or not len(A[0]):
        raise ValueError("Matrices must have the same dimensions and be non-empty")
    
    result = []
    for row_A, row_B in zip(A, B):
        if len(row_A) != len(row_B):
            raise ValueError("Matrices must have the same dimensions")
        result.append([a + b for a, b in zip(row_A, row_B)])
    return result


def scalar_subtract_matrices(A, B):
    """Subtract two matrices element-wise.
    
    Computes the element-wise difference of two matrices of the same dimensions.
    
    Parameters
    ----------
    A : list[list[float]]
        First matrix as a list of lists.
    B : list[list[float]]
        Second matrix as a list of lists.
        
    Returns
    -------
    list[list[float]]
        Resulting matrix after element-wise subtraction.
        
    Examples
    --------
    >>> A = [[5.0, 6.0], [7.0, 8.0]]
    >>> B = [[1.0, 2.0], [3.0, 4.0]]
    >>> scalar_subtract_matrices(A, B)
    [[4.0, 4.0], [4.0, 4.0]]
    
    Raises
    ------
    ValueError
        If matrices A and B have different shapes or if they are empty.
    TypeError
        If A or B contains non-numeric values or is not a list of lists.
    """
    if not A or not B or len(A) != len(B) or len(A[0]) != len(B[0]) or not len(A[0]):
        raise ValueError("Matrices must have the same dimensions and be non-empty")
    
    result = []
    for row_A, row_B in zip(A, B):
        if len(row_A) != len(row_B):
            raise ValueError("Matrices must have the same dimensions")
        result.append([a - b for a, b in zip(row_A, row_B)])
    return result


def scalar_multiply_matrix(scalar, matrix):
    """Multiply a matrix by a scalar.
    
    Multiplies each element of the matrix by the specified scalar.
    
    Parameters
    ----------
    scalar : float
        Scalar value to multiply each element by.
    matrix : list[list[float]]
        Input matrix as a list of lists.
        
    Returns
    -------
    list[list[float]]
        Resulting matrix after scalar multiplication.
        
    Examples
    --------
    >>> matrix = [[1, 2], [3, 4]]
    >>> scalar_multiply_matrix(2.0, matrix)
    [[2.0, 4.0], [6.0, 8.0]]
    
    Raises
    ------
    ValueError
        If the input matrix is empty.
    TypeError
        If the matrix contains non-numeric values or is not a list of lists.
    """
    if not matrix or not matrix[0]:
        raise ValueError("Input matrix cannot be empty")
    
    result = []
    for row in matrix:
        result.append([scalar * val for val in row])
    return result


def scalar_divide_matrix(scalar, matrix):
    """Divide a matrix by a scalar.
    
    Divides each element of the matrix by the specified scalar.
    
    Parameters
    ----------
    scalar : float
        Scalar value to divide each element by (cannot be zero).
    matrix : list[list[float]]
        Input matrix as a list of lists.
        
    Returns
    -------
    list[list[float]]
        Resulting matrix after scalar division.
        
    Examples
    --------
    >>> matrix = [[2, 4], [6, 8]]
    >>> scalar_divide_matrix(2.0, matrix)
    [[1.0, 2.0], [3.0, 4.0]]
    
    Raises
    ------
    ValueError
        If the input matrix is empty or if the scalar is zero.
    TypeError
        If the matrix contains non-numeric values or is not a list of lists.
    ZeroDivisionError
        If scalar is zero.
    """
    if scalar == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    if not matrix or not matrix[0]:
        raise ValueError("Input matrix cannot be empty")
    
    result = []
    for row in matrix:
        result.append([val / scalar for val in row])
    return result

def stack_vectors(vectors):
    """Stack a list of vectors into a matrix, similar to numpy.stack.
    
    Parameters
    ----------
    vectors : list[list[float]]
        List of vectors to stack into rows of a matrix
        
    Returns
    -------
    list[list[float]]
        Matrix where each row is one of the input vectors
        
    Examples
    --------
    >>> stack_vectors([[1, 2], [3, 4]])
    [[1, 2], [3, 4]]
    """
    if not vectors:
        return []
    return [list(v) for v in vectors]  # Create copies of vectors
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

def lu(A):
    """
    Compute LU decomposition of a matrix with partial pivoting.
    Returns P, L, U such that PA = LU
    
    Parameters:
        A: list of lists representing square matrix
        
    Returns:
        P: permutation matrix as list of lists
        L: lower triangular matrix with unit diagonal as list of lists
        U: upper triangular matrix as list of lists
    """
    N = len(A)
    
    # Create working copy and pivots array
    A_copy = [row.copy() for row in A]
    pivots = [0] * N
    
    # Perform LU decomposition
    inplace_LU(A_copy, pivots)
    
    # Extract L (unit diagonal and below diagonal elements)
    L = [[1.0 if i == j else 0.0 for j in range(N)] for i in range(N)]
    for i in range(N):
        for j in range(i):
            L[i][j] = A_copy[i][j]
    
    # Extract U (upper triangular including diagonal)
    U = [[0.0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i, N):
            U[i][j] = A_copy[i][j]
    
    # Create permutation matrix directly from pivot sequence
    P = [[1.0 if j == i else 0.0 for j in range(N)] for i in range(N)]
    for i, pivot in enumerate(pivots):
        if pivot != i:
            P[i], P[pivot] = P[pivot], P[i]
            
    return P, L, U


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
    """Extract the three diagonals from a tridiagonal matrix.
    
    A tridiagonal matrix is a matrix that has nonzero elements only on the 
    main diagonal, the first diagonal below this (subdiagonal), and the first 
    diagonal above this (superdiagonal).
    
    Parameters
    ----------
    arr : list[list[float]]
        Square matrix in tridiagonal form, where elements not on the three
        main diagonals are zero
    
    Returns
    -------
    tuple[list[float], list[float], list[float]]
        Three lists containing:
        a: subdiagonal elements (length n-1)
        b: main diagonal elements (length n)
        c: superdiagonal elements (length n-1)
        
    Examples
    --------
    >>> arr = [[2, 1, 0], [1, 2, 1], [0, 1, 2]]
    >>> a, b, c = array_as_tridiagonals(arr)
    >>> a  # subdiagonal
    [1, 1]
    >>> b  # main diagonal
    [2, 2, 2]
    >>> c  # superdiagonal
    [1, 1]
    
    Notes
    -----
    For a matrix of size n×n, returns:
    - a[i] contains elements at position (i+1,i) for i=0..n-2
    - b[i] contains elements at position (i,i) for i=0..n-1
    - c[i] contains elements at position (i,i+1) for i=0..n-2
    
    No validation is performed to ensure the input matrix is actually tridiagonal.
    Elements outside the three diagonals are ignored.
    """
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
    r"""Construct a square matrix from three diagonals.
    
    Creates a tridiagonal matrix using the provided sub-, main, and super-diagonal 
    elements. All other elements are set to zero.
    
    Parameters
    ----------
    a : list[float]
        Subdiagonal elements (length n-1)
    b : list[float]
        Main diagonal elements (length n)
    c : list[float]
        Superdiagonal elements (length n-1)
    zero : float, optional
        Value to use for non-diagonal elements. Defaults to 0.0
    
    Returns
    -------
    list[list[float]]
        Square matrix of size n×n where n is the length of b
        
    Examples
    --------
    >>> a = [1, 1]  # subdiagonal
    >>> b = [2, 2, 2]  # main diagonal
    >>> c = [1, 1]  # superdiagonal
    >>> tridiagonals_as_array(a, b, c)
    [[2, 1, 0.0], [1, 2, 1], [0.0, 1, 2]]
    
    Notes
    -----
    For output matrix M of size n×n:
    - a[i] becomes M[i+1][i] for i=0..n-2
    - b[i] becomes M[i][i] for i=0..n-1
    - c[i] becomes M[i][i+1] for i=0..n-2
    
    No validation is performed on input lengths. For correct results:
    - len(b) should be n
    - len(a) and len(c) should be n-1
    
    The function is the inverse of array_as_tridiagonals() when zero=0.0
    """
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
    """Solve a tridiagonal system of equations using the Thomas algorithm.
    
    Solves the equation system Ax = d where A is a tridiagonal matrix composed of
    diagonals a, b, and c. This is an efficient O(n) method also known as the
    tridiagonal matrix algorithm (TDMA).
    
    The system of equations has the form:
    b[0]x[0] + c[0]x[1] = d[0]
    a[i]x[i-1] + b[i]x[i] + c[i]x[i+1] = d[i], for i=1..n-2
    a[n-1]x[n-2] + b[n-1]x[n-1] = d[n-1]
    
    Parameters
    ----------
    a : list[float]
        Lower diagonal (subdiagonal) elements a[i] at (i+1,i), length n-1, [-]
    b : list[float]
        Main diagonal elements b[i] at (i,i), length n, [-]
    c : list[float]
        Upper diagonal (superdiagonal) elements c[i] at (i,i+1), length n-1, [-]
    d : list[float]
        Right-hand side vector, length n, [-]
        
    Returns
    -------
    x : list[float]
        Solution vector, length n, [-]
        
    Examples
    --------
    >>> # Solve the system:
    >>> # [9 -1  0] [x0]   [1]
    >>> # [-1 2 -1] [x1] = [0]
    >>> # [0 -1  2] [x2]   [1]
    >>> a = [-1, -1]  # lower diagonal
    >>> b = [9, 2, 2]  # main diagonal
    >>> c = [-1, -1]  # upper diagonal
    >>> d = [1, 0, 1]  # right hand side
    >>> solve_tridiagonal(a, b, c, d)
    [0.16, 0.44, 0.72]
    
    Notes
    -----
    The algorithm modifies the input arrays b and d in-place to save memory,
    but makes copies first to preserve the originals.
    
    
    The algorithm fails if any diagonal element becomes zero during elimination.
    
    This implementation uses the Thomas algorithm, which is a specialized form
    of Gaussian elimination that exploits the tridiagonal structure for O(n)
    efficiency.
    
    No validation is performed on input lengths. For correct results:
    - len(b) should be n
    - len(a), len(c) should be n-1
    - len(d) should be n
    where n is the size of the system.
    
    References
    ----------
    .. [1] "Tridiagonal matrix algorithm", Wikipedia,
           https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    """
    # Make copies since the algorithm modifies arrays in-place
    b, d = [i for i in b], [i for i in d]
    N = len(d)
    
    # Forward elimination phase
    for i in range(N - 1):
        m = a[i]/b[i]
        b[i+1] -= m*c[i]
        d[i+1] -= m*d[i]
    
    # Back substitution phase
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

def sort_paired_lists(list1, list2):
    """
    Sort two lists based on the values in the first list while maintaining 
    the relationship between corresponding elements.
    
    Parameters
    ----------
    list1 : list
        First list that determines the sorting order
    list2 : list
        Second list that will be sorted according to list1's ordering
        
    Returns
    -------
    tuple
        A tuple containing (sorted_list1, sorted_list2)
        
    Raises
    ------
    ValueError
        If the lists have different lengths
    TypeError
        If either input is not a list
        
    Examples
    --------
    >>> temps = [300, 100, 200]
    >>> props = ['hot', 'cold', 'warm']
    >>> sort_paired_lists(temps, props)
    ([100, 200, 300], ['cold', 'warm', 'hot'])
    
    Notes
    -----
    This function maintains the one-to-one relationship between elements
    in both lists while sorting them based on list1's values.
    """
    # Input validation
    if len(list1) != len(list2):
        raise ValueError("Lists must have equal length")
        
    # Handle empty lists
    if len(list1) == 0:
        return ([], [])
        
    # Get sorting indices using argsort1d
    sorted_indices = argsort1d(list1)
    
    # Apply the sorting to both lists
    sorted_list1 = [list1[i] for i in sorted_indices]
    sorted_list2 = [list2[i] for i in sorted_indices]
    
    return sorted_list1, sorted_list2

def svd(matrix):
    """Compute the singular value decomposition of a matrix.
    
    This function wraps numpy.linalg.svd but maintains pure Python input/output
    interfaces.
    
    Parameters
    ----------
    matrix : list[list[float]]
        Input matrix A to decompose
        
    Returns
    -------
    tuple[list[list[float]], list[float], list[list[float]]]
        Returns (U, s, Vt) where:
        - U is the left singular vectors as a matrix
        - s is the singular values as a 1D array
        - Vt is the transpose of the right singular vectors as a matrix
        
    Notes
    -----

    Examples
    --------
    >>> A = [[1, 2], [3, 4]]
    >>> U, s, Vt = svd(A)
    """
    import numpy as np
    # Compute SVD
    U, s, Vt = np.linalg.svd(np.array(matrix, dtype=np.float64), full_matrices=True)
    # Convert back to Python lists
    return U.tolist(), s.tolist(), Vt.tolist()


def gelsd(a, b, rcond=None):
    """Solve a linear least-squares problem using SVD (Singular Value Decomposition).
    This is a simplified implementation that uses numpy's SVD internally.
    
    The function solves the equation arg min(|b - Ax|) for x, where A is
    an M x N matrix and b is a length M vector.
    
    Parameters
    ----------
    a : list[list[float]]
        Input matrix A of shape (M, N)
    b : list[float]
        Input vector b of length M
    rcond : float, optional
        Cutoff ratio for small singular values. Singular values smaller
        than rcond * largest_singular_value are considered zero.
        Default: max(M,N) * eps where eps is the machine precision
    
    Returns
    -------
    x : list[float]
        Solution vector of length N
    residuals : float
        Sum of squared residuals of the solution. Only computed for overdetermined 
        systems (M > N)
    rank : int
        Effective rank of matrix A
    s : list[float]
        Singular values of A in descending order
    
    Notes
    -----
    The implementation uses numpy.linalg.svd for the core computation but
    maintains a pure Python interface for input and output.
    """
    # Get dimensions and handle empty cases
    m = len(a)
    n = len(a[0]) if m > 0 else 0
    
    if m == 0:
        if n == 0:
            return [], 0.0, 0, []  # Empty matrix
        return [0.0] * n, 0.0, 0, []  # Empty rows
    elif n == 0:
        return [], 0.0, 0, []  # Empty columns
    
    # Check compatibility
    if len(b) != m:
        raise ValueError(f"Incompatible dimensions: A is {m}x{n}, b has length {len(b)}")
    
    U, s, Vt = svd(a)

    # Set default rcond
    if rcond is None:
        rcond = max(m, n) * 2.2e-16  # Approximate machine epsilon for float64
    
    # Determine rank using rcond
    tol = rcond * s[0]
    rank = sum(sv > tol for sv in s)
    
    # Handle zero matrix case (all singular values below threshold)
    if rank == 0:
        return [0.0] * n, sum(bi * bi for bi in b), 0, s
    
    # We only need the first rank columns of U and V
    # If U is economy sized (M×min(M,N)), this is fine
    # If U is full sized (M×M), we still only use first rank columns
    Ut = transpose(U)
    Utb = matrix_vector_dot(Ut[:rank], b)
    
    # Apply 1/singular values with truncation
    s_inv_Utb = [Utb[i] / s[i] for i in range(rank)]
    
    # Get the first rank rows of V (transpose of first rank columns of Vt)
    # Again, works with both economy and full-size Vt
    V = transpose(Vt[:rank])
    x = matrix_vector_dot(V, s_inv_Utb)
    
    # Compute residuals for overdetermined systems
    residuals = 0.0
    if m > n and rank == n:
        # Compute Ax
        Ax = matrix_vector_dot(a, x)
        
        # Compute residuals as |b - Ax|^2
        diff = [b[i] - Ax[i] for i in range(m)]
        residuals = dot_product(diff, diff)
    return x, residuals, rank, s

def null_space(a, rcond=None):
    """
    Construct an orthonormal basis for the null space of A using SVD.

    Parameters
    ----------
    a : list[list[float]]
        Input matrix A of shape (M, N)
    rcond : float, optional
        Relative condition number. Singular values ``s`` smaller than
        ``rcond * max(s)`` are considered zero.
        Default: floating point eps * max(M,N).

    Returns
    -------
    Z : list[list[float]]
        Orthonormal basis for the null space of A.
        K = dimension of effective null space, as determined by rcond
    """
    # Get dimensions and handle empty cases
    m = len(a)
    n = len(a[0]) if m > 0 else 0
    
    if m == 0 or n == 0:
        return []  # Empty matrix
    U, s, Vt = svd(a)
    # Set default rcond
    if rcond is None:
        rcond = max(m, n) * 2.2e-16  # Approximate machine epsilon for float64
    
    # Determine effective null space dimension using rcond
    tol = max(s) * rcond if s else 0.0
    num = sum(sv > tol for sv in s)
    # Extract null space basis
    V = transpose(Vt)  # V is transpose of Vt
    Z = [row[num:] for row in V]  # Extract last N - num columns
    
    return Z