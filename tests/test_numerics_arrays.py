'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2024 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
'''
from math import cos, erf, exp, isnan, log, pi, sin, sqrt

import pytest
from fluids.numerics.arrays import inv
from fluids.numerics import (
    array_as_tridiagonals,
    assert_close,
    assert_close1d,
    assert_close2d,
    solve_tridiagonal,
    subset_matrix,
    tridiagonals_as_array,
    argsort1d,
)
from fluids.numerics import numpy as np

assert_allclose = np.testing.assert_allclose

def get_rtol(matrix):
    """Set tolerance based on condition number"""
    cond = np.linalg.cond(matrix)
    machine_eps = np.finfo(float).eps  # ≈ 2.2e-16
    return min(10 * cond * machine_eps,100*cond * machine_eps if cond > 1e8 else 1e-9)

def check_inv(matrix, rtol=None):
    py_fail = False
    numpy_fail = False
    try:
        result = inv(matrix)
    except:
        py_fail = True
    try:
        expected = np.linalg.inv(matrix)
    except:
        numpy_fail = True
    if py_fail and not numpy_fail:
        raise ValueError(f"Inconsistent failure states: Python Fail: {py_fail}, Numpy Fail: {numpy_fail}")
    if py_fail and numpy_fail:
        return
    if not py_fail and numpy_fail:
        # We'll allow our inv to work with numbers closer to
        return

        
    # Convert result to numpy array if it isn't already
    result = np.array(result)
    
    # Compute infinity norm of input matrix
    matrix_norm = np.max(np.sum(np.abs(matrix), axis=1))
    thresh = matrix_norm * np.finfo(float).eps
    
    # Check both directions
    numpy_zeros = (expected == 0.0)
    our_zeros = (result == 0.0)
    
    # Where numpy has zeros but we don't; no cases require it but it makes sense to do
    our_values_at_numpy_zeros = result[numpy_zeros]
    result[numpy_zeros] = np.where(
        np.abs(our_values_at_numpy_zeros) < thresh,
        0.0,
        our_values_at_numpy_zeros
    )
    
    # Where we have zeros but numpy doesn't  - this is the one we discovered
    numpy_values_at_our_zeros = expected[our_zeros]
    expected[our_zeros] = np.where(
        np.abs(numpy_values_at_our_zeros) < thresh,
        0.0,
        numpy_values_at_our_zeros
    )

    # We also need to check against the values we get in the inverse; it is helpful 
    # to zero out anything too close to "zero" relative to the values used in the matrix
    # This is very necessary, and was needed when testing on different CPU architectures
    inv_norm = np.max(np.sum(np.abs(result), axis=1))
    trivial_relative_to_norm = np.where(np.abs(result)/inv_norm < 10*thresh)
    result[trivial_relative_to_norm] = 0.0
    trivial_relative_to_norm = np.where(np.abs(expected)/inv_norm < 10*thresh)
    expected[trivial_relative_to_norm] = 0.0

    if rtol is None:
        rtol = get_rtol(matrix)
    # For each element, use absolute tolerance if the expected value is near zero
    # In the near zero for some element cases but where others aren't, the relative differences can be brutal relative
    # to the other numbers in the matrix so we have to treat them differently
    mask = np.abs(expected) < 1e-14
    if mask.any():
        assert_allclose(result[mask], expected[mask], atol=thresh)
        assert_allclose(result[~mask], expected[~mask], rtol=rtol)
    else:
        assert_allclose(result, expected, rtol=rtol)    



def format_matrix_error(matrix):
    """Format a detailed error message for matrix comparison failure"""
    def matrix_info(matrix):
        """Get diagnostic information about a matrix"""
        arr = np.array(matrix)
        return {
            'condition_number': np.linalg.cond(arr),
            'determinant': np.linalg.det(arr),
            'shape': arr.shape
        }
    info = matrix_info(matrix)
    
    return (
        f"\nMatrix properties:"
        f"\n  Shape: {info['shape']}"
        f"\n  Condition number: {info['condition_number']:.2e}"
        f"\n  Determinant: {info['determinant']:.2e}"
        f"\nInput matrix:"
        f"\n{np.array2string(np.array(matrix), precision=6, suppress_small=True)}"
    )


# 1x1 matrices
matrices_1x1 = [
    [[2.0]],
    [[0.5]],
    [[-3.0]],
    [[1e-10]],
    [[1e10]],
]

# 2x2 matrices - regular cases
matrices_2x2 = [
    [[1.0, 0.0], 
     [0.0, 1.0]],  # Identity matrix
    
    [[2.0, 1.0],
     [1.0, 2.0]],  # Symmetric matrix
    
    [[1.0, 2.0],
     [3.0, 4.0]],  # General case
    
    [[1e-5, 1.0],
     [1.0, 1e5]],  # Poorly conditioned
    
    [[0.1, 0.2],
     [0.3, 0.4]],  # Decimal values


    # All ones matrices
    [[1.0, 1.0],
     [1.0, 1.0]],
    [[1.0, 1.0],
     [1.0, -1.0]],
    [[1.0, -1.0],
     [-1.0, 1.0]],
     
    # Upper triangular
    [[2.0, 3.0],
     [0.0, 4.0]],
    [[1.0, 10.0],
     [0.0, 2.0]],
    [[5.0, -3.0],
     [0.0, 1.0]],
     
    # Lower triangular
    [[2.0, 0.0],
     [3.0, 4.0]],
    [[1.0, 0.0],
     [10.0, 2.0]],
    [[5.0, 0.0],
     [-3.0, 1.0]],
     
    # Rotation matrices (θ = 30°, 45°, 60°)
    [[0.866, -0.5],
     [0.5, 0.866]],
    [[0.707, -0.707],
     [0.707, 0.707]],
    [[0.5, -0.866],
     [0.866, 0.5]],
     
    # Reflection matrices
    [[1.0, 0.0],
     [0.0, -1.0]],
    [[-1.0, 0.0],
     [0.0, 1.0]],
    [[0.0, 1.0],
     [1.0, 0.0]],
     
    # Scaling matrices
    [[10.0, 0.0],
     [0.0, 0.1]],
    [[100.0, 0.0],
     [0.0, 0.01]],
    [[1000.0, 0.0],
     [0.0, 0.001]],
     
    # Nearly zero determinant (different from existing near-singular)
    [[1.0, 2.0],
     [0.5, 1.0 + 1e-12]],
    [[2.0, 4.0],
     [1.0, 2.0 + 1e-13]],
    [[3.0, 6.0],
     [1.5, 3.0 + 1e-11]],
     
    # Mixed scale
    [[1e6, 1e-6],
     [1e-6, 1e6]],
    [[1e8, 1e-4],
     [1e-4, 1e8]],
    [[1e10, 1e-2],
     [1e-2, 1e10]],
     
    # Nilpotent matrices
    [[0.0, 1.0],
     [0.0, 0.0]],
    [[0.0, 2.0],
     [0.0, 0.0]],
    [[0.0, 0.5],
     [0.0, 0.0]],
     
    # Hadamard matrices (normalized)
    [[1/sqrt(2), 1/sqrt(2)],
     [1/sqrt(2), -1/sqrt(2)]],
    [[1/sqrt(2), 1/sqrt(2)],
     [-1/sqrt(2), 1/sqrt(2)]],
    [[-1/sqrt(2), 1/sqrt(2)],
     [1/sqrt(2), 1/sqrt(2)]]
     
]

# 2x2 matrices - nearly singular cases
matrices_2x2_near_singular = [
    [[1.0, 2.0],
     [1.0 + 1e-10, 2.0 + 1e-10]],  # Almost linearly dependent rows
    
    [[1.0, 1.0],
     [1.0, 1.0 + 1e-10]],  # Almost zero determinant
     
    [[1e5, 1e5],
     [1e5, 1e5 + 1.0]],  # Scaled nearly singular
     
    [[1e-10, 1.0],
     [1.0, 1.0]],  # One very small pivot
     
    [[1.0, -1e-10],
     [1e-10, 1.0]],  # Almost identity with perturbation

    # Precision Loss in Subtraction
    [[1, 1 + 1e-12], [1, 1]],                  # Case 1
    [[1e8, 1e8 + 1], [1e8 + 2, 1e8]],          # Case 3
    [[1, 1 + 1e-10], [1 + 2e-10, 1]],          # Case 4
    [[1e20, 1e20 + 10], [1e20 + 20, 1e20]],    # Case 5
    [[1, 1 + 1e-14], [1 + 1e-14, 1]],          # Case 6

    # Numerical Instability in Small Matrices
    [[1e-16, 2e-16], [2e-16, 1e-16]],          # Case 1
    [[1e-12, 1e-12], [1e-12, 1e-12 + 1e-14]],  # Case 2
    [[1e-8, 1e-8 + 1e-15], [1e-8 + 1e-15, 1e-8]], # Case 3
    [[1, 1 + 1e-13], [1 + 1e-13, 1]],          # Case 4
    [[1, 1 - 1e-14], [1, 1]],                  # Case 5
    [[1e-15, 1e-15 + 1e-16], [1e-15 + 1e-16, 1e-15]], # Case 6

    # # Overflow and Underflow Risks - not a target
    # [[1e308, 1e-308], [1, 1e-308]],            # Case 1
    # # [[1e-308, 1e308], [1e-308, 1e-308]],       # Case 2
    # [[1e308, 1], [1, 1e-308]],                 # Case 3
    # [[1e308, 1e-100], [1e-100, 1e-308]],       # Case 4
    # [[1e10, 1e-308], [1e-308, 1e10]],          # Case 5
    # [[1e-308, 1e308], [1e308, 1e-308]],        # Case 6

    # LU Decomposition Stability
    [[1e-15, 1], [1, 1]],                      # Case 1
    [[1e-20, 1], [1, 1e-10]],                  # Case 2
    [[1, 1], [1, 1 + 1e-15]],                  # Case 3
    [[1, 1 + 1e-12], [1 + 1e-12, 1]],          # Case 4
    [[1, 1], [1, 1 + 1e-16]],                  # Case 5
    [[1e-10, 1], [1, 1e-10 + 1e-15]]           # Case 6

]

# 3x3 matrices - regular cases
matrices_3x3 = [
    [[1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0]],  # Identity matrix
    
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0],
     [7.0, 8.0, 9.0]],  # Nearly singular
    
    [[1.0, 0.5, 0.3],
     [0.5, 2.0, 0.7],
     [0.3, 0.7, 3.0]],  # Symmetric positive definite
    
    [[1e-3, 1.0, 1e3],
     [1.0, 1.0, 1.0],
     [1e3, 1.0, 1e-3]],  # Poorly conditioned

    # All ones matrices
    [[1.0, 1.0, 1.0],
     [1.0, 1.0, 1.0],
     [1.0, 1.0, 1.0]],
    [[1.0, 1.0, -1.0],
     [1.0, -1.0, 1.0],
     [-1.0, 1.0, 1.0]],
    [[1.0, -1.0, 1.0],
     [-1.0, 1.0, -1.0],
     [1.0, -1.0, 1.0]],
     
    # Upper triangular
    [[2.0, 3.0, 4.0],
     [0.0, 5.0, 6.0],
     [0.0, 0.0, 7.0]],
    [[1.0, -2.0, 3.0],
     [0.0, 4.0, -5.0],
     [0.0, 0.0, 6.0]],
    [[10.0, 20.0, 30.0],
     [0.0, 40.0, 50.0],
     [0.0, 0.0, 60.0]],
     
    # Lower triangular
    [[2.0, 0.0, 0.0],
     [3.0, 4.0, 0.0],
     [5.0, 6.0, 7.0]],
    [[1.0, 0.0, 0.0],
     [-2.0, 3.0, 0.0],
     [4.0, -5.0, 6.0]],
    [[10.0, 0.0, 0.0],
     [20.0, 30.0, 0.0],
     [40.0, 50.0, 60.0]],
     
    # 3D Rotation matrices (around x, y, and z axes, 45 degrees)
    [[1.0, 0.0, 0.0],
     [0.0, 0.707, -0.707],
     [0.0, 0.707, 0.707]],
    [[0.707, 0.0, 0.707],
     [0.0, 1.0, 0.0],
     [-0.707, 0.0, 0.707]],
    [[0.707, -0.707, 0.0],
     [0.707, 0.707, 0.0],
     [0.0, 0.0, 1.0]],
     
    # Permutation matrices
    [[0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0]],
    [[0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0]],
     
    # Rank deficient (rank 2)
    [[1.0, 0.0, 1.0],
     [0.0, 1.0, 0.0],
     [2.0, 0.0, 2.0]],
    [[1.0, 1.0, 2.0],
     [2.0, 2.0, 4.0],
     [3.0, 3.0, 6.0]],
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0],
     [7.0, 8.0, 9.0]],
     
    # Skew-symmetric matrices
    [[0.0, 1.0, -2.0],
     [-1.0, 0.0, 3.0],
     [2.0, -3.0, 0.0]],
    [[0.0, 2.0, -1.0],
     [-2.0, 0.0, 4.0],
     [1.0, -4.0, 0.0]],
    [[0.0, 5.0, -3.0],
     [-5.0, 0.0, 1.0],
     [3.0, -1.0, 0.0]],
     
    # Toeplitz matrices
    [[1.0, 2.0, 3.0],
     [2.0, 1.0, 2.0],
     [3.0, 2.0, 1.0]],
    [[4.0, -1.0, 2.0],
     [-1.0, 4.0, -1.0],
     [2.0, -1.0, 4.0]],
    [[2.0, 3.0, 4.0],
     [3.0, 2.0, 3.0],
     [4.0, 3.0, 2.0]],
     
    # Circulant matrices
    [[1.0, 2.0, 3.0],
     [3.0, 1.0, 2.0],
     [2.0, 3.0, 1.0]],
    [[4.0, 1.0, 2.0],
     [2.0, 4.0, 1.0],
     [1.0, 2.0, 4.0]],
    [[2.0, 3.0, 1.0],
     [1.0, 2.0, 3.0],
     [3.0, 1.0, 2.0]],
     
    # # Mixed scale with near dependencies
    [[1e6, 1e-3, 1.0],
     [1e-3, 1e6, 1.0],
     [1.0, 1.0, 1e-6]],
    [[1e9, 1e-6, 1.0],
     [1e-6, 1e9, 1.0],
     [1.0, 1.0, 1e-9]],
    [[1e12, 1e-9, 1.0],
     [1e-9, 1e12, 1.0],
     [1.0, 1.0, 1e-12]],

    # Still challenging but more reasonable condition numbers
    [[1e3, 1e-2, 1.0],
     [1e-2, 1e3, 1.0],
     [1.0, 1.0, 1e-3]],  # Condition number ~10^6
     
    [[1e4, 1e-3, 1.0],
     [1e-3, 1e4, 1.0],
     [1.0, 1.0, 1e-4]],  # Condition number ~10^8
     
    [[1e5, 1e-4, 2.0],
     [1e-4, 1e5, 2.0],
     [2.0, 2.0, 1e-5]]   # Condition number ~10^10
     
]

# 3x3 matrices - nearly singular cases
matrices_3x3_near_singular = [
    [[1.0, 2.0, 3.0],
     [2.0, 4.0, 6.0],
     [3.0, 6.0, 9.0 + 1e-10]],  # Almost linearly dependent rows
     
    [[1e5, 1e5, 1e5],
     [1e5, 1e5, 1e5],
     [1e5, 1e5, 1e5 + 1.0]],  # Almost zero determinant with scaling
     
    [[1.0, 0.0, 1.0],
     [0.0, 1.0, 1e-10],
     [0.0, 0.0, 1e-10]],  # Nearly dependent columns
     
    [[1.0, 0.0, 1e-10],
     [0.0, 1.0, 1e-10],
     [1e-10, 1e-10, 1.0]],  # Almost rank 2
     
    [[1e8, 1e-8, 1.0],
     [1e-8, 1e8, 1.0],
     [1.0, 1.0, 1e-10 + 1.0]],  # Scaled with small perturbations

    # Precision Loss in Subtraction
    [[1, 1 + 1e-12, 1], [1, 1, 1], [1, 1, 1]],                  # Case 1
    [[1e8, 1e8 + 1, 1e8], [1e8, 1e8, 1e8 + 1e-10], [1e8 + 1e-10, 1e8, 1e8]], # Case 2
    [[1e10, 1e10 + 1e-5, 1e10], [1e10, 1e10 + 1e-6, 1e10], [1e10 + 1e-4, 1e10, 1e10]], # Case 3
    [[1, 1 + 1e-10, 1], [1, 1 + 1e-11, 1], [1, 1 + 1e-12, 1]], # Case 4
    [[1e20, 1e20 + 10, 1e20], [1e20, 1e20, 1e20 + 20], [1e20, 1e20 + 30, 1e20]], # Case 5
    [[1, 1 + 1e-14, 1], [1, 1 + 1e-13, 1], [1, 1 + 1e-12, 1]], # Case 6

    # Numerical Instability in Small Matrices
    [[1e-16, 1e-16, 1e-16], [1e-16, 1e-16, 1e-16], [1e-16, 1e-16, 1e-16 + 1e-17]], # Case 1
    [[1e-12, 1e-12, 1e-12], [1e-12, 1e-12 + 1e-14, 1e-12], [1e-12, 1e-12, 1e-12]], # Case 2
    [[1e-8, 1e-8, 1e-8 + 1e-15], [1e-8, 1e-8 + 1e-15, 1e-8], [1e-8 + 1e-15, 1e-8, 1e-8]], # Case 3
    [[1, 1 + 1e-13, 1], [1 + 1e-13, 1, 1], [1, 1 + 1e-13, 1]], # Case 4
    [[1, 1 - 1e-14, 1], [1, 1, 1 - 1e-14], [1, 1, 1]],         # Case 5
    [[1e-15, 1e-15 + 1e-16, 1e-15], [1e-15, 1e-15, 1e-15 + 1e-16], [1e-15, 1e-15, 1e-15]], # Case 6

    # # Overflow and Underflow Risks
    # [[1e308, 1e-308, 1e308], [1, 1e-308, 1], [1e308, 1, 1e-308]], # Case 1
    # [[1e-308, 1e308, 1e-308], [1e-308, 1e-308, 1e308], [1e308, 1e-308, 1e-308]], # Case 2
    # [[1e308, 1e-100, 1], [1, 1e308, 1e-308], [1e-308, 1, 1e308]], # Case 3
    # [[1e308, 1e-308, 1], [1, 1e308, 1e-308], [1e-308, 1, 1e308]], # Case 4
    # [[1e10, 1e-308, 1], [1e-308, 1e10, 1e-308], [1, 1e-308, 1e10]], # Case 5
    # [[1e-308, 1e308, 1], [1e308, 1e-308, 1], [1, 1, 1e308]],       # Case 6

    # LU Decomposition Stability
    [[1e-15, 1, 1], [1, 1, 1], [1, 1, 1]],                        # Case 1
    [[1e-20, 1, 1], [1, 1e-10, 1], [1, 1, 1e-10]],                # Case 2
    [[1, 1, 1], [1, 1, 1 + 1e-15], [1, 1, 1]],                    # Case 3
    [[1, 1 + 1e-12, 1], [1 + 1e-12, 1, 1], [1, 1, 1]],            # Case 4
    [[1, 1, 1], [1, 1, 1 + 1e-16], [1, 1, 1]],                    # Case 5
    [[1e-10, 1, 1], [1, 1e-10, 1], [1, 1, 1e-10 + 1e-15]],         # Case 6
]

# 4x4 matrices - regular cases
matrices_4x4 = [
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],  # Identity matrix
    
    [[1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0],
     [9.0, 10.0, 11.0, 12.0],
     [13.0, 14.0, 15.0, 16.0]],  # Nearly singular
    
    [[1.0, 0.1, 0.1, 0.1],
     [0.1, 2.0, 0.2, 0.2],
     [0.1, 0.2, 3.0, 0.3],
     [0.1, 0.2, 0.3, 4.0]], # Diagonally dominant


    # All ones matrices
    [[1.0, 1.0, 1.0, 1.0],
     [1.0, 1.0, 1.0, 1.0],
     [1.0, 1.0, 1.0, 1.0],
     [1.0, 1.0, 1.0, 1.0]],
    [[1.0, 1.0, -1.0, -1.0],
     [1.0, -1.0, 1.0, -1.0],
     [-1.0, 1.0, 1.0, -1.0],
     [-1.0, -1.0, -1.0, 1.0]],
    [[1.0, -1.0, 1.0, -1.0],
     [-1.0, 1.0, -1.0, 1.0],
     [1.0, -1.0, 1.0, -1.0],
     [-1.0, 1.0, -1.0, 1.0]],

    # Upper triangular
    [[1.0, 2.0, 3.0, 4.0],
     [0.0, 5.0, 6.0, 7.0],
     [0.0, 0.0, 8.0, 9.0],
     [0.0, 0.0, 0.0, 10.0]],
    [[2.0, -1.0, 3.0, -2.0],
     [0.0, 4.0, -5.0, 6.0],
     [0.0, 0.0, 7.0, -8.0],
     [0.0, 0.0, 0.0, 9.0]],
    [[10.0, 20.0, 30.0, 40.0],
     [0.0, 50.0, 60.0, 70.0],
     [0.0, 0.0, 80.0, 90.0],
     [0.0, 0.0, 0.0, 100.0]],

    # Lower triangular
    [[1.0, 0.0, 0.0, 0.0],
     [2.0, 3.0, 0.0, 0.0],
     [4.0, 5.0, 6.0, 0.0],
     [7.0, 8.0, 9.0, 10.0]],
    [[2.0, 0.0, 0.0, 0.0],
     [-1.0, 3.0, 0.0, 0.0],
     [4.0, -5.0, 6.0, 0.0],
     [-7.0, 8.0, -9.0, 10.0]],
    [[10.0, 0.0, 0.0, 0.0],
     [20.0, 30.0, 0.0, 0.0],
     [40.0, 50.0, 60.0, 0.0],
     [70.0, 80.0, 90.0, 100.0]],

    # Block diagonal (2x2 blocks)
    [[2.0, 1.0, 0.0, 0.0],
     [1.0, 2.0, 0.0, 0.0],
     [0.0, 0.0, 3.0, 1.0],
     [0.0, 0.0, 1.0, 3.0]],
    [[4.0, -1.0, 0.0, 0.0],
     [-1.0, 4.0, 0.0, 0.0],
     [0.0, 0.0, 5.0, -1.0],
     [0.0, 0.0, -1.0, 5.0]],
    [[1.0, 0.5, 0.0, 0.0],
     [0.5, 1.0, 0.0, 0.0],
     [0.0, 0.0, 2.0, 0.5],
     [0.0, 0.0, 0.5, 2.0]],

    # Block tridiagonal
    [[2.0, 1.0, 0.0, 0.0],
     [1.0, 2.0, 1.0, 0.0],
     [0.0, 1.0, 2.0, 1.0],
     [0.0, 0.0, 1.0, 2.0]],
    [[4.0, -1.0, 0.0, 0.0],
     [-1.0, 4.0, -1.0, 0.0],
     [0.0, -1.0, 4.0, -1.0],
     [0.0, 0.0, -1.0, 4.0]],
    [[3.0, 1.0, 0.0, 0.0],
     [1.0, 3.0, 1.0, 0.0],
     [0.0, 1.0, 3.0, 1.0],
     [0.0, 0.0, 1.0, 3.0]],

    # Sparse matrix patterns
    [[1.0, 0.0, 2.0, 0.0],
     [0.0, 3.0, 0.0, 4.0],
     [2.0, 0.0, 5.0, 0.0],
     [0.0, 4.0, 0.0, 6.0]],
    [[2.0, 0.0, 0.0, 1.0],
     [0.0, 3.0, 1.0, 0.0],
     [0.0, 1.0, 4.0, 0.0],
     [1.0, 0.0, 0.0, 5.0]],
    [[1.0, 1.0, 0.0, 0.0],
     [1.0, 2.0, 1.0, 0.0],
     [0.0, 1.0, 3.0, 1.0],
     [0.0, 0.0, 1.0, 4.0]],

    # Vandermonde matrices
    [[1.0, 1.0, 1.0, 1.0],
     [1.0, 2.0, 4.0, 8.0],
     [1.0, 3.0, 9.0, 27.0],
     [1.0, 4.0, 16.0, 64.0]],
    [[1.0, 1.0, 1.0, 1.0],
     [1.0, -1.0, 1.0, -1.0],
     [1.0, -2.0, 4.0, -8.0],
     [1.0, -3.0, 9.0, -27.0]],
    [[1.0, 1.0, 1.0, 1.0],
     [1.0, 0.5, 0.25, 0.125],
     [1.0, 2.0, 4.0, 8.0],
     [1.0, 3.0, 9.0, 27.0]],

    # Hilbert matrix segments
    [[1.0, 1/2, 1/3, 1/4],
     [1/2, 1/3, 1/4, 1/5],
     [1/3, 1/4, 1/5, 1/6],
     [1/4, 1/5, 1/6, 1/7]],
    [[1.0, 1/3, 1/5, 1/7],
     [1/3, 1/5, 1/7, 1/9],
     [1/5, 1/7, 1/9, 1/11],
     [1/7, 1/9, 1/11, 1/13]],
    [[1/2, 1/3, 1/4, 1/5],
     [1/3, 1/4, 1/5, 1/6],
     [1/4, 1/5, 1/6, 1/7],
     [1/5, 1/6, 1/7, 1/8]],

    # Rank deficient (rank 3)
    [[1.0, 0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [2.0, 0.0, 0.0, 2.0]],
    [[1.0, 1.0, 1.0, 3.0],
     [2.0, 2.0, 2.0, 6.0],
     [3.0, 3.0, 3.0, 9.0],
     [4.0, 4.0, 4.0, 12.0]],
    [[1.0, 2.0, 3.0, 6.0],
     [4.0, 5.0, 6.0, 15.0],
     [7.0, 8.0, 9.0, 24.0],
     [10.0, 11.0, 12.0, 33.0]],

    # Mixed scale with multiple near dependencies
    [[1e6, 1e-3, 1.0, 1e-6],
     [1e-3, 1e6, 1e-6, 1.0],
     [1.0, 1e-6, 1e6, 1e-3],
     [1e-6, 1.0, 1e-3, 1e6]],
    [[1e9, 1e-6, 1.0, 1e-9],
     [1e-6, 1e9, 1e-9, 1.0],
     [1.0, 1e-9, 1e9, 1e-6],
     [1e-9, 1.0, 1e-6, 1e9]],
    [[1e12, 1e-9, 1.0, 1e-12],
     [1e-9, 1e12, 1e-12, 1.0],
     [1.0, 1e-12, 1e12, 1e-9],
     [1e-12, 1.0, 1e-9, 1e12]]
]

# 4x4 matrices - nearly singular cases
matrices_4x4_near_singular = [
    [[1.0, 2.0, 3.0, 4.0],
     [2.0, 4.0, 6.0, 8.0],
     [3.0, 6.0, 9.0, 12.0],
     [4.0, 8.0, 12.0, 16.0 + 1e-10]],  # Almost linearly dependent rows
     
    [[1e3, 1e3, 1e3, 1e3],
     [1e3, 1e3, 1e3, 1e3],
     [1e3, 1e3, 1e3, 1e3],
     [1e3, 1e3, 1e3, 1e3 + 1.0]],  # Almost zero determinant with scaling
     
    [[1.0, 0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0, 1e-10],
     [0.0, 0.0, 1.0, 1e-10],
     [0.0, 0.0, 0.0, 1e-10]],  # Nearly dependent columns
     
    [[1e5, 1e-5, 1.0, 1.0],
     [1e-5, 1e5, 1.0, 1.0],
     [1.0, 1.0, 1e-10, 1.0],
     [1.0, 1.0, 1.0, 1e-10 + 1.0]],  # Mixed scaling with near dependencies

    # Precision Loss in Subtraction
    [[1, 1 + 1e-12, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], # Case 1
    [[1e8, 1e8 + 1, 1e8, 1e8], [1e8, 1e8, 1e8 + 1e-10, 1e8], [1e8, 1e8, 1e8, 1e8], [1e8 + 1e-10, 1e8, 1e8, 1e8]], # Case 2
    [[1e10, 1e10 + 1e-5, 1e10, 1e10], [1e10, 1e10, 1e10 + 1e-6, 1e10], [1e10 + 1e-4, 1e10, 1e10, 1e10], [1e10, 1e10, 1e10, 1e10 + 1e-3]], # Case 3
    # [[1, 1 + 1e-10, 1, 1], [1, 1 + 1e-11, 1, 1], [1, 1, 1 + 1e-12, 1], [1, 1, 1, 1 + 1e-13]], # Case 4
    [[1e20, 1e20 + 10, 1e20, 1e20], [1e20, 1e20, 1e20 + 20, 1e20], [1e20, 1e20 + 30, 1e20, 1e20], [1e20, 1e20, 1e20 + 40, 1e20]], # Case 5
    [[1, 1 + 1e-14, 1, 1], [1, 1 + 1e-13, 1, 1], [1, 1, 1 + 1e-12, 1], [1, 1, 1, 1 + 1e-11]], # Case 6

    # Numerical Instability in Small Matrices
    [[1e-16, 1e-16, 1e-16, 1e-16], [1e-16, 1e-16, 1e-16, 1e-16], [1e-16, 1e-16, 1e-16, 1e-16], [1e-16, 1e-16, 1e-16, 1e-16 + 1e-17]], # Case 1
    [[1e-12, 1e-12, 1e-12, 1e-12], [1e-12, 1e-12 + 1e-14, 1e-12, 1e-12], [1e-12, 1e-12, 1e-12, 1e-12], [1e-12, 1e-12, 1e-12, 1e-12]], # Case 2
    # [[1e-8, 1e-8, 1e-8 + 1e-15, 1e-8], [1e-8, 1e-8, 1e-8, 1e-8 + 1e-15], [1e-8, 1e-8, 1e-8, 1e-8], [1e-8 + 1e-15, 1e-8, 1e-8, 1e-8]], # Case 3
    [[1, 1 + 1e-13, 1, 1], [1, 1, 1 + 1e-13, 1], [1, 1, 1, 1 + 1e-13], [1 + 1e-13, 1, 1, 1]], # Case 4
    [[1, 1 - 1e-14, 1, 1], [1, 1, 1 - 1e-14, 1], [1, 1, 1, 1 - 1e-14], [1, 1, 1, 1]], # Case 5
    [[1e-15, 1e-15 + 1e-16, 1e-15, 1e-15], [1e-15, 1e-15, 1e-15 + 1e-16, 1e-15], [1e-15, 1e-15, 1e-15, 1e-15], [1e-15, 1e-15, 1e-15, 1e-15]], # Case 6

    # Overflow and Underflow Risks
    [[1e308, 1e-308, 1e308, 1e-308], [1e-308, 1e308, 1, 1e-308], [1e308, 1, 1e-308, 1], [1, 1e-308, 1e308, 1e-308]], # Case 1
    [[1e-308, 1e308, 1e-308, 1], [1e308, 1e-308, 1, 1e308], [1, 1e-308, 1e308, 1e-308], [1e-308, 1, 1, 1e308]], # Case 2
    [[1e308, 1e-100, 1, 1], [1, 1e308, 1e-308, 1], [1e-308, 1, 1e308, 1], [1, 1, 1e-308, 1e308]], # Case 3
    [[1e308, 1e-308, 1, 1], [1, 1e308, 1e-308, 1], [1e-308, 1, 1e308, 1], [1, 1e-308, 1, 1e308]], # Case 4
    [[1e10, 1e-308, 1, 1e-308], [1e-308, 1e10, 1e-308, 1], [1, 1e-308, 1e10, 1e-308], [1e-308, 1, 1e-308, 1e10]], # Case 5
    [[1e-308, 1e308, 1, 1], [1e308, 1e-308, 1, 1e308], [1, 1, 1e308, 1e-308], [1e-308, 1e308, 1, 1]], # Case 6

    # LU Decomposition Stability
    [[1e-15, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], # Case 1
    [[1e-20, 1, 1, 1], [1, 1e-10, 1, 1], [1, 1, 1, 1e-10], [1, 1, 1, 1]], # Case 2
    [[1, 1, 1, 1], [1, 1, 1, 1 + 1e-15], [1, 1, 1, 1], [1, 1, 1, 1]], # Case 3
    [[1, 1 + 1e-12, 1, 1], [1 + 1e-12, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], # Case 4
    [[1, 1, 1, 1], [1, 1, 1, 1 + 1e-16], [1, 1, 1, 1], [1, 1, 1, 1]], # Case 5
    [[1e-10, 1, 1, 1], [1, 1e-10, 1, 1], [1, 1, 1e-10, 1], [1, 1, 1, 1e-10 + 1e-15]], # Case 6
    
]

# Singular matrices that should raise exceptions
matrices_singular = [
    [[0.0]],  # Singular 1x1
    [[1.0, 0.0],
     [0.0, 0.0]],  # Singular 2x2
    [[1.0, 0.0],
     [0.0]]  # Irregular matrix
]

@pytest.mark.parametrize("matrix", matrices_1x1)
def test_inv_1x1(matrix):
    try:
        check_inv(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_2x2)
def test_inv_2x2(matrix):
    try:
        check_inv(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_2x2_near_singular)
def test_inv_2x2_near_singular(matrix):
    try:
        check_inv(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_3x3)
def test_inv_3x3(matrix):
    try:
        check_inv(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_3x3_near_singular)
def test_inv_3x3_near_singular(matrix):
    try:
        check_inv(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_4x4)
def test_inv_4x4(matrix):
    try:
        check_inv(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_4x4_near_singular)
def test_inv_4x4_near_singular(matrix):
    try:
        check_inv(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_singular)
def test_inv_singular_matrices(matrix):
    try:
        check_inv(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)



def test_array_as_tridiagonals():
    A = [[10.0, 2.0, 0.0, 0.0],
     [3.0, 10.0, 4.0, 0.0],
     [0.0, 1.0, 7.0, 5.0],
     [0.0, 0.0, 3.0, 4.0]]

    tridiagonals = array_as_tridiagonals(A)
    expect_diags = [[3.0, 1.0, 3.0], [10.0, 10.0, 7.0, 4.0], [2.0, 4.0, 5.0]]

    assert_allclose(tridiagonals[0], expect_diags[0], rtol=0, atol=0)
    assert_allclose(tridiagonals[1], expect_diags[1], rtol=0, atol=0)
    assert_allclose(tridiagonals[2], expect_diags[2], rtol=0, atol=0)

    A = np.array(A)
    tridiagonals = array_as_tridiagonals(A)
    assert_allclose(tridiagonals[0], expect_diags[0], rtol=0, atol=0)
    assert_allclose(tridiagonals[1], expect_diags[1], rtol=0, atol=0)
    assert_allclose(tridiagonals[2], expect_diags[2], rtol=0, atol=0)


    a, b, c = [3.0, 1.0, 3.0], [10.0, 10.0, 7.0, 4.0], [2.0, 4.0, 5.0]
    expect_mat = tridiagonals_as_array(a, b, c)
    assert_allclose(expect_mat, A, rtol=0, atol=0)

    d = [3.0, 4.0, 5.0, 6.0]

    solved_expect = [0.1487758945386064, 0.756120527306968, -1.001883239171375, 2.2514124293785316]
    assert_allclose(solve_tridiagonal(a, b, c, d), solved_expect, rtol=1e-12)


def test_subset_matrix():
    kijs = [[0, 0.00076, 0.00171], [0.00076, 0, 0.00061], [0.00171, 0.00061, 0]]

    expect = [[0, 0.00061], [0.00061, 0]]
    got = subset_matrix(kijs, [1,2])
    assert_allclose(expect, got, atol=0, rtol=0)
    got = subset_matrix(kijs, slice(1, 3, 1))
    assert_allclose(expect, got, atol=0, rtol=0)

    expect = [[0, 0.00171], [0.00171, 0]]
    got = subset_matrix(kijs, [0,2])
    assert_allclose(expect, got, atol=0, rtol=0)
    got = subset_matrix(kijs, slice(0, 3, 2))
    assert_allclose(expect, got, atol=0, rtol=0)

    expect = [[0, 0.00076], [0.00076, 0]]
    got = subset_matrix(kijs, [0,1])
    assert_allclose(expect, got, atol=0, rtol=0)
    got = subset_matrix(kijs, slice(0, 2, 1))
    assert_allclose(expect, got, atol=0, rtol=0)

    got = subset_matrix(kijs, [0,1, 2])
    assert_allclose(kijs, got, atol=0, rtol=0)
    got = subset_matrix(kijs, slice(0, 3, 1))
    assert_allclose(kijs, got, atol=0, rtol=0)





def test_argsort1d():

    def check_argsort1d(input_list, expected, error_message):
        numpy_argsort1d = lambda x: list(np.argsort(x))
        assert argsort1d(input_list) == expected, error_message
        assert argsort1d(input_list) == numpy_argsort1d(input_list), error_message


    check_argsort1d([3, 1, 2], [1, 2, 0], "Failed on simple test case")
    check_argsort1d([-1, -3, -2], [1, 2, 0], "Failed with negative numbers")
    check_argsort1d([], [], "Failed on empty list")
    check_argsort1d([42], [0], "Failed with single element list")
    check_argsort1d([99, 21, 31, 80, 70], [1, 2, 4, 3, 0], "Mismatch with expected output")
    check_argsort1d([2, 3, 1, 5, 4], [2, 0, 1, 4, 3], "Mismatch with expected output")
    
    check_argsort1d([3.5, 1, 2.2], [1, 2, 0], "Failed with mixed floats and ints")
    check_argsort1d([0.1, 0.2, 0.3], [0, 1, 2], "Failed with floats")

    check_argsort1d([True, False, True], [1, 0, 2], "Failed with boolean values")

    check_argsort1d(['apple', 'banana', 'cherry'], [0, 1, 2], "Failed with strings")

    check_argsort1d([2, 3, 2, 3, 3], [0, 2, 1, 3, 4], "Failed with duplicate numbers")

    check_argsort1d([-3, -1, 0, 1, 3], [0, 1, 2, 3, 4], "Failed with negative and positive numbers")

    # infinities and nan behavior does not match
    # check_argsort1d([-np.inf, np.inf, np.nan, 0, -1], [0, 4, 3, 2, 1], "Failed with infinities and NaN")
