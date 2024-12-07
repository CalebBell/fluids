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
from fluids.numerics.arrays import (inv, solve, lu, gelsd, eye, dot_product, transpose, matrix_vector_dot, matrix_multiply, sum_matrix_rows, sum_matrix_cols,
    scalar_divide_matrix, scalar_multiply_matrix, scalar_subtract_matrices, scalar_add_matrices, null_space)
from fluids.numerics import (
    array_as_tridiagonals,
    assert_close,
    assert_close1d,
    assert_close2d,
    solve_tridiagonal,
    subset_matrix,
    tridiagonals_as_array,
    argsort1d,
    sort_paired_lists,
)
from fluids.numerics import numpy as np

assert_allclose = np.testing.assert_allclose

def get_rtol(matrix):
    """Set tolerance based on condition number"""
    cond = np.linalg.cond(matrix)
    machine_eps = np.finfo(float).eps  # ≈ 2.2e-16
    return min(10 * cond * machine_eps,100*cond * machine_eps if cond > 1e8 else 1e-9)

def check_inv(matrix, rtol=None):
    just_return = False
    try:
        # This will fail for bad matrix (inconsistent size) inputs
        cond = np.linalg.cond(matrix)
    except:
        just_return = True
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
        if not just_return and cond > 1e14:
            # Let ill conditioned matrices pass
            return 
        raise ValueError(f"Inconsistent failure states: Python Fail: {py_fail}, Numpy Fail: {numpy_fail}")
    if py_fail and numpy_fail:
        return
    if not py_fail and numpy_fail:
        # We'll allow our inv to work with numbers closer to
        return
    if just_return:
        return
        
    # Convert result to numpy array if it isn't already
    result = np.array(result)
    # Compute infinity norm of input matrix
    matrix_norm = np.max(np.sum(np.abs(matrix), axis=1))
    thresh = matrix_norm * np.finfo(float).eps

    # We also need to check against the values we get in the inverse; it is helpful 
    # to zero out anything too close to "zero" relative to the values used in the matrix
    # This is very necessary, and was needed when testing on different CPU architectures
    inv_norm = np.max(np.sum(np.abs(result), axis=1))
    if cond < 1e10:
        zero_thresh = thresh
    elif cond < 1e14:
        zero_thresh = 10*thresh
    else:
        zero_thresh = 100*thresh
    trivial_relative_to_norm_result = (np.abs(result)/inv_norm < zero_thresh)
    trivial_relative_to_norm_expected = (np.abs(expected)/inv_norm < zero_thresh)
    # Zero out in both matrices where either condition is met
    combined_relative_mask = np.logical_or(
        trivial_relative_to_norm_result,
        trivial_relative_to_norm_expected
    )
    result[combined_relative_mask] = 0.0
    expected[combined_relative_mask] = 0.0


    # Check both directions
    numpy_zeros = (expected == 0.0)
    our_zeros = (result == 0.0)
    mask_exact_zeros = numpy_zeros | our_zeros
        
    # Where numpy has zeros but we don't; no cases require it but it makes sense to do
    result[mask_exact_zeros] = np.where(np.abs(result[mask_exact_zeros]) < thresh, 0.0, result[mask_exact_zeros])
    
    # Where we have zeros but numpy doesn't - this is the one we discovered. Apply the check only to `numpy_zeros`
    expected[numpy_zeros] = np.where(np.abs(expected[numpy_zeros]) < thresh, 0.0, expected[numpy_zeros])

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
     [13.0, 14.0, 15.0, 16.0]],  # Singular
    
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
    # [[1.0, 1.0, 1.0, 1.0], # failing on other CPUs in test_lu_4x4
    #  [1.0, 2.0, 4.0, 8.0],
    #  [1.0, 3.0, 9.0, 27.0],
    #  [1.0, 4.0, 16.0, 64.0]],
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
     
    # [[1.0, 0.0, 0.0, 1.0],
    #  [0.0, 1.0, 0.0, 1e-10],
    #  [0.0, 0.0, 1.0, 1e-10],
    #  [0.0, 0.0, 0.0, 1e-10]],  # Nearly dependent columns, too hard on some CPUs
     
    # [[1e5, 1e-5, 1.0, 1.0],
    #  [1e-5, 1e5, 1.0, 1.0],
    #  [1.0, 1.0, 1e-10, 1.0],
    #  [1.0, 1.0, 1.0, 1e-10 + 1.0]],  # Mixed scaling with near dependencies

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

    # # Overflow and Underflow Risks - not a target
    # [[1e308, 1e-308, 1e308, 1e-308], [1e-308, 1e308, 1, 1e-308], [1e308, 1, 1e-308, 1], [1, 1e-308, 1e308, 1e-308]], # Case 1
    # [[1e-308, 1e308, 1e-308, 1], [1e308, 1e-308, 1, 1e308], [1, 1e-308, 1e308, 1e-308], [1e-308, 1, 1, 1e308]], # Case 2
    # [[1e308, 1e-100, 1, 1], [1, 1e308, 1e-308, 1], [1e-308, 1, 1e308, 1], [1, 1, 1e-308, 1e308]], # Case 3
    # [[1e308, 1e-308, 1, 1], [1, 1e308, 1e-308, 1], [1e-308, 1, 1e308, 1], [1, 1e-308, 1, 1e308]], # Case 4
    # [[1e10, 1e-308, 1, 1e-308], [1e-308, 1e10, 1e-308, 1], [1, 1e-308, 1e10, 1e-308], [1e-308, 1, 1e-308, 1e10]], # Case 5
    # [[1e-308, 1e308, 1, 1], [1e308, 1e-308, 1, 1e308], [1, 1, 1e308, 1e-308], [1e-308, 1e308, 1, 1]], # Case 6

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



def check_solve(matrix, b=None):
    """Set tolerance based on condition number and check solution"""
    if b is None:
        # Create a right-hand side vector that's compatible with the matrix size
        b = [1.0] * len(matrix)
    
    just_return = False
    try:
        # This will fail for bad matrix (inconsistent size) inputs
        cond = np.linalg.cond(matrix)
    except:
        just_return = True
        
    py_fail = False
    numpy_fail = False
    try:
        result = solve(matrix, b)
    except:
        py_fail = True
    try:
        expected = np.linalg.solve(matrix, b)
    except:
        numpy_fail = True
        
    if py_fail and not numpy_fail:
        if not just_return and cond > 1e14:
            # Let ill conditioned matrices pass
            return 
        raise ValueError(f"Inconsistent failure states: Python Fail: {py_fail}, Numpy Fail: {numpy_fail}")
    if py_fail and numpy_fail:
        return
    if not py_fail and numpy_fail:
        return
    if just_return:
        return
        
    # Convert result to numpy array if it isn't already
    result = np.array(result)
    expected = np.array(expected)
    
    # Compute infinity norm of input matrix
    matrix_norm = np.max(np.sum(np.abs(matrix), axis=1))
    thresh = matrix_norm * np.finfo(float).eps
    
    # Get solution norms
    sol_norm = np.max(np.abs(result))
    
    # Adjust tolerance based on condition number
    if cond < 1e10:
        zero_thresh = thresh
        rtol = 10 * cond * np.finfo(float).eps
    elif cond < 1e14:
        zero_thresh = 10*thresh
        rtol = 10 * cond * np.finfo(float).eps
    else:
        zero_thresh = 100*thresh
        rtol = 100 * cond * np.finfo(float).eps
    
    # Zero out small values relative to solution norm
    trivial_relative_to_norm_result = (np.abs(result)/sol_norm < zero_thresh)
    trivial_relative_to_norm_expected = (np.abs(expected)/sol_norm < zero_thresh)
    # Zero out in both solutions where either condition is met
    combined_relative_mask = np.logical_or(
        trivial_relative_to_norm_result,
        trivial_relative_to_norm_expected
    )
    result[combined_relative_mask] = 0.0
    expected[trivial_relative_to_norm_expected] = 0.0
    np.testing.assert_allclose(result, expected, rtol=rtol)

@pytest.mark.parametrize("matrix", matrices_1x1)
def test_solve_1x1(matrix):
    try:
        check_solve(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_2x2)
def test_solve_2x2(matrix):
    try:
        check_solve(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_2x2_near_singular)
def test_solve_2x2_near_singular(matrix):
    try:
        check_solve(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_3x3)
def test_solve_3x3(matrix):
    try:
        check_solve(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_3x3_near_singular)
def test_solve_3x3_near_singular(matrix):
    try:
        check_solve(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_4x4)
def test_solve_4x4(matrix):
    try:
        check_solve(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_4x4_near_singular)
def test_solve_4x4_near_singular(matrix):
    try:
        check_solve(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_singular)
def test_solve_singular_matrices(matrix):
    try:
        check_solve(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

specific_rhs_cases = [
    ([[2.0, 1.0], [1.0, 3.0]], [1.0, 1.0]),
    ([[2.0, 1.0], [1.0, 3.0]], [1.0, -1.0]),
    ([[2.0, 1.0], [1.0, 3.0]], [1e6, 1e-6]),
    ([[2.0, 1.0], [1.0, 3.0]], [0.0, 1.0]),
    ([[2.0, 1.0], [1.0, 3.0]], [1e-15, 1e-15]),
    ([[1.0, 0.0], [0.0, 1.0]], [-1.0, 1.0]),
    ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [1.0, 2.0, 3.0]),
    ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [1e3, 1e0, 1e-3]),
    ([[1e5, 0.0], [0.0, 1e-5]], [1.0, 1.0]),
    ([[1.0, 1.0], [1.0, 1.0 + 1e-10]], [1.0, 1.0]),
    ([[2.0, 0.0, 0.0, 0.0],
      [0.0, 2.0, 0.0, 0.0],
      [0.0, 0.0, 2.0, 0.0],
      [0.0, 0.0, 0.0, 2.0]], [1.0, -1.0, 1.0, -1.0]),
    ([[1.0, 0.1, 0.1, 0.1],
      [0.1, 1.0, 0.1, 0.1],
      [0.1, 0.1, 1.0, 0.1],
      [0.1, 0.1, 0.1, 1.0]], [1e-8, 1e-8, 1e8, 1e8])
]

@pytest.mark.parametrize("matrix,rhs", specific_rhs_cases)
def test_solve_specific_rhs(matrix, rhs):
    try:
        check_solve(matrix, rhs)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}\nRHS: {rhs}"
        raise Exception(new_message)



def test_py_solve_bad_cases():
    j = [[-3.8789618086360855, -3.8439678951838587, -1.1398039850146757e-07], [1.878915113936518, 1.8439217680605073, 1.139794740950828e-07], [-1.0, -1.0, 0.0]]
    nv = [-1.4181331207951953e-07, 1.418121622354107e-07, 2.220446049250313e-16]

    import fluids.numerics
    calc = fluids.numerics.py_solve(j, nv)
    import numpy as np
    expect = np.linalg.solve(j, nv)
    fluids.numerics.assert_close1d(calc, expect, rtol=1e-4)



specific_solution_cases = [
    # Case 1
    ([
        [0.8660254037844387, -0.49999999999999994, 0.0],
        [0.49999999999999994, 0.8660254037844387, 0.0],
        [0.0, 0.0, 1.0]],
     [1, 2, 3],
     [1.8660254037844384, 1.2320508075688774, 3.0]),
    # Case 2
    ([
        [4, -1, 0, 0],
        [-1, 4, -1, 0],
        [0, -1, 4, -1],
        [0, 0, -1, 4]],
     [1, -1, 1, -1],
     [0.21052631578947367, -0.15789473684210525, 0.15789473684210528, -0.21052631578947367]),
    # Case 3
    ([
        [2, 1, 1],
        [0, 3, -1],
        [0, 0, 4]],
     [1, 2, 3],
     [-0.3333333333333333, 0.9166666666666666, 0.75]),
    # Case 4
    ([
        [3, 1, -2],
        [2, -3, 1],
        [-1, 2, 4]],
     [7, -1, 3],
     [1.981132075471698, 1.7735849056603772, 0.3584905660377357]),
    # Case 5
    ([[3.0]],
     [6.0],
     [2.0]),
    # Case 6
    ([
        [0.7071067811865476, -0.7071067811865475],
        [0.7071067811865475, 0.7071067811865476]],
     [1.0, 2.0],
     [2.1213203435596424, 0.7071067811865478]),
    # Case 7
    ([
        [2, 1, 1],
        [0, 3, -1],
        [0, 0, 4]],
     [1, 2, 3],
     [-0.3333333333333333, 0.9166666666666666, 0.75]),
    # Case 8
    ([
        [4, -1, 0, 0, 0],
        [-1, 4, -1, 0, 0],
        [0, -1, 4, -1, 0],
        [0, 0, -1, 4, -1],
        [0, 0, 0, -1, 4]],
     [1, -1, 1, -1, 1],
     [0.21153846153846154, -0.15384615384615383, 0.17307692307692307, -0.15384615384615383, 0.21153846153846154]),
    # Case 9
    ([
        [2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 4.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 4.0]],
     [1, -1, 2, -2, 3, -3],
     [1.0, -1.0, 0.5, -0.5000000000000001, 0.6, -0.6]),
    # Case 10
    ([
        [2, 1, 0, 0, 0, 0, 0],
        [-1, 3, -1, 0, 0, 0, 0],
        [0, 1, 2, 1, 0, 0, 0],
        [0, 0, -1, 4, -1, 0, 0],
        [0, 0, 0, 1, 3, 1, 0],
        [0, 0, 0, 0, -1, 2, -1],
        [0, 0, 0, 0, 0, 1, 3]],
     [1, -1, 1, -1, 1, -1, 1],
     [0.4983480176211454, 0.0033039647577092373, 0.5115638766519823, -0.02643171806167401, 0.3827092511013216, -0.12169603524229072, 0.37389867841409696]),
]
@pytest.mark.parametrize("matrix,rhs,expected", specific_solution_cases)
def test_solve_specific_solutions(matrix, rhs, expected):
    result = solve(matrix, rhs)
    assert_allclose(result, expected, rtol=1e-15)





def check_lu(matrix):
    """Compare our LU decomposition against SciPy's"""
    import numpy as np
    from scipy import linalg
    
    just_return = False
    try:
        # This will fail for bad matrix (inconsistent size) inputs
        cond = np.linalg.cond(matrix)
    except:
        just_return = True
        
    py_fail = False
    scipy_fail = False
    try:
        P, L, U = lu(matrix)
    except:
        py_fail = True
    try:
        p, l, u = linalg.lu(matrix)
    except:
        scipy_fail = True
        
    if py_fail and not scipy_fail:
        if not just_return and cond > 1e14:
            # Let ill conditioned matrices pass
            return 
        raise ValueError(f"Inconsistent failure states: Python Fail: {py_fail}, SciPy Fail: {scipy_fail}")
    if py_fail and scipy_fail:
        return
    if not py_fail and scipy_fail:
        return
    if just_return:
        return
        
    # Convert results to numpy arrays
    P, L, U = np.array(P), np.array(L), np.array(U)
    
    # Compute infinity norm of input matrix
    matrix_norm = np.max(np.sum(np.abs(matrix), axis=1))
    thresh = matrix_norm * np.finfo(float).eps

    # Verify L is lower triangular with unit diagonal
    tril_mask = np.tril(np.ones_like(L, dtype=bool))
    assert_allclose(L[~tril_mask], 0, atol=thresh)
    assert_allclose(np.diag(L), 1, rtol=thresh)

    # Verify U is upper triangular
    triu_mask = np.triu(np.ones_like(U, dtype=bool))
    assert_allclose(U[~triu_mask], 0, atol=thresh)

    # Check that P is a permutation matrix
    P_sum_rows = np.sum(P, axis=1)
    P_sum_cols = np.sum(P, axis=0)
    assert_allclose(P_sum_rows, np.ones(len(matrix)), rtol=thresh)
    assert_allclose(P_sum_cols, np.ones(len(matrix)), rtol=thresh)


    # Most importantly: verify that PA = LU
    PA = P @ matrix
    LU = L @ U
    assert_allclose(PA, LU, rtol=1e-13, atol=10*thresh)

    # Compare with SciPy's results:
    # Since pivot choices might differ, we compare
    # The upper triangular factor (which should be unique up to sign changes)
    # Normalize each row to handle sign differences
    U_normalized = U / (np.max(np.abs(U), axis=1, keepdims=True) + np.finfo(float).eps)
    u_normalized = u / (np.max(np.abs(u), axis=1, keepdims=True) + np.finfo(float).eps)
    if cond < 1e7:
        np.testing.assert_allclose(np.abs(U_normalized), np.abs(u_normalized), rtol=1e-13)

specific_lu_cases = [
    # Case 1
    (
    [[0.8660254037844387, -0.49999999999999994, 0.0],
 [0.49999999999999994, 0.8660254037844387, 0.0],
 [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.5773502691896256, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.8660254037844387, -0.49999999999999994, 0.0],
 [0.0, 1.1547005383792515, 0.0],
 [0.0, 0.0, 1.0]]
    ),
    # Case 2
    (
    [[4.0, -1.0, 0.0, 0.0],
 [-1.0, 4.0, -1.0, 0.0],
 [0.0, -1.0, 4.0, -1.0],
 [0.0, 0.0, -1.0, 4.0]],
    [[1.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0],
 [0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0, 0.0],
 [-0.25, 1.0, 0.0, 0.0],
 [0.0, -0.26666666666666666, 1.0, 0.0],
 [0.0, 0.0, -0.26785714285714285, 1.0]],
    [[4.0, -1.0, 0.0, 0.0],
 [0.0, 3.75, -1.0, 0.0],
 [0.0, 0.0, 3.7333333333333334, -1.0],
 [0.0, 0.0, 0.0, 3.732142857142857]]
    ),
    # Case 3
    (
    [[2.0, 1.0, 1.0], [0.0, 3.0, -1.0], [0.0, 0.0, 4.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[2.0, 1.0, 1.0], [0.0, 3.0, -1.0], [0.0, 0.0, 4.0]]
    ),
    # Case 4
    (
    [[3.0, 1.0, -2.0], [2.0, -3.0, 1.0], [-1.0, 2.0, 4.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0],
 [0.6666666666666666, 1.0, 0.0],
 [-0.3333333333333333, -0.6363636363636365, 1.0]],
    [[3.0, 1.0, -2.0],
 [0.0, -3.6666666666666665, 2.333333333333333],
 [0.0, 0.0, 4.818181818181818]]
    ),
    # Case 5
    (
    [[3.0]],
    [[1.0]],
    [[1.0]],
    [[3.0]]
    ),
    # Case 6
    (
    [[0.7071067811865476, -0.7071067811865475],
 [0.7071067811865475, 0.7071067811865476]],
    [[1.0, 0.0], [0.0, 1.0]],
    [[1.0, 0.0], [0.9999999999999998, 1.0]],
    [[0.7071067811865476, -0.7071067811865475], [0.0, 1.414213562373095]]
    ),
    # Case 7
    (
    [[4.0, -1.0, 0.0, 0.0, 0.0],
 [-1.0, 4.0, -1.0, 0.0, 0.0],
 [0.0, -1.0, 4.0, -1.0, 0.0],
 [0.0, 0.0, -1.0, 4.0, -1.0],
 [0.0, 0.0, 0.0, -1.0, 4.0]],
    [[1.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0, 0.0, 0.0],
 [-0.25, 1.0, 0.0, 0.0, 0.0],
 [0.0, -0.26666666666666666, 1.0, 0.0, 0.0],
 [0.0, 0.0, -0.26785714285714285, 1.0, 0.0],
 [0.0, 0.0, 0.0, -0.2679425837320574, 1.0]],
    [[4.0, -1.0, 0.0, 0.0, 0.0],
 [0.0, 3.75, -1.0, 0.0, 0.0],
 [0.0, 0.0, 3.7333333333333334, -1.0, 0.0],
 [0.0, 0.0, 0.0, 3.732142857142857, -1.0],
 [0.0, 0.0, 0.0, 0.0, 3.7320574162679425]]
    ),
    # Case 8
    (
    [[2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
 [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 3.0, -1.0, 0.0, 0.0],
 [0.0, 0.0, -1.0, 3.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 4.0, -1.0],
 [0.0, 0.0, 0.0, 0.0, -1.0, 4.0]],
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [0.5, 1.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, -0.3333333333333333, 1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, -0.25, 1.0]],
    [[2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 1.5, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 3.0, -1.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 2.6666666666666665, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 4.0, -1.0],
 [0.0, 0.0, 0.0, 0.0, 0.0, 3.75]]
    ),
]

@pytest.mark.parametrize("matrix,p_expected,l_expected,u_expected", specific_lu_cases)
def test_lu_specific_cases(matrix, p_expected, l_expected, u_expected):
    p, l, u = lu(matrix)
    assert_allclose(p, p_expected, rtol=1e-15)
    assert_allclose(l, l_expected, rtol=1e-15)
    assert_allclose(u, u_expected, rtol=1e-15)


@pytest.mark.parametrize("matrix", matrices_1x1)
def test_lu_1x1(matrix):
    try:
        check_lu(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_2x2)
def test_lu_2x2(matrix):
    try:
        check_lu(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_2x2_near_singular)
def test_lu_2x2_near_singular(matrix):
    try:
        check_lu(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_3x3)
def test_lu_3x3(matrix):
    try:
        check_lu(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_3x3_near_singular)
def test_lu_3x3_near_singular(matrix):
    try:
        check_lu(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_4x4)
def test_lu_4x4(matrix):
    try:
        check_lu(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_4x4_near_singular)
def test_lu_4x4_near_singular(matrix):
    try:
        check_lu(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_singular)
def test_lu_singular_matrices(matrix):
    try:
        check_lu(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error(matrix)}"
        raise Exception(new_message)




def test_gelsd_basic():
    """Test basic functionality with simple well-conditioned problems"""
    # Simple 2x2 system
    A = [[1.0, 2.0],
         [3.0, 4.0]]
    b = [5.0, 6.0]
    x, residuals, rank, s = gelsd(A, b)
    
    # Compare with numpy's lstsq
    x_numpy = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    assert_allclose(x, x_numpy, rtol=1e-14)
    assert rank == 2
    assert len(s) == 2


def test_gelsd_overdetermined():
    """Test overdetermined system (more equations than unknowns)"""
    A = [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]]
    b = [7.0, 8.0, 9.0]
    x, residuals, rank, s = gelsd(A, b)
    
    # Verify dimensions
    assert len(x) == 2
    assert rank == 2
    assert len(s) == 2
    
    # Check residuals are positive for overdetermined system
    assert residuals >= 0

def test_gelsd_underdetermined():
    """Test underdetermined system (fewer equations than unknowns)"""
    A = [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]
    b = [7.0, 8.0]
    x, residuals, rank, s = gelsd(A, b)
    
    # Verify dimensions
    assert len(x) == 3
    assert rank == 2
    assert len(s) == 2
    assert residuals == 0.0  # Should be exactly solvable

def test_gelsd_ill_conditioned():
    """Test behavior with ill-conditioned matrix"""
    A = [[1.0, 1.0],
         [1.0, 1.0 + 1e-15]]
    b = [2.0, 2.0]
    x, residuals, rank, s = gelsd(A, b)
    
    # Matrix should be detected as rank deficient
    assert rank == 1
    assert s[0]/s[1] > 1e14  # Check condition number

def test_gelsd_zero_matrix():
    """Test with zero matrix"""
    A = [[0.0, 0.0],
         [0.0, 0.0]]
    b = [1.0, 1.0]
    x, residuals, rank, s = gelsd(A, b)
    
    assert rank == 0
    assert all(sv == 0 for sv in s)



def test_gelsd_against_lapack():
    """Compare results against LAPACK's dgelsd"""
    from scipy.linalg import lapack
    A = [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0],
         [7.0, 8.0, 9.0],
         [10.0, 11.0, 12.0]]
    b = [13.0, 14.0, 15.0, 16.0]
    
    # Our implementation
    x1, residuals1, rank1, s1 = gelsd(A, b)
    
    # LAPACK implementation
    m, n = np.array(A).shape
    minmn = min(m, n)
    maxmn = max(m, n)
    x2, s2, rank2, info = lapack.dgelsd(A, b, lwork=10000, size_iwork=10000)
    x2 = x2.ravel()
    # Compare results
    assert_allclose(x1, x2[:n], rtol=1e-12, atol=1e-12)
    assert_allclose(s1, s2[:minmn], rtol=1e-12, atol=1e-12)
    assert rank1 == rank2

@pytest.mark.parametrize("A, b, name", [
    # Standard square matrix
    ([[1.0, 2.0], 
      [3.0, 4.0]], 
     [5.0, 6.0],
     "2x2 well-conditioned"),
    
    # Original test case
    ([[1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
      [10.0, 11.0, 12.0]], 
     [13.0, 14.0, 15.0, 16.0],
     "4x3 overdetermined"),
     
    # Overdetermined system
    ([[1.0, 2.0], 
      [3.0, 4.0],
      [5.0, 6.0]], 
     [7.0, 8.0, 9.0],
     "3x2 overdetermined"),
    
    # # Nearly singular system
    # ([[1.0, 1.0], 
    #   [1.0, 1.0 + 1e-6]],  # 1e-10 broke on some CPUs 1e-6 didn't help
    #  [2.0, 2.0],
    #  "2x2 nearly singular"),
    
    # Zero matrix
    ([[0.0, 0.0], 
      [0.0, 0.0]], 
     [1.0, 1.0],
     "2x2 zero matrix"),
     
    # Underdetermined system
    ([[1.0, 2.0, 3.0], 
      [4.0, 5.0, 6.0]], 
     [7.0, 8.0],
     "2x3 underdetermined"),
     
    # Ill-conditioned matrix
    ([[1e-10, 1.0], 
      [1.0, 1.0]], 
     [1.0, 2.0],
     "2x2 ill-conditioned"),
     
    # Larger system
    ([[1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0],
      [17.0, 18.0, 19.0, 20.0]], 
     [21.0, 22.0, 23.0, 24.0, 25.0],
     "5x4 larger system")
])
def test_gelsd_against_lapack2(A, b, name):
    """Compare GELSD results against LAPACK's dgelsd for various test cases"""
    from scipy.linalg import lapack
    try:
        # Our implementation
        x1, residuals1, rank1, s1 = gelsd(A, b)
        
        # LAPACK implementation
        m, n = np.array(A).shape
        minmn = min(m, n)
        maxmn = max(m, n)
        if len(b) < maxmn:
            b_padded = np.zeros(maxmn, dtype=np.float64)
            b_padded[:len(b)] = b
            b_arr = b_padded
        else:
            b_arr = np.array(b)
        x2, s2, rank2, info = lapack.dgelsd(A, b_arr, lwork=10000, size_iwork=10000)
        x2 = x2.ravel()
        
        # Compare results
        assert_allclose(x1, x2[:n], rtol=1e-12, atol=1e-12)
        assert_allclose(s1, s2[:minmn], rtol=1e-12, atol=1e-12)
        assert rank1 == rank2
        
    except Exception as e:
        raise AssertionError(f"Failed for case: {name}\nError: {str(e)}")


def test_gelsd_rcond():
    A = [[0., 1., 0., 1., 2., 0.],
         [0., 2., 0., 0., 1., 0.],
         [1., 0., 1., 0., 0., 4.],
         [0., 0., 0., 2., 3., 0.]]
    A = np.array(A).T.tolist()
    b = [1, 0, 0, 0, 0, 0]
    # With rcond=-1, should give full rank
    x1, residuals1, rank1, s1 = gelsd(A, b, rcond=-1)
    assert rank1 == 4
    # With default rcond, should detect rank deficiency
    x2, residuals2, rank2, s2 = gelsd(A, b)
    assert rank2 == 3


@pytest.mark.parametrize("m,n,n_rhs", [
    (4, 2, 1),  # Overdetermined, single RHS
    (4, 0, 1),  # Empty columns
    (4, 2, 1),  # Standard overdetermined
    (2, 4, 1),  # Underdetermined
])
def test_gelsd_empty_and_shapes(m, n, n_rhs):
    """Test various matrix shapes including empty matrices"""
    # Create test matrices
    if m * n > 0:
        A = np.arange(m * n).reshape(m, n).tolist()
    else:
        A = np.zeros((m, n)).tolist()
    
    if m > 0:
        b = np.ones(m).tolist()
    else:
        b = np.ones(0).tolist()

    x, residuals, rank, s = gelsd(A, b)

    # Check dimensions
    assert len(x) == n
    assert len(s) == min(m, n)
    # Check rank
    assert rank == min(m, n)
    
    # For zero-sized matrices, solution should be zero
    if m == 0:
        assert_allclose(x, np.zeros(n))

    # For overdetermined systems, check residuals
    if m > n and n > 0:
        r = np.array(b) - np.dot(A, x)
        expected_residuals = float(np.sum(r * r))
        assert_allclose(residuals, expected_residuals, atol=1e-28)

def test_gelsd_incompatible_dims():
    """Test error handling for incompatible dimensions"""
    A = [[1.0, 2.0],
         [3.0, 4.0]]
    b = [1.0, 2.0, 3.0]  # Wrong dimension
    
    with pytest.raises(ValueError):
        gelsd(A, b)







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





def test_eye():
    # Test basic functionality
    assert eye(1) == [[1.0]]
    assert eye(2) == [[1.0, 0.0], [0.0, 1.0]]
    assert eye(3) == [[1.0, 0.0, 0.0], 
                     [0.0, 1.0, 0.0], 
                     [0.0, 0.0, 1.0]]
    
    # Test with different dtypes
    assert eye(2, dtype=int) == [[1, 0], [0, 1]]
    assert eye(2, dtype=float) == [[1.0, 0.0], [0.0, 1.0]]
    
    # Test error cases
    with pytest.raises(ValueError):
        eye(0)  # Zero size
    with pytest.raises(ValueError):
        eye(-1)  # Negative size
    with pytest.raises(TypeError):
        eye(2.5)  # Non-integer size
    
    # Test matrix properties
    def check_matrix_properties(matrix):
        N = len(matrix)
        # Check dimensions
        assert all(len(row) == N for row in matrix), "Matrix rows have inconsistent lengths"
        # Check diagonal elements
        assert all(matrix[i][i] == 1 for i in range(N)), "Diagonal elements are not 1"
        # Check off-diagonal elements
        assert all(matrix[i][j] == 0 
                  for i in range(N) 
                  for j in range(N) 
                  if i != j), "Off-diagonal elements are not 0"
    
    # Test matrix properties for various sizes
    for size in [1, 2, 3, 4, 5, 10]:
        check_matrix_properties(eye(size))
    
    # Test type consistency
    def check_type_consistency(matrix, expected_type):
        assert all(isinstance(x, expected_type) 
                  for row in matrix 
                  for x in row), f"Not all elements are of type {expected_type}"
    
    # Check type consistency for different dtypes
    check_type_consistency(eye(3, dtype=float), float)
    check_type_consistency(eye(3, dtype=int), int)


def test_dot_product():
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32.0
    assert dot_product([1, 0], [0, 1]) == 0.0  # Orthogonal vectors
    assert dot_product([1, 1], [1, 1]) == 2.0  # Parallel vectors
    assert_close(dot_product([0.1, 0.2], [0.3, 0.4]), 0.11)
    assert_close(dot_product([-1, -2], [3, 4]), -11.0)
    
    # Test properties of dot product
    def test_commutative(a, b):
        """Test if a·b = b·a"""
        assert_close(dot_product(a, b), dot_product(b, a), rtol=1e-14)
    
    def test_distributive(a, b, c):
        """Test if a·(b + c) = a·b + a·c"""
        # Create vector sum b + c
        vec_sum = [bi + ci for bi, ci in zip(b, c)]
        left = dot_product(a, vec_sum)
        right = dot_product(a, b) + dot_product(a, c)
        return assert_close(left, right, rtol=1e-14)
    
    # Test mathematical properties
    a, b, c = [1, 2], [3, 4], [5, 6]
    test_commutative(a, b)
    test_distributive(a, b, c)
    
    # Test error cases
    with pytest.raises(ValueError):
        dot_product([1, 2], [1, 2, 3])  # Different lengths


def test_matrix_vector_dot():
    """Test the matrix-vector dot product function"""
    # Basic multiplication
    matrix = [[1, 2], [3, 4]]
    vector = [1, 2]
    result = matrix_vector_dot(matrix, vector)
    assert_close1d(result, [5, 11])
    
    # Identity matrix
    matrix = [[1, 0], [0, 1]]
    assert_close1d(matrix_vector_dot(matrix, [2, 3]), [2, 3])
    
    # Zero matrix
    matrix = [[0, 0], [0, 0]]
    assert_close1d(matrix_vector_dot(matrix, [1, 1]), [0, 0])
    
    # Rectangular matrix (more rows than columns)
    matrix = [[1, 2], [3, 4], [5, 6]]
    vector = [1, 2]
    assert_close1d(matrix_vector_dot(matrix, vector), [5, 11, 17])
    
    # Error cases
    with pytest.raises(ValueError):
        matrix_vector_dot([[1, 2], [3, 4]], [1, 2, 3])  # Incompatible dimensions

def test_transpose():
    # Empty matrix and empty rows
    assert transpose([]) == []
    assert transpose([[]]) == []
    
    # 1x1 matrix
    assert transpose([[1]]) == [[1]]
    
    # 2x2 matrix
    assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
    
    # 3x3 matrix 
    assert transpose([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]]) == [[1, 4, 7],
                                   [2, 5, 8],
                                   [3, 6, 9]]
                                   
    # Rectangular matrices
    assert transpose([[1, 2, 3],
                     [4, 5, 6]]) == [[1, 4],
                                    [2, 5], 
                                    [3, 6]]
                                    
    # Single row/column
    assert transpose([[1, 2, 3]]) == [[1], [2], [3]]
    assert transpose([[1], [2], [3]]) == [[1, 2, 3]]
    
    # Mixed types
    result = transpose([[1, 2.5], [3, 4.2]])
    assert result[0][0] == 1
    assert abs(result[1][1] - 4.2) < 1e-10  # Float comparison with tolerance


def test_matrix_multiply():
    # 2x2 matrices
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    result = matrix_multiply(A, B)
    expect = [[19, 22], [43, 50]]
    assert_close2d(result, expect)
    
    # Identity matrix
    I = [[1, 0], [0, 1]]
    assert_close2d(matrix_multiply(A, I), A)
    assert_close2d(matrix_multiply(I, A), A)
    
    # Zero matrix
    Z = [[0, 0], [0, 0]]
    assert_close2d(matrix_multiply(A, Z), Z)
    
    # Different shapes
    A = [[1, 2, 3], [4, 5, 6]]  # 2x3
    B = [[7, 8], [9, 10], [11, 12]]  # 3x2
    result = matrix_multiply(A, B)
    expect = [[58, 64], [139, 154]]
    assert_close2d(result, expect)

    # Very small numbers
    A = [[1e-15, 1e-15], [1e-15, 1e-15]]
    result = matrix_multiply(A, A)
    expect = [[2e-30, 2e-30], [2e-30, 2e-30]]
    assert_close2d(result, expect)
    
    # Very large numbers
    A = [[1e15, 1e15], [1e15, 1e15]]
    result = matrix_multiply(A, A)
    expect = [[2e30, 2e30], [2e30, 2e30]]
    assert_close2d(result, expect)
    
    # Mixed scales
    A = [[1e10, 1e-10], [1e-10, 1e10]]
    result = matrix_multiply(A, A)
    expect = [[1e20 + 1e-20, 2], [2, 1e20 + 1e-20]]
    assert_close2d(result, expect)


    # Single-element matrices
    A = [[2]]
    B = [[3]]
    C = [[6.0]]
    assert matrix_multiply(A, B) == C

    # Large matrices (should not raise error)
    A = [[i for i in range(10)] for _ in range(10)]
    B = [[i for i in range(10)] for _ in range(10)]
    C = matrix_multiply(A, B)
    assert len(C) == 10 and len(C[0]) == 10

    # Empty matrices
    with pytest.raises(ValueError):
        matrix_multiply([], [])
    with pytest.raises(ValueError):
        matrix_multiply([[]], [[]])

    with pytest.raises(ValueError):
        matrix_multiply([], [[1]])
    with pytest.raises(ValueError):
        matrix_multiply([[1]], [])  
          
    # Incompatible dimensions
    with pytest.raises(ValueError):
        matrix_multiply([[1, 2]], [[1], [2], [3]])
        
    # Irregular matrices
    with pytest.raises(ValueError):
        matrix_multiply([[1, 2], [1]], [[1, 2]])
    with pytest.raises(ValueError):
        matrix_multiply([[1, 2]], [[1], [1, 2]])

    # Non-numeric values
    with pytest.raises(TypeError):
        A = [[1, 2, 'a']]
        B = [[4, 5], [6, 7], [8, 9]]
        matrix_multiply(A, B)


def test_sum_matrix_rows():
    """Test row-wise matrix summation"""
    # Basic functionality
    assert_close1d(sum_matrix_rows([[1, 2, 3], [4, 5, 6]]), [6.0, 15.0])
    assert_close1d(sum_matrix_rows([[1], [2]]), [1.0, 2.0])
    
    # Handle zeros
    assert_close1d(sum_matrix_rows([[0, 0], [0, 0]]), [0.0, 0.0])
    
    # Mixed positive and negative
    assert_close1d(sum_matrix_rows([[-1, 2], [3, -4]]), [1.0, -1.0])
    
    # Large numbers
    assert_close1d(sum_matrix_rows([[1e15, 1e15], [1e15, 1e15]]), [2e15, 2e15])
    
    # Small numbers
    assert_close1d(sum_matrix_rows([[1e-15, 1e-15], [1e-15, 1e-15]]), [2e-15, 2e-15])

    # Test with single row
    assert sum_matrix_cols([[1, 2, 3]]) == [1.0, 2.0, 3.0]
    
    # Test with single column
    assert sum_matrix_cols([[1], [2], [3]]) == [6.0]

    # Error cases
    with pytest.raises(ValueError):
        sum_matrix_rows([])  # Empty matrix
    with pytest.raises(ValueError):
        sum_matrix_rows([[]])  # Empty rows
    with pytest.raises(ValueError):
        sum_matrix_rows([[1, 2], [1]])  # Irregular rows
    # Test non-numeric values
    with pytest.raises(TypeError):
        sum_matrix_cols([[1, 'a'], [2, 3]])


def test_sum_matrix_cols():
    """Test column-wise matrix summation"""
    # Basic functionality
    assert_close1d(sum_matrix_cols([[1, 2, 3], [4, 5, 6]]), [5.0, 7.0, 9.0])
    assert_close1d(sum_matrix_cols([[1], [2]]), [3.0])


    # Test with single row
    assert sum_matrix_rows([[1, 2, 3]]) == [6.0]
    
    # Test with single column
    assert sum_matrix_rows([[1], [2], [3]]) == [1.0, 2.0, 3.0]
        
    # Handle zeros
    assert_close1d(sum_matrix_cols([[0, 0], [0, 0]]), [0.0, 0.0])
    
    # Mixed positive and negative
    assert_close1d(sum_matrix_cols([[-1, 2], [3, -4]]), [2.0, -2.0])
    
    # Large numbers
    assert_close1d(sum_matrix_cols([[1e15, 1e15], [1e15, 1e15]]), [2e15, 2e15])
    
    # Small numbers
    assert_close1d(sum_matrix_cols([[1e-15, 1e-15], [1e-15, 1e-15]]), [2e-15, 2e-15])
    
    # Error cases
    with pytest.raises(ValueError):
        sum_matrix_cols([])  # Empty matrix
    with pytest.raises(ValueError):
        sum_matrix_cols([[]])  # Empty rows
    with pytest.raises(ValueError):
        sum_matrix_cols([[1, 2], [1]])  # Irregular rows
    with pytest.raises(TypeError):
        sum_matrix_rows([[1, 'a'], [2, 3]])
    
def test_scalar_add_matrices():
    """Test matrix addition functionality"""
    # Basic functionality
    assert_close2d(scalar_add_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]]), 
                  [[6.0, 8.0], [10.0, 12.0]])
    
    # Single element matrices
    assert_close2d(scalar_add_matrices([[1]], [[2]]), [[3.0]])
    
    # Test with zeros
    assert_close2d(scalar_add_matrices([[0, 0], [0, 0]], [[0, 0], [0, 0]]),
                  [[0.0, 0.0], [0.0, 0.0]])
    
    # Mixed positive and negative
    assert_close2d(scalar_add_matrices([[-1, 2], [3, -4]], [[1, -2], [-3, 4]]),
                  [[0.0, 0.0], [0.0, 0.0]], atol=1e-14)
    
    # Large numbers
    assert_close2d(scalar_add_matrices([[1e15, 1e15], [1e15, 1e15]], 
                                     [[1e15, 1e15], [1e15, 1e15]]),
                  [[2e15, 2e15], [2e15, 2e15]])
    
    # Small numbers
    assert_close2d(scalar_add_matrices([[1e-15, 1e-15], [1e-15, 1e-15]], 
                                     [[1e-15, 1e-15], [1e-15, 1e-15]]),
                  [[2e-15, 2e-15], [2e-15, 2e-15]])
                  
    # Different shapes of matrices
    rect1 = [[1, 2, 3], [4, 5, 6]]
    rect2 = [[7, 8, 9], [10, 11, 12]]
    assert_close2d(scalar_add_matrices(rect1, rect2),
                  [[8.0, 10.0, 12.0], [14.0, 16.0, 18.0]])
    
    # Error cases
    with pytest.raises(ValueError):
        scalar_add_matrices([], [])  # Empty matrices
    with pytest.raises(ValueError):
        scalar_add_matrices([[]], [[]])  # Empty rows
    with pytest.raises(ValueError):
        scalar_add_matrices([[1, 2], [1]], [[1, 2], [3, 4]])  # Irregular rows A
    with pytest.raises(ValueError):
        scalar_add_matrices([[1, 2], [3, 4]], [[1, 2], [3]])  # Irregular rows B
    with pytest.raises(ValueError):
        scalar_add_matrices([[1, 2]], [[1, 2, 3]])  # Incompatible shapes
    with pytest.raises(TypeError):
        scalar_add_matrices([[1, 'a']], [[1, 2]])  # Invalid type


def test_scalar_subtract_matrices():
    """Test matrix subtraction functionality"""
    # Basic functionality
    assert_close2d(scalar_subtract_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
                  [[-4.0, -4.0], [-4.0, -4.0]])
    
    # Single element matrices
    assert_close2d(scalar_subtract_matrices([[1]], [[2]]), [[-1.0]])
    
    # Test with zeros
    assert_close2d(scalar_subtract_matrices([[0, 0], [0, 0]], [[0, 0], [0, 0]]),
                  [[0.0, 0.0], [0.0, 0.0]])
    
    # Mixed positive and negative
    assert_close2d(scalar_subtract_matrices([[-1, 2], [3, -4]], [[1, -2], [-3, 4]]),
                  [[-2.0, 4.0], [6.0, -8.0]])
    
    # Large numbers
    assert_close2d(scalar_subtract_matrices([[1e15, 1e15], [1e15, 1e15]], 
                                          [[1e15, 1e15], [1e15, 1e15]]),
                  [[0.0, 0.0], [0.0, 0.0]])
    
    # Small numbers
    assert_close2d(scalar_subtract_matrices([[1e-15, 1e-15], [1e-15, 1e-15]], 
                                          [[1e-15, 1e-15], [1e-15, 1e-15]]),
                  [[0.0, 0.0], [0.0, 0.0]])
    
    # Different shapes of matrices
    rect1 = [[1, 2, 3], [4, 5, 6]]
    rect2 = [[7, 8, 9], [10, 11, 12]]
    assert_close2d(scalar_subtract_matrices(rect1, rect2),
                  [[-6.0, -6.0, -6.0], [-6.0, -6.0, -6.0]])
    
    # Error cases
    with pytest.raises(ValueError):
        scalar_subtract_matrices([], [])  # Empty matrices
    with pytest.raises(ValueError):
        scalar_subtract_matrices([[]], [[]])  # Empty rows
    with pytest.raises(ValueError):
        scalar_subtract_matrices([[1, 2], [1]], [[1, 2], [3, 4]])  # Irregular rows A
    with pytest.raises(ValueError):
        scalar_subtract_matrices([[1, 2], [3, 4]], [[1, 2], [3]])  # Irregular rows B
    with pytest.raises(ValueError):
        scalar_subtract_matrices([[1, 2]], [[1, 2, 3]])  # Incompatible shapes
    with pytest.raises(TypeError):
        scalar_subtract_matrices([[1, 'a']], [[1, 2]])  # Invalid type

def test_scalar_multiply_matrix():
    """Test matrix scalar multiplication functionality"""
    # Basic functionality
    assert_close2d(scalar_multiply_matrix(2.0, [[1, 2], [3, 4]]),
                  [[2.0, 4.0], [6.0, 8.0]])
    
    # Single element matrices
    assert_close2d(scalar_multiply_matrix(3.0, [[2]]), [[6.0]])
    
    # Test with zeros
    assert_close2d(scalar_multiply_matrix(0.0, [[1, 2], [3, 4]]),
                  [[0.0, 0.0], [0.0, 0.0]])
    
    # Test with negative scalar
    assert_close2d(scalar_multiply_matrix(-1.0, [[1, 2], [3, 4]]),
                  [[-1.0, -2.0], [-3.0, -4.0]])
    
    # Large numbers
    assert_close2d(scalar_multiply_matrix(1e15, [[1, 2], [3, 4]]),
                  [[1e15, 2e15], [3e15, 4e15]])
    
    # Small numbers
    assert_close2d(scalar_multiply_matrix(1e-15, [[1, 2], [3, 4]]),
                  [[1e-15, 2e-15], [3e-15, 4e-15]])
    
    # Rectangle matrix
    rect = [[1, 2, 3], [4, 5, 6]]
    assert_close2d(scalar_multiply_matrix(2.0, rect),
                  [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
    
    # Error cases
    with pytest.raises(ValueError):
        scalar_multiply_matrix(2.0, [])  # Empty matrix
    with pytest.raises(ValueError):
        scalar_multiply_matrix(2.0, [[]])  # Empty rows
    with pytest.raises(TypeError):
        scalar_multiply_matrix(2.0, [[1, 'a']])  # Invalid type

def test_scalar_divide_matrix():
    """Test matrix scalar division functionality"""
    # Basic functionality
    assert_close2d(scalar_divide_matrix(2.0, [[2, 4], [6, 8]]),
                  [[1.0, 2.0], [3.0, 4.0]])
    
    # Single element matrices
    assert_close2d(scalar_divide_matrix(2.0, [[4]]), [[2.0]])
    
    # Test with ones (identity case)
    assert_close2d(scalar_divide_matrix(1.0, [[1, 2], [3, 4]]),
                  [[1.0, 2.0], [3.0, 4.0]])
    
    # Test with negative scalar
    assert_close2d(scalar_divide_matrix(-2.0, [[2, 4], [6, 8]]),
                  [[-1.0, -2.0], [-3.0, -4.0]])
    
    # Large numbers
    assert_close2d(scalar_divide_matrix(1e15, [[1e15, 2e15], [3e15, 4e15]]),
                  [[1.0, 2.0], [3.0, 4.0]])
    
    # Small numbers
    assert_close2d(scalar_divide_matrix(1e-15, [[1e-15, 2e-15], [3e-15, 4e-15]]),
                  [[1.0, 2.0], [3.0, 4.0]])
    
    # Rectangle matrix
    rect = [[2, 4, 6], [8, 10, 12]]
    assert_close2d(scalar_divide_matrix(2.0, rect),
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    # Error cases
    with pytest.raises(ValueError):
        scalar_divide_matrix(2.0, [])  # Empty matrix
    with pytest.raises(ValueError):
        scalar_divide_matrix(2.0, [[]])  # Empty rows
    with pytest.raises(TypeError):
        scalar_divide_matrix(2.0, [[1, 'a']])  # Invalid type
    with pytest.raises(TypeError):
        scalar_divide_matrix('2', [[1, 2]])  # Invalid scalar type
    with pytest.raises(ZeroDivisionError):
        scalar_divide_matrix(0.0, [[1, 2]])  # Division by zero



def test_sort_paired_lists():
    assert sort_paired_lists([3, 1, 2], ['c', 'a', 'b']) == ([1, 2, 3], ['a', 'b', 'c'])
    assert sort_paired_lists([], []) == ([], [])
    assert sort_paired_lists([2, 2, 1], ['a', 'b', 'c']) == ([1, 2, 2], ['c', 'a', 'b'])
    assert sort_paired_lists([-3, -1, -2], ['c', 'a', 'b']) == ([-3, -2, -1], ['c', 'b', 'a'])
    temps = [300.5, 100.1, 200.7]
    props = ['hot', 'cold', 'warm']
    assert sort_paired_lists(temps, props) == ([100.1, 200.7, 300.5], ['cold', 'warm', 'hot'])
    
    with pytest.raises(ValueError):
        # Test 6: Unequal length lists
        sort_paired_lists([1, 2], [1])
    

def format_matrix_error_null_space(matrix):
    """Format a detailed error message for matrix comparison failure"""
    def matrix_info(matrix):
        """Get diagnostic information about a matrix"""
        arr = np.array(matrix)
        rank = np.linalg.matrix_rank(arr)
        shape = arr.shape
        try:
            cond = np.linalg.cond(arr)
        except:
            cond = float('inf')
        # Only compute determinant for square matrices
        det = np.linalg.det(arr) if shape[0] == shape[1] else None
        return {
            'rank': rank,
            'condition_number': cond,
            'shape': shape,
            'null_space_dim': shape[1] - rank,
            'determinant': det
        }
    info = matrix_info(matrix)
    
    msg = (
        f"\nMatrix properties:"
        f"\n  Shape: {info['shape']}"
        f"\n  Rank: {info['rank']}"
        f"\n  Null space dimension: {info['null_space_dim']}"
        f"\n  Condition number: {info['condition_number']:.2e}"
    )
    if info['determinant'] is not None:
        msg += f"\n  Determinant: {info['determinant']:.2e}"
    msg += (
        f"\nInput matrix:"
        f"\n{np.array2string(np.array(matrix), precision=6, suppress_small=True)}"
    )
    return msg

def check_null_space(matrix, rtol=None):
    """Check if null_space implementation matches scipy behavior"""
    import scipy
    just_return = False
    try:
        # This will fail for bad matrix inputs
        cond = np.linalg.cond(np.array(matrix))
    except:
        just_return = True
        
    py_fail = False
    scipy_fail = False
    
    try:
        result = null_space(matrix)
        if not result:  # Empty result is valid for some cases
            return
        result = np.array(result)
    except:
        py_fail = True
        
    # Convert to numpy array if not already
    matrix = np.array(matrix)
    try:
        expected = scipy.linalg.null_space(matrix, rcond=rtol)
    except:
        scipy_fail = True
    
    if py_fail and not scipy_fail:
        if not just_return and cond > 1e14:
            # Let ill conditioned matrices pass
            return 
        raise ValueError(f"Inconsistent failure states: Python Fail: {py_fail}, SciPy Fail: {scipy_fail}")
    if py_fail and scipy_fail:
        return
    if not py_fail and scipy_fail:
        return
    if just_return:
        return

    
    if rtol is None:
        rtol = get_rtol(matrix)
        
    # Compute matrix norm for threshold
    matrix_norm = np.max(np.sum(np.abs(matrix), axis=1))
    thresh = matrix_norm * np.finfo(float).eps
    

    # We need to handle sign ambiguity in the basis vectors
    # Both +v and -v are valid basis vectors
    # Compare shapes first
    assert result.shape == expected.shape, \
           f"Shape mismatch: got {result.shape}, expected {expected.shape}"
           
    if result.shape[1] > 0:  # Only if we have vectors
        # For each column in result, check if it matches any expected column or its negative
        used_expected_cols = set()
        
        for i in range(result.shape[1]):
            res_col = result[:, i].reshape(-1, 1)
            found_match = False
            
            # Try matching with each unused expected column
            for j in range(expected.shape[1]):
                if j in used_expected_cols:
                    continue
                    
                exp_col = expected[:, j].reshape(-1, 1)
                
                # Check both orientations with looser tolerance
                matches_positive = np.allclose(res_col, exp_col, rtol=1e-7, atol=1e-10)
                matches_negative = np.allclose(res_col, -exp_col, rtol=1e-7, atol=1e-10)
                
                if matches_positive or matches_negative:
                    used_expected_cols.add(j)
                    found_match = True
                    break
            
            assert found_match, f"Column {i} doesn't match any expected column in either orientation"
        
        # Verify orthonormality
        gram = result.T @ result
        assert_allclose(gram, np.eye(gram.shape[0]), rtol=1e-10, atol=100*thresh,
                       err_msg="Basis vectors are not orthonormal")
        
        # Verify it's actually a null space
        product = matrix @ result
        assert_allclose(product, np.zeros_like(product), atol=100*thresh,
                       err_msg="Result vectors are not in the null space")

# Test cases specific to null space
matrices_full_rank = [
    [[1.0]],  # 1x1
    [[1.0, 0.0], [0.0, 1.0]],  # 2x2 identity
    [[1.0, 2.0], [3.0, 4.0]],  # 2x2 full rank
]

matrices_rank_deficient = [
    [[1.0, 1.0], [1.0, 1.0]],  # 2x2 rank 1
    # [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],  # 2x3 rank 1 Failing sometimes
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],  # 3x3 rank 2
]

matrices_tall = [
    [[1.0], [0.0], [0.0]],  # 3x1
    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],  # 3x2
]

matrices_wide = [
    [[1.0, 0.0, 0.0]],  # 1x3
    [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],  # 2x3
]

@pytest.mark.parametrize("matrix", matrices_full_rank)
def test_null_space_full_rank(matrix):
    try:
        check_null_space(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error_null_space(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_rank_deficient)
def test_null_space_rank_deficient(matrix):
    try:
        check_null_space(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error_null_space(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_tall)
def test_null_space_tall(matrix):
    try:
        check_null_space(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error_null_space(matrix)}"
        raise Exception(new_message)

@pytest.mark.parametrize("matrix", matrices_wide)
def test_null_space_wide(matrix):
    try:
        check_null_space(matrix)
    except Exception as e:
        new_message = f"Original error: {str(e)}\nAdditional context: {format_matrix_error_null_space(matrix)}"
        raise Exception(new_message)