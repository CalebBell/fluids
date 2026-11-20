"""Independent geometry regressions for the oblique-conical guppy head."""
import numpy as np
import pytest

from fluids import A_partial_circle, SA_guppy_head, SA_partial_horiz_guppy_head, TANK


@pytest.mark.parametrize("h", [0.5, 1.0, 1.5])
def test_guppy_equal_extension_radius_partial_area(h):
    from scipy.integrate import dblquad

    # Integrate the surface gradient directly, independently of the elliptic
    # inner integral used in production. D=2, R=a=1.
    area = 2.0*dblquad(
        lambda y, x: np.sqrt(1.0 + 0.25*(1.0-y*y/(1.0-x)**2)**2
                             + y*y/(1.0-x)**2),
        -1.0, h-1.0, lambda x: 0.0, lambda x: np.sqrt(1.0-x*x))[0]
    assert SA_partial_horiz_guppy_head(2.0, 1.0, h) == pytest.approx(area, rel=1e-8)


@pytest.mark.parametrize("a", [0.2, 0.6, 0.999, 1.0, 1.001, 2.0])
def test_guppy_area_against_triangle_mesh(a):
    # Circular base in x=0; the apex is level with the top of the circle.
    # Triangle cross products provide an independent surface-area oracle.
    theta = np.linspace(0.0, 2.0*np.pi, 20001)
    base = np.column_stack((np.zeros_like(theta), np.cos(theta), np.sin(theta)))
    apex = np.array([a, 0.0, 1.0])
    mesh_area = 0.5*np.linalg.norm(
        np.cross(base[:-1] - apex, base[1:] - apex), axis=1).sum()
    assert SA_guppy_head(2.0, a) == pytest.approx(mesh_area, rel=3e-8)
    assert SA_partial_horiz_guppy_head(2.0, a, 2.0) == pytest.approx(mesh_area, rel=3e-8)


@pytest.mark.parametrize("a", [0.0, 0.6, 1.0, 2.0])
def test_guppy_tank_full_wetted_area(a):
    tank = TANK(D=2.0, L=5.0, horizontal=True,
                sideA="guppy", sideA_a=a, sideB="same")
    assert tank.SA_from_h(tank.h_max) == pytest.approx(tank.A, rel=1e-10)
    assert SA_partial_horiz_guppy_head(2.0, a, -0.1) == 0.0
    assert SA_partial_horiz_guppy_head(2.0, a, 3.0) == pytest.approx(SA_guppy_head(2.0, a))


@pytest.mark.parametrize("h", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_guppy_zero_extension_is_flat_head(h):
    assert SA_partial_horiz_guppy_head(2.0, 0.0, h) == pytest.approx(A_partial_circle(2.0, h))
    assert SA_guppy_head(2.0, 0.0) == pytest.approx(np.pi)
