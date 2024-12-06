import nimpy
import times
import strformat
import math

# Initialize Python and import fluids
let fluids = pyImport("fluids")

proc test_fluids() =
  try:
    # Test basic module import
    echo "Running fluids tests from Nim..."
    echo "✓ Successfully imported fluids"
    echo &"✓ Fluids version: {fluids.getAttr(\"__version__\").to(string)}"
    
    # Test basic Reynolds number calculation
    let Re = fluids.Reynolds(V=2.5, D=0.1, rho=1000, mu=0.001).to(float)
    echo &"✓ Reynolds number calculation successful: {Re}"
    assert Re > 0
    
    # Test friction factor calculation
    let fd = fluids.friction_factor(Re=1e5, eD=0.0001).to(float)
    echo &"✓ Friction factor calculation successful: {fd}"
    assert 0 < fd and fd < 1
    
    echo "\nAll basic tests completed successfully!"
    
  except:
    echo "Error in fluids tests: ", getCurrentExceptionMsg()
    raise

proc test_atmosphere() =
  try:
    # Test ATMOSPHERE_1976 class
    let atm = fluids.ATMOSPHERE_1976(Z=5000)
    
    echo "\nTesting atmosphere at 5000m elevation:"
    echo &"✓ Temperature: {atm.T.to(float):.4f}"
    echo &"✓ Pressure: {atm.P.to(float):.4f}"
    echo &"✓ Density: {atm.rho.to(float):.6f}"
    
    # Test derived properties
    echo &"✓ Gravity: {atm.g.to(float):.6f}"
    echo &"✓ Viscosity: {atm.mu.to(float):.6e}"
    echo &"✓ Thermal conductivity: {atm.k.to(float):.6f}"
    echo &"✓ Sonic velocity: {atm.v_sonic.to(float):.4f}"
    
    # Test static methods
    let g_high = fluids.ATMOSPHERE_1976.gravity(Z=1E5).to(float)
    echo &"✓ High altitude gravity: {g_high:.6f}"
    
    let v_sonic = fluids.ATMOSPHERE_1976.sonic_velocity(T=300).to(float)
    echo &"✓ Sonic velocity at 300K: {v_sonic:.4f}"
    
    let mu_400 = fluids.ATMOSPHERE_1976.viscosity(T=400).to(float)
    echo &"✓ Viscosity at 400K: {mu_400:.6e}"
    
    let k_400 = fluids.ATMOSPHERE_1976.thermal_conductivity(T=400).to(float)
    echo &"✓ Thermal conductivity at 400K: {k_400:.6f}"
    
  except:
    echo "Error in atmosphere tests: ", getCurrentExceptionMsg()
    raise

proc test_tank() =
  try:
    # Test basic tank creation
    let T1 = fluids.TANK(V=10, L_over_D=0.7, sideB="conical", horizontal=false)
    echo "\nTesting tank calculations:"
    echo &"✓ Tank length: {T1.L.to(float):.6f}"
    echo &"✓ Tank diameter: {T1.D.to(float):.6f}"
    
    # Test ellipsoidal tank
    let tank_ellip = fluids.TANK(
      D=10, V=500, horizontal=false,
      sideA="ellipsoidal", sideB="ellipsoidal",
      sideA_a=1, sideB_a=1
    )
    echo &"✓ Ellipsoidal tank L: {tank_ellip.L.to(float):.6f}"
    
    # Test torispherical tank
    let DIN = fluids.TANK(
      L=3, D=5, horizontal=false,
      sideA="torispherical", sideB="torispherical",
      sideA_f=1, sideA_k=0.1, sideB_f=1, sideB_k=0.1
    )
    
    echo &"✓ Tank representation: {$DIN}"
    echo &"✓ Tank max height: {DIN.h_max.to(float):.6f}"
    echo &"✓ Height at V=40: {DIN.h_from_V(40).to(float):.6f}"
    echo &"✓ Volume at h=4.1: {DIN.V_from_h(4.1).to(float):.5f}"
    echo &"✓ Surface area at h=2.1: {DIN.SA_from_h(2.1).to(float):.5f}"
    
  except:
    echo "Error in tank tests: ", getCurrentExceptionMsg()
    raise

proc test_reynolds() =
  try:
    echo "\nTesting Reynolds number calculations:"
    
    # Test with density and viscosity
    let Re1 = fluids.Reynolds(V=2.5, D=0.25, rho=1.1613, mu=1.9E-5).to(float)
    echo &"✓ Re (with rho, mu): {Re1:.4f}"
    assert abs(Re1 - 38200.6579) < 0.1
    
    # Test with kinematic viscosity
    let Re2 = fluids.Reynolds(V=2.5, D=0.25, nu=1.636e-05).to(float)
    echo &"✓ Re (with nu): {Re2:.4f}"
    assert abs(Re2 - 38202.934) < 0.1
    
  except:
    echo "Error in Reynolds tests: ", getCurrentExceptionMsg()
    raise

proc test_psd() =
  try:
    echo "\nTesting particle size distribution functionality:"
    
    # Create a discrete PSD
    let ds = @[240.0, 360.0, 450.0, 562.5, 703.0, 878.0, 1097.0, 1371.0,
               1713.0, 2141.0, 2676.0, 3345.0, 4181.0, 5226.0, 6532.0]
    let numbers = @[65.0, 119.0, 232.0, 410.0, 629.0, 849.0, 990.0, 981.0,
                   825.0, 579.0, 297.0, 111.0, 21.0, 1.0]
    
    let psd = fluids.particle_size_distribution.ParticleSizeDistribution(
      ds=ds,
      fractions=numbers,
      order=0
    )
    echo "✓ Created discrete PSD"
    
    # Test mean sizes
    let d21 = psd.mean_size(2, 1).to(float)
    echo &"✓ Size-weighted mean diameter: {d21:.4f}"
    assert abs(d21 - 1857.788) < 0.1
    
    let d10 = psd.mean_size(1, 0).to(float)
    echo &"✓ Arithmetic mean diameter: {d10:.4f}"
    assert abs(d10 - 1459.372) < 0.1
    
    # Test percentile calculations
    let d10_percentile = psd.dn(0.1).to(float)
    let d90_percentile = psd.dn(0.9).to(float)
    echo &"✓ D10: {d10_percentile:.4f}"
    echo &"✓ D90: {d90_percentile:.4f}"
    
    # Test probability functions
    let pdf_val = psd.pdf(1000).to(float)
    let cdf_val = psd.cdf(5000).to(float)
    echo &"✓ PDF at 1000: {pdf_val:.4e}"
    echo &"✓ CDF at 5000: {cdf_val:.6f}"
    
    # Test lognormal distribution
    let psd_log = fluids.particle_size_distribution.PSDLognormal(s=0.5, d_characteristic=5E-6)
    echo "✓ Created lognormal PSD"
    
    let vssa = psd_log.vssa.to(float)
    echo &"✓ Volume specific surface area: {vssa:.2f}"
    
    let span = psd_log.dn(0.9).to(float) - psd_log.dn(0.1).to(float)
    echo &"✓ Span: {span:.4e}"
    
    let ratio_7525 = psd_log.dn(0.75).to(float) / psd_log.dn(0.25).to(float)
    echo &"✓ D75/D25 ratio: {ratio_7525:.6f}"
    
  except:
    echo "Error in PSD tests: ", getCurrentExceptionMsg()
    raise

proc benchmark_fluids() =
  echo "\nRunning benchmarks:"
  
  # Benchmark friction factor calculation
  echo "\nBenchmarking friction_factor:"
  let start = epochTime()
  
  for i in 1..1_000_000:
    discard fluids.friction_factor(Re=1e5, eD=0.0001)
  
  let duration = (epochTime() - start) * 1_000_000  # Convert to microseconds
  echo &"Time for 1e6 friction_factor calls: {duration.int} microseconds"
  echo &"Average time per call: {duration / 1_000_000:.6f} microseconds"

  # Benchmark tank creation
  echo "\nBenchmarking TANK creation:"
  let tank_start = epochTime()
  
  for i in 1..1000:
    discard fluids.TANK(
      L=3, D=5, horizontal=false,
      sideA="torispherical", sideB="torispherical",
      sideA_f=1, sideA_k=0.1,
      sideB_f=1, sideB_k=0.1
    )
  
  let tank_duration = (epochTime() - tank_start) * 1_000_000  # Convert to microseconds
  echo &"Average time per tank creation: {tank_duration / 1000:.6f} microseconds"

when isMainModule:
  try:
    test_fluids()
    test_atmosphere()
    test_tank()
    test_reynolds()
    test_psd()
    benchmark_fluids()
    echo "\nAll tests completed!"
  except:
    echo "Fatal error: ", getCurrentExceptionMsg()
    quit(1)
