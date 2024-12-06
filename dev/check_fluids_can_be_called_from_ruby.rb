require 'pycall'

# Import fluids
Fluids = PyCall.import_module('fluids')

def test_fluids
  begin
    # 1. Test basic module import
    puts "✓ Successfully imported fluids"
    puts "✓ Fluids version: #{Fluids.__version__}"
    
    # 2. Test basic Reynolds number calculation
    re = Fluids.Reynolds(V: 2.5, D: 0.1, rho: 1000, mu: 0.001)
    puts "✓ Reynolds number calculation successful: #{re}"
    raise unless re > 0
    
    # 3. Test friction factor calculation
    fd = Fluids.friction_factor(Re: 1e5, eD: 0.0001)
    puts "✓ Friction factor calculation successful: #{fd}"
    raise unless (0 < fd && fd < 1)
    
    puts "\nAll basic tests completed successfully!"
  rescue => e
    puts "Error occurred: #{e}"
    raise
  end
end

def test_atmosphere
  begin
    # Test ATMOSPHERE_1976 class
    atm = Fluids.ATMOSPHERE_1976.new(Z: 5000)
    
    puts "\nTesting atmosphere at 5000m elevation:"
    puts "✓ Temperature: #{PyCall.getattr(atm, :T).round(4)}"
    puts "✓ Pressure: #{PyCall.getattr(atm, :P).round(4)}"
    puts "✓ Density: #{PyCall.getattr(atm, :rho).round(6)}"
    
    # Test derived properties
    puts "✓ Gravity: #{PyCall.getattr(atm, :g).round(6)}"
    puts format("✓ Viscosity: %.6e", PyCall.getattr(atm, :mu))
    puts "✓ Thermal conductivity: #{PyCall.getattr(atm, :k).round(6)}"
    puts "✓ Sonic velocity: #{PyCall.getattr(atm, :v_sonic).round(4)}"
    
    # Test static methods
    g_high = Fluids.ATMOSPHERE_1976.gravity(Z: 1E5)
    puts "✓ High altitude gravity: #{g_high.round(6)}"
    
    v_sonic = Fluids.ATMOSPHERE_1976.sonic_velocity(T: 300)
    puts "✓ Sonic velocity at 300K: #{v_sonic.round(4)}"
    
    mu_400 = Fluids.ATMOSPHERE_1976.viscosity(T: 400)
    puts format("✓ Viscosity at 400K: %.6e", mu_400)
    
    k_400 = Fluids.ATMOSPHERE_1976.thermal_conductivity(T: 400)
    puts "✓ Thermal conductivity at 400K: #{k_400.round(6)}"
  rescue => e
    puts "Error in atmosphere tests: #{e}"
    raise
  end
end

def test_tank
  begin
    # Test basic tank creation
    t1 = Fluids.TANK.new(V: 10, L_over_D: 0.7, sideB: "conical", horizontal: false)
    puts "\nTesting tank calculations:"
    puts "✓ Tank length: #{t1.L.round(6)}"
    puts "✓ Tank diameter: #{t1.D.round(6)}"
    
    # Test ellipsoidal tank
    tank_ellip = Fluids.TANK.new(D: 10, V: 500, horizontal: false,
                            sideA: "ellipsoidal", sideB: "ellipsoidal",
                            sideA_a: 1, sideB_a: 1)
    puts "✓ Ellipsoidal tank L: #{tank_ellip.L.round(6)}"
    
    # Test torispherical tank
    din = Fluids.TANK.new(L: 3, D: 5, horizontal: false,
                      sideA: "torispherical", sideB: "torispherical",
                      sideA_f: 1, sideA_k: 0.1, sideB_f: 1, sideB_k: 0.1)
    
    puts "✓ Tank representation: #{din}"
    puts "✓ Tank max height: #{din.h_max.round(6)}"
    puts "✓ Height at V=40: #{din.h_from_V(40).round(6)}"
    puts "✓ Volume at h=4.1: #{din.V_from_h(4.1).round(5)}"
    puts "✓ Surface area at h=2.1: #{din.SA_from_h(2.1).round(5)}"
  rescue => e
    puts "Error in tank tests: #{e}"
    raise
  end
end

def test_reynolds
  begin
    puts "\nTesting Reynolds number calculations:"
    
    # Test with density and viscosity
    re1 = Fluids.Reynolds(V: 2.5, D: 0.25, rho: 1.1613, mu: 1.9E-5)
    puts "✓ Re (with rho, mu): #{re1.round(4)}"
    raise unless (re1 - 38200.6579).abs < 0.1
    
    # Test with kinematic viscosity
    re2 = Fluids.Reynolds(V: 2.5, D: 0.25, nu: 1.636e-05)
    puts "✓ Re (with nu): #{re2.round(4)}"
    raise unless (re2 - 38202.934).abs < 0.1
  rescue => e
    puts "Error in Reynolds tests: #{e}"
    raise
  end
end

def test_psd
  begin
    puts "\nTesting particle size distribution functionality:"
    
    # Create a discrete PSD
    ds = [240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532]
    numbers = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
    
    psd = Fluids.particle_size_distribution.ParticleSizeDistribution.new(
      ds: ds,
      fractions: numbers,
      order: 0
    )
    puts "✓ Created discrete PSD"
    
    # Test mean sizes
    d21 = psd.mean_size(2, 1)
    puts "✓ Size-weighted mean diameter: #{d21.round(4)}"
    raise unless (d21 - 1857.788).abs < 0.1
    
    d10 = psd.mean_size(1, 0)
    puts "✓ Arithmetic mean diameter: #{d10.round(4)}"
    raise unless (d10 - 1459.372).abs < 0.1
    
    # Test percentile calculations
    d10_percentile = psd.dn(0.1)
    d90_percentile = psd.dn(0.9)
    puts "✓ D10: #{d10_percentile.round(4)}"
    puts "✓ D90: #{d90_percentile.round(4)}"
    
    # Test probability functions
    pdf_val = psd.pdf(1000)
    cdf_val = psd.cdf(5000)
    puts "✓ PDF at 1000: %.4e" % pdf_val
    puts "✓ CDF at 5000: #{cdf_val.round(6)}"
    
    # Test lognormal distribution
    psd_log = Fluids.particle_size_distribution.PSDLognormal.new(s: 0.5, d_characteristic: 5E-6)
    puts "✓ Created lognormal PSD"
    
    vssa = psd_log.vssa
    puts "✓ Volume specific surface area: #{vssa.round(2)}"
    
    span = psd_log.dn(0.9) - psd_log.dn(0.1)
    puts "✓ Span: %.4e" % span
    
    ratio_7525 = psd_log.dn(0.75)/psd_log.dn(0.25)
    puts "✓ D75/D25 ratio: #{ratio_7525.round(6)}"
  rescue => e
    puts "Error in PSD tests: #{e}"
    raise
  end
end

def benchmark_fluids
  puts "\nRunning benchmarks:"
  
  # Benchmark friction factor calculation
  puts "\nBenchmarking friction_factor:"
  t1 = Time.now
  1_000_000.times do
    Fluids.friction_factor(Re: 1e5, eD: 0.0001)
  end
  elapsed = Time.now - t1
  puts "Time for 1e6 friction_factor calls: #{elapsed.round(6)} seconds"
  puts "Average time per call: #{(elapsed/1_000_000).round(6)} seconds"
  
  # Benchmark tank creation
  puts "\nBenchmarking TANK creation:"
  t2 = Time.now
  1000.times do
    Fluids.TANK.new(L: 3, D: 5, horizontal: false,
                sideA: "torispherical", sideB: "torispherical",
                sideA_f: 1, sideA_k: 0.1, sideB_f: 1, sideB_k: 0.1)
  end
  tank_elapsed = Time.now - t2
  puts "Average time per creation: #{(tank_elapsed/1000).round(6)} seconds"
end

# Run all tests
puts "Running fluids tests from Ruby..."
test_fluids
test_atmosphere
test_tank
test_reynolds
test_psd
benchmark_fluids
puts "\nAll tests completed!"
