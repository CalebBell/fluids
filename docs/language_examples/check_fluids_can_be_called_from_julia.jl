using PyCall
using Printf
const fluids = pyimport("fluids")

function test_fluids()
    try
        # 1. Test basic module import
        println("✓ Successfully imported fluids")
        println("✓ Fluids version: ", fluids.__version__)
        
        # 2. Test basic Reynolds number calculation
        Re = fluids.Reynolds(V=2.5, D=0.1, rho=1000, mu=0.001)
        println("✓ Reynolds number calculation successful: ", Re)
        @assert Re > 0
        
        # 3. Test friction factor calculation
        fd = fluids.friction_factor(Re=1e5, eD=0.0001)
        println("✓ Friction factor calculation successful: ", fd)
        @assert 0 < fd < 1
                
        println("\nAll basic tests completed successfully!")
        
    catch e
        println("Error occurred: ", e)
        rethrow(e)
    end
end

function test_atmosphere()
    try
        # Test ATMOSPHERE_1976 class
        atm = fluids.ATMOSPHERE_1976(Z=5000)
        
        println("\nTesting atmosphere at 5000m elevation:")
        println("✓ Temperature: ", round(atm.T, digits=4))
        println("✓ Pressure: ", round(atm.P, digits=4))
        println("✓ Density: ", round(atm.rho, digits=6))
        
        # Test derived properties
        println("✓ Gravity: ", round(atm.g, digits=6))
        println("✓ Viscosity: ", @sprintf("%.6e", atm.mu))
        println("✓ Thermal conductivity: ", round(atm.k, digits=6))
        println("✓ Sonic velocity: ", round(atm.v_sonic, digits=4))
        
        # Test static methods
        g_high = fluids.ATMOSPHERE_1976.gravity(Z=1E5)
        println("✓ High altitude gravity: ", round(g_high, digits=6))
        
        v_sonic = fluids.ATMOSPHERE_1976.sonic_velocity(T=300)
        println("✓ Sonic velocity at 300K: ", round(v_sonic, digits=4))
        
        mu_400 = fluids.ATMOSPHERE_1976.viscosity(T=400)
        println("✓ Viscosity at 400K: ", @sprintf("%.6e", mu_400))
        
        k_400 = fluids.ATMOSPHERE_1976.thermal_conductivity(T=400)
        println("✓ Thermal conductivity at 400K: ", round(k_400, digits=6))
        
    catch e
        println("Error in atmosphere tests: ", e)
        rethrow(e)
    end
end

function test_tank()
    try
        # Test basic tank creation
        T1 = fluids.TANK(V=10, L_over_D=0.7, sideB="conical", horizontal=false)
        println("\nTesting tank calculations:")
        println("✓ Tank length: ", round(T1.L, digits=6))
        println("✓ Tank diameter: ", round(T1.D, digits=6))
        
        # Test ellipsoidal tank
        tank_ellip = fluids.TANK(D=10, V=500, horizontal=false,
                                sideA="ellipsoidal", sideB="ellipsoidal",
                                sideA_a=1, sideB_a=1)
        println("✓ Ellipsoidal tank L: ", round(tank_ellip.L, digits=6))
        
        # Test torispherical tank
        DIN = fluids.TANK(L=3, D=5, horizontal=false,
                         sideA="torispherical", sideB="torispherical",
                         sideA_f=1, sideA_k=0.1, sideB_f=1, sideB_k=0.1)
        
        println("✓ Tank representation: ", string(DIN))
        println("✓ Tank max height: ", round(DIN.h_max, digits=6))
        println("✓ Height at V=40: ", round(DIN.h_from_V(40), digits=6))
        println("✓ Volume at h=4.1: ", round(DIN.V_from_h(4.1), digits=5))
        println("✓ Surface area at h=2.1: ", round(DIN.SA_from_h(2.1), digits=5))
        
    catch e
        println("Error in tank tests: ", e)
        rethrow(e)
    end
end

function test_reynolds()
    try
        println("\nTesting Reynolds number calculations:")
        
        # Test with density and viscosity
        Re1 = fluids.Reynolds(V=2.5, D=0.25, rho=1.1613, mu=1.9E-5)
        println("✓ Re (with rho, mu): ", round(Re1, digits=4))
        @assert abs(Re1 - 38200.6579) < 0.1
        
        # Test with kinematic viscosity
        Re2 = fluids.Reynolds(V=2.5, D=0.25, nu=1.636e-05)
        println("✓ Re (with nu): ", round(Re2, digits=4))
        @assert abs(Re2 - 38202.934) < 0.1
        
    catch e
        println("Error in Reynolds tests: ", e)
        rethrow(e)
    end
end

function test_psd()
    try
        println("\nTesting particle size distribution functionality:")
        
        # Create a discrete PSD
        ds = [240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532]
        numbers = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
        
        psd = fluids.particle_size_distribution.ParticleSizeDistribution(
            ds=ds,
            fractions=numbers,
            order=0
        )
        println("✓ Created discrete PSD")
        
        # Test mean sizes
        d21 = psd.mean_size(2, 1)
        println("✓ Size-weighted mean diameter: ", round(d21, digits=4))
        @assert abs(d21 - 1857.788) < 0.1
        
        d10 = psd.mean_size(1, 0)
        println("✓ Arithmetic mean diameter: ", round(d10, digits=4))
        @assert abs(d10 - 1459.372) < 0.1
        
        # Test percentile calculations
        d10_percentile = psd.dn(0.1)
        d90_percentile = psd.dn(0.9)
        println("✓ D10: ", round(d10_percentile, digits=4))
        println("✓ D90: ", round(d90_percentile, digits=4))
        
        # Test probability functions
        pdf_val = psd.pdf(1000)
        cdf_val = psd.cdf(5000)
        println("✓ PDF at 1000: ", @sprintf("%.4e", pdf_val))
        println("✓ CDF at 5000: ", round(cdf_val, digits=6))
        
        # Test lognormal distribution
        psd_log = fluids.particle_size_distribution.PSDLognormal(s=0.5, d_characteristic=5E-6)
        println("✓ Created lognormal PSD")
        
        vssa = psd_log.vssa
        println("✓ Volume specific surface area: ", round(vssa, digits=2))
        
        span = psd_log.dn(0.9) - psd_log.dn(0.1)
        println("✓ Span: ", @sprintf("%.4e", span))
        
        ratio_7525 = psd_log.dn(0.75)/psd_log.dn(0.25)
        println("✓ D75/D25 ratio: ", round(ratio_7525, digits=6))
        
    catch e
        println("Error in PSD tests: ", e)
        rethrow(e)
    end
end
function benchmark_fluids()
    println("\nRunning benchmarks:")
    
    # Benchmark friction factor calculation
    println("\nBenchmarking friction_factor:")
    t1 = @elapsed for i in 1:1000000
        fluids.friction_factor(Re=1e5, eD=0.0001)
    end
    println("Time for 1e6 friction_factor calls: ", round(t1, digits=6), " seconds")
    println("Average time per call: ", round(t1/1000000, digits=6), " seconds")
    
    # Benchmark tank creation
    println("\nBenchmarking TANK creation:")
    t2 = @elapsed for i in 1:1000
        fluids.TANK(L=3, D=5, horizontal=false,
                   sideA="torispherical", sideB="torispherical",
                   sideA_f=1, sideA_k=0.1, sideB_f=1, sideB_k=0.1)
    end
    println("Average time per creation: ", round(t2/1000, digits=6), " seconds")
    
end

# Add this line at the end of your script to run the benchmark
# Run all tests
println("Running fluids tests from Julia...")
test_fluids()
test_atmosphere()
test_tank()
test_reynolds()
test_psd()
benchmark_fluids()
println("\nAll tests completed!") 
