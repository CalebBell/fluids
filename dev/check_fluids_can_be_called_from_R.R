 
# Install and load reticulate
library(reticulate)

# Function to run all tests
test_fluids <- function() {
  tryCatch({
    # 1. Test import
    fluids <- import("fluids")
    print("✓ Successfully imported fluids")
    
    # 2. Test version access
    version <- py_get_attr(fluids, "__version__")
    print(paste("✓ Fluids version:", version))
    
    # 3. Test basic Reynolds number calculation
    Re <- fluids$Reynolds(V=2.5, D=0.1, rho=1000, mu=0.001)
    print(paste("✓ Reynolds number calculation successful:", Re))
    stopifnot(Re > 0)  # Basic sanity check
    
    # 4. Test friction factor calculation
    fd <- fluids$friction_factor(Re=1e5, eD=0.0001)
    print(paste("✓ Friction factor calculation successful:", fd))
    stopifnot(fd > 0 && fd < 1)  # Basic range check
    
    print("\nAll tests completed successfully!")
    
  }, error = function(e) {
    print(paste("Error occurred:", e$message))
    stop("Test suite failed")
  })
}
test_atmosphere <- function() {
  tryCatch({
    fluids <- import("fluids")
    # Test ATMOSPHERE_1976 class
    atm <- fluids$ATMOSPHERE_1976(Z=5000)
    
    # Test basic properties
    print("Testing atmosphere at 5000m elevation:")
    print(paste("✓ Temperature:", round(atm$T, 4)))
    print(paste("✓ Pressure:", round(atm$P, 4)))
    print(paste("✓ Density:", round(atm$rho, 6)))
    
    # Test derived properties
    print(paste("✓ Gravity:", round(atm$g, 6)))
    print(paste("✓ Viscosity:", formatC(atm$mu, format="e", digits=6)))
    print(paste("✓ Thermal conductivity:", round(atm$k, 6)))
    print(paste("✓ Sonic velocity:", round(atm$v_sonic, 4)))
    
    # Test static methods
    g_high <- fluids$ATMOSPHERE_1976$gravity(Z=1E5)
    print(paste("✓ High altitude gravity:", round(g_high, 6)))
    
    v_sonic <- fluids$ATMOSPHERE_1976$sonic_velocity(T=300)
    print(paste("✓ Sonic velocity at 300K:", round(v_sonic, 4)))
    
    mu_400 <- fluids$ATMOSPHERE_1976$viscosity(T=400)
    print(paste("✓ Viscosity at 400K:", formatC(mu_400, format="e", digits=6)))
    
    k_400 <- fluids$ATMOSPHERE_1976$thermal_conductivity(T=400)
    print(paste("✓ Thermal conductivity at 400K:", round(k_400, 6)))
  }, error = function(e) {
    print(paste("Error in atmosphere tests:", e$message))
    stop("Atmosphere test suite failed")
  })
}
test_tank <- function() {
  tryCatch({
    fluids <- import("fluids")
    # Test basic tank creation
    T1 <- fluids$TANK(V=10, L_over_D=0.7, sideB='conical', horizontal=FALSE)
    print("\nTesting tank calculations:")
    print(paste("✓ Tank length:", round(T1$L, 6)))
    print(paste("✓ Tank diameter:", round(T1$D, 6)))
    
    # Test ellipsoidal tank
    tank_ellip <- fluids$TANK(D=10, V=500, horizontal=FALSE, 
                             sideA='ellipsoidal', sideB='ellipsoidal', 
                             sideA_a=1, sideB_a=1)
    print(paste("✓ Ellipsoidal tank L:", round(tank_ellip$L, 6)))
    
    # Test torispherical tank
    DIN <- fluids$TANK(L=3, D=5, horizontal=FALSE, 
                       sideA='torispherical', sideB='torispherical',
                       sideA_f=1, sideA_k=0.1, sideB_f=1, sideB_k=0.1)
    print(paste("✓ Tank representation:", capture.output(print(DIN))))
    print(paste("✓ Tank max height:", round(DIN$h_max, 6)))
    print(paste("✓ Height at V=40:", round(DIN$h_from_V(40), 6)))
    print(paste("✓ Volume at h=4.1:", round(DIN$V_from_h(4.1), 5)))
    print(paste("✓ Surface area at h=2.1:", round(DIN$SA_from_h(2.1), 5)))
    
  }, error = function(e) {
    print(paste("Error in tank tests:", e$message))
    stop("Tank test suite failed")
  })
}
# Test function for Reynolds number calculations
test_reynolds <- function() {
  tryCatch({
    print("\nTesting Reynolds number calculations:")
    fluids <- import("fluids")
    
    # Test with density and viscosity
    Re1 <- fluids$Reynolds(V=2.5, D=0.25, rho=1.1613, mu=1.9E-5)
    print(paste("✓ Re (with rho, mu):", round(Re1, 4)))
    stopifnot(abs(Re1 - 38200.6579) < 0.1)
    
    # Test with kinematic viscosity
    Re2 <- fluids$Reynolds(V=2.5, D=0.25, nu=1.636e-05)
    print(paste("✓ Re (with nu):", round(Re2, 4)))
    stopifnot(abs(Re2 - 38202.934) < 0.1)
    
  }, error = function(e) {
    print(paste("Error in Reynolds tests:", e$message))
    stop("Reynolds test suite failed")
  })
}

# Test function for particle size distributions
test_psd <- function() {
  tryCatch({
    print("\nTesting particle size distribution functionality:")
    fluids <- import("fluids")

    # Create a discrete PSD
    ds <- c(240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532)
    numbers <- c(65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1)
    
    psd <- fluids$particle_size_distribution$ParticleSizeDistribution(
      ds=ds, 
      fractions=numbers, 
      order=0
    )
    print("✓ Created discrete PSD")
    
    # Test mean sizes
    d21 <- psd$mean_size(2, 1)
    print(paste("✓ Size-weighted mean diameter:", round(d21, 4)))
    stopifnot(abs(d21 - 1857.788) < 0.1)
    
    d10 <- psd$mean_size(1, 0)
    print(paste("✓ Arithmetic mean diameter:", round(d10, 4)))
    stopifnot(abs(d10 - 1459.372) < 0.1)
    
    # Test percentile calculations
    d10_percentile <- psd$dn(0.1)
    d90_percentile <- psd$dn(0.9)
    print(paste("✓ D10:", round(d10_percentile, 4)))
    print(paste("✓ D90:", round(d90_percentile, 4)))
    
    # Test probability functions
    pdf_val <- psd$pdf(1000)
    cdf_val <- psd$cdf(5000)
    print(paste("✓ PDF at 1000:", formatC(pdf_val, format="e", digits=4)))
    print(paste("✓ CDF at 5000:", round(cdf_val, 6)))
    
    # Test lognormal distribution
    psd_log <- fluids$particle_size_distribution$PSDLognormal(s=0.5, d_characteristic=5E-6)
    print("✓ Created lognormal PSD")
    
    vssa <- psd_log$vssa
    print(paste("✓ Volume specific surface area:", round(vssa, 2)))
    
    span <- psd_log$dn(0.9) - psd_log$dn(0.1)
    print(paste("✓ Span:", formatC(span, format="e", digits=4)))
    
    ratio_7525 <- psd_log$dn(0.75)/psd_log$dn(0.25)
    print(paste("✓ D75/D25 ratio:", round(ratio_7525, 6)))
    
  }, error = function(e) {
    print(paste("Error in PSD tests:", e$message))
    stop("PSD test suite failed")
  })
}
benchmark_fluids <- function() {
  fluids <- import("fluids")
  cat("\nRunning benchmarks:\n")
  
  # Benchmark friction factor calculation
  cat("\nBenchmarking friction_factor:\n")
  t1 <- system.time({
    for(i in 1:10000) {
      fluids$friction_factor(Re=1e5, eD=0.0001)
    }
  })
  cat(sprintf("Time for 10000 friction_factor calls: %.6f seconds\n", t1["elapsed"]))
  cat(sprintf("Average time per call: %.6f seconds\n", t1["elapsed"]/10000))
  
  # Benchmark tank creation
  cat("\nBenchmarking TANK creation:\n")
  t2 <- system.time({
    for(i in 1:1000) {
      fluids$TANK(L=3, D=5, horizontal=FALSE,
                  sideA="torispherical", sideB="torispherical",
                  sideA_f=1, sideA_k=0.1, sideB_f=1, sideB_k=0.1)
    }
  })
  cat(sprintf("Time for 1000 TANK creations: %.6f seconds\n", t2["elapsed"]))
  cat(sprintf("Average time per creation: %.6f seconds\n", t2["elapsed"]/1000))
}

# Run the tests
test_fluids()
test_atmosphere()
test_tank()
test_reynolds()
test_psd()
benchmark_fluids()
