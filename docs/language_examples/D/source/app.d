import std.stdio;
import pyd.pyd;
import pyd.embedded;
import std.format;
import std.math;
import std.array;
import core.time;
import std.datetime.stopwatch;

// Initialize Python before using it
shared static this() {
    py_init();
}

// Helper to run Python expressions and handle errors
auto pyEval(T)(string expr, string namespace = "") {
    try {
        return py_eval!T(expr, namespace);
    } catch (Exception e) {
        writeln("Error evaluating: ", expr);
        writeln("Error: ", e.msg);
        throw e;
    }
}

void testFluids() {
    writeln("\nTesting basic fluids functionality...");
    
    // Import fluids and create a PydObject reference
    PydObject fluidsModule = py_import("fluids");
    writeln("✓ Successfully imported fluids");
    
    // Get version using attribute access
    string fluidsVersion = fluidsModule.__version__.to_d!string();
    writeln("✓ Fluids version: ", fluidsVersion);
    
    double Re = pyEval!double("Reynolds(V=2.5, D=0.1, rho=1000, mu=0.001)", "fluids");
    writeln("✓ Reynolds number calculation: ", Re);
    assert(Re > 0);
    
    // Test friction factor
    double fd = pyEval!double("friction_factor(Re=1e5, eD=0.0001)", "fluids");
    writeln("✓ Friction factor: ", fd);
    assert(0 < fd && fd < 1);
}

void testAtmosphere() {
    writeln("\nTesting atmosphere calculations...");
    PydObject fluidsModule = py_import("fluids");    
    PydObject atm = py_eval("ATMOSPHERE_1976(5000.0)", "fluids");
    
    // Get properties
    double temp = atm.T.to_d!double();
    double pressure = atm.P.to_d!double();
    double density = atm.rho.to_d!double();
    double gravity = atm.g.to_d!double();
    double viscosity = atm.mu.to_d!double();
    double thermalConductivity = atm.k.to_d!double();
    double sonicVelocity = atm.v_sonic.to_d!double();
    
    writeln("At 5000m elevation:");
    writefln("✓ Temperature: %.4f", temp);
    writefln("✓ Pressure: %.4f", pressure);
    writefln("✓ Density: %.6f", density);
    writefln("✓ Gravity: %.6f", gravity);
    writefln("✓ Viscosity: %.6e", viscosity);
    writefln("✓ Thermal conductivity: %.6f", thermalConductivity);
    writefln("✓ Sonic velocity: %.4f", sonicVelocity);
    
    // Test static methods
    double gHigh = pyEval!double("ATMOSPHERE_1976.gravity(1E5)", "fluids");
    writefln("✓ High altitude gravity: %.6f", gHigh);
    
    double vSonic = pyEval!double("ATMOSPHERE_1976.sonic_velocity(300)", "fluids");
    writefln("✓ Sonic velocity at 300K: %.4f", vSonic);
    
    double mu400 = pyEval!double("ATMOSPHERE_1976.viscosity(400)", "fluids");
    writefln("✓ Viscosity at 400K: %.6e", mu400);
    
    double k400 = pyEval!double("ATMOSPHERE_1976.thermal_conductivity(400)", "fluids");
    writefln("✓ Thermal conductivity at 400K: %.6f", k400);
}

void testTank() {
    writeln("\nTesting tank calculations...");
    PydObject fluidsModule = py_import("fluids");
    
    // Test basic tank creation
    PydObject T1 = py_eval("TANK(V=10, L_over_D=0.7, sideB='conical', horizontal=False)", "fluids");
    writefln("✓ Tank length: %.6f", T1.L.to_d!double());
    writefln("✓ Tank diameter: %.6f", T1.D.to_d!double());
    
    // Test ellipsoidal tank
    PydObject tankEllip = py_eval(
        "TANK(D=10, V=500, horizontal=False, sideA='ellipsoidal', sideB='ellipsoidal', sideA_a=1, sideB_a=1)",
        "fluids"
    );
    writefln("✓ Ellipsoidal tank L: %.6f", tankEllip.L.to_d!double());
    
    // Test torispherical tank
    PydObject DIN = py_eval(
        "TANK(L=3, D=5, horizontal=False, sideA='torispherical', sideB='torispherical', " ~
        "sideA_f=1, sideA_k=0.1, sideB_f=1, sideB_k=0.1)",
        "fluids"
    );
    
    writeln("✓ Tank representation: ", DIN.toString());
    writefln("✓ Height at V=40: %.6f", DIN.method("h_from_V", 40.0).to_d!double());
    writefln("✓ Volume at h=4.1: %.5f", DIN.method("V_from_h", 4.1).to_d!double());
    writefln("✓ Surface area at h=2.1: %.5f", DIN.method("SA_from_h", 2.1).to_d!double());
    
}

void testReynolds() {
    writeln("\nTesting Reynolds number calculations:");
    
    // Test with density and viscosity
    double Re1 = pyEval!double("Reynolds(V=2.5, D=0.25, rho=1.1613, mu=1.9E-5)", "fluids");
    writefln("✓ Re (with rho, mu): %.4f", Re1);
    assert(abs(Re1 - 38200.6579) < 0.1);
    
    // Test with kinematic viscosity
    double Re2 = pyEval!double("Reynolds(V=2.5, D=0.25, nu=1.636e-05)", "fluids");
    writefln("✓ Re (with nu): %.4f", Re2);
    assert(abs(Re2 - 38202.934) < 0.1);
}

void testPSD() {
    writeln("\nTesting particle size distribution functionality:");
    
    // Create arrays for discrete PSD
    double[] ds = [240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532];
    double[] numbers = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1];
    
    // Create Python lists from D arrays
    string dsStr = format("[%(%s,%)]", ds);
    string numbersStr = format("[%(%s,%)]", numbers);
    
    // Create the PSD object
    PydObject particleDist = py_eval("particle_size_distribution", "fluids");
    PydObject psd = particleDist.ParticleSizeDistribution(ds, numbers, 0);
    writeln("✓ Created discrete PSD");
    
    // Test mean sizes
    double d21 = psd.method("mean_size", 2, 1).to_d!double();
    writefln("✓ Size-weighted mean diameter: %.4f", d21);
    
    double d10 = psd.method("mean_size", 1, 0).to_d!double();
    writefln("✓ Arithmetic mean diameter: %.4f", d10);
    
    // Test percentile calculations
    double d10Percentile = psd.method("dn", 0.1).to_d!double();
    double d90Percentile = psd.method("dn", 0.9).to_d!double();
    writefln("✓ D10: %.4f", d10Percentile);
    writefln("✓ D90: %.4f", d90Percentile);
    
    // Test probability functions
    double pdfVal = psd.method("pdf", 1000.0).to_d!double();
    double cdfVal = psd.method("cdf", 5000.0).to_d!double();
    writefln("✓ PDF at 1000: %.4e", pdfVal);
    writefln("✓ CDF at 5000: %.6f", cdfVal);
    
    // Test lognormal distribution
    PydObject psdLog = particleDist.PSDLognormal(0.5, 5e-6);
    writeln("✓ Created lognormal PSD");
    
    double vssa = psdLog.vssa.to_d!double();
    writefln("✓ Volume specific surface area: %.2f", vssa);
    
    // Calculate span using individual method calls
    double dn90 = psdLog.method("dn", 0.9).to_d!double();
    double dn10 = psdLog.method("dn", 0.1).to_d!double();
    double span = dn90 - dn10;
    writefln("✓ Span: %.4e", span);
    
    // Calculate ratio using individual method calls
    double dn75 = psdLog.method("dn", 0.75).to_d!double();
    double dn25 = psdLog.method("dn", 0.25).to_d!double();
    double ratio7525 = dn75 / dn25;
    writefln("✓ D75/D25 ratio: %.6f", ratio7525);
}

void benchmarkFluids() {
    writeln("\nRunning benchmarks:");
    
    // Benchmark friction factor calculation
    writeln("\nBenchmarking friction_factor:");
    auto sw = StopWatch(AutoStart.yes);
    
    foreach (i; 0..10000) {
        pyEval!double("friction_factor(Re=1e5, eD=0.0001)", "fluids");
    }
    
    sw.stop();
    double elapsed = sw.peek().total!"usecs" / 1_000_000.0;
    writefln("Time for 1e4 friction_factor calls: %.6f seconds", elapsed);
    writefln("Average time per call: %.6f seconds", elapsed/10000);
    
    // Benchmark tank creation
    writeln("\nBenchmarking TANK creation:");
    sw.reset();
    sw.start();
    
    foreach (i; 0..1_000) {
        py_eval(
            "TANK(L=3, D=5, horizontal=False, sideA='torispherical', sideB='torispherical', " ~
            "sideA_f=1, sideA_k=0.1, sideB_f=1, sideB_k=0.1)",
            "fluids"
        );
    }
    
    sw.stop();
    elapsed = sw.peek().total!"usecs" / 1_000_000.0;
    writefln("Average time per creation: %.6f seconds", elapsed/1_000);
}

void main() {
    try {
        writeln("Running fluids tests from D...");
        testFluids();
        testAtmosphere();
        testTank();
        testReynolds();
        testPSD();
        benchmarkFluids();
        writeln("\nAll tests completed!");
    } catch (Exception e) {
        writeln("Test failed with error: ", e.msg);
    }
}
