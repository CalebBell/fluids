#!/usr/bin/perl
use strict;
use warnings;
use Inline Python => <<'END_PYTHON';
import fluids
END_PYTHON
use Time::HiRes qw(time);
use POSIX qw(ceil floor);

# Function to test basic fluids functionality
sub test_fluids {
    eval {
        # Test basic module import and version
        my $version = Inline::Python::py_eval('fluids.__version__', 0);
        print "✓ Successfully imported fluids\n";
        print "✓ Fluids version: $version\n";
        
        # Test basic Reynolds number calculation
        # Create kwargs dictionary in Python
        Inline::Python::py_eval('kwargs = {"V": 2.5, "D": 0.1, "rho": 1000, "mu": 0.001}');
        my $Re = Inline::Python::py_eval('fluids.Reynolds(**kwargs)', 0);
        print "✓ Reynolds number calculation successful: $Re\n";
        die "Invalid Reynolds number" unless $Re > 0;
        
        # Test friction factor calculation
        Inline::Python::py_eval('kwargs = {"Re": 1e5, "eD": 0.0001}');
        my $fd = Inline::Python::py_eval('fluids.friction_factor(**kwargs)', 0);
        print "✓ Friction factor calculation successful: $fd\n";
        die "Invalid friction factor" unless ($fd > 0 && $fd < 1);
        
        print "\nAll basic tests completed successfully!\n";
    };
    if ($@) {
        print "Error occurred: $@\n";
        die $@;
    }
}

# Function to test atmosphere calculations
sub test_atmosphere {
    eval {
        # Test ATMOSPHERE_1976 class
        Inline::Python::py_eval('atm = fluids.ATMOSPHERE_1976(Z=5000)');
        
        print "\nTesting atmosphere at 5000m elevation:\n";
        printf "✓ Temperature: %.4f\n", Inline::Python::py_eval('atm.T', 0);
        printf "✓ Pressure: %.4f\n", Inline::Python::py_eval('atm.P', 0);
        printf "✓ Density: %.6f\n", Inline::Python::py_eval('atm.rho', 0);
        
        # Test derived properties
        printf "✓ Gravity: %.6f\n", Inline::Python::py_eval('atm.g', 0);
        printf "✓ Viscosity: %.6e\n", Inline::Python::py_eval('atm.mu', 0);
        printf "✓ Thermal conductivity: %.6f\n", Inline::Python::py_eval('atm.k', 0);
        printf "✓ Sonic velocity: %.4f\n", Inline::Python::py_eval('atm.v_sonic', 0);
        
        # Test static methods
        my $g_high = Inline::Python::py_eval('fluids.ATMOSPHERE_1976.gravity(Z=1E5)', 0);
        printf "✓ High altitude gravity: %.6f\n", $g_high;
        
        my $v_sonic = Inline::Python::py_eval('fluids.ATMOSPHERE_1976.sonic_velocity(T=300)', 0);
        printf "✓ Sonic velocity at 300K: %.4f\n", $v_sonic;
    };
    if ($@) {
        print "Error in atmosphere tests: $@\n";
        die $@;
    }
}

# Function to test tank calculations


# Function to test tank calculations
sub test_tank {
    eval {
        # Test basic tank creation
        Inline::Python::py_eval(<<'END_PYTHON');
T1 = fluids.TANK(V=10, L_over_D=0.7, sideB='conical', horizontal=False)
END_PYTHON
        print "\nTesting tank calculations:\n";
        printf "✓ Tank length: %.6f\n", Inline::Python::py_eval('T1.L', 0);
        printf "✓ Tank diameter: %.6f\n", Inline::Python::py_eval('T1.D', 0);
        
        # Test ellipsoidal tank
        Inline::Python::py_eval(<<'END_PYTHON');
tank_ellip = fluids.TANK(D=10, V=500, horizontal=False,
                       sideA='ellipsoidal', sideB='ellipsoidal',
                       sideA_a=1, sideB_a=1)
END_PYTHON
        printf "✓ Ellipsoidal tank L: %.6f\n", Inline::Python::py_eval('tank_ellip.L', 0);
        
        # Test torispherical tank
        Inline::Python::py_eval(<<'END_PYTHON');
DIN = fluids.TANK(L=3, D=5, horizontal=False,
                sideA='torispherical', sideB='torispherical',
                sideA_f=1, sideA_k=0.1, sideB_f=1, sideB_k=0.1)
END_PYTHON
        printf "✓ Tank max height: %.6f\n", Inline::Python::py_eval('DIN.h_max', 0);
        printf "✓ Height at V=40: %.6f\n", Inline::Python::py_eval('DIN.h_from_V(40)', 0);
        printf "✓ Volume at h=4.1: %.5f\n", Inline::Python::py_eval('DIN.V_from_h(4.1)', 0);
        printf "✓ Surface area at h=2.1: %.5f\n", Inline::Python::py_eval('DIN.SA_from_h(2.1)', 0);
    };
    if ($@) {
        print "Error in tank tests: $@\n";
        die $@;
    }
}
# Function to benchmark fluids operations


# Function to benchmark fluids operations
sub benchmark_fluids {
    print "\nRunning benchmarks:\n";
    
    # Benchmark friction factor calculation
    print "\nBenchmarking friction_factor:\n";
    my $start_time = time();
    Inline::Python::py_eval(<<'END_PYTHON');
for i in range(10000):
    fluids.friction_factor(Re=1e5, eD=0.0001)
END_PYTHON
    my $duration = time() - $start_time;
    printf "Time for 10000 friction_factor calls: %.6f seconds\n", $duration;
    printf "Average time per call: %.6f seconds\n", $duration/10000;
    
    # Benchmark tank creation
    print "\nBenchmarking TANK creation:\n";
    $start_time = time();
    Inline::Python::py_eval(<<'END_PYTHON');
for i in range(1000):
    fluids.TANK(L=3, D=5, horizontal=False,
               sideA='torispherical', sideB='torispherical',
               sideA_f=1, sideA_k=0.1, sideB_f=1, sideB_k=0.1)
END_PYTHON
    $duration = time() - $start_time;
    printf "Time for 1000 TANK creations: %.6f seconds\n", $duration;
    printf "Average time per creation: %.6f seconds\n", $duration/1000;
}

# Run all tests
print "Running fluids tests from Perl...\n";
test_fluids();
test_atmosphere();
test_tank();
benchmark_fluids();
print "\nAll tests completed!\n";
