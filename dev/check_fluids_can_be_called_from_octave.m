% Test suite for fluids library in Octave
% Requires pythonic package or oct2py

pkg load pythonic  % Load Python interface

function test_fluids()
    try
        % Test basic module import
        fluids = py.importlib.import_module('fluids');
        printf('✓ Successfully imported fluids\n');
        printf('✓ Fluids version: %s\n', char(fluids.__version__));
        
        % Test basic Reynolds number calculation
        Re = double(fluids.Reynolds(pyargs('V', 2.5, 'D', 0.1, 'rho', 1000, 'mu', 0.001)));
        printf('✓ Reynolds number calculation successful: %f\n', Re);
        assert(Re > 0);
        
        % Test friction factor calculation
        fd = double(fluids.friction_factor(pyargs('Re', 1e5, 'eD', 0.0001)));
        printf('✓ Friction factor calculation successful: %f\n', fd);
        assert(fd > 0 && fd < 1);
        
        printf('\nAll basic tests completed successfully!\n');
        
    catch err
        printf('Error occurred: %s\n', err.message);
        rethrow(err);
    end
end

function test_atmosphere()
    try
        fluids = py.importlib.import_module('fluids');
        % Test ATMOSPHERE_1976 class
        atm = fluids.ATMOSPHERE_1976(pyargs('Z', 5000));
        
        printf('\nTesting atmosphere at 5000m elevation:\n');
        printf('✓ Temperature: %.4f\n', double(atm.T));
        printf('✓ Pressure: %.4f\n', double(atm.P));
        printf('✓ Density: %.6f\n', double(atm.rho));
        
        % Test derived properties
        printf('✓ Gravity: %.6f\n', double(atm.g));
        printf('✓ Viscosity: %.6e\n', double(atm.mu));
        printf('✓ Thermal conductivity: %.6f\n', double(atm.k));
        printf('✓ Sonic velocity: %.4f\n', double(atm.v_sonic));
        
        % Test static methods
        g_high = double(fluids.ATMOSPHERE_1976.gravity(pyargs('Z', 1E5)));
        printf('✓ High altitude gravity: %.6f\n', g_high);
        
        v_sonic = double(fluids.ATMOSPHERE_1976.sonic_velocity(pyargs('T', 300)));
        printf('✓ Sonic velocity at 300K: %.4f\n', v_sonic);
        
    catch err
        printf('Error in atmosphere tests: %s\n', err.message);
        rethrow(err);
    end
end

function test_tank()
    try
        fluids = py.importlib.import_module('fluids');
        % Test basic tank creation
        T1 = fluids.TANK(pyargs('V', 10, 'L_over_D', 0.7, 'sideB', 'conical', 'horizontal', false));
        printf('\nTesting tank calculations:\n');
        printf('✓ Tank length: %.6f\n', double(T1.L));
        printf('✓ Tank diameter: %.6f\n', double(T1.D));
        
        % Test ellipsoidal tank
        tank_ellip = fluids.TANK(pyargs('D', 10, 'V', 500, 'horizontal', false, ...
                                      'sideA', 'ellipsoidal', 'sideB', 'ellipsoidal', ...
                                      'sideA_a', 1, 'sideB_a', 1));
        printf('✓ Ellipsoidal tank L: %.6f\n', double(tank_ellip.L));
        
        % Test torispherical tank
        DIN = fluids.TANK(pyargs('L', 3, 'D', 5, 'horizontal', false, ...
                                'sideA', 'torispherical', 'sideB', 'torispherical', ...
                                'sideA_f', 1, 'sideA_k', 0.1, 'sideB_f', 1, 'sideB_k', 0.1));
        
        printf('✓ Tank max height: %.6f\n', double(DIN.h_max));
        printf('✓ Height at V=40: %.6f\n', double(DIN.h_from_V(40)));
        printf('✓ Volume at h=4.1: %.5f\n', double(DIN.V_from_h(4.1)));
        printf('✓ Surface area at h=2.1: %.5f\n', double(DIN.SA_from_h(2.1)));
        
    catch err
        printf('Error in tank tests: %s\n', err.message);
        rethrow(err);
    end
end

function benchmark_fluids()
    fluids = py.importlib.import_module('fluids');
    printf('\nRunning benchmarks:\n');
    
    % Benchmark friction factor calculation
    printf('\nBenchmarking friction_factor:\n');
    tic;
    for i = 1:1000
        fluids.friction_factor(pyargs('Re', 1e5, 'eD', 0.0001));
    end
    t1 = toc;
    printf('Time for 1000 friction_factor calls: %.6f seconds\n', t1);
    printf('Average time per call: %.6f seconds\n', t1/1000);
    
    % Benchmark tank creation
    printf('\nBenchmarking TANK creation:\n');
    tic;
    for i = 1:1000
        fluids.TANK(pyargs('L', 3, 'D', 5, 'horizontal', false, ...
                          'sideA', 'torispherical', 'sideB', 'torispherical', ...
                          'sideA_f', 1, 'sideA_k', 0.1, 'sideB_f', 1, 'sideB_k', 0.1));
    end
    t2 = toc;
    printf('Time for 1000 TANK creations: %.6f seconds\n', t2);
    printf('Average time per creation: %.6f seconds\n', t2/1000);
end

% Run all tests
printf('Running fluids tests from Octave...\n');
test_fluids();
test_atmosphere();
test_tank();
benchmark_fluids();
printf('\nAll tests completed!\n'); 
