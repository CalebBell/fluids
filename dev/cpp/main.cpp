#include <pybind11/pybind11.h>
#include <pybind11/embed.h> // Added this for interpreter management
#include <pybind11/stl.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

namespace py = pybind11;
PYBIND11_EMBEDDED_MODULE(dummy, m) {} // Needed for embedding

class FluidsTester {
private:
    py::scoped_interpreter guard; // RAII management of the Python interpreter
    py::module_ fluids;
    
public:
    FluidsTester() {
        try {
            // The interpreter is automatically initialized by scoped_interpreter
            fluids = py::module_::import("fluids");
            std::cout << "✓ Successfully imported fluids\n";
            std::cout << "✓ Fluids version: " << fluids.attr("__version__").cast<std::string>() << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error initializing Python: " << e.what() << std::endl;
            throw;
        }
    }

    void test_fluids() {
        try {
            // Test basic Reynolds number calculation
            auto Re = fluids.attr("Reynolds")(
                py::arg("V") = 2.5,
                py::arg("D") = 0.1,
                py::arg("rho") = 1000,
                py::arg("mu") = 0.001
            ).cast<double>();
            std::cout << "✓ Reynolds number calculation successful: " << Re << "\n";
            assert(Re > 0);

            // Test friction factor calculation
            auto fd = fluids.attr("friction_factor")(
                py::arg("Re") = 1e5,
                py::arg("eD") = 0.0001
            ).cast<double>();
            std::cout << "✓ Friction factor calculation successful: " << fd << "\n";
            assert(fd > 0 && fd < 1);

            std::cout << "\nAll basic tests completed successfully!\n";
        } catch (const std::exception& e) {
            std::cerr << "Error in fluids tests: " << e.what() << std::endl;
            throw;
        }
    }

    void test_atmosphere() {
        try {
            // Test ATMOSPHERE_1976 class
            auto atm = fluids.attr("ATMOSPHERE_1976")(py::arg("Z") = 5000);
            
            std::cout << "\nTesting atmosphere at 5000m elevation:\n";
            std::cout << "✓ Temperature: " << std::fixed << std::setprecision(4) 
                      << atm.attr("T").cast<double>() << "\n";
            std::cout << "✓ Pressure: " << atm.attr("P").cast<double>() << "\n";
            std::cout << "✓ Density: " << std::setprecision(6) 
                      << atm.attr("rho").cast<double>() << "\n";
            
            // Test derived properties
            std::cout << "✓ Gravity: " << atm.attr("g").cast<double>() << "\n";
            std::cout << "✓ Viscosity: " << std::scientific 
                      << atm.attr("mu").cast<double>() << "\n";
            std::cout << "✓ Thermal conductivity: " << std::fixed 
                      << atm.attr("k").cast<double>() << "\n";
            std::cout << "✓ Sonic velocity: " << std::setprecision(4) 
                      << atm.attr("v_sonic").cast<double>() << "\n";

            // Test static methods
            auto atm_class = fluids.attr("ATMOSPHERE_1976");
            auto g_high = atm_class.attr("gravity")(py::arg("Z") = 1E5).cast<double>();
            std::cout << "✓ High altitude gravity: " << std::setprecision(6) 
                      << g_high << "\n";

        } catch (const std::exception& e) {
            std::cerr << "Error in atmosphere tests: " << e.what() << std::endl;
            throw;
        }
    }

    void test_tank() {
        try {
            // Test basic tank creation
            auto T1 = fluids.attr("TANK")(
                py::arg("V") = 10,
                py::arg("L_over_D") = 0.7,
                py::arg("sideB") = "conical",
                py::arg("horizontal") = false
            );
            
            std::cout << "\nTesting tank calculations:\n";
            std::cout << "✓ Tank length: " << std::fixed << std::setprecision(6) 
                      << T1.attr("L").cast<double>() << "\n";
            std::cout << "✓ Tank diameter: " << T1.attr("D").cast<double>() << "\n";

            // Test torispherical tank
            auto DIN = fluids.attr("TANK")(
                py::arg("L") = 3,
                py::arg("D") = 5,
                py::arg("horizontal") = false,
                py::arg("sideA") = "torispherical",
                py::arg("sideB") = "torispherical",
                py::arg("sideA_f") = 1,
                py::arg("sideA_k") = 0.1,
                py::arg("sideB_f") = 1,
                py::arg("sideB_k") = 0.1
            );

            std::cout << "✓ Tank max height: " << DIN.attr("h_max").cast<double>() << "\n";
            std::cout << "✓ Height at V=40: " << DIN.attr("h_from_V")(40).cast<double>() << "\n";
            std::cout << "✓ Volume at h=4.1: " << std::setprecision(5) 
                      << DIN.attr("V_from_h")(4.1).cast<double>() << "\n";
            
        } catch (const std::exception& e) {
            std::cerr << "Error in tank tests: " << e.what() << std::endl;
            throw;
        }
    }
void test_reynolds() {
    try {
        std::cout << "\nTesting Reynolds number calculations:\n";
        
        // Test with density and viscosity
        auto Re1 = fluids.attr("Reynolds")(
            py::arg("V") = 2.5,
            py::arg("D") = 0.25,
            py::arg("rho") = 1.1613,
            py::arg("mu") = 1.9E-5
        ).cast<double>();
        std::cout << "✓ Re (with rho, mu): " << std::fixed << std::setprecision(4) << Re1 << "\n";
        assert(std::abs(Re1 - 38200.6579) < 0.1);
        
        // Test with kinematic viscosity
        auto Re2 = fluids.attr("Reynolds")(
            py::arg("V") = 2.5,
            py::arg("D") = 0.25,
            py::arg("nu") = 1.636e-05
        ).cast<double>();
        std::cout << "✓ Re (with nu): " << Re2 << "\n";
        assert(std::abs(Re2 - 38202.934) < 0.1);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in Reynolds tests: " << e.what() << std::endl;
        throw;
    }
}

void test_psd() {
    try {
        std::cout << "\nTesting particle size distribution functionality:\n";
        
        // Create vectors for discrete PSD
        std::vector<double> ds = {240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 
                                 2141, 2676, 3345, 4181, 5226, 6532};
        std::vector<double> numbers = {65, 119, 232, 410, 629, 849, 990, 981, 825, 
                                     579, 297, 111, 21, 1};
        
        // Create a discrete PSD
        auto psd = fluids.attr("particle_size_distribution").attr("ParticleSizeDistribution")(
            py::arg("ds") = ds,
            py::arg("fractions") = numbers,
            py::arg("order") = 0
        );
        std::cout << "✓ Created discrete PSD\n";
        
        // Test mean sizes
        auto d21 = psd.attr("mean_size")(2, 1).cast<double>();
        std::cout << "✓ Size-weighted mean diameter: " << std::fixed 
                  << std::setprecision(4) << d21 << "\n";
        assert(std::abs(d21 - 1857.788) < 0.1);
        
        auto d10 = psd.attr("mean_size")(1, 0).cast<double>();
        std::cout << "✓ Arithmetic mean diameter: " << d10 << "\n";
        assert(std::abs(d10 - 1459.372) < 0.1);
        
        // Test percentile calculations
        auto d10_percentile = psd.attr("dn")(0.1).cast<double>();
        auto d90_percentile = psd.attr("dn")(0.9).cast<double>();
        std::cout << "✓ D10: " << d10_percentile << "\n";
        std::cout << "✓ D90: " << d90_percentile << "\n";
        
        // Test probability functions
        auto pdf_val = psd.attr("pdf")(1000).cast<double>();
        auto cdf_val = psd.attr("cdf")(5000).cast<double>();
        std::cout << "✓ PDF at 1000: " << std::scientific << pdf_val << "\n";
        std::cout << "✓ CDF at 5000: " << std::fixed << std::setprecision(6) << cdf_val << "\n";
        
        // Test lognormal distribution
        auto psd_log = fluids.attr("particle_size_distribution").attr("PSDLognormal")(
            py::arg("s") = 0.5,
            py::arg("d_characteristic") = 5E-6
        );
        std::cout << "✓ Created lognormal PSD\n";
        
        auto vssa = psd_log.attr("vssa").cast<double>();
        std::cout << "✓ Volume specific surface area: " << std::setprecision(2) 
                  << vssa << "\n";
        
        auto span = psd_log.attr("dn")(0.9).cast<double>() - 
                   psd_log.attr("dn")(0.1).cast<double>();
        std::cout << "✓ Span: " << std::scientific << span << "\n";
        
        auto ratio_7525 = (psd_log.attr("dn")(0.75).cast<double>() / 
                          psd_log.attr("dn")(0.25).cast<double>());
        std::cout << "✓ D75/D25 ratio: " << std::fixed << std::setprecision(6) 
                  << ratio_7525 << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error in PSD tests: " << e.what() << std::endl;
        throw;
    }
}
    void benchmark_fluids() {
        std::cout << "\nRunning benchmarks:\n";
        
        // Benchmark friction factor calculation
        std::cout << "\nBenchmarking friction_factor:\n";
        auto start = std::chrono::high_resolution_clock::now();
        
        for(int i = 0; i < 1000000; i++) {
            fluids.attr("friction_factor")(
                py::arg("Re") = 1e5,
                py::arg("eD") = 0.0001
            );
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Time for 1e6 friction_factor calls: " 
                  << duration.count() << " microseconds\n";
        std::cout << "Average time per call: " 
                  << duration.count() / 1000000.0 << " microseconds\n";

        // Benchmark tank creation
        std::cout << "\nBenchmarking TANK creation:\n";
        start = std::chrono::high_resolution_clock::now();
        
        for(int i = 0; i < 1000; i++) {
            fluids.attr("TANK")(
                py::arg("L") = 3,
                py::arg("D") = 5,
                py::arg("horizontal") = false,
                py::arg("sideA") = "torispherical",
                py::arg("sideB") = "torispherical",
                py::arg("sideA_f") = 1,
                py::arg("sideA_k") = 0.1,
                py::arg("sideB_f") = 1,
                py::arg("sideB_k") = 0.1
            );
        }
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Average time per tank creation: " 
                  << duration.count() / 1000.0 << " microseconds\n";
    }
};

int main() {
    try {
        std::cout << "Running fluids tests from C++...\n";
        FluidsTester tester;
        tester.test_fluids();
        tester.test_atmosphere();
        tester.test_tank();
        tester.test_reynolds();
        tester.test_psd();
        tester.benchmark_fluids();
        std::cout << "\nAll tests completed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
