#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <math.h>

// Helper function to initialize Python and import fluids
PyObject* init_fluids(void) {
    Py_Initialize();
    PyObject* module = PyImport_ImportModule("fluids");
    if (!module) {
        PyErr_Print();
        fprintf(stderr, "Failed to import fluids module\n");
        return NULL;
    }
    return module;
}

// Helper to measure time in microseconds
long long microseconds_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000LL;
}

void test_atmosphere(PyObject* fluids) {
    printf("\nTesting atmosphere at 5000m elevation:\n");
    
    // Create ATMOSPHERE_1976 instance
    PyObject* atm_class = PyObject_GetAttrString(fluids, "ATMOSPHERE_1976");
    if (!atm_class) {
        PyErr_Print();
        return;
    }
    
    PyObject* args = PyTuple_New(0);
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "Z", PyFloat_FromDouble(5000));
    
    PyObject* atm = PyObject_Call(atm_class, args, kwargs);
    if (!atm) {
        PyErr_Print();
        Py_DECREF(args);
        Py_DECREF(kwargs);
        Py_DECREF(atm_class);
        return;
    }
    
    // Test various properties
    PyObject* temp = PyObject_GetAttrString(atm, "T");
    PyObject* pressure = PyObject_GetAttrString(atm, "P");
    PyObject* density = PyObject_GetAttrString(atm, "rho");
    PyObject* gravity = PyObject_GetAttrString(atm, "g");
    PyObject* viscosity = PyObject_GetAttrString(atm, "mu");
    PyObject* conductivity = PyObject_GetAttrString(atm, "k");
    PyObject* sonic = PyObject_GetAttrString(atm, "v_sonic");
    
    printf("✓ Temperature: %.4f\n", PyFloat_AsDouble(temp));
    printf("✓ Pressure: %.4f\n", PyFloat_AsDouble(pressure));
    printf("✓ Density: %.6f\n", PyFloat_AsDouble(density));
    printf("✓ Gravity: %.6f\n", PyFloat_AsDouble(gravity));
    printf("✓ Viscosity: %.6e\n", PyFloat_AsDouble(viscosity));
    printf("✓ Thermal conductivity: %.4f\n", PyFloat_AsDouble(conductivity));
    printf("✓ Sonic velocity: %.4f\n", PyFloat_AsDouble(sonic));
    
    // Test static gravity method
    PyObject* gravity_method = PyObject_GetAttrString(atm_class, "gravity");
    PyObject* gravity_args = PyTuple_New(0);
    PyObject* gravity_kwargs = PyDict_New();
    PyDict_SetItemString(gravity_kwargs, "Z", PyFloat_FromDouble(1E5));
    
    PyObject* high_gravity = PyObject_Call(gravity_method, gravity_args, gravity_kwargs);
    printf("✓ High altitude gravity: %.6f\n", PyFloat_AsDouble(high_gravity));
    
    // Cleanup
    Py_DECREF(temp);
    Py_DECREF(pressure);
    Py_DECREF(density);
    Py_DECREF(gravity);
    Py_DECREF(viscosity);
    Py_DECREF(conductivity);
    Py_DECREF(sonic);
    Py_DECREF(gravity_method);
    Py_DECREF(gravity_args);
    Py_DECREF(gravity_kwargs);
    Py_DECREF(high_gravity);
    Py_DECREF(atm);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(atm_class);
}

void test_expanded_reynolds(PyObject* fluids) {
    printf("\nTesting Reynolds number calculations:\n");
    
    // Get Reynolds function
    PyObject* reynolds_func = PyObject_GetAttrString(fluids, "Reynolds");
    if (!reynolds_func) {
        PyErr_Print();
        return;
    }
    
    // Test with density and viscosity
    PyObject* args1 = PyTuple_New(0);
    PyObject* kwargs1 = PyDict_New();
    PyDict_SetItemString(kwargs1, "V", PyFloat_FromDouble(2.5));
    PyDict_SetItemString(kwargs1, "D", PyFloat_FromDouble(0.25));
    PyDict_SetItemString(kwargs1, "rho", PyFloat_FromDouble(1.1613));
    PyDict_SetItemString(kwargs1, "mu", PyFloat_FromDouble(1.9E-5));
    
    PyObject* re1 = PyObject_Call(reynolds_func, args1, kwargs1);
    double re1_val = PyFloat_AsDouble(re1);
    printf("✓ Re (with rho, mu): %.4f\n", re1_val);
    assert(fabs(re1_val - 38200.6579) < 0.1);
    
    // Test with kinematic viscosity
    PyObject* args2 = PyTuple_New(0);
    PyObject* kwargs2 = PyDict_New();
    PyDict_SetItemString(kwargs2, "V", PyFloat_FromDouble(2.5));
    PyDict_SetItemString(kwargs2, "D", PyFloat_FromDouble(0.25));
    PyDict_SetItemString(kwargs2, "nu", PyFloat_FromDouble(1.636e-05));
    
    PyObject* re2 = PyObject_Call(reynolds_func, args2, kwargs2);
    double re2_val = PyFloat_AsDouble(re2);
    printf("✓ Re (with nu): %.4f\n", re2_val);
    assert(fabs(re2_val - 38202.934) < 0.1);
    
    // Cleanup
    Py_DECREF(reynolds_func);
    Py_DECREF(args1);
    Py_DECREF(kwargs1);
    Py_DECREF(re1);
    Py_DECREF(args2);
    Py_DECREF(kwargs2);
    Py_DECREF(re2);
}

void test_psd(PyObject* fluids) {
    printf("\nTesting particle size distribution functionality:\n");
    
    // Get particle_size_distribution module
    PyObject* psd_module = PyObject_GetAttrString(fluids, "particle_size_distribution");
    if (!psd_module) {
        PyErr_Print();
        return;
    }
    
    // Create discrete PSD
    PyObject* psd_class = PyObject_GetAttrString(psd_module, "ParticleSizeDistribution");
    if (!psd_class) {
        PyErr_Print();
        Py_DECREF(psd_module);
        return;
    }
    
    // Create lists for ds and numbers
    PyObject* ds_list = PyList_New(15);
    double ds[] = {240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 
                   2141, 2676, 3345, 4181, 5226, 6532};
    for (int i = 0; i < 15; i++) {
        PyList_SetItem(ds_list, i, PyFloat_FromDouble(ds[i]));
    }
    
    PyObject* numbers_list = PyList_New(14);
    double numbers[] = {65, 119, 232, 410, 629, 849, 990, 981, 825, 
                       579, 297, 111, 21, 1};
    for (int i = 0; i < 14; i++) {
        PyList_SetItem(numbers_list, i, PyFloat_FromDouble(numbers[i]));
    }
    
    PyObject* args = PyTuple_New(0);
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "ds", ds_list);
    PyDict_SetItemString(kwargs, "fractions", numbers_list);
    PyDict_SetItemString(kwargs, "order", PyLong_FromLong(0));
    
    PyObject* psd = PyObject_Call(psd_class, args, kwargs);
    printf("✓ Created discrete PSD\n");
    
    // Test mean sizes
    PyObject* mean_size_method = PyObject_GetAttrString(psd, "mean_size");
    PyObject* mean_args = PyTuple_Pack(2, PyLong_FromLong(2), PyLong_FromLong(1));
    PyObject* d21 = PyObject_Call(mean_size_method, mean_args, NULL);
    printf("✓ Size-weighted mean diameter: %.4f\n", PyFloat_AsDouble(d21));
    
    PyObject* mean_args2 = PyTuple_Pack(2, PyLong_FromLong(1), PyLong_FromLong(0));
    PyObject* d10 = PyObject_Call(mean_size_method, mean_args2, NULL);
    printf("✓ Arithmetic mean diameter: %.4f\n", PyFloat_AsDouble(d10));
    
    // Test percentile calculations
    PyObject* dn_method = PyObject_GetAttrString(psd, "dn");
    PyObject* d10_args = PyTuple_Pack(1, PyFloat_FromDouble(0.1));
    PyObject* d90_args = PyTuple_Pack(1, PyFloat_FromDouble(0.9));
    
    PyObject* d10_percentile = PyObject_Call(dn_method, d10_args, NULL);
    PyObject* d90_percentile = PyObject_Call(dn_method, d90_args, NULL);
    
    printf("✓ D10: %.4f\n", PyFloat_AsDouble(d10_percentile));
    printf("✓ D90: %.4f\n", PyFloat_AsDouble(d90_percentile));
    
    // Cleanup
    Py_DECREF(psd_module);
    Py_DECREF(psd_class);
    Py_DECREF(ds_list);
    Py_DECREF(numbers_list);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(psd);
    Py_DECREF(mean_size_method);
    Py_DECREF(mean_args);
    Py_DECREF(d21);
    Py_DECREF(mean_args2);
    Py_DECREF(d10);
    Py_DECREF(dn_method);
    Py_DECREF(d10_args);
    Py_DECREF(d90_args);
    Py_DECREF(d10_percentile);
    Py_DECREF(d90_percentile);
}

void test_fluids(PyObject* fluids) {
    printf("Running fluids tests from C...\n");

    // Get version
    PyObject* version = PyObject_GetAttrString(fluids, "__version__");
    if (version) {
        const char* version_str = PyUnicode_AsUTF8(version);
        printf("✓ Successfully imported fluids\n");
        printf("✓ Fluids version: %s\n", version_str);
        Py_DECREF(version);
    }

    // Test Reynolds number calculation
    PyObject* reynolds_func = PyObject_GetAttrString(fluids, "Reynolds");
    if (reynolds_func) {
        PyObject* args = PyTuple_New(0);
        PyObject* kwargs = PyDict_New();
        PyDict_SetItemString(kwargs, "V", PyFloat_FromDouble(2.5));
        PyDict_SetItemString(kwargs, "D", PyFloat_FromDouble(0.1));
        PyDict_SetItemString(kwargs, "rho", PyFloat_FromDouble(1000));
        PyDict_SetItemString(kwargs, "mu", PyFloat_FromDouble(0.001));
        
        PyObject* result = PyObject_Call(reynolds_func, args, kwargs);
        if (result) {
            double re = PyFloat_AsDouble(result);
            printf("✓ Reynolds number calculation successful: %f\n", re);
            assert(re > 0);
            Py_DECREF(result);
        }
        
        Py_DECREF(args);
        Py_DECREF(kwargs);
        Py_DECREF(reynolds_func);
    }
}

void benchmark_fluids(PyObject* fluids) {
    printf("\nRunning benchmarks:\n");
    
    // Get friction_factor function
    PyObject* friction_func = PyObject_GetAttrString(fluids, "friction_factor");
    if (!friction_func) {
        PyErr_Print();
        return;
    }

    // Get TANK class
    PyObject* tank_class = PyObject_GetAttrString(fluids, "TANK");
    if (!tank_class) {
        PyErr_Print();
        Py_DECREF(friction_func);
        return;
    }

    // Benchmark friction_factor
    printf("\nBenchmarking friction_factor:\n");
    long long start = microseconds_now();
    
    for(int i = 0; i < 1000000; i++) {
        PyObject* args = PyTuple_New(0);
        PyObject* kwargs = PyDict_New();
        PyDict_SetItemString(kwargs, "Re", PyFloat_FromDouble(1e5));
        PyDict_SetItemString(kwargs, "eD", PyFloat_FromDouble(0.0001));
        
        PyObject* result = PyObject_Call(friction_func, args, kwargs);
        
        Py_DECREF(result);
        Py_DECREF(args);
        Py_DECREF(kwargs);
    }
    
    long long duration = microseconds_now() - start;
    printf("Time for 1e6 friction_factor calls: %lld microseconds\n", duration);
    printf("Average time per call: %.6f microseconds\n", duration / 1000000.0);

    // Benchmark TANK creation
    printf("\nBenchmarking TANK creation:\n");
    start = microseconds_now();
    
    for(int i = 0; i < 1000; i++) {
        PyObject* args = PyTuple_New(0);
        PyObject* kwargs = PyDict_New();
        PyDict_SetItemString(kwargs, "L", PyFloat_FromDouble(3));
        PyDict_SetItemString(kwargs, "D", PyFloat_FromDouble(5));
        PyDict_SetItemString(kwargs, "horizontal", Py_False);
        PyDict_SetItemString(kwargs, "sideA", PyUnicode_FromString("torispherical"));
        PyDict_SetItemString(kwargs, "sideB", PyUnicode_FromString("torispherical"));
        PyDict_SetItemString(kwargs, "sideA_f", PyFloat_FromDouble(1));
        PyDict_SetItemString(kwargs, "sideA_k", PyFloat_FromDouble(0.1));
        PyDict_SetItemString(kwargs, "sideB_f", PyFloat_FromDouble(1));
        PyDict_SetItemString(kwargs, "sideB_k", PyFloat_FromDouble(0.1));
        
        PyObject* tank = PyObject_Call(tank_class, args, kwargs);
        
        Py_DECREF(tank);
        Py_DECREF(args);
        Py_DECREF(kwargs);
    }
    
    duration = microseconds_now() - start;
    printf("Average time per tank creation: %.6f microseconds\n", duration / 1000.0);

    Py_DECREF(friction_func);
    Py_DECREF(tank_class);
}

void test_tank(PyObject* fluids) {
    printf("\nTesting tank calculations:\n");
    
    // Get TANK class
    PyObject* tank_class = PyObject_GetAttrString(fluids, "TANK");
    if (!tank_class) {
        PyErr_Print();
        return;
    }
    
    // Test basic tank creation
    PyObject* args1 = PyTuple_New(0);
    PyObject* kwargs1 = PyDict_New();
    PyDict_SetItemString(kwargs1, "V", PyFloat_FromDouble(10));
    PyDict_SetItemString(kwargs1, "L_over_D", PyFloat_FromDouble(0.7));
    PyDict_SetItemString(kwargs1, "sideB", PyUnicode_FromString("conical"));
    PyDict_SetItemString(kwargs1, "horizontal", Py_False);
    
    PyObject* T1 = PyObject_Call(tank_class, args1, kwargs1);
    if (!T1) {
        PyErr_Print();
        goto cleanup1;
    }
    
    PyObject* length = PyObject_GetAttrString(T1, "L");
    PyObject* diameter = PyObject_GetAttrString(T1, "D");
    printf("✓ Tank length: %.6f\n", PyFloat_AsDouble(length));
    printf("✓ Tank diameter: %.6f\n", PyFloat_AsDouble(diameter));
    
    // Test torispherical tank
    PyObject* args2 = PyTuple_New(0);
    PyObject* kwargs2 = PyDict_New();
    PyDict_SetItemString(kwargs2, "L", PyFloat_FromDouble(3));
    PyDict_SetItemString(kwargs2, "D", PyFloat_FromDouble(5));
    PyDict_SetItemString(kwargs2, "horizontal", Py_False);
    PyDict_SetItemString(kwargs2, "sideA", PyUnicode_FromString("torispherical"));
    PyDict_SetItemString(kwargs2, "sideB", PyUnicode_FromString("torispherical"));
    PyDict_SetItemString(kwargs2, "sideA_f", PyFloat_FromDouble(1));
    PyDict_SetItemString(kwargs2, "sideA_k", PyFloat_FromDouble(0.1));
    PyDict_SetItemString(kwargs2, "sideB_f", PyFloat_FromDouble(1));
    PyDict_SetItemString(kwargs2, "sideB_k", PyFloat_FromDouble(0.1));
    
    PyObject* DIN = PyObject_Call(tank_class, args2, kwargs2);
    if (!DIN) {
        PyErr_Print();
        goto cleanup2;
    }
    
    // Get max height
    PyObject* h_max = PyObject_GetAttrString(DIN, "h_max");
    printf("✓ Tank max height: %.6f\n", PyFloat_AsDouble(h_max));
    
    // Test h_from_V method
    PyObject* h_from_V = PyObject_GetAttrString(DIN, "h_from_V");
    PyObject* h_args = PyTuple_Pack(1, PyFloat_FromDouble(40));
    PyObject* h_result = PyObject_CallObject(h_from_V, h_args);
    printf("✓ Height at V=40: %.6f\n", PyFloat_AsDouble(h_result));
    
    // Test V_from_h method
    PyObject* V_from_h = PyObject_GetAttrString(DIN, "V_from_h");
    PyObject* v_args = PyTuple_Pack(1, PyFloat_FromDouble(4.1));
    PyObject* v_result = PyObject_CallObject(V_from_h, v_args);
    printf("✓ Volume at h=4.1: %.5f\n", PyFloat_AsDouble(v_result));
    
    // Cleanup
    Py_DECREF(v_result);
    Py_DECREF(v_args);
    Py_DECREF(V_from_h);
    Py_DECREF(h_result);
    Py_DECREF(h_args);
    Py_DECREF(h_from_V);
    Py_DECREF(h_max);
    Py_DECREF(DIN);
cleanup2:
    Py_DECREF(args2);
    Py_DECREF(kwargs2);
    Py_DECREF(diameter);
    Py_DECREF(length);
    Py_DECREF(T1);
cleanup1:
    Py_DECREF(args1);
    Py_DECREF(kwargs1);
    Py_DECREF(tank_class);
}

int main() {
    PyObject* fluids = init_fluids();
    if (!fluids) {
        return 1;
    }

    test_fluids(fluids);
    test_atmosphere(fluids);
    test_tank(fluids);
    test_expanded_reynolds(fluids);
    test_psd(fluids);
    benchmark_fluids(fluids);

    Py_DECREF(fluids);
    Py_Finalize();
    printf("\nAll tests completed!\n");
    return 0;
}        
