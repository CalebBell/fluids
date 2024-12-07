package main

/*
#cgo pkg-config: python3
#cgo LDFLAGS: -lpython3.11
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Helper function to convert C string to Go string without memory leaks
static char* getPyUnicodeAsString(PyObject* unicode) {
    return (char*)PyUnicode_AsUTF8(unicode);
}

// Helper to get double from PyFloat
static double getPyFloatAsDouble(PyObject* float_obj) {
    return PyFloat_AsDouble(float_obj);
}
*/
import "C"
import (
	"fmt"
	"time"
	"unsafe"
)

// Helper functions to convert between Go and Python types
func goStringToPyUnicode(s string) *C.PyObject {
	cs := C.CString(s)
	defer C.free(unsafe.Pointer(cs))
	return C.PyUnicode_FromString(cs)
}

func goFloat64ToPyFloat(v float64) *C.PyObject {
	return C.PyFloat_FromDouble(C.double(v))
}

func goIntToPyLong(v int64) *C.PyObject {
	return C.PyLong_FromLongLong(C.longlong(v))
}

func pyObjectToGoString(obj *C.PyObject) string {
	cstr := C.getPyUnicodeAsString(obj)
	return C.GoString(cstr)
}

func pyObjectToGoFloat64(obj *C.PyObject) float64 {
	return float64(C.getPyFloatAsDouble(obj))
}

// Initialize Python interpreter and import fluids module
func initFluids() *C.PyObject {
	C.Py_Initialize()
	
	moduleName := C.CString("fluids")
	defer C.free(unsafe.Pointer(moduleName))
	
	module := C.PyImport_ImportModule(moduleName)
	if module == nil {
		C.PyErr_Print()
		panic("Failed to import fluids module")
	}
	
	return module
}

func testAtmosphere(fluids *C.PyObject) {
	fmt.Println("\nTesting atmosphere at 5000m elevation:")
	
	// Get ATMOSPHERE_1976 class
	atmClass := C.PyObject_GetAttrString(fluids, C.CString("ATMOSPHERE_1976"))
	defer C.free(unsafe.Pointer(C.CString("ATMOSPHERE_1976")))
	if atmClass == nil {
		C.PyErr_Print()
		return
	}
	defer C.Py_DecRef(atmClass)
	
	// Create instance with Z=5000
	args := C.PyTuple_New(0)
	kwargs := C.PyDict_New()
	C.PyDict_SetItemString(kwargs, C.CString("Z"), goFloat64ToPyFloat(5000))
	defer C.free(unsafe.Pointer(C.CString("Z")))
	
	atm := C.PyObject_Call(atmClass, args, kwargs)
	if atm == nil {
		C.PyErr_Print()
		C.Py_DecRef(args)
		C.Py_DecRef(kwargs)
		return
	}
	defer C.Py_DecRef(atm)
	defer C.Py_DecRef(args)
	defer C.Py_DecRef(kwargs)
	
	// Get and print properties
	properties := []struct {
		name string
		format string
	}{
		{"T", "Temperature: %.4f"},
		{"P", "Pressure: %.4f"},
		{"rho", "Density: %.6f"},
		{"g", "Gravity: %.6f"},
		{"mu", "Viscosity: %.6e"},
		{"k", "Thermal conductivity: %.4f"},
		{"v_sonic", "Sonic velocity: %.4f"},
	}
	
	for _, prop := range properties {
		cname := C.CString(prop.name)
		value := C.PyObject_GetAttrString(atm, cname)
		C.free(unsafe.Pointer(cname))
		if value == nil {
			C.PyErr_Print()
			continue
		}
		fmt.Printf("✓ "+prop.format+"\n", pyObjectToGoFloat64(value))
		C.Py_DecRef(value)
	}
	
	// Test static gravity method
	gravityMethod := C.PyObject_GetAttrString(atmClass, C.CString("gravity"))
	defer C.free(unsafe.Pointer(C.CString("gravity")))
	if gravityMethod == nil {
		C.PyErr_Print()
		return
	}
	defer C.Py_DecRef(gravityMethod)
	
	gravityArgs := C.PyTuple_New(0)
	gravityKwargs := C.PyDict_New()
	C.PyDict_SetItemString(gravityKwargs, C.CString("Z"), goFloat64ToPyFloat(1E5))
	defer C.free(unsafe.Pointer(C.CString("Z")))
	
	highGravity := C.PyObject_Call(gravityMethod, gravityArgs, gravityKwargs)
	if highGravity == nil {
		C.PyErr_Print()
		C.Py_DecRef(gravityArgs)
		C.Py_DecRef(gravityKwargs)
		return
	}
	
	fmt.Printf("✓ High altitude gravity: %.6f\n", pyObjectToGoFloat64(highGravity))
	
	C.Py_DecRef(highGravity)
	C.Py_DecRef(gravityArgs)
	C.Py_DecRef(gravityKwargs)
}

func testExpandedReynolds(fluids *C.PyObject) {
	fmt.Println("\nTesting Reynolds number calculations:")
	
	reynoldsFunc := C.PyObject_GetAttrString(fluids, C.CString("Reynolds"))
	defer C.free(unsafe.Pointer(C.CString("Reynolds")))
	if reynoldsFunc == nil {
		C.PyErr_Print()
		return
	}
	defer C.Py_DecRef(reynoldsFunc)
	
	// Test with density and viscosity
	args1 := C.PyTuple_New(0)
	kwargs1 := C.PyDict_New()
	
	params1 := map[string]float64{
		"V": 2.5,
		"D": 0.25,
		"rho": 1.1613,
		"mu": 1.9E-5,
	}
	
	for k, v := range params1 {
		ckey := C.CString(k)
		C.PyDict_SetItemString(kwargs1, ckey, goFloat64ToPyFloat(v))
		C.free(unsafe.Pointer(ckey))
	}
	
	re1 := C.PyObject_Call(reynoldsFunc, args1, kwargs1)
	if re1 == nil {
		C.PyErr_Print()
		C.Py_DecRef(args1)
		C.Py_DecRef(kwargs1)
		return
	}
	
	re1Val := pyObjectToGoFloat64(re1)
	fmt.Printf("✓ Re (with rho, mu): %.4f\n", re1Val)
	
	C.Py_DecRef(re1)
	C.Py_DecRef(args1)
	C.Py_DecRef(kwargs1)
	
	// Test with kinematic viscosity
	args2 := C.PyTuple_New(0)
	kwargs2 := C.PyDict_New()
	
	params2 := map[string]float64{
		"V": 2.5,
		"D": 0.25,
		"nu": 1.636e-05,
	}
	
	for k, v := range params2 {
		ckey := C.CString(k)
		C.PyDict_SetItemString(kwargs2, ckey, goFloat64ToPyFloat(v))
		C.free(unsafe.Pointer(ckey))
	}
	
	re2 := C.PyObject_Call(reynoldsFunc, args2, kwargs2)
	if re2 == nil {
		C.PyErr_Print()
		C.Py_DecRef(args2)
		C.Py_DecRef(kwargs2)
		return
	}
	
	re2Val := pyObjectToGoFloat64(re2)
	fmt.Printf("✓ Re (with nu): %.4f\n", re2Val)
	
	C.Py_DecRef(re2)
	C.Py_DecRef(args2)
	C.Py_DecRef(kwargs2)
}


func testTank(fluids *C.PyObject) {
    fmt.Println("\nTesting tank calculations:")
    
    // Get TANK class
    tankClass := C.PyObject_GetAttrString(fluids, C.CString("TANK"))
    if tankClass == nil {
        C.PyErr_Print()
        return
    }
    
    // Test basic tank creation
    args1 := C.PyTuple_New(0)
    kwargs1 := C.PyDict_New()
    
    // Set basic tank parameters
    C.PyDict_SetItemString(kwargs1, C.CString("V"), goFloat64ToPyFloat(10))
    C.PyDict_SetItemString(kwargs1, C.CString("L_over_D"), goFloat64ToPyFloat(0.7))
    C.PyDict_SetItemString(kwargs1, C.CString("sideB"), goStringToPyUnicode("conical"))
    C.PyDict_SetItemString(kwargs1, C.CString("horizontal"), C.Py_False)
    
    tank1 := C.PyObject_Call(tankClass, args1, kwargs1)
    if tank1 == nil {
        C.PyErr_Print()
        return
    }
    
    // Get and print dimensions
    length := C.PyObject_GetAttrString(tank1, C.CString("L"))
    diameter := C.PyObject_GetAttrString(tank1, C.CString("D"))
    fmt.Printf("✓ Tank length: %.6f\n", pyObjectToGoFloat64(length))
    fmt.Printf("✓ Tank diameter: %.6f\n", pyObjectToGoFloat64(diameter))
    
    // Test ellipsoidal tank
    argsEllip := C.PyTuple_New(0)
    kwargsEllip := C.PyDict_New()
    
    ellipParams := map[string]interface{}{
        "D": 10.0,
        "V": 500.0,
        "horizontal": false,
        "sideA": "ellipsoidal",
        "sideB": "ellipsoidal",
        "sideA_a": 1.0,
        "sideB_a": 1.0,
    }
    
    for k, v := range ellipParams {
        switch val := v.(type) {
        case float64:
            C.PyDict_SetItemString(kwargsEllip, C.CString(k), goFloat64ToPyFloat(val))
        case string:
            C.PyDict_SetItemString(kwargsEllip, C.CString(k), goStringToPyUnicode(val))
        case bool:
            if val {
                C.PyDict_SetItemString(kwargsEllip, C.CString(k), C.Py_True)
            } else {
                C.PyDict_SetItemString(kwargsEllip, C.CString(k), C.Py_False)
            }
        }
    }
    
    tankEllip := C.PyObject_Call(tankClass, argsEllip, kwargsEllip)
    if tankEllip == nil {
        C.PyErr_Print()
        return
    }
    
    ellipL := C.PyObject_GetAttrString(tankEllip, C.CString("L"))
    fmt.Printf("✓ Ellipsoidal tank L: %.6f\n", pyObjectToGoFloat64(ellipL))
    
    // Test torispherical tank
    argsTori := C.PyTuple_New(0)
    kwargsTori := C.PyDict_New()
    
    toriParams := map[string]interface{}{
        "L": 3.0,
        "D": 5.0,
        "horizontal": false,
        "sideA": "torispherical",
        "sideB": "torispherical",
        "sideA_f": 1.0,
        "sideA_k": 0.1,
        "sideB_f": 1.0,
        "sideB_k": 0.1,
    }
    
    for k, v := range toriParams {
        switch val := v.(type) {
        case float64:
            C.PyDict_SetItemString(kwargsTori, C.CString(k), goFloat64ToPyFloat(val))
        case string:
            C.PyDict_SetItemString(kwargsTori, C.CString(k), goStringToPyUnicode(val))
        case bool:
            if val {
                C.PyDict_SetItemString(kwargsTori, C.CString(k), C.Py_True)
            } else {
                C.PyDict_SetItemString(kwargsTori, C.CString(k), C.Py_False)
            }
        }
    }
    
    DIN := C.PyObject_Call(tankClass, argsTori, kwargsTori)
    if DIN == nil {
        C.PyErr_Print()
        return
    }
    
    // Get tank string representation
    strMethod := C.PyObject_Str(DIN)
    fmt.Printf("✓ Tank representation: %s\n", pyObjectToGoString(strMethod))
    
    // Test various methods
    hMax := C.PyObject_GetAttrString(DIN, C.CString("h_max"))
    fmt.Printf("✓ Tank max height: %.6f\n", pyObjectToGoFloat64(hMax))
    
    // Test h_from_V method
    hFromV := C.PyObject_GetAttrString(DIN, C.CString("h_from_V"))
    hArgs := C.PyTuple_New(1)
    C.PyTuple_SetItem(hArgs, 0, goFloat64ToPyFloat(40))
    hResult := C.PyObject_CallObject(hFromV, hArgs)
    fmt.Printf("✓ Height at V=40: %.6f\n", pyObjectToGoFloat64(hResult))
    
    // Test V_from_h method
    vFromH := C.PyObject_GetAttrString(DIN, C.CString("V_from_h"))
    vArgs := C.PyTuple_New(1)
    C.PyTuple_SetItem(vArgs, 0, goFloat64ToPyFloat(4.1))
    vResult := C.PyObject_CallObject(vFromH, vArgs)
    fmt.Printf("✓ Volume at h=4.1: %.5f\n", pyObjectToGoFloat64(vResult))
    
    // Test SA_from_h method
    saFromH := C.PyObject_GetAttrString(DIN, C.CString("SA_from_h"))
    saArgs := C.PyTuple_New(1)
    C.PyTuple_SetItem(saArgs, 0, goFloat64ToPyFloat(2.1))
    saResult := C.PyObject_CallObject(saFromH, saArgs)
    fmt.Printf("✓ Surface area at h=2.1: %.5f\n", pyObjectToGoFloat64(saResult))
}

func benchmarkFluids(fluids *C.PyObject) {
    fmt.Println("\nRunning benchmarks:")
    
    // Benchmark friction_factor
    frictionFunc := C.PyObject_GetAttrString(fluids, C.CString("friction_factor"))
    if frictionFunc == nil {
        C.PyErr_Print()
        return
    }
    
    fmt.Println("\nBenchmarking friction_factor:")
    start := time.Now()
    
    for i := 0; i < 1000000; i++ {
        args := C.PyTuple_New(0)
        kwargs := C.PyDict_New()
        
        C.PyDict_SetItemString(kwargs, C.CString("Re"), goFloat64ToPyFloat(1e5))
        C.PyDict_SetItemString(kwargs, C.CString("eD"), goFloat64ToPyFloat(0.0001))
        
        result := C.PyObject_Call(frictionFunc, args, kwargs)
        C.Py_DecRef(result)
    }
    
    duration := time.Since(start)
    fmt.Printf("Time for 1e6 friction_factor calls: %v\n", duration)
    fmt.Printf("Average time per call: %.6f microseconds\n", float64(duration.Microseconds())/1000000.0)

    // Benchmark TANK creation
    tankClass := C.PyObject_GetAttrString(fluids, C.CString("TANK"))
    if tankClass == nil {
        C.PyErr_Print()
        return
    }
    
    fmt.Println("\nBenchmarking TANK creation and methods:")
    start = time.Now()
    
    for i := 0; i < 1000; i++ {
        args := C.PyTuple_New(0)
        kwargs := C.PyDict_New()
        
        // Create tank
        C.PyDict_SetItemString(kwargs, C.CString("L"), goFloat64ToPyFloat(3))
        C.PyDict_SetItemString(kwargs, C.CString("D"), goFloat64ToPyFloat(5))
        C.PyDict_SetItemString(kwargs, C.CString("horizontal"), C.Py_False)
        C.PyDict_SetItemString(kwargs, C.CString("sideA"), goStringToPyUnicode("torispherical"))
        C.PyDict_SetItemString(kwargs, C.CString("sideB"), goStringToPyUnicode("torispherical"))
        C.PyDict_SetItemString(kwargs, C.CString("sideA_f"), goFloat64ToPyFloat(1))
        C.PyDict_SetItemString(kwargs, C.CString("sideA_k"), goFloat64ToPyFloat(0.1))
        C.PyDict_SetItemString(kwargs, C.CString("sideB_f"), goFloat64ToPyFloat(1))
        C.PyDict_SetItemString(kwargs, C.CString("sideB_k"), goFloat64ToPyFloat(0.1))
        
        tank := C.PyObject_Call(tankClass, args, kwargs)
        
        // Test some methods while we have the tank
        vFromH := C.PyObject_GetAttrString(tank, C.CString("V_from_h"))
        vArgs := C.PyTuple_New(1)
        C.PyTuple_SetItem(vArgs, 0, goFloat64ToPyFloat(2.5))
        vResult := C.PyObject_CallObject(vFromH, vArgs)
        C.Py_DecRef(vResult)
    }
    
    duration = time.Since(start)
    fmt.Printf("Time for 1000 tank operations: %v\n", duration)
    fmt.Printf("Average time per tank operation: %.6f microseconds\n", float64(duration.Microseconds())/1000.0)
}

func main() {
	fluids := initFluids()
	if fluids == nil {
		return
	}
	defer C.Py_DecRef(fluids)
	
	// Get version
	version := C.PyObject_GetAttrString(fluids, C.CString("__version__"))
	defer C.free(unsafe.Pointer(C.CString("__version__")))
	if version != nil {
		fmt.Printf("✓ Successfully imported fluids\n")
		fmt.Printf("✓ Fluids version: %s\n", pyObjectToGoString(version))
		C.Py_DecRef(version)
	}
	
	testAtmosphere(fluids)
	testExpandedReynolds(fluids)
	testTank(fluids)
	benchmarkFluids(fluids)
	
	C.Py_Finalize()
	fmt.Println("\nAll tests completed!")
} 
