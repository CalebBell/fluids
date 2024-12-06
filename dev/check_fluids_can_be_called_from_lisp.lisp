#!/usr/bin/sbcl --script

;; Load quicklisp
#-quicklisp
(let ((quicklisp-init (merge-pathnames "quicklisp/setup.lisp"
                                     (user-homedir-pathname))))
  (when (probe-file quicklisp-init)
    (load quicklisp-init)))

;; Load py4cl and ensure we're using Python 3
(ql:quickload :py4cl :silent t)
(setf py4cl:*python-command* "python3")

;; Import required modules
;; (py4cl:import-module "fluids" :as "fl")
(py4cl:import-module "fluids" :as "fluids")

;; Check Python environment
(py4cl:python-exec "
import sys
print('Python path:', sys.executable)
try:
    import numpy
    print('NumPy found at:', numpy.__file__)
    import fluids
    print('fluids found at:', fluids.__file__)
except ImportError as e:
    print('Error importing NumPy:', e)
")



(defun test-fluids ()
  (handler-case
      (progn
        (format t "✓ Successfully imported fluids~%")
        (format t "✓ Fluids version: ~A~%" (py4cl:python-eval "fluids.__version__"))
        ;; Test Reynolds number calculation
        (let ((re (py4cl:python-eval 
                   (format nil "fluids.Reynolds(**{'V': ~F, 'D': ~F, 'rho': ~F, 'mu': ~F})"
                          2.5 0.1 1000 0.001))))
          (format t "✓ Reynolds number calculation successful: ~A~%" re)
          (assert (> re 0)))
        
        ;; Test friction factor calculation
        (let ((fd (py4cl:python-eval 
                   (format nil "fluids.friction_factor(**{'Re': ~E, 'eD': ~F})"
                          1e5 0.0001))))
          (format t "✓ Friction factor calculation successful: ~A~%" fd)
          (assert (and (> fd 0) (< fd 1))))
        
        (format t "~%All basic tests completed successfully!~%"))
    (error (e)
      (format t "Error occurred: ~A~%" e)
      (error e))))
      
      
      
(defun test-atmosphere ()
  (handler-case
      (progn
        ;; Create and store the atmosphere object in Python's namespace
        (py4cl:python-exec 
         (format nil "atm = fluids.ATMOSPHERE_1976(**{'Z': ~F})" 5000))
        
        (format t "~%Testing atmosphere at 5000m elevation:~%")
        ;; Access properties using attributes
        (format t "✓ Temperature: ~,4F~%" 
                (py4cl:python-eval "atm.T"))
        (format t "✓ Pressure: ~,4F~%" 
                (py4cl:python-eval "atm.P"))
        (format t "✓ Density: ~,6F~%" 
                (py4cl:python-eval "atm.rho"))
        
        ;; Test derived properties
        (format t "✓ Gravity: ~,6F~%" 
                (py4cl:python-eval "atm.g"))
        (format t "✓ Viscosity: ~,6E~%" 
                (py4cl:python-eval "atm.mu"))
        (format t "✓ Thermal conductivity: ~,6F~%" 
                (py4cl:python-eval "atm.k"))
        (format t "✓ Sonic velocity: ~,4F~%" 
                (py4cl:python-eval "atm.v_sonic"))
        
        ;; Test static methods
        (let ((g-high (py4cl:python-eval 
                      (format nil "fluids.ATMOSPHERE_1976.gravity(**{'Z': ~E})" 1e5))))
          (format t "✓ High altitude gravity: ~,6F~%" g-high))
        
        (let ((v-sonic (py4cl:python-eval 
                       (format nil "fluids.ATMOSPHERE_1976.sonic_velocity(**{'T': ~F})" 300))))
          (format t "✓ Sonic velocity at 300K: ~,4F~%" v-sonic))
        
        (let ((mu-400 (py4cl:python-eval 
                      (format nil "fluids.ATMOSPHERE_1976.viscosity(**{'T': ~F})" 400))))
          (format t "✓ Viscosity at 400K: ~,6E~%" mu-400))
        
        (let ((k-400 (py4cl:python-eval 
                     (format nil "fluids.ATMOSPHERE_1976.thermal_conductivity(**{'T': ~F})" 400))))
          (format t "✓ Thermal conductivity at 400K: ~,6F~%" k-400)))
    (error (e)
      (format t "Error in atmosphere tests: ~A~%" e)
      (error e))))
      
      
(defun test-tank ()
  (handler-case
      (progn
        ;; Test basic tank creation
        (py4cl:python-exec 
         (format nil "t1 = fluids.TANK(**{'V': ~F, 'L_over_D': ~F, 'sideB': 'conical', 'horizontal': False})"
                 10 0.7))
        (format t "~%Testing tank calculations:~%")
        (format t "✓ Tank length: ~,6F~%" 
                (py4cl:python-eval "t1.L"))
        (format t "✓ Tank diameter: ~,6F~%" 
                (py4cl:python-eval "t1.D"))
        
        ;; Test ellipsoidal tank
        (py4cl:python-exec 
         (format nil "tank_ellip = fluids.TANK(**{'D': ~F, 'V': ~F, 'horizontal': False, 
                                                 'sideA': 'ellipsoidal', 'sideB': 'ellipsoidal',
                                                 'sideA_a': ~F, 'sideB_a': ~F})"
                 10 500 1 1))
        (format t "✓ Ellipsoidal tank L: ~,6F~%" 
                (py4cl:python-eval "tank_ellip.L"))
        
        ;; Test torispherical tank
        (py4cl:python-exec 
         (format nil "din = fluids.TANK(**{'L': ~F, 'D': ~F, 'horizontal': False,
                                         'sideA': 'torispherical', 'sideB': 'torispherical',
                                         'sideA_f': ~F, 'sideA_k': ~F, 'sideB_f': ~F, 'sideB_k': ~F})"
                 3 5 1 0.1 1 0.1))
        
        (format t "✓ Tank representation: ~A~%" 
                (py4cl:python-eval "str(din)"))
        (format t "✓ Tank max height: ~,6F~%" 
                (py4cl:python-eval "din.h_max"))
        (format t "✓ Height at V=40: ~,6F~%" 
                (py4cl:python-eval "din.h_from_V(40)"))
        (format t "✓ Volume at h=4.1: ~,5F~%" 
                (py4cl:python-eval "din.V_from_h(4.1)"))
        (format t "✓ Surface area at h=2.1: ~,5F~%" 
                (py4cl:python-eval "din.SA_from_h(2.1)")))
    (error (e)
      (format t "Error in tank tests: ~A~%" e)
      (error e))))
      
      
(defun test-psd ()
  (handler-case
      (progn
        (format t "~%Testing particle size distribution functionality:~%")
        
        ;; Create arrays in Python's namespace
        (py4cl:python-exec "
ds = [240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532]
numbers = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]")
        
        ;; Create discrete PSD
        (py4cl:python-exec "
psd = fluids.particle_size_distribution.ParticleSizeDistribution(
    ds=ds,
    fractions=numbers,
    order=0
)")
        (format t "✓ Created discrete PSD~%")
        
        ;; Test mean sizes
        (let ((d21 (py4cl:python-eval "psd.mean_size(2, 1)")))
          (format t "✓ Size-weighted mean diameter: ~,4F~%" d21)
          (assert (< (abs (- d21 1857.788)) 0.1)))
        
        (let ((d10 (py4cl:python-eval "psd.mean_size(1, 0)")))
          (format t "✓ Arithmetic mean diameter: ~,4F~%" d10)
          (assert (< (abs (- d10 1459.372)) 0.1)))
        
        ;; Test percentile calculations
        (let ((d10-percentile (py4cl:python-eval "psd.dn(0.1)"))
              (d90-percentile (py4cl:python-eval "psd.dn(0.9)")))
          (format t "✓ D10: ~,4F~%" d10-percentile)
          (format t "✓ D90: ~,4F~%" d90-percentile))
        
        ;; Test probability functions
        (let ((pdf-val (py4cl:python-eval "psd.pdf(1000)"))
              (cdf-val (py4cl:python-eval "psd.cdf(5000)")))
          (format t "✓ PDF at 1000: ~,4E~%" pdf-val)
          (format t "✓ CDF at 5000: ~,6F~%" cdf-val))
        
        ;; Test lognormal distribution
        (py4cl:python-exec 
         (format nil "psd_log = fluids.particle_size_distribution.PSDLognormal(**{'s': ~F, 'd_characteristic': ~E})"
                 0.5 5e-6))
        (format t "✓ Created lognormal PSD~%")
        
        (let ((vssa (py4cl:python-eval "psd_log.vssa")))
          (format t "✓ Volume specific surface area: ~,2F~%" vssa))
        
        (let ((span (py4cl:python-eval "psd_log.dn(0.9) - psd_log.dn(0.1)")))
          (format t "✓ Span: ~,4E~%" span))
        
        (let ((ratio-7525 (py4cl:python-eval "psd_log.dn(0.75)/psd_log.dn(0.25)")))
          (format t "✓ D75/D25 ratio: ~,6F~%" ratio-7525)))
    (error (e)
      (format t "Error in PSD tests: ~A~%" e)
      (error e))))
      
(defun benchmark-fluids ()
 (format t "~%Running benchmarks:~%")
 
 ;; Benchmark friction factor calculation
 (format t "~%Benchmarking friction_factor:~%")
 (let ((t1 (get-internal-real-time)))
   (dotimes (i 10000)
     (py4cl:python-eval 
      (format nil "fluids.friction_factor(**{'Re': ~E, 'eD': ~F})" 1e5 0.0001)))
   (let ((elapsed (/ (- (get-internal-real-time) t1) internal-time-units-per-second)))
     (format t "Time for 1e6 friction_factor calls: ~,6F seconds~%" elapsed)
     (format t "Average time per call: ~,6F seconds~%" (/ elapsed 10000))))
 
 ;; Benchmark tank creation
 (format t "~%Benchmarking TANK creation:~%")
 (let ((t2 (get-internal-real-time)))
   (dotimes (i 10000)
     (py4cl:python-eval 
      (format nil "fluids.TANK(**{'L': ~F, 'D': ~F, 'horizontal': False,
                                 'sideA': 'torispherical', 'sideB': 'torispherical',
                                 'sideA_f': ~F, 'sideA_k': ~F, 
                                 'sideB_f': ~F, 'sideB_k': ~F})"
              3 5 1 0.1 1 0.1)))
   (let ((elapsed (/ (- (get-internal-real-time) t2) internal-time-units-per-second)))
     (format t "Average time per creation: ~,6F seconds~%" (/ elapsed 10000)))))
     
     
    ;; Run all tests
(format t "Running fluids tests from Common Lisp...~%")
(test-fluids)
(test-atmosphere)
(test-tank)
(test-psd)
(benchmark-fluids)
(format t "~%All tests completed!~%")

;; Clean up
(py4cl:python-stop)
