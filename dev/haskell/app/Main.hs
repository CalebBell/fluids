{-# LANGUAGE OverloadedStrings #-}
module Main where

import qualified CPython as Py
import qualified CPython.Types.Module as Py
import qualified CPython.Protocols.Object as Py
import qualified CPython.Protocols.Number as PyNum
import qualified CPython.Types.Dictionary as PyDict
import qualified CPython.Types.Tuple as PyTuple
import qualified CPython.Types.Unicode as PyUnicode
import qualified CPython.Types.Float as PyFloat
import qualified CPython.Types.Exception as PyExc
import Control.Exception (handle)
import Text.Printf (printf)
import Data.Maybe (fromMaybe)
import qualified Data.Text as T
import System.Clock

testBasics :: IO ()
testBasics = do
    putStrLn "Testing basic fluids functionality..."
    
    -- Import fluids module
    fluidsModule <- Py.importModule (T.pack "fluids")
    
    -- Get version
    versionName <- PyUnicode.toUnicode (T.pack "__version__")
    versionObj <- Py.getAttribute fluidsModule versionName 
    Just versionStr <- Py.cast versionObj
    version <- PyUnicode.fromUnicode versionStr
    putStrLn $ "✓ Fluids version: " ++ T.unpack version
    
    -- Test Reynolds number calculation
    let calcReynolds = do
            -- Get Reynolds function
            reynoldsName <- PyUnicode.toUnicode (T.pack "Reynolds")
            reynolds <- Py.getAttribute fluidsModule reynoldsName
            
            -- Create arguments
            v <- PyFloat.toFloat 2.5
            d <- PyFloat.toFloat 0.1
            rho <- PyFloat.toFloat 1000.0
            mu <- PyFloat.toFloat 0.001
            args <- PyTuple.toTuple [Py.toObject v, Py.toObject d, Py.toObject rho, Py.toObject mu]
            kwargs <- PyDict.new
            
            -- Call function
            result <- Py.call reynolds args kwargs
            x <- PyNum.castToNumber result
            let num = fromMaybe (error "Could not convert Reynolds number") x
            PyFloat.fromFloat =<< PyNum.toFloat num
    
    re <- calcReynolds
    putStrLn $ printf "✓ Reynolds number calculation: %.1f" re
    
    -- Test friction factor
    let calcFriction = do
            -- Get friction_factor function
            frictionName <- PyUnicode.toUnicode (T.pack "friction_factor")
            friction <- Py.getAttribute fluidsModule frictionName
            
            -- Create arguments
            re <- PyFloat.toFloat 1e5
            ed <- PyFloat.toFloat 0.0001
            args <- PyTuple.toTuple [Py.toObject re, Py.toObject ed]
            kwargs <- PyDict.new
            
            -- Call function
            result <- Py.call friction args kwargs
            x <- PyNum.castToNumber result
            let num = fromMaybe (error "Could not convert friction factor") x
            PyFloat.fromFloat =<< PyNum.toFloat num
    
    fd <- calcFriction
    putStrLn $ printf "✓ Friction factor calculation: %.6f" fd
    putStrLn "Basic tests completed successfully!\n"

testAtmosphere :: IO ()
testAtmosphere = do
    putStrLn "\nTesting atmosphere at 5000m elevation:"
    
    -- Import fluids module
    fluidsModule <- Py.importModule (T.pack "fluids")
    
    -- Create argument for constructor
    zArg <- PyFloat.toFloat 5000.0
    args <- PyTuple.toTuple [Py.toObject zArg]
    kwargs <- PyDict.new
    
    -- Get ATMOSPHERE_1976 class and create instance
    atmosClass <- PyUnicode.toUnicode (T.pack "ATMOSPHERE_1976") >>= Py.getAttribute fluidsModule
    atm <- Py.call atmosClass args kwargs
    
    -- Get and print properties
    let getProperty name = do
            nameObj <- PyUnicode.toUnicode name
            prop <- Py.getAttribute atm nameObj
            x <- PyNum.castToNumber prop
            let num = fromMaybe (error $ "Could not get " ++ T.unpack name ++ " as number") x
            PyFloat.fromFloat =<< PyNum.toFloat num
    
    temp <- getProperty (T.pack "T") 
    pressure <- getProperty (T.pack "P")
    density <- getProperty (T.pack "rho")
    gravity <- getProperty (T.pack "g")
    viscosity <- getProperty (T.pack "mu")
    conductivity <- getProperty (T.pack "k")
    sonicVel <- getProperty (T.pack "v_sonic")
    
    putStrLn $ printf "✓ Temperature: %.4f" temp
    putStrLn $ printf "✓ Pressure: %.4f" pressure
    putStrLn $ printf "✓ Density: %.6f" density
    putStrLn $ printf "✓ Gravity: %.6f" gravity
    putStrLn $ printf "✓ Viscosity: %.6e" viscosity
    putStrLn $ printf "✓ Thermal conductivity: %.6f" conductivity
    putStrLn $ printf "✓ Sonic velocity: %.4f" sonicVel

benchmarkFluids :: IO ()
benchmarkFluids = do
    putStrLn "\nRunning benchmarks:"
    
    -- Import fluids module
    fluidsModule <- Py.importModule (T.pack "fluids")
    
    -- Get friction_factor function
    frictionName <- PyUnicode.toUnicode (T.pack "friction_factor")
    friction <- Py.getAttribute fluidsModule frictionName
    
    -- Prepare arguments that will be reused
    re <- PyFloat.toFloat 1e5
    ed <- PyFloat.toFloat 0.0001
    args <- PyTuple.toTuple [Py.toObject re, Py.toObject ed]
    kwargs <- PyDict.new
    
    -- Time the operations
    putStrLn "\nBenchmarking friction_factor:"
    start <- getTime Monotonic
    
    -- Do 1,000,000 iterations like in Julia version
    sequence_ $ replicate 1000000 $ do
        result <- Py.call friction args kwargs
        x <- PyNum.castToNumber result
        let num = fromMaybe (error "Could not convert friction factor") x
        PyFloat.fromFloat =<< PyNum.toFloat num
        
    end <- getTime Monotonic
    let diff = (fromIntegral (toNanoSecs (diffTimeSpec end start)) :: Double) / 1000000000
    
    putStrLn $ printf "Time for 1e6 friction_factor calls: %.6f seconds" diff
    putStrLn $ printf "Average time per call: %.6f seconds" (diff / 1000000)    

main :: IO ()
main = handle pyExceptionHandler $ do
    putStrLn "Running fluids tests from Haskell..."
    Py.initialize
    testBasics
    testAtmosphere
    benchmarkFluids
    Py.finalize
    putStrLn "\nAll tests completed!"
  where
    pyExceptionHandler :: PyExc.Exception -> IO ()
    pyExceptionHandler e = do
        putStrLn $ "Python error occurred: " ++ show e
        Py.finalize
