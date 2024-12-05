{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import qualified CPython as Py
import qualified CPython.Simple as PySim
import Data.Time.Clock.System (getSystemTime, systemSeconds)
import Control.Exception (handle)
import qualified CPython.Types.Exception as PyExc
import Text.Printf (printf)
import Data.Text (Text)

-- Helper function for timing
time :: IO a -> IO (Double, a)
time action = do
    start <- getSystemTime
    result <- action
    end <- getSystemTime
    let diff = fromIntegral (systemSeconds end - systemSeconds start)
    return (diff, result)

-- Basic fluids test
testFluids :: IO ()
testFluids = do
    putStrLn "Testing basic fluids functionality..."
    
    -- Test version
    version <- (PySim.getAttribute "fluids" "__version__" :: IO Text)
    putStrLn $ "✓ Fluids version: " ++ show version
    
    -- Test Reynolds number
    re <- (PySim.call "fluids" "Reynolds" 
            [PySim.arg (2.5 :: Double), PySim.arg (0.1 :: Double), 
             PySim.arg (1000.0 :: Double), PySim.arg (0.001 :: Double)] [] :: IO Double)
    putStrLn $ "✓ Reynolds number calculation: " ++ show re
    
    -- Test friction factor
    fd <- (PySim.call "fluids" "friction_factor" 
            [PySim.arg (1e5 :: Double), PySim.arg (0.0001 :: Double)] [] :: IO Double)
    putStrLn $ "✓ Friction factor calculation: " ++ show fd
    
    putStrLn "Basic tests completed successfully!\n"


-- Benchmark fluids functions
benchmarkFluids :: IO ()
benchmarkFluids = do
    putStrLn "Running benchmarks..."
    
    -- Benchmark friction factor
    putStrLn "\nBenchmarking friction_factor:"
    (t1, _) <- time $ do
        sequence $ replicate 1000 $ 
            (PySim.call "fluids" "friction_factor" 
                [PySim.arg (1e5 :: Double), PySim.arg (0.0001 :: Double)] [] :: IO Double)
    putStrLn $ printf "Time for 1000 friction_factor calls: %.6f seconds" t1
    putStrLn $ printf "Average time per call: %.6f seconds" (t1 / 1000)
    
-- Main function to run all tests
main :: IO ()
main = handle pyExceptionHandler $ do
    putStrLn "Running fluids tests from Haskell..."
    Py.initialize
    
    testFluids
    benchmarkFluids
    
    Py.finalize
    putStrLn "All tests completed!"
  where
    pyExceptionHandler :: PyExc.Exception -> IO ()
    pyExceptionHandler e = do
        putStrLn $ "Python error occurred: " ++ show e
        Py.finalize
