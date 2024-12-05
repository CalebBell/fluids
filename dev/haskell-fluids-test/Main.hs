-- module Main where
-- 
-- main :: IO ()
-- main = putStrLn "Hello, Haskell!"
{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified CPython as Py
import qualified CPython.Simple as PySim

main :: IO ()
main = do
    putStrLn "Testing fluids from Haskell..."
    -- Initialize Python
    Py.initialize
    
    -- Basic test
    re <- PySim.call "fluids" "Reynolds" 
            [PySim.arg (2.5 :: Double), PySim.arg (0.1 :: Double), 
             PySim.arg (1000.0 :: Double), PySim.arg (0.001 :: Double)] []
    
    putStrLn $ "Reynolds number calculation: " ++ show (re :: Double)
    
    -- Clean up
    Py.finalize
