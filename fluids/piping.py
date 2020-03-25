# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

from __future__ import division
from math import pi
from fluids.constants import inch

__all__ = ['nearest_pipe', 'gauge_from_t', 't_from_gauge', 'wire_schedules']

# Schedules 5, 10, 20, 30, 40, 60, 80, 100, 120, 140, 160 from
# ASME B36.10M - Welded and Seamless Wrought Steel Pipe
# All schedule lists stored in mm, other than NPS.
# i = inner diameter, o = outer diameter, and t = wall thickness in variable names

NPS5 = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 30]
S5i = [18, 23.4, 30.1, 38.9, 45, 57, 68.78, 84.68, 97.38, 110.08, 135.76, 162.76, 213.56, 266.2, 315.88, 347.68, 398.02, 448.62, 498.44, 549.44, 598.92, 749.3]
S5o = [21.3, 26.7, 33.4, 42.2, 48.3, 60.3, 73, 88.9, 101.6, 114.3, 141.3, 168.3, 219.1, 273, 323.8, 355.6, 406.4, 457, 508, 559, 610, 762]
S5t = [1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 2.11, 2.11, 2.11, 2.11, 2.77, 2.77, 2.77, 3.4, 3.96, 3.96, 4.19, 4.19, 4.78, 4.78, 5.54, 6.35]

NPS10 = [0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
S10i = [7.82, 10.4, 13.8, 17.08, 22.48, 27.86, 36.66, 42.76, 54.76, 66.9, 82.8, 95.5, 108.2, 134.5, 161.5, 211.58, 264.62, 314.66, 342.9, 393.7, 444.3, 495.3, 546.3, 597.3, 644.16, 695.16, 746.16, 797.16, 848.16, 898.16]
S10o = [10.3, 13.7, 17.1, 21.3, 26.7, 33.4, 42.2, 48.3, 60.3, 73, 88.9, 101.6, 114.3, 141.3, 168.3, 219.1, 273, 323.8, 355.6, 406.4, 457, 508, 559, 610, 660, 711, 762, 813, 864, 914]
S10t = [1.24, 1.65, 1.65, 2.11, 2.11, 2.77, 2.77, 2.77, 2.77, 3.05, 3.05, 3.05, 3.05, 3.4, 3.4, 3.76, 4.19, 4.57, 6.35, 6.35, 6.35, 6.35, 6.35, 6.35, 7.92, 7.92, 7.92, 7.92, 7.92, 7.92]

NPS20 = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
S20i = [206.4, 260.3, 311.1, 339.76, 390.56, 441.16, 488.94, 539.94, 590.94, 634.6, 685.6, 736.6, 787.6, 838.6, 888.6]
S20o = [219.1, 273, 323.8, 355.6, 406.4, 457, 508, 559, 610, 660, 711, 762, 813, 864, 914]
S20t = [6.35, 6.35, 6.35, 7.92, 7.92, 7.92, 9.53, 9.53, 9.53, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7]

NPS30 = [0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 30, 32, 34, 36]
S30i = [7.4, 10, 13.4, 16.48, 21.88, 27.6, 36.26, 41.94, 53.94, 63.44, 79.34, 92.04, 104.74, 205.02, 257.4, 307.04, 336.54, 387.34, 434.74, 482.6, 533.6, 581.46, 679.24, 730.24, 781.24, 832.24, 882.24]
S30o = [10.3, 13.7, 17.1, 21.3, 26.7, 33.4, 42.2, 48.3, 60.3, 73, 88.9, 101.6, 114.3, 219.1, 273, 323.8, 355.6, 406.4, 457, 508, 559, 610, 711, 762, 813, 864, 914]
S30t = [1.45, 1.85, 1.85, 2.41, 2.41, 2.9, 2.97, 3.18, 3.18, 4.78, 4.78, 4.78, 4.78, 7.04, 7.8, 8.38, 9.53, 9.53, 11.13, 12.7, 12.7, 14.27, 15.88, 15.88, 15.88, 15.88, 15.88]

NPS40 = [0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 32, 34, 36]
S40i = [6.84, 9.22, 12.48, 15.76, 20.96, 26.64, 35.08, 40.94, 52.48, 62.68, 77.92, 90.12, 102.26, 128.2, 154.08, 202.74, 254.46, 303.18, 333.34, 381, 428.46, 477.82, 575.04, 778.04, 829.04, 875.9]
S40o = [10.3, 13.7, 17.1, 21.3, 26.7, 33.4, 42.2, 48.3, 60.3, 73, 88.9, 101.6, 114.3, 141.3, 168.3, 219.1, 273, 323.8, 355.6, 406.4, 457, 508, 610, 813, 864, 914]
S40t = [1.73, 2.24, 2.31, 2.77, 2.87, 3.38, 3.56, 3.68, 3.91, 5.16, 5.49, 5.74, 6.02, 6.55, 7.11, 8.18, 9.27, 10.31, 11.13, 12.7, 14.27, 15.09, 17.48, 17.48, 17.48, 19.05]

NPS60 = [8, 10, 12, 14, 16, 18, 20, 22, 24]
S60i = [198.48, 247.6, 295.26, 325.42, 373.08, 418.9, 466.76, 514.54, 560.78]
S60o = [219.1, 273, 323.8, 355.6, 406.4, 457, 508, 559, 610]
S60t = [10.31, 12.7, 14.27, 15.09, 16.66, 19.05, 20.62, 22.23, 24.61]

NPS80 = [0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
S80i = [5.48, 7.66, 10.7, 13.84, 18.88, 24.3, 32.5, 38.14, 49.22, 58.98, 73.66, 85.44, 97.18, 122.24, 146.36, 193.7, 242.82, 288.84, 317.5, 363.52, 409.34, 455.62, 501.84, 548.08]
S80o = [10.3, 13.7, 17.1, 21.3, 26.7, 33.4, 42.2, 48.3, 60.3, 73, 88.9, 101.6, 114.3, 141.3, 168.3, 219.1, 273, 323.8, 355.6, 406.4, 457, 508, 559, 610]
S80t = [2.41, 3.02, 3.2, 3.73, 3.91, 4.55, 4.85, 5.08, 5.54, 7.01, 7.62, 8.08, 8.56, 9.53, 10.97, 12.7, 15.09, 17.48, 19.05, 21.44, 23.83, 26.19, 28.58, 30.96]

NPS100 = [8, 10, 12, 14, 16, 18, 20, 22, 24]
S100i = [188.92, 236.48, 280.92, 307.94, 354.02, 398.28, 442.92, 489.14, 532.22]
S100o = [219.1, 273, 323.8, 355.6, 406.4, 457, 508, 559, 610]
S100t = [15.09, 18.26, 21.44, 23.83, 26.19, 29.36, 32.54, 34.93, 38.89]

NPS120 = [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
S120i = [92.04, 115.9, 139.76, 182.58, 230.12, 273, 300.02, 344.48, 387.14, 431.8, 476.44, 517.96]
S120o = [114.3, 141.3, 168.3, 219.1, 273, 323.8, 355.6, 406.4, 457, 508, 559, 610]
S120t = [11.13, 12.7, 14.27, 18.26, 21.44, 25.4, 27.79, 30.96, 34.93, 38.1, 41.28, 46.02]

NPS140 = [8, 10, 12, 14, 16, 18, 20, 22, 24]
S140i = [177.86, 222.2, 266.64, 292.1, 333.34, 377.66, 419.1, 463.74, 505.26]
S140o = [219.1, 273, 323.8, 355.6, 406.4, 457, 508, 559, 610]
S140t = [20.62, 25.4, 28.58, 31.75, 36.53, 39.67, 44.45, 47.63, 52.37]

NPS160 = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
S160i = [11.74, 15.58, 20.7, 29.5, 34.02, 42.82, 53.94, 66.64, 87.32, 109.54, 131.78, 173.08, 215.84, 257.16, 284.18, 325.42, 366.52, 407.98, 451.04, 490.92]
S160o = [21.3, 26.7, 33.4, 42.2, 48.3, 60.3, 73, 88.9, 114.3, 141.3, 168.3, 219.1, 273, 323.8, 355.6, 406.4, 457, 508, 559, 610]
S160t = [4.78, 5.56, 6.35, 6.35, 7.14, 8.74, 9.53, 11.13, 13.49, 15.88, 18.26, 23.01, 28.58, 33.32, 35.71, 40.49, 45.24, 50.01, 53.98, 59.54]

# Schedules designated STD, XS, and XXS
NPSSTD = [0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
STDi = [6.84, 9.22, 12.48, 15.76, 20.96, 26.64, 35.08, 40.94, 52.48, 62.68, 77.92, 90.12, 102.26, 128.2, 154.08, 202.74, 254.46, 304.74, 336.54, 387.34, 437.94, 488.94, 539.94, 590.94, 640.94, 691.94, 742.94, 793.94, 844.94, 894.94, 945.94, 996.94, 1047.94, 1098.94, 1148.94, 1199.94]
STDo = [10.3, 13.7, 17.1, 21.3, 26.7, 33.4, 42.2, 48.3, 60.3, 73, 88.9, 101.6, 114.3, 141.3, 168.3, 219.1, 273, 323.8, 355.6, 406.4, 457, 508, 559, 610, 660, 711, 762, 813, 864, 914, 965, 1016, 1067, 1118, 1168, 1219]
STDt = [1.73, 2.24, 2.31, 2.77, 2.87, 3.38, 3.56, 3.68, 3.91, 5.16, 5.49, 5.74, 6.02, 6.55, 7.11, 8.18, 9.27, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53]

NPSXS = [0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
XSi = [5.48, 7.66, 10.7, 13.84, 18.88, 24.3, 32.5, 38.14, 49.22, 58.98, 73.66, 85.44, 97.18, 122.24, 146.36, 193.7, 247.6, 298.4, 330.2, 381, 431.6, 482.6, 533.6, 584.6, 634.6, 685.6, 736.6, 787.6, 838.6, 888.6, 939.6, 990.6, 1041.6, 1092.6, 1142.6, 1193.6]
XSo = [10.3, 13.7, 17.1, 21.3, 26.7, 33.4, 42.2, 48.3, 60.3, 73, 88.9, 101.6, 114.3, 141.3, 168.3, 219.1, 273, 323.8, 355.6, 406.4, 457, 508, 559, 610, 660, 711, 762, 813, 864, 914, 965, 1016, 1067, 1118, 1168, 1219]
XSt = [2.41, 3.02, 3.2, 3.73, 3.91, 4.55, 4.85, 5.08, 5.54, 7.01, 7.62, 8.08, 8.56, 9.53, 10.97, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7]

NPSXXS = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12]
XXSi = [6.36, 11.06, 15.22, 22.8, 28, 38.16, 44.96, 58.42, 80.06, 103.2, 124.4, 174.64, 222.2, 273]
XXSo = [21.3, 26.7, 33.4, 42.2, 48.3, 60.3, 73, 88.9, 114.3, 141.3, 168.3, 219.1, 273, 323.8]
XXSt = [7.47, 7.82, 9.09, 9.7, 10.15, 11.07, 14.02, 15.24, 17.12, 19.05, 21.95, 22.23, 25.4, 25.4]

NPSS5 = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 30]
SS5DN = [15, 20, 25, 32, 40, 50, 65, 80, 90, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 750]
SS5i = [18, 23.4, 30.1, 38.9, 45, 57, 68.78, 84.68, 97.38, 110.08, 135.76, 162.76, 213.56, 266.3, 315.98, 347.68, 398.02, 448.62, 498.44, 549.44, 598.92, 749.3]
SS5o = [21.3, 26.7, 33.4, 42.2, 48.3, 60.3, 73, 88.9, 101.6, 114.3, 141.3, 168.3, 219.1, 273.1, 323.9, 355.6, 406.4, 457, 508, 559, 610, 762]
SS5t = [1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 2.11, 2.11, 2.11, 2.11, 2.77, 2.77, 2.77, 3.4, 3.96, 3.96, 4.19, 4.19, 4.78, 4.78, 5.54, 6.35]

# Schedules 10, 40 and 80 from ASME B36.19M - Stainless Steel Pipe
NPSS10 = [0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 30]
SS10DN = [6, 8, 10, 15, 20, 25, 32, 40, 50, 65, 80, 90, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 750]
SS10i = [7.82, 10.4, 13.8, 17.08, 22.48, 27.86, 36.66, 42.76, 54.76, 66.9, 82.8, 95.5, 108.2, 134.5, 161.5, 211.58, 264.72, 314.76, 346.04, 396.84, 447.44, 496.92, 547.92, 597.3, 746.16]
SS10o = [10.3, 13.7, 17.1, 21.3, 26.7, 33.4, 42.2, 48.3, 60.3, 73, 88.9, 101.6, 114.3, 141.3, 168.3, 219.1, 273.1, 323.9, 355.6, 406.4, 457, 508, 559, 610, 762]
SS10t = [1.24, 1.65, 1.65, 2.11, 2.11, 2.77, 2.77, 2.77, 2.77, 3.05, 3.05, 3.05, 3.05, 3.4, 3.4, 3.76, 4.19, 4.57, 4.78, 4.78, 4.78, 5.54, 5.54, 6.35, 7.92]

NPSS40 = [0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24]
SS40DN = [6, 8, 10, 15, 20, 25, 32, 40, 50, 65, 80, 90, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 600]
SS40i = [6.84, 9.22, 12.48, 15.76, 20.96, 26.64, 35.08, 40.94, 52.48, 62.68, 77.92, 90.12, 102.26, 128.2, 154.08, 202.74, 254.56, 304.84, 336.54, 387.34, 437.94, 488.94, 590.94]
SS40o = [10.3, 13.7, 17.1, 21.3, 26.7, 33.4, 42.2, 48.3, 60.3, 73, 88.9, 101.6, 114.3, 141.3, 168.3, 219.1, 273.1, 323.9, 355.6, 406.4, 457, 508, 610]
SS40t = [1.73, 2.24, 2.31, 2.77, 2.87, 3.38, 3.56, 3.68, 3.91, 5.16, 5.49, 5.74, 6.02, 6.55, 7.11, 8.18, 9.27, 9.53, 9.53, 9.53, 9.53, 9.53, 9.53]

NPSS80 = [0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24]
SS80DN = [6, 8, 10, 15, 20, 25, 32, 40, 50, 65, 80, 90, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 600]
SS80i = [5.48, 7.66, 10.7, 13.84, 18.88, 24.3, 32.5, 38.14, 49.22, 58.98, 73.66, 85.44, 97.18, 122.24, 146.36, 193.7, 247.7, 298.5, 330.2, 381, 431.6, 482.6, 584.6]
SS80o = [10.3, 13.7, 17.1, 21.3, 26.7, 33.4, 42.2, 48.3, 60.3, 73, 88.9, 101.6, 114.3, 141.3, 168.3, 219.1, 273.1, 323.9, 355.6, 406.4, 457, 508, 610]
SS80t = [2.41, 3.02, 3.2, 3.73, 3.91, 4.55, 4.85, 5.08, 5.54, 7.01, 7.62, 8.08, 8.56, 9.53, 10.97, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7]

schedule_lookup = { '40': (NPS40, S40i, S40o, S40t),
                    '5': (NPS5, S5i, S5o, S5t),
                    '10': (NPS10, S10i, S10o, S10t),
                    '20': (NPS20, S20i, S20o, S20t),
                    '30': (NPS30, S30i, S30o, S30t),
                    '60': (NPS60, S60i, S60o, S60t),
                    '80': (NPS80, S80i, S80o, S80t),
                    '100': (NPS100, S100i, S100o, S100t),
                    '120': (NPS120, S120i, S120o, S120t),
                    '140': (NPS140, S140i, S140o, S140t),
                    '160': (NPS160, S160i, S160o, S160t),
                    'STD': (NPSSTD, STDi, STDo, STDt),
                    'XS': (NPSXS, XSi, XSo, XSt),
                    'XXS': (NPSXXS, XXSi, XXSo, XXSt),
                    '5S': (NPSS5, SS5i, SS5o, SS5t),
                    '10S': (NPSS10, SS10i, SS10o, SS10t),
                    '40S': (NPSS40, SS40i, SS40o, SS40t),
                    '80S': (NPSS80, SS80i, SS80o, SS80t)}


def nearest_pipe(Do=None, Di=None, NPS=None, schedule='40'):
    r'''Searches for and finds the nearest standard pipe size to a given
    specification. Acceptable inputs are:

    - Nominal pipe size
    - Nominal pipe size and schedule
    - Outer diameter `Do`
    - Outer diameter `Do` and schedule
    - Inner diameter `Di`
    - Inner diameter `Di` and schedule

    Acceptable schedules are: '5', '10', '20', '30', '40', '60', '80', '100',
    '120', '140', '160', 'STD', 'XS', 'XXS', '5S', '10S', '40S', '80S'.

    Parameters
    ----------
    Do : float
        Pipe outer diameter, [m]
    Di : float
        Pipe inner diameter, [m]
    NPS : float
        Nominal pipe size, [-]
    schedule : str
        String representing schedule size

    Returns
    -------
    NPS : float
        Nominal pipe size, [-]
    Di : float
        Pipe inner diameter, [m]
    Do : float
        Pipe outer diameter, [m]
    t : float
        Pipe wall thickness, [m]

    Notes
    -----
    Internal units within this function are mm.
    The imperial schedules are not quite identical to these value, but
    all rounding differences happen in the sub-0.1 mm level.

    Examples
    --------
    >>> nearest_pipe(Di=0.021)
    (1, 0.02664, 0.0334, 0.0033799999999999998)
    >>> nearest_pipe(Do=.273, schedule='5S')
    (10, 0.26630000000000004, 0.2731, 0.0034)

    References
    ----------
    .. [1] American National Standards Institute, and American Society of
       Mechanical Engineers. B36.10M-2004: Welded and Seamless Wrought Steel
       Pipe. New York: American Society of Mechanical Engineers, 2004.
    .. [2] American National Standards Institute, and American Society of
       Mechanical Engineers. B36-19M-2004: Stainless Steel Pipe.
       New York, N.Y.: American Society of Mechanical Engineers, 2004.
    '''
    if Di: 
        Di *= 1E3
    if Do: 
        Do *= 1E3
    if NPS: 
        NPS = float(NPS)

    def Di_lookup(Di, NPSes, Dis, Dos, ts):
        for i in range(len(Dis)): # Go up ascending list; once larger than specified, return
            if Dis[-1] < Di:
                return None
            if Dis[i] >= Di:
                _nps, _di, _do, _t = NPSes[i], Dis[i], Dos[i], ts[i]
                return (_nps, _di, _do, _t)
        raise Exception('Di lookup failed')

    def Do_lookup(Do, NPSes, Dis, Dos, ts):
        for i in range(len(Dos)): # Go up ascending list; once larger than specified, return
            if Dos[-1] < Do:
                return None
            if Dos[i] >= Do:
                _nps, _di, _do, _t = NPSes[i], Dis[i], Dos[i], ts[i]
                return (_nps, _di, _do, _t)
        raise Exception('Do lookup failed')

    def NPS_lookup(NPS, NPSes, Dis, Dos, ts):
        for i in range(len(NPSes)): # Go up ascending list; once larger than specified, return
            if NPSes[i] == NPS:
                _nps, _di, _do, _t = NPSes[i], Dis[i], Dos[i], ts[i]
                return (_nps, _di, _do, _t)
        raise Exception('NPS not in list')


    # If accidentally given an numerical schedule, convert it to a string
    schedule_type = type(schedule)
    if schedule_type in (int, float):
        schedule = str(int(schedule))
    
    if schedule not in schedule_lookup:
        raise ValueError('Schedule not recognized')
    else:
        NPSes, Dis, Dos, ts = schedule_lookup[schedule]
    
    # Handle the three cases of different inputs
    if Di:
        nums = Di_lookup(Di, NPSes, Dis, Dos, ts)
    elif Do:
        nums = Do_lookup(Do, NPSes, Dis, Dos, ts)
    elif NPS:
        nums = NPS_lookup(NPS, NPSes, Dis, Dos, ts)

    if nums is None:
        raise ValueError('Pipe input is larger than max of selected schedule')
    _nps, _di, _do, _t = nums
    return _nps, _di/1E3, _do/1E3, _t/1E3


### Wire gauge schedules

# Stub's Steel Wire Gage
SSWG_integers = list(range(1, 81))
SSWG_inch = [0.227, 0.219, 0.212, 0.207, 0.204, 0.201, 0.199, 0.197, 0.194,
             0.191, 0.188, 0.185, 0.182, 0.18, 0.178, 0.175, 0.172, 0.168,
             0.164, 0.161, 0.157, 0.155, 0.153, 0.151, 0.148, 0.146, 0.143,
             0.139, 0.134, 0.127, 0.12, 0.115, 0.112, 0.11, 0.108, 0.106,
             0.103, 0.101, 0.099, 0.097, 0.095, 0.092, 0.088, 0.085, 0.081,
             0.079, 0.077, 0.075, 0.072, 0.069, 0.066, 0.063, 0.058, 0.055,
             0.05, 0.045, 0.042, 0.041, 0.04, 0.039, 0.038, 0.037, 0.036,
             0.035, 0.033, 0.032, 0.031, 0.03, 0.029, 0.027, 0.026, 0.024,
             0.023, 0.022, 0.02, 0.018, 0.016, 0.015, 0.014, 0.013]
#SSWG_SI = [round(i*inch, 7) for i in SSWG_inch] # 7 decimals for equal conversion
SSWG_SI = [0.0057658, 0.0055626, 0.0053848, 0.0052578, 0.0051816, 0.0051054, 
           0.0050546, 0.0050038, 0.0049276, 0.0048514, 0.0047752, 0.004699, 
           0.0046228, 0.004572, 0.0045212, 0.004445, 0.0043688, 0.0042672, 
           0.0041656, 0.0040894, 0.0039878, 0.003937, 0.0038862, 0.0038354,
           0.0037592, 0.0037084, 0.0036322, 0.0035306, 0.0034036, 0.0032258, 
           0.003048, 0.002921, 0.0028448, 0.002794, 0.0027432, 0.0026924,
           0.0026162, 0.0025654, 0.0025146, 0.0024638, 0.002413, 0.0023368,
           0.0022352, 0.002159, 0.0020574, 0.0020066, 0.0019558, 0.001905, 
           0.0018288, 0.0017526, 0.0016764, 0.0016002, 0.0014732, 0.001397,
           0.00127, 0.001143, 0.0010668, 0.0010414, 0.001016, 0.0009906, 
           0.0009652, 0.0009398, 0.0009144, 0.000889, 0.0008382, 0.0008128, 
           0.0007874, 0.000762, 0.0007366, 0.0006858, 0.0006604, 0.0006096, 
           0.0005842, 0.0005588, 0.000508, 0.0004572, 0.0004064, 0.000381,
           0.0003556, 0.0003302]


# British Standard Wire Gage (Imperial Wire Gage)
BSWG_integers = [0.143, .167, 0.2, 0.25, 0.33, 0.5] + list(range(51))
BSWG_inch = [0.5, 0.464, 0.432, 0.4, 0.372, 0.348, 0.324, 0.3, 0.276, 0.252, 0.232,
        0.212, 0.192, 0.176, 0.16, 0.144, 0.128, 0.116, 0.104, 0.092, 0.08,
        0.072, 0.064, 0.056, 0.048, 0.04, 0.036, 0.032, 0.028, 0.024, 0.022,
        0.02, 0.018, 0.0164, 0.0149, 0.0136, 0.0124, 0.0116, 0.0108, 0.01,
        0.0092, 0.0084, 0.0076, 0.0068, 0.006, 0.0052, 0.0048, 0.0044, 0.004,
        0.0036, 0.0032, 0.0028, 0.0024, 0.002, 0.0016, 0.0012, 0.001]
#BSWG_SI = [round(i*inch,8) for i in BSWG_inch] # 8 decimals for equal conversion
BSWG_SI = [0.0127, 0.0117856, 0.0109728, 0.01016, 0.0094488, 0.0088392, 
           0.0082296, 0.00762, 0.0070104, 0.0064008, 0.0058928, 0.0053848,
           0.0048768, 0.0044704, 0.004064, 0.0036576, 0.0032512, 0.0029464,
           0.0026416, 0.0023368, 0.002032, 0.0018288, 0.0016256, 0.0014224,
           0.0012192, 0.001016, 0.0009144, 0.0008128, 0.0007112, 0.0006096,
           0.0005588, 0.000508, 0.0004572, 0.00041656, 0.00037846, 0.00034544,
           0.00031496, 0.00029464, 0.00027432, 0.000254, 0.00023368,
           0.00021336, 0.00019304, 0.00017272, 0.0001524, 0.00013208,
           0.00012192, 0.00011176, 0.0001016, 9.144e-05, 8.128e-05, 7.112e-05,
           6.096e-05, 5.08e-05, 4.064e-05, 3.048e-05, 2.54e-05]


# Music Wire Gauge
MWG_integers = [45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 
                29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,
                13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0.5, 0.33, 0.25,
                0.2, 0.167]
#MWG_integers = [.167, 0.2, 0.25, 0.33, 0.5] + list(range(46))
#MWG_inch = [0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012,
#            0.013, 0.014, 0.016, 0.018, 0.02, 0.022, 0.024, 0.026, 0.029,
#            0.031, 0.033, 0.035, 0.037, 0.039, 0.041, 0.043, 0.045, 0.047,
#            0.049, 0.051, 0.055, 0.059, 0.063, 0.067, 0.071, 0.075, 0.08,
#            0.085, 0.09, 0.095, 0.1, 0.106, 0.112, 0.118, 0.124, 0.13, 0.138,
#            0.146, 0.154, 0.162, 0.17, 0.18]
#MWG_SI = [round(i*inch,7) for i in MWG_inch] # 7 decimals for equal conversion
# Scale gets bigger instead of smaller; reverse for convenience
#MWG_integers.reverse()
#MWG_inch.reverse()

MWG_inch = [0.18, 0.17, 0.162, 0.154, 0.146, 0.138, 0.13, 0.124, 0.118, 0.112,
            0.106, 0.1, 0.095, 0.09, 0.085, 0.08, 0.075, 0.071, 0.067, 0.063,
            0.059, 0.055, 0.051, 0.049, 0.047, 0.045, 0.043, 0.041, 0.039, 
            0.037, 0.035, 0.033, 0.031, 0.029, 0.026, 0.024, 0.022, 0.02,
            0.018, 0.016, 0.014, 0.013, 0.012, 0.011, 0.01, 0.009, 0.008,
            0.007, 0.006, 0.005, 0.004]
#MWG_SI.reverse()
MWG_SI = [0.004572, 0.004318, 0.0041148, 0.0039116, 0.0037084, 0.0035052,
          0.003302, 0.0031496, 0.0029972, 0.0028448, 0.0026924, 0.00254, 
          0.002413, 0.002286, 0.002159, 0.002032, 0.001905, 0.0018034, 
          0.0017018, 0.0016002, 0.0014986, 0.001397, 0.0012954, 0.0012446,
          0.0011938, 0.001143, 0.0010922, 0.0010414, 0.0009906, 0.0009398,
          0.000889, 0.0008382, 0.0007874, 0.0007366, 0.0006604, 0.0006096,
          0.0005588, 0.000508, 0.0004572, 0.0004064, 0.0003556, 0.0003302,
          0.0003048, 0.0002794, 0.000254, 0.0002286, 0.0002032, 0.0001778, 
          0.0001524, 0.000127, 0.0001016]

# Steel Wire Gage -Also Washburn & Moen gage, American Steel gage;
# Wire Co.gage;  Roebling Wire Gages.
SWG_integers = [0.143, .167, 0.2, 0.25, 0.33, 0.5] + list(range(51))
SWG_inch = [0.49, 0.4615, 0.4305, 0.3938, 0.3625, 0.331, 0.3065, 0.283, 0.2625,
            0.2437, 0.2253, 0.207, 0.192, 0.177, 0.162, 0.1483, 0.135, 0.1205,
            0.1055, 0.0915, 0.08, 0.072, 0.0625, 0.054, 0.0475, 0.041, 0.0348,
            0.0318, 0.0286, 0.0258, 0.023, 0.0204, 0.0181, 0.0173, 0.0162,
            0.015, 0.014, 0.0132, 0.0128, 0.0118, 0.0104, 0.0095, 0.009,
            0.0085, 0.008, 0.0075, 0.007, 0.0066, 0.0062, 0.006, 0.0058,
            0.0055, 0.0052, 0.005, 0.0048, 0.0046, 0.0044]
#SWG_SI = [round(i*inch,8) for i in SWG_inch] # 8 decimals for equal conversion
SWG_SI = [0.012446, 0.0117221, 0.0109347, 0.01000252, 0.0092075, 0.0084074,
          0.0077851, 0.0071882, 0.0066675, 0.00618998, 0.00572262, 0.0052578,
          0.0048768, 0.0044958, 0.0041148, 0.00376682, 0.003429, 0.0030607,
          0.0026797, 0.0023241, 0.002032, 0.0018288, 0.0015875, 0.0013716, 
          0.0012065, 0.0010414, 0.00088392, 0.00080772, 0.00072644, 0.00065532, 
          0.0005842, 0.00051816, 0.00045974, 0.00043942, 0.00041148, 0.000381,
          0.0003556, 0.00033528, 0.00032512, 0.00029972, 0.00026416, 0.0002413, 
          0.0002286, 0.0002159, 0.0002032, 0.0001905, 0.0001778, 0.00016764,
          0.00015748, 0.0001524, 0.00014732, 0.0001397, 0.00013208, 0.000127,
          0.00012192, 0.00011684, 0.00011176]

# American Wire or Brown & Sharpe Gage
AWG_integers = [.167, 0.2, 0.25, 0.33, 0.5] + list(range(51))
AWG_inch = [0.58, 0.5165, 0.46, 0.4096, 0.3648, 0.3249, 0.2893, 0.2576, 0.2294,
            0.2043, 0.1819, 0.162, 0.1443, 0.1285, 0.1144, 0.1019, 0.0907,
            0.0808, 0.072, 0.0641, 0.0571, 0.0508, 0.0453, 0.0403, 0.0359,
            0.032, 0.0285, 0.0253, 0.0226, 0.0201, 0.0179, 0.0159, 0.0142,
            0.0126, 0.0113, 0.01, 0.00893, 0.00795, 0.00708, 0.0063, 0.00561,
            0.005, 0.00445, 0.00396, 0.00353, 0.00314, 0.0028, 0.00249,
            0.00222, 0.00198, 0.00176, 0.00157, 0.0014, 0.00124, 0.00111,
            0.00099]
#AWG_SI = [round(i*inch,9) for i in AWG_inch] # 9 decimals for equal conversion
AWG_SI = [0.014732, 0.0131191, 0.011684, 0.01040384, 0.00926592, 0.00825246,
          0.00734822, 0.00654304, 0.00582676, 0.00518922, 0.00462026, 
          0.0041148, 0.00366522, 0.0032639, 0.00290576, 0.00258826, 0.00230378,
          0.00205232, 0.0018288, 0.00162814, 0.00145034, 0.00129032,
          0.00115062, 0.00102362, 0.00091186, 0.0008128, 0.0007239, 0.00064262, 
          0.00057404, 0.00051054, 0.00045466, 0.00040386, 0.00036068, 
          0.00032004, 0.00028702, 0.000254, 0.000226822, 0.00020193, 
          0.000179832, 0.00016002, 0.000142494, 0.000127, 0.00011303,
          0.000100584, 8.9662e-05, 7.9756e-05, 7.112e-05, 6.3246e-05, 
          5.6388e-05, 5.0292e-05, 4.4704e-05, 3.9878e-05, 3.556e-05, 
          3.1496e-05, 2.8194e-05, 2.5146e-05]

# Birmingham or Stub's Iron Wire Gage
BWG_integers = [0.2, 0.25, 0.33, 0.5] + list(range(37))
BWG_inch = [0.5, 0.454, 0.425, 0.38, 0.34, 0.3, 0.284, 0.259, 0.238, 0.22,
            0.203, 0.18, 0.165, 0.148, 0.134, 0.12, 0.109, 0.095, 0.083,
            0.072, 0.065, 0.058, 0.049, 0.042, 0.035, 0.032, 0.028, 0.025,
            0.022, 0.02, 0.018, 0.016, 0.014, 0.013, 0.012, 0.01, 0.009,
            0.008, 0.007, 0.005, 0.004]
#BWG_SI = [round(i*inch,6) for i in BWG_inch]
BWG_SI = [0.0127, 0.011532, 0.010795, 0.009652, 0.008636, 0.00762, 0.007214,
          0.006579, 0.006045, 0.005588, 0.005156, 0.004572, 0.004191, 0.003759,
          0.003404, 0.003048, 0.002769, 0.002413, 0.002108, 0.001829, 0.001651,
          0.001473, 0.001245, 0.001067, 0.000889, 0.000813, 0.000711, 0.000635,
          0.000559, 0.000508, 0.000457, 0.000406, 0.000356, 0.00033, 0.000305,
          0.000254, 0.000229, 0.000203, 0.000178, 0.000127, 0.000102]

wire_schedules = {'BWG': (BWG_integers, BWG_inch, BWG_SI, True),
                 'AWG': (AWG_integers, AWG_inch, AWG_SI, True),
                 'SWG': (SWG_integers, SWG_inch, SWG_SI, True),
                 'MWG': (MWG_integers, MWG_inch, MWG_SI, False),
                 'BSWG': (BSWG_integers, BSWG_inch, BSWG_SI, True),
                 'SSWG': (SSWG_integers, SSWG_inch, SSWG_SI, True)}


def gauge_from_t(t, SI=True, schedule='BWG'):
    r'''Looks up the gauge of a given wire thickness of given schedule.
    Values are all non-linear, and tabulated internally.

    Parameters
    ----------
    t : float
        Thickness, [m]
    SI : bool, optional
        If False, requires that the thickness is given in inches not meters
    schedule : str
        Gauge schedule, one of 'BWG', 'AWG', 'SWG', 'MWG', 'BSWG', or 'SSWG'

    Returns
    -------
    gauge : float-like
        Wire Gauge, [-]

    Notes
    -----
    An internal variable, tol, is used in the selection of the wire gauge. If
    the next smaller wire gauge is within 10% of the difference between it and
    the previous wire gauge, the smaller wire gauge is selected. Accordingly,
    this function can return a gauge with a thickness smaller than desired
    in some circumstances.

    * Birmingham Wire Gauge (BWG) ranges from 0.2 (0.5 inch) to 36 (0.004 inch).
    * American Wire Gauge (AWG) ranges from 0.167 (0.58 inch) to 51 (0.00099
      inch). These are used for electrical wires.
    * Steel Wire Gauge (SWG) ranges from 0.143 (0.49 inch) to 51 (0.0044 inch).
      Also called Washburn & Moen wire gauge, American Steel gauge, Wire Co.
      gauge, and Roebling wire gauge.
    * Music Wire Gauge (MWG) ranges from 0.167 (0.004 inch) to 46 (0.18
      inch). Also called Piano Wire Gauge.
    * British Standard Wire Gage (BSWG) ranges from 0.143 (0.5 inch) to
      51 (0.001 inch). Also called Imperial Wire Gage (IWG).
    * Stub's Steel Wire Gage (SSWG) ranges from 1 (0.227 inch) to 80 (0.013 inch)

    Examples
    --------
    >>> gauge_from_t(.5, SI=False, schedule='BWG')
    0.2

    References
    ----------
    .. [1] Oberg, Erik, Franklin D. Jones, and Henry H. Ryffel. Machinery's
       Handbook. Industrial Press, Incorporated, 2012.
    '''
    tol = 0.1
    # Handle units
    if SI:
        t_inch = round(t/inch, 9) # all schedules are in inches
    else:
        t_inch = t

    # Get the schedule
    try:
        sch_integers, sch_inch, sch_SI, decreasing = wire_schedules[schedule]
    except:
        raise ValueError('Wire gauge schedule not found')

    # Check if outside limits
    sch_max, sch_min = sch_inch[0], sch_inch[-1]
    if t_inch > sch_max:
        raise ValueError('Input thickness is above the largest in the selected schedule')


    # If given thickness is exactly in the index, be happy
    if t_inch in sch_inch:
        gauge = sch_integers[sch_inch.index(t_inch)]

    else:
        for i in range(len(sch_inch)):
            if sch_inch[i] >= t_inch:
                larger = sch_inch[i]
            else:
                break
        if larger == sch_min:
            gauge = sch_min # If t is under the lowest schedule, be happy
        else:
            smaller = sch_inch[i]
            if (t_inch - smaller) <= tol*(larger - smaller):
                gauge = sch_integers[i]
            else:
                gauge = sch_integers[i-1]
    return gauge


def t_from_gauge(gauge, SI=True, schedule='BWG'):
    r'''Looks up the thickness of a given wire gauge of given schedule.
    Values are all non-linear, and tabulated internally.

    Parameters
    ----------
    gauge : float-like
        Wire Gauge, []
    SI : bool, optional
        If False, will return a thickness in inches not meters
    schedule : str
        Gauge schedule, one of 'BWG', 'AWG', 'SWG', 'MWG', 'BSWG', or 'SSWG'

    Returns
    -------
    t : float
        Thickness, [m]

    Notes
    -----
    
    * Birmingham Wire Gauge (BWG) ranges from 0.2 (0.5 inch) to 36 (0.004 inch).
    * American Wire Gauge (AWG) ranges from 0.167 (0.58 inch) to 51 (0.00099
      inch). These are used for electrical wires.
    * Steel Wire Gauge (SWG) ranges from 0.143 (0.49 inch) to 51 (0.0044 inch).
      Also called Washburn & Moen wire gauge, American Steel gauge, Wire Co.
      gauge, and Roebling wire gauge.
    * Music Wire Gauge (MWG) ranges from 0.167 (0.004 inch) to 46 (0.18
      inch). Also called Piano Wire Gauge.
    * British Standard Wire Gage (BSWG) ranges from 0.143 (0.5 inch) to
      51 (0.001 inch). Also called Imperial Wire Gage (IWG).
    * Stub's Steel Wire Gage (SSWG) ranges from 1 (0.227 inch) to 80 (0.013 inch)

    Examples
    --------
    >>> t_from_gauge(.2, False, 'BWG')
    0.5

    References
    ----------
    .. [1] Oberg, Erik, Franklin D. Jones, and Henry H. Ryffel. Machinery's
       Handbook. Industrial Press, Incorporated, 2012.
    '''
    try:
        sch_integers, sch_inch, sch_SI, decreasing = wire_schedules[schedule]
    except:
        raise ValueError("Wire gauge schedule not found; supported gauges are \
'BWG', 'AWG', 'SWG', 'MWG', 'BSWG', and 'SSWG'.")

    try:
        i = sch_integers.index(gauge)
    except:
        raise ValueError('Input gauge not found in selected schedule')
    if SI:
        return sch_SI[i] # returns thickness in m
    else:
        return sch_inch[i] # returns thickness in inch

