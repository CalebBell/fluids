import sys
import os
if sys.version_info.major != 3 and sys.version_info.minor != 7:
	raise ValueError("This prerelease script will only run on Python 3.7")

import fluids
main_dir = fluids.fluids_dir
os.chdir(os.path.join(main_dir, 'tests'))
os.system("python3 make_test_stubs.py")

