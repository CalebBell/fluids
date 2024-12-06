CC=gcc
PYTHON_VERSION=$(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_INCLUDES=$(shell python3-config --includes)
PYTHON_LDFLAGS=$(shell python3-config --ldflags)
PYTHON_LIBS=$(shell python3-config --libs)

# Add -lpython3.11 explicitly and move LDFLAGS to end of command
CFLAGS=-Wall -O3 $(PYTHON_INCLUDES)
LDFLAGS=$(PYTHON_LDFLAGS) $(PYTHON_LIBS) -lpython$(PYTHON_VERSION)

fluids_test: fluids_test.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f fluids_test
