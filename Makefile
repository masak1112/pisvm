# D. Brugger, december 2006
# $Id: Makefile 573 2010-12-29 10:54:20Z dome $
CXXC = mpicxx
CC = mpicc
CFLAGS = -Wall -O3
# Use following flags for debugging/profiling
#CFLAGS = -Wall -pg -g
#CFLAGS = -Wall -O3 -pg
# Can be set to SOLVER_PSMO or SOLVER_LOQO
SOLVER = SOLVER_PSMO
# WARNING: This version is only tested with the SOLVER_PSMO, so i might have introduced changed that broke the LOQO solver!

# If you set SOLVER = SOLVER_LOQO you need a properly
# installed PLAPACK library and need to set the
# following. PLAPACK has to be patched to support
# manteuffel shifting when doing a Cholesky decomposition!
#PLAPACK_INCLUDE = -I/usr/local/PLAPACKR30/INCLUDE
#PLAPACK_LIB = -L/usr/local/PLAPACKR30 -lPLAPACK
# In addition PLAPACK needs a working blas library
#BLASLIB = -lblas
INCLUDE = $(PLAPACK_INCLUDE) -I.
LIBS = $(PLAPACK_LIB) $(BLASLIB) -lm

all: pisvm-train pisvm-predict pisvm-scale
# test_parallel_loqo test_manteuffel

pisvm-predict: pisvm-predict.cpp svm.o svm_cache.o svm_kernel.o svm_solver.o svm_solver_nu.o
	$(CXXC) $(CFLAGS) pisvm-predict.cpp svm.o svm_cache.o svm_kernel.o svm_solver.o svm_solver_nu.o -o pisvm-predict $(LIBS)
pisvm-train: pisvm-train.cpp svm.o svm_cache.o svm_kernel.o svm_solver.o svm_solver_nu.o
	$(CXXC) $(CFLAGS) pisvm-train.cpp svm.o svm_cache.o svm_kernel.o svm_solver.o svm_solver_nu.o -o pisvm-train $(LIBS)
pisvm-scale: pisvm-scale.cpp
	$(CXXC) $(CFLAGS) pisvm-scale.cpp -o pisvm-scale
svm.o: svm.cpp svm.h svm_cache.h svm_kernel.h svm_solver.h svm_solver_nu.h svm_q_kernels.h psmo/psmo_solver.cpp psmo/psmo_solver.h
	$(CXXC) $(CFLAGS) $(INCLUDE) -D $(SOLVER) -c svm.cpp
svm_cache.o: svm_cache.cpp svm_cache.h
	$(CXXC) $(CFLAGS) -c svm_cache.cpp
svm_kernel.o: svm_kernel.cpp svm_kernel.h
	$(CXXC) $(CFLAGS) -c svm_kernel.cpp
svm_solver.o: svm_solver.cpp svm_solver.h
	$(CXXC) $(CFLAGS) -c svm_solver.cpp
svm_solver_nu.o: svm_solver_nu.cpp svm_solver_nu.h svm_solver.h
	$(CXXC) $(CFLAGS) -c svm_solver_nu.cpp

clean:
	rm -f *~ *.o pisvm-train pisvm-predict pisvm-scale
