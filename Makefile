OBJS = dual_sys.o newtonTest.o ICFS.o myUtils.o quasiNewton.o solve_scd.o
CC = g++
DEBUG = -g
DFLAGS = -ggdb
CFLAGS = -Wall -std=c++14 -O2 -O3 -DNDEBUG -g -I/home/hari/libraries/eigen -fopenmp -c
LFLAGS = -Wall -std=c++14 -O2 -O3 -DNDEBUG -g -fopenmp

LIBS = -L/usr/local/lib/ 

#CFLAGS = -Wall -std=c++0x -I /home/hari/libraries/eigen -fopenmp -c $(DEBUG) $(DFLAGS)
#LFLAGS = -Wall -std=c++0x -fopenmp $(DEBUG) $(DFLAGS)

#CFLAGS = -Wall -std=c++0x -I/home/hari/libraries/eigen -fopenmp -c
#LFLAGS = -Wall -std=c++0x -fopenmp

newtonTest : $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) $(LIBS) -o newtonTest

myUtils.o : myUtils.cpp myUtils.hpp
	$(CC) $(CFLAGS) myUtils.cpp

dual_sys.o : dual_sys.cpp dual_sys.h myUtils.hpp ICFS.h
	$(CC) $(CFLAGS) dual_sys.cpp

ICFS.o : ICFS.cpp ICFS.h
	$(CC) $(CFLAGS) ICFS.cpp

solve_scd.o : solvers/solve_scd.cpp solvers/solve_scd.h
	$(CC) $(CFLAGS) solvers/solve_scd.cpp

quasiNewton.o : quasiNewton.cpp quasiNewton.hpp
	$(CC) $(CFLAGS) quasiNewton.cpp

newtonTest.o : newtonTest.cpp dual_sys.h myUtils.hpp solvers/solve_scd.h
	$(CC) $(CFLAGS) newtonTest.cpp

clean:
	\rm *.o
