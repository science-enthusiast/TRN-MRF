OBJS = dualSys.o newtonTest.o ICFS.o myUtils.o quasiNewton.o
CC = g++
DEBUG = -g
DFLAGS = -ggdb
CFLAGS = -Wall -std=c++0x -O2 -O3 -DNDEBUG -g -I /home/hari/libraries/eigen -I /home/hari/libraries/opengm-master/include -I/home/hari/libraries/opengm-master/src/external/AD3-patched/ad3/ -fopenmp -c
LFLAGS = -Wall -std=c++0x -O2 -O3 -DNDEBUG -g -fopenmp

LIBS = -L/usr/local/lib/ -L/home/hari/libraries/opengm-master/src/external/AD3-patched/ad3/ -lad3 

#CFLAGS = -Wall -std=c++0x -I /home/hari/libraries/eigen -I /home/hari/libraries/opengm-master/include -I/home/hari/libraries/opengm-master/src/external/AD3-patched/ad3/ -fopenmp -c $(DEBUG) $(DFLAGS)
#LFLAGS = -Wall -std=c++0x -fopenmp $(DEBUG) $(DFLAGS)

#CFLAGS = -Wall -std=c++0x -I/home/hari/libraries/eigen -fopenmp -c
#LFLAGS = -Wall -std=c++0x -fopenmp

newtonTest : $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) $(LIBS) -o newtonTest -lhdf5

myUtils.o : myUtils.cpp myUtils.hpp
	$(CC) $(CFLAGS) myUtils.cpp

dualSys.o : dualSys.cpp dualSys.hpp myUtils.hpp ICFS.h
	$(CC) $(CFLAGS) dualSys.cpp

ICFS.o : ICFS.cpp ICFS.h
	$(CC) $(CFLAGS) ICFS.cpp

quasiNewton.o: quasiNewton.cpp quasiNewton.hpp
	$(CC) $(CFLAGS) quasiNewton.cpp

newtonTest.o : newtonTest.cpp dualSys.hpp myUtils.hpp
	$(CC) $(CFLAGS) newtonTest.cpp

clean:
	\rm *.o
