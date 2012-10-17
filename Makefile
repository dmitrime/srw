CC=g++
CXXFLAGS=-march=native -O2 -std=c++0x -fopenmp -I./alglib
CXXFLAGS_D=-march=native -g -Wall -std=c++0x -fopenmp -I./alglib
TARGETA=SRW
LIBS=-lm

srw: 
	$(CC) -o $(TARGETA) $(CXXFLAGS) srw.cpp alglib/*.o $(LIBS)
srwd: 
	$(CC) -o $(TARGETA) $(CXXFLAGS_D) srw.cpp alglib/*.o $(LIBS)
clean:
	rm -f $(TARGETA) 
