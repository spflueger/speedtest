default: all

stlbenchmark: stlbenchmark.cpp
	g++ -fopenmp-simd -O3 stlbenchmark.cpp -I${TBBINCDIR} -I${PSTLINCDIR} -L${TBBLIBDIR} -ltbb -o stlbenchmark

eigenbenchmark: eigenbenchmark.cpp
	g++ -fopenmp-simd -O3 eigenbenchmark.cpp -I${EIGENINCDIR} -o eigenbenchmark

all: stlbenchmark eigenbenchmark

clean:
	rm stlbenchmark eigenbenchmark