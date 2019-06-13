default: all

stlbenchmark: stlbenchmark.cpp
	g++ -fopenmp-simd -O3 stlbenchmark.cpp -I${TBBINCDIR} -I${PSTLINCDIR} -L${TBBLIBDIR} -ltbb -o stlbenchmark

eigenbenchmark: eigenbenchmark.cpp
	g++ -fopenmp-simd -O3 eigenbenchmark.cpp -I${EIGENINCDIR} -o eigenbenchmark

pipelinegraph: mypipelinetest.cpp
	g++ -std=c++17 -fopenmp-simd -O3 mypipelinetest.cpp -I${TBBINCDIR} -I${PSTLINCDIR} -L${TBBLIBDIR} -ltbb -o mytest

all: stlbenchmark eigenbenchmark pipelinegraph

clean:
	rm stlbenchmark eigenbenchmark mytest