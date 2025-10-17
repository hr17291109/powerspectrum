# g++ -O3 test.cpp -lfftw3f_omp -lfftw3f -o test.exe -fopenmp -lgsl -lgslcblas -I.
#g++ -O3 -I/home/nishimichi/local/include main.cpp -L/home/nishimichi/local/lib -lfftw3f_omp -lfftw3f -o measure_multipoles.exe -fopenmp -lgsl -lgslcblas -I.
g++ -DZSPACE -O3 halos_for.cpp -lfftw3f_omp -lfftw3f -o halos_for.exe -fopenmp -lgsl -lgslcblas -I.
