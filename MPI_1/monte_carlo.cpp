#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <sys/time.h>

#include <mpi.h>
#define min(a,b) (a < b ? a : b)

static double rtclock() {
    struct timeval Tp;
    int stat;

    stat = gettimeofday(&Tp, NULL);
    if (stat != 0)
        printf ("Error return from gettimeofday: %d", stat);

    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}


int* get_interval(int rank, int size, int point) {
    int* range = new int[2];

    int intervals_count = point % size;
    int interval_size = point / size;
    range[0] = interval_size * rank + min(rank, intervals_count);
    range[1] = interval_size * (rank + 1) + min(rank + 1, intervals_count);

    return range;
}


double frand(double a, double b) {
    return a + (b - a) * (rand() / double(RAND_MAX));
}

int do_walk(int a, int b, int x, double p, double& t) {
    int step = 0;
    while(x > a && x < b) {
        if(frand(0,1) < p) {
            x += 1;
        }
        else {
            x -= 1;
        }
        t += 1.0;
        step += 1;
    }
    return x;
}

void run_mc(int a, int b, int x, double p, int N, int rank, int size) {
    srand(2 * rank);
	
    double t = 0.0; 
    double w = 0.0; 
    int* proc_range;
    double* w_tmp = new double[size];
    double* t_tmp = new double[size];
    double w_result = 0.0;
    double t_result = 0.0;
    MPI_Status status;
    
    proc_range = get_interval(rank, size, N);
    for (int i = proc_range[0]; i < proc_range[1]; i++ ) {
        int out = do_walk(a, b, x, p, t);
        if (out == b)
            w += 1;
    }

    MPI_Gather(&w, 1, MPI_DOUBLE, w_tmp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&t, 1, MPI_DOUBLE, t_tmp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            w_result += w_tmp[i];
            t_result += t_tmp[i];
        }

        std::ofstream f("output.txt");
        f << w_result/N << " " << t_result/N << std::endl;
        f.close();
    }

    delete [] w_tmp;
    delete [] t_tmp;
    delete [] proc_range;
}

int main(int argc, char** argv) {
    int a = atoi(argv[1]);
    int b = atoi(argv[2]);
    int x = atoi(argv[3]);
    double p = atof(argv[4]);
    int N = atoi(argv[5]);
    double bench_t_start, bench_t_end;

    int proc_rank, proc_size;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
	
    if (proc_rank == 0) {
        bench_t_start = rtclock();
    }

    run_mc(a, b, x, p, N, proc_rank, proc_size);

    MPI_Barrier(MPI_COMM_WORLD);
    if (proc_rank == 0) {
        bench_t_end = rtclock();
        std::cout << "Elapsed time: " << bench_t_end - bench_t_start << " s\n";

        std::ofstream file("stat.txt");
        file << "[a, b] = [" << a << ", " << b << "]" << std::endl;
        file << "x = " << x << std::endl;
        file << "p = " << p << std::endl;
        file << "N = " << N << std::endl;
        file << "P = " << proc_size << std::endl;
        file << "Elapsed time: " << bench_t_end - bench_t_start << std::endl;
    }

    MPI_Finalize();

    return 0;
}
