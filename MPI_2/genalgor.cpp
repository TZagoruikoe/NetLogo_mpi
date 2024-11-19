#include <iostream>
#include <cmath>
#include <mpi.h>

#define min(a,b) (a < b ? a : b)

double frand() { // вещественное случайное число в диапазоне [0,1)
	return double(rand()) / RAND_MAX;
}

int eval(int* a, int n) {
	int sum = 0;
	for (int i = 0; i < n; i++)
		sum += a[i];
	return sum;
}

void init(int* P, int m, int n) {
	for (int k = 0; k < m; k++)
		for (int i = 0; i < n; i++)
			P[k * n + i] = rand() % 2;
}

void shuffle(int* P, int m, int n) {
	for( int k = 0; k < m; k++ ) {
		int l = rand() % m;
		for(int i = 0; i < n; i++)
			std::swap(P[k * n + i], P[l * n + i]);
	}
}

void select(int* P, int m, int n) {
	double pwin = 0.95;
	shuffle(P, m, n);
	for(int k = 0; k < m / 2; k++) {
		int a = 2 * k;
		int b = 2 * k + 1;
		int fa = eval(P + a * n, n);
		int fb = eval(P + b * n, n);
		double p = frand();
		if ((fa < fb && p < pwin) || (fa > fb && p > pwin))
			for (int i = 0; i < n; i++)
				P[b * n + i] = P[a * n + i];
		else
			for (int i = 0; i < n; i++)
				P[a * n + i] = P[b * n + i];
	}
}

void crossover(int* P, int m, int n) {
	shuffle(P, m, n);
	for (int k = 0; k < m / 2; k++) {
		int a = 2 * k;
		int b = 2 * k + 1;
		int j = rand() % n;
		for (int i = j; i < n; i++)
			std::swap(P[a * n + i], P[b * n + i]);
	}
}

void mutate(int* P, int m, int n) {
	double pmut = 0.001;
	for (int k = 0; k < m; k++)
		for (int i = 0; i < n; i++)
			if (frand() < pmut)
				P[k * n + i] = 1 - P[k * n + i];
}

void printthebest(int* P, int m, int n, int rank, int size) {
	int sum = 0;

	//int k0 = -1;
	int f0 = 1000000000;
	for (int k = 0; k < m; k++) {
		int f = eval(P + k * n, n);
		if (f < f0) {
			f0 = f;
			//k0 = k;
		}
	}

    MPI_Allreduce(&f0, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
		std::cout << " Dependence of the average error: " << sum / size << std::endl;
    }
}

void migrate(int* P, int m, int n, int people, int* buf_tmp, int rank, int size) {
    shuffle(P, m, n);
    MPI_Request request[4];

    int left = (rank - 1 < 0) ? size - 1 : rank - 1;
    int right = (rank + 1 > size - 1) ? 0 : rank + 1;

    MPI_Isend(P, people * n, MPI_INT, left, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Isend(P + (m - people) * n, people * n, MPI_INT, right, 1, MPI_COMM_WORLD, &request[1]);
    MPI_Irecv(buf_tmp, people * n, MPI_INT, left, 1, MPI_COMM_WORLD, &request[2]);
    MPI_Irecv(buf_tmp + people * n, people * n, MPI_INT, right, 0, MPI_COMM_WORLD, &request[3]);

    MPI_Waitall(4, request, MPI_STATUS_IGNORE);

    memcpy(P, buf_tmp, people * n);
    memcpy(P + (m - people) * n, buf_tmp + people * n, people * n);
}

void runGA(int n, int m, int T, int people, int dt, int rank, int size) {
    int* range = new int[2];
    int intrv_count = m % size;
    int intrv_size = m / size;
    range[0] = intrv_size * rank + min(rank, intrv_count);
    range[1] = intrv_size * (rank + 1) + min(rank + 1, intrv_count);
    int diff = range[1] - range[0];

    int* buf_tmp = new int[2 * people * n];

	int* P = new int[diff * n];
	init(P, diff, n);
	for (int t = 0; t < T; t++) {
        if (t % dt == 0) {
            migrate(P, diff, n, people, buf_tmp, rank, size);
        }
		select(P, diff, n);
		crossover(P, diff, n);
		mutate(P, diff, n);
		if (rank == 0)
		    std::cout << "Iter --> " << t;
		printthebest(P, diff, n, rank, size);
	}

    delete[] range;
    delete[] buf_tmp;
	delete[] P;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand(rank);

	int n = 1000; // atoi(argv[1]);
	int m = 2000; // atoi(argv[2]);
	int T = 100; // atoi(argv[3]);
    int people = 100; // atoi(argv[4]);
    int dt = 3; // atoi(argv[5]);

	runGA(n, m, T, people, dt, rank, size);

    MPI_Finalize();
	return 0;
}