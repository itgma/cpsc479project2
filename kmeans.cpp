#include <iostream>
#include <mpi.h>
#include <time.h>
#include <cstdlib.h>

class Data {
	int x;
	int y;
}

// Can create random sample or input from file 
double euclidean_distance(double x1, double y1, double x2, double y2) {
	double x = x1 - x2; //calculating number to square in next step
	double y = y1 - y2;
	double dist;

	dist = pow(x, 2) + pow(y, 2);       //calculating Euclidean distance
	dist = sqrt(dist);

	return dist;
}


int main(int argc, char **argv) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size;)

	if (argc != 4 && rank == 0) {
		printf("Usage is as follows: \nmpirun -n 100 ./kmeans element_num cluster_num dimension_num");
		return 0;
	}

	int elements = atoi(argv[1]);
	int clusters = atoi(argv[2]);
	int dimensions = atoi(argv[3]);
}
