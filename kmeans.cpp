#include <iostream>
#include <mpi.h>
#include <time.h>
#include <cstdlib.h>
#include<unistd.h>
#include<math.h>
#include<errno.h>
#define MAX_ITERATIONS 1000

using namespace std;
int numOfClusters = 0;
int numOfElements = 0;
int num_of_processes = 0;


class Data {
	int x;
	int y;
};
double random_d() //generate random number
 {
     static bool need_random = true;

    if(need_random)
    {
        srand(static_cast<unsigned int>( time(NULL)) );
        need_random = false;
    }

    int n = (rand() % 12) + 1;
    if((rand() % 100) >= 90) n = 1;
    double a = 0;
    double b = 0;
    for(long long i = 0, j = 0; i < n; i++)
    {
        j = (rand() % 10);
        b = b * 10 + j;
    }
    std::streamsize input_precision = n;
    a += (b / std:: pow(10, input_precision));

    if((rand() % 100) >= 60) a = (-a);
    return a;
 }
// Can create random sample or input from file 
double euclidean_distance(double x1, double y1, double x2, double y2) {
	double x = x1 - x2; //calculating number to square in next step
	double y = y1 - y2;
	double dist;

// Heavily referencing https://github.com/rexdwyer/MPI-K-means-clustering/blob/master/kmeans.c

// Stop checking when the centroids move less than the threshold
#define THRESHOLD 0.00001


// Distance between two individual sites
double euclidean_distance(double *, double *, int);

// Assigns a site to its proper cluster 
// Calculate the distance to each cluster from the site
// Takes the cluster with the lowest distance
int assign_site(double *, double *, int, int);

// Adds a site to a vector of all sites 
void add_site(double *, double *, int);


/* This function goes through that data points and assigns them to a cluster */
void assign2Cluster(double k_x[], double k_y[], double recv_x[], double recv_y[], int assign[])
{
	double min_dist = 10000000;
	double x=0, y=0, temp_dist=0;
	int k_min_index = 0;

	for(int i = 0; i < (numOfElements/num_of_processes) + 1; i++)
	{
		for(int j = 0; j < numOfClusters; j++)
		{
			x = abs(recv_x[i] - k_x[j]);
			y = abs(recv_y[i] - k_y[j]);
			temp_dist = sqrt((x*x) + (y*y));

			// new minimum distance found
			if(temp_dist < min_dist)
			{
				min_dist = temp_dist;
				k_min_index = j;
			}
		}

		// update the cluster assignment of this data points
		assign[i] = k_min_index;
	}

}

/* Recalcuate k-means of each cluster because each data point may have
   been reassigned to a new cluster for each iteration of the algorithm */
void calcKmeans(double k_means_x[], double k_means_y[], double data_x_points[], double data_y_points[], int k_assignment[])
{
	double total_x = 0;
	double total_y = 0;
	int numOfpoints = 0;

	for(int i = 0; i < numOfClusters; i++)
	{
		total_x = 0;
		total_y = 0;
		numOfpoints = 0;

		for(int j = 0; j < numOfElements; j++)
		{
			if(k_assignment[j] == i)
			{
				total_x += data_x_points[j];
				total_y += data_y_points[j];
				numOfpoints++;
			}
		}

		if(numOfpoints != 0)
		{
			k_means_x[i] = total_x / numOfpoints;
			k_means_y[i] = total_y / numOfpoints;
		}
	}

}

int main(int argc, char **argv) {
	 cout.setf(ios::fixed); // You do it only once

	double x = random_d();
	cout.precision(precision_d(x)); 
	cout << x << endl;

	double y = random_d();
	cout.precision(precision_d(y));
	cout << y << endl;
	
	// int rank, size;
	// MPI_Init(&argc, &argv);
	// MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// MPI_Comm_size(MPI_COMM_WORLD, &size;)

	// if (argc != 4 && rank == 0) {
		// printf("Usage is as follows: \nmpirun -n 100 ./kmeans element_num cluster_num dimension_num");
		// return 0;
	// }

	// int elements = atoi(argv[1]);
	// int clusters = atoi(argv[2]);
	// int dimensions = atoi(argv[3]);
	// initialize the MPI environment
	MPI_Init(NULL, NULL);

	// get number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// get rank
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// send buffers
	double *k_means_x = NULL;		// k means corresponding x values
	double *k_means_y = NULL;		// k means corresponding y values
	int *k_assignment = NULL;		// each data point is assigned to a cluster
	double *data_x_points = NULL;
	double *data_y_points = NULL;

	// receive buffer
	double *recv_x = NULL;
	double *recv_y = NULL;
	int *recv_assign = NULL;

	if(world_rank == 0)
	{
		if(argc != 2)
		{
			printf("Please include an argument after the program name to list how many processes.\n");
			printf("e.g. To indicate 4 processes, run: mpirun -n 4 ./kmeans 4\n");
			exit(-1);
		}

		num_of_processes = atoi(argv[1]);

		char buffer[2];
		printf("How many clusters would you like to analyze for? ");
		scanf("%s", buffer);
		printf("\n");

		numOfClusters = atoi(buffer);
		printf("Ok %d clusters it is.\n", numOfClusters);


		// broadcast the number of clusters to all nodes
		MPI_Bcast(&numOfClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// allocate memory for arrays
		k_means_x = (double *)malloc(sizeof(double) * numOfClusters);
		k_means_y = (double *)malloc(sizeof(double) * numOfClusters);

		if(k_means_x == NULL || k_means_y == NULL)
		{
			perror("malloc");
			exit(-1);
		}

		printf("Reading input data from file...\n\n");

		FILE* fp = fopen("input.txt", "r");

		if(!fp)
		{
			perror("fopen");
			exit(-1);
		}

		// count number of lines to find out how many elements
		int c = 0;
		numOfElements = 0;
		while(!feof(fp))
		{
			c = fgetc(fp);
			if(c == '\n')
			{
				numOfElements++;
			}
		}

		printf("There are a total number of %d elements in the file.\n", numOfElements);

		// broadcast the number of elements to all nodes
		MPI_Bcast(&numOfElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// allocate memory for an array of data points
		data_x_points = (double *)malloc(sizeof(double) * numOfElements);
		data_y_points = (double *)malloc(sizeof(double) * numOfElements);
		k_assignment = (int *)malloc(sizeof(int) * numOfElements);

		if(data_x_points == NULL || data_y_points == NULL || k_assignment == NULL)
		{
			perror("malloc");
			exit(-1);
		}

		// reset file pointer to origin of file
		fseek(fp, 0, SEEK_SET);

		// now read in points and fill the arrays
		int i = 0;

		double point_x=0, point_y=0;

		while(fscanf(fp, "%lf %lf", &point_x, &point_y) != EOF)
		{
			data_x_points[i] = point_x;
			data_y_points[i] = point_y;

			// assign the initial k means to zero
			k_assignment[i] = 0;
			i++;
		}

		// close file pointer
		fclose(fp);

		// randomly select initial k-means
		time_t t;
		srand((unsigned) time(&t));
		int random;
		for(int i = 0; i < numOfClusters; i++) {
			random = rand() % numOfElements;
			k_means_x[i] = data_x_points[random];
			k_means_y[i] = data_y_points[random];
		}

		printf("Running k-means algorithm for %d iterations...\n\n", MAX_ITERATIONS);
		for(int i = 0; i < numOfClusters; i++)
		{
			printf("Initial K-means: (%f, %f)\n", k_means_x[i], k_means_y[i]);
		}

		// allocate memory for receive buffers
		recv_x = (double *)malloc(sizeof(double) * ((numOfElements/num_of_processes) + 1));
		recv_y = (double *)malloc(sizeof(double) * ((numOfElements/num_of_processes) + 1));
		recv_assign = (int *)malloc(sizeof(int) * ((numOfElements/num_of_processes) + 1));

		if(recv_x == NULL || recv_y == NULL || recv_assign == NULL)
		{
			perror("malloc");
			exit(-1);
		}

// Creates random sites
// All values are from 0-1
double * rand_sites(int);

int main(int argc, char **argv) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size;)

	if (argc != 4 && rank == 0) {
		printf("Usage is as follows: \nmpirun -n 100 ./kmeans elements_per_cluster cluster_num dimension_num");
		return 0;

	}
	else
	{	// I am a worker node

		num_of_processes = atoi(argv[1]);

		// receive broadcast of number of clusters
		MPI_Bcast(&numOfClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// receive broadcast of number of elements
		MPI_Bcast(&numOfElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// allocate memory for arrays
		k_means_x = (double *)malloc(sizeof(double) * numOfClusters);
		k_means_y = (double *)malloc(sizeof(double) * numOfClusters);

		if(k_means_x == NULL || k_means_y == NULL)
		{
			perror("malloc");
			exit(-1);
		}

		// allocate memory for receive buffers
		recv_x = (double *)malloc(sizeof(double) * ((numOfElements/num_of_processes) + 1));
		recv_y = (double *)malloc(sizeof(double) * ((numOfElements/num_of_processes) + 1));
		recv_assign = (int *)malloc(sizeof(int) * ((numOfElements/num_of_processes) + 1));

		if(recv_x == NULL || recv_y == NULL || recv_assign == NULL)
		{
			perror("malloc");
			exit(-1);
		}
	}

	/* Distribute the work among all nodes. The data points itself will stay constant and
	   not change for the duration of the algorithm. */
	MPI_Scatter(data_x_points, (numOfElements/num_of_processes) + 1, MPI_DOUBLE,
		recv_x, (numOfElements/num_of_processes) + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Scatter(data_y_points, (numOfElements/num_of_processes) + 1, MPI_DOUBLE,
		recv_y, (numOfElements/num_of_processes) + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	int count = 0;
	while(count < MAX_ITERATIONS)
	{
		// broadcast k-means arrays
		MPI_Bcast(k_means_x, numOfClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(k_means_y, numOfClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// scatter k-cluster assignments array
		MPI_Scatter(k_assignment, (numOfElements/num_of_processes) + 1, MPI_INT,
			recv_assign, (numOfElements/num_of_processes) + 1, MPI_INT, 0, MPI_COMM_WORLD);

		// assign the data points to a cluster
		assign2Cluster(k_means_x, k_means_y, recv_x, recv_y, recv_assign);

		// gather back k-cluster assignments
		MPI_Gather(recv_assign, (numOfElements/num_of_processes)+1, MPI_INT,
			k_assignment, (numOfElements/num_of_processes)+1, MPI_INT, 0, MPI_COMM_WORLD);

		// let the root process recalculate k means
		if(world_rank == 0)
		{
			calcKmeans(k_means_x, k_means_y, data_x_points, data_y_points, k_assignment);
			//printf("Finished iteration %d\n",count);
		}

		count++;
	}

	if(world_rank == 0)
	{
		printf("--------------------------------------------------\n");
		printf("FINAL RESULTS:\n");
		for(int i = 0; i < numOfClusters; i++)
		{
			printf("Cluster #%d: (%f, %f)\n", i, k_means_x[i], k_means_y[i]);
		}
		printf("--------------------------------------------------\n");
	}

	// deallocate memory and clean up
	free(k_means_x);
	free(k_means_y);
	free(data_x_points);
	free(data_y_points);
	free(k_assignment);
	free(recv_x);
	free(recv_y);
	free(recv_assign);


	//MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Finalize();

	int elements_per = atoi(argv[1]);
	int cluster_num = atoi(argv[2]);
	int dimension_num = atoi(argv[3]);

	// Seed the random generator 
	srand(time(NULL));

	// Declaring buffers for both individual process and the process handler - process 0
	// First will be for process handler
	// Second for individual processes
	// Matrixes are declared as a single array b/c the dimensionality is variable


	// All sites possible 
	// Matrix but accessed and parsed like an array
	double * all_sites = NULL;
	double * sites = sites = new double[elements_per * dimension_num];

	// Sum of site distances assigned to each cluster
	// Matrix but accessed and parsed like an array
	double * all_sums = NULL;
	double * sums = new double[cluster_num * dimension_num];

	// Number of sites assigned to a cluster
	int * all_site_count = NULL;
	int * site_count = new int[cluster_num];

	// Cluster assignments for each site
	int * all_assignments = NULL;
	int * cluster_assignment = new int[elements_per];

	// All centroids. Will be broadcasted later
	// Matrix but accessed and parsed like an array
	double * centroids = new double[cluster_num * dimension_num];

	if (rank == 0) {
		// Size is number of processes
		all_sites = rand_sites(elements_per * dimension_num * size);
		// initial seeding for clusters is the first every other site (k times)
		// hopefully a little more random than just the first k sites
		for (int i = 0; i < cluster_num * dimension_num; i++) {
			centroids[i] = all_sites[i * 2];
		}
		// TODO: Print centroids? If we want to
		all_site_count = new int[cluster_num];
		all_sums = new double[cluster_num * dimension_num];
		all_assignments = new int[elements_per * size];
	}

	// Scatter all_sites to each process's sites 
	MPI_Scatter(all_sites, elements_per * dimension_num, MPI_DOUBLE, sites, elements_per * dimension_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Measures the distance the new centroids are from the old centroids
	double norm = 1.0;
	while (norm > THRESHOLD) {
		// Broadcast all centroids to processes to compare and adjust

		// Reset all sum and site counts to 0
		for (int i = 0; i < cluster_num; i++) {
			site_count[i] = 0;
		}
		for (int i = 0; i < cluster_num * dimension_num) {
			sums[i] = 0;
		}

		double * site = sites;
		// Increment site by dimension to get the next site 
		for (int i = 0; i < elements_per; i++, site += d) {
			int cluster = assign_site(site, centroids, cluster_num, dimension_num);
			site_count[cluster] += 1;
			add_site(site, &sums[cluster * dimension_num], dimension_num)
		}

		MPI_Reduce(sums, all_sums, cluster_num * dimension_num, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(site_count, all_site_count, cluster_num, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		// Compute the new centroids by dividing sums 
		// Compute how much the centroids have moved
		if (rank == 0) {
			for (int i = 0; i < cluster_num; i++) {
				for (int j = 0; j < dimension_num; j++) {
					all_sums[dimension_num * i + j] /= all_site_count[i];
				}
			}

			// Calculate how much the centroids have changed
			// all_sums now holds the location for new centroids
			// centroids holds the location for old centroids
			// Compare the distance between the two to see if they've moved
			norm = euclidean_distance(all_sums, centroids, cluster_num * dimension_num);

			// New centroids are in all_sums
			// Copy all coordinates from all_sums to centroids
			for (int i = 0; i < cluster_num * dimension_num; i++) {
				centroids[i] = all_sums[i];
			}
		}
		// Broadcast the new locations of centroids 
		MPI_Bcast(&norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	// New centroids have been calculated
	// Assign labels - Tell each site which centroid they belong to
	// Done locally in each process
	site = sites;
	for (int i = 0; i < elements_per; i++) {
		cluster_assignment[i] = assign_site(site, centroids, cluster_num, dimension_num);
	}

	// Gather all labels into the master process - p0
	MPI_Gather(cluster_assignment, elements_per, MPI_INT, all_assignments, elements_per, MPI_INT, 0, MPI_COMM_WORLD);
	// TODO: Print out new clusters
	// TODO: Print out all sites and labels
	MPI_Finalize();
	return 0;
}

double euclidean_distance(double * site1, double * site2, int dimension) {
	double distance = 0;
	double difference = 0;
	for (int i = 0; i < dimension; i++) {
		difference = site1[i] - site2[i];
		distance += difference * difference;
	}
	return distance;
}

int assign_site(double * site, double * centroids, int cluster_num, int dimension) {
	int assigned_cluster = 0;
	// Start lowest distance on the first cluster
	double low_distance = euclidean_distance(site, centroids, dimension);
	// Increment by dimension to get the next centroid
	double * current_centroid = centroids + dimension;

	for (int i = 1; i < cluster_num, i++, current_centroid += dimension) {
		double distance = euclidean_distance(site, current_centroid, dimension);
		if (distance < low_distance) {
			assigned_cluster = i;
			low_distance = distance;
		}
	}
	return assigned_cluster;
}

void add_site(double * site, double * sums, int dimension) {
	for (int i = 0; i < dimension; i++) {
		sums[i] += site[i];
	}
}

double * rand_sites(int total_elements) {
	double * all_elements = new double[total_elements];
	double range = 5;
	for (int i = 0; i < total_elements; i++) {
		// rand_num is 0-1
		double rand_num = (double)rand() / (double) RAND_MAX; 
		// all elements are now between 0-5
		all_elements[i] = rand_num * range;
	}
	return all_elements;

}
