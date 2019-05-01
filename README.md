# cpsc479project2
Topic: Something related to data science with the use of either mpi, openmp, cuda(preferably mpi for simplicity)
MKmeans algorithm
Input: number of clusters K, number of data objects N
Output: K centroids
1: MPI_INIT// start the procedure;
2: Read N objects from the file;
3: Partition N data objects evenly among all
processes, and assume that each process has N’
data objects;
4: For each process, install steps 5-11;
5: Randomly select K points as the initial cluster
centroids, denoted as μk (1İkİK);
6: Calculate J in Formula (1), denoted as J’;
7: Assign each object n (1İnİN) to the closest
cluster;
8: Calculate the new centroid of each cluster μk in
Formula (2) ;
9: Recalculate J in Formula (1);
10:Repeat steps 6-9 until J’- J < threshold;
11:Generate the cluster id for each data object;
12:Generate new cluster centroids according to the
clustering results of all processes at the end of
each iteration;
13:Generate a final centroid set Centroid by
Function Merge and output the clustering results:
K centroids;
14:MPI_FINALIZE// finish the procedure;

Main steps of the K-means algorithm are described by
Jain and Dubes [11] as follows:
a) Select an initial partition with K clusters; repeat
steps b and c until cluster memberships stabilize.
b) Generate a new partition by assigning each pattern
to its closest cluster center.
c) Compute new cluster centers. 
