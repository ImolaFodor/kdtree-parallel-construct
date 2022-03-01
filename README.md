# kdtree-parallel-construct
Assignment 2 for Foundations of High Performance Computing @ units. Construct of kdtree with OpenMP and OpenMPI. Performance modelling.

To execute code:
>>module load openmpi
>>mpicc -fopenmp -lm -g kdtree_build.c -o kdtree.x
>>export OMP_NUM_THREADS=2
>>mpirun -np 4
