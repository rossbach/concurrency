all: kmeans-pthreads

kmeans-pthreads: kmeans-pthreads.c info.h
	gcc -O3  kmeans-pthreads.c -lm -o kmeans-pthreads -pthread 

clean:
	rm kmeans-pthreads

