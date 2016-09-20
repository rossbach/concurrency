all: kmeans-pthreads

kmeans-pthreads: kmeans-pthreads.c info.h
	gcc  kmeans-pthreads.c -lm -o kmeans-pthreads -pthread -g

clean:
	rm kmeans-pthreads

