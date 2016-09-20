all: kmeans-serial kmeans-forkjoin

kmeans-serial: kmeans-serial.c
	gcc kmeans-serial.c -lm -o kmeans-serial 

kmeans-forkjoin: kmeans-forkjoin.c sysvbarrier.h info.h
	gcc kmeans-forkjoin.c -lm -o kmeans-forkjoin 

clean:
	rm kmeans-serial kmeans-forkjoin

