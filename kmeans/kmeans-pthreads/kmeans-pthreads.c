#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <sys/wait.h>
#include <signal.h>
#include <sched.h>
#include <sys/mman.h>
#include <pthread.h>
#include <sys/syscall.h>
#include "info.h"

#define DIMS 16
#define MAXWORKERS 32
#define MIN(a,b) ((a)<(b) ? (a) : (b))
#define gettid() syscall(__NR_gettid)

typedef struct {    /* A 2D vector */
  int cluster;
  float v[DIMS];
} POINT;


int     _d = DIMS;         /* dimensionality */
int     _k = 4;            /* Number of clusters */
double  _threshold = 0.05; /* Threshold for convergence */
char*   _inputname;        /* Input filename to read from */
POINT*  _centers;          /* Global array of centers */
POINT*  _ncenters;         /* Global temporary centers */
POINT*  _points;           /* Global array of 2D data points */
int     _numpoints;        /* Number of 2D data points */
pthread_barrier_t _barrierS;   /* sync start of all threads */
pthread_barrier_t _barrierE;   /* sync end of all threads */
pthread_barrier_t _barrierX;   /* for synchronization of procs across iterations */
pthread_barrier_t _barrierY;   /* for synchronization of procs across iterations */
pthread_mutex_t   _lock;       /* sync access to sums arrays at aggregation */
pthread_t *       _workers;    /* worker threads */
int *             _wids;       /* worker thread ids */
POINT*  _sums;             /* aggregation of all dimensions */
int*    _counts;           /* aggregation counter */
int*    _converged;        /* need shared variable for convergence test */
int     _numprocs;         /* in case the user provides a processor count */
int     _go = 0;           /* signal handler: parent->child: "GO!" */
int     _printres=1;
unsigned _startticks=0;
unsigned _stopticks=0;
int      _startlogged=0;
int      _stoplogged=0;

/*
 * Signal handler for go signal from parent
 */
void siggoproc(int x) {  
  _info("%d: WORKER GO SIGNAL!\n", gettid());
  _go=1;
}

/* 
 * Return a point at the given index
 */
POINT get_point(POINT * v, int idx) {
  return v[idx];
}

void accumulate_point(POINT* a, POINT* b) {
  int i;
  for(i=0;i<_d;i++)
    a->v[i] += b->v[i];
}

POINT div_point(POINT* a, int div) {
  int i;
  POINT p;
  for(i=0;i<_d;i++)
    p.v[i] = a->v[i] /= div;
  return p;
}

/*
 * Return a random center to be associated
 * with a cluster
 */
POINT  random_center(int cluster) {
    /* Vector *point = &_points[rand() % _numpoints];
       point->cluster = cluster; */

  return get_point(_points, rand() % _numpoints);
}

/*
 * Return a center at (0,0,...) to be associated
 * with a cluster
 */
void zero_center(POINT * point, int cluster) {
  int i;
  for(i=0;i<_d;i++)
    point->v[i] = 0.0f;
  point->cluster = cluster;
}

/*
 * Create the initial, random centers
 */
void init_centers() {
    int i;
    for (i = 0; i < _k; i++) {
      zero_center(&_centers[i],i);
      _ncenters[i] = random_center(i);
    }
}

/*
 * Find the nearest center for each point
 */
void find_nearest_center(POINT *centers, POINT *point) {
    double distance = DBL_MAX;
    int cluster_idx = 0;
    int i,j;
    for (i = 0; i < _k; i++) {
	double d = 0;
	for(j=0; j<_d; j++)
	  d += pow(centers[i].v[j] - point->v[j], 2.0);
	d = sqrt(d);
	if (d < distance) {
	    distance = d;
	    cluster_idx = i;
	} 
    }

    point->cluster = cluster_idx;
}


/*
 * Check if the centers have changed
 */
int centers_changed(POINT *centers) {
    int changed = 0;
    int i,j;
    for (i = 0; i < _k; i++) {
      for(j=0;j<_d;j++) {
	double diff = fabs(centers[i].v[j] - _centers[i].v[j]);
	if (diff > _threshold) {
	    changed = 1;
	}
	_centers[i].v[j] = centers[i].v[j];
      }
    }

    return changed;
}

void reset_accumulators(POINT * vp, int * c) {
  memset(vp, 0, sizeof(POINT)*_k);
  memset(c, 0, sizeof(int)*_k);  
}

/*
 * Compute k-means and print out the centers
 */
void kmeans(int id, int nprocs) {

    /* While the centers have moved, re-cluster 
	the points and compute the averages.
    */
    int max_itr = 10;
    int itr = 0;
    int i;
    int chunksize = _numpoints/nprocs;  
    int start = id * chunksize;
    int stop = MIN(((id+1)*chunksize)-1, _numpoints);
    POINT sums[_k];       /* per-thread aggregation of all dimensions */
    int   counts[_k];       /* per-thread aggregation counter */
    
    do {

      _info("#%d:%d top-loop\n", id, gettid());
      reset_accumulators(sums, counts);
      for(i=start; i<stop; i++) {
	find_nearest_center(_ncenters, &_points[i]);
	accumulate_point(&sums[_points[i].cluster], &_points[i]);
	counts[_points[i].cluster]++;	
      }

      /* need to aggregate local accumulators to find new 
	 centers. Canonically, we would do this with a lock/mutex.
	 The _lock variable uses system V IPC semaphores to implement
	 a coarse mutex that covers the _x_sums, _y_sums, _counts arrays.
	 A better implementation would use atomic instructions and potentially
	 SIMD instructions to admit more parallelism.
      */      
      _info("#%d:%d entering _lock wait on %d\n", id, gettid(), _lock);
      pthread_mutex_lock(&_lock);
      _info("#%d:%d holds _lock\n", id, gettid());
      for(i=0;i<_k;i++) {
	accumulate_point(&_sums[i], &sums[i]);
	_counts[i] += counts[i];
      }
      _info("#%d:%d releasing _lock\n", id, gettid());
      pthread_mutex_unlock(&_lock);

      /* now each thread has completed it's parallel work 
	 for this iteration ostensibly. The final aggregation and 
	 convergence test  requires that no straggling processor is 
	 still updating shared data, so we use a barrier, and perform
	 the final reduction and convergence test on a single process. 
	 use a second barrier wait after the test to ensure that no
	 process restarts the loop before the test completes.
      */
      _info("#%d:%d entering barrier wait 1k\n", id, gettid());      
      pthread_barrier_wait(&_barrierX);
      _info("#%d:%d exiting barrier wait 1k\n", id, gettid());      
      if(id==0) {
	for(i=0; i<_k; i++) 
	  _ncenters[i] = div_point(&_sums[i], _counts[i]);
	*_converged = ((!centers_changed(_ncenters)) || itr >= max_itr);
	reset_accumulators(_sums, _counts);
      }
      _info("#%d:%d entering barrier wait 2\n", id, gettid());      
      pthread_barrier_wait(&_barrierY);
      _info("#%d:%d exiting barrier wait 2\n", id, gettid());      
      itr++;

    } while(!(*_converged));

    if(id==0)
      _info("Converged in %d iterations (max=%d)\n", itr, max_itr);
}

/*
 * Read data points from the input file
 * By using mmap to allocate the points and centers, 
 * they are automatically mapped into the child address space
 * giving us the shared memory abstraction we typically want 
 * from threads.
 */
void
read_inputfile(char *inputname) {
  
    _centers = mmap(NULL,
		    sizeof(POINT) * _k,
		    PROT_READ|PROT_WRITE,
		    MAP_ANONYMOUS|MAP_SHARED,
		    -1, 0);      

    _ncenters = mmap(NULL,
		     sizeof(POINT) * _k,
		     PROT_READ|PROT_WRITE,
		     MAP_ANONYMOUS|MAP_SHARED,
		     -1, 0);

    _sums = mmap(NULL,
		 sizeof(POINT) * _k,
		 PROT_READ|PROT_WRITE,
		 MAP_ANONYMOUS|MAP_SHARED,
		 -1, 0);

    _counts = mmap(NULL,
		   sizeof(int) * _k,
		   PROT_READ|PROT_WRITE,
		   MAP_ANONYMOUS|MAP_SHARED,
		   -1, 0);

    _converged = mmap(NULL,
		      sizeof(int),
		      PROT_READ|PROT_WRITE,
		      MAP_ANONYMOUS|MAP_SHARED,
		      -1, 0);
    *_converged = 0;
    
    /* Open the input file */
    if (_inputname == NULL) {
	fprintf(stderr, "Must provide an input filename\n");
	free(_inputname);
	free(_centers);
	exit(EXIT_FAILURE);
    }
    
    FILE *inputfile = fopen(_inputname, "r");
    if (inputfile == NULL) {
	fprintf(stderr, "Invalid filename\n");
	free(_inputname);
	free(_centers);
	exit(EXIT_FAILURE);
    }

    /* Read the line count */
    char *line = NULL;
    size_t len = 0;
    ssize_t read = getline(&line, &len, inputfile);
    _numpoints = atoi(line);
    _points = mmap(NULL,
		   sizeof(POINT) * _numpoints,
		   PROT_READ|PROT_WRITE,
		   MAP_ANONYMOUS|MAP_SHARED,
		   -1, 0);      
    

    /* Read each data point in */
    while ((read = getline(&line, &len, inputfile)) != -1) {
        int k;
	char *saveptr;
	char *token;
	token = strtok_r(line, " ", &saveptr);
	int i = atoi(token) - 1;
	for(k=0; k<_d; k++) {	
	  token = strtok_r(NULL, " ", &saveptr);
	  _points[i].v[k] = atof(token);
	}
	_points[i].cluster = 0;
    }
    
    free(line);
    fclose(inputfile);
}

void *
kmeans_thread_proc(void * p) {

  int tid = *((int*)p);
  
  pthread_barrier_wait(&_barrierS);
  if(!_startlogged) {
    pthread_mutex_lock(&_lock);
    if(!_startlogged) 
      _startticks = ticks();
    _startlogged = 1;
    pthread_mutex_unlock(&_lock);
  }

  kmeans(tid, _numprocs);

  pthread_barrier_wait(&_barrierE);
  if(!_stoplogged) {
    pthread_mutex_lock(&_lock);
    if(!_stoplogged) 
      _stopticks = ticks();
    _stoplogged = 1;
    pthread_mutex_unlock(&_lock);
  }
  
  return NULL;
}

unsigned int
kmeans_workers(int numprocs) {

  int i=0;
  _wids = malloc(sizeof(int)*numprocs);
  _workers = malloc(sizeof(pthread_t)*numprocs);  
  for(i=0; i<numprocs; i++) {
    _wids[i] = i;
    if(pthread_create(&_workers[i], NULL, kmeans_thread_proc, &_wids[i])) {
      _error("cannot create pthread for worker %d\n", i);
      exit(1);
    }    
  }
  for(i=0; i<numprocs; i++)
    pthread_join(_workers[i], NULL);
  _info("all threads joined\n");
  free(_wids);
  free(_workers);
  return 0;  
}

void main (int argc, char *const *argv) {

    size_t len;
    int opt;
    unsigned ticks = 0;
    _numprocs = get_nprocs_conf();
    
    while ((opt = getopt(argc, argv, "k:t:i:vc:")) != -1) {
	switch (opt) {
	case 'k':
	    _k = atoi(optarg);
	    break;
	case 'p':
	  _printres = 0;
	  break;
	case 'c':
	  _numprocs = atoi(optarg);
	  break;
	case 't':
	    _threshold = atof(optarg);
	    break;
	case 'i':
	    len = strlen(optarg);
	    _inputname = (char*) malloc(len + 1);
	    strcpy(_inputname, optarg);
	    break;
	case 'v':
	    _verbose = 1;
	    break;
	default:
	    fprintf(stderr, "Usage: %s [-k clusters] [-t threshold]"
                            " [-i inputfile] -v (verbose)\n", argv[0]);
	    exit(EXIT_FAILURE);
	}
    }

    int numprocs = get_nprocs_conf();
    _numprocs = MIN(_numprocs, numprocs);
    
    read_inputfile(_inputname);
    init_centers(_ncenters);
    pthread_barrier_init(&_barrierS, NULL, _numprocs);
    pthread_barrier_init(&_barrierE, NULL, _numprocs);
    pthread_barrier_init(&_barrierX, NULL, _numprocs);
    pthread_barrier_init(&_barrierY, NULL, _numprocs);
    pthread_mutex_init(&_lock, NULL);
    kmeans_workers(_numprocs);
    ticks = _stopticks - _startticks;
    printf("parallel work completed in %d usec\n", ticks);

    /* Print the center of each cluster */
    if(_verbose||_printres) {
      int j,k;
      for (j = 0; j < _k; j++) {
	printf("Cluster %d center: ", j);
	for(k=0;k<_d;k++) {
	  printf("[%.2f]", _centers[j].v[k]);
	}
	printf("\n");
      }
    }
    

    free(_inputname);
    munmap(_centers, _k*sizeof(POINT));
    munmap(_ncenters, _k*sizeof(POINT));
    munmap(_points, _numpoints*sizeof(POINT));
    munmap(_sums, _k*sizeof(POINT));
    munmap(_counts, _k*sizeof(int));
    munmap(_converged, sizeof(int));
    pthread_barrier_destroy(&_barrierX);
    pthread_barrier_destroy(&_barrierY);
    pthread_mutex_destroy(&_lock);
    
    exit(EXIT_SUCCESS);
}
