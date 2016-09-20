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
#include "info.h"
#include "sysvbarrier.h"

#define DIMS 16
#define MAXWORKERS 32
#define MIN(a,b) ((a)<(b) ? (a) : (b))

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
BARRIER _barrierX;         /* for synchronization of procs across iterations */
BARRIER _barrierY;         /* for synchronization of procs across iterations */
LOCK    _lock;             /* sync access to sums arrays at aggregation */
POINT*  _sums;             /* aggregation of all dimensions */
int*    _counts;           /* aggregation counter */
int*    _converged;        /* need shared variable for convergence test */
int     _numprocs;         /* in case the user provides a processor count */
int     _go = 0;           /* signal handler: parent->child: "GO!" */

/*
 * Signal handler for go signal from parent
 */
void siggoproc(int x) {  
  _info("%d: WORKER GO SIGNAL!\n", getpid());
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

      _info("#%d:%d top-loop\n", id, getpid());
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
      _info("#%d:%d entering _lock wait on %d\n", id, getpid(), _lock);
      lock(_lock);
      _info("#%d:%d holds _lock\n", id, getpid());
      for(i=0;i<_k;i++) {
	accumulate_point(&_sums[i], &sums[i]);
	_counts[i] += counts[i];
      }
      _info("#%d:%d releasing _lock\n", id, getpid());
      unlock(_lock);

      /* now each thread has completed it's parallel work 
	 for this iteration ostensibly. The final aggregation and 
	 convergence test  requires that no straggling processor is 
	 still updating shared data, so we use a barrier, and perform
	 the final reduction and convergence test on a single process. 
	 use a second barrier wait after the test to ensure that no
	 process restarts the loop before the test completes.
      */
      _info("#%d:%d entering barrier wait 1k\n", id, getpid());      
      barrier_wait(_barrierX, id, nprocs);
      _info("#%d:%d exiting barrier wait 1k\n", id, getpid());      
      if(id==0) {
	for(i=0; i<_k; i++) 
	  _ncenters[i] = div_point(&_sums[i], _counts[i]);
	*_converged = ((!centers_changed(_ncenters)) || itr >= max_itr);
	reset_accumulators(_sums, _counts);
      }
      _info("#%d:%d entering barrier wait 2\n", id, getpid());      
      barrier_wait(_barrierY, id, nprocs);
      _info("#%d:%d exiting barrier wait 2\n", id, getpid());      
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


unsigned int
kmeans_workers(int numprocs) {

  int i=0;
  int index=0;
  int child_status=0;
  int parent=1;
  int childpids[MAXWORKERS];
  int _pfd[MAXWORKERS*2];
  int * pfd[MAXWORKERS];
  int _dpfd[MAXWORKERS*2];
  int * dpfd[MAXWORKERS];
  cpu_set_t set[MAXWORKERS];
  int buf = 0;
  unsigned startticks = 0;
  unsigned stopticks = 0;

  memset(childpids, 0, MAXWORKERS*sizeof(int));
  memset(_pfd, 0, MAXWORKERS*sizeof(int)*2);
  memset(pfd, 0, MAXWORKERS*sizeof(int*));
  
  for(i=0; i<numprocs; i++) {
    pfd[i] = &_pfd[i*2];
    dpfd[i] = &_dpfd[i*2];    
    if(pipe(pfd[i]) < 0 || pipe(dpfd[i]) < 0) {
      _error("can't make alive/dead pipes %d\n", i);
      exit(1);
    }
  }
  
  for(index=0; index<numprocs; index++) {

    int child_pid = fork();
    if(child_pid == 0) {

	close(pfd[index][0]); /* child writes pipe to notify parent alive */
	close(dpfd[index][0]); /* child writes pipe to notify parent complete */

        /* set cpu affinity for the child */
	CPU_ZERO(&set[index]);
	CPU_SET(index, &set[index]);
	if(sched_setaffinity(getpid(), sizeof(cpu_set_t), &set[index]) == -1) {
	  _error("#%d:%d set affinity failed\n", index, getpid());
	  exit(0);
	}

	/* set up go/stop signals */
	_go=0;
	_info("#%d:%d: _go:%d\n", index, getpid(), _go);
	signal(SIGALRM, siggoproc);     /* override SIGALRM to receive "go" signal from parent */
	
	_info("#%d:%d notifying  alive\n", index, getpid());
	write(pfd[index][1], &buf, 1);  /* hello, parents! I'm alive! */
	while(!_go);                   /* wait for go signal from parent after all children alive */
	
	_info("pid:%d Going! DO SOME WORK HERE\n", getpid());
	kmeans(index, numprocs);

	_info("#%d:%d done, notifying parent dead!\n", index, getpid());
	write(dpfd[index][1], &buf, 1); /* notify parent complete! */
	close(dpfd[index][1]);          /* keep close overheads off parallel critical path */
	close(pfd[index][1]);           /* keep close overheads off parallel critical path */
	sleep(1);
	exit(0);                        /* exit the child process */
	
    } else {

      parent=1;
      close(pfd[index][1]); /* parent reads 'alive signal from child' */
      close(dpfd[index][1]); /* parent reads 'dead' signal from child' */
      childpids[index]=child_pid;
    }
  }
    
  for(index=0; index<numprocs; index++) {
    int res = 0;
    do {
      res = read(pfd[index][0], &buf, 1);
      _info("collected alive signal (%d) from #%d:%d\n", res, index, childpids[index]);
      if(res>0) break;
    } while(res > 0);
    _info("#%d alive!\n", index);
  }

  startticks = ticks();
  for(index=0; index<numprocs; index++) {
    _info("Telling child#%d (pid%d) to go!\n", index, childpids[index]);
    kill(childpids[index], SIGALRM);
  }

  for(index=0; index<numprocs; index++) {
    
    /* collect 'done' signal over pipe (rather than wait() to avoid 
       process creation overheads sneaking into parallel
       performance measurement 
    */
    
    while(1) {
      if(read(dpfd[index][0], &buf, 1) > 0)
	break;
    } 
    _info("#%d dead!\n", index);
  }
  stopticks = ticks();
    
  for(index=0; index<numprocs; index++) {
    close(pfd[index][0]);    /* close read end of child alive pipe */
    close(dpfd[index][0]);   /* close read end of child dead pipe */
    wait(&child_status);     /* wait for process exit--not strictly necessary here */
    _info("child exited with status %d\n", child_status);
  }
  return stopticks - startticks;  
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
    _barrierX = barrier_create(149, _numprocs);
    _barrierY = barrier_create(150, _numprocs);
    _lock = lock_create(1449);
    ticks = kmeans_workers(_numprocs);
    printf("parallel work completed in %d msec\n", ticks);

    /* Print the center of each cluster */
    if(_verbose) {
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
    
    exit(EXIT_SUCCESS);
}
