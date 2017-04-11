///-------------------------------------------------------------------------------------------------
// file:	native.cu.h
//
// summary:	Declares the kmeans.cu class
///-------------------------------------------------------------------------------------------------

#ifndef __NATIVE_CU_H__
#define __NATIVE_CU_H__

#include <algorithm>
#include <vector>
#include <set>
#include <type_traits>
#include <cuda_runtime.h>
#include "genericvector.h"
#include "kmeans-common.cu.h"

    static const int MAX_CHAR_PER_LINE = 512;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Reads an input file. </summary>
    ///
    /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. Adapted from code in
    ///             serban/kmeans which appears adapted from code in STAMP (variable names only
    ///             appear to be changed), or perhaps each was adapted from code in other places.
    ///             
    ///             ****************************** Note that AMP requires template int args to be
    ///             statically known at compile time. Hence, for now, you have to have built this
    ///             program with DEFAULTRANK defined to match the DEFAULTRANK of your input file! This restriction
    ///             is easy to fix, but not the best use of my time at the moment...
    ///             ******************************.
    ///             </remarks>
    ///
    /// <param name="filename">     If non-null, filename of the file. </param>
    /// <param name="points">       The points. </param>
    /// <param name="numObjs">      If non-null, number of objects. </param>
    /// <param name="numCoords">    If non-null, number of coords. </param>
    /// <param name="_debug">       The debug flag. </param>
    ///
    /// <returns>   The input. </returns>
    ///-------------------------------------------------------------------------------------------------

    static int
    ReadInput(
        char * filename,
        std::vector<pt<R>>& points,
        int * numObjs,
        int * numCoords,
        int _debug
        ) 
    {
        #pragma warning(disable:4996)
        float **objects;
        int     i, j, len;

        FILE *infile;
        char *line, *ret;
        int   lineLen;

        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            return NULL;
        }

        /* first find the number of objects */
        lineLen = MAX_CHAR_PER_LINE;
        line = (char*) malloc(lineLen);
        assert(line != NULL);

        (*numObjs) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            /* check each line to find the max line length */
            while (strlen(line) == lineLen-1) {
                /* this line read is not complete */
                len = (int)strlen(line);
                fseek(infile, -len, SEEK_CUR);

                /* increase lineLen */
                lineLen += MAX_CHAR_PER_LINE;
                line = (char*) realloc(line, lineLen);
                assert(line != NULL);

                ret = fgets(line, lineLen, infile);
                assert(ret != NULL);
            }

            if (strtok(line, " \t\n") != 0)
                (*numObjs)++;
        }
        rewind(infile);
        if (_debug) printf("lineLen = %d\n",lineLen);

        /* find the no. objects of each object */
        (*numCoords) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                /* ignore the id (first coordiinate): numCoords = 1; */
                while (strtok(NULL, " ,\t\n") != NULL) (*numCoords)++;
                break; /* this makes read from 1st object */
            }
        }
        rewind(infile);
        if (_debug) {
            printf("File %s numObjs   = %d\n",filename,*numObjs);
            printf("File %s numCoords = %d\n",filename,*numCoords);
        }

        /* allocate space for objects[][] and read all objects */
        len = (*numObjs) * (*numCoords);
        objects    = (float**)malloc((*numObjs) * sizeof(float*));
        assert(objects != NULL);
        objects[0] = (float*) malloc(len * sizeof(float));
        assert(objects[0] != NULL);
        for (i=1; i<(*numObjs); i++)
            objects[i] = objects[i-1] + (*numCoords);

        i = 0;
        /* read all objects */
        while (fgets(line, lineLen, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue;
            for (j=0; j<(*numCoords); j++)
                objects[i][j] = (float)atof(strtok(NULL, " ,\t\n"));
            i++;
        }

		points.reserve(static_cast<int>(*numObjs));
        for(int idx=0; idx<(*numObjs); idx++) {
            pt<R> point(objects[idx]);
            points.push_back(point);
        }

        fclose(infile);
        free(line);
        free(objects[0]);
        free(objects);
        return 0;
        #pragma warning(default:4996)
    }

    static void
    ChooseInitialCenters(
        std::vector<pt<R>> &points,
        std::vector<pt<R>> &centers,
        std::vector<pt<R>> &refcenters,
        int nRandomSeed
        )
    {
        srand(nRandomSeed);
        std::set<int> chosenidx;
        while(chosenidx.size() < (size_t)C) {
            // sets don't allow dups...
            int idx = rand() % points.size();
            chosenidx.insert(idx);
        }
        std::set<int>::iterator si;
        for(si=chosenidx.begin(); si!=chosenidx.end(); si++) {
            centers.push_back(points[*si]);
            refcenters.push_back(points[*si]);
        }    
    }

    static void
    PrintCenters(
        FILE * fp,
        std::vector<pt<R>> &centers,
        int nColLimit=8,
        int nRowLimit=16
        )
    {
        fprintf(fp, "\n");
        int nRows = 0;
        for(auto x = centers.begin(); x != centers.end(); x++) {
            if(++nRows > nRowLimit) {
                fprintf(fp, "...");
                break;
            }
            x->dump(fp,nColLimit);
        }
        fprintf(fp, "\n");
    }

    static void PrintCenters(
        FILE * fp,
        pt<R> * pCenters,
        int nColLimit=8,
        int nRowLimit=16
        )
    {
        nRowLimit=(nRowLimit==0)?C:min(nRowLimit,C);
	    for(int i=0; i<nRowLimit; i++)
            pCenters[i].dump(fp,nColLimit);
        if(nRowLimit < C) 
            fprintf(fp, "...");
        fprintf(fp, "\n");
    }

    static void
    PrintResults(
        FILE * fp,
        std::vector<pt<R>> &centers,
        std::vector<pt<R>> &refcenters,
        bool bSuccess,
        double dKmeansExecTime,
        double dKmeansCPUTime,
        bool bVerbose=false
        )
    {
        fprintf(fp, "%s: GPU: %.5f sec, CPU: %.5f sec\n", 
                (bSuccess?"SUCCESS":"FAILURE"),
                dKmeansExecTime, dKmeansCPUTime);
        if(!bSuccess || bVerbose) {
            fprintf(fp, "final centers:\n");
		    PrintCenters(fp, centers);
            fprintf(fp, "reference centers:\n");
		    PrintCenters(fp, refcenters);
        }
    }


    static double
    benchmark(
	    const int nSteps,
	    void * lpvPoints,
	    void * lpvCenters,
	    const int nPoints,
	    const int nCenters,
        StopWatchInterface * hTimer,
	    bool bVerify,
	    bool bVerbose
	    )
    {
        assert(nCenters == C);

        float * h_TxPoints = NULL;
        float * h_TxCenters = NULL;
	    pt<R> * h_InPoints = reinterpret_cast<pt<R>*>(lpvPoints);
	    pt<R> * h_InCenters = reinterpret_cast<pt<R>*>(lpvCenters);        
	    float * h_Points = reinterpret_cast<float*>(h_InPoints);
	    float * h_Centers = reinterpret_cast<float*>(h_InCenters);
        if(!ROWMAJ) {
            h_TxPoints = transpose(h_InPoints, nPoints);
            h_TxCenters = transpose(h_InCenters, nCenters);
            h_Points = h_TxPoints;
            h_Centers = h_TxCenters;
        }

        float * d_Points = NULL;
	    float * d_Centers = NULL;
	    int *   d_ClusterIds = NULL;
	    int *   d_ClusterCounts = NULL;
	    size_t  uiPointsBytes = nPoints * R * sizeof(float);
	    size_t  uiCentersBytes = nCenters * R * sizeof(float);
	    size_t  uiClusterIdsBytes = nPoints * sizeof(int);
	    size_t  uiClusterCountsBytes = nCenters * sizeof(int);

        INFORM(bVerbose, "Initializing data...\n");
        checkCudaErrors( cudaMalloc((void **)&d_Points, uiPointsBytes ) );
        checkCudaErrors( cudaMalloc((void **)&d_Centers, uiCentersBytes ) );
        checkCudaErrors( cudaMalloc((void **)&d_ClusterIds, uiClusterIdsBytes ) );
        checkCudaErrors( cudaMalloc((void **)&d_ClusterCounts, uiClusterCountsBytes ) );
	    checkCudaErrors( cudaMemcpy(d_Points, h_Points, uiPointsBytes, cudaMemcpyHostToDevice) );
	    checkCudaErrors( cudaMemcpy(d_Centers, h_Centers, uiCentersBytes, cudaMemcpyHostToDevice) );
	    INFORM(bVerbose, "Starting up kmeans-raw...\n\n");

        // fprintf(stdout, "initial centers:\n");
	    // PrintCenters<DEFAULTRANK>(stdout, h_Centers, nCenters);

        kmeansraw<R,C,CM,SM,ROWMAJ>* pKMeans = 
            new kmeansraw<R,C,CM,SM,ROWMAJ>(nSteps,
                                            d_Points,
                                            d_Centers,
                                            d_ClusterCounts,
                                            d_ClusterIds,
                                            nPoints,
                                            nCenters);

	    sdkResetTimer(&hTimer);
	    sdkStartTimer(&hTimer);
        pKMeans->execute();
	    checkCudaErrors( cudaDeviceSynchronize() );
	    sdkStopTimer(&hTimer);
	    double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);

	    if(bVerbose) {
		    uint byteCount = (uint)(uiPointsBytes + uiCentersBytes);
		    shrLog("kmeans() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
		    shrLogEx(LOGBOTH | MASTER, 0, "kmeans, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n", 
					    (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, THREADBLOCK_SIZE); 
	    }

	    if(bVerify) {
		    shrLog(" ...reading back GPU results\n");
		    checkCudaErrors( cudaMemcpy(h_Centers, d_Centers, uiCentersBytes, cudaMemcpyDeviceToHost) );
            if(!ROWMAJ) {
                rtranspose(h_TxCenters, nCenters, (float*)h_InCenters);
            }
	    }
	    shrLog("cleaning up device resources...\n");
	    checkCudaErrors( cudaFree((void*)d_Points) );
	    checkCudaErrors( cudaFree((void*)d_Centers) );
	    checkCudaErrors( cudaFree((void*)d_ClusterIds) );
	    checkCudaErrors( cudaFree((void*)d_ClusterCounts) );
        if(!ROWMAJ) {
            free(h_TxCenters);
            free(h_TxPoints);
        }
	    return dAvgSecs;
    }

    static float 
    hdistance(
        pt<R> &a,
        pt<R> &b
        ) 
    {
        float accum = 0.0f;
        for(int i=0; i<R; i++) {
            float delta = a.m_v[i]-b.m_v[i];
            accum += delta*delta;
        }
        return sqrt(accum);
    }

    static int 
    NearestCenter(
        pt<R> &point,
        std::vector<pt<R>> &centers
        ) 
    {
        float mindist = FLT_MAX;
        int minidx = 0;
        for(size_t i=0; i<centers.size();i++) {
            float dist = hdistance(point, centers[i]);
            if(dist < mindist) {
                minidx = static_cast<int>(i);
                mindist = dist;
            }
        }
        return minidx;
    }

    static void
    MapPointsToCentersSequential(
        std::vector<pt<R>> &vcenters,
        std::vector<pt<R>> &vpoints,
        std::vector<int>& vclusterids
        )
    {
        // std::vector<pt<R>>::iterator vi;
        int index = 0;
        for(auto vi=vpoints.begin(); vi!=vpoints.end(); vi++) 
            vclusterids[index++] = NearestCenter(*vi, vcenters);
    }

    static void
    UpdateCentersSequential(
        std::vector<pt<R>> &vcenters,
        std::vector<pt<R>> &vpoints,
        std::vector<int>& vclusterids
        )
    {
        std::vector<int> counts;
        for(size_t i=0; i<vcenters.size(); i++) {
            vcenters[i].set(0.0f);
            counts.push_back(0);
        }
        for(size_t i=0; i<vpoints.size(); i++) {
            int clusterid = vclusterids[i];
            vcenters[clusterid] += vpoints[i];
            counts[clusterid] += 1;
        }
        for(size_t i=0; i<vcenters.size(); i++) {
            vcenters[i] /= counts[i];
        }
    }

    static void 
    bncmain(
        char * lpszInputFile, 
        LPFNKMEANS lpfnKMeans,
        int nSteps,
        int nSeed,
        StopWatchInterface * hTimer,
        bool bVerify,
        bool bVerbose
        )
    {
        int nP = 0;
        int nD = 0;
        std::vector<pt<R>> points;
        std::vector<pt<R>> centers;
        std::vector<pt<R>> refcenters;

        ReadInput(lpszInputFile, points, &nP, &nD, 0);
	    if (points.size()==0) {
		    fprintf(stderr, "Error loading points from %s!\n", lpszInputFile);
	    }
        std::vector<int> clusterids(points.size());
        std::vector<int> refclusterids(points.size());

        // be careful to make sure you're choosing the same
        // random seed every run. If you upgrade this code to actually
        // do divergence tests to decide when to terminate, it will be
        // important to use the same random seed every time, since the 
        // choice of initial centers can have a profound impact on the
        // number of iterations required to converge. Failure to be 
        // consistent will introduce a ton of noice in your data. 
    
        ChooseInitialCenters(points,                   // points to choose from              
                             centers,                  // destination array of initial centers
                             refcenters,               // save a copy for the reference impl to check
                             nSeed);                 // random seed. ACHTUNG! Beware benchmarkers

	    int nPoints = (int)points.size();
	    int nCenters = C;
	    size_t uiPointsBytes = nPoints * R * sizeof(float);
	    size_t uiCentersBytes = C * R * sizeof(float);
	    size_t uiClusterIdsBytes = nPoints * sizeof(int);
		
		bool bSuccess = false;
		double dAvgSecs = 0.0;
	    pt<R> *h_Points = NULL;
	    pt<R> *h_Centers = NULL;
	    int * h_ClusterIds = NULL;
		bool bTooBig = (uiPointsBytes > 0XFFFFFFFF);

		// if the points won't fit in GPU memory, there is
		// no point in going through the exercise of watching 
		// the GPU exec fail (particularly if we still want to
		// collect the CPU comparison number). If it's obviously
		// too big, skip the CUDA rigmaroll.

		if(!bTooBig) {

			h_Points = (pt<R>*)malloc(uiPointsBytes);
			h_Centers = (pt<R>*)malloc(uiCentersBytes);
			h_ClusterIds = (int*)malloc(uiClusterIdsBytes);
			memset(h_ClusterIds, 0, uiClusterIdsBytes);
			//std::vector<pt<R>>::iterator vi;
			pt<R>* pPoints = h_Points;
			for(auto vi=points.begin(); vi!=points.end(); vi++) 
				*pPoints++ = *vi;
			pt<R>* pCenters = h_Centers;
			for(auto vi=centers.begin(); vi!=centers.end(); vi++) 
				*pCenters++ = *vi;

			//fprintf(stdout, "initial centers:\n");
			//PrintCenters<DEFAULTRANK>(stdout, h_Centers, nCenters);

			dAvgSecs = (*lpfnKMeans)(nSteps, 
									 h_Points, 
									 h_Centers, 
									 nPoints, 
									 nCenters, 
									 hTimer, 
									 bVerify,
									 bVerbose);
		}

	    if(bVerify) {
		    shrLog("\nValidating GPU results...\n");
		    sdkResetTimer(&hTimer);
		    sdkStartTimer(&hTimer);
		    for(int nStep=0; nStep<nSteps; nStep++) {
			    MapPointsToCentersSequential(refcenters, points, refclusterids);
			    UpdateCentersSequential(refcenters, points, refclusterids);
		    }
		    sdkStopTimer(&hTimer);
		    double dRefAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);

		    // compare the results, complaining loudly on a mismatch,
		    // and print the final output along with timing data for each impl.
		    
			if(!bTooBig) {
				pt<R>* pCenters = h_Centers;
				for(auto vi=centers.begin(); vi!=centers.end(); vi++) 
					*vi = *pCenters++;
				bSuccess = CompareResults(centers, refcenters,  0.1f, bVerbose);
			}
		    PrintResults(stdout, centers, refcenters, bSuccess, dAvgSecs, dRefAvgSecs, bVerbose);
	    }

	    shrLog("Cleaning up...\n");
        free(h_Centers);
        free(h_Points);
        free(h_ClusterIds);
    }
};


#endif
