///-------------------------------------------------------------------------------------------------
// file:	kmeansaw.cu.h
//
// summary:	Declares the kmeans.cu class
///-------------------------------------------------------------------------------------------------

#ifndef _RAWSHR_CU_H__
#define _RAWSHR_CU_H__

#include <algorithm>
#include <vector>
#include <set>
#include <type_traits>
#include <cuda_runtime.h>
#include "genericvector.h"
#include "kmeans-common.cu.h"

template <int R, int C> 
class centersmanagerGM {
protected: 
    float * m_pG;
    float * m_pRO;
public:
    centersmanagerGM(float * pG) : m_pG(pG), m_pRO(NULL) {} 
    bool useROMem() { return false; }
    float * dataRW() { return m_pG; }
    float * dataRO() { return m_pG; }
    bool update(float * p, bool bHtoD=false) { return true; }
    bool updateHtoD(float * p) { return true; }
    bool updateDToD(float * p) { return true; }
};


template <int R, int C>
__global__ void resetExplicit(float * pC, int * pCC)  {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= C) return;
    // printf("resetExplicit: pCC[%d]=>%.3f/%d\n", idx, pC[idx*R], pCC[idx]);
	for(int i=0;i<R;i++) 
		pC[idx*R+i] = 0.0f;
    pCC[idx] = 0;
}

template <int R, int C>
__global__ void finalizeCentersBasic(float * pC, int * pCC) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= C) return;
    int nNumerator = pCC[idx];
    if(nNumerator == 0) {       
	    for(int i=0;i<R;i++) 
            pC[(idx*R)+i] = 0.0f;
    } else {
	    for(int i=0;i<R;i++) 
		    pC[(idx*R)+i] /= pCC[idx];
    }
}

template <int R, int C>
__global__ void accumulateCenters(float * pP, float * pC, int * pCC, int * pCI, int nP) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= nP) return;
	int clusterid = pCI[idx];
	for(int i=0;i<R;i++) 
		atomicAdd(&pC[(clusterid*R)+i], pP[(idx*R)+i]);
	atomicAdd(&pCC[clusterid], 1);
}

template <int R, int C, bool ROWMAJ=true> 
class accumulatorGM {
public:

    static void reset(float * pC, int * pCC) { 
            resetExplicit<R,C><<<iDivUp(C, THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(pC, pCC);
    }

    static void accumulate(float * pP, float * pC, int * pCC, int * pCI, int nP) {
        const uint nPointsBlocks = iDivUp(nP, THREADBLOCK_SIZE);
            accumulateCenters<R,C><<<nPointsBlocks, THREADBLOCK_SIZE>>>(pP, pC, pCC, pCI, nP);
    }

    static void finalize(float * pC, int * pCC) {
        const int nAccums = iDivUp(C, THREADBLOCK_SIZE);
            finalizeCentersBasic<R,C><<<nAccums, THREADBLOCK_SIZE>>>(pC, pCC);
    }
};

template<int R, int C> __global__
void accumulateSM_RCeqBlockSize(float * pP, float * pC, int * pCC, int * pCI, int nP) {
    __shared__ float accums[R*C];
    __shared__ int cnts[C];
    dassert(R*C*sizeof(float) <= ACCUM_SHMEMSIZE);
    dassert(C*sizeof(int) <= COUNTER_SHMEMSIZE);
    dassert(R*C <= 1024);
    dassert(threadIdx.x < R*C);
    if(threadIdx.x < R*C) accums[threadIdx.x] = 0.0f;
    if(threadIdx.x < C) cnts[threadIdx.x] = 0;
    __syncthreads();
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < nP) {
        int clusterid = pCI[idx];
        for(int i=0;i<R;i++) 
            atomicAdd(&accums[(clusterid*R)+i], pP[(idx*R)+i]);
        atomicAdd(&cnts[clusterid], 1);
    }
    __syncthreads();
    if(threadIdx.x < R*C) atomicAdd(&pC[threadIdx.x], accums[threadIdx.x]);
    if(threadIdx.x < C) atomicAdd(&pCC[threadIdx.x], cnts[threadIdx.x]);
}

template<int R, int C, int nLDElemsPerThread> __global__
void accumulateSM(float * pP, float * pC, int * pCC, int * pCI, int nP) {
    __shared__ float accums[R*C];
    __shared__ int cnts[C];
    dassert(R*C*sizeof(float) <= ACCUM_SHMEMSIZE);
    dassert(C*sizeof(int) <= COUNTER_SHMEMSIZE);
	for(int ridx=0; ridx<nLDElemsPerThread; ridx++) {
		int nLDIdx = threadIdx.x + (ridx*blockDim.x);
        if(nLDIdx < R*C) accums[nLDIdx] = 0.0f;
    }
    if(threadIdx.x < C) cnts[threadIdx.x] = 0;
    __syncthreads();
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < nP) {
        int clusterid = pCI[idx];
        for(int i=0;i<R;i++) 
            atomicAdd(&accums[(clusterid*R)+i], pP[(idx*R)+i]);
        atomicAdd(&cnts[clusterid], 1);
    }
    __syncthreads();
	for(int ridx=0; ridx<nLDElemsPerThread; ridx++) {
		int nLDIdx = threadIdx.x + (ridx*blockDim.x);
        if(nLDIdx < R*C) atomicAdd(&pC[nLDIdx], accums[nLDIdx]);
    }
    if(threadIdx.x < C) atomicAdd(&pCC[threadIdx.x], cnts[threadIdx.x]);
}

template<int R, int C, int nLDElemsPerThread> __global__
void accumulateSMexact(float * pP, float * pC, int * pCC, int * pCI, int nP) {
    __shared__ float accums[R*C];
    __shared__ int cnts[C];
    dassert(R*C*sizeof(float) <= ACCUM_SHMEMSIZE);
    dassert(C*sizeof(int) <= COUNTER_SHMEMSIZE);
	for(int ridx=0; ridx<nLDElemsPerThread; ridx++) {
		int nLDIdx = threadIdx.x + (ridx*blockDim.x);
        accums[nLDIdx] = 0.0f;
    }
    if(threadIdx.x < C) cnts[threadIdx.x] = 0;
    __syncthreads();
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < nP) {
        int clusterid = pCI[idx];
        for(int i=0;i<R;i++) 
            atomicAdd(&accums[(clusterid*R)+i], pP[(idx*R)+i]);
        atomicAdd(&cnts[clusterid], 1);
    }
    __syncthreads();
	for(int ridx=0; ridx<nLDElemsPerThread; ridx++) {
		int nLDIdx = threadIdx.x + (ridx*blockDim.x);
		atomicAdd(&pC[nLDIdx], accums[nLDIdx]);
    }
    if(threadIdx.x < C) atomicAdd(&pCC[threadIdx.x], cnts[threadIdx.x]);
}

template <int R, int C, bool ROWMAJ=true> 
class accumulatorSM {
public:
    static void reset(float * pC, int * pCC) { accumulatorGM<R,C,ROWMAJ>::reset(pC, pCC); }
    static void finalize(float * pC, int * pCC) { accumulatorGMMS<R,C,ROWMAJ>::finalize(pC, pCC); }

    static void
    accumulate(float * pP, float * pC, int * pCC, int * pCI, int nP) { 
        if(R*C>SHMEMACCUM_FLOATS) {
            accumulatorGM<R,C,ROWMAJ>::accumulate(pP, pC, pCC, pCI, nP);
        } else {
            if(R*C<MAXBLOCKTHREADS) {
                accumulateRCeqBS(pP, pC, pCC, pCI, nP);
            } else {
                accumulateGeneral(pP, pC, pCC, pCI, nP);
            }
        }
    }

protected:

    static void
    accumulateRCeqBS(float * pP, float * pC, int * pCC, int * pCI, int nP) {
        // if R*C < max thread group, we can reset and merge
        // accumulators in a more efficient way than if R*C is larger
        const int nMinRCSize = max(R*C, THREADBLOCK_SIZE);
        const int nBlockSize = MAXBLOCKTHREADS; // min(nMinRCSize, MAXBLOCKTHREADS);
        const int nMinAccumBlocks = iDivUp(R*C, nBlockSize);
        const int nMinPointsBlocks = iDivUp(nP, nBlockSize);
        const int nBlocks = max(nMinAccumBlocks, nMinPointsBlocks);
            // printf("accumulateSM_RCeqBlockSize<%d,%d><<<%d,%d>>>(...)\n", R, C, nBlocks, nBlockSize);
            accumulateSM_RCeqBlockSize<R,C><<<nBlocks, nBlockSize>>>(pP, pC, pCC, pCI, nP);
    }

    static void
    accumulateGeneral(float * pP, float * pC, int * pCC, int * pCI, int nP) {
        // const int nMinRCSize = max(R*C, THREADBLOCK_SIZE);
        const int nBlockSize = THREADBLOCK_SIZE;
		const int nElemsPerThread = (R*C)/nBlockSize;
		const int nElemsPerThreadPP = ((R*C)/nBlockSize)+1;
        const int nMinAccumBlocks = iDivUp(C, nBlockSize);
        const int nMinPointsBlocks = iDivUp(nP, nBlockSize);
        const int nBlocks = max(nMinAccumBlocks, nMinPointsBlocks);
			if(R*C%nBlockSize) {
				// printf("accumulateSM<%d, %d, %d><<<%d,%d>>>()\n", R,C,nElemsPerThreadPP,nBlocks,nBlockSize);
				accumulateSM<R,C,nElemsPerThreadPP><<<nBlocks, nBlockSize>>>(pP, pC, pCC, pCI, nP);
			} else {
				// printf("accumulateSM<%d, %d, %d><<<%d,%d>>>()\n", R,C,nElemsPerThread,nBlocks,nBlockSize);
				accumulateSMexact<R,C,nElemsPerThread><<<nBlocks, nBlockSize>>>(pP, pC, pCC, pCI, nP);
			}
    }
};

template <int R, int C> 
__device__ float 
_vdistancef(float * a, float * b) {
    float accum = 0.0f;
    for(int i=0; i<R; i++) {
        float delta = a[i]-b[i];
        accum += delta*delta;
    }
    return sqrt(accum);
}

template <int R, int C>
__device__ int 
nearestCenter(float * pP, float * pC) {
    float mindist = FLT_MAX;
    int minidx = 0;
	int clistidx = 0;
    for(int i=0; i<C;i++) {
		clistidx = i*R;
        float dist = _vdistancef<R,C>(pP, &pC[clistidx]);
        if(dist < mindist) {
            minidx = static_cast<int>(i);
            mindist = dist;
        }
    }
    return minidx;
}

template <int R, int C, bool bRO> 
__global__ void 
mapPointsToCenters(float * pP, float * pC, int * pCI, int nP) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= nP) return;
	pCI[idx] = nearestCenter<R,C>(&pP[idx*R], bRO ? d_cnst_centers : pC);
}

template<int R, 
         int C, 
         typename CM, 
         typename SM,
         bool ROWMAJ=true> 
class kmeansraw {
public:    

    kmeansraw(
	    __in int     nSteps, 
	    __in float * d_Points,  
	    __in float * d_Centers,
	    __in int *   d_ClusterCounts,
	    __in int *   d_ClusterIds,
	    __in int     nPoints,
	    __in int     nCenters) : m_nSteps(nSteps),
                                 m_dPoints(d_Points),
                                 m_dCenters(d_Centers),
                                 m_dClusterCounts(d_ClusterCounts),
                                 m_dClusterIds(d_ClusterIds),
                                 m_nPoints(nPoints),
                                 m_nCenters(nCenters),
                                 m_centers(d_Centers) {}

    bool execute() {
        assert(m_nCenters == C);
        const uint nCentersBlocks = iDivUp(m_nCenters, THREADBLOCK_SIZE);
        const uint nPointsBlocks = iDivUp(m_nPoints, THREADBLOCK_SIZE);	    
        for(int i=0; i<m_nSteps; i++) {
            _V(updatecentersIn(m_dCenters));
		    _V(mapCenters(m_dPoints, m_dClusterIds, m_nPoints));
		    _V(resetAccumulators(m_dCenters, m_dClusterCounts));
		    _V(accumulate(m_dPoints, m_dCenters, m_dClusterCounts, m_dClusterIds, m_nPoints));
		    _V(finalizeAccumulators(m_dCenters, m_dClusterCounts));
	    }
        return true;
    }


protected:

    int     m_nSteps;
    float * m_dPoints;
    float * m_dCenters;
    int *   m_dClusterCounts;
    int *   m_dClusterIds;
    int     m_nPoints;
    int     m_nCenters;

    //std::conditional<((((R)*(C))<=CONSTMEMSIZE)&CM), centersmanagerRO<R,C,CM>, centersmanagerGM<R,C,CM>>::type m_centers;
    //std::conditional<((((R)*(C))<=SHMEMSIZE)&SM), accumulatorSM<R,C,SM,MS,SHMAP>, accumulatorGM<R,C,SM,MS,SHMAP>>::type m_accumulator;
    CM m_centers; 
    SM m_accumulator;
    bool updatecentersIn(float * p_Centers) { return m_centers.update(p_Centers); }
    bool initcentersInput(float * p_Centers) { return m_centers.updateHtoD(p_Centers); }
    void resetAccumulators(float * pC, int * pCC) { m_accumulator.reset(pC, pCC); }
    void accumulate(float * pP, float * pC, int * pCC, int * pCI, int nP) { m_accumulator.accumulate(pP, pC, pCC, pCI, nP); }
    void finalizeAccumulators(float * pC, int * pCI) { m_accumulator.finalize(pC, pCI); }
    float * centers() { return m_centers.data(); }

    void mapCenters(float * pP, int * pCI, int nP) {
        const uint nCentersBlocks = iDivUp(C, THREADBLOCK_SIZE);
        const uint nPointsBlocks = iDivUp(nP, THREADBLOCK_SIZE);	    
        float * pC = m_centers.dataRO();
                mapPointsToCenters<R,C,false><<<nPointsBlocks, THREADBLOCK_SIZE>>>(pP, pC, pCI, nP);
    }

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

    static bool
    CompareResults(
        std::vector<pt<R>> &centers,    
        std::vector<pt<R>> &refcenters,
        float EPSILON=0.0001f,
        bool bVerbose=true,
        int nRowLimit=16
        )
    {
        int nRows=0;
        std::map<int, pt<R>*> unmatched;
        std::map<int, float> unmatched_deltas;
        std::map<int, int> matched;
        std::map<int, int> revmatched;
        // std::vector<pt<N>>::iterator vi, xi;
        int nCenterIdx=0;    
        for(auto vi=centers.begin(); vi!=centers.end(); vi++, nCenterIdx++) {
            bool bFound = false;        
            if(EPSILON*R > hdistance(*vi, refcenters[nCenterIdx])) {
                bFound = true;
                matched[nCenterIdx] = nCenterIdx;
                revmatched[nCenterIdx] = nCenterIdx;
            } else {
                int nRefIdx=0;
                for(auto xi=refcenters.begin(); xi!=refcenters.end(); xi++, nRefIdx++) {
                    if(EPSILON*R > hdistance(*vi, *xi)) {
                        bFound = true;
                        matched[nCenterIdx] = nRefIdx;
                        revmatched[nRefIdx] = nCenterIdx;
                        break;
                    }
                }
            }
            if(!bFound) {
                unmatched[nCenterIdx] = (&(*vi));
                unmatched_deltas[nCenterIdx] = hdistance(*vi, refcenters[nCenterIdx]);
            }
        }
        bool bSuccess = unmatched.size() == 0;
        if(bVerbose && !bSuccess) {
            // std::map<int, pt<R>*>::iterator si;
            fprintf(stderr, "Could not match %d centers:\n", unmatched.size());
            for(auto si=unmatched.begin(); si!=unmatched.end(); si++) {
                if(++nRows > nRowLimit) {
                    fprintf(stderr, "...\n");
                    break;
                }
                fprintf(stdout, "IDX(%d): ", si->first);
                (si->second)->dump(stderr);        
            }
        }
        return bSuccess;
    }

public:

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
