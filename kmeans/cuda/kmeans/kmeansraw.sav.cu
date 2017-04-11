///-------------------------------------------------------------------------------------------------
// file:	kmeansraw.cu
//
// summary:	kmeans implementation over extents of floats (no underlying point/vector struct)
///-------------------------------------------------------------------------------------------------


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <set>
#include <map>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <shrUtils.h>
#include <shrQATest.h>
#include <sdkHelper.h>  // helper for shared that are common to CUDA SDK samples
#include "kmeansraw.cu.h"

#define buildfnnamebase(name) kmeans_##name
#define buildfnnamebaserank(name, rnk) buildfnnamebase(name)_r##rnk
#define buildfnname(name, rnk, cent) buildfnnamebaserank(name,rnk)_c##cent


#define declare_testcase_hdr(name, rnk, cent, cman, accman, rowmaj)  \
double                                                               \
buildfnname(name, rnk, cent)(                                        \
	const int nSteps,                                                \
	void * lpvPoints,                                                \
	void * lpvCenters,                                               \
	const int nPoints,                                               \
	const int nCenters,                                              \
    StopWatchInterface * hTimer,                                     \
	bool bVerify,                                                    \
	bool bVerbose                                                    \
	);                                                               \


#define declare_testcase(name, rnk, cent, cman, accman, rowmaj)  \
double                                                           \
buildfnname(name, rnk, cent)(                                    \
	const int nSteps,                                            \
	void * lpvPoints,                                            \
	void * lpvCenters,                                           \
	const int nPoints,                                           \
	const int nCenters,                                          \
    StopWatchInterface * hTimer,                                 \
	bool bVerify,                                                \
	bool bVerbose                                                \
	)                                                            \
{                                                                \
                                                                 \
    return kmeansraw<rnk,                                        \
                     cent,                                       \
                     cman<rnk, cent>,                            \
                     accman<rnk, cent, rowmaj>,                  \
                     rowmaj>::benchmark(nSteps,                  \
                                       lpvPoints,                \
                                       lpvCenters,               \
                                       nPoints,                  \
                                       nCenters,                 \
                                       hTimer,                   \
                                       bVerify,                  \
                                       bVerbose);                \
}                                                       

#define declare_testsuite(rank, cent)                                                                     \
declare_testcase(raw, rank, cent, centersmanagerGM, accumulatorGM, true)                                  \
declare_testcase(constmem, rank, cent, centersmanagerRO, accumulatorGM, true)                             \
declare_testcase(constmem_memsetreset, rank, cent, centersmanagerRO, accumulatorGMMS, true)               \
declare_testcase(constmem_memsetreset_shr, rank, cent, centersmanagerRO, accumulatorSM, true)             \
declare_testcase(constmem_memsetreset_shr_map, rank, cent, centersmanagerRO, accumulatorSMMAP, true)      \
declare_testcase(cm, rank, cent, centersmanagerGM, accumulatorGM, false)                                  \
declare_testcase(cm_constmem, rank, cent, centersmanagerRO, accumulatorGM, false)                         \
declare_testcase(cm_constmem_shr, rank, cent, centersmanagerRO, accumulatorSM, false)                     \
declare_testcase(cm_constmem_shr_map, rank, cent, centersmanagerRO, accumulatorSMMAP, false)              \


declare_testsuite(24, 64)


///-------------------------------------------------------------------------------------------------
/// <summary>   basic kmeans implementation, based on generic vector. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="argc"> The argc. </param>
/// <param name="argv"> [in,out] If non-null, the argv. </param>
///
/// <returns>   benchmark time in seconds</returns>
///-------------------------------------------------------------------------------------------------

double
kmeans_raw(
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
#if 1
    return kmeansraw<DEFAULTRANK, 
                     DEFAULTCENTERS, 
                     centersmanagerGM<DEFAULTRANK, DEFAULTCENTERS>, 
                     accumulatorGM<DEFAULTRANK,DEFAULTCENTERS>>::benchmark(nSteps,
                                                                           lpvPoints,
                                                                           lpvCenters,
                                                                           nPoints,
                                                                           nCenters,
                                                                           hTimer,
                                                                           bVerify,
                                                                           bVerbose);
#else
	    pt<DEFAULTRANK> * h_Points = reinterpret_cast<pt<DEFAULTRANK>*>(lpvPoints);
	    pt<DEFAULTRANK> * h_Centers = reinterpret_cast<pt<DEFAULTRANK>*>(lpvCenters);
        float * d_Points = NULL;
	    float * d_Centers = NULL;
	    int *   d_ClusterIds = NULL;
	    int *   d_ClusterCounts = NULL;
	    size_t  uiPointsBytes = nPoints * DEFAULTRANK * sizeof(float);
	    size_t  uiCentersBytes = nCenters * DEFAULTRANK * sizeof(float);
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

        kmeansraw<DEFAULTRANK, 
                  DEFAULTCENTERS, 
                  centersmanagerGM<DEFAULTRANK, DEFAULTCENTERS>, 
                  accumulatorGM<DEFAULTRANK,DEFAULTCENTERS>> * pKMeans = 
            new kmeansraw<DEFAULTRANK, 
                          DEFAULTCENTERS, 
                          centersmanagerGM<DEFAULTRANK, DEFAULTCENTERS>, 
                          accumulatorGM<DEFAULTRANK,DEFAULTCENTERS>>(nSteps,
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
	    }
	    shrLog("cleaning up device resources...\n");
	    checkCudaErrors( cudaFree((void*)d_Points) );
	    checkCudaErrors( cudaFree((void*)d_Centers) );
	    checkCudaErrors( cudaFree((void*)d_ClusterIds) );
	    checkCudaErrors( cudaFree((void*)d_ClusterCounts) );
	    return dAvgSecs;
#endif
}

///-------------------------------------------------------------------------------------------------
/// <summary>   basic kmeans implementation, based on generic vector. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="argc"> The argc. </param>
/// <param name="argv"> [in,out] If non-null, the argv. </param>
///
/// <returns>   benchmark time in seconds</returns>
///-------------------------------------------------------------------------------------------------


double
kmeans_constmem(
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
#if 1
    return kmeansraw<DEFAULTRANK,
                     DEFAULTCENTERS,
                     centersmanagerRO<DEFAULTRANK, DEFAULTCENTERS>,
                     accumulatorGM<DEFAULTRANK, DEFAULTCENTERS>>::benchmark(nSteps,
                                                                            lpvPoints,
                                                                            lpvCenters,
                                                                            nPoints,
                                                                            nCenters,
                                                                            hTimer,
                                                                            bVerify,
                                                                            bVerbose);
#else
	    pt<DEFAULTRANK> * h_Points = reinterpret_cast<pt<DEFAULTRANK>*>(lpvPoints);
	    pt<DEFAULTRANK> * h_Centers = reinterpret_cast<pt<DEFAULTRANK>*>(lpvCenters);
        float * d_Points = NULL;
	    float * d_Centers = NULL;
	    int *   d_ClusterIds = NULL;
	    int *   d_ClusterCounts = NULL;
	    size_t  uiPointsBytes = nPoints * DEFAULTRANK * sizeof(float);
	    size_t  uiCentersBytes = nCenters * DEFAULTRANK * sizeof(float);
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

        kmeansraw<DEFAULTRANK, 
                  DEFAULTCENTERS, 
                  centersmanagerRO<DEFAULTRANK, DEFAULTCENTERS>, 
                  accumulatorGM<DEFAULTRANK,DEFAULTCENTERS>> * pKMeans = 
            new kmeansraw<DEFAULTRANK, 
                          DEFAULTCENTERS, 
                          centersmanagerRO<DEFAULTRANK, DEFAULTCENTERS>, 
                          accumulatorGM<DEFAULTRANK,DEFAULTCENTERS>>(nSteps,
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
	    }
	    shrLog("cleaning up device resources...\n");
	    checkCudaErrors( cudaFree((void*)d_Points) );
	    checkCudaErrors( cudaFree((void*)d_Centers) );
	    checkCudaErrors( cudaFree((void*)d_ClusterIds) );
	    checkCudaErrors( cudaFree((void*)d_ClusterCounts) );
	    return dAvgSecs;
#endif
}

///-------------------------------------------------------------------------------------------------
/// <summary>   basic kmeans implementation, based on generic vector. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="argc"> The argc. </param>
/// <param name="argv"> [in,out] If non-null, the argv. </param>
///
/// <returns>   benchmark time in seconds</returns>
///-------------------------------------------------------------------------------------------------

double
kmeans_constmem_memsetreset(
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
    return kmeansraw<DEFAULTRANK,
                     DEFAULTCENTERS,
                     centersmanagerRO<DEFAULTRANK, DEFAULTCENTERS>,
                     accumulatorGMMS<DEFAULTRANK, DEFAULTCENTERS>>::benchmark(nSteps,
                                                                              lpvPoints,
                                                                              lpvCenters,
                                                                              nPoints,
                                                                              nCenters,
                                                                              hTimer,
                                                                              bVerify,
                                                                              bVerbose);
}

///-------------------------------------------------------------------------------------------------
/// <summary>   basic kmeans implementation, based on generic vector. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="argc"> The argc. </param>
/// <param name="argv"> [in,out] If non-null, the argv. </param>
///
/// <returns>   benchmark time in seconds</returns>
///-------------------------------------------------------------------------------------------------


double
kmeans_constmem_memsetreset_shr(
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
#if 1
    return kmeansraw<DEFAULTRANK,
                     DEFAULTCENTERS,
                     centersmanagerRO<DEFAULTRANK, DEFAULTCENTERS>,
                     accumulatorSM<DEFAULTRANK, DEFAULTCENTERS>>::benchmark(nSteps,
                                                                               lpvPoints,
                                                                               lpvCenters,
                                                                               nPoints,
                                                                               nCenters,
                                                                               hTimer,
                                                                               bVerify,
                                                                               bVerbose);
#else
	    pt<DEFAULTRANK> * h_Points = reinterpret_cast<pt<DEFAULTRANK>*>(lpvPoints);
	    pt<DEFAULTRANK> * h_Centers = reinterpret_cast<pt<DEFAULTRANK>*>(lpvCenters);
        float * d_Points = NULL;
	    float * d_Centers = NULL;
	    int *   d_ClusterIds = NULL;
	    int *   d_ClusterCounts = NULL;
	    size_t  uiPointsBytes = nPoints * DEFAULTRANK * sizeof(float);
	    size_t  uiCentersBytes = nCenters * DEFAULTRANK * sizeof(float);
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

        kmeansraw<DEFAULTRANK, 
                  DEFAULTCENTERS, 
                  centersmanagerRO<DEFAULTRANK, DEFAULTCENTERS>, 
                  accumulatorSM<DEFAULTRANK,DEFAULTCENTERS>> * pKMeans = 
            new kmeansraw<DEFAULTRANK, 
                          DEFAULTCENTERS, 
                          centersmanagerRO<DEFAULTRANK, DEFAULTCENTERS>, 
                          accumulatorSM<DEFAULTRANK,DEFAULTCENTERS>>(nSteps,
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
	    }
	    shrLog("cleaning up device resources...\n");
	    checkCudaErrors( cudaFree((void*)d_Points) );
	    checkCudaErrors( cudaFree((void*)d_Centers) );
	    checkCudaErrors( cudaFree((void*)d_ClusterIds) );
	    checkCudaErrors( cudaFree((void*)d_ClusterCounts) );
	    return dAvgSecs;
#endif
}

///-------------------------------------------------------------------------------------------------
/// <summary>   basic kmeans implementation, based on generic vector. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="argc"> The argc. </param>
/// <param name="argv"> [in,out] If non-null, the argv. </param>
///
/// <returns>   benchmark time in seconds</returns>
///-------------------------------------------------------------------------------------------------


double
kmeans_constmem_memsetreset_shr_map(
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
#if 0
    return 0.0;
#else
    return kmeansraw<DEFAULTRANK,
                     DEFAULTCENTERS,
                     centersmanagerRO<DEFAULTRANK, DEFAULTCENTERS>,
                     accumulatorSMMAP<DEFAULTRANK, DEFAULTCENTERS>>::benchmark(nSteps,
                                                                               lpvPoints,
                                                                               lpvCenters,
                                                                               nPoints,
                                                                               nCenters,
                                                                               hTimer,
                                                                               bVerify,
                                                                               bVerbose);
#endif
}

#if 0
///-------------------------------------------------------------------------------------------------
/// <summary>   basic kmeans implementation, column-major vector layout. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="argc"> The argc. </param>
/// <param name="argv"> [in,out] If non-null, the argv. </param>
///
/// <returns>   benchmark time in seconds</returns>
///-------------------------------------------------------------------------------------------------

double
kmeans_cm(
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
#if 1
    return kmeansraw<DEFAULTRANK,
                     DEFAULTCENTERS,
                     centersmanagerGM<DEFAULTRANK, DEFAULTCENTERS>,
                     accumulatorGM<DEFAULTRANK, DEFAULTCENTERS, false>,
                     false>::benchmark(nSteps,
                                       lpvPoints,
                                       lpvCenters,
                                       nPoints,
                                       nCenters,
                                       hTimer,
                                       bVerify,
                                       bVerbose);
#else
	pt<DEFAULTRANK> * h_InPoints = reinterpret_cast<pt<DEFAULTRANK>*>(lpvPoints);
	pt<DEFAULTRANK> * h_InCenters = reinterpret_cast<pt<DEFAULTRANK>*>(lpvCenters);
    float * h_TxPoints = kmeansraw<DEFAULTRANK,
              DEFAULTCENTERS,
              centersmanagerRO<DEFAULTRANK, DEFAULTCENTERS>,
              accumulatorSMMAP<DEFAULTRANK, DEFAULTCENTERS, false>,
              false>::transpose(h_InPoints, nPoints);
    float * h_TxCenters = kmeansraw<DEFAULTRANK,
              DEFAULTCENTERS,
              centersmanagerRO<DEFAULTRANK, DEFAULTCENTERS>,
              accumulatorSMMAP<DEFAULTRANK, DEFAULTCENTERS, false>,
              false>::transpose(h_InCenters, nCenters);

    float * d_Points = NULL;
	float * d_Centers = NULL;
	int *   d_ClusterIds = NULL;
	int *   d_ClusterCounts = NULL;
	size_t  uiPointsBytes = nPoints * DEFAULTRANK * sizeof(float);
	size_t  uiCentersBytes = nCenters * DEFAULTRANK * sizeof(float);
	size_t  uiClusterIdsBytes = nPoints * sizeof(int);
	size_t  uiClusterCountsBytes = nCenters * sizeof(int);

    INFORM(bVerbose, "Initializing data...\n");
    checkCudaErrors( cudaMalloc((void **)&d_Points, uiPointsBytes ) );
    checkCudaErrors( cudaMalloc((void **)&d_Centers, uiCentersBytes ) );
    checkCudaErrors( cudaMalloc((void **)&d_ClusterIds, uiClusterIdsBytes ) );
    checkCudaErrors( cudaMalloc((void **)&d_ClusterCounts, uiClusterCountsBytes ) );
	checkCudaErrors( cudaMemcpy(d_Points, h_TxPoints, uiPointsBytes, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_Centers, h_TxCenters, uiCentersBytes, cudaMemcpyHostToDevice) );
	INFORM(bVerbose, "Starting up kmeans-cm...\n\n");

    kmeansraw<DEFAULTRANK,
              DEFAULTCENTERS,
              centersmanagerGM<DEFAULTRANK, DEFAULTCENTERS>,
              accumulatorGM<DEFAULTRANK, DEFAULTCENTERS, false>,
              false> * pKMeans = 
        new kmeansraw<DEFAULTRANK,
              DEFAULTCENTERS,
              centersmanagerGM<DEFAULTRANK, DEFAULTCENTERS>,
              accumulatorGM<DEFAULTRANK, DEFAULTCENTERS, false>,
              false>(nSteps,
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
		checkCudaErrors( cudaMemcpy(h_TxCenters, d_Centers, uiCentersBytes, cudaMemcpyDeviceToHost) );
        // PrintCentersTransposed(stdout, DEFAULTRANK, h_TxCenters, nCenters);
        kmeansraw<DEFAULTRANK,
              DEFAULTCENTERS,
              centersmanagerRO<DEFAULTRANK, DEFAULTCENTERS>,
              accumulatorSMMAP<DEFAULTRANK, DEFAULTCENTERS, false>,
              false>::rtranspose(h_TxCenters, nCenters, (float*)h_InCenters);
	}
	shrLog("cleaning up device resources...\n");
	checkCudaErrors( cudaFree((void*)d_Points) );
	checkCudaErrors( cudaFree((void*)d_Centers) );
	checkCudaErrors( cudaFree((void*)d_ClusterIds) );
	checkCudaErrors( cudaFree((void*)d_ClusterCounts) );
    free(h_TxCenters);
    free(h_TxPoints);
	return dAvgSecs;
#endif
}
#endif

///-------------------------------------------------------------------------------------------------
/// <summary>   kmeans implementation, column-major vector layout, uses
///             constant memory to store centers during map phase. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <returns>   benchmark time in seconds</returns>
///-------------------------------------------------------------------------------------------------

double
kmeans_cm_constmem(
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
#if 1
    return kmeansraw<DEFAULTRANK,
                     DEFAULTCENTERS,
                     centersmanagerRO<DEFAULTRANK, DEFAULTCENTERS>,
                     accumulatorGM<DEFAULTRANK, DEFAULTCENTERS, false>,
                     false>::benchmark(nSteps,
                                       lpvPoints,
                                       lpvCenters,
                                       nPoints,
                                       nCenters,
                                       hTimer,
                                       bVerify,
                                       bVerbose);
#else
	pt<DEFAULTRANK> * h_InPoints = reinterpret_cast<pt<DEFAULTRANK>*>(lpvPoints);
	pt<DEFAULTRANK> * h_InCenters = reinterpret_cast<pt<DEFAULTRANK>*>(lpvCenters);
    float * h_TxPoints = transpose(h_InPoints, nPoints);
    float * h_TxCenters = transpose(h_InCenters, nCenters);

    float * d_Points = NULL;
	float * d_Centers = NULL;
	int *   d_ClusterIds = NULL;
	int *   d_ClusterCounts = NULL;
	size_t  uiPointsBytes = nPoints * DEFAULTRANK * sizeof(float);
	size_t  uiCentersBytes = nCenters * DEFAULTRANK * sizeof(float);
	size_t  uiClusterIdsBytes = nPoints * sizeof(int);
	size_t  uiClusterCountsBytes = nCenters * sizeof(int);

    INFORM(bVerbose, "Initializing data...\n");
    checkCudaErrors( cudaMalloc((void **)&d_Points, uiPointsBytes ) );
    checkCudaErrors( cudaMalloc((void **)&d_Centers, uiCentersBytes ) );
    checkCudaErrors( cudaMalloc((void **)&d_ClusterIds, uiClusterIdsBytes ) );
    checkCudaErrors( cudaMalloc((void **)&d_ClusterCounts, uiClusterCountsBytes ) );
	checkCudaErrors( cudaMemcpy(d_Points, h_TxPoints, uiPointsBytes, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_Centers, h_TxCenters, uiCentersBytes, cudaMemcpyHostToDevice) );
    updateConstCentersCM((float*)h_TxCenters, DEFAULTRANK, nCenters);
	
    INFORM(bVerbose, "Starting up kmeans-cm-constmem...\n\n");

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	_vkmeansfcm_cnstmem(nSteps, d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
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
		checkCudaErrors( cudaMemcpy(h_TxCenters, d_Centers, uiCentersBytes, cudaMemcpyDeviceToHost) );
        // PrintCentersTransposed(stdout, DEFAULTRANK, h_TxCenters, nCenters);
        transpose(h_TxCenters, nCenters, DEFAULTRANK, (float*)h_InCenters);
	}
	shrLog("cleaning up device resources...\n");
	checkCudaErrors( cudaFree((void*)d_Points) );
	checkCudaErrors( cudaFree((void*)d_Centers) );
	checkCudaErrors( cudaFree((void*)d_ClusterIds) );
	checkCudaErrors( cudaFree((void*)d_ClusterCounts) );
    free(h_TxCenters);
    free(h_TxPoints);
	return dAvgSecs;
#endif
}

///-------------------------------------------------------------------------------------------------
/// <summary>   kmeans implementation, column-major vector layout, uses
///             constant memory to store centers during map phase. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <returns>   benchmark time in seconds</returns>
///-------------------------------------------------------------------------------------------------

double
kmeans_cm_constmem_shr(
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
#if 1
    return kmeansraw<DEFAULTRANK,
                     DEFAULTCENTERS,
                     centersmanagerRO<DEFAULTRANK, DEFAULTCENTERS>,
                     accumulatorSM<DEFAULTRANK, DEFAULTCENTERS, false>,
                     false>::benchmark(nSteps,
                                       lpvPoints,
                                       lpvCenters,
                                       nPoints,
                                       nCenters,
                                       hTimer,
                                       bVerify,
                                       bVerbose);
#else
	pt<DEFAULTRANK> * h_InPoints = reinterpret_cast<pt<DEFAULTRANK>*>(lpvPoints);
	pt<DEFAULTRANK> * h_InCenters = reinterpret_cast<pt<DEFAULTRANK>*>(lpvCenters);
    float * h_TxPoints = transpose(h_InPoints, nPoints);
    float * h_TxCenters = transpose(h_InCenters, nCenters);

    float * d_Points = NULL;
	float * d_Centers = NULL;
	int *   d_ClusterIds = NULL;
	int *   d_ClusterCounts = NULL;
	size_t  uiPointsBytes = nPoints * DEFAULTRANK * sizeof(float);
	size_t  uiCentersBytes = nCenters * DEFAULTRANK * sizeof(float);
	size_t  uiClusterIdsBytes = nPoints * sizeof(int);
	size_t  uiClusterCountsBytes = nCenters * sizeof(int);

    INFORM(bVerbose, "Initializing data...\n");
    checkCudaErrors( cudaMalloc((void **)&d_Points, uiPointsBytes ) );
    checkCudaErrors( cudaMalloc((void **)&d_Centers, uiCentersBytes ) );
    checkCudaErrors( cudaMalloc((void **)&d_ClusterIds, uiClusterIdsBytes ) );
    checkCudaErrors( cudaMalloc((void **)&d_ClusterCounts, uiClusterCountsBytes ) );
	checkCudaErrors( cudaMemcpy(d_Points, h_TxPoints, uiPointsBytes, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_Centers, h_TxCenters, uiCentersBytes, cudaMemcpyHostToDevice) );
    updateConstCentersCM((float*)h_TxCenters, DEFAULTRANK, nCenters);

	INFORM(bVerbose, "Starting up kmeans-cm-constmem-shr...\n\n");

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	_vkmeansfcm_cnstmem_shr(nSteps, d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
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
		checkCudaErrors( cudaMemcpy(h_TxCenters, d_Centers, uiCentersBytes, cudaMemcpyDeviceToHost) );
        // PrintCentersTransposed(stdout, DEFAULTRANK, h_TxCenters, nCenters);
        transpose(h_TxCenters, nCenters, DEFAULTRANK, (float*)h_InCenters);
	}
	shrLog("cleaning up device resources...\n");
	checkCudaErrors( cudaFree((void*)d_Points) );
	checkCudaErrors( cudaFree((void*)d_Centers) );
	checkCudaErrors( cudaFree((void*)d_ClusterIds) );
	checkCudaErrors( cudaFree((void*)d_ClusterCounts) );
    free(h_TxCenters);
    free(h_TxPoints);
	return dAvgSecs;
#endif
}

///-------------------------------------------------------------------------------------------------
/// <summary>   kmeans implementation, column-major vector layout, uses
///             constant memory to store centers during map phase. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <returns>   benchmark time in seconds</returns>
///-------------------------------------------------------------------------------------------------

double
kmeans_cm_constmem_shr_map(
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
#if 1
    return kmeansraw<DEFAULTRANK,
                     DEFAULTCENTERS,
                     centersmanagerRO<DEFAULTRANK, DEFAULTCENTERS>,
                     accumulatorSMMAP<DEFAULTRANK, DEFAULTCENTERS, false>,
                     false>::benchmark(nSteps,
                                       lpvPoints,
                                       lpvCenters,
                                       nPoints,
                                       nCenters,
                                       hTimer,
                                       bVerify,
                                       bVerbose);
#else
	pt<DEFAULTRANK> * h_InPoints = reinterpret_cast<pt<DEFAULTRANK>*>(lpvPoints);
	pt<DEFAULTRANK> * h_InCenters = reinterpret_cast<pt<DEFAULTRANK>*>(lpvCenters);
    float * h_TxPoints = transpose(h_InPoints, nPoints);
    float * h_TxCenters = transpose(h_InCenters, nCenters);

    float * d_Points = NULL;
	float * d_Centers = NULL;
	int *   d_ClusterIds = NULL;
	int *   d_ClusterCounts = NULL;
	size_t  uiPointsBytes = nPoints * DEFAULTRANK * sizeof(float);
	size_t  uiCentersBytes = nCenters * DEFAULTRANK * sizeof(float);
	size_t  uiClusterIdsBytes = nPoints * sizeof(int);
	size_t  uiClusterCountsBytes = nCenters * sizeof(int);

    INFORM(bVerbose, "Initializing data...\n");
    checkCudaErrors( cudaMalloc((void **)&d_Points, uiPointsBytes ) );
    checkCudaErrors( cudaMalloc((void **)&d_Centers, uiCentersBytes ) );
    checkCudaErrors( cudaMalloc((void **)&d_ClusterIds, uiClusterIdsBytes ) );
    checkCudaErrors( cudaMalloc((void **)&d_ClusterCounts, uiClusterCountsBytes ) );
	checkCudaErrors( cudaMemcpy(d_Points, h_TxPoints, uiPointsBytes, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_Centers, h_TxCenters, uiCentersBytes, cudaMemcpyHostToDevice) );
    updateConstCentersCM((float*)h_TxCenters, DEFAULTRANK, nCenters);

	INFORM(bVerbose, "Starting up kmeans-cm-constmem-shr-map...\n\n");

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	_vkmeansfcm_cnstmem_shr_map(nSteps, d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
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
		checkCudaErrors( cudaMemcpy(h_TxCenters, d_Centers, uiCentersBytes, cudaMemcpyDeviceToHost) );
        // PrintCentersTransposed(stdout, DEFAULTRANK, h_TxCenters, nCenters);
        transpose(h_TxCenters, nCenters, DEFAULTRANK, (float*)h_InCenters);
	}
	shrLog("cleaning up device resources...\n");
	checkCudaErrors( cudaFree((void*)d_Points) );
	checkCudaErrors( cudaFree((void*)d_Centers) );
	checkCudaErrors( cudaFree((void*)d_ClusterIds) );
	checkCudaErrors( cudaFree((void*)d_ClusterCounts) );
    free(h_TxCenters);
    free(h_TxPoints);
	return dAvgSecs;
#endif
}


#if 0
//const int _RANK = DEFAULTRANK;
//const int _NUMCENTERS = DEFAULTCENTERS;
//
//typedef centersmanagerGM<_RANK, _NUMCENTERS> CMGM;
//typedef accumulatorGM<_RANK, _NUMCENTERS, false, false> AMGM;
//typedef kmeansraw<_RANK, _NUMCENTERS, CMGM, AMGM> ktype;
//
//ktype * g_kmeansbasic = NULL;
//
//ktype * 
//CreateKMeans(
//	__in int     nSteps, 
//	__in float * d_Points,  
//	__in float * d_Centers,
//	__in int *   d_ClusterCounts,
//	__in int *   d_ClusterIds,
//	__in int     nPoints,
//	__in int     nCenters
//    )
//{
//    return new ktype(nSteps,
//                     d_Points,
//                     d_Centers,
//                     d_ClusterCounts,
//                     d_ClusterIds,
//                     nPoints,
//                     nCenters);
//}
//
//void 
//dokmeans(
//	__in int     nSteps, 
//	__in float * d_Points,  
//	__in float * d_Centers,
//	__in int *   d_ClusterCounts,
//	__in int *   d_ClusterIds,
//	__in int     nPoints,
//	__in int     nCenters
//    )
//{
//    g_kmeansbasic = CreateKMeans(nSteps, d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
//    g_kmeansbasic->execute();
//}

#if 0

    void 
    kmeansraw::updateconstmemH(
        float * h_Centers
        )
    {
        cudaMemcpyToSymbol(kmeansraw<DIM,V,C>::d_cnst_centers, h_Centers, sizeof(float)*DIM*C);
    }
    
    void 
    kmeansraw::updateconstmemD(
        float * d_Centers
        )
    {
    }

void 
kmeansraw::updateConstCenters(float * h_Centers, int nRank, int nCenters)
{
    assert(nRank == RANK);
    assert(nCenters == NUMCENTERS);
    cudaMemcpyToSymbol( d_cnst_centers, h_Centers, sizeof(float)*nRank*nCenters );
}

///-------------------------------------------------------------------------------------------------
/// <summary>   euclidean distance over floats. </summary>
///
/// <param name="a">    [in,out] The pt&lt;R&gt; &amp; to process. </param>
/// <param name="b">    [in,out] The pt&lt;R&gt; &amp; to process. </param>
///
/// <returns>   distance. </returns>
///-------------------------------------------------------------------------------------------------

float __device__ __host__ 
_vdistancef(
	int R,
    float * a,
    float * b
    ) 
{
    float accum = 0.0f;
    for(int i=0; i<R; i++) {
        float delta = a[i]-b[i];
        accum += delta*delta;
    }
    return sqrt(accum);
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Find the nearest center to the given point. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <typeparam name="int N">    Type of the int n. </typeparam>
/// <param name="point">        [in,out] The point. </param>
/// <param name="centers">      [in,out] The centers. </param>
///
/// <returns>   the index of the nearest center </returns>
///-------------------------------------------------------------------------------------------------

__device__ __host__ int 
__vNearestCenterF(
	int R,
	int C,
    float * point,
    float * centers
    ) 
{
    float mindist = FLT_MAX;
    int minidx = 0;
	int clistidx = 0;
    for(int i=0; i<C;i++) {
		clistidx = i*R;
        float dist = _vdistancef(R, point, &centers[clistidx]);
        if(dist < mindist) {
            minidx = static_cast<int>(i);
            mindist = dist;
        }
    }
    return minidx;
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Find the nearest center to the given point. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <typeparam name="int N">    Type of the int n. </typeparam>
/// <param name="point">        [in,out] The point. </param>
/// <param name="centers">      [in,out] The centers. </param>
///
/// <returns>   the index of the nearest center </returns>
///-------------------------------------------------------------------------------------------------

__device__ int 
__vNearestCenterF_cnst(
	int R,
	int C,
    float * point
    ) 
{
    float mindist = FLT_MAX;
    int minidx = 0;
	int clistidx = 0;
    for(int i=0; i<C;i++) {
		clistidx = i*R;
        float dist = _vdistancef(R, point, &d_cnst_centers[clistidx]);
        if(dist < mindist) {
            minidx = static_cast<int>(i);
            mindist = dist;
        }
    }
    return minidx;
}


///-------------------------------------------------------------------------------------------------
/// <summary>	Map points to centers. </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="d_Points">	   	[in,out] If non-null, the points. </param>
/// <param name="d_Centers">   	[in,out] If non-null, the centers. </param>
/// <param name="d_ClusterIds">	[in,out] If non-null, list of identifiers for the clusters. </param>
/// <param name="nPoints">	   	The points. </param>
/// <param name="nCenters">	   	The centers. </param>
///-------------------------------------------------------------------------------------------------

template<int R, int C> 
__global__ void 
__vmapPointsToCentersF(
    __in  float * d_Points,
    __in  float * d_Centers,
    __out int *   d_ClusterIds,
	__in  int     nPoints,
	__in  int     nCenters
	)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx > nPoints) return;
	d_ClusterIds[idx] = __vNearestCenterF(R, C, &d_Points[idx*R], d_Centers);
}

///-------------------------------------------------------------------------------------------------
/// <summary>	Map points to centers. </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="d_Points">	   	[in,out] If non-null, the points. </param>
/// <param name="d_Centers">   	[in,out] If non-null, the centers. </param>
/// <param name="d_ClusterIds">	[in,out] If non-null, list of identifiers for the clusters. </param>
/// <param name="nPoints">	   	The points. </param>
/// <param name="nCenters">	   	The centers. </param>
///-------------------------------------------------------------------------------------------------

template<int R, int C> 
__global__ void 
__vmapPointsToCentersF_cnst(
    __in  float * d_Points,
    __out int *   d_ClusterIds,
	__in  int     nPoints
	)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx > nPoints) return;
	d_ClusterIds[idx] = __vNearestCenterF_cnst(R, C, &d_Points[idx*R]);
}

///-------------------------------------------------------------------------------------------------
/// <summary>	Sets a point. </summary>
///
/// <remarks>	crossbac, 7/29/2013. </remarks>
///
/// <typeparam name="int R">	Type of the int r. </typeparam>
/// <param name="fValue">	[in,out] If non-null, the value. </param>
///-------------------------------------------------------------------------------------------------

template<int R> 
__device__ void
__setPoint(
	__in float * d_pPoint,
	__in float   fValue
	)
{ 
	for(int i=0;i<R;i++) {
		d_pPoint[i] = fValue;
	}
}

///-------------------------------------------------------------------------------------------------
/// <summary>	zero the center accumulators and count accumulators. </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="d_Points">	   	[in,out] If non-null, the points. </param>
/// <param name="d_Centers">   	[in,out] If non-null, the centers. </param>
/// <param name="d_ClusterIds">	[in,out] If non-null, list of identifiers for the clusters. </param>
/// <param name="nPoints">	   	The points. </param>
/// <param name="nCenters">	   	The centers. </param>
///-------------------------------------------------------------------------------------------------

template<int R, int C> 
__global__ void 
__vResetAccumulatorsF(
    __out float * d_Centers,
	__out int *   d_ClusterCounts,
	__in  int     nCenters
	)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx > C) return;
	// TODO: refine this to map a hw thread per 
	// element of each vector rather than serializing
	__setPoint<R>(&d_Centers[idx*R], 0.0f);
	d_ClusterCounts[idx] = 0;
}

///-------------------------------------------------------------------------------------------------
/// <summary>	zero the center accumulators and count accumulators. </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="d_Points">	   	[in,out] If non-null, the points. </param>
/// <param name="d_Centers">   	[in,out] If non-null, the centers. </param>
/// <param name="d_ClusterIds">	[in,out] If non-null, list of identifiers for the clusters. </param>
/// <param name="nPoints">	   	The points. </param>
/// <param name="nCenters">	   	The centers. </param>
///-------------------------------------------------------------------------------------------------

template<int R, int C> void 
__vMemsetResetAccumulatorsF(
    __out float * d_Centers,
	__out int *   d_ClusterCounts,
	__in  int     nCenters
	)
{
    cudaMemset(d_Centers, 0, R*C*sizeof(float));
    cudaMemset(d_ClusterCounts, 0,C*sizeof(int));
}

///-------------------------------------------------------------------------------------------------
/// <summary>	accumulate centers. </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="d_Points">	   	[in,out] If non-null, the points. </param>
/// <param name="d_Centers">   	[in,out] If non-null, the centers. </param>
/// <param name="d_ClusterIds">	[in,out] If non-null, list of identifiers for the clusters. </param>
/// <param name="nPoints">	   	The points. </param>
/// <param name="nCenters">	   	The centers. </param>
///-------------------------------------------------------------------------------------------------

template<int R, int C> 
__global__ void 
__vAccumulateCentersF(
    __in    float * d_Points,
    __in    float * d_Centers,
	__inout int *   d_ClusterCounts,
    __inout int *   d_ClusterIds,
	__in    int     nPoints,
	__in    int     nCenters
	)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx > nPoints) return;
	int clusterid = d_ClusterIds[idx];

	// TODO: refine this to map a hw thread per 
	// element of each vector rather than serializing
	for(int i=0;i<R;i++) {
		atomicAdd(&d_Centers[(clusterid*R)+i], d_Points[(idx*R)+i]);
	}
	atomicAdd(&d_ClusterCounts[clusterid], 1);
}

///-------------------------------------------------------------------------------------------------
/// <summary>	accumulate centers into a shared mem array. </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="d_Points">	   	[in,out] If non-null, the points. </param>
/// <param name="d_Centers">   	[in,out] If non-null, the centers. </param>
/// <param name="d_ClusterIds">	[in,out] If non-null, list of identifiers for the clusters. </param>
/// <param name="nPoints">	   	The points. </param>
/// <param name="nCenters">	   	The centers. </param>
///-------------------------------------------------------------------------------------------------

template<int R, int C> 
__global__ void 
__vAccumulateCentersF_shr(
    __in    float * d_Points,
    __in    float * d_Centers,
	__inout int *   d_ClusterCounts,
    __inout int *   d_ClusterIds,
	__in    int     nPoints,
	__in    int     nCenters
	)
{
    __shared__ float accums[R*C];
    __shared__ int cnts[C];
    dassert(blockDim.x == RANK*NUMCENTERS);
    accums[threadIdx.x] = 0.0f;
    if(threadIdx.x < C) cnts[threadIdx.x/R] = 0;
    __syncthreads();
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < nPoints) {
        int clusterid = d_ClusterIds[idx];
        for(int i=0;i<R;i++) 
            atomicAdd(&accums[(clusterid*R)+i], d_Points[(idx*R)+i]);
        atomicAdd(&cnts[clusterid], 1);
    }
    __syncthreads();
    atomicAdd(&d_Centers[threadIdx.x], accums[threadIdx.x]);
    if(threadIdx.x < C) atomicAdd(&d_ClusterCounts[threadIdx.x], cnts[threadIdx.x]);
}

///-------------------------------------------------------------------------------------------------
/// <summary>	final computation of cluster means. </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="d_Points">	   	[in,out] If non-null, the points. </param>
/// <param name="d_Centers">   	[in,out] If non-null, the centers. </param>
/// <param name="d_ClusterIds">	[in,out] If non-null, list of identifiers for the clusters. </param>
/// <param name="nPoints">	   	The points. </param>
/// <param name="nCenters">	   	The centers. </param>
///-------------------------------------------------------------------------------------------------

template<int R, int C> 
__global__ void 
__vUpdateCentersF(
    __inout float * d_Centers,
	__in    int *   d_ClusterCounts,
	__in    int     nCenters
	)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx > C) return;
	for(int i=0;i<R;i++) {
		// TODO: refine this to map a hw thread per 
		// element of each vector rather than serializing
		d_Centers[(idx*R)+i] /= d_ClusterCounts[idx];
	}
}

///-------------------------------------------------------------------------------------------------
/// <summary>	final computation of cluster means. </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="d_Points">	   	[in,out] If non-null, the points. </param>
/// <param name="d_Centers">   	[in,out] If non-null, the centers. </param>
/// <param name="d_ClusterIds">	[in,out] If non-null, list of identifiers for the clusters. </param>
/// <param name="nPoints">	   	The points. </param>
/// <param name="nCenters">	   	The centers. </param>
///-------------------------------------------------------------------------------------------------

template<int R, int C> 
__global__ void 
__vUpdateCentersF_shr(
    __inout float * d_Centers,
	__in    int *   d_ClusterCounts,
	__in    int     nCenters
	)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int cidx = idx/R;
    d_Centers[idx] /= d_ClusterCounts[cidx];
}


///-------------------------------------------------------------------------------------------------
/// <summary>	Kmeans on GPU. </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="nSteps">	   	The steps. </param>
/// <param name="d_Points">	   	[in,out] If non-null, the points. </param>
/// <param name="d_Centers">   	[in,out] If non-null, the centers. </param>
/// <param name="d_ClusterIds">	[in,out] If non-null, list of identifiers for the clusters. </param>
///-------------------------------------------------------------------------------------------------

extern "C" void 
_vkmeansf(
	__in int     nSteps, 
	__in float * d_Points,  
	__in float * d_Centers,
	__in int *   d_ClusterCounts,
	__in int *   d_ClusterIds,
	__in int     nPoints,
	__in int     nCenters
	)
{
    const uint nPointsBlocks = iDivUp(nPoints, THREADBLOCK_SIZE);
    const uint nCentersBlocks = iDivUp(nCenters, THREADBLOCK_SIZE);
	for(int i=0; i<nSteps; i++) {

		__vmapPointsToCentersF<RANK,NUMCENTERS><<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__vmapPointsToCentersF() execution failed\n");

		__vResetAccumulatorsF<RANK,NUMCENTERS><<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__vResetAccumulatorsF() execution failed\n");

		__vAccumulateCentersF<RANK, NUMCENTERS><<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__vAccumulateCentersF() execution failed\n");

		__vUpdateCentersF<RANK,NUMCENTERS><<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__vUpdateCentersF() execution failed\n");
	}
}

///-------------------------------------------------------------------------------------------------
/// <summary>	Kmeans on GPU, points in constant memory </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="nSteps">	   	The steps. </param>
/// <param name="d_Points">	   	[in,out] If non-null, the points. </param>
/// <param name="d_Centers">   	[in,out] If non-null, the centers. </param>
/// <param name="d_ClusterIds">	[in,out] If non-null, list of identifiers for the clusters. </param>
///-------------------------------------------------------------------------------------------------

extern "C" void 
_vkmeansf_cnstmem(
	__in int     nSteps, 
	__in float * d_Points,  
	__in float * d_Centers,
	__in int *   d_ClusterCounts,
	__in int *   d_ClusterIds,
	__in int     nPoints,
	__in int     nCenters
	)
{
    const uint nPointsBlocks = iDivUp(nPoints, THREADBLOCK_SIZE);
    const uint nCentersBlocks = iDivUp(nCenters, THREADBLOCK_SIZE);
	for(int i=0; i<nSteps; i++) {

		__vmapPointsToCentersF_cnst<RANK,NUMCENTERS><<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_ClusterIds, nPoints);
		getLastCudaError("__vmapPointsToCentersF() execution failed\n");

		__vResetAccumulatorsF<RANK,NUMCENTERS><<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__vResetAccumulatorsF() execution failed\n");

		__vAccumulateCentersF<RANK, NUMCENTERS><<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__vAccumulateCentersF() execution failed\n");

		__vUpdateCentersF<RANK,NUMCENTERS><<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__vUpdateCentersF() execution failed\n");
	}
}

///-------------------------------------------------------------------------------------------------
/// <summary>	Kmeans on GPU, points in constant memory, accumulators reset
///             using device memset rather than through kernel code.  </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="nSteps">	   	The steps. </param>
/// <param name="d_Points">	   	[in,out] If non-null, the points. </param>
/// <param name="d_Centers">   	[in,out] If non-null, the centers. </param>
/// <param name="d_ClusterIds">	[in,out] If non-null, list of identifiers for the clusters. </param>
///-------------------------------------------------------------------------------------------------

extern "C" void 
_vkmeansf_cnstmem_memsetreset(
	__in int     nSteps, 
	__in float * d_Points,  
	__in float * d_Centers,
	__in int *   d_ClusterCounts,
	__in int *   d_ClusterIds,
	__in int     nPoints,
	__in int     nCenters
	)
{
    const uint nPointsBlocks = iDivUp(nPoints, THREADBLOCK_SIZE);
    const uint nCentersBlocks = iDivUp(nCenters, THREADBLOCK_SIZE);
	for(int i=0; i<nSteps; i++) {

		__vmapPointsToCentersF_cnst<RANK,NUMCENTERS><<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_ClusterIds, nPoints);
		getLastCudaError("__vmapPointsToCentersF() execution failed\n");

		__vMemsetResetAccumulatorsF<RANK,NUMCENTERS>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__vMemsetResetAccumulatorsF() execution failed\n");

		__vAccumulateCentersF<RANK, NUMCENTERS><<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__vAccumulateCentersF() execution failed\n");

		__vUpdateCentersF<RANK,NUMCENTERS><<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__vUpdateCentersF() execution failed\n");
	}
}

///-------------------------------------------------------------------------------------------------
/// <summary>	Kmeans on GPU, points in constant memory, accumulators reset
///             using device memset rather than through kernel code, accumulations
///             privatized per block in shared memory.  </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="nSteps">	   	The steps. </param>
/// <param name="d_Points">	   	[in,out] If non-null, the points. </param>
/// <param name="d_Centers">   	[in,out] If non-null, the centers. </param>
/// <param name="d_ClusterIds">	[in,out] If non-null, list of identifiers for the clusters. </param>
///-------------------------------------------------------------------------------------------------

extern "C" void 
_vkmeansf_cnstmem_memsetreset_shraccum(
	__in int     nSteps, 
	__in float * d_Points,  
	__in float * d_Centers,
	__in int *   d_ClusterCounts,
	__in int *   d_ClusterIds,
	__in int     nPoints,
	__in int     nCenters
	)
{
    const uint nPointsBlocks = iDivUp(nPoints, THREADBLOCK_SIZE);
    const uint nCentersBlocks = iDivUp(nCenters, THREADBLOCK_SIZE);
    const uint nAccumPointsBlocks = iDivUp(nPoints, RANK*NUMCENTERS);
	for(int i=0; i<nSteps; i++) {

		__vmapPointsToCentersF_cnst<RANK,NUMCENTERS><<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_ClusterIds, nPoints);
		getLastCudaError("__vmapPointsToCentersF() execution failed\n");

		__vMemsetResetAccumulatorsF<RANK,NUMCENTERS>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__vMemsetResetAccumulatorsF() execution failed\n");

		__vAccumulateCentersF_shr<RANK, NUMCENTERS><<<nAccumPointsBlocks, RANK*NUMCENTERS>>>(d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__vAccumulateCentersF_shr() execution failed\n");

		 __vUpdateCentersF<RANK,NUMCENTERS><<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
		 getLastCudaError("__vUpdateCentersF() execution failed\n");
	}
}

///-------------------------------------------------------------------------------------------------
/// <summary>	Kmeans on GPU, points in constant memory, accumulators reset
///             using device memset rather than through kernel code, accumulations
///             privatized per block in shared memory, and final update HW thread
///             to item mapping changed to improve coalescing.  </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="nSteps">	   	The steps. </param>
/// <param name="d_Points">	   	[in,out] If non-null, the points. </param>
/// <param name="d_Centers">   	[in,out] If non-null, the centers. </param>
/// <param name="d_ClusterIds">	[in,out] If non-null, list of identifiers for the clusters. </param>
///-------------------------------------------------------------------------------------------------

extern "C" void 
_vkmeansf_cnstmem_memsetreset_shraccum_map(
	__in int     nSteps, 
	__in float * d_Points,  
	__in float * d_Centers,
	__in int *   d_ClusterCounts,
	__in int *   d_ClusterIds,
	__in int     nPoints,
	__in int     nCenters
	)
{
    const uint nPointsBlocks = iDivUp(nPoints, THREADBLOCK_SIZE);
    const uint nCentersBlocks = iDivUp(nCenters, THREADBLOCK_SIZE);
    const uint nAccumPointsBlocks = iDivUp(nPoints, RANK*NUMCENTERS);
	for(int i=0; i<nSteps; i++) {

		__vmapPointsToCentersF_cnst<RANK,NUMCENTERS><<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_ClusterIds, nPoints);
		getLastCudaError("__vmapPointsToCentersF() execution failed\n");

		__vMemsetResetAccumulatorsF<RANK,NUMCENTERS>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__vMemsetResetAccumulatorsF() execution failed\n");

		__vAccumulateCentersF_shr<RANK, NUMCENTERS><<<nAccumPointsBlocks, RANK*NUMCENTERS>>>(d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__vAccumulateCentersF_shr() execution failed\n");

		 __vUpdateCentersF_shr<RANK,NUMCENTERS><<<1, RANK*NUMCENTERS>>>(d_Centers, d_ClusterCounts, nCenters);
		 getLastCudaError("__vUpdateCentersF() execution failed\n");
	}
}
#endif
#endif