///-------------------------------------------------------------------------------------------------
// file:	kmeanscm.cu
//
// summary:	kmeans implementation over extents of floats (no underlying point/vector struct)
//          This variant lays out vectors in column-major format to promote better memory
//          coalescing on the GPU. 
///-------------------------------------------------------------------------------------------------

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cutil_inline.h>
#include "kmeanscm.cu.h"

#ifdef RANK

__constant__ float d_cnst_centers_cm[RANK*NUMCENTERS];

extern "C"
void updateConstCentersCM(float * h_Centers, int nRank, int nCenters)
{
    assert(nRank == RANK);
    assert(nCenters == NUMCENTERS);
    cudaMemcpyToSymbol( d_cnst_centers_cm, h_Centers, sizeof(float)*nRank*nCenters );
}

///-------------------------------------------------------------------------------------------------
/// <summary>   euclidean distance over float vectors, in column-major order. </summary>
///
/// <param name="a">    [in,out] The pt&lt;R&gt; &amp; to process. </param>
/// <param name="b">    [in,out] The pt&lt;R&gt; &amp; to process. </param>
///
/// <returns>   distance. </returns>
///-------------------------------------------------------------------------------------------------

__device__ float 
_vdistancefcm(
    int R,
    int nAIndex,
    float * pAVectors,    
    int nAVectorCount,
    int nBIndex,
    float * pBVectors,
    int nBVectorCount
    ) 
{
    // assumes perfect packing 
    // (no trailing per-row pitch) 
    float accum = 0.0f;
    float * pAStart = &pAVectors[nAIndex];
    float * pBStart = &pBVectors[nBIndex];
    for(int i=0; i<R; i++) {
        float a = (*(pAStart + i*nAVectorCount));
        float b = (*(pBStart + i*nBVectorCount));
        float delta = a-b;
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

__device__ int 
__vNearestCenterFcm(
	int R,
	int C,
    float * points,
    int nPointIndex,
    int nPoints,
    float * centers
    ) 
{
    float mindist = FLT_MAX;
    int minidx = 0;
    for(int i=0; i<C;i++) {
        float dist = _vdistancefcm(R, nPointIndex, points, nPoints, i, centers, C);
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
__vNearestCenterFcm_cnst(
	int R,
	int C,
    float * points,
    int nPointIndex,
    int nPoints
    ) 
{
    float mindist = FLT_MAX;
    int minidx = 0;
    for(int i=0; i<C;i++) {
        float dist = _vdistancefcm(R, nPointIndex, points, nPoints, i, d_cnst_centers_cm, C);
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
__vmapPointsToCentersFcm(
    __in  float * d_Points,
    __in  float * d_Centers,
    __out int *   d_ClusterIds,
	__in  int     nPoints,
	__in  int     nCenters
	)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx > nPoints) return;
	d_ClusterIds[idx] = __vNearestCenterFcm(R, C, d_Points, idx, nPoints, d_Centers);
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
__vmapPointsToCentersFcm_cnst(
    __in  float * d_Points,
    __out int *   d_ClusterIds,
	__in  int     nPoints
	)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx > nPoints) return;
	d_ClusterIds[idx] = __vNearestCenterFcm_cnst(R, C, d_Points, idx, nPoints);
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
__vMemsetResetAccumulatorsFcm(
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
__vAccumulateCentersFcm(
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
		atomicAdd(&d_Centers[clusterid+(i*nCenters)], d_Points[idx+(i*nPoints)]);
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
__vAccumulateCentersFcm_shr(
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
            atomicAdd(&accums[clusterid+(nCenters*i)], d_Points[idx+(nPoints*i)]);
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
__vUpdateCentersFcm(
    __inout float * d_Centers,
	__in    int *   d_ClusterCounts,
	__in    int     nCenters
	)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= C) return;
	for(int i=0;i<R;i++) {
		// TODO: refine this to map a hw thread per 
		// element of each vector rather than serializing
		d_Centers[idx+(i*nCenters)] /= d_ClusterCounts[idx];
	}
    //printf("C[%d]:"
    //       " %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f"
    //       " %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f"
    //       " %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
    //       idx,
    //       d_Centers[idx+(0*nCenters)],
    //       d_Centers[idx+(1*nCenters)],
    //       d_Centers[idx+(2*nCenters)],
    //       d_Centers[idx+(3*nCenters)],
    //       d_Centers[idx+(4*nCenters)],
    //       d_Centers[idx+(5*nCenters)],
    //       d_Centers[idx+(6*nCenters)],
    //       d_Centers[idx+(7*nCenters)],
    //       d_Centers[idx+(8*nCenters)],
    //       d_Centers[idx+(9*nCenters)],
    //       d_Centers[idx+(10*nCenters)],
    //       d_Centers[idx+(11*nCenters)],
    //       d_Centers[idx+(12*nCenters)],
    //       d_Centers[idx+(13*nCenters)],
    //       d_Centers[idx+(14*nCenters)],
    //       d_Centers[idx+(15*nCenters)],
    //       d_Centers[idx+(16*nCenters)],
    //       d_Centers[idx+(17*nCenters)],
    //       d_Centers[idx+(18*nCenters)],
    //       d_Centers[idx+(19*nCenters)],
    //       d_Centers[idx+(20*nCenters)],
    //       d_Centers[idx+(21*nCenters)],
    //       d_Centers[idx+(22*nCenters)],
    //       d_Centers[idx+(23*nCenters)]);
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
__vUpdateCentersFcm_shr(
    __inout float * d_Centers,
	__in    int *   d_ClusterCounts,
	__in    int     nCenters
	)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int cidx = idx % nCenters;
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
_vkmeansfcm(
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

		__vmapPointsToCentersFcm<RANK,NUMCENTERS><<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__vmapPointsToCentersFcm() execution failed\n");

		__vMemsetResetAccumulatorsFcm<RANK,NUMCENTERS>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__vMemsetResetAccumulatorsFcm() execution failed\n");

		__vAccumulateCentersFcm<RANK, NUMCENTERS><<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__vAccumulateCentersFcm() execution failed\n");

		__vUpdateCentersFcm<RANK,NUMCENTERS><<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__vUpdateCentersFcm() execution failed\n");
	}
}

///-------------------------------------------------------------------------------------------------
/// <summary>	Kmeans on GPU, column-major vector layout, 
///             points in constant memory, device memset (instead of device code) 
///             used to reset the accumulator arrays between steps. 
///             </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="nSteps">	   	The steps. </param>
/// <param name="d_Points">	   	[in,out] If non-null, the points. </param>
/// <param name="d_Centers">   	[in,out] If non-null, the centers. </param>
/// <param name="d_ClusterIds">	[in,out] If non-null, list of identifiers for the clusters. </param>
///-------------------------------------------------------------------------------------------------

extern "C" void 
_vkmeansfcm_cnstmem(
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

		__vmapPointsToCentersFcm_cnst<RANK,NUMCENTERS><<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_ClusterIds, nPoints);
		getLastCudaError("__vmapPointsToCentersFcm_cnst() execution failed\n");

		__vMemsetResetAccumulatorsFcm<RANK,NUMCENTERS>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__vMemsetResetAccumulatorsFcm() execution failed\n");

		__vAccumulateCentersFcm<RANK, NUMCENTERS><<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__vAccumulateCentersF() execution failed\n");

		__vUpdateCentersFcm<RANK,NUMCENTERS><<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
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
_vkmeansfcm_cnstmem_shr(
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

		__vmapPointsToCentersFcm_cnst<RANK,NUMCENTERS><<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_ClusterIds, nPoints);
		getLastCudaError("__vmapPointsToCentersF() execution failed\n");

		__vMemsetResetAccumulatorsFcm<RANK,NUMCENTERS>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__vMemsetResetAccumulatorsF() execution failed\n");

		__vAccumulateCentersFcm_shr<RANK, NUMCENTERS><<<nAccumPointsBlocks, RANK*NUMCENTERS>>>(d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__vAccumulateCentersF_shr() execution failed\n");

		 __vUpdateCentersFcm<RANK,NUMCENTERS><<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
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
_vkmeansfcm_cnstmem_shr_map(
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

		__vmapPointsToCentersFcm_cnst<RANK,NUMCENTERS><<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_ClusterIds, nPoints);
		getLastCudaError("__vmapPointsToCentersF() execution failed\n");

		__vMemsetResetAccumulatorsFcm<RANK,NUMCENTERS>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__vMemsetResetAccumulatorsF() execution failed\n");

		__vAccumulateCentersFcm_shr<RANK, NUMCENTERS><<<nAccumPointsBlocks, RANK*NUMCENTERS>>>(d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__vAccumulateCentersF_shr() execution failed\n");

		 __vUpdateCentersFcm_shr<RANK,NUMCENTERS><<<1, RANK*NUMCENTERS>>>(d_Centers, d_ClusterCounts, nCenters);
		 getLastCudaError("__vUpdateCentersF() execution failed\n");
	}
}
#endif