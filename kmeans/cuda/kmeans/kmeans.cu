///-------------------------------------------------------------------------------------------------
// file:	kmeans.cu
//
// summary:	kmeans class
///-------------------------------------------------------------------------------------------------

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cutil_inline.h>
#include "kmeans.cu.h"

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

template<int R> __global__ void 
__mapPointsToCenters(
    pt<R> *d_Points,
    pt<R> *d_Centers,
    int *d_ClusterIds,
	int nPoints,
	int nCenters
	)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx > nPoints) return;
    float mindist = FLT_MAX;
    int minidx = 0;
    for(int i=0; i<nCenters;i++) {
        float dist = distance<R>(d_Points[idx], d_Centers[i]);
        if(dist < mindist) {
            minidx = i;
            mindist = dist;
        }
    }
    d_ClusterIds[idx] = minidx;
}

///-------------------------------------------------------------------------------------------------
/// <summary>	Map points to centers, using shared memory to cache the centers
/// 			explicitly beforehand. </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="d_Points">	   	[in,out] If non-null, the points. </param>
/// <param name="d_Centers">   	[in,out] If non-null, the centers. </param>
/// <param name="d_ClusterIds">	[in,out] If non-null, list of identifiers for the clusters. </param>
/// <param name="nPoints">	   	The points. </param>
/// <param name="nCenters">	   	The centers. </param>
///-------------------------------------------------------------------------------------------------
#if 0
template<int _RANK, int _CENTERS> __global__ void 
__mapPointsToCenters_shmem(
    pt<_RANK> *d_Points,
    pt<_RANK> *d_Centers,
    int *d_ClusterIds,
	int nPoints,
	int nCenters
	)
{
	const int _NCFLOATS = _RANK*_CENTERS;
	__shared__ float d_shmemCenters[_NCFLOATS];
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx > nPoints) return;
	if(idx < _NCFLOATS) {
		int nVectorIdx = idx / _RANK;
		int nFloatIdx = idx % _RANK;
		d_shmemCenters[idx] = d_Centers[nVectorIdx].m_v[nFloatIdx];
	}
	__syncthreads();
    float mindist = FLT_MAX;
    int minidx = 0;
    for(int i=0; i<nCenters;i++) {
		int nVectorIdx = idx / _RANK;
        float dist = distance<_RANK>(d_Points[idx], &d_shmemCenters[]);
        if(dist < mindist) {
            minidx = i;
            mindist = dist;
        }
    }
    d_ClusterIds[idx] = minidx;
}
#endif

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

template<int R> __global__ void 
__resetAccumulators(
    pt<R> *d_Centers,
	int *d_ClusterCounts,
	int nCenters
	)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx > nCenters) return;
	d_Centers[idx].set(0.0f);
	d_ClusterCounts[idx] = 0;
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

template<int R> __global__ void 
__accumulateCenters(
    pt<R> *d_Points,
    pt<R> *d_Centers,
	int *d_ClusterCounts,
    int *d_ClusterIds,
	int nPoints,
	int nCenters
	)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx > nPoints) return;
	int clusterid = d_ClusterIds[idx];
	for(int i=0;i<R;i++) 
		atomicAdd(&d_Centers[clusterid].m_v[i], d_Points[idx].m_v[i]);
	atomicAdd(&d_ClusterCounts[clusterid], 1);
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

template<int R> __global__ void 
__updateCenters(
    pt<R> *d_Centers,
	int *d_ClusterCounts,
	int nCenters
	)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx > nCenters) return;
	d_Centers[idx] /= d_ClusterCounts[idx];
}

#if 0
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
kmeans(
	int nSteps, 
	pt<RANK>* d_Points,  
	pt<RANK>* d_Centers,
	int* d_ClusterCounts,
	int* d_ClusterIds,
	int nPoints,
	int nCenters
	)
{
    const uint nPointsBlocks = iDivUp(nPoints, THREADBLOCK_SIZE);
    const uint nCentersBlocks = iDivUp(nCenters, THREADBLOCK_SIZE);
	for(int i=0; i<nSteps; i++) {

		__mapPointsToCenters<<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__mapPointsToCenters() execution failed\n");

		__resetAccumulators<<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__resetAccumulators() execution failed\n");

		__accumulateCenters<<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__updateCenters() execution failed\n");

		__updateCenters<<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__updateCenters() execution failed\n");
	}
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
kmeans_shmem(
	int nSteps, 
	pt<RANK>* d_Points,  
	pt<RANK>* d_Centers,
	int* d_ClusterCounts,
	int* d_ClusterIds,
	int nPoints,
	int nCenters
	)
{
    const uint nPointsBlocks = iDivUp(nPoints, THREADBLOCK_SIZE);
    const uint nCentersBlocks = iDivUp(nCenters, THREADBLOCK_SIZE);
	for(int i=0; i<nSteps; i++) {

		__mapPointsToCenters<<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__mapPointsToCenters() execution failed\n");

		__resetAccumulators<<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__resetAccumulators() execution failed\n");

		__accumulateCenters<<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__updateCenters() execution failed\n");

		__updateCenters<<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__updateCenters() execution failed\n");
	}
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
kmeans_constmem(
	int nSteps, 
	pt<RANK>* d_Points,  
	pt<RANK>* d_Centers,
	int* d_ClusterCounts,
	int* d_ClusterIds,
	int nPoints,
	int nCenters
	)
{
    const uint nPointsBlocks = iDivUp(nPoints, THREADBLOCK_SIZE);
    const uint nCentersBlocks = iDivUp(nCenters, THREADBLOCK_SIZE);
	for(int i=0; i<nSteps; i++) {

		__mapPointsToCenters<<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__mapPointsToCenters() execution failed\n");

		__resetAccumulators<<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__resetAccumulators() execution failed\n");

		__accumulateCenters<<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
		getLastCudaError("__updateCenters() execution failed\n");

		__updateCenters<<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
		getLastCudaError("__updateCenters() execution failed\n");
	}
}


///-------------------------------------------------------------------------------------------------
/// <summary>	compute for each center, the sum and count of all the nearest points. </summary>
///
/// <remarks>	Chris Rossbach originally. </remarks>
///
/// <param name="d_Points">	   	    [in,out] the points. </param>
/// <param name="d_Centers">   	    [in] the centers, [out] the sum of all the nearest points </param>
/// <param name="d_ClusterIds">	    [in,out] for each point, which center is nearest. </param>
/// <param name="d_ClusterCounts">	[in,out] for each center, how many points are nearest it. </param>
///-------------------------------------------------------------------------------------------------
extern "C" void 
kmeans_step1(
	pt<RANK>* d_Points,  
	pt<RANK>* d_Centers,
	int* d_ClusterCounts,
	int* d_ClusterIds,
	int nPoints,
	int nCenters
	)
{
    const uint nPointsBlocks = iDivUp(nPoints, THREADBLOCK_SIZE);
    const uint nCentersBlocks = iDivUp(nCenters, THREADBLOCK_SIZE);

	__mapPointsToCenters<<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterIds, nPoints, nCenters);
	getLastCudaError("__mapPointsToCenters() execution failed\n");

	__resetAccumulators<<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
	getLastCudaError("__resetAccumulators() execution failed\n");

	__accumulateCenters<<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
	getLastCudaError("__updateCenters() execution failed\n");

}

///-------------------------------------------------------------------------------------------------
/// <summary>	Given the sum and count of all the nearest points, computes the new centers </summary>
///
/// <remarks>	Chris Rossbach originally. </remarks>
///
/// <param name="d_Points">	   	    [in,out] the points. </param>
/// <param name="d_Centers">   	    [in] the sum of all the nearest points [out] the centers </param>
/// <param name="d_ClusterIds">	    [in,out] for each point, which center is nearest. </param>
/// <param name="d_ClusterCounts">	[in] for each center, how many points are nearest it. </param>
///-------------------------------------------------------------------------------------------------
extern "C" void 
kmeans_step2(
	pt<RANK>* d_Points,  
	pt<RANK>* d_Centers,
	int* d_ClusterCounts,
	int* d_ClusterIds,
	int nPoints,
	int nCenters
	)
{
    const uint nPointsBlocks = iDivUp(nPoints, THREADBLOCK_SIZE);
    const uint nCentersBlocks = iDivUp(nCenters, THREADBLOCK_SIZE);

	__updateCenters<<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
	getLastCudaError("__updateCenters() execution failed\n");
}
#endif