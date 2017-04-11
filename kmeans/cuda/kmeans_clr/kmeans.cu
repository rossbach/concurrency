/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cutil_inline.h>
#include "genericvector.cuh"
#include "kmeans.h"

////////////////////////////////////////////////////////////////////////////////
// GPU-specific common definitions
////////////////////////////////////////////////////////////////////////////////
//Data type used for input data fetches
typedef uint4 data_t;

//May change on future hardware, so better parametrize the code
#define SHARED_MEMORY_BANKS 16

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions
    // This will output the proper CUDA error strings in the event that a CUDA host call returns an error
    #define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

    inline void __checkCudaErrors( cudaError err, const char *file, const int line )
    {
        if( cudaSuccess != err) {
	        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                    file, line, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

    // This will output the proper error string when calling cudaGetLastError
    #define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

    inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
    {
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err) {
            fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                    file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }
// end of CUDA Helper Functions

////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute gridDim.x partial histograms
////////////////////////////////////////////////////////////////////////////////
//Count a byte into shared-memory storage
inline __device__ void addByte(uchar *s_ThreadBase, uint data){
    s_ThreadBase[UMUL(data, THREADBLOCK_SIZE)]++;
}

//Count four bytes of a word
inline __device__ void addWord(uchar *s_ThreadBase, uint data){
    //Only higher 6 bits of each byte matter, as this is a 64-bin histogram
    addByte(s_ThreadBase, (data >>  2) & 0x3FU);
    addByte(s_ThreadBase, (data >> 10) & 0x3FU);
    addByte(s_ThreadBase, (data >> 18) & 0x3FU);
    addByte(s_ThreadBase, (data >> 26) & 0x3FU);
}

//Round a / b to nearest higher integer value
inline uint iDivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Snap a to nearest lower multiple of b
inline uint iSnapDown(uint a, uint b){
    return a - a % b;
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
/// <summary>	copy to the card and then compute for each center, the sum and count of all the nearest points. </summary>
///
/// <remarks>	JPM. </remarks>
///
/// <param name="h_Points">	   	    [in,out] the points. </param>
/// <param name="h_Centers">   	    [in] the input clusterSum if firstIter, otherwise the centers </param>
/// <param name="h_ClusterSums">    [out] the sum of all the nearest points </param>
/// <param name="h_ClusterCounts">	[out] for each center, how many points are nearest it. </param>
///-------------------------------------------------------------------------------------------------
extern "C" void 
kmeans_iter(
	float* h_Points,  
	float* h_Centers,
    float* h_ClusterSums,
	int* h_ClusterCounts,
	int nPoints,
	int nCenters,
    bool firstIter
	)
{
    void* d_PointsVoid;
    void* d_CentersVoid;
    int* d_ClusterIds;
    int* d_ClusterCounts;

    cudaMalloc( &d_PointsVoid, nPoints*RANK*sizeof(float) );
    getLastCudaError("cudaMalloc failed\n");
    cudaMalloc( &d_ClusterIds, nPoints*RANK*sizeof(float) );
    getLastCudaError("cudaMalloc failed\n");
    cudaMalloc( &d_ClusterCounts, nCenters*sizeof(int) );
    getLastCudaError("cudaMalloc failed\n");
    cudaMalloc( &d_CentersVoid, nCenters*RANK*sizeof(float) );
    getLastCudaError("cudaMalloc failed\n");

    pt<RANK>* d_Points = ( pt<RANK>* )d_PointsVoid;
    pt<RANK>* d_Centers = ( pt<RANK>* )d_CentersVoid;

    cudaMemcpy( d_Points, h_Points, nPoints*RANK*sizeof(float), cudaMemcpyHostToDevice );
    getLastCudaError("cudaMemcpy failed\n");
    cudaMemcpy( d_Centers, h_Centers, nCenters*RANK*sizeof(float), cudaMemcpyHostToDevice );
    getLastCudaError("cudaMemcpy failed\n");

    const uint nPointsBlocks = iDivUp(nPoints, THREADBLOCK_SIZE);
    const uint nCentersBlocks = iDivUp(nCenters, THREADBLOCK_SIZE);

    if (!firstIter) {
        // our input isn't a center, but a centerSum: we must divide by the counts first.
        cudaMemcpy( d_ClusterCounts, h_ClusterCounts, nCenters*sizeof(int), cudaMemcpyHostToDevice );
        getLastCudaError("cudaMemcpy failed\n");

        __updateCenters<<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
	    getLastCudaError("__updateCenters() execution failed\n");
    }

	__mapPointsToCenters<<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterIds, nPoints, nCenters);
	getLastCudaError("__mapPointsToCenters() execution failed\n");

	__resetAccumulators<<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
	getLastCudaError("__resetAccumulators() execution failed\n");

	__accumulateCenters<<<nPointsBlocks, THREADBLOCK_SIZE>>>(d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
	getLastCudaError("__updateCenters() execution failed\n");

    cudaMemcpy( h_ClusterSums, d_Centers, nCenters*RANK*sizeof(float), cudaMemcpyDeviceToHost );
    getLastCudaError("cudaMemcpy failed\n");
    cudaMemcpy( h_ClusterCounts, d_ClusterCounts, nCenters*sizeof(int), cudaMemcpyDeviceToHost );
    getLastCudaError("cudaMemcpy failed\n");
    
    cudaFree( d_PointsVoid );
    getLastCudaError("cudaFree failed\n");
    cudaFree( d_ClusterIds );
    getLastCudaError("cudaFree failed\n");
    cudaFree( d_ClusterCounts );
    getLastCudaError("cudaFree failed\n");
    cudaFree( d_CentersVoid );
    getLastCudaError("cudaFree failed\n");
}


///-------------------------------------------------------------------------------------------------
/// <summary>	divide clusterSums by clusterCounts. </summary>
///
/// <remarks>	JPM. </remarks>
///
/// <param name="h_ClusterSums">    [out] the sum of all the nearest points </param>
/// <param name="h_ClusterCounts">	[out] for each center, how many points are nearest it. </param>
///-------------------------------------------------------------------------------------------------
extern "C" void 
kmeans_end(
    float* h_ClusterSums,
	int* h_ClusterCounts,
	int nPoints,
	int nCenters
	)
{
    void* d_CentersVoid;
    int* d_ClusterCounts;

    cudaMalloc( &d_ClusterCounts, nCenters*sizeof(int) );
    getLastCudaError("cudaMalloc failed\n");
    cudaMalloc( &d_CentersVoid, nCenters*RANK*sizeof(float) );
    getLastCudaError("cudaMalloc failed\n");

    pt<RANK>* d_Centers = ( pt<RANK>* )d_CentersVoid;

    cudaMemcpy( d_Centers, h_ClusterSums, nCenters*RANK*sizeof(float), cudaMemcpyHostToDevice );
    getLastCudaError("cudaMemcpy failed\n");

    const uint nPointsBlocks = iDivUp(nPoints, THREADBLOCK_SIZE);
    const uint nCentersBlocks = iDivUp(nCenters, THREADBLOCK_SIZE);

    cudaMemcpy( d_ClusterCounts, h_ClusterCounts, nCenters*sizeof(int), cudaMemcpyHostToDevice );
    getLastCudaError("cudaMemcpy failed\n");

    __updateCenters<<<nCentersBlocks, THREADBLOCK_SIZE>>>(d_Centers, d_ClusterCounts, nCenters);
	getLastCudaError("__updateCenters() execution failed\n");

    cudaMemcpy( h_ClusterSums, d_Centers, nCenters*RANK*sizeof(float), cudaMemcpyDeviceToHost );
    getLastCudaError("cudaMemcpy failed\n");
    cudaMemcpy( h_ClusterCounts, d_ClusterCounts, nCenters*sizeof(int), cudaMemcpyDeviceToHost );
    getLastCudaError("cudaMemcpy failed\n");
    
    cudaFree( d_ClusterCounts );
    getLastCudaError("cudaFree failed\n");
    cudaFree( d_CentersVoid );
    getLastCudaError("cudaFree failed\n");
}
