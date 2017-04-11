#ifndef __KMEANS_H__
#define __KMEANS_H__

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
	);


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
	);

#endif