///-------------------------------------------------------------------------------------------------
// file:	kmeans.cu.h
//
// summary:	Declares the kmeans.cu class
///-------------------------------------------------------------------------------------------------

#ifndef __KMEANS_CU_H__
#define __KMEANS_CU_H__

#include "genericvector.h"
#include "kmeans-common.cu.h"

///-------------------------------------------------------------------------------------------------
/// <summary>   euclidean distance. </summary>
///
/// <param name="a">    [in,out] The pt&lt;R&gt; &amp; to process. </param>
/// <param name="b">    [in,out] The pt&lt;R&gt; &amp; to process. </param>
///
/// <returns>   distance. </returns>
///-------------------------------------------------------------------------------------------------

template<int R> float __device__
distance(
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

template <int N> int 
NearestCenter(
    pt<N> &point,
    std::vector<pt<N>> &centers
    ) 
{
    float mindist = FLT_MAX;
    int minidx = 0;
    for(size_t i=0; i<centers.size();i++) {
        float dist = distance<N>(point, centers[i]);
        if(dist < mindist) {
            minidx = static_cast<int>(i);
            mindist = dist;
        }
    }
    return minidx;
}

template<int R> int ReadInput(char * filename,
                              std::vector<pt<R>>& points,
                              int * numObjs,
                              int * numCoords,
                              int _debugflag
                              );

extern "C" void kmeans(
	int nSteps, 
	pt<DEFAULTRANK>* d_Points,  
	pt<DEFAULTRANK>* d_Centers,
	int* d_ClusterCounts,
	int* d_ClusterIds,
	int nPoints,
	int nCenters
	);

#endif
