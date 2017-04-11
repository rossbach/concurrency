///-------------------------------------------------------------------------------------------------
// file:	genericvector.h
//
// summary:	Declares the genericvector class
///-------------------------------------------------------------------------------------------------

#ifndef __GENERIC_VECTOR_H__
#define __GENERIC_VECTOR_H__

#include <vector>
#include "parameters.h"


////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
#define UINT_BITS 32
typedef unsigned int uint;
typedef unsigned char uchar;

////////////////////////////////////////////////////////////////////////////////
// GPU-specific common definitions
////////////////////////////////////////////////////////////////////////////////
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)
#define SHARED_MEMORY_BANKS 16
#define THREADBLOCK_SIZE (4 * SHARED_MEMORY_BANKS)
#define WARP_COUNT 6
#define UMUL(a, b) ( (a) * (b) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

template<int R> 
struct pt {
    float m_v[R];
    struct pt<R>(float * p) { for(int i=0;i<R;i++) m_v[i]=*p++; }
    struct pt<R>() { set(0.0f); }
    __device__ __host__ void set(float v) { for(int i=0;i<R;i++) m_v[i]=v; }
    __device__ __host__ void operator+=(pt<R> opt) { 
        for(int i=0;i<R;i++) 
            m_v[i]+=opt.m_v[i];
    }
    __device__ __host__ struct pt<R> operator+(pt<R> opt) { 
        pt<R> res; 
        for(int i=0;i<R;i++) 
            res.m_v[i]=m_v[i]+opt.m_v[i];
        return res;
    }
    __device__ __host__ void operator/=(int numerator) {
        for(int i=0;i<R;i++) 
            m_v[i]/=numerator;
    }
    void dump(FILE * fp) {
        for(int i=0;i<R;i++) {
            if(i>0) fprintf(fp, ", ");
            fprintf(fp, "%.3f", m_v[i]);
        }
        fprintf(fp, "\n");
    }
};

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
	pt<RANK>* d_Points,  
	pt<RANK>* d_Centers,
	int* d_ClusterCounts,
	int* d_ClusterIds,
	int nPoints,
	int nCenters
	);

#endif
