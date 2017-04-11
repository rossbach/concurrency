// interop.cpp : main project file.

#include "parameters.h"
#include "kmeans.h"

using namespace System;

#pragma managed

namespace KMeansWrapper_x {

	public ref class KMeansWrapper {
	public:
		///-------------------------------------------------------------------------------------------------
		/// <summary>	compute for each center, the sum and count of all the nearest points. </summary>
        /// <remark>    (see kmeans_iter for details)                                         </remark>
		///-------------------------------------------------------------------------------------------------
		static void FirstIter(
			/* in  */ float* h_Points,
			/* in  */ float* h_Centers,
            /* out */ float* h_ClusterSums,
			/* out */ int* h_ClusterCounts,
			/* in  */ int nPoints,
			/* in  */ int nCenters) 
		{
			kmeans_iter( h_Points, h_Centers, h_ClusterSums, h_ClusterCounts, nPoints, nCenters, true );
		}

        ///-------------------------------------------------------------------------------------------------
		/// <summary>	compute for each center, the sum and count of all the nearest points. </summary>
        /// <remark>    (see kmeans_iter for details)                                         </remark>
		///-------------------------------------------------------------------------------------------------
		static void NextIter(
			/* in    */ float* h_Points,
            /* inout */ float* h_ClusterSums,
			/* inout */ int* h_ClusterCounts,
			/* in    */ int nPoints,
			/* in    */ int nCenters) 
		{   
			kmeans_iter( h_Points, h_ClusterSums, h_ClusterSums, h_ClusterCounts, nPoints, nCenters, false );
		}

        ///-------------------------------------------------------------------------------------------------
		/// <summary>	divide clusterSums by clusterCounts. </summary>
        /// <remark>    (see kmeans_iter for details)                                         </remark>
		///-------------------------------------------------------------------------------------------------
        static void LastIter(
            /* inout */ float* h_ClusterSums,
			/* inout */ int* h_ClusterCounts,
			/* in    */ int nPoints,
			/* in    */ int nCenters) 
        {
            kmeans_end( h_ClusterSums, h_ClusterCounts, nPoints, nCenters );
        }

	};

}

