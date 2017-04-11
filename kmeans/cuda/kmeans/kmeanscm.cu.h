///-------------------------------------------------------------------------------------------------
// file:	kmeanscm.cu.h
//
// summary:	column-major kmeans impl
///-------------------------------------------------------------------------------------------------

#ifndef __KMEANS_CM_CU_H__
#define __KMEANS_CM_CU_H__
#include "kmeans-common.cu.h"

extern "C" void 
_vkmeansfcm(
	int nSteps, 
	float * d_Points,  
	float * d_Centers,
	int * d_ClusterCounts,
	int * d_ClusterIds,
	int nPoints,
	int nCenters
	);

extern "C" void 
_vkmeansfcm_cnstmem(
	int nSteps, 
	float * d_Points,  
	float * d_Centers,
	int * d_ClusterCounts,
	int * d_ClusterIds,
	int nPoints,
	int nCenters
	);

extern "C" void 
updateConstCentersCM(
    float * h_Centers, 
    int nRank, 
    int nCenters
    );

extern "C" void 
_vkmeansfcm_cnstmem_shr(
	__in int     nSteps, 
	__in float * d_Points,  
	__in float * d_Centers,
	__in int *   d_ClusterCounts,
	__in int *   d_ClusterIds,
	__in int     nPoints,
	__in int     nCenters
	);

extern "C" void 
_vkmeansfcm_cnstmem_shr_map(
	__in int     nSteps, 
	__in float * d_Points,  
	__in float * d_Centers,
	__in int *   d_ClusterCounts,
	__in int *   d_ClusterIds,
	__in int     nPoints,
	__in int     nCenters
	);


#endif

