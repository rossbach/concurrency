/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
 /*
 * This sample implements 64-bin histogram calculation
 * of arbitrary-sized 8-bit data array
 */

// Utility and system includes
#include <shrUtils.h>
#include <shrQATest.h>

#include <sdkHelper.h>  // helper for shared that are common to CUDA SDK samples
#include <cuda_runtime.h>
#include <vector>
#include <set>
#include <map>
#include "getopt.h"
#include "hrperftimer.h"
#include "kmeans.cu.h"
#include "kmeanscm.cu.h"

typedef double (*LPFNKMEANS)(const int nSteps,
                             void * h_Points,
                             void * h_Centers,
	                         const int nPoints,
	                         const int nCenters,
                             StopWatchInterface * hTimer,
	                         bool bVerify,
	                         bool bVerbose);

typedef void (*LPFNBNC)(char * szFile, 
                        LPFNKMEANS lpfn, 
                        int nSteps,
                        int nSeed,
                        StopWatchInterface * hTimer,
                        bool bVerify,
                        bool bVerbose);

#include "testsuitedecl.h"

//declare_suite_hdrs(4);
declare_suite_hdrs(16);
//declare_suite_hdrs(24);
//declare_suite_hdrs(32);
//declare_suite_hdrs(64);
declare_suite_hdrs(128);

std::map<std::string, std::map<int, std::map<int, LPFNKMEANS>>> g_lpfns;
std::map<std::string, std::map<int, std::map<int, LPFNBNC>>> g_bncfns;
bool g_blpfnInit = false;
decl_init_lpfn_table_begin(g_lpfns, g_bncfns, g_blpfnInit);

    //create_suite_entries(g_lpfns, g_bncfns, 4);
    create_suite_entries(g_lpfns, g_bncfns, 16);
    //create_suite_entries(g_lpfns, g_bncfns, 24);
    //create_suite_entries(g_lpfns, g_bncfns, 32);
    //create_suite_entries(g_lpfns, g_bncfns, 64);
    create_suite_entries(g_lpfns, g_bncfns, 128);

decl_init_lpfn_table_end(g_lpfns, g_bncfns, g_blpfnInit);
declare_lpfn_finder(g_lpfns, g_blpfnInit)             
declare_bnc_finder(g_bncfns, g_blpfnInit)

int gVerbose = 0;

typedef double (*LPFNKMEANS)(const int nSteps,
                             void * h_Points,
                             void * h_Centers,
	                         const int nPoints,
	                         const int nCenters,
                             StopWatchInterface * hTimer,
	                         bool bVerify,
	                         bool bVerbose);

bool g_bVerbose = false;
bool g_bVerify = false;
int g_nSteps = DEFAULTSTEPS;
LPFNKMEANS g_lpfnKMeans = NULL;
LPFNBNC g_lpfnBnc = NULL;
char * g_lpszDefaultInput = "..\\..\\Csharp\\kmeans\\inputs\\random-n16-d16-c16.txt";
//char * g_lpszDefaultInput = "..\\..\\ampkmeans\\inputs\\random-n65536-d32-c16.txt";
char g_vInputFile[4096];
char g_vKMeansVersion[4096];
int g_nRank = DEFAULTRANK;
int g_nCenters = DEFAULTCENTERS;
int g_nSeed = static_cast<int>(::GetTickCount());

#if 1

LPFNKMEANS
choose_kmeans_impl(
    char * lpszImpl,
    int nRank,
    int nCenters
    )
{
    // "raw" 
    // "constmem"
    // "constmemset"
    // "constmemsetshr"
    // "constmemsetshrmap"
    // "cm"
    // "cmconstmem"
    // "cmconstmemshr"
    // "cmconstmemshrmap"
    std::string strName(lpszImpl);
    return find_lpfn(strName, nRank, nCenters);
}

LPFNBNC
choose_kmeans_bnc(
    char * lpszImpl,
    int nRank,
    int nCenters
    )
{
    // "raw" 
    // "constmem"
    // "constmemset"
    // "constmemsetshr"
    // "constmemsetshrmap"
    // "cm"
    // "cmconstmem"
    // "cmconstmemshr"
    // "cmconstmemshrmap"
    std::string strName(lpszImpl);
    return find_bncfn(strName, nRank, nCenters);
}
#else
///-------------------------------------------------------------------------------------------------
/// <summary>   print usage. </summary>
///
/// <remarks>   crossbac, 7/29/2013. </remarks>
///
/// <param name="argv"> [in,out] If non-null, the argv. </param>
///-------------------------------------------------------------------------------------------------

LPFNKMEANS
choose_kmeans_impl(
    char * lpszImpl
    )
{
    if(lpszImpl == NULL) return NULL;
    if(!strcmp(lpszImpl, "simple")) return kmeans_simple;
    if(!strcmp(lpszImpl, "raw")) return kmeans_raw;
    if(!strcmp(lpszImpl, "constmem")) return kmeans_constmem;
    if(!strcmp(lpszImpl, "constmemset")) return kmeans_constmem_memsetreset;
    if(!strcmp(lpszImpl, "constmemsetshr")) return kmeans_constmem_memsetreset_shr;
    if(!strcmp(lpszImpl, "constmemsetshrmap")) return kmeans_constmem_memsetreset_shr_map;
    if(!strcmp(lpszImpl, "cm")) return kmeans_cm_r24_c64;
    if(!strcmp(lpszImpl, "cmconstmem")) return kmeans_cm_constmem;
    if(!strcmp(lpszImpl, "cmconstmemshr")) return kmeans_cm_constmem_shr;
    if(!strcmp(lpszImpl, "cmconstmemshrmap")) return kmeans_cm_constmem_shr_map;
    return NULL;
}

#endif

///-------------------------------------------------------------------------------------------------
/// <summary>   print usage. </summary>
///
/// <remarks>   crossbac, 7/29/2013. </remarks>
///
/// <param name="argv"> [in,out] If non-null, the argv. </param>
///-------------------------------------------------------------------------------------------------

void 
usage(
    char ** argv
    )
{
    printf("\nUsage for %s:\n", argv[0]);
    printf("%s [options]\n\n", argv[0]);
    printf("     -v          : verbose output\n");
    printf("     -V          : verify GPU-side output by comparing with CPU (time-consuming!)\n");
    printf("     -s (steps)  : number of iterations of kmeans to run\n");
    printf("     -t (task)   : task to run [simple|raw|...] different versions of kmeans...\n");
    printf("     -f (file)   : input file (determines K and number of dimensions) \n");
    printf("     -r (seed)   : seed value for rand() -- used to choose initial centers\n");
    printf("     -R (rank)   : expected rank of vector objects\n");
    printf("     -C (centers): expected number of centers to find\n");
    printf("     -h          : prints this message\n");
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Gets the options. </summary>
///
/// <remarks>   crossbac, 7/29/2013. </remarks>
///
/// <param name="argc"> The argc. </param>
/// <param name="argv"> [in,out] If non-null, the argv. </param>
///
/// <returns>   true if it succeeds, false if it fails. </returns>
///-------------------------------------------------------------------------------------------------

BOOL 
get_options(
    int argc, 
    char * argv[]
    )  
{
    int c = 0;
    opterr = 0;
    strcpy_s(g_vInputFile, 4096, g_lpszDefaultInput);
    while((c = getopt(argc, argv, "vVs:t:f:hr:R:C:")) != -1) {
        switch(c) {	
        case 'h': return 0;
        case 'v': g_bVerbose = true; break;
        case 'V': g_bVerify = true; break;
        case 'R':
            if(optarg == NULL) {
                printf("\n-R requires an integer-valued rank!\n\n"); 
                return FALSE;
            }
            g_nRank = atoi(optarg);
            break;	
        case 'C':
            if(optarg == NULL) {
                printf("\n-C requires an integer-valued centers argument!\n\n"); 
                return FALSE;
            }
            g_nCenters = atoi(optarg);
            break;	
        case 'r':
            if(optarg == NULL) {
                printf("\n-r requires an integer-valued rand seed!\n\n"); 
                return FALSE;
            }
            g_nSeed = atoi(optarg);
            break;	
        case 's':
            if(optarg == NULL) {
                printf("\n-s requires an integer-valued number of steps!\n\n"); 
                return FALSE;
            }
            g_nSteps = atoi(optarg);
            break;			
        case 't':
            if(optarg == NULL) {
                printf("\n-t requires a valid version of kemans to run!\n\n"); 
                return FALSE;
            }			
            strcpy_s(g_vKMeansVersion, 4096, optarg);
            break;
        case 'f':
            if(optarg == NULL) {
                printf("\n-s requires a path to an input file!\n\n"); 
                return FALSE;
            }			
            strcpy_s(g_vInputFile, 4096, optarg);
            break;
        default:			
            printf("Unknown command line switch: %c", c);			
            return FALSE;
        }
    }
    return TRUE;
}

// General GPU Device CUDA Initialization
int gpuDeviceInit(int devID)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
        exit(-1);
    }
    if (devID < 0) 
        devID = 0;
    if (devID > deviceCount-1) {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
        fprintf(stderr, "\n");
        return -devID;
    }

    cudaDeviceProp deviceProp;
    checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
    if (deviceProp.major < 1) {
        fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
        exit(-1);                                                  \
    }

    checkCudaErrors( cudaSetDevice(devID) );
    printf("> gpuDeviceInit() CUDA device [%d]: %s\n", devID, deviceProp.name);
    return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
int gpuGetMaxGflopsDeviceId()
{
	int current_device   = 0, sm_per_multiproc = 0;
	int max_compute_perf = 0, max_perf_device  = 0;
	int device_count     = 0, best_SM_arch     = 0;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount( &device_count );
	// Find the best major SM Architecture GPU device
	while ( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major > 0 && deviceProp.major < 9999) {
			best_SM_arch = MAX(best_SM_arch, deviceProp.major);
		}
		current_device++;
	}

    // Find the best CUDA capable GPU device
    current_device = 0;
    while( current_device < device_count ) {
        cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
            sm_per_multiproc = 1;
		} else {
            sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
        }

        int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
        if( compute_perf  > max_compute_perf ) {
            // If we find GPU with SM major > 2, search only these
            if ( best_SM_arch > 2 ) {
                // If our device==dest_SM_arch, choose this, or else pass
                if (deviceProp.major == best_SM_arch) {	
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                }
            } else {
                max_compute_perf  = compute_perf;
                max_perf_device   = current_device;
            }
        }
        ++current_device;
	}
	return max_perf_device;
}

// Initialization code to find the best CUDA Device
int findCudaDevice(int argc, const char **argv)
{
    cudaDeviceProp deviceProp;
    int devID = 0;
    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, argv, "device")) {
        devID = getCmdLineArgumentInt(argc, argv, "device=");
        if (devID < 0) {
            printf("Invalid command line parameters\n");
            exit(-1);
        } else {
            devID = gpuDeviceInit(devID);
            if (devID < 0) {
                printf("exiting...\n");
                shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
                exit(-1);
            }
        }
    } else {
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
        checkCudaErrors( cudaSetDevice( devID ) );
        checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
        printf("> Using CUDA device [%d]: %s\n", devID, deviceProp.name);
    }
    return devID;
}

// const int numRuns = 16;

static char *sSDKsample = "[kmeans]\0";

///-------------------------------------------------------------------------------------------------
/// <summary>   Map points to centers sequentially, using the CPU.
///             </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="vcenters">     The vcenters. </param>
/// <param name="vpoints">      The vpoints. </param>
/// <param name="vclusterids">  The vclusterids. </param>
///-------------------------------------------------------------------------------------------------

template<int N> void
MapPointsToCentersSequential(
    std::vector<pt<N>> &vcenters,
    std::vector<pt<N>> &vpoints,
    std::vector<int>& vclusterids
    )
{
    std::vector<pt<N>>::iterator vi;
    int index = 0;
    for(vi=vpoints.begin(); vi!=vpoints.end(); vi++) 
        vclusterids[index++] = NearestCenter<N>(*vi, vcenters);
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Updates the centers based on the results of MapPointsToCenters. 
/// 			Currently just does this sequentially on the CPU.
/// 			</summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="vcenters">     [in,out] The vcenters. </param>
/// <param name="vpoints">      [in,out] The vpoints. </param>
/// <param name="vclusterids">  [in,out] The vclusterids. </param>
///
/// <returns>   . </returns>
///-------------------------------------------------------------------------------------------------

template<int N> void
UpdateCentersSequential(
    std::vector<pt<N>> &vcenters,
    std::vector<pt<N>> &vpoints,
    std::vector<int>& vclusterids
    )
{
    std::vector<int> counts;
    for(size_t i=0; i<vcenters.size(); i++) {
        vcenters[i].set(0.0f);
        counts.push_back(0);
    }
    for(size_t i=0; i<vpoints.size(); i++) {
        int clusterid = vclusterids[i];
        vcenters[clusterid] += vpoints[i];
        counts[clusterid] += 1;
    }
    for(size_t i=0; i<vcenters.size(); i++) {
        vcenters[i] /= counts[i];
    }
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Choose initial centers randomly. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <typeparam name="int R">    Type of the int r. </typeparam>
/// <param name="points">       The points. </param>
/// <param name="centers">      [out] The initial centers. </param>
/// <param name="refcenters">   [out] The a copy of the initial centers for the reference
///                             implementation to check--the GPU implementation is destructive,
///                             so we need to save a copy. </param>
/// <param name="nCenters">     The number of centers to choose. </param>
/// <param name="nRandomSeed">  The random seed. </param>
///-------------------------------------------------------------------------------------------------

template <int R> void
ChooseInitialCenters(
    std::vector<pt<R>> &points,
    std::vector<pt<R>> &centers,
    std::vector<pt<R>> &refcenters,
    int nCenters,
    int nRandomSeed
    )
{
    srand(nRandomSeed);
    std::set<int> chosenidx;
    while(chosenidx.size() < (size_t)nCenters) {
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

///-------------------------------------------------------------------------------------------------
/// <summary>	Print centers. </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="fp">	  	[in,out] If non-null, the fp. </param>
/// <param name="centers">	[in,out] The centers. </param>
///-------------------------------------------------------------------------------------------------

template<int N> void
PrintCenters(
    FILE * fp,
    std::vector<pt<N>> &centers,
    int nRowLimit=16,
    int nColLimit=8
    )
{
    int nRows=0;
    std::vector<pt<N>>::iterator vi;
    for(vi=centers.begin(); vi!=centers.end(); vi++) {
        if(++nRows > nRowLimit) {
            fprintf(fp, "...");
            break;
        }
        vi->dump(fp,nColLimit);
    }
    fprintf(fp, "\n");
}

///-------------------------------------------------------------------------------------------------
/// <summary>	Print centers. </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="fp">	  	[in,out] If non-null, the fp. </param>
/// <param name="centers">	[in,out] The centers. </param>
///-------------------------------------------------------------------------------------------------

template<int N> void
PrintCenters(
    FILE * fp,
    pt<N> * pCenters,
	int nCenters
    )
{
	for(int i=0; i<nCenters; i++)
        pCenters[i].dump(fp);
    fprintf(fp, "\n");
}

///-------------------------------------------------------------------------------------------------
/// <summary>	Print centers. </summary>
///
/// <remarks>	Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="fp">	  	[in,out] If non-null, the fp. </param>
/// <param name="centers">	[in,out] The centers. </param>
///-------------------------------------------------------------------------------------------------

void
PrintCentersTransposed(
    FILE * fp,
    int nDEFAULTRANK,
    float * pCenters,
	int nCenters
    )
{
    for(int c=0; c<nCenters; c++) {
	    for(int i=0; i<nDEFAULTRANK; i++) {
            int fidx = i*nCenters+c;
            if(i>0) fprintf(fp, ", ");
            fprintf(fp, "%.3f", pCenters[fidx]);
        }
        fprintf(fp, "\n");
    }
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Print results. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="fp">               [in,out] If non-null, the fp. </param>
/// <param name="centers">          The centers. </param>
/// <param name="bSuccess">         true if the operation was a success, false if it failed. </param>
/// <param name="dKmeansExecTime">  Time of the kmeans execute. </param>
/// <param name="dKmeansCPUTime">   Time of the kmeans CPU. </param>
/// <param name="bVerbose">         (optional) the verbose. </param>
///-------------------------------------------------------------------------------------------------

template<int N> void
PrintResults(
    FILE * fp,
    std::vector<pt<N>> &centers,
    std::vector<pt<N>> &refcenters,
    bool bSuccess,
    double dKmeansExecTime,
    double dKmeansCPUTime,
    bool bVerbose=false,
    int nRowLimit=16,
    int nColLimit=8
    )
{
    fprintf(fp, "%s: GPU: %.5f sec, CPU: %.5f sec\n", 
            (bSuccess?"SUCCESS":"FAILURE"),
            dKmeansExecTime, dKmeansCPUTime);
    if(!bSuccess || bVerbose) {
        fprintf(fp, "final centers:\n");
		PrintCenters<N>(fp, centers, nRowLimit, nColLimit);
        fprintf(fp, "reference centers:\n");
		PrintCenters<N>(fp, refcenters, nRowLimit, nColLimit);
    }
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Compare results, returning true if the reference and candidate centers match
///             within a given epsilon parameter. Note that we cannot assume anything about the
///             order in which the centers will be generated on the GPU (in general), and there
///             are abundant sourcs of floating point error, so we just check that for each
///             member of the candidate centers, there exists a member of the reference that is
///             within the epsilon in terms of euclidean distance.
///             </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="centers">      The centers. </param>
/// <param name="refcenters">   The refcenters. </param>
/// <param name="EPSILON">      The epsilon. </param>
/// <param name="bVerbose">     true to verbose. </param>
///
/// <returns>   true if it succeeds, false if it fails. </returns>
///-------------------------------------------------------------------------------------------------

template<int N> bool
CompareResults(
    std::vector<pt<N>> &centers,    
    std::vector<pt<N>> &refcenters,
    float EPSILON=0.0001f,
    bool bVerbose=true,
    int nRowLimit=16
    )
{
    int nRows=0;
    std::map<int, pt<N>*> unmatched;
    std::map<int, float> unmatched_deltas;
    std::map<int, int> matched;
    std::map<int, int> revmatched;
    std::vector<pt<N>>::iterator vi, xi;
    int nCenterIdx=0;    
    for(vi=centers.begin(); vi!=centers.end(); vi++, nCenterIdx++) {
        bool bFound = false;        
        if(EPSILON*N > distance(*vi, refcenters[nCenterIdx])) {
            bFound = true;
            matched[nCenterIdx] = nCenterIdx;
            revmatched[nCenterIdx] = nCenterIdx;
        } else {
            int nRefIdx=0;
            for(xi=refcenters.begin(); xi!=refcenters.end(); xi++, nRefIdx++) {
                if(EPSILON*N > distance(*vi, *xi)) {
                    bFound = true;
                    matched[nCenterIdx] = nRefIdx;
                    revmatched[nRefIdx] = nCenterIdx;
                    break;
                }
            }
        }
        if(!bFound) {
            unmatched[nCenterIdx] = (&(*vi));
            unmatched_deltas[nCenterIdx] = distance(*vi, refcenters[nCenterIdx]);
        }
    }
    bool bSuccess = unmatched.size() == 0;
    if(bVerbose && !bSuccess) {
        std::map<int, pt<N>*>::iterator si;
        fprintf(stderr, "Could not match %d centers:\n", unmatched.size());
        for(si=unmatched.begin(); si!=unmatched.end(); si++) {
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

///-------------------------------------------------------------------------------------------------
/// <summary>   Main entry-point for this application. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="argc"> The argc. </param>
/// <param name="argv"> [in,out] If non-null, the argv. </param>
///
/// <returns>   . </returns>
///-------------------------------------------------------------------------------------------------

StopWatchInterface *
ConfigureDevice(
	int argc, 
	char **argv
	)
{
    StopWatchInterface * hTimer;
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;

    // set logfile name and start logs
    shrQAStart(argc, argv);
    shrSetLogFileName ("kmeans.txt");

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    // refer to helper_cuda.h for implementation
    int dev = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors( cudaGetDeviceProperties(&deviceProp, dev) );

    printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n", 
          deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = deviceProp.major * 0x10 + deviceProp.minor;

    if(version < 0x11) {
        printf("There is no device supporting a minimum of CUDA compute capability 1.1 for this SDK sample\n");
        cudaDeviceReset();
        shrQAFinishExit(argc, (const char **)argv, QA_WAIVED);
    }

    sdkCreateTimer(&hTimer);
	return hTimer;
}

///-------------------------------------------------------------------------------------------------
/// <summary>   shut down the device, GPU SDK, report result</summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="argc"> The argc. </param>
/// <param name="argv"> [in,out] If non-null, the argv. </param>
///
/// <returns>   . </returns>
///-------------------------------------------------------------------------------------------------

void 
TeardownDevice(
	int argc,
	char ** argv,
	int PassFailFlag,
	StopWatchInterface * hTimer
	)
{
    shrLog("Shutting down...\n");
    sdkDeleteTimer(&hTimer);
    cudaDeviceReset();
	shrLog("%s - Test Summary\n", sSDKsample);
    fprintf(stderr, "%s %s\n", argv[0], (PassFailFlag?"PASSED":"FAILED"));
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
kmeans_simple(
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
	pt<DEFAULTRANK> * h_Points = reinterpret_cast<pt<DEFAULTRANK>*>(lpvPoints);
	pt<DEFAULTRANK> * h_Centers = reinterpret_cast<pt<DEFAULTRANK>*>(lpvCenters);
    pt<DEFAULTRANK> *d_Points = NULL;
	pt<DEFAULTRANK> *d_Centers = NULL;
	int * d_ClusterIds = NULL;
	int * d_ClusterCounts = NULL;
	size_t uiPointsBytes = nPoints * DEFAULTRANK * sizeof(float);
	size_t uiCentersBytes = nCenters * DEFAULTRANK * sizeof(float);
	size_t uiClusterIdsBytes = nPoints * sizeof(int);
	size_t uiClusterCountsBytes = nCenters * sizeof(int);

    INFORM(bVerbose, "Initializing data...\n");
    checkCudaErrors( cudaMalloc((void **)&d_Points, uiPointsBytes ) );
    checkCudaErrors( cudaMalloc((void **)&d_Centers, uiCentersBytes ) );
    checkCudaErrors( cudaMalloc((void **)&d_ClusterIds, uiClusterIdsBytes ) );
    checkCudaErrors( cudaMalloc((void **)&d_ClusterCounts, uiClusterCountsBytes ) );
	checkCudaErrors( cudaMemcpy(d_Points, h_Points, uiPointsBytes, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_Centers, h_Centers, uiCentersBytes, cudaMemcpyHostToDevice) );
	INFORM(bVerbose, "Starting up kmeans-simple...\n\n");

    // fprintf(stdout, "initial centers:\n");
	// PrintCenters<DEFAULTRANK>(stdout, h_Centers, nCenters);

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	kmeans(nSteps, d_Points, d_Centers, d_ClusterCounts, d_ClusterIds, nPoints, nCenters);
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
#else
    return 0.0;
#endif 
}

///-------------------------------------------------------------------------------------------------
/// <summary>   return a transposed (column-major) version of the vector array. </summary>
///
/// <remarks>   Caller must free the result if no output buffer is provided.
///             Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="argc"> The argc. </param>
/// <param name="argv"> [in,out] If non-null, the argv. </param>
///
/// <returns>   (potentially newly allocated) transposed version of the points buffer.</returns>
///-------------------------------------------------------------------------------------------------

float *
transpose(
    float * h_Points,
    int nDEFAULTRANK,
    int nPoints,
    float * pTranspose=NULL
    )
{	
    size_t  uiPointsBytes = nPoints * nDEFAULTRANK * sizeof(float);
    float * pInput = reinterpret_cast<float*>(h_Points);
    if(pTranspose == NULL) {
        pTranspose = (float*)malloc(uiPointsBytes);
    }
    for(int i=0; i<nPoints; i++) {
        for(int j=0; j<nDEFAULTRANK; j++) { 
            int nInputIdx = (i*nDEFAULTRANK)+j;
            int nTxIdx = j*nPoints+i;
            pTranspose[nTxIdx] = pInput[nInputIdx];
        }
    }
    return pTranspose;
}

///-------------------------------------------------------------------------------------------------
/// <summary>   return a transposed (column-major) version of the vector array. </summary>
///
/// <remarks>   Caller must free the result if no output buffer is provided.
///             Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="argc"> The argc. </param>
/// <param name="argv"> [in,out] If non-null, the argv. </param>
///
/// <returns>   (potentially newly allocated) transposed version of the points buffer.</returns>
///-------------------------------------------------------------------------------------------------

float *
transpose(
    pt<DEFAULTRANK>* h_Points,
    int nPoints,
    float * pTranspose=NULL
    )
{	
    float * pInput = reinterpret_cast<float*>(h_Points);
    return transpose(pInput, DEFAULTRANK, nPoints, pTranspose);
}

///-------------------------------------------------------------------------------------------------
/// <summary>   Main entry-point for this application. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="argc"> The argc. </param>
/// <param name="argv"> [in,out] If non-null, the argv. </param>
///
/// <returns>   . </returns>
///-------------------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
    if(!get_options(argc, argv)) {
        usage(argv);
        exit(1);
    }

    int PassFailFlag = 1;
	StopWatchInterface * hTimer = ConfigureDevice(argc, argv);
    g_lpfnKMeans = choose_kmeans_impl(g_vKMeansVersion, g_nRank, g_nCenters);
    g_lpfnBnc = choose_kmeans_bnc(g_vKMeansVersion, g_nRank, g_nCenters);
    if(!g_lpfnKMeans || !g_lpfnBnc) {
		fprintf(stderr, 
                "failed to select valid implementation for %s(RANK=%d, CENTERS=%d)!\n", 
                g_vKMeansVersion,
                g_nRank,
                g_nCenters);
        exit(-1);
    }

    (*g_lpfnBnc)(g_vInputFile,
                 g_lpfnKMeans,
                 g_nSteps,
                 g_nSeed,
                 hTimer,
                 g_bVerify,
                 g_bVerbose);

	TeardownDevice(argc, argv, PassFailFlag, hTimer);
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

template<int R> int
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

