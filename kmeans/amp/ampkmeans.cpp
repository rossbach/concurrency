///-------------------------------------------------------------------------------------------------
// file:	ampkmeans.cpp
//
// summary:	Implements kmeans using AMP/C++ 
// 
// maintainer: crossbac@microsoft.com
///-------------------------------------------------------------------------------------------------

#include <stdio.h>
#include <tchar.h>
#include <SDKDDKVer.h>
#define NOMINMAX
#include <amp.h>
#include <iostream>
#include <numeric>
#include <assert.h>
#include <vector>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <amp_math.h>
#include "hrperftimer.h"
using namespace concurrency;
using namespace concurrency::precise_math;

///-------------------------------------------------------------------------------------------------
/// <summary>   Point structure. </summary>
///-------------------------------------------------------------------------------------------------

template<int R> 
struct pt {
    float m_v[R];
    struct pt<R>(float * p) { for(int i=0;i<R;i++) m_v[i]=*p++; }
    void set(float v) { for(int i=0;i<R;i++) m_v[i]=v; }
    void operator+=(pt<R> opt) { 
        for(int i=0;i<R;i++) 
            m_v[i]+=opt.m_v[i];
    }
    struct pt<R> operator+(pt<R> opt) { 
        pt<R> res; 
        for(int i=0;i<R;i++) 
            res.m_v[i]=m_v[i]+opt.m_v[i];
        return res;
    }
    void operator/=(int numerator) {
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

// forward decl
template<int R> int
ReadInput(
    char * filename,
    std::vector<pt<R>>& points,
    int * numObjs,
    int * numCoords,
    int _debugflag
    );

///-------------------------------------------------------------------------------------------------
/// <summary>   euclidean distance. </summary>
///
/// <param name="a">    [in,out] The pt&lt;R&gt; &amp; to process. </param>
/// <param name="b">    [in,out] The pt&lt;R&gt; &amp; to process. </param>
///
/// <returns>   distance. </returns>
///-------------------------------------------------------------------------------------------------

template<int R>
float
distance(
    pt<R> &a,
    pt<R> &b
    ) restrict(amp) restrict(cpu)
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

///-------------------------------------------------------------------------------------------------
/// <summary>   Maps points to centers on an accelerator. </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="centers">      [in,out] The centers. </param>
/// <param name="centerids">    [in,out] The centerids. </param>
///
/// <returns>   . </returns>
///-------------------------------------------------------------------------------------------------

template<int N> void
__MapPointsToCenters(
    array<pt<N>, 1> &centers,
    array<pt<N>, 1> &points,
    array<int, 1> &centerids
    )
{
    parallel_for_each(points.extent, 
                      [&](index<1> idx) restrict(amp) {
        float mindist = FLT_MAX;
        int minidx = 0;
        for(int i=0; i<centers.extent[0];i++) {
            float dist = distance<N>(points[idx], centers[i]);
            if(dist < mindist) {
                minidx = i;
                mindist = dist;
            }
        }
        centerids[idx] = minidx;
    });
}               

///-------------------------------------------------------------------------------------------------
/// <summary>   Map points to centers by managing the offload to an accelerator, doing the
///             computation on the accelerator, and then moving data back from the accelerator's
///             memory.
///             </summary>
///
/// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/8/2012. </remarks>
///
/// <param name="vcenters">     The centers. </param>
/// <param name="vpoints">      The points. </param>
/// <param name="vclusterids">  The clusterids. </param>
///-------------------------------------------------------------------------------------------------

template<int N> void
MapPointsToCentersParallel(
    std::vector<pt<N>> &vcenters,
    array<pt<N>,1> &points,
    std::vector<int>& vclusterids
    )
{
    array<pt<N>,1> centers((int)vcenters.size(), vcenters.begin());
    array<int,1> clusterids((int)points.extent[0]);
    __MapPointsToCenters<N>(centers, points, clusterids);
    copy(clusterids, vclusterids.begin());
}

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
    bool bSuccess,
    double dKmeansExecTime,
    double dKmeansCPUTime,
    bool bVerbose=false
    )
{
    fprintf(fp, "%s: GPU: %.3f msec, CPU: %.3f msec\n", 
            (bSuccess?"SUCCESS":"FAILURE"),
            dKmeansExecTime, dKmeansCPUTime);
    if(!bSuccess || bVerbose) {
        fprintf(fp, "final centers:\n");
        std::vector<pt<N>>::iterator vi;
        for(vi=centers.begin(); vi!=centers.end(); vi++) 
            vi->dump(fp);
        fprintf(fp, "\n");
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
    bool bVerbose=true
    )
{
    std::set<pt<N>*> unmatched;
    std::vector<pt<N>>::iterator vi, xi;
    for(vi=centers.begin(); vi!=centers.end(); vi++) {
        bool bFound = false;
        for(xi=refcenters.begin(); xi!=refcenters.end(); xi++) {
            if(EPSILON > distance(*vi, *xi)) {
                bFound = true;
                break;
            }
        }
        if(!bFound)
            unmatched.insert(&(*vi));
    }
    bool bSuccess = unmatched.size() == 0;
    if(bVerbose && !bSuccess) {
        std::set<pt<N>*>::iterator si;
        fprintf(stderr, "Could not match %d centers:\n", unmatched.size());
        for(si=unmatched.begin(); si!=unmatched.end(); si++) 
            (*si)->dump(stderr);        
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

static const int RANK = 16;        // known to match input file. see comment on ReadInput<>
static const int STEPS = 10;       // number of steps to perform. Can also do convergence test.
static const int NUMCENTERS = 10;  // number of clusters to find

int _tmain(int argc, _TCHAR* argv[])
{
    int numObjs = 0;
    int numCoords = 0;
    int nDefClusterId = 0;
    std::vector<pt<RANK>> points;
    std::vector<pt<RANK>> centers;
    std::vector<pt<RANK>> refcenters;
    CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);

    char * lpszInputFile = "C:\\SVC\\Dandelion\\accelerators\\kmeans\\ampkmeans\\inputs\\random-n2048-d16-c16.txt";
    ReadInput<RANK>(lpszInputFile, points, &numObjs, &numCoords, 0);
    std::vector<int> clusterids(points.size());
    std::vector<int> refclusterids(points.size());

    // be careful to make sure you're choosing the same
    // random seed every run. If you upgrade this code to actually
    // do divergence tests to decide when to terminate, it will be
    // important to use the same random seed every time, since the 
    // choice of initial centers can have a profound impact on the
    // number of iterations required to converge. Failure to be 
    // consistent will introduce a ton of noice in your data. 
    ChooseInitialCenters(points,                   // points to choose from              
                         centers,                  // destination array of initial centers
                         refcenters,               // save a copy for the reference impl to check
                         NUMCENTERS,               // number of points to choose from.
                         (int)(::GetTickCount())); // random seed. ACHTUNG! Beware benchmarkers
    

    // do the kmeans computation. map each point to its nearest
    // center, and then compute the new centers as the average
    // of all points that map to each center. 
    pTimer->reset();
    double dStart = pTimer->elapsed(false);
    array<pt<RANK>,1> apoints((int)points.size(), points.begin());
    for(int nStep=0; nStep<STEPS; nStep++) {
        MapPointsToCentersParallel(centers, apoints, clusterids);
        UpdateCentersSequential(centers, points, clusterids);
    }
    double dEnd = pTimer->elapsed(false);

    // no repeat the same computation on the CPU
    // making sure we get the same result, and (of course)
    // timing the endeavor to give ourselves a comparison point.
    double dRefStart = pTimer->elapsed(false);
    for(int nStep=0; nStep<STEPS; nStep++) {
        MapPointsToCentersSequential(refcenters, points, refclusterids);
        UpdateCentersSequential(refcenters, points, refclusterids);
    }
    double dRefEnd = pTimer->elapsed(false);

    // compare the results, complaining loudly on a mismatch,
    // and print the final output along with timing data for each impl.
    bool bSuccess = CompareResults<RANK>(centers, refcenters);
    PrintResults<RANK>(stdout, centers, bSuccess, dEnd-dStart, dRefEnd-dRefStart);

    delete pTimer;
	return 0;
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
///             program with RANK defined to match the rank of your input file! This restriction
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