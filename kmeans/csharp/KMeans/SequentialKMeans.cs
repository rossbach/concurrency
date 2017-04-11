///-------------------------------------------------------------------------------------------------
// file:	SequentialKMeans.cs
//
// summary:	Implements the sequential k means class
///-------------------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace KMeans {
    public class SequentialKMeans : KMeansCalculator {

        /// <summary>   Options for controlling the operation. </summary>
        protected KMeansOptions m_options = null;
        protected long m_nStartParallelPhase = 0;
        protected long m_nEndParallelPhase = 0;
        Stopwatch m_pStopwatch = new Stopwatch();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the parallel runtime in milliseconds. </summary>
        ///
        /// <value> The runtime milliseconds. </value>
        ///-------------------------------------------------------------------------------------------------

        public override long RuntimeMilliseconds { get { return m_nEndParallelPhase - m_nStartParallelPhase; } }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <param name="options">  Options for controlling the operation. </param>
        ///-------------------------------------------------------------------------------------------------

        public SequentialKMeans(KMeansOptions options) { m_options = options; }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates an accumulator centers. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <typeparam name="S">    Type of the s. </typeparam>
        /// <typeparam name="R">    Type of the r. </typeparam>
        /// <param name="nCenters">     The centers. </param>
        /// <param name="nRank">        The rank. </param>
        /// <param name="nPoints">      The points. </param>
        /// <param name="accum">        [out] The accum. </param>
        /// <param name="counts">       [out] The counts. </param>
        /// <param name="clusterIds">   [out] List of identifiers for the clusters. </param>
        ///-------------------------------------------------------------------------------------------------

        public void
        CreateAccumulatorCenters(
            int nCenters,
            int nRank,
            int nPoints,
            out Vector[] accum,
            out int[] counts,
            out int[] clusterIds
            ) {
            accum = CreateCenterAccumulatorList(nCenters, nRank);
            counts = CreateGroupCountList(nCenters, nRank);
            int[] tclusterIds = new int[nPoints];
            for(int i = 0; i < nPoints; i++)
                tclusterIds[i] = -1;
            clusterIds = tclusterIds;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Calculates the new centers. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <typeparam name="S">    Type of the s. </typeparam>
        /// <typeparam name="R">    Type of the r. </typeparam>
        /// <param name="options">      Options for controlling the operation. </param>
        /// <param name="vectors">      The vectors. </param>
        /// <param name="nCenters">     The centers. </param>
        /// <param name="clusterIds">   [out] List of identifiers for the clusters. </param>
        /// <param name="nIterations">  [out] The iterations. </param>
        ///
        /// <returns>   The calculated new centers&lt; s,r&gt; </returns>
        ///-------------------------------------------------------------------------------------------------

        public Vector[]
        ComputeNewCenters(
            KMeansOptions options,
            IEnumerable<Vector> vectors,
            Vector[] centers,
            out int[] clusterIds,
            out int nIterations
            ) {
            float delta = 0.0f;
            int iterations = 0;
            int nRank = vectors.ElementAt(0).Rank;
            int nPoints = vectors.Count();
            int nCenters = centers.Length;

            clusterIds = new int[nPoints];
            for(int i = 0; i < nPoints; i++)
                clusterIds[i] = -1;
            Vector[] newCenters = new Vector[nCenters];
            Vector[] oldCenters = new Vector[nCenters];
            int[] groupCounts = new int[nCenters];
            for(int i = 0; i < nCenters; i++) {
                groupCounts[i] = 0;
                newCenters[i] = Vector.ZeroVector(nRank);
                oldCenters[i] = new Vector(centers[i]);
            }

            m_pStopwatch.Start();
            m_nStartParallelPhase = m_pStopwatch.ElapsedMilliseconds;

            do {
                delta = 0.0f;
                for(int i = 0; i < nPoints; i++) {
                    Vector vec = vectors.ElementAt(i);
                    int nIndex = FindNearestCenter(vec, oldCenters);
                    if(clusterIds[i] != nIndex)
                        delta += 1.0f;
                    clusterIds[i] = nIndex;
                    newCenters[nIndex] += vec;
                    groupCounts[nIndex] += 1;
                }

                for(int i = 0; i < nCenters; i++)
                    newCenters[i] /= groupCounts[i];
                delta /= nPoints;
                iterations++;

                Vector[] tCenters = oldCenters;
                oldCenters = newCenters;
                newCenters = tCenters;
                for(int i = 0; i < nCenters; i++) {
                    groupCounts[i] = 0;
                    newCenters[i].clear();
                }

            } while((delta > options.m_fConvergenceThreshold) && (iterations < options.m_nMaxIterations));

            for(int i = 0; i < nCenters; i++)
                centers[i] = oldCenters[i];

            nIterations = iterations;
            m_nEndParallelPhase = m_pStopwatch.ElapsedMilliseconds;

            return centers;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Executes. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <param name="options">          Options for controlling the operation. </param>
        /// <param name="vectors">          The vectors. </param>
        /// <param name="nBestNClusters">   [out] The best n clusters. </param>
        /// <param name="centers">          [out] The centers. </param>
        /// <param name="clusterIds">       [out] List of identifiers for the clusters. </param>
        ///
        /// <returns>   The number of iterations/steps executed to arrive at the final result. </returns>
        ///-------------------------------------------------------------------------------------------------

        public override int
        execute(
            KMeansOptions options,
            IEnumerable<Vector> vectors,
            ref Vector[] centers,
            out int[] clusterIds,
            int nMaxParallelism,
            bool bAvoidLazyEval
            ) {
            int nCenters = options.m_nClusters;
            int nIterations = 0;
            clusterIds = null;
            centers = ComputeNewCenters(options, vectors, centers, out clusterIds, out nIterations);
            return nIterations;
        }

    }

}
