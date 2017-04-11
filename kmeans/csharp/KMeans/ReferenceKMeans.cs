///-------------------------------------------------------------------------------------------------
// file:	ReferenceKMeans.cs
//
// summary:	Implements the reference k means class
///-------------------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;


namespace KMeans
{
    class ReferenceKMeans : KMeansCalculator {

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
        /// <summary>   Nearest center. </summary>
        ///
        /// <remarks>   crossbac, 8/6/2013. </remarks>
        ///
        /// <param name="vec">      The vector. </param>
        /// <param name="centers">  The centers. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        public static int NearestCenter(Vector vec, IEnumerable<Vector> centers) {
            int nIndex = -1;
            float fMinDistance = Single.MaxValue;
            int nLength= centers.Count();
            for(int i = 0; i < nLength; i++) {
                float fDistance = 0;
                Vector center = centers.ElementAt(i);
                for(int j = 0; j < vec.m_v.Length; j++) {
                    float delta = vec.m_v[j] - center.m_v[j];
                    fDistance += (delta * delta);
                }
                if(fDistance < fMinDistance) {
                    fMinDistance = fDistance;
                    nIndex = i;
                }
            }
            return nIndex;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Step. </summary>
        ///
        /// <remarks>   crossbac, 8/6/2013. </remarks>
        ///
        /// <param name="data">     The data. </param>
        /// <param name="centers">  The centers. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        public static IQueryable<Vector>
        Step(
            IQueryable<Vector> data,
            IQueryable<Vector> centers
            ) {
            return data.GroupBy(point => NearestCenter(point, centers))
                       .Select(group => group.Aggregate((x, y) => x + y) / group.Count());
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Steps. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <param name="data">             The data. </param>
        /// <param name="centers">          The centers. </param>
        /// <param name="nSteps">           The steps. </param>
        /// <param name="bAvoidLazyEval">   true to avoid lazy eval. </param>
        ///
        /// <returns>   list of centers </returns>
        ///-------------------------------------------------------------------------------------------------

        public static IQueryable<Vector> 
        Steps(
            IQueryable<Vector> data, 
            IQueryable<Vector> centers, 
            int nSteps,
            bool bAvoidLazyEval
            )
        {
            for (int i = 0; i < nSteps; i++)
            {
                centers = Step(data, centers);
                if (bAvoidLazyEval)
                    centers = centers.ToArray().AsQueryable();
            }
            return centers;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Calculates the reference. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <param name="m_input">          The input. </param>
        /// <param name="m_centers">        The centers. </param>
        /// <param name="nSteps">           The steps. </param>
        /// <param name="bAvoidLazyEval">   true to avoid lazy eval. </param>
        ///
        /// <returns>   The calculated reference. </returns>
        ///-------------------------------------------------------------------------------------------------

        public IEnumerable<Vector>
        Compute(
            IEnumerable<Vector> m_input,
            IEnumerable<Vector> m_centers,
            int nSteps,
            bool bAvoidLazyEval
            )
        {
            m_pStopwatch.Reset();
            m_pStopwatch.Start();
            m_nStartParallelPhase = m_pStopwatch.ElapsedMilliseconds;
            var referenceResult = Steps(m_input.AsQueryable(),
                                        m_centers.AsQueryable(),
                                        nSteps, 
                                        bAvoidLazyEval);
            referenceResult.Count();
            m_pStopwatch.Stop();
            m_nEndParallelPhase = m_pStopwatch.ElapsedMilliseconds;
            return referenceResult;
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
            int nMaxParallelism = 1,
            bool bAvoidLazyEval = true
            ) {
            int nIterations = 0;
            clusterIds = null;
            centers = Compute(vectors,
                              centers,
                              options.m_nMaxIterations,
                              bAvoidLazyEval).ToArray();
            nIterations = options.m_nMaxIterations;
            return nIterations;
        }
    }
}
