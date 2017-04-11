///-------------------------------------------------------------------------------------------------
// file:	LINQKMeans.cs
//
// summary:	Implements the linqk means class
///-------------------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;


namespace KMeans
{
    public class LINQKMeans : KMeansCalculator  {

        protected KMeansOptions m_options = null;
        public LINQKMeans(KMeansOptions options) { m_options = options; }
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
        /// <summary>   Steps. </summary>
        ///
        /// <remarks>   crossbac, 8/6/2013. </remarks>
        ///
        /// <param name="data">     The data. </param>
        /// <param name="centers">  The centers. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        public static IEnumerable<Vector>
        Step(
            IEnumerable<Vector> data,
            IEnumerable<Vector> centers
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

        public static IEnumerable<Vector>
        Steps(
            IEnumerable<Vector> data,
            IEnumerable<Vector> centers,
            int nSteps,
            int nParallelDegree,
            bool bAvoidLazyEval
            ) {
            for(int i = 0; i < nSteps; i++) {
                centers = Step(data, centers);
                if(bAvoidLazyEval)
                    centers = centers.ToArray();
            }
            return centers;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Executes. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <param name="options">          Options for controlling the operation. </param>
        /// <param name="vectors">          The vectors. </param>
        /// <param name="centers">          [out] The centers. </param>
        /// <param name="clusterIds">       [out] List of identifiers for the clusters. </param>
        /// <param name="nMaxParallelism">  (Optional) the maximum parallelism. </param>
        /// <param name="bAvoidLazyEval">   (Optional) the avoid lazy eval. </param>
        ///
        /// <returns>   The number of iterations/steps executed to arrive at the final result. </returns>
        ///-------------------------------------------------------------------------------------------------

        public override int
        execute(
            KMeansOptions options,
            IEnumerable<Vector> vectors,
            ref Vector[] centers,
            out int[] clusterIds,
            int nParallelDegree = 1,
            bool bAvoidLazyEval = true
            ) {

            IEnumerable<Vector> result = null;

            m_pStopwatch.Start();
            m_nStartParallelPhase = m_pStopwatch.ElapsedMilliseconds;
            result = LINQKMeans.Steps(vectors,
                                      centers,
                                      options.m_nMaxIterations,
                                      nParallelDegree,
                                      bAvoidLazyEval);
            result.Count();
            m_pStopwatch.Stop();
            m_nEndParallelPhase = m_pStopwatch.ElapsedMilliseconds;
            clusterIds = null;
            centers = result.ToArray();

            return options.m_nMaxIterations;
        }
    }

}
