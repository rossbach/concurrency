///-------------------------------------------------------------------------------------------------
// file:	KMeansCalculator.cs
//
// summary:	Implements the means calculator class
///-------------------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.IO;

namespace KMeans
{
    public abstract class KMeansCalculator
    {
        public abstract long RuntimeMilliseconds { get; }

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

        public abstract int
        execute(
            KMeansOptions options,
            IEnumerable<Vector> vectors,
            ref Vector[] centers,
            out int[] clusterIds,
            int nMaxParallelism=1,
            bool bAvoidLazyEval=true
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Choose random centers. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <param name="vectors">      The vectors. </param>
        /// <param name="nCenters">     The centers. </param>
        /// <param name="nRandomSeed">  The random seed. </param>
        /// <param name="bVerbose">     (optional) the verbose. </param>
        ///
        /// <returns>   A list of. </returns>
        ///-------------------------------------------------------------------------------------------------

        public static Vector[]
        SelectRandomCenters(
            IEnumerable<Vector> vectors,
            int nCenters,
            int nRandomSeed,
            bool bVerbose=false
            )
        {
            // randomly choose cluster centers:
            if (nRandomSeed == -1)
                nRandomSeed = (int)(DateTime.Now.Millisecond);
            if(bVerbose) Console.WriteLine("seeding initial selection with {0}", nRandomSeed);
            Random rand = new Random(nRandomSeed);
            HashSet<int> chosenIdx = new HashSet<int>();
            Vector[] centers = new Vector[nCenters];
            int nChosen = 0;
            while (chosenIdx.Count() < nCenters)
            {
                int nIndex = rand.Next(vectors.Count() - 1);
                if (chosenIdx.Contains(nIndex))
                    continue;
                Vector center = vectors.ElementAt(nIndex);
                centers[nChosen++] = new Vector(center);
                chosenIdx.Add(nIndex);
            }
            return centers;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Searches for the nearest center. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <param name="vec">      The vector. </param>
        /// <param name="centers">  [out] The centers. </param>
        ///
        /// <returns>   The found center. </returns>
        ///-------------------------------------------------------------------------------------------------

        public static int
        FindNearestCenter(
            Vector vec,
            Vector[] centers
            )
        {
            int nIndex = -1;
            float fMinDistance = Single.MaxValue;
            for (int i = 0; i < centers.Length; i++)
            {
                float fDistance = 0;
                for (int j = 0; j < vec.m_v.Length; j++)
                {
                    float delta = vec.m_v[j] - centers[i].m_v[j];
                    fDistance += (delta * delta);
                }
                if (fDistance < fMinDistance) {
                    fMinDistance = fDistance;
                    nIndex = i;
                }
            }
            return nIndex;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enumerates create center accumulator list in this collection. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <typeparam name="S">    Type of the s. </typeparam>
        /// <param name="nCenters"> The centers. </param>
        /// <param name="nRank">    The rank. </param>
        ///
        /// <returns>   The new center accumulator list&lt; s&gt; </returns>
        ///-------------------------------------------------------------------------------------------------

        public static Vector[]
        CreateCenterAccumulatorList(
            int nCenters,
            int nRank
            )
        {
            Vector[] accum = new Vector[nCenters];
            for (int i = 0; i < nCenters; i++) {
                accum[i] = Vector.ZeroVector(nRank);
            }
            return accum;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a group count list. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <typeparam name="S">    Type of the s. </typeparam>
        /// <param name="nCenters"> The centers. </param>
        /// <param name="nRank">    The rank. </param>
        ///
        /// <returns>   The new group count list&lt; s&gt; </returns>
        ///-------------------------------------------------------------------------------------------------

        public static int[]
        CreateGroupCountList(
            int nCenters,
            int nRank
            )
        {
            int[] counts = new int[nCenters];
            for (int i = 0; i < nCenters; i++)
                counts[i] = 0; 
            return counts;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a group count list. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <typeparam name="S">    Type of the s. </typeparam>
        /// <param name="nCenters"> The centers. </param>
        /// <param name="nRank">    The rank. </param>
        ///
        /// <returns>   The new group count list&lt; s&gt; </returns>
        ///-------------------------------------------------------------------------------------------------

        public static int[]
        CreateClusterAssignments(
            int nVectors
            )
        {
            int[] clusterIds = new int[nVectors];
            for (int i = 0; i < nVectors; i++)
                clusterIds[i] = -1;
            return clusterIds;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Is a given value present in the list of vectors? 
        /// 			We need this to check correctness because we have to tolerate two sources
        /// 			of divergence between a list generated by our candidate and reference implementations:
        /// 			1) Parallel calculation -> centers are not guaranteed to appear in order  
        /// 			2) Parallel calculation -> floating point error accumulated from lack  
        /// 			                           of associativity.
        /// 			
        ///             This method returns true if there is a vector in the reference that is within
        ///             the tolerance parameter in terms or euclidean distance from the given vector. 
        ///             This metric is not perfect, since it's only a probabilistic guarantee that we're
        ///             getting the right answer, and that probability will depend heavily on the input!
        ///             </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <param name="options">      Options for controlling the operation. </param>
        /// <param name="center">       The center. </param>
        /// <param name="refcenters">   The refcenters. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        static bool
        ValuePresent(
            KMeansOptions options,
            Vector center,
            IEnumerable<Vector> refcenters
            )
        {
            foreach (Vector rc in refcenters)
            {
                if (Vector.Dist(center, rc) < options.m_fEpsilon)
                    return true;
            }
            return false;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Check result. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <param name="options">          Options for controlling the operation. </param>
        /// <param name="centers">          [out] The centers. </param>
        /// <param name="refnewcenters">    The refnewcenters. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        public static bool
        CheckResult(
            KMeansOptions options,
            IEnumerable<Vector> centers,
            IEnumerable<Vector> refnewcenters,
            bool bVerbose = true,
            bool bDumpLists = false
            )
        {
            List<int> diffIndexes = new List<int>();
            if (centers.Count() != refnewcenters.Count())
            {
                if(bVerbose)
                    Console.WriteLine("different lengths!");
                return false;
            }

            bool bResult = true;
            int nIndex = 0;
            foreach (Vector center in centers)
            {
                if (!ValuePresent(options, center, refnewcenters))
                {
                    bResult = false;
                    diffIndexes.Add(nIndex);
                }
                nIndex++;
            }
            if (!bResult && bDumpLists)
            {
                if (bVerbose)
                    Console.WriteLine("center lists differ!");
                Console.WriteLine("REF:");
                foreach (Vector center in refnewcenters)
                    Console.WriteLine("\t{0}", center);
                Console.WriteLine("CAND: could not match:");
                foreach(int index in diffIndexes) 
                    Console.WriteLine("\t{0}", centers.ElementAt(index));
            }
            return bResult;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Reads a text input. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <param name="strInputFile"> The input file. </param>
        ///
        /// <returns>   The text input. </returns>
        ///-------------------------------------------------------------------------------------------------

        public static Vector[]
        ReadTextInput(
            string strInputFile
            )
        {
            string line;
            List<Vector> attributes = new List<Vector>();
            System.IO.StreamReader file = new System.IO.StreamReader(strInputFile);
            while ((line = file.ReadLine()) != null)
            {
                Vector vItem = new Vector(line);
                attributes.Add(vItem);
            }
            file.Close(); 
            return attributes.ToArray();
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Reads a binary input. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <param name="strInputFile"> The input file. </param>
        ///
        /// <returns>   The binary input. </returns>
        ///-------------------------------------------------------------------------------------------------

        public static Vector[] 
        ReadBinaryInput(
            string strInputFile
            )
        {
            List<Vector> result = new List<Vector>();
            using (BinaryReader r = new BinaryReader(File.Open(strInputFile, FileMode.Open)))
            {
                int nRank = r.ReadInt32();
                int nVectors = r.ReadInt32();
                for (int i = 0; i < nVectors; i++)
                {
                    Vector vec = new Vector(nRank);
                    for (int j = 0; j < nRank; j++)
                        vec[j] = r.ReadSingle();
                    result.Add(vec);
                }
            }
            return result.ToArray();
        }

    }
}
