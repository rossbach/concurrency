///-------------------------------------------------------------------------------------------------
// file:	Program.cs
//
// summary:	Implements the program class
///-------------------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Collections;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using System.Diagnostics;
using System.IO;

namespace KMeans
{
    class Program
    {
        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Select initial centers. </summary>
        ///
        /// <remarks>   crossbac, 8/6/2013. </remarks>
        ///
        /// <param name="attributes">   The attributes. </param>
        /// <param name="nMinClusters"> The minimum clusters. </param>
        /// <param name="nRandomSeed">  The random seed. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        static Vector[]
        SelectInitialCenters(
            IEnumerable<Vector> attributes,
            int nMinClusters,
            int nRandomSeed
            )
        {
            return KMeansCalculator.SelectRandomCenters(attributes, nMinClusters, nRandomSeed);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Enumerates duplicate centers in this collection. </summary>
        ///
        /// <remarks>   crossbac, 8/5/2013. </remarks>
        ///
        /// <param name="centers">  The centers. </param>
        ///
        /// <returns>   An enumerator that allows foreach to be used to process duplicate centers in this
        ///             collection.
        ///             </returns>
        ///-------------------------------------------------------------------------------------------------

        static Vector[]
        DuplicateCenters(
            IEnumerable<Vector> goldcenters
            )
        {
            int nIndex = 0;
            Vector[] refcenters = new Vector[goldcenters.Count()];
            foreach(Vector center in goldcenters)
                refcenters[nIndex++] = new Vector(center);
            return refcenters;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if 'sVersion' is a parallelizable version of kmeans. </summary>
        ///
        /// <remarks>   crossbac, 8/6/2013. </remarks>
        ///
        /// <param name="sVersion"> The version. </param>
        ///
        /// <returns>   true if parallelizable version, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        static bool
        IsParallelizableVersion(
            String sVersion
            ) {
            switch(sVersion) {
            // must be in [threaded|tasks|seq|LINQ|PLINQ]
            case "seq": return false;
            case "threaded": return true;
            case "tasks": return true;
            case "PLINQ": return true;
            case "LINQ": return false;
            default:
                throw new Exception("unknown version " + sVersion);
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Select implementation. </summary>
        ///
        /// <remarks>   crossbac, 8/6/2013. </remarks>
        ///
        /// <param name="sVersion">         The version. </param>
        /// <param name="nMaxParallelism">  [out] The maximum parallelism. </param>
        /// <param name="bAvoidLazyEval">   [out] The avoid lazy eval. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        static KMeansCalculator
        SelectImplementation(
            KMeansOptions options,
            String sVersion,
            out int nMaxParallelism,
            out bool bAvoidLazyEval
            ) {
            if(sVersion.StartsWith("PLINQ") ||
               sVersion.StartsWith("threaded") ||
               sVersion.StartsWith("tasks")) {
                int nDashIndex = sVersion.IndexOf('-');
                String strMaxParallelism = sVersion.Substring(nDashIndex + 1);
                nMaxParallelism = Int32.Parse(strMaxParallelism);
                bAvoidLazyEval = true;
                if(sVersion.StartsWith("PLINQ")) return new PLINQKMeans(options);
                if(sVersion.StartsWith("threaded")) return new ThreadedKMeans(options);
                if(sVersion.StartsWith("tasks")) return new TaskingKMeans(options);
                return null;
            } else {
                nMaxParallelism = 1;
                bAvoidLazyEval = true;
                if(sVersion == "seq") return new SequentialKMeans(options);
                if(sVersion == "LINQ") return new LINQKMeans(options);
            }
            return null;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Select versions to compare. </summary>
        ///
        /// <remarks>   crossbac, 8/6/2013. </remarks>
        ///
        /// <param name="options">  Options for controlling the operation. </param>
        ///
        /// <returns>   . </returns>
        ///-------------------------------------------------------------------------------------------------

        static List<String>
        SelectVersionsToCompare(
            KMeansOptions options
            ) {
            List<String> runnableVersions = new List<string>();
            List<String> specifiedVersions = options.m_vImplementations.Keys.ToList();
            foreach(String sVersion in specifiedVersions) {
                if(IsParallelizableVersion(sVersion)) {
                    int procs = options.m_nMinimumThreadsSweep;
                    while(procs <= System.Environment.ProcessorCount &&
                          procs <= options.m_nMaximumThreadsSweep) {
                        runnableVersions.Add(sVersion + "-" + procs);
                        if(procs < 16) procs *= 2;
                        else procs += 8;
                    }
                } else {
                    runnableVersions.Add(sVersion);
                }
            }
            return runnableVersions;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Compare performance of several implementations. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/7/2012. </remarks>
        ///
        /// <param name="options">      Options for controlling the operation. </param>
        /// <param name="attributes">   The attributes. </param>
        ///-------------------------------------------------------------------------------------------------

        static void ComparePerformance(
            KMeansOptions options,
            IEnumerable<Vector> attributes
            )
        {
            List<String> runnableVersions = SelectVersionsToCompare(options);
            Dictionary<String, Sample[]> perfdata = new Dictionary<string, Sample[]>();
            Dictionary<String, Performance> stats = new Dictionary<string, Performance>();
            foreach (String version in runnableVersions) {
                Sample[] vdata = new Sample[options.m_nTotalRuns];
                for (int i = 0; i < options.m_nTotalRuns; i++)
                    vdata[i] = new Sample();
                perfdata[version] = vdata;
            }

            IEnumerable<Vector> goldcenters = 
                SelectInitialCenters(attributes, 
                                     options.m_nClusters, 
                                     options.m_nRandomSeed);

            for(int i = 0; i < options.m_nTotalRuns; i++) {
                long lReferenceImplTime = 0;
                IEnumerable<Vector> refnewcenters = null;
                if(options.m_bCheckResult) {
                    Vector[] refcenters = DuplicateCenters(goldcenters);
                    ReferenceKMeans refkmeans = new ReferenceKMeans();
                    refnewcenters = refkmeans.Compute(attributes, refcenters, options.m_nMaxIterations, true);
                    lReferenceImplTime = refkmeans.RuntimeMilliseconds;
                }

                foreach(String sVersion in runnableVersions) {
                    bool bAvoidLazyEval = true;
                    int nMaxParallelism = 1;
                    int[] rClusterIds = null;
                    Vector[] newcenters = DuplicateCenters(goldcenters);
                    KMeansCalculator kmeans = SelectImplementation(options, sVersion, out nMaxParallelism, out bAvoidLazyEval);
                    int nIterations = kmeans.execute(options, attributes, ref newcenters, out rClusterIds, nMaxParallelism, bAvoidLazyEval);
                    Sample isample = perfdata[sVersion][i];
                    isample.m_impltime = kmeans.RuntimeMilliseconds;
                    isample.m_success = true;
                    isample.m_reftime = 0;
                    if(options.m_bCheckResult) {
                        isample.m_reftime = lReferenceImplTime;
                        isample.m_success = KMeansCalculator.CheckResult(options,
                                                                         newcenters,
                                                                         refnewcenters,
                                                                         options.m_bVerbose,
                                                                         options.m_bVerbose);
                        if(!isample.m_success) {
                            Console.WriteLine("FAILED");
                            return;
                        }
                    }
                }
            }
                
            Console.WriteLine("SUCCEEDED");
            foreach(String v in perfdata.Keys) {
                Sample[] samples = perfdata[v];
                Performance perf = new Performance(samples);
                Console.WriteLine("{0,-15}: {1}, avg={2}", v, perf.RawRuntimes(), perf.m_impltime.ToString("f1"));
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   write random vector file. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/2/2012. </remarks>
        ///
        /// <param name="strInputFile"> The input file. </param>
        ///
        /// <returns>   The text input. </returns>
        ///-------------------------------------------------------------------------------------------------

        public static Vector[]
        GenerateRandomInput(
            string strPath,
            int nVectors,
            int nDimensions
            ) {
            List<Vector> attributes = new List<Vector>();
            System.IO.StreamWriter writer = new System.IO.StreamWriter(strPath);
            StringBuilder sb = new StringBuilder();
            Random random = new Random();
            for(int i=0; i < nVectors; i++) {
                sb.Clear();
                sb.Append(i+1);
                sb.Append(" ");
                for(int j=0; j < nDimensions; j++) {
                    if(j > 0)
                        sb.Append(" ");
                    sb.Append(random.NextDouble());
                }
                writer.WriteLine(sb.ToString());
            }
            writer.Close();
            return attributes.ToArray();
        }


        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Main entry-point for this application. </summary>
        ///
        /// <remarks>   crossbac, 8/6/2013. </remarks>
        ///
        /// <param name="args"> Array of command-line argument strings. </param>
        ///-------------------------------------------------------------------------------------------------

        static void Main(string[] args)
        {
            KMeansOptions options = KMeansOptions.getOptions(args);
            if (options == null)
                return;

            if(options.m_bGenerateData) {
                GenerateRandomInput(options.m_strFileName,
                                    options.m_nGenerateElems,
                                    options.m_nGenerateDims);
                return; 
            }

            Vector[] attributes =
                options.m_bBinaryInput ?
                    KMeansCalculator.ReadBinaryInput(options.m_strFileName) :
                    KMeansCalculator.ReadTextInput(options.m_strFileName);

            ComparePerformance(options, attributes);
        }
    }
}


