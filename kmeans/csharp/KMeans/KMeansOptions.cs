///-------------------------------------------------------------------------------------------------
// file:	KMeansOptions.cs
//
// summary:	Implements the means options class
///-------------------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KMeans
{
    public class KMeansOptions
    {
        public String m_strFileName = "..\\..\\..\\inputs\\random-n2048-d16-c16.txt";
        public bool m_bVerbose = false;
        public bool m_bBinaryInput = false;
        public int m_nClusters = 16;
        public float m_fConvergenceThreshold = 0.00001f;
        public float m_fEpsilon = 0.001f;
        public int m_nPartitionSize = 1;
        public int m_nMaxIterations = 100;
        public int m_nMinimumThreadsSweep = 1;
        public int m_nMaximumThreadsSweep = -1;
        public int m_nCurrentThreadsSweep = 1;
        public int m_nTotalRuns = 3;
        public int m_nRandomSeed = -1;
        public bool m_bCheckResult = false;
        public bool m_bComparePerformance = true;
        public bool m_bGenerateData = false;
        public int m_nGenerateDims = 0;
        public int m_nGenerateElems = 0;
        public Dictionary<String, KMeansCalculator> m_vImplementations =
            new Dictionary<string, KMeansCalculator>();

        static void PrintUsage()
        {
            Console.WriteLine("Usage: %s [switches] -i filename");
            Console.WriteLine("       -i filename    : file containing data to be clustered");
            Console.WriteLine("       -b             : input file is in binary format");
            Console.WriteLine("       -m clusters    : number of clusters");
            Console.WriteLine("       -T threshold   : threshold value for termination");
            Console.WriteLine("       -K elems       : number of elems per chunk for vector input partitioning (default==1)");
            Console.WriteLine("       -r runs        : number of runs to measure");
            Console.WriteLine("       -c             : check result");
            Console.WriteLine("       -R rand-seed   : provide seed for selection of initial centers");
            Console.WriteLine("       -x min-threads : min number of threads in sweep");
            Console.WriteLine("       -X max-threads : max thread count in sweep");
            Console.WriteLine("       -e epsilon     : epsilon for correctness check to accommodate FP error.");
            Console.WriteLine("       -I iterations  : max iterations to run.");
            Console.WriteLine("       -v             : max verbosity");
            Console.WriteLine("       -g             : generate random data (to -i file, with -e elems, -d dims)");
            Console.WriteLine("       -d dims        : number of dimensions in generated random data (-g option)");
            Console.WriteLine("       -E elems       : number of vectors to generate (-g)");
            Console.WriteLine("       -V version     : version to include (multiple use OK) [threaded|tasks|seq|LINQ|PLINQ]");
        }

        public static KMeansOptions getOptions(string[] args) {
            Getopt.Getopt g = new Getopt.Getopt("KMeans", args, "cr:x:i:m:T:e:bR:X:K:I:vV:gd:E:");
            KMeansOptions options = new KMeansOptions();
            try {
                int c;
                while((c = g.getopt()) != -1) {
                    switch(c) {
                    case 'g':
                        options.m_bGenerateData = true;
                        break;
                    case 'd':
                        options.m_nGenerateDims = Int32.Parse(g.Optarg);
                        break;
                    case 'E':
                        options.m_nGenerateElems = Int32.Parse(g.Optarg);
                        break;
                    case 'v':
                        options.m_bVerbose = true;
                        break;
                    case 'V':
                        options.m_vImplementations[g.Optarg] = null;
                        break;
                    case 'X':
                        options.m_nMaximumThreadsSweep = Int32.Parse(g.Optarg);
                        break;
                    case 'K':
                        options.m_nPartitionSize = Int32.Parse(g.Optarg);
                        break;
                    case 'R':
                        options.m_nRandomSeed = Int32.Parse(g.Optarg);
                        break;
                    case 'c':
                        options.m_bCheckResult = true;
                        break;
                    case 'C':
                        options.m_bComparePerformance = true;
                        break;
                    case 'r':
                        options.m_nTotalRuns = Int16.Parse(g.Optarg);
                        break;
                    case 'T':
                        options.m_fConvergenceThreshold = float.Parse(g.Optarg);
                        break;
                    case 'e':
                        options.m_fEpsilon = float.Parse(g.Optarg);
                        break;
                    case 'x':
                        options.m_nMinimumThreadsSweep = int.Parse(g.Optarg);
                        break;
                    case 'i':
                        options.m_strFileName = g.Optarg;
                        break;
                    case 'I':
                        options.m_nMaxIterations = int.Parse(g.Optarg);
                        break;
                    case 'm':
                        options.m_nClusters = int.Parse(g.Optarg);
                        break;
                    case 'b':
                        options.m_bBinaryInput = true;
                        break;
                    case 'h':
                        PrintUsage();
                        return null;
                    default:
                        Console.WriteLine("Unknown option: " + (Char) c);
                        PrintUsage();
                        return null;
                    }
                }
            } catch(Exception e) {
                Console.WriteLine("Options exception: " + e.Message);
                PrintUsage();
                return null;
            }
            return options;
        }
    }
}
