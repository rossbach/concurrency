///-------------------------------------------------------------------------------------------------
// file:	TaskingKMeans.cs
//
// summary:	Implements the tasking k means class
///-------------------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics;

namespace KMeans
{

    public class TaskingKMeans : KMeansCalculator  {

        /// <summary>   Options for controlling the operation. </summary>
        protected KMeansOptions m_options = null;
        protected static bool m_bMapPhasesComplete = false;
        protected static bool m_bUpdatePhasesComplete = false;        
        protected ManualResetEvent m_evtMapWorkAvailable    = new ManualResetEvent(false);
        protected ManualResetEvent m_evtMapWorkComplete     = new ManualResetEvent(false);
        protected ManualResetEvent m_evtUpdateWorkAvailable = new ManualResetEvent(false);
        protected ManualResetEvent m_evtUpdateWorkComplete  = new ManualResetEvent(false);   
        protected Barrier m_barrier;

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

        public TaskingKMeans(KMeansOptions options) { m_options = options; }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates a map partition description. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/7/2012. </remarks>
        ///
        /// <param name="options">          Options for controlling the operation. </param>
        /// <param name="vectors">          The vectors. </param>
        /// <param name="initCenters">      The initialise centers. </param>
        /// <param name="clusterIds">       [out] List of identifiers for the clusters. </param>
        /// <param name="nCenters">         The centers. </param>
        /// <param name="nRank">            The rank. </param>
        /// <param name="nId">              The identifier. </param>
        /// <param name="nStartIndex">      The start index. </param>
        /// <param name="nPartitionSize">   Size of the partition. </param>
        ///
        /// <returns>   The new map partition description. </returns>
        ///-------------------------------------------------------------------------------------------------

        public KMeansMapPartition
        CreateMapPartitionDesc(
            KMeansOptions options,
            Vector[] vectors,
            Vector[] initCenters,
            int[] clusterIds,
            int nCenters,
            int nRank,
            int nId,
            int nStartIndex,
            int nPartitionSize
            )
        {
            KMeansMapPartition partition = new KMeansMapPartition();

            partition.m_evtMapWorkComplete     = m_evtMapWorkComplete;
            partition.m_evtMapWorkAvailable    = m_evtMapWorkAvailable;
            partition.m_evtUpdateWorkComplete  = m_evtUpdateWorkComplete;
            partition.m_evtUpdateWorkAvailable = m_evtUpdateWorkAvailable;
            partition.m_barrier                = m_barrier;
            partition.m_nId                    = nId;
            partition.m_nStartIndex            = nStartIndex;
            partition.m_nPartitionSize         = nPartitionSize;
            partition.m_nCenters               = nCenters;
            partition.m_nRank                  = nRank;
            partition.m_nPoints                = vectors.Count();
            partition.m_oldCenters             = initCenters;
            partition.m_newCenters             = null;
            partition.m_newGroupCounts         = null;
            partition.m_clusterIds             = clusterIds;
            partition.m_fDelta                 = 0.0f;
            partition.m_vectors                = vectors;

            return partition;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates an update description. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/7/2012. </remarks>
        ///
        /// <param name="options">      Options for controlling the operation. </param>
        /// <param name="workers">      The workers. </param>
        /// <param name="vectors">      The vectors. </param>
        /// <param name="clusterIds">   [out] List of identifiers for the clusters. </param>
        /// <param name="nCenters">     The centers. </param>
        /// <param name="nRank">        The rank. </param>
        /// <param name="nIteration">   The iteration. </param>
        ///
        /// <returns>   The new update description. </returns>
        ///-------------------------------------------------------------------------------------------------

        KMeansUpdateCenters
        CreateUpdateDesc(
            KMeansOptions options,
            KMeansMapPartition[] workers,
            Vector[] vectors,
            int[] clusterIds,
            int nCenters,
            int nRank,
            int nIteration
            )
        {
            KMeansUpdateCenters update = new KMeansUpdateCenters();

            update.m_evtMapWorkComplete     = m_evtMapWorkComplete;
            update.m_evtMapWorkAvailable    = m_evtMapWorkAvailable;
            update.m_evtUpdateWorkComplete  = m_evtUpdateWorkComplete;
            update.m_evtUpdateWorkAvailable = m_evtUpdateWorkAvailable;
            update.m_barrier                = m_barrier;
            update.m_nIterations            = nIteration;
            update.m_nCenters               = nCenters;
            update.m_nRank                  = nRank;
            update.m_nPoints                = vectors.Count();
            update.m_vectors                = vectors;
            update.m_sharedCenters          = null;
            update.m_sharedGroupCounts      = null;
            update.m_counts                 = null;
            update.m_clusterIds             = clusterIds;
            update.m_fDelta                 = 0.0f;
            update.m_workers                = workers;
            update.m_options                = options;

            return update;
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Map partition. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/7/2012. </remarks>
        ///
        /// <param name="partition">    The partition. </param>
        ///-------------------------------------------------------------------------------------------------

        public static void MapPartition(
            object oPartition            
            ) {
            KMeansMapPartition partition = oPartition as KMeansMapPartition;
            while(!m_bMapPhasesComplete) {
                partition.m_newCenters = KMeansCalculator.CreateCenterAccumulatorList(partition.m_nCenters, partition.m_nRank);
                partition.m_newGroupCounts = KMeansCalculator.CreateGroupCountList(partition.m_nCenters, partition.m_nRank);
                partition.m_fDelta = 0.0f;
                partition.m_evtMapWorkAvailable.WaitOne();
                if(!m_bMapPhasesComplete) {
                    for (int i = partition.m_nStartIndex; i < partition.m_nPartitionSize; i++) {
                        Vector vec = partition.m_vectors.ElementAt(i);
                        int nIndex = FindNearestCenter(vec, partition.m_oldCenters);
                        if (partition.m_clusterIds.ElementAt(i) != nIndex)
                            partition.m_fDelta += 1.0f;
                        partition.m_clusterIds[i] = nIndex;
                        partition.m_newCenters[nIndex] += vec;
                        partition.m_newGroupCounts[nIndex] += 1;
                    }
                    partition.m_barrier.SignalAndWait();
                }
            }
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Updates the centers. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/7/2012. </remarks>
        ///
        /// <param name="update">       The update. </param>
        ///-------------------------------------------------------------------------------------------------

        public static void UpdateCenters(
            object oUpdate
            )
        {
            KMeansUpdateCenters update = oUpdate as KMeansUpdateCenters;
            while(!m_bUpdatePhasesComplete) {
                update.m_sharedCenters = KMeansCalculator.CreateCenterAccumulatorList(update.m_nCenters, update.m_nRank);
                update.m_sharedGroupCounts = KMeansCalculator.CreateGroupCountList(update.m_nCenters, update.m_nRank);
                update.m_fDelta = 0.0f;
                update.m_evtMapWorkComplete.WaitOne();
                for (int w = 0; w < update.m_workers.Count(); w++) {
                    KMeansMapPartition worker = update.m_workers[w];
                    update.m_fDelta += worker.m_fDelta;
                    for (int i = 0; i < update.m_nCenters; i++) {
                        update.m_sharedCenters[i] += worker.m_newCenters[i];
                        update.m_sharedGroupCounts[i] += worker.m_newGroupCounts[i];
                    }
                }
                for (int i = 0; i < update.m_nCenters; i++) 
                    update.m_sharedCenters[i] /= update.m_sharedGroupCounts[i];
                update.m_fDelta /= update.m_nPoints;
                update.m_nIterations++;

                if(!(update.m_nIterations > update.m_options.m_nMaxIterations || 
                     update.m_fDelta < update.m_options.m_fConvergenceThreshold)) {
                    update.m_evtMapWorkComplete.Reset();
                    update.m_evtMapWorkAvailable.Set();
                } else {
                    m_bMapPhasesComplete = true;
                    m_bUpdatePhasesComplete = true;
                    update.m_evtMapWorkAvailable.Set(); // shouldn't be needed, but harmless
                    return;
                }
            }
        }


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
            ) 
        {
            accum = CreateCenterAccumulatorList(nCenters, nRank);
            counts = CreateGroupCountList(nCenters, nRank);
            clusterIds = CreateInitialClusterIds(nPoints);
        }

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Creates an initial cluster identifiers. </summary>
        ///
        /// <remarks>   Chris Rossbach (crossbach@microsoft.com), 8/7/2012. </remarks>
        ///
        /// <param name="nPoints">  The points. </param>
        ///
        /// <returns>   The new initial cluster identifiers. </returns>
        ///-------------------------------------------------------------------------------------------------

        public int[]
        CreateInitialClusterIds(
            int nPoints
            ) {
            int[] tclusterIds = new int[nPoints];
            for(int i = 0; i < nPoints; i++)
                tclusterIds[i] = -1;
            return tclusterIds;
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
            int nCenters,
            out int[] clusterIds,
            out int nIterations
            )
        {
            int nRank = vectors.ElementAt(0).Rank;
            int nPoints = vectors.Count();
            clusterIds = CreateInitialClusterIds(nPoints);

            m_barrier = new Barrier(options.m_nCurrentThreadsSweep, (b) => {
                m_evtMapWorkAvailable.Reset();
                m_evtMapWorkComplete.Set();
                Console.WriteLine("reached barrier!");
            });

            KMeansMapPartition[] partitions;
            Thread[] mappers;
            int nVectors = vectors.Count();
            int nWorkers = options.m_nCurrentThreadsSweep;
            int nPartitionSize = (int) Math.Round((double) nVectors / (double) nWorkers);
            partitions = new KMeansMapPartition[nWorkers];
            mappers = new Thread[nWorkers];

            m_pStopwatch.Start();
            m_nStartParallelPhase = m_pStopwatch.ElapsedMilliseconds;

            for(int i=0; i<nWorkers; i++) {
                int nStartIndex = i * nPartitionSize;
                int nWorkerPartitionSize = nPartitionSize;
                if(nStartIndex + nPartitionSize > nVectors)
                    nWorkerPartitionSize = nVectors - nStartIndex;
                partitions[i] = CreateMapPartitionDesc(options,
                                                       vectors as Vector[],
                                                       centers,
                                                       clusterIds,
                                                       nCenters,
                                                       nRank,
                                                       i,
                                                       nStartIndex,
                                                       nPartitionSize);
                mappers[i] = new Thread(TaskingKMeans.MapPartition);
                mappers[i].Start(partitions[i]);
            }

            KMeansUpdateCenters update;
            update = CreateUpdateDesc(options,
                                      partitions,
                                      vectors as Vector[],
                                      clusterIds,
                                      nCenters,
                                      nRank,
                                      0);

            Thread updater = new Thread(TaskingKMeans.UpdateCenters);
            updater.Start(update);
            updater.Join();

            m_pStopwatch.Stop();
            m_nEndParallelPhase = m_pStopwatch.ElapsedMilliseconds;

            nIterations = update.m_nIterations;
            centers = update.m_sharedCenters;
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
            int nMaxParallelism=1,
            bool bAvoidLazyEval=true
            )
        {
            int nIterations = 0;
            options.m_nCurrentThreadsSweep = nMaxParallelism;
            centers = ComputeNewCenters(options, 
                                        vectors, 
                                        centers, 
                                        options.m_nClusters, 
                                        out clusterIds, 
                                        out nIterations);
            return nIterations;
        }
    }
}
