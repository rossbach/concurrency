using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMeansWrapper_x;
using System.Diagnostics;

namespace clr_test
{

    // -----------------------------------------------------------
    // A simple program to run kmeans_clr and make sure it works
    // JPM
    // -----------------------------------------------------------
    class Program
    {

        // how many coordinates per point
        static int dim=100;
        // how many iterations of kmeans
        static int iter = 2;
        static int pointCount= 10000000/9; 
        static int centerCount=   10000; 

        static unsafe void Main(string[] args)
        {
            Random rnd = new Random(12345);
            float[] points =  randomPoints( rnd, pointCount*dim );
           // float[] points = new float[pointCount*dim];
            float[] centers = randomPoints( rnd, centerCount*dim );
            float[] sums = new float[centerCount*dim];
            int[] counts = new int[centerCount];

            Stopwatch timer = Stopwatch.StartNew();

            fixed (float* p_points = points, p_centers = centers, p_sums = sums) fixed(int* p_counts = counts) {
                Console.WriteLine("Iteration 0");
                KMeansWrapper.FirstIter( p_points, p_centers, p_sums, p_counts, pointCount, centerCount );
                for (int i=1; i<iter; i++) {
                    Console.WriteLine("Iteration {0}. Elapsed: {1} ms",i,timer.ElapsedMilliseconds);
                    KMeansWrapper.NextIter( p_points, p_sums, p_counts, pointCount, centerCount );
                }
            }

            fixed (float* p_points = points, p_centers = centers, p_sums = sums) fixed(int* p_counts = counts) {
                KMeansWrapper.LastIter( p_sums, p_counts, pointCount, centerCount );
            }

            Console.WriteLine("Done. Elapsed: {0} ms. Press enter.",timer.ElapsedMilliseconds);

            Console.WriteLine("First three centers: ");
            for (int c=0; c<3; c++) {
                for (int i=0; i<dim; i++) {
                    Console.Write(centers[c*dim+i]+",");
                }
                Console.WriteLine("\n");
            }

            Console.ReadLine();
        }

        static private float[] randomPoints( Random rnd, int count) {
            return Enumerable.Range(0,count).Select(x=>((float)rnd.NextDouble())).ToArray();            
        }
    }
}
