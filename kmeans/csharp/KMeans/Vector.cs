///-------------------------------------------------------------------------------------------------
// file:	Vector.cs
//
// summary:	Implements yet another vector class
///-------------------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KMeans
{
    public struct Vector
    {
        public float[] m_v;
        public Vector(int nRank) { m_v = new float[nRank]; }
        public float v(int nIndex) { if (nIndex >= m_v.Length) throw new Exception(); return m_v[nIndex]; }
        public void setv(int nIndex, float value) { if (nIndex >= m_v.Length) throw new Exception(); m_v[nIndex] = value; }
        public int Rank { get { return m_v.Length; } }
        public Vector(Vector clone)
        {
            m_v = new float[clone.Rank];
            for (int i = 0; i < clone.Rank; i++)
                m_v[i] = clone[i];
        }
        public float this[int r]
        {
            get { return v(r); }
            set { setv(r, value); }
        }
        public Vector(string strLine)
        {
            // lines are like:
            // <attr index> float_0 float_1 ... float_n
            // so the rank is the number of atoms - 1
            char[] seps = new char[] { ' ', '\t' };
            string[] atoms = strLine.Split(seps);
            m_v = new float[atoms.Length - 1];
            for (int i = 0; i < atoms.Length - 1; i++)
                m_v[i] = Single.Parse(atoms[i + 1]);
        }
        public void clear()
        {
            for (int i = 0; i < m_v.Length; i++)
                m_v[i] = 0.0f;
        }
        public void copy(Vector clone)
        {
            int nBound  = Math.Min(Rank, clone.Rank);
            for (int i = 0; i < nBound; i++)
                m_v[i] = clone.m_v[i];
        }
        public static Vector ZeroVector(int nRank)
        {
            Vector vec = new Vector(nRank);
            vec.clear();
            return vec;
        }
        public static Vector operator +(Vector v1, Vector vec)
        {
            Vector ret = new Vector(v1.Rank);
            for (int i = 0; i < ret.Rank; i++)
                ret[i] = vec[i] + v1[i];
            return ret;
        }
        public static Vector operator /(Vector v1, int n)
        {
            Vector ret = new Vector(v1.Rank);
            for (int i = 0; i < ret.Rank; i++)
                ret[i] = v1[i] / n;
            return ret;
        }
        public static float DistSq(Vector v1, Vector v2)
        {
            float accum = 0.0f;
            for (int i = 0; i < v1.Rank; i++)
            {
                float delta = v1[i] - v2[i];
                accum += delta * delta;
            }
            return accum;
        }
        public static float Dist(Vector v1, Vector v2)
        {
            return (float)Math.Sqrt(DistSq(v1, v2));
        }
        public static Vector operator -(Vector v1, Vector v2)
        {
            Vector ret = new Vector(v1.Rank);
            for (int i = 0; i < ret.Rank; i++)
                ret[i] = v1[i] - v2[i];
            return ret;
        }
        public float Norm2()
        {
            float res = 0.0f;
            for (int i = 0; i < this.m_v.Length; i++)
                res += m_v[i] * m_v[i];
            return res;
        }
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            bool bFirst = true;
            foreach (float elem in m_v)
            {
                if (!bFirst) sb.Append(", ");
                sb.Append(elem.ToString());
                bFirst = false;
            }
            return sb.ToString();
        }
        public Object epsilon { get { return 1E-09; } }
    }

}
