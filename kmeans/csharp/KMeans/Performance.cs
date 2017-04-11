///-------------------------------------------------------------------------------------------------
// file:	Performance.cs
//
// summary:	Implements the performance class
///-------------------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMeans {
    public class Performance {
        public double m_impltime;
        public double m_reftime;
        public bool m_success;
        public Sample[] m_samples;
        public void clear() { init(0.0, 0.0, true); }
        public void init(
            double _impltime,
            double _reftime,
            bool _success
            ) {
            m_impltime = _impltime;
            m_reftime = _reftime;
            m_success = _success;
        }
        public Performance() { clear(); }
        public Performance(
            Sample[] _samples
            ) {
            clear();
            m_samples = _samples;
            foreach(Sample sample in _samples) {
                m_impltime += sample.m_impltime;
                m_reftime += sample.m_reftime;
                m_success &= sample.m_success;
            }
            m_impltime /= _samples.Length;
            m_reftime /= _samples.Length;
        }
        public String RawRuntimes() {
            bool bFirst = true;
            String strRuntimes = "";
            foreach(Sample sample in m_samples) {
                strRuntimes += bFirst ? "" : ", ";
                strRuntimes += sample.m_impltime.ToString("f2");
                bFirst = false;
            }
            return strRuntimes;
        }
        public String RawReftimes() {
            bool bFirst = true;
            String strRuntimes = "";
            foreach(Sample sample in m_samples) {
                strRuntimes += bFirst ? "" : ", ";
                strRuntimes += sample.m_reftime.ToString("f2");
                bFirst = false;
            }
            return strRuntimes;
        }
    };
}
