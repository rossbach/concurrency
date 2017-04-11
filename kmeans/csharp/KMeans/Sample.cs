///-------------------------------------------------------------------------------------------------
// file:	Sample.cs
//
// summary:	Implements the sample class
///-------------------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMeans {
    public class Sample {
        public long m_impltime;
        public long m_reftime;
        public bool m_success;
        public int m_threads;
        public void clear() { init(0, 0, false, 1); }
        public void init(
            long _impltime,
            long _reftime,
            bool _success,
            int _threads
            ) {
            m_impltime = _impltime;
            m_reftime = _reftime;
            m_success = _success;
            m_threads = _threads;
        }
        public Sample() { clear(); }
        public Sample(
            long _impltime,
            long _reftime,
            bool _success,
            int _threads
            ) {
            init(_impltime, _reftime, _success, _threads);
        }
    };
}
