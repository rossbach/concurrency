/********************************************************
* hrperftimer.h
**********************************************************/
#ifndef _HRPERFT_H_
#define _HRPERFT_H_

// performance timers are architecture and platform
// specific. Need to define a routine to access
// the perf counters on whatever processor is in use here:
#include "windows.h"
#define ctrtype double
#define hpfresult(x) x.QuadPart
#define query_hpc(x) QueryPerformanceCounter(x)
#define query_freq(x) QueryPerformanceFrequency(x)
typedef long (__stdcall *LPFNtQuerySystemTime)(PLARGE_INTEGER SystemTime);

typedef enum gran_t {
    gran_msec,
    gran_sec 
} hpf_granularity;

///-------------------------------------------------------------------------------------------------
/// <summary>   High resolution timer. 
/// 			For collecting performance measurements.
/// 			</summary>
///
/// <remarks>   Crossbac, 12/23/2011. </remarks>
///-------------------------------------------------------------------------------------------------

class CHighResolutionTimer {
public:

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Constructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="gran"> The granularity of the timer
    /// 					(seconds or milliseconds). </param>
    ///-------------------------------------------------------------------------------------------------

    CHighResolutionTimer(
        hpf_granularity gran
        )
    {
	    m_freq = 0;
	    m_start = 0;
	    m_gran = gran;
	    init_query_system_time();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Destructor. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    ~CHighResolutionTimer(void) { free_query_system_time(); }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the tick frequency of the underlying
    /// 			counter primitive. 
    /// 			</summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    double tickfreq() {
	    LARGE_INTEGER tps;
	    query_freq(&tps); 
	    return (double)hpfresult(tps);    
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Gets the tick count. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    __int64 tickcnt() {
        LARGE_INTEGER t;
        query_hpc(&t); 
	    return (DWORD)hpfresult(t);
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Resets this timer. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void reset() {
	    if(!m_freq) 
		    m_freq = tickfreq();
	    m_start = tickcnt();
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the time elapsed since the
    /// 			last reset. Optionally, reset the timer
    /// 			as a side-effect of the query. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="reset">    true to reset. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    double elapsed(bool reset) {
	    __int64 end = tickcnt();
	    if(!m_freq) return -1.0;
	    double res = ((double)(end-m_start))/m_freq;
	    if(reset)
		    m_start = end;
	    return m_gran == gran_sec ? res : res * 1000;
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Queries the system time. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="li">   The li. </param>
    ///
    /// <returns>   true if it succeeds, false if it fails. </returns>
    ///-------------------------------------------------------------------------------------------------

    BOOL query_system_time(PLARGE_INTEGER li) {
	    if(!m_lpfnQuerySystemTime) 
		    return FALSE;
	    (*m_lpfnQuerySystemTime)(li);
	    return TRUE;
    }


protected:

    /// <summary> The granularity of the timer,
    /// 		  either seconds or milliseconds 
    /// 		  </summary>
    hpf_granularity m_gran;
    
    /// <summary> the value of the underlying 
    /// 		  timing primitive at the time the 
    /// 		  timer was last reset.</summary>
    __int64 m_start; 
    
    /// <summary> The frequency of the underlying
    /// 		  timing primitive </summary>
    double m_freq;

    /// <summary> Module for windows DLL for querying
    /// 		  system time getting perf counter
    /// 		  frequency. 
    /// 		  </summary>
    HMODULE m_hModule;
    
    /// <summary> Function pointer for querying
    /// 		  system time 
    /// 		  </summary>
    LPFNtQuerySystemTime m_lpfnQuerySystemTime;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Free resources allocated to support
    /// 			query of system time. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///-------------------------------------------------------------------------------------------------

    void free_query_system_time() {
	    if(m_hModule) {
		    FreeLibrary(m_hModule);
	    }
    }

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Initialises the query system time. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    LPFNtQuerySystemTime init_query_system_time() {
	    m_hModule = LoadLibraryW(L"NTDLL.DLL");
	    FARPROC x = GetProcAddress(m_hModule, "NtQuerySystemTime");
	    m_lpfnQuerySystemTime = (LPFNtQuerySystemTime) x;
	    return m_lpfnQuerySystemTime;
    }
    
    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Return the difference in milliseconds. </summary>
    ///
    /// <remarks>   Crossbac, 12/23/2011. </remarks>
    ///
    /// <param name="lEarly">   The early. </param>
    /// <param name="lLate">    The late. </param>
    ///
    /// <returns>   . </returns>
    ///-------------------------------------------------------------------------------------------------

    DWORD delta_milliseconds(LARGE_INTEGER lEarly, LARGE_INTEGER lLate) {
	    LONGLONG ll1 = lEarly.QuadPart;
	    LONGLONG ll2 = lLate.QuadPart;
	    LONGLONG ll = ll2 - ll1;
	    ll = (ll * 100000) / 1000000000;
	    return (DWORD) ll;        
    }

};

#endif
