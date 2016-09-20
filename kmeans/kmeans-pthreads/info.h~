#include <stdarg.h>
#include <sys/time.h>

static int _verbose = 0;

#define FMSG_BODY(x)                \
  if(_verbose) {                    \
    va_list args;                   \
    va_start(args, fmt);            \
    vfprintf((x), fmt, args);	    \
    va_end(args);                   \
    fflush(stdout);}
    

static inline void _info(const char* fmt, ...) { FMSG_BODY(stdout); }
static inline void _error(const char* fmt, ...) { FMSG_BODY(stderr); }
static inline unsigned ticks() {
  struct timeval tv;
  if(gettimeofday(&tv, NULL) != 0)
    return 0;
  return (tv.tv_sec*1000)+(tv.tv_usec / 1000);
}
