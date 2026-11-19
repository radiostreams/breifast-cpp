// VTune ITT collection control, or nothing when built without VTune (meson -Dvtune_root unset).
#pragma once
#ifdef BREIFAST_HAVE_ITT
#include <ittnotify.h>
#else
#define __itt_pause()
#define __itt_resume()
#endif
