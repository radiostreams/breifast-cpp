// No-op stand-in for Tracy's public header, used when the tracy/ checkout is absent and -Dtracy=false.
// Tracy's own header defines exactly these as empty without TRACY_ENABLE, so nothing changes in the build.
#pragma once
#define ZoneScoped
#define ZoneScopedN(name)
#define ZoneNamed(varname, active)
#define ZoneNamedN(varname, name, active)
#define FrameMark
#define FrameMarkNamed(name)
#define TracyPlot(name, value)
#define TracyMessageL(txt)
