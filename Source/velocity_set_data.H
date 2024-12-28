
#ifndef VELOCITY_SET_DATA_H_
#define VELOCITY_SET_DATA_H_
#if (AMREX_SPACEDIM == 2)
// D2Q9 parameters
const unsigned int ndir = 9;
const double w0 = 4.0 / 9.0;  // zero weight
const double ws = 1.0 / 9.0;  // adjacent weight
const double wd = 1.0 / 36.0; // diagonal weight
const double wi[] = {w0, ws, ws, ws, ws, wd, wd, wd, wd};
const int dirx[] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int diry[] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
#elif (AMREX_SPACEDIM == 3)
// D3Q27 parameters
const unsigned int ndir = 27;
const double w0 = 8.0 / 27.0;  // zero weight
const double ws = 2.0 / 27.0;  // adjacent weight
const double wd = 1.0 / 54.0;  // diagonal weight
const double wv = 1.0 / 216.0; // vertex weight
const double wi[] = {w0, ws, ws, ws, ws, ws, ws, wd, wd, wd, wd, wd, wd, wd,
                     wd, wd, wd, wd, wd, wv, wv, wv, wv, wv, wv, wv, wv};
const int dirx[] = {0,  +1, -1, 0, 0, 0,  0,  +1, -1, +1, -1, 0,  0, +1,
                    -1, +1, -1, 0, 0, +1, -1, +1, -1, +1, -1, -1, +1};
const int diry[] = {0,  0, 0, +1, -1, 0,  0,  +1, -1, 0,  0,  +1, -1, -1,
                    +1, 0, 0, +1, -1, +1, -1, +1, -1, -1, +1, +1, -1};
const int diry[] = {0, 0,  0,  0,  0,  +1, -1, 0,  0,  +1, -1, +1, -1, 0,
                    0, -1, +1, -1, +1, +1, -1, -1, +1, +1, -1, +1, -1};
#endif
#endif // VELOCITY_SET_DATA_H_