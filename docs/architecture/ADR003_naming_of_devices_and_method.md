# Decision

We use the device names for the naming certain paths.
There are also some explanation here for some variables/types within the code

|               |                                 |                           |
|---------------|---------------------------------|---------------------------|
| Producer      | Halo-Photonics (prev. Lumibird) | Vaisala (prev. Leosphere) |
| Device name   | StreamLine                      | Windcube                  |
| Output format | hpl file (text, line based)     | Level 1 NetCDF            |

# Scan modes

## DBS

Has 4 or 5 rays. Goes in north-south and west-east direction. Turns 90 degree per scan.
The additional ray would be vertical and thus a stare

## VAD

A VAD with 4 rays is equivalent to a DBS scan, but can have more rays.
It usually has more than 4 rays and thus a better base for the extrapolation of the vertical and horizontal winds

## Stare

It is a zenith scan, straight to the sky. 

# Scan Parameters

## Elevation angle
Vertical angle of ray

## Azimuth angle
Horizontal angle of the rays

## Accumulation time
Collection time of signal

# Scan Modes

## Step and Shoot
Single shots at certain positions (e.g. DBS and VAD)

## Continuous Scan

### PPI
For a specific elevation angle, all 360° are measured with continuous beam and reflection measurement

### RHI
For a specific azimuth angle, all 180° are measured with continuous beam and reflection measurement. (Only a part of the 180° are covered)

# Other naming

## Gate or range gate
Vertical space resolution, equally spaced.
