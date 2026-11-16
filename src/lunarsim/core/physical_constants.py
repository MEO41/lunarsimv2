# SPDX-License-Identifier: BSD-3-Clause
"""Physical constants. Every number has its public source next to it."""

# Gravitational parameter of the Moon, m^3/s^2.
# Source: Park, Folkner, Williams, Boggs (2021), "The JPL Planetary and Lunar Ephemerides DE440
# and DE441", The Astronomical Journal 161:105, Table 2: GM_Moon = 4902.800118 km^3/s^2.
MOON_GRAVITATIONAL_PARAMETER = 4.902800118e12

# Mean radius of the Moon, m.
# Source: NASA NSSDCA Moon Fact Sheet (volumetric mean radius 1737.4 km),
# https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
MOON_MEAN_RADIUS = 1.7374e6

# Gravitational acceleration at the mean lunar surface, m/s^2 (derived: GM / R^2 = 1.6242).
MOON_SURFACE_GRAVITY = MOON_GRAVITATIONAL_PARAMETER / MOON_MEAN_RADIUS**2

# Standard gravity, m/s^2, used to convert specific impulse (s) to exhaust velocity.
# Source: 3rd General Conference on Weights and Measures (CGPM), 1901; BIPM SI Brochure.
STANDARD_GRAVITY = 9.80665

# Sidereal rotation rate of the Moon, rad/s (derived: 2 pi / 655.720 h = 2.6617e-6).
# Source: NASA NSSDCA Moon Fact Sheet, "Sidereal rotation period (hrs) 655.720",
# https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
MOON_SIDEREAL_ROTATION_PERIOD = 655.720 * 3600.0
MOON_ROTATION_RATE = 2.0 * 3.141592653589793 / MOON_SIDEREAL_ROTATION_PERIOD
