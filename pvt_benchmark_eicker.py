"""External benchmark: bare-panel night-cooling flux versus published measurements.

Eicker & Dalibard (Solar Energy 85, 1322-1335, 2011) measured night radiative cooling
with uncovered PVT collectors in Madrid: about 60-65 W m-2 when cooling a warm storage
tank, 40-45 W m-2 when cooling a ceiling, and about 41 W m-2 averaged over a summer
with a mean tank temperature of 20.9 C.

The present model is run with every mechanism switched off (bare panel, ordinary
backsheet emissivity 0.90; no plenum, stack, shroud, emitter, thermosiphon, wetting or
EHD) over a stated envelope of Madrid summer-night conditions. To represent the measured
configuration, two terms absent from the reference array calculation are added here:
the ground view of the tilted surface (ground at ambient temperature, sky view factor
(1 + cos tilt)/2), and an areal coolant-to-surface resistance R through which the
surface temperature is solved from the collector fluid temperature.

This is an order-of-magnitude plausibility check of the convective and radiative
sub-models, not a calibration: the measured time series of ambient, sky, wind and fluid
temperature is not reproduced. Run: python pvt_benchmark_eicker.py
"""
from itertools import product
import numpy as np
from pvt_full_system_model import Environment, Geometry, MechanismState, evaluate

SIGMA = 5.670374419e-8
BARE = MechanismState(plenum_stack=False, shrouds=False, emitter_deployed=False, film=False,
                      thermosiphon=False, ehd=False, mist=False)

T_AMB = (16.0, 18.0, 20.0, 22.0)        # C, Madrid summer night
RH = (0.30, 0.45, 0.60)
WIND = (0.5, 1.0, 2.0, 3.0)              # m/s
CLOUD = (0.0, 0.2)
FLUID_OVER_AMB = (3.0, 5.0, 8.0)         # K, collector fluid above ambient
R_AREAL = (0.0, 0.01, 0.02, 0.04)        # m2 K/W, coolant-to-surface resistance
MEASURED = {"tank cooling": (60.0, 65.0), "ceiling cooling": (40.0, 45.0), "summer mean": (41.0, 41.0)}


def ground_view_correction(env, geom, util):
    """Radiative flux lost because a tilted surface sees ground at ambient instead of sky."""
    F_sky = (1.0 + np.cos(np.radians(geom.tilt_deg))) / 2.0
    Ta, Ts = env.T_amb_C + 273.15, env.T_sky_C + 273.15
    return geom.eps_backsheet * SIGMA * (1.0 - F_sky) * (Ta ** 4 - Ts ** 4) * util


def bare_flux(env, geom, T_fluid_C, R):
    def q_at(T_surf):
        r = evaluate(env, geom, BARE, T_panel_C=T_surf)
        return r["q_flux"] - ground_view_correction(env, geom, r["util"])
    if R == 0.0:
        return q_at(T_fluid_C)
    lo, hi = env.T_amb_C - 15.0, T_fluid_C
    for _ in range(60):
        Ts = 0.5 * (lo + hi)
        if T_fluid_C - Ts - q_at(Ts) * R > 0.0:
            lo = Ts
        else:
            hi = Ts
    return q_at(Ts)


def main():
    geom = Geometry()                    # reference non-isothermal utilisation 0.75 for a bare panel
    print("Bare-panel model versus Eicker & Dalibard (2011) measured night cooling")
    print(f"  surface utilisation {geom.baseline_util:.2f}; ground view included; "
          f"{len(T_AMB) * len(RH) * len(WIND) * len(CLOUD)} ambient states per case")
    allq = []
    print("   R (m2K/W)  fluid-ambient (K)   median  5-95% (W/m2)")
    for R in R_AREAL:
        for dT in FLUID_OVER_AMB:
            qs = [bare_flux(Environment(T_amb_C=ta, RH=rh, wind_speed=u, cloud_fraction=c), geom, ta + dT, R)
                  for ta, rh, u, c in product(T_AMB, RH, WIND, CLOUD)]
            allq += qs
            print(f"   {R:9.2f}  {dT:17.0f}   {np.median(qs):6.0f}  {np.percentile(qs, 5):.0f}-{np.percentile(qs, 95):.0f}")
    allq = np.array(allq)
    print(f"  all cases: median {np.median(allq):.0f} W/m2, range {allq.min():.0f}-{allq.max():.0f} W/m2")
    for k, (a, b) in MEASURED.items():
        print(f"  measured {k}: {a:.0f}-{b:.0f} W/m2 -> model/measured median ratio {np.median(allq) / ((a + b) / 2):.2f}")
    # effect of the ground view on the reference bare panel (baseline night, 35 C surface)
    env0 = Environment()
    r0 = evaluate(env0, geom, BARE, T_panel_C=35.0)
    corr = ground_view_correction(env0, geom, r0["util"])
    print(f"  reference baseline bare panel: {r0['q_flux']:.0f} W/m2; ground view would remove "
          f"{corr:.1f} W/m2 ({100 * corr / r0['q_flux']:.1f}%)")


if __name__ == "__main__":
    main()
