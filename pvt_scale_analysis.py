#!/usr/bin/env python3
"""
=============================================================================
SCALE ANALYSIS — Roof-Area Matching, Acoustic Benefit and Water Benefit
Across Data-Centre Archetypes from Hyperscale AI to Micro Co-Location
=============================================================================

QUESTION
--------
A fanless PVT rejection array rejects heat at a flux of order 500-750 W/m2.
A data centre generates heat at a flux set by its power density and floor
plate. Whether the array can absorb a useful fraction of the load is
therefore a pure area-matching problem, and the answer differs by two orders
of magnitude between a hyperscale AI facility and a micro installation in a
laundromat or car wash.

THE TWO BENEFITS SCALE DIFFERENTLY
----------------------------------
Let f be the fraction of heat load the silent array absorbs.

  WATER saving scales LINEARLY with f:
      the array displaces evaporative duty in direct proportion.

  ACOUSTIC benefit scales LOGARITHMICALLY with f:
      residual mechanical plant still runs, and
      dL_W = 10 log10(1 - f)
      so f = 0.10 gives 0.5 dB, f = 0.50 gives 3.0 dB, and only
      f -> 1 gives elimination.

This asymmetry is the central result of the analysis. Partial coverage is
worth having for water and almost worthless for noise.

=============================================================================
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict

from pvt_full_system_model import (
    Environment, Geometry, Acoustics, MechanismState, evaluate,
    standard_environments,
)
from pvt_mist_combined_model import (
    dry_cooler, evaporative_cooler, fan_sound_power, sum_levels_dB,
    sound_pressure_at, perceived_loudness_ratio,
)


# ============================================================================
# ARCHETYPES
# ============================================================================
@dataclass
class DataCentre:
    label: str
    it_load_kW: float          # IT load
    pue: float                 # power usage effectiveness
    footprint_m2: float        # building footprint (ground area)
    storeys: int               # roof area = footprint / storeys
    roof_usable: float         # fraction of roof available for array
    heat_recovered: float = 0.0   # fraction of heat usefully reused on site
    site_area_m2: float | None = None   # whole parcel, for ground-mount option
    rack_kW: float = 27.0      # representative rack density
    note: str = ""

    @property
    def heat_to_reject_kW(self) -> float:
        """Total heat that must leave the site, after any useful recovery."""
        return self.it_load_kW * self.pue * (1.0 - self.heat_recovered)

    @property
    def roof_area_m2(self) -> float:
        return self.footprint_m2 / max(self.storeys, 1)

    @property
    def usable_roof_m2(self) -> float:
        return self.roof_area_m2 * self.roof_usable

    @property
    def roof_heat_flux_W_m2(self) -> float:
        """Heat to reject per square metre of USABLE roof."""
        return self.heat_to_reject_kW * 1000.0 / max(self.usable_roof_m2, 1e-9)


def archetypes() -> list[DataCentre]:
    """
    Representative facilities. Rack densities reflect 2026 industry survey
    data: ~27 kW/rack average, 50-70 kW common in AI deployments, and
    120-140 kW for current-generation GPU racks.
    White-space allowance is taken as 3.0 m2 per rack including aisles, and
    white space as roughly half of gross floor area for large facilities.
    """
    return [
        DataCentre("AI training hall (GB200-class)", 100_000, 1.15, 40_000, 2,
                   0.60, 0.0, 260_000, 130.0,
                   "current-generation GPU racks, liquid cooled"),
        DataCentre("Hyperscale, AI-dense", 60_000, 1.20, 45_000, 1,
                   0.60, 0.0, 220_000, 60.0, "50-70 kW racks"),
        DataCentre("Hyperscale, conventional", 40_000, 1.30, 45_000, 1,
                   0.65, 0.0, 180_000, 27.0, "2026 average rack density"),
        DataCentre("Enterprise colocation", 5_000, 1.45, 9_000, 2,
                   0.60, 0.0, 22_000, 16.0, ""),
        DataCentre("Regional edge facility", 500, 1.45, 1_500, 1,
                   0.70, 0.0, 4_000, 12.0, ""),
        DataCentre("Micro DC — car wash", 30, 1.35, 600, 1,
                   0.75, 0.35, 1_400, 10.0,
                   "waste heat pre-heats wash water"),
        DataCentre("Micro DC — laundromat", 50, 1.35, 450, 1,
                   0.75, 0.55, 900, 10.0,
                   "waste heat pre-heats laundry hot water"),
        DataCentre("Micro DC — combined site", 80, 1.30, 1_000, 1,
                   0.75, 0.45, 2_200, 12.0,
                   "laundromat + car wash on one parcel"),
    ]


# ============================================================================
# ARRAY CAPABILITY
# ============================================================================
def array_flux(env: Environment, geom: Geometry, wet: bool) -> float:
    """Rejection flux [W/m2] of the array in dry or wet mode."""
    st = MechanismState(plenum_stack=True, shrouds=True, emitter_deployed=True,
                        thermosiphon=True, film=wet, mist=wet,
                        film_duty=0.30 if wet else 0.0,
                        mist_duty=0.35 if wet else 0.0, ehd=False)
    return evaluate(env, geom, st)["q_flux"]


# ============================================================================
# ACOUSTIC MODEL FOR THE RESIDUAL PLANT
# ============================================================================
def plant_sound_power(heat_kW: float, ac: Acoustics, wet: bool) -> float:
    """A-weighted sound power of mechanical-draught plant of a given duty."""
    if heat_kW <= 1e-6:
        return -np.inf

    class _C: T_amb_C = 5.0
    class _A:
        K_W_axial = ac.K_W_axial; dry_air_rise_K = ac.dry_air_rise_K
        dry_static_dp_Pa = ac.dry_dp_Pa
        evap_air_enthalpy_rise = ac.evap_enthalpy_rise
        evap_static_dp_Pa = ac.evap_dp_Pa
        evap_water_noise_LWA = ac.evap_water_LWA

    if wet:
        pl = evaporative_cooler(heat_kW, _C(), _A())
        LW = sum_levels_dB([fan_sound_power(pl["Q_vol"], pl["dp_Pa"],
                                            ac.K_W_axial) + ac.fan_a_weight,
                            ac.evap_water_LWA + 10 * np.log10(heat_kW / 100.0)])
    else:
        pl = dry_cooler(heat_kW, _C(), _A())
        LW = fan_sound_power(pl["Q_vol"], pl["dp_Pa"], ac.K_W_axial) + ac.fan_a_weight
    return LW


def boundary_level(LW: float, distance_m: float, ac: Acoustics) -> float:
    """Sound pressure at the site boundary, energetically summed with background."""
    if not np.isfinite(LW):
        return ac.background_dBA
    src = sound_pressure_at(LW, distance_m, True)
    return sum_levels_dB([src, ac.background_dBA])


# ============================================================================
# MAIN ANALYSIS
# ============================================================================
def analyse(dc: DataCentre, env: Environment, geom: Geometry, ac: Acoustics,
            wet: bool = False, ground_mount: bool = False,
            hours: float = 12.0) -> dict:
    """
    Area-match one facility against the array and compute the resulting
    acoustic and water outcomes.
    """
    q = array_flux(env, geom, wet)                    # W/m2
    area = dc.usable_roof_m2
    if ground_mount and dc.site_area_m2:
        # 25 % of the remaining parcel, after the building footprint
        area += 0.25 * max(dc.site_area_m2 - dc.footprint_m2, 0.0)

    cap_kW = q * area / 1000.0                        # array capacity
    demand_kW = dc.heat_to_reject_kW
    f = float(np.clip(cap_kW / max(demand_kW, 1e-9), 0.0, 1.0))
    residual_kW = demand_kW * (1.0 - f)

    # --- acoustics -------------------------------------------------------
    LW_base = plant_sound_power(demand_kW, ac, wet=True)     # all-mechanical
    LW_res = plant_sound_power(residual_kW, ac, wet=True)
    d_boundary = max(np.sqrt(dc.site_area_m2 or dc.footprint_m2) / 2.0, 10.0)
    Lp_base = boundary_level(LW_base, d_boundary, ac)
    Lp_res = boundary_level(LW_res, d_boundary, ac)
    dL = Lp_base - Lp_res
    # theoretical sound-power reduction, independent of geometry
    dLW = -10.0 * np.log10(max(1.0 - f, 1e-9)) if f < 1 else np.inf

    # --- water -----------------------------------------------------------
    class _C: T_amb_C = env.T_amb_C
    class _A:
        K_W_axial = ac.K_W_axial; dry_air_rise_K = ac.dry_air_rise_K
        dry_static_dp_Pa = ac.dry_dp_Pa
        evap_air_enthalpy_rise = ac.evap_enthalpy_rise
        evap_static_dp_Pa = ac.evap_dp_Pa
        evap_water_noise_LWA = ac.evap_water_LWA
    base_water_L = evaporative_cooler(demand_kW, _C(), _A())["water_kg_s"] * hours * 3600.0
    res_water_L = evaporative_cooler(residual_kW, _C(), _A())["water_kg_s"] * hours * 3600.0

    st = MechanismState(film=wet, mist=wet, film_duty=0.30 if wet else 0.0,
                        mist_duty=0.35 if wet else 0.0)
    r = evaluate(env, geom, st)
    array_water_L = (r["m_makeup"] * (area / geom.A_total)) * hours * 3600.0
    total_water_L = res_water_L + array_water_L
    water_saved_pct = (1.0 - total_water_L / max(base_water_L, 1e-9)) * 100.0

    return {
        "label": dc.label, "it_kW": dc.it_load_kW, "rack_kW": dc.rack_kW,
        "demand_kW": demand_kW, "usable_roof_m2": area,
        "roof_flux_W_m2": dc.roof_heat_flux_W_m2,
        "array_flux_W_m2": q, "array_cap_kW": cap_kW,
        "coverage": f, "coverage_pct": f * 100.0,
        "residual_kW": residual_kW,
        "boundary_m": d_boundary,
        "Lp_base": Lp_base, "Lp_residual": Lp_res,
        "noise_reduction_dB": dL, "sound_power_reduction_dB": dLW,
        "loudness_ratio": perceived_loudness_ratio(dL),
        "base_water_L": base_water_L, "total_water_L": total_water_L,
        "water_saved_pct": water_saved_pct,
        "area_needed_for_full_m2": demand_kW * 1000.0 / max(q, 1e-9),
        "area_ratio_needed": (demand_kW * 1000.0 / max(q, 1e-9)) /
                             max(dc.usable_roof_m2, 1e-9),
        "ground_mount": ground_mount, "wet": wet,
    }


def coverage_to_noise_curve(fracs=np.linspace(0, 1, 101)) -> list:
    """The asymmetry: water scales linearly with coverage, noise does not."""
    return [{"f": float(f),
             "noise_dB": float(-10.0 * np.log10(max(1.0 - f, 1e-9))),
             "water_pct": float(f * 100.0),
             "loudness_ratio": float(perceived_loudness_ratio(
                 -10.0 * np.log10(max(1.0 - f, 1e-9))))}
            for f in fracs]


# ============================================================================
# REPORT
# ============================================================================
def main() -> None:
    W = 96
    geom, ac = Geometry(), Acoustics()
    env = Environment(5.0, 0.55, 1.5, 0.0, "clear winter night")

    q_dry = array_flux(env, geom, False)
    q_wet = array_flux(env, geom, True)

    print("=" * W)
    print("  SCALE ANALYSIS — CAN THE ROOF ABSORB THE LOAD?")
    print("=" * W)
    print(f"  Array flux at {env.label}: {q_dry:.0f} W/m2 dry, {q_wet:.0f} W/m2 wet")
    print(f"  Rack densities per 2026 survey data: 27 kW average, 50-70 kW AI,")
    print(f"  120-140 kW current-generation GPU racks.")
    print()

    print("-" * W)
    print("  AREA MATCHING (dry mode, roof only)")
    print("-" * W)
    print(f"  {'Facility':32s} {'rack':>6s} {'reject':>9s} {'roof':>9s} "
          f"{'roof flux':>10s} {'array':>8s} {'cover':>7s}")
    print(f"  {'':32s} {'kW':>6s} {'kW':>9s} {'m2':>9s} "
          f"{'W/m2':>10s} {'kW':>8s} {'%':>7s}")
    rows = []
    for dc in archetypes():
        r = analyse(dc, env, geom, ac, wet=False)
        rows.append(r)
        print(f"  {r['label']:32s} {r['rack_kW']:6.0f} {r['demand_kW']:9.0f} "
              f"{r['usable_roof_m2']:9.0f} {r['roof_flux_W_m2']:10.0f} "
              f"{r['array_cap_kW']:8.0f} {r['coverage_pct']:6.1f}%")
    print()

    print("-" * W)
    print("  ACOUSTIC AND WATER OUTCOME")
    print("-" * W)
    print(f"  {'Facility':32s} {'cover':>7s} {'L_p base':>9s} {'L_p new':>8s} "
          f"{'noise':>8s} {'loudness':>9s} {'water':>8s}")
    for r in rows:
        aud = "inaudible" if r["residual_kW"] < 1 else f"{r['Lp_residual']:8.1f}"
        print(f"  {r['label']:32s} {r['coverage_pct']:6.1f}% "
              f"{r['Lp_base']:9.1f} {aud:>8s} "
              f"{r['noise_reduction_dB']:7.1f} dB "
              f"{r['loudness_ratio']:8.2f}x {r['water_saved_pct']:7.0f}%")
    print()

    print("-" * W)
    print("  HOW MUCH ARRAY AREA WOULD FULL COVERAGE NEED?")
    print("-" * W)
    print(f"  {'Facility':32s} {'need m2':>10s} {'have m2':>10s} {'ratio':>10s}")
    for r in rows:
        print(f"  {r['label']:32s} {r['area_needed_for_full_m2']:10.0f} "
              f"{r['usable_roof_m2']:10.0f} {r['area_ratio_needed']:9.1f}x")
    print()

    print("-" * W)
    print("  THE ASYMMETRY: WATER IS LINEAR IN COVERAGE, NOISE IS NOT")
    print("-" * W)
    print(f"  {'coverage':>9s} {'water saved':>12s} {'noise cut':>10s} "
          f"{'perceived loudness':>19s}")
    for c in coverage_to_noise_curve(np.array([0.05, 0.10, 0.25, 0.50,
                                               0.75, 0.90, 0.95, 0.99])):
        print(f"  {c['f']*100:8.0f}% {c['water_pct']:11.0f}% "
              f"{c['noise_dB']:9.1f} dB {c['loudness_ratio']:17.2f}x")
    print()
    print("  A 3 dB cut (the smallest reliably noticeable change) needs 50 % coverage.")
    print("  A 10 dB cut (half as loud) needs 90 % coverage.")
    print()

    print("-" * W)
    print("  GROUND-MOUNT OPTION (25 % of remaining parcel, hyperscale only)")
    print("-" * W)
    print(f"  {'Facility':32s} {'roof only':>11s} {'+ ground':>10s}")
    for dc in archetypes()[:4]:
        a = analyse(dc, env, geom, ac, wet=False, ground_mount=False)
        b = analyse(dc, env, geom, ac, wet=False, ground_mount=True)
        print(f"  {dc.label:32s} {a['coverage_pct']:10.1f}% "
              f"{b['coverage_pct']:9.1f}%")
    print()

    print("=" * W)
    print("  CONCLUSION")
    print("=" * W)
    micro = [r for r in rows if "Micro" in r["label"]]
    mega = [r for r in rows if "Micro" not in r["label"]]
    print(f"  Mega/edge coverage: {min(r['coverage_pct'] for r in mega):.1f}% to "
          f"{max(r['coverage_pct'] for r in mega):.1f}%")
    print(f"    -> acoustic benefit {min(r['noise_reduction_dB'] for r in mega):.1f} "
          f"to {max(r['noise_reduction_dB'] for r in mega):.1f} dB; "
          f"coverage above 50% exceeds the nominal 3 dB threshold")
    print(f"  Micro-scale coverage: {min(r['coverage_pct'] for r in micro):.0f}% to "
          f"{max(r['coverage_pct'] for r in micro):.0f}%")
    print(f"    -> mechanical plant eliminated entirely; site returns to background")
    print()
    print("  The array is a WATER technology at hyperscale and an ACOUSTIC")
    print("  technology only at micro scale, where full coverage is achievable.")
    print("=" * W)


if __name__ == "__main__":
    main()
