#!/usr/bin/env python3
"""
=============================================================================
FULL SYSTEM MODEL — Hybrid PVT Fanless Heat-Rejection Array
Seven candidate heat-rejection mechanisms, individually
switchable, with an ambient-condition controller and design optimisation.
=============================================================================

MECHANISMS MODELLED (all independently toggleable)

  A  Sub-panel plenum + vertical draft stack (buoyancy draught)
                  -> includes optimisation of stack height, filling the
                     -> includes a stack-height sensitivity sweep
  B  Fixed aerodynamic wind-augmentation shrouds
  C  Selective 8-13 um radiative emitter, optionally DEPLOYABLE
                  (stowed by day for collection, deployed by night)
  D  Intermittent evaporative film, meltwater-fed, with
                  freeze-management lockout
  E  Passive two-phase thermosiphon area spreading
  F  Electrohydrodynamic (corona-discharge) ionic wind, with
                  ozone generation and catalytic mitigation
  G  (this work)  Overhead horizontal mist array with gravity water recovery

  H  CONTROLLER apportioning the above on measured ambient air
                  temperature, wind speed, sky clarity and humidity, to meet
                  a target duty while holding acoustic emission below a limit

SKY MODEL
---------
Effective radiative sky temperature is derived from dew point and cloud
fraction (Berdahl-Martin clear-sky emissivity with a cloud correction)
rather than being an assumed constant, so that "sky clarity" is a genuine
controller input.

STATUS
------
Lumped, quasi-steady, design-stage. Textbook correlations cited inline.
Not CFD, not experimentally validated. Absolute fluxes +/-30-50 %.

=============================================================================
"""

from __future__ import annotations

import itertools
import numpy as np
from dataclasses import dataclass, asdict, field, replace

from pvt_rejection_model import (
    SIGMA, G, RHO_AIR, CP_AIR, MU_AIR, K_AIR, NU_AIR, PR_AIR, H_FG, BETA,
    h_natural_convection, h_forced_convection, q_radiative_flux,
    q_evaporative_flux, p_sat_water, stack_air_velocity,
)
from pvt_mist_combined_model import (
    RHO_W, P_ATM, R_V, humidity_ratio, wet_bulb_stull, air_density,
    vapour_diffusivity, droplet_terminal_velocity, spalding_mass_number,
    droplet_evaporated_fraction, fan_sound_power, sum_levels_dB,
    sound_pressure_at, perceived_loudness_ratio,
    dry_cooler, evaporative_cooler,
)

MU_ION = 2.0e-4          # ion mobility in air [m2/(V s)]
O3_MW = 48.0             # ozone molar mass [g/mol]


# ============================================================================
# ENVIRONMENT
# ============================================================================
@dataclass
class Environment:
    """Measured ambient state used by the controller."""
    T_amb_C: float = 5.0
    RH: float = 0.55
    wind_speed: float = 1.5
    cloud_fraction: float = 0.0      # 0 = clear, 1 = overcast  ("sky clarity")
    label: str = "baseline night"

    @property
    def T_dew_C(self) -> float:
        """Dew point from dry bulb and RH (Magnus inversion)."""
        a, b = 17.27, 237.3
        g = (a * self.T_amb_C) / (b + self.T_amb_C) + np.log(max(self.RH, 1e-4))
        return b * g / (a - g)

    @property
    def eps_sky(self) -> float:
        """
        Effective sky emissivity. Berdahl & Martin clear-sky correlation in
        dew point, with a linear cloud correction toward black-body overcast.
        """
        td = np.clip(self.T_dew_C, -30.0, 30.0)
        eps_clear = 0.711 + 0.56 * (td / 100.0) + 0.73 * (td / 100.0) ** 2
        eps_clear = float(np.clip(eps_clear, 0.55, 0.95))
        c = float(np.clip(self.cloud_fraction, 0.0, 1.0))
        return eps_clear + (1.0 - eps_clear) * c * 0.96

    @property
    def T_sky_C(self) -> float:
        """Effective radiative sky temperature [C]."""
        T = self.T_amb_C + 273.15
        return T * self.eps_sky ** 0.25 - 273.15

    @property
    def T_wb_C(self) -> float:
        return wet_bulb_stull(self.T_amb_C, self.RH)


# ============================================================================
# GEOMETRY / HARDWARE
# ============================================================================
@dataclass
class Geometry:
    panel_L: float = 1.7
    panel_W: float = 1.0
    n_panels: int = 100
    tilt_deg: float = 30.0
    plenum_gap: float = 0.10
    plenum_loss_K: float = 3.0

    stack_height: float = 5.0        # sensitivity tested below
    shroud_speedup: float = 1.6

    eps_backsheet: float = 0.90      # ordinary backsheet
    eps_emitter: float = 0.95        # selective 8-13 um emitter
    selective_sky_bonus_C: float = 12.0   # extra sink depression, clear sky

    thermosiphon_util: float = 0.97  # assumed near-isothermal
    baseline_util: float = 0.75      # without thermosiphon

    # EHD electrodes
    ehd_gap: float = 0.020           # electrode-to-collector gap [m]
    ehd_voltage_kV: float = 15.0
    ehd_current_density: float = 1.0e-3   # [A/m2] at full setting
    ehd_momentum_eff: float = 0.35   # fraction of ideal EHD head realised
    ehd_ozone_g_per_kWh: float = 5.0 # positive corona, conservative
    ehd_catalyst_removal: float = 0.98    # MnO2 catalytic converter

    # Mist array [this work]
    mist_rows: int = 2
    mist_nozzles_per_row: int = 25
    mist_flow_Lph: float = 4.5
    mist_d32_um: float = 45.0
    mist_height: float = 1.20
    mist_interception: float = 0.82
    mist_recovery_eff: float = 0.90
    mist_geometric_coverage: float = 0.92
    mist_cycles_conc: float = 6.0

    # Film
    film_wet_fraction: float = 0.70
    film_cycles_conc: float = 8.0    # meltwater-fed: low TDS, high COC

    freeze_lockout_C: float = 2.0    # freeze management: no wetting below this ambient air temperature

    # Overhead spray zone: the mist falls through ambient air above the array.
    # That air is renewed by the ambient wind crossing a layer of depth
    # mist_height over the array's plan width, with a minimum exchange
    # velocity representing buoyant/entrainment renewal on calm nights.
    spray_exchange_min_m_s: float = 0.3
    spray_exchange_factor: float = 1.0   # sensitivity multiplier on the spray-zone air exchange

    # Combination of natural- and forced-convection coefficients:
    # "power" with exponent h_comb_exp (2 = quadrature, reference case), or
    # "velocity_first", which combines forced-flow velocities before
    # computing a single forced coefficient.
    h_comb_mode: str = "power"
    h_comb_exp: float = 2.0

    @property
    def A_panel(self): return self.panel_L * self.panel_W
    @property
    def A_total(self): return self.A_panel * self.n_panels
    @property
    def tilt_rad(self): return np.radians(self.tilt_deg)
    @property
    def n_nozzles(self): return self.mist_rows * self.mist_nozzles_per_row
    @property
    def mist_flow_kg_s(self):
        return self.n_nozzles * self.mist_flow_Lph / 3600.0 * (RHO_W / 1000.0)


# ============================================================================
# MECHANISM STATE — what the controller switches
# ============================================================================
@dataclass
class MechanismState:
    plenum_stack: bool = True     # A
    shrouds: bool = True          # B
    emitter_deployed: bool = True # C
    film: bool = False            # D
    thermosiphon: bool = True     # E
    ehd: bool = False             # F
    mist: bool = False            # G

    film_duty: float = 0.30
    mist_duty: float = 0.35
    ehd_setting: float = 1.0      # 0-1 fraction of full current density

    def label(self) -> str:
        on = []
        if self.plenum_stack: on.append("stack")
        if self.shrouds: on.append("shroud")
        if self.emitter_deployed: on.append("emitter")
        if self.thermosiphon: on.append("thermo")
        if self.film: on.append(f"film{self.film_duty:.2f}")
        if self.ehd: on.append(f"ehd{self.ehd_setting:.2f}")
        if self.mist: on.append(f"mist{self.mist_duty:.2f}")
        return "+".join(on) if on else "bare"


# ============================================================================
# MECHANISM F — ELECTROHYDRODYNAMIC IONIC WIND
# ============================================================================
def ehd_ionic_wind(geom: Geometry, setting: float, T_amb_C: float) -> dict:
    """
    Corona-discharge ionic wind.

    Space-charge drift in the inter-electrode field exerts a body force on
    the neutral gas. Integrating over the gap gives a static head

        dp = J d / mu_ion

    from which an ideal velocity follows; a momentum-recovery efficiency
    accounts for the fraction actually converted to useful surface flow.
    Electrical power is V I. Ozone is generated at a rate proportional to
    discharge power and is reduced by an assumed catalytic converter.
    """
    if setting <= 0.0:
        return {"v": 0.0, "dp_Pa": 0.0, "P_elec_W_m2": 0.0,
                "O3_gen_g_h_m2": 0.0, "O3_out_g_h_m2": 0.0}

    J = geom.ehd_current_density * float(np.clip(setting, 0.0, 1.0))  # [A/m2]
    dp = J * geom.ehd_gap / MU_ION                                    # [Pa]
    rho = air_density(T_amb_C)
    v_ideal = np.sqrt(2.0 * dp / rho)
    v = v_ideal * geom.ehd_momentum_eff

    P_elec = geom.ehd_voltage_kV * 1000.0 * J                          # [W/m2]
    O3_gen = geom.ehd_ozone_g_per_kWh * P_elec / 1000.0                # [g/h/m2]
    O3_out = O3_gen * (1.0 - geom.ehd_catalyst_removal)

    return {"v": v, "dp_Pa": dp, "P_elec_W_m2": P_elec,
            "O3_gen_g_h_m2": O3_gen, "O3_out_g_h_m2": O3_out}


# ============================================================================
# CORE EVALUATION — all mechanisms, one operating point
# ============================================================================
def evaluate(env: Environment, geom: Geometry, st: MechanismState,
             T_panel_C: float = 35.0) -> dict:
    """
    Full surface energy balance with every enabled mechanism.
    Returns fluxes, resource consumption, acoustics and diagnostics.
    """
    A = geom.A_total
    dT_dry = T_panel_C - env.T_amb_C

    # ---- freeze management -----------------------------------------------
    frozen = env.T_amb_C < geom.freeze_lockout_C
    use_film = st.film and not frozen
    use_mist = st.mist and not frozen

    # ---- E: thermosiphon area utilisation --------------------------------
    util = geom.thermosiphon_util if st.thermosiphon else geom.baseline_util

    # ---- A: buoyancy plenum + draft stack ---------------------------------
    if st.plenum_stack:
        v_stack = stack_air_velocity(geom.stack_height, dT_dry,
                                     env.T_amb_C, geom.plenum_loss_K)
        h_plenum = h_forced_convection(v_stack, geom.panel_L)
    else:
        v_stack, h_plenum = 0.0, 0.0

    # ---- B: aerodynamic shrouds -------------------------------------------
    if st.shrouds:
        v_shroud = env.wind_speed * geom.shroud_speedup
    else:
        v_shroud = env.wind_speed
    h_shroud = h_forced_convection(v_shroud, geom.panel_L)

    # ---- F: EHD ionic wind ------------------------------------------------
    ehd = ehd_ionic_wind(geom, st.ehd_setting if st.ehd else 0.0, env.T_amb_C)
    h_ehd = h_forced_convection(ehd["v"], geom.panel_L) if ehd["v"] > 0 else 0.0

    # ---- G: mist -> air pre-cooling + film supply -------------------------
    m_mist_on = geom.mist_flow_kg_s if use_mist else 0.0
    drop = droplet_evaporated_fraction(geom.mist_d32_um, geom.mist_height,
                                       env.T_amb_C, env.RH)
    f_kinetic = drop["f_evap"] if use_mist else 0.0

    # Spray-zone air mass flow. The nozzles are above the array, so droplets
    # fall through ambient air in a layer of depth mist_height over the
    # array's plan width. Isolated-droplet kinetics provide an upper bound on
    # evaporation; the whole spray is additionally capped by the moisture
    # capacity of the air renewed through that layer at its adiabatic
    # saturation state.
    plan_width = np.sqrt(geom.A_total * np.cos(geom.tilt_rad))
    v_exchange = max(env.wind_speed, geom.spray_exchange_min_m_s) * geom.spray_exchange_factor
    m_air = air_density(env.T_amb_C) * v_exchange * geom.mist_height * plan_width
    w_in = humidity_ratio(env.T_amb_C, env.RH)
    w_sat_wb = humidity_ratio(env.T_wb_C, 1.0)
    m_psych_cap_on = max(m_air * (w_sat_wb - w_in), 0.0)
    m_kinetic_on = m_mist_on * f_kinetic
    m_flight_on = min(m_kinetic_on, m_psych_cap_on) if use_mist else 0.0
    f_flight = m_flight_on / max(m_mist_on, 1e-12) if use_mist else 0.0
    m_to_surface_on = m_mist_on * (1.0 - f_flight) * geom.mist_interception
    m_carryover_on = m_mist_on * (1.0 - f_flight) * (1.0 - geom.mist_interception)

    # The model reports cycle-averaged capacity and resources. Therefore the
    # pre-cooling effect of intermittent mist is duty-averaged consistently
    # with the water and acoustic calculations.
    dT_max = max(env.T_amb_C - env.T_wb_C, 0.0)
    dT_pre_on = min(m_flight_on * H_FG / max(m_air * CP_AIR, 1e-9), dT_max)
    mist_duty = st.mist_duty if use_mist else 0.0
    dT_pre = dT_pre_on * mist_duty
    T_air_eff = env.T_amb_C - dT_pre
    m_flight_avg = m_flight_on * mist_duty
    w_out = w_in + m_flight_avg / max(m_air, 1e-9)
    pv_out = w_out * P_ATM / (0.622 + w_out)
    RH_eff = float(np.clip(pv_out / p_sat_water(T_air_eff), 0.0, 1.0))

    dT_eff = T_panel_C - T_air_eff

    # ---- convective total --------------------------------------------------
    h_nc = h_natural_convection(T_panel_C, T_air_eff, geom.panel_L, geom.tilt_rad)
    if geom.h_comb_mode == "velocity_first":
        v_forced = np.sqrt(v_stack ** 2 + v_shroud ** 2 + ehd["v"] ** 2)
        h_forced = h_forced_convection(v_forced, geom.panel_L) if v_forced > 0 else 0.0
        h_conv = np.sqrt(h_nc ** 2 + h_forced ** 2)
    else:
        n = geom.h_comb_exp
        h_conv = (h_nc ** n + h_plenum ** n + h_shroud ** n + h_ehd ** n) ** (1.0 / n)
    q_conv = h_conv * dT_eff

    # ---- C: radiative emitter ---------------------------------------------
    if st.emitter_deployed:
        eps = geom.eps_emitter
        # the selective advantage is real only when the window is open:
        # it scales with sky clarity
        bonus = geom.selective_sky_bonus_C * (1.0 - env.cloud_fraction)
    else:
        eps = geom.eps_backsheet
        bonus = 0.0
    T_sky_eff = env.T_sky_C - bonus
    q_rad = q_radiative_flux(T_panel_C, T_sky_eff, eps)

    # ---- D + G: evaporative surface --------------------------------------
    q_evap_full = q_evaporative_flux(T_panel_C, T_air_eff, RH_eff, h_conv)
    m_demand_full = q_evap_full * A / H_FG

    wet_from_mist = 0.0
    if use_mist and m_demand_full > 0:
        wet_from_mist = min(m_to_surface_on / m_demand_full,
                            geom.mist_geometric_coverage)
    wet_from_film = geom.film_wet_fraction if use_film else 0.0

    # mist and film wet the same surface; take the greater coverage and the
    # greater duty rather than double-counting
    wet_fraction = min(max(wet_from_mist, wet_from_film),
                       geom.mist_geometric_coverage)
    duty = max(st.mist_duty if use_mist else 0.0,
               st.film_duty if use_film else 0.0)
    q_evap = q_evap_full * wet_fraction * duty

    # ---- totals ------------------------------------------------------------
    q_total = (q_conv + q_rad + q_evap) * util
    P_total = q_total * A

    # ---- water balance -----------------------------------------------------
    m_surface_evap = q_evap * A / H_FG
    m_carry_avg = m_carryover_on * mist_duty
    m_surf_avg = m_to_surface_on * mist_duty
    m_runoff = max(m_surf_avg - m_surface_evap, 0.0)
    m_recovered = m_runoff * geom.mist_recovery_eff
    m_runoff_lost = m_runoff - m_recovered

    coc = geom.mist_cycles_conc if use_mist else geom.film_cycles_conc
    m_evap_total = m_flight_avg + m_surface_evap
    m_blowdown = m_evap_total / max(coc - 1.0, 1e-6)
    m_makeup = m_evap_total + m_carry_avg + m_runoff_lost + m_blowdown

    # ---- electrical --------------------------------------------------------
    P_ehd_W = ehd["P_elec_W_m2"] * A
    O3_out_g_h = ehd["O3_out_g_h_m2"] * A

    return {
        "state": st.label(), "frozen": frozen,
        "q_conv": q_conv * util, "q_rad": q_rad * util, "q_evap": q_evap * util,
        "q_flux": q_total, "P_total": P_total, "P_kW": P_total / 1000.0,
        "h_nc": h_nc, "h_plenum": h_plenum, "h_shroud": h_shroud,
        "h_ehd": h_ehd, "h_conv": h_conv,
        "v_stack": v_stack, "v_shroud": v_shroud, "v_ehd": ehd["v"],
        "T_sky": env.T_sky_C, "eps_sky": env.eps_sky, "T_sky_eff": T_sky_eff,
        "T_air_eff": T_air_eff, "RH_eff": RH_eff, "dT_precool": dT_pre,
        "dT_eff": dT_eff, "m_air": m_air,
        "wet_fraction": wet_fraction, "duty": duty, "f_flight": f_flight,
        "f_flight_kinetic": f_kinetic, "m_psych_cap": m_psych_cap_on,
        "dT_precool_on": dT_pre_on,
        "m_makeup": m_makeup, "m_evap_total": m_evap_total,
        "m_recovered": m_recovered,
        "P_ehd_W": P_ehd_W, "O3_out_g_h": O3_out_g_h,
        "ehd_dp_Pa": ehd["dp_Pa"],
        "util": util,
    }


# ============================================================================
# ACOUSTICS OF THE APPARATUS (all silent sources)
# ============================================================================
@dataclass
class Acoustics:
    K_W_axial: float = 40.0
    dry_air_rise_K: float = 10.0
    dry_dp_Pa: float = 150.0
    evap_enthalpy_rise: float = 30.0
    evap_dp_Pa: float = 200.0
    evap_water_LWA: float = 84.0
    nozzle_Lp_1m_dBA: float = 48.0
    buoyancy_LWA: float = 25.0
    ehd_LWA: float = 30.0            # corona hiss, essentially inaudible
    fan_a_weight: float = -4.0
    receiver_m: float = 10.0
    background_dBA: float = 35.0
    limit_dBA: float = 45.0          # apparatus-only controller constraint at receiver


def apparatus_sound(geom: Geometry, st: MechanismState, ac: Acoustics,
                    duty_averaged: bool = True) -> float:
    """A-weighted sound power of the apparatus [dB(A) re 1 pW]."""
    parts = [ac.buoyancy_LWA]
    if st.mist:
        LW1 = ac.nozzle_Lp_1m_dBA + 10.0 * np.log10(4.0 * np.pi)
        LW = LW1 + 10.0 * np.log10(max(geom.n_nozzles, 1))
        if duty_averaged:
            LW += 10.0 * np.log10(max(st.mist_duty, 1e-6))
        parts.append(LW)
    if st.ehd:
        parts.append(ac.ehd_LWA)
    return sum_levels_dB(parts)


def apparatus_Lp(geom, st, ac, duty_averaged=True) -> float:
    """
    Sound pressure at the receiver from the apparatus alone.
    Returned unfloored so the controller can rank options; use
    apparatus_Lp_total() for the level an observer would actually measure.
    """
    return sound_pressure_at(apparatus_sound(geom, st, ac, duty_averaged),
                             ac.receiver_m, True)


def apparatus_Lp_total(geom, st, ac, duty_averaged=True) -> dict:
    """
    Level an observer measures: apparatus energetically summed with the
    ambient night background. When the apparatus falls below background the
    total is background-dominated and the apparatus is effectively inaudible.
    """
    src = apparatus_Lp(geom, st, ac, duty_averaged)
    total = sum_levels_dB([src, ac.background_dBA])
    return {"source_dBA": src, "total_dBA": total,
            "excess_over_background_dB": total - ac.background_dBA,
            "inaudible": src < ac.background_dBA - 10.0}


# ============================================================================
# H — CONTROLLER
# ============================================================================
@dataclass
class Resources:
    """Weights the controller uses to rank feasible mechanism combinations."""
    water_cost_per_m3: float = 2.50      # currency per m3 make-up
    elec_cost_per_kWh: float = 0.18      # currency per kWh
    hours: float = 12.0


def controller(env: Environment, geom: Geometry, ac: Acoustics,
               Q_target_kW: float, res: Resources = Resources(),
               T_panel_C: float = 35.0,
               allow_ehd: bool = True) -> dict:
    """
    Select the mechanism combination that meets the target duty at least
    resource cost while holding the apparatus-only A-weighted sound pressure
    at the receiver below the configured limit. Ambient background is added
    afterward for reporting the total level heard by an observer.

    Controller inputs are ambient air temperature, wind speed, sky clarity
    and relative humidity.

    The passive mechanisms (plenum/stack, shrouds, emitter, thermosiphon)
    cost nothing to run and are always enabled. The search is therefore over
    the three consumptive mechanisms: film duty, mist duty and EHD setting.
    """
    film_opts = [0.0, 0.15, 0.30, 0.50, 0.75, 1.0]
    mist_opts = [0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
    ehd_opts = [0.0, 0.25, 0.50, 1.0] if allow_ehd else [0.0]

    feasible, all_rows = [], []
    for fd, md, es in itertools.product(film_opts, mist_opts, ehd_opts):
        st = MechanismState(
            plenum_stack=True, shrouds=True, emitter_deployed=True,
            thermosiphon=True,
            film=fd > 0, mist=md > 0, ehd=es > 0,
            film_duty=fd, mist_duty=md, ehd_setting=es)
        r = evaluate(env, geom, st, T_panel_C)
        Lp = apparatus_Lp(geom, st, ac, duty_averaged=True)
        Ltot = apparatus_Lp_total(geom, st, ac, duty_averaged=True)

        water_m3 = r["m_makeup"] * res.hours * 3600.0 / 1000.0
        elec_kWh = r["P_ehd_W"] / 1000.0 * res.hours
        cost = water_m3 * res.water_cost_per_m3 + elec_kWh * res.elec_cost_per_kWh

        row = {"film_duty": fd, "mist_duty": md, "ehd": es,
               "P_kW": r["P_kW"], "q_flux": r["q_flux"],
               "water_m3": water_m3, "elec_kWh": elec_kWh,
               "O3_g_h": r["O3_out_g_h"], "Lp_dBA": Lp,
               "Lp_total_dBA": Ltot["total_dBA"],
               "excess_dB": Ltot["excess_over_background_dB"],
               "inaudible": Ltot["inaudible"], "cost": cost,
               "frozen": r["frozen"], "state": st, "result": r}
        all_rows.append(row)
        if r["P_kW"] >= Q_target_kW and Lp <= ac.limit_dBA:
            feasible.append(row)

    if feasible:
        best = min(feasible, key=lambda x: x["cost"])
        status = "target met"
    else:
        # cannot meet target: choose the highest-capacity acoustically legal
        legal = [r for r in all_rows if r["Lp_dBA"] <= ac.limit_dBA]
        best = max(legal or all_rows, key=lambda x: x["P_kW"])
        status = "target NOT met - capacity limited"

    return {"best": best, "status": status, "n_feasible": len(feasible),
            "all": all_rows, "env": env, "Q_target_kW": Q_target_kW}


# ============================================================================
# DESIGN SENSITIVITY — stack-height bracket
# ============================================================================
def optimise_stack_height(env: Environment, geom: Geometry,
                          heights=np.arange(1.0, 25.5, 0.5),
                          T_panel_C: float = 35.0) -> list:
    """
    Sweep draft-stack height; stack velocity varies approximately as the
    square root of stack height times temperature difference. The sweep is
    extended beyond the expected 20% marginal-return point so the bracket is
    determined by the model rather than by the end of the tested range.
    """
    st = MechanismState(plenum_stack=True, shrouds=True, emitter_deployed=True,
                        thermosiphon=True, film=False, mist=False, ehd=False)
    out = []
    for H in heights:
        g = replace(geom, stack_height=float(H))
        r = evaluate(env, g, st, T_panel_C)
        out.append({"H": float(H), "v_stack": r["v_stack"],
                    "h_plenum": r["h_plenum"], "q_flux": r["q_flux"],
                    "P_kW": r["P_kW"]})
    # marginal gain per additional metre, normalised to the first metre
    for i in range(1, len(out)):
        out[i]["dP_per_m"] = (out[i]["P_kW"] - out[i-1]["P_kW"]) / \
                             (out[i]["H"] - out[i-1]["H"])
    out[0]["dP_per_m"] = np.nan
    base = out[1]["dP_per_m"]
    for r in out:
        r["marginal_frac"] = (r["dP_per_m"] / base) if np.isfinite(r.get("dP_per_m", np.nan)) else np.nan
    return out


def stack_height_bracket(sweep: list, lo_frac=0.50, hi_frac=0.20) -> tuple:
    """
    Identify the practical height bracket: the range over which marginal
    return per metre falls from `lo_frac` to `hi_frac` of its initial value.
    """
    Xs = [r["H"] for r in sweep if np.isfinite(r.get("marginal_frac", np.nan))
          and r["marginal_frac"] <= lo_frac]
    Ys = [r["H"] for r in sweep if np.isfinite(r.get("marginal_frac", np.nan))
          and r["marginal_frac"] <= hi_frac]
    X = Xs[0] if Xs else sweep[1]["H"]
    Y = Ys[0] if Ys else sweep[-1]["H"]
    return X, Y


def optimise_shroud(env: Environment, geom: Geometry,
                    speedups=np.arange(1.0, 2.6, 0.1),
                    winds=(0.0, 0.5, 1.5, 3.0, 6.0),
                    T_panel_C: float = 35.0) -> list:
    """
    Shroud speed-up sweep against wind speed, to test whether
    buoyancy predominates when calm and wind augmentation when breezy.
    """
    st = MechanismState(plenum_stack=True, shrouds=True, emitter_deployed=True,
                        thermosiphon=True)
    out = []
    for w in winds:
        e = replace(env, wind_speed=float(w))
        for s in speedups:
            g = replace(geom, shroud_speedup=float(s))
            r = evaluate(e, g, st, T_panel_C)
            out.append({"wind": float(w), "speedup": float(s),
                        "q_flux": r["q_flux"],
                        "h_plenum": r["h_plenum"], "h_shroud": r["h_shroud"],
                        "buoyancy_share": r["h_plenum"] ** 2 /
                                          max(r["h_conv"] ** 2, 1e-9)})
    return out


def optimise_ehd(env: Environment, geom: Geometry, ac: Acoustics,
                 settings=np.linspace(0, 1, 11),
                 T_panel_C: float = 35.0) -> list:
    """
    EHD ionic-wind sweep: capacity gain against electrical cost and
    ozone production, with a comparison to the fan power that would be
    needed for the same air movement.
    """
    out = []
    # Compare EHD at the same operating point used in the mechanism ladder:
    # all other mechanisms active, with film and mist at their design duties.
    base_state = MechanismState(film=True, mist=True, ehd=False,
                                film_duty=0.30, mist_duty=0.35)
    base = evaluate(env, geom, base_state, T_panel_C)
    for s in settings:
        st = MechanismState(film=True, mist=True, ehd=s > 0,
                            film_duty=0.30, mist_duty=0.35,
                            ehd_setting=float(s))
        r = evaluate(env, geom, st, T_panel_C)
        # Equivalent fan comparison uses the same distributed flow area as the
        # EHD electrodes, so both powers are evaluated on the full array area.
        A_face = geom.A_total
        Q_vol = r["v_ehd"] * A_face
        P_fan = Q_vol * max(r["ehd_dp_Pa"], 1e-9) / 0.55
        out.append({"setting": float(s), "v_ehd": r["v_ehd"],
                    "dp_Pa": r["ehd_dp_Pa"],
                    "q_flux": r["q_flux"],
                    "gain": r["q_flux"] / base["q_flux"],
                    "P_ehd_kW": r["P_ehd_W"] / 1000.0,
                    "P_fan_equiv_kW": P_fan / 1000.0,
                    "O3_g_h": r["O3_out_g_h"],
                    "COP_thermal": (r["P_total"] - base["P_total"]) /
                                   max(r["P_ehd_W"], 1e-9)})
    return out


def emitter_study(env_list: list, geom: Geometry,
                  T_panel_C: float = 35.0) -> list:
    """
    Deployable emitter: compare deployed against stowed across sky
    conditions, isolating the radiative contribution.
    """
    out = []
    for e in env_list:
        dep = evaluate(e, geom, MechanismState(emitter_deployed=True), T_panel_C)
        stw = evaluate(e, geom, MechanismState(emitter_deployed=False), T_panel_C)
        out.append({"label": e.label, "cloud": e.cloud_fraction,
                    "T_amb": e.T_amb_C, "RH": e.RH,
                    "eps_sky": e.eps_sky, "T_sky": e.T_sky_C,
                    "q_rad_deployed": dep["q_rad"], "q_rad_stowed": stw["q_rad"],
                    "q_tot_deployed": dep["q_flux"], "q_tot_stowed": stw["q_flux"],
                    "gain": dep["q_flux"] / stw["q_flux"],
                    "rad_share": dep["q_rad"] / dep["q_flux"]})
    return out


# ============================================================================
# MECHANISM ATTRIBUTION — one mechanism at a time
# ============================================================================
def mechanism_ladder(env: Environment, geom: Geometry,
                     T_panel_C: float = 35.0) -> list:
    """Cumulative contribution of each mechanism, from a bare tilted panel."""
    steps = [
        ("Bare tilted panel (no mechanisms)",
         MechanismState(False, False, False, False, False, False, False)),
        ("+ Selective emitter",
         MechanismState(False, False, True, False, False, False, False)),
        ("+ Thermosiphon spreading",
         MechanismState(False, False, True, False, True, False, False)),
        ("+ Plenum & draft stack",
         MechanismState(True, False, True, False, True, False, False)),
        ("+ Aerodynamic shrouds",
         MechanismState(True, True, True, False, True, False, False)),
        ("+ Evaporative film",
         MechanismState(True, True, True, True, True, False, False)),
        ("+ Overhead mist array",
         MechanismState(True, True, True, True, True, False, True)),
        ("+ EHD ionic wind",
         MechanismState(True, True, True, True, True, True, True)),
    ]
    rows, prev = [], None
    for name, st in steps:
        r = evaluate(env, geom, st, T_panel_C)
        d = 0.0 if prev is None else r["P_kW"] - prev
        rows.append({"name": name, "P_kW": r["P_kW"], "delta_kW": d,
                     "q_flux": r["q_flux"], "water_kg_s": r["m_makeup"],
                     "P_ehd_kW": r["P_ehd_W"] / 1000.0})
        prev = r["P_kW"]
    return rows



# ============================================================================
# REPRODUCIBILITY SWEEPS USED BY TABLE 2 / FIGURES 8 AND 11
# ============================================================================
def nozzle_row_sweep(env: Environment, geom: Geometry,
                     rows=(0, 1, 2, 3, 4, 6, 8),
                     T_panel_C: float = 35.0) -> list:
    """Mist-only, continuous-operation sweep used for the nozzle-row table."""
    out=[]
    for n in rows:
        g=replace(geom, mist_rows=int(max(n,1)))
        if n == 0:
            st=MechanismState(True, True, True, False, True, False, False)
        else:
            st=MechanismState(True, True, True, False, True, False, True,
                              mist_duty=1.0)
        r=evaluate(env,g,st,T_panel_C)
        out.append({"rows":int(n),"P_kW":r["P_kW"],
                    "wet_fraction":0.0 if n==0 else r["wet_fraction"],
                    "m_makeup":0.0 if n==0 else r["m_makeup"]})
    return out


def water_tradeoff_sweep(env: Environment, geom: Geometry,
                         duties=np.linspace(0,1,21),
                         T_panel_C: float = 35.0) -> list:
    """Common film/mist duty sweep used for the capacity-water trade-off."""
    dry=evaluate(env,geom,MechanismState(True,True,True,False,True,False,False),T_panel_C)
    # Reference evaporative cooler make-up for the same instantaneous heat duty.
    class _C: T_amb_C=env.T_amb_C
    class _A:
        K_W_axial=40.0; dry_air_rise_K=10.0; dry_static_dp_Pa=150.0
        evap_air_enthalpy_rise=30.0; evap_static_dp_Pa=200.0; evap_water_noise_LWA=84.0
    out=[]
    for duty in duties:
        st=MechanismState(True,True,True,duty>0,True,False,duty>0,
                          film_duty=float(duty),mist_duty=float(duty))
        r=evaluate(env,geom,st,T_panel_C)
        ref=evaporative_cooler(r["P_kW"],_C(),_A())["water_kg_s"]
        saving=(1-r["m_makeup"]/max(ref,1e-12))*100.0
        out.append({"duty":float(duty),"P_kW":r["P_kW"],
                    "gain":r["P_kW"]/dry["P_kW"],
                    "water_saving_pct":saving,"m_makeup":r["m_makeup"]})
    return out


def passive_gain_sensitivity(env: Environment, geom: Geometry,
                             T_panel_C: float = 35.0) -> dict:
    """Direct sensitivity of thermosiphon utilisation and emitter bonus."""
    thermo=[]
    for u in (0.80,0.85,0.90,0.95,0.97):
        g=replace(geom,thermosiphon_util=float(u))
        a=evaluate(env,g,MechanismState(False,False,True,False,False,False,False),T_panel_C)
        b=evaluate(env,g,MechanismState(False,False,True,False,True,False,False),T_panel_C)
        thermo.append({"util":float(u),"delta_kW":b["P_kW"]-a["P_kW"]})
    emitter=[]
    for bonus in (0,3,6,9,12):
        g=replace(geom,selective_sky_bonus_C=float(bonus))
        a=evaluate(env,g,MechanismState(False,False,False,False,False,False,False),T_panel_C)
        b=evaluate(env,g,MechanismState(False,False,True,False,False,False,False),T_panel_C)
        emitter.append({"bonus_C":float(bonus),"delta_kW":b["P_kW"]-a["P_kW"]})
    return {"thermosiphon":thermo,"emitter":emitter}


def model_form_sensitivity(env: Environment, geom: Geometry,
                           T_panel_C: float = 35.0) -> list:
    """Alternative convective-combination rules and spray-zone air exchange."""
    cases = [("Reference: quadrature (n = 2)", {}),
             ("Power-law combination, n = 3", {"h_comb_exp": 3.0}),
             ("Velocities combined first", {"h_comb_mode": "velocity_first"}),
             ("Spray-zone air exchange x0.5", {"spray_exchange_factor": 0.5}),
             ("Spray-zone air exchange x2", {"spray_exchange_factor": 2.0})]
    out = []
    for label, kw in cases:
        g = replace(geom, **kw)
        lad = mechanism_ladder(env, g, T_panel_C)
        bare, passive = lad[0]["P_kW"], lad[4]["P_kW"]
        mist_inc = lad[6]["P_kW"] - lad[5]["P_kW"]
        total = lad[-1]["P_kW"]
        r = evaluate(env, g, MechanismState(film=True, mist=True, film_duty=0.30,
                                             mist_duty=0.35), T_panel_C)
        met = sum(1 for e in standard_environments()
                  if not controller(e, g, Acoustics(), 100.0)["status"].startswith("target NOT"))
        out.append({"label": label, "bare_kW": bare, "passive_kW": passive,
                    "mist_kW": mist_inc, "total_kW": total,
                    "f_flight": r["f_flight"], "nights_met": met})
    return out

# ============================================================================
# ANNUAL / SEASONAL OPERATING ENVELOPE
# ============================================================================
def standard_environments() -> list:
    return [
        Environment(30.0, 0.60, 1.0, 0.0, "Summer night, clear"),
        Environment(22.0, 0.80, 1.5, 0.3, "Warm humid night"),
        Environment(22.0, 0.35, 2.0, 0.0, "Warm dry night, clear"),
        Environment(12.0, 0.65, 1.5, 0.0, "Cool autumn night, clear"),
        Environment(12.0, 0.85, 1.5, 1.0, "Cool autumn night, overcast"),
        Environment(5.0, 0.55, 1.5, 0.0, "Mild winter night, clear"),
        Environment(5.0, 0.90, 4.0, 1.0, "Mild winter night, overcast windy"),
        Environment(-2.0, 0.45, 1.5, 0.0, "Cold winter night, clear"),
        Environment(-10.0, 0.40, 2.0, 0.0, "Hard frost, clear"),
        Environment(-20.0, 0.35, 3.0, 0.2, "Deep cold night"),
    ]


def annual_envelope(geom: Geometry, ac: Acoustics, Q_target_kW: float = 100.0,
                    res: Resources = Resources()) -> list:
    """Run the controller across the standard environment set."""
    rows = []
    for e in standard_environments():
        c = controller(e, geom, ac, Q_target_kW, res)
        b = c["best"]
        rows.append({
            "label": e.label, "T_amb": e.T_amb_C, "RH": e.RH,
            "wind": e.wind_speed, "cloud": e.cloud_fraction,
            "T_sky": e.T_sky_C, "T_wb": e.T_wb_C,
            "film_duty": b["film_duty"], "mist_duty": b["mist_duty"],
            "ehd": b["ehd"], "P_kW": b["P_kW"], "q_flux": b["q_flux"],
            "water_m3": b["water_m3"], "elec_kWh": b["elec_kWh"],
            "Lp_dBA": b["Lp_dBA"], "Lp_total_dBA": b["Lp_total_dBA"],
            "excess_dB": b["excess_dB"], "inaudible": b["inaudible"],
            "cost": b["cost"], "capacity_limited": "NOT met" in c["status"],
            "frozen": b["frozen"], "status": c["status"],
        })
    return rows


# ============================================================================
# REPORT
# ============================================================================
def main() -> None:
    W = 80
    geom, ac = Geometry(), Acoustics()
    env = Environment()

    print("=" * W)
    print("  FULL SYSTEM MODEL — ALL MECHANISMS")
    print("=" * W)
    print(f"  Array {geom.n_panels} x {geom.A_panel:.2f} m2 = {geom.A_total:.0f} m2, "
          f"tilt {geom.tilt_deg:.0f} deg")
    print(f"  Baseline env: {env.label} — air {env.T_amb_C:.0f} C, RH {env.RH*100:.0f} %, "
          f"wind {env.wind_speed:.1f} m/s, cloud {env.cloud_fraction:.1f}")
    print(f"  Derived: dew point {env.T_dew_C:.1f} C, sky emissivity "
          f"{env.eps_sky:.3f}, T_sky {env.T_sky_C:.1f} C, T_wb {env.T_wb_C:.1f} C")
    print()

    print("-" * W)
    print("  MECHANISM LADDER (cumulative, 170 m2 array)")
    print("-" * W)
    print(f"    {'Mechanism':38s} {'kW':>8s} {'delta':>8s} {'W/m2':>8s} "
          f"{'water':>8s} {'elec kW':>8s}")
    for r in mechanism_ladder(env, geom):
        print(f"    {r['name']:38s} {r['P_kW']:8.1f} {r['delta_kW']:+8.1f} "
              f"{r['q_flux']:8.0f} {r['water_kg_s']:8.4f} {r['P_ehd_kW']:8.2f}")
    print()

    print("-" * W)
    print("  A. DRAFT-STACK HEIGHT SENSITIVITY")
    print("-" * W)
    sw = optimise_stack_height(env, geom)
    print(f"    {'H (m)':>7s} {'v (m/s)':>9s} {'h_plenum':>9s} {'W/m2':>8s} "
          f"{'kW':>8s} {'marginal':>9s}")
    for r in sw[::3]:
        mf = r.get("marginal_frac", np.nan)
        print(f"    {r['H']:7.1f} {r['v_stack']:9.2f} {r['h_plenum']:9.2f} "
              f"{r['q_flux']:8.0f} {r['P_kW']:8.1f} "
              f"{'—' if not np.isfinite(mf) else f'{mf*100:7.0f}%'}")
    X, Y = stack_height_bracket(sw)
    print()
    print(f"    >> Practical stack-height bracket: about {X:.1f} m to {Y:.1f} m")
    print(f"       (marginal return per metre falls to 50 % at {X:.1f} m "
          f"and 20 % at {Y:.1f} m)")
    print()

    print("-" * W)
    print("  B. AERODYNAMIC SHROUDS — buoyancy vs wind dominance")
    print("-" * W)
    sh = optimise_shroud(env, geom)
    print(f"    {'wind':>6s} {'speedup':>8s} {'W/m2':>8s} {'buoyancy share':>15s}")
    for w in (0.0, 1.5, 3.0, 6.0):
        for s in (1.0, 1.6, 2.2):
            m = [r for r in sh if abs(r["wind"] - w) < 1e-6
                 and abs(r["speedup"] - s) < 1e-6][0]
            print(f"    {w:6.1f} {s:8.1f} {m['q_flux']:8.0f} "
                  f"{m['buoyancy_share']*100:14.0f}%")
    print("    (Buoyancy share falls as wind rises, confirming that")
    print("     the two mechanisms cover complementary regimes.)")
    print()

    print("-" * W)
    print("  C. DEPLOYABLE RADIATIVE EMITTER")
    print("-" * W)
    print(f"    {'Condition':30s} {'cloud':>6s} {'eps_sky':>8s} {'T_sky':>7s} "
          f"{'rad dep':>8s} {'rad stow':>9s} {'gain':>7s}")
    for r in emitter_study(standard_environments(), geom):
        print(f"    {r['label']:30s} {r['cloud']:6.1f} {r['eps_sky']:8.3f} "
              f"{r['T_sky']:7.1f} {r['q_rad_deployed']:8.0f} "
              f"{r['q_rad_stowed']:9.0f} {r['gain']:6.2f}x")
    print()

    print("-" * W)
    print("  F. EHD IONIC WIND — capacity, power, ozone")
    print("-" * W)
    print(f"    {'set':>5s} {'v (m/s)':>8s} {'dp (Pa)':>8s} {'gain':>7s} "
          f"{'P_ehd kW':>9s} {'fan kW':>8s} {'O3 g/h':>8s} {'COP':>7s}")
    for r in optimise_ehd(env, geom, ac):
        print(f"    {r['setting']:5.2f} {r['v_ehd']:8.3f} {r['dp_Pa']:8.3f} "
              f"{r['gain']:6.3f}x {r['P_ehd_kW']:9.2f} "
              f"{r['P_fan_equiv_kW']:8.4f} {r['O3_g_h']:8.3f} "
              f"{r['COP_thermal']:7.2f}")
    print()

    print("-" * W)
    print("  G1. NOZZLE-ROW SWEEP (mist only, continuous operation)")
    print("-" * W)
    print(f"    {'rows':>5s} {'kW':>8s} {'wet %':>8s} {'make-up kg/s':>13s}")
    for r in nozzle_row_sweep(env, geom):
        print(f"    {r['rows']:5d} {r['P_kW']:8.1f} {r['wet_fraction']*100:8.1f} {r['m_makeup']:13.4f}")
    print()

    print("-" * W)
    print("  G2. CAPACITY-WATER TRADE-OFF (common film/mist duty)")
    print("-" * W)
    for r in water_tradeoff_sweep(env, geom):
        print(f"    duty={r['duty']:.2f}  P={r['P_kW']:.1f} kW  gain={r['gain']:.3f}x  water saving={r['water_saving_pct']:.1f}%")
    print()

    print("-" * W)
    print("  SENSITIVITY OF ASSUMED PASSIVE GAINS")
    print("-" * W)
    ps=passive_gain_sensitivity(env, geom)
    print("    thermosiphon utilisation -> incremental kW")
    for r in ps['thermosiphon']:
        print(f"      {r['util']:.2f} -> {r['delta_kW']:.2f}")
    print("    selective-emitter sky bonus -> incremental kW")
    for r in ps['emitter']:
        print(f"      {r['bonus_C']:.0f} K -> {r['delta_kW']:.2f}")
    print()

    print("-" * W)
    print("  MODEL-FORM SENSITIVITY (baseline night)")
    print("-" * W)
    print(f"    {'Case':34s} {'bare':>6s} {'passive':>8s} {'mist':>6s} {'total':>7s} {'spray f':>8s} {'nights met':>11s}")
    for r in model_form_sensitivity(env, geom):
        print(f"    {r['label']:34s} {r['bare_kW']:6.1f} {r['passive_kW']:8.1f} {r['mist_kW']:6.1f} "
              f"{r['total_kW']:7.1f} {r['f_flight']*100:7.0f}% {r['nights_met']:8d}/10")
    print()

    print("-" * W)
    print("  H. CONTROLLER ACROSS THE OPERATING ENVELOPE")
    print("-" * W)
    print(f"    {'Condition':30s} {'T':>5s} {'Tsky':>6s} {'film':>5s} "
          f"{'mist':>5s} {'ehd':>5s} {'kW':>7s} {'m3':>6s} {'dBA':>6s}")
    for r in annual_envelope(geom, ac, 100.0):
        flag = " (dry-locked)" if r["frozen"] else ""
        print(f"    {r['label']:30s} {r['T_amb']:5.0f} {r['T_sky']:6.1f} "
              f"{r['film_duty']:5.2f} {r['mist_duty']:5.2f} {r['ehd']:5.2f} "
              f"{r['P_kW']:7.1f} {r['water_m3']:6.2f} "
              f"{r['Lp_total_dBA']:6.1f}{flag}"
              f"{'  [CAPACITY LIMITED]' if r['capacity_limited'] else ''}")
    print()
    print("=" * W)


if __name__ == "__main__":
    main()
