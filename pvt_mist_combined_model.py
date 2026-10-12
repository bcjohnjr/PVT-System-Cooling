import numpy as np
from pvt_rejection_model import p_sat_water, H_FG, CP_AIR
RHO_W=1000.0
P_ATM=101325.0
R_V=461.5

def humidity_ratio(T_C,RH):
    pv=np.clip(RH,0,1)*p_sat_water(T_C)
    return 0.622*pv/max(P_ATM-pv,1.0)

def wet_bulb_stull(T_C,RH):
    rh=np.clip(RH*100.0,0.1,100.0)
    T=T_C
    return (T*np.arctan(0.151977*np.sqrt(rh+8.313659)) +
            np.arctan(T+rh) - np.arctan(rh-1.676331) +
            0.00391838*rh**1.5*np.arctan(0.023101*rh) - 4.686035)

def air_density(T_C):
    T=T_C+273.15
    return P_ATM/(287.058*T)

def vapour_diffusivity(T_C):
    T=T_C+273.15
    return 2.42e-5*(T/273.15)**1.81

def droplet_terminal_velocity(d_um,T_C=5.0):
    d=d_um*1e-6
    rho=air_density(T_C); mu=1.75e-5*( (T_C+273.15)/278.15 )**0.7
    # iterate using Schiller-Naumann drag, good for this range
    v=(RHO_W-rho)*9.80665*d*d/(18*mu)
    for _ in range(30):
        Re=rho*abs(v)*d/mu
        Cd=24/max(Re,1e-12)*(1+0.15*max(Re,1e-12)**0.687) if Re<1000 else 0.44
        vn=np.sqrt(4*(RHO_W-rho)*9.80665*d/(3*rho*Cd))
        if abs(vn-v)<1e-9: break
        v=0.5*(v+vn)
    return float(v)

def spalding_mass_number(T_C,RH,T_surface_C=None):
    # isolated-drop surface approximated at wet bulb; mass fractions from vapor partial pressures
    Twb=wet_bulb_stull(T_C,RH) if T_surface_C is None else T_surface_C
    pv_inf=RH*p_sat_water(T_C)
    pv_s=p_sat_water(Twb)
    def Y(pv):
        # vapor mass fraction from ideal mixture
        w=0.622*pv/max(P_ATM-pv,1.0)
        return w/(1+w)
    Yinf=Y(pv_inf); Ys=Y(pv_s)
    return max((Ys-Yinf)/max(1-Ys,1e-12),0.0)

def droplet_evaporated_fraction(d_um,height_m,T_C,RH):
    d=d_um*1e-6
    v=droplet_terminal_velocity(d_um,T_C)
    t_fall=height_m/max(v,1e-9)
    D=vapour_diffusivity(T_C)
    rho_air=air_density(T_C)
    BM=spalding_mass_number(T_C,RH)
    # d^2 law for isolated sphere with stagnant-gas Sherwood number Sh=2:
    # K = 8 rho_air D ln(1+B_M)/rho_l
    K=8.0*rho_air*D*np.log1p(BM)/RHO_W
    tau=d*d/max(K,1e-30)
    frac_d2=min(t_fall/max(tau,1e-30),1.0)
    f_mass=1.0-(max(1.0-frac_d2,0.0))**1.5
    return {"f_evap":float(np.clip(f_mass,0,1)),"tau_s":float(tau),"t_fall_s":float(t_fall),
            "v_terminal_m_s":float(v),"B_M":float(BM),"K_m2_s":float(K)}

def fan_sound_power(Q_vol,dp_Pa,K_W=40.0):
    return K_W + 10*np.log10(max(Q_vol,1e-12)) + 20*np.log10(max(dp_Pa,1e-12))

def sum_levels_dB(levels):
    vals=[10**(L/10) for L in levels if np.isfinite(L)]
    if not vals: return -np.inf
    return 10*np.log10(sum(vals))

def sound_pressure_at(LW,distance_m,hemispherical=True):
    if not np.isfinite(LW): return -np.inf
    Q=2.0 if hemispherical else 1.0
    return LW - 10*np.log10(4*np.pi*distance_m**2/Q)

def perceived_loudness_ratio(delta_dB):
    return 2**(delta_dB/10.0)

def dry_cooler(heat_kW, env, ac):
    m_air=heat_kW*1000/(CP_AIR*ac.dry_air_rise_K)
    Q=m_air/air_density(env.T_amb_C)
    P=Q*ac.dry_static_dp_Pa/0.55
    return {"Q_vol":Q,"dp_Pa":ac.dry_static_dp_Pa,"P_fan_W":P,"water_kg_s":0.0}

def evaporative_cooler(heat_kW, env, ac):
    m_air=heat_kW/ac.evap_air_enthalpy_rise # kW/(kJ/kg)=kg/s
    Q=m_air/air_density(env.T_amb_C)
    # direct evaporative water corresponding to heat duty, kg/s
    water=heat_kW/H_FG*1000.0
    P=Q*ac.evap_static_dp_Pa/0.55
    return {"Q_vol":Q,"dp_Pa":ac.evap_static_dp_Pa,"P_fan_W":P,"water_kg_s":water}
