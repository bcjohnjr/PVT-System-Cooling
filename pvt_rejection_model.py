import numpy as np
SIGMA=5.670374419e-8
G=9.80665
RHO_AIR=1.269 # approx at 5 C
CP_AIR=1006.0
MU_AIR=1.75e-5
K_AIR=0.0249
NU_AIR=MU_AIR/RHO_AIR
PR_AIR=0.71
H_FG=2.45e6
BETA=1/278.15

def p_sat_water(T_C):
    # Buck equation over water, Pa
    return 611.21*np.exp((18.678-T_C/234.5)*(T_C/(257.14+T_C)))

def h_natural_convection(Ts_C, Ta_C, L, tilt_rad=0.0):
    Tf=(Ts_C+Ta_C)/2+273.15
    beta=1/Tf
    rho=101325/(287.058*Tf)
    mu=1.8e-5*((Tf/300.0)**0.7)
    nu=mu/rho
    k=0.0257*((Tf/300.0)**0.85)
    Pr=0.71
    alpha=nu/Pr
    dT=max(Ts_C-Ta_C,1e-9)
    # Inclined-plate Churchill-Chu correction: g*cos(theta_v) when theta_v
    # is measured from vertical. Geometry stores tilt beta from horizontal, so
    # cos(theta_v) = sin(beta). A small floor avoids a singular zero-Rayleigh
    # calculation if this helper is used at exactly horizontal orientation.
    geff=G*max(np.sin(tilt_rad),0.05)
    Ra=geff*beta*dT*L**3/(nu*alpha)
    Nu=(0.825 + 0.387*Ra**(1/6)/(1+(0.492/Pr)**(9/16))**(8/27))**2
    return Nu*k/L

def h_forced_convection(v,L):
    if v<=0: return 0.0
    T=293.15
    rho=1.204
    mu=1.81e-5
    nu=mu/rho
    k=0.0259
    Pr=0.71
    Re=v*L/nu
    if Re < 5e5:
        Nu=0.664*np.sqrt(Re)*Pr**(1/3)
    else:
        Nu=(0.037*Re**0.8-871)*Pr**(1/3)
        Nu=max(Nu,0)
    return Nu*k/L

def q_radiative_flux(Ts_C,Tsky_C,eps):
    return eps*SIGMA*((Ts_C+273.15)**4-(Tsky_C+273.15)**4)

def q_evaporative_flux(Ts_C,Ta_C,RH,h_conv):
    # Lewis/Chilton-Colburn: h_m,kg/m2/s per humidity-ratio difference ~ h/(cp)*Le^-2/3
    Le=0.865
    p=101325.0
    ps=p_sat_water(Ts_C)
    pa=RH*p_sat_water(Ta_C)
    ws=0.622*ps/max(p-ps,1.0)
    wa=0.622*pa/max(p-pa,1.0)
    mflux=(h_conv/CP_AIR)*(Le**(-2/3))*max(ws-wa,0.0)
    return mflux*H_FG

def stack_air_velocity(H,dT,Tamb_C,K):
    T=Tamb_C+273.15
    return np.sqrt(max(2*G*H*(dT/T)/max(K,1e-9),0.0))
