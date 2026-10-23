"""0D deceleration-phase ICF surrogate.

The asymmetric dual-piston stagnation mechanics follow
    O. A. Hurricane et al., "An analytic asymmetric-piston model for the impact of
    mode-1 shell asymmetry on ICF implosions," Phys. Plasmas 27, 062704 (2020).
    https://doi.org/10.1063/5.0001335
with Spitzer thermal ablation and marginal-ignition scalings from
    R. Betti et al., "Deceleration phase of inertial confinement fusion implosions,"
    Phys. Plasmas 9, 2277 (2002),
and the Gamow DT reactivity fit from the NRL Plasma Formulary (2019).
"""

import sys
import argparse
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import k, m_p, e, Stefan_Boltzmann
import matplotlib.pyplot as plt
import os

# --- Physical Constants ---
E_ALPHA = 3.5e6 * e  
M_DT = 2.5 * m_p     
SIGMA_SB = Stefan_Boltzmann

def nrl_dt_reactivity(T_keV):
    """Unconditionally stable Gamow fit from the NRL Plasma Formulary."""
    if T_keV < 0.5: return 0.0
    T = min(T_keV, 100.0) 
    # NRL formula yields cm^3/s, multiply by 1e-6 for m^3/s
    sigmav_cm3 = 3.68e-12 * (T**(-2.0/3.0)) * np.exp(-19.94 * (T**(-1.0/3.0)))
    return sigmav_cm3 * 1e-6

# Once the cold shell has been fully ablated into the hot spot there is no piston left to
# confine it and the model is out of validity; floor the effective piston mass at this
# fraction of its initial value to keep the integration finite rather than singular.
MIN_PISTON_FRACTION = 0.01

# --- RT -> mix coupling ---
# Prior versions integrated the RT exponent xi but never fed it back into the hot-spot energy
# balance, so it was a pure diagnostic: driving RT amplification from 42x to 2196x moved the
# yield by 0.0%. The three coefficients below close that loop.
#
# HAAN_SAT       bubble amplitude saturates near a fraction of the perturbation wavelength
#                (Haan, Phys. Rev. A 39, 5812 (1989)). Without this, exp(xi) is unbounded and
#                every mode quenches identically, which gives the adversary no strategy.
# C_ENTRAIN      rate at which bubbles of a given penetration depth drag cold ablator across
#                the hot-spot boundary. This is the single free coefficient of the mix model
#                and the one calibrated against experiment (see MIX CALIBRATION below).
# Z_ENH          bremsstrahlung enhancement per unit mix fraction. Mixed-in ablator is high-Z
#                and brems scales as Z^2, so this is the term that competes with alpha
#                heating and produces the ignition cliff.
#
# MIX CALIBRATION. Ma et al., Phys. Rev. Lett. 111, 085004 (2013) report that DT neutron yield
# and ion temperature fall abruptly once hot-spot mix mass exceeds several hundred ng, with
# inferred mix masses spanning roughly 100-2000 ng -- though with large shot-to-
# shot scatter, so it is a trend and not a calibration curve. Their measured yields sit 3-20x
# below 2D post-shot simulations, and the observed Y ~ T_ion^2.4 scaling is recovered in
# simulation only when extra mix is preloaded, which is the evidence that radiative cooling
# through mix is the dominant loss. Zylstra/Casey et al., Phys. Plasmas 27, 092709 (2020)
# attribute a ~40%
# yield degradation on affected NIF HDC shots to hot-spot mix. C_ENTRAIN is set so that a
# mid-range defect (roughness ~1 um) injects a few hundred ng and costs a few percent of
# yield, while the worst admissible defect drives the design over an ignition cliff. The
# coefficient is a calibration, NOT a first-principles result -- see README.
HAAN_SAT = 0.1
C_ENTRAIN = 0.012
Z_ENH = 8.0

# --- Mode-1 mass defect ---
# A mode-1 asymmetry does not merely repartition the shell between the two pistons: part of
# the shell misses the hot spot entirely and does no compression work. Without this term the
# total decelerating mass M1+M2 is independent of delta, so mean deceleration -- which sets
# compression and therefore yield -- is unchanged and only RKE responds. Quadratic because
# the yield penalty must not depend on the sign of delta.
C_DELTA_LOSS = 1.5

# --- Ablation-front feedthrough ---
# A rough ablation front burns through unevenly and pushes cold ablator across the boundary
# ahead of the clean ablation wave. This must NOT be modelled by scaling C_abl: in this model
# ablated mass is added to M_fuel and burns, so boosting C_abl makes roughness INCREASE the
# yield -- the opposite of the intended effect. Feedthrough is therefore routed through the
# non-fuel entrainment channel, where the mass adds inertia and radiating electrons but
# never reacts.
ROUGH_REF = 1.0e-6      # reference roughness (m) for the feedthrough scaling


def rt_penetration(xi, R_eff, mode_number, roughness):
    """Fraction of the hot-spot radius penetrated by RT bubbles, in [0, 1].

    Growth is linear in exp(xi) until the bubble amplitude approaches a fixed fraction of the
    perturbation wavelength lambda = 2*pi*R/l, after which it saturates (Haan). The tanh is a
    smooth stand-in for the saturation knee -- it is C-infinity, which the implicit solver
    needs, and it recovers a_lin for a_lin << a_sat.

    Saturation is what makes `mode` a genuine strategic choice rather than a monotone knob:
    a very high mode has a short wavelength, so it saturates early at a small amplitude and
    ultimately entrains LESS material than an intermediate mode.
    """
    if roughness <= 0.0 or R_eff <= 0.0:
        return 0.0
    a_lin = roughness * np.exp(min(xi, 50.0))       # min() guards exp() overflow
    a_sat = HAAN_SAT * (2.0 * np.pi * R_eff / max(mode_number, 1e-9))
    if a_sat <= 0.0:
        return 0.0
    amp = a_sat * np.tanh(a_lin / a_sat)
    return min(amp / R_eff, 1.0)


def mix_fraction(M_mix, M_hs):
    """Mass fraction of the hot spot that is cold, high-Z ablator rather than fuel.

    Driving the penalties off the ACCUMULATED mix mass -- rather than off the instantaneous
    bubble amplitude -- is both the physically meaningful quantity and the one NIF actually
    reports (ng of ablator in the hot spot; Ma et al., PRL 111, 085004 (2013)). It also keeps
    roughness influential: bubble amplitude saturates at a value fixed by the mode, but the
    mass entrained through that boundary keeps integrating in time.

    Capped below 1 so the reacting volume never collapses to exactly zero, which would stall
    the integrator.
    """
    if M_hs <= 0.0:
        return 0.0
    return min(0.95, max(0.0, M_mix) / M_hs)


def asymmetric_odes(t, y, M1, M2, mode_number, M_hs0, roughness):
    R1, v1, R2, v2, E_hs, yield_J, rt_int, M_fuel, M_hs, M_mix = y

    # Hard clamps to protect the solver from unphysical undershoots
    R1 = max(R1, 1e-9)
    R2 = max(R2, 1e-9)
    E_hs = max(E_hs, 1e-9)
    M_hs = max(M_hs, 1e-12)
    M_fuel = max(M_fuel, 0.0)

    # --- Shell/hot-spot mass conservation ---
    # Every kilogram ablated into the hot spot leaves the cold shell, so the pistons lose
    # inertia as the implosion proceeds. Both halves are depleted proportionally, which
    # preserves the mode-1 mass ratio M1/M2 = (1+delta)/(1-delta) set at t=0.
    M_shell_0 = M1 + M2
    ablated = max(0.0, M_hs - M_hs0)
    shell_frac = max(MIN_PISTON_FRACTION, 1.0 - ablated / M_shell_0) if M_shell_0 > 0 else MIN_PISTON_FRACTION
    M1_eff = M1 * shell_frac
    M2_eff = M2 * shell_frac
    shell_exhausted = ablated >= M_shell_0

    R_eff = (R1 + R2) / 2.0
    V = (4.0 / 3.0) * np.pi * R_eff**3
    P_hs = (2.0 / 3.0) * E_hs / V 
    
    rho_hs = M_hs / V
    n_ion = rho_hs / M_DT
    T_K = P_hs / (2.0 * n_ion * k) if n_ion > 0 else 0.0
    T_keV = T_K / 1.16045e7
    T_keV = max(T_keV, 1e-5)
    
    # --- RT-driven mix ---
    # `pen` is how deep the RT bubbles reach into the hot spot; it drives how fast cold
    # ablator is entrained. `f_mix` is the resulting contamination of the hot spot by mass,
    # and it is what actually penalises the burn.
    pen = rt_penetration(rt_int, R_eff, mode_number, roughness)
    f_mix = mix_fraction(M_mix, M_hs)
    f_clean = 1.0 - f_mix

    # Flux-Limited Spitzer Ablation
    T_keV_abl = min(T_keV, 15.0)
    C_abl = 50000.0
    # Prevent ablation shock at early, cold times, and stop once the cold shell reservoir
    # is exhausted — there is no more material to ablate into the hot spot.
    dM_abl_dt = C_abl * R_eff * (T_keV_abl**2.5) if (T_keV > 0.5 and not shell_exhausted) else 0.0

    # RT spikes and ablation-front feedthrough drag cold ablator across the boundary. This
    # material adds inertia and radiating electrons to the hot spot but is NOT fuel, so it is
    # deliberately excluded from dM_fuel_dt below. Entrainment is driven by how deep the
    # bubbles reach (pen) and how fast the boundary is moving; the feedthrough term adds a
    # roughness-proportional contribution so a rough front contaminates the hot spot even
    # before RT has amplified appreciably.
    v_spike = abs((v1 + v2) / 2.0)
    inj_strength = pen * C_ENTRAIN * (roughness / ROUGH_REF)
    dM_inj_dt = (inj_strength * (M_shell_0 * shell_frac / max(R_eff, 1e-9)) * v_spike
                 if not shell_exhausted else 0.0)
    dM_hs_dt = dM_abl_dt + dM_inj_dt

    sigmav = nrl_dt_reactivity(T_keV)

    fuel_fraction = M_fuel / M_hs if M_hs > 0 else 0.0
    n_fuel_ion = n_ion * fuel_fraction

    # Fusion proceeds only in the uncontaminated fraction of the hot spot.
    rate_density = (n_fuel_ion / 2.0)**2 * sigmav
    R_rxn = rate_density * V * f_clean

    W_alpha = R_rxn * E_ALPHA

    # Blackbody Radiative Cap. Mixed-in ablator is high-Z and bremsstrahlung scales as Z^2,
    # so contamination multiplies the radiative loss that competes with alpha heating.
    Z_factor = 1.0 + Z_ENH * f_mix
    W_rad_volumetric = 5.34e-37 * (n_ion**2) * np.sqrt(T_keV) * V * Z_factor
    W_blackbody = SIGMA_SB * (T_K**4) * (4.0 * np.pi * R_eff**2) * Z_factor
    W_rad = min(W_rad_volumetric, W_blackbody)

    # Force the yield derivative to be strictly positive
    dW_yield_dt = max(0.0, 5.0 * W_alpha)

    # Replenish fuel with ablated shell mass. Only the ablated component is fuel; the
    # spike-injected component is cold ablator and dilutes the reacting fraction.
    dM_fuel_dt = dM_abl_dt - R_rxn * (2.0 * M_DT)

    A_eff = 4.0 * np.pi * R_eff**2
    dv1_dt = (A_eff * P_hs) / M1_eff
    dv2_dt = (A_eff * P_hs) / M2_eff
    
    dV_dt = A_eff * ((v1 + v2) / 2.0)
    dE_dt = -P_hs * dV_dt + W_alpha - W_rad 
    
    g_eff = (dv1_dt + dv2_dt) / 2.0
    dRT_dt = 0.0
    if g_eff > 0:
        wave_number = mode_number / R_eff 
        dRT_dt = np.sqrt(wave_number * g_eff)
    
    return [v1, dv1_dt, v2, dv2_dt, dE_dt, dW_yield_dt, dRT_dt, dM_fuel_dt, dM_hs_dt, dM_inj_dt]

def plot_synthetic_density(R1, R2, rt_exponent, time_val, step, mode_number, initial_roughness, output_dir="implosion_frames"):
    os.makedirs(output_dir, exist_ok=True)
    grid_size = 600
    L = 200e-6  
    x = np.linspace(-L, L, grid_size)
    y = np.linspace(-L, L, grid_size)
    X, Y = np.meshgrid(x, y)
    
    distance = np.sqrt(X**2 + Y**2)
    angles = np.arctan2(Y, X)
    
    R_theta = np.where(X < 0, R1, R2) 
    rt_amp = initial_roughness * np.exp(rt_exponent)
    R_perturbed = R_theta + rt_amp * np.cos(mode_number * angles)
    
    density_map = np.zeros_like(X)
    shell_thickness = 15e-6
    
    shell_mask = (distance > R_perturbed) & (distance < R_perturbed + shell_thickness)
    hotspot_mask = distance <= R_perturbed
    
    density_map[shell_mask] = 100.0  
    density_map[hotspot_mask] = 10.0 
    
    plt.figure(figsize=(6,6))
    plt.pcolormesh(X*1e6, Y*1e6, density_map, cmap='magma', shading='nearest')
    plt.title(f"Synthetic RT + Mode-1 Model: t = {time_val*1e9:.3f} ns\n(Delta={abs(R1-R2)/(R1+R2):.2f}, l={mode_number})")
    plt.xlabel("X (um)")
    plt.ylabel("Y (um)")
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    
    plt.savefig(f"{output_dir}/frame_{step:03d}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_histories(sol, M1, M2, output_dir="implosion_frames"):
    os.makedirs(output_dir, exist_ok=True)
    
    t_ns = sol.t * 1e9
    R1 = sol.y[0]
    R2 = sol.y[2]
    R_eff = (R1 + R2) / 2.0
    V = (4.0 / 3.0) * np.pi * R_eff**3
    E_hs = sol.y[4]
    M_hs_total = sol.y[8]
    
    rho_hs = M_hs_total / V
    rhoR_hs_gcm2 = (rho_hs * R_eff) * 0.1
    rhoR_shell_gcm2 = ((M1 + M2) / (4.0 * np.pi * R_eff**2)) * 0.1 
    
    P_hs = (2.0 / 3.0) * E_hs / V
    n_ion = rho_hs / M_DT
    T_keV = np.zeros_like(P_hs)
    valid = n_ion > 0
    T_keV[valid] = (P_hs[valid] / (2.0 * n_ion[valid] * k)) / 1.16045e7
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    axs[0].plot(t_ns, R1*1e6, label='R1 (Left Piston)', color='blue')
    axs[0].plot(t_ns, R2*1e6, label='R2 (Right Piston)', color='red', linestyle='--')
    axs[0].set_ylabel('Radius (um)')
    axs[0].set_title('Implosion Trajectory')
    axs[0].legend()
    axs[0].grid(True, alpha=0.5)
    
    axs[1].plot(t_ns, rhoR_shell_gcm2, label='Shell $\\rho R$', color='green')
    axs[1].plot(t_ns, rhoR_hs_gcm2, label='Hot Spot $\\rho R$', color='orange')
    axs[1].set_ylabel('Areal Density (g/cm$^2$)')
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].grid(True, alpha=0.5)
    
    axs[2].plot(t_ns, T_keV, label='$T_{ion} \\approx T_{elec}$', color='purple')
    axs[2].set_ylabel('Temperature (keV)')
    axs[2].set_xlabel('Time (ns)')
    axs[2].legend()
    axs[2].grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_histories.png", dpi=300, bbox_inches='tight')
    plt.close()

def solve_implosion(R0, v0, T0, M_sh, M_hs, delta, mode, roughness):
    """Integrate the deceleration-phase ODE system for a given capsule design.

    Returns (sol, M1, M2): the solve_ivp solution object and the two piston masses.
    All inputs are in SI units except T0 which is in keV.
    """
    # A mode-1 defect both repartitions the shell AND spills some of it past the hot spot,
    # where it does no compression work. The engaged fraction reduces the TOTAL decelerating
    # mass, which is what actually moves the yield; the (1 +/- delta) split alone leaves mean
    # deceleration untouched and only shows up in the residual kinetic energy.
    M_engaged = M_sh * max(0.05, 1.0 - C_DELTA_LOSS * delta**2)
    M1 = (M_engaged / 2.0) * (1.0 + delta)
    M2 = (M_engaged / 2.0) * (1.0 - delta)

    V0 = (4.0 / 3.0) * np.pi * R0**3
    P_hs0 = 2.0 * (M_hs / V0 / M_DT) * k * (T0 * 1.16045e7)
    E_hs0 = 1.5 * P_hs0 * V0

    y0 = [R0, v0, R0, v0, E_hs0, 0.0, 0.0, M_hs, M_hs, 0.0]

    t_span = (0, 0.6e-9)
    t_eval = np.linspace(0, 0.6e-9, 200)

    # Safely back on Radau. No custom tolerance arrays to trip it up.
    sol = solve_ivp(asymmetric_odes, t_span, y0, t_eval=t_eval, method='Radau',
                    args=(M1, M2, mode, M_hs, roughness), rtol=1e-6, atol=1e-8, max_step=1e-12)

    return sol, M1, M2


def compute_metrics(sol, M1, M2, M_hs, mode, roughness=0.0):
    """Compute stagnation metrics from an integrated solution. Returns a dict."""
    R_eff_array = (sol.y[0] + sol.y[2]) / 2.0
    stag_idx = np.argmin(R_eff_array)

    R1_stag = sol.y[0][stag_idx]
    R2_stag = sol.y[2][stag_idx]
    v1_stag = sol.y[1][stag_idx]
    v2_stag = sol.y[3][stag_idx]

    # The pistons lose mass to ablation, so the residual kinetic energy at stagnation must be
    # evaluated with the DEPLETED piston masses (mirroring the depletion in asymmetric_odes),
    # not the initial ones.
    M_shell_0 = M1 + M2
    ablated_stag = max(0.0, sol.y[8][stag_idx] - M_hs)
    shell_frac = (max(MIN_PISTON_FRACTION, 1.0 - ablated_stag / M_shell_0)
                  if M_shell_0 > 0 else MIN_PISTON_FRACTION)
    RKE = 0.5 * (M1 * shell_frac) * v1_stag**2 + 0.5 * (M2 * shell_frac) * v2_stag**2

    E_hs_stag = sol.y[4][stag_idx]
    V_stag = (4.0 / 3.0) * np.pi * ((R1_stag + R2_stag) / 2.0)**3
    P_stag = (2.0 / 3.0) * E_hs_stag / V_stag

    rt_exponent = sol.y[6][stag_idx]
    rt_amplification = np.exp(rt_exponent)

    # Contaminated hot-spot volume fraction at stagnation, and its peak over the run. The
    # peak matters because burn can be quenched during the compression even if the mix
    # fraction relaxes slightly by the time minimum radius is reached.
    R_eff_stag = (R1_stag + R2_stag) / 2.0
    f_mix_stag = mix_fraction(sol.y[9][stag_idx], sol.y[8][stag_idx])

    # Non-fuel ablator mass injected into the hot spot. This is the quantity NIF reports as
    # "hot-spot mix mass" (Ma et al. quote a ~100 ng detection threshold on CH(Si)), so it is
    # the natural handle for calibrating this model against experiment.
    mix_mass_stag_ug = max(0.0, sol.y[9][stag_idx] * 1e9)
    mix_mass_ug = max(0.0, sol.y[9][-1] * 1e9)
    f_mix_peak = max(
        mix_fraction(sol.y[9][i], sol.y[8][i]) for i in range(stag_idx + 1)
    )

    # Force diagnostic yield to non-negative just in case
    final_yield_MJ = max(0.0, sol.y[5][-1] / 1e6)

    initial_fuel_ug = M_hs * 1e9
    final_fuel_ug = max(0.0, sol.y[7][-1] * 1e9)
    total_ablated_mass_ug = max(0.0, (sol.y[8][-1] - M_hs) * 1e9)

    total_pool = initial_fuel_ug + total_ablated_mass_ug
    burned_fraction = ((total_pool - final_fuel_ug) / total_pool * 100.0) if total_pool > 0 else 0.0

    # Fraction of the cold shell consumed by ablation. Approaching 1 means the piston has
    # essentially been eaten away and the 0D model is at the edge of its validity.
    shell_consumed = (total_ablated_mass_ug / (M_shell_0 * 1e9)) if M_shell_0 > 0 else 0.0

    return {
        "stag_idx": stag_idx,
        "R1_stag": R1_stag,
        "R2_stag": R2_stag,
        "shell_consumed": shell_consumed,
        "P_stag_Gbar": P_stag / 1e14,
        "yield_MJ": final_yield_MJ,
        "ablated_ug": total_ablated_mass_ug,
        "burn_fraction": burned_fraction,
        "total_pool_ug": total_pool,
        "RKE_J": RKE,
        "rt_amplification": rt_amplification,
        "f_mix_stag": f_mix_stag,
        "f_mix_peak": f_mix_peak,
        "mix_mass_ug": mix_mass_ug,
        "mix_mass_stag_ug": mix_mass_stag_ug,
        "mode": mode,
    }


def main():
    parser = argparse.ArgumentParser(description="2D Asymmetric Piston Surrogate")
    parser.add_argument('--R0', type=float, default=150e-6)
    parser.add_argument('--v0', type=float, default=-450e3)
    parser.add_argument('--T0', type=float, default=1.0)
    parser.add_argument('--M_sh', type=float, default=0.25e-6)
    parser.add_argument('--M_hs', type=float, default=0.005e-6)
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--mode', type=float, default=10.0)
    parser.add_argument('--roughness', type=float, default=0.1e-6)
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()

    print(f"Simulating implosion (Delta = {args.delta}, Mode = {args.mode})...")

    sol, M1, M2 = solve_implosion(args.R0, args.v0, args.T0, args.M_sh, args.M_hs,
                                  args.delta, args.mode, args.roughness)

    t_eval_len = 200
    if len(sol.t) < t_eval_len:
        print(f"Warning: ODE solver stopped early at step {len(sol.t)}.")

    m = compute_metrics(sol, M1, M2, args.M_hs, args.mode, args.roughness)
    stag_idx = m["stag_idx"]

    if args.plot:
        print("Generating high-resolution PNGs and time-history traces...")
        output_dir = "implosion_frames"
        plot_time_histories(sol, M1, M2, output_dir=output_dir)

        end_idx = min(stag_idx + 20, len(sol.t))
        for i in range(0, end_idx, 2):
            plot_synthetic_density(
                sol.y[0][i], sol.y[2][i], sol.y[6][i], sol.t[i], i,
                args.mode, args.roughness, output_dir=output_dir
            )
        print(f"Saved visualization files to the '{output_dir}/' directory.")

    print(f"\n--- Asymmetric Mode-1 Metrics (Delta = {args.delta}) ---")
    print(f"Stagnation Radius (Left):  {m['R1_stag']*1e6:.2f} um")
    print(f"Stagnation Radius (Right): {m['R2_stag']*1e6:.2f} um")
    print(f"Stagnation Pressure:       {m['P_stag_Gbar']:.2f} Gbar")
    print(f"Total Fusion Yield:        {m['yield_MJ']:.3f} MJ")
    print(f"Ablated Mass Injected:     {m['ablated_ug']:.2f} ug")
    print(f"Fuel Burn Fraction:        {m['burn_fraction']:.2f}% (Total Pool: {m['total_pool_ug']:.2f} ug)")
    print(f"Shell Mass Consumed:       {m['shell_consumed']*100:.1f}% of the cold shell")
    print(f"Residual Kinetic Energy:   {m['RKE_J']:.2f} Joules")
    print(f"RT Amplification (l={args.mode}): {m['rt_amplification']:.2f}x initial perturbation")
    print(f"Hot-Spot Mix Fraction:     {m['f_mix_stag']*100:.1f}% at stagnation "
          f"(peak {m['f_mix_peak']*100:.1f}%)")
    print(f"Hot-Spot Mix Mass:         {m['mix_mass_stag_ug']*1000:.1f} ng at stagnation")


if __name__ == "__main__":
    main()
