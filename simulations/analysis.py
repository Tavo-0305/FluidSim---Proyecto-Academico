"""
analisis.py
Grafica energia/enstrofia/vorticidad y calcula el criterio Okubo-Weiss
para una simulacion NS2D ya corrida.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from fluidsim import load_sim_for_plot
from fluidsim.operators.operators2d import OperatorsPseudoSpectral2D
from fluidsim.solvers.ns2d.solver import Simul


def graficar_generales(sim, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

    # Vorticidad
    sim.output.phys_fields.plot(field="rot")
    fig = plt.gcf()  # captura la figura recien creada por .plot()
    fig.savefig(os.path.join(fig_dir, "vorticidad.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Energia / enstrofia
    sim.output.spatial_means.plot()
    for i, num in enumerate(plt.get_fignums()):
        fig = plt.figure(num)
        fig.savefig(os.path.join(fig_dir, f"spatial_means_{i}.png"),
                    dpi=150, bbox_inches="tight")
    plt.close("all")

def calcular_okubo_weiss(sim, oper_full, time):
    """
    Calcula el campo OW = s_n^2 + s_s^2 - omega^2
    usando derivadas espectrales.
    """

    # Campos de velocidad en el tiempo solicitado
    ux, t_ux = sim.output.phys_fields.get_field_to_plot("ux", time)
    uy, t_uy = sim.output.phys_fields.get_field_to_plot("uy", time)

    # Transformadas al espacio de Fourier
    ux_fft = oper_full.fft(ux)
    uy_fft = oper_full.fft(uy)

    # Gradientes espectrales
    dux_dx_fft, dux_dy_fft = oper_full.gradfft_from_fft(ux_fft)
    duy_dx_fft, duy_dy_fft = oper_full.gradfft_from_fft(uy_fft)

    # Regresar al espacio físico
    dux_dx = oper_full.ifft(dux_dx_fft)
    dux_dy = oper_full.ifft(dux_dy_fft)
    duy_dx = oper_full.ifft(duy_dx_fft)
    duy_dy = oper_full.ifft(duy_dy_fft)

    # Deformaciones y vorticidad
    s_n = dux_dx - duy_dy
    s_s = duy_dx + dux_dy
    omega = duy_dx - dux_dy

    # Criterio Okubo-Weiss
    OW = s_n**2 + s_s**2 - omega**2

    return OW, omega

def graficar_ow(OW, fig_dir):
    threshold = -0.2 * np.std(OW)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(OW, cmap="RdBu_r", vmin=-np.std(OW)*3, vmax=np.std(OW)*3)
    ax.contour(OW, levels=[threshold], colors="k", linewidths=0.8)
    plt.colorbar(im, ax=ax, label="OW")
    ax.set_title("Criterio Okubo-Weiss")
    fig.savefig(os.path.join(fig_dir, "okubo_weiss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Umbral OW usado: {threshold:.4e}")
    frac_vortice = np.mean(OW < threshold)
    print(f"Fraccion del dominio clasificada como vortice: {frac_vortice:.2%}")


if __name__ == "__main__":
    path_run = "/home/gustavo/Sim_data/NS2D_1024x1024_S2pix2pi_2026-09-28_18-37-18"
    fig_dir = os.path.join(path_run, "figures")

    sim = load_sim_for_plot(path_run)

    # ----------------------------------------
    # Operador espectral de resolución completa
    # ----------------------------------------
    params_full = Simul.create_default_params()

    params_full.oper.nx = 1024
    params_full.oper.ny = 1024
    params_full.oper.Lx = sim.params.oper.Lx
    params_full.oper.Ly = sim.params.oper.Ly
    params_full.oper.type_fft = sim.params.oper.type_fft

    oper_full = OperatorsPseudoSpectral2D(params_full)

    # ----------------------------------------
    # Análisis
    # ----------------------------------------
    graficar_generales(sim, fig_dir)

    OW, omega = calcular_okubo_weiss(sim, oper_full, 10)

    graficar_ow(OW, fig_dir)

    print(f"Figuras guardadas en: {fig_dir}")