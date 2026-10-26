"""
Analisis completo para los Objetivos 1, 2 y 3 del anteproyecto.

Edita el diccionario RUNS (resolucion -> path_run) y T_STAT (tiempo a partir del
cual el flujo es estacionario) en la seccion __main__.

Todo el calculo de Q_OW y de espectros se hace con numpy a partir de la
vorticidad, asi no depende del operador de fluidsim ni del backend MPI.
Convencion de arreglos: campo[iy, ix] (como entrega fluidsim), dominio 2pi x 2pi.
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage


# =====================================================================
# Utilidades espectrales (solo numpy)
# =====================================================================
def wavenumbers(n, L=2 * np.pi):
    k1 = np.fft.fftfreq(n, d=1.0 / n) * 2 * np.pi / L
    KY, KX = np.meshgrid(k1, k1, indexing="ij")
    return KX, KY, KX**2 + KY**2


def coarse_grain(omega, m):
    """Trunca espectralmente omega (n x n) a una malla m x m (m < n).
    Representa el MISMO flujo visto en una malla mas gruesa."""
    n = omega.shape[0]
    if m == n:
        return omega
    w_hat = np.fft.fftshift(np.fft.fft2(omega))
    c, h = n // 2, m // 2
    w_hat = w_hat[c - h:c + h, c - h:c + h] * (m / n) ** 2
    return np.fft.ifft2(np.fft.ifftshift(w_hat)).real


def okubo_weiss(omega, L=2 * np.pi):
    """Q_OW = 1/2 (1/2 omega^2 - |S|^2) a partir de omega.
    psi_hat = omega_hat / k^2;  u = (d_y psi, -d_x psi)  =>  omega = -lap(psi)."""
    n = omega.shape[0]
    KX, KY, K2 = wavenumbers(n, L)
    K2s = K2.copy()
    K2s[0, 0] = 1.0
    psi_hat = np.fft.fft2(omega) / K2s
    psi_hat[0, 0] = 0.0
    d = lambda f_hat: np.fft.ifft2(f_hat).real
    dux_dx = d(-KX * KY * psi_hat)       # d_x d_y psi
    dux_dy = d(-KY * KY * psi_hat)       # d_yy psi
    duy_dx = d(KX * KX * psi_hat)        # -d_xx psi
    duy_dy = d(KX * KY * psi_hat)        # -d_x d_y psi
    om = duy_dx - dux_dy
    S2 = dux_dx**2 + duy_dy**2 + 0.5 * (dux_dy + duy_dx) ** 2 * 2.0
    return 0.5 * (0.5 * om**2 - S2)


def energy_spectrum_from_omega(omega, L=2 * np.pi):
    """E(k) por cascarones: E = sum 1/2 |u_hat|^2 = sum 1/2 |omega_hat|^2 / k^2."""
    n = omega.shape[0]
    _, _, K2 = wavenumbers(n, L)
    K = np.sqrt(K2)
    K2s = K2.copy()
    K2s[0, 0] = 1.0
    w_hat = np.fft.fft2(omega) / n**2
    e2d = 0.5 * np.abs(w_hat) ** 2 / K2s
    e2d[0, 0] = 0.0
    dk = 2 * np.pi / L
    kbin = np.rint(K / dk).astype(int)
    kmax = n // 2
    Ek = np.bincount(kbin.ravel(), weights=e2d.ravel(), minlength=kmax + 1)[:kmax + 1]
    return np.arange(kmax + 1) * dk, Ek / dk


def fit_slope(k, Ek, kmin, kmax):
    sel = (k >= kmin) & (k <= kmax) & (Ek > 0)
    p = np.polyfit(np.log(k[sel]), np.log(Ek[sel]), 1)
    return p[0]


# =====================================================================
# Estructuras coherentes
# =====================================================================
def label_periodic(mask):
    """Etiqueta componentes conexas 4-vecinas respetando periodicidad."""
    lab, n = ndimage.label(mask)
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = set(zip(lab[0, :], lab[-1, :])) | set(zip(lab[:, 0], lab[:, -1]))
    for a, b in pairs:
        if a > 0 and b > 0:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    roots = np.array([find(i) for i in range(n + 1)])
    return roots[lab]  # etiquetas no consecutivas, 0 = fondo


def structure_stats(mask, L=2 * np.pi, min_pixels=4):
    """Numero, areas y radios equivalentes de las estructuras (mask=True)."""
    n = mask.shape[0]
    dA = (L / n) ** 2
    lab = label_periodic(mask)
    counts = np.bincount(lab.ravel())[1:]
    counts = counts[counts >= min_pixels]  # descarta etiquetas vacias y ruido de 1-3 pixeles
    areas = counts * dA
    return dict(
        n_struct=len(areas),
        areas=areas,
        r_eq=np.sqrt(areas / np.pi),
        area_frac=mask.mean(),
    )


# =====================================================================
# Lectura de datos de fluidsim
# =====================================================================
def load_run(path_run):
    from fluidsim import load_sim_for_plot
    return load_sim_for_plot(path_run)


def get_times(sim):
    return np.array(sim.output.phys_fields.set_of_phys_files.times)


def get_omega(sim, t):
    rot, _ = sim.output.phys_fields.get_field_to_plot("rot", time=t,
                                                       interpolate_time=False)
    return np.asarray(rot)


# =====================================================================
# Objetivo 1: estacionariedad y balance de energia
# =====================================================================
def plot_stationarity(sim, fig_dir, kf, nu2, t_stat=None):
    d = sim.output.spatial_means.load()
    t, E, Z = d["t"], d["E"], d["Z"]
    fig, ax = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
    ax[0].plot(t, E); ax[0].set_ylabel("E(t)")
    ax[1].plot(t, Z); ax[1].set_ylabel("Z(t)")
    ax[2].plot(t, Z / E); ax[2].axhline(kf**2, ls="--", c="k", label=r"$k_f^2$")
    ax[2].set_ylabel("Z/E"); ax[2].legend()
    # balance: inyeccion vs disipacion (si estan en el archivo)
    for key, lab in [("PK_tot", "inyeccion energia"), ("epsK_tot", "disipacion energia")]:
        if key in d:
            ax[3].plot(t, d[key], label=lab)
    ax[3].set_ylabel("balance E"); ax[3].set_xlabel("t"); ax[3].legend()
    if t_stat is not None:
        for a in ax:
            a.axvline(t_stat, c="r", ls=":")
    fig.savefig(os.path.join(fig_dir, "estacionariedad.png"), bbox_inches="tight")
    plt.close(fig)

    # Numeros de control (diagnostico)
    tt = t > (t_stat if t_stat else 0.5 * t[-1])
    Em, Zm = E[tt].mean(), Z[tt].mean()
    print(f"<E>={Em:.3e}  <Z>={Zm:.3e}  Z/E={Zm/Em:.3e}  (kf^2={kf**2})")
    print(f"Re_kf = omega_rms/(nu kf^2) = {np.sqrt(2*Zm)/(nu2*kf**2):.2f}")
    print(f"tiempo viscoso en kf = {1/(2*nu2*kf**2):.3f}")
    if "PK_tot" in d:
        print(f"<PK_tot>={d['PK_tot'][tt].mean():.3e}  <PZ_tot>={d['PZ_tot'][tt].mean():.3e}")


# =====================================================================
# Objetivos 2 y 3: barrido en snapshots y resoluciones
# =====================================================================
def analyze_snapshot(omega, thr=0.0):
    Q = okubo_weiss(omega)
    mask = Q > thr
    st = structure_stats(mask)
    k, E_tot = energy_spectrum_from_omega(omega)
    _, E_c = energy_spectrum_from_omega(np.where(mask, omega, 0.0))
    _, E_r = energy_spectrum_from_omega(np.where(mask, 0.0, omega))
    return dict(Q=Q, mask=mask, st=st, k=k, E_tot=E_tot, E_c=E_c, E_r=E_r)


def plot_ow_map(Q, fig_dir, name, crop=None):
    f = Q if crop is None else Q[:crop, :crop]
    s = np.std(Q)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(f, cmap="RdBu_r", vmin=-3 * s, vmax=3 * s, origin="lower")
    ax.contour(f, levels=[0.0], colors="k", linewidths=0.4)
    plt.colorbar(im, ax=ax, label=r"$Q_{OW}$")
    ax.set_title(name)
    fig.savefig(os.path.join(fig_dir, name + ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def run_sensitivity(runs, t_stat, out_dir, kf, n_snap=10, thr=0.0,
                    coarse_levels=(1024, 512, 256), fit_range=None):
    """
    runs : {N: path_run}.
    Para cada simulacion (y opcionalmente cada nivel de coarse-graining del
    campo de mayor resolucion) promedia sobre n_snap instantes estacionarios.
    """
    os.makedirs(out_dir, exist_ok=True)
    results = {}  # etiqueta -> dict de listas

    cases = []  # (etiqueta, N_malla, path, N_original)
    for N, p in runs.items():
        cases.append((f"DNS_{N}", N, p, N))
    Nmax = max(runs)
    for m in coarse_levels:
        if m < Nmax:
            cases.append((f"truncado_{m}", m, runs[Nmax], Nmax))

    for label, M, path, N0 in cases:
        sim = load_run(path)
        times = get_times(sim)
        times = times[times >= t_stat]
        pick = times[np.linspace(0, len(times) - 1, min(n_snap, len(times))).astype(int)]
        acc = dict(frac=[], nstruct=[], areas=[], E_tot=[], E_c=[], E_r=[])
        for i, t in enumerate(pick):
            w = coarse_grain(get_omega(sim, t), M)
            r = analyze_snapshot(w, thr)
            acc["frac"].append(r["st"]["area_frac"])
            acc["nstruct"].append(r["st"]["n_struct"])
            acc["areas"].append(r["st"]["areas"])
            for key in ("E_tot", "E_c", "E_r"):
                acc[key].append(r[key])
            k = r["k"]
            if i == len(pick) - 1:
                plot_ow_map(r["Q"], out_dir, f"OW_{label}_t{t:.1f}")
        acc["k"] = k
        acc["areas"] = np.concatenate(acc["areas"]) if acc["areas"] else np.array([])
        results[label] = acc
        print(f"{label}: area coherente = {np.mean(acc['frac']):.3%} +- {np.std(acc['frac']):.3%} | "
              f"N estructuras = {np.mean(acc['nstruct']):.0f} | "
              f"area media = {acc['areas'].mean() if acc['areas'].size else np.nan:.3e}")

    # --- Fig: fraccion y numero de estructuras vs resolucion
    labels = list(results)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].errorbar(labels, [np.mean(results[l]["frac"]) for l in labels],
                   yerr=[np.std(results[l]["frac"]) for l in labels], fmt="o-")
    ax[0].set_ylabel("fraccion de area coherente (Q_OW>0)")
    ax[1].errorbar(labels, [np.mean(results[l]["nstruct"]) for l in labels],
                   yerr=[np.std(results[l]["nstruct"]) for l in labels], fmt="o-")
    ax[1].set_ylabel("numero de estructuras")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    fig.savefig(os.path.join(out_dir, "obj2_fraccion_numero.png"), bbox_inches="tight")
    plt.close(fig)

    # --- Fig: PDF de tamanos
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for l in labels:
        a = results[l]["areas"]
        if a.size:
            bins = np.logspace(np.log10(a.min()), np.log10(a.max()), 30)
            h, b = np.histogram(a, bins=bins, density=True)
            ax.loglog(0.5 * (b[1:] + b[:-1]), h, "o-", ms=3, label=l)
    ax.set_xlabel("area de la estructura"); ax.set_ylabel("PDF"); ax.legend()
    fig.savefig(os.path.join(out_dir, "obj2_pdf_areas.png"), bbox_inches="tight")
    plt.close(fig)

    # --- Fig: espectros coherente / residual / total
    fig, axs = plt.subplots(1, len(labels), figsize=(4.5 * len(labels), 4), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, l in zip(axs, labels):
        k = results[l]["k"]
        Et, Ec, Er = (np.mean(results[l][x], axis=0) for x in ("E_tot", "E_c", "E_r"))
        ax.loglog(k[1:], Et[1:], "k", label="total")
        ax.loglog(k[1:], Ec[1:], "r", label="coherente")
        ax.loglog(k[1:], Er[1:], "b", label="residual")
        kr = np.array([3.0, kf])
        ax.loglog(kr, Et[3] * (kr / 3) ** (-5 / 3), "k--", lw=0.8, label=r"$k^{-5/3}$")
        kd = np.array([kf, 0.9 * k[-1]])
        ax.loglog(kd, Et[int(kf)] * (kd / kf) ** (-3), "k:", lw=0.8, label=r"$k^{-3}$")
        ax.axvline(kf, c="gray", lw=0.5)
        ax.set_title(l); ax.set_xlabel("k")
        if fit_range:
            s = fit_slope(k, Er, *fit_range)
            print(f"{l}: pendiente residual en k in {fit_range} = {s:.2f} (ref -5/3 = -1.67)")
    axs[0].set_ylabel("E(k)"); axs[0].legend(fontsize=7)
    fig.savefig(os.path.join(out_dir, "obj3_espectros.png"), bbox_inches="tight")
    plt.close(fig)
    return results


# =====================================================================
if __name__ == "__main__":
    # ---- EDITAR ----
    RUNS = {
        #256: "/ruta/a/la/corrida_256",
        512: "/home/gustavo/Sim_data/ns2d_gaussian_damping/NS2D_512x512_S2pix2pi_2026-10-19_14-14-10",
        #1024: "/home/gustavo/clusterDATA/NS2D_1024x1024_S2pix2pi_2026-10-09_09-35-47",
        #2048: "/home/gustavo/clusterDATA/NS2D_1024x1024_S2pix2pi_2026-10-09_09-35-47",
    }
    KF = 200          # numero de onda de forzamiento
    NU2 = 5e-5
    T_STAT = 20.0    # tiempo desde el cual hay estacionariedad (ver estacionariedad.png)
    # -----------------
    out = os.path.join(os.path.dirname(list(RUNS.values())[0]), "Figures")
    os.makedirs(out, exist_ok=True)

    sim_ref = load_run(RUNS[max(RUNS)])
    plot_stationarity(sim_ref, out, KF, NU2, T_STAT)
    run_sensitivity(RUNS, T_STAT, out, KF, n_snap=10, thr=0.0,
                    coarse_levels=(1024, 512, 256),
                    fit_range=(3, KF / 2))   # rango de la cascada inversa
    print("Figuras en:", out)
