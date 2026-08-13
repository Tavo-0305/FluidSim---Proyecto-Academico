#################################################
#           Librerías e importaciones          #
################################################

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from fluidsim.solvers.ns2d.solver import Simul
from fluidsim import load_sim_for_plot
from fluidsim.util import load_state_phys_file

#################################################
#           Parametros de la simulacion        #
################################################

params = Simul.create_default_params()

#Ahora se modifican los parametros:

#Dominio Espacial:
params.oper.Lx = 2*np.pi
params.oper.Ly = 2*np.pi
#Resolucion de la malla:
params.oper.nx = 256
params.oper.ny = 256
#Parametros fisicos:
params.nu_2 = 1e-4 #Viscosidad

#Integracion temporal:
params.time_stepping.USE_CFL = True #Se usa la condicion CFL para evitar inestavilidades numericas
params.time_stepping.t_end = 50.0 #En segundos
params.time_stepping.deltat_max = 0.05 #paso maximo permitido

#Condiciones iniciales
params.init_fields.type = 'noise'
params.init_fields.noise.velo_max = 1.0
params.init_fields.noise.length = 1.0 #Esta es la escala de distribucion

'''

NOTAS:
- Noise genera un campo de velocidades aleatorio
- Fuidsim Calcula el rot(u) internamente

''' 

#Salidas:

#Se configura la frecuencia de guardado de Datos en segundos:
#Para los campos fisicos (Vorticidad,velocidades bidimensionales):
params.output.periods_save.phys_fields = 2.0
#Medias espaciales (Energia cinetica,enstrofia):
params.output.periods_save.spatial_means = 0.5
#Espectros de energia
params.output.periods_save.spectra = 1.0
#Campo a visualizar en las graficas:
params.output.phys_fields.field_to_plot = 'rot'

#Se imprimen los datos generales:
print(f"  Dominio  : {params.oper.Lx:.4f} × {params.oper.Ly:.4f} [m]")
print(f"  Malla    : {params.oper.nx} X {params.oper.ny}")
print(f"  ν        : {params.nu_2}")
print(f"  t_end    : {params.time_stepping.t_end} [s]")
print(f"  CI       : ruido aleatorio, escala {params.init_fields.noise.length} m")

#################################################
#           Simulacion                         #
################################################

#Se instancia el simulador:

sim = Simul(params)

'''
NOTAS:
Esto construye internamente:
- La malla espectral (números de onda kx, ky)
- Los operadores pseudoespectrales (gradiente, laplaciano, rotacional)
- Los manejadores de salida (archivos .nc y .h5)
- El integrador temporal RK4
'''

#Se guarda la ruta:
path_run = sim.output.path_run
print(f"  Directorio de resultados: {path_run}")

#Se inicia la integracion temporal:
sim.time_stepping.start()

# Se actualiza el path_run con la ubicación final
path_run = sim.output.path_run
print(f"\n  Simulación completada.")
print(f"  Resultados en: {path_run}")

# 'load' reconstruye el objeto simulación desde los archivos en disco, sin necesidad de correrla de nuevo.
sim_loaded = load_sim_for_plot(path_run)
print("Simulación cargada desde:", path_run)

## EVOLUCIÓN TEMPORAL DE LA ENERGÍA
# Cargar los datos de medias espaciales
means = sim_loaded.output.spatial_means.load()

# 'means' es un diccionario con claves 't', 'E', 'Z', etc.
t  = means['t']    # vector de tiempos
E  = means['E']    # energía cinética total E(t) = ½∫(u²+v²)dxdy

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, E, color='royalblue', linewidth=1.8, label='$E(t)$')
ax.set_xlabel('Tiempo $t$', fontsize=13)
ax.set_ylabel('Energía cinética $E$', fontsize=13)
ax.set_title('Evolución temporal de la energía cinética\n'
             r'(turbulencia 2D, $\nu=10^{-4}$)', fontsize=13)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('exp1_energia.png', dpi=150)
plt.show()

print(f"Energía inicial: {E[0]:.4f}")
print(f"Energía final:   {E[-1]:.4f}")
print(f"Fracción conservada: {E[-1]/E[0]*100:.1f}%")

## CAMPO DE VORTICIDAD
# Obtener tiempos disponibles desde los nombres de archivo
nc_files = sorted(glob.glob(os.path.join(path_run, 'state_phys_t*.nc')))

times_saved = []
for f in nc_files:
    base = os.path.basename(f)
    t_str = base.replace('state_phys_t', '').replace('.nc', '')
    try:
        times_saved.append(float(t_str))
    except ValueError:
        pass

times_saved = sorted(times_saved)
print("Tiempos guardados disponibles:", times_saved)

# Seleccionar 4 instantes distribuidos en la simulación
n = len(times_saved)
indices = [0, n//4, n//2, n-1]

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

for ax, idx in zip(axes, indices):
    t_snap = times_saved[idx]

    # Cargar el estado físico en ese instante
    sim_snap = load_state_phys_file(path_run, t_approx=t_snap, hide_stdout=True)
    rot = sim_snap.state.get_var('rot')

    vmax = np.percentile(np.abs(rot), 98)

    im = ax.imshow(
        rot.T,
        origin='lower',
        cmap='RdBu_r',
        vmin=-vmax, vmax=vmax,
        extent=[0, params.oper.Lx, 0, params.oper.Ly]
    )
    ax.set_title(f'$t = {t_snap:.1f}$', fontsize=12)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.colorbar(im, ax=ax, shrink=0.8, label='$\\omega$')

fig.suptitle(
    'Evolución del campo de vorticidad $\\omega(x,y,t)$\n'
    'Rojo = vorticidad positiva (anticiclónica)  |  Azul = negativa (ciclónica)',
    fontsize=12
)
plt.tight_layout()
plt.savefig('exp1_vorticidad_snapshots.png', dpi=150)
plt.show()