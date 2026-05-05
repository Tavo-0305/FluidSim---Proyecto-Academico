#Importaciones necesarias

import matplotlib.pyplot as plt
from fluidsim import load_sim_for_plot
from fluidsim.util import load_state_phys_file

'''
NOTA:
Aqui la forma mas sencilla es copiar manualmente la direccion del path_run, 
dado que cada vez que se simula, se crea un archivo nuevo
'''
path_run = '/home/gustavo/Sim_data/NS2D_256x256_S2pix2pi_2026-05-05_14-10-40'

#Se cargan los espectros:
sim_plot = load_sim_for_plot(path_run)

# Cargar datos de medias espaciales desde spatial_means.txt
# Retorna un diccionario con claves: 't', 'E', 'Z', etc.

means = sim_plot.output.spatial_means.load()

t_arr = means['t']   # vector de tiempos [s]
E_arr = means['E']   # energía cinética total E(t) = ½∫(u²+v²)dxdy [m²/s²]

#Grafica para la evolucion de la energia:
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_arr, E_arr, color='royalblue', linewidth=1.8, label='$E(t)$')
ax.set_xlabel('Tiempo $t$ [s]', fontsize=13)
ax.set_ylabel('Energía cinética $E$ [m²/s²]', fontsize=13)
ax.set_title(
    'Evolución de la energía cinética — Turbulencia 2D\n'
    r'$\nu = 10^{-4}$,  $256 \times 256$',
    fontsize=12
)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fname_e = 'exp1_energia.png'
plt.savefig(fname_e, dpi=150)
plt.close()

#Graficas para el campo de Vorticidad:

#Graficas para el espectro de energia:

