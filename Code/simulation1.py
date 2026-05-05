#################################################
#           Librerías e importaciones          #
################################################

import numpy as np
from fluidsim.solvers.ns2d.solver import Simul

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

#Aqui se configura la frecuencia temporal de guardado
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