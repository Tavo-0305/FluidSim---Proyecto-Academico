import numpy as np
from fluidsim.solvers.ns2d.solver import Simul 

def generate_physical_parameters(injection_rate=13.05):
    '''
    Esta función genera un diccionario de python con los parámetros físicos
    de la simulación, todos se mantienen fijos a excepción del "injection rate"
    de energía. 
    '''
    phys_parameters = {
        'KF' : 200.0, #numero de onda de forzamiento
        'DKF' : 4.0, #ancho de banda
        'NU2' : 5e-5, #viscocidad a pequena escala
        'EPSILON TARGET' : injection_rate, 
        'ALPHA OMEGA': 500.0, #factor de friccion de gran escala 
        'K0 OMEGA' : 0.1, #centro del perfil gausiano
        'SIGMA SQUARED OMEGA' : 1.0 #varianza del perfil gausiano
    }
    return phys_parameters

def params_construction(physical_params,N,t_end,fft_backend='fftwmpi2d', use_mpi=True):
    '''
    Esta funcion construye los params del simulador para una resolucion N dada
    '''
    params = Simul.create_default_params()
    #Se define la malla
    params.oper.nx = N
    params.oper.ny = N
    #Se utiliza un dominio cuadrado periodico
    params.oper.Lx = 2 * np.pi 
    params.oper.Ly = 2 * np.pi
    #Metodo para eliminar aliasing en metodos pseudoespectrales
    params.oper.coef_dealiasing = 2 / 3

    #Se configura el backend:
    if fft_backend is None:
        pass
    elif use_mpi:
        params.oper.type_fft = f"fft2d.mpi_with_{fft_backend}"
    else:
        params.oper.type_fft = f"fft2d.with_{fft_backend}"

    #Parametros para el forzamiento
    params.forcing.enable = True
    params.forcing.type = "tcrandom"
    params.forcing.nkmin_forcing = physical_params['KF'] - physical_params['DKF']
    params.forcing.nkmax_forcing = physical_params['KF'] + physical_params['DKF']
    params.forcing.forcing_rate = physical_params['EPSILON TARGET']

    #Viscocidad a pequena escala
    params.nu_2 = physical_params['NU2']
    params.nu_4 = 0.0 #Se desactiva la hiperviscocidad, esta se implementa luego con un filtro gausiano

    params.time_stepping.USE_CFL = True
    params.time_stepping.t_end = t_end

    params.output.periods_save.phys_fields = 0.5
    params.output.periods_save.spatial_means = 0.05
    params.output.phys_fields.field_to_plot = "rot"

    return params

def gaussian_damping(physical_params, sim):
    K = sim.oper.K  # magnitud de k, misma forma que los campos espectrales

    alpha_omega = physical_params['ALPHA OMEGA']
    k0_omega = physical_params['K0 OMEGA']
    sigma2_omega = physical_params['SIGMA SQUARED OMEGA']

    d_omega_gauss = alpha_omega * np.exp(
        -((K - k0_omega) ** 2) / (2 * sigma2_omega)
    )

    def compute_freq_diss_paper():
        f_d = sim.params.nu_2 * sim.oper.K2
        f_d_hypo = d_omega_gauss
        return f_d, f_d_hypo

    sim.compute_freq_diss = compute_freq_diss_paper
    print("Amortiguamiento gaussiano implementado")

def run_simulation(N, t_end, injection_rate=13.05, fft_backend='fftwmpi2d', use_mpi=True):
    '''
    Corre una simulacion completa para una resolucion N dada,
    construye los parametros fisicos y numericos, instala el amortiguamiento
    gaussiano de gran escala, y avanza la integracion temporal hasta t_end.
    '''
    # Construccion de parametros
    physical_params = generate_physical_parameters(injection_rate)
    params = params_construction(physical_params, N, t_end, fft_backend, use_mpi)

    # Creacion del objeto de simulacion
    sim = Simul(params)

    # Amortiguamiento gaussiano (reemplaza nu_m4 default)
    gaussian_damping(physical_params, sim)
    path_run = sim.output.path_run
    print(f"N={N} | path_run={path_run}")

    # Integracion temporal -- esto es lo que faltaba
    sim.time_stepping.start()

    print(f"Simulacion terminada. Datos guardados en: {path_run}")

    return sim


if __name__ == "__main__":
    #test local: serial, sin MPI
    #sim = run_simulation(N=512, t_end=0.2, injection_rate=13.05,fft_backend=None, use_mpi=False)
    #Para cluster:
    sim = run_simulation(N=512, t_end=0.2, injection_rate=13.05,fft_backend='fftwmpi2d', use_mpi=True)