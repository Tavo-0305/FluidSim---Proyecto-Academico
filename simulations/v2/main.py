import argparse
import os
import time

import numpy as np

# Backend sin pantalla (el cluster no tiene display). Debe fijarse antes de
# que se importe matplotlib; fluidsim solo lo importa si se grafica algo.
os.environ.setdefault("MPLBACKEND", "Agg")

from fluiddyn.util.mpi import rank, nb_proc
from fluidsim.solvers.ns2d.solver import Simul


def generate_physical_parameters(injection_rate=13.05):
    '''
    Esta función genera un diccionario de python con los parámetros físicos
    de la simulación, todos se mantienen fijos a excepción del "injection rate"
    de energía.
    '''
    phys_parameters = {
        'KF': 200.0,  # numero de onda de forzamiento
        'DKF': 4.0,  # ancho de banda
        'NU2': 5e-5,  # viscosidad a pequena escala
        'EPSILON TARGET': injection_rate,
        'ALPHA OMEGA': 500.0,  # factor de friccion de gran escala
        'K0 OMEGA': 0.1,  # centro del perfil gaussiano
        'SIGMA SQUARED OMEGA': 1.0,  # varianza del perfil gaussiano
    }
    return phys_parameters


class SimulGaussDamping(Simul):
    '''
    Solver ns2d con amortiguamiento gaussiano de gran escala:

        d_omega(k) = alpha * exp(-(k - k0)^2 / (2 sigma^2))

    IMPORTANTE: fluidsim llama a compute_freq_diss() DENTRO de Simul.__init__
    (al construir el integrador temporal y los coeficientes lineales exactos).
    Por eso el amortiguamiento se define sobreescribiendo el metodo en una
    subclase, y NO con un monkey-patch posterior a Simul(params): en ese caso
    el integrador nunca vería el amortiguamiento.

    Los parametros fisicos se pasan como atributo de clase (_physical_params)
    antes de instanciar, porque Simul.__init__ no admite argumentos extra.
    '''

    _physical_params = None

    def compute_freq_diss(self):
        pp = type(self)._physical_params
        if pp is None:
            raise RuntimeError(
                "Hay que asignar SimulGaussDamping._physical_params antes de "
                "crear la simulacion."
            )

        # self.oper.K tiene la forma LOCAL del subdominio de cada rank, igual
        # que los campos espectrales, asi que funciona tal cual bajo MPI.
        K = self.oper.K

        f_d = self.params.nu_2 * self.oper.K2
        f_d_hypo = pp['ALPHA OMEGA'] * np.exp(
            -((K - pp['K0 OMEGA']) ** 2) / (2 * pp['SIGMA SQUARED OMEGA'])
        )
        return f_d, f_d_hypo


def params_construction(
    physical_params,
    N,
    t_end,
    fft_backend='fftw1d',
    use_mpi=True,
    sub_directory='ns2d_gaussian_damping',
    period_phys_fields=0.5,
    max_elapsed=None,
    restart_file=None,
):
    '''
    Esta funcion construye los params del simulador para una resolucion N dada
    '''
    params = SimulGaussDamping.create_default_params()

    # Se define la malla
    params.oper.nx = N
    params.oper.ny = N
    # Se utiliza un dominio cuadrado periodico
    params.oper.Lx = 2 * np.pi
    params.oper.Ly = 2 * np.pi
    # Metodo para eliminar aliasing en metodos pseudoespectrales
    params.oper.coef_dealiasing = 2 / 3

    # Se configura el backend FFT:
    if fft_backend is None:
        pass
    elif use_mpi:
        params.oper.type_fft = f"fft2d.mpi_with_{fft_backend}"
    else:
        params.oper.type_fft = f"fft2d.with_{fft_backend}"

    # Parametros para el forzamiento
    params.forcing.enable = True
    params.forcing.type = "tcrandom"
    params.forcing.nkmin_forcing = physical_params['KF'] - physical_params['DKF']
    params.forcing.nkmax_forcing = physical_params['KF'] + physical_params['DKF']
    params.forcing.forcing_rate = physical_params['EPSILON TARGET']

    # Viscosidad a pequena escala
    params.nu_2 = physical_params['NU2']
    # Se desactiva la hiperviscosidad, la disipacion de gran escala se
    # implementa con el perfil gaussiano (ver SimulGaussDamping)
    params.nu_4 = 0.0

    params.time_stepping.USE_CFL = True
    params.time_stepping.t_end = t_end
    # Parada ordenada: fluidsim guarda el estado final y termina limpiamente
    # al cumplirse este tiempo de pared (formato "HH:MM:SS").
    if max_elapsed is not None:
        params.time_stepping.max_elapsed = max_elapsed

    # Salida
    params.output.sub_directory = sub_directory
    params.output.periods_save.phys_fields = period_phys_fields
    params.output.periods_save.spatial_means = 0.05
    params.output.phys_fields.field_to_plot = "rot"

    # Reinicio desde un estado guardado (state_phys_t*.h5 o .nc). Se restaura el
    # tiempo de la simulacion; t_end sigue siendo un tiempo ABSOLUTO.
    if restart_file is not None:
        params.init_fields.type = "from_file"
        params.init_fields.from_file.path = restart_file

    return params


def run_simulation(
    N,
    t_end,
    injection_rate=13.05,
    fft_backend='fftw1d',
    use_mpi=True,
    **kwargs,
):
    '''
    Corre una simulacion completa para una resolucion N dada: construye los
    parametros fisicos y numericos, usa el solver con amortiguamiento gaussiano
    de gran escala, y avanza la integracion temporal hasta t_end.
    '''
    if use_mpi and fft_backend is not None and N % nb_proc != 0:
        raise ValueError(
            f"N={N} no es divisible por el numero de procesos MPI ({nb_proc}); "
            "la descomposicion en slabs de fluidfft lo requiere."
        )

    physical_params = generate_physical_parameters(injection_rate)
    params = params_construction(
        physical_params, N, t_end, fft_backend, use_mpi, **kwargs
    )

    # Los parametros del amortiguamiento deben estar fijados ANTES de Simul()
    SimulGaussDamping._physical_params = physical_params
    sim = SimulGaussDamping(params)

    path_run = sim.output.path_run
    if rank == 0:
        print(f"N={N} | procesos MPI={nb_proc} | FFT={sim.oper.type_fft}")
        print(f"path_run={path_run}")
        print("Amortiguamiento gaussiano implementado")

    t0 = time.time()
    sim.time_stepping.start()

    if rank == 0:
        print(f"Simulacion terminada en {(time.time() - t0) / 60:.1f} min. "
              f"Datos guardados en: {path_run}")

    return sim


def parse_args():
    p = argparse.ArgumentParser(description="NS 2D forzado con amortiguamiento gaussiano")
    p.add_argument("--N", type=int, default=1024, help="resolucion (N x N)")
    p.add_argument("--t-end", type=float, default=600.0, help="tiempo final (absoluto)")
    p.add_argument("--injection-rate", type=float, default=13.05)
    p.add_argument("--fft-backend", default="fftw1d",
                   help="fftw1d (por defecto, el mismo de CLLJ_simulation.py) "
                        "o fftwmpi2d si esta instalado en el entorno")
    p.add_argument("--serial", action="store_true",
                   help="sin MPI (test local); usa el backend FFT por defecto de fluidsim")
    p.add_argument("--sub-directory", default="ns2d_gaussian_damping")
    p.add_argument("--period-phys-fields", type=float, default=0.5,
                   help="cada cuanto tiempo de simulacion se guardan los campos fisicos")
    p.add_argument("--max-elapsed", default=None,
                   help='tiempo de pared maximo "HH:MM:SS"; guarda y sale limpio')
    p.add_argument("--restart-file", default=None,
                   help="ruta a un archivo state_phys_t*.h5 (o .nc) para reanudar")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_simulation(
        N=args.N,
        t_end=args.t_end,
        injection_rate=args.injection_rate,
        fft_backend=None if args.serial else args.fft_backend,
        use_mpi=not args.serial,
        sub_directory=args.sub_directory,
        period_phys_fields=args.period_phys_fields,
        max_elapsed=args.max_elapsed,
        restart_file=args.restart_file,
    )
