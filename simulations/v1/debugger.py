from  fluidsim import load_sim_for_plot
path_run = "/home/gustavo/Sim_data/NS2D_1024x1024_S2pix2pi_2026-09-28_18-37-18"
sim = load_sim_for_plot(path_run)

#print(sim.output.phys_fields.get_field_to_plot('ux'))
ux_test1, t_test1 = sim.output.phys_fields.get_field_to_plot('ux') #Devuelve una tupla
'''
print(ux_test1)
print(t_test1)

print(dir(sim.output))
print(dir(sim.output.phys_fields))

ux, t_ux = sim.output.phys_fields.get_field_to_plot("ux", time=10)
uy, t_uy = sim.output.phys_fields.get_field_to_plot("uy", time=10)

print("ux:", ux.shape)
print("uy:", uy.shape)
print("sim.oper:", sim.oper.shapeX_loc)
'''

#print(sim.output.phys_fields.get_key_field_to_plot())

#print(sim.params)
#print(sim.params.oper)
rot, t_rot = sim.output.phys_fields.get_field_to_plot("rot", time=10)
#print(rot)
#print(t_rot)
'''
from fluidsim.operators.operators2d import OperatorsPseudoSpectral2D

# Usamos los parámetros completos de la simulación
params = sim.params

# Creamos un operador completo
oper_full = OperatorsPseudoSpectral2D(params)

print("Operador original:", sim.oper.shapeX_loc)
print("Operador nuevo:", oper_full.shapeX_loc)


print("=== sim.params.oper ===")
print("nx:", sim.params.oper.nx)
print("ny:", sim.params.oper.ny)
print("Lx:", sim.params.oper.Lx)
print("Ly:", sim.params.oper.Ly)

print("\n=== sim.oper ===")
print("shapeX_loc:", sim.oper.shapeX_loc)
print("nx:", sim.oper.nx)
print("ny:", sim.oper.ny)
'''
from fluidsim.operators.operators2d import OperatorsPseudoSpectral2D
from fluidsim.solvers.ns2d.solver import Simul

# Crear parámetros nuevos desde cero
params_full = Simul.create_default_params()

# Usar la resolución real de la simulación
params_full.oper.nx = 1024
params_full.oper.ny = 1024

# Mantener el mismo dominio
params_full.oper.Lx = sim.params.oper.Lx
params_full.oper.Ly = sim.params.oper.Ly

# Mantener el mismo tipo de FFT
params_full.oper.type_fft = sim.params.oper.type_fft

# Crear operador de resolución completa
oper_full = OperatorsPseudoSpectral2D(params_full)

print("Operador original:")
print("  nx =", sim.oper.nx)
print("  ny =", sim.oper.ny)

print("\nOperador nuevo:")
print("  nx =", oper_full.nx)
print("  ny =", oper_full.ny)
print("  shapeX_loc =", oper_full.shapeX_loc)

# Campos de velocidad en t = 10
ux, t_ux = sim.output.phys_fields.get_field_to_plot("ux", time=10)
uy, t_uy = sim.output.phys_fields.get_field_to_plot("uy", time=10)

# Transformadas de Fourier
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

# Comprobamos tamaños
print("dux_dx:", dux_dx.shape)
print("dux_dy:", dux_dy.shape)
print("duy_dx:", duy_dx.shape)
print("duy_dy:", duy_dy.shape)