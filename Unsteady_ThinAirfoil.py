import numpy as np
from matplotlib import pyplot as plt
from mesh import mesh
from solution import solution
import matplotlib.cm as cm
from Generic_Functions import*
from functools import partial
#%% Inputs
N = 10;
dt = 0.35;
Ndt = 200;
k_arr = np.array([0.02, 0.05, 0.1]);			# 0.02 0.05 0.1
A = np.array([0.006, 0.015, 0.035]);				# 0.006 0.015 0.035
gust_amps = np.array([0.01, 0.08, 0.15]);
N_arr = np.array([1, 5, 10, 15, 20]);
dt_arr = np.array([0.65, 0.5, 0.35, 0.2, 0.1]);
gust_delay = 10;
delta = 0;
delta_arr = np.radians(np.array([5, 8, 10]));
U0 = 1.0;
airfoil = '0009';
alpha_steady = np.radians(np.arange(-15.0, 15.0 + 2.5, 2.5));
alpha_qury = [-5.0, 0.0, 5.0];
c = 1;
save_anim = False;
anim_plot = False;
save_plots = False;
N_period = 3;
vinf = lambda t: U0*np.ones_like(t);
alpha = lambda t, a0, t_delay: 0.0*np.ones_like(t) + a0*(t >= t_delay);
Omega = lambda t, k, a: a*(np.cos((2*k*U0/c)*t));
main_args = [airfoil, N, c, delta, U0, alpha_steady];
#%% Main Function
def main(airfoil, N, c, delta, U0, alpha_arr, dt, Ndt, vinf, alpha, Omega, unsteady = True):
	Mesh = mesh(airfoil, NACA4Camber_line, N, c);
	Mesh.GenMesh();
	Mesh.deflect_falp(delta);

	Solution = solution(Mesh, dt, Ndt);
	Solution.init_FlowField();
	Solution.Solve_Stready_Flow(U0, alpha_arr);
	if (unsteady): Solution.time_march(vinf, alpha, Omega);
	return Mesh, Solution;

def Run_k_iters(main_args, dt, k_arr, A, vinf, alpha, Omega, N_period, alpha_qury, save_plots):
	Solutions = np.zeros(k_arr.shape, dtype = object);
	Meshes = np.zeros(k_arr.shape, dtype = object);
	alpha_k = partial(alpha, a0 = 0, t_delay = 0);
	for i, k in enumerate(k_arr):
		Omega_k = partial(Omega, k = k, a = A[i]);
		Ndt = int((N_period*np.pi/(k*main_args[4]/main_args[2]))/dt);
		Meshes[i], Solutions[i] = main(*main_args, dt, Ndt, vinf, alpha_k, Omega_k);
		Generate_Unsteady_Plots(Solutions[i], k, save_plots);
		# if i == 0: 
		# 	Genetrate_Steady_Plots(Solutions[i], alpha_qury, save_plots);
	if save_plots: plt.close('all');
	return Meshes, Solutions;

def Run_Gust_cases(main_args, gust_amps, gust_delay, dt, Ndt, vinf, alpha, Omega, save_plots):
	Solutions = np.zeros(gust_amps.shape, dtype = object);
	Meshes = np.zeros(gust_amps.shape, dtype = object);
	Omega_i = partial(Omega, k = 0, a = 0);
	for i, gust_amp in enumerate(gust_amps):
		alpha_i = partial(alpha, a0 = gust_amp, t_delay = gust_delay);
		Meshes[i], Solutions[i] = main(*main_args, dt, Ndt, vinf, alpha_i, Omega_i);
		Generate_Gust_Plots(Solutions[i], gust_amp, gust_delay, save_plots)
	if save_plots: plt.close('all');
	return Meshes, Solutions;

def Run_Sensitivity(main_args, N_arr, dt_arr, save_plots, k = 0.25, dt = dt, N_period = N_period, vinf = vinf, alpha = partial(alpha, a0 = 0, t_delay = 0), Omega = partial(Omega, a = 0.035)):
	Omega = partial(Omega, k = k);
	Ndt = int((N_period*np.pi/(k*main_args[4]/main_args[2]))/dt)
	fig = plt.figure();		ax = fig.add_subplot(111);
	main_args_i = main_args.copy();
	markers = ['-x', '-o', '-^', '-d', '-v'];
	for i, N in enumerate(N_arr):
		main_args_i[1] = N;
		_, soln = main(*main_args_i, dt, Ndt, vinf, alpha, Omega);
		Generic_single_1D_plot(soln.t, soln.lift, r'$t$ [s]', r'$C_l$ [-]', save_plots, 'Sensitivity_N_k%0.2f'%k, line_label = r'$N$ = %i'%(N), ax = ax, marker = markers[i], marker_size = 2.5);
	# dt Convergence
	fig = plt.figure();		ax = fig.add_subplot(111);
	for i, dt_i in enumerate(dt_arr):
		Ndt = int((N_period*np.pi/(k*main_args[4]/main_args[2]))/dt_i);
		_, soln = main(*main_args, dt_i, Ndt, vinf, alpha, Omega);
		Generic_single_1D_plot(soln.t, soln.lift, r'$t$ [s]', r'$C_l$ [-]', save_plots, 'Sensitivity_dt_k%0.2f'%k, line_label = r'$\Delta t$ = %0.2f'%(dt_i), ax = ax, marker = markers[i], marker_size = 2.5);
	if save_plots: plt.close('all');
	return 0;

def Run_Flap_cases(main_args, delta_arr, save_plots, unsteady = False, alpha_qury = alpha_qury):
	main_args_i = main_args.copy();
	fig = plt.figure();		ax1 = fig.add_subplot(111);
	markers = ['-x', '-o', '-^'];
	for i, delta in enumerate(delta_arr):
		main_args_i[3] = delta;
		_, soln = main(*main_args_i, 0, 0, 0, 0, 0, unsteady = unsteady);
		Generic_single_1D_plot(np.degrees(soln.alpha_steady), soln.lift_steady, r'$\alpha \: [^\circ]$', r'$C_l$ [-]', save_plots, '%s_Cl_vs_alpha_steady_flap'%(soln.mesh.airfoil), line_label = r'$\delta_f = %0.1f^{\circ}$'%(np.degrees(delta)), ax = ax1, marker = markers[i], marker_size = 5);
		# Genetrate_Steady_Plots(soln, alpha_qury, save_plots, flap = True, delta = delta);
	if save_plots: plt.close('all');
	return 0;

if __name__ == '__main__':
	Meshes, Solutions = Run_k_iters(main_args, dt, k_arr, A, vinf, alpha, Omega, N_period, alpha_qury, save_plots);
	# Meshes, Solutions = Run_Gust_cases(main_args, gust_amps, gust_delay, dt, Ndt, vinf, alpha, Omega, save_plots);
	# Run_Sensitivity(main_args, N_arr, dt_arr, save_plots);
	# Run_Flap_cases(main_args, delta_arr, save_plots);
	# Plots
	if (anim_plot):
		Animate_Solution(Solutions[-1], save_anim);
	plt.show();