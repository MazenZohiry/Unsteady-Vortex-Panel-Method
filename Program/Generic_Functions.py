import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as ani
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def NACA4Camber_line(x, max_camber, pos_camber):
	""" Function to generate camber line for NACA 4 digit serie airfoil
		using explicit function of chord wise station.
	"""
	x0 = x[np.where(x < pos_camber)];
	x1 = x[np.where(x >= pos_camber)];
	x_f = max_camber/pos_camber**2*(2*pos_camber*x0 - x0**2);
	x_b = max_camber/(1 - pos_camber)**2*(1 - 2*pos_camber + 2*pos_camber*x1 - x1**2);
	z = np.concatenate((x_f, x_b));
	return z;

def Generic_single_1D_plot(x, y, xlabel, ylabel, save_plots, name, line_label = None, linewidth = 0.75, ax = None, marker = '-x', marker_size = 5, alpha = 1, xscale = 'linear', yscale = 'linear'):
	if ax == None:	fig = plt.figure(); ax = fig.add_subplot(111);
	ax.plot(x, y, marker, label = line_label, mfc = 'none', linewidth = linewidth, markersize = marker_size, alpha = alpha);
	ax.set_xscale(xscale);
	ax.set_yscale(yscale);
	ax.grid(True);
	ax.minorticks_on();
	ax.grid(b = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2);
	ax.set_xlabel('%s'%(xlabel), fontsize = 13);
	ax.set_ylabel('%s'%(ylabel), fontsize = 13);
	if line_label != None:
		plt.legend(loc = 'best');
	if (save_plots):
		plt.savefig('./Figures/{}.png'.format(name), dpi = 300, bbox_inches = 'tight');
	return ax;

def Animate_Solution(Solution, save_anim):
	idx = 0;
	fig = plt.figure(figsize = (12, 8));
	ax = fig.add_subplot(111);
	im, quv, line, time = Solution.Plot_Flow_Field(idx, fig = fig, ax = ax);
	anim = ani.FuncAnimation(fig, Solution.Animate_flow_field, Solution.Ndt, interval = 50, fargs = (im, quv, line, time));
	if save_anim:
		anim.save('./Flow_field.gif');
	plt.show();
	return 0;

def Genetrate_Steady_Plots(Solution, alpha_qury, save_plots, flap = False, delta = 0):
	alp_idxs = np.where(np.in1d(np.degrees(Solution.alpha_steady), alpha_qury))[0];
	for idx in alp_idxs:
		Solution.Plot_Flow_Field(idx, unsteady = False, save_name = '%s_Flow_Field_steady_aoa_%0.1f_delta_%0.2f'%(Solution.mesh.airfoil, np.degrees(Solution.alpha_steady[idx]), np.degrees(delta)), save_plot = save_plots);
		Solution.Plot_Flow_Field(idx, unsteady = False, plot_pressure = True, save_name = '%s_Flow_Field_steady_pressure_aoa_%0.1f_delta_%0.2f'%(Solution.mesh.airfoil, np.degrees(Solution.alpha_steady[idx]), np.degrees(delta)), save_plot = save_plots);
	if flap == False:
		polar_data = np.genfromtxt('NACA%s.dat'%(Solution.mesh.airfoil));
		exp_data = np.genfromtxt('NACA0012_Exp.csv', delimiter = ',');
		ax = Generic_single_1D_plot(np.degrees(Solution.alpha_steady), Solution.lift_steady, '', '', False, '', marker = '-o', line_label = 'VPM Results', marker_size = 5);
		Generic_single_1D_plot(np.degrees(Solution.alpha_steady), 2*np.pi*Solution.alpha_steady, r'$\alpha \: [^\circ]$', r'$C_l$ [-]', save_plots, '%s_Cl_vs_alpha_steady_lone'%(Solution.mesh.airfoil), marker = '-x', line_label = 'Thin airfoil theory', ax = ax, marker_size = 5);
		ax = Generic_single_1D_plot(np.degrees(Solution.alpha_steady), Solution.lift_steady, '', '', False, '', marker = '-o', line_label = 'VPM Results', marker_size = 5);
		Generic_single_1D_plot(polar_data[:, 0], polar_data[:, 1], '', '', False, '', ax = ax, marker = '-*', line_label = 'XFoil Results', marker_size = 2.5);
		Generic_single_1D_plot(exp_data[:, 0], exp_data[:, 1], r'$\alpha \: [^\circ]$', r'$C_l$ [-]', save_plots, '%s_Cl_vs_alpha_steady'%(Solution.mesh.airfoil), ax = ax, marker = '-^', line_label = 'Experimental Results', marker_size = 5);
	return 0;

def Generate_Unsteady_Plots(Solution, k, save_plots):
	Solution.Plot_Flow_Field(Solution.Ndt - 1, unsteady = True, save_name = '%s_Flow_Field_unsteady_k_%0.3f'%(Solution.mesh.airfoil, k), save_plot = save_plots);

	ax = Generic_single_1D_plot(Solution.alpha*180/np.pi, Solution.lift, '', '', False, '', alpha = 0.75, marker = '--r', marker_size = 4, line_label = 'Unsteady solution', linewidth = 0.85);
	Generic_single_1D_plot(Solution.alpha*180/np.pi, 2*np.pi*(Solution.alpha + Solution.omega/2), '', '', False, '', alpha = 0.75, ax = ax, marker = '--g', marker_size = 5, line_label = 'Quasi-Steady solution', linewidth = 0.85);
	Generic_single_1D_plot(Solution.alpha_steady[2:-2]*180/np.pi, Solution.lift_steady[2:-2], '', '', False, '', ax = ax, marker = '-k^', marker_size = 5, line_label = 'Steady solution', linewidth = 0.85);
	Generic_single_1D_plot(Solution.alpha[0]*180/np.pi, 2*np.pi*(Solution.alpha + Solution.omega/2)[0], '', '', False, '', ax = ax, marker = '^g', marker_size = 8, line_label = 'Start Quasi-Steady');
	Generic_single_1D_plot(Solution.alpha[-1]*180/np.pi, 2*np.pi*(Solution.alpha + Solution.omega/2)[-1], '', '', False, '', ax = ax, marker = 'og', marker_size = 8, line_label = 'End Quasi-Steady');
	Generic_single_1D_plot(Solution.alpha[0]*180/np.pi, Solution.lift[0], '', '', False, '', marker = '^r', marker_size = 8, ax = ax, line_label = 'Start Unsteady');
	Generic_single_1D_plot(Solution.alpha[-1]*180/np.pi, Solution.lift[-1], r'$\alpha \: [^{\circ}]$', r'$C_l$ [-]', save_plots, '%s_Cl_vs_alpha_t_k_%0.3f'%(Solution.mesh.airfoil, k), marker = 'or', marker_size = 8, ax = ax, line_label = 'End Unsteady');

	ax = Generic_single_1D_plot(Solution.t, Solution.theta, '', '', False, '', line_label = r'$\theta(t)$', marker_size = 3);
	Generic_single_1D_plot(Solution.t, Solution.omega, r'$t$ [s]', r'$\theta$ [rad],    $\Omega$ [rad/s]', save_plots, '%s_Omega_theta_vs_t_k_%0.3f'%(Solution.mesh.airfoil, k), ax = ax, marker = '-o', marker_size = 3.5, line_label = r'$\Omega(t)$');

	ax = Generic_single_1D_plot(Solution.t, Solution.lift, r'$t$ [s]', r'$C_l$ [-]', save_plots, '%s_Cl_vs_t_k_%0.3f'%(Solution.mesh.airfoil, k), marker_size = 3.5);

	ax = Generic_single_1D_plot(Solution.t, np.degrees(Solution.alpha), '', '', False, '', line_label = r'$\alpha_{s}$', marker_size = 1);
	Generic_single_1D_plot(Solution.t, np.degrees(Solution.alpha + Solution.omega/2), '', '', False, '', ax = ax, line_label = r'$\alpha_{qs}$', marker = '-o', marker_size = 1);
	Generic_single_1D_plot(Solution.t, np.degrees(Solution.alpha_unsteady), r'$t$ [s]', r'$\alpha \: [^{\circ}]$', save_plots, 'Alpha_sd_unsd_k_%0.3f'%(k), ax = ax, line_label = r'$\alpha_{eq}$', marker = '-^', marker_size = 1);
	return 0;

def Generate_Gust_Plots(Solution, gust_amp, gust_delay, save_plots):
	ax = Generic_single_1D_plot(Solution.t, Solution.lift, '', '', False, '', marker_size = 3.5, line_label = 'Numerical Solution');
	Generic_single_1D_plot(Solution.t, (Solution.t > gust_delay)*2*np.pi*np.sin(gust_amp)*(1 - 0.5*(np.exp(-0.13*2*(Solution.t - gust_delay)) + np.exp(-2*(Solution.t - gust_delay)))), r'$t$ [s]', r'$C_l$ [-]', save_plots, '%s_gust_responce_amp_%0.3f_delay_%0.3f'%(Solution.mesh.airfoil, gust_amp, gust_delay), ax = ax, marker = '-o', marker_size = 3.5, line_label = 'Analytical Solution');
	t_idx = np.where(Solution.t > gust_delay)[0];
	Solution.Plot_Flow_Field(t_idx[0] + 2, unsteady = True, save_name = '%s_Flow_Field_gust_amp_%0.3f_delay_%0.3f'%(Solution.mesh.airfoil, gust_amp, gust_delay), save_plot = save_plots);
	Solution.Plot_Flow_Field(1, unsteady = True, save_name = '%s_Flow_Field_gust_t0_amp_%0.3f_delay_%0.3f'%(Solution.mesh.airfoil, gust_amp, gust_delay), save_plot = save_plots);
	return 0;

