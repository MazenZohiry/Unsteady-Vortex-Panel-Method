import numpy as np
from matplotlib import pyplot as plt
from mesh import mesh
from solution import solution

#%% Functions
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

def Sort_XfoilCp_data(Cp_data):
	""" Function to read exported Xfoil .dat files
		and return dCp for each .dat file.
	"""
	Cp = np.genfromtxt(Cp_data, comments = "#");
	#splt = where(Cp[:, 0] == 0)[0][0];
	splt = len(Cp[:, 0])//2;
	try:
		dCp_vis = Cp[0:splt, 1] - Cp[splt:, 1][::-1];				# Viscous data
		dCp_invis = Cp[0:splt, -1] - Cp[splt:, -1][::-1];			# Inviscid data
	except:
		diff = len(Cp[0:splt, 1]) - len(Cp[splt:, 1][::-1]);
		dCp_vis = Cp[0:splt, 1] - Cp[splt - diff:, 1][::-1];
		dCp_invis = Cp[0:splt, -1] - Cp[splt - diff:, -1][::-1];
	x = Cp[0:splt, 0];
	return x, dCp_vis, dCp_invis;

#%% Inputs
V_inf = 1;												# Free stream velocity
alpha = np.linspace(np.radians(-12), np.radians(12), 25);		# Range of angles of attack
c = 1;													# Chord length
airfoil = "0009";										# NACA 4 digit code
max_camber = int(airfoil[0])/100;						# Max camber
pos_camber = max(int(airfoil[1])/10, 0.01);				# Max camber location
plot_title = "{}".format(airfoil);						# Title for plot
N = 70;													# Number of panels
N_arr = [10, 20, 30];									# Number of panels for convergence test
save_plot = False;											# Boolien to save plots
file = "NACA{}.dat".format(airfoil);							# Polars
airfoil_file = "NACA{}_xz.dat".format(airfoil);				# Airfoil coordinates
Cp_data_aoa2 = "NACA{}_Cp_aoa2.txt".format(airfoil);			# Cp for aoa 2 deg 
Cp_data_aoa5 = "NACA{}_Cp_aoa5.txt".format(airfoil);			# Cp for aoa 5 deg
Cp_data_aoa8 = "NACA{}_Cp_aoa8.txt".format(airfoil);			# Cp for aoa 8 deg

# Read Xfoil .dat files data
airfoil_xz = np.genfromtxt(airfoil_file);
data = np.genfromtxt(file);
x_Cp, Cp_vis2, Cp_invis2 = Sort_XfoilCp_data(Cp_data_aoa2);
_, Cp_vis5, Cp_invis5 = Sort_XfoilCp_data(Cp_data_aoa5);
_, Cp_vis8, Cp_invis8 = Sort_XfoilCp_data(Cp_data_aoa8);

#%% NACA 0015 data from Table 1
Cp_file_0015 = "Assignment1v2.csv";
Cp_data_0015 = np.genfromtxt(Cp_file_0015, delimiter = ",");

#%% Main
def main(N_arr):
	""" Main function that handels all the objects
		and function calls. Setup discretisation, assemble linear system and solve
	"""
	Mesh = np.zeros(len(N_arr), dtype = object);
	soln = np.zeros(len(N_arr), dtype = object);
	for i in range(len(N_arr)):
		Mesh[i] = mesh(NACA4Camber_line, N_arr[i], c);				# Mesh object containing discretised camber line
		soln[i] = solution(Mesh[i]);								# Solution object, defines linear system and stores solution data
		Mesh[i].GenMesh(max_camber, pos_camber);					# Generate and Discretise camber line
		soln[i].Assemble_System_matrix();							# Assemble linear system A Gamma = v
		soln[i].batchSolve(alpha, V_inf);							# Batch solve for all specified aoa
	return Mesh, soln;

Mesh_objs, soln_objs = main(N_arr);									# main() function call, objects used for convergence study. Uses N_arr
Mesh, soln = main([N]);												# main() function call, objects used to create analyse all final plots. Uses N
Mesh = Mesh[0]; soln = soln[0];

#%% Plots (Individual Ns)
Mesh.Plot_CamberLine(0);																	# Camber line plot
soln.Plot_Cl_alph(data[:, 0], data[:, 1], plot_title, save_plot);							# Cl - alpha plot
soln.FlowField(V_inf, alpha, 24, airfoil_xz, plot_title);									# Generate and plot flow field (velocity contours)
soln.Plot_dCp(V_inf, x_Cp, Cp_vis2, Cp_invis2, alpha, 2, plot_title, save_plot);			# Cp plot 2 deg
soln.Plot_dCp(V_inf, x_Cp, Cp_vis5, Cp_invis5, alpha, 5, plot_title, save_plot);			# Cp plot 5 deg
soln.Plot_dCp(V_inf, x_Cp, Cp_vis8, Cp_invis8, alpha, 8, plot_title, save_plot);			# Cp plot 8 deg
if airfoil[0] == "0":
	soln.Cp_plotNACA0015(V_inf, Cp_data_0015, alpha, plot_title, save_plot);				# SSM plot for NACA 0015


#%% Plots (for all Ns)
#if len(N_arr) > 1:
#	alph = [2, 5, 8];
#	for aoa in alph:
#		fig = plt.figure();
#		plot_title1 = plot_title + "aoa{}Ns".format(aoa);
#		fig.canvas.set_window_title(plot_title1);
#		for i in range(len(soln_objs)):
#			soln = soln_objs[i];
#			soln.Plot_dCpVarN(V_inf, alpha, aoa, fig, N_arr[i])								# Plot Cp solution for each aoa and every element in N_arr
#		plt.grid(True);
#		plt.gca().invert_yaxis();
#		plt.xlabel(r"$\frac{x}{c}$", fontsize = 17);
#		plt.ylabel(r"$\Delta C_p$", fontsize = 13);
#		plt.legend();
#		if save_plot == True:
#			plt.savefig(plot_title1 + ".png", dpi = 300, bbox_inches = "tight");
plt.show();