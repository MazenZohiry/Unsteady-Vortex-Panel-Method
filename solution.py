import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv, norm
import matplotlib.cm as cm

class solution():
	def __init__(self, mesh, dt, Ndt):
		self.mesh = mesh;						# Mesh object
		self.A = None;							# System matrix
		self.Atan = None;						# System matrix tan
		self.gamma_dist = np.zeros((self.mesh.N, Ndt));					# Gamma distribution at each panel
		self.gamma_shed = np.zeros(Ndt);
		self.gamma_steady = None;						# Steady Gamma value
		self.lift_steady = None;
		self.gamma_pos = np.zeros((Ndt, 2, Ndt));
		self.lift = np.zeros(Ndt);
		self.theta = np.zeros(Ndt);
		self.alpha = np.zeros(Ndt);
		self.vinf = np.zeros(Ndt);
		self.omega = np.zeros(Ndt);
		self.alpha_steady = None;
		self.alpha_unsteady = np.zeros(Ndt);
		self.vinf_steady = None;
		self.t = np.zeros(Ndt);
		self.dt = dt;
		self.Ndt = Ndt;
		# Bounds for plot
		self.x_in = -0.5;
		self.x_out = 3.5;
		self.z_max = 1;
		self.z_min = -1;
		self.x_in_sd = -0.5;
		self.x_out_sd = 1.5;
		self.z_max_sd = 1;
		self.z_min_sd = -1;
		self.fieldRes = 60;
		self.fieldRes_sd = 50;

	def Compute_norm_tan_velocities(self, control_points, bound_points):
		N = control_points.shape[0];
		M = bound_points.shape[0];
		vi_normal = np.zeros((N, M));
		vi_tan =  np.zeros((N, M));
		for i in range(N):
			cp = control_points[i];
			for j in range(M):
				Gp = bound_points[j];
				r = Gp - cp;
				norm_r = 1/norm(r);
				vi = self.mesh.rotation_matrix.dot(r)*norm_r;
				vi_normal[i][j] = 1/(2*np.pi)*norm_r*vi.dot(self.mesh.unit_n[i]);
				vi_tan[i][j] = 1/(2*np.pi)*norm_r*vi.dot(self.mesh.unit_t[i]);
		return vi_normal, vi_tan;

	def Assemble_System_matrix(self):
		""" Assembles matrix A for the linear system 
		"""
		self.A, self.Atan = self.Compute_norm_tan_velocities(self.mesh.control_point, self.mesh.Gamma_pos);
		return 0;

	def Assemble_ith_System_matrix(self, j):
		if j == 0: self.Assemble_System_matrix();
		vi_normal_shed, _ = self.Compute_norm_tan_velocities(self.mesh.control_point, self.gamma_pos[j, :, j][np.newaxis, :]);
		Ai = np.hstack((self.A, vi_normal_shed));
		Ai = np.vstack((Ai, np.ones(Ai.shape[1])));
		return Ai;

	def Shed_induced_vel(self):
		M = len(self.gamma_shed[self.gamma_shed != 0]);
		if M > 0:
			vi_normal_shed, vi_tan_shed = self.Compute_norm_tan_velocities(self.mesh.control_point, self.gamma_pos[M, :, 0:M].T);
			vi_normal_shed = vi_normal_shed@self.gamma_shed[self.gamma_shed != 0];
			vi_tan_shed = vi_tan_shed@self.gamma_shed[self.gamma_shed != 0];
		else:
			vi_normal_shed = [0.0];
			vi_tan_shed = [0.0];
		return vi_normal_shed, vi_tan_shed;

	def Get_applied_norm_tan_velocities(self, v_inf, alpha, omega, unsteady = True):
		v_inf_comp = v_inf*np.sin(alpha - np.arctan(self.mesh.loc_slope));
		v_inf_tan = v_inf*np.cos(alpha - np.arctan(self.mesh.loc_slope));
		if (unsteady):
			ind_comp, ind_comp_tan = self.Shed_induced_vel();
			v_rot = np.cross(np.array([0, 0, omega]), np.hstack((self.mesh.control_point - self.mesh.pivot, np.zeros((self.mesh.N, 1)))), axisa = 0, axisb = 1);
			v_rot_normal = np.sum(v_rot[:, 0:2]*self.mesh.unit_n, axis = 1);
			v_rot_tan = np.sum(v_rot[:, 0:2]*self.mesh.unit_t, axis = 1);
			net_norm_v = np.concatenate((-v_inf_comp - v_rot_normal - ind_comp, np.array([-np.sum(self.gamma_shed)])));
			net_tan_v = ((ind_comp_tan + v_inf_tan + v_rot_tan)*self.mesh.unit_t.T).T;
			return net_norm_v, net_tan_v, ind_comp;
		else:
			return -v_inf_comp, (v_inf_tan*self.mesh.unit_t.T).T;
	
	def Circulatory_loads(self, Gammas, net_tan_v):
		net_tan_v += ((self.Atan@Gammas)*self.mesh.unit_t.T).T;
		v_tan_vect = np.hstack((net_tan_v, np.zeros((net_tan_v.shape[0], 1))));
		gamma_vect = np.vstack((np.zeros((2, len(Gammas))), Gammas)).T;
		Lift_vect = sum(np.cross(gamma_vect, v_tan_vect, axisa = 1, axisb = 1), 0);
		Lift = norm(Lift_vect)*np.sign(Lift_vect[1]);
		return Lift;

	def time_march(self, vinf, alpha, omega):
		for i in range(self.Ndt):
			if i > 0:
				self.theta[i] = self.theta[i - 1] + omega(i*self.dt)*self.dt;
			else:
				self.theta[i] = alpha(i*self.dt);
			self.t[i] = i*self.dt;
			self.alpha[i] = alpha(i*self.dt) + self.theta[i];
			self.vinf[i] = vinf(i*self.dt);
			self.omega[i] = omega(i*self.dt);
			self.gamma_pos[i:, 0, i] = self.mesh.x[-1] + self.vinf[i]*np.cos(self.alpha[i])*self.dt;
			self.gamma_pos[i:, 1, i] = self.mesh.z[-1] + self.vinf[i]*np.sin(self.alpha[i])*self.dt;

			A = self.Assemble_ith_System_matrix(i);

			b, net_tan_v, ind_comp = self.Get_applied_norm_tan_velocities(self.vinf[i], self.alpha[i], omega(i*self.dt));
			self.alpha_unsteady[i] = np.arctan2(np.sin(self.alpha[i]) + ind_comp[0], np.cos(self.alpha[i]));
			Gammas = inv(A)@b;
			self.gamma_dist[:, i] = Gammas[:-1];
			self.gamma_shed[i] = Gammas[-1];
			self.lift[i] = self.Circulatory_loads(Gammas[:-1], net_tan_v)/(0.5*self.vinf[i]**2*self.mesh.c);
			if i > 0:
				self.gamma_pos[i:, :, 0:i] += self.vinf[i]*self.dt*np.array([[np.cos(self.alpha[i])], [np.sin(self.alpha[i])]]);
		return 0;

	def Solve_Stready_Flow(self, vinf, alpha_arr):
		self.alpha_steady = alpha_arr;
		self.gamma_steady = np.zeros((alpha_arr.shape[0], self.mesh.N));
		self.lift_steady = np.zeros_like(alpha_arr);
		self.vinf_steady = vinf;
		self.Assemble_System_matrix();
		for i, alpha in enumerate(alpha_arr):
			b, tan_v = self.Get_applied_norm_tan_velocities(vinf, alpha, 0, unsteady = False);
			Gammas = inv(self.A)@b;
			self.gamma_steady[i, :] = Gammas;
			self.lift_steady[i] = self.Circulatory_loads(Gammas, tan_v)/(0.5*vinf**2*self.mesh.c);
		return 0;

	def Compute_Velocity_Potential(self, idx):
		u, v = self.Compute_Field_velocities(idx);
		phi = np.zeros_like(self.x_mesh);
		phi[:, 0] = v[:, 0]*self.z_mesh[:, 0] + u[:, 0]*self.x_mesh[0, 0];
		delta_x = abs(self.x_mesh[0, 0] - self.x_mesh[0, 1]);
		phi[0, :] = u[0, :]*self.x_mesh[0, :] + v[0, :]*self.z_mesh[0, 0];
		delta_z = abs(self.z_mesh[0, 0] - self.z_mesh[1, 0]);
		for i in range(1, self.fieldRes):
			for j in range(self.fieldRes - 1):
				phi[i, j + 1] = u[i, j]*delta_x + v[i, j]*delta_z + phi[i - 1, j];
		return phi;

	def Compute_Field_velocities(self, idx, core = 1e-2, unsteady = True):
		if (unsteady):
			gamma_positions = np.vstack((self.mesh.Gamma_pos, self.gamma_pos[idx][:, :idx].T));
			gammas = np.concatenate((self.gamma_dist[:, idx], self.gamma_shed[:idx]));
			vinf = self.vinf[idx];
			field_nodes = self.field_nodes;
			shp = self.x_mesh.shape;
			alpha = self.alpha[idx];
		else:
			gamma_positions = self.mesh.Gamma_pos;
			gammas = self.gamma_steady[idx, :];
			vinf = self.vinf_steady;
			field_nodes = self.field_nodes_sd;
			shp = self.x_mesh_sd.shape;
			alpha = self.alpha_steady[idx];

		R = np.asarray([field_nodes - r for r in gamma_positions]);
		R_mag = norm(R, axis = 2);		R_mag = R_mag.reshape(np.shape(R)[0], np.shape(R)[1], 1);
		R /= R_mag;
		theta = R[:, :, [1, 0]];		theta[:, :, 1] *= -1;
		V = np.asarray([theta[i]*gammas[i]/(2*np.pi*R_mag[i]) for i in range(R.shape[0])]);
		V = np.sum(V, axis = 0);
		V_core = max(np.absolute(gammas))/(2*np.pi*core);
		V[norm(V, axis = 1) >= V_core, :] = 0.0;
		u_uni = vinf*np.cos(alpha)*np.ones(shp);
		v_uni = vinf*np.sin(alpha)*np.ones(shp);
		u = u_uni + V[:, 0].reshape(shp);		v = v_uni + V[:, 1].reshape(shp);
		return u, v;

	def init_FlowField(self):
		x = np.linspace(self.x_in, self.x_out, self.fieldRes);
		z = np.linspace(self.z_min, self.z_max, self.fieldRes);
		x_sd = np.linspace(self.x_in_sd, self.x_out_sd, self.fieldRes_sd);
		z_sd = np.linspace(self.z_min_sd, self.z_max_sd, self.fieldRes_sd);
		self.x_mesh, self.z_mesh = np.meshgrid(x, z);
		self.x_mesh_sd, self.z_mesh_sd = np.meshgrid(x_sd, z_sd);
		self.field_nodes = np.vstack((self.x_mesh.reshape(1, -1)[0], self.z_mesh.reshape(1, -1)[0])).T;
		self.field_nodes_sd = np.vstack((self.x_mesh_sd.reshape(1, -1)[0], self.z_mesh_sd.reshape(1, -1)[0])).T;
		return 0;

	def Plot_Flow_Field(self, idx, unsteady = True, plot_pressure = False, fig = None, ax = None, save_name = None, save_plot = False):
		if fig == None and (unsteady):
			fig = plt.figure(figsize = (12, 8));
			ax = fig.add_subplot(111);
		elif fig == None and not (unsteady):
			fig = plt.figure(figsize = (8, 9));
			ax = fig.add_subplot(111);
		if (unsteady): 
			vinf = self.vinf[idx]; 	
			time = ax.text(0.05, 1.005, r't = %.3f s'%(self.t[idx]), transform = ax.transAxes, fontsize = 15);
			x_mesh = self.x_mesh;
			z_mesh = self.z_mesh;
			line = ax.plot(np.concatenate((self.gamma_pos[idx, 0, 0:idx + 1], [self.mesh.x[-1]])), np.concatenate((self.gamma_pos[idx, 1, 0:idx + 1], [self.mesh.z[-1]])), '-wo', alpha = 0.3, linewidth = 0.75)[0];
			ax.set_xlim([self.x_in, self.x_out]);
			ax.set_ylim([self.z_min, self.z_max]);
		else: 
			vinf = self.vinf_steady;
			time = None;
			x_mesh = self.x_mesh_sd;
			z_mesh = self.z_mesh_sd;
			line = None;
			ax.set_xlim([self.x_in_sd, self.x_out_sd]);
			ax.set_ylim([self.z_min_sd, self.z_max_sd]);

		u, v = self.Compute_Field_velocities(idx, unsteady = unsteady);
		
		airfoil_x = self.mesh.airfoil_xz[:, 0];		airfoil_z =  self.mesh.airfoil_xz[:, 1];
		ax.plot(self.mesh.x, self.mesh.z, label = "Camber Line");
		ax.plot(self.mesh.Gamma_pos[:, 0], self.mesh.Gamma_pos[:, 1], "o", label = "Vortex");
		ax.plot(self.mesh.control_point[:, 0], self.mesh.control_point[:, 1], "x", label = "Control Point");
		ax.legend(loc = 'upper right');
		ax.set_xlabel(r'$\frac{x}{c}$ [-]', fontsize = 15);
		ax.set_ylabel(r'$\frac{z}{c}$ [-]', fontsize = 15);
		# ax.set_title(r"Velocity contours around {} at $\alpha$ = {}".format(plot_title, round(np.degrees(alpha[idx]), 2)));

		if plot_pressure: 
			P = 1 - (u**2 + v**2)/(vinf**2);
			im = ax.imshow(P, origin = 'lower', extent = (x_mesh[0, 0], x_mesh[0, -1], z_mesh[0, 0], z_mesh[-1, 0]), interpolation = 'bilinear', cmap = cm.inferno);
			im.set_clim(-1.75, 0.5);
			clab = r'$C_p$ [-]'
		else:
			im = ax.imshow(np.sqrt(u**2 + v**2)/vinf, origin = 'lower', extent = (x_mesh[0, 0], x_mesh[0, -1], z_mesh[0, 0], z_mesh[-1, 0]), interpolation = 'bilinear', cmap = cm.inferno);
			im.set_clim(0.65, 1.35);
			clab = r'$\frac{|\vec{u}|}{u_{\infty}}$ [-]'
		cbar = fig.colorbar(im, orientation = 'horizontal', fraction = 0.06, pad = 0.1, aspect = 35);
		cbar.set_label(clab, fontsize = 15);
		
		quv = ax.quiver(x_mesh, z_mesh, u, v, width = 0.0012);		#angles = "xy", scale_units = "xy" -np.sqrt(u**2 + v**2), cmap = cm.gist_gray
		ax.fill_between(airfoil_x, 0, airfoil_z, facecolor = "white");
		plt.tight_layout();
		if (save_plot):
			plt.savefig('./Figures/%s.png'%(save_name), dpi = 300, bbox_inches = 'tight');
		return im, quv, line, time;

	def Animate_flow_field(self, idx, im, quv, line, time):
		u, v = self.Compute_Field_velocities(idx);
		u_mag = np.sqrt(u**2 + v**2)
		# P = self.Compute_Field_Pressure(v_inf, alpha, idx);
		time.set_text(r't = %.3f s'%(self.t[idx]));
		im.set_data(u_mag);
		line.set_xdata(np.concatenate((self.gamma_pos[idx, 0, 0:idx + 1], [self.mesh.x[-1]])));
		line.set_ydata(np.concatenate((self.gamma_pos[idx, 1, 0:idx + 1], [self.mesh.z[-1]])));
		# im.set_clim(np.amin(P), np.amax(P));
		# im.set_clim(np.amin(u_mag), min(1.5, np.amax(u_mag)));
		quv.set_UVC(u, v);
		return 0;