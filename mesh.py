import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt

class mesh():
	def __init__(self, airfoil, camber_func, N, c):
		self.camber_func = camber_func;											# Camber line function handel
		self.N = N;																# Number of panels
		self.c = c;																# Chord length
		self.x = None;															# Camber x positions
		self.z = None;															# Camber z positions
		self.control_point = None;												# Control points
		self.Gamma_pos = None;													# Lump points
		self.unit_n = None;
		self.unit_t = None;														# Unit normal vector
		self.loc_slope = None;													# Local slope of each panel
		self.a = None;															# Length of each panel
		self.rotation_matrix = self.Rotate_Mat(np.pi/2);							# Unit axis transformation matrix in z direction
		self.pivot = np.array([[c/4, 0]]);
		self.max_camber = int(airfoil[0])/100;						# Max camber
		self.pos_camber = max(int(airfoil[1])/10, 0.01);				# Max camber location
		self.airfoil_xz = np.genfromtxt('NACA%s_xz.dat'%airfoil);
		self.airfoil = airfoil;
		self.flap_pos = 0.8*c;
		self.flap_idx = None;
		self.airf_flap_idx = np.where(self.airfoil_xz[:, 0] >= self.flap_pos)[0];

	def Rotate_Mat(self, theta):
		T = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]);		# Rotation matrix in z direction
		return T;

	def GenMesh(self):
		""" Function to discretise camber line. First generates 
			camber line with explicit function, then discretises into N panels
		"""
		self.x = np.linspace(0, self.c, self.N + 1);
		self.z = self.camber_func(self.x, self.max_camber, self.pos_camber);
		self.flap_idx = np.where(self.x >= self.flap_pos)[0];
		self.Gamma_pos = np.zeros((self.N, 2));
		self.control_point = np.zeros_like(self.Gamma_pos);
		self.unit_n = np.zeros((self.N, 2));
		self.unit_t = np.zeros_like(self.unit_n);
		self.loc_slope = np.zeros(self.N);
		self.a = np.zeros(self.N);
		for i in range(self.N):
			dx = self.x[i + 1] - self.x[i];		dz = self.z[i + 1] - self.z[i];
			self.loc_slope[i] = np.arctan2(dz, dx);
			self.a[i] = np.sqrt(dx**2 + dz**2);
			u = np.array([dx, dz]);
			self.Gamma_pos[i] = 0.25*u + np.array([self.x[i], self.z[i]]);
			self.control_point[i] = 0.75*u + np.array([self.x[i], self.z[i]]);
			self.unit_n[i] = self.rotation_matrix.dot(u)/norm(u);
			self.unit_t[i] = u/norm(u);
		return 0;

	def deflect_falp(self, delta):
		T = self.Rotate_Mat(-delta);
		x0 = self.x[self.flap_idx[0]];
		z0 = self.z[self.flap_idx[0]];
		X0 = np.array([[x0], [z0]]);
		xz_deflect = T@(np.vstack((self.x[self.flap_idx], self.z[self.flap_idx])) - X0) + X0;
		self.x[self.flap_idx] = xz_deflect[0, :];
		self.z[self.flap_idx] = xz_deflect[1, :];

		self.Gamma_pos[self.flap_idx[:-1], :] = (T@(self.Gamma_pos[self.flap_idx[:-1], :].T - X0) + X0).T;
		self.control_point[self.flap_idx[:-1], :] = (T@(self.control_point[self.flap_idx[:-1], :].T - X0) + X0).T;

		self.unit_n[self.flap_idx[:-1], :] = (T@(self.unit_n[self.flap_idx[:-1], :].T)).T;
		self.unit_t[self.flap_idx[:-1], :] = (T@(self.unit_t[self.flap_idx[:-1], :].T)).T;

		self.loc_slope[self.flap_idx[:-1]] = np.arctan2((self.z[self.flap_idx][1:] - self.z[self.flap_idx][:-1]), (self.x[self.flap_idx][1:] - self.x[self.flap_idx][:-1]));
		
		self.airfoil_xz[self.airf_flap_idx, :] = (T@(self.airfoil_xz[self.airf_flap_idx, :].T - X0) + X0).T
		return 0;

	def Plot_CamberLine(self, save_plot):
		fig = plt.figure();
		ax = fig.add_subplot(111);
		ax.plot(self.x, self.z, label = "Camber Line");
		ax.plot(self.Gamma_pos[:, 0], self.Gamma_pos[:, 1], "o", label = "Vortex");
		ax.plot(self.control_point[:, 0], self.control_point[:, 1], "x", label = "Control Point");
		ax.set_xlabel("x");
		ax.set_ylabel("z");
		ax.legend(loc = "lower right");
		ax.set_ylim([-0.2, 0.2]);
		ax.grid(True);
		ax.quiver(self.control_point[:, 0], self.control_point[:, 1], self.unit_n[:, 0], self.unit_n[:, 1], units = "xy", angles = "uv", pivot = "tail");
		ax.set_aspect('equal', adjustable = 'box');
		if save_plot == True:
			plt.savefig('./Figures/%s_Camber.png'%(self.airfoil), dpi = 300, bbox_inches = "tight")
		return 0;
