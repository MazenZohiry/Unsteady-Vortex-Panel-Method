# Unsteady-Vortex-Panel-Method

In unsteady aerodynamics, the load over the airfoil changes resulting in changing the vorticity distribution and hence the circulation. In order to ensure that the impermeability condition is satisfied, vortices are shed in the wake according to Kelvin's theorem. The shed vortices induce velocities everywhere in the domain including the surface of the airfoil. These induced velocities result in a lag that is experienced in the unsteady cases. The model presented is used to investigate a number of steady and unsteady flow cases, namely a steady flat plate, a steady flat plate with a flap, a pitching flat plate and finally a flat plate under a gust.

![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/64721988/189983786-edffd7bb-0aa1-4afe-b74c-4a0463037b5b.gif)


# Model Walkthrough

The implementation of the Vortex Panel Method (VPM) was done in Python using several scripts, the first one being *mesh.py* which builds a Python object containing the control points, the normal and the tangential unit vectors at each control point, and the positions of the vortices. The other script named *solution.py* builds another object and contains the methods to construct the matrices $\mathbf{A}$ and $\mathbf{A}^\prime$, solving both the steady and unsteady cases as well as plotting the solution. The time marching procedure is performed in *solution.py* where the new $\Gamma$ distribution is updated and the new velocity and pressure field are computed. Lastly, the script *ThinAirfoil.py* contains the definition of the unsteady motion, the reduced frequency, and the call to the mesh and solution classes where most of the computations are performed. The flow chart below describes the algorithm.

<img src="https://user-images.githubusercontent.com/64721988/191044639-56f94c03-9c9a-4c67-8390-7a84c44e4776.png" width=40% height=40%>

 
