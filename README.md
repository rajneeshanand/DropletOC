# Optimal_Transport_and_Control_of_a_Drop_in_a_Microchannel
# 1. Mass Conservation — Weak Form Derivation

We begin with the mass conservation equation for a height field \( h(x,t) \) and flux \( q(x,t) \):

$$
\frac{\partial h}{\partial t} + \frac{\partial q}{\partial x} = 0,
\qquad x \in \Omega.
$$

Here:
- \( h(x,t) \): film thickness (or height),
- \( q(x,t) \): volumetric flux in the \(x\)-direction,
- \( \Omega \): 1D spatial domain (e.g. \([0,L]\)).

---

## 1.1 Weak Form

To derive the weak form, multiply the PDE by a **test function** \( v(x) \) and integrate over the domain:

$$
\int_\Omega 
\frac{\partial h}{\partial t} \, v \, dx
\;+\;
\int_\Omega
\frac{\partial q}{\partial x} \, v \, dx
= 0.
$$

Now integrate the second term by parts:

$$
\int_\Omega \frac{\partial q}{\partial x} v \, dx
=
- \int_\Omega q \, \frac{\partial v}{\partial x} \, dx
+ \left[ q\,v \right]_{\partial\Omega}.
$$

Assuming **no-flux boundary conditions**

$$
q = 0 \quad \text{on } \partial\Omega,
$$

the boundary term vanishes:

$$
\left[ q\,v \right]_{\partial\Omega} = 0.
$$

Therefore, the weak form becomes:

$$
\int_\Omega 
\frac{\partial h}{\partial t} \, v \, dx
\;-\;
\int_\Omega 
q \, v_x \, dx
= 0,
\qquad \forall v(x).
$$

---

## 1.2 Time Discretization (Backward Euler)

Approximate the time derivative using

$$
\frac{\partial h}{\partial t}
\approx
\frac{h^{n+1} - h^n}{\Delta t}.
$$

Substituting gives:

$$
\int_\Omega 
\frac{h^{n+1} - h^n}{\Delta t} \, v \, dx
\;-\;
\int_\Omega 
q^{n+1} \, v_x \, dx
= 0.
$$

We now treat \( h^{n+1} \) as the **trial function** (unknown at the new time step).  
Thus the semi-discrete weak form is:

$$
\int_\Omega 
\frac{h - h^n}{\Delta t} \, v \, dx
\;-\;
\int_\Omega 
q(h) \, v_x \, dx
= 0.
$$

---

## 1.3 Auxiliary Second-Derivative Equation

To compute the curvature term \( h_{xx} \), we introduce an auxiliary variable

$$
\text{d2h}(x) = h_{xx}(x).
$$

The strong form is:

$$
\text{d2h} - h_{xx} = 0.
$$

Multiply by a test function \( u(x) \) and integrate:

$$
\int_\Omega \text{d2h} \, u\, dx
-
\int_\Omega h_{xx} \, u \, dx
= 0.
$$

Integrate the second term by parts:

$$
- \int_\Omega h_{xx} \, u \, dx
=
\int_\Omega h_x \, u_x \, dx
-
\left[ h_x u \right]_{\partial\Omega}.
$$

Under natural boundary conditions \( h_x = 0 \) at \( \partial\Omega \):

$$
\left[ h_x u \right]_{\partial\Omega} = 0.
$$

So the weak form becomes:

$$
\int_\Omega \text{d2h}\, u \, dx
+
\int_\Omega h_x \, u_x \, dx
= 0,
\qquad \forall u(x),
$$

where **d2h** is the trial function approximating \(h_{xx}\).

