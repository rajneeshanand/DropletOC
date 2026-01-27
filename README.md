# Optimal_Transport_and_Control_of_a_Drop_in_a_Microchannel
# 1. Mass Conservation — Weak and Variational Forms

We begin with the governing conservation law for the film height \( h(x,t) \) and flux \( q(x,t) \):

$$
\frac{\partial h}{\partial t} + \frac{\partial q}{\partial x} = 0,
\qquad x \in \Omega.
$$

Here:
- \(h(x,t)\): film thickness,
- \(q(x,t)\): volumetric flux,
- \(\Omega\): 1D domain (e.g., \([0,L]\)).

---

## 1.1 Weak Form

Multiply the PDE by a **test function** \(v(x)\) and integrate over \(\Omega\):

$$
\int_\Omega 
\frac{\partial h}{\partial t}\, v \, dx
+
\int_\Omega
\frac{\partial q}{\partial x}\, v \, dx
= 0.
$$

### Integration by parts

For the second term,

$$
\int_\Omega \frac{\partial q}{\partial x} \, v \, dx
=
- \int_\Omega q \, v_x \, dx
+
\left[ q\,v \right]_{\partial\Omega}.
$$

Assuming **no–flux boundary condition**

$$
q = 0 \quad \text{on } \partial\Omega,
$$

the boundary term vanishes:

$$
\left[ q\,v \right]_{\partial\Omega}=0.
$$

Thus the weak form is:

$$
\int_\Omega 
\frac{\partial h}{\partial t}\, v \, dx
-
\int_\Omega 
q \, v_x\, dx
= 0.
$$

---

## 1.2 Time Discretization (Backward Euler)

Approximate the time derivative with:

$$
\frac{\partial h}{\partial t}
\approx 
\frac{h^{n+1} - h^{n}}{\Delta t}.
$$

Insert into the weak form:

$$
\int_\Omega 
\frac{h^{n+1} - h^{n}}{\Delta t} \, v \, dx
-
\int_\Omega 
q^{\,n+1} \, v_x\, dx
= 0.
$$

We now treat \( h^{n+1}(x) \) as the **trial function** to be solved.

---

## 1.3 Auxiliary Equation for the Second Derivative

To compute curvature terms involving \(h_{xx}\), introduce an auxiliary variable:

$$
\text{d2h}(x) = h_{xx}(x).
$$

The strong form is:

$$
\text{d2h} - h_{xx} = 0.
$$

Multiply by a test function \(u(x)\) and integrate:

$$
\int_\Omega \text{d2h}\, u\, dx
-
\int_\Omega h_{xx}\, u\, dx
= 0.
$$

### Integration by parts

$$
- \int_\Omega h_{xx}\, u\, dx
=
\int_\Omega h_x\, u_x\, dx
-
\left[ h_x\,u \right]_{\partial\Omega}.
$$

Assuming natural BCs \(h_x = 0\) on \(\partial\Omega\):

$$
\left[ h_x\,u \right]_{\partial\Omega} = 0.
$$

Final weak form:

$$
\int_\Omega \text{d2h}\, u\, dx
+
\int_\Omega h_x\, u_x\, dx
= 0,
\qquad \forall u(x).
$$

Here **d2h** is the trial function that approximates the second derivative \(h_{xx}\).
