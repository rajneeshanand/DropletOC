# Optimal_Transport_and_Control_of_a_Drop_in_a_Microchannel
## 1. Mass conservation and weak form

We start from the mass conservation equation for the film height \(h(x,t)\) and flux \(q(x,t)\):

\[
\frac{\partial h}{\partial t} + \frac{\partial q}{\partial x} = 0 \quad \text{in } \Omega .
\]

Here:
- \(h(x,t)\) = film/thickness (or height) as a function of space and time,  
- \(q(x,t)\) = volumetric flux in the \(x\)-direction,  
- \(\Omega\) = spatial domain in 1D, e.g. \(\Omega = (0,L)\).

### 1.1 Weak form

To derive the weak form, we multiply the PDE by a **test function** \(v(x)\) (smooth and arbitrary, but vanishing on the appropriate boundary if needed) and integrate over \(\Omega\):

\[
\int_{\Omega} \frac{\partial h}{\partial t} \, v \, dx
\;+\;
\int_{\Omega} \frac{\partial q}{\partial x} \, v \, dx
= 0.
\]

We now integrate the second term by parts:

\[
\int_{\Omega} \frac{\partial q}{\partial x} \, v \, dx
=
- \int_{\Omega} q \, \frac{\partial v}{\partial x} \, dx
+ \left[ q\,v \right]_{\partial \Omega}.
\]

If we assume a **no–flux boundary condition** on \(\partial\Omega\),

\[
q = 0 \quad \text{on } \partial \Omega,
\]

then the boundary term vanishes:

\[
\left[ q\,v \right]_{\partial \Omega} = 0.
\]

Therefore, the weak form of mass conservation becomes

\[
\int_{\Omega} \frac{\partial h}{\partial t} \, v \, dx
\;-\;
\int_{\Omega} q \, \frac{\partial v}{\partial x} \, dx
= 0
\quad \text{for all test functions } v(x).
\]

This is the starting point for our time discretization and finite element formulation.

### 1.2 Time discretization (backward Euler)

We now discretize the time derivative using a backward Euler step from time level \(n\) to \(n+1\):

\[
\frac{\partial h}{\partial t}
\approx
\frac{h^{n+1} - h^{n}}{\Delta t}.
\]

Substituting this into the weak form gives

\[
\int_{\Omega} \frac{h^{n+1} - h^n}{\Delta t} \, v \, dx
\;-\;
\int_{\Omega} q^{\theta} \, \frac{\partial v}{\partial x} \, dx
= 0.
\]

Here:
- \(h^n(x)\) is the known solution at the previous time step,
- \(h^{n+1}(x)\) is the unknown solution at the new time step,
- \(q^{\theta}\) denotes the flux evaluated at some appropriate time level (e.g. fully implicit, \(\theta=1\), or semi-implicit).

For convenience we now **rename** \(h^{n+1}\) simply as \(h\), to emphasize that it is the new unknown (the *trial function*) we want to solve for:

\[
\int_{\Omega} \frac{h - h^n}{\Delta t} \, v \, dx
\;-\;
\int_{\Omega} q(h,\dots)\, \frac{\partial v}{\partial x} \, dx
= 0.
\]

> **Key idea:**  
> - \(v(x)\) is the **test function** (arbitrary, used to derive the weak form).  
> - \(h(x)\) is the **trial function**, i.e. the unknown solution at the new time step that we approximate in a finite element space.

---

## 2. Auxiliary equation for the second derivative

To handle second derivatives like \(h_{xx}\) in a finite element framework, we introduce an **auxiliary variable**

\[
\text{d2h}(x) = \frac{\partial^2 h}{\partial x^2} = h_{xx}.
\]

We enforce this definition through the *strong form* equation

\[
\text{d2h} - h_{xx} = 0 \quad \text{in } \Omega.
\]

### 2.1 Weak form of the auxiliary equation

Again, multiply by a test function \(u(x)\) and integrate over the domain:

\[
\int_{\Omega} \text{d2h} \, u \, dx
\;-\;
\int_{\Omega} h_{xx} \, u \, dx
= 0.
\]

We integrate the second term by parts:

\[
- \int_{\Omega} h_{xx} \, u \, dx
=
\int_{\Omega} h_x \, u_x \, dx
- \left[ h_x u \right]_{\partial \Omega}.
\]

If we assume a boundary condition such that the boundary term vanishes (for example \(h_x = 0\) on \(\partial\Omega\)), then

\[
\left[ h_x u \right]_{\partial \Omega} = 0,
\]

and the weak form for the auxiliary equation becomes

\[
\int_{\Omega} \text{d2h} \, u \, dx
\;+\;
\int_{\Omega} h_x \, u_x \, dx
= 0
\quad \text{for all test functions } u(x).
\]

In this formulation:

- \(u(x)\) is the **test function** for the auxiliary problem,
- \(\text{d2h}(x)\) is treated as the **trial function**, i.e. the unknown field that approximates \(h_{xx}\).

This auxiliary equation lets us compute a finite element approximation of the second derivative \(h_{xx}\) in a weak sense, which is useful in higher-order PDEs (such as thin-film equations).
