# Optimal_Transport_and_Control_of_a_Drop_in_a_Microchannel
# Numerical Methods for Partial Differential Equations

## 1. Mass Conservation (Weak Form Derivation)

The starting point is the **Strong Form** of the continuity equation:

$$\frac{\partial h}{\partial t} + \frac{\partial q}{\partial x} = 0$$

### Step 1: Weighted Residual Method
Multiply by a **test function** $v(x)$ and integrate over the domain $\Omega$:

$$\int_{\Omega} \frac{\partial h}{\partial t} v \, dx + \int_{\Omega} \frac{\partial q}{\partial x} v \, dx = 0$$

### Step 2: Integration by Parts
To reduce the continuity requirement on $q$, we shift the derivative to the test function $v$:

$$\int_{\Omega} \frac{\partial h}{\partial t} v \, dx - \int_{\Omega} q \frac{\partial v}{\partial x} \, dx + \left[ qv \right]_{\partial\Omega} = 0$$

> **Boundary Condition:** Assuming no flux on the boundary ($q=0$), the term $[qv]_{\partial\Omega}$ vanishes.

### Step 3: Temporal Discretization
Using a finite difference approximation for the time derivative:
$$\frac{\partial h}{\partial t} \approx \frac{h^{n+1} - h^n}{\Delta t}$$

Substituting this into the weak form:
$$\int_{\Omega} \left( \frac{h^{n+1} - h^n}{\Delta t} \right) v \, dx - \int_{\Omega} q^\theta v_x \, dx = 0$$

* **Trial Function:** $h$ (The unknown for the new time step).
* **Test Function:** $v$.

---

## 2. Auxiliary Equation (Second-Order Terms)

To handle second-order derivatives like $h_{xx}$, we define an auxiliary variable $d2h$:

**Strong Form:**
$$d2h - \frac{\partial^2 h}{\partial x^2} = 0$$

### Weak Form Derivation
1. **Weighted Integral:**
   $$\int_{\Omega} (d2h)u \, dx - \int_{\Omega} h_{xx} u \, dx = 0$$

2. **Integration by Parts:**
   $$\int_{\Omega} (d2h)u \, dx + \int_{\Omega} h_x u_x \, dx - [h_x u]_{\partial\Omega} = 0$$

3. **Final Auxiliary Weak Form:**
   $$\int_{\Omega} h_x u_x \, dx + \int_{\Omega} (d2h)u \, dx = 0$$

* **Trial Function:** $d2h$.
