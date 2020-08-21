Discrete Approximation
===================================

To solve the system of equations defined above, we use finite differences, similarly to most modelling tools. The system is now defined over a grid :math:`x = x_i` for  :math:`0 \leq i \leq N`, where a difference is made between material properties and variables defined over the grid points :math:`x_i` and the slabs  between consecutive:math:`x_i \rightarrow x_{i+1}` over which the current densities are defined and computed. Thus, material parameters and variables defoned throughout the system are now vectors: :math:`u(x) \rightarrow \{u(x_i)\}_{0\leq i \leq N}`.
For the discretization of the current, we use the Scharfetter-Gummel scheme over the "slabs", which defines trial functions for the current to evaluate the gradient of the currents that appears in the continuity equations, in order to ensure numerical convergence.
In this work, we we followthe discretization as outlined in SESAME, where the currents are defined as:

.. math::

    \begin{cases} 
    \Psi_{n, i} &= q\phi_i + \chi_i + k_BT\ln(N_{c, i})\\
    \Psi_{p, i} &= q\phi_i + \chi_i + E_{g, i}- k_BT\ln(N_{v, i}) \\
    J_n^{i \rightarrow i+1}  &= -\frac{q\mu_{n, i}}{x_{i+1} - x_i}\frac{\Psi_{n, i+1} - \Psi_{n, i}}{e^{-\frac{\Psi_{n, i+1}}{k_BT}} - e^{-\frac{\Psi_{n, i}}{k_BT}}}\Big[e^{\frac{E_{f, n, i+1}}{k_BT} - \frac{E_{f, n, i}}{k_BT}}\Big] \\
    J_p^{i \rightarrow i+1}  &= \frac{q\mu_{p, i}}{x_{i+1} - x_i}\frac{\Psi_{p, i+1} - \Psi_{p, i}}{e^{\frac{\Psi_{p, i+1}}{k_BT}} - e^{\frac{\Psi_{p, i}}{k_BT}}}\Big[e^{\frac{E_{f, p, i+1}}{k_BT}} - e^{\frac{E_{f, p, i}}{k_BT}}\Big] \\
    \end{cases} 

Therefore, the discretized gradient of the current is:

.. math::
    \begin{align}
        \frac{dJ_n}{dx}\Big | _{i} &= \frac{J_n^{i \rightarrow i+1} - J_n^{i - 1 \rightarrow i}}{\frac{x_{i+1} - x_{i-1}}{2}} \\
        \frac{dJ_p}{dx}\Big | _{i} &= \frac{J_p^{i \rightarrow i+1} - J_p^{i - 1 \rightarrow i}}{\frac{x_{i+1} - x_{i-1}}{2}} \\
    \end{align}

Finally, the discretized Laplacian in the left side of the Poisson equation is simply (where the divergence is taken as a central derivative):

.. math::
    \frac{d(\epsilon \frac{d\phi}{dx})}{dx} | _{i} &= \frac{1}{\frac{x_{i+1} - x_{i-1}}{2}}\Big [\normalsize\frac{\epsilon_{i + 1} + \epsilon_i}{2}\frac{\phi_{i+1} - \phi_i}{x_{i+1}- x_i} - \frac{\epsilon_i + \epsilon_{i-1}}{2}\frac{\phi_{i} - \phi_{i-1}}{x_i- x_{i-1}}\Big] \\ 

The system of differential equations becomes a set of equations for the zeros of the :math:`F` function of :math:`3N` variables :math:`\{E_{f, n, i}, E_{f, p, i}, \phi_i\}_{0\leq i \leq N} = \{u_i\}_{0\leq i \leq N}, F:\mathbf{R}^{3N}\rightarrow\mathbf{R}^{3N}`, where 6 components of F are the discretized boundary conditions, and the other 3(N - 2) are the discretized continuity and Poisson equations.
This problem is solved using the Newton-Raphson method, which is an iterative method which, starting from an initial guess :math:`u^0` takes successive steps until convergence:

.. math::

    u^{i + 1} = u^i + [J\{F\}(u^i)]^{-1} \cdot F(u^i)

where :math:`J\{F\}` is the Jacobian matrix of :math:`F`.