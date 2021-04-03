System of Equations
===================================

=================
Internal points 
=================

For internal points on the grid, the values of :math:`phi`, :math:`E_{f, n}`, and :math:`E_{f, p}` satisfying the following three equations are solved for.

.. math::

    \begin{cases}
    \epsilon_0 \overrightarrow{\nabla} \cdot (\epsilon \overrightarrow{\nabla} \phi) &= q(n - p + N_a - N_d) \\
    \overrightarrow{\nabla} \cdot \overrightarrow{J_n} &= -q(G-R) \\ 
    \overrightarrow{\nabla} \cdot \overrightarrow{J_p} &= q(G-R)
    \end{cases} 

where, 

* :math:`n` and :math:`p` are defined as

    .. math::

        \begin{cases}
        n &= N_ce^{\frac{E_{f,n} + \chi + q\phi}{k_BT}} \\
        p &= N_ve^{\frac{-E_{f,p} - \chi - E_g - q\phi}{k_BT}}\
        \end{cases} 

* :math:`\overrightarrow{J_n}` and :math:`\overrightarrow{J_p}` are defined as

    .. math::

        \begin{cases}
        \overrightarrow{J_n} &= q\mu_nn:`E_{f, n}` \\
        \overrightarrow{J_p} &= q\mu_pp`E_{f, p}`
        \end{cases} 

* :math:`R` is defined as

    .. math::

        \begin{cases}
        R &= R_{radiative} + R_{Auger} + R_{SHR}\\
        n_{int} &= \sqrt{N_cN_v}e^{\frac{E_g}{2k_BT}} \\
        R_{radiative} &= B(jnp - n_{int}^2) \\ 
        R_{Auger} &= (C_nn + C_pp)(jnp-n_{int}^2) \\
        R_{SHR} &= \frac{jnp-n_{int}^2}{t_n(n+n_{int}e^{\frac{E_{SHR}}{k_BT}}) + t_p(p+n_{int}e^{-\frac{E_{SHR}}{k_BT}})} 
        \end{cases}

    where :math:`R_{radiative}`, :math:`R_{Auger}`, and :math:`R_{SHR}` are the radiative, Auger, and Shockley-Read-Hall (SHR) recombination rate density, and :math:`n_{int}` is the intrinsic carrier density.


=================
Boundary points 
=================

The boundary conditions differ based on whether or not the circuit is open / in equilibrium, meaning there is no net flow of charge carriers, or if it is closed / out of equilibrium, meaning that there is a voltage being applied.

--------------
In equilibrium
--------------

In equilibrium, the quasi-Fermi energies are constant across the system and are set to 0: :math:`E_{f,n} = E_{f,p} = 0`, and Dirichlet boundary conditions are applied for :math:`\phi` depending on if the contacts are n-doped or p-doped.

.. math::
        \phi(x = 0, L) = 
        \begin{cases}
        -\chi + k_BT\ln(\frac{N_d}{N_c}) & \mbox{if contact n-doped} \\ 
       -\chi - E_g - k_BT\ln(\frac{N_a}{N_v}) & \mbox{if contact p-doped}
        \end{cases}

------------------
Not in equilibrium
------------------

When the system is out of equilibrium and we impose a voltage V , we require Dirichlet boundary conditions on the three variables defined as follows, where :math:`n_{eq}` and  :math:`p_{eq}` are the equilibrium electron/hole densities, and :math:`S_{n, 0/L}` and :math:`S_{p, 0/L}` are the electron and hole surface recombination velocities for :math:`x = 0, L`.

.. math::
         
        \begin{cases}
        \phi(x = 0) &= \phi_{eq}(x = 0) \\  
        \phi(x = L) &= \phi_{eq}(x = L) + V \\  
        J_n(x = 0) &= qS_{n, 0}(n(x=0) - n_{eq}(x=0)) \\  
        J_n(x = 0) &= -qS_{n, L}(n(x=L) - n_{eq}(x=L)) \\  
        J_p(x = 0) &= -qS_{p, 0}(p(x=0) - p_{eq}(x=0)) \\  
        J_p(x = L) &= qS_{p, L}(p(x=L) - p_{eq}(x=L)) \\ 
        \end{cases}

