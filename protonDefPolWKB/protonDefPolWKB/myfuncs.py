def Potential_Coul_multip(r,Z,R0,e0,epsilon0,beta2,beta4):
    """Coulomb potential with multipole expansion for deformed nucleus
    V_Coul_multip(r, Z, R0, e0, epsilon0, beta2, beta4)
    
    Inputs:
    r : float or np.array
        Radial distance [fm]
    Z : int
        Atomic number of daughter nucleus
    R0 : float
        Spherical equivalent radius of daughter nucleus [fm]
    e0 : float
        Elementary charge [C]
    epsilon0 : float
        Vacuum permittivity [C^2/(N m^2)]
    beta2, beta4 : float
        Deformation parameter
        
    Returns:
    V_Coul_multip : float or np.array
        Coulomb potential with multipole expansion [MeV]
        0th, 2nd, and 4th order components
        V_Coul_multip = V_Coul_0 + V_Coul_2 * Y20(theta) + V_Coul_4 * Y40(theta)
    """

    import numpy as np
    Y00=lambda theta: 1/np.sqrt(4*np.pi)
    Y20=lambda theta: np.sqrt(5/(16*np.pi))*(3*np.cos(theta)**2-1)
    Y40=lambda theta: (3/(16*np.sqrt(np.pi)))*(35*np.cos(theta)**4-30*np.cos(theta)**2+3)

    Rcoul_theta = lambda theta: R0 * (1 + beta2 * Y20(theta) + beta4 * Y40(theta))
    
    # def Klambda(lam,r,Rcoul):
    #     [Rc_mes,r_mes]=np.meshgrid(Rcoul,r)
    #     if lam==2:
    #         return np.piecewise(
    #             r_mes,
    #             [r_mes < Rc_mes, r_mes >= Rc_mes],
    #             [lambda r_mes: r_mes**2/5+r_mes**2*np.log(Rc_mes/r_mes),
    #              lambda r_mes: 1/5*Rc_mes**5/r_mes**3]
    #         )
    #     else:
    #         return np.piecewise(
    #             r_mes,
    #             [r_mes < Rc_mes, r_mes >= Rc_mes],
    #             [lambda r_mes: (2*lam+1)*r_mes**2/(lam+3)/(lam-2)-1/(lam-2)*r_mes**lam/Rc_mes**(lam-2),
    #              lambda r_mes: 1/(lam+3)*Rc_mes**(lam+3)/r_mes**(lam+1)]
    #         )

    def Klambda(lam,r,Rcoul):
        Rc_mes,r_mes=np.meshgrid(Rcoul,r)
        res=np.ones(np.shape(r_mes))
        if lam==2:
            res[r_mes < Rc_mes]= r_mes[r_mes < Rc_mes]**2/5+r_mes[r_mes < Rc_mes]**2*np.log(Rc_mes[r_mes < Rc_mes]/r_mes[r_mes < Rc_mes])
            res[r_mes >=Rc_mes]= 1/5*Rc_mes[r_mes >=Rc_mes]**5/r_mes[r_mes >=Rc_mes]**3
        else:
            res[r_mes < Rc_mes]= (2*lam+1)*r_mes[r_mes < Rc_mes]**2/(lam+3)/(lam-2)-1/(lam-2)*r_mes[r_mes < Rc_mes]**lam/Rc_mes[r_mes < Rc_mes]**(lam-2)
            res[r_mes >= Rc_mes]= 1/(lam+3)*Rc_mes[r_mes >= Rc_mes]**(lam+3)/r_mes[r_mes >= Rc_mes]**(lam+1)
        return res
         
    theta_grid=np.linspace(0, np.pi/2, 300)
    dtheta_grid=theta_grid[1]-theta_grid[0]

    # order 0
    V_Coul_0 = np.piecewise(
        r,
        [r < R0, r >= R0],
        [lambda r: Z * e0**2 / (4 * np.pi * epsilon0) / (e0 * 1e6) * (3 * R0**2 - r**2) / (2 * R0**3 * 1e-15),
         lambda r: Z * e0**2 / (4 * np.pi * epsilon0) / (r * 1e-15) / (e0 * 1e6)]
    )
    # order 2
    lam=2
    V_Coul_2 =1/(4*np.pi*epsilon0)/1e-15/(e0*1e6)*3*Z*e0**2/(R0**3)*2*np.pi/(2*lam+1)*2*np.trapezoid(Klambda(lam,r,Rcoul_theta(theta_grid))*Y20(theta_grid)*np.sin(theta_grid)*dtheta_grid)
    
    # order 4
    lam=4
    V_Coul_4 =1/(4*np.pi*epsilon0)/1e-15/(e0*1e6)*3*Z*e0**2/(R0**3)*2*np.pi/(2*lam+1)*2*np.trapezoid(Klambda(lam,r,Rcoul_theta(theta_grid))*Y40(theta_grid)*np.sin(theta_grid)*dtheta_grid)

    return V_Coul_0,V_Coul_2,V_Coul_4