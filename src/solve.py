from .F import *
if USE_JAX:
    from jax import jit , jacfwd

def damp( move ):
    """
    Computes a damped move of potentials from the Newton method displacement.

    Parameters
    ----------
        move : numpy array , shape = ( 3N )
            displacement in e-/hole quasi-Fermi energy and electrostatic potential computed from the Newton method

    Returns
    -------
        numpy array , shape = ( 3N )
            damped displacement in potentials

    """
    approx_sign = np.tanh( 1e50 * move )
    approx_abs = approx_sign * move
#    approx_H = 1 - ( 1 + np.exp( - 1e40 * ( move**2 - 1 ) ) )**(-1)
#    return np.log( 1 + approx_abs ) * approx_sign + approx_H * ( move - np.log( 1 + approx_abs ) * approx_sign )
    thr = 1
    around_zero = 0.5 * ( np.tanh( 1e50 * ( move + thr ) ) - np.tanh( 1e50 * ( move - thr ) ) )
    return ( 1 - around_zero ) * approx_sign * np.log( 1 + approx_abs ) + around_zero * move




#@jit
def step( dgrid , neq0 , neqL , peq0 , peqL , phis , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G ):
    """
    Computes the next potentials in the Newton method iterative scheme.

    Parameters
    ----------
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        neq0     : float
            e- equilibrium density at left boundary
        neqL     : float
            e- equilibrium density at right boundary
        peq0     : float
            hole equilibrium density at left boundary
        peqL     : float
            hole equilibrium density at right boundary
        phis     : numpy array , shape = ( 3N )
            current potentials ( e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential )
        eps      : numpy array , shape = ( N )
            relative dieclectric constant
        Chi      : numpy array , shape = ( N )
            electron affinity
        Eg       : numpy array , shape = ( N )
            band gap
        Nc       : numpy array , shape = ( N )
            e- density of states
        Nv       : numpy array , shape = ( N )
            hole density of states
        Ndop     : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )
        mn       : numpy array , shape = ( N )
            e- mobility
        mp       : numpy array , shape = ( N )
            hole mobility
        Et       : numpy array , shape = ( N )
            SHR trap state energy level
        tn       : numpy array , shape = ( N )
            SHR e- lifetime
        tp       : numpy array , shape = ( N )
            SHR hole lifetime
        Br       : numpy array , shape = ( N )
            radiative recombination coefficient
        Cn       : numpy array , shape = ( N )
            electron Auger coefficient
        Cp       : numpy array , shape = ( N )
            hole Auger coefficient
        Snl      : float
            e- surface recombination velocity at left boundary
        Spl      : float
            hole surface recombination velocity at left boundary
        Snr      : float
            e- surface recombination velocity at right boundary
        Spr      : float
            hole surface recombination velocity at right boundary
        G        : numpy array , shape = ( N )
            e-/hole pair generation rate density

    Returns
    -------
        error : float
            norm of the displacement in the potentials space ( used to estimate the error )
        float
            norm of the value of the polynomial function which defines the out-of-equilibrium solution for the system;
            the solution is reached when the zero of the system of equations is found ( printed for user )
        numpy array , shape = ( 3N )
            next potentials ( e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential )

    """
    N = dgrid.size + 1

    _F = F( dgrid , neq0 , neqL , peq0 , peqL , phis[0:N] , phis[N:2*N] , phis[2*N:] , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G )
#    gradF = F_deriv( dgrid , neq0 , neqL , peq0 , peqL , phis[0:N] , phis[N:2*N] , phis[2*N:] , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G )

    gradF = np.zeros( (3*N,3*N) )

    gradF[0 ,0]= -54.64270665442045
    gradF[0 ,2]= -54.133641069703266
    gradF[0 ,3]= 0.5090655847171832
    gradF[1 ,1]= -3.484902850195993e-65
    gradF[1 ,2]= -3.483886467509612e-65
    gradF[1 ,4]= 1.0163826863809517e-68
    gradF[2 ,2]= 1.0
    gradF[3 ,0]= 0.003988312200564364
    gradF[3 ,3]= -0.007976611323921952
    gradF[3 ,4]= 1.3168313745278962e-74
    gradF[3 ,5]= 2.4586373148030487e-90
    gradF[3 ,6]= 0.003988299123357586
    gradF[4 ,1]= 7.962925780550618e-71
    gradF[4 ,3]= 1.3168313745278958e-74
    gradF[4 ,4]= -1.5927194502030293e-70
    gradF[4 ,5]= -2.4586373148030487e-90
    gradF[4 ,7]= 7.96295189010515e-71
    gradF[5 ,2]= -0.0004505455579565719
    gradF[5 ,3]= 0.3702499970526018
    gradF[5 ,4]= 2.3828232303881067e-67
    gradF[5 ,5]= 0.3711510881685149
    gradF[5 ,8]= -0.0004505455579565719
    gradF[6 ,3]= 0.003988299123357587
    gradF[6 ,6]= -0.007965835287312234
    gradF[6 ,7]= 1.3168399995533878e-74
    gradF[6 ,8]= 2.45865341846668e-90
    gradF[6 ,9]= 0.003977536163954649
    gradF[7 ,4]= 7.96295189010515e-71
    gradF[7 ,6]= 1.3168399995533876e-74
    gradF[7 ,7]= -1.594874849002587e-70
    gradF[7 ,8]= -2.45865341846668e-90
    gradF[7 ,10]= 7.984479759921169e-71
    gradF[8 ,5]= -0.00045054555795657195
    gradF[8 ,6]= 0.37024757199286484
    gradF[8 ,7]= 2.3828388374821527e-67
    gradF[8 ,8]= 0.3711486631087779
    gradF[8 ,11]= -0.00045054555795657206
    gradF[9 ,6]= 0.003977536163954649
    gradF[9 ,9]= -0.003983079399771458
    gradF[9 ,10]= 1.32396432808735e-74
    gradF[9 ,11]= 4.943910303884919e-90
    gradF[9 ,12]= 5.543235816810049e-06
    gradF[10, 7]= 7.984479759921167e-71
    gradF[10, 9]= 1.3239643280873494e-74
    gradF[10, 10]= -8.31539550005392e-69
    gradF[10, 11]= -4.943910303884919e-90
    gradF[10, 13]= 8.235537462811429e-69
    gradF[11, 8]= -0.000450545557956572
    gradF[11, 9]= 0.36825524841901947
    gradF[11, 10]= 2.3957304011705766e-67
    gradF[11, 11]= 0.3692379695365029
    gradF[11, 14]= -0.0005321755595268894
    gradF[12, 9]= 5.543235816810048e-06
    gradF[12, 12]= -5.544249019279215e-06
    gradF[12, 13]= 6.253841376152589e-30
    gradF[12, 14]= 4.580654007267571e-44
    gradF[12, 15]= 1.0132024691667028e-09
    gradF[13, 10]= 8.235537462811428e-69
    gradF[13, 12]= 6.253841376152544e-30
    gradF[13, 13]= -4.582392285508529e-26
    gradF[13, 14]= -4.580654007267571e-44
    gradF[13, 16]= 4.581766901370914e-26
    gradF[14, 11]= -0.0005321755595268893
    gradF[14, 12]= 5.886001511054781e-05
    gradF[14, 13]= 1.1316406048170259e-22
    gradF[14, 14]= 0.001204841135734644
    gradF[14, 17]= -0.0006138055610972069
    gradF[15, 12]= 1.013202469166703e-09
    gradF[15, 15]= -1.0564076661087338e-09
    gradF[15, 16]= 1.4709416524273058e-28
    gradF[15, 17]= 1.0457095707353858e-42
    gradF[15, 18]= 4.320519694203051e-11
    gradF[16, 13]= 4.581766901370915e-26
    gradF[16, 15]= 1.4709416524272952e-28
    gradF[16, 16]= -1.1224282838297037e-24
    gradF[16, 17]= -1.0457095707353858e-42
    gradF[16, 19]= 1.0764635206507519e-24
    gradF[17, 14]= -0.0006138055610972071
    gradF[17, 15]= 2.502486660041197e-06
    gradF[17, 16]= 2.6616878388531777e-21
    gradF[17, 17]= 0.0012301136088544561
    gradF[17, 20]= -0.0006138055610972075
    gradF[18, 15]= 4.320519694203051e-11
    gradF[18, 18]= -4.50498427995028e-11
    gradF[18, 19]= 3.44566541500358e-27
    gradF[18, 20]= 2.449561907885246e-41
    gradF[18, 21]= 1.8446458574722903e-12
    gradF[19, 16]= 1.0764635206507519e-24
    gradF[19, 18]= 3.445665415003555e-27
    gradF[19, 19]= -2.6294777183534372e-23
    gradF[19, 20]= -2.449561907885246e-41
    gradF[19, 22]= 2.5214867997468612e-23
    gradF[20, 17]= -0.0006138055610972075
    gradF[20, 18]= 1.0683013875503949e-07
    gradF[20, 19]= 6.234980491036523e-20
    gradF[20, 20]= 0.00122771795233317
    gradF[20, 23]= -0.0006138055610972071
    gradF[21, 18]= 1.8446458574722895e-12
    gradF[21, 21]= -1.9234071722410527e-12
    gradF[21, 22]= 8.069896824025045e-26
    gradF[21, 23]= 5.9107251119464516e-40
    gradF[21, 24]= 7.876131476868253e-14
    gradF[22, 19]= 2.5214867997468603e-23
    gradF[22, 21]= 8.069896824024985e-26
    gradF[22, 22]= -6.158476572608617e-22
    gradF[22, 23]= -5.9107251119464516e-40
    gradF[22, 25]= 5.905520902951528e-22
    gradF[23, 20]= -0.0006138055610972069
    gradF[23, 21]= 4.5613290283199685e-09
    gradF[23, 22]= 1.4602845505265567e-18
    gradF[23, 23]= 0.0012276156835234436
    gradF[23, 26]= -0.0006138055610972069
    gradF[24, 21]= 7.876131476868253e-14
    gradF[24, 24]= -8.212421430626144e-14
    gradF[24, 25]= 1.889257779058128e-24
    gradF[24, 26]= 1.3832098441846903e-38
    gradF[24, 27]= 3.3628995356896715e-15
    gradF[25, 22]= 5.905520902951528e-22
    gradF[25, 24]= 1.8892577790581145e-24
    gradF[25, 25]= -1.4423560979376564e-20
    gradF[25, 26]= -1.3832098441846903e-38
    gradF[25, 28]= 1.383111963130235e-20
    gradF[26, 23]= -0.0006138055610972069
    gradF[26, 24]= 1.9475662354930002e-10
    gradF[26, 25]= 3.4200830701081953e-17
    gradF[26, 26]= 0.0012276113169510717
    gradF[26, 29]= -0.0006138055610972069
    gradF[27, 24]= 3.3628995356896715e-15
    gradF[27, 27]= -3.5064864966470076e-15
    gradF[27, 28]= 4.3831846538893187e-23
    gradF[27, 29]= 3.0854658609523836e-37
    gradF[27, 30]= 1.4358691712548922e-16
    gradF[28, 25]= 1.383111963130235e-20
    gradF[28, 27]= 4.3831846538892875e-23
    gradF[28, 28]= -3.378088237554654e-19
    gradF[28, 29]= -3.0854658609523836e-37
    gradF[28, 31]= 3.239338722776241e-19
    gradF[29, 26]= -0.0006138055610972069
    gradF[29, 27]= 8.315592717784831e-12
    gradF[29, 28]= 8.010058375848791e-16
    gradF[29, 29]= 0.0012276111305108074
    gradF[29, 32]= -0.0006138055610972069
    gradF[30, 27]= 1.4358691712548922e-16
    gradF[30, 30]= -1.497185398466105e-16
    gradF[30, 31]= 8.410444407917437e-22
    gradF[30, 32]= 4.8478700355641627e-36
    gradF[30, 33]= 6.130781676680487e-18
    gradF[31, 28]= 3.239338722776241e-19
    gradF[31, 30]= 8.4104444079173885e-22
    gradF[31, 31]= -7.911518098485926e-18
    gradF[31, 32]= -4.8478700355641627e-36
    gradF[31, 34]= 7.58674318176751e-18
    gradF[32, 29]= -0.0006138055610972069
    gradF[32, 30]= 3.5505382011968424e-13
    gradF[32, 31]= 1.8760080676441318e-14
    gradF[32, 32]= 0.0012276111225682277
    gradF[32, 35]= -0.0006138055610972069
    gradF[33, 30]= 6.130781676680492e-18
    gradF[33, 33]= -6.396159255032565e-18
    gradF[33, 34]= 3.6094123778264585e-21
    gradF[33, 35]= 2.7088245117023134e-36
    gradF[33, 36]= 2.6176816597424757e-19
    gradF[34, 31]= 7.586743181767515e-18
    gradF[34, 33]= 3.609412377826456e-21
    gradF[34, 34]= -1.8527684054704366e-16
    gradF[34, 35]= -2.7088245117023134e-36
    gradF[34, 37]= 1.776864879528983e-16
    gradF[35, 32]= -0.0006138055610972074
    gradF[35, 33]= 1.5159859258261908e-14
    gradF[35, 34]= 4.393733606922437e-13
    gradF[35, 35]= 0.0012276111226489487
    gradF[35, 38]= -0.0006138055610972081
    gradF[36, 33]= 2.6176816597424737e-19
    gradF[36, 36]= -2.749242098935944e-19
    gradF[36, 37]= 1.979235563968144e-21
    gradF[36, 38]= -7.73634295461634e-36
    gradF[36, 39]= 1.1176808355378881e-20
    gradF[37, 34]= 1.7768648795289819e-16
    gradF[37, 36]= 1.9792355639681517e-21
    gradF[37, 37]= -4.339222264326774e-15
    gradF[37, 38]= 7.73634295461634e-36
    gradF[37, 40]= 4.161533797138312e-15
    gradF[38, 35]= -0.0006138055610972077
    gradF[38, 36]= 6.472859030048344e-16
    gradF[38, 37]= 1.0290411515225309e-11
    gradF[38, 38]= 0.0012276111324854722
    gradF[38, 41]= -0.000613805561097206
    gradF[39, 36]= 1.1176808355378881e-20
    gradF[39, 39]= -1.180170956450272e-20
    gradF[39, 40]= 1.4768119158141682e-22
    gradF[39, 41]= -1.045900963117743e-36
    gradF[39, 42]= 4.7722001754242135e-22
    gradF[40, 37]= 4.161533797138312e-15
    gradF[40, 39]= 1.4768119158141788e-22
    gradF[40, 40]= -1.0162738972493982e-13
    gradF[40, 41]= 1.045900963117743e-36
    gradF[40, 43]= 9.746585578012033e-14
    gradF[41, 38]= -0.000613805561097206
    gradF[41, 39]= 2.7637395972308584e-17
    gradF[41, 40]= 2.410081730058005e-10
    gradF[41, 41]= 0.0012276113632026142
    gradF[41, 44]= -0.0006138055610972077
    gradF[42, 39]= 4.7722001754242135e-22
    gradF[42, 42]= -5.041077273622699e-22
    gradF[42, 43]= 6.511821647658081e-24
    gradF[42, 44]= -4.76265750190852e-38
    gradF[42, 45]= 2.037588817219036e-23
    gradF[43, 40]= 9.746585578012033e-14
    gradF[43, 42]= 6.51182164765813e-24
    gradF[43, 43]= -2.380186608354207e-12
    gradF[43, 44]= 4.76265750190852e-38
    gradF[43, 46]= 2.282720752567575e-12
    gradF[44, 41]= -0.0006138055610972077
    gradF[44, 42]= 1.1800432431349256e-18
    gradF[44, 43]= 5.644571373696992e-09
    gradF[44, 44]= 0.0012276167667657884
    gradF[44, 47]= -0.000613805561097206
    gradF[45, 42]= 2.037588817219036e-23
    gradF[45, 45]= -2.1524163618134875e-23
    gradF[45, 46]= 2.784235662157365e-25
    gradF[45, 47]= -1.9792222504502923e-39
    gradF[45, 48]= 8.698518797287794e-25
    gradF[46, 43]= 2.282720752567575e-12
    gradF[46, 45]= 2.784235662157385e-25
    gradF[46, 46]= -5.57491852707861e-11
    gradF[46, 47]= 1.9792222504502923e-39
    gradF[46, 49]= 5.346646451821825e-11
    gradF[47, 44]= -0.000613805561097206
    gradF[47, 45]= 5.0384242120299e-20
    gradF[47, 46]= 1.322008237023856e-07
    gradF[47, 47]= 0.001227743323018116
    gradF[47, 50]= -0.0006138055610972077
    gradF[48, 45]= 8.698518797287806e-25
    gradF[48, 48]= -9.187338105220987e-25
    gradF[48, 49]= 1.188598517392898e-26
    gradF[48, 50]= -8.449863228948734e-41
    gradF[48, 51]= 3.699594561938911e-26
    gradF[49, 46]= 5.3466464518218325e-11
    gradF[49, 48]= 1.1885985173929065e-26
    gradF[49, 49]= -1.3076900560874695e-09
    gradF[49, 50]= 8.449863228948734e-41
    gradF[49, 52]= 1.2542235915692515e-09
    gradF[50, 47]= -0.0006138055610972086
    gradF[50, 48]= 2.150789958442611e-21
    gradF[50, 49]= 3.0969264496413577e-06
    gradF[50, 50]= 0.0012307080486440585
    gradF[50, 53]= -0.0006138055610972086
    gradF[51, 48]= 3.699594561938905e-26
    gradF[51, 51]= -3.8941418982233067e-26
    gradF[51, 52]= 5.0483371236692095e-28
    gradF[51, 53]= -3.697676628306264e-42
    gradF[51, 54]= 1.440639650477089e-27
    gradF[52, 49]= 1.2542235915692496e-09
    gradF[52, 51]= 5.048337123669246e-28
    gradF[52, 52]= -3.174320309022573e-08
    gradF[52, 53]= 3.697676628306264e-42
    gradF[52, 55]= 3.048897949865647e-08
    gradF[53, 50]= -0.0006138055610972077
    gradF[53, 51]= 9.135031687199149e-23
    gradF[53, 52]= 7.29153279156956e-05
    gradF[53, 53]= 0.001300526450110109
    gradF[53, 56]= -0.000613805561097206
    gradF[54, 51]= 1.440639650477089e-27
    gradF[54, 54]= -1.4596797689653897e-27
    gradF[54, 55]= 1.9040118488300583e-29
    gradF[54, 56]= -1.3946019663232395e-43
    gradF[54, 57]= 2.5496538413167965e-58
    gradF[55, 52]= 3.048897949865647e-08
    gradF[55, 54]= 1.9040118488300723e-29
    gradF[55, 55]= -1.087057227922668e-06
    gradF[55, 56]= 1.3946019663232395e-43
    gradF[55, 58]= 1.0565682484240118e-06
    gradF[56, 53]= -0.000613805561097206
    gradF[56, 54]= 3.445333830458902e-24
    gradF[56, 55]= 0.0019332925741587046
    gradF[56, 56]= 0.003349992637504724
    gradF[56, 59]= -0.0008028945022488135
    gradF[57, 54]= 2.5496538413167965e-58
    gradF[57, 57]= -2.6359584129142512e-58
    gradF[57, 58]= 1.5534430749401372e-61
    gradF[57, 59]= -4.422776590027069e-75
    gradF[57, 60]= 8.475112852251474e-60
    gradF[58, 55]= 1.0565682484240118e-06
    gradF[58, 57]= 1.5534430749401813e-61
    gradF[58, 58]= -0.004255663675085719
    gradF[58, 59]= 4.422776590027069e-75
    gradF[58, 61]= 0.004254607106837292
    gradF[59, 56]= -0.0008028945022488135
    gradF[59, 57]= 2.8109751314059247e-56
    gradF[59, 58]= 0.15699916870731995
    gradF[59, 59]= 0.15879404665296915
    gradF[59, 62]= -0.0009919834434004165
    gradF[60, 57]= 8.475112852251474e-60
    gradF[60, 60]= -1.699813611161111e-59
    gradF[60, 61]= 1.5158209779460518e-61
    gradF[60, 62]= -4.3156634730178863e-75
    gradF[60, 63]= 8.371441161565019e-60
    gradF[61, 58]= 0.004254607106837292
    gradF[61, 60]= 1.515820977946095e-61
    gradF[61, 61]= -0.008562118890980935
    gradF[61, 62]= 4.3156634730178863e-75
    gradF[61, 64]= 0.004307511784143642
    gradF[62, 59]= -0.0009919834434004165
    gradF[62, 60]= 2.7428974652540505e-56
    gradF[62, 61]= 0.16089582803519195
    gradF[62, 62]= 0.16287979492199275
    gradF[62, 65]= -0.0009919834434004193
    gradF[63, 60]= 8.371441161565019e-60
    gradF[63, 63]= -1.689381305618246e-59
    gradF[63, 64]= 1.5155946713177193e-61
    gradF[63, 65]= -4.315019159959942e-75
    gradF[63, 66]= 8.370812427485662e-60
    gradF[64, 61]= 0.004307511784143642
    gradF[64, 63]= 1.5155946713177625e-61
    gradF[64, 64]= -0.008615347114711895
    gradF[64, 65]= 4.315019159959942e-75
    gradF[64, 67]= 0.004307835330568253
    gradF[65, 62]= -0.0009919834434004193
    gradF[65, 63]= 2.7424879605128875e-56
    gradF[65, 64]= 0.1609198527913121
    gradF[65, 65]= 0.1629038196781129
    gradF[65, 68]= -0.0009919834434004165
    gradF[66, 63]= 8.370812427485662e-60
    gradF[66, 66]= -1.6893180355657602e-59
    gradF[66, 67]= 1.5155932933034054e-61
    gradF[66, 69]= 8.370808598841597e-60
    gradF[67, 64]= 0.004307835330568253
    gradF[67, 66]= 1.5155932933034054e-61
    gradF[67, 67]= -0.00861567263145642
    gradF[67, 70]= 0.004307837300888167
    gradF[68, 65]= -0.0009919834434004165
    gradF[68, 66]= 2.7424854669782667e-56
    gradF[68, 67]= 0.16091999910355928
    gradF[68, 68]= 0.16290396599036008
    gradF[68, 71]= -0.0009919834434004193
    gradF[69, 66]= 8.370808598841609e-60
    gradF[69, 69]= -1.6893176502859678e-59
    gradF[69, 70]= 1.5155932849118614e-61
    gradF[69, 71]= -4.3150152127516586e-75
    gradF[69, 72]= 8.370808575526878e-60
    gradF[70, 67]= 0.004307837300888173
    gradF[70, 69]= 1.5155932849119044e-61
    gradF[70, 70]= -0.008615674613774715
    gradF[70, 71]= 4.3150152127516586e-75
    gradF[70, 73]= 0.004307837312886542
    gradF[71, 68]= -0.0009919834434004206
    gradF[71, 69]= 2.7424854517937384e-56
    gradF[71, 70]= 0.1609199999945419
    gradF[71, 71]= 0.16290396688134273
    gradF[71, 74]= -0.0009919834434004206
    gradF[72, 69]= 8.370808575526867e-60
    gradF[72, 72]= -1.6893176479397753e-59
    gradF[72, 73]= 1.5155932848607732e-61
    gradF[72, 74]= -4.315015212606207e-75
    gradF[72, 75]= 8.370808575384808e-60
    gradF[73, 70]= 0.004307837312886535
    gradF[73, 72]= 1.5155932848608164e-61
    gradF[73, 73]= -0.008615674625846153
    gradF[73, 74]= 4.315015212606207e-75
    gradF[73, 76]= 0.004307837312959618
    gradF[74, 71]= -0.0009919834434004193
    gradF[74, 72]= 2.742485451701294e-56
    gradF[74, 73]= 0.16091999999996623
    gradF[74, 74]= 0.16290396688676703
    gradF[74, 77]= -0.0009919834434004165
    gradF[75, 72]= 8.370808575384797e-60
    gradF[75, 75]= -1.6893176479254688e-59
    gradF[75, 76]= 1.5155932848604717e-61
    gradF[75, 78]= 8.370808575383847e-60
    gradF[76, 73]= 0.004307837312959612
    gradF[76, 75]= 1.5155932848604717e-61
    gradF[76, 76]= -0.008615674625919712
    gradF[76, 79]= 0.004307837312960102
    gradF[77, 74]= -0.0009919834434004152
    gradF[77, 75]= 2.7424854517006705e-56
    gradF[77, 76]= 0.16091999999999823
    gradF[77, 77]= 0.16290396688679903
    gradF[77, 80]= -0.0009919834434004152
    gradF[78, 75]= 8.370808575383858e-60
    gradF[78, 78]= -1.6893176479253786e-59
    gradF[78, 79]= 1.5155932848604717e-61
    gradF[78, 81]= 8.370808575383882e-60
    gradF[79, 76]= 0.004307837312960108
    gradF[79, 78]= 1.5155932848604717e-61
    gradF[79, 79]= -0.008615674625920229
    gradF[79, 82]= 0.004307837312960121
    gradF[80, 77]= -0.0009919834434004165
    gradF[80, 78]= 2.7424854517006705e-56
    gradF[80, 79]= 0.16091999999999823
    gradF[80, 80]= 0.16290396688679903
    gradF[80, 83]= -0.0009919834434004193
    gradF[81, 78]= 8.370808575383894e-60
    gradF[81, 81]= -1.689317647925383e-59
    gradF[81, 82]= 1.5155932848604717e-61
    gradF[81, 84]= 8.370808575383894e-60
    gradF[82, 79]= 0.004307837312960127
    gradF[82, 81]= 1.5155932848604717e-61
    gradF[82, 82]= -0.008615674625920253
    gradF[82, 85]= 0.004307837312960127
    gradF[83, 80]= -0.0009919834434004206
    gradF[83, 81]= 2.7424854517006705e-56
    gradF[83, 82]= 0.16091999999999823
    gradF[83, 83]= 0.16290396688679906
    gradF[83, 86]= -0.0009919834434004206
    gradF[84, 81]= 8.370808575383894e-60
    gradF[84, 84]= -1.689317647925383e-59
    gradF[84, 85]= 1.5155932848604717e-61
    gradF[84, 87]= 8.370808575383894e-60
    gradF[85, 82]= 0.004307837312960127
    gradF[85, 84]= 1.5155932848604717e-61
    gradF[85, 85]= -0.008615674625920253
    gradF[85, 88]= 0.004307837312960127
    gradF[86, 83]= -0.0009919834434004206
    gradF[86, 84]= 2.7424854517006705e-56
    gradF[86, 85]= 0.16091999999999823
    gradF[86, 86]= 0.16290396688679906
    gradF[86, 89]= -0.0009919834434004206
    gradF[87, 84]= -1.0684445819914541e-57
    gradF[87, 87]= 4.010811134149818e-54
    gradF[87, 89]= 4.0097426895678265e-54
    gradF[88, 85]= -0.5498495630001654
    gradF[88, 88]= 24.07769701995263
    gradF[88, 89]= 23.527847456952465
    gradF[89, 89]= 1.0



#    for i in range(3*N):
#        for j in range(3*N):
#            if (gradF[i,j]!=0):
#                print(i,j,gradF[i,j])

#    print(gradF[87,89])

#    quit()

    move = np.linalg.solve( gradF , - _F )
    print(move)
    quit()
    error = np.linalg.norm( move )
    damp_move = damp( move )

    return error , np.linalg.norm(_F) , np.concatenate( ( phis[0:N] + damp_move[0:3*N:3] , phis[N:2*N] + damp_move[1:3*N:3] , phis[2*N:]+ damp_move[2:3*N:3] ) , axis = 0 )





#@jit
def step_forgrad( dgrid , neq0 , neqL , peq0 , peqL , phis , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G ):
    """
    Computes the next potentials in the Newton method iterative scheme.

    This function is to be used for gradient calculations with JAX. It doesn't print variables.

    Parameters
    ----------
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        neq0     : float
            e- equilibrium density at left boundary
        neqL     : float
            e- equilibrium density at right boundary
        peq0     : float
            hole equilibrium density at left boundary
        peqL     : float
            hole equilibrium density at right boundary
        phis     : numpy array , shape = ( 3N )
            current potentials ( e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential )
        eps      : numpy array , shape = ( N )
            relative dieclectric constant
        Chi      : numpy array , shape = ( N )
            electron affinity
        Eg       : numpy array , shape = ( N )
            band gap
        Nc       : numpy array , shape = ( N )
            e- density of states
        Nv       : numpy array , shape = ( N )
            hole density of states
        Ndop     : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )
        mn       : numpy array , shape = ( N )
            e- mobility
        mp       : numpy array , shape = ( N )
            hole mobility
        Et       : numpy array , shape = ( N )
            SHR trap state energy level
        tn       : numpy array , shape = ( N )
            SHR e- lifetime
        tp       : numpy array , shape = ( N )
            SHR hole lifetime
        Br       : numpy array , shape = ( N )
            radiative recombination coefficient
        Cn       : numpy array , shape = ( N )
            electron Auger coefficient
        Cp       : numpy array , shape = ( N )
            hole Auger coefficient
        Snl      : float
            e- surface recombination velocity at left boundary
        Spl      : float
            hole surface recombination velocity at left boundary
        Snr      : float
            e- surface recombination velocity at right boundary
        Spr      : float
            hole surface recombination velocity at right boundary
        G        : numpy array , shape = ( N )
            e-/hole pair generation rate density

    Returns
    -------
        numpy array , shape = ( 3N )
            next potentials ( e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential )

    """
    N = dgrid.size + 1
    _F = F( dgrid , neq0 , neqL , peq0 , peqL , phis[0:N] , phis[N:2*N] , phis[2*N:] , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G )
    gradF = F_deriv( dgrid , neq0 , neqL , peq0 , peqL , phis[0:N] , phis[N:2*N] , phis[2*N:] , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G )

    move = np.linalg.solve( gradF , - _F )
    damp_move = damp( move )

    return np.concatenate( ( phis[0:N] + damp_move[0:3*N:3] , phis[N:2*N] + damp_move[1:3*N:3] , phis[2*N:]+ damp_move[2:3*N:3] ) , axis = 0 )





def solve( dgrid , neq0 , neqL , peq0 , peqL , phis_ini , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G ):
    """
    Solves for the e-/hole quasi-Fermi energies and electrostatic potential using the Newton method.

    Parameters
    ----------
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        neq0     : float
            e- equilibrium density at left boundary
        neqL     : float
            e- equilibrium density at right boundary
        peq0     : float
            hole equilibrium density at left boundary
        peqL     : float
            hole equilibrium density at right boundary
        phis_ini : numpy array , shape = ( 3N )
            current potentials ( e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential )
        eps      : numpy array , shape = ( N )
            relative dieclectric constant
        Chi      : numpy array , shape = ( N )
            electron affinity
        Eg       : numpy array , shape = ( N )
            band gap
        Nc       : numpy array , shape = ( N )
            e- density of states
        Nv       : numpy array , shape = ( N )
            hole density of states
        Ndop     : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )
        mn       : numpy array , shape = ( N )
            e- mobility
        mp       : numpy array , shape = ( N )
            hole mobility
        Et       : numpy array , shape = ( N )
            SHR trap state energy level
        tn       : numpy array , shape = ( N )
            SHR e- lifetime
        tp       : numpy array , shape = ( N )
            SHR hole lifetime
        Br       : numpy array , shape = ( N )
            radiative recombination coefficient
        Cn       : numpy array , shape = ( N )
            electron Auger coefficient
        Cp       : numpy array , shape = ( N )
            hole Auger coefficient
        Snl      : float
            e- surface recombination velocity at left boundary
        Spl      : float
            hole surface recombination velocity at left boundary
        Snr      : float
            e- surface recombination velocity at right boundary
        Spr      : float
            hole surface recombination velocity at right boundary
        G        : numpy array , shape = ( N )
            e-/hole pair generation rate density

    Returns
    -------
        phis     : numpy array , shape = ( 3N )
            solution for the e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential

    """
    error = 1
    iter = 0

    phis = phis_ini
    while (error > 1e-6):
        error_dx , error_F , next_phis = step( dgrid , neq0 , neqL , peq0 , peqL , phis , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G )
        phis = next_phis
        error = error_dx
        iter += 1
        #print(iter)
        print( '                {0:02d}              {1:.9f}           {2:.9f}'.format( iter , float( error_F ) , float( error_dx ) ) )

    return phis





def solve_forgrad( dgrid , neq0 , neqL , peq0 , peqL , phis_ini , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G ):
    """
    Solves for the e-/hole quasi-Fermi energies and electrostatic potential using the Newton method and computes derivatives.

    This function is to be used for gradient calculations with JAX. It doesn't print variables.

    Parameters
    ----------
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        neq0     : float
            e- equilibrium density at left boundary
        neqL     : float
            e- equilibrium density at right boundary
        peq0     : float
            hole equilibrium density at left boundary
        peqL     : float
            hole equilibrium density at right boundary
        phis_ini : numpy array , shape = ( 3N )
            current potentials ( e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential )
        eps      : numpy array , shape = ( N )
            relative dieclectric constant
        Chi      : numpy array , shape = ( N )
            electron affinity
        Eg       : numpy array , shape = ( N )
            band gap
        Nc       : numpy array , shape = ( N )
            e- density of states
        Nv       : numpy array , shape = ( N )
            hole density of states
        Ndop     : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )
        mn       : numpy array , shape = ( N )
            e- mobility
        mp       : numpy array , shape = ( N )
            hole mobility
        Et       : numpy array , shape = ( N )
            SHR trap state energy level
        tn       : numpy array , shape = ( N )
            SHR e- lifetime
        tp       : numpy array , shape = ( N )
            SHR hole lifetime
        Br       : numpy array , shape = ( N )
            radiative recombination coefficient
        Cn       : numpy array , shape = ( N )
            electron Auger coefficient
        Cp       : numpy array , shape = ( N )
            hole Auger coefficient
        Snl      : float
            e- surface recombination velocity at left boundary
        Spl      : float
            hole surface recombination velocity at left boundary
        Snr      : float
            e- surface recombination velocity at right boundary
        Spr      : float
            hole surface recombination velocity at right boundary
        G        : numpy array , shape = ( N )
            e-/hole pair generation rate density

    Returns
    -------
        phis     : numpy array , shape = ( 3N )
            solution for the e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential

    """
    N = dgrid.size + 1
    error = 1
    iter = 0
    grad_step = jit( jacfwd( step_forgrad , ( 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11, 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24 ) ) )

    phis = phis_ini

    dphis_dphiini = np.eye( 3 * N )
    dphis_dneq0 = np.zeros( ( 3 * N , 1 ) )
    dphis_dneqL = np.zeros( ( 3 * N , 1 ) )
    dphis_dpeq0 = np.zeros( ( 3 * N , 1 ) )
    dphis_dpeqL = np.zeros( ( 3 * N , 1 ) )
    dphis_dSnl = np.zeros( ( 3 * N , 1 ) )
    dphis_dSpl = np.zeros( ( 3 * N , 1 ) )
    dphis_dSnr = np.zeros( ( 3 * N , 1 ) )
    dphis_dSpr = np.zeros( ( 3 * N , 1 ) )
    dphis_deps = np.zeros( ( 3 * N , N ) )
    dphis_dChi = np.zeros( ( 3 * N , N ) )
    dphis_dEg = np.zeros( ( 3 * N , N ) )
    dphis_dNc = np.zeros( ( 3 * N , N ) )
    dphis_dNv = np.zeros( ( 3 * N , N ) )
    dphis_dNdop = np.zeros( ( 3 * N , N ) )
    dphis_dmn = np.zeros( ( 3 * N , N ) )
    dphis_dmp = np.zeros( ( 3 * N , N ) )
    dphis_dEt = np.zeros( ( 3 * N , N ) )
    dphis_dtn = np.zeros( ( 3 * N , N ) )
    dphis_dtp = np.zeros( ( 3 * N , N ) )
    dphis_dBr = np.zeros( ( 3 * N , N ) )
    dphis_dCn = np.zeros( ( 3 * N , N ) )
    dphis_dCp = np.zeros( ( 3 * N , N ) )
    dphis_dG = np.zeros( ( 3 * N , N ) )

    while (error > 1e-6):
        error_dx , error_F , next_phis = step( dgrid , neq0 , neqL , peq0 , peqL , phis , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G )
        gradstep = grad_step( dgrid , neq0 , neqL , peq0 , peqL , phis , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G )
        phis = next_phis

        dphis_dphiini = np.dot( gradstep[4] , dphis_dphiini )
        dphis_dneq0 = np.reshape( gradstep[0] , ( 3*N , 1 ) ) + np.dot( gradstep[4] , dphis_dneq0 )
        dphis_dneqL = np.reshape( gradstep[1] , ( 3*N , 1 ) ) + np.dot( gradstep[4] , dphis_dneqL )
        dphis_dpeq0 = np.reshape( gradstep[2] , ( 3*N , 1 ) ) + np.dot( gradstep[4] , dphis_dpeq0 )
        dphis_dpeqL = np.reshape( gradstep[3] , ( 3*N , 1 ) ) + np.dot( gradstep[4] , dphis_dpeqL )
        dphis_dSnl = np.reshape( gradstep[19] , ( 3*N , 1 ) ) + np.dot( gradstep[4] , dphis_dSnl )
        dphis_dSpl = np.reshape( gradstep[20] , ( 3*N , 1 ) ) + np.dot( gradstep[4] , dphis_dSpl )
        dphis_dSnr = np.reshape( gradstep[21] , ( 3*N , 1 ) ) + np.dot( gradstep[4] , dphis_dSnr )
        dphis_dSpr = np.reshape( gradstep[22] , ( 3*N , 1 ) ) + np.dot( gradstep[4] , dphis_dSpr )
        dphis_deps = gradstep[5] + np.dot( gradstep[4] , dphis_deps )
        dphis_dChi = gradstep[6] + np.dot( gradstep[4] , dphis_dChi )
        dphis_dEg = gradstep[7] + np.dot( gradstep[4] , dphis_dEg )
        dphis_dNc = gradstep[8] + np.dot( gradstep[4] , dphis_dNc )
        dphis_dNv = gradstep[9] + np.dot( gradstep[4] , dphis_dNv )
        dphis_dNdop = gradstep[10] + np.dot( gradstep[4] , dphis_dNdop )
        dphis_dmn = gradstep[11] + np.dot( gradstep[4] , dphis_dmn )
        dphis_dmp = gradstep[12] + np.dot( gradstep[4] , dphis_dmp )
        dphis_dEt = gradstep[13] + np.dot( gradstep[4] , dphis_dEt )
        dphis_dtn = gradstep[14] + np.dot( gradstep[4] , dphis_dtn )
        dphis_dtp = gradstep[15] + np.dot( gradstep[4] , dphis_dtp )
        dphis_dBr = gradstep[16] + np.dot( gradstep[4] , dphis_dBr )
        dphis_dCn = gradstep[17] + np.dot( gradstep[4] , dphis_dCn )
        dphis_dCp = gradstep[18] + np.dot( gradstep[4] , dphis_dCp )
        dphis_dG = gradstep[23] + np.dot( gradstep[4] , dphis_dG )

        error = error_dx
        iter += 1
        print( '                {0:02d}              {1:.9f}           {2:.9f}'.format( iter , float( error_F ) , float( error_dx ) ) )


    grad_phis = {}
    grad_phis['neq0'] = dphis_dneq0
    grad_phis['neqL'] = dphis_dneqL
    grad_phis['peq0'] = dphis_dpeq0
    grad_phis['peqL'] = dphis_dpeqL
    grad_phis['phi_ini'] = dphis_dphiini
    grad_phis['eps'] = dphis_deps
    grad_phis['Chi'] = dphis_dChi
    grad_phis['Eg'] = dphis_dEg
    grad_phis['Nc'] = dphis_dNc
    grad_phis['Nv'] = dphis_dNv
    grad_phis['Ndop'] = dphis_dNdop
    grad_phis['mn'] = dphis_dmn
    grad_phis['mp'] = dphis_dmp
    grad_phis['Et'] = dphis_dEt
    grad_phis['tn'] = dphis_dtn
    grad_phis['tp'] = dphis_dtp
    grad_phis['Br'] = dphis_dBr
    grad_phis['Cn'] = dphis_dCn
    grad_phis['Cp'] = dphis_dCp
    grad_phis['Snl'] = dphis_dSnl
    grad_phis['Spl'] = dphis_dSpl
    grad_phis['Snr'] = dphis_dSnr
    grad_phis['Spr'] = dphis_dSpr
    grad_phis['G'] = dphis_dG

    return phis , grad_phis
