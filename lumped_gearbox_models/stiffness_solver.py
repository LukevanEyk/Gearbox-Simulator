"""

This solver was written by Luke van Eyk. 

"""

############### DEPENDENT LIBRARIES ##########################
import  numpy as np
from    matplotlib import pyplot as plt
from    scipy.optimize import fsolve
##############################################################

####################### USER AID #######################################
def RadiusFinder(Z1,Z2,m1,m2,a0,ha = 1, c = 0.25):
    '''
    Z1 - Driving gear number of teeth
    Z2 - Driven gear number of teeth
    m1 - Driving gear module (in metres)
    m2 - Driven gear module (in metres)
    a0 - Pressure angle (radians)
    ha - Addendum coefficient (The default of 1 is a commonly accepted value)
    c  - Tip clearance coefficient (The default of 0.25 is a commonly accepted value)
    '''
    Rb1 = 0.5*m1*Z1*np.cos(a0)  # Base circle radius for pinion
    Rr1 = 0.5*m1*Z1 - (ha+c)*m1 # Root circle radius for pinion

    Rb2 = 0.5*m2*Z2*np.cos(a0)  # Base circle radius for gear
    Rr2 = 0.5*m2*Z2 - (ha+c)*m2 # Root circle radius for gear
    
    return Rb1,Rr1,Rb2,Rr2
########################################################################

###################### ANGULAR FUNCTIONS #################################
def A1(T,Z1,Z2,a0,GP=1,GT=0):
    '''
    T - Theta
    GT - Gear Type: 0 = Pinion, 1 = Gear
    GP - Gear Pairs Engaged: 1 = 1 Pair, 2 = 2 Pairs
    '''
    if GT == 0 and GP == 1: # Pinion - 1st Gear Pair
        botterm = np.sqrt((Z2+2)**2 +(Z1+Z2)**2-2*(Z2+2)*(Z1+Z2)*np.cos(np.arccos(Z2*np.cos(a0)/(Z2+2))-a0))
        result = T - np.pi/(2*Z1) - np.tan(a0) + a0 + np.tan(np.arccos(Z1*np.cos(a0)/botterm))

    elif GT == 0 and GP == 2: # Pinion - 2nd Gear Pair
        botterm = np.sqrt((Z2+2)**2 +(Z1+Z2)**2-2*(Z2+2)*(Z1+Z2)*np.cos(np.arccos(Z2*np.cos(a0)/(Z2+2))-a0))
        result = T + 3*np.pi/(2*Z1) - np.tan(a0) + a0 + np.tan(np.arccos(Z1*np.cos(a0)/botterm))
    
    elif GT == 1 and GP == 1: # Driven Gear - 1st Gear Pair
        result = np.tan(np.arccos(Z2*np.cos(a0)/(Z2+2))) - np.pi/(2*Z2) - np.tan(a0) + a0 - Z1/Z2 * T
        
    elif GT == 1 and GP == 2: # Driven Gear - 2nd Gear Pair
        result = np.tan(np.arccos(Z2*np.cos(a0)/(Z2+2))) - 5*np.pi/(2*Z2) - np.tan(a0) + a0 - Z1/Z2 * T
    else:
        raise ValueError('Invalid value for GT or GP entered')
    return result

def A2(Z,a0):
    return np.pi/(2*Z) + np.tan(a0) - a0

def A3(a2,Rb,Rr):
    return np.arcsin(Rb*np.sin(a2)/Rr)



def A5(Rb,Rr,a2):
    
    def func(a5,Rb,Rr,a2):
        return (Rb/Rr)**2 * (((a2-a5)*np.cos(a5) - np.sin(a5))**2 + ((a5-a2)*np.sin(a5) + np.cos(a5))**2) - 1
    a5 = fsolve(func,1, args = (Rb,Rr,a2))
    return a5[0]


def DoubleMeshPeriod(Z1,Z2,a0):
    term1 = np.tan(np.arccos((Z1*np.cos(a0))/(Z1+2)))
    term2 = -2*np.pi/Z1
    term3 = -np.tan(np.arccos(Z1*np.cos(a0)/np.sqrt((Z2+2)**2 +(Z1+Z2)**2-2*(Z2+2)*(Z1+Z2)*np.cos(np.arccos(Z2*np.cos(a0)/(Z2+2))-a0))))
    return (term1 + term2 + term3)
#####################################################################

########################### STIFFNESSES ############################
def invkb(Z1,Z2,L,T,Rb,Rr,GP,GT,alpha_mem,dIx_mem,damage,a0,E,granularity = 100):
    # def invkb(Z1,Z2,L,T,Rb,Rr,GP,GT,alpha_mem,dIx_mem,damage,a0,granularity = 100,E = 2.068*10**11):
    if GT == 0:
        Z = Z1
    else:
        Z = Z2
        
    
    a1 = A1(T,Z1,Z2,a0,GP,GT)
    a2 = A2(Z,a0)
    intStop = a2
    a3 = A3(a2,Rb,Rr)
    constTerm = 0
    
    if Rb>=Rr: # We have to add some are that we are neglecting since the tooth actually starts at Rr
        # Constant Term
        if damage == 1 and GP == 1 and GT == 0: # Ensure only pinion gets damaged.
            #################################################
            Ix1 = getdIxVal(a2,alpha_mem,dIx_mem) 
            #################################################
            constTop = Rb**3 * ((1-Rr/Rb * np.cos(a1)*np.cos(a3))**3 - (1-np.cos(a1)*np.cos(a2))**3)
            constBot = E*np.cos(a1) *3  * Ix1
            constTerm = constTop/constBot
        else:
            constTop = (1-Rr/Rb * np.cos(a1)*np.cos(a3))**3 - (1-np.cos(a1)*np.cos(a2))**3
            constBot = 2*E*L*np.cos(a1) * np.sin(a2)**3
            constTerm = constTop/constBot
            
    else: # Otherwise, now we just have to integrate up to a5, and not a2
        a5 = A5(Rb,Rr,a2)
        a4 = np.arccos(Rb/Rr*((a5-a2)*np.sin(a5) + np.cos(a5)))
        intStop = a5
        
    # Integration Term
    IntTerm = np.zeros(len(T))
    irange = np.arange(len(T))

    for i in irange:
        a1i = a1[i]
        a2i = intStop
        
        a = np.linspace(-a1i,a2i,granularity)
        if damage == 1 and GP == 1 and GT == 0: # Ensure only pinion gets damaged.
            #################################################
            Ixmod = getdIxVal(a,alpha_mem,dIx_mem) 
            #################################################
            IntTop = (1+np.cos(a1i)*((a2-a)*np.sin(a) - np.cos(a)))**2*(a2-a)*np.cos(a) * Rb**3
            IntBot = E*Ixmod
        else:
            IntTop = 3*(1+np.cos(a1i)*((a2-a)*np.sin(a) - np.cos(a)))**2*(a2-a)*np.cos(a) 
            IntBot = 2*E*L*(np.sin(a) + (a2-a)*np.cos(a))**3 

    
        dInt = np.diff(a)[0]
        IntTerm[i] = np.sum(IntTop/IntBot * dInt)
    return constTerm + IntTerm


def invks(Z1,Z2,L,T,Rb,Rr,GP,GT,alpha_mem,dAx_mem,damage,a0,E,nu,granularity = 100):
    # invks(Z1,Z2,L,T,Rb,Rr,GP,GT,alpha_mem,dAx_mem,damage,a0,granularity = 100,E = 2.068*10**11,nu = 0.3)
    if GT == 0:
        Z = Z1
    else:
        Z = Z2
        
    a1 = A1(T,Z1,Z2,a0,GP,GT)
    a2 = A2(Z,a0)
    intStop = a2
    a3 = A3(a2,Rb,Rr)
    constTerm = 0
    d1 = Rb*np.cos(a2) - Rr*np.cos(a3)
    
    if Rb>=Rr: # We have to add some are that we are neglecting since the tooth actually starts at Rr
        # Constant Term
        if damage == 1 and GP == 1 and GT == 0:
            #################################################
            Ax1 = getdAxVal(a2,alpha_mem,dAx_mem)
            #################################################
            constTop = Rb* 2.4*(1+nu)*np.cos(a1)**2 *d1
            constBot = E*Ax1 
            constTerm = constTop/constBot
        else:
            constTop = 1.2*(1+nu)*np.cos(a1)**2 *d1
            constBot = E*L*np.sin(a2) 
            constTerm = constTop/constBot
            
    else: # Otherwise, now we just have to integrate up to a5, and not a2
        a5 = A5(Rb,Rr,a2)
        a4 = np.arccos(Rb/Rr*((a5-a2)*np.sin(a5) + np.cos(a5)))
        intStop = a5
        
        
    # Integration Term
    IntTerm = np.zeros(len(T))
    irange = np.arange(len(T))
    for i in irange:
        a1i = a1[i]
        a2i = intStop
        a3i = a3
        
        a = np.linspace(-a1i,a2i,granularity)
        if damage == 1 and GP == 1 and GT == 0:
            #################################################
            Ax = getdAxVal(a,alpha_mem,dAx_mem) # Term used to switch damage on or off
            #################################################
            IntTop = 2.4*(1+nu)*(a2-a)*np.cos(a)*np.cos(a1i)**2  * Rb
            IntBot = E*Ax
        else:
            IntTop = 1.2*(1+nu)*(a2-a)*np.cos(a)*np.cos(a1i)**2
            IntBot = E*L*(np.sin(a)+(a2-a)*np.cos(a))
            
        dInt = np.diff(a)[0]
        IntTerm[i] = np.sum(IntTop/IntBot * dInt)
    return constTerm + IntTerm


def invka(Z1,Z2,L,T,Rb,Rr,GP,GT,alpha_mem,dAx_mem,damage,a0,E,granularity = 100):
    # invka(Z1,Z2,L,T,Rb,Rr,GP,GT,alpha_mem,dAx_mem,damage,a0,granularity = 100,E = 2.068*10**11):  
    if GT == 0:
        Z = Z1
    else:
        Z = Z2
        
    a1 = A1(T,Z1,Z2,a0,GP,GT)
    a2 = A2(Z,a0)
    intStop = a2
    a3 = A3(a2,Rb,Rr)
    constTerm = 0
    d1 = Rb*np.cos(a2) - Rr*np.cos(a3)
    
    if Rb>Rr: # We have to add some are that we are neglecting since the tooth actually starts at Rr
        # Constant Term
        if damage == 1 and GP == 1 and GT == 0:
            #################################################
            Ax1 = getdAxVal(a2,alpha_mem,dAx_mem)
            #################################################
            constTop = Rb*np.sin(a1)**2*d1
            constBot = E*Ax1 
            constTerm = constTop/constBot
        else:
            constTop = np.sin(a1)**2*d1
            constBot = 2*E*L*np.sin(a2) 
            constTerm = constTop/constBot
        
    else: # Otherwise, now we just have to integrate up to a5, and not a2
        a5 = A5(Rb,Rr,a2)
        a4 = np.arccos(Rb/Rr*((a5-a2)*np.sin(a5) + np.cos(a5)))
        intStop = a5
        
    # Integration Term
    IntTerm = np.zeros(len(T))
    irange = np.arange(len(T))
    for i in irange:
        a1i = a1[i]
        a2i = intStop
        a3i = a3
        
        a = np.linspace(-a1i,a2i,granularity)
        
        if damage == 1 and GP == 1 and GT == 0:
            #################################################
            Ax = getdAxVal(a,alpha_mem,dAx_mem) # Term used to switch damage on or off
            #################################################
            IntTop = (a2-a)*np.cos(a)*np.sin(a1i)**2  * Rb
            IntBot = E*Ax
        else:
            IntTop = (a2-a)*np.cos(a)*np.sin(a1i)**2
            IntBot = E*(2*L*(np.sin(a)+(a2-a)*np.cos(a)))

            
        dInt = np.diff(a)[0]
        IntTerm[i] = np.sum(IntTop/IntBot * dInt)
    return constTerm + IntTerm


def invkh(Z1,Z2,L,T,GP,GT,damage,alpha_mem,dL_mem,a0,E,nu,granularity = 100):
    # invkh(Z1,Z2,L,T,GP,GT,damage,alpha_mem,dL_mem,a0,granularity = 100,E = 2.068*10**11,nu = 0.3)
    a1 = A1(T,Z1,Z2,a0,GP,GT)
    cterm = 4*(1-nu**2)/(np.pi*E)
    Lnew = L
    if damage == 1 and GP == 1 and GT == 0:
        Lnew = getdLVal(-a1,alpha_mem,dL_mem)
    result = cterm * 1/Lnew
    return result



def Xi(h,Tf,Ai,Bi,Ci,Di,Ei,Fi):
    
    return Ai/Tf**2 + Bi*h**2 + Ci*h/Tf + Di/Tf + Ei*h + Fi



def invkf(Z1,Z2,L,T,Rb,Rr,GP,GT,hi,damage,alpha_mem,dL_mem,a0,E,hat = 1, rcbar = 0.2):
    # invkf(Z1,Z2,L,T,Rb,Rr,GP,GT,hi,damage,alpha_mem,dL_mem,a0,E = 2.068*10**11,hat = 1, rcbar = 0.2)
    '''
    hat - Tool tip addendum coefficient - From original article
    rcbar - Tool tip radius - From original article
    '''
    
    if GT == 0:
        Z = Z1
    else:
        Z = Z2
    
    a1 = A1(T,Z1,Z2,a0,GP,GT)
    a2 = A2(Z,a0)
    a3 = A3(a2,Rb,Rr)
    Lnew = L
    Tf = 1/Z * (np.pi/2 + 2*np.tan(a0)*(hat - rcbar) + 2*rcbar/np.cos(a0))
    ###########################################
    Sf = 2*Tf*Rr # I'm Guessing this is correct

    h = Rb*((a1+a2)*np.cos(a1) - np.sin(a1))
    if Rb>=Rr: # We have to add some are that we are neglecting since the tooth actually starts at Rr
        d  = Rb*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr*np.cos(a3)
    else: # Otherwise, now we just have to integrate up to a5, and not a2
        a5 = A5(Rb,Rr,a2)
        a4 = np.arcsin(Rb/Rr*((a2-a5)*np.cos(a5)-np.sin(a5)))
        d  = Rb*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr*np.cos(a4) 
        
    uf = d-h*np.tan(a1) # This is my own derivation.

    
    # CONSTANT VALUES
    ###########################################################################################################
    AL,BL,CL,DL,EL,FL =  -5.574*10**(-5), -1.9986*10**(-3), -2.3015*10**(-4),  4.7702*10**(-3),  0.0271, 6.8045
    AM,BM,CM,DM,EM,FM =  60.111*10**(-5),  28.100*10**(-3), -83.431*10**(-4), -9.9256*10**(-3),  0.1624, 0.9086
    AP,BP,CP,DP,EP,FP = -50.952*10**(-5),  185.50*10**(-3),  0.0538*10**(-4),  53.300*10**(-3),  0.2895, 0.9236
    AQ,BQ,CQ,DQ,EQ,FQ = -6.2042*10**(-5),  9.0889*10**(-3), -4.0964*10**(-4),  7.8297*10**(-3), -0.1472, 0.6904
    ###########################################################################################################

    Lstar = Xi(hi,Tf,AL,BL,CL,DL,EL,FL)
    Mstar = Xi(hi,Tf,AM,BM,CM,DM,EM,FM)
    Pstar = Xi(hi,Tf,AP,BP,CP,DP,EP,FP)
    Qstar = Xi(hi,Tf,AQ,BQ,CQ,DQ,EQ,FQ)
    
#     if damage == 1 and GP == 1 and GT == 0:
#         Lnew = getdLVal(-a1,alpha_mem,dL_mem)
    
    result = np.cos(a1)**2/(E*Lnew)*(Lstar*(uf/Sf)**2 + Mstar*(uf/Sf) + Pstar*(1+Qstar*np.tan(a1)**2))
    return result
####################################################################


######################## FAULT INTERPOLATORS ########################
def getdLVal(alpha_vals,alpha_mem,dL_mem):
    dL = np.interp(alpha_vals,alpha_mem,dL_mem)
    return dL

def getdAxVal(alpha_vals,alpha_mem,dAx_mem):
    dAx = np.interp(alpha_vals,alpha_mem,dAx_mem)
    return dAx

def getdIxVal(alpha_vals,alpha_mem,dIx_mem):
    dIx = np.interp(alpha_vals,alpha_mem,dIx_mem)
    return dIx
####################################################################


######################### FAULT GENERATION FUNCTIONS ###########################
# def generateCrackingFaults(q0,q2,v,Wc,Z1,Z2,L,Rb,Rr,no_points,a0):
#     term1 = np.tan(np.arccos((Z1*np.cos(a0))/(Z1+2)))
#     term2 = -2*np.pi/Z1
#     term3 = -np.tan(np.arccos(Z1*np.cos(a0)/np.sqrt((Z2+2)**2 +(Z1+Z2)**2-2*(Z2+2)*(Z1+Z2)*np.cos(np.arccos(Z2*np.cos(a0)/(Z2+2))-a0))))

#     Td = (term1 + term2 + term3) # Double Meshing Angle
#     Ts = 2*np.pi/Z1 - Td # Single Meshing Angle

#     Theta_mem = Td+Ts+Td
    
#     a1 = A1(Theta_mem,Z1,Z2,a0,1,0) # I fix the gear tooth to the driving gear from pair 1
#     a2 = A2(Z1,a0)
#     a3 = A3(a2,Rb,Rr)
    


    
#     # THE ASSUMPTION IS THAT WE START CRACK AT TOOTH ROOT, THUS AT HX = RBSINA2
    
    
#     if Rb>Rr:
#         d1 = np.sqrt(Rb**2 + Rr**2 - 2*Rb*Rr*np.cos(a3-a2)) # This is 0.48% different from just simply saying Rb-Rr    
#         alpha_mem = np.linspace(-a1,a2,no_points)
#         d  = Rb*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr*np.cos(a3)
#         xarr = Rb*(-(-alpha_mem+a2)*np.sin(alpha_mem)+np.cos(alpha_mem) ) - Rr*np.cos(a3)
#         hc = Rb*np.sin(a2) # a = a2 in equation for hx.

        
#     else:
#         d1 = -100
#         a5 = A5(Rb,Rr,a2)
# #         a4 = np.arcsin(Rb/Rr*((a2-a5)*np.cos(a5)-np.sin(a5))) - Delivers same result
#         a4 = np.arccos(Rb/Rr*((a5-a2)*np.sin(a5) + np.cos(a5))) # Comes from def of x=0
#         alpha_mem = np.linspace(-a1,a5,no_points)        
#         d  = Rb*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr*np.cos(a4) 
#         xarr = Rb*(-(-alpha_mem+a2)*np.sin(alpha_mem)+np.cos(alpha_mem) ) - Rr*np.cos(a4)
#         hc = Rb*((a2-a5)*np.cos(a5) + np.sin(a5)) # a = a5 in equation for hx.

#     L_mem = np.ones(len(alpha_mem))*L
#     Ax_mem = np.zeros(len(alpha_mem))
#     Ix_mem = np.zeros(len(alpha_mem))


#     hmax = Rb*np.sin(a2)
#     for i in range(len(alpha_mem)):
#         x = xarr[i]
#         if 0<=x and x<= d1:
#             hx = Rb*np.sin(a2) # This corresponds to a straight line to the root circle. Cool.

#         elif d1 < x and x<=d:
#             hx = Rb*((a2-alpha_mem[i])*np.cos(alpha_mem[i]) + np.sin(alpha_mem[i]))
            
#         else:
#             print('Error in Code')
#             print(i,x,d)
        
#         q0n = q0*np.sin(v)
#         q2n = q2*np.sin(v)
        
#         if q2n == 0: # Not full width crack   
#             zi = 0
#             if hx >= (hmax-q0n): # In cracked region
#                 zi = Wc*(1-((hmax-hx)/q0n)**2)
#             hxbar = zi/L*(hmax-hx) + hx + 2*q0n/(3*(L*Wc**0.5)) * ((Wc-zi)**1.5 - (Wc)**1.5)
                
#         else: # Full width crack type. Formulation of qz changes (and zi in turn)
#             if q0n == q2n:
#                 hxbar = hmax - q0n
                
#             else:
#                 zi = 0
#                 if hx >= (hmax-q0n) and hx<=(hmax-q2n): #Region between q0 and q2 (A cracked region)
#                     zi = (L*((hmax-hx)**2 - q0n**2))/(q2n**2 - q0n**2)
#                 elif hx >= (hmax-q0n) and hx>(hmax-q2n): # Region above q2
#                     zi = L

#                 hxbar = zi/L*(hmax-hx) + hx -  (2)/(3*(q2n**2 - q0n**2)) * (((q2n**2 - q0n**2)/(L) * zi + q0n**2)**1.5 - q0n**3)


        
#         Ax_mem[i] = (hxbar+hx)*L
#         Ix_mem[i] = 1/12 * (hxbar+hx)**3 *L


#     return alpha_mem, L_mem, Ax_mem, Ix_mem

def generateHealthy(Z1,Z2,L,Rb,Rr,no_points,a0):
    term1 = np.tan(np.arccos((Z1*np.cos(a0))/(Z1+2)))
    term2 = -2*np.pi/Z1
    term3 = -np.tan(np.arccos(Z1*np.cos(a0)/np.sqrt((Z2+2)**2 +(Z1+Z2)**2-2*(Z2+2)*(Z1+Z2)*np.cos(np.arccos(Z2*np.cos(a0)/(Z2+2))-a0))))

    Td = (term1 + term2 + term3) # Double Meshing Angle
    Ts = 2*np.pi/Z1 - Td # Single Meshing Angle

    Theta_mem = Td+Ts+Td
    
    a1 = A1(Theta_mem,Z1,Z2,a0,1,0) # I fix the gear tooth to the driving gear from pair 1
    a2 = A2(Z1,a0)
    a3 = A3(a2,Rb,Rr)
    
    
    
    ########################################
    
    if Rb>Rr:
        d1 = np.sqrt(Rb**2 + Rr**2 - 2*Rb*Rr*np.cos(a3-a2)) # This is 0.48% different from just simply saying Rb-Rr    
        alpha_mem = np.linspace(-a1,a2,no_points)
        d  = Rb*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr*np.cos(a3)
        xarr = Rb*(-(-alpha_mem+a2)*np.sin(alpha_mem)+np.cos(alpha_mem) ) - Rr*np.cos(a3)

    else:
        d1 = -100 # just negative enough to keep any small negative x's out of the else statement below...
        a5 = A5(Rb,Rr,a2)
        a4 = np.arccos(Rb/Rr*((a5-a2)*np.sin(a5) + np.cos(a5)))
        alpha_mem = np.linspace(-a1,a5,no_points)        
        d  = Rb*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr*np.cos(a4) 
        xarr = Rb*(-(-alpha_mem+a2)*np.sin(alpha_mem)+np.cos(alpha_mem) ) - Rr*np.cos(a4)
    ######################################

 
    dL_mem = np.zeros(len(xarr))
    dAx_mem = np.zeros(len(xarr))
    dIx_mem = np.zeros(len(xarr))
    
    
    for i in range(len(xarr)):
        x = xarr[i]
        if 0<=x and x<= d1:
            hx = Rb*np.sin(a2) # This corresponds to a straight line to the root circle. Cool.

        elif d1 < x and x<=d:
            hx = Rb*((a2-alpha_mem[i])*np.cos(alpha_mem[i]) + np.sin(alpha_mem[i]))

        else:
            raise ValueError('Error in Code')
            raise ValueError(i,x,d)
        dL_mem = L
        dAx_mem[i] = 2*hx*L
        dIx_mem[i] = 2/3*hx**3 * L

  
    return alpha_mem, dL_mem, dAx_mem, dIx_mem




def generateCrackingFaults(q0,q2,v,Wc,Z1,Z2,L,Rb,Rr,no_points,a0):
    term1 = np.tan(np.arccos((Z1*np.cos(a0))/(Z1+2)))
    term2 = -2*np.pi/Z1
    term3 = -np.tan(np.arccos(Z1*np.cos(a0)/np.sqrt((Z2+2)**2 +(Z1+Z2)**2-2*(Z2+2)*(Z1+Z2)*np.cos(np.arccos(Z2*np.cos(a0)/(Z2+2))-a0))))

    Td = (term1 + term2 + term3) # Double Meshing Angle
    Ts = 2*np.pi/Z1 - Td # Single Meshing Angle

    Theta_mem = Td+Ts+Td
    
    a1 = A1(Theta_mem,Z1,Z2,a0,1,0) # I fix the gear tooth to the driving gear from pair 1
    a2 = A2(Z1,a0)
    a3 = A3(a2,Rb,Rr)
    


    
    # THE ASSUMPTION IS THAT WE START CRACK AT TOOTH ROOT, THUS AT HX = RBSINA2
    
    
    if Rb>Rr:
        d1 = np.sqrt(Rb**2 + Rr**2 - 2*Rb*Rr*np.cos(a3-a2)) # This is 0.48% different from just simply saying Rb-Rr    
        alpha_mem = np.linspace(-a1,a2,no_points)
        d  = Rb*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr*np.cos(a3)
        xarr = Rb*(-(-alpha_mem+a2)*np.sin(alpha_mem)+np.cos(alpha_mem) ) - Rr*np.cos(a3)
        hc = Rb*np.sin(a2) # a = a2 in equation for hx.

        
    else:
        d1 = -100
        a5 = A5(Rb,Rr,a2)
#         a4 = np.arcsin(Rb/Rr*((a2-a5)*np.cos(a5)-np.sin(a5))) - Delivers same result
        a4 = np.arccos(Rb/Rr*((a5-a2)*np.sin(a5) + np.cos(a5))) # Comes from def of x=0
        alpha_mem = np.linspace(-a1,a5,no_points)        
        d  = Rb*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr*np.cos(a4) 
        xarr = Rb*(-(-alpha_mem+a2)*np.sin(alpha_mem)+np.cos(alpha_mem) ) - Rr*np.cos(a4)
        hc = Rb*((a2-a5)*np.cos(a5) + np.sin(a5)) # a = a5 in equation for hx.

    L_mem = np.ones(len(alpha_mem))*L
    Ax_mem = np.zeros(len(alpha_mem))
    Ix_mem = np.zeros(len(alpha_mem))


    hmax = 2*Rb*np.sin(a2)
    for i in range(len(alpha_mem)):
        x = xarr[i]
        if 0<=x and x<= d1:
            hx = 2*Rb*np.sin(a2) # This corresponds to a straight line to the root circle. Cool.

        elif d1 < x and x<=d:
            hx = 2*Rb*((a2-alpha_mem[i])*np.cos(alpha_mem[i]) + np.sin(alpha_mem[i]))
            
        else:
            raise ValueError('Error in Code: {} {} {}'.format(i,x,d))
            #print(i,x,d)
        
        q0n = q0*np.sin(v)
        q2n = q2*np.sin(v)
        if q2n == 0: # Not full width crack   
            zi = 0
            if hx >= (hmax-q0n): # In cracked region
                zi = Wc*(1-((hmax-hx)/q0n)**2)
            hxbar = zi/L*(hmax-hx) + hx + 2*q0n/(3*(L*Wc**0.5)) * ((Wc-zi)**1.5 - (Wc)**1.5)
            hxbar = hxbar/2 # Have to do this to compensate for the *2 below
                
        else: # Full width crack type. Formulation of qz changes (and zi in turn)
            if q0n == q2n:
                hxbar = hmax - q0n
                hxbar = hxbar/2 # Have to do this to compensate for the *2 below
                
            else:
                zi = 0
                if hx >= (hmax-q0n) and hx<=(hmax-q2n): #Region between q0 and q2 (A cracked region)
                    zi = (L*((hmax-hx)**2 - q0n**2))/(q2n**2 - q0n**2)
                elif hx >= (hmax-q0n) and hx>(hmax-q2n): # Region above q2
                    zi = L

                hxbar = zi/L*(hmax-hx) + hx -  (2)/(3*(q2n**2 - q0n**2)) * (((q2n**2 - q0n**2)/(L) * zi + q0n**2)**1.5 - q0n**3)
                hxbar = hxbar/2 # Have to do this to compensate for the *2 below

        
        Ax_mem[i] = (hxbar*2)*L
        Ix_mem[i] = 1/12 * (hxbar*2)**3 *L


    return alpha_mem, L_mem, Ax_mem, Ix_mem


def generateChippingFaults(b,c,Z1,Z2,L,Rb,Rr,no_points,a0):
    term1 = np.tan(np.arccos((Z1*np.cos(a0))/(Z1+2)))
    term2 = -2*np.pi/Z1
    term3 = -np.tan(np.arccos(Z1*np.cos(a0)/np.sqrt((Z2+2)**2 +(Z1+Z2)**2-2*(Z2+2)*(Z1+Z2)*np.cos(np.arccos(Z2*np.cos(a0)/(Z2+2))-a0))))

    Td = (term1 + term2 + term3) # Double Meshing Angle
    Ts = 2*np.pi/Z1 - Td # Single Meshing Angle

    Theta_mem = Td+Ts+Td
    
    a1 = A1(Theta_mem,Z1,Z2,a0,1,0) # I fix the gear tooth to the driving gear from pair 1
    a2 = A2(Z1,a0)
    a3 = A3(a2,Rb,Rr)
    
    
    
    ########################################
    
    if Rb>Rr:
        d1 = np.sqrt(Rb**2 + Rr**2 - 2*Rb*Rr*np.cos(a3-a2)) # This is 0.48% different from just simply saying Rb-Rr    
        alpha_mem = np.linspace(-a1,a2,no_points)
        d  = Rb*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr*np.cos(a3)
        xarr = Rb*(-(-alpha_mem+a2)*np.sin(alpha_mem)+np.cos(alpha_mem) ) - Rr*np.cos(a3)

    else:
        d1 = -100 # just negative enough to keep any small negative x's out of the else statement below...
        a5 = A5(Rb,Rr,a2)
        a4 = np.arccos(Rb/Rr*((a5-a2)*np.sin(a5) + np.cos(a5)))
        alpha_mem = np.linspace(-a1,a5,no_points)        
        d  = Rb*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr*np.cos(a4) 
        xarr = Rb*(-(-alpha_mem+a2)*np.sin(alpha_mem)+np.cos(alpha_mem) ) - Rr*np.cos(a4)
    ######################################

    dmax  = np.max(xarr)  
    dL_mem = np.zeros(len(xarr))
    dAx_mem = np.zeros(len(xarr))
    dIx_mem = np.zeros(len(xarr))
    
    u = dmax - b
    
    for i in range(len(xarr)):
        x = xarr[i]
        if 0<=x and x<= d1:
            hx = Rb*np.sin(a2) # This corresponds to a straight line to the root circle. Cool.

        elif d1 < x and x<=d:
            hx = Rb*((a2-alpha_mem[i])*np.cos(alpha_mem[i]) + np.sin(alpha_mem[i]))

        else:
            raise ValueError('Error in Code')
            print(i,x,d)
        dL = L
        dAx_mem[i] = 2*hx*L
        dIx_mem[i] = 2/3*hx**3 * L

        if u <= x:
            dh = u+b
            dL = L - (dh*c/b - (dh**2*c/b - dh*c)/x) 
        dL_mem[i] = dL
    return alpha_mem, dL_mem, dAx_mem, dIx_mem


def generatePittingFaults(u_range, pa_range, r_range,Z1,Z2,L,Rb,Rr,no_points,a0):
    term1 = np.tan(np.arccos((Z1*np.cos(a0))/(Z1+2)))
    term2 = -2*np.pi/Z1
    term3 = -np.tan(np.arccos(Z1*np.cos(a0)/np.sqrt((Z2+2)**2 +(Z1+Z2)**2-2*(Z2+2)*(Z1+Z2)*np.cos(np.arccos(Z2*np.cos(a0)/(Z2+2))-a0))))

    Td = (term1 + term2 + term3) # Double Meshing Angle
    Ts = 2*np.pi/Z1 - Td # Single Meshing Angle

    Theta_mem = Td+Ts+Td
    
    a1 = A1(Theta_mem,Z1,Z2,a0,1,0) # I fix the gear tooth to the driving gear from pair 1
    a2 = A2(Z1,a0)
    a3 = A3(a2,Rb,Rr)
    ########################################
    
    if Rb>Rr:
        d1 = np.sqrt(Rb**2 + Rr**2 - 2*Rb*Rr*np.cos(a3-a2)) # This is 0.48% different from just simply saying Rb-Rr    
        alpha_mem = np.linspace(-a1,a2,no_points)
        d  = Rb*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr*np.cos(a3)
        xarr = Rb*(-(-alpha_mem+a2)*np.sin(alpha_mem)+np.cos(alpha_mem) ) - Rr*np.cos(a3)

    else:
        d1 = -100 # just negative enough to keep any small negative x's out of the else statement below...
        a5 = A5(Rb,Rr,a2)
        a4 = np.arccos(Rb/Rr*((a5-a2)*np.sin(a5) + np.cos(a5)))
        alpha_mem = np.linspace(-a1,a5,no_points)        
        d  = Rb*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr*np.cos(a4) 
        xarr = Rb*(-(-alpha_mem+a2)*np.sin(alpha_mem)+np.cos(alpha_mem) ) - Rr*np.cos(a4)
    ######################################
    
    
    
    dL_mem = np.zeros((len(pa_range),len(xarr)))
    dAx_mem = np.zeros((len(pa_range),len(xarr)))
    dIx_mem = np.zeros((len(pa_range),len(xarr)))
    
    L_mem = np.zeros(len(xarr))
    Ax_mem = np.zeros(len(xarr))
    Ix_mem = np.zeros(len(xarr))
    
    for i in range(len(xarr)):
        x = xarr[i]
        if 0<=x and x<= d1:
            hx = Rb*np.sin(a2)
        elif d1 < x  and x<=d:
            
            hx = Rb*((a2-alpha_mem[i])*np.cos(alpha_mem[i]) + np.sin(alpha_mem[i]))
        else:
            raise ValueError('Error in Code')
            print(i,x,d)
        
        Ax = 2*hx*L
        Ix = 2/3*hx**3*L
            
        for j in range(len(pa_range)):
            pa = pa_range[j]
            r = r_range[j]
            u = u_range[j]

            
            if u-(r*np.sin(pa)) <= x and x <= u+(r*np.sin(pa)):
                dL = 2*np.sqrt((r*np.sin(pa))**2 - (u-x)**2)
                dL_mem[j,i] = dL

            if u-r <= x and x <= u+r:
                dR = np.sqrt((r)**2 - (u-x)**2) 
                dAx = dR**2*(2*pa-np.sin(2*pa))/2
                dAx_mem[j,i] = dAx

                term1 = dR**4/72 * (18*pa - 9*np.sin(2*pa)*np.cos(2*pa) - 64*np.sin(pa)**6/(2*pa - np.sin(2*pa))) 
                term2 = dAx*Ax*(hx-((4*dR*np.sin(pa)**3)/(3*(2*pa-np.sin(2*pa))) - dR*np.cos(pa)))**2/(Ax-dAx)

                dIx_mem[j,i] = (term1 + term2)
        
        L_mem[i] = L 
        Ax_mem[i] =  Ax
        Ix_mem[i] = Ix 
    L_mem = L_mem - np.sum(dL_mem,axis = 0)
    Ax_mem = Ax_mem - np.sum(dAx_mem,axis = 0)
    Ix_mem = Ix_mem - np.sum(dIx_mem,axis = 0)
    
    return alpha_mem, L_mem, Ax_mem, Ix_mem


def generateSpallFaults(x1,ws,ls,hs,Z1,Z2,L,Rb,Rr,no_points,a0):
    term1 = np.tan(np.arccos((Z1*np.cos(a0))/(Z1+2)))
    term2 = -2*np.pi/Z1
    term3 = -np.tan(np.arccos(Z1*np.cos(a0)/np.sqrt((Z2+2)**2 +(Z1+Z2)**2-2*(Z2+2)*(Z1+Z2)*np.cos(np.arccos(Z2*np.cos(a0)/(Z2+2))-a0))))

    Td = (term1 + term2 + term3) # Double Meshing Angle
    Ts = 2*np.pi/Z1 - Td # Single Meshing Angle

    Theta_mem = Td+Ts+Td
    
    a1 = A1(Theta_mem,Z1,Z2,a0,1,0) # I fix the gear tooth to the driving gear from pair 1
    a2 = A2(Z1,a0)
    a3 = A3(a2,Rb,Rr)
    
    
    
    ########################################
    
    if Rb>Rr:
        d1 = np.sqrt(Rb**2 + Rr**2 - 2*Rb*Rr*np.cos(a3-a2)) # This is 0.48% different from just simply saying Rb-Rr    
        alpha_mem = np.linspace(-a1,a2,no_points)
        d  = Rb*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr*np.cos(a3)
        xarr = Rb*(-(-alpha_mem+a2)*np.sin(alpha_mem)+np.cos(alpha_mem) ) - Rr*np.cos(a3)

    else:
        d1 = -100 # just negative enough to keep any small negative x's out of the else statement below...
        a5 = A5(Rb,Rr,a2)
        a4 = np.arccos(Rb/Rr*((a5-a2)*np.sin(a5) + np.cos(a5)))
        alpha_mem = np.linspace(-a1,a5,no_points)        
        d  = Rb*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr*np.cos(a4) 
        xarr = Rb*(-(-alpha_mem+a2)*np.sin(alpha_mem)+np.cos(alpha_mem) ) - Rr*np.cos(a4)
    ######################################
    L_mem = np.zeros(len(xarr))
    Ax_mem = np.zeros(len(xarr))
    Ix_mem = np.zeros(len(xarr))

    for i in range(len(xarr)):
        x = xarr[i]
        if 0<=x and x<= d1:
            hx = Rb*np.sin(a2)
        elif d1 < x  and x<=d:
            
            hx = Rb*((a2-alpha_mem[i])*np.cos(alpha_mem[i]) + np.sin(alpha_mem[i]))
        else:
            raise ValueError('Error in Code')
            print(i,x,d)
            
        L_mem[i]  = L     
        Ax_mem[i] = 2*L*hx
        Ix_mem[i] = 2/3*L*hx**3

            
        x2 = x1-ws
        
        if x <= x1 and x >= x2:
            dL = ls
            L_mem[i] = L-dL
            Ax_mem[i] = 2*L*hx - hs*dL
            Ix_mem[i] = 2/3*L*hx**3 - 1/12 * hs**3*dL


    return alpha_mem, L_mem, Ax_mem, Ix_mem
###############################################################################

######################## STIFFNESS DETERMINATION AND COMPILATION ###################
def getToothStiff(Z1,Z2,L1,L2,ThetaFeed,Rb1,Rb2,Rr1,Rr2,Ra,damage,FaultType,FaultMat,a0,E,nu,verbose=False):
    # I set the following values to 0 as they are always passed. This should be faster than calculating them
    # when we know for a fact that only 1 of the teeth needs these values. 
    alpha_mem = 0
    dL_mem = 0
    dAx_mem = 0
    dIx_mem = 0
    q0 = 0
    q2 = 0
    v = 0
    Wc = 0
    b = 0
    c = 0
    Uarr = 0
    PAarr = 0
    Rarr = 0
    x1 = 0
    ws = 0
    ls = 0
    hs = 0
    
    if FaultType == 0:
        if damage == 1: # Just a way to get it to print. Not really damaged
            if verbose: print('Healthy Gear Tooth')
            damage = 0 # So that downstream works correctly
        alpha_mem, dL_mem, dAx_mem, dIx_mem = generateHealthy(Z1,Z2,L1,Rb1,Rr1,1000,a0)
        

    if FaultType == 0 and damage == 1: # HEALTHY TOOTH FOR PRINTING
        if verbose: print('Healthy Gear Tooth')
        alpha_mem, dL_mem, dAx_mem, dIx_mem = generateHealthy(Z1,Z2,L1,Rb1,Rr1,1000,a0)
 
    if FaultType == 1 and damage == 1:
        if verbose: print('Inducing Cracking Fault on Gear Tooth')
        q0,q2,v,Wc = FaultMat[0], FaultMat[1], FaultMat[2], FaultMat[3]
        alpha_mem, dL_mem, dAx_mem, dIx_mem = generateCrackingFaults(q0,q2,v,Wc,Z1,Z2,L1,Rb1,Rr1,1000,a0) 
    
    if FaultType == 2 and damage == 1:
        if verbose: print('Inducing Chipping Fault on Gear Tooth')
        b,c = FaultMat[0], FaultMat[1]
        alpha_mem, dL_mem, dAx_mem, dIx_mem = generateChippingFaults(b,c,Z1,Z2,L1,Rb1,Rr1,1000,a0)    

    if FaultType == 3 and damage == 1:
        if verbose: print('Inducing Pitting Fault on Gear Tooth')
        Uarr, PAarr, Rarr = FaultMat[0,:], FaultMat[1,:], FaultMat[2,:]
        alpha_mem, dL_mem, dAx_mem, dIx_mem = generatePittingFaults(Uarr,PAarr,Rarr,Z1,Z2,L1,Rb1,Rr1,1000,a0)
    
    if FaultType == 4 and damage == 1:
        if verbose: print('Inducing Broken Tooth Fault on Gear Tooth')
        # I add a small value to avoid div by 0 errors. 
        alpha_mem, dL_mem, dAx_mem, dIx_mem = np.zeros(len(ThetaFeed))+1e-6,np.zeros(len(ThetaFeed))+1e-6,np.zeros(len(ThetaFeed))+1e-6,np.zeros(len(ThetaFeed))+1e-6


    if FaultType == 5 and damage == 1:
        if verbose: print('Inducing Spall Fault on Gear Tooth')
        x1,ws,ls,hs = FaultMat[0], FaultMat[1], FaultMat[2], FaultMat[3]
        alpha_mem, dL_mem, dAx_mem, dIx_mem = generateSpallFaults(x1,ws,ls,hs,Z1,Z2,L1,Rb1,Rr1,1000,a0)
    
    invkbp1 = invkb(Z1,Z2,L1,ThetaFeed, Rb1,Rr1,1,0,alpha_mem,dIx_mem,damage,a0,E)  # Driving Gear
    invkbg1 = invkb(Z1,Z2,L2,ThetaFeed, Rb2,Rr2,1,1,alpha_mem,dIx_mem,damage,a0,E) # Driven Gear

    invksp1 = invks(Z1,Z2,L1,ThetaFeed, Rb1,Rr1,1,0,alpha_mem,dAx_mem,damage,a0,E,nu) # Driving Gear
    invksg1 = invks(Z1,Z2,L2,ThetaFeed, Rb2,Rr2,1,1,alpha_mem,dAx_mem,damage,a0,E,nu) # Driven Gear

    invkap1 = invka(Z1,Z2,L1,ThetaFeed, Rb1,Rr1,1,0,alpha_mem,dAx_mem,damage,a0,E) # Driving Gear
    invkag1 = invka(Z1,Z2,L2,ThetaFeed, Rb2,Rr2,1,1,alpha_mem,dAx_mem,damage,a0,E) # Driven Gear

    invkhpg1 = invkh(Z1,Z2,L1,ThetaFeed,1,0,damage,alpha_mem,dL_mem,a0,E,nu) # Both Gears simultaneously

    h1 = Rr1/Ra
    h2 = Rr2/Ra
    invkfp1 = invkf(Z1,Z2,L1,ThetaFeed, Rb1,Rr1,1,0,h1,damage,alpha_mem,dL_mem,a0,E) # Driving Gear
    invkfg1 = invkf(Z1,Z2,L2,ThetaFeed, Rb2,Rr2,1,1,h2,damage,alpha_mem,dL_mem,a0,E) # Driven Gear    
    TotStiff = 1/(invkbp1+invkbg1+invksp1+invksg1+invkap1+invkag1+invkfp1+invkfg1+invkhpg1) 

    return TotStiff

def stiffnessCompiler(E,nu,Z1,Z2,L1,L2,Rb1,Rb2,Rr1,Rr2,Ra,a0 = np.deg2rad(20),FTN = 0,mesh_fineness = 100, FaultType = 0, FaultMat = [],verbose=False):
    '''
    Z1 - Number of teeth on DRIVING gear
    Z2 - Number of teeth on DRIVEN gear
    L1 - Width of DRIVING gear
    L2 - Width of DRIVEN gear
    Rb1 - Base Radius of DRIVING gear
    Rb2 - Base Radius of DRIVEN gear
    Rr1 - Root Radius of DRIVING gear
    Rr2 - Root Radius of DRIVEN gear
    a0 - Pressure Angle (Default is 20 degrees - Should be sufficient for most applications.)
    FTN - Tooth number which must have the fault (default 0): 0 indicates a healthy gear (Note we always assume a fault on the DRIVING gear)
    mesh_fineness - How many interpolation points you wish a single meshing period to have;
                    This includes a single and double meshing pair combo. Default is 100 points.
    
    FaultType (Default 1, but won't matter if FTN = 0):
    Dimensions:
    x - This direction refers to the direction radially outward from the gear center
    y - This direction refers to the direction describing the tooth height (perpendicular to x)
    z - This direction refers to the direction describing the tooth width (where Hertzian contact acts)
    I make these dimensions clear, as I add them to the fault parameters so the user knows in which dimension they are changing fault parameters.
    
    Labels:
    1 - Crack (Requires a fault matrix of the shape [float:crackAngle, float: Crack width, float: Left side crack height, float: Right side crack height]
    2 - Surface Chip (Requires a fault matrix of the shape [float: Chip length (x), float: Chip width (z)])
    3 - Pitting Defects (Requires a fault matrix of the shape [array: depths of pits (x), array: radius of pits, array: angle of pits])
    4 - Broken Tooth (Requires no fault matrix - Simply pass a 0 as a placeholder (or leave the value out completely))
    5 - Spalling (Requires a fault matrix of the shape [float: Spall Start (x), float: Spall Width (z), float: Spall Length (x), int: Spall Depth (y)])
    
    All quantities are to be read in as SI units. Angles are to be read in as radians.
    FaultMat - Matrix containing the fault information (Default [0,0,0,0])
    '''
    # Creating the necessary simulation angles for a single tooth meshing pair.
    Td = DoubleMeshPeriod(Z1,Z2,a0) # Double Meshing Angle
    Ts = 2*np.pi/Z1 - Td # Single Meshing Angle
    new_fineness = int(mesh_fineness*(1+Td/(Td+Ts))) # I'm increasing the fineness to still equate to mesh_fineness for Td+Ts
    ThetaFeed = np.linspace(0,Td+Ts+Td,new_fineness)
    
    
    # Compiling the stiffness array for any given rotation angle.
    kti = np.zeros((Z1,mesh_fineness*Z1))
    for i in range(1,Z1+1): # Go through each tooth:
        if i == FTN and FTN != 0: # If we have a faulty tooth, we need to pass a damage parameter.
            kti[(i-1),(i-1)*mesh_fineness:((i-1))*mesh_fineness+new_fineness] = getToothStiff(Z1,Z2,L1,L2,ThetaFeed,Rb1,Rb2,Rr1,Rr2,Ra,1,FaultType, FaultMat,a0,E,nu,verbose=verbose)
            
        elif i==Z1: # If we're at the very last tooth, we need to change our indicies slightly
            kti[(i-1),(i-1)*mesh_fineness:((i-1))*mesh_fineness+mesh_fineness] = getToothStiff(Z1,Z2,L1,L2,ThetaFeed,Rb1,Rb2,Rr1,Rr2,Ra,0,FaultType, FaultMat,a0,E,nu,verbose=verbose)[0:mesh_fineness]
            kti[(i-1),0:int(new_fineness-mesh_fineness)] = getToothStiff(Z1,Z2,L1,L2,ThetaFeed,Rb1,Rb2,Rr1,Rr2,Ra,0,FaultType, FaultMat,a0,E,nu,verbose=verbose)[mesh_fineness:new_fineness]

        else: # For any other normal tooth
            kti[(i-1),(i-1)*mesh_fineness:((i-1))*mesh_fineness+new_fineness] = getToothStiff(Z1,Z2,L1,L2,ThetaFeed,Rb1,Rb2,Rr1,Rr2,Ra,0,FaultType, FaultMat,a0,E,nu,verbose=verbose)

    return np.linspace(0,2*np.pi,mesh_fineness*Z1),np.sum(kti,axis = 0)



