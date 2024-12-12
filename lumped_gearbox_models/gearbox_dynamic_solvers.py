"""

These set of functions contain a wrapper around the "stiffness_solver.py" functions written by Luke van Eyk.

The rationale behind the wrappers are as follows:

GearboxModel Class:
A class is created for each Gearbox Model in the format GearboxModel_<Original_Authors or Descriptive Name>_NumberDOF.
The NECESSARY default parameters are defined by a function, whereafter the additional parameters are DERIVED.
Thereafter, the mass and stiffness matrices are calculated and important quantities such as the angle of the pinion are also returned.
The class inherents the GearboxModelUtils class to leverage the available functions that will be shared between the classes (e.g. Newmark integration)
Please try to keep the notation consistent, e.g. number of gear teeth. The GearboxModelUtils has a basic check for this.

GearStiffness Class:
The gear mesh stiffness is calculated with a class in the following format: GearStiffness_<Description of the case>.
This class calculates the gear mesh stiffness, plots the damage area and calculates the size of the damage. 
This class derives the information of the gears from the GearboxModel Class that is given as an input.

It is possible that the workflow of the functions can be improved, but the wrappers were written to achieve the objective of simulating gears in different scenerios very quickly.

"""

# We can use this to print additional functions.
DEBUG = False

import numpy as np 
import matplotlib.pyplot as plt
from . import stiffness_solver as lss

# =======================================================================================================
# Auxiliary functions:
# =======================================================================================================

def calculate_mass_and_inertia_of_disk(rho,diameter,thickness):
    """_summary_

    Parameters
    -------------
        rho (_type_): _description_
        diameter (_type_): _description_
        thickness (_type_): _description_

    Returns
    -------------    
        _type_: _description_
    """
    A = np.pi * (diameter/2.0)**2.0

    m = rho * A * thickness

    I = 1/2.0 * m * (diameter/2.0)**2.0

    return m,I

def calculate_gear_radii(Z1,Z2,m1,m2,pressure_angle,ha = 1, c = 0.25):
    '''
    Parameters
    ----------------
    Z1 - Driving gear number of teeth
    Z2 - Driven gear number of teeth
    m1 - Driving gear module (in metres)
    m2 - Driven gear module (in metres)
    a0 - Pressure angle (radians)
    ha - Addendum coefficient (The default of 1 is a commonly accepted value)
    c  - Tip clearance coefficient (The default of 0.25 is a commonly accepted value)
    
    Returns
    ----------------
    Rb1: float
        Base circle radius for pinion
        
    Rr1: float
        Root circle radius for pinion
        
    Rb2: float
        Base circle radius for gear
        
    Rr2: float
        Root circle radius for gear
        
    Rp1: float
        Pitch circle radius for pinion
        
    Rp2: float
        Pitch circle radius for gear        
    
    '''
    a0 = float(pressure_angle)
    Rb1 = 0.5*m1*Z1*np.cos(a0)  # Base circle radius for pinion
    Rr1 = 0.5*m1*Z1 - (ha+c)*m1 # Root circle radius for pinion

    Rb2 = 0.5*m2*Z2*np.cos(a0)  # Base circle radius for gear
    Rr2 = 0.5*m2*Z2 - (ha+c)*m2 # Root circle radius for gear
    
    Rp1 = None # Still need to calculate
    Pp2 = None # Still need to calculate
    
    return Rb1,Rr1,Rb2,Rr2,Rp1,Pp2

def make_gear_geometry_dictionary(pressure_angle_rad,
                                  number_of_teeth_pinion,
                                  number_of_teeth_gear,
                                  width_of_gear1_m,
                                  width_of_gear2_m,
                                  module1_m,
                                  module2_m,
                                  radius_hub_bore_m):
    """_summary_

    Parameters
    ----------------    
        pressure_angle_rad (float): The pressure angle of the gears in radians.
        number_of_teeth_pinion (int): The number of teeth on the pinion.
        number_of_teeth_gear (int): The number of teeth on the gear.
        width_of_gear1_m (float): The width of the pinion in meters.
        width_of_gear2_m (float): The width of the gear in meters.
        module1_m (float): The module of the pinion in meters.
        module2_m (float): The module of the gear in meters.
        radius_hub_bore_m (float): The hub bore radias in meters.

    Returns
    ----------------    
        _type_: _description_
    """
    a0 = pressure_angle_rad # Pressure angle
    Z1 = number_of_teeth_pinion # Number of pinion teeth
    Z2 = number_of_teeth_gear # Number of gear teeth
    L1 = width_of_gear1_m #0.016#0.038 # Pinion width
    L2 = width_of_gear2_m #0.016#0.038 # Gear width
    mod1 = module1_m # Module of pinion
    mod2 = module2_m # Module of gear
    Ra = radius_hub_bore_m # Hube bore radius (No formula for this. You simply need to have it.)
    
    Rb1,Rr1,Rb2,Rr2,Rp1,Rp2 = calculate_gear_radii(Z1,Z2,mod1,mod2,a0,ha = 1, c = 0.25) # lss.RadiusFinder(Z1,Z2,mod1,mod2,a0) # If you dont have the module, you need to be given the radii directly.
    a2 = lss.A2(Z1,a0)
    a3 = lss.A3(a2,Rb1,Rr1)
    d11 = np.sqrt(Rb1**2 + Rr1**2 - 2*Rb1*Rr1*np.cos(a3-a2)) # Gives us a sort of reference to utilize if we'd like.
    
    Td = lss.DoubleMeshPeriod(Z1,Z2,a0) # Double Meshing Angle
    Ts = 2.0*np.pi/Z1 - Td # Single Meshing Angle
    Tlarge = 2.0*Td+Ts
    
    a1 = lss.A1(Tlarge,Z1,Z2,a0,1,0)
    d1 = np.sqrt(Rb1**2 + Rr1**2 - 2*Rb1*Rr1*np.cos(a3-a2)) # Gives us a sort of reference to utilize if we'd like.
    #xmax = Rb1*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr1*np.cos(a3)
    #xmin = Rb1*((-a2+a2)*np.sin(-a2)+np.cos(a2)) - Rr1*np.cos(a3) # Below this we are in d1 region, where we have already integrated analytically assuming no faults!!    
    
    return {
            "a0": pressure_angle_rad,
            "Z1": number_of_teeth_pinion,
            "Z2": number_of_teeth_gear,
            "L1": width_of_gear1_m,        
            "L2": width_of_gear2_m,        
            "mod1": module1_m,
            "mod2": module2_m,
            "Ra": radius_hub_bore_m,
            "Rb1": Rb1,
            "Rr1": Rr1,
            "Rb2": Rb2,
            "Rr2": Rr2,
            "Td": Td,
            "Ts": Ts,
            "Tlarge": Tlarge,
            "a1": a1,
            "a2": a2,
            "a3": a3,
            "d1": d1,
            "Rp1": Rp1,
            "Rp2": Rp2,
            }

# =======================================================================================================
# Gearbox Model Utility Functions and Classes:
# =======================================================================================================

class GearboxModelUtils: 
    """
    """
    def __init__(self):
        pass
    
    def _set_parms_of_dict(self,parminput):
        """
        If the user supplied some parameters (defined in the dictionary parminput), 
        then this function writes those parameters to self.parms. If there is a key
        in parminput not expected, then an Exception is thrown. 
        
        Note: The default parameters are assumed if no parameters/incomplete parameters are given.
        """
        if self.parms is None:
            default_parms = self.get_parameter_dict_default()
        else:
            default_parms = self.parms
        self.parms = dict(default_parms)
        keys_of_parms = default_parms.keys()
        for ikey in parminput:
            if ikey not in keys_of_parms:
                raise ValueError("The key '{}' is not expected in the dictionary.".format(ikey))
            self.parms[ikey] = parminput[ikey]        
        return self.parms
    def _set_gear_geometry_dictionary(self):
        """
        This function adds gear geometry specific values.
        """

        ### ==============================================================================================
        ### Checking whether the default parameters are defind in the function.
        ### ==============================================================================================        
        try:
            assert "Z1"     in self.parms, "The number of teeth on the pinion needs to be defined as Z1."
            assert "Z2"     in self.parms, "The number of teeth on the gear needs to be defined as Z1."
            assert "L1"     in self.parms, "The width of the pinion is defined as L1"        
            assert "L2"     in self.parms, "The width of the gear is defined as L1"        
            assert "mod1"   in self.parms, "The module of the pinion needs to be defined as mod1."
            assert "mod2"   in self.parms, "The module of the gear needs to be defined as mod2."
            assert "a0"     in self.parms, "The pressure angle needs to be defined in self.parms"
            assert "Ra"     in self.parms, "The hub bore radius need to be defined."
        except Exception as e:
            print("The following exception was raised:",e)
            raise ValueError("Please make sure that your gearbox class have the necessary gear parameters (e.g. Z1, Z2) in the correct format.") 


        pressure_angle_rad = self.parms["a0"]
        Z1 = self.parms["Z1"]
        Z2 = self.parms["Z2"]
        L1 = self.parms["L1"]
        L2 = self.parms["L2"]
        module1 = self.parms["mod1"]
        module2 = self.parms["mod2"]
        Ra = self.parms["Ra"]
        # These parameters are defined from the gear parameters:
        geargeom = make_gear_geometry_dictionary(pressure_angle_rad,
                                              Z1,
                                              Z2,
                                              L1,
                                              L2,
                                              module1,
                                              module2,
                                              Ra)                     

        for ikey in geargeom:
            self.parms[ikey] = geargeom[ikey]  
        return self.parms

    def integrate(self,
                           gear_mesh_class,
                           tvals, 
                           applied_torque_array,
                           motor_torque_function,
                           beta1 = 0.5, 
                           beta2 = 2.25,                   
                           mean_gear_mesh_stiff = None,
                           verbose=True,
                           return_parms_in_dict=True,
                           ):
        """_summary_

        Parameters
        ---------------
            gear_mesh_class (_type_): _description_
            tvals (_type_): _description_
            applied_torque_array (_type_): _description_
            motor_torque_function (_type_): _description_
            beta1 (float, optional): _description_. Defaults to 0.5.
            beta2 (float, optional): _description_. Defaults to 2.25.
            mean_gear_mesh_stiff (_type_, optional): _description_. Defaults to None.
            verbose (bool, optional): _description_. Defaults to True.
            return_parms_in_dict (bool,optional): _description_. Defaults to True.
        """
        # Gear mesh stiffness dictionary:
        gear_mesh_class.set_verbose(verbose)
        gm_dict = gear_mesh_class.get_gear_mesh_stiffness()

        # M, Ks, Cs S:
        M = self.get_mass_matrix()
        Ks = self.get_stiffness_matrix_static()
        Cs = self.get_damping_matrix_static()
        # *** Cs = self.get_damping_matrix_static() - We should write something like this
        S = self.get_stiffness_matrix_dynamic_coefficients()    

        #         # Damping coefficients for Rayleigh damping:
        #         cm = self.parms["rayleigh_damping_m"]
        #         ck = self.parms["rayleigh_damping_k"]    

        thetaMain = gm_dict["angle_rad"]
        stiffMain = gm_dict["gear_mesh_stiffness"]

        # If the mean gear stiffness of the healthy gear is not supplied, then the mean of the current gear mesh stiffness is used.
        if mean_gear_mesh_stiff is None:
            mean_gear_mesh_stiff = np.mean(stiffMain)
        Kbar = Ks + mean_gear_mesh_stiff*S
        # *** We need to incorporate static damping values. Something like C = Cs + Cv, where Cv is as defined below and Cs is constructed from the dict like Ks.
        # *** I think different authors handle this differently, but Tian 2004 (the bedrock of this work) explicitly follows C = Cs + Cv with Cv containing the Rayleigh damping.
        Zeta = self.parms["Zeta"]
        m1, m2 = self.parms["m1"], self.parms["m2"]
        

        dt = np.diff(tvals)[0]
        wm0 = 0

        TL0 =  applied_torque_array[0] # Load Torque
        #TM0 =  Tm(Tb, gb, wm0, ws, ca1, ca2) # Motor Torque
        TM0 = motor_torque_function(wm0) 
        F0 = self.get_torque_array(TM0,TL0)

        gearbox_parm = self.parms
        #Td = gearbox_model_class.parms["Td"]
        #Ts = gearbox_model_class.parms["Ts"]    

        # NOW FOR THE ALGORITHM:
        # I need to create arrays to store my values in, and to get i and i+1
        u   = np.zeros((M.shape[1], len(tvals))) # Displacement Values
        ud  = np.zeros((M.shape[1], len(tvals))) # Velocity Values
        udd = np.zeros((M.shape[1], len(tvals))) # Acceleration Values
        p   = np.zeros((M.shape[1], len(tvals))) # Force Values
        kvals = np.zeros(len(tvals)) # Stiffness Values
        kvals[0] = gear_mesh_stiffness_extrapolate(0,thetaMain,stiffMain,gearbox_parm)
        avals = np.zeros_like(kvals)
        avals[0] = 0.0

        p[:,0] = F0
        # Now I can loop through
        pnumb = 0
        ivals = range(1,len(tvals))

        if verbose:
            print("\nStarting the integration process.")
            import time 
            time_start = time.time() 
            
        for i in ivals:

            if (verbose) & ((i % int(len(tvals)*0.1))==0):
                time_stop = time.time() - time_start
                time_avg = time_stop/(i + 1E-16)
                time_left = time_avg * (len(tvals)- i)
                print("Calculating: #{}/{}, Time elapsed: {:.3f} s, {:.3f} min, Time left: {:.3f} s, {:.3f} min".format(i,len(tvals),time_stop,time_stop/60,time_left,time_left/60))

            t = tvals[i]
            #wm = ud[2,i-1]/(2*np.pi)*60 # rpm, as TM needs rpm vals
            wm = self.get_motor_speedradps(ud[:,i-1])/(2*np.pi)*60 

            uddhat = -2/(beta2*dt**2) * u[:,i-1]    -   2/(beta2*dt) * ud[:,i-1]  -  (1-beta2)/(beta2) * udd[:,i-1]
            udhat  = -2*beta1/(beta2*dt) * u[:,i-1]    +  (1-(2*beta1)/(beta2)) * ud[:,i-1]   +  (1-(beta1)/(beta2)) * dt*udd[:,i-1]


            angle_of_pinion = self.get_pinion_angle_rad(u[:,i-1])
            kgmt = gear_mesh_stiffness_extrapolate(angle_of_pinion,thetaMain,stiffMain,gearbox_parm)


            avals[i] = angle_of_pinion
            kvals[i] = kgmt
            K = kgmt*S + Ks
            
            cm = 2 * Zeta * np.sqrt(mean_gear_mesh_stiff*(m1*m2)/(m1+m2))
            km = mean_gear_mesh_stiff
            mu = cm/km
            ct = mu * kgmt # Note kgmt varies, mean_gear_stiffness from mu does not!
            C = ct * S +  Cs 
            
            A = 2/(beta2 * dt**2) * M    +   (2*beta1)/(beta2*dt) * C     + K 

            Tv = applied_torque_array[i]
            TL = Tv # Load Torque
            TM =  motor_torque_function(wm)
            
            F = self.get_torque_array(TM,TL) #np.array([0,0,TM,0,0,0,0,TL])
            p[:,i] = F

            negative_dynamic_force = -F + np.dot(C,udhat) + np.dot(M,uddhat) # It's -F because we have to have mxdd + cxd + kx + F = 0

            # UPDATE
            u[:,i]   = np.dot(-np.linalg.inv(A),negative_dynamic_force)
            ud[:,i]  = udhat + (2*beta1)/(beta2*dt) * u[:,i]
            udd[:,i] = uddhat + 2/(beta2*dt**2) * u[:,i]     

        if verbose:
            time_stop = time.time() - time_start
            print("Finished integration after: {:.3f} sec, {:.3f}min\n".format(time_stop,time_stop/60.0))
        
        dict_out = {
            "u": u,
            "ud": ud,
            "udd": udd,
            "gear_mesh_stiffness": kvals,
            "angle_of_pinion": avals,
            "p": p,
            "t_ls": [tvals.min(),tvals.max(),len(tvals)],
            "mean_gear_mesh_stiff": mean_gear_mesh_stiff
        }
        damage_size = gear_mesh_class.get_damage_size()        
        if isinstance(damage_size,dict):
            for ikey in damage_size:
                dict_out["damage_size_" + ikey] = damage_size[ikey]

        if return_parms_in_dict:
            for ikey in self.parms:
                ikey_save = "parm_" + ikey
                if ikey_save not in dict_out:                    
                    value = self.parms[ikey]
                    if self.parms[ikey] == None: 
                        value = "None"
                    dict_out[ikey_save] = value
            dict_out["parm_KEYS"] = list(self.parms.keys())

        self.integrated_result = dict_out        
        
        return dict_out

# =======================================================================================================
# Gearbox Model Classes:
# =======================================================================================================

class GearboxModel_Chaari_8DOF(GearboxModelUtils):
    """_summary_

    Parameters
    ----------------
        GearboxModelUtils (_type_): _description_
    """
    def __init__(self):
        self._parms_set = False
        self.parms = None
        
    def get_parameter_dict_default(self):
        """
        The purpose of this function is two-fold. 
        Firstly, create a dictionary with all the essential parameters (not the derived parameters)
        Secondly, we can check that the appropriate parameters are supplied to the class using the defined keys.
        """
        parms = {}
        parms["m1"] = 1.8 # [kg] - Mass of Bearing Block 1
        parms["m2"] = 2.5 # [kg] - Mass of Bearing Block 2
        parms["mp"] = 0.6 # [kg] - Mass of Pinion
        parms["mg"] = 1.5 # [kg] - Mass of Gear
        parms["I11"] = 4.3e-3 # [kgm^2] - Motor Moment of Inertia
        parms["I12"] = 2.7e-4# [kgm^2] - Pinion Moment of Inertia
        parms["I21"] = 2.7e-3 # [kgm^2] - Gear Moment of Inertia
        parms["I22"] = 4.5e-3# [kgm^2] - Machine Moment of Inertia        
        parms["E"] = 2.068*10**11
        parms["nu"] = 0.33
        parms["rho"] = 7800        
  
        parms["a0"] = np.deg2rad(20) # Pressure angle
        parms["Z1"] = 20 # Number of pinion teeth
        parms["Z2"] = 40 # Number of gear teeth
        parms["L1"] = 0.023 #0.016#0.038 # Pinion width
        parms["L2"] = 0.023 #0.016#0.038 # Gear width
        parms["mod1"] = 0.003 # Module of pinion
        parms["mod2"] = 0.003 # Module of gear
        parms["Ra"] = 0.188*0.026 # Hube bore radius (No formula for this. You simply need to have it.)

        parms["kx1"] = 10**8 # [Nm] - Bearing Translational Stiffness
        parms["ky1"] = 10**8 # [Nm] - Bearing Translational Stiffness
        parms["kx2"] = 10**8 # [Nm] - Bearing Translational Stiffness
        parms["ky2"] = 10**8 # [Nm] - Bearing Translational Stiffness
        parms["ktheta1"] = 10**5 # [Nm/rad] - Torsional Stiffness of Shafts
        parms["ktheta2"] = 10**5 # [Nm/rad] - Torsional Stiffness of Shafts


        # I don't have values for these! The work from Chaari doesn't give these values, so I'm setting them as 0.
        parms["cx1"] = 0 # [Nm] - Bearing Translational Stiffness
        parms["cy1"] = 0 # [Nm] - Bearing Translational Stiffness
        parms["cx2"] = 0 # [Nm] - Bearing Translational Stiffness
        parms["cy2"] = 0 # [Nm] - Bearing Translational Stiffness
        parms["ctheta1"] = 0 # [Nm/rad] - Torsional Stiffness of Shafts
        parms["ctheta2"] = 0 # [Nm/rad] - Torsional Stiffness of Shafts

        parms["Zeta"] = 0.07 # [-] - Damping Constant - Typically used as 0.07 in literature.
        #         parms["rayleigh_damping_k"] = 10**-6 # [-] - Stiffness Proportionality Constant (Used to determine C from K and M)                

        return parms
    
    def set_parameter_dict(self,**parminput):
        """
        Supply the parameters with the appropriate names as keys or as a dictionary.
        """        
        # Set the parameters:
        self._set_parms_of_dict(parminput)
        # Set the gear geometry's parameters:
        self._set_gear_geometry_dictionary()                  
        #-------------
        # Calculate the mass moment of inertia using Eq. (7) and (8)
        # Chaari, F., Bartelmus, W., Zimroz, R., Fakhfakh, T. and Haddar, M., 2012. Gearbox vibration signal amplitude and frequency modulation. Shock and Vibration, 19(4), pp.635-652.
        rho = self.parms["rho"]
        if self.parms["Rp1"] is not None:
            # Pinion
            Dp1 = self.parms["Rp1"]*2.0
            thickness = self.parms["L1"]  
            self.parms["I12"] = calculate_mass_and_inertia_of_disk(rho,self.Dp1,thickness)[1]
        if self.parms["Rp2"] is not None:
            # Gear
            Dp2 = self.parms["Rp2"]*2.0
            thickness = self.parms["L2"] 
            self.parms["I21"] = calculate_mass_and_inertia_of_disk(rho,Dp2,thickness)[1]            
        self._parms_set = True
        return self.parms
    
    def get_parameters_dict(self):        
        return self.parms
            
    def get_mass_matrix(self):
        m1 = self.parms.get("m1")
        m2 = self.parms.get("m2")
        I11 = self.parms.get("I11")
        I12 = self.parms.get("I12")
        I21 = self.parms.get("I21")
        I22 = self.parms.get("I22")
        M = np.diag(np.array([m1, m1, I11, I12, m2, m2, I21,I22]))
        return M

    def get_stiffness_matrix_static(self):
        """
        The stiffness matrix is defined by K(t) = Ks + kgm * S, where this matrix is Ks.
        """        
        assert self._parms_set == True, "You first need to run set_parameters_dict."
        kx1 = self.parms.get("kx1")
        ky1 = self.parms.get("ky1")
        ktheta1 = self.parms.get("ktheta1")
        ktheta2 = self.parms.get("ktheta2")
        kx2 = self.parms.get("kx2")
        ky2 = self.parms.get("ky2")
        
        kdiag = np.diag([kx1, ky1, ktheta1, ktheta1, kx2, ky2, ktheta2, ktheta2])
        
        kdiag[2,3] = -ktheta1
        kdiag[3,2] = -ktheta1
        kdiag[7,6] = -ktheta2
        kdiag[6,7] = -ktheta2
        return kdiag

    def get_damping_matrix_static(self):
        """
        The damping matrix is defined by C(t) = Cs + cgm * S, where this matrix is Cs.
        """        
        assert self._parms_set == True, "You first need to run set_parameters_dict."
        cx1 = self.parms.get("cx1")
        cy1 = self.parms.get("cy1")
        ctheta1 = self.parms.get("ctheta1")
        ctheta2 = self.parms.get("ctheta2")
        cx2 = self.parms.get("cx2")
        cy2 = self.parms.get("cy2")
        
        cdiag = np.diag([cx1, cy1, ctheta1, ctheta1, cx2, cy2, ctheta2, ctheta2])
        
        cdiag[2,3] = -ctheta1
        cdiag[3,2] = -ctheta1
        cdiag[7,6] = -ctheta2
        cdiag[6,7] = -ctheta2
        
        return cdiag


    def get_stiffness_matrix_dynamic_coefficients(self):
        """
        The stiffness matrix is defined by K(t) = Ks + kgm * S, where this matrix is S.
        """
        assert self._parms_set == True, "You first need to run set_parameters_dict."        
        a0 = self.parms.get("a0")
        Rb1 = self.parms.get("Rb1")
        Rb2 = self.parms.get("Rb2")
        P = np.array([[np.sin(a0),np.cos(a0),0,Rb1,-np.sin(a0),-np.cos(a0),-Rb2,0]])
        S = np.dot(P.T,P)
        return S

    def get_torque_array(self,
                            torque_of_motor,
                            torque_of_load):
        """
        Supply the motor torque and the applied torque for the given time step.
        """
        torque_of_motor = float(torque_of_motor)
        torque_of_load  = float(torque_of_load)        
        return np.array([0,0,torque_of_motor,0,0,0,0,torque_of_load])

    def get_pinion_angle_rad(self,
                             displacement_response_vector):
        """
        Get the DOF that is associated with the pinion. This is used in the Newmark integration function.
        """
        assert len(displacement_response_vector.shape) == 1,"This should be an (DOF,) array."
        
        return displacement_response_vector[3]
    
    def get_motor_speedradps(self,
                        velocity_response_vector):
        """
        Get the DOF that is associated with the motor. This is used in the Newmark integration function.
        """
        assert len(velocity_response_vector.shape) == 1,"This should be an (DOF,) array."     
        
        return velocity_response_vector[2]
    
    def dof_info(self):
        str_print = (
        """This function provides information behind the physical meaning of the DOFs:
            0: X Vibration (Pinion Bearing)
            1: Y Vibration (Pinion Bearing)
            2: Motor Rotation
            3: Pinion Rotation
            4: X Vibration (Driven Bearing)
            5: Y vibration (Driven Bearing)
            6: Driven Gear Rotation
            7: Load Rotation
           """
        )
        print(str_print)
        return str_print

class GearboxModel_Luo_10DOF(GearboxModelUtils):
    '''
    Based on: Y. Luo et al./Mechanical Systems and Signal Processing 119 (2019) 155–181
    '''

    def __init__(self):
        self._parms_set = False
        self.parms = None
        
    def get_parameter_dict_default(self):
        """
        The purpose of this function is two-fold. 
        Firstly, create a dictionary with all the essential parameters (not the derived parameters)
        Secondly, we can check that the appropriate parameters are supplied to the class using the defined keys.
        """
        parms = {}
        parms["m1"] = 1.2728 # [kg] - Mass of Bearing Block 1
        parms["m2"] = 3.5367 # [kg] - Mass of Bearing Block 2
        parms["mp"] = None # [kg] - Mass of Pinion
        parms["mg"] = None # [kg] - Mass of Gear
        parms["mf"] = 18.509 # - [kg] - Mass of Gearbox Casing
        parms["I11"] = 0.016107 # [kgm^2] - Motor Moment of Inertia
        parms["I12"] = 0.0001751# [kgm^2] - Pinion Moment of Inertia
        parms["I21"] = 0.006828 # [kgm^2] - Gear Moment of Inertia
        parms["I22"] = 0.005153# [kgm^2] - Machine Moment of Inertia        
        parms["E"] = 2.068*10**11
        parms["nu"] = 0.3
        parms["rho"] = 7800        
  
        parms["a0"] = np.deg2rad(20) # Pressure angle
        parms["Z1"] = 16 # Number of pinion teeth
        parms["Z2"] = 48 # Number of gear teeth
        parms["L1"] = 0.016 #0.016#0.038 # Pinion width
        parms["L2"] = 0.016 #0.016#0.038 # Gear width
        parms["mod1"] = 0.003175 # Module of pinion
        parms["mod2"] = 0.003175 # Module of gear
        parms["Ra"] = 0.00976 # From Masters report#0.188*0.026 # Hube bore radius (No formula for this. You simply need to have it.)

        parms["kx1"] = 8.5364*10**7 # [N/m] - Bearing Translational Stiffness
        parms["ky1"] = 8.5364*10**7 # [N/m] - Bearing Translational Stiffness
        parms["kx2"] = 8.5364*10**7 # [N/m] - Bearing Translational Stiffness
        parms["ky2"] = 8.5364*10**7 # [N/m] - Bearing Translational Stiffness
        parms["kfx"] = 1.9912*10**8 # [N/m] - Casing Bolts Translational Stiffness
        parms["kfy"] = 2.036*10**8 # [N/m] - Casing Bolts Translational Stiffness
        parms["ktheta1"] = 330 # [Nm/rad] - Torsional Stiffness of Shafts
        parms["ktheta2"] = 330 # [Nm/rad] - Torsional Stiffness of Shafts

        parms["cx1"] =  2.134*10**4 # [Ns/m] - Bearing Translational Damping
        parms["cy1"] =  2.134*10**4 # [Ns/m] - Bearing Translational Damping
        parms["cx2"] =  2.134*10**4 # [Ns/m] - Bearing Translational Damping
        parms["cy2"] =  2.134*10**4 # [Ns/m] - Bearing Translational Damping
        parms["cfx"] =  1995.56 # [Ns/m] - Casing Bolts Translational Damping
        parms["cfy"] =  2005.8 # [Ns/m] - Casing Bolts Translational Damping
        parms["ctheta1"] =  23.1 # [Nms/rad] - Torsional Damping of Shafts
        parms["ctheta2"] =  23.1 # [Nms/rad] - Torsional Damping of Shafts

        parms["Zeta"] = 0.07 # [-] - Damping Constant - Typically used as 0.07 in literature.
        #         parms["rayleigh_damping_m"] = 0.05 # [-] - Mass Proportionality Constant (Used to determine C from K and M)
        #         parms["rayleigh_damping_k"] = 10**-6 # [-] - Stiffness Proportionality Constant (Used to determine C from K and M)                

        return parms
    
    def set_parameter_dict(self,**parminput):
        """
        Supply the parameters with the appropriate names as keys or as a dictionary.
        """        
        # Set the parameters:
        self._set_parms_of_dict(parminput)
        # Set the gear geometry's parameters:
        self._set_gear_geometry_dictionary()                  
        #-------------
        # Calculate the mass moment of inertia using Eq. (7) and (8)
        # Chaari, F., Bartelmus, W., Zimroz, R., Fakhfakh, T. and Haddar, M., 2012. Gearbox vibration signal amplitude and frequency modulation. Shock and Vibration, 19(4), pp.635-652.
        rho = self.parms["rho"]
        if self.parms["Rp1"] is not None:
            # Pinion
            Dp1 = self.parms["Rp1"]*2.0
            thickness = self.parms["L1"]
            self.parms["I12"] = calculate_mass_and_inertia_of_disk(rho,self.Dp1,thickness)[1]
        if self.parms["Rp2"] is not None:
            # Gear
            Dp2 = self.parms["Rp2"]*2.0
            thickness = self.parms["L2"]            
            self.parms["I21"] = calculate_mass_and_inertia_of_disk(rho,Dp2,thickness)[1]            
        self._parms_set = True
        return self.parms
    
    def get_parameters_dict(self):        
        return self.parms
            
    def get_mass_matrix(self):
        m1 = self.parms.get("m1")
        m2 = self.parms.get("m2")
        mf = self.parms.get("mf")
        I11 = self.parms.get("I11")
        I12 = self.parms.get("I12")
        I21 = self.parms.get("I21")
        I22 = self.parms.get("I22")
        M = np.diag(np.array([m1, m1, I11, I12, m2, m2, I21,I22, mf, mf]))
        return M

    def get_stiffness_matrix_static(self):
        """
        The stiffness matrix is defined by K(t) = Ks + kgm * S, where this matrix is Ks.
        """        
        assert self._parms_set == True, "You first need to run set_parameters_dict."
        kx1 = self.parms.get("kx1")
        ky1 = self.parms.get("ky1")
        ktheta1 = self.parms.get("ktheta1")
        ktheta2 = self.parms.get("ktheta2")
        kx2 = self.parms.get("kx2")
        ky2 = self.parms.get("ky2")
        kfx = self.parms.get("kfx")
        kfy = self.parms.get("kfy")
        
        kdiag = np.diag([kx1, ky1, ktheta1, ktheta1, kx2, ky2, ktheta2, ktheta2, kfx+kx1+kx2, kfy+ky1+ky2])
        
        kdiag[2,3] = -ktheta1
        kdiag[3,2] = -ktheta1
        kdiag[7,6] = -ktheta2
        kdiag[6,7] = -ktheta2
        
        # Bolts effect on x and y vibrations
        kdiag[0,8] = -kx1
        kdiag[4,8] = -kx2
        kdiag[1,9] = -ky1
        kdiag[5,9] = -ky2
        # # x and y vibrations effect on Bolts - Removing this makes all acc values 0 in bolts DOF!
        kdiag[8,0] = -kx1
        kdiag[8,4] = -kx2
        kdiag[9,1] = -ky1
        kdiag[9,5] = -ky2
        return kdiag

    def get_damping_matrix_static(self):
        """
        The damping matrix is defined by C(t) = Cs + cgm * S, where this matrix is Cs.
        """        
        assert self._parms_set == True, "You first need to run set_parameters_dict."
        cx1 = self.parms.get("cx1")
        cy1 = self.parms.get("cy1")
        ctheta1 = self.parms.get("ctheta1")
        ctheta2 = self.parms.get("ctheta2")
        cx2 = self.parms.get("cx2")
        cy2 = self.parms.get("cy2")
        cfx = self.parms.get("cfx")
        cfy = self.parms.get("cfy")
        
        cdiag = np.diag([cx1, cy1, ctheta1, ctheta1, cx2, cy2, ctheta2, ctheta2, cfx+cx1+cx2, cfy+cy1+cy2])
        
        cdiag[2,3] = -ctheta1
        cdiag[3,2] = -ctheta1
        cdiag[7,6] = -ctheta2
        cdiag[6,7] = -ctheta2
        
        # Bolts effect on x and y vibrations
        cdiag[0,8] = -cx1
        cdiag[4,8] = -cx2
        cdiag[1,9] = -cy1
        cdiag[5,9] = -cy2
        # # x and y vibrations effect on Bolts - Removing this makes all acc values 0 in bolts DOF!
        cdiag[8,0] = -cx1
        cdiag[8,4] = -cx2
        cdiag[9,1] = -cy1
        cdiag[9,5] = -cy2
        return cdiag

    def get_stiffness_matrix_dynamic_coefficients(self):
        """
        The stiffness matrix is defined by K(t) = Ks + kgm * S, where this matrix is S.
        """
        assert self._parms_set == True, "You first need to run set_parameters_dict."        
        a0 = self.parms.get("a0")
        Rb1 = self.parms.get("Rb1")
        Rb2 = self.parms.get("Rb2")
        P = np.array([[np.sin(a0),np.cos(a0),0,Rb1,-np.sin(a0),-np.cos(a0),-Rb2,0, 0, 0]])
        S = np.dot(P.T,P)
        return S

    def get_torque_array(self,
                            torque_of_motor,
                            torque_of_load):
        """
        Supply the motor torque and the applied torque for the given time step.
        """
        torque_of_motor = float(torque_of_motor)
        torque_of_load  = float(torque_of_load)        
        return np.array([0,0,torque_of_motor,0,0,0,0,torque_of_load, 0, 0])

    def get_pinion_angle_rad(self,
                             displacement_response_vector):
        """
        Get the DOF that is associated with the pinion. This is used in the Newmark integration function.
        """
        assert len(displacement_response_vector.shape) == 1,"This should be an (DOF,) array."
        
        return displacement_response_vector[3]
    
    def get_motor_speedradps(self,
                        velocity_response_vector):
        """
        Get the DOF that is associated with the motor. This is used in the Newmark integration function.
        """
        assert len(velocity_response_vector.shape) == 1,"This should be an (DOF,) array."     
        
        return velocity_response_vector[2]
    
    def dof_info(self):
        str_print = (
        """This function provides information behind the physical meaning of the DOFs:
            0: X Vibration (Pinion Bearing)
            1: Y Vibration (Pinion Bearing)
            2: Motor Rotation
            3: Pinion Rotation
            4: X Vibration (Driven Bearing)
            5: Y vibration (Driven Bearing)
            6: Driven Gear Rotation
            7: Load Rotation
            8: X Vibration (Gearbox Casing)
            9: Y Vibration (Gearbox Casing)
           """
        )
        print(str_print)
        return str_print

class GearboxModel_Luo_8DOF(GearboxModelUtils):
    '''
    Based on: Y. Luo et al./Mechanical Systems and Signal Processing 119 (2019) 155–181
    '''


    def __init__(self):
        self._parms_set = False
        self.parms = None
        
    def get_parameter_dict_default(self):
        """
        The purpose of this function is two-fold. 
        Firstly, create a dictionary with all the essential parameters (not the derived parameters)
        Secondly, we can check that the appropriate parameters are supplied to the class using the defined keys.
        """
        parms = {}
        parms["m1"] = 1.2728 # [kg] - Mass of Bearing Block 1
        parms["m2"] = 3.5367 # [kg] - Mass of Bearing Block 2
        parms["mp"] = None # [kg] - Mass of Pinion
        parms["mg"] = None # [kg] - Mass of Gear
        parms["I11"] = 0.016107 # [kgm^2] - Motor Moment of Inertia
        parms["I12"] = 0.0001751# [kgm^2] - Pinion Moment of Inertia
        parms["I21"] = 0.006828 # [kgm^2] - Gear Moment of Inertia
        parms["I22"] = 0.005153# [kgm^2] - Machine Moment of Inertia        
        parms["E"] = 2.068*10**11
        parms["nu"] = 0.3
        parms["rho"] = 7800        
  
        parms["a0"] = np.deg2rad(20) # Pressure angle
        parms["Z1"] = 16 # Number of pinion teeth
        parms["Z2"] = 48 # Number of gear teeth
        parms["L1"] = 0.016 #0.016#0.038 # Pinion width
        parms["L2"] = 0.016 #0.016#0.038 # Gear width
        parms["mod1"] = 0.003175 # Module of pinion
        parms["mod2"] = 0.003175 # Module of gear
        parms["Ra"] = 0.188*0.026 # Hube bore radius (No formula for this. You simply need to have it.)

        parms["kx1"] = 8.5364*10**7 # [N/m] - Bearing Translational Stiffness
        parms["ky1"] = 8.5364*10**7 # [N/m] - Bearing Translational Stiffness
        parms["kx2"] = 8.5364*10**7 # [N/m] - Bearing Translational Stiffness
        parms["ky2"] = 8.5364*10**7 # [N/m] - Bearing Translational Stiffness
        parms["ktheta1"] = 330 # [Nm/rad] - Torsional Stiffness of Shafts
        parms["ktheta2"] = 330 # [Nm/rad] - Torsional Stiffness of Shafts

        parms["cx1"] =  2.134*10**4 # [Ns/m] - Bearing Translational Damping
        parms["cy1"] =  2.134*10**4 # [Ns/m] - Bearing Translational Damping
        parms["cx2"] =  2.134*10**4 # [Ns/m] - Bearing Translational Damping
        parms["cy2"] =  2.134*10**4 # [Ns/m] - Bearing Translational Damping
        parms["ctheta1"] =  23.1 # [Nms/rad] - Torsional Damping of Shafts
        parms["ctheta2"] =  23.1 # [Nms/rad] - Torsional Damping of Shafts

        parms["Zeta"] = 0.07 # [-] - Damping Constant - Typically used as 0.07 in literature.
        #         parms["rayleigh_damping_m"] = 0.05 # [-] - Mass Proportionality Constant (Used to determine C from K and M)
        #         parms["rayleigh_damping_k"] = 10**-6 # [-] - Stiffness Proportionality Constant (Used to determine C from K and M)               

        return parms
    
    def set_parameter_dict(self,**parminput):
        """
        Supply the parameters with the appropriate names as keys or as a dictionary.
        """        
        # Set the parameters:
        self._set_parms_of_dict(parminput)
        # Set the gear geometry's parameters:
        self._set_gear_geometry_dictionary()                  
        #-------------
        # Calculate the mass moment of inertia using Eq. (7) and (8)
        # Chaari, F., Bartelmus, W., Zimroz, R., Fakhfakh, T. and Haddar, M., 2012. Gearbox vibration signal amplitude and frequency modulation. Shock and Vibration, 19(4), pp.635-652.
        rho = self.parms["rho"]
        if self.parms["Rp1"] is not None:
            # Pinion
            Dp1 = self.parms["Rp1"]*2.0
            thickness = self.parms["L1"]
            self.parms["I12"] = calculate_mass_and_inertia_of_disk(rho,self.Dp1,thickness)[1]
        if self.parms["Rp2"] is not None:
            # Gear
            Dp2 = self.parms["Rp2"]*2.0
            thickness = self.parms["L2"]            
            self.parms["I21"] = calculate_mass_and_inertia_of_disk(rho,Dp2,thickness)[1]            
        self._parms_set = True
        return self.parms
    
    def get_parameters_dict(self):        
        return self.parms
            
    def get_mass_matrix(self):
        m1 = self.parms.get("m1")
        m2 = self.parms.get("m2")
        mf = self.parms.get("mf")
        I11 = self.parms.get("I11")
        I12 = self.parms.get("I12")
        I21 = self.parms.get("I21")
        I22 = self.parms.get("I22")
        M = np.diag(np.array([m1, m1, I11, I12, m2, m2, I21,I22]))
        return M

    def get_stiffness_matrix_static(self):
        """
        The stiffness matrix is defined by K(t) = Ks + kgm * S, where this matrix is Ks.
        """        
        assert self._parms_set == True, "You first need to run set_parameters_dict."
        kx1 = self.parms.get("kx1")
        ky1 = self.parms.get("ky1")
        ktheta1 = self.parms.get("ktheta1")
        ktheta2 = self.parms.get("ktheta2")
        kx2 = self.parms.get("kx2")
        ky2 = self.parms.get("ky2")
        
        
        kdiag = np.diag([kx1, ky1, ktheta1, ktheta1, kx2, ky2, ktheta2, ktheta2])
        
        kdiag[2,3] = -ktheta1
        kdiag[3,2] = -ktheta1
        kdiag[7,6] = -ktheta2
        kdiag[6,7] = -ktheta2
        
        return kdiag

    def get_damping_matrix_static(self):
        """
        The damping matrix is defined by C(t) = Cs + cgm * S, where this matrix is Cs.
        """        
        assert self._parms_set == True, "You first need to run set_parameters_dict."
        cx1 = self.parms.get("cx1")
        cy1 = self.parms.get("cy1")
        ctheta1 = self.parms.get("ctheta1")
        ctheta2 = self.parms.get("ctheta2")
        cx2 = self.parms.get("cx2")
        cy2 = self.parms.get("cy2")
        
        cdiag = np.diag([cx1, cy1, ctheta1, ctheta1, cx2, cy2, ctheta2, ctheta2])
        
        cdiag[2,3] = -ctheta1
        cdiag[3,2] = -ctheta1
        cdiag[7,6] = -ctheta2
        cdiag[6,7] = -ctheta2
        
        return cdiag

    def get_stiffness_matrix_dynamic_coefficients(self):
        """
        The stiffness matrix is defined by K(t) = Ks + kgm * S, where this matrix is S.
        """
        assert self._parms_set == True, "You first need to run set_parameters_dict."        
        a0 = self.parms.get("a0")
        Rb1 = self.parms.get("Rb1")
        Rb2 = self.parms.get("Rb2")
        P = np.array([[np.sin(a0),np.cos(a0),0,Rb1,-np.sin(a0),-np.cos(a0),-Rb2,0]])
        S = np.dot(P.T,P)
        return S

    def get_torque_array(self,
                            torque_of_motor,
                            torque_of_load):
        """
        Supply the motor torque and the applied torque for the given time step.
        """
        torque_of_motor = float(torque_of_motor)
        torque_of_load  = float(torque_of_load)        
        return np.array([0,0,torque_of_motor,0,0,0,0,torque_of_load])

    def get_pinion_angle_rad(self,
                             displacement_response_vector):
        """
        Get the DOF that is associated with the pinion. This is used in the Newmark integration function.
        """
        assert len(displacement_response_vector.shape) == 1,"This should be an (DOF,) array."
        
        return displacement_response_vector[3]
    
    def get_motor_speedradps(self,
                        velocity_response_vector):
        """
        Get the DOF that is associated with the motor. This is used in the Newmark integration function.
        """
        assert len(velocity_response_vector.shape) == 1,"This should be an (DOF,) array."     
        
        return velocity_response_vector[2]
    
    def dof_info(self):
        str_print = (
        """This function provides information behind the physical meaning of the DOFs:
            0: X Vibration (Pinion Bearing)
            1: Y Vibration (Pinion Bearing)
            2: Motor Rotation
            3: Pinion Rotation
            4: X Vibration (Driven Bearing)
            5: Y vibration (Driven Bearing)
            6: Driven Gear Rotation
            7: Load Rotation
           """
        )
        print(str_print)
        return str_print

class GearboxModel_Luo_6DOF(GearboxModelUtils):
    '''
    Based on: Y. Luo et al./Mechanical Systems and Signal Processing 119 (2019) 155–181
    '''

    def __init__(self):
        self._parms_set = False
        self.parms = None
        
    def get_parameter_dict_default(self):
        """
        The purpose of this function is two-fold. 
        Firstly, create a dictionary with all the essential parameters (not the derived parameters)
        Secondly, we can check that the appropriate parameters are supplied to the class using the defined keys.
        """
        parms = {}
        parms["m1"] = 1.2728 # [kg] - Mass of Bearing Block 1
        parms["m2"] = 3.5367 # [kg] - Mass of Bearing Block 2
        parms["mp"] = None # [kg] - Mass of Pinion
        parms["mg"] = None # [kg] - Mass of Gear
        parms["I12"] = 0.0001751# [kgm^2] - Pinion Moment of Inertia
        parms["I21"] = 0.006828 # [kgm^2] - Gear Moment of Inertia
        parms["E"] = 2.068*10**11
        parms["nu"] = 0.3
        parms["rho"] = 7800        
  
        parms["a0"] = np.deg2rad(20) # Pressure angle
        parms["Z1"] = 16 # Number of pinion teeth
        parms["Z2"] = 48 # Number of gear teeth
        parms["L1"] = 0.016 #0.016#0.038 # Pinion width
        parms["L2"] = 0.016 #0.016#0.038 # Gear width
        parms["mod1"] = 0.003175 # Module of pinion
        parms["mod2"] = 0.003175 # Module of gear
        parms["Ra"] = 0.188*0.026 # Hube bore radius (No formula for this. You simply need to have it.)

        parms["kx1"] = 8.5364*10**7 # [N/m] - Bearing Translational Stiffness
        parms["ky1"] = 8.5364*10**7 # [N/m] - Bearing Translational Stiffness
        parms["kx2"] = 8.5364*10**7 # [N/m] - Bearing Translational Stiffness
        parms["ky2"] = 8.5364*10**7 # [N/m] - Bearing Translational Stiffness

        parms["cx1"] =  2.134*10**4 # [Ns/m] - Bearing Translational Damping
        parms["cy1"] =  2.134*10**4 # [Ns/m] - Bearing Translational Damping
        parms["cx2"] =  2.134*10**4 # [Ns/m] - Bearing Translational Damping
        parms["cy2"] =  2.134*10**4 # [Ns/m] - Bearing Translational Damping
    

        parms["Zeta"] = 0.07 # [-] - Damping Constant - Typically used as 0.07 in literature.
        #         parms["rayleigh_damping_m"] = 0.05 # [-] - Mass Proportionality Constant (Used to determine C from K and M)
        #         parms["rayleigh_damping_k"] = 10**-6 # [-] - Stiffness Proportionality Constant (Used to determine C from K and M)               

        return parms
    
    def set_parameter_dict(self,**parminput):
        """
        Supply the parameters with the appropriate names as keys or as a dictionary.
        """        
        # Set the parameters:
        self._set_parms_of_dict(parminput)
        # Set the gear geometry's parameters:
        self._set_gear_geometry_dictionary()                  
        #-------------
        # Calculate the mass moment of inertia using Eq. (7) and (8)
        # Chaari, F., Bartelmus, W., Zimroz, R., Fakhfakh, T. and Haddar, M., 2012. Gearbox vibration signal amplitude and frequency modulation. Shock and Vibration, 19(4), pp.635-652.
        rho = self.parms["rho"]
        if self.parms["Rp1"] is not None:
            # Pinion
            Dp1 = self.parms["Rp1"]*2.0
            thickness = self.parms["L1"]
            self.parms["I12"] = calculate_mass_and_inertia_of_disk(rho,self.Dp1,thickness)[1]
        if self.parms["Rp2"] is not None:
            # Gear
            Dp2 = self.parms["Rp2"]*2.0
            thickness = self.parms["L2"]            
            self.parms["I21"] = calculate_mass_and_inertia_of_disk(rho,Dp2,thickness)[1]            
        self._parms_set = True
        return self.parms
    
    def get_parameters_dict(self):        
        return self.parms
            
    def get_mass_matrix(self):
        m1 = self.parms.get("m1")
        m2 = self.parms.get("m2")
        mf = self.parms.get("mf")
        I12 = self.parms.get("I12")
        I21 = self.parms.get("I21")
        M = np.diag(np.array([m1, m1, I12, m2, m2, I21]))
        return M

    def get_stiffness_matrix_static(self):
        """
        The stiffness matrix is defined by K(t) = Ks + kgm * S, where this matrix is Ks.
        """        
        assert self._parms_set == True, "You first need to run set_parameters_dict."
        kx1 = self.parms.get("kx1")
        ky1 = self.parms.get("ky1")
        kx2 = self.parms.get("kx2")
        ky2 = self.parms.get("ky2")
        
        
        kdiag = np.diag([kx1, ky1, 0, kx2, ky2, 0])
        
        return kdiag

    def get_damping_matrix_static(self):
        """
        The damping matrix is defined by C(t) = Cs + cgm * S, where this matrix is Cs.
        """        
        assert self._parms_set == True, "You first need to run set_parameters_dict."
        cx1 = self.parms.get("cx1")
        cy1 = self.parms.get("cy1")
        cx2 = self.parms.get("cx2")
        cy2 = self.parms.get("cy2")
        
        
        cdiag = np.diag([cx1, cy1, 0, cx2, cy2, 0])
        
        return cdiag

    def get_stiffness_matrix_dynamic_coefficients(self):
        """
        The stiffness matrix is defined by K(t) = Ks + kgm * S, where this matrix is S.
        """
        assert self._parms_set == True, "You first need to run set_parameters_dict."        
        a0 = self.parms.get("a0")
        Rb1 = self.parms.get("Rb1")
        Rb2 = self.parms.get("Rb2")
        P = np.array([[np.sin(a0),np.cos(a0),Rb1,-np.sin(a0),-np.cos(a0),-Rb2]])
        S = np.dot(P.T,P)
        return S

    def get_torque_array(self,
                            torque_of_motor,
                            torque_of_load):
        """
        Supply the motor torque and the applied torque for the given time step.
        """
        torque_of_motor = float(torque_of_motor)
        torque_of_load  = float(torque_of_load)        
        return np.array([0,0,torque_of_motor,0,0,torque_of_load])

    def get_pinion_angle_rad(self,
                             displacement_response_vector):
        """
        Get the DOF that is associated with the pinion. This is used in the Newmark integration function.
        """
        assert len(displacement_response_vector.shape) == 1,"This should be an (DOF,) array."
        
        return displacement_response_vector[2]
    
    def get_motor_speedradps(self,
                        velocity_response_vector):
        """
        Get the DOF that is associated with the motor. This is used in the Newmark integration function.
        """
        assert len(velocity_response_vector.shape) == 1,"This should be an (DOF,) array."     
        
        return velocity_response_vector[2]
    
    def dof_info(self):
        str_print = (
        """This function provides information behind the physical meaning of the DOFs:
            0: X Vibration (Pinion Bearing)
            1: Y Vibration (Pinion Bearing)
            2: Pinion Rotation
            3: X Vibration (Driven Bearing)
            4: Y vibration (Driven Bearing)
            5: Driven Gear Rotation
           """
        )
        print(str_print)
        return str_print
class GearboxModel_Meng_8DOF(GearboxModelUtils):
    '''
    Based on: Z. Meng, G. Shi and F. Wang/ Mechanism and Machine Theory 148 (2020) 103786
    Motor and load inertial parameters taken from Chaari et al. 2012 (can confirm specifics later)
    Coupling stiffness/damping values obtained from Mohammed et al. 2015 (can confirm specifics later)
    '''


    def __init__(self):
        self._parms_set = False
        self.parms = None
        
    def get_parameter_dict_default(self):
        """
        The purpose of this function is two-fold. 
        Firstly, create a dictionary with all the essential parameters (not the derived parameters)
        Secondly, we can check that the appropriate parameters are supplied to the class using the defined keys.
        """
        parms = {}
        parms["m1"] = 0.3083 # [kg] - Mass of Bearing Block 1
        parms["m2"] = 0.4439 # [kg] - Mass of Bearing Block 2
        parms["mp"] = None # [kg] - Mass of Pinion
        parms["mg"] = None # [kg] - Mass of Gear
        parms["I11"] = 0.0043 # [kgm^2] - Motor Moment of Inertia
        parms["I12"] = 0.96*10**-4 # [kgm^2] - Pinion Moment of Inertia
        parms["I21"] = 2*10**-4 # [kgm^2] - Gear Moment of Inertia
        parms["I22"] = 0.0045# [kgm^2] - Machine Moment of Inertia        
        parms["E"] = 2.068*10**11
        parms["nu"] = 0.3
        parms["rho"] = 7800        
  
        parms["a0"] = np.deg2rad(20) # Pressure angle
        parms["Z1"] = 25 # Number of pinion teeth
        parms["Z2"] = 30 # Number of gear teeth
        parms["L1"] = 0.02 #0.016#0.038 # Pinion width
        parms["L2"] = 0.02 #0.016#0.038 # Gear width
        parms["mod1"] = 0.002 # Module of pinion
        parms["mod2"] = 0.002 # Module of gear
        parms["Ra"] = 0.188*0.026 # Hube bore radius (No formula for this. You simply need to have it.)

        parms["kx1"] = 6.56*10**8 # [N/m] - Bearing Translational Stiffness
        parms["ky1"] = 6.56*10**8 # [N/m] - Bearing Translational Stiffness
        parms["kx2"] = 6.56*10**8 # [N/m] - Bearing Translational Stiffness
        parms["ky2"] = 6.56*10**8 # [N/m] - Bearing Translational Stiffness
        parms["ktheta1"] = 1*10**4 # [Nm/rad] - Torsional Stiffness of Shafts
        parms["ktheta2"] = 1*10**4 # [Nm/rad] - Torsional Stiffness of Shafts

        parms["cx1"] =  1.8*10**3 # [Ns/m] - Bearing Translational Damping
        parms["cy1"] =  1.8*10**3 # [Ns/m] - Bearing Translational Damping
        parms["cx2"] =  1.8*10**3 # [Ns/m] - Bearing Translational Damping
        parms["cy2"] =  1.8*10**3 # [Ns/m] - Bearing Translational Damping
        parms["ctheta1"] =  10 # [Nms/rad] - Torsional Damping of Shafts
        parms["ctheta2"] =  10 # [Nms/rad] - Torsional Damping of Shafts

        parms["Zeta"] = 0.07 # [-] - Damping Constant - Typically used as 0.07 in literature.
        #         parms["rayleigh_damping_m"] = 0.05 # [-] - Mass Proportionality Constant (Used to determine C from K and M)
        #         parms["rayleigh_damping_k"] = 10**-6 # [-] - Stiffness Proportionality Constant (Used to determine C from K and M)               

        return parms
    
    def set_parameter_dict(self,**parminput):
        """
        Supply the parameters with the appropriate names as keys or as a dictionary.
        """        
        # Set the parameters:
        self._set_parms_of_dict(parminput)
        # Set the gear geometry's parameters:
        self._set_gear_geometry_dictionary()                  
        #-------------
        # Calculate the mass moment of inertia using Eq. (7) and (8)
        # Chaari, F., Bartelmus, W., Zimroz, R., Fakhfakh, T. and Haddar, M., 2012. Gearbox vibration signal amplitude and frequency modulation. Shock and Vibration, 19(4), pp.635-652.
        rho = self.parms["rho"]
        if self.parms["Rp1"] is not None:
            # Pinion
            Dp1 = self.parms["Rp1"]*2.0
            thickness = self.parms["L1"]
            self.parms["I12"] = calculate_mass_and_inertia_of_disk(rho,self.Dp1,thickness)[1]
        if self.parms["Rp2"] is not None:
            # Gear
            Dp2 = self.parms["Rp2"]*2.0
            thickness = self.parms["L2"]            
            self.parms["I21"] = calculate_mass_and_inertia_of_disk(rho,Dp2,thickness)[1]            
        self._parms_set = True
        return self.parms
    
    def get_parameters_dict(self):        
        return self.parms
            
    def get_mass_matrix(self):
        m1 = self.parms.get("m1")
        m2 = self.parms.get("m2")
        mf = self.parms.get("mf")
        I11 = self.parms.get("I11")
        I12 = self.parms.get("I12")
        I21 = self.parms.get("I21")
        I22 = self.parms.get("I22")
        M = np.diag(np.array([m1, m1, I11, I12, m2, m2, I21,I22]))
        return M

    def get_stiffness_matrix_static(self):
        """
        The stiffness matrix is defined by K(t) = Ks + kgm * S, where this matrix is Ks.
        """        
        assert self._parms_set == True, "You first need to run set_parameters_dict."
        kx1 = self.parms.get("kx1")
        ky1 = self.parms.get("ky1")
        ktheta1 = self.parms.get("ktheta1")
        ktheta2 = self.parms.get("ktheta2")
        kx2 = self.parms.get("kx2")
        ky2 = self.parms.get("ky2")
        
        
        kdiag = np.diag([kx1, ky1, ktheta1, ktheta1, kx2, ky2, ktheta2, ktheta2])
        
        kdiag[2,3] = -ktheta1
        kdiag[3,2] = -ktheta1
        kdiag[7,6] = -ktheta2
        kdiag[6,7] = -ktheta2
        
        return kdiag

    def get_damping_matrix_static(self):
        """
        The damping matrix is defined by C(t) = Cs + cgm * S, where this matrix is Cs.
        """        
        assert self._parms_set == True, "You first need to run set_parameters_dict."
        cx1 = self.parms.get("cx1")
        cy1 = self.parms.get("cy1")
        ctheta1 = self.parms.get("ctheta1")
        ctheta2 = self.parms.get("ctheta2")
        cx2 = self.parms.get("cx2")
        cy2 = self.parms.get("cy2")
        
        cdiag = np.diag([cx1, cy1, ctheta1, ctheta1, cx2, cy2, ctheta2, ctheta2])
        
        cdiag[2,3] = -ctheta1
        cdiag[3,2] = -ctheta1
        cdiag[7,6] = -ctheta2
        cdiag[6,7] = -ctheta2
        
        return cdiag

    def get_stiffness_matrix_dynamic_coefficients(self):
        """
        The stiffness matrix is defined by K(t) = Ks + kgm * S, where this matrix is S.
        """
        assert self._parms_set == True, "You first need to run set_parameters_dict."        
        a0 = self.parms.get("a0")
        Rb1 = self.parms.get("Rb1")
        Rb2 = self.parms.get("Rb2")
        P = np.array([[np.sin(a0),np.cos(a0),0,Rb1,-np.sin(a0),-np.cos(a0),-Rb2,0]])
        S = np.dot(P.T,P)
        return S

    def get_torque_array(self,
                            torque_of_motor,
                            torque_of_load):
        """
        Supply the motor torque and the applied torque for the given time step.
        """
        torque_of_motor = float(torque_of_motor)
        torque_of_load  = float(torque_of_load)        
        return np.array([0,0,torque_of_motor,0,0,0,0,torque_of_load])

    def get_pinion_angle_rad(self,
                             displacement_response_vector):
        """
        Get the DOF that is associated with the pinion. This is used in the Newmark integration function.
        """
        assert len(displacement_response_vector.shape) == 1,"This should be an (DOF,) array."
        
        return displacement_response_vector[3]
    
    def get_motor_speedradps(self,
                        velocity_response_vector):
        """
        Get the DOF that is associated with the motor. This is used in the Newmark integration function.
        """
        assert len(velocity_response_vector.shape) == 1,"This should be an (DOF,) array."     
        
        return velocity_response_vector[2]
    
    def dof_info(self):
        str_print = (
        """This function provides information behind the physical meaning of the DOFs:
            0: X Vibration (Pinion Bearing)
            1: Y Vibration (Pinion Bearing)
            2: Motor Rotation
            3: Pinion Rotation
            4: X Vibration (Driven Bearing)
            5: Y vibration (Driven Bearing)
            6: Driven Gear Rotation
            7: Load Rotation
           """
        )
        print(str_print)
        return str_print

class GearboxModel_Meng_6DOF(GearboxModelUtils):
    '''
    Based on: Z. Meng, G. Shi and F. Wang/ Mechanism and Machine Theory 148 (2020) 103786
    Coupling stiffness/damping values obtained from Mohammed et al. 2015 (can confirm specifics later)
    '''

    def __init__(self):
        self._parms_set = False
        self.parms = None
        
    def get_parameter_dict_default(self):
        """
        The purpose of this function is two-fold. 
        Firstly, create a dictionary with all the essential parameters (not the derived parameters)
        Secondly, we can check that the appropriate parameters are supplied to the class using the defined keys.
        """
        parms = {}
        parms["m1"] = 0.3083 # [kg] - Mass of Bearing Block 1
        parms["m2"] = 0.4439 # [kg] - Mass of Bearing Block 2
        parms["mp"] = None # [kg] - Mass of Pinion
        parms["mg"] = None # [kg] - Mass of Gear
        parms["I12"] = 0.96*10**-4# [kgm^2] - Pinion Moment of Inertia
        parms["I21"] = 2*10**-4 # [kgm^2] - Gear Moment of Inertia
        parms["E"] = 2.068*10**11
        parms["nu"] = 0.3
        parms["rho"] = 7800        
  
        parms["a0"] = np.deg2rad(20) # Pressure angle
        parms["Z1"] = 25 # Number of pinion teeth
        parms["Z2"] = 30 # Number of gear teeth
        parms["L1"] = 0.02 #0.016#0.038 # Pinion width
        parms["L2"] = 0.02 #0.016#0.038 # Gear width
        parms["mod1"] = 0.002 # Module of pinion
        parms["mod2"] = 0.002 # Module of gear
        parms["Ra"] = 0.188*0.026 # Hube bore radius (No formula for this. You simply need to have it.)

        parms["kx1"] = 6.56*10**8 # [N/m] - Bearing Translational Stiffness
        parms["ky1"] = 6.56*10**8 # [N/m] - Bearing Translational Stiffness
        parms["kx2"] = 6.56*10**8 # [N/m] - Bearing Translational Stiffness
        parms["ky2"] = 6.56*10**8 # [N/m] - Bearing Translational Stiffness

        parms["cx1"] =  1.8*10**3 # [Ns/m] - Bearing Translational Damping
        parms["cy1"] =  1.8*10**3 # [Ns/m] - Bearing Translational Damping
        parms["cx2"] =  1.8*10**3 # [Ns/m] - Bearing Translational Damping
        parms["cy2"] =  1.8*10**3 # [Ns/m] - Bearing Translational Damping
    

        parms["Zeta"] = 0.07 # [-] - Damping Constant - Typically used as 0.07 in literature.
        #         parms["rayleigh_damping_m"] = 0.05 # [-] - Mass Proportionality Constant (Used to determine C from K and M)
        #         parms["rayleigh_damping_k"] = 10**-6 # [-] - Stiffness Proportionality Constant (Used to determine C from K and M)                and M)                

        return parms
    
    def set_parameter_dict(self,**parminput):
        """
        Supply the parameters with the appropriate names as keys or as a dictionary.
        """        
        # Set the parameters:
        self._set_parms_of_dict(parminput)
        # Set the gear geometry's parameters:
        self._set_gear_geometry_dictionary()                  
        #-------------
        # Calculate the mass moment of inertia using Eq. (7) and (8)
        # Chaari, F., Bartelmus, W., Zimroz, R., Fakhfakh, T. and Haddar, M., 2012. Gearbox vibration signal amplitude and frequency modulation. Shock and Vibration, 19(4), pp.635-652.
        rho = self.parms["rho"]
        if self.parms["Rp1"] is not None:
            # Pinion
            Dp1 = self.parms["Rp1"]*2.0
            thickness = self.parms["L1"]
            self.parms["I12"] = calculate_mass_and_inertia_of_disk(rho,self.Dp1,thickness)[1]
        if self.parms["Rp2"] is not None:
            # Gear
            Dp2 = self.parms["Rp2"]*2.0
            thickness = self.parms["L2"]            
            self.parms["I21"] = calculate_mass_and_inertia_of_disk(rho,Dp2,thickness)[1]            
        self._parms_set = True
        return self.parms
    
    def get_parameters_dict(self):        
        return self.parms
            
    def get_mass_matrix(self):
        m1 = self.parms.get("m1")
        m2 = self.parms.get("m2")
        mf = self.parms.get("mf")
        I12 = self.parms.get("I12")
        I21 = self.parms.get("I21")
        M = np.diag(np.array([m1, m1, I12, m2, m2, I21]))
        return M

    def get_stiffness_matrix_static(self):
        """
        The stiffness matrix is defined by K(t) = Ks + kgm * S, where this matrix is Ks.
        """        
        assert self._parms_set == True, "You first need to run set_parameters_dict."
        kx1 = self.parms.get("kx1")
        ky1 = self.parms.get("ky1")
        kx2 = self.parms.get("kx2")
        ky2 = self.parms.get("ky2")
        
        
        kdiag = np.diag([kx1, ky1, 0, kx2, ky2, 0])
        
        return kdiag

    def get_damping_matrix_static(self):
        """
        The damping matrix is defined by C(t) = Cs + cgm * S, where this matrix is Cs.
        """        
        assert self._parms_set == True, "You first need to run set_parameters_dict."
        cx1 = self.parms.get("cx1")
        cy1 = self.parms.get("cy1")
        cx2 = self.parms.get("cx2")
        cy2 = self.parms.get("cy2")
        
        
        cdiag = np.diag([cx1, cy1, 0, cx2, cy2, 0])
        
        return cdiag

    def get_stiffness_matrix_dynamic_coefficients(self):
        """
        The stiffness matrix is defined by K(t) = Ks + kgm * S, where this matrix is S.
        """
        assert self._parms_set == True, "You first need to run set_parameters_dict."        
        a0 = self.parms.get("a0")
        Rb1 = self.parms.get("Rb1")
        Rb2 = self.parms.get("Rb2")
        P = np.array([[np.sin(a0),np.cos(a0),Rb1,-np.sin(a0),-np.cos(a0),-Rb2]])
        S = np.dot(P.T,P)
        return S

    def get_torque_array(self,
                            torque_of_motor,
                            torque_of_load):
        """
        Supply the motor torque and the applied torque for the given time step.
        """
        torque_of_motor = float(torque_of_motor)
        torque_of_load  = float(torque_of_load)        
        return np.array([0,0,torque_of_motor,0,0,torque_of_load])

    def get_pinion_angle_rad(self,
                             displacement_response_vector):
        """
        Get the DOF that is associated with the pinion. This is used in the Newmark integration function.
        """
        assert len(displacement_response_vector.shape) == 1,"This should be an (DOF,) array."
        
        return displacement_response_vector[2]
    
    def get_motor_speedradps(self,
                        velocity_response_vector):
        """
        Get the DOF that is associated with the motor. This is used in the Newmark integration function.
        """
        assert len(velocity_response_vector.shape) == 1,"This should be an (DOF,) array."     
        
        return velocity_response_vector[2]
    
    def dof_info(self):
        str_print = (
        """This function provides information behind the physical meaning of the DOFs:
            0: X Vibration (Pinion Bearing)
            1: Y Vibration (Pinion Bearing)
            2: Pinion Rotation
            3: X Vibration (Driven Bearing)
            4: Y vibration (Driven Bearing)
            5: Driven Gear Rotation
           """
        )
        print(str_print)
        return str_print

# =======================================================================================================
# Gear Stiffness Classes
# =======================================================================================================

class GearStiffnessUtils:

    def __init__(self):

        pass 

    """
    def check_stiffness(self,kgm):

        if np.any(kgm < 0):
            print(....)
            raise ValueError()
    """

    def set_verbose(self,verbose):
        assert isinstance(verbose,bool),"The verbose variable must be a boolean variable."
        self.verbose = verbose

    def _check_input_parameter_dictionary(self):
        if DEBUG == True:
            print("Checking the parameters in the gear mesh stiffness function...")
        assert isinstance(self.parms,dict), "The dict_parm needs to be a dictionary."
        assert self.parms["Ra"] != None, "The parameter <XXX> is not defined in the input dictionary"
        assert self.parms["Z1"] != None, "The parameter <XXX> is not defined in the input dictionary"
        assert self.parms["Z2"] != None, "The parameter <XXX> is not defined in the input dictionary"
        assert self.parms["a2"] != None, "The parameter <XXX> is not defined in the input dictionary"
        assert self.parms["Rb1"] != None, "The parameter <XXX> is not defined in the input dictionary"
        assert self.parms["Rr1"]   != None, "The parameter <XXX> is not defined in the input dictionary" 
        assert self.parms["a3"]   != None, "The parameter <XXX> is not defined in the input dictionary"  
        assert self.parms["a0"]   != None , "The parameter <XXX> is not defined in the input dictionary"          
        assert self.parms["Tlarge"]  != None , "The parameter <XXX> is not defined in the input dictionary"   
        assert self.parms["L1"] != None, "The parameter <XXX> is not defined in the input dictionary"
        assert self.parms["L2"] != None, "The parameter <XXX> is not defined in the input dictionary"
        assert self.parms["Rb1"] != None, "The parameter <XXX> is not defined in the input dictionary"
        assert self.parms["Rb2"] != None, "The parameter <XXX> is not defined in the input dictionary"
        assert self.parms["Rr1"] != None, "The parameter <XXX> is not defined in the input dictionary"
        assert self.parms["Rr2"] != None, "The parameter <XXX> is not defined in the input dictionary"
        assert self.parms["E"] != None, "The parameter <XXX> is not defined in the input dictionary"
        assert self.parms["nu"] != None, "The parameter <XXX> is not defined in the input dictionary"       

class GearStiffness_Healthy(GearStiffnessUtils):
    
    def __init__(self,
                    dict_of_gear_model_parameters,
                    mesh_fineness = 100,
                    verbose=False):
        """
        Parameters
        ----------------
            dict_of_gear_model_parameters (_type_): _description_
            mesh_fineness (int, optional): _description_. Defaults to 100.
        """

        self.parms = dict_of_gear_model_parameters
        self.mesh_fineness = mesh_fineness
        self.verbose = verbose
        self._check_input_parameter_dictionary()

    def get_gear_mesh_stiffness(self):
        pressure_angle_rad = self.parms["a0"]
        Ra = self.parms["Ra"]
        Z1 = self.parms["Z1"]
        Z2 = self.parms["Z2"]
        a2 = self.parms["a2"]
        Rb1 = self.parms["Rb1"]
        Rr1 = self.parms["Rr1"]        
        a3 = self.parms["a3"]    
        a0 = self.parms["a0"]             
        Tlarge = self.parms["Tlarge"]     
        L1 = self.parms["L1"]
        L2 = self.parms["L2"]
        Rb1 = self.parms["Rb1"]
        Rb2 = self.parms["Rb2"]
        Rr1 = self.parms["Rr1"]
        Rr2 = self.parms["Rr2"]
        E = self.parms["E"]
        nu = self.parms["nu"]        
        mesh_fineness = self.mesh_fineness
        
        thetaHealth, stiffHealth = lss.stiffnessCompiler(E,nu,Z1,Z2,L1,L2,Rb1,Rb2,Rr1,Rr2,Ra,pressure_angle_rad,1,mesh_fineness,0,[],verbose=self.verbose)

        ##### Helper quantities for faults. Not required. I just use them as a reference length for faults.######
        a1 = lss.A1(Tlarge,Z1,Z2,a0,1,0)
        a2 = lss.A2(Z1,a0)
        a3 = lss.A3(a2,Rb1,Rr1)
        # d1 = np.sqrt(Rb1**2 + Rr1**2 - 2*Rb1*Rr1*np.cos(a3-a2)) # Gives us a sort of reference to utilize if we'd like.
        self.xmax = Rb1*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr1*np.cos(a3)
        self.xmin = Rb1*((-a2+a2)*np.sin(-a2)+np.cos(a2)) - Rr1*np.cos(a3) # Below this we are in d1 region, where we have already integrated analytically assuming no faults!!

        return {
            "angle_rad": thetaHealth,
            "gear_mesh_stiffness": stiffHealth,
        }        
        
        
    def plot_gear_cross_section(self):
        """

        Returns
        ----------------
            _type_: _description_
        """
        return {} 
    
    def plot_gear_mesh_stiffness(self,fignum=None):
        """_summary_

        Parameters
        ----------------
            fignum (_type_, optional): _description_. Defaults to None.
        """
        dict_out = self.get_gear_mesh_stiffness()
        plt.figure(fignum)
        plt.plot(dict_out["angle_rad"],dict_out["gear_mesh_stiffness"])
    
    def get_damage_size(self):
        """_summary_

        Returns
        ----------------
            _type_: _description_
        """
        L1 = self.parms["L1"]
        FullArea = (self.xmax-self.xmin)*L1
        return  {'fault_area':0.0, 'full_area':FullArea, 'fault_area_ratio': 0.0/FullArea}

class GearStiffness_CrackSingle(GearStiffnessUtils):

    def __init__(self,
                     dict_of_gear_model_parameters,
                     fraction_crack_height_leftside,
                     fraction_crack_height_rightside,
                     fraction_crack_width,
                     fault_tooth_number = 5,
                     mesh_fineness = 100,
                     crack_angle_degrees=90,
                     verbose=False):
        """_summary_

        Parameters
        ----------------
            dict_of_gear_model_parameters (_type_): _description_
            fraction_crack_height_leftside (_type_): _description_
            fraction_crack_height_rightside (_type_): _description_
            fraction_crack_width (_type_): _description_
            fault_tooth_number (int, optional): _description_. Defaults to 5.
            mesh_fineness (int, optional): _description_. Defaults to 100.
            crack_angle_degrees (int, optional): _description_. Defaults to 90.
        """
        self.fraction_clh = fraction_crack_height_leftside
        self.fraction_crh = fraction_crack_height_rightside
        self.fraction_cw  = fraction_crack_width
        self.parms = dict_of_gear_model_parameters 
        self.fault_tooth_number = fault_tooth_number
        self.mesh_fineness = mesh_fineness
        self.crack_angle_degrees = crack_angle_degrees
        self._check_input_parameter_dictionary()
        self.verbose = verbose

        if self.fraction_cw>0 and self.fraction_crh>0: 
            raise ValueError('The right side crack height and crack width cannot simultaneously be defined. Choose one or the other.')

    def get_gear_mesh_stiffness(self):
        """_summary_
        """
        fault_tooth_number = self.fault_tooth_number
        pressure_angle_rad = self.parms["a0"]
        Ra = self.parms["Ra"]
        Z1 = self.parms["Z1"]
        Z2 = self.parms["Z2"]
        a2 = self.parms["a2"]
        Rb1 = self.parms["Rb1"]
        Rr1 = self.parms["Rr1"]        
        a3 = self.parms["a3"]    
        a0 = self.parms["a0"]          
        fraction_clh = self.fraction_clh# = 
        fraction_crh = self.fraction_crh# = fraction_crh
        fraction_cw  = self.fraction_cw#  = fraction_cw        
        crack_angle_degrees = self.crack_angle_degrees
        Tlarge = self.parms["Tlarge"]     
        L1 = self.parms["L1"]
        L2 = self.parms["L2"]
        Rb1 = self.parms["Rb1"]
        Rb2 = self.parms["Rb2"]
        Rr1 = self.parms["Rr1"]
        Rr2 = self.parms["Rr2"]        
        E = self.parms["E"]
        nu = self.parms["nu"]
        mesh_fineness = self.mesh_fineness
        
        ##### Helper quantities for faults. Not required. I just use them as a reference length for faults.######
        a1 = lss.A1(Tlarge,Z1,Z2,a0,1,0)
        a2 = lss.A2(Z1,a0)
        a3 = lss.A3(a2,Rb1,Rr1)
        d1 = np.sqrt(Rb1**2 + Rr1**2 - 2*Rb1*Rr1*np.cos(a3-a2)) # Gives us a sort of reference to utilize if we'd like.
        xmax = Rb1*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr1*np.cos(a3)
        xmin = Rb1*((-a2+a2)*np.sin(-a2)+np.cos(a2)) - Rr1*np.cos(a3) # Below this we are in d1 region, where we have already integrated analytically assuming no faults!!

        # Cracking Helper Parameter
        toothAngle = lss.A2(Z1,a0)
        #         toothThickness = 2*np.sin(toothAngle)*Rb1
        self.maximum_tooth_height = 2*np.sin(toothAngle)*Rb1
        
        
        self.crack_angle = np.deg2rad(crack_angle_degrees)    
        self.crack_left_height = fraction_clh * self.maximum_tooth_height # *** Essentially nothing changed, just the names to be more clear
        self.crack_right_height = fraction_crh * self.maximum_tooth_height # *** Essentially nothing changed, just the names to be more clear
        self.crack_width = L1 * fraction_cw # *** This is correct. The cw paremter operates along the "thickness" dimension
        
        crack_parameters = np.array([self.crack_left_height,self.crack_right_height,self.crack_angle,self.crack_width])
        pinion_rotation_angle, stiff_crack = lss.stiffnessCompiler(E,nu,Z1,Z2,L1,L2,Rb1,Rb2,Rr1,Rr2,Ra,pressure_angle_rad,fault_tooth_number,mesh_fineness,1,crack_parameters,verbose=self.verbose) 

        return {
            "angle_rad": pinion_rotation_angle,
            "gear_mesh_stiffness": stiff_crack,
        }

    def plot_gear_cross_section(self, fignum=None):
        """_summary_

        Parameters
        ----------------
            fignum (_type_, optional): _description_. Defaults to None.
        """
        crackAngle = np.deg2rad(self.crack_angle_degrees)
        pWc = self.crack_width
        pq0 = self.crack_left_height
        pq2 = self.crack_right_height
        
        L1 = self.parms['L1']
        pz = np.linspace(0,L1,1000)
        pq = np.zeros(len(pz))
        for i in range(len(pz)):
            if pq2 == 0:
                if pz[i] <= pWc:
                    q = pq0*np.sqrt((pWc-pz[i])/pWc)
                else:
                    q = 0
            else:
                q = np.sqrt((pq2**2-pq0**2)/(L1) * pz[i] + pq0**2)
            pq[i] = q * np.sin(crackAngle)
        
        plt.figure(fignum)
        plt.title('Crack Progression')
        plt.xlabel('Tooth Width (h dimension)')
        plt.ylabel('Tooth Thickness (L dimension)')
        plt.grid()
        plt.plot([0, L1, L1, 0, 0], [0,0,self.maximum_tooth_height, self.maximum_tooth_height, 0])
        plt.plot(pz,pq)
        
    
    def plot_gear_mesh_stiffness(self,fignum=None):
        dict_out = self.get_gear_mesh_stiffness()
        plt.figure(fignum)
        plt.plot(dict_out["angle_rad"],dict_out["gear_mesh_stiffness"])
    
    def get_damage_size(self):
        a2 = self.parms['a2']
        Rb1 =  self.parms['Rb1']
        L1 = self.parms['L1']
        v = np.deg2rad(self.crack_angle_degrees)

        hmax = Rb1*np.sin(a2)
        hb = hmax # Corresponds to my report

        q0 = self.crack_left_height
        q2 = self.crack_right_height
        Wc = self.crack_width        

        if q2 == 0: # Not full width crack   
            hq = hb - ((2*q0*Wc)/(3*L1))*np.sin(v)

        else:# q0 != q2:
            eta = 1e-9
            hq = hb - (2/3 * ((q2+eta)**3 -q0**3)/(((q2+eta)**2 -q0**2)))*np.sin(v)

        FullArea = L1*2*hmax
        CrackArea = (hq + hmax)*L1
        Ratio = 1 - CrackArea/FullArea
        return {'fault_area':CrackArea, 'full_area':FullArea, 'fault_area_ratio': CrackArea/FullArea}

class GearStiffness_ChipSingle(GearStiffnessUtils):

    def __init__(self,
                     dict_of_gear_model_parameters,
                     fraction_chip_width_along_L,
                     fraction_chip_depth_along_x,
                     fault_tooth_number = 5,
                     mesh_fineness = 100,
                     verbose=False):
        """_summary_

        Parameters
        ----------------
            dict_of_gear_model_parameters (_type_): _description_
            fraction_chip_width_along_L (_type_): _description_
            fraction_chip_depth_along_x (_type_): _description_
            fault_tooth_number (int, optional): _description_. Defaults to 5.
            mesh_fineness (int, optional): _description_. Defaults to 100.
        """
        self.fraction_chip_width_along_L = fraction_chip_width_along_L
        self.fraction_chip_depth_along_x = fraction_chip_depth_along_x
        self.parms = dict_of_gear_model_parameters 
        self.fault_tooth_number = fault_tooth_number
        self.mesh_fineness = mesh_fineness
        self._check_input_parameter_dictionary()
        self.verbose = verbose

    def get_gear_mesh_stiffness(self):
        fault_tooth_number = self.fault_tooth_number
        pressure_angle_rad = self.parms["a0"]
        Ra = self.parms["Ra"]
        Z1 = self.parms["Z1"]
        Z2 = self.parms["Z2"]
        a2 = self.parms["a2"]
        Rb1 = self.parms["Rb1"]
        Rr1 = self.parms["Rr1"]        
        a3 = self.parms["a3"]    
        a0 = self.parms["a0"]    
        fraction_chip_width = self.fraction_chip_width_along_L
        fraction_chip_depth = self.fraction_chip_depth_along_x
        Tlarge = self.parms["Tlarge"]     
        L1 = self.parms["L1"]
        L2 = self.parms["L2"]
        Rb1 = self.parms["Rb1"]
        Rb2 = self.parms["Rb2"]
        Rr1 = self.parms["Rr1"]
        Rr2 = self.parms["Rr2"]        
        E = self.parms["E"]
        nu = self.parms["nu"]
        mesh_fineness = self.mesh_fineness
        
        ##### Helper quantities for faults. Not required. I just use them as a reference length for faults.######
        a1 = lss.A1(Tlarge,Z1,Z2,a0,1,0)
        a2 = lss.A2(Z1,a0)
        a3 = lss.A3(a2,Rb1,Rr1)
        d1 = np.sqrt(Rb1**2 + Rr1**2 - 2*Rb1*Rr1*np.cos(a3-a2)) # Gives us a sort of reference to utilize if we'd like.
        self.xmax = Rb1*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr1*np.cos(a3)
        self.xmin = Rb1*((-a2+a2)*np.sin(-a2)+np.cos(a2)) - Rr1*np.cos(a3) # Below this we are in d1 region, where we have already integrated analytically assuming no faults!!

        self.chip_width = fraction_chip_width * L1 
        self.chip_depth = fraction_chip_depth * self.xmax 
        
        chip_parameters = np.array([self.chip_depth, self.chip_width])
        pinion_rotation_angle, stiff_chip = lss.stiffnessCompiler(E,nu,Z1,Z2,L1,L2,Rb1,Rb2,Rr1,Rr2,Ra,pressure_angle_rad,fault_tooth_number,mesh_fineness,2,chip_parameters,verbose=self.verbose) 

        return {
            "angle_rad": pinion_rotation_angle,
            "gear_mesh_stiffness": stiff_chip,
        }

    def plot_gear_cross_section(self, fignum=None):
        plt.figure(fignum)
        
        a1 = self.parms['a1']
        a2 = self.parms['a2']
        a3 = self.parms['a3']
        Rb1 = self.parms['Rb1']
        Rr1 = self.parms['Rr1']
        L1 = self.parms['L1']

        xarr = np.linspace(0.000001,self.xmax,1000)#Rb1*(-(-alpha_mem+a2)*np.sin(alpha_mem)+np.cos(alpha_mem) ) - Rr1*np.cos(a3)
    
        plt.plot([0, L1, L1, 0, 0], [0,0,self.xmax, self.xmax, 0])
        plotval = L1 - (self.xmax*self.chip_width/self.chip_depth - (self.xmax**2*self.chip_width/self.chip_depth - self.xmax*self.chip_width)/xarr)
        plotval = np.where(self.xmax - self.chip_depth <= xarr,plotval,L1)
        plt.plot(plotval, xarr)
        plt.title('Chip Progression')
        plt.xlabel('Tooth Width (z)')
        plt.ylabel('Length Along Tooth (x)')
        plt.grid()
    
    def plot_gear_mesh_stiffness(self,fignum=None):
        dict_out = self.get_gear_mesh_stiffness()
        plt.figure(fignum)
        plt.plot(dict_out["angle_rad"],dict_out["gear_mesh_stiffness"])
    
    def get_damage_size(self):
        L1 = self.parms['L1']
        
        ChipArea = -self.xmax*self.chip_width*(self.xmax/self.chip_depth-1)*(np.log(self.xmax)-np.log(self.xmax-self.chip_depth)) + self.xmax*self.chip_width
        FullArea = (self.xmax-self.xmin)*L1
        Ratio = ChipArea/FullArea
        return {'fault_area':ChipArea, 'full_area':FullArea, 'fault_area_ratio': ChipArea/FullArea}

class GearStiffness_SpallSingle(GearStiffnessUtils):

    def __init__(self,
                     dict_of_gear_model_parameters,
                     fraction_spall_start,
                     fraction_spall_length,
                     fraction_spall_width,
                     fraction_spall_depth,
                     fault_tooth_number = 5,
                     mesh_fineness = 100,
                     verbose=False):
        """_summary_

        Parameters
        ----------------
            dict_of_gear_model_parameters (_type_): _description_
            fraction_spall_start (_type_): _description_
            fraction_spall_length (_type_): _description_
            fraction_spall_width (_type_): _description_
            fraction_spall_depth (_type_): _description_
            fault_tooth_number (int, optional): _description_. Defaults to 5.
            mesh_fineness (int, optional): _description_. Defaults to 100.
        """
        self.fraction_spall_start = fraction_spall_start
        self.fraction_spall_length = fraction_spall_length
        self.fraction_spall_width = fraction_spall_width
        self.fraction_spall_depth  = fraction_spall_depth 
        self.parms = dict_of_gear_model_parameters 
        self.fault_tooth_number = fault_tooth_number
        self.mesh_fineness = mesh_fineness
        self._check_input_parameter_dictionary()
        self.verbose = verbose

        if self.fraction_spall_start< self.fraction_spall_length: 
            raise ValueError('The spall start fraction must be a larger value than the spall length value. \n The spall is defined from the furthest point up the tooth downward. \n Therefore, choose fraction_spall_start as the upper bound value, and fraction_spall_length as the distance BACK DOWN the tooth.')

    def get_gear_mesh_stiffness(self):
        fault_tooth_number = self.fault_tooth_number
        pressure_angle_rad = self.parms["a0"]
        Ra = self.parms["Ra"]
        Z1 = self.parms["Z1"]
        Z2 = self.parms["Z2"]
        a2 = self.parms["a2"]
        Rb1 = self.parms["Rb1"]
        Rr1 = self.parms["Rr1"]        
        a3 = self.parms["a3"]    
        a0 = self.parms["a0"]    
        fraction_spall_start = self.fraction_spall_start
        fraction_spall_length = self.fraction_spall_length
        fraction_spall_width = self.fraction_spall_width
        fraction_spall_depth = self.fraction_spall_depth 
        Tlarge = self.parms["Tlarge"]     
        L1 = self.parms["L1"]
        L2 = self.parms["L2"]
        Rb1 = self.parms["Rb1"]
        Rb2 = self.parms["Rb2"]
        Rr1 = self.parms["Rr1"]
        Rr2 = self.parms["Rr2"]        
        E = self.parms["E"]
        nu = self.parms["nu"]
        mesh_fineness = self.mesh_fineness
        
        ##### Helper quantities for faults. Not required. I just use them as a reference length for faults.######
        a1 = lss.A1(Tlarge,Z1,Z2,a0,1,0)
        a2 = lss.A2(Z1,a0)
        a3 = lss.A3(a2,Rb1,Rr1)
        d1 = np.sqrt(Rb1**2 + Rr1**2 - 2*Rb1*Rr1*np.cos(a3-a2)) # Gives us a sort of reference to utilize if we'd like.
        self.xmax = Rb1*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr1*np.cos(a3)
        self.xmin = Rb1*((-a2+a2)*np.sin(-a2)+np.cos(a2)) - Rr1*np.cos(a3) # Below this we are in d1 region, where we have already integrated analytically assuming no faults!!
        self.hmax = Rb1*np.sin(a2)

        self.spall_start = fraction_spall_start * self.xmax 
        self.spall_length = fraction_spall_length * self.xmax 
        self.spall_width = fraction_spall_width * L1
        self.spall_depth = fraction_spall_depth * self.hmax
        
        spall_parameters = np.array([self.spall_start, self.spall_length, self.spall_width, self.spall_depth])
        pinion_rotation_angle, stiff_spall = lss.stiffnessCompiler(E,nu,Z1,Z2,L1,L2,Rb1,Rb2,Rr1,Rr2,Ra,pressure_angle_rad,fault_tooth_number,mesh_fineness,5,spall_parameters,verbose=self.verbose) 

        return {
            "angle_rad": pinion_rotation_angle,
            "gear_mesh_stiffness": stiff_spall,
        }

    def plot_gear_cross_section(self, fignum=None):
        plt.figure(fignum)
        
        a1 = self.parms['a1']
        a2 = self.parms['a2']
        a3 = self.parms['a3']
        Rb1 = self.parms['Rb1']
        Rr1 = self.parms['Rr1']
        L1 = self.parms['L1']

        plt.title('Spall Progression')
        plt.xlabel('Tooth Width (z)')
        plt.ylabel('Length Along Tooth (x)')
        plt.grid()

        left_margin = L1/2 - self.spall_width/2
        right_margin = L1/2 + self.spall_width/2
        high_margin = self.spall_start
        low_margin = self.spall_start - self.spall_length
        plt.plot([0, L1, L1, 0, 0], [0,0,self.xmax, self.xmax, 0])
        
        plt.plot([left_margin,left_margin,right_margin,right_margin,left_margin],
                [low_margin,high_margin,high_margin,low_margin,low_margin])

    def plot_gear_mesh_stiffness(self,fignum=None):
        dict_out = self.get_gear_mesh_stiffness()
        plt.figure(fignum)
        plt.plot(dict_out["angle_rad"],dict_out["gear_mesh_stiffness"])
    
    def get_damage_size(self):
        L1 = self.parms['L1']
        SpallArea = self.spall_width * self.spall_length
        FullArea = (self.xmax-self.xmin)*L1
        Ratio = SpallArea/FullArea
        return {'fault_area':SpallArea, 'full_area':FullArea, 'fault_area_ratio': SpallArea/FullArea}

class GearStiffness_PitSingle(GearStiffnessUtils):

    """
    #The workflow of GearStiffness_PitSingle is as follows:

    #We need to initialise the class:

    gms_pit = module_name.GearStiffness_PitSingle(
                        dict_of_gear_model_parameters,
                        lower_pit_angle_rad,
                        upper_pit_angle_rad,
                        lower_pit_radius_frac, # relative to L1
                        upper_pit_radius_frac,
                        mean_pit_location_frac, # relative to xmax
                        std_pit_location_frac, # relative to xmax
                        num_pits,
                        fault_tooth_number = 5,
                        mesh_fineness = 100,
                        verbose=False)  

    # This calculates a dictionary with the statistical properties of the pits:
    #    - lower_pit_radius
    #    - upper_pit_radius
    #    - mean_pit_location
    #    - std_pit_location
    #    - num_pits
    #    - upper_pit_angle_rad
    #    - lower_pit_angle_rad
    dict_pit_parms = gms_pit.generate_pitting_radius_location_and_scale()

    # Generate geometry of the pits: If we call the get_gear_mesh_stiffness() function, we need to calculate the location, size, angle of each pit.
    # If we have not calculated the values, it will be calculated on the fly in get_gear_mesh_stiffness()
    # We can use set_geometry_of_each_pit to set the exact geometry (and location) of each pit.

    # If None is supplied, the values are NOT updated. They are NOT set to None.
    gms_pit.set_geometry_of_each_pit(pit_location_dist=None,
                            pit_angle_dist=None,
                            pit_radius_dist=None)

    # This function sets pit_location_dist=None, pit_angle_dist=None, and pit_radius_dist=None
    gms_pit.clear_geometry_of_each_pit()

    # Only the pit_location_dist is updated and set to "value":
    gms_pit.set_geometry_of_each_pit(pit_location_dist=value,
                            pit_angle_dist=None,
                            pit_radius_dist=None)

    # Here is an example where we randomly generate the pit_location_dist values externally and only update the pit_location_dist values:
    pit_location_dist = np.random.normal(loc = dict_pit_parms["mean_pit_location"], scale = dict_pit_parms["std_pit_location"], size = dict_pit_parms["num_pits"])
    gms_pit.set_geometry_of_each_pit(pit_location_dist=pit_location_dist,
                            pit_angle_dist=None,
                            pit_radius_dist=None)
    
    # Here we update ALL the values:
    num_pits = dict_pit_parms["num_pits"]

    pit_location_dist = np.random.normal(loc = dict_pit_parms["mean_pit_location"], scale = dict_pit_parms["std_pit_location"], size = num_pits)

    pit_angle_dist = np.random.uniform(low = dict_pit_parms["lower_pit_angle_rad"], high = dict_pit_parms["upper_pit_angle_rad"], size = num_pits)
            
    pit_radius_dist = np.random.uniform(low = dict_pit_parms["lower_pit_radius"], high = dict_pit_parms["upper_pit_radius"], size = num_pits)

    gms_pit.set_geometry_of_each_pit(pit_location_dist=pit_location_dist,
                            pit_angle_dist=pit_angle_dist,
                            pit_radius_dist=pit_radius_dist)    

    # We can update all the values using a dictionary with the keys pit_location_dist, pit_angle_dist, pit_radius_dist:
    gms_pit.set_geometry_of_each_pit(**dict_gms_to_update)    

    # If we want to generate standardised values, we can scale them within the function. Then we need to use the set_standarised_geometry_of_each_pit function:
    num_pits = dict_pit_parms["num_pits"] # Number of pits.

    # Standardised values:
    s_pit_location_dist = np.random.normal(loc = 0, scale = 1, size = num_pits)
    s_pit_angle_dist = np.random.uniform(low = -1, high = 1, size = num_pits)
    s_pit_radius_dist = np.random.uniform(low = -1, high = 1, size = num_pits)

    # Here we set the values to standardised values.
    gms_pit.set_standardised_geometry_of_each_pit(pit_location_dist_standardised=s_pit_location_dist,
                            pit_angle_dist_standardised=s_pit_angle_dist,
                            pit_radius_dist_standardised=s_pit_radius_dist)    


    Args:
        GearStiffnessUtils (_type_): _description_
    """


    def __init__(self,
                     dict_of_gear_model_parameters,
                     lower_pit_angle_rad,
                     upper_pit_angle_rad,
                     lower_pit_radius_frac, # relative to L1
                     upper_pit_radius_frac,
                     mean_pit_location_frac, # relative to xmax
                     std_pit_location_frac, # relative to xmax
                     num_pits,
                     fault_tooth_number = 5,
                     mesh_fineness = 100,
                     verbose=False):
        # *** I had to think about this one. There are multiple ways to approach it. Ideally, the user should be able to 
        # specify their own distributions and accompanying parameters, as fractions. This is tricky through. So for now
        # I am limiting the user to uniform distributions for fault shape parameters and a normal distribution for 
        # pit distributions. 
        # Therefore, the lower_ and upper_ parameters are describing the lower and upper params of the uniform distribution.
        # and the mean_ and std_ related to the normal pit location distribution.

        """_summary_

        Parameters
        ----------------
            dict_of_gear_model_parameters (_type_): _description_
            lower_pit_angle_rad (_type_): _description_
            upper_pit_angle_rad (_type_): _description_
            lower_pit_radius_frac (_type_): _description_
            mean_pit_location_frac (_type_): _description_
            fault_tooth_number (int, optional): _description_. Defaults to 5.
            mesh_fineness (int, optional): _description_. Defaults to 100.
        """
        self.lower_pit_angle_rad = lower_pit_angle_rad
        self.upper_pit_angle_rad = upper_pit_angle_rad
        self.lower_pit_radius_frac = lower_pit_radius_frac
        self.upper_pit_radius_frac = upper_pit_radius_frac
        self.mean_pit_location_frac = mean_pit_location_frac
        self.std_pit_location_frac = std_pit_location_frac
        self.num_pits = num_pits
        self.parms = dict_of_gear_model_parameters 
        self.fault_tooth_number = fault_tooth_number
        self.mesh_fineness = mesh_fineness
        self._check_input_parameter_dictionary()
        self.verbose = verbose
        self.pit_location_dist = None
        self.pit_angle_dist = None
        self.pit_radius_dist = None
        self.sampled_new_pit_location_angle_radius = [False,False,False]
    
    def generate_pitting_radius_location_and_scale(self):
        """This function returns the actual lower_pit_radius, upper_pit_radius, mean_pit_location, and std_pit_location. 
        This can be used to generate pits outside of the function.

        Returns:
            _type_: _description_
        """
        fault_tooth_number = self.fault_tooth_number
        pressure_angle_rad = self.parms["a0"]
        Ra = self.parms["Ra"]
        Z1 = self.parms["Z1"]
        Z2 = self.parms["Z2"]
        a2 = self.parms["a2"]
        Rb1 = self.parms["Rb1"]
        Rr1 = self.parms["Rr1"]        
        a3 = self.parms["a3"]    
        a0 = self.parms["a0"]     
        Tlarge = self.parms["Tlarge"]     
        L1 = self.parms["L1"]
        L2 = self.parms["L2"]
        Rb1 = self.parms["Rb1"]
        Rb2 = self.parms["Rb2"]
        Rr1 = self.parms["Rr1"]
        Rr2 = self.parms["Rr2"]        
        E = self.parms["E"]
        nu = self.parms["nu"]
        mesh_fineness = self.mesh_fineness
        
        ##### Helper quantities for faults. Not required. I just use them as a reference length for faults.######
        a1 = lss.A1(Tlarge,Z1,Z2,a0,1,0)
        a2 = lss.A2(Z1,a0)
        a3 = lss.A3(a2,Rb1,Rr1)
        d1 = np.sqrt(Rb1**2 + Rr1**2 - 2*Rb1*Rr1*np.cos(a3-a2)) # Gives us a sort of reference to utilize if we'd like.
        self.xmax = Rb1*((a1+a2)*np.sin(a1)+np.cos(a1)) - Rr1*np.cos(a3)
        self.xmin = Rb1*((-a2+a2)*np.sin(-a2)+np.cos(a2)) - Rr1*np.cos(a3) # Below this we are in d1 region, where we have already integrated analytically assuming no faults!!

        dictout = {}
        dictout["lower_pit_radius"]  = self.lower_pit_radius_frac * L1
        dictout["upper_pit_radius"]  = self.upper_pit_radius_frac * L1
        dictout["mean_pit_location"] = self.mean_pit_location_frac * self.xmax
        dictout["std_pit_location"]  = self.std_pit_location_frac * self.xmax
        dictout["num_pits"]    = self.num_pits
        dictout["upper_pit_angle_rad"] = self.upper_pit_angle_rad
        dictout["lower_pit_angle_rad"] = self.lower_pit_angle_rad        
        return dictout

    def set_geometry_of_each_pit(self,pit_location_dist = None,
                                pit_angle_dist = None,
                                pit_radius_dist = None,
                                **kwargs):
        """In this function, the pit_location_dist, the pit_angle_dist and the pit_radius_dist can be changed using external data.
        Args:
            pit_location_dist (_type_, optional): _description_. Defaults to None.
            pit_angle_dist (_type_, optional): _description_. Defaults to None.
            pit_radius_dist (_type_, optional): _description_. Defaults to None.
        """

        if pit_location_dist is not None:
            assert len(pit_location_dist) == self.num_pits,"The length of the pit_location_dist array should match the number of pits: {}".format(self.num_pits)
            self.pit_location_dist = pit_location_dist
        if pit_angle_dist is not None:
            assert len(pit_angle_dist) == self.num_pits,"The length of the pit_angle_dist array should match the number of pits: {}".format(self.num_pits)            
            self.pit_angle_dist = pit_angle_dist            
        if pit_radius_dist is not None:            
            assert len(pit_radius_dist) == self.num_pits,"The length of the pit_radius_dist array should match the number of pits: {}".format(self.num_pits)                        
            self.pit_radius_dist = pit_radius_dist

    def clear_geometry_of_each_pit(self):
        self.pit_angle_dist = None
        self.pit_location_dist = None
        self.pit_radius_dist = None                

    def _generate_and_add_pit_attributes(self):
        """
            This function calculates pit_location_dist, pit_angle_dist and pit_radius_dist if it was not set with the set_geometry_of_each_pit function.
        """ 
        # Generate the pitting radius, location and scale based on the input parameters.        
        dict_pit_parm = self.generate_pitting_radius_location_and_scale()
        self.lower_pit_radius  = dict_pit_parm["lower_pit_radius"]#self.lower_pit_radius_frac * L1
        self.upper_pit_radius  = dict_pit_parm["upper_pit_radius"]#self.upper_pit_radius_frac * L1
        self.mean_pit_location = dict_pit_parm["mean_pit_location"]#self.mean_pit_location_frac * self.xmax
        self.std_pit_location  = dict_pit_parm["std_pit_location"]#self.std_pit_location_frac * self.xmax
        # self.lower_pit_radius = self.lower_pit_radius_frac * L1
        # self.upper_pit_radius = self.upper_pit_radius_frac * L1
        # self.mean_pit_location = self.mean_pit_location_frac * self.xmax
        # self.std_pit_location = self.std_pit_location_frac * self.xmax

        if self.pit_location_dist is None:
            if self.verbose: print("New pit location values are sampled.")
            self.pit_location_dist = np.random.normal(loc = self.mean_pit_location, scale = self.std_pit_location, size = self.num_pits)
            self.sampled_new_pit_location_angle_radius[0] = True            
        else:
            self.sampled_new_pit_location_angle_radius[0] = False
        if self.pit_angle_dist is None:
            if self.verbose: print("New pit angle values are sampled.")            
            self.pit_angle_dist = np.random.uniform(low = self.lower_pit_angle_rad, high = self.upper_pit_angle_rad, size = self.num_pits)
            self.sampled_new_pit_location_angle_radius[1] = True     
        else:
            self.sampled_new_pit_location_angle_radius[1] = False
        if self.pit_radius_dist is None:            
            if self.verbose: print("New pit radius values are sampled.")                     
            self.pit_radius_dist = np.random.uniform(low = self.lower_pit_radius, high = self.upper_pit_radius, size = self.num_pits)
            self.sampled_new_pit_location_angle_radius[2] = True     
        else:
            self.sampled_new_pit_location_angle_radius[2] = False

    def set_standardised_geometry_of_each_pit(self,pit_location_dist_standardised = None,
                                pit_angle_dist_standardised = None,
                                pit_radius_dist_standardised = None,
                                ):

        """
            Note, if a value of None is received (e.g. pit_radius_dist_standardised is None), then the pit_radius_dist is randomly sampled when generating the gear mesh function.

            The values are calculated as follows using the standardised values.

            pit_location_dist
            --------------------
            scale =  self.std_pit_location 
            location = self.mean_pit_location
            pit_location_dist = pit_location_dist_standardised * scale + location 

            If pit_location_dist_standardised is a Gaussian with a mean of 0 and a standard deviation of 1, then pit_location_dist will have a mean of location and a standard deviation of scale.
            If pit_location_dist_standardised is a uniform distribution between -1 and 1, then pit_location_dist will be uniformly distributed between [location - scale, location + scale].

            pit_angle_dist
            --------------------            
            location = (self.upper_pit_angle_rad + self.lower_pit_angle_rad)/2.0
            range_of_angle = 0.5 * (self.upper_pit_angle_rad - self.lower_pit_angle_rad)
            pit_angle_dist = pit_angle_dist_standardised * range_of_angle + location

            If pit_angle_dist_standardised is a uniform distributed between -1 and 1, then pit_angle_dist will be uniform distributed between lower_pit_angle_rad and upper_pit_angle_rad
            If pit_angle_dist_standardised is a standardised Gaussian distribution (loc = 0, scale = 1), then pit_angle_dist has a mean of location and a standard deviation of range_of_angle.

            pit_radius_dist
            --------------------   
            location = (self.upper_pit_radius + self.lower_pit_radius)/2.0
            range_of_radius = 0.5 * (self.upper_pit_radius - self.lower_pit_radius)
            pit_radius_dist = pit_radius_dist_standardised * range_of_radius + location

            If pit_radius_dist_standardised is a uniform distributed between -1 and 1, then pit_radius_dist will be uniform distributed between lower_pit_radius and upper_pit_radius
            If pit_radius_dist_standardised is a standardised Gaussian distribution (loc = 0, scale = 1), then pit_radius_dist has a mean of location and a standard deviation of range_of_radius.
        """
        # Initialise the values to None, so that if the standardised values are not supplied, the set_geometry_of_each_pit function receives a None.
        pit_location_dist = None 
        pit_angle_dist = None 
        pit_radius_dist = None                 

        if pit_location_dist_standardised is not None:
            scale =  self.std_pit_location 
            location = self.mean_pit_location
            pit_location_dist = pit_location_dist_standardised * scale + location 
        if pit_angle_dist_standardised is not None:
            location = (self.upper_pit_angle_rad + self.lower_pit_angle_rad)/2.0
            range_of_angle = 0.5 * (self.upper_pit_angle_rad - self.lower_pit_angle_rad)
            pit_angle_dist = pit_angle_dist_standardised * range_of_angle + location
        if pit_radius_dist_standardised is not None:            
            location = (self.upper_pit_radius + self.lower_pit_radius)/2.0
            range_of_radius = 0.5 * (self.upper_pit_radius - self.lower_pit_radius)
            pit_radius_dist = pit_radius_dist_standardised * range_of_radius + location

        self.set_geometry_of_each_pit(pit_location_dist,
                                        pit_angle_dist,
                                        pit_radius_dist)

    def get_gear_mesh_stiffness(self):
        fault_tooth_number = self.fault_tooth_number
        pressure_angle_rad = self.parms["a0"]
        Ra = self.parms["Ra"]
        Z1 = self.parms["Z1"]
        Z2 = self.parms["Z2"]
        a2 = self.parms["a2"]
        Rb1 = self.parms["Rb1"]
        Rr1 = self.parms["Rr1"]        
        a3 = self.parms["a3"]    
        a0 = self.parms["a0"]     
        Tlarge = self.parms["Tlarge"]     
        L1 = self.parms["L1"]
        L2 = self.parms["L2"]
        Rb1 = self.parms["Rb1"]
        Rb2 = self.parms["Rb2"]
        Rr1 = self.parms["Rr1"]
        Rr2 = self.parms["Rr2"]        
        E = self.parms["E"]
        nu = self.parms["nu"]
        mesh_fineness = self.mesh_fineness

        self._generate_and_add_pit_attributes()

        pit_parameters = np.array([self.pit_location_dist, self.pit_angle_dist, self.pit_radius_dist])
        pinion_rotation_angle, stiff_pit = lss.stiffnessCompiler(E,nu,Z1,Z2,L1,L2,Rb1,Rb2,Rr1,Rr2,Ra,pressure_angle_rad,fault_tooth_number,mesh_fineness,3,pit_parameters,verbose=self.verbose) 
        # For the pit, I need to determine beforehand if it will work:
        if np.min(lss.generatePittingFaults(self.pit_location_dist, self.pit_angle_dist, self.pit_radius_dist, Z1, Z2, L1, Rb1, Rr1, 1000, a0)[1]) < 0: # 5 is a thumb-sucked value... 
            msg = """
                The combination of pitting parameters produced a non-physical result. 
                This happens when the effective line contact reduction is larger than the tooth width. 
                To fix this, consider 
                - lowering the number of pits being sampled, 
                - lowering the size of the pits, 
                - lowering the angle of the pits or finally, 
                - choose a larger pit standard deviation to spread them out more.
                  """
            raise ValueError(msg)

        return {
            "angle_rad": pinion_rotation_angle,
            "gear_mesh_stiffness": stiff_pit,
            "pit_location_dist": self.pit_location_dist,
            "pit_angle_dist": self.pit_angle_dist,            
            "pit_radius_dist": self.pit_radius_dist,                        
        }

    def plot_gear_cross_section(self, fignum=None, plot_marginal_location=False):
        plt.figure(fignum)
        plt.subplot(2,1,1)
        
        a0 = self.parms['a0']
        a1 = self.parms['a1']
        a2 = self.parms['a2']
        a3 = self.parms['a3']
        Rb1 = self.parms['Rb1']
        Rr1 = self.parms['Rr1']
        L1 = self.parms['L1']
        Z1 = self.parms['Z1']
        Z2 = self.parms['Z2']
        alpha_mem = np.linspace(-a1,a2,1000)
        xarr = Rb1*(-(-alpha_mem+a2)*np.sin(alpha_mem)+np.cos(alpha_mem) ) - Rr1*np.cos(a3)
        
        plt.ylim([0,L1])
        plt.xlim([self.xmin,self.xmax])
        plt.title('Pit Distribution (Worst Case)')
        plt.ylabel('Tooth Width (z)')
        plt.grid()
        
        # Luke, the "np.random.uniform(0,L1,self.num_pits)" makes it look like we have the pits at different locations each time we plot. I do not think this is happening and therefore can cause confusion.
        # Would it be possible to change to:
        # plt.scatter(self.pit_location_dist,np.linspace(0,L1,self.num_pits), s=10*self.pit_radius_dist/np.max(self.pit_radius_dist),c=5*self.pit_radius_dist/np.max(self.pit_radius_dist), alpha=0.9)        
        # np.random.seed(0)  -        
        plt.scatter(self.pit_location_dist,np.random.uniform(0,L1,self.num_pits), s=10*self.pit_radius_dist/np.max(self.pit_radius_dist),c=5*self.pit_radius_dist/np.max(self.pit_radius_dist), alpha=0.9)

        plt.subplot(2,1,2)
        plt.tight_layout(h_pad=1)
        plt.plot(xarr,lss.generatePittingFaults(self.pit_location_dist, self.pit_angle_dist, self.pit_radius_dist, Z1, Z2, L1, Rb1, Rr1, 1000, a0)[1])
        plt.ylabel('Effective Tooth Width (z)')
        plt.xlabel('Length Along Tooth (x)')
        plt.grid()
        plt.ylim([0,L1])
        plt.xlim([self.xmin,self.xmax])

        if plot_marginal_location:
            plt.figure()
            plt.scatter(np.linspace(0,1,self.num_pits),self.pit_location_dist, s=10*self.pit_radius_dist/np.max(self.pit_radius_dist),c=5*self.pit_radius_dist/np.max(self.pit_radius_dist), alpha=0.9)


    def plot_gear_mesh_stiffness(self,fignum=None):
        dict_out = self.get_gear_mesh_stiffness()
        plt.figure(fignum)
        plt.plot(dict_out["angle_rad"],dict_out["gear_mesh_stiffness"])
    
    def get_damage_size(self):
        L1 = self.parms['L1']
        PitArea = np.sum(np.pi*(self.pit_radius_dist*np.sin(self.pit_angle_dist))**2)
        FullArea = (self.xmax-self.xmin)*L1
        Ratio = PitArea/FullArea
        return {'fault_area':PitArea, 'full_area':FullArea, 'fault_area_ratio': PitArea/FullArea}

def gear_mesh_stiffness_extrapolate(thetarad,thetaMain,stiffMain,gearbox_parm,SWM=False):
    """_summary_

    Parameters
    ----------------
        thetarad (_type_): _description_
        thetaMain (_type_): _description_
        stiffMain (_type_): _description_
        gearbox_parm (_type_): _description_
        SWM (bool, optional): _description_. Defaults to False.

    Returns
    ----------------
        _type_: _description_
    """
    Td = gearbox_parm["Td"]
    Ts = gearbox_parm["Ts"]
    Z1 = gearbox_parm["Z1"]
    
    thetanorm = thetarad%(2*np.pi*1)
    
    kbar = np.mean(stiffMain) # I will use this as a reference line.
    maindoubstop = np.where(thetaMain <= Td)[0][-1] + 5 # 5 is just a safety margin. Quite arbitrary.
    mainsingstop = np.where(thetaMain <= Td+Ts)[0][-1] + 5  # 5 is just a safety margin. Quite arbitrary.
           
    ind1 = np.where(kbar < stiffMain)[0]
    ind1 = np.where(ind1 <= maindoubstop)[0][-1] 
    ind1 = ind1#np.arange(Z1)*ind3 + ind1
    x1 = thetaMain[ind1]
    
    ind2 = ind1+1
    x2 = thetaMain[ind2]
    xm1 = (x1+x2) / 2
        
    indprelim = np.where(stiffMain < kbar)[0]
    ind3 = indprelim[np.where(indprelim <= mainsingstop)[0][-1]]

    x3 = thetaMain[ind3]
    
    ind4 = ind3+1
    x4 = thetaMain[ind4]
    xm2 = (x3+x4) / 2
    
    dx = x4# Defines by how many x values we may jump for interpolation
    GTN = np.floor(thetarad/(2*np.pi/Z1))%Z1
    
    xind = thetanorm%(2*np.pi/Z1)
    if x1 <= xind and xind <= xm1:
        result = np.interp(x1+GTN*dx,thetaMain,stiffMain)
    
    elif xm1 < xind and xind <= x2:
        result = np.interp(x1+GTN*dx,thetaMain,stiffMain)
    
    elif x3 <= xind and xind <= xm2:
        result = np.interp(x3+GTN*dx,thetaMain,stiffMain)
    
    elif xm2 < xind and xind <= x4:
        result = np.interp(x3+GTN*dx,thetaMain,stiffMain)
    
    else:
        result = np.interp(xind+GTN*dx,thetaMain,stiffMain)
        
    if SWM:
        meanstiff = np.mean(stiffMain)
        if result > meanstiff:
            result = np.max(stiffMain)

        else:
            result = np.min(stiffMain)
    return result

# =======================================================================================================
# Torque model templates.
# =======================================================================================================

def torque_motor_chaari(Tb, gb, wr, ws, ca1, ca2):  
    """_summary_

    Parameters
    ----------------
        Tb (float): _description_
        gb (float): _description_
        wr (float): _description_
        ws (float): _description_
        ca1 (float): _description_
        ca2 (float): _description_

    Returns
    ----------------
        _type_: _description_
    """
    gn = 1 - wr/ws
    gn = np.where(gn==0,gn+0.00001,gn) # Replace zero gn values with a small value to avoid division errors.
    Tmgn = Tb/(1+(gb-gn)**2 * (ca1/gn - ca2*gn**2))
    return Tmgn

def torque_model_template(speed_rpm):
    Tf = 10 # [Nm] - Full Load
    Ts = 2.7*Tf #27 # [Nm] - Starting
    Tb = 3.2*Tf #32 # [Nm] - Breakdown

    Po = 1500 # [W] - Rated Power
    ws = 1500 # [rpm] - Synchronous Speed
    gb = 0.315 # [-] - Slip
    ca1 = 1.711 # [-] - Motor Constant
    ca2 = 1.316 # [-] - Motor Constant    
    
    return torque_motor_chaari(Tb, gb, speed_rpm, ws, ca1, ca2)

if __name__ == "__main__":

    def motor_torque_function(speed_rpm):
        Tf = 10 # [Nm] - Full Load
        Ts = 2.7*Tf #27 # [Nm] - Starting
        Tb = 3.2*Tf #32 # [Nm] - Breakdown

        Po = 1500 # [W] - Rated Power
        ws = 1500 # [rpm] - Synchronous Speed
        gb = 0.315 # [-] - Slip
        ca1 = 1.711 # [-] - Motor Constant
        ca2 = 1.316 # [-] - Motor Constant    
        
        return torque_motor_chaari(Tb, gb, speed_rpm, ws, ca1, ca2)

    # Initialise the gearbox class:
    gb_mod = GearboxModel_Chaari_8DOF()
    # Obtain the default parameters:
    parmdict = gb_mod.get_parameter_dict_default()

    # This is an example where you supply the wrong parameter as an input.
    try:
        gb_mod.set_parameter_dict(wrong_parameter_given_as_input=1)
    except Exception as e:
        print("***ERROR***")
        print(e)
        print("***ERROR***")    

    # Set the default values as inputs:
    gb_mod.set_parameter_dict(**parmdict)

    # This is the gear mesh stiffness of a cracked model:
    gms = GearStiffness_CrackSingle(gb_mod.parms,
                                   fraction_crack_height_leftside = 0.5,
                                   fraction_crack_height_rightside = 0.5,
                                   fraction_crack_width=0.0)

    gms.plot_gear_mesh_stiffness(1)

    # This is the gear mesh stiffness of a healthy model:
    gms_h = GearStiffness_Healthy(gb_mod.parms)

    gms_h.plot_gear_mesh_stiffness(1)

    # Initialise the gearbox class:
    gb_mod = GearboxModel_Luo_10DOF()
    # Obtain the default parameters:
    parmdict = gb_mod.get_parameter_dict_default()

    # Set the default values as inputs:
    gb_mod.set_parameter_dict(**parmdict)

    # This is the gear mesh stiffness of a cracked model:
    gms_crack = GearStiffness_CrackSingle(gb_mod.parms,
                                fraction_crack_height_leftside = 0.2,
                                fraction_crack_height_rightside = 0.6,
                                fraction_crack_width=0.0,
                                        fault_tooth_number = 2)


    gms_chip = GearStiffness_ChipSingle(gb_mod.parms,
                                fraction_chip_width_along_L = 0.8,
                                fraction_chip_depth_along_x = 0.8,
                                    fault_tooth_number = 4)

    gms_spall = GearStiffness_SpallSingle(gb_mod.parms,
                                        fraction_spall_start=0.8,
                                        fraction_spall_length=0.5,
                                        fraction_spall_width=0.9,
                                        fraction_spall_depth=0.3,
                                        fault_tooth_number = 6)


    gms_pit = GearStiffness_PitSingle(gb_mod.parms,
                                    lower_pit_angle_rad=np.pi/6,
                                    upper_pit_angle_rad=np.pi/5,
                                    lower_pit_radius_frac= 0.005,
                                    upper_pit_radius_frac= 0.015,
                                    mean_pit_location_frac= 0.5,
                                    std_pit_location_frac= 0.01,
                                    num_pits= 120,
                                        fault_tooth_number = 8)

    # *** IMPORTANT!!!!! We need to incorporate a check to ensure the start value is always larger then the length value, otherwise this does not work correctly.

    gms_crack.plot_gear_mesh_stiffness(1)
    gms_chip.plot_gear_mesh_stiffness(1)
    gms_spall.plot_gear_mesh_stiffness(1)
    gms_pit.plot_gear_mesh_stiffness(1)

    # This is the gear mesh stiffness of a healthy model:
    gms_h = GearStiffness_Healthy(gb_mod.parms)

    gms_h.plot_gear_mesh_stiffness(1)

    print('Crack damage properties',gms_crack.get_damage_size())
    gms_crack.plot_gear_cross_section(2)

    print('Chipe damage properties',gms_chip.get_damage_size())
    gms_chip.plot_gear_cross_section(3)

    print('Spall damage properties',gms_spall.get_damage_size())
    gms_spall.plot_gear_cross_section(4)

    print('Pit damage properties',gms_pit.get_damage_size())
    gms_pit.plot_gear_cross_section(5)

    gb_mod.dof_info()

    fs = 200000 # Hz
    endtime = 2.0 #s - Defines where simulation ends. Should maybe change this to 1.5s
    tnewmark = np.linspace(0,endtime,int(fs*endtime))
    tnewmark = tnewmark[:50000]
    loadCase2 = -20 * np.ones(len(tnewmark))

    # The integrate function is available due to class inheritance:
    output = gb_mod.integrate(
                        gms_h,
                        tnewmark, 
                        loadCase2,
                        motor_torque_function,
                        )

    # This is the response:
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(output["ud"][2,:], label = 'Motor speed [rad/s]')
    plt.plot(output["ud"][7,:], label = 'Load speed [rad/s]')

    Z1 = gb_mod.get_parameters_dict()['Z1']
    Z2 = gb_mod.get_parameters_dict()['Z2']
    plt.title('Gear ratio (Z1/Z2) = {}'.format(Z1/Z2))
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(output["ud"][7,:]/output["ud"][2,:] + 0.0000001, label = 'Load to Motor ratio')
    plt.plot(np.ones(len(output['ud'][2,:]))*(Z1/Z2), label = 'Gear ratio (Z1/Z2)')
    plt.legend()

    # This is the target acceleration for subsequent optimisations:
    TARGET_acc = output["udd"][2,:]

    def cost_function(x_parm):

    #     gb_mod = GearboxModel_Chaari_8DOF()
        gb_mod = GearboxModel_Luo_10DOF()
        parmdict = gb_mod.get_parameter_dict_default()
        # In this cost function the objective is to determine kx1 and kx2 using the correct model.    
        parmdict["kx1"] = x_parm[0]
        parmdict["ky1"] = x_parm[1]    
        
        gb_mod.set_parameter_dict(**parmdict)
        gms_h2 = GearStiffness_Healthy(gb_mod.parms)
        
        output = gb_mod.integrate(
                            gms_h2,
                            tnewmark, 
                            loadCase2,
                            motor_torque_function,
                            )   
        
        acc = output["udd"][2,:]
        return np.mean((acc - TARGET_acc)**2.0)
    
    x_parm = []
    x_parm.append([1E8,1E8])
    x_parm.append([2E8,1E8])
    x_parm.append([1E8,2E8])
    # x_parm.append([1E5,2E5])
    x_parm.append([8.5364*10**7, 8.5364*10**7])

    for x in x_parm:
        print("*"*20)
        print(x)
        print("ERROR. = {}".format(cost_function(x)))

    def cost_function_CRACKED(x_parm):

        gb_mod = GearboxModel_Chaari_8DOF()
        parmdict = gb_mod.get_parameter_dict_default()
        
        # In this cost function the objective is to determine kx1 and kx2 using a cracked model. 
        parmdict["kx1"] = x_parm[0]
        parmdict["ky1"] = x_parm[1]    
        
        gb_mod.set_parameter_dict(**parmdict)

        gms = GearStiffness_CrackSingle(gb_mod.parms,
                                    fraction_crack_height_leftside = 0.5,
                                    fraction_crack_height_rightside = 0.5,
                                    fraction_crack_width=0.0)    
        
        
        output = gb_mod.integrate(
                            gms,
                            tnewmark, 
                            loadCase2,
                            motor_torque_function,
                            )   
        
        acc = output["udd"][2,:]
        return np.mean((acc - TARGET_acc)**2.0)

    for x in x_parm:
        print("*"*20)
        print(x)
        print("ERROR. = {}".format(cost_function_CRACKED(x)))

    def cost_function_DETERMINE_E(x_parm):

        gb_mod = GearboxModel_Chaari_8DOF()
        parmdict = gb_mod.get_parameter_dict_default()
        
        # In this objective function we want to estimate "E":
        parmdict["E"] = x_parm[0]
        
        gb_mod.set_parameter_dict(**parmdict)
        gms_h2 = GearStiffness_Healthy(gb_mod.parms)
        
        output = gb_mod.integrate(
                            gms_h2,
                            tnewmark, 
                            loadCase2,
                            motor_torque_function,
                            )   
        
        acc = output["udd"][2,:]
        return np.mean((acc - TARGET_acc)**2.0)

    list_output = []
    for Ev in np.linspace(150,250,10)*1E9:
        
        c = cost_function_DETERMINE_E([Ev])
        
        list_output.append([Ev,c])
        
    list_output = np.array(list_output)

    plt.figure(1)
    plt.plot(list_output[:,0]/1E9,list_output[:,1],'bo-')
    plt.xlabel("E [GPa]")
    plt.ylabel("Cost function")
