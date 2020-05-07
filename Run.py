#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:05:24 2020

@author: fassaad
"""

import os
import subprocess
import shutil
import sys
import json
from colorama import Fore, Back, Style


def Set_Default_Variables():
    """Defines a dictionary containing all parameters with default values."""
    
    Params = {}
    Params_model = {}
    
    Params["VAR_Lattice"] = {
        # Parameters that define the Bravais lattice
        "L1": 6,
        "L2": 6,
        "Lattice_type": "Square",
        "Model":"Hubbard"
        }
    
    Params["VAR_Model_Generic"] = {
        # General parameters concerning any model
        "Checkerboard": True,
        "Symm"        : True,
        "N_SUN"       : 2,
        "N_FL"        : 1,
        "Phi_X"       : 0.0,
        "Phi_Y"       : 0.0,
        "Bulk"        : True,
        "N_Phi"       : 0,
        "Dtau"        : 0.1,
        "Beta"        : 5.0,
        "Projector"   : False,
        "Theta"       : 10.0,
        }
    
    Params["VAR_QMC"] = {
        # General parameters for the Monte Carlo algorithm
        "Nwrap"              : 10 ,
        "NSweep"             : 20 ,
        "NBin"               : 5  ,
        "Ltau"               : 1  ,
        "LOBS_ST"            : 0  ,
        "LOBS_EN"            : 0  ,
        "CPU_MAX"            : 0.0,
        "Propose_S0"         : False,
        "Global_moves"       : False,
        "N_Global"           : 1  ,
        "Global_tau_moves"   : False,
        "N_Global_tau"       : 1  ,
        "Nt_sequential_start": 0  ,
        "Nt_sequential_end"  : -1 ,
        }
    
    Params["VAR_errors"] = {
        # Post-processing parameters
        "n_skip" : 1,
        "N_rebin": 1,
        "N_Cov"  : 0,
        }
    
    Params["VAR_TEMP"] = {
        # Parallel tempering parameters
        "N_exchange_steps"      : 6   ,
        "N_Tempering_frequency" : 10  ,
        "mpi_per_parameter_set" : 2   ,
        "Tempering_calc_det"    : True,
        }
    
    Params["VAR_Max_Stoch"] = {
        # MaxEnt parameters
        "NGamma"     : 400  ,
        "Om_st"      : -10.0,
        "Om_en"      :  10.0,
        "Ndis"       : 2000 ,
        "NBins"      : 250  ,
        "NSweeps"    : 70   ,
        "NWarm"      : 20   ,
        "N_alpha"    : 14   ,
        "alpha_st"   : 1.0  ,
        "R"          : 1.2  ,
        "Channel"    : "P"  ,
        "Checkpoint" : False,
        "Tolerance"  : 0.1  ,
        }
    
    Params_model["VAR_Hubbard"] = {
        # Parameters of the Hubbard hamiltonian
        "HS"       :  "Mz" ,
        "ham_T"    :  1.0  ,
        "ham_chem" :  0.0  ,
        "ham_U"    :  4.0  ,
        "ham_T2"   :  1.0  ,
        "ham_U2"   :  4.0  ,
        "ham_Tperp":  1.0  ,
        }
        
    #json.dumps(Params)
    
    return Params, Params_model

def convert_par_to_str(parameter):
    """Converts a given parameter value to a string that can be written into a parameter file"""
    if type(parameter) == type(1) or type(parameter) == type(1.):
        return str(parameter)
    elif type(parameter) == type(''):
        return '"' + parameter + '"'
    elif type(parameter) == type(True):
        if parameter == True:
            return '.T.'
        else:
            return '.F.'
    
    raise Exception('Error in "convert_par_to_str": unrecognized type')

def Print_parameters(Params, file):
    print ("Setting up parameter file for", file )
    with open(file, 'w') as f:
        for namespace in Params:
            f.write( "&{}\n".format(namespace) )
            for var in Params[namespace]:
                f.write(var + ' = ' + convert_par_to_str(Params[namespace][var]) + '\n')
            f.write("/\n\n")

def Directory_name(Sim):
    Dir=''
    for name in Sim:
        if name in ["L1", "L2","Lattice_type","Model",
                    "Checkerboard","Symm","N_SUN","N_FL", "Phi_X","N_Phi",
                    "Dtau","Beta","Projector",
                    "Theta", "ham_T","ham_chem","ham_U",
                    "ham_T2", "ham_U2", "ham_Tperp"]:
            if name in ["Lattice_type","Model"]:
                Dir='{}{}_'.format(Dir, Sim[name])
            else:
                Dir='{}{}={}_'.format(Dir, name.strip("ham_"), Sim[name])
    return Dir[:-1]
        
def update_var(Params, var, value):
    """Tries to update value of parameter called var in Params"""
    for name in Params:
        for var2 in Params[name]:
            if var2 == var:
                Params[name][var2] = value
                return Params
    raise Exception ('"{}" does not correspond to a parameter'.format(var) )

def Set_param(Sim):
    model = Sim['Model']
    Params, Params_model = Set_Default_Variables()
    Params['VAR_'+model] = Params_model['VAR_'+model]
    
    for var in Sim:
        Params = update_var(Params, var, Sim[var])
    return Params

def Run(rundir,Alfdir,Runbranch,Config,Executable,Params):
    if os.path.exists(rundir):
        os.chdir(rundir)
        os.system("rm parameters")
        os.system("bash out_to_in.sh")
    else:
        os.mkdir(rundir)
        os.chdir(rundir)
        os.system("cp ../Start/* .")
    Print_parameters(Params, "parameters")
    os.chdir(Alfdir)
    os.system("make clean")
    command=str("git checkout " + Runbranch ) 
    print (Fore.RED+command)
    print(Style.RESET_ALL)
    os.system(command)
    command=str(". ./configureHPC.sh " + Config + "; make " + Executable )
    print (Fore.RED+command)
    print(Style.RESET_ALL)
    os.system(command)
    os.chdir(rundir)
    command=str(Alfdir+"/Prog/" + str(Executable).strip() + ".out" )
    print (Fore.RED+command)
    print(Style.RESET_ALL)
    os.system(command)
    os.chdir(Alfdir)
    command=str(". ./configureHPC.sh " + Config + "; make ana " )
    print (Fore.RED+command)
    print(Style.RESET_ALL)
    os.system(command)
    os.chdir(rundir)
    command=str( "cd " + Alfdir+ "; . ./configureHPC.sh " + Config + "; cd "
                    + rundir + "; bash analysis.sh" )
    print (Fore.RED+command)
    os.system(command)


if __name__ == "__main__":
    print ('Number of arguments:', len(sys.argv), 'arguments.' )
    print (sys.argv)
    if len(sys.argv) == 1:
        print(Fore.RED+"Usage: python3 Run.py Type=  Alfdir=  Runbranch=  Program= Testbranch= Config=")
        print(Style.RESET_ALL)
        print("Type        : R+T, R, T        with R=Run,  T=Test. Default=R+A")
        print("Alfdir      : Path to ALF directory.                Mandatory")
        print("branch_R    : Will run git checkout \"Runbranch\".    Default=master")
        print("Executable_R: Name of ref executable                Default=\"Model\".out")
        print("Config      : Will run ./configureHPC \"Config\"      Default=Intel")
        print("branch_T    : Branch to test against Runbranch      Default=master")
        print("Executable_T: Name of test executable               Default=\"Model\".out")

    # Default
    Type         = "R"
    Alfdir       = "none"
    branch_R     = "master"
    branch_T     = "master"
    Config       = "Intel"
    Executable_R = "none"
    Executable_T = "none"
    print(sys.argv)
    for arg in sys.argv:
        if arg.split('=')[0].lower() == "type" :
            Type=str(arg.split('=')[1])
        if arg.split('=')[0].lower() == "alfdir" :
            Alfdir=os.path.expanduser(arg.split('=')[1])
        if arg.split('=')[0].lower() == "branch_r" :
            branch_R=str(arg.split('=')[1])
        if arg.split('=')[0].lower() == "config" :
            Config=str(arg.split('=')[1])
        if arg.split('=')[0].lower() == "branch_t" :
            branch_T=str(arg.split('=')[1])
        if arg.split('=')[0].lower() == "executable_r" :
            Executable_R=str(arg.split('=')[1])
        if arg.split('=')[0].lower() == "executable_t" :
            Executable_T=str(arg.split('=')[1])
    if  Alfdir == "none" :
        print("Alfdir is mandatory")
        exit()
    
    with open("Sims") as f:
        Simulations = f.read().splitlines()
    print ( "Number of simulations ", len(Simulations))
    for Sim in Simulations:
        if Sim.strip() ==  "stop":
            print("Done")
            exit()
        Sim = json.loads(Sim)
        model = Sim['Model']
        print("Model is", model) 
        Params = Set_param(Sim)
        Dir = Directory_name(Sim)
        print(Dir)
        cwd = os.getcwd()
        rundir = str(cwd+"/"+Dir)
        print ("rundir is ", rundir)
        print(Type, Type.split("+"),Type.split("+")[0] )
        if "R" in str(Type.split("+")).strip():
            if Executable_R == "none":
                Executable1 = model
            else:
                Executable1 = Executable_R
            Run(rundir,Alfdir,branch_R,Config,Executable1,Params)
            f=open("Kin_scalJ")
            Kin_R=f.read().splitlines()[2]
            f.close               
        os.chdir(cwd)
        if "T" in str(Type.split("+")).strip():
            rundirT=str(rundir+"_Test")
            if Executable_T == "none":
                Executable1 = model
            else:
                Executable1 = Executable_T
            Run(rundirT,Alfdir,branch_T,Config,Executable1,Params)
            f=open("Kin_scalJ")
            Kin_T=f.read().splitlines()[2]
            f.close 
        os.chdir(cwd)
        with open(str(rundir)+".txt","w") as f:
            f.write(Kin_T)
            f.write(Kin_R)
        
        
# You need to write a general  run routine 
#  in ALFdir, Branch,  Config, rundir,  Executable
# You need to write ageneral analysis routine
#  in ALFdir, Branch,  Config, rundir


#    directory = '/Users/fassaad'
#    os.chdir(directory)
#    cwd = os.getcwd()
#    print ("Current working directory is:", cwd)
#    for filename in os.listdir(directory):
#        print(filename)
#        if filename.endswith(".txt")
#            f = open(filename)
#            lines = f.readlines()
#            print (filename)
#            print (lines[0])
#            continue
#        else:
#            continue


#directory = '/Users/fassaad'
#os.chdir(directory)
#cwd = os.getcwd()
#s = open("TEST.txt").read()
#replace = '20'
#s = s.replace('L2_R', replace)
#f = open("TEST.txt", 'w')
#f.write(s)
#f.close()
#s = open("TEST.txt").read()
#replace = '10'
#s = s.replace('L1_R', replace)
#f = open("TEST.txt", 'w')
#f.write(s)
#f.close()
#
#
#
#
#import subprocess
#directory = '/Users/fassaad'
#os.chdir(directory)
#hi = 'hithere'
#Command = ' export Hi="hi"  ; echo $Hi '
#Tmp = subprocess.check_output( str(Command) , shell=True)
#print(Tmp)
