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
import argparse
from shutil import copyfile
from colorama import Fore, Back, Style


def Set_Default_Variables():
    """Defines a dictionary containing all parameters with default values."""
    
    params = {}
    params_model = {}
    
    params["VAR_Lattice"] = {
        # Parameters that define the Bravais lattice
        "L1": 6,
        "L2": 6,
        "Lattice_type": "Square",
        "Model":"Hubbard"
        }
    
    params["VAR_Model_Generic"] = {
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
    
    params["VAR_QMC"] = {
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
    
    params["VAR_errors"] = {
        # Post-processing parameters
        "n_skip" : 1,
        "N_rebin": 1,
        "N_Cov"  : 0,
        }
    
    params["VAR_TEMP"] = {
        # Parallel tempering parameters
        "N_exchange_steps"      : 6   ,
        "N_Tempering_frequency" : 10  ,
        "mpi_per_parameter_set" : 2   ,
        "Tempering_calc_det"    : True,
        }
    
    params["VAR_Max_Stoch"] = {
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
    
    params_model["VAR_Hubbard"] = {
        # Parameters of the Hubbard hamiltonian
        "HS"       :  "Mz" ,
        "ham_T"    :  1.0  ,
        "ham_chem" :  0.0  ,
        "ham_U"    :  4.0  ,
        "ham_T2"   :  1.0  ,
        "ham_U2"   :  4.0  ,
        "ham_Tperp":  1.0  ,
        }
        
    #json.dumps(params)
    
    return params, params_model

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

def write_parameters(params, file):
    print ("Setting up parameter file for", file )
    with open(file, 'w') as f:
        for namespace in params:
            f.write( "&{}\n".format(namespace) )
            for var in params[namespace]:
                f.write(var + ' = ' + convert_par_to_str(params[namespace][var]) + '\n')
            f.write("/\n\n")

def Directory_name(sim):
    Dir=''
    for name in sim:
        if name in ["L1", "L2","Lattice_type","Model",
                    "Checkerboard","Symm","N_SUN","N_FL", "Phi_X","N_Phi",
                    "Dtau","Beta","Projector",
                    "Theta", "ham_T","ham_chem","ham_U",
                    "ham_T2", "ham_U2", "ham_Tperp"]:
            if name in ["Lattice_type","Model"]:
                Dir='{}{}_'.format(Dir, sim[name])
            else:
                Dir='{}{}={}_'.format(Dir, name.strip("ham_"), sim[name])
    return Dir[:-1]
        
def update_var(params, var, value):
    """Tries to update value of parameter called var in params"""
    for name in params:
        for var2 in params[name]:
            if var2 == var:
                params[name][var2] = value
                return params
    raise Exception ('"{}" does not correspond to a parameter'.format(var) )

def Set_param(sim):
    model = sim['Model']
    params, params_model = Set_Default_Variables()
    params['VAR_'+model] = params_model['VAR_'+model]
    
    for var in sim:
        params = update_var(params, var, sim[var])
    return params

def Compile(alfdir, branch, Config, executable):
    os.chdir(alfdir)
    os.system("make clean")
    command=str("git checkout " + branch ) 
    print (Fore.RED+command)
    print(Style.RESET_ALL)
    os.system(command)
    command=str(". ./configureHPC.sh " + Config + "; make " + executable )
    print (Fore.RED+command)
    print(Style.RESET_ALL)
    os.system(command)
    command=str(". ./configureHPC.sh " + Config + "; make ana " )
    print (Fore.RED+command)
    print(Style.RESET_ALL)
    os.system(command)

def Run(rundir, alfdir, executable, params):
    #Preparing run directory
    if not os.path.exists(rundir):
        os.mkdir(rundir)
    os.chdir(rundir)
    out_to_in()
    copyfile('../seeds', 'seeds')
    write_parameters(params, "parameters")
    
    #Running Monte Carlo
    command=str(alfdir+"/Prog/" + str(executable).strip() + ".out" )
    print (Fore.RED+command)
    print(Style.RESET_ALL)
    os.system(command)
    analysis(alfdir)
    print (Fore.RED+command)
    os.system(command)

    
def out_to_in(verbose=False):
    """Renames all the output configurations confout_* to confin_* 
    to continue the Monte Carlo simulation where the previous stopped"""
    for name in os.listdir():
        if name[:8] == 'confout_':
            name2 = 'confin_' + name[8:]
            if verbose:
                print( 'mv {} {}'.format(name, name2) )
            os.replace(name, name2)


def analysis(alfdir):
    """Performs the default analysis on all files ending in 
    _scal, _eq or _tau in current working directory"""
    if os.path.exists('Var_scal'):
        os.remove('Var_scal')
    for name in os.listdir():
        if name[-5:] == '_scal':
            print( 'Analysing {}'.format(name) )
            os.symlink(name, 'Var_scal')
            command = alfdir + '/Analysis/cov_scal.out'
            os.system(command)
            os.remove('Var_scal')
            os.replace('Var_scalJ', name+'J')
            
            for name2 in os.listdir():
                if name2[:14] == 'Var_scal_Auto_':
                    name3 = name + name2[8:]
                    os.replace(name2, name3)
    
    if os.path.exists('ineq'):
        os.remove('ineq')
    for name in os.listdir():
        if name[-3:] == '_eq':
            print( 'Analysing {}'.format(name) )
            os.symlink(name, 'ineq')
            command = alfdir + '/Analysis/cov_eq.out'
            os.system(command)
            os.remove('ineq')
            
            for name2 in os.listdir():
                if name2[:14] == 'Var_eq_Auto_Tr':
                    name3 = name + name2[6:]
                    os.replace(name2, name3)
    
    if os.path.exists('intau'):
        os.remove('intau')
    for name in os.listdir():
        if name[-4:] == '_tau':
            print( 'Analysing {}'.format(name) )
            os.symlink(name, 'intau')
            command = alfdir + '/Analysis/cov_tau.out'
            os.system(command)
            os.remove('intau')
            
            for name2 in os.listdir():
                if name2[:2] == 'g_':
                    directory = name[:-4] + name2[1:]
                    if not os.path.exists(directory):
                        os.mkdir(directory)
                    os.replace(name2, directory +'/'+ name2)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Helper script for compiling, running and testing ALF.',
        #formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--alfdir', required=True,      
                        help="Path to ALF directory")
    parser.add_argument('--type',     default="R"     , 
                        help='Options R+T, R, T with R=Run,  T=Test. (default: R)')
    parser.add_argument('--branch_R', default="master", 
                        help='Git branch to checkout for run.        (default: master)')
    parser.add_argument('--branch_T', default="master", 
                        help='Branch to test against Runbranch.      (default: master)')
    parser.add_argument('--config',   default="Intel" , 
                        help='Will run ./configureHPC.sh CONFIG      (default: Intel)')
    parser.add_argument('--executable_R',               
                        help='Name of  ref executable.               (default: <Model>.out)')
    parser.add_argument('--executable_T',               
                        help='Name of test executable.               (default: <Model>.out)')
    
    args = parser.parse_args()
    
    Type         = args.type
    alfdir       = os.path.expanduser(args.alfdir)
    branch_R     = args.branch_R
    branch_T     = args.branch_T
    Config       = args.config
    executable_R = args.executable_R
    executable_T = args.executable_T
    
    with open("Sims") as f:
        simulations = f.read().splitlines()
    print ( "Number of simulations ", len(simulations))
    for sim in simulations:
        if sim.strip() ==  "stop":
            print("Done")
            exit()
        sim = json.loads(sim)
        model = sim['Model']
        print("Model is", model) 
        params = Set_param(sim)
        Dir = Directory_name(sim)
        print(Dir)
        cwd = os.getcwd()
        rundir = str(cwd+"/"+Dir)
        print ("rundir is ", rundir)
        print(Type, Type.split("+"),Type.split("+")[0] )
        if "R" in str(Type.split("+")).strip():
            if executable_R == None:
                executable1 = model
            else:
                executable1 = executable_R
            #Run(rundir,alfdir,branch_R,Config,executable1,params)
            Compile(alfdir, branch_R, Config, executable1)
            Run(rundir, alfdir, executable1, params)
            with open("Kin_scalJ") as f:
                Kin_R = f.read().splitlines()[2]
        os.chdir(cwd)
        if "T" in str(Type.split("+")).strip():
            rundirT=str(rundir+"_Test")
            if executable_T == None:
                executable1 = model
            else:
                executable1 = executable_T
            Compile(alfdir, branch_T, Config, executable1)
            Run(rundir, alfdir, executable1, params)
            with open("Kin_scalJ") as f:
                Kin_T=f.read().splitlines()[2]
        os.chdir(cwd)
        with open(str(rundir)+".txt","w") as f:
            f.write(Kin_T)
            f.write(Kin_R)
        
        
# You need to write a general  run routine 
#  in ALFdir, Branch,  Config, rundir,  executable
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
