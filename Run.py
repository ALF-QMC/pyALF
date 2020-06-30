#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:05:24 2020

@author: fassaad
"""

from pyALF import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Helper script for compiling, running and testing ALF.',
        #formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--alfdir', required=True,      
                        help="Path to ALF directory")
    parser.add_argument('-R', action='store_true', 
                        help='Do a run')
    parser.add_argument('-T', action='store_true', 
                        help='Do a test')
    parser.add_argument('--branch_R', default="master", 
                        help='Git branch to checkout for run.        (default: master)')
    parser.add_argument('--branch_T', default="master", 
                        help='Branch to test against Runbranch.      (default: master)')
    parser.add_argument('--config',   default='GNU noMPI' , 
                        help='Will run ./configureHPC.sh CONFIG      (default: Intel)')
    parser.add_argument('--executable_R',               
                        help='Name of  ref executable.               (default: <Model>.out)')
    parser.add_argument('--executable_T',               
                        help='Name of test executable.               (default: <Model>.out)')
    
    args = parser.parse_args()
    
    do_R         = args.R
    do_T         = args.T
    alf_dir      = os.path.abspath(args.alfdir)
    branch_R     = args.branch_R
    branch_T     = args.branch_T
    config       = args.config
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
        if do_R:
            compile_alf(alf_dir, branch_R, config)
            run(sim, alf_dir, executable_R)
            ana(sim)
            obs1 = get_obs(sim)
        if do_T:
            raise Exception('Test not yet implemted')
        
        #with open(sim_dir+".txt","w") as f:
            #f.write( 'Run:  {} +- {}\n'.format( *Kin_R[0] ) )
            #f.write( 'Test: {} +- {}\n'.format( *Kin_T[0] ) )
            #f.write( 'Diff: {} +- {}\n'.format( *(Kin_R[0] - Kin_T[0])  ) )
