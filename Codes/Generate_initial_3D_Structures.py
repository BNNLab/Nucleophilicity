# This code takes SMILES strings and make input files for Gaussian 09 via generation of initial 3D structures with CIRpy and molecular mechanics. Shown below is for solvent optimisations. Simple modification to remove “SCRF=(solvent=solvent)” from the command line provided gas input files.
# import modules
import cirpy
import os
import pandas as pd
# params to change
# location of .csv file (save as in excel) must have a column called "Smile" or "Smiles" with the SMILES codes in, a column called "mol_name" with a unique identifier for the molecule in, and a column called "Solvent" with the solvent in
input_file = "input.csv"
# folder where you want the output files to go
output_file_dir = "output"
# commands for Gaussian
commands = "#opt freq m062x/def2svp Volume pop=(nbo,savenbo) SCRF=(solvent="
# end of params to change
# read in input file
data = pd.read_csv(input_file)
# make folder to put files in
os.mkdir(output_file_dir)
# get list of SMILES
try:
    smi2 = data["Smiles"]
except:
    smi2 = data["Smile"]
# get list of names
mol_name = data["mol_name"]
# get list of solvents
solvents = data["Solvent"]
# for each SMILES code
for j in range(len(smi2)):
    # get name, SMILES, and solvent
    name = mol_name[j]
    smi = smi2[j]
    solvent = solvents[j]
    # try to get initial 3D structure
    try:
        m = cirpy.resolve(smi,'xyz')
    except:
        # if there's a problem it will print the name
        print(name)
        # skip to next molecule
        continue
    try:
        # check xyz structure obtained
        m = m.splitlines()
    except AttributeError:
        # if there's a problem it will print the name
        print(name)
        # skip to next molecule
        continue
    # format xyz structure
    del(m[0])
    del(m[0])
    # deduce charge from SMILES
    pos=smi.count('+')
    neg=smi.count('-')
    charge=(-neg)+pos
    # open input file and write contents
    f = open (output_file_dir + "\\" + name + ".com","a")
    f.write("%nprocshared=4\n")
    f.write("%mem=100MW\n")
    f.write("%chk=" + name + ".chk\n")
    f.write(commands + solvent + ")" + \n")
    f.write("\nOptimisation\n\n")
    f.write(str(charge) + " 1\n")
    for i in range(len(m)):
        f.write(m[i])
        f.write('\n')
    f.write('\n')
    f.write('\n')
    f.close()
