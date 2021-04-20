Files and folders:
DFT/Gas_structures/ - xyz structures optimised in the gas phase using DFT
DFT/Solution_structures/ - xyz structures optimised in the solution phase using DFT
Li_structures/ - xyz structures for structures optimised in gas phase with Li probe inserted at the appropriate location
PM6/Gas_structures/ - xyz structures optimised in the gas phase using PM6
PM6/Solution_structures/ - xyz structures optimised in the solution phase using PM6
Example_input_files/ - example .com files used as an input for Gaussian 09
GAS_SET_16_Descriptors.csv - Gas phase descriptors
SOL_SET_17_Descriptors.csv - Solution phase descriptors

Descriptor Columns (see Supporting Information for more details):
Name - Unique name given to each molecule
Train_test - Whether the molecule is in the training or test set in the fixed training:test split models
SMILES - SMILES code for molecule
Solvent - Solvent N was measured in
Type - Type of nucleophile
N - Experimental Mayr's nucleophilic parameter
sol_PCA1 - First solvent principal component
sol_PCA2 - Second solvent principal component
sol_PCA3 - Third solvent principal component
sol_PCA4 - Fourth solvent principal component
sol_PCA5 - Fifth solvent principal component
N_HOMO - HOMO of optimised gas phase structure
N_LUMO - LUMO of optimised gas phase structure
N_DM - Dipole of optimised gas phase structure
N_EB - Electronic basicity of optimised gas phase structure
N_EA_H - Electronic acidity of optimised gas phase structure
N_EA_nonH  Electronic acidity of non-hydrogen atom of optimised gas phase structure
N_TCA - Tolman Cone Angle
N_BAD - Single maximum angle from TCA analysis
N_fukui - Condensed Fukui functional of most nucleophilic atom according to Fukui analysis
N_fukui_Li - Condensed Fukui functional of most nucleophilic atom according to Li structure
N_fukui_charge - Hirshfeld charge of most nucleophilic atom according to Li structure
NS_HOMO - HOMO of optimised solution phase structure
NS_LUMO - LUMO of optimised solution phase structure
NS_DM - Dipole of optimised solution phase structure
NS_EB - Electronic basicity of optimised solution phase structure
NS_EA_H - Electronic acidity of optimised solution phase structure
NS_EA_nonH - Electronic acidity of non-hydrogen atom of optimised solution phase structure
NS_DeltaG - Difference between Gibbs energy of solution phase structure and Gibbs energy of gas phase structure
*_PM6 - Corresponding descriptors calculated with PM6 instead of DFT which replaced DFT descriptors in PM6 models (all other descriptors were the same in PM6 and DFT models)