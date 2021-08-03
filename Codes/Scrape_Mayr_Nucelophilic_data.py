# Script to extract Mayr nucleophilicity data from html version
# Note that this script gave a rough extraction which was further curated by hand
# Import regular expressions and BeautifulSoup
import re
from bs4 import BeautifulSoup
# Initiate lists to store scraped data
mol_id = []
mol_name = []
mol_smiles = []
mol_N = []
mol_SN = []
mol_solvent = []
# open a .txt file made from the table section of the original html file
with open('concise_database.txt') as f:
    # open data
    data = BeautifulSoup(f, 'html.parser')
    # get table
    table = data.find_all('table')[0]
    # for each table row
    for tr in table.find_all('tr'):
        # for each row cell
        for td in tr.find_all('td'):
            td = str(td)
            # use regular expressions to match correct data to correct list
            # molecule id
            id = re.findall('<td\sid=\"(.+)\"><a', td)
            if len(id) > 0: mol_id.append(id[0])
            # molecule name
            name = re.findall('[0-9].+>(.+)</a><div', td)
            if len(name) > 0: mol_name.append(name[0])
            # molecule solvent
            solvent = re.findall('^<td>(.+)</td>$',td)
            if len(solvent) > 0 and re.match('.+=.+', solvent[0]) is None and re.match('.+<.+',solvent[0]) is None: mol_solvent.append(solvent[0])
            # molecule N
            N = re.findall('Param\.:\s(.+)</p><i>s</i><sub>N</sub>', td)
            if len(N) > 0: mol_N.append(N[0])
        # molecule SN
        SN = re.findall('sub>\sParam\.:\s(.+)</td><td><img', str(tr))
        if len(SN) > 0: mol_SN.append(SN[0])
        #SMILES
        smiles = re.findall('>[A-Z].+</td><td>(.+)</td></tr></table>',str(tr))
        if len(smiles) > 0: mol_smiles.append(smiles[0])
# write data to file
with open('mayr_updated_database.txt', 'w') as wf:
    wf.write('id\tname\tsolvent\tsmiles\tN_parameter\tSN_parameter\n')
    for num in range(len(mol_id)):
        wf.write(mol_id[num])
        wf.write('\t')
        wf.write(mol_name[num])
        wf.write('\t')
        wf.write(mol_solvent[num])
        wf.write('\t')
        wf.write(mol_smiles[num])
        wf.write('\t')
        wf.write(mol_N[num])
        wf.write('\t')
        wf.write(mol_SN[num])
        wf.write('\n')
