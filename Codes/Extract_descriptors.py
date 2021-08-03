# This code obtains the gas or solution descriptors from a folder of log files (the output from Gaussian 09 optimisation). One final descriptor, "NS_DeltaG", was calculated form the difference between NS_G0 and N_G0.
# import modules
import pandas as pd
import re,os
# parameters to chang
# path to folder containing log files (folder must be called "log")
log_path="gas\\log"
# where and what you want the output .csv file to be
output="descs.csv"
# end of parameters to change
# method to get first half of descriptors using log file and regular expressions
def getDesc(log_path):
    df_data=[]
    for files in os.listdir(log_path):
        files=files.replace(".log","")
        f = open(log_path+'\\'+files+'.log',"r") # open solv log
        data = []
        for line in f:
            data.append(line)
        # E0
        scfd=re.compile(r'SCF D.*')
        scfd2=re.compile(r'-?\d+\.\d+E?-?\d?\d?')
        for f in range(len(data)):
            if re.findall(scfd,str(data[f])) != []:
                E0_solv = re.findall(scfd,str(data[f]))
        try:
            E0_solv=re.findall(scfd2,str(E0_solv))
        except UnboundLocalError:
            continue
        E0_solv=str(E0_solv)
        E0_solv=E0_solv.replace("['","")
        E0_solv=E0_solv.replace("']","")
        E0_solv=float(E0_solv)
        # dipole
        dip=re.compile(r'Tot=.*')
        dip2=re.compile(r'-?\d+\.\d+')
        for f in range(len(data)):
            if re.findall(dip,str(data[f])) != []:
                solv_dipole = re.findall(dip,str(data[f]))
        solv_dipole=re.findall(dip2,str(solv_dipole))
        solv_dipole=str(solv_dipole)
        solv_dipole=solv_dipole.replace("['","")
        solv_dipole=solv_dipole.replace("']","")
        solv_dipole=float(solv_dipole)
        # G0
        therm=re.compile(r'mal Free.*')
        therm2=re.compile(r'-?\d+\.\d+')
        for f in range(len(data)):
            if re.findall(therm,str(data[f])) != []:
                G_solv = re.findall(therm,str(data[f]))
        try:
            G_solv=re.findall(therm2,str(G_solv))
        except UnboundLocalError:
            continue
        G_solv=str(G_solv)
        G_solv=G_solv.replace("['","")
        G_solv=G_solv.replace("']","")
        G_solv=float(G_solv)
        # volume
        vol=re.compile(r'Molar volume.*')
        for f in range(len(data)):
                if re.findall(vol,str(data[f])) != []:
                        Volume = re.findall(vol,str(data[f]))
        vol2=re.compile(r'\d+\.\d+')
        try:
            Volume=re.findall(vol2,str(Volume))[-1]
        except UnboundLocalError:
            print('bad', files)
            Volume = 0
        Volume=str(Volume)
        Volume=Volume.replace("['","")
        Volume=Volume.replace("']","")
        Volume=float(Volume)
        # HOMO
        HOMO=re.compile(r'Alpha  occ.*')
        for f in range(len(data)):
               if re.findall(HOMO,str(data[f])) != []:
                    HO = re.findall(HOMO,str(data[f]))
                    j=f
        HOMO2=re.compile(r'-?\d+\.\d+')
        HO=re.findall(HOMO2,str(HO))[-1]
        HO=str(HO)
        HO=HO.replace("['","")
        HO=HO.replace("']","")
        HO=float(HO)
        # LUMO
        LUMO=re.compile(r'Alpha virt.*')
        try: 
            LU = re.findall(LUMO,str(data[j+1]))
        except IndexError:
            continue
        LUMO2=re.compile(r'-?\d+\.\d+')
        LU=re.findall(LUMO2,str(LU))[0]
        LU=str(LU)
        LU=LU.replace("['","")
        LU=LU.replace("']","")
        LU=float(LU)
        # add descriptors
        Descs=[]
        Descs.append(files)
        Descs.append(E0_solv)
        Descs.append(G_solv)
        Descs.append(Volume)
        Descs.append(HO)
        Descs.append(LU)
        Descs.append(solv_dipole)
        df_data.append(Descs)
    df_data=pd.DataFrame(data=df_data, columns=["Name","N_E0","N_G0","N_vol","N_HOMO","N_LUMO","N_DM"])
    return(df_data)
# method to get xyz structure for each molecule and save in a new folder in same directory as log
# key for convert atomic number to atomic symbol
code = {"1" : "H", "2" : "He", "3" : "Li", "4" : "Be", "5" : "B", \
"6"  : "C", "7"  : "N", "8"  : "O", "9" : "F", "10" : "Ne", \
"11" : "Na" , "12" : "Mg" , "13" : "Al" , "14" : "Si" , "15" : "P", \
"16" : "S"  , "17" : "Cl" , "18" : "Ar" , "19" : "K"  , "20" : "Ca", \
"21" : "Sc" , "22" : "Ti" , "23" : "V"  , "24" : "Cr" , "25" : "Mn", \
"26" : "Fe" , "27" : "Co" , "28" : "Ni" , "29" : "Cu" , "30" : "Zn", \
"31" : "Ga" , "32" : "Ge" , "33" : "As" , "34" : "Se" , "35" : "Br", \
"36" : "Kr" , "37" : "Rb" , "38" : "Sr" , "39" : "Y"  , "40" : "Zr", \
"41" : "Nb" , "42" : "Mo" , "43" : "Tc" , "44" : "Ru" , "45" : "Rh", \
"46" : "Pd" , "47" : "Ag" , "48" : "Cd" , "49" : "In" , "50" : "Sn", \
"51" : "Sb" , "52" : "Te" , "53" : "I"  , "54" : "Xe" , "55" : "Cs", \
"56" : "Ba" , "57" : "La" , "58" : "Ce" , "59" : "Pr" , "60" : "Nd", \
"61" : "Pm" , "62" : "Sm" , "63" : "Eu" , "64" : "Gd" , "65" : "Tb", \
"66" : "Dy" , "67" : "Ho" , "68" : "Er" , "69" : "Tm" , "70" : "Yb", \
"71" : "Lu" , "72" : "Hf" , "73" : "Ta" , "74" : "W"  , "75" : "Re", \
"76" : "Os" , "77" : "Ir" , "78" : "Pt" , "79" : "Au" , "80" : "Hg", \
"81" : "Tl" , "82" : "Pb" , "83" : "Bi" , "84" : "Po" , "85" : "At", \
"86" : "Rn" , "87" : "Fr" , "88" : "Ra" , "89" : "Ac" , "90" : "Th", \
"91" : "Pa" , "92" : "U"  , "93" : "Np" , "94" : "Pu" , "95" : "Am", \
"96" : "Cm" , "97" : "Bk" , "98" : "Cf" , "99" : "Es" ,"100" : "Fm", \
"101": "Md" ,"102" : "No" ,"103" : "Lr" ,"104" : "Rf" ,"105" : "Db", \
"106": "Sg" ,"107" : "Bh" ,"108" : "Hs" ,"109" : "Mt" ,"110" : "Ds", \
"111": "Rg" ,"112" : "Uub","113" : "Uut","114" : "Uuq","115" : "Uup", \
"116": "Uuh","117" : "Uus","118" : "Uuo"}
def get_xyzs(path):
    # make folder in same location for xyzs
    os.makedirs(path.replace('\\log','') + '\\xyz')  # change the first str
    for files in os.listdir(path):
        #for each log files in folder
        files=files.replace(".log","")
        try:
            f = open(path + "\\" + files + ".log","r")
        except IOError:
            continue
        data = []
        for line in f:
            data.append(line)
        start=re.compile(r'Standard orientation')
        dash=re.compile(r'------------')
        lineno=[]
        lineno2=[]
        for f in range(len(data)):
            if re.findall(start,str(data[f])) != []:
                lineno.append(f)
        for f in range(len(data)):
            if re.findall(dash,str(data[f])) != []:
                lineno2.append(f)
        coord=[]
        try: 
            linestart=lineno[-1]
        except IndexError:
            continue
        x=[]
        for f in range(len(lineno2)):
            if lineno2[f]>linestart:
                x.append(lineno2[f])
        lineend=x[2]
        for f in range(linestart,lineend):
            coord.append(data[f])
        for f in range(5):
            del(coord[0])
        for f in range(len(coord)):
            coord[f]=coord[f].split()
        #write xyz file
        f=open(path.replace('\\log','') + '\\xyz\\' + files + '.xyz','a')
        f.write(str(len(coord)) + '\n\n')
        for i in range(len(coord)):
            f.write(code[coord[i][1]] + ' ' + coord[i][3] + ' ' + coord[i][4] + ' ' + coord[i][5])
            if i==len(coord)-1:
                break
            f.write('\n')
        f.close()
# method to get charges
def getCharges(data4):
    lineno=[]
    lineno2=[]
    summ=re.compile(r'Summary o') 
    equals=re.compile(r'\* Total \*')
    Mull=re.compile(r'Mulliken charges:')
    Mull2=re.compile(r'Sum of Mulliken charges')
    z=0
    for f in range(len(data4)):
        if re.findall(summ,str(data4[f])) !=[]:
            z=5
            for f in range(len(data4)):
                if re.findall(summ,str(data4[f])) != []:
                    lineno.append(f)
            for f in range(len(data4)):
                if re.findall(equals,str(data4[f])) != []:
                    lineno2.append(f)
            charge=[]
            try:
                for f in range(lineno[-1],lineno2[-1]+1):
                    charge.append(data4[f])
            except IndexError:
                continue
                
            for f in range(len(charge)):
                charge[f]=charge[f].replace('\n','')
            for f in range(6):
                del(charge[0])
            del(charge[-1])
            del(charge[-1])
            for f in range(len(charge)):
                charge[f]=str.split(charge[f])
            for i in range(len(charge)):
                charge[i][2]=float(charge[i][2])
            charge = [item[0:3] for item in charge]
            return(charge)

    if z==0:
        for f in range(len(data4)):
            if re.findall(Mull,str(data4[f])) != []:
                    lineno.append(f)
        for f in range(len(data4)):
            if re.findall(Mull2,str(data4[f])) != []:
                lineno2.append(f)
        charge=[]
        
        for f in range(lineno[-1],lineno2[-1]+1):
            charge.append(data4[f])

        for f in range(len(charge)):
            charge[f]=charge[f].replace('\n','')
        for f in range(2):
            del(charge[0])
        del(charge[-1])
        for f in range(len(charge)):
            charge[f]=str.split(charge[f])
        for i in range(len(charge)):
            charge[i][2]=float(charge[i][2])
        for i in range(len(charge)):
            charge[i][0], charge[i][1] = charge[i][1], charge[i][0]
        return(charge)
# method to get charges save in a new folder in same directory as log
def get_charge_files(log_loc):    
    os.makedirs(log_loc.replace("log","charge"))
    for files in os.listdir(log_loc):
        files=files.replace(".log","")
        # open log file
        try:
            f = open(log_loc + "\\" + files + ".log","r")
        except FileNotFoundError:
            print('bad', files)
            continue
        data4 = []
        for line in f:
            data4.append(line)
        f.close()
        # open xyz file
        try:
            f = open(xyz_loc + "\\" + files + ".xyz","r")
        except FileNotFoundError:
            print('bad', files)
            continue
        xyz = []
        for line in f:
            xyz.append(line)
        f.close()
        del(xyz[0])
        del(xyz[0])
        for x in range(len(xyz)):
            xyz[x]=xyz[x].replace("\n","")
            xyz[x]=xyz[x].split()
        xyz = [x for x in xyz if x != []]
        xyz_df=pd.DataFrame(data=xyz,columns=["Atom","X","Y","Z"])
        charges=getCharges(data4)
        charge_df=pd.DataFrame(data=charges,columns=["Atom","Number","Charge"])
        charge_df["X"]=xyz_df["X"].tolist()
        charge_df["Y"]=xyz_df["Y"].tolist()
        charge_df["Z"]=xyz_df["Z"].tolist()
        charge_df=charge_df.drop(columns=['Number'])
        charge_df.to_csv(log_loc.replace("log","charge") + "\\" + files + ".csv",index=False)
# get the charge descriptors that were saved to file
def get_charge_descs(charge_path):    
    path = os.walk(charge_path)
    charge_dict = {'Name':[],
                   'Nu_site':[], 'N_EB':[],
                   'EAcidity_site':[], 'N_EA_H:[],
                   'Ephile_site':[], 'N_EA_nonH':[]}

    for i in path:
        file_name = i[2]
    for f in file_name:
        csv_path = charge_path + '\\' + f
        df = pd.read_csv(csv_path)
        mol_name = re.findall('(.+)\.csv',f)[0]
        Nu = df['Charge'][df['Atom']!='H'].min()
        Nu_site = df['Atom'][df['Charge']==Nu].values[0]
        EA = df['Charge'][df['Atom'] == 'H'].max()
        EA_site = 'H'
        EP = df['Charge'][df['Atom'] != 'H'].max()
        EP_site =  df['Atom'][df['Charge']==EP].values[0]

        charge_dict['Name'].append(mol_name)
        charge_dict['Nu_site'].append(Nu_site)
        charge_dict['N_EB'].append(Nu)
        charge_dict['EAcidity_site'].append(EA_site)
        charge_dict['N_EA_H'].append(EA)
        charge_dict['Ephile_site'].append(EP_site)
        charge_dict['N_EA_nonH'].append(EP)

    cd = pd.DataFrame(charge_dict)
    return(cd)
# run methods and save descriptors to desired location
descs1=getDesc(log_path)
get_xyzs(log_path)
xyz_loc=log_path.replace("log","xyz")
get_charge_files(log_path)
charge_path=log_path.replace("log","charge")
descs2=get_charge_descs(charge_path)
descs=pd.merge(descs1,descs2)
descs.to_csv(output,index=False)
