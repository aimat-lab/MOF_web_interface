import os
from molSimplify.Informatics.MOF.MOF_descriptors import get_MOF_descriptors
import pandas as pd

### Defining pymatgen function for getting primitive, since molSimplify does not depend on pymatgen
from pymatgen.io.cif import CifParser


def get_primitive(datapath, writepath):
    s = CifParser(datapath, occupancy_tolerance=1).get_structures()[0]
    sprim = s.get_primitive_structure()
    sprim.to("cif", writepath)


def compute_features(MOF_random_name, startpath):

    featurization_list = []
    #### Adapt this directory for your use.


    #### Place your clean cif files of interest inside of this directory, under cif/
    #### Thus, the path where the cif file should be is: /Users/<your username>/Desktop/example_MOF/cif/

    ### The code expects to look for a directory called /cif/ in the example directory

    #### This first part gets the primitive cells ####
    if not os.path.exists('%s/cif_files/%s/molsimplify/'%(startpath, MOF_random_name)):
        os.mkdir('%s/cif_files/%s/molsimplify/'%(startpath, MOF_random_name))
    if not os.path.exists('%s/cif_files/%s/xyz/'%(startpath, MOF_random_name)):
        os.mkdir('%s/cif_files/%s/xyz/'%(startpath, MOF_random_name))

    cif_file = "%s/cif_files/%s/mof.cif"%(startpath, MOF_random_name)
    get_primitive("%s/cif_files/%s/mof.cif"%(startpath, MOF_random_name), "%s/cif_files/%s/mof_primitive.cif"%(startpath, MOF_random_name))
    #### With the primitive cells, we can generate descriptors and write them
    full_names, full_descriptors = get_MOF_descriptors("%s/cif_files/%s/mof_primitive.cif"%(startpath, MOF_random_name), 3, path="%s/cif_files/%s/molsimplify"%(startpath, MOF_random_name), xyzpath="%s/cif_files/%s/xyz/mof.xyz"%(startpath, MOF_random_name))
    full_names.append('filename')
    full_descriptors.append(cif_file)
    featurization = dict(zip(full_names, full_descriptors))
    featurization_list.append(featurization)
    df = pd.DataFrame(featurization_list) 
    ### Write the RACs to the directory. Full featurization frame contains everything.
    df.to_csv('%s/cif_files/%s/full_featurization_frame.csv'%(startpath, MOF_random_name), index=False) 


    #### The full featurization frame contains all features. 
    # The following table can help decode features:
    # mc --> metal centered products 
    # D_mc --> metal centered differences
    # lc --> linker connecting atom centered products
    # D_lc --> linker connecting atom centered differences
    # f- --> full MOF unit cell (not used in https://www.nature.com/articles/s41467-020-17755-8)
    # f-lig --> full linker RACs
    # func --> functional group centered products
    # D_func --> functional group centered differences

    # All Zeo++ features should be computed separately.


    return(df)

#{"Attachments":[{"__type":"ItemIdAttachment:#Exchange","ItemId":{"__type":"ItemId:#Exchange","ChangeKey":null,"Id":"AAMkAGRjYWRlNzJhLWEwN2UtNDZjYi1iMmIyLTdlZGJjOWU3ZjRmNgBGAAAAAADMnOBbYa6MSKjp8OJsYK9nBwBuWsT8v0xyTLJK6O6dtNFSAAAYgBZVAACtRyp1CYZDSYWi4GE9gPa5AAGoGF7VAAA="},"Name":"AEM Inverse-design project","IsInline":false}]}
