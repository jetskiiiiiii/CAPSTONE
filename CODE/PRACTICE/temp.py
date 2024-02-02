import pandas as pd
from rdkit.Chem import PandasTools

pp = pd.read_csv("PRACTICE/VIETHERB_SMILES.csv", names=["Smiles"])
PandasTools.AddMoleculeColumnToFrame(pp, "Smiles")  # pp = doesn't work for me
PandasTools.WriteSDF(pp, "pp_out.sdf", properties=list(pp.columns))
