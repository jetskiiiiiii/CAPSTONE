from OneHotEncoding import get_one_hot_sequence
from BuildMatrixFromDotBracket import get_couples, build_matrix

# getting input

# defining folder path
bprna_folder_path = "../../DATASETS/bpRNA_1m_90/bpRNA_1m_90_DBNFILES_as_TXT/"
pattern = r"bpRNA_CRW_.*\.txt$"  # get files that start with "bpRNA_CRW_"
extracted_data = {}
bprna_df = pd.DataFrame(columns=["name", "length", "sequence", "dbn"])

# iterate through
for filename in os.listdir(bprna_folder_path):
    if re.match(pattern, filename):
        with open(os.path.join(bprna_folder_path, filename), "r") as file:
            for i, line in enumerate(file):
                if i == 0:
                    key, value = line.strip().split(" ", 1)
                    extracted_data["name"] = value
                elif i == 1:
                    key, value = line.strip().split(" ", 1)
                    extracted_data["length"] = value
                elif i == 3:
                    extracted_data["sequence"] = line
                elif i == 4:
                    extracted_data["dbn"] = line
        bprna_df = pd.concat(
            [bprna_df, pd.DataFrame([extracted_data])], ignore_index=True
        )
        extracted_data = {}


example_structure = bprna_df.loc[8, "dbn"]
example_sequence_length = int(bprna_df.loc[8, "length"])
example_couples = get_couples(example_structure)
example_matrix = build_matrix(example_couples, example_sequence_length)
example_one_hot = get_one_hot_sequence(bprna_df.loc[8, "sequence"])
example_one_hot = example_one_hot[:-1, :]
