giab_datasets = ['/home/ndo/GIAB-GT/GIAB_SNP_NA12878.vcf', '/home/ndo/GIAB-GT/GIAB_SNP_NA24149.vcf', '/home/ndo/GIAB-GT/GIAB_SNP_NA24631.vcf', '/home/ndo/GIAB-GT/GIAB_SNP_NA24694.vcf', '/home/ndo/GIAB-GT/GIAB_SNP_NA24695.vcf']

SEs = ['SRR13586011','SRR13586012','SRR13586013', \
            'SRR13586014', 'SRR13586015', 'SRR13586016', 'SRR13586018', 'SRR13586019', 'SRR13586020']

PEs = ['SRR13586007', 'SRR13586026']

SEs = []
PEs = ['SRR13586007']
datasets = []

for SE in SEs:
    vcf_link = f"/home/ndo/nextflow_SE/outDir/VarCall/{SE}_vardict.vcf"
    datasets.append(vcf_link)

for PE in PEs:
    vcf_link = f"/home/ndo/nextflow/outDir/VarCall/{PE}_vardict.vcf"
    datasets.append(vcf_link)


giab_to_dataset = {'NA12878': {'SRR13586007','SRR13586016'}, 'NA24695': {'SRR13586011'}, 'NA24694': {'SRR13586012','SRR13586013','SRR13586014', 'SRR13586015'}, \
           'NA24631': {'SRR13586018', 'SRR13586019', 'SRR13586020'}, 'NA24149': {'SRR13586026'}}

# Create an empty dictionary to store the reversed mapping
dataset_to_giab = {}

# Iterate through the original dictionary
for giab, ids in giab_to_dataset.items():
    # For each dataset, map it to the corresponding GIAB value
    for id in ids:
        dataset_to_giab[id] = giab

import pandas as pd
import numpy as np
import swifter

def vcf_to_df(vcf_file, extract_flank_seqs=True):
    """ Converts vcf file into a pandas dataframe
    """
    with open(vcf_file) as vcf:
        for line in vcf:
            if line.startswith('#CHROM'):
                header_names = line.strip().split()
                header_names[0] = header_names[0][1:]
                break
        read_lines = vcf.readlines()
        # list containing each row as 1 giant string

    
    # split the giant string into columns -- now we have a 2D list
    create_columns = [row.strip().split() for row in read_lines]

    final = []
    
    # extracts LSEQ and RSEQ for SRR vcf file if needed, in addition to CHROM, POS, REF, ALT
    if extract_flank_seqs == True:
        for row in create_columns:
            info = row[7].split(';')
            if info[1] != 'TYPE=SNV':
                continue
            final.append([row[0], row[1], row[3], row[4], row[7]]) 
        df = pd.DataFrame(data=final, columns=header_names[0:2]+header_names[3:5]+['INFO'])
        return df
    
    # extracts CHROM, POS, REF, ALT    
    for row in create_columns:
        final.append([row[0], row[1], row[3], row[4]]) # CHROM, POS, REF, ALT                     
    df = pd.DataFrame(data=final, columns=header_names[0:2]+header_names[3:5])   
    return df

def parse_and_convert_type(df):


    # Assuming your DataFrame is named df
    # Split the INFO column by semicolons to get key-value pairs
    info_split = df['INFO'].str.split(';')

    # Create a dictionary to store the key-value pairs
    info_dict = {}

    # Iterate through each row of the split INFO column
    for row in info_split:

        # Filter out SNV:
        if row[1] != 'TYPE=SNV':
            continue
        # Iterate through each key-value pair in the row
        for item in row:
            # Split the key-value pair by '='
            key, value = item.split('=')
            # Add the key-value pair to the dictionary
            if key in info_dict:
                info_dict[key].append(value)
            else:
                info_dict[key] = [value]

    # Create a DataFrame from the dictionary
    info_df = pd.DataFrame(info_dict)

    # Rename the columns
    info_df.rename(columns={'TYPE': 'TYPE', 'DP': 'DP', 'VD': 'VD', 'AF': 'AF', 'BIAS': 'BIAS', 'REFBIAS': 'REFBIAS', 'VARBIAS': 'VARBIAS', 'PMEAN': 'PMEAN', 'PSTD': 'PSTD', 'QUAL': 'QUAL', 'QSTD': 'QSTD', 'SBF': 'SBF', 'ODDRATIO': 'ODDRATIO', 'MQ': 'MQ', 'SN': 'SN', 'HIAF': 'HIAF', 'ADJAF': 'ADJAF', 'SHIFT3': 'SHIFT3', 'MSI': 'MSI', 'MSILEN': 'MSILEN', 'NM': 'NM', 'HICNT': 'HICNT', 'HICOV': 'HICOV', 'LSEQ': 'LSEQ', 'RSEQ': 'RSEQ', 'DUPRATE': 'DUPRATE', 'SPLITREAD': 'SPLITREAD', 'SPANPAIR': 'SPANPAIR'}, inplace=True)

    # Merge with the original DataFrame
    df = pd.concat([df, info_df], axis=1)

    # Drop the original INFO column
    df.drop(columns=['INFO'], inplace=True)


    # Convert CHROM and POS columns to create MultiIndex
    df.index = pd.MultiIndex.from_tuples(zip(df['CHROM'], df['POS']))
    # df.drop(['CHROM', 'POS'], axis=1, inplace=True)
    

    # Convert columns to desired data types
    df['POS'] = df['POS'].astype('int64')
    df['AF'] = df['AF'].astype(float)
    df['ADJAF'] = df['ADJAF'].astype(float)
    df['DP'] = df['DP'].astype(int)
    df['DUPRATE'] = df['DUPRATE'].astype(int)
    df['HIAF'] = df['HIAF'].astype(float)
    df['HICNT'] = df['HICNT'].astype(int)
    df['HICOV'] = df['HICOV'].astype(int)
    df['MQ'] = df['MQ'].astype(float)
    df['MSI'] = df['MSI'].astype(float)
    df['MSILEN'] = df['MSILEN'].astype(int)
    df['NM'] = df['NM'].astype(float)
    df['ODDRATIO'] = df['ODDRATIO'].astype(float)
    df['PMEAN'] = df['PMEAN'].astype(float)
    df['PSTD'] = df['PSTD'].astype(int)
    df['QSTD'] = df['QSTD'].astype(int)
    df['QUAL'] = df['QUAL'].astype(float)
    df['SBF'] = df['SBF'].astype(float)
    df['SHIFT3'] = df['SHIFT3'].astype(int)
    df['SN'] = df['SN'].astype(float)
    df['SPANPAIR'] = df['SPANPAIR'].astype(int)
    df['SPLITREAD'] = df['SPLITREAD'].astype(int)
    return df

def preprocessing_sample_dataset(vcf_file):
    df = vcf_to_df(vcf_file, extract_flank_seqs=True)
    return parse_and_convert_type(df)


sampleId_to_df = {}
for dataset in datasets:
    # Split the path and extract the sample ID (assuming format like SRR<sample_id>_vardict.vcf)
    print("dataset", dataset)
    sample_id = dataset.split('/')[-1].split('_')[0]
    df = preprocessing_sample_dataset(dataset)
    sampleId_to_df[sample_id] = df

sampleId_to_df = {}
for dataset in datasets:
    # Split the path and extract the sample ID (assuming format like SRR<sample_id>_vardict.vcf)
    print("dataset", dataset)
    sample_id = dataset.split('/')[-1].split('_')[0]
    df = preprocessing_sample_dataset(dataset)
    sampleId_to_df[sample_id] = df

# Create a hashmap to store sample ID as key and corresponding DataFrame as value
hashmap = {}

for dataset in giab_datasets:
    # Extract the sample ID from the file path (assuming format like GIAB_SNP_<sample_id>.vcf)
    sample_id = dataset.split('_')[-1].split('.')[0]
    
    # Read the VCF file into a DataFrame
    df = read_vcf_with_header(dataset)
    df= df[['#CHROM','POS','REF','ALT']]
    df.rename(columns={'#CHROM': 'CHROM'}, inplace=True)
    # Store the DataFrame in the hashmap with the sample ID as the key
    hashmap[sample_id] = df

merge_dfs = {}
for sample_id, giab_id in dataset_to_giab.items():
    sample_df = sampleId_to_df[sample_id]
    giab_df = hashmap[giab_id]
    merge_df = pd.merge(sample_df, giab_df, how="outer", on=["CHROM", "POS"])
    merge_dfs[sample_id] = merge_df

def categorize_variants(df):
    if (df["ALT_GIAB"] == df["ALT_S"]) and (df["REF_GIAB"] == df["REF_S"]):
        return "TP"
    if (df["ALT_GIAB"] != df["ALT_S"]) and (df["REF_GIAB"] == df["REF_S"]):
        return "FP"
    if  df["ALT_GIAB"] and pd.isna(df["ALT_S"]):
        return "FN"
    if  pd.isna(df["ALT_GIAB"]) and df["ALT_S"]:
        return "FP"
    return "None"

for id, m_df in merge_dfs.items():
    m_df = m_df.rename(columns={"REF_x" : "REF_GIAB", "ALT_x" : "ALT_GIAB", "REF_y" : "REF_S", "ALT_y" : "ALT_S"})
    m_df['VAR_CATE'] = m_df.swifter.apply(categorize_variants, axis = 1)
    merge_dfs[id] = m_df

#DROP the INDELS in GIAB
for id, m_df in merge_dfs.items():
    #DROP the INDELS in GIAB
    m_df.drop(m_df[m_df.VAR_CATE == 'None'].index, inplace=True)
    merge_dfs[id] = m_df

# Save the preprocessing dataframe to a text file
# Iterate through the merge_dfs dictionary
for sample_id, df in merge_dfs.items():
    # Construct the CSV file name
    csv_file_name = f"/home/ndo/vardict_ML/mergeDf_pre_filter_FN/{sample_id}_df.csv"
    # Write the DataFrame to the CSV file
    df.to_csv(csv_file_name, index=False)
    print(f"Data for {sample_id} written to {csv_file_name}")