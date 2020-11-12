# DigiPico
Analysis Scripts for DigiPico Sequencing Data (https://elifesciences.org/articles/55207)

## MutLX
MutLX tool for accurate identification of true islet specific variants from DigiPico data. 

### Installation

You can either download or clone this repository.

### Requirements

This project has been tested on Python 3. The `MutLX/requirements.txt` file contains all Python libraries that you need. They can be installed by running the following command in the project's folder:

```
pip install -r requirements.txt
```

### Input

The input should be a csv file in the following format without header:

Mutation, Type, [41 MutLX features extracted from DigiPico data]

Type column must be of any of the following categories:
* SNP-Unq: For UTD variants
* SNP-Hm: For homozygous germline variants
* SNP-Ht-H: For high-confidence heterezygous germline variants (Important for tumours with complex genomes)
* SNP-Ht: For other heterozygous germline variants
* SNP-Somatic: For known somatic mutations

Example files for analysis with DigiPico can be downloaded from below links:
* [Sample file for DigiPico run without expected islet specific variants](https://drive.google.com/open?id=11m_fSPoW2oqmk8H8Ffqpu2FdN9dfuyet)
* [Sample file for DigiPico run with expected islet specific variants](https://drive.google.com/open?id=1j2LFKdEDBOrWKA2yG525jWQlfDOlTnvb)

### Running MutLX

You can run the following command in the project's folder:
```
python mutLX.py --input test1.csv --out_path test1_Results --sample_name DigiPico_test1
```

### Arguments

* --input: Path to input csv file
* --out_path: Output directory path (default = run directory)
* --sample_name: Sample name to be used as prefix for output files (default = "DigiPico")
* --batch_size: Training batch size (default = 8) 
* --epochs: Number of epochs in training (default = 10)
* --subset_num: Number of training subsets to be used for training (default = 25)
* --drop_it: Number of iterations for dropout analysis (default = 100)
* --pscore_cf: Probability score cut-off value (default = 0.2)
* --auc_cf: Cut-off value for area under the ROC curve to identify samples with true UTDs (default = 0.9)
* --tpr_cf: The required true positive rate based on germline SNPs for the recovery of true UTDs (default = 0.95)

### Output

mutLX.py will generate a final sample_name_scores.csv file in the out_path directory with the below header as described in our manuscript:

Mutation, Type, Probability_Score, Uncertainty_Score, Result

It will also generate several plots to represent the data. 

