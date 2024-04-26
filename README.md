# UltraMutate

Implementation for the paper "Integrating Reinforcement Learning and Monte Carlo Tree Search for Enhanced Neoantigen Vaccine Design". UltraMutate uses an optimized policy network and Monte Carlo Tree Search to identify mutations to an neoantigen with low binding affinity to a specified HLA allele, while maintaining a high degree of similarity with the original neoantigen.

## Installing

Requires python 3.9 or later version.

```shell
pip install -r requirements.txt
```

This will automatically setup the environment.

## Example Usage

For a specified peptide-HLA pair, run main.py with:

```shell
python main.py --peptide VMMGLLMFS --HLA HLA-A*26:01 --num_simulations 10
```

Parameter --num_simulations is the number of simulations of MCTS performed. Ajusting this value will influence both the process time and the obtained results.

For a list of peptides and HLA alleles, a fasta file containing all peptides and a list of HLAs should be provided.

For example a fasta file containing peptides (peptides.fa) should look like the following:

```
>peptide1
RDLKDLGV
>peptide2
KNSMENGRPPDP
>peptide3
GEELALLRRF
```

and its corresponding HLA list (HLAs.txt):

```
HLA-C*01:02
HLA-A*26:01
HLA-B*07:02
```

note that the number of HLA alleles should match the number of peptides in the fasta file. Then you can safely run main.py with:

```shell
python main.py --peptide_fasta peptides.fa --HLA_file HLAs.txt --num_simulations 10
```

