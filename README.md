# gapless 
Combined scaffolding, gap-closing and assembly correction with long reads

## Table of Contents

- [Abstract](#abstract)
- [Requirements](#requirements)
- [Installation](#installation)
- [Bioconda](#conda)
- [Quick start examples](#quickstart)
- [Parameter](#parameter)
- [Output files](#output)
- [FAQ](#faq)
- [Publication](#publication)

## <a name="abstract"></a>Abstract
Continuity, correctness and completeness of genome assemblies are important for
many biological projects. Long reads represent a major driver towards delivering
high-quality genomes, but not everybody can achieve the necessary coverage for good
long-read-only assemblies. Therefore, improving existing assemblies with low-coverage
long reads is a promising alternative. The improvements include correction, scaffold-
ing and gap filling. However, most tools perform only one of these tasks and the
useful information of reads that supported the scaffolding is lost when running sepa-
rate programs successively. Therefore, we propose a new tool for combined execution
of all three tasks using PacBio or Oxford Nanopore reads.

## <a name="requirements"></a>Requirements

The first list are the python requirements for `gapless.py` and the second list are the external programs called in `gapless.sh`.

| Requirement               | Tested version | Installation instructions                                         |
|---------------------------|----------------|-------------------------------------------------------------------|
| Python 3                  | 3.9.6          |                                                                   |
| biopython                 | 1.77           | https://biopython.org/wiki/Download                               |
| matplotlib                | 3.4.2          | https://matplotlib.org/stable/users/installing.html               |
| numpy                     | 1.21.1         | https://numpy.org/install/                                        |
| pandas                    | 1.3.1          | https://pandas.pydata.org/docs/getting_started/install.html       |
| pillow                    | 8.3.1          | https://pillow.readthedocs.io/en/stable/installation.html         |
| scipy                     | 1.6.3          | https://www.scipy.org/install.html                                |
| seaborn                   | 0.11.1         | https://seaborn.pydata.org/installing.html                        |
|---------------------------|----------------|-------------------------------------------------------------------|
| Linux system              |                |                                                                   |
| minimap2                  | 2.18-r1015     | https://github.com/lh3/minimap2                                   |
| racon                     | v1.4.13        | https://github.com/lbcb-sci/racon                                 |
| seqtk                     | 1.3-r106       | https://github.com/lh3/seqtk                                      |

## <a name="installation"></a>Installation
No installation except for the requirements is necessary. The program can be directly called from its folder after downloading:
```
cd /where/you/want/to/download/
git clone https://github.com/schmeing/gapless.git
ls gapless
```

You may want to add the folder to your PATH variable to be able to call it from everywhere:
```
export PATH=/where/you/downloaded/gapless/:$PATH
```
If you insert this command into `~/.bashrc` it will be automatically called when you login.
 
An alterantive is to create links to these files:
```
ln -s /where/you/downloaded/gapless/gapless.py /where/you/want/to/have/gapless/gapless.py
ln -s /where/you/downloaded/gapless/gapless.sh /where/you/want/to/have/gapless/gapless.sh
```

## <a name="conda"></a>Bioconda
Gapless can also be downloaded with all python requirements in an automatic fashion via anaconda/miniconda(https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html).
However, updates will not be as frequent and the option to switch to the devel branch to get the most recent bugfixes is missing.

```
conda install -c bioconda gapless
```

To add the additional software used in gapless.sh from conda use:

```
conda install -c bioconda minimap2 racon seqtk
```

## <a name="quickstart"></a>Quick start examples
The pipeline can be run with one of the following three commands depending on the type of long reads:
```
gapless.sh -j 30 -i assembly.fa.gz -t pb_clr data_pacbio_clr.fq.gz
gapless.sh -j 30 -i assembly.fa.gz -t pb_hifi data_pacbio_hifi.fq.gz
gapless.sh -j 30 -i assembly.fa.gz -t nanopore data_oxford_nanopore.fq.gz
```

The final output is linked to `gapless_run/gapless.fa`. Depending on the available number of cores you should change the `-j` parameter. 30 threads can finish a human sized genome with 30x coverage with the default 3 iterations in approximately half a day.

The pipeline essentially makes the following calls for each iteration:
```
gapless.py split -o gapless_split.fa {input_assembly}
minimap2 -t {threads} -DP -k19 -w19 -m200 gapless_split.fa gapless_split.fa > gapless_split_repeats.paf
minimap2 -t {threads} -x {read_type} -c -N 5 --secondary=no gapless_split.fa {input_reads} > gapless_reads.paf
gapless.py scaffold -p gapless -s gapless_stats.pdf gapless_split.fa gapless_reads.paf gapless_split_repeats.paf
minimap2 -t {threads} -x {read_type2} <(seqtk subseq {input_reads} gapless_extending_reads.lst) <(seqtk subseq {input_reads} gapless_extending_reads.lst) > gapless_extending_reads.paf
gapless.py extend -p gapless gapless_extending_reads.paf
gapless.py finish -o gapless_raw.fa -H 0 -s gapless_extended_scaffold_paths.csv gapless_split.fa <(seqtk subseq {input_reads} gapless_used_reads.lst)
minimap2 -t {threads} -x {read_type} gapless_raw.fa {input_reads} > gapless_consensus.paf
racon -t {threads} {input_reads} gapless_consensus.paf gapless_raw.fa > gapless.fa
```

`{read_type}` is `map-pb`, `asm20` or `map-ont` depending on the type of long reads.
`{read_type2}` is `ava-pb`, `asm20 --min-occ-floor=0 -X -m100 -g10000 --max-chain-skip 25` or `ava-ont` depending on the type of long reads.

## <a name="parameter"></a>Parameter
`gapless.sh [OPTIONS] {long_reads}.fq`

| Parameter          | Default                | Description |
|--------------------|------------------------|-------------|
| `-h` `-?`          |                        | Display this help and exit |
| `-i`               | (mandatory)            | Input assembly (fasta) |
| `-j`               | 4                      | Number of threads |
| `-n`               | 3                      | Number of iterations |
| `-o`               | gapless_run            | Output directory (improved assembly is written to gapless.fa in this directory) |
| `-r`               |                        | Restart at the start iteration and overwrite instead of incorporat already present files |
| `-s`               | 1                      | Start iteration (Previous runs must be present in output directory) |
| `-t`               | (mandatory)            | Type of long reads (`pb_clr`,`pb_hifi`,`nanopore`) |

`gapless.py split [OPTIONS] {assembly}.fa`

| Parameter          | Default                | Description |
|--------------------|------------------------|-------------|
| `-h` `--help`      |                        | Display this help and exit |
| `-n` `--minN`      | 1                      | Minimum number of N's to split at that position |
| `-o` `--output`    | `{assembly}_split.fa`  | File to which the split sequences should be written to |

`gapless.py scaffold [OPTIONS] {assembly}.fa {mapping}.paf {repeat}.paf`

| Parameter          | Default                 | Description |
|--------------------|-------------------------|-------------|
| `-h` `--help`      |                         | Display this help and exit |
| `-p` `--prefix`    | `{assembly}`            | Prefix for output files |
| `-s` `--stats`     | (no stats)              | Output file for plots with statistics regarding input parameters (pdf) |
| `--minLenBreak`    | 600                     | Minimum length for a read to diverge from a contig to consider a contig break |
| `--minMapLength`   | 400                     | Minimum length of individual mappings of reads |
| `--minMapQ`        | 20                      | Minimum mapping quality of reads |

`gapless.py extend -p {prefix} {all_vs_all}.paf`

| Parameter          | Default                 | Description |
|--------------------|-------------------------|-------------|
| `-h` `--help`      |                         | Display this help and exit |
| `-p` `--prefix`    | (mandatory)             | Prefix for output files of scaffolding step |
| `--minLenBreak`    | 1000                    | Minimum length for two reads to diverge to consider them incompatible for this contig |

`gapless.py finish [OPTIONS] -s {scaffolds}.csv {assembly}.fa {reads}.fq`

| Parameter          | Default                 | Description |
|--------------------|-------------------------|-------------|
| `-h` `--help`      |                         | Display this help and exit |
| `-f` `--format`    | `fastq` or read ending  | Format of `{reads}`.fq (`fasta`/`fastq`) |
| `-H` `--hap`       | mixed                   | Haplotype starting from 0 written to `--output` |
| `--hap[1-9]`       | mixed                   | Haplotypes starting from 0 written to `--out[1-9]` |
| `-o` `--output`    | `{assembly}`_gapless.fa | Output file for modified assembly |
| `--out[1-9]`       | (no additional output)  | Additional output files for modified assembly |
| `-p` `--polishing` | (mandatory)             | Input file for polishing read information |
| `-s` `--scaffolds` | (mandatory)             | Csv file from previous steps describing the scaffolding |

## <a name="output"></a>Intermediate and final output files in the pipeline
| File                                | Program             | Type         | Information                                          |
|-------------------------------------|---------------------|--------------|------------------------------------------------------|
| gapless_split.fa                    | gapless.py split    | temporary    | Input assembly with the scaffolds split into contigs |
| gapless_split_repeats.paf           | minimap2            | temporary    | Mapping of the split assembly to itself              |
| gapless_reads.paf                   | minimap2            | temporary    | Mapping of the long reads to the split assembly      |
| gapless_scaffold_paths.csv          | gapless.py scaffold | intermediate | Table summarising the first and last base included for contigs and reads in the new scaffolds as well as their order and orientation for all haplotypes. Positions start at 0 and end positions are one after the last included position. The orientation is encoded with +/-. The first and last contig/read in a scaffold contain information about the distance to the next scaffold in case this information is present. Otherwise, the fields are encoded with -1. Identical phases mean that contig/reads are phased to be on the same haplotype. Negative phases mark contigs/reads identical to main haplotye (0). Empty contig/read names in combination with positive phases mark deletions. |
| gapless_extensions.csv              | gapless.py scaffold | intermediate | Table summarising the mapping of extending reads to the assembly. How much they extend the new scaffolds, how far from the end they stop aligning (trim), the read distance to the next alignment (unmap_ext). |
| gapless_extending_reads.lst         | gapless.py scaffold | intermediate | List of extending reads with one read name per line  |
| gapless_stats.pdf                   | gapless.py scaffold | final        | File containing plots with information about the run |
| gapless_extending_reads.paf         | minimap2            | temporary    | All-vs.-all alignment of the extending reads         |
| gapless_extended_scaffold_paths.csv | gapless.py extend   | intermediate | Table including the extensions in the same format as gapless_scaffold_paths.csv |
| gapless_used_reads.lst              | gapless.py extend   | intermediate | List of reads inlcuded in gapless_extended_scaffold_paths.csv with one read name per line |
| gapless_raw.fa                      | gapless.py finish   | temporary    | Unpolished output assembly                           |
| gapless_consensus.paf               | minimap2            | temporary    | Mapping of the long reads to the unpolished assembly |
| gapless.fa                          | racon               | final        | Polished output assembly                             |

Intermediate files are required for the following steps in the pipeline, but are not removed when they are not needed anymore (in contrast to temporary files).
Log files are stored in the `logs` folder and the ressource usage acquired with GNU time is written to the `timing` folder.

## <a name="faq"></a>FAQ
Coming soon ...

## <a name="publication"></a>Publication
Schmeing, S., Robinson, M.D. Gapless provides combined scaffolding, gap filling and assembly correction with long reads. bioRxiv (2022). https://doi.org/10.1101/2022.03.08.483466
