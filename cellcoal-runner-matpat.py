#!/usr/bin/env python3
# coding: utf-8
import argparse
import math
import os
import shutil
import subprocess
import sys

import ete3
import numpy as np

parser = argparse.ArgumentParser(
    description="Generate trees with parallel maternal and paternal sequences"
)

parser.add_argument("--ncells", type=int, required=True, help="number of cells in sample")
parser.add_argument("--nsamples", type=int, required=True, help="number of samples")
parser.add_argument("--nsites", type=int, required=True, help="number of sites")
parser.add_argument("--eff_pop", type=int, required=True, help="effective population size")
parser.add_argument("--exp_growth_rate", type=float, required=False, help="exponential growth rate")
parser.add_argument(
    "--birth_rate", type=float, required=False, help="birth rate (Ohtsaki Innan 2017)"
)
parser.add_argument(
    "--death_rate", type=float, required=False, help="death rate (Ohtsaki Innan 2017)"
)
parser.add_argument("--somatic_mut_rate", type=float, required=True, help="somatic mutation rate")
parser.add_argument("--lin_rate_var", type=float, required=False, help="lineage rate variation")
parser.add_argument("--doublet_rate_mean", type=float, default=0.0, help="doublet rate mean")
parser.add_argument("--doublet_rate_var", type=float, default=0.0, help="doublet rate variation")
parser.add_argument("--ado", type=float, required=True, help="Allelic dropout")
parser.add_argument("--amp_err_mean", type=float, required=True, help="Amplification error mean")
parser.add_argument("--amp_err_var", type=float, required=True, help="Amplification error variance")
parser.add_argument("--seq_err", type=float, required=True, help="Sequencing error")
parser.add_argument("--gamma", type=float, required=False, help="Rate var sites Gamma")
parser.add_argument(
    "--base_freqs",
    type=float,
    nargs=4,
    default=[0.293, 0.2, 0.207, 0.3],
    metavar=("pi_a", "pi_c", "pi_g", "pi_t"),
    help="base freqs ACGT",
)
parser.add_argument(
    "--mut_matrix",
    type=float,
    nargs=16,
    default=[
        # fmt: off
        0.00e-3, 0.03e-3, 0.12e-3, 0.04e-3,
        0.11e-3, 0.00e-3, 0.02e-3, 0.68e-3,
        0.68e-3, 0.02e-3, 0.00e-3, 0.11e-3,
        0.04e-3, 0.12e-3, 0.03e-3, 0.00e-3,
        # fmt: on
    ],
    metavar=(
        # fmt: off
        "q_aa", "q_ac", "q_ag", "q_at",
        "q_ca", "q_cc", "q_cg", "q_ct",
        "q_ga", "q_gc", "q_gg", "q_gt",
        "q_ta", "q_tc", "q_tg", "q_tt",
        # fmt: on
    ),
    help="mutation matrix ACGTxACGT (default GTnR)",
)

# version number is the x.y.z part. This makes assumptions about the current working directory.
parser.add_argument(
    "--cellcoal",
    type=str,
    default="../cellcoal-modified/bin/cellcoal-1.3.0",
    help="cellcoal binary location",
)

parser.add_argument("--output", type=str, help="output filename prefix for tree")
parser.add_argument("--log", action="store_true")

if hasattr(sys, "ps1"):
    opt = parser.parse_args(
        "--ncells 10 "
        "--nsamples 10 "
        "--nsites 1000 "
        "--ado 0 "
        "--amp_err_mean 0 "
        "--amp_err_var 0 "
        "--seq_err 0 "
        "--log".split()
    )
else:
    opt = parser.parse_args()

if opt.log:
    print(opt)

NCELLS_SAMPLE = opt.ncells
NUM_SAMPLES = opt.nsamples
NUM_SITES = opt.nsites
ALLELIC_DROPOUT = opt.ado
AMPLIFICATION_ERROR_MEAN = opt.amp_err_mean
AMPLIFICATION_ERROR_VARIANCE = opt.amp_err_var
SEQUENCING_ERROR = opt.seq_err
RATE_VAR_SITES_GAMMA_PARAM = (
    opt.gamma if hasattr(opt, "gamma") and opt.gamma is not None else float("-inf")
)
NUC_BASE_FREQ = opt.base_freqs
MUT_MATRIX = np.array(opt.mut_matrix, dtype=np.float64).reshape(4, 4)
EFFECTIVE_POP_SIZE = opt.eff_pop
EXPONENTIAL_GROWTH_RATE = (
    opt.exp_growth_rate
    if hasattr(opt, "exp_growth_rate") and opt.exp_growth_rate is not None
    else float("-inf")
)
BIRTH_RATE = (
    opt.birth_rate if hasattr(opt, "birth_rate") and opt.birth_rate is not None else float("-inf")
)
DEATH_RATE = (
    opt.death_rate if hasattr(opt, "death_rate") and opt.death_rate is not None else float("-inf")
)
SOMATIC_MUT_RATE = opt.somatic_mut_rate
LINEAGE_RATE_VARIATION = (
    opt.lin_rate_var
    if hasattr(opt, "lin_rate_var") and opt.lin_rate_var is not None
    else float("-inf")
)
DOUBLET_RATE_MEAN = opt.doublet_rate_mean
DOUBLET_RATE_VAR = opt.doublet_rate_var

CELLCOAL_BIN = opt.cellcoal

DEBUG = opt.log

# # ## Cellcoal parameters
# #
# # See https://dapogon.github.io/cellcoal/cellcoal.manual.v1.1.html#63_mutation_models for parameters
# #

NUM_REPLICATES = 1
ALPHABET_DNA = True
# TRANSFORMING_BRANCH_LENGTH -k
# GERMLINE_SNP_RATE -c
# DELETION_RATE = 0.0000001 infinite sites model only
# COPY_NEUTRAL_LOH = 0.0
# FIXED_NUM_MUT -j
# TRANSITION_TRANSVERSION_RATIO = 0.5
FLAT_MUT_MATRIX = np.reshape(MUT_MATRIX, -1)
# ALLELIC_DROPOUT_SITES = 1 # ???
# ALLELIC_DROPOUT_CELLS = 1 # ???
AMPLIFICATION_ERROR = [
    AMPLIFICATION_ERROR_MEAN,
    AMPLIFICATION_ERROR_VARIANCE,
    0,
]  # (mean of beta-binomial, variance, 2/4 template model type) ****
# GENOTYPING_ERROR_MEAN = 0.1
# GENOTYPING_ERROR_VAR = 0.01
# ???: ERROR_MATRIX= [[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]
# SEED = 42
#


DELETE_VCF_FILES = True


# reference for cell coal command line parameters:
# https://dapogon.github.io/cellcoal/cellcoal.manual.v1.1.html#67_parameter_file_name_at_the_command_line
# Note the version in use is not the 1.1 version, but the 1.3 version with minor modifications.
def params(user_genome_filename):
    params = [
        CELLCOAL_BIN,
        "-s" + str(NCELLS_SAMPLE),
        "-n" + str(NUM_REPLICATES),
        "-l" + str(NUM_SITES),
        "-b" + str(int(ALPHABET_DNA)),
        "-e" + str(EFFECTIVE_POP_SIZE),
        "-u" + str(SOMATIC_MUT_RATE),  # somatic mutation rate
        # "-d" + str(DELETION_RATE),
        # "-H" + str(COPY_NEUTRAL_LOH),
        "-m2",
        "-p1.0",  # 100% finite DNA model
        "-f" + str(NUC_BASE_FREQ[0]),  # nucleotide base frequencies
        *[str(freq) for freq in NUC_BASE_FREQ[1:]],  # nucleotide base frequencies (rest of them)
        # "-t" + str(TRANSITION_TRANSVERSION_RATIO),
        "-r" + str(FLAT_MUT_MATRIX[0]),
        *[str(entry) for entry in FLAT_MUT_MATRIX[1:]],
        "-D" + str(ALLELIC_DROPOUT),
        # "-P" + str(ALLELIC_DROPOUT_SITES),
        # "-Q" + str(ALLELIC_DROPOUT_CELLS),
        # "-G" + str(GENOTYPING_ERROR_MEAN), str(GENOTYPING_ERROR_VAR),
        "-E" + str(SEQUENCING_ERROR),
        "-1",  # print SNV genotypes to a file
        "-2",  # print SNV haplotypes to a file
        "-3",  # print full genotypes to a file
        "-4",  # print full haplotypes to a file
        "-5",  # print ancestral genotypes
        "-6",  # print trees to a file
        # "-7", # print times to a file
        # "-8", # print read counts in CATG format <- error
        # "-9", # print read counts in PILEUP format <- error
        # "-oRun" + ident, # comment this out so that it goes in the "Results" folder
        "-v",  # separate files (print replicates in individual folders)
        "-Y",  # print true haplotypes to file
        "-x",  # print consensus/IUPAC haplotypes
        # "-T" + nwk_filename,  # omit this and cellcoal generates a random coalescent tree.
        "-U" + user_genome_filename,
        # "-#" + str(SEED),
        "-y0",
    ]
    if DOUBLET_RATE_MEAN > 0.0:
        params += [
            "-B" + str(DOUBLET_RATE_MEAN),
            str(DOUBLET_RATE_VAR),
        ]
    if AMPLIFICATION_ERROR[0] > 0.0:
        params += [
            "-A" + str(AMPLIFICATION_ERROR[0]),
            *[str(c) for c in AMPLIFICATION_ERROR[1:]],  # value giving error
        ]
    if EXPONENTIAL_GROWTH_RATE > 0.0:
        params += ["-g" + str(EXPONENTIAL_GROWTH_RATE)]
    if BIRTH_RATE > 0.0:
        params += ["-K" + str(BIRTH_RATE)]
    if DEATH_RATE > 0.0:
        params += ["-L" + str(DEATH_RATE)]
    if RATE_VAR_SITES_GAMMA_PARAM > 0.0:
        params += ["-a" + str(RATE_VAR_SITES_GAMMA_PARAM)]
    if LINEAGE_RATE_VARIATION > 0.0:
        params += ["-i" + str(LINEAGE_RATE_VARIATION)]

    return params


# | Symbol | Description               | | | | |Complement |
# |:------:|:-------------------------:|-|-|-|-|----------:|
# | A      | Adenine                   |A| | | |          T|
# | C      | Cytosine                  | |C| | |          G|
# | G      | Guanine                   | | |G| |          C|
# | T      | Thymine                   | | | |T|          A|
# | U      | Uracil                    | | | |U|          A|
# | W      | Weak                      |A| | |T|          W|
# | S      | Strong                    | |C|G| |          S|
# | M      | aMino                     |A|C| | |          K|
# | K      | Keto                      | | |G|T|          M|
# | R      | puRine                    |A| |G| |          Y|
# | Y      | pYrimidine                | |C| |T|          R|
# | B      | not A                     | |C|G|T|          V|
# | D      | not C                     |A| |G|T|          H|
# | H      | not G                     |A|C| |T|          D|
# | V      | not T                     |A|C|G| |          B|
# | N      | any Nucleotide (not a gap)|A|C|G|T|          N|
# | Z      | Zero                      | | | | |          Z|
nucleotide_symbol_table = {
    frozenset({"."}): "?",  # Gap symbol?
    frozenset(): "Z",  # Is this also a gap, or just not a read?
    frozenset({"A"}): "A",
    frozenset({"C"}): "C",
    frozenset({"G"}): "G",
    frozenset({"T"}): "T",
    frozenset({"U"}): "U",
    frozenset({"N"}): "N",
    frozenset({"A", "T"}): "W",
    frozenset({"A", "U"}): "W",
    frozenset({"C", "G"}): "S",
    frozenset({"A", "C"}): "M",
    frozenset({"G", "T"}): "K",
    frozenset({"G", "U"}): "K",
    frozenset({"A", "G"}): "R",
    frozenset({"C", "T"}): "Y",
    frozenset({"C", "U"}): "Y",
    frozenset({"C", "G", "T"}): "B",
    frozenset({"C", "G", "U"}): "B",
    frozenset({"A", "G", "T"}): "D",
    frozenset({"A", "G", "U"}): "D",
    frozenset({"A", "C", "T"}): "H",
    frozenset({"A", "C", "U"}): "H",
    frozenset({"A", "C", "G"}): "V",
    frozenset({"A", "C", "G", "T"}): "N",
    frozenset({"A", "C", "G", "U"}): "N",
}


def nuc_set_to_letter(s) -> str:
    """
    translate a set of nucleotides to its IUPAC letter
    """
    global nucleotide_symbol_table
    if "?" in s:
        return "?"
    fs = frozenset(s)
    if fs in nucleotide_symbol_table:
        return nucleotide_symbol_table[fs]
    else:
        raise RuntimeError(f"{s} an invalid nucleotide set")


def to_10_state(st) -> str:
    if "?" in st:
        return "?"
    if "N" in st:
        return "N"  # we might want to deal with 'AN', etc. eventually, but it doesn't fit into the alphabet.
    site = set(st)
    return nuc_set_to_letter(site)


def to_3_state(st, ref):
    if "N" in st or "." in st or "?" in st:
        return "?"  # undetermined, give gap symbol
    if st[0] != st[1]:  # all heterozygous are 1
        return 1
    alt_count = 0
    if st[0] != ref:
        alt_count += 1
    if st[1] != ref:
        alt_count += 1
    return alt_count


def to_3_state_alt(st, ref):
    if "N" in st or "." in st or "?" in st:
        return "?"  # undetermined, give gap symbol
    alt_count = 0
    if st[0] != ref:
        alt_count += 1
    if st[1] != ref:
        alt_count += 1
    return alt_count


def to_2_state(st, ref):
    if "N" in st or "." in st or "?" in st:
        return "?"  # undetermined, give gap symbol
    elif st[0] == ref and st[1] == ref:
        return 0
    else:
        return 1


def generate_matpat():
    """
    Generate a maternal and paternal genotype
    :return:
    """
    translation = {0: "A", 1: "C", 2: "G", 3: "T"}
    cdf = np.cumsum(NUC_BASE_FREQ)
    cdf[-1] = 1.0  # careful
    mat_genome = "".join(
        [translation[int(np.argmax(r < cdf))] for r in np.random.random(NUM_SITES)]
    )
    pat_genome = "".join(
        [translation[int(np.argmax(r < cdf))] for r in np.random.random(NUM_SITES)]
    )
    return mat_genome, pat_genome


nexus_template = """#NEXUS

BEGIN DATA;
    dimensions ntax={NTAX} nchar={NCHAR};
    format datatype={DATATYPE} missing=? {SYMBOLS};
    matrix
{SITES}
    ;
END;
"""

fasta_template = """> maternal
{mat_genome}
> paternal
{pat_genome}
"""

fill_width = math.ceil(np.log10(NUM_SAMPLES + 1))

for biopsy_number in range(NUM_SAMPLES):
    # `FILENAME_PREFIX` a prefix for all generated files
    FILENAME_PREFIX = f"tree-{str(biopsy_number).zfill(fill_width)}"

    print(f"Sample {biopsy_number} started", end=" ... ")
    newick_filename = f"{os.getcwd()}/{FILENAME_PREFIX}.nwk"

    # create the ancestral genome
    mat_genome, pat_genome = generate_matpat()
    fasta_filename = f"{os.getcwd()}/{FILENAME_PREFIX}-ancestral.fasta"
    with open(fasta_filename, "w") as file:
        file.write(fasta_template.format(mat_genome=mat_genome, pat_genome=pat_genome))

    # run cellcoal
    result = subprocess.run(params(fasta_filename), capture_output=True)
    log = result.stdout.decode("utf-8") + result.stderr.decode("utf-8")
    if result.returncode > 0:
        print("Log: '" + log + "'")
        sys.exit(result.returncode)

    if len(log) > 0:
        print("Log: '" + log + "'")
    else:
        if DELETE_VCF_FILES:
            print("removing vcf files", end=" ... ")
            try:
                shutil.rmtree(os.getcwd() + f"/Results/vcf_dir")
            except FileNotFoundError as e:
                print(e)
                pass
        print("complete.")

    NEXUS_DIR = f"{os.getcwd()}/{FILENAME_PREFIX}-nexus-files"

    if os.path.isdir(NEXUS_DIR) or os.path.isfile(NEXUS_DIR):
        try:
            shutil.rmtree(NEXUS_DIR)
        except Exception as e:
            print(e)

    os.mkdir(NEXUS_DIR)

    # get ancestral genotype:
    # out_root_maternal = None
    # out_root_paternal = None
    # in_root_maternal = None
    # in_root_paternal = None
    out_root_unphased = None
    in_root_unphased = None
    with open(f"{os.getcwd()}/Results/true_haplotypes_dir/true_hap.0001") as file:
        for line in file:
            # cellcoal-1.3:
            if line[:8].strip() == "outgroot":
                out_root_unphased = line[8:].strip()
            elif line[:8] == "ingroot":
                in_root_unphased = line[8:].strip()
            # # cellcoal-1.1:
            # if line[:9] == "outgrootm":
            #     out_root_maternal = line[9:].strip()
            # elif line[:9] == "outgrootp":
            #     out_root_paternal = line[9:].strip()
            # elif line[:9] == "ingrrootm":
            #     in_root_maternal = line[9:].strip()
            # elif line[:9] == "ingrrootp":
            #     in_root_paternal = line[9:].strip()

    # cellcoal-1.1:
    # out_root_unphased = "".join(
    #     [
    #         nuc_set_to_letter({mat, pat})
    #         for mat, pat in zip(out_root_maternal, out_root_paternal)
    #     ]
    # )
    # in_root_unphased = "".join(
    #     [
    #         nuc_set_to_letter({mat, pat})
    #         for mat, pat in zip(in_root_maternal, in_root_paternal)
    #     ]
    # )

    reference = out_root_unphased

    # # get observed snv genotypes:
    # observed_sites_10_state_snv = dict()
    # observed_sites_3_state_snv = dict()
    # observed_sites_2_state_snv = dict()
    # with open(os.getcwd() + "/Results/snv_genotypes_dir/snv_gen.0001") as file:
    #     cell_count, num_sites = map(int, next(file).split())

    #     # get locations of snvs and filter reference, for use with 3 and 2 state models
    #     snv_locations = list(map(int, next(file).split()))
    #     snv_reference = [reference[loc - 1] for loc in snv_locations]

    #     for line in file:
    #         cell_name, *genes = line.split()
    #         observed_sites_10_state_snv[cell_name] = "".join(
    #             [to_10_state(site) for site in genes]
    #         )
    #         observed_sites_3_state_snv[cell_name] = "".join(
    #             map(
    #                 str,
    #                 [
    #                     to_3_state(site, ref_site)
    #                     for site, ref_site in zip(genes, snv_reference)
    #                 ],
    #             )
    #         )
    #         observed_sites_2_state_snv[cell_name] = "".join(
    #             map(
    #                 str,
    #                 [
    #                     to_2_state(site, ref_site)
    #                     for site, ref_site in zip(genes, snv_reference)
    #                 ],
    #             )
    #         )

    # get observed full genotypes:
    observed_sites_16_state = dict()
    observed_sites_10_state = dict()
    observed_sites_3_state = dict()
    observed_sites_3_state_alt = dict()
    observed_sites_2_state = dict()
    with open(os.getcwd() + f"/Results/full_genotypes_dir/full_gen.0001") as file:
        cell_count, num_sites = map(int, next(file).split())
        for line in file:
            cell_name, *genes = line.split()
            observed_sites_16_state[cell_name] = " ".join(genes)
            observed_sites_10_state[cell_name] = "".join(to_10_state(site) for site in genes)
            observed_sites_3_state[cell_name] = "".join(
                map(
                    str,
                    [to_3_state(site, ref_site) for site, ref_site in zip(genes, reference)],
                )
            )
            observed_sites_3_state_alt[cell_name] = "".join(
                map(
                    str,
                    [to_3_state_alt(site, ref_site) for site, ref_site in zip(genes, reference)],
                )
            )
            observed_sites_2_state[cell_name] = "".join(
                map(
                    str,
                    [to_2_state(site, ref_site) for site, ref_site in zip(genes, reference)],
                )
            )

    # copy the tree file
    shutil.copyfile(os.getcwd() + "/Results/trees_dir/trees.0001", NEXUS_DIR + "/tree-outgcell.nwk")

    # make a tree file without the outgroup
    with open(NEXUS_DIR + "/tree-no-outgcell.nwk", "w") as file:
        tree = ete3.Tree(NEXUS_DIR + "/tree-outgcell.nwk")
        tree.prune(
            leaf for leaf in map(lambda lf: lf.name, tree.iter_leaves()) if leaf != "outgcell"
        )
        file.write(tree.write(format=5))
        file.write("\n")

    # save the Results folder
    shutil.move(os.getcwd() + "/Results", NEXUS_DIR)

    # write nexus for genotypes
    with open(NEXUS_DIR + f"/16state.nex", "w") as file:
        file.write(
            nexus_template.format(
                NTAX=NCELLS_SAMPLE + 1,
                NCHAR=NUM_SITES,
                SYMBOLS="",
                DATATYPE="dna",
                SITES="\n".join(
                    "    " + cell_name.ljust(9) + " " + sites
                    for cell_name, sites in observed_sites_16_state.items()
                    if cell_name != "outgroot" and cell_name != "ingrroot"
                ),
            )
        )
    with open(NEXUS_DIR + f"/16state-nooutgcell.nex", "w") as file:
        file.write(
            nexus_template.format(
                NTAX=NCELLS_SAMPLE,
                NCHAR=NUM_SITES,
                SYMBOLS="",
                DATATYPE="dna",
                SITES="\n".join(
                    "    " + cell_name.ljust(9) + " " + sites
                    for cell_name, sites in observed_sites_16_state.items()
                    if cell_name != "outgcell"
                    and cell_name != "outgroot"
                    and cell_name != "ingrroot"
                ),
            )
        )

    print("16state", end=" ")

    with open(NEXUS_DIR + f"/10state.nex", "w") as file:
        file.write(
            nexus_template.format(
                NTAX=NCELLS_SAMPLE + 1,
                NCHAR=NUM_SITES,
                SYMBOLS="",
                DATATYPE="dna",
                SITES="\n".join(
                    "    " + cell_name.ljust(9) + " " + sites
                    for cell_name, sites in observed_sites_10_state.items()
                    if cell_name != "outgroot" and cell_name != "ingrroot"
                ),
            )
        )

    with open(NEXUS_DIR + f"/10state-nooutgcell.nex", "w") as file:
        file.write(
            nexus_template.format(
                NTAX=NCELLS_SAMPLE,
                NCHAR=NUM_SITES,
                SYMBOLS="",
                DATATYPE="dna",
                SITES="\n".join(
                    "    " + cell_name.ljust(9) + " " + sites
                    for cell_name, sites in observed_sites_10_state.items()
                    if cell_name != "outgcell"
                    and cell_name != "outgroot"
                    and cell_name != "ingrroot"
                ),
            )
        )

    print("10state", end=" ")

    with open(NEXUS_DIR + f"/3state.nex", "w") as file:
        file.write(
            nexus_template.format(
                NTAX=NCELLS_SAMPLE + 1,
                NCHAR=NUM_SITES,
                SYMBOLS='symbols="0 1 2"',
                DATATYPE="standard",
                SITES="\n".join(
                    "    " + cell_name.ljust(9) + " " + sites
                    for cell_name, sites in observed_sites_3_state.items()
                    if cell_name != "outgroot" and cell_name != "ingrroot"
                ),
            )
        )

    with open(NEXUS_DIR + f"/3state-nooutgcell.nex", "w") as file:
        file.write(
            nexus_template.format(
                NTAX=NCELLS_SAMPLE,
                NCHAR=NUM_SITES,
                SYMBOLS='symbols="0 1 2"',
                DATATYPE="standard",
                SITES="\n".join(
                    "    " + cell_name.ljust(9) + " " + sites
                    for cell_name, sites in observed_sites_3_state.items()
                    if cell_name != "outgcell"
                    and cell_name != "outgroot"
                    and cell_name != "ingrroot"
                ),
            )
        )

    print("3state", end=" ")

    with open(NEXUS_DIR + f"/3state_alt.nex", "w") as file:
        file.write(
            nexus_template.format(
                NTAX=NCELLS_SAMPLE + 1,
                NCHAR=NUM_SITES,
                SYMBOLS='symbols="0 1 2"',
                DATATYPE="standard",
                SITES="\n".join(
                    "    " + cell_name.ljust(9) + " " + sites
                    for cell_name, sites in observed_sites_3_state_alt.items()
                    if cell_name != "outgroot" and cell_name != "ingrroot"
                ),
            )
        )

    with open(NEXUS_DIR + f"/3state_alt-nooutgcell.nex", "w") as file:
        file.write(
            nexus_template.format(
                NTAX=NCELLS_SAMPLE,
                NCHAR=NUM_SITES,
                SYMBOLS='symbols="0 1 2"',
                DATATYPE="standard",
                SITES="\n".join(
                    "    " + cell_name.ljust(9) + " " + sites
                    for cell_name, sites in observed_sites_3_state_alt.items()
                    if cell_name != "outgcell"
                    and cell_name != "outgroot"
                    and cell_name != "ingrroot"
                ),
            )
        )

    print("3state_alt", end=" ")

    with open(NEXUS_DIR + f"/2state.nex", "w") as file:
        file.write(
            nexus_template.format(
                NTAX=NCELLS_SAMPLE + 1,
                NCHAR=NUM_SITES,
                SYMBOLS='symbols="0 1"',
                DATATYPE="standard",
                SITES="\n".join(
                    "    " + cell_name.ljust(9) + " " + sites
                    for cell_name, sites in observed_sites_2_state.items()
                    if cell_name != "outgroot" and cell_name != "ingrroot"
                ),
            )
        )

    with open(NEXUS_DIR + f"/2state-nooutgcell.nex", "w") as file:
        file.write(
            nexus_template.format(
                NTAX=NCELLS_SAMPLE,
                NCHAR=NUM_SITES,
                SYMBOLS='symbols="0 1"',
                DATATYPE="standard",
                SITES="\n".join(
                    "    " + cell_name.ljust(9) + " " + sites
                    for cell_name, sites in observed_sites_2_state.items()
                    if cell_name != "outgcell"
                    and cell_name != "outgroot"
                    and cell_name != "ingrroot"
                ),
            )
        )

    print("2state")
