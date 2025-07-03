#!/bin/bash

usage() {
  cat << EOF
Usage: bash scripts/$(basename "$0") <input_file> <output_directory> <database_file> <cpu_cores>

Description:
  This script using BLASTN and Infernal to search for MSA from a FASTA file.

Arguments:
  <input_file>        Required. Path to the input file (sequence in FASTA format).
  <output_directory>  Required. Directory where the output results will be saved.
                      The directory will be created if it doesn't exist.
  <database_file>     Required. Path to the database file (rnacentral_xx.fasta).
  <cpu_cores>         Required. Number of CPU cores to utilize for the computation.

Options:
  -h, --help          Display this help message and exit.

Example:
  bash scripts/$(basename "$0") example/seq.fasta example/msa/ library/rnacentral_99_rep_seq.fasta 4
EOF
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

if [ "$#" -ne 4 ]; then
  echo "Error: Incorrect number of arguments provided." >&2
  echo "Expected 4 arguments, but got $#." >&2
  usage
  exit 1
fi

start=`date +%s`

input=$1
output_dir=$2
db_pth=$3
cpu=$4
seq_id='seq'

if [ ! -f "$input" ]; then
  echo "Error: Input file '$input' not found." >&2
  exit 1
fi
if [ ! -f "$db_pth" ]; then
  echo "Error: Database file '$db_pth' not found." >&2
  exit 1
fi

if ! [[ "$cpu" =~ ^[1-9][0-9]*$ ]]; then
   echo "Error: CPU cores '$cpu' must be a positive integer." >&2
   exit 1
fi

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

path_blastn_database=$db_pth
path_infernal_database=$db_pth

source_dir=$script_dir/bin

chmod +x $script_dir/bin/*
chmod +x $script_dir/utils/*

mkdir -p $output_dir
echo ">"$seq_id > $output_dir/$seq_id.fasta
tail -n1 $input >> $output_dir/$seq_id.fasta
echo "" >> $output_dir/$seq_id.fasta


###### check if aligned homologous sequences file already exists ############
if [ -f $output_dir/$seq_id.a3m ];	then
        echo ""
        echo "======================================================================"
        echo "    MSA file $output_dir/$seq_id.a3m from Infernal Pipeline already  "
        echo "    exists for query sequence $output_dir/$seq_id.fasta.             "
        echo "                                                                      "
        echo "    Delete existing $output_dir/$seq_id.a3m if want to generate new  "
        echo "    alignment file                                                    "
        echo "======================================================================"
    	echo ""
else
    #################### check if blastn database file ready exists ######################
    if [ -f $path_blastn_database.ndb ];       then
        echo ""
        echo "==========================================================================================================================="
        echo "        BLASTN database file $path_blastn_database.ndb already exists      "
        echo "==========================================================================================================================="
        echo ""
    else
        echo ""
        echo "==========================================================================================================================="
        echo "        BLASTN database file $path_blastn_database.ndb does not exist .          "
        echo "        Running MAKEBLASTDB to generate BLASTN databse for $path_blastn_database.          "
        echo "        May take 5-10 mins.                                                                "
        echo "==========================================================================================================================="
        echo ""
        $source_dir/makeblastdb -in $path_blastn_database -dbtype nucl
    fi

    #################### check if blastn alignment file ready exists ######################
    if [ -f $output_dir/$seq_id.bla ];       then
        echo ""
        echo "======================================================================="
        echo "    MSA-1 file $output_dir/$seq_id.bla from Infernal Pipeline already "
        echo "    exists for query sequence $output_dir/$seq_id.fasta.              "
        echo "                                                                       "
        echo "    Delete existing $output_dir/$seq_id.a3m if want to generate new   "
        echo "    alignment file.                                                    "
        echo "======================================================================="
        echo ""
    else
        echo ""
        echo "==========================================================================================================================="
        echo "      Running BLASTN for first round of homologous sequence search for query sequence $output_dir/$seq_id.fasta.          "
        echo "      May take 5 mins to few hours depending on sequence length and no. of homologous sequences in database.               "
        echo "==========================================================================================================================="
        echo ""
        $source_dir/blastn -db $path_blastn_database -query $output_dir/$seq_id.fasta -out $output_dir/$seq_id.bla -evalue 0.001 -num_descriptions 1 -num_threads $cpu -line_length 1000 -num_alignments 50000
    fi

    if [ $? -eq 0 ]; then
        echo ""
        echo "==========================================================="
          echo "      First round of MSA-1 search completed successfully.  "
        echo "==========================================================="
        echo ""
    else
          echo ""
          echo "=================================================================="
          echo "        Error occured while formatting the nt database.           "
          echo ""
          echo "  Please raise issue at 'https://github.com/quailwwk/trRosettaRNA2/issues'"
          echo "=================================================================="
          echo ""
          exit 1
    fi

    ######## reformat the output ################
    echo ""
    echo "========================================================================================"
    echo "         Converting $output_dir/$seq_id.bla from BLASTN to $output_dir/$seq_id.sto.   "
    echo "========================================================================================"
    echo ""
    perl $script_dir/utils/parse_blastn_local.pl $output_dir/$seq_id.bla $output_dir/$seq_id.fasta $output_dir/$seq_id.aln
    perl $script_dir/utils/reformat.pl fas sto $output_dir/$seq_id.aln $output_dir/$seq_id.sto


    if [ $? -eq 0 ]; then
          echo ""
          echo "=========================================="
          echo "      Converison completed successfully.  "
          echo "=========================================="
          echo ""
    else
          echo ""
          echo "============================================================================================="
          echo "   Error occured while Converting $output_dir/$seq_id.bla to $output_dir/$seq_id.sto       "
          echo " "
          echo "   Please raise issue at 'https://github.com/quailwwk/trRosettaRNA2/issues'             "
          echo "============================================================================================="
          echo ""
          exit 1
    fi

    ######## predict secondary structure from RNAfold ################
    echo ""
    echo "==============================================================================================================================="
    echo "       Predicting Consensus Secondary Structure (CSS) of query sequence $output_dir/$seq_id.fasta using RNAfold predictor.   "
    echo "==============================================================================================================================="
    echo ""

    $source_dir/RNAfold $output_dir/$seq_id.fasta | awk '{print $1}' | tail -n +3 > $output_dir/$seq_id.db

    ################ reformat ss with according to gaps in reference sequence of .sto file from blastn ################
    for i in `awk '{print $2}' $output_dir/$seq_id.sto | head -n5 | tail -n1 | grep -b -o - | sed 's/..$//'`; do sed -i "s/./&-/$i" $output_dir/$seq_id.db; done

    #########  add reformated ss from last step to .sto file of blastn ##############
    head -n -1 $output_dir/$seq_id.sto > $output_dir/temp.sto
    echo "#=GC SS_cons                     "`cat $output_dir/$seq_id.db` > $output_dir/temp.txt
    cat $output_dir/temp.sto $output_dir/temp.txt > $output_dir/$seq_id.sto
    echo "//" >> $output_dir/$seq_id.sto

    if [ $? -eq 0 ]; then
        echo ""
        echo "=================================================================="
        echo "      Consensus Secondary Structure (CSS) generated successfully. "
        echo "=================================================================="
        echo ""
    else
        echo ""
        echo "=============================================================================="
        echo "             Error occured while generating structure from RNAfold.          "
        echo " "
        echo "  Please raise issue at 'https://github.com/quailwwk/trRosettaRNA2/issues'"
        echo "=============================================================================="
        echo ""
        exit 1
    fi

    ######## run infernal ################
    echo ""
    echo "=============================================================================================================="
    echo "      Building Covariance Model from BLASTN alignment from $output_dir/$seq_id.sto file.         "
    echo "=============================================================================================================="
    echo ""
    $source_dir/cmbuild --hand -F $output_dir/$seq_id.cm $output_dir/$seq_id.sto

    if [ $? -eq 0 ]; then
        echo ""
        echo "============================================================================"
        echo "    Covariance Model (CM) built successfully from $output_dir/$seq_id.sto. "
        echo "============================================================================"
        echo ""
    else
        echo ""
        echo "==============================================================================================="
        echo "     Error occured while building Covariance Model (CM) from cmbuild.           "
        echo " "
        echo "  Please raise issue at 'https://github.com/quailwwk/trRosettaRNA2/issues'"
        echo "==============================================================================================="
        echo ""
        exit 1
    fi

    echo ""
    echo "===================================================================="
    echo "       Calibrating the Covariance Model $output_dir/$seq_id.cm.    "
    echo "===================================================================="
    echo ""
    $source_dir/cmcalibrate --cpu $cpu $output_dir/$seq_id.cm

    if [ $? -eq 0 ]; then
        echo ""
        echo "==========================================================="
        echo "    CM calibrated $output_dir/$seq_id.cm successfully.    "
        echo "==========================================================="
        echo ""
    else
        echo ""
        echo "==============================================================="
        echo "     Error occured while calibrating $output_dir/$seq_id.cm.  "
        echo " "
        echo "  Please raise issue at 'https://github.com/quailwwk/trRosettaRNA2/issues'"
        echo "==============================================================="
        echo ""
        exit 1
    fi

    echo ""
    echo "======================================================================================================================"
    echo "        Second round of homologous sequences search using the calibrated covariance model $output_dir/$seq_id.cm.    "
    echo "                 May take 15 mins to few hours for this step.                                                         "
    echo "======================================================================================================================"
    echo ""
    $source_dir/cmsearch -o $output_dir/$seq_id.out -A $output_dir/$seq_id.msa --cpu $cpu --incE 0.01 $output_dir/$seq_id.cm $path_infernal_database

    if [ $? -eq 0 ]; then
        echo ""
        echo "==========================================================="
        echo "      Second round of MSA-2 search completed successfully.  "
        echo "==========================================================="
        echo ""
    else
        echo ""
        echo "===================================================================================="
        echo "     Error occured during the second round search using CM $output_dir/$seq_id.cm. "
        echo " "
        echo "  Please raise issue at 'https://github.com/quailwwk/trRosettaRNA2/issues'"
        echo "===================================================================================="
        echo ""
        exit 1
    fi

    ######### reformat the alignment without gaps and dashes  ###############
    echo ""
    echo "======================================================================="
    echo "          Reformatting the output alignment $output_dir/$seq_id.msa   "
    echo "          for PSSM and DCA features by removing the gaps and dashes.   "
    echo "======================================================================="
    echo ""
    ##### check if .msa	is not empty  #########
    if [[ -s $output_dir/$seq_id.msa ]]
      then
      $source_dir/esl-reformat --replace acgturyswkmbdhvn:................ a2m $output_dir/$seq_id.msa > $output_dir/temp.a2m
    else
      cat $output_dir/$seq_id.fasta > $output_dir/temp.a2m
      cat $output_dir/$seq_id.fasta >> $output_dir/temp.a2m
      sed -i '$ s/.$/./' $output_dir/temp.a2m
    fi

    if [ $? -eq 0 ]; then
        echo ""
        echo "==========================================================="
        echo "   Reformatted the $output_dir/$seq_id.msa successfully.  "
        echo "==========================================================="
        echo ""
    else
        echo ""
        echo "========================================================================================"
        echo "     Error occured during the refomatting the alignment file $output_dir/$seq_id.msa.  "
        echo " "
        echo "  Please raise issue at 'https://github.com/quailwwk/trRosettaRNA2/issues'"
        echo "========================================================================================"
        echo ""
        exit 1
    fi

    ######### remove duplicates sequences from the alignment ###############
    echo ""
    echo "======================================================================="
    echo "          Removing duplicates from the alignment.                      "
    echo "======================================================================="
    echo ""
    $script_dir/utils/seqkit rmdup -s $output_dir/temp.a2m > $output_dir/$seq_id.a3m

    if [ $? -eq 0 ]; then
        echo ""
        echo "==============================================="
        echo "   Duplicate sequences removed successfully.   "
        echo "==============================================="
        echo ""
    else
        echo ""
        echo "========================================================================================"
        echo "     Error occured during the removel of duplicates from MSA-2.  "
        echo " "
        echo "  Please raise issue at 'https://github.com/quailwwk/trRosettaRNA2/issues'"
        echo "========================================================================================"
        echo ""
        exit 1
    fi

    ############# multiline fasta to single line fasta file   #############
    awk '/^>/ {printf("\n%s\n",$0);next; } { printf("%s",$0);}  END {printf("\n");}' < $output_dir/$seq_id.a3m | sed '/^$/d' > $output_dir/temp.a2m
    ############# add query sequence at the top of MSA file  #############
    cat $output_dir/$seq_id.fasta $output_dir/temp.a2m > $output_dir/$seq_id.a3m
fi

end=`date +%s`

runtime=$((end-start))

echo -e "\nMSA file saved to $output_dir/$seq_id.a3m"
echo -e "\ncomputation time = "$runtime" seconds\n"

