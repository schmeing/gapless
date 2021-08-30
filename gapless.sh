#!/bin/bash
set -uo pipefail

# Declare functions
function echo_usage {
	echo "Usage: gapless.sh [OPTIONS] {long_reads}.fq"
    echo "Improves input assembly with reads in {long_reads}.fq using gapless, minimap2, racon and seqtk"
    echo "  -h -?                    Display this help and exit"
    echo "  -i [STRING]              Input assembly (fasta)"
    echo "  -j [INT]                 Number of threads [4]"
    echo "  -n [INT]                 Number of iterations [3]"
    echo "  -o [STRING]              Output directory (improved assembly is written to gapless.fa in this directory) [gapless_run]"
    echo "  -r                       Restart at the start iteration and overwrite instead of incorporat already present files"
    echo "  -s [INT]                 Start iteration (Previous runs must be present in output directory) [1]"
    echo "  -t [STRING]              Type of long reads ('pb_clr','pb_hifi','nanopore')"
}

# Reading in arguments
OPTIND=1         # Reset in case getopts has been used previously in the shell.

reads=""
asm=""
threads=4
iterations=3
output="gapless_run"
reset=false
start=1
type=""

while getopts "h?i:j:n:o:rs:t:" opt; do
  case "$opt" in
  h|\?)
	echo_usage
    exit 0
    ;;
  i) asm=$OPTARG
    ;;
  j) threads=$OPTARG
    ;;
  n) iterations=$OPTARG
    ;;
  o) output=$OPTARG
    ;;
  r) reset=true
    ;;
  s) start=$OPTARG
    ;;
  t) type=$OPTARG
    ;;
  esac
done

shift $((OPTIND-1))
[ "$1" = "--" ] && shift

# Checking arguments
if [ -z "$asm" ]; then
  echo "Input assembly required."
  echo
  echo_usage
  exit 1
fi

if [ -z "$1" ]; then
  echo "Long reads required."
  echo
  echo_usage
  exit 1
else
  reads=$1
fi

if [ -z "$type" ]; then
  echo "Reads type required."
  echo
  echo_usage
  exit 1
fi

case "$type" in
  pb_clr)
	mm_map="map-pb"
	mm_ava="ava-pb"
    ;;
  pb_hifi)
    #mm_map="map-hifi"
    #mm_ava="map-hifi -X -e0 -m100"
    mm_map="asm20"
    mm_ava="asm20 --min-occ-floor=0 -X -m100 -g10000 --max-chain-skip 25"
    ;;
  nanopore)
    mm_map="map-ont"
    mm_ava="ava-ont"
    ;;
  *)
	echo "Unknow read type:" $type
	echo_usage
    exit 1
    ;;
  esac

# Running pipeline
mkdir -p "${output}"
org_path=$(pwd)
cd "${output}"
rm -f gapless.fa

for (( i=${start}; i<=$(expr ${start} + ${iterations} - 1); i++ ))
do
  if [ ! -f pass${i}/gapless.fa ] || [ $reset = true ]; then
    mkdir -p pass${i}/logs
    mkdir -p pass${i}/timing
    # Split
    if [ ! -f pass${i}/gapless_split.fa ] || [ $reset = true ]; then
      if [ $i -eq 1 ]; then
        env time -v -o pass${i}/timing/gapless_split.txt gapless.py split -o pass1/gapless_split.fa "${org_path}/${asm}" >pass${i}/logs/gapless_split.log 2>&1 || rm -f pass1/gapless_split.fa
      else
        env time -v -o pass${i}/timing/gapless_split.txt gapless.py split -o pass${i}/gapless_split.fa pass$(expr $i - 1)/gapless.fa >pass${i}/logs/gapless_split.log 2>&1 || rm -f pass${i}/gapless_split.fa
      fi
      if [ ! -f pass${i}/gapless_split.fa ]; then
        echo "pipeline crashed: split"
        exit 1
      fi
    fi
    # Scaffold
    if [ ! -f pass${i}/gapless_scaffold_paths.csv ] || [ $reset = true ]; then
      if [ ! -f pass${i}/gapless_split_repeats.paf ] || [ $reset = true ]; then
        env time -v -o pass${i}/timing/minimap2_repeats.txt minimap2 -t $threads -DP -k19 -w19 -m200 pass${i}/gapless_split.fa pass${i}/gapless_split.fa > pass${i}/gapless_split_repeats.paf 2>pass${i}/logs/minimap2_repeats.log || rm -f pass${i}/gapless_split_repeats.paf
      fi
      if [ ! -f pass${i}/gapless_reads.paf ] || [ $reset = true ]; then
        env time -v -o pass${i}/timing/minimap2_reads.txt minimap2 -t $threads -x $mm_map -c -N 5 --secondary=no pass${i}/gapless_split.fa "${org_path}/${reads}" > pass${i}/gapless_reads.paf 2>pass${i}/logs/minimap2_reads.log || rm -f pass${i}/gapless_reads.paf
        #env time -v -o pass${i}/timing/minimap2_reads.txt minimap2 -t $threads -x $mm_map -N 5 --secondary=no pass${i}/gapless_split.fa "${org_path}/${reads}" > pass${i}/gapless_reads.paf 2>pass${i}/logs/minimap2_reads.log || rm -f pass${i}/gapless_reads.paf
      fi
      rm -f pass${i}/gapless.fa # Remove the final file here, so that we can be certain that if it exists later it is the new one not an old one
      env time -v -o pass${i}/timing/gapless_scaffold.txt gapless.py scaffold -p pass${i}/gapless -s pass${i}/gapless_stats.pdf pass${i}/gapless_split.fa pass${i}/gapless_reads.paf pass${i}/gapless_split_repeats.paf >pass${i}/logs/gapless_scaffold.log 2>&1 &&\
      rm -f pass${i}/gapless_reads.paf pass${i}/gapless_split_repeats.paf
      if [ -f pass${i}/gapless_reads.paf ]; then
		rm -f pass${i}/gapless_scaffold_paths.csv
	  fi
      if [ ! -f pass${i}/gapless_scaffold_paths.csv ]; then
        echo "pipeline crashed : scaffold"
        exit 1
      fi
    fi
    # Extend
    if [ ! -f pass${i}/gapless_extended_scaffold_paths.csv ] || [ $reset = true ]; then
      if [ ! -f pass${i}/gapless_extending_reads.paf ] || [ $reset = true ]; then
        #mkfifo pipe
        #seqtk subseq "${org_path}/${reads}" pass${i}/gapless_extending_reads.lst | tee pipe | env time -v -o pass${i}/timing/minimap2_extension.txt minimap2 -t $threads -x $mm_ava - <(cat pipe) > pass${i}/gapless_extending_reads.paf 2>pass${i}/logs/minimap2_extension.log
        #rm -f pipe
        env time -v -o pass${i}/timing/minimap2_extension.txt minimap2 -t $threads -x $mm_ava <(seqtk subseq "${org_path}/${reads}" pass${i}/gapless_extending_reads.lst) <(seqtk subseq "${org_path}/${reads}" pass${i}/gapless_extending_reads.lst) > pass${i}/gapless_extending_reads.paf 2>pass${i}/logs/minimap2_extension.log || rm -f pass${i}/gapless_extending_reads.paf
      fi
      env time -v -o pass${i}/timing/gapless_extend.txt gapless.py extend -p pass${i}/gapless pass${i}/gapless_extending_reads.paf >pass${i}/logs/gapless_extend.log 2>&1 &&\
      rm -f pass${i}/gapless_extending_reads.paf
      if [ -f pass${i}/gapless_extending_reads.paf ]; then
		rm -f pass${i}/gapless_extended_scaffold_paths.csv
	  fi
      if [ ! -f pass${i}/gapless_extended_scaffold_paths.csv ]; then
        echo "pipeline crashed: extend"
        exit 1
      fi
    fi
    # Finish
    if [ ! -f pass${i}/gapless_raw.fa ] || [ $reset = true ]; then
      env time -v -o pass${i}/timing/gapless_finish.txt gapless.py finish -o pass${i}/gapless_raw.fa -H 0 -s pass${i}/gapless_extended_scaffold_paths.csv pass${i}/gapless_split.fa <(seqtk subseq "${org_path}/${reads}" pass${i}/gapless_used_reads.lst) >pass${i}/logs/gapless_finish.log 2>&1 || rm -f pass${i}/gapless_raw.fa
      if [ ! -f pass${i}/gapless_raw.fa ]; then
        echo "pipeline crashed: finish"
        exit 1
      fi
    fi
	# Consensus
	if [ ! -f pass${i}/gapless_consensus.paf ] || [ $reset = true ]; then
      env time -v -o pass${i}/timing/minimap2_consensus.txt minimap2 -t $threads -x $mm_map pass${i}/gapless_raw.fa "${org_path}/${reads}" > pass${i}/gapless_consensus.paf 2>pass${i}/logs/minimap2_consensus.log || rm -f pass${i}/gapless_consensus.paf
    fi
    env time -v -o pass${i}/timing/racon.txt racon -t $threads "${org_path}/${reads}" pass${i}/gapless_consensus.paf pass${i}/gapless_raw.fa > pass${i}/gapless.fa 2>pass${i}/logs/racon.log &&\
    rm -f pass${i}/gapless_consensus.paf pass${i}/gapless_raw.fa pass${i}/gapless_split.fa
    if [ -f pass${i}/gapless_raw.fa ]; then
		rm -f pass${i}/gapless.fa # In case of an error remove final output file to avoid going into the next round
    fi
    if [ ! -f pass${i}/gapless.fa ]; then
      echo "pipeline crashed: consensus"
      exit 1
    fi
  fi
done

if [ -f pass${iterations}/gapless.fa ]; then
  ln -s pass${iterations}/gapless.fa gapless.fa
  echo "pipeline successful"
fi
cd "${org_path}"

exit 0














