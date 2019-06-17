#!/usr/bin/python

from Bio import SeqIO

import getopt
import gzip
import re
import sys

def splitScaffolds(faFile,oFile=False,minN=False):
    if False == oFile:
        if ".gz" == faFile[-3:len(faFile)]:
            oFile = faFile.rsplit('.',2)[0]+"_split.fa"
        else:
            oFile = faFile.rsplit('.',1)[0]+"_split.fa"
        pass
    if False == minN:
        minN = 1;
        pass

    with open(oFile, 'w') as fout:
        with gzip.open(faFile, 'rb') if 'gz' == faFile.rsplit('.',1)[-1] else open(faFile, 'rU') as fin:
            for record in SeqIO.parse(fin, "fasta"):
                # Split scaffolds into contigs
                contigs=re.split('([nN]+)',str(record.seq))
                # Split identifier and further description
                seqid=record.description.split(' ', 1 )
                if 2 == len(seqid):
                    seqid[1] = " " + seqid[1] # Add the space here, so in case seqid[1] is empty no trailing space is in the sequence identifier
                else:
                    seqid.append("")
                
                # Combine sequences that do not reach minimum of N's at split sites, attach start and end position to seq id and print it
                startPos = 0
                seq = ""
                numN = 0
                for contig in contigs:
                    if -1 == numN:
                        # Contig of only N's
                        numN = len(contig)
                    else:
                        # Real contig
                        if numN < minN:
                            # Minimum number of N's not reached at this split site: Merging sequences
                            if len(seq):
                                if len(contig):
                                    seq += 'N' * numN + contig
                            else:
                                seq = contig
                                startPos += numN
                             
                        else:
                            # Split valid: Print contig
                            if len(seq): # Ignore potentially empty sequences, when scaffold starts or ends with N
                                fout.write(">{0}_chunk{1}-{2}{3}\n".format(seqid[0],startPos+1,startPos+len(seq), seqid[1]))
                                fout.write(seq)
                                fout.write('\n');
                            # Insert new sequence
                            startPos += len(seq) + numN
                            seq = contig
                        numN = -1
                
                # Print final sequence
                if len(seq):
                    if len(seq)==len(record.seq):
                        fout.write(">{0}\n".format(record.description))
                    else:
                        fout.write(">{0}_chunk{1}-{2}{3}\n".format(seqid[0],startPos+1,startPos+len(seq), seqid[1]))
                    fout.write(seq)
                    fout.write('\n');


def usage():
    print "Usage: python splitScaffolds.py [OPTIONS] File"
    print "Splits sequences in a fasta file at N's."
    print "  -h, --help            display this help and exit"
    print "  -n, --minN [int]      minimum number of N's to split at that position (1)"
    print "  -o, --output FILE     file to which the split sequences should be written to ({File}_split.fa)"

def main(argv):
    try:
        optlist, args = getopt.getopt(argv, 'hkn:o:', ['help','keepN','minN=','output='])
    except getopt.GetoptError:
        print "Unknown option\n"
        usage()
        sys.exit(2)

    oFile = False
    keepN = False
    minN = False
    for opt, par in optlist:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-n", "--minN"):
            try:
                minN = int(par)
            except ValueError:
                print "-n,--minN option only accepts integers"
                sys.exit()
        elif opt in ("-o", "--output"):
            oFile = par

    if 1 != len(args):
        print "Wrong number of files. Exactly one file is required.\n"
        usage()
        sys.exit(2)

    splitScaffolds(args[0],oFile,minN)

if __name__ == "__main__":
    main(sys.argv[1:])
