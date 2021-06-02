#!/usr/bin/env python3
from Bio import SeqIO

from datetime import timedelta
import getopt
import glob
import gzip
from matplotlib import use as mpl_use
mpl_use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import re
from scipy.optimize import minimize
from scipy.special import erf
import seaborn as sns
import sys
from time import clock

def MiniGapSplit(fa_file,o_file=False,min_n=False):
    if False == o_file:
        if ".gz" == fa_file[-3:len(fa_file)]:
            o_file = fa_file.rsplit('.',2)[0]+"_split.fa"
        else:
            o_file = fa_file.rsplit('.',1)[0]+"_split.fa"
        pass
    if False == min_n:
        min_n = 1;
        pass
    
    # Remove output files, such that we do not accidentially use an old one after a crash
    if os.path.exists(o_file):
        os.remove(o_file)

    with open(o_file, 'w') as fout:
        with gzip.open(fa_file, 'rb') if 'gz' == fa_file.rsplit('.',1)[-1] else open(fa_file, 'rU') as fin:
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
                start_pos = 0
                seq = ""
                num_n = 0
                for contig in contigs:
                    if -1 == num_n:
                        # Contig of only N's
                        num_n = len(contig)
                    else:
                        # Real contig
                        if num_n < min_n:
                            # Minimum number of N's not reached at this split site: Merging sequences
                            if len(seq):
                                if len(contig):
                                    seq += 'N' * num_n + contig
                            else:
                                seq = contig
                                start_pos += num_n
                             
                        else:
                            # Split valid: Print contig
                            if len(seq): # Ignore potentially empty sequences, when scaffold starts or ends with N
                                fout.write(">{0}_chunk{1}-{2}{3}\n".format(seqid[0],start_pos+1,start_pos+len(seq), seqid[1]))
                                fout.write(seq)
                                fout.write('\n')
                            # Insert new sequence
                            start_pos += len(seq) + num_n
                            seq = contig
                        num_n = -1
                
                # Print final sequence
                if len(seq):
                    if len(seq)==len(record.seq):
                        fout.write(">{0}\n".format(record.description))
                    else:
                        fout.write(">{0}_chunk{1}-{2}{3}\n".format(seqid[0],start_pos+1,start_pos+len(seq), seqid[1]))
                    fout.write(seq)
                    fout.write('\n')

def ReadContigs(assembly_file):
    # Create contig table of assembly
    names = []
    descriptions = []
    seq_lengths = []
    distances_next = []
    contig_end = -1
    last_scaffold = ""

    with gzip.open(assembly_file, 'rb') if 'gz' == assembly_file.rsplit('.',1)[-1] else open(assembly_file, 'rU') as fin:
        for record in SeqIO.parse(fin, "fasta"):
            seqid = record.description.split(' ', 1 ) # Split identifier(name) and further description
            names.append(seqid[0])
            if 2 == len(seqid):
                descriptions.append(seqid[1])
            else:
                descriptions.append("") 
            seq_lengths.append( len(record.seq) )

            # Get distance to next contig from the encoding in the identifier (_chunk[start]-[end])
            chunk = seqid[0].rsplit('_chunk')
            if 2 == len(chunk):
                coords = chunk[1].split('-')
                if 2 == len(coords):
                    try:
                        if 0 < contig_end and chunk[0] == last_scaffold:
                            distances_next.append(int(coords[0]) - contig_end - 1)
                        elif 0 <= contig_end:
                            distances_next.append(-1)
    
                        contig_end = int(coords[1])
                        last_scaffold = chunk[0]
                    except ValueError:
                        print('Cannot interpret "_chunk{}-{}" as a position encoding in: '.format(coords[0], coords[1]), record.description)
                        if 0 <= contig_end:
                            distances_next.append(-1)
                        contig_end = 0
                else:
                    print("Position encoding in contig identifier wrong:", record.description)
                    if 0 <= contig_end:
                        distances_next.append(-1)
                    contig_end = 0
            else:
                if 0 <= contig_end:
                    distances_next.append(-1)
                contig_end = 0

        distances_next.append(-1) # Last contig has no next

    contigs = pd.DataFrame({ 'name' : names,
                             'description' : descriptions,
                             'length' : seq_lengths,
                             'org_dist_right' : distances_next })
    contigs['org_dist_left'] = contigs['org_dist_right'].shift(1, fill_value=-1)

    # Create dictionary for mapping names to table row
    contig_ids = {k: v for k, v in zip(names, range(len(names)))}

    return contigs, contig_ids

def calculateN50(len_values):
    # len_values must be sorted from lowest to highest
    len_total = np.sum(len_values)
    len_sum = 0
    len_N50 = 0 # Use as index first
    while len_sum < len_total/2:
        len_N50 -= 1
        len_sum += len_values[len_N50]
    len_N50 = len_values[len_N50]

    return len_N50, len_total

def GetInputInfo(result_info, contigs):
    # Calculate lengths
    con_len = contigs['length'].values

    scaf_len = contigs[['org_dist_left']].copy()
    scaf_len['length'] = con_len
    scaf_len.loc[scaf_len['org_dist_left'] >= 0, 'length'] += scaf_len.loc[scaf_len['org_dist_left'] >= 0, 'org_dist_left']
    scaf_len['scaffold'] = (scaf_len['org_dist_left'] == -1).cumsum()
    scaf_len = scaf_len.groupby('scaffold', sort=False)['length'].sum().values

    # Calculate N50
    con_len.sort()
    con_N50, con_total = calculateN50(con_len)

    scaf_len.sort()
    scaf_N50, scaf_total = calculateN50(scaf_len)

    # Store output stats
    result_info['input'] = {}
    result_info['input']['contigs'] = {}
    result_info['input']['contigs']['num'] = len(con_len)
    result_info['input']['contigs']['total'] = con_total
    result_info['input']['contigs']['min'] = con_len[0]
    result_info['input']['contigs']['max'] = con_len[-1]
    result_info['input']['contigs']['N50'] = con_N50
    
    result_info['input']['scaffolds'] = {}
    result_info['input']['scaffolds']['num'] = len(scaf_len)
    result_info['input']['scaffolds']['total'] = scaf_total
    result_info['input']['scaffolds']['min'] = scaf_len[0]
    result_info['input']['scaffolds']['max'] = scaf_len[-1]
    result_info['input']['scaffolds']['N50'] = scaf_N50

    return result_info

def ReadPaf(file_name):
    return pd.read_csv(file_name, sep='\t', header=None, usecols=range(12), names=['q_name','q_len','q_start','q_end','strand','t_name','t_len','t_start','t_end','matches','alignment_length','mapq'], dtype={'q_len':np.int32, 'q_start':np.int32, 'q_end':np.int32, 't_len':np.int32, 't_start':np.int32, 't_end':np.int32, 'matches':np.int32, 'alignment_length':np.int32, 'mapq':np.int16})

def stackhist(x, y, **kws):
    grouped = pd.groupby(x, y)
    data = [d for _, d in grouped]
    labels = [l for l, _ in grouped]
    plt.hist(data, histtype="barstacked", label=labels)

def PlotHist(pdf, xtitle, ytitle, values, category=[], catname='category', threshold=None, logx=False, logy=False):
    plt.close()

    if logx:      
        values = np.log10(np.extract(values>0,values))

    if len(category):
        #sns.FacetGrid(pd.DataFrame({"values":values, "category":category}), hue="category")
        ax = sns.distplot(values, bins=100, kde=False, color='w')
        
        cats = np.unique(category)
        cat_ids = {c: i for c,i in zip(cats, range(len(cats)))}
        cat_ids = np.array(itemgetter(*category)(cat_ids))

        for x in range(len(cats)):
            plt.hist(values[cat_ids>=x],bins=np.linspace(min(values),max(values),101))

        plt.legend(handles=[patches.Patch(color=col, label=lab) for col, lab in zip(sns.color_palette().as_hex()[:len(cats)],cats)], title=catname, loc='upper right')
        ax=plt.gca()
    else:
        ax = sns.distplot(values, bins=100, kde=False)
        
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    if logx:
        plt.xlim(int(min(values)), int(max(values))+1)
        locs, labels = plt.xticks()
        ax.set(xticklabels=np.where(locs.astype(int) == locs, (10 ** locs).astype(str), ""))

    if threshold:
        if logx:
            plt.axvline(np.log10(threshold))
        else:
            plt.axvline(threshold)

    if logy:
        ax.set_yscale('log', nonpositive='clip')
    else:
        # Set at what range of exponents they are not plotted in exponential format for example (-3,4): [0.001-1000[
        ax.get_yaxis().get_major_formatter().set_powerlimits((-3,4))

    pdf.savefig()

    return

def PlotXY(pdf, xtitle, ytitle, x, y, category=[], count=[], logx=False, linex=[], liney=[]):
    plt.close()

    if logx:    
        y = np.extract(x>0, y)
        if len(liney):
            liney = np.extract(linex>0, liney)
        if len(category):
            category = np.extract(x>0, category)           
        x = np.log10(np.extract(x>0,x))
        if len(linex):
            linex = np.extract(linex>0, linex)

    if len(linex):
        sns.lineplot(x=linex, y=liney, color='red', linewidth=2.5)

    if len(count):
        if len(category):
            ax = sns.scatterplot(x, y, hue=count, style=category, linewidth=0, alpha = 0.7)
        else:
            ax = sns.scatterplot(x, y, hue=count, linewidth=0, alpha = 0.7);
    elif len(category):
        ax = sns.scatterplot(x, y, hue=category, linewidth=0, alpha = 0.7)
    else:
        ax = sns.scatterplot(x, y, linewidth=0, alpha = 0.7)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    if logx:
        plt.xlim(0, int(max(x))+1)
        locs, labels = plt.xticks()
        ax.set(xticklabels=10 ** locs)

    # Set at what range of exponents they are not plotted in exponential format for example (-3,4): [0.001-1000[
    ax.get_yaxis().get_major_formatter().set_powerlimits((-3,4))

    pdf.savefig()

    return

def MaskRepeatEnds(contigs, repeat_file, contig_ids, max_repeat_extension, min_len_repeat_connection, repeat_len_factor_unique, remove_duplicated_contigs, pdf):
    # Read in minimap 2 repeat table (.paf)
    repeat_table = ReadPaf(repeat_file)
    
    repeat_table = repeat_table[(repeat_table['q_name'] != repeat_table['t_name']) | (repeat_table['q_start'] != repeat_table['t_start'])].copy()  # Ignore repeats on same contig and same position
     
    if pdf and len(repeat_table):
        tmp_dist = np.minimum(repeat_table['q_start'], repeat_table['q_len']-repeat_table['q_end']) # Repeat distance to contig end
        PlotHist(pdf, "Repeat distance to contig end", "# Repeat mappings", tmp_dist, logx=True, threshold=max_repeat_extension)
        if np.sum(tmp_dist < 10*max_repeat_extension):
            PlotHist(pdf, "Repeat distance to contig end", "# Repeat mappings", np.extract(tmp_dist < 10*max_repeat_extension, tmp_dist), threshold=max_repeat_extension)

    # We can do everything on query as all mappings are also reversed in the table
    # Check if the repeat belongs to a contig end
    repeat_table['left_repeat'] = repeat_table['q_start'] <= max_repeat_extension
    repeat_table['right_repeat'] = repeat_table['q_len'] - repeat_table['q_end'] <= max_repeat_extension
    
    repeat_table['q_id'] = itemgetter(*repeat_table['q_name'])(contig_ids)
    repeat_table['t_id'] = itemgetter(*repeat_table['t_name'])(contig_ids)
    
    contigs['repeat_mask_left'] = int(0)
    contigs['repeat_mask_right'] = contigs['length']
    contigs['remove'] = False
    
    # Complete duplicate -> Schedule for removal
    # If the target is also a complete duplicate, keep the longer one or at same length decide by name
    complete_repeats = np.unique(repeat_table.loc[repeat_table['left_repeat'] & repeat_table['right_repeat'] &
             ((repeat_table['t_start'] > max_repeat_extension) |
              (repeat_table['q_len'] - repeat_table['q_end'] > max_repeat_extension) |
              (repeat_table['t_len'] > repeat_table['q_len']) |
              ((repeat_table['t_len'] == repeat_table['q_len']) & (repeat_table['t_name'] < repeat_table['q_name']))), 'q_id'])
    if remove_duplicated_contigs:
        contigs.iloc[complete_repeats, contigs.columns.get_loc('remove')] = True
    
    ## Handle complete repeats, complexity contigs and tandem repeats
    # Remove the complete_repeats from the repeat table, as the repeat won't exist anymore after the removal of the contigs (sp we don't want to count those contigs for repeats in other contigs)
    repeat_table = repeat_table[np.logical_not(np.isin(repeat_table['q_id'], complete_repeats)) & np.logical_not(np.isin(repeat_table['t_id'], complete_repeats))].copy()
    
    # Add low complexity contigs and tandem repeats to complete_repeats, where you get a similar sequence when shifting the frame a bit
    # We want to keep them as they are no duplications but don't do anything with them afterwards, because the mappings cannot be anchored and are therefore likely at the wrong place
    # The idea is to run minigap multiple times until another contig with a proper anchor is extended enough to overlap this contig and then it is removed as a duplicate
    complete_repeats = np.unique(np.concatenate([ complete_repeats, np.unique(repeat_table.loc[repeat_table['left_repeat'] & repeat_table['right_repeat'] & (repeat_table['q_id'] == repeat_table['t_id']), 'q_id'])]))
    # Mask the whole read, so mappings to it will be ignored, as we remove it anyways
    contigs.iloc[complete_repeats, contigs.columns.get_loc('repeat_mask_right')] = 0
    contigs.iloc[complete_repeats, contigs.columns.get_loc('repeat_mask_left')] = contigs.iloc[complete_repeats, contigs.columns.get_loc('length')]
    
    # Remove the low complexity contigs and tandem repeats from the repeat table
    repeat_table = repeat_table[np.logical_not(repeat_table['left_repeat'] & repeat_table['right_repeat'])].copy()
    
    ## Check for repeat connections
    con_search = repeat_table[repeat_table['left_repeat'] | repeat_table['right_repeat']].copy()
    con_search['rep_len'] = con_search['q_end'] - con_search['q_start']
    con_search['con_left'] = con_search['t_start'] <= max_repeat_extension
    con_search['con_right'] = con_search['t_len'] - con_search['t_end'] <= max_repeat_extension
    
    # Only keep longest for one type of connection to not undermine the validity of a repeat connection with a repeat between the same contigs
    con_search.sort_values(['q_id','left_repeat','right_repeat','t_id','con_left','con_right','rep_len'], inplace=True)
    con_search = con_search.groupby(['q_id','left_repeat','right_repeat','t_id','con_left','con_right'], sort=False).last().reset_index()
    
    # Get second longest repeats to potentially veto repeat connections
    con_search.sort_values(['q_id','left_repeat','right_repeat','rep_len'], inplace=True)
    con_search['longest'] = (con_search['q_id'] != con_search['q_id'].shift(-1, fill_value=-1)) | (con_search['left_repeat'] != con_search['left_repeat'].shift(-1, fill_value=-1))
    second_longest = con_search[con_search['longest']==False].groupby(['q_id','left_repeat','right_repeat'], sort=False)['rep_len'].last().reset_index(name="len_second")
    
    # Get potential repeat connections
    con_search = con_search.loc[con_search['longest'] & (con_search['con_left'] | con_search['con_right']) & (con_search['q_id'] != con_search['t_id']), ['q_id','left_repeat','right_repeat','t_id','con_left','con_right','rep_len','strand']].copy()
    con_search = con_search[np.logical_not(con_search['con_left'] & con_search['con_right'])].copy() # Remove complete duplicates
    con_search = con_search[(con_search['left_repeat'] != con_search['con_left']) & (con_search['strand'] == '+') | (con_search['left_repeat'] == con_search['con_left']) & (con_search['strand'] == '-')].copy() # Strand must match a connection
    
    # Apply filter
    if len(con_search):
        con_search = con_search.merge(second_longest, on=['q_id','left_repeat','right_repeat'], how='left').fillna(0)
        con_search['accepted'] = (con_search['rep_len'] >= min_len_repeat_connection) & (con_search['len_second']*repeat_len_factor_unique < con_search['rep_len'])
        if pdf:
            PlotHist(pdf, "Connection length", "# Potential repeat connections", con_search['rep_len'], logx=True, threshold=min_len_repeat_connection)
            if np.sum(con_search['rep_len'] < 10*min_len_repeat_connection):
                PlotHist(pdf, "Connection length", "# Potential repeat connections", np.extract(con_search['rep_len'] < 10*min_len_repeat_connection, con_search['rep_len']), threshold=min_len_repeat_connection)
            PlotXY(pdf, "Longest repeat connection", "Second longest connection", con_search['rep_len'], con_search['len_second'], category=np.where(con_search['accepted'], "accepted", "declined"))
            #PlotXY(pdf, "Longest repeat connection", "Second longest connection", con_search['rep_len'], con_search['len_second'], category=con_search['accepted'], logx=True)
        con_search = con_search[con_search['accepted']].copy()
        
        # Check that partner sides also survived all filters
        con_search = con_search[(con_search.merge(con_search, left_on=['t_id','con_left','con_right'], right_on=['q_id','left_repeat','right_repeat'], how='left', indicator=True)['_merge'] == "both").values].copy()
    
    # Store repeat connections
    contigs['rep_con_left'] = -1
    contigs['rep_con_side_left'] = ''
    contigs['rep_con_right'] = -1
    contigs['rep_con_side_right'] = ''
    if len(con_search):
        contigs.iloc[con_search.loc[con_search['left_repeat'], 'q_id'], contigs.columns.get_loc('rep_con_left')] = con_search.loc[con_search['left_repeat'], 't_id'].values
        contigs.iloc[con_search.loc[con_search['left_repeat'], 'q_id'], contigs.columns.get_loc('rep_con_side_left')] = np.where(con_search.loc[con_search['left_repeat'], 'con_left'].values, 'l', 'r')
        contigs.iloc[con_search.loc[con_search['right_repeat'], 'q_id'], contigs.columns.get_loc('rep_con_right')] = con_search.loc[con_search['right_repeat'], 't_id'].values
        contigs.iloc[con_search.loc[con_search['right_repeat'], 'q_id'], contigs.columns.get_loc('rep_con_side_right')] = np.where(con_search.loc[con_search['right_repeat'], 'con_left'].values, 'l', 'r')

    ## Mask repeat ends
    # Keep only repeat positions and sort repeats by contig and start position to merge everything that is closer than max_repeat_extension
    repeat_table = repeat_table[['q_id','q_len','q_start','q_end']].copy()
    repeat_table.sort_values(['q_id','q_start','q_end'], inplace=True)
    repeat_table['group'] = (repeat_table['q_id'] != repeat_table['q_id'].shift(1, fill_value=-1)) | (repeat_table['q_start'] > repeat_table['q_end'].shift(1, fill_value=-1) + max_repeat_extension)
    while np.sum(repeat_table['group']) < len(repeat_table):
        repeat_table['group'] = repeat_table['group'].cumsum()
        repeat_table = repeat_table.groupby('group').agg(['min','max']).reset_index()[[('q_id','min'),('q_len','min'),('q_start','min'),('q_end','max')]].copy()
        repeat_table.columns = repeat_table.columns.droplevel(-1)
        repeat_table['group'] = (repeat_table['q_id'] != repeat_table['q_id'].shift(1, fill_value=-1)) | (repeat_table['q_start'] > repeat_table['q_end'].shift(1, fill_value=-1) + max_repeat_extension)
    
    # Mask contig ends with repeats
    repeat_table['left_repeat'] = repeat_table['q_start'] <= max_repeat_extension
    mask = repeat_table[repeat_table['left_repeat']].groupby('q_id')['q_end'].max().reset_index()
    contigs.iloc[mask['q_id'].values, contigs.columns.get_loc('repeat_mask_left')] = mask['q_end'].values
    repeat_table['right_repeat'] = repeat_table['q_len'] - repeat_table['q_end'] <= max_repeat_extension
    mask = repeat_table[repeat_table['right_repeat']].groupby('q_id')['q_start'].max().reset_index()
    contigs.iloc[mask['q_id'].values, contigs.columns.get_loc('repeat_mask_right')] = mask['q_start'].values

    # If we had a full repeat after the combination mask it completely (but don't remove it, same strategy as for low complexity contigs and tandem repeats)
    contigs.loc[contigs['repeat_mask_left'] > contigs['repeat_mask_right'], 'repeat_mask_left'] = contigs.loc[contigs['repeat_mask_left'] > contigs['repeat_mask_right'], 'length']
    contigs.loc[contigs['repeat_mask_left'] > contigs['repeat_mask_right'], 'repeat_mask_right'] = 0
    
    # Only keep center repeats (repeats not at the contig ends)
    repeat_table = repeat_table.loc[np.logical_not(repeat_table['left_repeat'] | repeat_table['right_repeat']), ['q_id','q_start','q_end']].copy()
    repeat_table.rename(columns={'q_id':'con_id', 'q_start':'start', 'q_end':'end'}, inplace=True)

    if pdf:
        masked = ((contigs.loc[contigs['remove'] == False, 'repeat_mask_left']+contigs.loc[contigs['remove'] == False, 'length']-contigs.loc[contigs['remove'] == False, 'repeat_mask_right'])/contigs.loc[contigs['remove'] == False, 'length']).values*100
        masked[masked > 100] = 100 # Fully masked the way it's done would be 200, which does not make sense
        category = np.log10(contigs.loc[contigs['remove'] == False, 'length']).values.astype(int)
        category = [''.join(['>',a," bases"]) for a in np.power(10,category).astype(str)]

        if len(masked):
            PlotHist(pdf, "% of bases masked", "# Contigs", masked, category=category, catname='length', logy=True)

    return contigs, repeat_table

def GetThresholdsFromReadLenDist(mappings, num_read_len_groups):
    # Get read length distribution
    read_len_dist = mappings[['q_name','q_len']].drop_duplicates()
    read_len_dist = read_len_dist['q_len'].sort_values().values

    return read_len_dist[[len(read_len_dist)//num_read_len_groups*i for i in range(1,num_read_len_groups)]]

def GetBinnedCoverage(mappings, length_thresholds):
    # Count coverage for bins of threshold length
    cov_counts = []
    tmp_seqs = mappings[['t_name','t_len']].drop_duplicates().sort_values('t_name').reset_index(drop=True)
    for cur_cov_thres in length_thresholds:
        # Count mappings
        cov_count = mappings.loc[mappings['t_end']-mappings['t_start'] >= cur_cov_thres, ['t_name','t_len','t_start','t_end']].copy()
        cov_count['offset'] = cov_count['t_len']%cur_cov_thres // 2
        cov_count['start_bin'] = (cov_count['t_start'] - cov_count['offset'] - 1) // cur_cov_thres + 1
        cov_count['end_bin'] = (cov_count['t_end'] - cov_count['offset']) // cur_cov_thres
        cov_count = cov_count.loc[cov_count['start_bin'] < cov_count['end_bin'], ['t_name','start_bin','end_bin']]
        cov_count = cov_count.loc[np.repeat(cov_count.index.values, (cov_count['end_bin']-cov_count['start_bin']).values)]
        cov_count.reset_index(inplace=True)
        cov_count['bin'] = cov_count.groupby(['index'], sort=False).cumcount() + cov_count['start_bin']
        cov_count = cov_count.groupby(['t_name','bin'], sort=False).size().reset_index(name='count')
        # Add zero bins
        zero_bins = tmp_seqs.loc[np.repeat(tmp_seqs.index.values, tmp_seqs['t_len']//cur_cov_thres), ['t_name']].reset_index(drop=True)
        zero_bins['bin'] = zero_bins.groupby(['t_name'], sort=False).cumcount()
        cov_count = zero_bins.merge(cov_count, on=['t_name','bin'], how='left').fillna(0)
        cov_count['count'] = cov_count['count'].astype(int)
        # Add to cov_counts
        cov_count['bin_size'] = cur_cov_thres
        cov_counts.append(cov_count)

    return pd.concat(cov_counts, ignore_index=True)

def NormCDF(x, mu, sigma):
    return (1+erf((x-mu)/sigma))/2 # We skip the sqrt(2) factor for the sigma, since sigma will be fitted anyways and we don't care about the value as long as it describes the curve

def CovChi2(par, df): #, expo
    return np.sum((NormCDF(df['x'], par[0], par[1]) - df['y'])**2)
    
def GetCoverageProbabilities(cov_counts, min_num_reads, cov_bin_fraction, pdf):
    probs = cov_counts.groupby(['bin_size','count']).size().reset_index(name='nbins')
    probs['nbin_cumsum'] = probs.groupby(['bin_size'], sort=False)['nbins'].cumsum()
    cov_bins = probs.groupby(['bin_size'])['nbins'].agg(['sum','size']).reset_index()
    probs['nbin_sum'] = np.repeat( cov_bins['sum'].values, cov_bins['size'].values)

    # Fit CDF with CDF of normal distribution
    probs['lower_half'] = (np.repeat( (cov_bins['sum']*0.5).values.astype(int), cov_bins['size'].values) > probs['nbin_cumsum']).shift(1, fill_value=True) # Shift so that we also get the bin that is only partially in the lower half (Ensures that we always have at least one bin)
    probs.loc[probs['bin_size'] != probs['bin_size'].shift(1), 'lower_half'] = True # The first bin must be True, but likely has been turned to False by the shift 
    prob_len = np.unique(probs['bin_size'])
    prob_mu = []
    prob_sigma = []
    for bsize in prob_len:
        selection = (bsize == probs['bin_size']) & probs['lower_half']
        opt_par = minimize(CovChi2, [probs.loc[selection, 'count'].values[-1],1.0], args=(pd.DataFrame({'x':probs.loc[selection, 'count'], 'y':probs.loc[selection, 'nbin_cumsum']/probs.loc[selection, 'nbin_sum']})), method='Nelder-Mead').x
        prob_mu.append(opt_par[0])
        prob_sigma.append(opt_par[1])
        if pdf:
            x_values = np.arange(probs.loc[selection, 'count'].values[0], probs.loc[selection, 'count'].values[-1]+1)
            PlotXY(pdf, "Max. Coverage", "% Bins (Size: {})".format(bsize), probs.loc[selection, 'count'], probs.loc[selection, 'nbin_cumsum']/probs.loc[selection, 'nbin_sum'], linex=x_values, liney=NormCDF(x_values,opt_par[0],opt_par[1]))

    # Store results in DataFrame
    cov_probs = pd.DataFrame({'length':prob_len, 'mu':prob_mu, 'sigma':prob_sigma})

    return cov_probs

def GetBestSubreads(mappings, alignment_precision):
    # Identify groups of subread mappings (of which only one is required)
    mappings['subread'] = [ re.sub(r'^(.*?/.*?)/.+', r'\1', qname) for qname in mappings['q_name'].values ]
    mappings.sort_values(['subread','t_name','t_start','t_end'], inplace=True)
    mappings['group'] = (( (mappings['subread'] != mappings['subread'].shift(1, fill_value="")) | (mappings['t_name'] != mappings['t_name'].shift(1, fill_value="")) | ((mappings['t_start']-mappings['t_start'].shift(1, fill_value=0) > alignment_precision) & (np.abs(mappings['t_end'] - mappings['t_end'].shift(1, fill_value=0)) > alignment_precision)) )).cumsum()
    
    # Keep only the best subread (over all mappings)
    mappings.sort_values(['group','mapq','matches'], ascending=[True,False,False], inplace=True)
    mappings['pos'] = mappings.groupby(['group'], sort=False).cumcount()
    mappings.sort_values(['q_name'], inplace=True)
    tmp = mappings.groupby(['q_name'], sort=False)['pos'].agg(['size','max','mean'])
    mappings['max_pos'] = np.repeat(tmp['max'].values, tmp['size'].values)
    mappings['mean_pos'] = np.repeat(tmp['mean'].values, tmp['size'].values)
    mappings.sort_values(['group','max_pos','mean_pos'], inplace=True)
    mappings = mappings.groupby(['group'], sort=False).first().reset_index()
    
    mappings.drop(columns=['subread','group','pos','max_pos','mean_pos'], inplace=True)
    
    return mappings

def ReadMappings(mapping_file, contig_ids, min_mapq, keep_all_subreads, alignment_precision, min_num_reads, cov_bin_fraction, num_read_len_groups, pdf):
    mappings = ReadPaf(mapping_file)

    length_thresholds = GetThresholdsFromReadLenDist(mappings, num_read_len_groups)
    cov_counts = GetBinnedCoverage(mappings, length_thresholds)
    cov_probs = GetCoverageProbabilities(cov_counts, min_num_reads, cov_bin_fraction, pdf)
    cov_counts.rename(columns={'count':'count_all'}, inplace=True)

    # Filter low mapping qualities
    if pdf:
        PlotHist(pdf, "Mapping quality", "# Mappings", mappings['mapq'], threshold=min_mapq, logy=True)
    mappings = mappings[min_mapq <= mappings['mapq']].copy()

    # Remove all but the best subread
    if not keep_all_subreads:
        mappings = GetBestSubreads(mappings, alignment_precision)

    cov_counts = cov_counts.merge(GetBinnedCoverage(mappings, length_thresholds), on=['bin_size','t_name','bin'], how='outer').fillna(0)
    cov_counts['count'] = cov_counts['count'].astype(np.int64)

    mappings['t_id'] = itemgetter(*mappings['t_name'])(contig_ids)
    cov_counts['t_id'] = itemgetter(*cov_counts['t_name'])(contig_ids)

    return mappings, cov_counts, cov_probs

def RemoveUnmappedContigs(contigs, mappings, remove_zero_hit_contigs):
    # Schedule contigs for removal that don't have any high quality reads mapping to it
    contigs['remove'] = False
    mapped_reads = np.bincount(mappings['t_id'], minlength=len(contigs))
    if remove_zero_hit_contigs:
        contigs.loc[0==mapped_reads, 'remove'] = True

    return contigs

def RemoveUnanchoredMappings(mappings, contigs, center_repeats, min_mapping_length, pdf, max_dist_contig_end):
    # Filter reads only mapping to repeat ends
    mapping_lengths = np.minimum(
                mappings['t_end'].values - np.maximum(mappings['t_start'].values, contigs['repeat_mask_left'].iloc[mappings['t_id']].values),
                np.minimum(mappings['t_end'].values, contigs['repeat_mask_right'].iloc[mappings['t_id']].values) - mappings['t_start'].values )
    mappings = mappings[min_mapping_length <= mapping_lengths].copy()

    if pdf:
        mapping_lengths[ 0 > mapping_lengths ] = 0 # Set negative values to zero (Only mapping to repeat)
        if np.sum(mapping_lengths < 10*min_mapping_length):
            PlotHist(pdf, "Mapping length", "# Mappings", np.extract(mapping_lengths < 10*min_mapping_length, mapping_lengths), threshold=min_mapping_length)
        PlotHist(pdf, "Mapping length", "# Mappings", mapping_lengths, threshold=min_mapping_length, logx=True)
                 
        # Plot distance of mappings from contig ends, where the continuing read is longer than the continuing contig
        dists_contig_ends = mappings.loc[mappings['t_start'] < np.where('+' == mappings['strand'], mappings['q_start'], mappings['q_len']-mappings['q_end']), 't_start']
        dists_contig_ends = np.concatenate([dists_contig_ends, 
                                            np.extract( mappings['t_len']-mappings['t_end'] < np.where('+' == mappings['strand'], mappings['q_len']-mappings['q_end'], mappings['q_start']), mappings['t_len']-mappings['t_end'] )] )
        if len(dists_contig_ends):
            if np.sum(dists_contig_ends < 10*max_dist_contig_end):
                PlotHist(pdf, "Distance to contig end", "# Reads reaching over contig ends", np.extract(dists_contig_ends < 10*max_dist_contig_end, dists_contig_ends), threshold=max_dist_contig_end, logy=True)
            PlotHist(pdf, "Distance to contig end", "# Reads reaching over contig ends", dists_contig_ends, threshold=max_dist_contig_end, logx=True)

    # Find and remove reads only mapping to center repeats (Distance to merge repeats should be larger than min_mapping_length, so we can consider here one repeat at a time)
    repeats = center_repeats.merge(mappings, left_on=['con_id'], right_on=['t_id'], how='inner')
    repeats = repeats[ (repeats['start']-min_mapping_length < repeats['t_start']) & (repeats['end']+min_mapping_length > repeats['t_end']) ].copy()
    mappings.sort_values(['q_name','q_start'], inplace=True)
    mappings = mappings[(mappings.merge(repeats[['q_name','q_start']].drop_duplicates(), on=['q_name','q_start'], how='left', indicator=True)['_merge'] == "left_only").values].copy()

    return mappings

def BreakReadsAtAdapters(mappings, adapter_signal_max_dist, keep_all_subreads):
    # Sort mappings by starting position and provide information on next/previous mapping (if from same read)
    mappings.sort_values(['q_name','q_start'], inplace=True)
    
    mappings['next_con'] = mappings['t_id'].shift(-1, fill_value=-1)
    mappings.loc[mappings['q_name'].shift(-1, fill_value='') != mappings['q_name'], 'next_con'] = -1
    mappings['next_strand'] = mappings['strand'].shift(-1, fill_value='')
    mappings.loc[-1 == mappings['next_con'], 'next_strand'] = ''
    
    mappings['prev_con'] = mappings['t_id'].shift(1, fill_value=-1)
    mappings.loc[mappings['q_name'].shift(1, fill_value='') != mappings['q_name'], 'prev_con'] = -1
    mappings['prev_strand'] = mappings['strand'].shift(1, fill_value='')
    mappings.loc[-1 == mappings['prev_con'], 'prev_strand'] = ''
    
    # Find potential left-over adapters
    mappings['read_start'] = 0
    mappings['read_end'] = mappings['q_len']
    
    if keep_all_subreads:
        # If we only keep the best subreads, we already removed most of the duplication and cannot separate the reads at adapters anymore, which results in spurious breaks
        adapter = (mappings['next_con'] == mappings['t_id']) & (mappings['next_strand'] != mappings['strand'])
        location_shift = np.abs(np.where('+' == mappings['strand'], mappings['t_end'] - mappings['t_end'].shift(-1, fill_value=0), mappings['t_start'] - mappings['t_start'].shift(-1, fill_value=0)))
    
        adapter = adapter & (location_shift <= adapter_signal_max_dist)
    
        # We need to be very sure that this is an adapter signal, because we do not want to miss a signal for a break point due to a missed inverted repeat
        # For this we require that they must have at least one mapping before and after the two with the adapter that are compatible with the adapter hypothesis
        adapter = adapter & (mappings['prev_con'] >= 0) & (mappings['prev_con'] == mappings['next_con'].shift(-1, fill_value=-1)) & (mappings['prev_strand'] != mappings['next_strand'].shift(-1, fill_value=''))
        adapter = adapter & (np.abs(np.where('-' == mappings['strand'], mappings['t_end'] - mappings['t_end'].shift(-1, fill_value=0), mappings['t_start'] - mappings['t_start'].shift(-1, fill_value=0))) <= adapter_signal_max_dist)
    
        # Split the read at the adapter
        mappings.loc[adapter, 'next_con'] = -1
        mappings.loc[adapter, 'next_strand'] = ''
        mappings.loc[adapter, 'read_end'] = mappings.loc[adapter, 'q_end'] # If the adapter was missed it's probably a bad part of the read, so better ignore it and don't use it for extensions
        mappings['read_end'] = mappings.loc[::-1, ['q_name','read_end']].groupby('q_name', sort=False).cummin()[::-1]
    
        adapter = adapter.shift(1, fill_value=False)
        mappings.loc[adapter, 'prev_con'] = -1
        mappings.loc[adapter, 'prev_strand'] = ''
        mappings.loc[adapter, 'read_start'] = mappings.loc[adapter, 'q_start'] # If the adapter was missed it's probably a bad part of the read, so better ignore it and don't use it for extensions
        mappings['read_start'] = mappings[['q_name','read_start']].groupby('q_name', sort=False).cummax()

    return mappings

def GetBrokenMappings(mappings, max_dist_contig_end, min_length_contig_break, pdf):
    # Find reads that have a break point and map on the left/right side(contig orientation) of it
    pot_breaks = mappings.copy()
    pot_breaks['next_pos'] = np.where(0 > pot_breaks['next_con'], -1, np.where(pot_breaks['next_strand'] == '+', pot_breaks['t_start'].shift(-1, fill_value=-1), pot_breaks['t_end'].shift(-1, fill_value=0)-1))
    pot_breaks['prev_pos'] = np.where(0 > pot_breaks['prev_con'], -1, np.where(pot_breaks['prev_strand'] == '-', pot_breaks['t_start'].shift(1, fill_value=-1), pot_breaks['t_end'].shift(1, fill_value=0)-1))    
    pot_breaks['next_dist'] = np.where(0 > pot_breaks['next_con'], 0, pot_breaks['q_start'].shift(-1, fill_value=0) - pot_breaks['q_end'])
    pot_breaks['prev_dist'] = np.where(0 > pot_breaks['prev_con'], 0, pot_breaks['q_start'] - pot_breaks['q_end'].shift(1, fill_value=0))

    left_breaks = pot_breaks[ (pot_breaks['t_len']-pot_breaks['t_end'] > max_dist_contig_end) &
                             np.where('+' == pot_breaks['strand'],
                                      (0 <= pot_breaks['next_con']) | (pot_breaks['read_end']-pot_breaks['q_end'] > min_length_contig_break),
                                      (0 <= pot_breaks['prev_con']) | (pot_breaks['q_start']-pot_breaks['read_start'] > min_length_contig_break)) ].copy()
    left_breaks['side'] = 'l'

    right_breaks = pot_breaks[ (pot_breaks['t_start'] > max_dist_contig_end) &
                              np.where('+' == pot_breaks['strand'],
                                       (0 <= pot_breaks['prev_con']) | (pot_breaks['q_start']-pot_breaks['read_start'] > min_length_contig_break),
                                       (0 <= pot_breaks['next_con']) | (pot_breaks['read_end']-pot_breaks['q_end'] > min_length_contig_break)) ].copy()
    right_breaks['side'] = 'r'

    pot_breaks = pd.concat([left_breaks, right_breaks], ignore_index=False)
    pot_breaks = pot_breaks.reset_index().rename(columns={'index':'map_index', 't_id':'contig_id'}).drop(columns=['t_name','matches','alignment_length'])
    pot_breaks['position'] = np.where('l' == pot_breaks['side'], pot_breaks['t_end'], pot_breaks['t_start'])
    pot_breaks['read_pos'] = np.where(('+' == pot_breaks['strand']) == ('l' == pot_breaks['side']), pot_breaks['q_end'], pot_breaks['q_start'])
    pot_breaks['read_side'] = np.where(('+' == pot_breaks['strand']) == ('l' == pot_breaks['side']), 'r', 'l')
    pot_breaks['map_len'] = np.where('l' == pot_breaks['side'], pot_breaks['t_end']-pot_breaks['t_start'], pot_breaks['t_end']-pot_breaks['t_start'])
    pot_breaks['con'] = np.where(('+' == pot_breaks['strand']) == ('l' == pot_breaks['side']), pot_breaks['next_con'], pot_breaks['prev_con'])
    pot_breaks['con_strand'] = np.where(('+' == pot_breaks['strand']) == ('l' == pot_breaks['side']), pot_breaks['next_strand'], pot_breaks['prev_strand'])
    pot_breaks['con_pos'] = np.where(('+' == pot_breaks['strand']) == ('l' == pot_breaks['side']), pot_breaks['next_pos'], pot_breaks['prev_pos'])
    pot_breaks['con_dist'] = np.where(('+' == pot_breaks['strand']) == ('l' == pot_breaks['side']), pot_breaks['next_dist'], pot_breaks['prev_dist'])
    pot_breaks['opos_con'] = np.where(('+' == pot_breaks['strand']) == ('l' == pot_breaks['side']), pot_breaks['prev_con'], pot_breaks['next_con']) # Connection on the opposite side of the break
    pot_breaks.drop(columns=['next_con','prev_con','next_strand','prev_strand','next_pos','prev_pos','next_dist','prev_dist'], inplace=True)

    if pdf:
        dists_contig_ends = np.where(pot_breaks['side'] == 'l', pot_breaks['t_len']-pot_breaks['t_end'], pot_breaks['t_start'])
        if len(dists_contig_ends):
            if np.sum(dists_contig_ends < 10*max_dist_contig_end):
                PlotHist(pdf, "Distance to contig end", "# Potential contig breaks", np.extract(dists_contig_ends < 10*max_dist_contig_end, dists_contig_ends), threshold=max_dist_contig_end)
            PlotHist(pdf, "Distance to contig end", "# Potential contig breaks", dists_contig_ends, threshold=max_dist_contig_end, logx=True)

    return pot_breaks

def GetNonInformativeMappings(mappings, contigs, min_extension, break_groups, pot_breaks):
    # All reads that are not extending contigs enough and do not have multiple mappings are non informative (except they overlap breaks)
    non_informative_mappings = mappings[(min_extension > mappings['q_start']) & (min_extension > mappings['q_len']-mappings['q_end']) & (mappings['q_name'].shift(1, fill_value='') != mappings['q_name']) & (mappings['q_name'].shift(-1, fill_value='') != mappings['q_name'])].index
    non_informative_mappings = np.setdiff1d(non_informative_mappings,pot_breaks['map_index'].values) # Remove breaking reads
    non_informative_mappings = mappings.loc[non_informative_mappings, ['t_id', 't_start', 't_end']]

    if len(break_groups):
        # Remove continues reads overlapping breaks from non_informative_mappings
        touching_breaks = []
        for i in range(break_groups['num'].max()+1):
            # Set breaks for every mapping, depending on their contig_id
            breaks = pd.DataFrame({'contig':np.arange(len(contigs)), 'start':sys.maxsize//2, 'end':0, 'group':-1})
            breaks.iloc[break_groups.loc[i==break_groups['num'], 'contig_id'], [breaks.columns.get_loc('start'), breaks.columns.get_loc('end')]] = break_groups.loc[i==break_groups['num'], ['start', 'end']].values
            breaks = breaks.iloc[non_informative_mappings['t_id']]
            # Find mappings that touch the previously set breaks (we could filter more, but this is the maximum set, so we don't declare informative mappings as non_informative_mappings, and the numbers we could filter more don't really matter speed wise)
            touching_breaks.append(non_informative_mappings[(non_informative_mappings['t_start'] <= breaks['end'].values) & (non_informative_mappings['t_end'] >= breaks['start'].values)].index)

        non_informative_mappings = np.setdiff1d(non_informative_mappings.index,np.concatenate(touching_breaks))
    else:
        # Without breaks no reads can overlap them
        non_informative_mappings = non_informative_mappings.index

    return non_informative_mappings

def CallAllBreaksSpurious(mappings, contigs, max_dist_contig_end, min_length_contig_break, min_extension, pdf):
    pot_breaks = GetBrokenMappings(mappings, max_dist_contig_end, min_length_contig_break, pdf)

    break_groups = []
    spurious_break_indexes = pot_breaks[['map_index','q_name','read_side','q_start','q_end']].drop_duplicates()
    non_informative_mappings = GetNonInformativeMappings(mappings, contigs, min_extension, break_groups, pot_breaks)

    return break_groups, spurious_break_indexes, non_informative_mappings

def NormCDFCapped(x, mu, sigma):
    return np.minimum(0.5, NormCDF(x, mu, sigma)) # Cap it at the mean so that repeats do not get a bonus

def GetConProb(cov_probs, req_length, counts):
    probs = pd.DataFrame({'length':req_length, 'counts':counts, 'mu':0.0, 'sigma':1.0})
    for plen, mu, sigma in zip(reversed(cov_probs['length']), reversed(cov_probs['mu']), reversed(cov_probs['sigma'])):
        probs.loc[plen >= probs['length'], 'mu'] = mu
        probs.loc[plen >= probs['length'], 'sigma'] = sigma

    return NormCDFCapped(probs['counts'],probs['mu'], probs['sigma'])      

def FindBreakPoints(mappings, contigs, max_dist_contig_end, min_mapping_length, min_length_contig_break, max_break_point_distance, min_num_reads, min_extension, merge_block_length, org_scaffold_trust, cov_probs, prob_factor, allow_same_contig_breaks, pdf):
    if pdf:
        loose_reads_ends = mappings[(mappings['t_start'] > max_dist_contig_end) & np.where('+' == mappings['strand'], -1 == mappings['prev_con'], -1 == mappings['next_con'])]
        loose_reads_ends_length = np.where('+' == loose_reads_ends['strand'], loose_reads_ends['q_start']-loose_reads_ends['read_start'], loose_reads_ends['read_end']-loose_reads_ends['q_end'])
        loose_reads_ends = mappings[(mappings['t_len']-mappings['t_end'] > max_dist_contig_end) & np.where('+' == mappings['strand'], -1 == mappings['next_con'], -1 == mappings['prev_con'])]
        loose_reads_ends_length = np.concatenate([loose_reads_ends_length, np.where('+' == loose_reads_ends['strand'], loose_reads_ends['read_end']-loose_reads_ends['q_end'], loose_reads_ends['q_start']-loose_reads_ends['read_start'])])
        if len(loose_reads_ends_length):
            if np.sum(loose_reads_ends_length < 10*min_length_contig_break):
                PlotHist(pdf, "Loose end length", "# Ends", np.extract(loose_reads_ends_length < 10*min_length_contig_break, loose_reads_ends_length), threshold=min_length_contig_break, logy=True)
            PlotHist(pdf, "Loose end length", "# Ends", loose_reads_ends_length, threshold=min_length_contig_break, logx=True)
#
    pot_breaks = GetBrokenMappings(mappings, max_dist_contig_end, min_length_contig_break, pdf)
    break_points = pot_breaks[['contig_id','side','position','mapq','map_len']].copy()
    break_points['connected'] = 0 <= pot_breaks['con']
    break_points = break_points[ break_points['map_len'] >= min_mapping_length+max_break_point_distance ].copy() # require half the mapping length for breaks as we will later require for continuously mapping reads that veto breaks, since breaking reads only map on one side
    break_points.drop(columns=['map_len'], inplace=True)
    break_points.sort_values(['contig_id','position'], inplace=True)
    if min_num_reads > 1:
        break_points = break_points[ ((break_points['contig_id'] == break_points['contig_id'].shift(-1, fill_value=-1)) & (break_points['position']+max_break_point_distance >= break_points['position'].shift(-1, fill_value=-1))) | ((break_points['contig_id'] == break_points['contig_id'].shift(1, fill_value=-1)) & (break_points['position']-max_break_point_distance <= break_points['position'].shift(1, fill_value=-1))) ].copy() # require at least one supporting break within +- max_break_point_distance
#
    if pdf:
        break_point_dist = (break_points['position'] - break_points['position'].shift(1, fill_value=0))[break_points['contig_id'] == break_points['contig_id'].shift(1, fill_value=-1)]
        if len(break_point_dist):
            if np.sum(break_point_dist <= 10*max_break_point_distance):
                PlotHist(pdf, "Break point distance", "# Break point pairs", break_point_dist[break_point_dist <= 10*max_break_point_distance], threshold=max_break_point_distance )
            PlotHist(pdf, "Break point distance", "# Break point pairs", break_point_dist, threshold=max_break_point_distance, logx=True )
#
    if len(break_points):
        # Check how many reads support a break_point with breaks within +- max_break_point_distance
        break_points['support'] = 1
        break_points['con_supp'] = break_points['connected'].astype(int)
        break_points.drop(columns=['connected'], inplace=True)
        break_points = break_points.groupby(['contig_id','position','mapq']).sum().reset_index()
        break_supp = []
        for direction in ["plus","minus"]:
            s = (1 if direction == "plus" else -1)
            con_loop = True
            while con_loop:
                tmp_breaks = break_points.copy()
                tmp_breaks['supp_pos'] = tmp_breaks['position'].shift(s, fill_value=-1)
                if direction == "plus":
                    tmp_breaks.loc[(tmp_breaks['contig_id'] != tmp_breaks['contig_id'].shift(s)) | (tmp_breaks['position'] > tmp_breaks['supp_pos']+max_break_point_distance), 'supp_pos'] = -1
                else:
                    tmp_breaks.loc[(tmp_breaks['contig_id'] != tmp_breaks['contig_id'].shift(s)) | (tmp_breaks['position'] < tmp_breaks['supp_pos']-max_break_point_distance), 'supp_pos'] = -1
                tmp_breaks = tmp_breaks[tmp_breaks['supp_pos'] != -1].copy()
                if len(tmp_breaks):
                    break_supp.append(tmp_breaks.copy())
                    s += (1 if direction == "plus" else -1)
                else:
                    con_loop = False
        break_supp = pd.concat(break_supp, ignore_index=True).drop_duplicates()
        break_supp = break_supp[break_supp['position'] != break_supp['supp_pos']].copy() # If we are at the same position we would multicount the values, when we do a cumsum later
        break_supp['position'] = break_supp['supp_pos']
        break_supp.drop(columns=['supp_pos'], inplace=True)
#
        break_points = pd.concat([break_points, break_supp], ignore_index=True).groupby(['contig_id','position','mapq']).sum().reset_index()
        break_points.sort_values(['contig_id','position','mapq'], ascending=[True,True,False], inplace=True)
        # Add support from higher mapping qualities to lower mapping qualities
        break_points['support'] = break_points.groupby(['contig_id','position'])['support'].cumsum()
        break_points['con_supp'] = break_points.groupby(['contig_id','position'])['con_supp'].cumsum()
        # Require a support of at least min_num_reads at some mapping quality level
        max_support = break_points.groupby(['contig_id','position'], sort=False)['support'].agg(['max','size'])
        break_points = break_points[ np.repeat(max_support['max'].values, max_support['size'].values) >= min_num_reads].copy()
#
    if len(break_points):
        # Check how many reads veto a break (continuously map over the break region position +- (min_mapping_length+max_break_point_distance))
        break_pos = break_points[['contig_id','position']].drop_duplicates()
        break_pos['merge_block'] = break_pos['position'] // merge_block_length
        break_pos['num'] = break_pos.groupby(['contig_id','merge_block'], sort=False).cumcount()
        tmp_mappings = mappings.loc[ mappings['t_end'] - mappings['t_start'] >= 2*min_mapping_length + 2*max_break_point_distance, ['t_id','t_start','t_end','mapq'] ].copy() # Do not consider mappings that cannot fullfil veto requirements because they are too short
        tmp_mappings = tmp_mappings.loc[ np.repeat( tmp_mappings.index.values, tmp_mappings['t_end']//merge_block_length - tmp_mappings['t_start']//merge_block_length + 1 ) ].copy()
        tmp_mappings['merge_block'] = tmp_mappings.reset_index().groupby(['index']).cumcount().values + tmp_mappings['t_start']//merge_block_length
        tmp_mappings.rename(columns={'t_id':'contig_id'}, inplace=True)
        break_list = []
        for n in range(break_pos['num'].max()+1):
            tmp_breaks = break_pos[break_pos['num'] == n]
            tmp_mappings = tmp_mappings[np.isin(tmp_mappings['contig_id'],tmp_breaks['contig_id'])].copy()
            # Create one entry for every mapping that potentially overlaps the break
            tmp_breaks = tmp_breaks.merge( tmp_mappings, on=['contig_id','merge_block'], how='inner')
            break_list.append( tmp_breaks[(tmp_breaks['t_start']+min_mapping_length+max_break_point_distance <= tmp_breaks['position']) &
                                          (tmp_breaks['t_end']-min_mapping_length-max_break_point_distance >= tmp_breaks['position'])].\
                                   groupby(['contig_id','position','mapq']).size().reset_index(name='vetos') )
#
        break_points = break_points.merge(pd.concat(break_list, ignore_index=True), on=['contig_id','position','mapq'], how='outer').fillna(0)
        break_points.sort_values(['contig_id','position','mapq'], ascending=[True,True,False], inplace=True)
        break_points.reset_index(inplace=True, drop=True)
        break_points[['support', 'con_supp']] = break_points.groupby(['contig_id','position'], sort=False)[['support', 'con_supp']].cummax().astype(int)
        break_points['vetos'] = break_points.groupby(['contig_id','position'], sort=False)['vetos'].cumsum().astype(int)
#
        # Remove breaks where the vetos are much more likely than the breaks
        break_points['break_prob'] = GetConProb(cov_probs, min_mapping_length+max_break_point_distance+min_length_contig_break, break_points['support'])
        break_points['veto_prob'] = GetConProb(cov_probs, 2*(min_mapping_length+max_break_point_distance), break_points['vetos'])
        break_points['vetoed'] = (break_points['vetos'] >= min_num_reads) & (break_points['veto_prob'] >= prob_factor*break_points['break_prob']) # Always make sure that vetos reach min_num_reads, before removing something based on their probability
        break_points['vetoed'] = break_points.groupby(['contig_id','position'], sort=False)['vetoed'].cumsum().astype(bool) # Propagate vetos to lower mapping qualities
        break_points = break_points[break_points['vetoed'] == False].copy()
#
        # Remove break points that will reconnected anyways
        break_points['vetoed'] = (break_points['vetos'] >= min_num_reads) & (break_points['con_supp'] < min_num_reads) # If we have enough support, but not enough con_supp the bridge finding step would anyways reseal the break with a unique bridge, so don't even break it
        break_points['vetoed'] = break_points.groupby(['contig_id','position'], sort=False)['vetoed'].cummax() # Propagate vetos to lower mapping qualities
        unconnected_break_points = [break_points[break_points['vetoed'] == True].copy()] # Store unconnected break points to later check if we can insert additional sequence to provide a connection in a second run
        break_points = break_points[break_points['vetoed'] == False].copy()
#
        # Remove break points with not enough consistent breaks
        break_pos = break_points[['contig_id','position']].drop_duplicates()
        break_pos['merge_block'] = break_pos['position'] // merge_block_length
        tmp_mappings = pot_breaks.loc[ (pot_breaks['t_end'] - pot_breaks['t_start'] >= min_mapping_length+max_break_point_distance) & (pot_breaks['con'] >= 0), ['contig_id','position','side','strand','mapq','con','con_strand','con_pos','con_dist'] ].copy() # Do not consider mappings that are too short
        tmp_mappings = tmp_mappings.loc[ np.repeat( tmp_mappings.index.values, (tmp_mappings['position']+max_break_point_distance)//merge_block_length - (tmp_mappings['position']-max_break_point_distance)//merge_block_length + 1 ) ].copy()
        tmp_mappings['merge_block'] = tmp_mappings.reset_index().groupby(['index']).cumcount().values + (tmp_mappings['position']-max_break_point_distance)//merge_block_length
        break_pos = break_pos.merge( tmp_mappings.rename(columns={'position':'bpos'}), on=['contig_id','merge_block'], how='inner').drop(columns=['merge_block'])
        break_pos = break_pos[ (break_pos['position']+max_break_point_distance >= break_pos['bpos']) & (break_pos['position']-max_break_point_distance <= break_pos['bpos']) ].drop(columns=['bpos'])
        break_pos['strand_switch'] = break_pos['strand'] != break_pos['con_strand']
        break_pos.sort_values(['contig_id','position','side','con','strand_switch','con_pos','mapq'], inplace=True)
        break_pos['group'] = ( (break_pos['contig_id'] != break_pos['contig_id'].shift(1)) | (break_pos['position'] != break_pos['position'].shift(1)) | (break_pos['side'] != break_pos['side'].shift(1)) |
                               (break_pos['con'] != break_pos['con'].shift(1)) | (break_pos['strand_switch'] != break_pos['strand_switch'].shift(1)) | (break_pos['con_pos'] > break_pos['con_pos'].shift(1)+max_break_point_distance) ).cumsum()
        break_pos = break_pos.groupby(['contig_id','position','group','mapq']).size().reset_index(name='count').merge( break_pos.groupby(['contig_id','position','group'])['con_dist'].mean().reset_index(name='mean_dist'), on=['contig_id','position','group'], how='inner')
        break_pos.sort_values(['group','mapq'], ascending=[True,False], inplace=True)
        break_pos['count'] = break_pos.groupby(['group'], sort=False)['count'].cumsum()
        break_pos = break_pos[ break_pos['count'] >= min_num_reads ].copy()
        break_points['vetoed'] = (break_points.merge( break_pos[['contig_id','position','mapq']].drop_duplicates(), on=['contig_id','position','mapq'], how='left', indicator=True )['_merge'] == "both").values # Start with setting vetoed to not vetoed ones for a following cummax
        break_points['vetoed'] = (break_points.groupby(['contig_id','position'], sort=False)['vetoed'].cummax() == False) & (break_points['vetos'] >= min_num_reads)
        break_points['vetoed'] = break_points.groupby(['contig_id','position'], sort=False)['vetoed'].cummax() # Propagate vetos to lower mapping qualities
        unconnected_break_points.append( break_points[break_points['vetoed'] == True].copy() )
        break_points = break_points[break_points['vetoed'] == False].copy()
#
        # Remove break points with not enough consistent breaks to be likely enough
        break_pos['con_prob'] = GetConProb(cov_probs, np.where(0 <= break_pos['mean_dist'], break_pos['mean_dist']+2*min_mapping_length, min_mapping_length-break_pos['mean_dist']), break_pos['count'])
        break_points = break_points.merge(break_pos.groupby(['contig_id','position','mapq'])['con_prob'].max().reset_index(name='con_prob'), on=['contig_id','position','mapq'], how='left')
        break_points['con_prob'] =  break_points.fillna(0.0).groupby(['contig_id','position'], sort=False)['con_prob'].cummax()
        break_points['vetoed'] = (break_points['vetos'] >= min_num_reads) & (break_points['veto_prob'] >= prob_factor*break_points['con_prob']) # Always make sure that vetos reach min_num_reads, before removing something based on their probability
        break_points['vetoed'] = break_points.groupby(['contig_id','position'], sort=False)['vetoed'].cummax() # Propagate vetos to lower mapping qualities
        unconnected_break_points.append( break_points[break_points['vetoed'] == True].drop(columns=['con_prob']) )
        break_points = break_points[break_points['vetoed'] == False].copy()
#
        # Remove breaks that do not fullfil requirements and reduce accepted ones to a single entry per break_id
        break_points = break_points[break_points['support'] >= min_num_reads].copy()
        break_points = break_points.groupby(['contig_id','position'], sort=False).first().reset_index()
#
    if len(break_points):
        # Cluster break_points into groups, so that groups have a maximum length of 2*max_break_point_distance
        break_points['dist'] = np.where( break_points['contig_id'] != break_points['contig_id'].shift(1, fill_value=-1), -1, break_points['position'] - break_points['position'].shift(1, fill_value=0) )
        break_points['break_id'] = range(len(break_points))
        break_points['group'] = pd.Series(np.where( (break_points['dist'] > 2*max_break_point_distance) | (-1 == break_points['dist']), break_points['break_id'], 0 )).cummax()
        break_points.loc[break_points['group'] != break_points['group'].shift(1, fill_value=-1), 'dist'] = -1
        break_groups = break_points.groupby(['contig_id','group'], sort=False)['position'].agg(['min','max','size']).reset_index()
        while np.sum(break_groups['max']-break_groups['min'] > 2*max_break_point_distance):
            # Break at the longest distance until the break_groups are below the maximum length
            break_points['group'] = pd.Series(np.where( np.repeat(break_groups['max']-break_groups['min'] <= 2*max_break_point_distance, break_groups['size']).values |
                                                        (break_points['dist'] < np.repeat(break_points.groupby(['contig_id','group'], sort=False)['dist'].max().values, break_groups['size'].values)), break_points['group'], break_points['break_id'])).cummax()
            break_points.loc[break_points['group'] != break_points['group'].shift(1, fill_value=-1), 'dist'] = -1
            break_groups = break_points.groupby(['contig_id','group'], sort=False)['position'].agg(['min','max','size']).reset_index()
#
        break_groups.drop(columns=['group','size'], inplace=True)
        break_groups.rename(columns={'min':'start', 'max':'end'}, inplace=True)
        break_groups['end'] += 1 # Set end to one position after the last included one to be consistent with other ends (like q_end, t_end)
        break_groups['pos'] = (break_groups['end']+break_groups['start'])//2
        break_groups['num'] = break_groups.groupby(['contig_id'], sort=False).cumcount()
    else:
        break_groups = []
#
    if len(break_groups):
        # Find mappings belonging to accepted breaks and mappings that have a connection to the same part of the contig (not split by breaks)
        pot_breaks['keep'] = False
        pot_breaks['self'] = allow_same_contig_breaks & (pot_breaks['contig_id'] == pot_breaks['con']) & (pot_breaks['strand'] == pot_breaks['con_strand'])
        for i in range(break_groups['num'].max()+1):
            breaks = pd.DataFrame({'contig':np.arange(len(contigs)), 'start':sys.maxsize, 'end':-1, 'pos':-1})
            breaks.loc[break_groups.loc[i==break_groups['num'], 'contig_id'].values, ['start','end','pos']] = break_groups.loc[i==break_groups['num'], ['start','end','pos']].values
            breaks = breaks.iloc[pot_breaks['contig_id'].values]
            pot_breaks['keep'] = ( pot_breaks['keep'] | ( (pot_breaks['side'] == 'l') & (breaks['pos'].values - max_dist_contig_end <= pot_breaks['position'].values) & (pot_breaks['position'].values <= breaks['end'].values) )
                                                      | ( (pot_breaks['side'] == 'r') & (breaks['start'].values <= pot_breaks['position'].values) & (pot_breaks['position'].values <= breaks['pos'].values + max_dist_contig_end) ) )
            pot_breaks['self'] = pot_breaks['self'] & ( ((pot_breaks['position'].values < breaks['pos'].values + np.where((pot_breaks['side'] == 'l'), 1, -1)*min_mapping_length) & (pot_breaks['con_pos'].values < breaks['pos'].values + np.where((pot_breaks['side'] == 'l'), -1, 1)*min_mapping_length)) |
                                                        ((pot_breaks['position'].values > breaks['pos'].values + np.where((pot_breaks['side'] == 'l'), 1, -1)*min_mapping_length) & (pot_breaks['con_pos'].values > breaks['pos'].values + np.where((pot_breaks['side'] == 'l'), -1, 1)*min_mapping_length)) )

        # A read is only correct in between two valid breaks/contig ends if all breaks in that region connect to another position inside that region (or has no breaks at all if allow_same_contig_breaks==False)
        pot_breaks.sort_values(['q_name','q_start','q_end','read_pos'], inplace=True)
        pot_breaks['group_id'] = ( (pot_breaks['q_name'] != pot_breaks['q_name'].shift(1)) | (pot_breaks['read_start'] != pot_breaks['read_start'].shift(1)) |
                                   (pot_breaks['contig_id'] != pot_breaks['contig_id'].shift(1)) |
                                   (pot_breaks['keep'] & (pot_breaks['read_pos'] == pot_breaks['q_start'])) |
                                   (pot_breaks['keep'] & (pot_breaks['read_pos'] == pot_breaks['q_end'])).shift(1, fill_value=False) ).cumsum()
        pot_breaks['self'] = pot_breaks['keep'] | pot_breaks['self']
        self_con = pot_breaks.groupby(['group_id'])['self'].agg(['sum','size']).reset_index()
        self_con = np.unique(self_con.loc[self_con['sum'] == self_con['size'], ['group_id']].values)
        pot_breaks['keep'] = np.isin(pot_breaks['group_id'], self_con)

        # Mappings not belonging to accepted breaks or connecting to itself are spurious
        spurious_break_indexes = pot_breaks.loc[pot_breaks['keep'] == False, ['map_index','q_name','read_side','q_start','q_end']].drop_duplicates()
    else:
        # We don't have breaks, so all are spurious
        spurious_break_indexes = pot_breaks[['map_index','q_name','read_side','q_start','q_end']].drop_duplicates()

    non_informative_mappings = GetNonInformativeMappings(mappings, contigs, min_extension, break_groups, pot_breaks)
#
    # Filter unconnected break points that overlap a break_group (probably from vetoed low mapping qualities, where the high mapping quality was not vetoed)
    unconnected_break_points = pd.concat(unconnected_break_points, ignore_index=True)[['contig_id','position']].sort_values(['contig_id','position']).drop_duplicates()
    unconnected_break_points['remove'] = False
    for i in range(break_groups['num'].max()+1):
        breaks = pd.DataFrame({'contig':np.arange(len(contigs)), 'start':sys.maxsize, 'end':-1})
        breaks.iloc[break_groups.loc[i==break_groups['num'], 'contig_id'], [breaks.columns.get_loc('start'), breaks.columns.get_loc('end')]] = break_groups.loc[i==break_groups['num'], ['start', 'end']].values
        breaks = breaks.iloc[unconnected_break_points['contig_id']]
        unconnected_break_points['remove'] = unconnected_break_points['remove'] | ( (breaks['start'].values-max_break_point_distance <= unconnected_break_points['position'].values) & (unconnected_break_points['position'].values <= breaks['end'].values+max_break_point_distance) )
    unconnected_break_points = unconnected_break_points[unconnected_break_points['remove'] == False].drop(columns=['remove'])
#
    # Handle unconnected break points
    unconnected_break_points['dist'] = np.where( unconnected_break_points['contig_id'] != unconnected_break_points['contig_id'].shift(1, fill_value=-1), -1, unconnected_break_points['position'] - unconnected_break_points['position'].shift(1, fill_value=0) )
    unconnected_break_points['group'] = ((unconnected_break_points['dist'] > max_break_point_distance) | (-1 == unconnected_break_points['dist'])).cumsum()
    unconnected_break_points = unconnected_break_points.groupby(['contig_id','group'])['position'].agg(['min','max']).reset_index().rename(columns={'min':'bfrom','max':'bto'}).drop(columns=['group'])
    unconnected_break_points = unconnected_break_points.merge(pot_breaks, on=['contig_id'], how='left')
    unconnected_break_points = unconnected_break_points[(unconnected_break_points['position'] >= unconnected_break_points['bfrom'] - max_break_point_distance) & (unconnected_break_points['position'] <= unconnected_break_points['bto'] + max_break_point_distance)].copy()
#
    # If the unconnected break points are connected to another contig on the other side, the extension will likely take care of it and we don't need to do anything
    unconnected_break_points.sort_values(['contig_id','bfrom','bto','side','opos_con'], inplace=True)
    count_breaks = unconnected_break_points.groupby(['contig_id','bfrom','bto','side','opos_con'], sort=False).size()
    unconnected_break_points['ocon_count'] = np.where(unconnected_break_points['opos_con'] < 0, 0, np.repeat(count_breaks.values, count_breaks.values))
    count_breaks = unconnected_break_points.groupby(['contig_id','bfrom','bto','side'], sort=False)['ocon_count'].agg(['max','size'])
    unconnected_break_points['ocon_count'] = np.repeat(count_breaks['max'].values, count_breaks['size'].values)
    unconnected_break_points = unconnected_break_points[(unconnected_break_points['ocon_count'] < min_num_reads) & (np.repeat(count_breaks['size'].values, count_breaks['size'].values) >= min_num_reads)].copy()

    return break_groups, spurious_break_indexes, non_informative_mappings, unconnected_break_points

def SplitReadsAtSpuriousBreakIndexes(mappings, spurious_break_indexes):
    if len(spurious_break_indexes):
        # Handle multiple breaks in the same read one after another
        split_reads = spurious_break_indexes[['q_name','read_side','q_start','q_end']].sort_values(['q_name','q_start'])
        split_reads.drop_duplicates(inplace=True)
        split_reads.rename(columns={'read_side':'split_side','q_start':'split_start','q_end':'split_end'}, inplace=True)
        split_reads['bcount'] = split_reads.groupby(['q_name'], sort=False).cumcount()
        split_reads = split_reads.merge(mappings[['q_name']].drop_duplicates(), on=['q_name'], how='inner')
        for b in range(split_reads['bcount'].max()+1):
            # Split reads at split_pos
            mappings[['split_side','split_start','split_end']] = mappings[['q_name']].merge(split_reads.loc[split_reads['bcount'] == b, ['q_name','split_side','split_start','split_end']], on=['q_name'], how='left')[['split_side','split_start','split_end']].values
            cur = ((mappings['split_side'] == 'l') & (mappings['q_start'] < mappings['split_start'])) | ((mappings['split_side'] == 'r') & (mappings['q_end'] <= mappings['split_end']))
            if np.sum(cur):
                mappings.loc[cur, 'read_end'] = np.minimum(mappings.loc[cur, 'read_end'].values, mappings.loc[cur, 'split_end'].astype(int).values)
            cur = ((mappings['split_side'] == 'l') & (mappings['q_start'] >= mappings['split_start'])) | ((mappings['split_side'] == 'r') & (mappings['q_end'] > mappings['split_end']))
            if np.sum(cur):
                mappings.loc[cur, 'read_start'] = np.maximum(mappings.loc[cur, 'read_start'].values, mappings.loc[cur, 'split_start'].astype(int).values)
            # Make sure mappings do not reach over read start/end
            splitted = mappings.loc[np.isnan(mappings['split_start']) == False, ['q_name','q_start','q_end','read_start','read_end']].groupby(['q_name','read_start','read_end']).agg({'q_start':['min'],'q_end':['max']}).droplevel(axis=1,level=1).reset_index()
            splitted = splitted[(splitted['read_start'] > splitted['q_start']) | (splitted['read_end'] < splitted['q_end'])].rename(columns={'q_start':'new_start','q_end':'new_end'})
            splitted['new_start'] = np.minimum(splitted['new_start'], splitted['read_start'])
            splitted['new_end'] = np.maximum(splitted['new_end'], splitted['read_end'])
            mappings[['new_start','new_end']] = mappings[['q_name','read_start','read_end']].merge(splitted, on=['q_name','read_start','read_end'], how='left')[['new_start','new_end']].values
            mappings.loc[np.isnan(mappings['new_start']) == False, 'read_start'] =  mappings.loc[np.isnan(mappings['new_start']) == False, 'new_start'].astype(int)
            mappings.loc[np.isnan(mappings['new_end']) == False, 'read_end'] =  mappings.loc[np.isnan(mappings['new_end']) == False, 'new_end'].astype(int)
        mappings.drop(columns=['split_side','split_start','split_end','new_start','new_end'], inplace=True)
#
    return mappings

def GetContigParts(contigs, break_groups, remove_short_contigs, min_mapping_length, alignment_precision, pdf):
    # Create a dataframe with the contigs being split into parts and contigs scheduled for removal not included
    if 0 == len(break_groups):
        # We do not have break points, so we don't need to split
        contigs['parts'] = 1
    else:
        contig_parts = break_groups.groupby('contig_id').size().reset_index(name='counts')
        contigs['parts'] = 0
        contigs.iloc[contig_parts['contig_id'], contigs.columns.get_loc('parts')] = contig_parts['counts'].values
        contigs['parts'] += 1

    contigs.loc[contigs['remove'], 'parts'] = 0
    
#   if pdf:
#        category = np.array(["deleted no coverage"]*len(contigs))
#        category[:] = "used"
#        category[ contigs['parts']>1 ] = "split"
#        category[ contigs['repeat_mask_right'] == 0 ] = "masked"
#        category[ (contigs['repeat_mask_right'] == 0) & contigs['remove'] ] = "deleted duplicate"
#        category[ (contigs['repeat_mask_right'] > 0) & contigs['remove'] ] = "deleted no coverage"

#        PlotHist(pdf, "Original contig length", "# Contigs", contigs['length'], category=category, catname="type", logx=True)
        
    contig_parts = pd.DataFrame({'contig':np.repeat(np.arange(len(contigs)), contigs['parts']), 'part':1, 'start':0})
    contig_parts['part'] = contig_parts.groupby('contig').cumsum()['part']-1
    if len(break_groups):
        for i in range(break_groups['num'].max()+1):
            contig_parts.loc[contig_parts['part']==i+1, 'start'] = break_groups.loc[break_groups['num']==i,'pos'].values
    contig_parts['end'] = contig_parts['start'].shift(-1,fill_value=0)
    contig_parts.loc[contig_parts['end']==0, 'end'] = contigs.iloc[contig_parts.loc[contig_parts['end']==0, 'contig'], contigs.columns.get_loc('length')].values
    contig_parts['name'] = contigs.iloc[contig_parts['contig'], contigs.columns.get_loc('name')].values

    # Remove short contig_parts < min_mapping_length + alignment_precision (adding alignment_precision gives a buffer so that a mapping to a short contig is not removed in one read and not removed in another based on a single base mapping or not)
    if remove_short_contigs:
        contig_parts = contig_parts[contig_parts['end']-contig_parts['start'] >= min_mapping_length + alignment_precision].copy()
        contig_parts['part'] = contig_parts.groupby(['contig'], sort=False).cumcount()
        contig_parts.reset_index(drop=True,inplace=True)
        contigs['parts'] = 0
        contig_sizes = contig_parts.groupby(['contig'], sort=False).size()
        contigs.loc[contig_sizes.index.values, 'parts'] = contig_sizes.values
    
    contigs['first_part'] = -1
    contigs.loc[contigs['parts']>0, 'first_part'] = contig_parts[contig_parts['part']==0].index
    contigs['last_part'] = contigs['first_part'] + contigs['parts'] - 1
    
    # Assign scaffold info from contigs to contig_parts
    contig_parts['org_dist_left'] = -1
    tmp_contigs = contigs[(contigs['parts']>0) & (contigs['parts'].shift(1, fill_value=-1)>0)]
    contig_parts.iloc[tmp_contigs['first_part'].values, contig_parts.columns.get_loc('org_dist_left')] = tmp_contigs['org_dist_left'].values
    contig_parts['org_dist_right'] = -1
    tmp_contigs = contigs[(contigs['parts']>0) & (contigs['parts'].shift(-1, fill_value=-1)>0)]
    contig_parts.iloc[tmp_contigs['last_part'].values, contig_parts.columns.get_loc('org_dist_right')] = tmp_contigs['org_dist_right'].values
    
    # Rescue scaffolds broken by removed contigs
    tmp_contigs = contigs[(contigs['parts']==0) & (contigs['org_dist_right'] != -1) & (contigs['org_dist_left'] != -1)].copy()
    tmp_contigs.reset_index(inplace=True)
    tmp_contigs['group'] = (tmp_contigs['index'] != tmp_contigs['index'].shift(1, fill_value=-1)).cumsum() # Group deleted contigs which are connected, so that we can pass the connectiong through multiple removed contigs
    tmp_contigs = tmp_contigs.groupby('group', sort=False)[['index','length','org_dist_right','org_dist_left']].agg(['first','last','sum'])
    tmp_contigs['from'] = tmp_contigs[('index','first')]-1
    tmp_contigs['to'] = tmp_contigs[('index','last')]+1
    tmp_contigs['len'] = tmp_contigs[('org_dist_left','first')] + tmp_contigs[('length','sum')] + tmp_contigs[('org_dist_right','sum')]
    tmp_contigs['from_part'] = contigs.iloc[tmp_contigs['from'], contigs.columns.get_loc('last_part')].values
    tmp_contigs['to_part'] = contigs.iloc[tmp_contigs['to'], contigs.columns.get_loc('first_part')].values

    tmp_contigs = tmp_contigs[(tmp_contigs['from_part'] >= 0) & (tmp_contigs['to_part'] >= 0)].copy()
    contig_parts.iloc[tmp_contigs['from_part'], contig_parts.columns.get_loc('org_dist_right')] = tmp_contigs['len'].values
    contig_parts.iloc[tmp_contigs['to_part'], contig_parts.columns.get_loc('org_dist_left')] = tmp_contigs['len'].values
    
    # Insert the break info as 0-length gaps
    contig_parts.loc[contig_parts['part'] > 0, 'org_dist_left'] = 0
    contig_parts.loc[contig_parts['part'].shift(-1, fill_value=0) > 0, 'org_dist_right'] = 0
    
    # Assign repeat connections
#    contig_parts['rep_con_left'] = -1
#    contig_parts['rep_con_side_left'] = ''
#    contig_parts['rep_con_right'] = -1
#    contig_parts['rep_con_side_right'] = ''
#    tmp_contigs = contigs[(contigs['parts']>0)]
#    contig_parts.iloc[tmp_contigs['first_part'].values, contig_parts.columns.get_loc('rep_con_left')] = np.where(tmp_contigs['rep_con_side_left'] == 'l',
#                                                          contigs.iloc[tmp_contigs['rep_con_left'].values, contigs.columns.get_loc('first_part')].values,
#                                                          contigs.iloc[tmp_contigs['rep_con_left'].values, contigs.columns.get_loc('last_part')].values)
#    contig_parts.iloc[tmp_contigs['first_part'].values, contig_parts.columns.get_loc('rep_con_side_left')] = tmp_contigs['rep_con_side_left'].values
#    contig_parts.iloc[tmp_contigs['last_part'].values, contig_parts.columns.get_loc('rep_con_right')] = np.where(tmp_contigs['rep_con_side_right'] == 'l',
#                                                          contigs.iloc[tmp_contigs['rep_con_right'].values, contigs.columns.get_loc('first_part')].values,
#                                                          contigs.iloc[tmp_contigs['rep_con_right'].values, contigs.columns.get_loc('last_part')].values)
#    contig_parts.iloc[tmp_contigs['last_part'].values, contig_parts.columns.get_loc('rep_con_side_right')] = tmp_contigs['rep_con_side_right'].values

    return contig_parts, contigs

def GetBreakAndRemovalInfo(result_info, contigs, contig_parts):
    part_count = contig_parts.groupby('contig', sort=False).size().reset_index(name='parts')
    part_count['breaks'] = part_count['parts']-1
    result_info['breaks'] = {}
    result_info['breaks']['opened'] = np.sum(part_count['breaks'])
    
    removed_length = contigs.loc[ np.isin(contigs.reset_index()['index'], part_count['contig']) == False, 'length' ].values
    result_info['removed'] = {}
    result_info['removed']['num'] = len(removed_length)
    result_info['removed']['total'] = np.sum(removed_length) if len(removed_length) else 0
    result_info['removed']['min'] = np.min(removed_length) if len(removed_length) else 0
    result_info['removed']['max'] = np.max(removed_length) if len(removed_length) else 0
    result_info['removed']['mean'] = np.mean(removed_length) if len(removed_length) else 0
    
    return result_info

def UpdateMappingsToContigParts(mappings, contig_parts, min_mapping_length, max_dist_contig_end, min_extension):
    # Duplicate mappings for every contig part (using inner, because some contigs have been removed (e.g. being too short)) 
    mappings = mappings.merge(contig_parts.reset_index()[['contig', 'index', 'part', 'start', 'end']].rename(columns={'contig':'t_id', 'index':'conpart', 'start':'part_start', 'end':'part_end'}), on=['t_id'], how='inner')

    # Remove mappings that do not touch a contig part or only less than min_mapping_length bases
    mappings = mappings[(mappings['t_start']+min_mapping_length < mappings['part_end']) & (mappings['t_end']-min_mapping_length > mappings['part_start'])].copy()

    # Update mapping information with limits from contig part
    mappings['con_from'] = np.maximum(mappings['t_start'], mappings['part_start'])  # We cannot work with 0 < mappings['num_part'], because of reads that start before mappings['con_start'] but do not cross the break region and therefore don't have a mapping before
    mappings['con_to'] = np.minimum(mappings['t_end'], mappings['part_end'])

    mappings['read_from'] = mappings['q_start']
    mappings['read_to'] = mappings['q_end']

    # Rough estimate of the split for the reads, better not use the new information, but not always avoidable
    # Forward strand
    multi_maps = mappings[(mappings['con_from'] > mappings['t_start']) & (mappings['strand'] == '+')]
    mappings.loc[(mappings['con_from'] > mappings['t_start']) & (mappings['strand'] == '+'), 'read_from'] = np.round(multi_maps['q_start'] + (multi_maps['q_end']-multi_maps['q_start'])/(multi_maps['t_end']-multi_maps['t_start'])*(multi_maps['con_from']-multi_maps['t_start'])).astype(int)
    multi_maps = mappings[(mappings['con_to'] < mappings['t_end']) & (mappings['strand'] == '+')]
    mappings.loc[(mappings['con_to'] < mappings['t_end']) & (mappings['strand'] == '+'), 'read_to'] = np.round(multi_maps['q_end'] - (multi_maps['q_end']-multi_maps['q_start'])/(multi_maps['t_end']-multi_maps['t_start'])*(multi_maps['t_end']-multi_maps['con_to'])).astype(int)
    # Reverse strand
    multi_maps = mappings[(mappings['con_from'] > mappings['t_start']) & (mappings['strand'] == '-')]
    mappings.loc[(mappings['con_from'] > mappings['t_start']) & (mappings['strand'] == '-'), 'read_to'] = np.round(multi_maps['q_end'] - (multi_maps['q_end']-multi_maps['q_start'])/(multi_maps['t_end']-multi_maps['t_start'])*(multi_maps['con_from']-multi_maps['t_start'])).astype(int)
    multi_maps = mappings[(mappings['con_to'] < mappings['t_end']) & (mappings['strand'] == '-')]
    mappings.loc[(mappings['con_to'] < mappings['t_end']) & (mappings['strand'] == '-'), 'read_from'] = np.round(multi_maps['q_start'] + (multi_maps['q_end']-multi_maps['q_start'])/(multi_maps['t_end']-multi_maps['t_start'])*(multi_maps['t_end']-multi_maps['con_to'])).astype(int)

    mappings['matches'] = np.round(mappings['matches']/(mappings['t_end']-mappings['t_start'])*(mappings['con_to']-mappings['con_from'])).astype(int) # Rough estimate use with care!
    mappings.sort_values(['q_name','read_start','read_from','read_to'], inplace=True)
    mappings.reset_index(inplace=True, drop=True)

    # Update connections
    mappings['next_con'] = np.where((mappings['q_name'].shift(-1, fill_value='') != mappings['q_name']) | (mappings['read_start'].shift(-1, fill_value=-1) != mappings['read_start']), -1, mappings['conpart'].shift(-1, fill_value=-1))
    mappings['next_strand'] = np.where(-1 == mappings['next_con'], '', mappings['strand'].shift(-1, fill_value=''))
    mappings['prev_con'] = np.where((mappings['q_name'].shift(1, fill_value='') != mappings['q_name']) | (mappings['read_start'].shift(1, fill_value=-1) != mappings['read_start']), -1, mappings['conpart'].shift(1, fill_value=-1))
    mappings['prev_strand'] = np.where(-1 == mappings['prev_con'], '', mappings['strand'].shift(1, fill_value=''))

    mappings['left_con'] = np.where('+' == mappings['strand'], mappings['prev_con'], mappings['next_con'])
    mappings['right_con'] = np.where('-' == mappings['strand'], mappings['prev_con'], mappings['next_con'])

    mappings['next_side'] = np.where('+' == mappings['next_strand'], 'l', 'r')
    mappings['prev_side'] = np.where('-' == mappings['prev_strand'], 'l', 'r')
    mappings['left_con_side'] = np.where('+' == mappings['strand'], mappings['prev_side'], mappings['next_side'])
    mappings.loc[-1 == mappings['left_con'], 'left_con_side'] = ''
    mappings['right_con_side'] = np.where('-' == mappings['strand'], mappings['prev_side'], mappings['next_side'])
    mappings.loc[-1 == mappings['right_con'], 'right_con_side'] = ''

    # Remove connections that are not starting within max_dist_contig_end or are consistent with a contig part
    mappings['left'] = mappings['con_from'] <= mappings['part_start']+max_dist_contig_end
    mappings['right'] = mappings['con_to'] >= mappings['part_end']-max_dist_contig_end
    mappings['remleft'] = (( (mappings['left'] == False) | ((mappings['left_con_side'] == 'r') & (False == np.where(mappings['strand'] == '+', mappings['right'].shift(1), mappings['right'].shift(-1))))
                                                        | ((mappings['left_con_side'] == 'l') & (False == np.where(mappings['strand'] == '+', mappings['left'].shift(1), mappings['left'].shift(-1)))) 
                          | ((mappings['left_con'] == mappings['conpart']) & (mappings['left_con_side'] == 'r')
                             & (mappings['con_from'] >= np.where(mappings['strand'] == '+', mappings['con_to'].shift(1), mappings['con_to'].shift(-1)))) ) & (mappings['left_con'] > 0))
    mappings['remright'] = ( ( (mappings['right'] == False) | ((mappings['right_con_side'] == 'r') & (False == np.where(mappings['strand'] == '+', mappings['right'].shift(-1), mappings['right'].shift(1))))
                                                          | ((mappings['right_con_side'] == 'l') & (False == np.where(mappings['strand'] == '+', mappings['left'].shift(-1), mappings['left'].shift(1)))) 
                          | ((mappings['right_con'] == mappings['conpart']) & (mappings['right_con_side'] == 'l')
                             & (mappings['con_to'] < np.where(mappings['strand'] == '+', mappings['con_from'].shift(-1), mappings['con_from'].shift(1)))) ) & (mappings['right_con'] > 0))
    mappings.loc[mappings['remleft'], 'left_con'] = -1
    mappings.loc[mappings['remleft'], 'left_con_side'] = ''
    mappings.loc[mappings['remright'], 'right_con'] = -1
    mappings.loc[mappings['remright'], 'right_con_side'] = ''

    # Remove mappings that only connect a contig part to itself without extensions or contig part duplications
    remove = ( (mappings['remleft'] | (mappings['left'] == False) | ((mappings['left_con'] < 0) & (min_extension+mappings['con_from']-mappings['part_start'] > np.where(mappings['strand'] == '+', mappings['read_from'] - mappings['read_start'], mappings['read_end']-mappings['read_to'])))) &
               (mappings['remright'] | (mappings['right'] == False) | ((mappings['right_con'] < 0) & (min_extension+mappings['part_end']-mappings['con_to'] > np.where(mappings['strand'] == '-', mappings['read_from'] - mappings['read_start'], mappings['read_end']-mappings['read_to'])))) )

    # Get distance to connected contigparts
    mappings['next_dist'] = mappings['read_from'].shift(-1, fill_value=0) - mappings['read_to']
    mappings['prev_dist'] = mappings['read_from'] - mappings['read_to'].shift(1, fill_value=0)
    mappings['left_con_dist'] = np.where('+' == mappings['strand'], mappings['prev_dist'], mappings['next_dist'])
    mappings.loc[-1 == mappings['left_con'], 'left_con_dist'] = 0
    mappings['right_con_dist'] = np.where('-' == mappings['strand'], mappings['prev_dist'], mappings['next_dist'])
    mappings.loc[-1 == mappings['right_con'], 'right_con_dist'] = 0

    # Select the columns we still need
    mappings.rename(columns={'q_name':'read_name'}, inplace=True)
    mappings = mappings.loc[remove == False, ['read_name', 'read_start', 'read_end', 'read_from', 'read_to', 'strand', 'conpart', 'con_from', 'con_to', 'left_con', 'left_con_side', 'left_con_dist', 'right_con', 'right_con_side', 'right_con_dist', 'mapq', 'matches']]

    # Count how many mappings each read has
    mappings['num_mappings'] = 1
    num_mappings = mappings.groupby(['read_name','read_start'], sort=False)['num_mappings'].size().values
    mappings['num_mappings'] = np.repeat(num_mappings, num_mappings)

    return mappings

def CreateBridges(left_bridge, right_bridge, min_distance_tolerance, rel_distance_tolerance):
    bridges = pd.concat([left_bridge,right_bridge], ignore_index=True, sort=False)

    # Duplicate bridges and switch from to, so that both sortings have all relevant bridges
    bridges = pd.concat([bridges[['from','from_side','to','to_side','mapq','distance']],
                         bridges.rename(columns={'from':'to', 'to':'from', 'from_side':'to_side', 'to_side':'from_side'})[['from','from_side','to','to_side','mapq','distance']]], ignore_index=True, sort=False)

    # Separate bridges if distances are far apart
    bridges.sort_values(['from','from_side','to','to_side','distance'], inplace=True)
    bridges['group'] = ( (bridges['from'] != bridges['from'].shift(1)) | (bridges['from_side'] != bridges['from_side'].shift(1)) |
                         (bridges['to'] != bridges['to'].shift(1)) | (bridges['to_side'] != bridges['to_side'].shift(1)) |
                         (bridges['distance'] > bridges['distance'].shift(1)+min_distance_tolerance+rel_distance_tolerance*np.maximum(np.abs(bridges['distance']),np.abs(bridges['distance'].shift(1)))) ).cumsum()

    # Bundle identical bridges
    bridged_dists = bridges.groupby(['from','from_side','to','to_side','group'], sort=False)['distance'].agg(['mean','min','max']).reset_index().rename(columns={'mean':'mean_dist','min':'min_dist','max':'max_dist'})
    bridged_dists['mean_dist'] = np.round(bridged_dists['mean_dist']).astype(int)
    bridges = bridges.groupby(['from','from_side','to','to_side','group','mapq']).size().reset_index(name='count')

    # Get cumulative counts (counts for this trust level and higher)
    bridges['count'] = bridges['count'] // 2 # Not counting all reads twice (because each bridge is bridged from both sides)
    bridges.sort_values(['from','from_side','to','to_side','group','mapq'], ascending=[True, True, True, True, True, False], inplace=True)
    bridges['cumcount'] = bridges.groupby(['from','from_side','to','to_side','group'], sort=False)['count'].cumsum().values
    bridges.drop(columns=['count'], inplace=True)

    # Add distances back in
    bridges = bridges.merge(bridged_dists, on=['from','from_side','to','to_side','group'], how='left')
    bridges.drop(columns=['group'], inplace=True) # We don't need the group anymore because it is encoded in ['from','from_side','to','to_side','mean_dist'] now

    return bridges

def MarkOrgScaffoldBridges(bridges, contig_parts, requirement):
    # requirement (-1: org scaffold, 0: unbroken org scaffold)
    bridges['org_scaffold'] = False
    bridges.loc[(bridges['from']+1 == bridges['to']) & ('r' == bridges['from_side']) & ('l' == bridges['to_side']) & (requirement < contig_parts.iloc[bridges['from'].values, contig_parts.columns.get_loc('org_dist_right')].values), 'org_scaffold'] = True
    bridges.loc[(bridges['from']-1 == bridges['to']) & ('l' == bridges['from_side']) & ('r' == bridges['to_side']) & (requirement < contig_parts.iloc[bridges['from'].values, contig_parts.columns.get_loc('org_dist_left')].values), 'org_scaffold'] = True

    return bridges

def CountAlternatives(bridges):
    bridges.sort_values(['to', 'to_side','from','from_side'], inplace=True)
    alternatives = bridges.groupby(['to','to_side'], sort=False).size().values
    bridges['to_alt'] = np.repeat(alternatives, alternatives)
    bridges.sort_values(['from','from_side','to','to_side'], inplace=True)
    alternatives = bridges.groupby(['from','from_side'], sort=False).size().values
    bridges['from_alt'] = np.repeat(alternatives, alternatives)

    return bridges

def FilterBridges(bridges, borderline_removal, min_factor_alternatives, min_num_reads, org_scaffold_trust, cov_probs, prob_factor, min_mapping_length, contig_parts, pdf=None):
    if org_scaffold_trust in ["blind", "full"]:
        bridges = MarkOrgScaffoldBridges(bridges, contig_parts, -1)
        if "blind" == org_scaffold_trust:
            # Make sure the org scaffolds are used if any read is there and do not use any other connection there even if we don't have a read
            bridges['org_scaf_conflict'] = False
            bridges.loc[('r' == bridges['from_side']) & (-1 != contig_parts.iloc[bridges['from'].values, contig_parts.columns.get_loc('org_dist_right')].values), 'org_scaf_conflict'] = True
            bridges.loc[('l' == bridges['from_side']) & (-1 != contig_parts.iloc[bridges['from'].values, contig_parts.columns.get_loc('org_dist_left')].values), 'org_scaf_conflict'] = True
            bridges.loc[('r' == bridges['to_side']) & (-1 != contig_parts.iloc[bridges['to'].values, contig_parts.columns.get_loc('org_dist_right')].values), 'org_scaf_conflict'] = True
            bridges.loc[('l' == bridges['to_side']) & (-1 != contig_parts.iloc[bridges['to'].values, contig_parts.columns.get_loc('org_dist_left')].values), 'org_scaf_conflict'] = True

            bridges = bridges[(False == bridges['org_scaf_conflict']) | bridges['org_scaffold']]

            bridges.drop(columns=['org_scaf_conflict'], inplace=True)
    elif "basic" == org_scaffold_trust:
        bridges = MarkOrgScaffoldBridges(bridges, contig_parts, 0) # Do not connect previously broken contigs through low quality reads
    else:
        bridges['org_scaffold'] = False # Do not use original scaffolds

    # Set lowq flags for all bridges that don't fulfill min_num_reads except for the ones with additional support from original scaffolds
    bridges['low_q'] = (bridges['cumcount'] < min_num_reads)

    # Remove lowq for the distance with the highest counts of each original scaffolds
    bridges.sort_values(['from','from_side','to','to_side','mapq','cumcount'], ascending=[True, True, True, True,False, False], inplace=True)
    bridges['max_count'] = bridges.groupby(['from','from_side','to','to_side','mapq'], sort=False)['cumcount'].cummax()
    bridges.loc[ bridges['org_scaffold'] & (bridges['max_count'] == bridges['cumcount']), 'low_q'] = False
    bridges.drop(columns=['max_count'], inplace=True)

    if borderline_removal:
        # Set lowq flags for all bridges that are not above min_factor_alternatives compared to other lowq bridges (so that we do not have borderline decisions where +-1 count decides that one bridge is accepted and the other not)
        old_num_lowq = np.sum(bridges['low_q'])
        bridges.sort_values(['from','from_side','mapq','cumcount'], ascending=[True, True, False, True], inplace=True)
        bridges['from_group'] = ( (bridges['from'] != bridges['from'].shift(1)) | (bridges['from_side'] != bridges['from_side'].shift(1)) | bridges['org_scaffold'] |
                                  (bridges['mapq'] != bridges['mapq'].shift(1)) | (bridges['cumcount'] > np.ceil(bridges['cumcount'].shift(1, fill_value = 0)*min_factor_alternatives)) ).astype(int).cumsum()
        bridges['low_q'] = bridges.groupby(['from_group'], sort=False)['low_q'].cumsum().astype(bool)

        bridges.sort_values(['to','to_side','mapq','cumcount'], ascending=[True, True, False, True], inplace=True)
        bridges['to_group'] = ( (bridges['to'] != bridges['to'].shift(1)) | (bridges['to_side'] != bridges['to_side'].shift(1)) | bridges['org_scaffold'] |
                              (bridges['mapq'] != bridges['mapq'].shift(1)) | (bridges['cumcount'] > np.ceil(bridges['cumcount'].shift(1, fill_value = 0)*min_factor_alternatives)) ).astype(int).cumsum()
        bridges['low_q'] = bridges.groupby(['to_group'], sort=False)['low_q'].cumsum().astype(bool)

        num_lowq = np.sum(bridges['low_q'])
        while old_num_lowq != num_lowq:
            bridges.sort_values(['from_group','cumcount'], ascending=[True, True], inplace=True)
            bridges['low_q'] = bridges.groupby(['from_group'], sort=False)['low_q'].cumsum().astype(bool)
            bridges.sort_values(['to_group','cumcount'], ascending=[True, True], inplace=True)
            bridges['low_q'] = bridges.groupby(['to_group'], sort=False)['low_q'].cumsum().astype(bool)
            old_num_lowq = num_lowq
            num_lowq = np.sum(bridges['low_q'])

        bridges.drop(columns=['from_group','to_group'], inplace=True)

    # Remove low quality bridges
    bridges = bridges[bridges['low_q'] == False].copy()
    bridges.drop(columns=['low_q'], inplace=True)

    # Set highq flags for the most likely bridges and the ones within prob_factor of the most likely
    bridges['probability'] = GetConProb(cov_probs, np.where(0 <= bridges['mean_dist'], bridges['mean_dist']+2*min_mapping_length, min_mapping_length-bridges['mean_dist']), bridges['cumcount'])
    bridges.sort_values(['to','to_side','mapq'], inplace=True)
    alternatives = bridges.groupby(['to','to_side','mapq'], sort=False)['probability'].agg(['max','size'])
    bridges['to_max_prob'] = np.repeat(alternatives['max'].values, alternatives['size'].values)
    bridges.sort_values(['from','from_side','mapq'], inplace=True)
    alternatives = bridges.groupby(['from','from_side','mapq'], sort=False)['probability'].agg(['max','size'])
    bridges['from_max_prob'] = np.repeat(alternatives['max'].values, alternatives['size'].values)
    bridges['highq'] = ((bridges['probability']*prob_factor >= bridges['from_max_prob']) & (bridges['probability']*prob_factor >= bridges['to_max_prob']))

    # Only keep high quality bridges
    bridges = bridges[bridges['highq']].copy()
    bridges.drop(columns=['highq','from_max_prob','to_max_prob'], inplace=True)

    #  Remove bridges that compete with a bridge with a higher trust level
    bridges.sort_values(['from','from_side','mapq'], ascending=[True, True, False], inplace=True)
    bridges['max_mapq'] = bridges.groupby(['from','from_side'], sort=False)['mapq'].cummax()
    bridges.sort_values(['to','to_side','mapq'], ascending=[True, True, False], inplace=True)
    bridges['max_mapq2'] = bridges.groupby(['to','to_side'], sort=False)['mapq'].cummax()
    bridges = bridges[np.maximum(bridges['max_mapq'], bridges['max_mapq2']) <= bridges['mapq']].copy()
    bridges.drop(columns=['max_mapq','max_mapq2'], inplace=True)

    if "full" == org_scaffold_trust:
        # Remove ambiguous bridges that compeat with the original scaffold
        bridges.sort_values(['from','from_side','to','to_side'], inplace=True)
        org_scaffolds = bridges.groupby(['from','from_side'], sort=False)['org_scaffold'].agg(['size','sum'])
        bridges['org_from'] = np.repeat(org_scaffolds['sum'].values, org_scaffolds['size'].values)
        bridges.sort_values(['to','to_side','from','from_side'], inplace=True)
        org_scaffolds = bridges.groupby(['to','to_side'], sort=False)['org_scaffold'].agg(['size','sum'])
        bridges['org_to'] = np.repeat(org_scaffolds['sum'].values, org_scaffolds['size'].values)
        bridges = bridges[ bridges['org_scaffold'] | ((0 == bridges['org_from']) & (0 == bridges['org_to'])) ].copy()

    # Count alternatives
    bridges.drop(columns=['org_scaffold'], inplace=True)
    bridges = CountAlternatives(bridges)
    bridges['from'] = bridges['from'].astype(int)
    bridges['to'] = bridges['to'].astype(int)
    bridges.rename(columns={'cumcount':'bcount'}, inplace=True)

    return bridges

def GetBridges(mappings, borderline_removal, min_factor_alternatives, min_num_reads, org_scaffold_trust, contig_parts, cov_probs, prob_factor, min_mapping_length, min_distance_tolerance, rel_distance_tolerance, pdf):
    # Get bridges
    left_bridge = mappings.loc[mappings['left_con'] >= 0, ['conpart','left_con','left_con_side','left_con_dist','mapq']].copy()
    left_bridge.rename(columns={'conpart':'from','left_con':'to','left_con_side':'to_side','mapq':'from_mapq','left_con_dist':'distance'}, inplace=True)
    left_bridge['from_side'] = 'l'
    left_bridge['to_mapq'] = np.where('+' == mappings['strand'], mappings['mapq'].shift(1, fill_value = -1), mappings['mapq'].shift(-1, fill_value = -1))[mappings['left_con'] >= 0]
    left_bridge['mapq'] = np.where(left_bridge['to_mapq'] < left_bridge['from_mapq'], left_bridge['to_mapq'].astype(int)*1000+left_bridge['from_mapq'], left_bridge['from_mapq'].astype(int)*1000+left_bridge['to_mapq'])
    left_bridge.drop(columns=['from_mapq','to_mapq'], inplace=True)
    left_bridge.loc[left_bridge['from'] > left_bridge['to'], ['from', 'from_side', 'to', 'to_side']] = left_bridge.loc[left_bridge['from'] > left_bridge['to'], ['to', 'to_side', 'from', 'from_side']].values
#
    right_bridge = mappings.loc[mappings['right_con'] >= 0, ['conpart','right_con','right_con_side','right_con_dist','mapq']].copy()
    right_bridge.rename(columns={'conpart':'from','right_con':'to','right_con_side':'to_side','mapq':'from_mapq','right_con_dist':'distance'}, inplace=True)
    right_bridge['from_side'] = 'r'
    right_bridge['to_mapq'] = np.where('-' == mappings['strand'], mappings['mapq'].shift(1, fill_value = -1), mappings['mapq'].shift(-1, fill_value = -1))[mappings['right_con'] >= 0]
    right_bridge['mapq'] = np.where(right_bridge['to_mapq'] < right_bridge['from_mapq'], right_bridge['to_mapq'].astype(int)*1000+right_bridge['from_mapq'], right_bridge['from_mapq'].astype(int)*1000+right_bridge['to_mapq'])
    right_bridge.drop(columns=['from_mapq','to_mapq'], inplace=True)
    right_bridge.loc[right_bridge['from'] >= right_bridge['to'], ['from', 'from_side', 'to', 'to_side']] = right_bridge.loc[right_bridge['from'] >= right_bridge['to'], ['to', 'to_side', 'from', 'from_side']].values # >= here so that contig loops are also sorted properly

    bridges = CreateBridges(left_bridge, right_bridge, min_distance_tolerance, rel_distance_tolerance)
    bridges = FilterBridges(bridges, borderline_removal, min_factor_alternatives, min_num_reads, org_scaffold_trust, cov_probs, prob_factor, min_mapping_length, contig_parts, pdf)

    return bridges

def ScaffoldAlongGivenConnections(scaffolds, scaffold_parts):
    # Handle right-right scaffold connections (keep scaffold with lower id and reverse+add the other one): We can only create new r-r or r-l connections
    keep = scaffolds.loc[(scaffolds['rscaf_side'] == 'r') & (scaffolds['scaffold'] < scaffolds['rscaf']), 'scaffold'].values
    while len(keep):
        scaffolds.loc[keep, 'size'] += scaffolds.loc[scaffolds.loc[keep, 'rscaf'].values, 'size'].values
        # Update scaffold_parts
        absorbed = np.isin(scaffold_parts['scaffold'], scaffolds.loc[keep, 'rscaf'])
        scaffold_parts.loc[absorbed, 'reverse'] = scaffold_parts.loc[absorbed, 'reverse'] == False
        absorbed_values = scaffold_parts.merge(scaffolds.loc[keep, ['scaffold','rscaf','size']].rename(columns={'scaffold':'new_scaf','rscaf':'scaffold'}), on=['scaffold'], how='left').loc[absorbed, ['new_scaf','size']]
        scaffold_parts.loc[absorbed, 'pos'] = absorbed_values['size'].astype(int).values - scaffold_parts.loc[absorbed, 'pos'].values - 1
        scaffold_parts.loc[absorbed, 'scaffold'] = absorbed_values['new_scaf'].astype(int).values
        # Update scaffolds (except 'size' which was updated before)
        absorbed = scaffolds.loc[keep, 'rscaf'].values
        scaffolds.loc[keep, 'right'] = scaffolds.loc[absorbed, 'left'].values
        scaffolds.loc[keep, 'rside'] = scaffolds.loc[absorbed, 'lside'].values
        scaffolds.loc[keep, 'rextendible'] = scaffolds.loc[absorbed, 'lextendible'].values
        new_scaffold_ids = scaffolds.loc[keep, ['scaffold','rscaf']].rename(columns={'scaffold':'new_scaf','rscaf':'scaffold'}).copy() # Store this to later change the connections to the removed scaffold
        scaffolds.loc[keep, 'rscaf'] = scaffolds.loc[absorbed, 'lscaf'].values
        scaffolds.loc[keep, 'rscaf_side'] = scaffolds.loc[absorbed, 'lscaf_side'].values
        # Drop removed scaffolds and update the connections
        scaffolds.drop(absorbed, inplace=True)
        scaffolds = scaffolds.merge(new_scaffold_ids.rename(columns={'scaffold':'lscaf'}), on=['lscaf'], how='left')
        scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'lscaf_side'] = 'r'
        scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'lscaf'] = scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'new_scaf'].astype(int)
        scaffolds.drop(columns=['new_scaf'], inplace=True)
        scaffolds = scaffolds.merge(new_scaffold_ids.rename(columns={'scaffold':'rscaf'}), on=['rscaf'], how='left')
        scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'rscaf_side'] = 'r'
        scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'rscaf'] = scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'new_scaf'].astype(int)
        scaffolds.drop(columns=['new_scaf'], inplace=True)
        scaffolds.index = scaffolds['scaffold'].values # Make sure we can access scaffolds with .loc[scaffold]
        # Prepare next round
        keep = scaffolds.loc[(scaffolds['rscaf_side'] == 'r') & (scaffolds['scaffold'] < scaffolds['rscaf']), 'scaffold'].values

    # Handle left-left scaffold connections (keep scaffold with lower id and reverse+add the other one): We can only create new l-l or l-r connections
    keep = scaffolds.loc[(scaffolds['lscaf_side'] == 'l') & (scaffolds['scaffold'] < scaffolds['lscaf']), 'scaffold'].values
    while len(keep):
        # Update scaffold_parts
        absorbed = np.isin(scaffold_parts['scaffold'], scaffolds.loc[keep, 'lscaf'])
        scaffold_parts.loc[absorbed, 'reverse'] = scaffold_parts.loc[absorbed, 'reverse'] == False
        scaffold_parts.loc[absorbed, 'scaffold'] = scaffold_parts.merge(scaffolds.loc[keep, ['scaffold','lscaf']].rename(columns={'scaffold':'new_scaf','lscaf':'scaffold'}), on=['scaffold'], how='left').loc[absorbed, 'new_scaf'].astype(int).values
        scaffold_parts.loc[absorbed, 'pos'] = scaffold_parts.loc[absorbed, 'pos']*-1 - 1
        scaffold_parts.sort_values(['scaffold','pos'], inplace=True)
        scaffold_parts['pos'] = scaffold_parts.groupby(['scaffold'], sort=False).cumcount() # Shift positions so that we don't have negative numbers anymore
        # Update scaffolds
        absorbed = scaffolds.loc[keep, 'lscaf'].values
        scaffolds.loc[keep, 'left'] = scaffolds.loc[absorbed, 'right'].values
        scaffolds.loc[keep, 'lside'] = scaffolds.loc[absorbed, 'rside'].values
        scaffolds.loc[keep, 'lextendible'] = scaffolds.loc[absorbed, 'rextendible'].values
        scaffolds.loc[keep, 'size'] += scaffolds.loc[absorbed, 'size'].values
        new_scaffold_ids = scaffolds.loc[keep, ['scaffold','lscaf']].rename(columns={'scaffold':'new_scaf','lscaf':'scaffold'}).copy() # Store this to later change the connections to the removed scaffold
        scaffolds.loc[keep, 'lscaf'] = scaffolds.loc[absorbed, 'rscaf'].values
        scaffolds.loc[keep, 'lscaf_side'] = scaffolds.loc[absorbed, 'rscaf_side'].values
        # Drop removed scaffolds and update the connections
        scaffolds.drop(absorbed, inplace=True)
        scaffolds = scaffolds.merge(new_scaffold_ids.rename(columns={'scaffold':'lscaf'}), on=['lscaf'], how='left')
        scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'lscaf_side'] = 'l'
        scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'lscaf'] = scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'new_scaf'].astype(int)
        scaffolds.drop(columns=['new_scaf'], inplace=True)
        scaffolds = scaffolds.merge(new_scaffold_ids.rename(columns={'scaffold':'rscaf'}), on=['rscaf'], how='left')
        scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'rscaf_side'] = 'l'
        scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'rscaf'] = scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'new_scaf'].astype(int)
        scaffolds.drop(columns=['new_scaf'], inplace=True)
        scaffolds.index = scaffolds['scaffold'].values # Make sure we can access scaffolds with .loc[scaffold]
        # Prepare next round
        keep = scaffolds.loc[(scaffolds['lscaf_side'] == 'l') & (scaffolds['scaffold'] < scaffolds['lscaf']), 'scaffold'].values

    # Remove simple ciruclarities
    circular_scaffolds = scaffolds['scaffold'] == scaffolds['lscaf']
    scaffolds.loc[circular_scaffolds & (scaffolds['lscaf'] == scaffolds['rscaf']), 'circular'] = True # Truely circular it is only for left-right connections
    scaffolds.loc[circular_scaffolds, 'lextendible'] = False
    scaffolds.loc[circular_scaffolds, 'lscaf'] = -1
    scaffolds.loc[circular_scaffolds, 'lscaf_side'] = ''
    circular_scaffolds = scaffolds['scaffold'] == scaffolds['rscaf']
    scaffolds.loc[circular_scaffolds, 'rextendible'] = False
    scaffolds.loc[circular_scaffolds, 'rscaf'] = -1
    scaffolds.loc[circular_scaffolds, 'rscaf_side'] = ''

    # Check for two-contig circularities (right-left connections: Others don't exist anymore)
    circular_scaffolds = (scaffolds['lscaf'] == scaffolds['rscaf']) & (scaffolds['lscaf'] != -1)
    scaffolds.loc[circular_scaffolds, 'circular'] = True
    if np.sum(circular_scaffolds):
        break_left = np.unique(np.minimum(scaffolds.loc[circular_scaffolds,'lscaf'].values.astype(int), scaffolds.loc[circular_scaffolds,'scaffold'].values)) # Choose arbitrarily the lower of the scaffold ids for left side break
        break_right = np.unique(np.maximum(scaffolds.loc[circular_scaffolds,'lscaf'].values.astype(int), scaffolds.loc[circular_scaffolds,'scaffold'].values))
        scaffolds.loc[break_left, 'lextendible'] = False
        scaffolds.loc[break_left, 'lscaf'] = -1
        scaffolds.loc[break_left, 'lscaf_side'] = ''
        scaffolds.loc[break_right, 'rextendible'] = False
        scaffolds.loc[break_right, 'rscaf'] = -1
        scaffolds.loc[break_right, 'rscaf_side'] = ''

    # Handle left-right scaffold connections (Follow the right-left connections until no connection exists anymore)
    while np.sum(scaffolds['rscaf_side'] == 'l'):
        keep = scaffolds.loc[(scaffolds['rscaf_side'] == 'l') & (scaffolds['lscaf_side'] == ''), 'scaffold'].values
        if 0 == len(keep):
            # All remaining right-left connections must be circular
            # Long circularities (>=3 scaffolds) cannot be solved in a single go, because we don't know which ones are connected. Thus we simply break the left connection of the first scaffold that still has a connection and run the loop until the whole circularity is scaffolded and repeat this until no more circularities are present
            circular_scaffolds = scaffolds.loc[scaffolds['rscaf_side'] == 'l', 'scaffold'].values
            scaffolds.loc[circular_scaffolds, 'circular'] = True
            if len(circular_scaffolds):
                break_left = circular_scaffolds[0]
                break_right = scaffolds.loc[break_left, 'lscaf']
                scaffolds.loc[break_left, 'lextendible'] = False
                scaffolds.loc[break_left, 'lscaf'] = -1
                scaffolds.loc[break_left, 'lscaf_side'] = ''
                scaffolds.loc[break_right, 'rextendible'] = False
                scaffolds.loc[break_right, 'rscaf'] = -1
                scaffolds.loc[break_right, 'rscaf_side'] = ''

            # Update the scaffolds that are kept and extended with another scaffold
            keep = scaffolds.loc[(scaffolds['rscaf_side'] == 'l') & (scaffolds['lscaf_side'] == ''), 'scaffold'].values

        # Update scaffold_parts
        absorbed = np.isin(scaffold_parts['scaffold'], scaffolds.loc[keep, 'rscaf'])
        absorbed_values = scaffold_parts.merge(scaffolds.loc[keep, ['scaffold','rscaf','size']].rename(columns={'scaffold':'new_scaf','rscaf':'scaffold'}), on=['scaffold'], how='left').loc[absorbed, ['new_scaf','size']]
        scaffold_parts.loc[absorbed, 'pos'] = absorbed_values['size'].astype(int).values + scaffold_parts.loc[absorbed, 'pos'].values
        scaffold_parts.loc[absorbed, 'scaffold'] = absorbed_values['new_scaf'].astype(int).values
        # Update scaffolds
        absorbed = scaffolds.loc[keep, 'rscaf'].values
        scaffolds.loc[keep, 'right'] = scaffolds.loc[absorbed, 'right'].values
        scaffolds.loc[keep, 'rside'] = scaffolds.loc[absorbed, 'rside'].values
        scaffolds.loc[keep, 'rextendible'] = scaffolds.loc[absorbed, 'rextendible'].values
        scaffolds.loc[keep, 'size'] += scaffolds.loc[absorbed, 'size'].values
        new_scaffold_ids = scaffolds.loc[keep, ['scaffold','rscaf']].rename(columns={'scaffold':'new_scaf','rscaf':'scaffold'}).copy() # Store this to later change the connections to the absorbed scaffold
        scaffolds.loc[keep, 'rscaf'] = scaffolds.loc[absorbed, 'rscaf'].values
        scaffolds.loc[keep, 'rscaf_side'] = scaffolds.loc[absorbed, 'rscaf_side'].values
        # Drop absorbed scaffolds and update the connections
        scaffolds.drop(absorbed, inplace=True)
        scaffolds = scaffolds.merge(new_scaffold_ids.rename(columns={'scaffold':'lscaf'}), on=['lscaf'], how='left')
        scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'lscaf'] = scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'new_scaf'].astype(int)
        scaffolds.drop(columns=['new_scaf'], inplace=True)
        scaffolds = scaffolds.merge(new_scaffold_ids.rename(columns={'scaffold':'rscaf'}), on=['rscaf'], how='left')
        scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'rscaf'] = scaffolds.loc[np.isnan(scaffolds['new_scaf']) == False, 'new_scaf'].astype(int)
        scaffolds.drop(columns=['new_scaf'], inplace=True)
        scaffolds.index = scaffolds['scaffold'].values # Make sure we can access scaffolds with .loc[scaffold]

    return scaffolds, scaffold_parts

def AddDistaneInformation(scaffold_parts, unique_bridges):
    scaffold_parts.sort_values(['scaffold','pos'], inplace=True)
    scaffold_parts['from'] = scaffold_parts['conpart']
    scaffold_parts['from_side'] = np.where(scaffold_parts['reverse'], 'l', 'r')
    scaffold_parts['to'] = np.where(scaffold_parts['scaffold'].shift(-1) == scaffold_parts['scaffold'], scaffold_parts['conpart'].shift(-1, fill_value=-1), -1)
    scaffold_parts['to_side'] = np.where(scaffold_parts['reverse'].shift(-1), 'r', 'l') # This is bogus if scaffolds do not match, but 'to' is correct and prevents the merge anyways
    scaffold_parts = scaffold_parts.merge(unique_bridges[['from','from_side','to','to_side','mean_dist']].rename(columns={'mean_dist':'next_dist'}), on=['from','from_side','to','to_side'], how='left')
    scaffold_parts['next_dist'] = scaffold_parts['next_dist'].fillna(0).astype(int)
    scaffold_parts.index = scaffold_parts['conpart'].values # Set the index again, after the merge reset it
    scaffold_parts.drop(columns=['from','from_side','to','to_side'], inplace=True)
    scaffold_parts['prev_dist'] = np.where(scaffold_parts['scaffold'].shift(1) == scaffold_parts['scaffold'], scaffold_parts['next_dist'].shift(1, fill_value=0), 0)

    return scaffold_parts

def LiftBridgesFromContigsToScaffolds(bridges, scaffolds):
    # Lift from
    scaf_bridges = bridges.merge(scaffolds[['scaffold','left','lside']].rename(columns={'scaffold':'lscaf', 'left':'from', 'lside':'from_side'}), on=['from','from_side'], how='left')
    scaf_bridges = scaf_bridges.merge(scaffolds[['scaffold','right','rside']].rename(columns={'scaffold':'rscaf', 'right':'from', 'rside':'from_side'}), on=['from','from_side'], how='left')
    scaf_bridges = scaf_bridges[(np.isnan(scaf_bridges['lscaf']) == False) | (np.isnan(scaf_bridges['rscaf']) == False)].copy() # Remove bridges that are now inside scaffolds
    scaf_bridges['from'] = np.where(np.isnan(scaf_bridges['lscaf']), scaf_bridges['rscaf'], scaf_bridges['lscaf']).astype(int)
    scaf_bridges['from_side'] = np.where(np.isnan(scaf_bridges['lscaf']), 'r', 'l')
    scaf_bridges.drop(columns=['lscaf','rscaf'], inplace=True)

    # Lift to
    scaf_bridges = scaf_bridges.merge(scaffolds[['scaffold','left','lside']].rename(columns={'scaffold':'lscaf', 'left':'to', 'lside':'to_side'}), on=['to','to_side'], how='left')
    scaf_bridges = scaf_bridges.merge(scaffolds[['scaffold','right','rside']].rename(columns={'scaffold':'rscaf', 'right':'to', 'rside':'to_side'}), on=['to','to_side'], how='left')
    scaf_bridges['to'] = np.where(np.isnan(scaf_bridges['lscaf']), scaf_bridges['rscaf'], scaf_bridges['lscaf']).astype(int)
    scaf_bridges['to_side'] = np.where(np.isnan(scaf_bridges['lscaf']), 'r', 'l')
    scaf_bridges.drop(columns=['lscaf','rscaf'], inplace=True)

    return scaf_bridges

def GetOriginalScaffoldConnections(contig_parts, scaffolds):
    org_scaf_conns = scaffolds[['scaffold','left','lside','right','rside']].copy()
    org_scaf_conns['org_dist_left'] = np.where(org_scaf_conns['lside'] == 'l', contig_parts.loc[org_scaf_conns['left'].values, 'org_dist_left'].values, contig_parts.loc[org_scaf_conns['left'].values, 'org_dist_right'].values)
    org_scaf_conns['org_dist_right'] = np.where(org_scaf_conns['rside'] == 'l', contig_parts.loc[org_scaf_conns['right'].values, 'org_dist_left'].values, contig_parts.loc[org_scaf_conns['right'].values, 'org_dist_right'].values)
    org_scaf_conns = [ org_scaf_conns.loc[org_scaf_conns['org_dist_left'] >= 0, ['scaffold','left','lside','org_dist_left']].rename(columns={'left':'conpart','lside':'cside','org_dist_left':'distance'}),
                       org_scaf_conns.loc[org_scaf_conns['org_dist_right'] >= 0, ['scaffold','right','rside','org_dist_right']].rename(columns={'right':'conpart','rside':'cside','org_dist_right':'distance'}) ]
    org_scaf_conns[0]['side'] = 'l'
    org_scaf_conns[1]['side'] = 'r'
    org_scaf_conns = pd.concat(org_scaf_conns, ignore_index=True, sort=False)
    org_scaf_conns['conpart1'] = org_scaf_conns['conpart'] - np.where(org_scaf_conns['cside'] == 'l', 1, 0)
    org_scaf_conns['conpart2'] = org_scaf_conns['conpart'] + np.where(org_scaf_conns['cside'] == 'r', 1, 0)
    org_scaf_conns.drop(columns=['conpart','cside'], inplace=True)
    org_scaf_conns = org_scaf_conns.rename(columns={'scaffold':'from','side':'from_side'}).merge(org_scaf_conns.rename(columns={'scaffold':'to','side':'to_side'}), on=['conpart1','conpart2','distance'], how='left')
    org_scaf_conns = org_scaf_conns.loc[org_scaf_conns['from'] != org_scaf_conns['to'], ['from','from_side','to','to_side','distance']].copy()
#
    return org_scaf_conns

def GetSizeAndEnumeratePositionsForLongRangeConnections(long_range_connections):
    con_size = long_range_connections.groupby(['conn_id'], sort=False).size().reset_index(name='size')
    long_range_connections['size'] = np.repeat(con_size['size'].values, con_size['size'].values)
    long_range_connections['pos'] = long_range_connections.groupby(['conn_id']).cumcount()

    return long_range_connections

def GetConnectionFromTo(long_range_connections):
    long_range_connections['from'] = np.where(long_range_connections['conn_id'] != long_range_connections['conn_id'].shift(1, fill_value=-1), -1, long_range_connections['conpart'].shift(1, fill_value=-1))
    long_range_connections['from_side'] = np.where(long_range_connections['strand'].shift(1, fill_value='') == '+', 'r', 'l') # When the previous entry does not belong to same conn_id this is garbage, but 'from' is already preventing the merge happening next, so it does not matter
    long_range_connections['to'] = long_range_connections['conpart']
    long_range_connections['to_side'] = np.where(long_range_connections['strand'] == '+', 'l', 'r')

    return long_range_connections

def GetLongRangeConnections(bridges, mappings):
    # Get long_range_mappings that include a bridge that has alternatives
    long_range_mappings = mappings[mappings['num_mappings']>=3].copy()
    if len(long_range_mappings):
        alternative_connections = bridges.loc[(bridges['from_alt'] > 1) | (bridges['to_alt'] > 1), ['from','from_side']].rename(columns={'from':'conpart','from_side':'side'}).drop_duplicates()
        interesting_reads = long_range_mappings[['read_name','read_start','conpart','right_con','left_con']].merge(alternative_connections, on=['conpart'], how='inner')
        interesting_reads = interesting_reads[ ((interesting_reads['side'] == 'r') & (interesting_reads['right_con'] >= 0)) | ((interesting_reads['side'] == 'l') & (interesting_reads['left_con'] >= 0)) ].copy() # Only keep reads that have a connection in the interesting direction
        long_range_mappings = long_range_mappings.merge(interesting_reads[['read_name','read_start']].drop_duplicates(), on=['read_name','read_start'], how='inner')

    if len(long_range_mappings):
        # Get long_range_connections that are supported by reads
        long_range_connections = long_range_mappings[['conpart','strand','left_con_dist','right_con_dist','left_con','right_con']].copy()
        long_range_connections['conn_id'] = ((long_range_mappings['read_name'] != long_range_mappings['read_name'].shift(1, fill_value='')) | (long_range_mappings['read_start'] != long_range_mappings['read_start'].shift(1, fill_value=-1))).cumsum()
    else:
        long_range_connections = []

    if len(long_range_connections):
        # If a long_range_connections goes through the same contig part multiple times without a proper repeat signature, merge those entries
        long_range_connections = GetConnectionFromTo(long_range_connections)
        long_range_connections['fake_rep'] = (long_range_connections['from'] == long_range_connections['to']) & (0 > np.where(long_range_connections['to_side'] == 'l', long_range_connections['left_con'], long_range_connections['right_con']) ) # The connection was removed due to its non-repeat signature
        remove = long_range_connections['fake_rep'] & long_range_connections['fake_rep'].shift(-1, fill_value=False) & (long_range_connections['conn_id'] == long_range_connections['conn_id'].shift(-1) )
        if np.sum(remove):
            # Remove the middle contig in case we have two fake repeat connections in a row
            long_range_connections = long_range_connections[remove == False].copy()
            GetConnectionFromTo(long_range_connections)
        selection = long_range_connections['fake_rep'] & (long_range_connections['from_side'] == 'r')
        long_range_connections.loc[selection.shift(-1, fill_value=False), 'right_con_dist'] = np.where(long_range_connections.loc[selection, 'to_side'] == 'l', long_range_connections.loc[selection, 'right_con_dist'], long_range_connections.loc[selection, 'left_con_dist']) # Fill in the other side that does not point towards the previous entry
        selection = long_range_connections['fake_rep'] & (long_range_connections['from_side'] == 'l')
        long_range_connections.loc[selection.shift(-1, fill_value=False), 'left_con_dist'] = np.where(long_range_connections.loc[selection, 'to_side'] == 'l', long_range_connections.loc[selection, 'right_con_dist'], long_range_connections.loc[selection, 'left_con_dist']) # Fill in the other side that does not point towards the previous entry
        long_range_connections = long_range_connections[long_range_connections['fake_rep'] == False].drop(columns = ['left_con','right_con','fake_rep'])

    if len(long_range_connections):
        # Break long_range_mappings when they go through invalid bridges and get number of prev alternatives (how many alternative to the connection of a mapping with its previous mapping exist)
        long_range_connections = GetConnectionFromTo(long_range_connections)
        long_range_connections['dist'] = np.where(long_range_connections['strand'] == '+', long_range_connections['left_con_dist'], long_range_connections['right_con_dist'])
        long_range_connections = long_range_connections.reset_index().merge(bridges[['from','from_side','to','to_side','to_alt','min_dist','max_dist','mean_dist']], on=['from','from_side','to','to_side'], how='left').fillna(0) # Keep index, so we know where we introduced duplicates with multiple possible distances for the bridges
        long_range_connections.rename(columns={'to_alt':'prev_alt', 'mean_dist':'prev_dist'}, inplace=True)
        long_range_connections['prev_alt'] = long_range_connections['prev_alt'].astype(int)
        long_range_connections['prev_dist'] = long_range_connections['prev_dist'].astype(int) # We store only the mean_dist of the distance category for later, since the exact distances are irrelevant and varying when we group connections
        long_range_connections.loc[(long_range_connections['min_dist'] > long_range_connections['dist']) | (long_range_connections['dist'] > long_range_connections['max_dist']),'prev_alt'] = 0 # Break connections that do not fit the distance
        long_range_connections = long_range_connections.sort_values(['index','prev_alt'], ascending=[True,False]).groupby(['index'], sort=False).first().reset_index() # Remove duplications from multiple possible bridge distances
        long_range_connections.drop(columns=['min_dist','max_dist'], inplace=True)
        long_range_connections['conn_id'] = (long_range_connections['prev_alt'] == 0).cumsum()

        # Remove connections that do not include at least 3 mappings anymore
        long_range_connections = GetSizeAndEnumeratePositionsForLongRangeConnections(long_range_connections)
        long_range_connections = long_range_connections[ long_range_connections['size'] >= 3 ].copy()

    if len(long_range_connections):
        # Get number of next alternatives
        long_range_connections['from'] = long_range_connections['conpart']
        long_range_connections['from_side'] = np.where(long_range_connections['strand'] == '+', 'r', 'l')
        long_range_connections['dist'] = np.where(long_range_connections['strand'] == '+', long_range_connections['right_con_dist'], long_range_connections['left_con_dist'])
        long_range_connections['to'] = np.where(long_range_connections['conn_id'] != long_range_connections['conn_id'].shift(-1, fill_value=-1), -1, long_range_connections['conpart'].shift(-1, fill_value=-1))
        long_range_connections['to_side'] = np.where(long_range_connections['strand'].shift(-1, fill_value='') == '+', 'l', 'r') # When the next entry does not belong to same conn_id this is garbage, but 'to' is already preventing the merge happening next, so it does not matter
        long_range_connections = long_range_connections.merge(bridges[['from','from_side','to','to_side','from_alt','min_dist','max_dist','mean_dist']], on=['from','from_side','to','to_side'], how='left').fillna(0)
        long_range_connections.rename(columns={'from_alt':'next_alt', 'mean_dist':'next_dist'}, inplace=True)
        long_range_connections['next_alt'] = long_range_connections['next_alt'].astype(int)
        long_range_connections['next_dist'] = long_range_connections['next_dist'].astype(int) # We store only the mean_dist of the distance category for later, since the exact distances are irrelevant and varying when we group connections
        long_range_connections.loc[(long_range_connections['min_dist'] > long_range_connections['dist']) | (long_range_connections['dist'] > long_range_connections['max_dist']),'next_alt'] = 0 # Mark wrong distances, so we can remove them and only keep the correct ones
        long_range_connections = long_range_connections.sort_values(['index','next_alt'], ascending=[True,False]).groupby(['index'], sort=False).first().reset_index(drop=True) # Remove duplications from multiple possible bridge distances
        long_range_connections.drop(columns=['from','from_side','to','to_side','dist','min_dist','max_dist','left_con_dist','right_con_dist'], inplace=True)

        # Trim parts at the beginning or end that do not include alternative connections
        long_range_connections['trim'] = ( ((long_range_connections['pos'] == 0) & (long_range_connections['next_alt'] == 1) & (long_range_connections['prev_alt'].shift(-1, fill_value=-1) == 1)) |
                                           ((long_range_connections['pos'] == long_range_connections['size']-1) & (long_range_connections['prev_alt'] == 1) & (long_range_connections['next_alt'].shift(1, fill_value=-1) == 1)) )
        while np.sum(long_range_connections['trim']):
            long_range_connections = long_range_connections[long_range_connections['trim'] == False].copy()
            long_range_connections = GetSizeAndEnumeratePositionsForLongRangeConnections(long_range_connections)
            long_range_connections = long_range_connections[ long_range_connections['size'] >= 3 ].copy()
            long_range_connections['trim'] = ( ((long_range_connections['pos'] == 0) & (long_range_connections['next_alt'] == 1) & (long_range_connections['prev_alt'].shift(-1, fill_value=-1) == 1)) |
                                               ((long_range_connections['pos'] == long_range_connections['size']-1) & (long_range_connections['prev_alt'] == 1) & (long_range_connections['next_alt'].shift(1, fill_value=-1) == 1)) )

    if len(long_range_connections):
        long_range_connections.drop(columns=['trim'], inplace=True)
        long_range_connections.loc[long_range_connections['pos'] == 0, ['prev_alt','prev_dist']] = 0 # Make sure that reads look similar independent if they were trimmed or not
        long_range_connections.loc[long_range_connections['pos'] == long_range_connections['size']-1, ['next_alt','next_dist']] = 0

        # Add all connections also in the reverse direction
        long_range_connections = long_range_connections.loc[np.repeat(long_range_connections.index.values, 2)].copy()
        long_range_connections['reverse'] = [False,True] * (len(long_range_connections)//2)
        long_range_connections.loc[long_range_connections['reverse'], 'strand'] = np.where(long_range_connections.loc[long_range_connections['reverse'], 'strand'] == '+', '-', '+')
        long_range_connections.loc[long_range_connections['reverse'], 'conn_id'] = long_range_connections.loc[long_range_connections['reverse'], 'conn_id'] + long_range_connections['conn_id'].max()
        tmp_alt = long_range_connections.loc[long_range_connections['reverse'], 'prev_alt']
        long_range_connections.loc[long_range_connections['reverse'], 'prev_alt'] = long_range_connections.loc[long_range_connections['reverse'], 'next_alt']
        long_range_connections.loc[long_range_connections['reverse'], 'next_alt'] = tmp_alt
        tmp_dist = long_range_connections.loc[long_range_connections['reverse'], 'prev_dist']
        long_range_connections.loc[long_range_connections['reverse'], 'prev_dist'] = long_range_connections.loc[long_range_connections['reverse'], 'next_dist']
        long_range_connections.loc[long_range_connections['reverse'], 'next_dist'] = tmp_dist
        long_range_connections.loc[long_range_connections['reverse'], 'pos'] = long_range_connections.loc[long_range_connections['reverse'], 'size'] - long_range_connections.loc[long_range_connections['reverse'], 'pos'] - 1
        long_range_connections.drop(columns=['reverse'], inplace=True)
        long_range_connections.sort_values(['conn_id', 'pos'], inplace=True)

        # Summarize identical long_range_connections
        long_range_connections['conn_code'] = long_range_connections['conpart'].astype(str) + long_range_connections['strand'] + '(' + long_range_connections['next_dist'].astype(str) + ')'
        codes = long_range_connections.groupby(['conn_id'], sort=False)['conn_code'].apply(''.join)
        long_range_connections['conn_code'] = codes.loc[long_range_connections['conn_id'].values].values
        long_range_connections = long_range_connections.groupby(['conn_code','pos','size','conpart','strand','prev_alt','next_alt','prev_dist','next_dist']).size().reset_index(name='count')
        long_range_connections['conn_code'] = (long_range_connections['conn_code'] != long_range_connections['conn_code'].shift(1)).cumsum()
        long_range_connections.rename(columns={'conn_code':'conn_id'}, inplace=True)

    return long_range_connections

def TransformContigConnectionsToScaffoldConnections(long_range_connections, scaffold_parts):
    long_range_connections['scaffold'] = scaffold_parts.loc[long_range_connections['conpart'].values,'scaffold'].values
    long_range_connections['scaf_pos'] = scaffold_parts.loc[long_range_connections['conpart'].values,'pos'].values
    long_range_connections['reverse'] = scaffold_parts.loc[long_range_connections['conpart'].values,'reverse'].values
    #long_range_connections[['scaffold','scaf_pos','reverse']] = scaffold_parts.loc[long_range_connections['conpart'].values,['scaffold','pos','reverse']].values
    # Reverse strand of contigs that are reversed in the scaffold to get the scaffold strand
    long_range_connections.loc[long_range_connections['reverse'], 'strand'] = np.where(long_range_connections.loc[long_range_connections['reverse'], 'strand'] == '+', '-', '+')
    # Group and combine contigs which are all part of the same scaffold (and are following it's order, so are not a repeat)
    long_range_connections['group'] = ( (long_range_connections['conn_id'] != long_range_connections['conn_id'].shift(1)) | (long_range_connections['scaffold'] != long_range_connections['scaffold'].shift(1)) | 
                                        ((long_range_connections['scaf_pos']+1 != long_range_connections['scaf_pos'].shift(1)) & (long_range_connections['scaf_pos']-1 != long_range_connections['scaf_pos'].shift(1))) ).cumsum()
    long_range_connections = long_range_connections.groupby(['group', 'conn_id', 'scaffold', 'strand', 'count'], sort=False)[['prev_alt','next_alt','prev_dist','next_dist']].agg({'prev_alt':['first'],'next_alt':['last'],'prev_dist':['first'],'next_dist':['last']}).droplevel(axis='columns',level=1).reset_index().drop(columns=['group'])

    # Get size and pos again
    con_size = long_range_connections.groupby(['conn_id']).size().reset_index(name='size')
    long_range_connections['size'] = np.repeat(con_size['size'].values, con_size['size'].values)
    long_range_connections['pos'] = long_range_connections.groupby(['conn_id'], sort=False).cumcount()

    return long_range_connections

def RemoveRedundantEntriesInScaffoldGraph(scaffold_graph):
    # Now remove all the paths that overlap a longer one (for equally long ones just take one of them)
    scaffold_graph.sort_values(['from','from_side']+[col for sublist in [['scaf'+str(s),'strand'+str(s),'dist'+str(s)] for s in range(1,scaffold_graph['length'].max())] for col in sublist], inplace=True)
    scaffold_graph['redundant'] = (scaffold_graph['from'] == scaffold_graph['from'].shift(1, fill_value=-1)) & (scaffold_graph['from_side'] == scaffold_graph['from_side'].shift(1, fill_value=''))
    for s in range(1,scaffold_graph['length'].max()):
        scaffold_graph['redundant'] = scaffold_graph['redundant'] & ( np.isnan(scaffold_graph['scaf'+str(s)]) | ( (scaffold_graph['scaf'+str(s)] == scaffold_graph['scaf'+str(s)].shift(1, fill_value=-1)) & (scaffold_graph['strand'+str(s)] == scaffold_graph['strand'+str(s)].shift(1, fill_value='')) & (scaffold_graph['dist'+str(s)] == scaffold_graph['dist'+str(s)].shift(1, fill_value='')))) 
    scaffold_graph = scaffold_graph[ scaffold_graph['redundant'] == False ].copy()
    scaffold_graph.drop(columns=['redundant'],inplace=True)

    # Remove overlapping paths with repeated starting scaffold
#    scaffold_graph.reset_index(inplace=True, drop=True)
#    scaffold_graph['rep_len'] = -1
#    reps = []
#    for s in range(1,scaffold_graph['length'].max()):
#        scaffold_graph.loc[(scaffold_graph['scaf'+str(s)] == scaffold_graph['from']) & (scaffold_graph['strand'+str(s)] == np.where(scaffold_graph['from_side'] == 'r', '+', '-')), 'rep_len'] = s
#        reps.append( scaffold_graph[(scaffold_graph['rep_len'] == s) & (scaffold_graph['rep_len']+1 < scaffold_graph['length'])].copy() )
#    reps = pd.concat(reps, ignore_index=True)
#    if len(reps):
#        reps = reps.merge(scaffold_graph.reset_index()[['from','from_side','index']].rename(columns={'index':'red_ind'}), on=['from','from_side'], how='inner') # Add all possibly redundant indexes in scaffold_graph
#        reps = reps[scaffold_graph.loc[reps['red_ind'], 'length'].values == reps['length'] - reps['rep_len']].copy() # Filter out all non-redundant entries
#        for s in range(1, (reps['length']-reps['rep_len']).max()):
#            for rl in range(1, reps['rep_len'].max()):
#                if s+rl < reps['length'].max():
#                    reps = reps[(reps['rep_len'] != rl) | (reps['length']-rl-1 < s) | ((scaffold_graph.loc[reps['red_ind'], 'scaf'+str(s)].values == reps['scaf'+str(s+rl)]) & (scaffold_graph.loc[reps['red_ind'], 'strand'+str(s)].values == reps['strand'+str(s+rl)]) & (scaffold_graph.loc[reps['red_ind'], 'dist'+str(s)].values == reps['dist'+str(s+rl)]))].copy()
#        scaffold_graph.drop(index=np.unique(reps['red_ind'].values), inplace=True)
#        scaffold_graph.drop(columns=['rep_len'], inplace=True)

    return scaffold_graph

def BuildScaffoldGraph(long_range_connections, scaf_bridges):
    # First start from every contig and extend in both directions on valid reads
    scaffold_graph = long_range_connections[['scaffold','strand','conn_id','pos','size']].copy()
    scaffold_graph = scaffold_graph[scaffold_graph['pos']+1 < scaffold_graph['size']].copy() # If we are already at the last position we cannot extend
    scaffold_graph['org_pos'] = scaffold_graph['pos']
    scaffold_graph.rename(columns={'scaffold':'from', 'strand':'from_side'}, inplace=True)
    scaffold_graph['from_side'] = np.where(scaffold_graph['from_side'] == '+','r','l')
    for s in range(1,scaffold_graph['size'].max()):
        scaffold_graph['pos'] += 1
        scaffold_graph = scaffold_graph.merge(long_range_connections[['conn_id','pos','scaffold','strand','prev_dist']].rename(columns={'scaffold':'scaf'+str(s), 'strand':'strand'+str(s), 'prev_dist':'dist'+str(s)}), on=['conn_id','pos'], how='left')
    scaffold_graph.drop(columns=['pos','conn_id'],inplace=True)
    scaffold_graph['length'] = scaffold_graph['size'] - scaffold_graph['org_pos']
    scaffold_graph.drop(columns=['size','org_pos'],inplace=True)
#
    # Then add scaf_bridges with alternatives
    short_bridges = scaf_bridges[['from','from_side','to','to_side','mean_dist']].copy()
    short_bridges.rename(columns={'to':'scaf1','to_side':'strand1','mean_dist':'dist1'}, inplace=True)
    short_bridges['strand1'] = np.where(short_bridges['strand1'] == 'l', '+', '-')
    short_bridges['length'] = 2
    scaffold_graph = pd.concat([scaffold_graph, short_bridges[['from','from_side','length','scaf1','strand1','dist1']]], ignore_index=True, sort=False)
#
    # Remove redundant entries
    scaffold_graph = RemoveRedundantEntriesInScaffoldGraph(scaffold_graph)

    return scaffold_graph

def RemoveEmptyColumns(df):
    # Clean up columns with all NaN
    cols = df.count()
    df.drop(columns=cols[cols == 0].index.values,inplace=True)
    return df

def FindValidExtensionsInScaffoldGraph(scaffold_graph):
    # Get all the possible continuations from a given scaffold
    extensions = scaffold_graph.rename(columns={'from':'scaf0','from_side':'strand0'})
    extensions['strand0'] = np.where(extensions['strand0'] == 'r', '+', '-')
    extensions.reset_index(drop=True, inplace=True)
    # All possible origins are just the inverse
    origins = extensions.rename(columns={col:f'o{col}' for col in extensions.columns if col not in ['scaf0','strand0']})
    origins.rename(columns={**{f'odist{s}':f'odist{s-1}' for s in range(1,origins['olength'].max())}}, inplace=True)
    origins['strand0'] = np.where(origins['strand0'] == '+', '-', '+')
    for s in range(1,origins['olength'].max()):
        origins.loc[origins[f'ostrand{s}'].isnull() == False, f'ostrand{s}'] = np.where(origins.loc[origins[f'ostrand{s}'].isnull() == False, f'ostrand{s}'] == '+', '-', '+')
    # Get all the branch points for the origins
    branches = origins[['scaf0','strand0']].reset_index().rename(columns={'index':'oindex1'}).merge(origins[['scaf0','strand0']].reset_index().rename(columns={'index':'oindex2'}), on=['scaf0','strand0'], how='inner').drop(columns=['scaf0','strand0'])
    nbranches = branches.groupby(['oindex1'], sort=False).size().reset_index(name='nbranches')
    branches = branches[np.isin(branches['oindex1'], nbranches.loc[nbranches['nbranches'] == 1, 'oindex1'].values) == False].copy() # With only one matching branch the branches do not have a branch point
    nbranches = nbranches[nbranches['nbranches'] > 1].copy()
    s = 1
    branch_points = []
    while len(branches):
        # Remove pairs where branch 2 stops matching branch 1 at scaffold s 
        branches = branches[(origins.loc[branches['oindex1'].values, [f'oscaf{s}',f'ostrand{s}',f'odist{s-1}']].values == origins.loc[branches['oindex2'].values, [f'oscaf{s}',f'ostrand{s}',f'odist{s-1}']].values).all(axis=1)].copy()
        # Add a branch point for the branches 1 that no have less pairs
        nbranches['nbranches_new'] = nbranches[['oindex1']].merge(branches.groupby(['oindex1'], sort=False).size().reset_index(name='nbranches'), on=['oindex1'], how='left')['nbranches'].fillna(0).astype(int).values
        branch_points.append( nbranches.loc[nbranches['nbranches_new'] < nbranches['nbranches'], ['oindex1']].rename(columns={'oindex1':'oindex'}) )
        branch_points[-1]['pos'] = s
        # Remove the branches that do not have further branch points
        nbranches['nbranches'] = nbranches['nbranches_new']
        nbranches.drop(columns=['nbranches_new'], inplace=True)
        branches = branches[np.isin(branches['oindex1'], nbranches.loc[nbranches['nbranches'] == 1, 'oindex1'].values) == False].copy() # With only one matching branch the branches do not have a branch point
        nbranches = nbranches[nbranches['nbranches'] > 1].copy()
        # Prepare next round
        s += 1
    branch_points = pd.concat(branch_points, ignore_index=True)
    branch_points.sort_values(['oindex','pos'], ascending=[True,False], inplace=True)
    # Get all the valid extensions for a given origin possible origin-extension-pairs and count how long they match
    pairs = origins[['scaf0','strand0']].reset_index().rename(columns={'index':'oindex'}).merge(extensions[['scaf0','strand0']].reset_index().rename(columns={'index':'eindex'}), on=['scaf0','strand0'], how='inner').drop(columns=['scaf0','strand0'])
    pairs[['scaf1','strand1','dist1']] = extensions.loc[pairs['eindex'].values, ['scaf1','strand1','dist1']].values
    while len(branch_points):
        # Get the furthest branch_point and find extensions starting from there
        highest_bp = branch_points.groupby(['oindex']).first().reset_index()
        cur_ext = []
        for bp in np.unique(highest_bp['pos']):
            cur = origins.loc[highest_bp.loc[highest_bp['pos'] == bp, 'oindex'].values].rename(columns={**{'scaf0':f'scaf{bp}','strand0':f'strand{bp}','odist0':f'dist{bp}'}, **{f'o{n}{s}':f'{n}{bp-s}' for s in range(1,bp+1) for n in ['scaf','strand','dist']}})
            cur.drop(columns=['olength']+[col for col in cur.columns if col[:5] == "oscaf" or col[:7] == "ostrand" or col[:5] == "odist"], inplace=True)
            cur.drop(columns=['dist0'], inplace=True, errors='ignore') # dist0 does not exist for a branch point at the highest position in scaffold_graph
            cur.reset_index(inplace=True)
            cur.rename(columns={'index':'oindex','scaf0':'from','strand0':'from_side'}, inplace=True)
            cur['from_side'] = np.where(cur['from_side'] == '+', 'r', 'l')
            cur = cur.merge(scaffold_graph[scaffold_graph['length'] > bp+1], on=[col for col in cur if col != 'oindex'], how='inner')
            if len(cur):
                # Trim cur to only contain the extensions
                cur.drop(columns=['from','from_side']+[f'{n}{s}' for s in range(1,bp+1) for n in ['scaf','strand','dist']], inplace=True)
                cur = RemoveEmptyColumns(cur)
                cur.rename(columns={f'{n}{s}':f'{n}{s-bp}' for s in range(bp+1,cur['length'].max()) for n in ['scaf','strand','dist']}, inplace=True)
                cur['length'] -= bp
                cur_ext.append(cur)
        if len(cur_ext):
            cur = pd.concat(cur_ext, ignore_index=True)
            del cur_ext
            if len(cur):
                # Check whether they map until the end
                cur_pairs = cur[['oindex','length','scaf1','strand1','dist1']].reset_index().rename(columns={'index':'cindex'}).merge(pairs[['oindex','eindex','scaf1','strand1','dist1']], on=['oindex','scaf1','strand1','dist1'], how='inner').drop(columns=['scaf1','strand1','dist1'])
                cur_pairs['matches'] = 1
                for s in range(2, cur['length'].max()):
                    comp = (cur_pairs['matches'] == s-1) & (cur_pairs['length'] > s)
                    cur_pairs.loc[comp,'matches'] += (cur.loc[cur_pairs.loc[comp, 'cindex'].values, [f'scaf{s}',f'strand{s}',f'dist{s}']].values == extensions.loc[cur_pairs.loc[comp, 'eindex'].values, [f'scaf{s}',f'strand{s}',f'dist{s}']].values).all(axis=1).astype(int)
                cur_pairs['end'] = cur_pairs['matches']+1 == cur_pairs['length']
                # Check if the extension matches at least one oft he possible extensions of origin (An origin can have multiple scaffold_graph entries that extend it and extensions are compared to each of them)
                cur_pairs = cur_pairs.groupby(['oindex','eindex'])[['end']].max().reset_index()
                # Add a end=False for every extension/origin pair that should have been checked, but was not, because already scaf1 did not match and they never went into cur_pairs
                cur_pairs = pairs[['oindex','eindex']].merge(cur[['oindex']].drop_duplicates(), on=['oindex'], how='inner').merge(cur_pairs, on=['oindex','eindex'], how='left')
                cur_pairs['end'] = cur_pairs['end'].fillna(False)
                # Remove pairs that do not match until the end
                pairs.drop(pairs[pairs[['oindex','eindex']].merge(cur_pairs, on=['oindex','eindex'], how='left')['end'].values == False].index.values, inplace=True)
        branch_points = branch_points[branch_points.merge(highest_bp, on=['oindex','pos'], how='left', indicator=True)['_merge'].values == "left_only"].copy()
    pairs.drop(columns=['scaf1','strand1','dist1'], inplace=True)
    # Add everything also in the other direction
    pairs = pairs.merge(pairs.rename(columns={'oindex':'eindex','eindex':'oindex'}), on=['oindex','eindex'], how='outer')
    pairs.sort_values(['oindex','eindex'], inplace=True)
    # Get continuations of origins and extensions to reduce computations later
    ocont = []
    missing_cont = pairs[['oindex']].rename(columns={'oindex':'oindex1'}) # All origins that have an extension must have a following origin in the direction of extension
    missing_cont[['nscaf','nstrand','ndist']] = extensions.loc[pairs['eindex'].values, ['scaf1','strand1','dist1']].values
    missing_cont.drop_duplicates(inplace=True)
    if len(missing_cont):
        for cut in range(origins['olength'].max()-1): # If we do not find a following origin for a given origin look for the longest match by cutting away scaffolds
            for l in np.unique(origins.loc[origins['olength'] > cut+1, 'olength'].values):
                cur = origins[origins['olength'] == l].reset_index()
                cur.rename(columns={**{'index':'oindex2','scaf0':'nscaf','strand0':'nstrand','odist0':'ndist'}, **{f'{n}{s}':f'{n}{s-1}' for s in range(1,l-cut) for n in ['oscaf','ostrand','odist']}}, inplace=True)
                cur.rename(columns={'oscaf0':'scaf0','ostrand0':'strand0'}, inplace=True)
                cur.drop(columns=['olength'] + [f'{n}{s}' for s in range(l-cut,l) for n in ['oscaf','ostrand']] + [f'odist{s}' for s in range(l-cut,l-1)], inplace=True)
                cur = RemoveEmptyColumns(cur)
                mcols = ['scaf0','strand0'] + [f'{n}{s}' for s in range(1,l-cut-1) for n in ['oscaf','ostrand']] + [f'odist{s}' for s in range(l-cut-2)]
                cur = cur.merge(origins.loc[np.isin(origins.index.values, np.unique(missing_cont['oindex1'])), mcols].reset_index().rename(columns={'index':'oindex1'}), on=mcols, how='inner').drop(columns=mcols)
                if cut > 0:
                    cur = cur.merge(missing_cont, on=['oindex1','nscaf','nstrand','ndist'], how='inner') # Do not use this for perfect matches, because sometimes an extension gets lost due to the simplification from all reads to the scaffold_graph
                cur['matches'] = l-cut
                ocont.append(cur)
            if cut == 0:
                # no cut means we have perfect matches, so we cannot improve on them anymore, so remove them from missing cont (For other cut values, we might cut away more from a longer origin and still achieve more matches, thus we cannnot only use it for perfect matches)
                ocont = pd.concat(ocont, ignore_index=True)
                mcols = ['oindex1','nscaf','nstrand','ndist']
                missing_cont = missing_cont[ missing_cont.merge(ocont[mcols].drop_duplicates(), on=mcols, how='left', indicator=True)['_merge'].values == "left_only" ].copy()
                perfect_ocont = ocont.drop(columns=['matches','nscaf','nstrand','ndist'])
                ocont = [ ocont ]
    if len(ocont):
        ocont = pd.concat(ocont, ignore_index=True)[['oindex1','oindex2','nscaf','nstrand','ndist','matches']]
        mcols = ['oindex1','nscaf','nstrand','ndist']
        ocont = ocont[ocont['matches'] == ocont[mcols].merge(ocont.groupby(mcols)['matches'].max().reset_index(), on=mcols, how='left')['matches'].values].copy()
        ocont.drop(columns=['matches'], inplace=True)
        ocont.sort_values(['oindex1','oindex2'], inplace=True)
    else:
        ocont = pd.DataFrame({'oindex1':[],'oindex2':[],'nscaf':[],'nstrand':[],'ndist':[]})
    econt = []
    for l in np.unique(extensions['length'].values):
        cur = extensions[extensions['length'] == l].reset_index()
        cur.drop(columns=['scaf0','strand0','dist1','length'], inplace=True)
        cur = RemoveEmptyColumns(cur)
        cur.rename(columns={**{'index':'eindex1'}, **{f'{n}{s}':f'{n}{s-1}' for s in range(1,l) for n in ['scaf','strand','dist']}}, inplace=True)
        mcols = ['scaf0','strand0'] + [f'{n}{s}' for s in range(1,l-1) for n in ['scaf','strand','dist']]
        cur = cur.merge(extensions[mcols].reset_index().rename(columns={'index':'eindex2'}), on=mcols, how='inner').drop(columns=mcols)
        econt.append(cur)
    if len(econt):
        econt = pd.concat(econt, ignore_index=True)
        econt.sort_values(['eindex1','eindex2'], inplace=True)
    else:
        econt = pd.DataFrame({'eindex1':[],'eindex2':[]})
    # Check that neighbours have consistent pairs
    while True:
        # Extend the extension to the next scaffold and then propagate back the origins to get the extension/origin pairs that should be there based on the neighbours
        new_pairs = econt.copy()
        new_pairs[['oscaf1','ostrand1','odist0']] = extensions.loc[new_pairs['eindex1'].values, ['scaf0','strand0','dist1']].values
        pairs[['oscaf1','ostrand1','odist0']] = origins.loc[pairs['oindex'].values, ['oscaf1','ostrand1','odist0']].values
        new_pairs = new_pairs.merge(pairs.rename(columns={'eindex':'eindex2','oindex':'oindex2'}), on=['eindex2','oscaf1','ostrand1','odist0'], how='inner').drop(columns=['oscaf1','ostrand1','odist0'])
        pairs.drop(columns=['oscaf1','ostrand1','odist0'], inplace=True)
        new_pairs = new_pairs.merge(perfect_ocont, on=['oindex2'], how='inner')
        new_pairs.drop(columns=['eindex2','oindex2'], inplace=True)
        new_pairs.drop_duplicates(inplace=True)
        #Add them to pairs if they are not already present
        new_pairs.rename(columns={'eindex1':'eindex','oindex1':'oindex'}, inplace=True)
        new_pairs = new_pairs.merge(new_pairs.rename(columns={'oindex':'eindex','eindex':'oindex'}), on=['oindex','eindex'], how='outer')
        old_len = len(pairs)
        pairs = pairs.merge(new_pairs, on=['oindex','eindex'], how='outer')
        pairs.sort_values(['oindex','eindex'], inplace=True)
        if old_len == len(pairs):
            break
    # Combine it into one dictionary for easier passing
    graph_ext = {}
    graph_ext['org'] = origins
    graph_ext['ext'] = extensions
    graph_ext['pairs'] = pairs
    graph_ext['ocont'] = ocont
#
    return graph_ext

def UnravelKnots(scaffold_graph, scaffolds):
    knots = []
    for oside in ['l','r']:
        # First take one side in scaffold_graph as origin
        pot_knots = scaffold_graph[scaffold_graph['from_side'] == oside].drop(columns=['from_side'])
        pot_knots['oindex'] = pot_knots.index
        pot_knots.rename(columns={'from':'oscaf0'}, inplace=True)
        pot_knots['ostrand0'] = '+' if oside == 'l' else '-'
        pot_knots['elen'] = 0 # Extension length
        pot_knots.rename(columns={'length':'olen'}, inplace=True)
        end_olen = pot_knots['olen'].max()
        pot_knots['olen'] -= 1
        pot_knots.rename(columns={f'scaf{s}':f'oscaf{s}' for s in range(1,end_olen)}, inplace=True)
        pot_knots.rename(columns={f'strand{s}':f'ostrand{s}' for s in range(1,end_olen)}, inplace=True)
        for s in range(1,end_olen):
            # In scaffold_graph strand + means the left side points to the center('from'), but now we want it to mean left side points to beginning of path / for second direction we invert the extension/origin, which does the same
            pot_knots.loc[pot_knots[f'ostrand{s}'].isnull() == False, f'ostrand{s}'] = np.where(pot_knots.loc[pot_knots[f'ostrand{s}'].isnull() == False, f'ostrand{s}'] == '+', '-', '+')
        pot_knots.rename(columns={f'dist{s}':f'odist{s-1}' for s in range(1,end_olen)}, inplace=True) # Distance is now also the one towards the beginning and not the center anymore (thus highest oscaf does not have a distance anymore)
        max_olen = end_olen-1
        pot_knots = pot_knots[['oindex','olen','elen']+[f'o{n}{s}' for s in range(max_olen) for n in ['scaf','strand','dist']]+[f'oscaf{max_olen}',f'ostrand{max_olen}']].copy()
        # Find the first position, where pot_knots origins are unique (upos)
        pot_knots.sort_values([f'o{n}{s}' for s in range(max_olen) for n in ['scaf','strand','dist']]+[f'oscaf{max_olen}',f'ostrand{max_olen}'], inplace=True)
        pot_knots['group'] = ((pot_knots['oscaf0'] != pot_knots['oscaf0'].shift(1)) | (pot_knots['ostrand0'] != pot_knots['ostrand0'].shift(1))).cumsum()
        pot_knots['upos'] = 0
        s = 1
        while np.sum(pot_knots['group'] == pot_knots['group'].shift(1)):
            pot_knots.loc[(pot_knots['group'] == pot_knots['group'].shift(1)) | (pot_knots['group'] == pot_knots['group'].shift(-1)), 'upos'] += 1
            pot_knots['group'] = ((pot_knots['group'] != pot_knots['group'].shift(1)) | (pot_knots[f'odist{s-1}'] != pot_knots[f'odist{s-1}'].shift(1)) |
                                  (pot_knots[f'oscaf{s}'] != pot_knots[f'oscaf{s}'].shift(1)) | (pot_knots[f'ostrand{s}'] != pot_knots[f'ostrand{s}'].shift(1))).cumsum()
            s += 1
        pot_knots.drop(columns=['group'], inplace=True)
        # Try to find a unique extension past oscaf0 starting from upos (we only consider one direction, if they are unique in the other direction we will check by merging the unique ones from both directions)
        extensions = []
        for s in range(0,pot_knots['upos'].max()+1):
            # Find extensions
            cur_ext = pot_knots.loc[s == pot_knots['upos'], ['oindex',f'oscaf{s}',f'ostrand{s}']+[f'o{n}{s1}' for s1 in range(s) for n in ['scaf','strand','dist']]].rename(columns={**{f'oscaf{s}':'from',f'ostrand{s}':'from_side'}, **{f'o{n}{s1}':f'{n}{s-s1}' for s1 in range(s) for n in ['scaf','strand','dist']}})
            cur_ext['from_side'] = np.where(cur_ext['from_side'] == '+', 'r','l')
            cur_ext = cur_ext.merge(scaffold_graph, on=['from','from_side']+[f'{n}{s1}' for s1 in range(1,s+1) for n in ['scaf','strand','dist']], how='inner')
            cur_ext.drop(columns=['from','from_side']+[f'{n}{s1}' for s1 in range(1,s+1) for n in ['scaf','strand','dist']], inplace=True)
            # Keep only extensions that are unique up to here
            cur_ext = cur_ext[(cur_ext['length'] > s+1)].copy() # Not an extensions
            cur_ext = cur_ext[1 == cur_ext[['oindex']].merge( cur_ext.groupby(['oindex']).size().reset_index(name='alts'), on=['oindex'], how='left' )['alts'].values].copy()
            # Store the extensions to check later if they match to a unique entry in scaffold_graph
            if len(cur_ext):
                end_elen = cur_ext['length'].max()
                cur_ext.rename(columns={f'{n}{s1}':f'e{n}{s1-s}' for s1 in range(s+1,end_elen) for n in ['scaf','strand','dist']}, inplace=True)
                cur_ext.drop(columns=[col for col in cur_ext.columns if (col[:4] == "scaf") | (col[:6] == "strand") | (col[:4] == "dist")], inplace=True)
                cur_ext['length'] -= s+1
                extensions.append(cur_ext)
        pot_knots.drop(columns=['upos'], inplace=True)
        pot_knots = pot_knots.merge(pd.concat(extensions, ignore_index=True, sort=False), on=['oindex'], how='inner')
        pot_knots['elen'] = pot_knots['length']
        pot_knots.drop(columns=['length'], inplace=True)
        # Find the entries in scaffold_graph for the other side that match the extensions
        other_side = scaffold_graph[scaffold_graph['from_side'] == ('r' if oside == 'l' else 'l')].drop(columns=['from_side'])
        other_side.reset_index(inplace=True)
        other_side.rename(columns={**{'from':'oscaf0'}, **{col:f'e{col}' for col in other_side.columns if col not in ['from','length']}}, inplace=True)
        extensions = []
        for s in range(1,pot_knots['elen'].max()+1):
            mcols = ['oscaf0']+[f'{n}{s1}' for s1 in range(1,s+1) for n in ['escaf','estrand','edist']]
            extensions.append( pot_knots.loc[pot_knots['elen'] == s, ['oindex']+mcols].merge(other_side[['eindex']+mcols], on=mcols, how='left').drop(columns=[col for col in mcols if col not in ['escaf1','estrand1','edist1']]) )
        extensions = pd.concat(extensions, ignore_index=True)
        # Filter the ambiguous matches to scaffold_graph (again considering only on direction)
        extensions = extensions[1 == extensions[['oindex']].merge( extensions.groupby(['oindex']).size().reset_index(name='alts'), on=['oindex'], how='left' )['alts'].values].copy()
        extensions[['escaf1','edist1']] = extensions[['escaf1','edist1']].astype(int)
        extensions.rename(columns={'escaf1':'escaf','estrand1':'estrand','edist1':'edist'}, inplace=True)
        # Store the remaining knots for this side
        pot_knots.drop(columns=['elen']+[col for col in pot_knots.columns if (col[:5] == "escaf") | (col[:7] == "estrand") | (col[:5] == "edist")], inplace=True)
        pot_knots = pot_knots.merge(extensions, on=['oindex'], how='inner')
        knots.append(pot_knots)
       
    # Finally keep only the traversals through the knot that are unique in both directions
    knots[0] = knots[0].merge(knots[1][['oindex','eindex']].rename(columns={'oindex':'eindex','eindex':'oindex'}), on=['oindex','eindex'], how='inner')
    knots[0].reset_index(inplace=True)
    knots[0].rename(columns={'index':'knot'}, inplace=True)
    knots[1] = knots[1].merge(knots[0][['knot','oindex','eindex']].rename(columns={'oindex':'eindex','eindex':'oindex'}), on=['oindex','eindex'], how='inner')
    knots = pd.concat(knots, ignore_index=True, sort=False)
    knots = RemoveEmptyColumns(knots)
    
    return knots

def FollowUniquePathsThroughGraph(knots, scaffold_graph):
    # Get connected knots, which have to be handled in sequence instead of parallel
    ravels = []
    for s in range(0,knots['olen'].max()+1):
        ravels.append(knots.loc[knots['olen'] >= s, ['knot',f'oscaf{s}']].rename(columns={f'oscaf{s}':'scaf'}))
    ravels = pd.concat(ravels, ignore_index=True)
    ravels.drop_duplicates(inplace=True)
    ravels['scaf'] = ravels['scaf'].astype(int)
    ravels['ravel'] = ravels['knot']
    while True:
        ravels['min_ravel'] = ravels[['scaf']].merge(ravels.groupby(['scaf'])['ravel'].min().reset_index(), on=['scaf'], how='left')['ravel'].values
        ravels['new_ravel'] = ravels[['ravel']].merge(ravels.groupby(['ravel'])['min_ravel'].min().reset_index(), on=['ravel'], how='left')['min_ravel'].values
        if np.sum(ravels['ravel'] != ravels['new_ravel']) == 0:
            break
        else:
            ravels['ravel'] = ravels['new_ravel']
    ravels = ravels[['ravel','knot']].drop_duplicates()
    ravels.sort_values(['ravel','knot'], inplace=True)
#
    # Follow the unique paths through knots
    scaffold_paths = []
    pid = 0
    while len(ravels):
        for cdir in ['+','-']:
            # Prepare extension
            extensions = ravels.groupby(['ravel']).first().reset_index(drop=True).merge(knots[knots['ostrand0'] == cdir], on=['knot'], how='left')
            extensions.drop(columns=['oindex','eindex'], inplace=True)
            extensions['pid'] = np.arange(len(extensions))+pid
            if cdir == '-': # Only on the second round change pid to get the same pid for both directions
                pid += len(extensions)
                extensions['pos'] = -1
            else: # Only on the first round we add the center scaffold and create used knots
                used_knots = extensions['knot'].values
                extensions['pos'] = 0
                scaffold_paths.append(extensions[['pid','pos','oscaf0','ostrand0','odist0']].rename(columns={'oscaf0':'scaf','ostrand0':'strand','odist0':'dist'}))
                extensions['pos'] = 1
            scaffold_paths.append(extensions[['pid','pos','escaf','estrand','edist']].rename(columns={'escaf':'scaf','estrand':'strand','edist':'dist'})) # In the negative direction unfortunately strands are the opposite and we do not have the correct distance yet, so we store the distance to the next instead of previous entry and later shift
#
            # Extend as long as possible
            while len(extensions):
                # Shift to next position
                extensions.drop(columns=['knot'], inplace=True)
                extensions = RemoveEmptyColumns(extensions)
                max_olen = extensions['olen'].max()
                extensions.rename(columns={**{'escaf':'oscaf0','estrand':'ostrand0','edist':'odist0'}, **{f'{n}{s}':f'{n}{s+1}' for s in range(max_olen) for n in ['oscaf','ostrand','odist']}, **{f'oscaf{max_olen}':f'oscaf{max_olen+1}',f'ostrand{max_olen}':f'ostrand{max_olen+1}'}}, inplace=True)
                extensions['pos'] += 1 if cdir == '+' else -1
                # Check if we have a valid extensions
                pos_ext = knots[np.isin(knots['oscaf0'], extensions['oscaf0'].values) & (knots['olen'] <= extensions['olen'].max()+1)].drop(columns=['oindex','eindex']) # The olen can be one higher, because we shifted the extensions by one, thus knots['olen'] == extensions['olen']+1 means we reach back to exactly the same scaffold
                if len(pos_ext) == 0:
                    extensions = []
                else:
                    new_ext = []
                    for s in range(1,pos_ext['olen'].max()+1):
                        mcols = [f'{n}{s1}' for s1 in range(s) for n in ['oscaf','ostrand','odist']]+[f'oscaf{s}',f'ostrand{s}']
                        new_ext.append( extensions[['pid','pos']+mcols].merge(pos_ext.loc[pos_ext['olen'] == s, ['knot','olen','escaf','estrand','edist']+mcols], on=mcols, how='inner') )
                    extensions = pd.concat(new_ext, ignore_index=True)
                    # Check if this knot was not already included (otherwise circular genomes would continue endlessly)
                    extensions = extensions[ np.isin(extensions['knot'], used_knots) == False ].copy()
                    used_knots = np.concatenate([used_knots, extensions['knot'].values])
                    # Add new scaffold to path
                    scaffold_paths.append(extensions[['pid','pos','escaf','estrand','edist']].rename(columns={'escaf':'scaf','estrand':'strand','edist':'dist'}))
        # Drop already used knots, so that we do not start from them anymore
        ravels = ravels[np.isin(ravels['knot'], used_knots) == False].copy()
#
    # Invert the scaffold paths in the negative direction to point in positive direction
    scaffold_paths = pd.concat(scaffold_paths, ignore_index=True)
    scaffold_paths.sort_values(['pid','pos'], inplace=True)
    invert = scaffold_paths['pos'] < 0
    scaffold_paths.loc[invert, 'strand'] = np.where(scaffold_paths.loc[invert, 'strand'] == '+', '-', '+')
    scaffold_paths['dist'] = np.where(invert, np.where(scaffold_paths['pid'] == scaffold_paths['pid'].shift(1), scaffold_paths['dist'].shift(1, fill_value=0), 0), scaffold_paths['dist'])
#
    # Finalize paths
    scaffold_paths['pos'] = scaffold_paths.groupby(['pid'], sort=False).cumcount()
    scaffold_paths.rename(columns={'scaf':'scaf0','strand':'strand0','dist':'dist0'}, inplace=True)
#
    return scaffold_paths

def FindRepeatedScaffolds(scaffold_graph):
    repeated_scaffolds = []
    for s in range(1, scaffold_graph['length'].max()):
        repeated_scaffolds.append(np.unique(scaffold_graph.loc[scaffold_graph['from'] == scaffold_graph[f'scaf{s}'], 'from']))
    repeated_scaffolds = np.unique(np.concatenate(repeated_scaffolds))
    repeat_graph = scaffold_graph[np.isin(scaffold_graph['from'], repeated_scaffolds)].copy()
#
    return repeated_scaffolds, repeat_graph

def ConnectLoopsThatShareScaffolds(loop_scafs):
    while True:
        loop_scafs['min_loop'] = loop_scafs[['scaf']].merge(loop_scafs.loc[loop_scafs['inside'], ['scaf','loop']].groupby(['scaf']).min().reset_index(), on=['scaf'], how='left')['loop'].values
        loop_scafs['min_loop'] = loop_scafs[['loop']].merge(loop_scafs.loc[loop_scafs['inside'], ['loop','min_loop']].groupby(['loop']).min().reset_index(), on=['loop'], how='left')['min_loop'].values
        if np.sum(loop_scafs['min_loop'] != loop_scafs['loop']) == 0:
            break
        else:
            loop_scafs['loop'] = loop_scafs['min_loop'].astype(int)
    loop_scafs.drop(columns=['min_loop'], inplace=True)
    loop_scafs = loop_scafs.groupby(['loop','scaf']).agg({'exit':['min'], 'inside':['max']}).droplevel(1, axis=1).reset_index()
    loop_scafs.sort_values(['loop','scaf','exit','inside'], ascending=[True,True,False,False], inplace=True)
#
    return loop_scafs

def AddConnectedScaffolds(loop_scafs, scaffold_graph):
    loop_graph = scaffold_graph.merge(loop_scafs.loc[loop_scafs['inside'], ['scaf','loop']].rename(columns={'scaf':'from'}), on=['from'], how='inner')
    new_scafs = []
    for s in range(1,loop_graph['length'].max()):
        new_scafs.append( loop_graph.loc[loop_graph['length'] > s, ['loop',f'scaf{s}']].drop_duplicates().astype(int).rename(columns={f'scaf{s}':'scaf'}) )
    new_scafs = pd.concat(new_scafs)
    new_scafs.drop_duplicates(inplace=True)
    loop_scafs = loop_scafs.merge(new_scafs, on=['loop','scaf'], how='outer')
    loop_scafs[['inside','exit']] = loop_scafs[['inside','exit']].fillna(False)
#
    return loop_scafs

def FindLoopExits(loop_scafs, scaffold_graph):
    loop_scafs.loc[np.isin(loop_scafs['loop'], loop_scafs.loc[loop_scafs['inside'] == loop_scafs['exit'], 'loop'].drop_duplicates().values), 'exit'] = False # If we added new scaffolds to the loop we cannot guarantee anymore that the exits are still exits, so check again
    exit_graph = scaffold_graph[['from','from_side','length']+[f'scaf{s}' for s in range(1,scaffold_graph['length'].max())]].merge(loop_scafs.loc[loop_scafs['inside'] == loop_scafs['exit'], ['scaf','loop']].rename(columns={'scaf':'from'}), on=['from'], how='inner')
    exit_graph = RemoveEmptyColumns(exit_graph)
    while len(exit_graph):
        num_undecided = np.sum(loop_scafs['inside'] == loop_scafs['exit'])
        exit_graph['exit'] = True
        exit_graph['inside'] = False
        for s in range(1,exit_graph['length'].max()):
            # Everything that is connected to something that is not an exit scaffolds is not a guaranteed exit paths
            exit_graph.loc[exit_graph[['loop',f'scaf{s}']].rename(columns={f'scaf{s}':'scaf'}).merge(loop_scafs.loc[loop_scafs['exit'] == False, ['loop','scaf']], on=['loop','scaf'], how='left', indicator=True)['_merge'].values == "both", 'exit'] = False
            # Everything that is connected to at least one scaffold inside the loop is an inside paths
            exit_graph.loc[exit_graph[['loop',f'scaf{s}']].rename(columns={f'scaf{s}':'scaf'}).merge(loop_scafs.loc[loop_scafs['inside'], ['loop','scaf']], on=['loop','scaf'], how='left', indicator=True)['_merge'].values == "both", 'inside'] = True
        # If all paths on one side are an exit paths the scaffold is an exit and if at least one paths on both sides is an inside paths the scaffold is inside the loop
        summary = exit_graph.groupby(['loop','from','from_side']).agg({'exit':['min'], 'inside':['max']}).droplevel(1, axis=1).reset_index()
        summary.sort_values(['loop','from','from_side'], inplace=True)
        summary = summary.loc[np.repeat(summary.index.values, 1+( ((summary['from_side'] == 'l') & ((summary['loop'] != summary['loop'].shift(-1)) | (summary['from'] != summary['from'].shift(-1)))) |
                                                                  ((summary['from_side'] == 'r') & ((summary['loop'] != summary['loop'].shift(1)) | (summary['from'] != summary['from'].shift(1)))) ))].reset_index(drop=True)
        summary.loc[(summary['from_side'] == 'l') & (summary['loop'] == summary['loop'].shift(1)) & (summary['from'] == summary['from'].shift(1)), ['from_side','exit','inside']] = ['r',True,False] # Add the exits where nothing connects on one side
        summary.loc[(summary['from_side'] == 'r') & (summary['loop'] == summary['loop'].shift(-1)) & (summary['from'] == summary['from'].shift(-1)), ['from_side','exit','inside']] = ['l',True,False]
        summary = summary.groupby(['loop','from']).agg({'exit':['max'], 'inside':['min']}).droplevel(1, axis=1).reset_index()
        summary.rename(columns={'from':'scaf'}, inplace=True)
        loop_scafs.loc[loop_scafs[['loop','scaf']].merge(summary.loc[summary['exit'], ['loop','scaf']], on=['loop','scaf'], how='left', indicator=True)['_merge'] == "both", 'exit'] = True
        loop_scafs.loc[loop_scafs[['loop','scaf']].merge(summary.loc[summary['inside'], ['loop','scaf']], on=['loop','scaf'], how='left', indicator=True)['_merge'] == "both", 'inside'] = True
        # If we cannot reduce the number of undecided scaffolds anymore they are looped within itself, but that means they are exits (into another loop)
        if np.sum(loop_scafs['inside'] == loop_scafs['exit']) == num_undecided:
            loop_scafs.loc[loop_scafs['inside'] == loop_scafs['exit'], 'exit'] = True
        # Delete all decided scaffolds from exit_graph
        exit_graph = exit_graph.merge(loop_scafs.loc[loop_scafs['inside'] == loop_scafs['exit'], ['scaf','loop']].rename(columns={'scaf':'from'}), on=['from','loop'], how='inner')
#
    return loop_scafs

def ReverseVerticalPaths(loops):
    reverse_loops = []
    for l in np.unique(loops['length']):
        reverse_loops.append( loops.loc[loops['length'] == l, ['length','scaf0','strand0']+[f'{n}{s}' for s in range(1,l) for n in ['scaf','strand','dist']]].rename(columns={**{f'{n}{s}':f'{n}{l-s-1}' for s in range(l) for n in ['scaf','strand']}, **{f'dist{s+1}':f'dist{l-s-1}' for s in range(l)}}).reset_index() )
    reverse_loops = pd.concat(reverse_loops, ignore_index=True, sort=False)
    reverse_loops = reverse_loops[['index','length','scaf0','strand0']+[f'{n}{s}' for s in range(1,reverse_loops['length'].max()) for n in ['scaf','strand','dist']]].copy()
    for s in range(reverse_loops['length'].max()):
        reverse_loops.loc[reverse_loops['length'] > s, f'strand{s}'] = np.where(reverse_loops.loc[reverse_loops['length'] > s, f'strand{s}'] == '+', '-', '+')
    reverse_loops.rename(columns={'index':'lindex'}, inplace=True)
#
    return reverse_loops

def GetLoopUnits(loop_scafs, scaffold_graph, max_loop_units):
    # Get loops by extending the inside scaffolds until we find an exit or they have multiple options after finding the starting scaffold again
    loops = []
    pot_loops = loop_scafs.loc[loop_scafs['inside'], ['loop','scaf']].rename(columns={'scaf':'scaf0'})
    pot_loops['strand0'] = '+'
    pot_loops['len'] = 1
    loop_graph = scaffold_graph[np.isin(scaffold_graph['from'], pot_loops['scaf0'].values)].copy()
    loop_graph.rename(columns={'from':'scaf0','from_side':'strand0'}, inplace=True)
    loop_graph['strand0'] = np.where(loop_graph['strand0'] == 'r', '+', '-')
    sfix = 0
    while len(pot_loops):
        # Merge with loop_graph to extend
        pot_loops['index'] = pot_loops.index.values
        new_loops = []
        for l in range(pot_loops['len'].min(),pot_loops['len'].max()+1):
            mcols = ['scaf0','strand0']+[f'{n}{s}' for s in range(1,l) for n in ['scaf','strand','dist']]
            new_loops.append( pot_loops.loc[pot_loops['len'] == l, ['loop']+[col for col in pot_loops.columns if col[:3] == "fix"]+['index','len']+(['dist0'] if 'dist0' in pot_loops.columns else [])+mcols].merge(loop_graph, on=mcols, how='inner') )
        pot_loops = pd.concat(new_loops, ignore_index=True, sort=False)
        pot_loops = RemoveEmptyColumns(pot_loops)
        # Prepare columns for next round
        pot_loops.rename(columns={f'{n}0':f'fix{n}{sfix}' for n in ['scaf','strand','dist']}, inplace=True)
        sfix += 1
        pot_loops.rename(columns={f'{n}{s}':f'{n}{s-1}' for s in range(1,pot_loops['length'].max())for n in ['scaf','strand','dist']}, inplace=True)
        pot_loops['len'] = pot_loops['length']-1
        pot_loops.drop(columns=['length'], inplace=True)
        # Check if we have an exit, close the loop or have a repeated scaffold
        repeats = []
        for s in range(1,sfix): # We explicitly compare for fixscaf0 later, so we do not need it here
            repeats.append( pot_loops.loc[np.isnan(pot_loops[f'fixscaf{s}']) == False, [f'fixscaf{s}',f'fixstrand{s}']].rename(columns={f'fixscaf{s}':'scaf',f'fixstrand{s}':'strand'}).reset_index() )
        pot_loops['exit'] = False
        pot_loops['closed'] = False
        for s in range(pot_loops['len'].max()):
            pot_loops.loc[(pot_loops['fixscaf0'] == pot_loops[f'scaf{s}']) & (pot_loops['fixstrand0'] == pot_loops[f'strand{s}']), 'closed'] = True
            pot_loops.loc[ pot_loops[['loop',f'scaf{s}']].rename(columns={f'scaf{s}':'scaf'}).merge(loop_scafs.loc[loop_scafs['exit'], ['loop','scaf']], on=['loop','scaf'], how='left', indicator=True)['_merge'] == "both", 'exit'] = True
            repeats.append( pot_loops.loc[np.isnan(pot_loops[f'scaf{s}']) == False, [f'scaf{s}',f'strand{s}']].rename(columns={f'scaf{s}':'scaf',f'strand{s}':'strand'}).reset_index() )
        repeats = pd.concat(repeats, ignore_index=True, sort=False).groupby(['index','scaf','strand']).size().reset_index(name='repeats')
        pot_loops['repeat'] = False
        pot_loops.loc[ np.unique(repeats.loc[repeats['repeats'] > 1, 'index'].values), 'repeat' ] = True
        pot_loops['max_units'] = pot_loops[['loop','fixscaf0']].merge(pot_loops.groupby(['loop','fixscaf0']).size().reset_index(name='count'), on=['loop','fixscaf0'], how='left')['count'] > max_loop_units
        # Add closed loops to loops and remove the dead ends or closed loops that do not involve the starting scaffold (and would lead to endless loops)
        loops.append( RemoveEmptyColumns(pot_loops[pot_loops['closed']].drop(columns=['index','exit','closed','repeat','max_units']).rename(columns={**{f'{n}{s}':f'{n}{s+sfix}' for s in range(pot_loops['len'].max()) for n in ['scaf','strand','dist']}, **{f'fix{n}{s}':f'{n}{s}' for s in range(sfix) for n in ['scaf','strand','dist']}})) )
        if len(loops[-1]):
            loops[-1]['len'] += sfix
        else:
            del loops[-1]
        pot_loops = pot_loops[ (pot_loops['closed'] == False) & (pot_loops['repeat'] == False) & (pot_loops['exit'] == False) & (pot_loops['max_units'] == False) ].copy()
        pot_loops.drop(columns=['exit','closed','repeat','max_units'], inplace=True)
    loops = pd.concat(loops, ignore_index=True, sort=False)
    loops.rename(columns={'len':'length'}, inplace=True)
    # Truncate the ends that reach into the next loop unit
    loops['truncate'] = True
    for s in range(loops['length'].max()-1, 1, -1):
        loops.loc[(loops['scaf0'] == loops[f'scaf{s}']) & (loops['strand0'] == loops[f'strand{s}']), 'truncate'] = False
        truncate = loops['truncate'] & (loops['length'] > s)
        loops.loc[truncate, [f'scaf{s}',f'strand{s}',f'dist{s}']] = np.nan
        loops.loc[truncate, 'length'] -= 1
    loops.drop(columns=['truncate'], inplace=True)
    loops = RemoveEmptyColumns(loops)
    loops.drop_duplicates(inplace=True)
    # Verify that the loop units are also valid in reverse direction
    loops.reset_index(drop=True, inplace=True)
    reverse_loops = ReverseVerticalPaths(loops)
    valid_indexes = []
    mcols = ['from','from_side','scaf1','strand1','dist1']
    while len(reverse_loops):
        # Remove invalid loop units
        reverse_loops.rename(columns={'scaf0':'from','strand0':'from_side'}, inplace=True)
        reverse_loops['from_side'] = np.where(reverse_loops['from_side'] == '+', 'r', 'l')
        check = reverse_loops[mcols].reset_index().rename(columns={'index':'rindex'}).merge(scaffold_graph[mcols].reset_index().rename(columns={'index':'sindex'}), on=mcols, how='inner').drop(columns=mcols)
        check['length'] = np.minimum(reverse_loops.loc[check['rindex'].values, 'length'].values, scaffold_graph.loc[check['sindex'].values, 'length'].values)
        reverse_loops['valid'] = False
        s = 2
        while len(check):
            reverse_loops.loc[np.unique(check.loc[check['length'] == s, 'rindex'].values), 'valid'] = True
            check = check[check['length'] > s].copy()
            if len(check):
                check = check[(reverse_loops.loc[check['rindex'].values, [f'scaf{s}',f'strand{s}',f'dist{s}']].values == scaffold_graph.loc[check['sindex'].values, [f'scaf{s}',f'strand{s}',f'dist{s}']].values).all(axis=1)].copy()
                s += 1
        reverse_loops = reverse_loops[reverse_loops['valid']].copy()
        # Store valid indexces
        valid_indexes.append( np.unique(reverse_loops.loc[reverse_loops['length'] == 2, 'lindex'].values) )
        reverse_loops = reverse_loops[reverse_loops['length'] > 2].copy()
        # Prepare next round
        if len(reverse_loops):
            reverse_loops.drop(columns=['from','from_side','dist1'], inplace=True)
            reverse_loops['length'] -= 1
            reverse_loops.rename(columns={f'{n}{s+1}':f'{n}{s}' for s in range(reverse_loops['length'].max()) for n in ['scaf','strand','dist']}, inplace=True)
    valid_indexes = np.unique(np.concatenate(valid_indexes))
    loops = loops.loc[valid_indexes].copy()
    # Finish
    loops.sort_values(['scaf0','length'], inplace=True)
    loops.reset_index(drop=True, inplace=True)
#
    return loops

def CheckConsistencyOfVerticalPaths(vpaths):
    if 'from' in vpaths.columns:
        inconsistent = vpaths[np.isnan(vpaths['from']) | ((vpaths['from_side'] != 'r') & (vpaths['from_side'] != 'l'))].copy()
    else:
        inconsistent = vpaths[np.isnan(vpaths['scaf0']) | ((vpaths['strand0'] != '+') & (vpaths['strand0'] != '-'))].copy()
    if len(inconsistent):
        print("Warning: Position 0 is inconsistent in vertical paths.")
        print(inconsistent)
#
    for s in range(1, vpaths['length'].max()):
        inconsistent = vpaths[ ((vpaths['length'] > s) & (np.isnan(vpaths[f'scaf{s}']) | ((vpaths[f'strand{s}'] != '+') & (vpaths[f'strand{s}'] != '-')) | np.isnan(vpaths[f'dist{s}']))) |
                              ((vpaths['length'] <= s) & ((np.isnan(vpaths[f'scaf{s}']) == False) | (vpaths[f'strand{s}'].isnull() == False) | (np.isnan(vpaths[f'dist{s}']) == False))) ].copy()
        if len(inconsistent):
            print(f"Warning: Position {s} is inconsistent in vertical paths.")
            print(inconsistent)

def GetLoopUnitsInBothDirections(loops):
    bidi_loops = ReverseVerticalPaths(loops)
    bidi_loops['loop'] = loops.loc[bidi_loops['lindex'].values, 'loop'].values
    bidi_loops = pd.concat([loops.reset_index().rename(columns={'index':'lindex'}), bidi_loops], ignore_index=True, sort=False)
    bidi_loops.sort_values(['lindex','strand0'], inplace=True)
    bidi_loops.reset_index(drop=True,inplace=True)
#
    return bidi_loops

def FindConnectionsBetweenLoopUnits(loops, scaffold_graph, full_info):
    # Get the positions where the scaffold_graph from the end of the loop units splits to later check only there to reduce the advantage of long reads just happen to be on one connection
    first_diff = scaffold_graph.loc[np.isin(scaffold_graph['from'], np.unique(loops['scaf0'].values)), ['from','from_side']].reset_index().rename(columns={'index':'index1'})
    first_diff = first_diff.merge(first_diff.rename(columns={'index1':'index2'}), on=['from','from_side'], how='inner').drop(columns=['from','from_side'])
    first_diff = first_diff[first_diff['index1'] != first_diff['index2']].copy()
    first_diff['diff'] = -1
    s = 1
    diffs = []
    while len(first_diff['diff']):
        first_diff.loc[(scaffold_graph.loc[first_diff['index1'].values, [f'scaf{s}',f'strand{s}',f'dist{s}']].values != scaffold_graph.loc[first_diff['index2'].values, [f'scaf{s}',f'strand{s}',f'dist{s}']].values).any(axis=1), 'diff'] = s
        diffs.append(first_diff.loc[first_diff['diff'] >= 0, ['index1','diff']].drop_duplicates())
        first_diff = first_diff[first_diff['diff'] < 0].copy()
        s += 1
    diffs = pd.concat(diffs, ignore_index=True, sort=False)
    # Find associated loop units
    units = diffs[['index1']].drop_duplicates()
    units[['scaf0','strand0']] = scaffold_graph.loc[ units['index1'].values, ['from','from_side']].values
    units['strand0'] = np.where(units['strand0'] == 'l', '+', '-') # For the first_diffs we went into the oppositee direction to get the diffs from which we can extend over the ends
    bidi_loops = GetLoopUnitsInBothDirections(loops)
    units = units.merge(bidi_loops[['scaf0','strand0']].reset_index().rename(columns={'index':'bindex'}), on=['scaf0','strand0'], how='left').drop(columns=['scaf0','strand0'])
    units['ls'] = bidi_loops.loc[units['bindex'].values, 'length'].values - 2
    units['len'] = np.minimum(scaffold_graph.loc[units['index1'].values, 'length'].values, units['ls']+2)
    valid_units = []
    s = 1
    units[['scaf','strand','dist']] = np.nan
    while len(units):
        for ls in np.unique(units['ls']):
            units.loc[units['ls'] == ls, ['scaf','strand','dist']] = bidi_loops.loc[ units.loc[units['ls'] == ls, 'bindex'].values, [f'scaf{ls}',f'strand{ls}',f'dist{ls+1}']].values
        units['strand'] = np.where(units['strand'] == '+', '-', '+') # We compare graph entries in different orientations
        units = units[(units[['scaf','strand','dist']].values == scaffold_graph.loc[units['index1'].values, [f'scaf{s}',f'strand{s}',f'dist{s}']].values).all(axis=1)].copy()
        units['ls'] -= 1
        s += 1
        valid_units.append( units.loc[units['len'] == s, ['index1','bindex']].copy() )
        units = units[units['len'] > s].copy()
    valid_units = pd.concat(valid_units, ignore_index=True, sort=False)
    diffs = diffs.merge(valid_units, on=['index1'], how='inner')
    # Check if the scaffold_graph extends from the position of difference into the next loop unit (otherwise it does not hold any information)
    info = diffs[['index1','diff']].drop_duplicates()
    info[['from','from_side']+[f'{n}{s}' for s in range(1,info['diff'].max()+1) for n in ['scaf','strand','dist']]] = np.nan
    for d in np.unique(info['diff']):
        info.loc[info['diff'] == d, ['from','from_side']+[f'{n}{s}' for s in range(1,d+1) for n in ['scaf','strand']]+[f'dist{s}' for s in range(1,d+1)]] = scaffold_graph.loc[info.loc[info['diff'] == d,'index1'].values, [f'{n}{s}' for s in range(d,0,-1) for n in ['scaf','strand']]+['from','from_side']+[f'dist{s}' for s in range(d,0,-1)]].values
        info.loc[info['diff'] == d, f'strand{d}'] = np.where(info.loc[info['diff'] == d, f'strand{d}'] == 'l', '+', '-')
    info['from_side'] = np.where(info['from_side'] == '+', 'l', 'r')
    for s in range(1,info['diff'].max()):
        info.loc[info['diff'] > s, [f'strand{s}']] = np.where(info.loc[info['diff'] > s, [f'strand{s}']] == '+', '-', '+')
    extensions = []
    for d in np.unique(info['diff']):
        mcols = ['from','from_side'] + [f'{n}{s}' for s in range(1,d+1) for n in ['scaf','strand','dist']]
        extensions.append( info.loc[info['diff'] == d, ['index1','diff']+mcols].merge(scaffold_graph, on=mcols, how='inner').drop(columns=[col for col in mcols if col not in [f'scaf{d}',f'strand{d}']]).rename(columns={f'{n}{s}':f'{n}{s-d}' for s in range(d,scaffold_graph['length'].max()) for n in ['scaf','strand','dist']}) )
    extensions = pd.concat(extensions, ignore_index=True, sort=False)
    extensions = extensions[extensions['diff']+1 < extensions['length']].copy()
    extensions['length'] -= extensions['diff']
    # Merge the extensions from different positions (only keep extensions with highest diff/longest mapping in the loop unit, but use all consistent extending scafs)
    if len(extensions):
        extensions.sort_values(['index1','scaf0','strand0']+[f'{n}{s}' for s in range(1,extensions['length'].max()) for n in ['scaf','strand','dist']]+['diff'], inplace=True)
        first_iter = True
        while True:
            extensions['consistent'] = (extensions['index1'] == extensions['index1'].shift(-1)) & (extensions['diff'] < extensions['diff'].shift(-1))
            for s in range(1, extensions['length'].max()):
                cur = extensions['consistent'] & (extensions['length'].shift(-1) > s)
                extensions.loc[cur, 'consistent'] = (extensions.loc[cur, [f'scaf{s}',f'strand{s}',f'dist{s}']].values == extensions.loc[cur.shift(1, fill_value=False).values, [f'scaf{s}',f'strand{s}',f'dist{s}']].values).all(axis=1)
            if first_iter:
                # Extensions that have a lower diff, but the same length and are fully consistent can be removed, because they do not contain additional information
                extensions = extensions[(extensions['consistent'] == False) | (extensions['length'] > extensions['length'].shift(-1))].copy()
                first_iter = False
            else:
                extensions['del'] = extensions['consistent'].shift(1) & (extensions['consistent'] == False)
                extensions.loc[extensions['del'].shift(-1, fill_value=False).values, 'diff'] = extensions.loc[extensions['del'], 'diff'].values
                extensions = extensions[extensions['del'] == False].drop(columns=['del'])
            if np.sum(extensions['consistent']) == 0:
                break
        extensions.drop(columns=['consistent'], inplace=True)
        max_diff = extensions.groupby(['index1'])['diff'].agg(['max','size'])
        extensions = extensions[extensions['diff'] == np.repeat(max_diff['max'].values, max_diff['size'].values)].drop(columns=['diff'])
        extensions = RemoveEmptyColumns(extensions)
        extendible = diffs.loc[np.isin(diffs['index1'], max_diff.index.values), ['index1','bindex']].drop_duplicates()
    # Get the loop units matching the extensions
    loop_conns = []
    if len(extensions):
        ext_units = extensions.reset_index().rename(columns={'index':'extindex'}).merge(bidi_loops[['scaf0','strand0','scaf1','strand1','dist1']].reset_index().rename(columns={'index':'bindex'}), on=['scaf0','strand0','scaf1','strand1','dist1'], how='inner')
        if len(ext_units):
            ext_units['length'] = np.minimum(ext_units['length'].values, bidi_loops.loc[ext_units['bindex'].values, 'length'].values)
            for s in range(2,ext_units['length'].max()):
                ext_units = ext_units[(ext_units['length'] <= s) | (ext_units[[f'scaf{s}',f'strand{s}',f'dist{s}']].values == bidi_loops.loc[ext_units['bindex'].values, [f'scaf{s}',f'strand{s}',f'dist{s}']].values).all(axis=1)].copy()
        if len(ext_units):
            # Get the connected loop units
            loop_conns = extendible.rename(columns={'bindex':'bindex1'}).merge(ext_units[['index1','bindex']].rename(columns={'bindex':'bindex2'}), on=['index1'], how='inner').drop(columns=['index1'])
            loop_conns['wildcard'] = False
            # Only keep extensions that extend into another loop unit, because we later use them to check if we have evidence for an additional copy of a loop unit
            extensions = RemoveEmptyColumns( extensions.loc[ext_units['extindex'].values].copy() )
        else:
            extensions = []
    # Non extendible loop units can connect to all other loop units, because we do not know what comes after them (we only take into account other loop units here, because extensions into exits are not an issue, if we do not have a true loop, but just a repeated scaffold, we will only create a duplicated paths that we later remove)
    if len(loop_conns):
        non_extendible = np.setdiff1d(bidi_loops.index.values, np.unique(loop_conns['bindex1'].values))
    else:
        non_extendible = bidi_loops.index.values
    non_extendible = bidi_loops.loc[non_extendible, ['scaf0','strand0']].reset_index().rename(columns={'index':'bindex1'})
    non_extendible = non_extendible.merge(bidi_loops[['scaf0','strand0']].reset_index().rename(columns={'index':'bindex2'}), on=['scaf0','strand0'], how='left').drop(columns=['scaf0','strand0'])
    non_extendible['wildcard'] = True
    if len(non_extendible):
        if len(loop_conns):
            loop_conns = pd.concat([loop_conns, non_extendible], ignore_index=True, sort=False)
        else:
            loop_conns = non_extendible
    # Get the loop indixes from the bidirectional loop indexes and verify that both directions are supported
    if len(loop_conns):
        loop_conns.drop_duplicates(inplace=True)
        for i in [1,2]:
            loop_conns[[f'lindex{i}',f'dir{i}']] = bidi_loops.loc[loop_conns[f'bindex{i}'].values, ['lindex','strand0']].values
            loop_conns[f'rdir{i}'] = np.where(loop_conns[f'dir{i}'] == '+', '-', '+')
        loop_conns = loop_conns[['lindex1','dir1','lindex2','dir2','wildcard']].merge(loop_conns[['lindex1','rdir1','lindex2','rdir2']].rename(columns={'lindex1':'lindex2','rdir1':'dir2','lindex2':'lindex1','rdir2':'dir1'}), on=['lindex1','dir1','lindex2','dir2'], how='inner')
#
    if len(loop_conns):
        if full_info == False:
            loop_conns = loop_conns[(loop_conns[['dir1','dir2']] == '+').all(axis=1)].drop(columns=['dir1','dir2'])
            loop_conns.sort_values(['lindex1','lindex2'], inplace=True)
        else:
            # Get loop units that overlap by more than the starting scaffold
            overlapping_units = []
            for sfrom in range(1,bidi_loops['length'].max()-1):
                for l in np.unique(bidi_loops.loc[bidi_loops['length'] > sfrom+1, 'length'].values):
                    overlapping_units.append( bidi_loops[bidi_loops['length'] == l].reset_index().rename(columns={'index':'bindex1'}).merge(bidi_loops[bidi_loops['length'] > l-sfrom+1].reset_index().rename(columns={'index':'bindex2'}), left_on=['loop',f'scaf{sfrom}',f'strand{sfrom}']+[f'{n}{s}' for s in range(sfrom+1, l) for n in ['scaf','strand','dist']], right_on=['loop','scaf0','strand0']+[f'{n}{s}' for s in range(1, l-sfrom) for n in ['scaf','strand','dist']], how='inner')[['bindex1','bindex2']].copy() )
                    overlapping_units[-1]['sfrom'] = sfrom
            if len(overlapping_units):
                overlapping_units = pd.concat(overlapping_units, ignore_index=True)
            # Filter out the ones that do not match an extension
            if len(overlapping_units):
                overlapping_units = overlapping_units.merge(extendible.rename(columns={'bindex':'bindex1'}), on=['bindex1'], how='left')
                overlapping_units['index1'] = overlapping_units['index1'].fillna(-1).astype(int)
                overlapping_units = overlapping_units.merge(extensions, on=['index1'], how='left')
                overlapping_units['length'] = overlapping_units['length'].fillna(0).astype(int)
                overlapping_units['s2'] = bidi_loops.loc[overlapping_units['bindex1'].values, 'length'].values - overlapping_units['sfrom']
                for s in range(1, overlapping_units['length'].max()):
                    overlapping_units['valid'] = overlapping_units['length'] <= s
                    overlapping_units.loc[overlapping_units['valid'], 's2'] = -1 # So we do not compare them in the next step
                    for s2 in np.unique(overlapping_units.loc[overlapping_units['valid'] == False, 's2'].values):
                        overlapping_units.loc[overlapping_units['s2'] == s2, 'valid'] = (overlapping_units.loc[overlapping_units['s2'] == s2, [f'scaf{s}',f'strand{s}',f'dist{s}']].values == bidi_loops.loc[overlapping_units.loc[overlapping_units['s2'] == s2, 'bindex2'].values, [f'scaf{s2}',f'strand{s2}',f'dist{s2}']].values).all(axis=1)
                    overlapping_units = overlapping_units[overlapping_units['valid']].copy()
                    overlapping_units['s2'] += 1
                overlapping_units = overlapping_units[['bindex1','bindex2','sfrom']].drop_duplicates()
            # Get the loop indixes from the biderectional loop indexes and verify that both directions are supported
            if len(overlapping_units):
                for i in [1,2]:
                    overlapping_units[[f'lindex{i}',f'dir{i}']] = bidi_loops.loc[overlapping_units[f'bindex{i}'].values, ['lindex','strand0']].values
                    overlapping_units[f'rdir{i}'] = np.where(overlapping_units[f'dir{i}'] == '+', '-', '+')
                overlapping_units['overlap'] = bidi_loops.loc[overlapping_units['bindex1'].values, 'length'].values - overlapping_units['sfrom']
                overlapping_units = overlapping_units[['lindex1','dir1','lindex2','dir2','sfrom','overlap']].merge(overlapping_units[['lindex1','rdir1','lindex2','rdir2','overlap']].rename(columns={'lindex1':'lindex2','rdir1':'dir2','lindex2':'lindex1','rdir2':'dir1'}), on=['lindex1','dir1','lindex2','dir2','overlap'], how='inner').drop(columns=['overlap'])
                overlapping_units['wildcard'] = False
            loop_conns['sfrom'] = loops.loc[loop_conns['lindex1'].values, 'length'].values - 1
            if len(overlapping_units):
                loop_conns = pd.concat([loop_conns, overlapping_units], ignore_index=True)
            # If multiple starting positions of unit 2 exist in unit 1 take the first, such that we have the minimum length of the combined unit
            loop_conns = loop_conns.groupby(['lindex1','dir1','lindex2','dir2','wildcard'])['sfrom'].min().reset_index()
            # Prepare possible extensions of the loop units
            if len(extensions):
                extensions = extendible.merge(extensions, on=['index1'], how='inner').drop(columns=['index1'])
                extensions = bidi_loops[['lindex']].reset_index().rename(columns={'index':'bindex'}).merge(extensions, on=['bindex'], how='inner').drop(columns=['bindex'])
#
    if full_info:
        return loop_conns, extensions
    else:
        return loop_conns

def TryReducingAlternativesToPloidy(alternatives, scaf_bridges, ploidy):
    # If we have less than ploidy alternatives  they are directly ok
    alternatives.sort_values(['group'], inplace=True)
    nalts = alternatives.groupby(['group'], sort=False).size().values
    nalts = np.repeat(nalts, nalts)
    valid_alts = alternatives[nalts <= ploidy].copy()
    alternatives['nalts'] = nalts
    alternatives = alternatives[nalts > ploidy].copy()
    if len(alternatives) == 0:
        alternatives = valid_alts
    else:
        # Find the alternatives that differ only by distance or additionally by a missing scaffold
        alternatives.reset_index(drop=True, inplace=True)
        pairs = alternatives[['group']].reset_index().rename(columns={'index':'index1'}).merge( alternatives[['group']].reset_index().rename(columns={'index':'index2'}), on=['group'], how='left' )
        pairs = pairs[pairs['index1'] != pairs['index2']].drop(columns=['group'])
        pairs['scaf_miss'] = False
        for i in [1,2]:
            pairs[f'len{i}'] = alternatives.loc[pairs[f'index{i}'].values, 'length'].values
        pairs['s2'] = 1
        reducible = []
        pairs[['scaf','strand']] = [-1,'']
        for s1 in range(1,alternatives['length'].max()):
            pairs['match'] = False
            while np.sum(pairs['match'] == False):
                for s2 in np.unique(pairs.loc[pairs['match'] == False, 's2'].values):
                    cur = (pairs['match'] == False) & (pairs['s2'] == s2)
                    pairs.loc[cur, ['scaf','strand']] = alternatives.loc[pairs.loc[cur, 'index2'].values, [f'scaf{s2}',f'strand{s2}']].values
                pairs.loc[pairs['match'] == False, 'match'] = (pairs.loc[pairs['match'] == False, ['scaf','strand']].values == alternatives.loc[pairs.loc[pairs['match'] == False, 'index1'].values, [f'scaf{s1}',f'strand{s1}']].values).all(axis=1)
                pairs.loc[pairs['match'] == False, 's2'] += 1
                pairs.loc[pairs['match'] == False, 'scaf_miss'] = True
                pairs = pairs[ pairs['s2'] < pairs['len2'] ].copy()
            reducible.append( pairs.loc[ s1+1 == pairs['len1'], ['index1','index2','scaf_miss'] ].copy() )
            pairs = pairs[ s1+1 < pairs['len1'] ].copy()
            pairs['s2'] += 1
        reducible = pd.concat(reducible, ignore_index=True)
        # Get bridge support for reducible alternatives
        bsupp_indexes = reducible[['index1']].copy()
        bsupp = alternatives.loc[bsupp_indexes['index1'].values, ['from','from_side','scaf1','strand1','dist1']].reset_index().rename(columns={'index':'index1','scaf1':'to','strand1':'to_side','dist1':'mean_dist'})
        bsupp['to_side'] = np.where(bsupp['to_side'] == '+', 'l', 'r')
        bsupp = [bsupp]
        bsupp_indexes['len1'] = alternatives.loc[bsupp_indexes['index1'].values, 'length'].values
        for s1 in range(1,alternatives['length'].max()-1):
            bsupp.append( alternatives.loc[bsupp_indexes.loc[bsupp_indexes['len1'] > s1+1, 'index1'].values, [f'scaf{s1}',f'strand{s1}',f'scaf{s1+1}',f'strand{s1+1}',f'dist{s1+1}']].reset_index().rename(columns={'index':'index1',f'scaf{s1}':'from',f'strand{s1}':'from_side',f'scaf{s1+1}':'to',f'strand{s1+1}':'to_side',f'dist{s1+1}':'mean_dist'}) )
            bsupp[-1]['from_side'] = np.where(bsupp[-1]['from_side'] == '+', 'r', 'l')
            bsupp[-1]['to_side'] = np.where(bsupp[-1]['to_side'] == '+', 'l', 'r')
        bsupp = pd.concat(bsupp, ignore_index=True)
        bsupp = bsupp.merge(scaf_bridges[['from','from_side','to','to_side','mean_dist','bcount']], on=['from','from_side','to','to_side','mean_dist'], how='left')
        bsupp = bsupp.groupby(['index1'])['bcount'].agg(['min','median','max']).reset_index()
        reducible = reducible.merge(bsupp, on=['index1'], how='left')
        # Distance only differences are only allowed if the reversed pair does not exist or the bridge support is lower (otherwise we might wrongly remove both of the alternatives)
        for col in ['min','median','max']:
            reducible = reducible[ reducible[col] <= reducible[['index1','index2']].rename(columns={'index1':'index2','index2':'index1'}).merge(reducible, on=['index1','index2'], how='left')[col].fillna(sys.maxsize).values ].copy()
        # In case of a tie remove the one with the higher index
        reducible = reducible[ (reducible['index1'] > reducible['index2']) | (reducible[['index1','index2']].rename(columns={'index1':'index2','index2':'index1'}).merge(reducible[['index1','index2']], on=['index1','index2'], how='left', indicator=True)['_merge'].values == "left_only") ].copy()
        reducible = reducible.groupby(['index1'])['scaf_miss'].min().reset_index()
        reducible[['group','loop','from','from_side','nalts']] = alternatives.loc[reducible['index1'].values, ['group','loop','from','from_side','nalts']].values
        # Remove the reducible ones with the lowest bridge support until we arrive at ploidy alternatives (distance only alternatives are always removed before missing scaffold alternatives)
        alternatives.drop(columns=['nalts'], inplace=True)
        reducible = reducible.merge(bsupp, on=['index1'], how='left')
        reducible.sort_values(['group','scaf_miss','min','median','max'], inplace=True)
        alternatives.drop(reducible.loc[reducible['nalts'] - reducible.groupby(['group'], sort=False).cumcount() > ploidy, 'index1'].values, inplace=True)
        # Add the already valid ones back in
        alternatives = pd.concat([valid_alts, alternatives], ignore_index=True).sort_values(['group']).reset_index(drop=True)
#
    return alternatives

def GetOtherSideOfExitConnection(conns, exit_conns):
    other_side = conns[['ito']].merge(exit_conns[['ifrom','length','from','from_side']+[f'{n}{s}' for s in range(1,exit_conns['length'].max()) for n in ['scaf','strand','dist']]].drop_duplicates().rename(columns={'ifrom':'ito'}), on=['ito'], how='left')
    other_side.rename(columns={'from':'scaf0','from_side':'strand0'}, inplace=True)
    other_side['strand0'] = np.where(other_side['strand0'] == 'r', '+', '-')
    other_side = ReverseVerticalPaths(other_side)
    other_side.sort_values(['lindex'], inplace=True)
    other_side.drop(columns=['lindex'], inplace=True)
    other_side.rename(columns={f'{col}':f'r{col}' for col in other_side.columns}, inplace=True)
    conns = pd.concat([conns.reset_index(drop=True), other_side.reset_index(drop=True)], axis=1)
#
    return conns

def AppendScaffoldsToExitConnection(conns):
    conns[[f'{n}{s}' for s in range(conns['length'].max(), (conns['sfrom']+conns['rlength']).max()) for n in ['scaf','strand','dist']]] = np.nan
    for sfrom in np.unique(conns.loc[conns['rlength'] > 0, 'sfrom']):
        inconsistent = conns[ (conns['rlength'] > 0) & (conns['sfrom'] == sfrom) & (conns[[f'{n}{sfrom}' for n in ['scaf','strand']]].values != conns[[f'r{n}0' for n in ['scaf','strand']]].values).any(axis=1) ].copy()
        if len(inconsistent):
            print("Warning: Appending vertical paths without proper match.")
            print(inconsistent)
        max_len = conns.loc[conns['sfrom'] == sfrom, 'rlength'].max()
        conns.loc[conns['sfrom'] == sfrom, [f'{n}{s}' for s in range(sfrom+1, sfrom+max_len) for n in ['scaf','strand','dist']]] = conns.loc[conns['sfrom'] == sfrom, [f'r{n}{s}' for s in range(1, max_len) for n in ['scaf','strand','dist']]].values
    conns['length'] = conns['sfrom']+conns['rlength']
    conns.drop(columns=['sfrom','rlength','rscaf0','rstrand0']+[f'r{n}{s}' for s in range(1, conns['rlength'].max()) for n in ['scaf','strand','dist']], inplace=True)
    conns = RemoveEmptyColumns(conns)
#
    return conns

def GetHandledScaffoldConnectionsFromVerticalPath(vpaths):
    handled_scaf_conns = []
    for s in range(1,vpaths['length'].max()):
        handled_scaf_conns.append(vpaths.loc[vpaths['length'] > s, [f'scaf{s-1}',f'strand{s-1}',f'scaf{s}',f'strand{s}',f'dist{s}']].rename(columns={f'scaf{s-1}':'scaf0',f'strand{s-1}':'strand0',f'scaf{s}':'scaf1',f'strand{s}':'strand1',f'dist{s}':'dist1'}))
    handled_scaf_conns = pd.concat(handled_scaf_conns, ignore_index=True)
    handled_scaf_conns[['scaf0','scaf1','dist1']] = handled_scaf_conns[['scaf0','scaf1','dist1']].astype(int)
    handled_scaf_conns.drop_duplicates(inplace=True)
    # Also get reverse direction
    handled_scaf_conns = pd.concat([handled_scaf_conns, handled_scaf_conns.rename(columns={'scaf0':'scaf1','scaf1':'scaf0','strand0':'strand1','strand1':'strand0'})], ignore_index=True)
    handled_scaf_conns.drop_duplicates(inplace=True)
#
    return handled_scaf_conns

def TurnHorizontalPathIntoVertical(vpaths, min_path_id):
    vpaths['dist0'] = 0
    vpaths['pid'] = np.arange(min_path_id, min_path_id+len(vpaths))
#
    hpaths = []
    for s in range(vpaths['length'].max()):
        hpaths.append( vpaths.loc[vpaths['length'] > s, ['pid',f'scaf{s}',f'strand{s}',f'dist{s}']].rename(columns={f'scaf{s}':'scaf0',f'strand{s}':'strand0',f'dist{s}':'dist0'}) )
        hpaths[-1]['pos'] = s
    hpaths = pd.concat(hpaths, ignore_index=True)
    hpaths[['scaf0','dist0']] = hpaths[['scaf0','dist0']].astype(int)
    hpaths = hpaths[['pid','pos','scaf0','strand0','dist0']].copy()
    hpaths.sort_values(['pid','pos'], inplace=True)
#
    vpaths.drop(columns=['dist0','pid'], inplace=True)
#
    return hpaths

def AddPathThroughLoops(scaffold_paths, scaffold_graph, scaf_bridges, org_scaf_conns, ploidy, max_loop_units):
    # Get the loop units and find scaffolds in them
    repeated_scaffolds, repeat_graph = FindRepeatedScaffolds(scaffold_graph)
    if len(repeated_scaffolds):
        loops = []
        for s in range(1, repeat_graph['length'].max()):
            loops.append(repeat_graph.loc[(repeat_graph['from'] == repeat_graph[f'scaf{s}']) & (repeat_graph['from_side'] == 'r') & (repeat_graph[f'strand{s}'] == '+'), ['from','length']+[f'scaf{s1}' for s1 in range(1,s+1) for n in ['scaf']]])
            loops[-1]['length'] = s+1
        loops = pd.concat(loops, ignore_index=True, sort=False)
        loops['scaf0'] = loops['from']
    loop_scafs = []
    if len(loops):
        for s in range(loops['length'].max()-1): # The last base is just the repeated scaffold again
            loop_scafs.append( loops.loc[loops['length'] > s+1, ['from',f'scaf{s}']].drop_duplicates().astype(int).rename(columns={'from':'loop',f'scaf{s}':'scaf'}) )
        loop_scafs = pd.concat(loop_scafs)
        loop_scafs.drop_duplicates(inplace=True)
        # Find all scaffolds connected to the loop
        loop_scafs['inside'] = True
        loop_scafs['exit'] = False
        loop_scafs = ConnectLoopsThatShareScaffolds(loop_scafs)
        loop_scafs = AddConnectedScaffolds(loop_scafs, scaffold_graph)
        while np.sum(loop_scafs['inside'] == loop_scafs['exit']): # New entries, where we do not know yet if they are inside the loop or an exit
            loop_scafs = FindLoopExits(loop_scafs, scaffold_graph)
            loop_size = loop_scafs.groupby(['loop']).size().reset_index(name='nbefore')
            loop_scafs = ConnectLoopsThatShareScaffolds(loop_scafs)
            loop_size['nafter'] = loop_size[['loop']].merge(loop_scafs.groupby(['loop']).size().reset_index(name='nafter'), on=['loop'], how='left')['nafter'].values
            loop_scafs.loc[np.isin(loop_scafs['loop'], loop_size.loc[loop_size['nbefore'] < loop_size['nafter'], 'loop'].values), 'exit'] = False # We cannot guarantee that the exits are still exits in merged loops
            loop_scafs = AddConnectedScaffolds(loop_scafs, scaffold_graph)
#
    if len(loop_scafs):
        # Get loop units by exploring possible paths in scaffold_graph
        loops = GetLoopUnits(loop_scafs, scaffold_graph, max_loop_units)
        if len(loops):
            CheckConsistencyOfVerticalPaths(loops)
    if len(loops):
        # Remove loop units that are already part of a multi-repeat unit
        multi_repeat = []
        max_len = loops['length'].max()
        for s in range(1, max_len-1):
            found_repeat = (loops['scaf0'] == loops[f'scaf{s}']) & (loops[f'strand{s}'] == '+') & (loops['length'] > s+1) # Do not take the last position into account, otherwise we would end up with the full loop unit
            multi_repeat.append( loops[found_repeat].drop(columns=[f'scaf{s1}' for s1 in range(0,s)]+[f'strand{s1}' for s1 in range(0,s)]+[f'dist{s1}' for s1 in range(1,s+1)]).rename(columns={f'{n}{s1}':f'{n}{s1-s}' for s1 in range(s, max_len) for n in ['scaf','strand','dist']}) )
            multi_repeat[-1]['length'] -= s
            multi_repeat.append( loops[found_repeat].drop(columns=[f'{n}{s1}' for s1 in range(s+1, max_len) for n in ['scaf','strand','dist']]) )
            multi_repeat[-1]['length'] = s+1
        if len(multi_repeat):
            multi_repeat = pd.concat(multi_repeat, ignore_index=True, sort=False)
        if len(multi_repeat):
            multi_repeat.drop_duplicates(inplace=True)
            multi_repeat = RemoveEmptyColumns(multi_repeat)
            loops = loops[ loops.merge(multi_repeat, on=list(multi_repeat.columns.values), how='left', indicator=True)['_merge'].values == "left_only" ].copy()
        # Find connections between loop units
        loop_conns = FindConnectionsBetweenLoopUnits(loops, scaffold_graph, False)
        loop_conns['loop'] = loops.loc[loop_conns['lindex1'].values, 'loop'].values
        loops = loops[np.isin(loops['loop'], np.unique(loop_conns['loop']))].copy()
    if len(loop_conns):
        # Check if we have evidence for an order for the repeat copies (only one valid connection)
        conn_count = loop_conns.groupby(['lindex1']).size().values
        loop_conns['unique'] = (loop_conns['lindex1'] != loop_conns['lindex2']) & np.repeat(conn_count == 1, conn_count) & loop_conns[['lindex2']].merge( (loop_conns.groupby(['lindex2']).size() == 1).reset_index(name='unique'), on=['lindex2'], how='left')['unique'].values
        while np.sum(loop_conns['unique']):
            loop_conns['append'] = loop_conns['unique'] & (np.isin(loop_conns['lindex2'].values, loop_conns.loc[loop_conns['unique'], 'lindex1'].values) == False) # Make sure we are not appending something that itselfs appends something in this round
            if np.sum(loop_conns['append']) == 0:
                loop_conns.reset_index(drop=True, inplace=True)
                loop_conns.loc[np.max(loop_conns[loop_conns['unique']].index.values), 'unique'] = False
            else:
                for i in [1,2]:
                    loop_conns[f'len{i}'] = loops.loc[loop_conns[f'lindex{i}'].values, 'length'].values
                for l in np.unique(loop_conns.loc[loop_conns['append'], 'len1']):
                    cur = loop_conns[loop_conns['append'] & (loop_conns['len1'] == l)]
                    loops.loc[cur['lindex1'].values, [f'{n}{s}' for s in range(l,l+cur['len2'].max()) for n in ['scaf','strand','dist']]] = loops.loc[cur['lindex2'].values, [f'{n}{s}' for s in range(1,1+cur['len2'].max()) for n in ['scaf','strand','dist']]].values
                cur = loop_conns[loop_conns['append']].copy()
                loops.loc[cur['lindex1'].values, 'length'] += cur['len2'].values - 1
                loop_conns.drop(columns=['len1','len2'], inplace=True) # Lengths change, so do not keep outdated information
                loops.drop(cur['lindex2'].values, inplace=True)
                loop_conns = loop_conns[loop_conns['append'] == False].copy()
                loop_conns['new_index'] = loop_conns[['lindex1']].merge(cur[['lindex1','lindex2']].rename(columns={'lindex1':'new_index','lindex2':'lindex1'}), on=['lindex1'], how='left')['new_index'].values
                loop_conns.loc[np.isnan(loop_conns['new_index']) == False, 'lindex1'] = loop_conns.loc[np.isnan(loop_conns['new_index']) == False, 'new_index'].astype(int)
        CheckConsistencyOfVerticalPaths(loops)
#
    # Get exits for the repeats
    if len(loop_scafs):
        exits = loop_scafs.loc[loop_scafs['exit'],['loop','scaf']].rename(columns={'scaf':'from'})
        if len(exits):
            exits = exits.merge(scaffold_graph, on=['from'], how='left')
            exits = exits[ exits[['loop','scaf1']].rename(columns={'scaf1':'scaf'}).merge(loop_scafs.loc[loop_scafs['exit'],['loop','scaf']], on=['loop','scaf'], how='left', indicator=True)['_merge'].values == "left_only" ].copy() # Only keep the first scaffold of the exit
            exits['loopside'] = False
            for s in range(1,exits['length'].max()):
                exits.loc[exits[['loop',f'scaf{s}']].rename(columns={f'scaf{s}':'scaf'}).merge(loop_scafs.loc[loop_scafs['inside'],['loop','scaf']], on=['loop','scaf'], how='left', indicator=True)['_merge'].values == "both", 'loopside'] = True
            exits = exits[exits['loopside']].drop(columns=['loopside'])
#
    # Handle bridged repeats
    bridged_repeats = []
    if len(exits):
        for s in range(2,exits['length'].max()):
            exits['bridged'] = exits[['loop',f'scaf{s}']].rename(columns={f'scaf{s}':'scaf'}).merge(loop_scafs.loc[loop_scafs['exit'],['loop','scaf']], on=['loop','scaf'], how='left', indicator=True)['_merge'].values == "both"
            bridged_repeats.append( exits.loc[exits['bridged'] & (exits['from'] < exits[f'scaf{s}']), ['loop','from','from_side']+[f'{n}{s}' for s in range(1,s+1) for n in ['scaf','strand','dist']]].drop_duplicates() )
            bridged_repeats[-1]['length'] = s+1
            exits = exits[exits['bridged'] == False].copy()
        exits.drop(columns=['bridged'], inplace=True)
        bridged_repeats = pd.concat(bridged_repeats, ignore_index=True, sort=False)
        if len(bridged_repeats):
            bridged_repeats.rename(columns={'from':'scaf0','from_side':'strand0'}, inplace=True)
            bridged_repeats['strand0'] = np.where(bridged_repeats['strand0'] == 'r', '+', '-')
#
    # Only keep exits of unbridged repeats, where we know from where to where they go (either because we only have two exits or because we have scaffolding information that tells us), for the rest we will later add the loop units as path
    if len(exits):
        exit_scafs = exits[['loop','from','from_side']].drop_duplicates()
        exit_conns = exit_scafs.merge(org_scaf_conns[org_scaf_conns['distance'] > 0], on=['from','from_side'], how='inner')
        if len(exit_conns):
            exit_conns = exit_conns.merge( exit_conns.rename(columns={'from':'to','from_side':'to_side','to':'from','to_side':'from_side'}), on=['loop','from','from_side','to','to_side','distance'], how='inner')
        if len(exit_conns):
            exit_scafs = exit_scafs[ exit_scafs[['loop','from','from_side']].merge(exit_conns[['loop','from','from_side']], on=['loop','from','from_side'], how='left', indicator=True)['_merge'].values == "left_only" ].copy()
        if len(exit_scafs):
            exit_scafs.sort_values(['loop','from','from_side'], inplace=True)
            exit_count = exit_scafs.groupby(['loop'], sort=False).size().values
            exit_scafs = exit_scafs[np.repeat(exit_count == 2, exit_count)].copy()
        if len(exit_scafs):
            exit_scafs['to'] = np.where(exit_scafs['loop'] == exit_scafs['loop'].shift(-1), exit_scafs['from'].shift(-1, fill_value=-1), exit_scafs['from'].shift(1, fill_value=-1))
            exit_scafs['to_side'] = np.where(exit_scafs['loop'] == exit_scafs['loop'].shift(-1), exit_scafs['from_side'].shift(-1, fill_value=''), exit_scafs['from_side'].shift(1, fill_value=''))
            exit_scafs['distance'] = 0
            exit_conns = pd.concat([exit_conns, exit_scafs], ignore_index=True)
    # Unbridged repeats are not allowed to have more alternatives than ploidy
    if len(exit_conns):
        exit_conns = exit_conns.merge(exits, on=['loop','from','from_side'], how='left')
        exit_conns.sort_values(['loop','from','from_side'], inplace=True)
        exit_conns['group'] = (exit_conns.groupby(['loop','from','from_side'], sort=False).cumcount() == 0).cumsum()
        exit_conns = TryReducingAlternativesToPloidy(exit_conns, scaf_bridges, ploidy)
        exit_conns['from_alts'] = exit_conns.merge(exit_conns.groupby(['group']).size().reset_index(name='count'), on=['group'], how='left')['count'].values
        exit_conns.drop(columns=['group'], inplace=True)
        exit_conns = exit_conns[exit_conns['from_alts'] <= ploidy].copy()
        exit_conns = exit_conns.merge( exit_conns[['loop','from','from_side','to','to_side','distance']].drop_duplicates().rename(columns={'from':'to','from_side':'to_side','to':'from','to_side':'from_side','from_alts':'to_alts'}), on=['loop','from','from_side','to','to_side','distance'], how='inner')
        exit_conns = RemoveEmptyColumns(exit_conns)
    # Reverse loop units to have both directions
    if len(loops):
        bidi_loops = GetLoopUnitsInBothDirections(loops)
        CheckConsistencyOfVerticalPaths(bidi_loops)
    # Get the loop units to fill the connections
    start_units = []
    if len(exit_conns) and len(loops):
        for sfrom in range(1,exit_conns['length'].max()):
            pairs = exit_conns.loc[exit_conns['length'] > sfrom, ['loop',f'scaf{sfrom}',f'strand{sfrom}']].reset_index().rename(columns={'index':'eindex',f'scaf{sfrom}':'scaf0',f'strand{sfrom}':'strand0'}).merge(bidi_loops[['loop','scaf0','strand0']].reset_index().rename(columns={'index':'bindex'}), on=['loop','scaf0','strand0'], how='inner').drop(columns=['loop','scaf0','strand0'])
            pairs['length'] = np.minimum(exit_conns.loc[pairs['eindex'].values, 'length'].values - sfrom, bidi_loops.loc[pairs['bindex'].values, 'length'].values)
            s = 1
            while len(pairs):
                start_units.append(pairs.loc[pairs['length'] == s, ['eindex','bindex']].copy())
                start_units[-1]['sfrom'] = sfrom
                pairs = pairs[pairs['length'] > s].copy()
                if len(pairs):
                    pairs = pairs[(exit_conns.loc[pairs['eindex'].values, [f'scaf{sfrom+s}',f'strand{sfrom+s}',f'dist{sfrom+s}']].values == bidi_loops.loc[pairs['bindex'].values, [f'scaf{s}',f'strand{s}',f'dist{s}']].values).all(axis=1)].copy()
                    s += 1
        exit_conns['from_units'] = 0
        if len(start_units):
            start_units = pd.concat(start_units, ignore_index=True)
            start_units.sort_values(['eindex','sfrom','bindex'], inplace=True)
            first_match = start_units.groupby(['eindex'])['sfrom'].agg(['min','size'])
            start_units = start_units[start_units['sfrom'] == np.repeat(first_match['min'].values, first_match['size'].values)].copy()
            # Add loop unit information and try to reduce to ploidy alternatives
            start_units[['loop','from','from_side','length']] = bidi_loops.loc[ start_units['bindex'].values, ['loop','scaf0','strand0','length']].values
            start_units['from_side'] = np.where(start_units['from_side'] == '+', 'r', 'l')
            cols = [f'{n}{s}' for s in range(1,start_units['length'].max()) for n in ['scaf','strand','dist']]
            start_units[cols] = bidi_loops.loc[ start_units['bindex'].values, cols].values
            start_units['group'] = start_units['eindex']
            start_units = TryReducingAlternativesToPloidy(start_units, scaf_bridges, ploidy)
            start_units.drop(columns=['group'], inplace=True)
            # Remove exit_conns with more than ploidy units
            nalts = start_units.groupby(['eindex']).size().reset_index(name='nalts')
            exit_conns.loc[nalts['eindex'].values, 'from_units'] = nalts['nalts'].values
            exit_conns = exit_conns[exit_conns['from_units'] <= ploidy].copy()
        exit_conns['ifrom'] = exit_conns.index.values
        exit_conns = exit_conns.merge( exit_conns[['loop','from','from_side','to','to_side','distance','from_alts','from_units','ifrom']].rename(columns={'from':'to','from_side':'to_side','to':'from','to_side':'from_side','from_alts':'to_alts','from_units':'to_units','ifrom':'ito'}), on=['loop','from','from_side','to','to_side','distance'], how='inner')
#
    # Handle direct connections, where no loop unit fits in by chosing one side and then extending it with the other side
    if len(exit_conns) and len(loops):
        direct_conns = exit_conns.loc[(exit_conns['from_units'] == 0) & ((exit_conns['to_units'] != 0) | (exit_conns['ifrom'] < exit_conns['ito'])), ['loop','ifrom','ito','length','from','from_side']+[f'{n}{s}' for s in range(1,exit_conns['length'].max()) for n in ['scaf','strand','dist']]].copy()
        if len(direct_conns):
            direct_conns = GetOtherSideOfExitConnection(direct_conns, exit_conns)
            direct_conns['sfrom'] = 1
    else:
        direct_conns = []
    if len(direct_conns):
        for sfrom in range(1,direct_conns['length'].max()):
            direct_conns['match'] = True
            cur = direct_conns['sfrom'] == sfrom
            direct_conns.loc[cur, 'match'] = (direct_conns.loc[cur, [f'scaf{sfrom}',f'strand{sfrom}']].values == direct_conns.loc[cur, ['rscaf0','rstrand0']].values).all(axis=1)
            for s in range(sfrom+1, direct_conns['length'].max()):
                cur = direct_conns['match'] & (direct_conns['sfrom'] == sfrom) & (direct_conns['length'] > s)
                direct_conns.loc[cur, 'match'] = (direct_conns.loc[cur, [f'scaf{s}',f'strand{s}',f'dist{s}']].values == direct_conns.loc[cur, [f'rscaf{s-sfrom}',f'rstrand{s-sfrom}',f'rdist{s-sfrom}']].values).all(axis=1)
            direct_conns.loc[direct_conns['match'] == False, 'sfrom'] += 1
        direct_conns = direct_conns[direct_conns['match']].drop(columns=['match'])
    if len(direct_conns):
        direct_conns = AppendScaffoldsToExitConnection(direct_conns)
        CheckConsistencyOfVerticalPaths(direct_conns)
    # If we do not have loop units and cannot build a direct connection, we cannot use the exit_conns
    if len(loops) == 0:
        exit_conns = []
    # Handle indirect connections by first finding a valid path through the loop units to connect from and to
    if len(exit_conns):
        indirect_conns = exit_conns.loc[(exit_conns['from_units'] > 0) & (exit_conns['to_units'] > 0) & (exit_conns['ifrom'] < exit_conns['ito']), ['loop','ifrom','ito','length','from','from_side']+[f'{n}{s}' for s in range(1,exit_conns['length'].max()) for n in ['scaf','strand','dist']]].copy()
    else:
        indirect_conns = []
    if len(loops):
        loop_conns, extensions = FindConnectionsBetweenLoopUnits(loops[np.isin(loops['loop'], np.unique(indirect_conns['loop'].values))], scaffold_graph, True)
    if len(loop_conns):
        for i in [1,2]:
            loop_conns = loop_conns.merge(bidi_loops[['lindex','strand0']].reset_index().rename(columns={'index':f'bindex{i}','lindex':f'lindex{i}','strand0':f'dir{i}'}), on=[f'lindex{i}',f'dir{i}'], how='left')
    if len(indirect_conns) and len(start_units):
        indirect_paths = indirect_conns[['ifrom','ito']].reset_index().rename(columns={'index':'cindex'})
        indirect_paths = indirect_paths.merge(start_units[['eindex','bindex']].rename(columns={'eindex':'ifrom','bindex':'bifrom'}), on=['ifrom'], how='left')
        indirect_paths = indirect_paths.merge(start_units[['eindex','bindex']].rename(columns={'eindex':'ito','bindex':'bito'}), on=['ito'], how='left')
        indirect_paths[['lindex','strand0']] = bidi_loops.loc[indirect_paths['bito'].values, ['lindex','strand0']].values # We need to get the bidi_loop that goes in the other direction, because we start from 'from' not 'to'
        indirect_paths.drop(columns=['ifrom','ito','bito'], inplace=True)
        indirect_paths['strand0'] = np.where(indirect_paths['strand0'] == '+', '-', '+')
        indirect_paths = indirect_paths.merge(bidi_loops[['lindex','strand0']].reset_index().rename(columns={'index':'bito'}), on=['lindex','strand0'], how='left').drop(columns=['lindex','strand0'])
        indirect_paths['len'] = 0
        finished_paths = [ indirect_paths[indirect_paths['bifrom'] == indirect_paths['bito']].copy() ]
        indirect_paths = indirect_paths[indirect_paths['bifrom'] != indirect_paths['bito']].copy()
        indirect_paths['bicur'] = indirect_paths['bifrom']
        indirect_paths['nwildcards'] = 0
        s = 0
        while True:
            # Extend path on valid connections with loop units
            new_path = indirect_paths[indirect_paths['len'] == s].rename(columns={'bicur':'bindex1'}).merge(loop_conns.loc[loop_conns['wildcard'] == False, ['bindex1','bindex2']], on=['bindex1'], how='inner')
            if len(new_path) == 0:
                # Add wildcard connections (loop units that do not have an extension and can connect to all other loop units) only if we have no other choice
                new_path = indirect_paths[indirect_paths['len'] == s].rename(columns={'bicur':'bindex1'}).merge(loop_conns.loc[loop_conns['wildcard'], ['bindex1','bindex2']], on=['bindex1'], how='inner')
                new_path['nwildcards'] += 1
                if len(new_path) == 0:
                    break
            new_path.rename(columns={'bindex2':'bicur','bindex1':f'bi{s}'}, inplace=True)
            new_path['len'] += 1
            # Only keep the shortest path of the ones with the lowest number of wildcard connections to each possible loop unit
            indirect_paths[f'bi{s}'] = -1 # We cannot use NaN here because the following first command ignores NaNs and would mix rows
            indirect_paths = pd.concat([indirect_paths, new_path], ignore_index=True)
            indirect_paths.sort_values(['cindex','bifrom','bito','bicur','nwildcards','len'], inplace=True)
            indirect_paths = indirect_paths.groupby(['cindex','bifrom','bito','bicur'], sort=False).first().reset_index()
            # If we reached 'bito' store the path and remove all other path for this search
            finished_paths.append( indirect_paths[indirect_paths['bicur'] == indirect_paths['bito']].drop(columns=['bicur','nwildcards']) )
            indirect_paths = indirect_paths[ indirect_paths.merge(finished_paths[-1][['cindex','bifrom','bito']], on=['cindex','bifrom','bito'], how='left', indicator=True)['_merge'].values == "left_only" ].copy()
            s += 1
        if len(finished_paths):
            finished_paths = pd.concat(finished_paths, ignore_index=True)
            finished_paths['bi'+str(finished_paths['len'].max())] = np.nan # Make room for bito
            for l in np.unique(finished_paths['len']):
                finished_paths.loc[finished_paths['len'] == l, f'bi{l}'] = finished_paths.loc[finished_paths['len'] == l, 'bito']
            finished_paths.drop(columns=['bifrom'], inplace=True) # bi0 is identical to bifrom
            finished_paths['len'] += 1
    else:
        finished_paths = []
    # Now extend along the valid path
    if len(finished_paths):
        indirect_conns['cindex'] = indirect_conns.index.values
        indirect_conns = indirect_conns.merge(finished_paths, on=['cindex'], how='inner')
        indirect_conns.drop(columns=['cindex'], inplace=True)
        for b in range(indirect_conns['len'].max()):
            if b == 0:
                indirect_conns['sfrom'] = indirect_conns[['ifrom','bi0']].rename(columns={'ifrom':'eindex','bi0':'bindex'}).merge(start_units[['eindex','bindex','sfrom']], on=['eindex','bindex'], how='left')['sfrom'].values
            else:
                indirect_conns['sfrom'] = indirect_conns['length']
                indirect_conns.loc[indirect_conns['len'] > b, 'sfrom'] = indirect_conns.loc[indirect_conns['len'] > b, [f'bi{b-1}',f'bi{b}']].rename(columns={f'bi{b-1}':'bindex1',f'bi{b}':'bindex2'}).merge(loop_conns[['bindex1','bindex2','sfrom']], on=['bindex1','bindex2'], how='left')['sfrom'].values + indirect_conns.loc[indirect_conns['len'] > b, 'old_sfrom']
            indirect_conns['old_sfrom'] = indirect_conns['sfrom']
            ext = RemoveEmptyColumns(bidi_loops.loc[indirect_conns.loc[indirect_conns['len'] > b, f'bi{b}']].drop(columns=['lindex','loop']).rename(columns={col:f'r{col}' for col in bidi_loops.columns}))
            ext.index = indirect_conns[indirect_conns['len'] > b].index.values
            indirect_conns = pd.concat([indirect_conns, ext], axis=1)
            indirect_conns['rlength'] = indirect_conns['rlength'].fillna(0).astype(int)
            indirect_conns = AppendScaffoldsToExitConnection(indirect_conns)
        indirect_conns.drop(columns=['old_sfrom'], inplace=True)
        # Check if we have evidence for another loop unit at the end (no extension of the last loop unit is already present in the indirect connection and the loop unit has a valid connection to itself)
        if len(extensions):
            extensions['lindex'] = extensions[['lindex','strand0']].merge(bidi_loops[['lindex','strand0']].reset_index(), on=['lindex','strand0'], how='left')['index'].values
            extensions.rename(columns={'lindex':'bindex'}, inplace=True)
            indirect_conns['extra_unit'] = np.isin(indirect_conns['bito'], np.unique(extensions['bindex'].values)) & np.isin(indirect_conns['bito'], loop_conns.loc[loop_conns['bindex1'] == loop_conns['bindex2'], 'bindex1'].values)
            extensions.rename(columns={f'{n}{s}':f'e{n}{s+1}' for s in range(extensions['length'].max()) for n in ['scaf','strand','dist']}, inplace=True) # We add the second to last position in the loop unit to the extension, such that it is likely that it came from a loop unit if something matches the whole extension
            extensions['length'] += 1
            extensions['blen'] = bidi_loops.loc[extensions['bindex'].values, 'length'].values
            extensions[['escaf0','estrand0','edist1']] = [-1,'',0]
            for l in np.unique(extensions['blen']):
                extensions.loc[extensions['blen'] == l, ['escaf0','estrand0','edist1']] = bidi_loops.loc[extensions.loc[extensions['blen'] == l, 'bindex'].values, [f'scaf{l-2}',f'strand{l-2}',f'dist{l-1}']].values
            extensions.drop(columns=['blen'], inplace=True)
            extensions.rename(columns={'bindex':'bito','length':'elength'}, inplace=True)
            extensions = indirect_conns[indirect_conns['extra_unit']].reset_index().rename(columns={'index':'iindex'}).merge(extensions, on=['bito'], how='left')
            sfrom = 1
            extensions = extensions[extensions['length'] >= sfrom + extensions['elength']].copy()
            while len(extensions):
                extensions['valid'] = False
                for l in np.unique(extensions['elength']):
                    extensions.loc[extensions['elength'] == l, 'valid'] = (extensions.loc[extensions['elength'] == l, [f'scaf{sfrom}',f'strand{sfrom}']+[f'{n}{s}' for s in range(sfrom+1, sfrom+l) for n in ['scaf','strand','dist']]].values == extensions.loc[extensions['elength'] == l, ['escaf0','estrand0']+[f'e{n}{s}' for s in range(1, l) for n in ['scaf','strand','dist']]].values).all(axis=1)
                found_index = np.unique(extensions.loc[extensions['valid'], 'iindex'].values)
                indirect_conns.loc[found_index, 'extra_unit'] = False
                sfrom += 1
                extensions = extensions[(np.isin(extensions['iindex'].values, found_index) == False) & (extensions['length'] >= sfrom + extensions['elength'])].copy()
            # Add the extra loop unit, where we found evidence
            if np.sum(indirect_conns['extra_unit']):
                indirect_conns['sfrom'] = indirect_conns['length'] - np.where(indirect_conns['extra_unit'], 1, 0)
                ext = RemoveEmptyColumns(bidi_loops.loc[indirect_conns.loc[indirect_conns['extra_unit'], 'bito']].drop(columns=['lindex','loop']).rename(columns={col:f'r{col}' for col in bidi_loops}))
                ext.index = indirect_conns[indirect_conns['extra_unit']].index.values
                indirect_conns = pd.concat([indirect_conns, ext], axis=1)
                indirect_conns['rlength'] = indirect_conns['rlength'].fillna(0).astype(int)
                indirect_conns = AppendScaffoldsToExitConnection(indirect_conns)
            indirect_conns.drop(columns=['extra_unit'], inplace=True)
        # Add the exit on the other side
        indirect_conns[['lindex','lstrand0']] = bidi_loops.loc[indirect_conns['bito'].values, ['lindex','strand0']].values # The bindex is the reverse of what is stored in start_units, so reverse it
        indirect_conns['lstrand0'] = np.where(indirect_conns['lstrand0'] == '+', '-', '+')
        indirect_conns['bito'] = indirect_conns[['lindex','lstrand0']].merge(bidi_loops[['lindex','strand0']].reset_index().rename(columns={'index':'bito','strand0':'lstrand0'}), on=['lindex','lstrand0'], how='left')['bito'].values
        indirect_conns.drop(columns=['lindex','lstrand0'], inplace=True)
        indirect_conns['sfrom'] = indirect_conns[['ito','bito']].rename(columns={'ito':'eindex','bito':'bindex'}).merge(start_units[['eindex','bindex','sfrom']], on=['eindex','bindex'], how='left')['sfrom'].values
        indirect_conns = GetOtherSideOfExitConnection(indirect_conns, exit_conns)
        indirect_conns['sfrom'] = indirect_conns['length'] - (indirect_conns['rlength'] - indirect_conns['sfrom'])
        indirect_conns = AppendScaffoldsToExitConnection(indirect_conns)
        indirect_conns.drop(columns=['bito'], inplace=True)
        CheckConsistencyOfVerticalPaths(indirect_conns)
#
    # Reduce number of alternatives in direct and indirect connections together to ploidy
    if len(direct_conns):
        direct_conns['lunits'] = 0
    if len(indirect_conns):
        used_units = []
        for u in range(indirect_conns['len'].max()):
            used_units.append( indirect_conns.loc[indirect_conns['len'] > u, [f'bi{u}']].reset_index().rename(columns={'index':'cindex',f'bi{u}':'bindex'}) )
        used_units = pd.concat(used_units, ignore_index=True)
        used_units['lindex'] = bidi_loops.loc[used_units['bindex'].values, 'lindex'].values
        used_units.drop(columns=['bindex'], inplace=True)
        used_units.sort_values(['cindex','lindex'], inplace=True)
        used_units.drop_duplicates(inplace=True)
        used_units['pos'] = used_units.groupby(['cindex'], sort=False).cumcount()
        indirect_conns.drop(columns=['len']+[f'bi{u}' for u in range(indirect_conns['len'].max())], inplace=True)
        indirect_conns['lunits'] = 0
        lunits = used_units.groupby(['cindex']).size().reset_index(name='lunits')
        indirect_conns.loc[lunits['cindex'].values, 'lunits'] = lunits['lunits'].values
        for u in range(lunits['lunits'].max()):
            indirect_conns[f'li{u}'] = np.nan
            indirect_conns.loc[used_units.loc[used_units['pos'] == u, 'cindex'].values, f'li{u}'] = used_units.loc[used_units['pos'] == u, 'lindex'].values
    if len(direct_conns):
        if len(indirect_conns):
            combined_conns = pd.concat([direct_conns, indirect_conns], ignore_index=True)
        else:
            combined_conns = direct_conns
    else:
        if len(indirect_conns):
            combined_conns = indirect_conns
        else:
            combined_conns = []
    if len(combined_conns):
        combined_conns[['to','to_side']] = [-1,'']
        for l in np.unique(combined_conns['length']):
            combined_conns.loc[combined_conns['length'] == l, ['to','to_side']] = combined_conns.loc[combined_conns['length'] == l, [f'scaf{l-1}',f'strand{l-1}']].values
        combined_conns['to'] = combined_conns['to'].astype(int)
        combined_conns['to_side'] = np.where(combined_conns['to_side'] == '+', 'l', 'r')
        combined_conns.sort_values(['from','from_side','to','to_side'], inplace=True)
        nalts = combined_conns.groupby(['from','from_side','to','to_side'], sort=False).size().values
        nalts = np.repeat(nalts <= ploidy, nalts)
        drop_cols = ['ifrom','ito','lunits']+[f'li{u}' for u in range(lunits['lunits'].max())]
        unbridged_repeats = combined_conns[nalts].drop(columns=drop_cols)
        combined_conns = combined_conns[nalts==False].copy()
    else:
        unbridged_repeats = []
    if len(combined_conns):
        # Get all possible combinations of conns to keep
        combinations = combined_conns[['from','from_side','to','to_side']].copy()
        combinations['c0'] = combinations.index.values
        for h in range(1,ploidy):
            combinations = combinations.merge(combined_conns[['from','from_side','to','to_side']].reset_index().rename(columns={'index':f'c{h}'}), on=['from','from_side','to','to_side'], how='left')
            combinations = combinations[combinations[f'c{h-1}'] < combinations[f'c{h}']].copy()
        combinations['group'] = (combinations.groupby(['from','from_side','to','to_side'], sort=False).cumcount() == 0).cumsum()
        combinations.drop(columns=['from','from_side','to','to_side'], inplace=True)
        # Pick the combinations that include the most exit paths
        comb_inc = []
        for h in range(ploidy):
            cur = combinations[[f'c{h}']].reset_index().rename(columns={f'c{h}':'conn'})
            cur[['e1','e2']] = combined_conns.loc[cur['conn'].values, ['ifrom','ito']].values
            for i in [1,2]:
                comb_inc.append(cur[['index',f'e{i}']].rename(columns={f'e{i}':'exit'}))
        comb_inc = pd.concat(comb_inc, ignore_index=True).drop_duplicates().groupby(['index']).size().reset_index(name='count')
        combinations['ninc'] = 0
        combinations.loc[comb_inc['index'].values, 'ninc'] = comb_inc['count'].values
        ninc = combinations.groupby(['group'], sort=False)['ninc'].agg(['size','max'])
        combinations = combinations[combinations['ninc'] == np.repeat(ninc['max'].values, ninc['size'].values)].copy()
        # Pick the combinations that include the most kinds of loop units
        comb_inc = []
        for h in range(ploidy):
            cur = combinations[[f'c{h}']].reset_index().rename(columns={f'c{h}':'conn'})
            cols = ['lunits']+[f'li{u}' for u in range(combined_conns['lunits'].max())]
            cur[cols] = combined_conns.loc[cur['conn'].values, cols].values
            usize = cur['lunits'].max()
            if np.isnan(usize) == False:
                for u in range(int(usize)):
                    comb_inc.append(cur.loc[cur['lunits'] > u, ['index',f'li{u}']].rename(columns={f'li{u}':'lunit'}))
        if len(comb_inc):
            comb_inc = pd.concat(comb_inc, ignore_index=True).drop_duplicates().groupby(['index']).size().reset_index(name='count')
        combinations['ninc'] = 0
        if len(comb_inc):
            combinations.loc[comb_inc['index'].values, 'ninc'] = comb_inc['count'].values
        ninc = combinations.groupby(['group'], sort=False)['ninc'].agg(['size','max'])
        combinations = combinations[combinations['ninc'] == np.repeat(ninc['max'].values, ninc['size'].values)].drop(columns=['ninc'])
        # Pick the ones with the shortest maximum length and if we have multiple where all length are identical, n arbitrary (the first) one (sorting length with simple bubble sort)
        for h in range(ploidy):
            combinations[f'len{h}'] = combined_conns.loc[combinations[f'c{h}'].values, 'length'].values
        n = ploidy-1
        while(n > 0):
            swapped = 0
            for i in range(n):
                swap = combinations[f'len{i}'] < combinations[f'len{i+1}']
                if np.sum(swap):
                    swapped = i
                    combinations.loc[swap, [f'len{i}',f'len{i+1}']] = combinations.loc[swap, [f'len{i+1}',f'len{i}']].values
            n = swapped
        combinations.sort_values(['group']+[f'len{h}' for h in range(ploidy)], inplace=True)
        combinations = combinations.groupby(['group']).first().reset_index()
        # Pick the chosen indexes and add those connections to unbridged repeats
        chosen_indexes = []
        for h in range(ploidy):
            chosen_indexes.append(combinations[f'c{h}'].values)
        combined_conns = combined_conns.loc[np.concatenate(chosen_indexes)].drop(columns=drop_cols)
        if len(combined_conns):
            if len(unbridged_repeats):
                unbridged_repeats = pd.concat( [unbridged_repeats, combined_conns], ignore_index=True )
            else:
                unbridged_repeats = combined_conns
    if len(unbridged_repeats):
        unbridged_repeats = RemoveEmptyColumns(unbridged_repeats)
#
    # Only keep loop units, where we have unconnected exits or never had exits at all
    if len(exits):
        unconnected_exits = exits[['loop','from','from_side']].drop_duplicates()
        if len(unbridged_repeats):
            unconnected_exits = unconnected_exits[unconnected_exits.merge(unbridged_repeats[['loop','from','from_side']].drop_duplicates(), on=['loop','from','from_side'], how='left', indicator=True)['_merge'].values == "left_only"].copy()
            unconnected_exits = unconnected_exits[unconnected_exits.merge(unbridged_repeats[['loop','to','to_side']].drop_duplicates().rename(columns={'to':'from','to_side':'from_side'}), on=['loop','from','from_side'], how='left', indicator=True)['_merge'].values == "left_only"].copy()
    else:
        unconnected_exits = []
    if len(loops):
        loops = loops[np.isin(loops['loop'], np.setdiff1d(np.unique(exits['loop'].values), np.unique(unconnected_exits['loop'].values))) == False].copy()
    if len(unbridged_repeats):
        unbridged_repeats.drop(columns=['to','to_side'], inplace=True)
        unbridged_repeats.rename(columns={'from':'scaf0','from_side':'strand0'}, inplace=True)
        unbridged_repeats['strand0'] = np.where(unbridged_repeats['strand0'] == 'r', '+', '-')
    if len(loops):
        # Only keep the start position with the most loop units per loop
        loops.sort_values(['loop','scaf0'], inplace=True)
        loop_count = loops.groupby(['loop','scaf0'], sort=False).size().reset_index(name='count')
        loop_count.sort_values(['loop','count'], ascending=[True,False], inplace=True)
        loop_count = loop_count.groupby(['loop'], sort=False).first().reset_index()
        loops = loops.merge(loop_count[['loop','scaf0']], on=['loop','scaf0'], how='inner')
    # Remove loop units already in bridged or unbridged repeats 
    if len(unbridged_repeats):
        if len(bridged_repeats):
            loop_paths = pd.concat([unbridged_repeats, bridged_repeats], ignore_index=True)
        else:
            loop_paths = unbridged_repeats
    else:
        if len(bridged_repeats):
            loop_paths = bridged_repeats
        else:
            loop_paths = []
    if len(loop_paths) and len(loops):
        max_len = loop_paths['length'].max()
        search_space = np.unique(loops['scaf0'].values)
        inner_repeats = []
        for s1 in range(1, max_len-1):
            for s2 in range(s1+1, max_len-1):
                found_repeats = loop_paths.loc[np.isin(loop_paths[f'scaf{s1}'], search_space) & (loop_paths[f'scaf{s1}'] == loop_paths[f'scaf{s2}']) & (loop_paths[f'strand{s1}'] == loop_paths[f'strand{s2}']), ['length']+[f'scaf{s1}',f'strand{s1}']+[f'{n}{s}' for s in range(s1+1, s2+1) for n in ['scaf','strand','dist']]].rename(columns={f'{n}{s}':f'{n}{s-s1}' for s in range(s1, s2+1) for n in ['scaf','strand','dist']})
                if len(found_repeats):
                    found_repeats['length'] = s2-s1
                    inner_repeats.append(found_repeats[found_repeats['strand0'] == '+'].copy())
                    found_repeats = found_repeats[found_repeats['strand0'] == '-'].copy()
                    if len(found_repeats):
                        found_repeats = ReverseVerticalPaths(found_repeats)
                        inner_repeats.append(found_repeats.drop(columns=['lindex']))
        if len(inner_repeats):
            inner_repeats = pd.concat(inner_repeats, ignore_index=True, sort=False)
        if len(inner_repeats):
            inner_repeats.drop_duplicates(inplace=True)
            loops = loops[ loops.merge(inner_repeats, on=list(inner_repeats.columns.values), how='left', indicator=True)['_merge'].values == "left_only" ].copy()
#
    # Add remaining loop units to loop_paths, get all scaffold connections in the path and make it a vertical path
    if len(loops):
        if len(loop_paths):
            loop_paths = pd.concat([loop_paths, loops], ignore_index=True).drop(columns=['loop'])
        else:
            loop_paths = loops.drop(columns=['loop'])
    if len(loop_paths):
        handled_scaf_conns = GetHandledScaffoldConnectionsFromVerticalPath(loop_paths)
        loop_paths = TurnHorizontalPathIntoVertical(loop_paths, scaffold_paths['pid'].max()+1)
        scaffold_paths = pd.concat([scaffold_paths, loop_paths], ignore_index=True)
    else:
        handled_scaf_conns = []
# 
    return scaffold_paths, handled_scaf_conns

def AddAlternativesToInversionsToPaths(inv_paths, inversions, scaffold_graph):
    # Add all paths to inv_path between the two outer scaffolds that go through the inverted scaffold
    pot_paths = scaffold_graph.merge(inversions, on=['from','from_side','scaf1'], how='inner')
    for l in range(3,scaffold_graph['length'].max()):
        pot_paths = pot_paths[pot_paths['length'] >= l].copy()
        cur = pot_paths.loc[(pot_paths[f'scaf{l-2}'] == pot_paths['scaf1']) & (pot_paths[f'scaf{l-1}'] == pot_paths['escaf']) & (pot_paths[f'strand{l-1}'] == pot_paths['estrand']), ['length','from','from_side']+[f'{n}{s}' for s in range(1,l) for n in ['scaf','strand','dist']]].rename(columns={'from':'scaf0','from_side':'strand0'})
        if len(cur):
            cur['strand0'] = np.where(cur['strand0'] == 'r', '+', '-')
            cur['length'] = l
            cur.drop_duplicates(inplace=True)
            inv_paths.append(cur)
#
    return inv_paths

def RemoveInvertedRepeatsThatMightBeTwoHaplotypesOnOneSide(cur, scaffold_graph, max_length):
    # Find the connected scaffold through the repeat also on the not repeat side
    cur['other_side'] = np.where(cur['to_side'] == 'r','l','r')
    remove = cur[['to','other_side','from']].reset_index().rename(columns={'from':'veto','to':'from','other_side':'from_side'}).merge(scaffold_graph, on=['from','from_side'], how='inner')
    if len(remove):
        remove['veto_pos'] = -1
        for s in range(remove['length'].max()-1,0,-1):
            remove.loc[remove[f'scaf{s}'] == remove['veto'], 'veto_pos'] = s
        remove = remove.loc[remove['veto_pos'] > 0].copy()
    # If the reversed path goes to somewhere else and not the inverted repeat it was a false alarm
    if len(remove):
        # Get the found path up to the veto
        pot_vetos = []
        for l1 in np.unique(remove['veto_pos'].values):
            pot_vetos.append( remove.loc[remove['veto_pos'] == l1, ['index','veto_pos','from','from_side']+[f'{n}{s}' for s in range(1,l1+1) for n in ['scaf','strand','dist']]].drop_duplicates() )
        pot_vetos = pd.concat(pot_vetos, ignore_index=True)
        pot_vetos.rename(columns={'index':'dindex','from':'scaf0','from_side':'strand0','veto_pos':'length'}, inplace=True)
        pot_vetos['strand0'] = np.where(pot_vetos['strand0'] == 'r', '+', '-')
        # Append the inverted repeat to the paths
        pot_vetos.rename(columns={f'{n}{s}':f'{n}{s+max_length-1}' for s in range(pot_vetos['length'].max()+1) for n in ['scaf','strand','dist']}, inplace=True)
        pot_vetos[[f'{n}{s}' for s in range(max_length-1) for n in ['scaf','strand','dist']]+[f'dist{max_length-1}']] = cur.loc[pot_vetos['dindex'].values, [f'{n}{s}' for s in range(1,max_length) for n in ['scaf','strand','dist']]+[f'dist{max_length}']].values
        pot_vetos.drop(columns=['dist0'], inplace=True)
        pot_vetos['length'] += max_length
        # Reverse it
        remove = ReverseVerticalPaths(pot_vetos)
        remove['lindex'] = pot_vetos.loc[remove['lindex'].values, 'dindex'].values
        remove.rename(columns={'lindex':'dindex'}, inplace=True)
        # Check if we find a consistent paths (which means we have evidence for two haplotypes on one side instead of both sides of an inverted repeat)
        remove.rename(columns={'scaf0':'from','strand0':'from_side'}, inplace=True)
        remove['from_side'] = np.where(remove['from_side'] == '+', 'r', 'l')
        pairs = remove[['from','from_side','scaf1','strand1','dist1']].reset_index().rename(columns={'index':'rindex'}).merge(scaffold_graph[['from','from_side','scaf1','strand1','dist1']].reset_index().rename(columns={'index':'sindex'}), on=['from','from_side','scaf1','strand1','dist1'], how='inner').drop(columns=['from','from_side','scaf1','strand1','dist1'])
        pairs['length'] = np.minimum(remove.loc[pairs['rindex'].values, 'length'].values, scaffold_graph.loc[pairs['sindex'].values, 'length'].values)
        s = 2
        valid_removal = [ np.unique(remove.loc[pairs.loc[pairs['length'] <= s, 'rindex'].values, 'dindex'].values) ]
        pairs = pairs[np.isin(remove.loc[pairs['rindex'].values, 'dindex'].values, valid_removal[-1]) == False].copy()
        while len(pairs):
            pairs = pairs[(remove.loc[pairs['rindex'].values, [f'scaf{s}',f'strand{s}',f'dist{s}']].values == scaffold_graph.loc[pairs['sindex'].values, [f'scaf{s}',f'strand{s}',f'dist{s}']].values).all(axis=1)].copy()
            s += 1
            valid_removal.append( np.unique(remove.loc[pairs.loc[pairs['length'] <= s, 'rindex'].values, 'dindex'].values) )
            pairs = pairs[np.isin(remove.loc[pairs['rindex'].values, 'dindex'].values, valid_removal[-1]) == False].copy()
        valid_removal = np.concatenate(valid_removal)
        # Drop the valid_removal (and their partners)
        cur.drop(columns=['other_side'], inplace=True)
        cur.drop(valid_removal, inplace=True)
        mcols = ['from','from_side','to','to_side']+[f'{n}{s}' for s in range(1,max_length) for n in ['scaf','strand','dist']]+[f'dist{max_length}']
        cur = cur.merge(cur[mcols].rename(columns={**{'from':'to','to':'from','from_side':'to_side','to_side':'from_side'}, **{f'dist{s}':f'dist{max_length+1-s}' for s in range(1,max_length+1)}}), on=mcols, how='inner')
#
    return cur

def AddPathThroughInvertedRepeats(scaffold_paths, handled_scaf_conns, scaffold_graph, scaf_bridges, ploidy):
    inv_paths = []
    # Check for bridged inverted repeats without a scaffold in between
    if scaffold_graph['length'].max() > 3:
        bridged_inversions = []
        l = 4
        while True:
            if scaffold_graph['length'].max() < l:
                cur = []
            else:
                cur = scaffold_graph.loc[(scaffold_graph['length'] >= l) & (scaffold_graph['scaf1'] == scaffold_graph[f'scaf{l-2}']) & (scaffold_graph['strand1'] != scaffold_graph[f'strand{l-2}']), ['from','from_side']+[f'{n}{s}' for s in range(1,l) for n in ['scaf','strand','dist']]].copy()
                for s in range(2,l//2):
                    cur = cur[(cur[f'scaf{s}'] == cur[f'scaf{l-s-1}']) & (cur[f'strand{s}'] != cur[f'strand{l-s-1}'])].copy()
            if len(cur):
                cur.drop_duplicates(inplace=True)
                # Filter the ones that will show up in the next iteration again (and if they are too short to do so, they are not truely bridged)
                cur = cur[(cur['from'] != cur[f'scaf{l-1}']) | ((cur['from_side'] == 'r') == (cur[f'strand{l-1}'] == '+'))].copy()
                # Only keep one direction for each inverted repeat
                if len(cur):
                    cur = cur[(cur['from'] < cur[f'scaf{l-1}']) | ((cur['from'] == cur[f'scaf{l-1}']) & (cur['from_side'] == 'r'))].copy()
                    # Only keep the relevant columns to later find all connections that match this
                    cur = cur[['from','from_side','scaf1']+[f'scaf{l-1}',f'strand{l-1}']].drop_duplicates().rename(columns={f'scaf{l-1}':'escaf',f'strand{l-1}':'estrand'})
                    bridged_inversions.append(cur)
                l += 2
            else:
                break
        if len(bridged_inversions):
            bridged_inversions = pd.concat(bridged_inversions, ignore_index=True)
            # Add all paths to inv_path between the two outer scaffolds that go through the inverted scaffold
            inv_paths = AddAlternativesToInversionsToPaths(inv_paths, bridged_inversions, scaffold_graph)
#
    # Check for scaffolds that are uniquely connected by an inverted (but not bridged) repeat
    if scaffold_graph['length'].max() > 2:
        unique_inversions = []
        l = 3
        while True:
            if scaffold_graph['length'].max() < l:
                cur = []
            else:
                max_length = 2*l - 3
                cur = scaffold_graph.loc[(scaffold_graph['length'] >= l) & (scaffold_graph['length'] <= max_length) & (scaffold_graph[f'scaf{l-2}'] == scaffold_graph[f'scaf{l-1}']) & (scaffold_graph[f'strand{l-2}'] != scaffold_graph[f'strand{l-1}']), ['length','from','from_side']+[f'{n}{s}' for s in range(1,np.minimum(max_length, scaffold_graph['length'].max())) for n in ['scaf','strand','dist']]].copy()
                for s in range(1,l-2):
                    if cur['length'].max() > max_length-s:
                        cur = cur[(cur['length'] <= max_length-s) | ((cur[f'scaf{s}'] == cur[f'scaf{max_length-s}']) & (cur[f'strand{s}'] != cur[f'strand{max_length-s}']))].copy()
            if len(cur):
                cur.drop_duplicates(inplace=True)
                cur.reset_index(drop=True, inplace=True)
                cur['ifrom'] = cur.index.values
                # Guarantee we have all possible dist columns
                cur[[f'dist{s}' for s in range(cur['length'].max(), max_length)]] = np.nan
                # Combine the paths with the same inverted repeats
                cols = ['ifrom','from','from_side']+[f'{n}{s}' for s in range(1,l) for n in ['scaf','strand']]+[f'dist{s}' for s in range(1,max_length)]
                cur = cur[cols].merge(cur[cols].rename(columns={**{'from':'to','from_side':'to_side','ifrom':'ito'}, **{f'dist{s}':f'tdist{max_length-s+1}' for s in range(1,max_length) if s != l-1}}), on=[f'{n}{s}' for s in range(1,l) for n in ['scaf','strand']]+[f'dist{l-1}'])
                cur = cur[(cur['ifrom'] != cur['ito'])].drop(columns=['ito','ifrom'])
                check_dist = [s for s in list(range(2,l-1)) if (np.isnan(cur[f'tdist{s}']) == False).any()] + [s for s in list(range(l,max_length)) if (np.isnan(cur[f'dist{s}']) == False).any()]
                if len(check_dist):
                    cur = cur[((cur[[f'dist{s}' for s in check_dist]].values == cur[[f'tdist{s}' for s in check_dist]].values) | np.isnan(cur[[f'dist{s}' for s in check_dist]].values) | np.isnan(cur[[f'tdist{s}' for s in check_dist]].values)).all(axis=1)].copy()
                cur.drop(columns=[f'dist{s}' for s in range(l,max_length)]+[f'tdist{s}' for s in range(2,l-1)], inplace=True)
                cur.rename(columns={f'tdist{s}':f'dist{s}' for s in range(l,max_length+1)}, inplace=True)
                if len(cur):
                    # Only keep the paths through the inverted repeats that do not have alternatives
                    cur.drop_duplicates(inplace=True)
                    cols = ['from','from_side']+[f'{n}{s}' for s in range(1,l-1) for n in ['scaf','strand','dist']]+[f'dist{l-1}']
                    cur.sort_values(cols, inplace=True)
                    nalts = cur.groupby(cols, sort=False).size().values
                    cur = cur[np.repeat(nalts==1, nalts)].copy()
                    # Filter the ones that will show up in the next iteration again (and if they are too short to do so, they are not guaranteed to be unique)
                    cur = cur[(cur['from'] != cur['to']) | (cur['from_side'] != cur['to_side'])].copy()
                if len(cur):
                    if l < max_length:
                        cur[[f'{n}{s}' for s in range(l,max_length) for n in ['scaf','strand']]] = cur[[f'{n}{s}' for s in range(l-3,0,-1) for n in ['scaf','strand']]].values
                        cur[[f'strand{s}' for s in range(l,max_length)]] = np.where(cur[[f'strand{s}' for s in range(l,max_length)]].values == '+', '-', '+')
                    # Remove the ones, where we have evidence that we might have two haplotypes on one side instead of both sides of the inverted repeat
                    if ploidy > 1:
                        cur = RemoveInvertedRepeatsThatMightBeTwoHaplotypesOnOneSide(cur, scaffold_graph, max_length)
                if len(cur):
                    # Only keep one direction
                    cur = cur[(cur['from'] < cur['to']) | ((cur['from'] == cur['to']) & (cur['from_side'] == 'r'))].copy()
                    # Add the valid paths to unique_inversions to later get all (bridged) alternatives and to inv_paths directly, because we will not retrieve them, when we check for the alternatives
                    cur['to_side'] = np.where(cur['to_side'] == 'l', '+', '-')
                    unique_inversions.append( cur[['from','from_side','scaf1','to','to_side']].rename(columns={'to':'escaf','to_side':'estrand'}) )
                    cur.rename(columns={'from':'scaf0','from_side':'strand0','to':f'scaf{max_length}','to_side':f'strand{max_length}'}, inplace=True)
                    cur['strand0'] = np.where(cur['strand0'] == 'r', '+', '-')
                    cur['length'] = max_length+1
                    inv_paths.append(cur[['length','scaf0','strand0']+[f'{n}{s}' for s in range(1,max_length+1) for n in ['scaf','strand','dist']]])
                l += 1
            else:
                break
        # Add all paths to inv_path between the two outer scaffolds that go through the inverted scaffold
        if len(unique_inversions):
            unique_inversions = pd.concat(unique_inversions, ignore_index=True)
            inv_paths = AddAlternativesToInversionsToPaths(inv_paths, unique_inversions, scaffold_graph)
#
    # Check for inverted repeat, which only have two connected scaffolds, so the paths is clear
    inv_scafs = scaf_bridges.loc[(scaf_bridges['from'] == scaf_bridges['to']) & (scaf_bridges['from_side'] == scaf_bridges['to_side']) & (scaf_bridges['from_alt'] == 1), ['from','from_side','mean_dist']].rename(columns={'mean_dist':'idist0'})
    inv_scafs['from_side'] = np.where(inv_scafs['from_side'] == 'r', 'l', 'r') # Flip 'from_side' so it is pointing in the direction of the continuation not the inversion
    s = 0
    two_connections = []
    inv_scafs[f'iscaf{s}'] = inv_scafs['from']
    inv_scafs[f'istrand{s}'] = np.where(inv_scafs['from_side'] == 'l', '+', '-') # Choice is arbitrary here, just needs to be consistent
    while len(inv_scafs):
        inv_scafs['length'] = s+1
        inv_scafs['nalt'] = inv_scafs[['from','from_side']].merge(scaf_bridges[['from','from_side','from_alt']].drop_duplicates(), on=['from','from_side'], how='left')['from_alt'].fillna(0).astype(int).values
        two_connections.append(inv_scafs[inv_scafs['nalt'] == 2].drop(columns=['nalt']))
        # If we have a single connection, we can continue and see if we find two connections in the next scaffold
        inv_scafs = inv_scafs[inv_scafs['nalt'] == 1].merge(scaf_bridges.loc[scaf_bridges['to_alt'] == 1, ['from','from_side','to','to_side','mean_dist']], on=['from','from_side'], how='inner')
        if len(inv_scafs):
            s += 1
            inv_scafs.rename(columns={'mean_dist':f'idist{s}'}, inplace=True)
            inv_scafs['from'] = inv_scafs['to']
            inv_scafs['from_side'] = np.where(inv_scafs['to_side'] == 'r', 'l', 'r')
            inv_scafs.drop(columns=['to','to_side'], inplace=True)
            inv_scafs[f'iscaf{s}'] = inv_scafs['from']
            inv_scafs[f'istrand{s}'] = np.where(inv_scafs['from_side'] == 'l', '+', '-') # Choice is arbitrary here, just needs to be consistent
    if len(two_connections):
        two_connections = pd.concat(two_connections, ignore_index=True)
    if len(two_connections):
        # Create the paths through the inverted repeat
        for l in np.unique(two_connections['length']):
            for s in range(l):
                two_connections.loc[two_connections['length'] == l, [f'scaf{l-s}',f'strand{l-s}',f'strand{l-s+1}']] = two_connections.loc[two_connections['length'] == l, [f'iscaf{s}',f'istrand{s}',f'idist{s}']].values
                two_connections.loc[two_connections['length'] == l, [f'scaf{l+1+s}',f'strand{l+1+s}',f'dist{l+1+s}']] = two_connections.loc[two_connections['length'] == l, [f'iscaf{s}',f'istrand{s}',f'idist{s}']].values
                two_connections.loc[two_connections['length'] == l, f'strand{l+1+s}'] = np.where(two_connections.loc[two_connections['length'] == l, f'strand{l+1+s}'] == '+', '-', '+')
        two_connections['length'] = two_connections['length']*2+2
        exits = two_connections[['from','from_side']].reset_index().merge(scaf_bridges[['from','from_side','to','to_side','mean_dist']], on=['from','from_side'], how='left')
        exits.sort_values(['index','to','to_side'], inplace=True)
        start = exits.groupby(['index']).first().reset_index()
        two_connections.loc[start['index'].values, 'scaf0'] = start['to'].values
        two_connections.loc[start['index'].values, 'strand0'] = np.where(start['to_side'].values == 'r', '+', '-') # Needs to be consistent with previous choice
        two_connections.loc[start['index'].values, 'dist1'] = start['mean_dist'].values
        ends = exits.groupby(['index']).last().reset_index()
        ends['len'] = two_connections.loc[ends['index'].values, 'length'].values
        for l in np.unique(two_connections['length']):
            two_connections.loc[ends.loc[ends['len'] == l, 'index'].values, f'scaf{l-1}'] = ends.loc[ends['len'] == l, 'to'].values
            two_connections.loc[ends.loc[ends['len'] == l, 'index'].values, f'strand{l-1}'] = np.where(ends.loc[ends['len'] == l, 'to_side'].values == 'l', '+', '-') # Needs to be consistent with previous choice
            two_connections.loc[ends.loc[ends['len'] == l, 'index'].values, f'dist{l-1}'] = ends.loc[ends['len'] == l, 'mean_dist'].values
        # Remove the ones, where we have evidence that we might have two haplotypes on one side instead of both sides of the inverted repeat
        if ploidy > 1:
            valid_connections = []
            for l in np.unique(two_connections['length']):
                cur = two_connections.loc[two_connections['length'] == l, ['length','scaf0','strand0']+[f'{n}{s}' for s in range(1,l) for n in ['scaf','strand','dist']]].rename(columns={'scaf0':'from','strand0':'from_side',f'scaf{l-1}':'to',f'strand{l-1}':'to_side'})
                cur['from_side'] = np.where(cur['from_side'] == '+', 'r', 'l')
                cur['to_side'] = np.where(cur['to_side'] == '+', 'l', 'r')
                cur = pd.concat([cur, cur.rename(columns={**{'from':'to','to':'from','from_side':'to_side','to_side':'from_side'}, **{f'dist{s}':f'dist{l-s}' for s in range(1,l)}})], ignore_index=True)
                cur = RemoveInvertedRepeatsThatMightBeTwoHaplotypesOnOneSide(cur, scaffold_graph, l-1)
                cur.rename(columns={'from':'scaf0','from_side':'strand0','to':f'scaf{l-1}','to_side':f'strand{l-1}'}, inplace=True)
                cur['strand0'] = np.where(cur['strand0'] == 'r', '+', '-')
                cur[f'strand{l-1}'] = np.where(cur[f'strand{l-1}'] == 'l', '+', '-')
                valid_connections.append(cur)
            two_connections = pd.concat(valid_connections)
    if len(two_connections):
        inv_paths.append( two_connections[['length','scaf0','strand0']+[f'{n}{s}' for s in range(1,two_connections['length'].max()) for n in ['scaf','strand','dist']]].copy() )
#
    # Filter the inv_paths that are fully included in another inv_paths
    if len(inv_paths):
        inv_paths = pd.concat(inv_paths, ignore_index=True).drop_duplicates()
        inv_paths['included'] = False
        for sfrom in range(inv_paths['length'].max()-2):
            for l in range(3,inv_paths['length'].max()+1-sfrom):
                check = inv_paths.loc[inv_paths['length'] >= sfrom+l, ['length',f'scaf{sfrom}',f'strand{sfrom}']+[f'{n}{s}' for s in range(sfrom+1,sfrom+l) for n in ['scaf','strand','dist']]].rename(columns={f'{n}{s}':f'{n}{s-sfrom}' for s in range(sfrom,sfrom+l) for n in ['scaf','strand','dist']})
                if sfrom == 0:
                    check = check[check['length'] > l].copy() # To avoid self-matching of inv_paths
                check['length'] = l
                inv_paths.loc[inv_paths[check.columns].merge(check, on=list(check.columns.values), how='left', indicator=True)['_merge'].values == "both", 'included'] = True
        inv_paths = inv_paths[inv_paths['included'] == False].copy()
#
    # Add information to scaffold_paths and handled_scaf_conns
    if len(inv_paths):
        if len(handled_scaf_conns):
            handled_scaf_conns = pd.concat([ handled_scaf_conns, GetHandledScaffoldConnectionsFromVerticalPath(inv_paths)], ignore_index=True).drop_duplicates()
        else:
            handled_scaf_conns = GetHandledScaffoldConnectionsFromVerticalPath(inv_paths)
        inv_paths = TurnHorizontalPathIntoVertical(inv_paths, scaffold_paths['pid'].max()+1)
        scaffold_paths = pd.concat([scaffold_paths, inv_paths], ignore_index=True)
#
    return scaffold_paths, handled_scaf_conns

def AddUntraversedConnectedPaths(scaffold_paths, knots, scaffold_graph, handled_scaf_conns):
    # Get untraversed connections to neighbouring scaffolds in graph
    conn_path = scaffold_graph.loc[np.isin(scaffold_graph.index.values, knots['oindex'].values) == False, ['from','from_side','scaf1','strand1','dist1']].drop_duplicates()
    # Require that it is untraversed in both directions, otherwise we would duplicate the end of a phased path
    conn_path['to_side'] = np.where(conn_path['strand1'] == '+', 'l', 'r')
    conn_path = conn_path.merge(conn_path[['from','from_side','scaf1','to_side','dist1']].rename(columns={'from':'scaf1','from_side':'to_side','scaf1':'from','to_side':'from_side'}), on=['from','from_side','scaf1','to_side','dist1'], how='inner')
    # Only use one of the two directions for the path
    conn_path = conn_path[(conn_path['from'] < conn_path['scaf1']) | ((conn_path['from'] == conn_path['scaf1']) & ((conn_path['from_side'] == conn_path['to_side']) | (conn_path['from_side'] == 'r')))].drop(columns=['to_side'])
    # Require that is has not been handled before
    conn_path.rename(columns={'from':'scaf0','from_side':'strand0'}, inplace=True)
    conn_path['strand0'] = np.where(conn_path['strand0'] == 'r', '+', '-')
    if len(handled_scaf_conns):
        conn_path = conn_path[ conn_path.merge(handled_scaf_conns, on=['scaf0','strand0','scaf1','strand1','dist1'], how='left', indicator=True)['_merge'].values == "left_only" ].copy()
    # Turn into proper format for scaffold_paths
    conn_path['pid'] = np.arange(len(conn_path)) + 1 + scaffold_paths['pid'].max()
    conn_path['dist0'] = 0
    conn_path['pos'] = 0
    conn_path['pos1'] = 1
    scaffold_paths = pd.concat([scaffold_paths, conn_path[['pid','pos','scaf0','strand0','dist0']], conn_path[['pid','pos1','scaf1','strand1','dist1']].rename(columns={'pos1':'pos','scaf1':'scaf0','strand1':'strand0','dist1':'dist0'})], ignore_index=True)
    scaffold_paths.sort_values(['pid','pos'], inplace=True)

    return scaffold_paths

def AddUnconnectedPaths(scaffold_paths, scaffolds, scaffold_graph):
    # All the scaffolds that are not in scaffold_graph, because they do not have connections are inserted as their own knot
    unconn_path = scaffolds[['scaffold']].rename(columns={'scaffold':'scaf0'}).merge(scaffold_graph[['from']].rename(columns={'from':'scaf0'}), on=['scaf0'], how='left', indicator=True)
    unconn_path = unconn_path.loc[unconn_path['_merge'] == "left_only", ['scaf0']].copy()
    unconn_path['strand0'] = '+'
    unconn_path['dist0'] = 0
    unconn_path['pid'] = np.arange(len(unconn_path)) + 1 + scaffold_paths['pid'].max()
    unconn_path['pos'] = 0

    scaffold_paths = pd.concat([scaffold_paths, unconn_path[['pid','pos','scaf0','strand0','dist0']]], ignore_index=True)

    return scaffold_paths

def CheckScaffoldPathsConsistency(scaffold_paths):
    # Check that the first positions have zero distance
    check = scaffold_paths[(scaffold_paths['pos'] == 0) & (scaffold_paths[[col for col in scaffold_paths.columns if col[:4] == "dist"]] != 0).any(axis=1)].copy()
    if len(check):
        print("Warning: First distances are not zero.")
        print(check)
#
    # Check that positions are consistent in scaffold_paths
    inconsistent = scaffold_paths[(scaffold_paths['pos'] < 0) |
                                  ((scaffold_paths['pos'] == 0) & (scaffold_paths['pid'] == scaffold_paths['pid'].shift(1))) |
                                  ((scaffold_paths['pos'] > 0) & ((scaffold_paths['pid'] != scaffold_paths['pid'].shift(1)) | (scaffold_paths['pos'] != scaffold_paths['pos'].shift(1)+1)))].copy()
    if len(inconsistent):
        print("Warning: Scaffold paths got inconsistent.")
        print(inconsistent)
    # Check that we do not have a phase 0, because it cannot be positive/negative as required for a phase
    inconsistent = scaffold_paths[(scaffold_paths[[col for col in scaffold_paths.columns if col[:5] == "phase"]] == 0).any(axis=1)].copy()
    if len(inconsistent):
        print("Warning: Scaffold paths has a zero phase.")
        print(inconsistent)

def CheckIfScaffoldPathsFollowsValidBridges(scaffold_paths, scaf_bridges, ploidy):
    test_paths = scaffold_paths.copy()
    if 'phase0' not in test_paths.columns:
        test_paths['phase0'] = 1
    for h in range(1,ploidy):
        test_paths.loc[test_paths[f'phase{h}'] < 0, [f'scaf{h}',f'strand{h}',f'dist{h}']] = test_paths.loc[test_paths[f'phase{h}'] < 0, ['scaf0','strand0','dist0']].values
    test_bridges = []
    for h in range(ploidy):
        cur = test_paths.loc[(test_paths[f'scaf{h}'] >= 0), ['pid','pos',f'phase{h}',f'scaf{h}',f'strand{h}',f'dist{h}']].copy()
        cur['from'] = cur[f'scaf{h}'].shift(1, fill_value=-1)
        cur['from_side'] = np.where(cur[f'strand{h}'].shift(1, fill_value='') == '+', 'r', 'l')
        cur = cur[(cur['pid'] == cur['pid'].shift(1)) & ((cur[f'phase{h}'] > 0) | (cur[f'phase{h}'].shift(1) > 0))].drop(columns=[f'phase{h}'])
        cur.rename(columns={f'scaf{h}':'to',f'strand{h}':'to_side',f'dist{h}':'mean_dist'}, inplace=True)
        cur['to_side'] = np.where(cur['to_side'] == '+', 'l', 'r')
        cur['hap'] = h
        test_bridges.append(cur)
    test_bridges = pd.concat(test_bridges, ignore_index=True)
    test_bridges = test_bridges.groupby(['pid','pos','from','from_side','to','to_side','mean_dist'])['hap'].min().reset_index()
    test_bridges = test_bridges[ test_bridges.merge(scaf_bridges[['from','from_side','to','to_side','mean_dist']], on=['from','from_side','to','to_side','mean_dist'], how='left', indicator=True)['_merge'].values == 'left_only' ].copy()
    if len(test_bridges):
        print("Scaffold path contains invalid bridges.")
        print(test_bridges[['pid','pos','hap','from','from_side','to','to_side','mean_dist']])

def ReverseScaffolds(scaffold_paths, reverse, ploidy):
    # Reverse Strand
    for h in range(ploidy):
        sreverse = reverse & (scaffold_paths[f'strand{h}'] != '')
        scaffold_paths.loc[sreverse, f'strand{h}'] = np.where(scaffold_paths.loc[sreverse, f'strand{h}'] == '+', '-', '+')
    # Reverse distance and phase
    for h in range(ploidy-1,-1,-1):
        # Shift distances in alternatives
        missing_information = reverse & (scaffold_paths['pid'] == scaffold_paths['pid'].shift(-1)) & (scaffold_paths[f'phase{h}'] < 0) & (scaffold_paths[f'phase{h}'].shift(-1) >= 0)
        if np.sum(missing_information):
            scaffold_paths.loc[missing_information, [f'scaf{h}', f'strand{h}', f'dist{h}']] = scaffold_paths.loc[missing_information, ['scaf0','strand0','dist0']].values
            scaffold_paths.loc[missing_information, [f'phase{h}']] = -scaffold_paths.loc[missing_information, [f'phase{h}']]
        scaffold_paths.loc[reverse & (scaffold_paths[f'phase{h}'] < 0), f'dist{h}'] = scaffold_paths.loc[reverse & (scaffold_paths[f'phase{h}'] < 0), 'dist0']
        scaffold_paths[f'dist{h}'] = np.where(reverse, scaffold_paths[f'dist{h}'].shift(-1, fill_value=0), scaffold_paths[f'dist{h}'])
        scaffold_paths.loc[reverse & (scaffold_paths['pid'] != scaffold_paths['pid'].shift(-1)) | (scaffold_paths[f'phase{h}'] < 0), f'dist{h}'] = 0
        while True:
            scaffold_paths['jumped'] = (scaffold_paths[f'phase{h}'] >= 0) & (scaffold_paths[f'scaf{h}'] < 0) & (scaffold_paths[f'dist{h}'] != 0) # Jump deletions
            if 0 == np.sum(scaffold_paths['jumped']):
                break
            else:
                scaffold_paths.loc[scaffold_paths['jumped'].shift(-1, fill_value=False), f'dist{h}'] = scaffold_paths.loc[scaffold_paths['jumped'], f'dist{h}'].values
                scaffold_paths.loc[scaffold_paths['jumped'], f'dist{h}'] = 0
        # We also need to shift the phase, because the distance variation at the end of the bubble is on the other side now
        scaffold_paths[f'phase{h}'] = np.where(reverse & (scaffold_paths['pid'] == scaffold_paths['pid'].shift(-1)), np.sign(scaffold_paths[f'phase{h}'])*np.abs(scaffold_paths[f'phase{h}'].shift(-1, fill_value=0)), scaffold_paths[f'phase{h}'])
    scaffold_paths.drop(columns=['jumped'], inplace=True)
    scaffold_paths = TrimAlternativesConsistentWithMain(scaffold_paths, ploidy)
    # Reverse ordering
    scaffold_paths.loc[reverse, 'pos'] *= -1
    scaffold_paths.sort_values(['pid','pos'], inplace=True)
    scaffold_paths['pos'] = scaffold_paths.groupby(['pid'], sort=False).cumcount()
#
    return scaffold_paths

def GetDuplications(scaffold_paths, ploidy, groups=[]):
    # Collect all the haplotypes for comparison
    if 1 == ploidy:
        duplications = scaffold_paths[['scaf0','pid','pos']].rename(columns={'scaf0':'scaf'})
        duplications['hap'] = 0
        pass
    else:
        duplications = []
        for h in range(ploidy):
            duplications.append(scaffold_paths.loc[scaffold_paths[f'phase{h}'] >= 0,[f'scaf{h}','pid','pos']].rename(columns={f'scaf{h}':'scaf'}))
            duplications[-1]['hap'] = h
        duplications = pd.concat(duplications, ignore_index=True)
    duplications = duplications[duplications['scaf'] >= 0].copy() # Ignore deletions
    # Get duplications
    if len(groups):
        duplications = duplications.merge(groups, on=['pid'], how='inner')
        mcols = ['scaf','group']
    else:
        mcols = ['scaf']
    duplications = duplications.rename(columns={col:f'a{col}' for col in duplications.columns if col not in mcols}).merge(duplications.rename(columns={col:f'b{col}' for col in duplications.columns if col not in mcols}), on=mcols, how='left') # 'left' keeps the order and we always have at least the self mapping, thus 'inner' does not reduce the size
    duplications = duplications[(duplications['apid'] != duplications['bpid'])].drop(columns=['scaf']) # Remove self mappings
#
    return duplications

def AddStrandToDuplications(duplications, scaffold_paths, ploidy):
    for p in ['a','b']:
        dup_strans = duplications[[f'{p}pid',f'{p}pos']].rename(columns={f'{p}pid':'pid',f'{p}pos':'pos'}).merge(scaffold_paths[['pid','pos']+[f'strand{h}' for h in range(ploidy)]], on=['pid','pos'], how='left')
        duplications[f'{p}strand'] = dup_strans['strand0'].values
        for h in range(1, ploidy):
            hap = duplications[f'{p}hap'].values == h
            duplications.loc[hap, f'{p}strand'] = dup_strans.loc[hap, f'strand{h}'].values
#
    return duplications

def SeparateDuplicationsByHaplotype(duplications, scaffold_paths, ploidy):
    if ploidy == 1 or len(duplications) == 0:
        duplications.reset_index(drop=True, inplace=True)
    else:
        #for h in range(1,ploidy):
        #        duplications[f'add{h}'] = False # Already create the variables to prevent an error creating multiple columns at once
        for p in ['a','b']:
            # All the duplications of the main. where another haplotype is identical to main we potentially need to clone
            duplications[[f'add{h}' for h in range(1,ploidy)]] = duplications[[f'{p}pid',f'{p}pos']].rename(columns={f'{p}pid':'pid',f'{p}pos':'pos'}).merge(scaffold_paths[['pid','pos']+[f'phase{h}' for h in range(1,ploidy)]], on=['pid','pos'], how='left')[[f'phase{h}' for h in range(1,ploidy)]].values < 0
            # Only clone the main haplotype
            duplications.loc[duplications[f'{p}hap'] != 0, [f'add{h}' for h in range(1,ploidy)]] = False
            # Only create duplicates for a haplotype that is different to main
            existing = (scaffold_paths.groupby(['pid'])[[f'phase{h}' for h in range(1,ploidy)]].max() >= 0).reset_index()
            existing.rename(columns={'pid':f'{p}pid'}, inplace=True)
            for h in range(1,ploidy):
                duplications.loc[duplications[[f'{p}pid']].merge(existing.loc[existing[f'phase{h}'], [f'{p}pid']], on=[f'{p}pid'], how='left', indicator=True)['_merge'].values == "left_only", f'add{h}'] = False
            # Clone duplicates and assign new haplotypes
            duplications['copies'] = 1 + duplications[[f'add{h}' for h in range(1,ploidy)]].sum(axis=1)
            duplications[[f'add{h}' for h in range(1,ploidy)]] = duplications[[f'add{h}' for h in range(1,ploidy)]].cumsum(axis=1)
            for h in range(2,ploidy):
                duplications.loc[duplications[f'add{h}'] == duplications[f'add{h-1}'], f'add{h}'] = -1
            if ploidy > 1:
                duplications.loc[duplications['add1'] == 0, 'add1'] = -1
            duplications.reset_index(drop=True, inplace=True)
            duplications = duplications.loc[np.repeat(duplications.index.values, duplications['copies'].values)].copy()
            duplications['index'] = duplications.index.values
            duplications.reset_index(drop=True, inplace=True)
            duplications['ipos'] = duplications.groupby(['index']).cumcount()
            for h in range(1,ploidy):
                duplications.loc[duplications['ipos'] == duplications[f'add{h}'], f'{p}hap'] = h
        duplications.drop(columns=[f'add{h}' for h in range(1,ploidy)]+['copies','index','ipos'], inplace=True)
#
    return duplications

def RequireContinuousDirectionForDuplications(duplications):
    # Sort them in the direction the positions should change
    duplications.loc[duplications['samedir'] == False, 'bpos'] *= -1
    duplications.sort_values(['apid','ahap','bpid','bhap','apos','bpos'], inplace=True)
    duplications.loc[duplications['samedir'] == False, 'bpos'] *= -1
    duplications['did'] = ((duplications['apid'] != duplications['apid'].shift(1)) | (duplications['ahap'] != duplications['ahap'].shift(1)) | (duplications['bpid'] != duplications['bpid'].shift(1)) | (duplications['bhap'] != duplications['bhap'].shift(1))).cumsum()
    # Find conflicts
    duplications.reset_index(drop=True, inplace=True) # Make sure the index is continuous with no missing numbers
    s = 1
    conflicts = []
    while True:
        duplications['dist'] = duplications['bpos'].shift(-s) - duplications['bpos']
        conflict_found = ((duplications['did'] == duplications['did'].shift(-s)) & ((duplications['apos'] == duplications['apos'].shift(-s)) | (duplications['dist'] == 0) | ((duplications['dist'] < 0) == duplications['samedir'])))
        conflicts.append(pd.DataFrame({'index1':duplications[conflict_found].index.values, 'index2':duplications[conflict_found].index.values + s}))
        s += 1
        if len(conflicts[-1]) == 0:
            break
    duplications.drop(columns=['dist'], inplace=True)
    conflicts = pd.concat(conflicts, ignore_index=True)
    conflicts['did'] = duplications.loc[conflicts['index1'].values, 'did'].values
    conflicts.sort_values(['did','index1','index2'], inplace=True)
    # Assign conflict id to group conflicts that handle the same indexes
    conflicts['cid'] = conflicts['index1'].values
    while True:
        new_cid = pd.concat([conflicts[['index1','cid']].rename(columns={'index1':'index'}), conflicts[['index2','cid']].rename(columns={'index2':'index'})], ignore_index=True).groupby(['index'])['cid'].min().reset_index()
        conflicts['new_cid'] = np.minimum(conflicts[['index1']].rename(columns={'index1':'index'}).merge(new_cid, on=['index'], how='left')['cid'].values, conflicts[['index2']].rename(columns={'index2':'index'}).merge(new_cid, on=['index'], how='left')['cid'].values)
        if np.sum(conflicts['new_cid'] != conflicts['cid']) == 0:
            break
        else:
            conflicts['cid'] = conflicts['new_cid']
    conflicts.drop(columns=['new_cid'], inplace=True)
    # Assign the cid to duplications and automatically keep the ones without, since they are not involved in any conflict
    duplications['cid'] = -1
    cids = pd.concat([conflicts[['index1','cid']].rename(columns={'index1':'index'}), conflicts[['index2','cid']].rename(columns={'index2':'index'})], ignore_index=True).drop_duplicates()
    duplications.loc[cids['index'].values, 'cid'] = cids['cid'].values
    valid_dups = duplications[duplications['cid'] == -1].drop(columns=['cid'])
    # Merge adjacent cids (not separated by a duplication without conflict) to get proper lengths measures later and prevent double counting of space in between
    while True:
        duplications['new_cid'] = np.where(duplications['did'] != duplications['did'].shift(1), -1, duplications['cid'].shift(1, fill_value=-1))
        new_cids = duplications.loc[(duplications['new_cid'] != duplications['cid']) & (duplications['new_cid'] >= 0) & (duplications['cid'] >= 0), ['cid','new_cid']].copy()
        if len(new_cids) == 0:
            break
        else:
            duplications['new_cid'] = duplications[['cid']].merge(new_cids, on=['cid'], how='left')['new_cid'].values
            duplications.loc[np.isnan(duplications['new_cid']) == False, 'cid'] = duplications.loc[np.isnan(duplications['new_cid']) == False, 'new_cid'].astype(int).values
            conflicts['new_cid'] = conflicts[['cid']].merge(new_cids, on=['cid'], how='left')['new_cid'].values
            conflicts.loc[np.isnan(conflicts['new_cid']) == False, 'cid'] = conflicts.loc[np.isnan(conflicts['new_cid']) == False, 'new_cid'].astype(int).values
            conflicts.drop(columns=['new_cid'], inplace=True)
    duplications.drop(columns=['new_cid'], inplace=True)
    # Check for later how much unreducible buffer we have on each side due to an adjacent non-conflicting duplication
    for p in ['a','b']:
        # Flip min/max to get the default (which is no buffer)
        duplications[[f'{p}max',f'{p}min']] = duplications[['cid']].merge(duplications.groupby(['cid'])[f'{p}pos'].agg(['min','max']).reset_index(), on=['cid'], how='left')[['min','max']].values
    duplications['left'] = (duplications['did'] == duplications['did'].shift(1)) & (duplications['cid'] != duplications['cid'].shift(1))
    duplications['amin'] = np.where(duplications['left'], duplications['apos'].shift(1, fill_value=-1), duplications['amin'])
    duplications['bmin'] = np.where(duplications['left'] & duplications['samedir'], duplications['bpos'].shift(1, fill_value=-1), duplications['bmin'])
    duplications['bmax'] = np.where(duplications['left'] & (duplications['samedir'] == False), duplications['bpos'].shift(1, fill_value=-1), duplications['bmax'])
    duplications['right'] =  (duplications['did'] == duplications['did'].shift(-1)) & (duplications['cid'] != duplications['cid'].shift(-1))
    duplications['amax'] = np.where(duplications['right'], duplications['apos'].shift(-1, fill_value=-1), duplications['amax'])
    duplications['bmax'] = np.where(duplications['right'] & duplications['samedir'], duplications['bpos'].shift(-1, fill_value=-1), duplications['bmax'])
    duplications['bmin'] = np.where(duplications['right'] & (duplications['samedir'] == False), duplications['bpos'].shift(-1, fill_value=-1), duplications['bmin'])
    duplications = duplications[duplications['cid'] >= 0].copy()
    duplications[['amin','bmin']] = duplications[['cid']].merge(duplications.groupby(['cid'])[['amin','bmin']].min().reset_index(), on=['cid'], how='left')[['amin','bmin']].values
    duplications[['amax','bmax','left','right']] = duplications[['cid']].merge(duplications.groupby(['cid'])[['amax','bmax','left','right']].max().reset_index(), on=['cid'], how='left')[['amax','bmax','left','right']].values
#
    # Get longest(most matches) valid index combinations for every conflict pool (cid)
    ext = duplications[['cid']].reset_index()
    alternatives = []
    if len(ext):
        cur_alts = ext.rename(columns={'index':'i0'})
        ext = ext.rename(columns={'index':'index1'}).merge(ext.rename(columns={'index':'index2'}), on=['cid'], how='left')
        ext = ext[ext['index1'] < ext['index2']].copy() # Only allow increasing index to avoid duplications, because the order is arbitrary
        ext = ext[ ext[['cid','index1','index2']].merge(conflicts[['cid','index1','index2']], on=['cid','index1','index2'], how='left', indicator=True)['_merge'].values == "left_only" ].copy()
        s=1
        while len(cur_alts):
            new_alts = cur_alts.merge(ext.rename(columns={'index1':f'i{s-1}','index2':f'i{s}'}), on=['cid',f'i{s-1}'], how='inner')
            # Remove conflicts
            for s2 in range(s-1):
                new_alts = new_alts[ new_alts[['cid',f'i{s2}',f'i{s}']].rename(columns={f'i{s2}':'index1',f'i{s}':'index2'}).merge(conflicts[['cid','index1','index2']], on=['cid','index1','index2'], how='left', indicator=True)['_merge'].values == "left_only" ].copy()
            alternatives.append( cur_alts[np.isin(cur_alts['cid'].values, np.unique(new_alts['cid'].values)) == False].copy() )
            alternatives[-1]['len'] = s
            cur_alts = new_alts
            s += 1
        alternatives = pd.concat(alternatives, ignore_index=True)
    if len(alternatives):
        # Get shortest merged path length
        alternatives['mlen'] = np.maximum( np.maximum(0, duplications.loc[alternatives['i0'].values, 'apos'].values - duplications.loc[alternatives['i0'].values, 'amin'].values),
                                           np.maximum(0, np.where(duplications.loc[alternatives['i0'].values, 'samedir'].values, duplications.loc[alternatives['i0'].values, 'bpos'].values - duplications.loc[alternatives['i0'].values, 'bmin'].values, duplications.loc[alternatives['i0'].values, 'bmax'].values - duplications.loc[alternatives['i0'].values, 'bpos'].values) ) )
        alternatives['li'] = alternatives['i0'].values
        for s in range(1,alternatives['len'].max()):
            cur = alternatives['len'] > s
            alternatives.loc[cur, 'mlen'] += np.maximum( np.abs(duplications.loc[alternatives.loc[cur, f'i{s}'].values, 'apos'].values - duplications.loc[alternatives.loc[cur, f'i{s-1}'].values, 'apos'].values),
                                                         np.abs(duplications.loc[alternatives.loc[cur, f'i{s}'].values, 'bpos'].values - duplications.loc[alternatives.loc[cur, f'i{s-1}'].values, 'bpos'].values) )
            alternatives.loc[alternatives['len'] == s+1, 'li'] = alternatives.loc[alternatives['len'] == s+1, f'i{s}'].astype(int).values
        alternatives['mlen'] += np.maximum( np.maximum(0, duplications.loc[alternatives['li'].values, 'amax'].values - duplications.loc[alternatives['li'].values, 'apos'].values),
                                            np.maximum(0, np.where(duplications.loc[alternatives['li'].values, 'samedir'].values, duplications.loc[alternatives['li'].values, 'bmax'].values - duplications.loc[alternatives['li'].values, 'bpos'].values, duplications.loc[alternatives['li'].values, 'bpos'].values - duplications.loc[alternatives['li'].values, 'bmin'].values) ) )
        alternatives = alternatives[alternatives['mlen'].values == alternatives[['cid']].merge(alternatives.groupby(['cid'])['mlen'].min().reset_index(name='minlen'), on=['cid'], how='left')['minlen'].values].drop(columns=['mlen'])
        duplications.drop(columns=['amin','amax','bmin','bmax'], inplace=True)
        # Take the alternatives that cut the least amount from the ends
        alternatives[['amin','bmin']] = duplications.loc[alternatives['i0'].values, ['apos','bpos']].values
        alternatives[['amax','bmax']] = duplications.loc[alternatives['li'].values, ['apos','bpos']].values
        alternatives['samedir'] = duplications.loc[alternatives['i0'].values, 'samedir'].values
        if np.sum(alternatives['samedir'] == False):
            alternatives.loc[alternatives['samedir'] == False, ['bmin','bmax']] = alternatives.loc[alternatives['samedir'] == False, ['bmax','bmin']].values
        alternatives[['left','right']] = duplications.loc[alternatives['i0'].values, ['left','right']].values == False # Here we are interested if we do not have a connected duplication (and not the other way round as before)
        alternatives['cut'] = np.where(alternatives['left'], alternatives['amin'].values - alternatives[['cid']].merge(alternatives.groupby(['cid'])[['amin']].min().reset_index(), on=['cid'], how='left')['amin'].values, 0)
        alternatives['cut'] += np.where(alternatives['right'], alternatives[['cid']].merge(alternatives.groupby(['cid'])[['amax']].max().reset_index(), on=['cid'], how='left')['amax'].values - alternatives['amax'].values, 0)
        if np.sum(alternatives['samedir'] == False):
            alternatives.loc[alternatives['samedir'] == False, ['left','right']] = alternatives.loc[alternatives['samedir'] == False, ['right','left']].values
        alternatives['cut'] += np.where(alternatives['left'], alternatives['bmin'].values - alternatives[['cid']].merge(alternatives.groupby(['cid'])[['bmin']].min().reset_index(), on=['cid'], how='left')['bmin'].values, 0)
        alternatives['cut'] += np.where(alternatives['right'], alternatives[['cid']].merge(alternatives.groupby(['cid'])[['bmax']].max().reset_index(), on=['cid'], how='left')['bmax'].values - alternatives['bmax'].values, 0)
        alternatives.drop(columns=['li','amin','bmin','amax','bmax','left','right'], inplace=True)
        alternatives = alternatives[alternatives['cut'].values == alternatives[['cid']].merge(alternatives.groupby(['cid'])['cut'].min().reset_index(), on=['cid'], how='left')['cut'].values].drop(columns=['cut'])
        duplications.drop(columns=['left','right'], inplace=True)
        # If we have an alternative with the same direction prefer it over alternatives with different directions
        alternatives = alternatives[alternatives['samedir'].values == alternatives[['cid']].merge(alternatives.groupby(['cid'])['samedir'].max().reset_index(name='samedir'), on=['cid'], how='left')['samedir'].values].copy()
        # Take the lowest indexes first for the lower pid and then for the higher pid (such that it is consistent no matter which one is a and b)
        alternatives['blower'] = (duplications.loc[alternatives['i0'].values, 'bpid'].values < duplications.loc[alternatives['i0'].values, 'apid'].values) | ((duplications.loc[alternatives['i0'].values, 'bpid'].values == duplications.loc[alternatives['i0'].values, 'apid'].values) & (duplications.loc[alternatives['i0'].values, 'bhap'].values < duplications.loc[alternatives['i0'].values, 'ahap'].values))
        for s in range(alternatives['len'].max()):
            cur = alternatives['len'] > s
            alternatives.loc[cur, [f'll{s}',f'lh{s}']] = np.where(alternatives.loc[cur, ['blower','blower']].values, duplications.loc[alternatives.loc[cur, f'i{s}'].values, ['bpos','apos']].values, duplications.loc[alternatives.loc[cur, f'i{s}'].values, ['apos','bpos']].values)
        for l in np.unique(alternatives['len']):
            cur = (alternatives['len'] == l) & (alternatives['samedir'] == False)
            for s in range(l//2):
                alternatives.loc[cur & alternatives['blower'], [f'll{s}',f'll{l-1-s}']] = alternatives.loc[cur, [f'll{l-1-s}',f'll{s}']].values
                alternatives.loc[cur & (alternatives['blower'] == False), [f'lh{s}',f'lh{l-1-s}']] = alternatives.loc[cur, [f'lh{l-1-s}',f'lh{s}']].values
        for s in range(alternatives['len'].max()-1,-1,-1): # Start comparing at the back, where we have the highest indexes
            for o in ['l','h']:
                cur = alternatives['len'] > s
                comp = alternatives.loc[cur, ['cid',f'l{o}{s}']].reset_index().merge(alternatives.loc[cur, ['cid',f'l{o}{s}']].groupby(['cid']).min().reset_index().rename(columns={f'l{o}{s}':'min'}), on=['cid'], how='left')
                alternatives.drop(comp.loc[comp[f'l{o}{s}'] > comp['min'], 'index'].values, inplace=True)
        # Take the duplications chosen by the remaining alternatives
        dup_ind = []
        for s in range(alternatives['len'].max()):
            dup_ind.append( alternatives.loc[alternatives['len'] > s, f'i{s}'].astype(int).values )
        dup_ind = np.concatenate(dup_ind)
        duplications = duplications.loc[dup_ind].drop(columns=['cid'])
        duplications = pd.concat([valid_dups, duplications], ignore_index=True)
    else:
        duplications = valid_dups
#
    # We need to sort again, since we destroyed the order by concatenating (but this time we have only one bpos per apos, which makes it easier)
    duplications.sort_values(['did','apos'], inplace=True)
    duplications.reset_index(drop=True, inplace=True)
#
    return duplications

def GetPositionFromPaths(df, scaffold_paths, ploidy, new_col, get_col, pid_col, pos_col, hap_col):
    #for col in [new_col]+[f'{n}{h}' for h in range(1,ploidy) for n in ['phase',get_col]]:
    #    if col not in df.columns:
    #        df[col] = np.nan # Already create the variables to prevent an error creating multiple columns at once
    df[[new_col]+[f'{n}{h}' for h in range(1,ploidy) for n in ['phase',get_col]]] = df[[pid_col,pos_col]].rename(columns={pid_col:'pid',pos_col:'pos'}).merge(scaffold_paths[['pid','pos',f'{get_col}0']+[f'{n}{h}' for h in range(1,ploidy) for n in ['phase',get_col]]], on=['pid','pos'], how='left')[[f'{get_col}0']+[f'{n}{h}' for h in range(1,ploidy) for n in ['phase',get_col]]].values
    for h in range(1,ploidy):
        alt_hap = (df[hap_col] == h) & (df[f'phase{h}'] >= 0)
        df.loc[alt_hap, new_col] = df.loc[alt_hap, f'{get_col}{h}']
        df.drop(columns=[f'phase{h}',f'{get_col}{h}'], inplace=True)
#
    return df

def GetPositionsBeforeDuplication(duplications, scaffold_paths, ploidy, invert):
    for p in ['a','b']:
        # Get previous position and skip duplications
        duplications[f'{p}prev_pos'] = duplications[f'{p}pos']
        update = np.repeat(True, len(duplications))
        while np.sum(update):
            duplications.loc[update, f'{p}prev_pos'] -= (1 if p == 'a' else np.where(duplications.loc[update, 'samedir'], 1, -1)) * (-1 if invert else 1)
            duplications = GetPositionFromPaths(duplications, scaffold_paths, ploidy, f'{p}prev_scaf', 'scaf', f'{p}pid', f'{p}prev_pos', f'{p}hap')
            update = duplications[f'{p}prev_scaf'] < 0
        # Get distance between current and previous position
        duplications.drop(columns=[f'{p}prev_scaf'], inplace=True)
        duplications['dist_pos'] = np.where((True if p == 'a' else duplications['samedir']) != invert, duplications[f'{p}pos'], duplications[f'{p}prev_pos'])
        duplications = GetPositionFromPaths(duplications, scaffold_paths, ploidy, f'{p}dist', 'dist', f'{p}pid', 'dist_pos', f'{p}hap')
        duplications.drop(columns=['dist_pos'], inplace=True)
#
    return duplications

def GetDuplicationDifferences(duplications):
    return duplications.groupby(['group','did','apid','ahap','bpid','bhap'])[['scaf_diff','dist_diff']].sum().reset_index()

def RemoveHaplotype(scaffold_paths, rem_pid, h):
    scaffold_paths.loc[rem_pid, f'phase{h}'] = -scaffold_paths.loc[rem_pid, 'phase0'].values
    scaffold_paths.loc[rem_pid, [f'scaf{h}', f'strand{h}', f'dist{h}']] = [-1,'',0]
#
    return scaffold_paths

def TrimAlternativesConsistentWithMain(scaffold_paths, ploidy):
    for h in range(1, ploidy):
        consistent = (scaffold_paths['scaf0'] == scaffold_paths[f'scaf{h}']) & (scaffold_paths['strand0'] == scaffold_paths[f'strand{h}']) & (scaffold_paths['dist0'] == scaffold_paths[f'dist{h}'])
        if np.sum(consistent):
            scaffold_paths.loc[consistent, [f'scaf{h}', f'dist{h}', f'strand{h}']] = [-1,0,'']
            scaffold_paths.loc[consistent, f'phase{h}'] = -scaffold_paths.loc[consistent, f'phase{h}']
#
    return scaffold_paths

def RemoveMainPath(scaffold_paths, rem_pid_org, ploidy):
    rem_pid = rem_pid_org.copy() # Do not change the input
    for h in range(1, ploidy):
        # Check if we can replace the main with another path
        cur_pid = (scaffold_paths.loc[rem_pid, ['pid',f'phase{h}']].groupby(['pid']).max() >= 0).reset_index().rename(columns={f'phase{h}':'replace'})
        cur_pid = scaffold_paths[['pid']].merge(cur_pid.loc[cur_pid['replace'], ['pid']], on=['pid'], how='left', indicator=True)['_merge'].values == "both"
        if np.sum(cur_pid):
            # For the cases, where we can replace we need to fill all later existing haplotypes (the previous ones are empty) with the main path to not lose the information
            for h1 in range(h+1, ploidy):
                fill = (scaffold_paths.loc[cur_pid, ['pid',f'phase{h1}']].groupby(['pid']).max() >= 0).reset_index().rename(columns={f'phase{h1}':'fill'})
                fill = (scaffold_paths[f'phase{h1}'] < 0) & (scaffold_paths[['pid']].merge(fill.loc[fill['fill'], ['pid']], on=['pid'], how='left', indicator=True)['_merge'].values == "both")
                scaffold_paths.loc[fill, f'phase{h1}'] = -scaffold_paths.loc[fill, f'phase{h1}'].values
                scaffold_paths.loc[fill, [f'scaf{h1}', f'strand{h1}', f'dist{h1}']] = scaffold_paths.loc[fill, ['scaf0', 'strand0', 'dist0']].values
            # Set alternative as main
            scaffold_paths.loc[cur_pid, 'phase0'] = np.abs(scaffold_paths.loc[cur_pid, f'phase{h}'].values)
            scaffold_paths.loc[cur_pid & (scaffold_paths[f'phase{h}'] >= 0), ['scaf0', 'strand0', 'dist0']] = scaffold_paths.loc[cur_pid & (scaffold_paths[f'phase{h}'] >= 0), [f'scaf{h}', f'strand{h}', f'dist{h}']].values
            scaffold_paths = RemoveHaplotype(scaffold_paths, cur_pid, h)
            # Remove already handled cases from rem_pid
            rem_pid = rem_pid & (cur_pid == False)
    scaffold_paths = TrimAlternativesConsistentWithMain(scaffold_paths, ploidy)
    # In the case where only one haplotype exists remove the whole path
    scaffold_paths = scaffold_paths[rem_pid == False].copy()
#
    return scaffold_paths

def AssignLowestHaplotypeToMain(duplications):
    # This becomes necessary when the main paths was removed
    min_hap = duplications[['apid','ahap']].drop_duplicates().groupby(['apid']).min().reset_index()
    min_hap = min_hap[min_hap['ahap'] > 0].copy()
    duplications.loc[duplications[['apid','ahap']].merge(min_hap, on=['apid','ahap'], how='left', indicator=True)['_merge'].values == "both", 'ahap'] = 0
    min_hap.rename(columns={'apid':'bpid','ahap':'bhap'}, inplace=True)
    duplications.loc[duplications[['bpid','bhap']].merge(min_hap, on=['bpid','bhap'], how='left', indicator=True)['_merge'].values == "both", 'bhap'] = 0
#
    return duplications

def GetHaplotypes(haps, scaffold_paths, ploidy):
    # Get full haplotype without deletions
    haps = haps.merge(scaffold_paths, on=['pid'], how='left')
    haps.rename(columns={'phase0':'phase','scaf0':'scaf','strand0':'strand','dist0':'dist'}, inplace=True)
    for h in range(1, ploidy):
        fill = (haps['hap'] == h) & (haps[f'phase{h}'] >= 0)
        haps.loc[fill, 'phase'] = np.abs(haps.loc[fill, f'phase{h}'])
        haps.loc[fill, ['scaf','strand','dist']] = haps.loc[fill, [f'scaf{h}',f'strand{h}',f'dist{h}']].values
        haps.drop(columns=[f'phase{h}',f'scaf{h}',f'strand{h}',f'dist{h}'], inplace=True)
    haps[['phase','scaf','dist']] = haps[['phase','scaf','dist']].astype(int)
    haps = haps[haps['scaf'] >= 0].copy()
#
    return haps

def GetBridgeSupport(bsupp, scaffold_paths, scaf_bridges, ploidy):
    bsupp = GetHaplotypes(bsupp, scaffold_paths, ploidy)
    bsupp.drop(columns=['phase'], inplace=True)
    bsupp.sort_values(['group','pid','hap','pos'], inplace=True)
    bsupp.rename(columns={'scaf':'to','dist':'mean_dist'}, inplace=True)
    bsupp['to_side'] = np.where(bsupp['strand'] == '+', 'l', 'r')
    bsupp['from'] = bsupp['to'].shift(1, fill_value=-1)
    bsupp.loc[(bsupp['pid'] != bsupp['pid'].shift(1)) | (bsupp['hap'] != bsupp['hap'].shift(1)), 'from'] = -1
    bsupp['from_side'] = np.where(bsupp['strand'].shift(1, fill_value='') == '+', 'r', 'l')
    bsupp = bsupp[bsupp['from'] >= 0].merge(scaf_bridges[['from','from_side','to','to_side','mean_dist','bcount']], on=['from','from_side','to','to_side','mean_dist'], how='left')
    # Sort by lowest distinct bridge support and assign a place
    if len(bsupp):
        bsupp.sort_values(['group','pid','hap','bcount'], inplace=True)
        bsupp['pos'] = bsupp.groupby(['group','pid','hap']).cumcount()
        vertical = bsupp[['group','pid','hap']].drop_duplicates()
        for p in range(bsupp['pos'].max()+1):
            vertical[f'bcount{p}'] = vertical[['group','pid','hap']].merge(bsupp.loc[bsupp['pos'] == p, ['group','pid','hap','bcount']], on=['group','pid','hap'], how='left')['bcount'].fillna(-1).astype(int).values
        vertical.sort_values([f'bcount{p}' for p in range(bsupp['pos'].max()+1)], inplace=True)
        bsupp = vertical[['group','pid','hap']].copy()
        bsupp['bplace'] = np.arange(len(bsupp),0,-1)
#
    return bsupp

def RemoveDuplicatedHaplotypesWithLowestSupport(scaffold_paths, duplications, rem_haps, bsupp, ploidy):
    if len(bsupp):
        # Find haplotype with lowest support in each group
        rem_haps = rem_haps.merge(bsupp, on=['group','pid','hap'], how='left')
        rem_haps.sort_values(['group','bplace'], ascending=[True,False], inplace=True)
        rem_haps = rem_haps.groupby(['group']).first().reset_index()
        # Remove those haplotypes
        for h in range(ploidy-1, -1, -1): # We need to go through in the inverse order, because we still need the main for the alternatives
            rem_pid = np.unique(rem_haps.loc[rem_haps['hap'] == h, 'pid'].values)
            duplications = duplications[((np.isin(duplications['apid'], rem_pid) == False) | (duplications['ahap'] != h)) & ((np.isin(duplications['bpid'], rem_pid) == False) | (duplications['bhap'] != h))].copy()
            rem_pid = np.isin(scaffold_paths['pid'], rem_pid)
            if h==0:
                scaffold_paths = RemoveMainPath(scaffold_paths, rem_pid, ploidy)
                duplications = AssignLowestHaplotypeToMain(duplications)
            else:
                scaffold_paths = RemoveHaplotype(scaffold_paths, rem_pid, h)
    differences = GetDuplicationDifferences(duplications)
#
    return scaffold_paths, duplications, differences

def ShiftHaplotypesToLowestPossible(scaffold_paths, ploidy):
    if ploidy > 1:
        existing_haps = []
        for h in range(1, ploidy):
            existing_haps.append(scaffold_paths.loc[scaffold_paths[f'phase{h}'] >= 0, ['pid']].drop_duplicates())
            existing_haps[-1]['hap'] = h
        existing_haps = pd.concat(existing_haps, ignore_index=True).sort_values(['pid','hap'])
        existing_haps['new_hap'] = existing_haps.groupby(['pid']).cumcount()
        existing_haps = existing_haps[existing_haps['new_hap'] != existing_haps['hap']].copy()
        if len(existing_haps):
            for hnew in range(1, ploidy):
                for hold in range(hnew+1, ploidy):
                    shift = np.isin(scaffold_paths['pid'], existing_haps.loc[(existing_haps['new_hap'] == hnew) & (existing_haps['new_hap'] == hold), 'pid'].values)
                    scaffold_paths.loc[shift, [f'phase{hnew}',f'scaf{hnew}', f'strand{hnew}', f'dist{hnew}']] = scaffold_paths.loc[shift, [f'phase{hold}',f'scaf{hold}', f'strand{hold}', f'dist{hold}']].values
                    scaffold_paths = RemoveHaplotype(scaffold_paths, shift, hold)
#
    return scaffold_paths

def CompressPaths(scaffold_paths, ploidy):
    # Remove positions with only deletions
    scaffold_paths = scaffold_paths[(scaffold_paths[[f'scaf{h}' for h in range(ploidy)]] >= 0).any(axis=1)].copy()
    scaffold_paths['pos'] = scaffold_paths.groupby(['pid'], sort=False).cumcount()
    scaffold_paths.reset_index(drop=True, inplace=True)
    # Compress paths where we have alternating deletions
    while True:
        shifts = scaffold_paths.loc[ ((np.where(scaffold_paths[[f'phase{h}' for h in range(ploidy)]].values < 0, scaffold_paths[['scaf0' for h in range(ploidy)]].values, scaffold_paths[[f'scaf{h}' for h in range(ploidy)]].values) < 0) |
                                      (np.where(scaffold_paths[[f'phase{h}' for h in range(ploidy)]].shift(1).values < 0, scaffold_paths[['scaf0' for h in range(ploidy)]].shift(1).values, scaffold_paths[[f'scaf{h}' for h in range(ploidy)]].shift(1).values) < 0)).all(axis=1) &
                                     (scaffold_paths['pos'] > 0), ['pid'] ]
        if len(shifts) == 0:
            break
        else:
            shifts['index'] = shifts.index.values
            shifts = shifts.groupby(['pid'], sort=False).first() # We can only take the first in each path, because otherwise we might block the optimal solution
            shifts['new_index'] = shifts['index'] - np.where(scaffold_paths.loc[shifts['index'].values, 'pos'].values > 1, 2, 1) # Make sure we do not go into the previous path
            while True:
                further = ( ((np.where(scaffold_paths.loc[shifts['index'].values, [f'phase{h}' for h in range(ploidy)]].values < 0, scaffold_paths.loc[shifts['index'].values, ['scaf0' for h in range(ploidy)]].values, scaffold_paths.loc[shifts['index'].values, [f'scaf{h}' for h in range(ploidy)]].values) < 0) |
                            (np.where(scaffold_paths.loc[shifts['new_index'].values, [f'phase{h}' for h in range(ploidy)]].values < 0, scaffold_paths.loc[shifts['new_index'].values, ['scaf0' for h in range(ploidy)]].values, scaffold_paths.loc[shifts['new_index'].values, [f'scaf{h}' for h in range(ploidy)]].values) < 0)).all(axis=1) &
                           (scaffold_paths.loc[shifts['new_index'].values, 'pos'] > 0) )
                if np.sum(further) == 0:
                    break
                else:
                    shifts.loc[further, 'new_index'] -= 1
            shifts['new_index'] += 1
            for h in range(ploidy):
                cur_shifts = (scaffold_paths.loc[shifts['index'].values, f'phase{h}'].values >= 0) & (scaffold_paths.loc[shifts['index'].values, f'scaf{h}'].values >= 0)
                if np.sum(cur_shifts):
                    scaffold_paths.loc[shifts.loc[cur_shifts, 'new_index'].values, f'phase{h}'] = np.abs(scaffold_paths.loc[shifts.loc[cur_shifts, 'index'].values, f'phase{h}'].values)
                    scaffold_paths.loc[shifts.loc[cur_shifts, 'new_index'].values, [f'scaf{h}',f'strand{h}',f'dist{h}']] = scaffold_paths.loc[shifts.loc[cur_shifts, 'index'].values, [f'scaf{h}',f'strand{h}',f'dist{h}']].values
                cur_shifts = (scaffold_paths.loc[shifts['index'].values, f'phase{h}'].values < 0) & (scaffold_paths.loc[shifts['index'].values, 'scaf0'].values >= 0)
                if np.sum(cur_shifts):
                    scaffold_paths.loc[shifts.loc[cur_shifts, 'new_index'].values, f'phase{h}'] = np.abs(scaffold_paths.loc[shifts.loc[cur_shifts, 'index'].values, f'phase{h}'].values)
                    scaffold_paths.loc[shifts.loc[cur_shifts, 'new_index'].values, [f'scaf{h}',f'strand{h}',f'dist{h}']] = scaffold_paths.loc[shifts.loc[cur_shifts, 'index'].values, ['scaf0','strand0','dist0']].values
            scaffold_paths.drop(shifts['index'].values, inplace=True)
            scaffold_paths['pos'] = scaffold_paths.groupby(['pid'], sort=False).cumcount()
            scaffold_paths.reset_index(drop=True, inplace=True)
#
    return scaffold_paths

def MergeHaplotypes(scaffold_paths, scaf_bridges, ploidy, ends_in=[]):
    if len(ends_in):
        ends = ends_in
    else:
        # Find start and end scaffolds for paths
        ends = scaffold_paths.loc[scaffold_paths['pos'] == 0, ['pid','scaf0','strand0']].rename(columns={'scaf0':'sscaf','strand0':'sside'})
        ends = ends.merge(scaffold_paths.loc[scaffold_paths['pid'] != scaffold_paths['pid'].shift(-1), ['pid','scaf0','strand0']].rename(columns={'scaf0':'escaf','strand0':'eside'}), on=['pid'], how='left')
        ends['sside'] = np.where(ends['sside'] == '+','l','r')
        ends['eside'] = np.where(ends['eside'] == '+','r','l')
        # Reverse scaffold ends if start has higher scaffold than end such that paths with identical ends always look the same
        ends['reverse'] = (ends['sscaf'] > ends['escaf']) | ((ends['sscaf'] == ends['escaf']) & (ends['sside'] == 'r') & (ends['eside'] == 'l'))
        ends.loc[ends['reverse'], [f'{p}{n}' for p in ['s','e'] for n in ['scaf','side']]] = ends.loc[ends['reverse'], [f'{p}{n}' for p in ['e','s'] for n in ['scaf','side']]].values
#
        # Find paths that start and end with the same scaffold
        ends.sort_values(['sscaf','sside','escaf','eside','pid'], inplace=True)
        groups = ends.groupby(['sscaf','sside','escaf','eside'], sort=False)['reverse'].agg(['sum','size'])
        ends['group'] = np.repeat(np.arange(len(groups)), groups['size'].values)
        ends['gsize'] = np.repeat(groups['size'].values, groups['size'].values)
        ends['grev'] = np.repeat(groups['sum'].values, groups['size'].values)
        ends = ends[ends['gsize'] > 1].copy()
        # Reverse the minority such that all scaffolds with identical start and end are facing in the same direction
        ends['reverse'] = np.where(ends['grev'] > ends['gsize']/2, ends['reverse'] == False, ends['reverse'])
        scaffold_paths['reverse'] = scaffold_paths[['pid']].merge(ends[['pid','reverse']], on=['pid'], how='left')['reverse'].fillna(False).values.astype(bool)
        scaffold_paths = ReverseScaffolds(scaffold_paths, scaffold_paths['reverse'], ploidy)
        scaffold_paths.drop(columns=['reverse'], inplace=True)
        ends.drop(columns=['gsize','grev'], inplace=True)
#
    # Get duplications within the groups
    duplications = GetDuplications(scaffold_paths, ploidy, ends[['pid','group']])
    # Require same strand (easy here, because we made them the same direction earlier)
    duplications = AddStrandToDuplications(duplications, scaffold_paths, ploidy)
    duplications = duplications[duplications['astrand'] == duplications['bstrand']].drop(columns=['astrand','bstrand'])
    # Require continuous direction of position change
    duplications = SeparateDuplicationsByHaplotype(duplications, scaffold_paths, ploidy)
    duplications['samedir'] = True
    duplications = RequireContinuousDirectionForDuplications(duplications)
    # When assuring continuous positions sometimes the duplication at the start/end gets assign to a place after/before the start/end, which means that those two paths should not be in the same group
    path_len = scaffold_paths.groupby(['pid'])['pos'].last().reset_index(name='max_pos')
    duplications['del'] = False
    for p in ['a','b']:
        duplications.loc[(duplications['did'] != duplications['did'].shift(1)) & (duplications[f'{p}pos'] > 0), 'del'] = True
        duplications['max_pos'] = duplications[[f'{p}pid']].rename(columns={f'{p}pid':'pid'}).merge(path_len, on=['pid'], how='left')['max_pos'].values
        duplications.loc[(duplications['did'] != duplications['did'].shift(-1)) & (duplications[f'{p}pos'] != duplications['max_pos']), 'del'] = True
    duplications['del']  = duplications[['did']].merge(duplications.groupby(['did'])['del'].max().reset_index(), on=['did'], how='left')['del'].values
    duplications = duplications[duplications['del'] == False].drop(columns=['max_pos','del'])
    # Add the duplications between haplotypes of the same path
    if len(duplications):
        new_duplications = [duplications]
        for h in range(1,ploidy):
            # Only take scaffold_paths that are not exactly identical to main
            new_dups = scaffold_paths.groupby(['pid'])[f'phase{h}'].max()
            new_dups = new_dups[new_dups >= 0].reset_index()
            new_dups = scaffold_paths[np.isin(scaffold_paths['pid'], new_dups['pid'])].copy()
            for h1 in range(h):
                # Add one direction (deletions cannot be duplications to be consistent)
                if h1 == 0:
                    add_dups = new_dups.loc[(new_dups['scaf0'] >= 0) & ((new_dups[f'phase{h}'] < 0) | ((new_dups[f'scaf{h}'] == new_dups['scaf0']) & (new_dups[f'strand{h}'] == new_dups['strand0']))), ['pid','pos']].copy()
                else:
                    add_dups = scaffold_paths.groupby(['pid'])[f'phase{h1}'].max()
                    add_dups = add_dups[add_dups >= 0].reset_index()
                    add_dups = new_dups[np.isin(new_dups['pid'], add_dups['pid'])].copy()
                    add_dups = add_dups.loc[((new_dups['scaf0'] >= 0) & (add_dups[f'phase{h}'] < 0) & (add_dups[f'phase{h1}'] < 0)) | ((new_dups[f'scaf{h}'] >= 0) & (add_dups[f'scaf{h}'] == add_dups[f'scaf{h1}']) & (new_dups[f'strand{h}'] == new_dups[f'strand{h1}'])) |
                                            ((add_dups[f'phase{h}'] < 0) & (new_dups['scaf0'] >= 0) & (add_dups[f'scaf{h1}'] == add_dups['scaf0']) & (new_dups[f'strand{h1}'] == new_dups['strand0'])) |
                                            ((add_dups[f'phase{h1}'] < 0) & (new_dups['scaf0'] >= 0) & (add_dups[f'scaf{h}'] == add_dups['scaf0']) & (new_dups[f'strand{h}'] == new_dups['strand0'])), ['pid','pos']].copy()
                add_dups.rename(columns={'pid':'apid','pos':'apos'}, inplace=True)
                add_dups['ahap'] = h1
                add_dups['bpid'] = add_dups['apid'].values
                add_dups['bpos'] = add_dups['apos'].values
                add_dups['bhap'] = h
                add_dups['group'] = add_dups[['apid','ahap']].merge(duplications[['apid','ahap','group']].drop_duplicates(), on=['apid','ahap'], how='left')['group'].values
                add_dups = add_dups[np.isnan(add_dups['group']) == False].copy() # We do not need to add scaffold that do not have duplications with other scaffolds
                add_dups['group'] = add_dups['group'].astype(int)
                add_dups['samedir'] = True
                if len(add_dups):
                    new_duplications.append(add_dups.copy())
                    # Add other direction
                    add_dups[['ahap','bhap']] = add_dups[['bhap','ahap']].values
                    add_dups['group'] = add_dups[['apid','ahap']].merge(duplications[['apid','ahap','group']].drop_duplicates(), on=['apid','ahap'], how='left')['group'].values
                    new_duplications.append(add_dups)
        duplications = pd.concat(new_duplications, ignore_index=True)
        duplications.sort_values(['apid','ahap','bpid','bhap','apos'], inplace=True)
        duplications['did'] = ((duplications['apid'] != duplications['apid'].shift(1)) | (duplications['ahap'] != duplications['ahap'].shift(1)) | (duplications['bpid'] != duplications['bpid'].shift(1)) | (duplications['bhap'] != duplications['bhap'].shift(1))).cumsum()
    # Assign new groups after some groups might have been split when we removed the duplications where the start/end duplication got reassigned to another position (always join the group with the scaffold with most matches as long as all scaffolds match with all other scaffolds)
    if len(duplications):
        duplications['agroup'] = duplications['apid']
        duplications['bgroup'] = duplications['bpid']
        error_shown = False
        while True:
            # Count the matches(scaffold duplications) in each pid pair with duplicate
            matches = duplications.groupby(['agroup','apid','ahap','bgroup','bpid','bhap']).size().reset_index(name='matches')
            # Get size(number of pid pairs) of each group
            for p in ['a','b']:
                matches[f'{p}size'] = matches[[f'{p}group']].merge(matches[[f'{p}group',f'{p}pid',f'{p}hap']].drop_duplicates().groupby([f'{p}group']).size().reset_index(name='size'), on=[f'{p}group'], how='left')['size'].values
            # Get min, median, max number of matches between groups
            matches = matches[matches['agroup'] != matches['bgroup']].copy()
            groups = matches.groupby(['agroup','bgroup','asize','bsize'])['matches'].agg(['size','min','median','max']).reset_index()
            # Delete duplications between groups, where not all pids match all pids of the other group
            delete = groups.loc[groups['size'] != groups['asize']*groups['bsize'], ['agroup','bgroup']].copy()
            if len(delete):
                duplications = duplications[duplications[['agroup','bgroup']].merge(delete, on=['agroup','bgroup'], how='left', indicator=True)['_merge'].values == "left_only"].copy()
            groups = groups[groups['size'] == groups['asize']*groups['bsize']].drop(columns=['size','asize','bsize'])
            if len(groups):
                # Merge the groups with the most min, median, max matches between them
                groups.sort_values(['agroup','min','median','max','bgroup'], ascending=[True,False,False,False,True], inplace=True)
                groups = groups.groupby(['agroup'], sort=False).first().reset_index()
                groups.drop(columns=['min','median','max'], inplace=True)
                groups = groups.merge(groups.rename(columns={'agroup':'bgroup','bgroup':'agroup'}), on=['agroup','bgroup'], how='inner')
                groups = groups[groups['agroup'] < groups['bgroup']].rename(columns={'bgroup':'group','agroup':'new_group'})
                if len(groups) == 0 and error_shown == False:
                    print("Error: Stuck in an endless loop while regrouping in MergeHaplotypes.")
                for p in ['a','b']:
                    duplications['new_group'] = duplications[[f'{p}group']].rename(columns={f'{p}group':'group'}).merge(groups, on=['group'], how='left')['new_group'].values
                    duplications.loc[np.isnan(duplications['new_group']) == False, f'{p}group'] = duplications.loc[np.isnan(duplications['new_group']) == False, 'new_group'].astype(int)
            else:
                break
        duplications['group'] = duplications['agroup']
        duplications.drop(columns=['agroup','bgroup','new_group'], inplace=True)
#
    if len(duplications):
        # Get minimum difference to another haplotype in group
        duplications = GetPositionsBeforeDuplication(duplications, scaffold_paths, ploidy, False)
        duplications['scaf_diff'] = (duplications['aprev_pos'] != duplications['apos'].shift(1)) | (duplications['bprev_pos'] != duplications['bpos'].shift(1))
        duplications['dist_diff'] = (duplications['scaf_diff'] == False) & (duplications['adist'] != duplications['bdist'])
        duplications.loc[duplications['apos'] == 0, 'scaf_diff'] = False
        differences = GetDuplicationDifferences(duplications)
#
        # Remove all except one version of haplotypes with no differences
        if len(differences):
            no_diff = differences[(differences['scaf_diff'] == 0) & (differences['dist_diff'] == 0)].copy()
            if len(no_diff):
                # Remove all except the one with the lowest pid
                for h in range(ploidy-1, -1, -1): # We need to go through in the inverse order, because we still need the main for the alternatives
                    rem_pid = np.unique(no_diff.loc[(no_diff['bpid'] > no_diff['apid']) & (no_diff['bhap'] == h), 'bpid'].values)
                    duplications = duplications[((np.isin(duplications['apid'], rem_pid) == False) | (duplications['ahap'] != h)) & ((np.isin(duplications['bpid'], rem_pid) == False) | (duplications['bhap'] != h))].copy()
                    rem_pid = np.isin(scaffold_paths['pid'], rem_pid)
                    if h==0:
                        scaffold_paths = RemoveMainPath(scaffold_paths, rem_pid, ploidy)
                        duplications = AssignLowestHaplotypeToMain(duplications)
                    else:
                        scaffold_paths = RemoveHaplotype(scaffold_paths, rem_pid, h)
                differences = GetDuplicationDifferences(duplications)
#
    if len(duplications):
        # Get bridge counts for different path/haplotypes to base decisions on it
        bsupp = duplications[['group','apid','ahap']].drop_duplicates()
        bsupp.rename(columns={'apid':'pid','ahap':'hap'}, inplace=True)
        bsupp = GetBridgeSupport(bsupp, scaffold_paths, scaf_bridges, ploidy)
#
        # Remove distance only variants with worst bridge support as long as we are above ploidy haplotypes
        while True:
            if len(duplications):
                groups = duplications.groupby(['group','apid','ahap']).size().groupby(['group']).size().reset_index(name='nhaps')
                rem_groups = groups[groups['nhaps'] > ploidy].copy()
            else:
                rem_groups = []
            if len(rem_groups) == 0:
                break
            else:
                # For each group find the path/haplotype with the lowest bridge support and remove it
                rem_dups = duplications[np.isin(duplications['did'], differences.loc[np.isin(differences['group'], rem_groups['group'].values) & (differences['scaf_diff'] == 0), 'did'].values)].copy()
                if len(rem_dups) == 0:
                    break
                else:
                    rem_haps = rem_dups[['group','apid','ahap']].drop_duplicates()
                    rem_haps.rename(columns={'apid':'pid','ahap':'hap'}, inplace=True)
                    scaffold_paths, duplications, differences = RemoveDuplicatedHaplotypesWithLowestSupport(scaffold_paths, duplications, rem_haps, bsupp, ploidy)
#
        # Remove variants that are identical to other haplotypes except of deletions(missing scaffolds) and distances as long as we are above ploidy haplotypes
        while True:
            if len(duplications):
                groups = duplications.groupby(['group','apid','ahap']).size().groupby(['group']).size().reset_index(name='nhaps')
                rem_groups = groups[groups['nhaps'] > ploidy].copy()
            else:
                rem_groups = []
            if len(rem_groups) == 0:
                break
            else:
                # For each group find the path/haplotype with the lowest bridge support and remove it
                rem_dups = duplications[np.isin(duplications['group'], rem_groups['group'].values)].copy()
                rem_dups['complete'] = (rem_dups['apos'].shift(1) == rem_dups['aprev_pos']) | (rem_dups['aprev_pos'] < 0)
                rem_dups['complete'] = rem_dups[['did']].merge(rem_dups.groupby(['did'])['complete'].min().reset_index(), on=['did'], how='left')['complete'].values
                rem_dups = rem_dups[rem_dups['complete']].copy()
                if len(rem_dups) == 0:
                    break
                else:
                    rem_haps = rem_dups[['group','apid','ahap']].drop_duplicates()
                    rem_haps.rename(columns={'apid':'pid','ahap':'hap'}, inplace=True)
                    scaffold_paths, duplications, differences = RemoveDuplicatedHaplotypesWithLowestSupport(scaffold_paths, duplications, rem_haps, bsupp, ploidy)
#
    if len(duplications) == 0:
        group_info = []
    else:
        # Merge all groups that do not have more than ploidy haplotypes
        groups = duplications.groupby(['group','apid','ahap']).size().groupby(['group']).size().reset_index(name='nhaps')
        groups = groups[groups['nhaps'] <= ploidy].copy()
        duplications = duplications[np.isin(duplications['group'], groups['group'].values)].copy()
        # Define insertion order by amount of bridge support (highest support == lowest new haplotype)
        if len(bsupp):
            bsupp = bsupp.merge(duplications[['group','apid','ahap']].drop_duplicates().rename(columns={'apid':'pid','ahap':'hap'}), on=['group','pid','hap'], how='inner')
            bsupp.sort_values(['group','bplace'], ascending=[True,False], inplace=True)
            for h in range(ploidy):
                groups[[f'pid{h}',f'hap{h}']] = groups[['group']].merge( bsupp.groupby(['group'])[['pid','hap']].last().reset_index(), on=['group'], how='left')[['pid','hap']].fillna(-1).values.astype(int)
                bsupp = bsupp[bsupp['group'].shift(-1) == bsupp['group']].copy()
        # Separate the individual haplotypes and update positions in duplications after removal of deletions
        haps = duplications[['group','apid','ahap']].drop_duplicates()
        haps.rename(columns={'apid':'pid','ahap':'hap'}, inplace=True)
        haps = GetHaplotypes(haps, scaffold_paths, ploidy)
        haps.sort_values(['group','pid','hap','pos'], inplace=True)
        haps['new_pos'] = haps.groupby(['group','pid','hap'], sort=False).cumcount()
        duplications['apos'] = duplications[['apid','apos','ahap']].rename(columns={'apid':'pid','apos':'pos','ahap':'hap'}).merge(haps[['pid','pos','hap','new_pos']], on=['pid','pos','hap'], how='left')['new_pos'].values
        duplications['bpos'] = duplications[['bpid','bpos','bhap']].rename(columns={'bpid':'pid','bpos':'pos','bhap':'hap'}).merge(haps[['pid','pos','hap','new_pos']], on=['pid','pos','hap'], how='left')['new_pos'].values
        duplications.drop(columns=['samedir','aprev_pos','adist','bprev_pos','bdist','scaf_diff','dist_diff'], inplace=True)
        haps['pos'] = haps['new_pos']
        haps.drop(columns={'new_pos'}, inplace=True)
        # Create new merged paths with most supported haplotype (so far only with the positions in haps)
        new_paths = groups[['group','pid0','hap0']].rename(columns={'pid0':'pid','hap0':'hap'})
        new_paths = new_paths.merge(haps[['pid','hap','pos']], on=['pid','hap'], how='left')
        new_paths.drop(columns=['hap'], inplace=True)
        plen = new_paths.groupby(['group'], sort=False).size().values
        new_paths['pid'] = np.repeat( np.arange(len(plen)) + scaffold_paths['pid'].max() + 1, plen )
        new_paths['pos0'] = new_paths['pos']
        for hadd in range(1, ploidy):
            new_paths[f'pos{hadd}'] = -1 # fill everything by default as a deletion and later replace the duplicated scaffolds with the correct position
            for hcomp in range(hadd):
                # Get all duplications relevant for this combination
                dups = groups[['group',f'pid{hcomp}',f'hap{hcomp}',f'pid{hadd}',f'hap{hadd}']].rename(columns={f'pid{hcomp}':'apid',f'hap{hcomp}':'ahap',f'pid{hadd}':'bpid',f'hap{hadd}':'bhap'}).merge( duplications, on=['group','apid','ahap','bpid','bhap'], how='left' )
                dups.drop(columns=['apid','ahap','bpid','bhap','did'], inplace=True)
                # Assing duplications to the place they belong (they might be different for different hcomp, but then we just take the last. For the polyploid case a better algorithm would be helpful, but the most important is that we do not introduce bugs, because a non-optimal alignment for the merging only increases path size, but no errors)
                new_paths[f'pos{hadd}'] = new_paths[['group',f'pos{hcomp}']].rename(columns={f'pos{hcomp}':'apos'}).merge(dups, on=['group','apos'], how='left')['bpos'].fillna(-1).values.astype(int)
            # If the order of the added positions is inverted, see if we can fix it
            while True:
                # Find conflicts
                new_paths['posmax'] =  new_paths.groupby(['group'])[f'pos{hadd}'].cummax()
                new_paths['conflict'] = (new_paths['posmax'] > new_paths[f'pos{hadd}']) & (0 <= new_paths[f'pos{hadd}'])
                if np.sum(new_paths['conflict']) == 0:
                    break
                else:
                    # Check if we can fix the problem
                    conflicts = new_paths.loc[new_paths['conflict'], ['group','pos',f'pos{hadd}']].rename(columns={f'pos{hadd}':'conpos'})
                    conflicts['oindex'] = conflicts.index.values
                    conflicts['cindex'] = conflicts.index.values-1
                    conflicts['nbefore'] = 0
                    conflicts[[f'in{h}' for h in range(hadd)]] = new_paths.loc[new_paths['conflict'], [f'pos{h}' for h in range(hadd)]].values >= 0
                    conflicts['done'] = False
                    conflicts['fixable'] = True
                    while np.sum(conflicts['done'] == False):
                        cons = new_paths.loc[conflicts['cindex'].values].copy()
                        cons['in'] = ((cons[[f'pos{h}' for h in range(hadd)]] >= 0) & (conflicts[[f'in{h}' for h in range(hadd)]].values)).any(axis=1)
                        conflicts.loc[cons['in'].values, 'nbefore'] += 1
                        conflicts.loc[cons['in'].values, [f'in{h}' for h in range(hadd)]] = conflicts.loc[cons['in'].values, [f'in{h}' for h in range(hadd)]] | (cons.loc[cons['in'], [f'pos{h}' for h in range(hadd)]] >= 0)
                        conflicts.loc[cons[f'pos{hadd}'].values >= 0, 'done'] = True
                        conflicts.loc[conflicts['done'] & cons['in'].values, 'fixable'] = False # The ordering in the previous haplotypes prevents a switch
                        conflicts.loc[conflicts['done'] & (cons[f'pos{hadd}'].values <= conflicts['conpos']), 'fixable'] = False # If the previous position is not the reason for the conflict, we need to fix that position first
                        conflicts = conflicts[conflicts['fixable']].copy()
                        conflicts.loc[conflicts['done'] == False, 'cindex'] -= 1
                    conflicts['newpos'] = new_paths.loc[conflicts['cindex'].values, 'pos'].values
                    # Remove conflicts that overlap with a previous conflict
                    conflicts = conflicts[ (conflicts['group'] != conflicts['group'].shift(1)) | (conflicts['newpos'] > conflicts['pos'].shift(1)) ].drop(columns=['conpos','done','fixable'])
                    # Fix the fixable conflicts
                    if len(conflicts) == 0:
                        break
                    else:
                        conflicts.rename(columns={'cindex':'sindex','oindex':'cindex','newpos':'pos_before','pos':'pos_after'}, inplace=True)
                        conflicts['pos_before'] += conflicts['nbefore']
                        conflicts[[f'in{h}' for h in range(hadd)]] = new_paths.loc[new_paths['conflict'], [f'pos{h}' for h in range(hadd)]].values >= 0
                        while len(conflicts):
                            cons = new_paths.loc[conflicts['cindex'].values].copy()
                            cons['in'] = ((cons[[f'pos{h}' for h in range(hadd)]] >= 0) & (conflicts[[f'in{h}' for h in range(hadd)]].values)).any(axis=1)
                            new_paths.loc[conflicts.loc[cons['in'].values, 'cindex'].values, 'pos'] = conflicts.loc[cons['in'].values, 'pos_before'].values
                            conflicts.loc[cons['in'].values, 'pos_before'] -= 1
                            new_paths.loc[conflicts.loc[cons['in'].values == False, 'cindex'].values, 'pos'] = conflicts.loc[cons['in'].values == False, 'pos_after'].values
                            conflicts.loc[cons['in'].values == False, 'pos_before'] -= 1
                            conflicts = conflicts[conflicts['cindex'] > conflicts['sindex']].copy()
                            conflicts['cindex'] -= 1
                        new_paths.sort_values(['group','pos'], inplace=True)
                        new_paths.reset_index(drop=True, inplace=True)
            # Remove the new positions that cannot be fixed
            while True:
                # Find conflicts (this time equal positions are also conflicts, they cannot be fixed by swapping, thus were ignored before)
                new_paths['posmax'] = new_paths.groupby(['group'])[f'pos{hadd}'].cummax().shift(1, fill_value=-1)
                new_paths.loc[new_paths['group'] != new_paths['group'].shift(1), 'posmax'] = -1
                new_paths['conflict'] = (new_paths['posmax'] >= new_paths[f'pos{hadd}']) & (0 <= new_paths[f'pos{hadd}'])
                if np.sum(new_paths['conflict']) == 0:
                    break
                else:
                    # Remove first conflict in each group
                    new_paths.loc[new_paths[['group', 'pos']].merge( new_paths.loc[new_paths['conflict'], ['group','pos']].groupby(['group'], sort=False).first().reset_index(), on=['group', 'pos'], how='left', indicator=True)['_merge'].values == "both", f'pos{hadd}'] = -1
            new_paths.drop(columns=['conflict'], inplace=True)
            # Add missing positions
            new_paths['posmax'] = new_paths.groupby(['group'])[f'pos{hadd}'].cummax().shift(1, fill_value=-1)
            new_paths.loc[new_paths['group'] != new_paths['group'].shift(1), 'posmax'] = -1
            new_paths['index'] = new_paths.index.values
            new_paths['repeat'] = new_paths[f'pos{hadd}'] - new_paths['posmax']
            new_paths.loc[new_paths[f'pos{hadd}'] < 0, 'repeat'] = 1
            new_paths.drop(columns=['posmax'], inplace=True)
            new_paths = new_paths.loc[np.repeat(new_paths.index.values, new_paths['repeat'].values)].copy()
            new_paths['ipos'] = new_paths.groupby(['index'], sort=False).cumcount()+1
            new_paths.loc[new_paths['ipos'] != new_paths['repeat'], [f'pos{h}' for h in range(hadd)]] = -1
            new_paths[f'pos{hadd}'] += new_paths['ipos'] - new_paths['repeat']
            new_paths['pos'] = new_paths.groupby(['group'], sort=False).cumcount()
            new_paths.reset_index(drop=True, inplace=True)
            new_paths.drop(columns=['index','repeat','ipos'], inplace=True)
        # Compress path by filling deletions
        while True:
            shifts = new_paths.loc[ ((new_paths[[f'pos{h}' for h in range(ploidy)]].values < 0) | (new_paths[[f'pos{h}' for h in range(ploidy)]].shift(1).values < 0)).all(axis=1) ].drop(columns=['pid','pos'])
            if len(shifts) == 0:
                break
            else:
                shifts['index'] = shifts.index.values
                shifts = shifts.groupby(['group'], sort=False).first() # We can only take the first in each group, because otherwise we might block the optimal solution
                shifts['new_index'] = shifts['index']-2
                while True:
                    further = ((new_paths.loc[shifts['index'].values, [f'pos{h}' for h in range(ploidy)]].values < 0) | (new_paths.loc[shifts['new_index'].values, [f'pos{h}' for h in range(ploidy)]].values < 0)).all(axis=1)
                    if np.sum(further) == 0: # We do not need to worry to go into another group, because the first position in every group must be a duplication for all included haplotypes, thus blocks continuation
                        break
                    else:
                        shifts.loc[further, 'new_index'] -= 1
                shifts['new_index'] += 1
                for h in range(ploidy):
                    cur_shifts = new_paths.loc[shifts['index'].values, f'pos{h}'].values >= 0
                    new_paths.loc[shifts.loc[cur_shifts, 'new_index'].values, f'pos{h}'] = new_paths.loc[shifts.loc[cur_shifts, 'index'].values, f'pos{h}'].values
                new_paths.drop(shifts['index'].values, inplace=True)
                new_paths['pos'] = new_paths.groupby(['group'], sort=False).cumcount()
                new_paths.reset_index(drop=True, inplace=True)
        # Insert scaffold information into new path
        new_paths[[f'{n}{h}' for h in range(ploidy) for n in ['pid','hap']]] = new_paths[['group']].merge(groups.drop(columns=['nhaps']), on=['group'], how='left')[[f'{n}{h}' for h in range(ploidy) for n in ['pid','hap']]].values
        group_info = new_paths.copy()
        for h in range(ploidy):
            new_paths[[f'phase{h}',f'scaf{h}',f'strand{h}',f'dist{h}']] = new_paths[[f'pid{h}',f'hap{h}',f'pos{h}']].rename(columns={f'{n}{h}':n for n in ['pid','hap','pos']}).merge(haps.drop(columns=['group']), on=['pid','hap','pos'], how='left')[['phase','scaf','strand','dist']].values
            while np.sum(np.isnan(new_paths[f'phase{h}'])):
                new_paths[f'phase{h}'] = np.where(np.isnan(new_paths[f'phase{h}']), new_paths[f'phase{h}'].shift(-1), new_paths[f'phase{h}'])
            new_paths[[f'phase{h}']] = new_paths[[f'phase{h}']].astype(int)
            new_paths[[f'scaf{h}']] = new_paths[[f'scaf{h}']].fillna(-1).astype(int)
            new_paths[[f'strand{h}']] = new_paths[[f'strand{h}']].fillna('')
            new_paths[[f'dist{h}']] = new_paths[[f'dist{h}']].fillna(0).astype(int)
            new_paths.drop(columns=[f'pos{h}',f'pid{h}',f'hap{h}'], inplace=True)
        new_paths.drop(columns=['group'], inplace=True)
        new_paths = TrimAlternativesConsistentWithMain(new_paths, ploidy)
        # Remove the old version of the merged haplotypes from scaffold_path and add the new merged path
        haps = pd.concat([groups[[f'pid{h}',f'hap{h}']].rename(columns={f'pid{h}':'pid',f'hap{h}':'hap'}) for h in range(ploidy)], ignore_index=True)
        for h in range(ploidy-1, -1, -1): # We need to go through in the inverse order, because we still need the main for the alternatives
            rem_pid = np.isin(scaffold_paths['pid'], haps.loc[haps['hap'] == h, 'pid'].values)
            if h==0:
                scaffold_paths = RemoveMainPath(scaffold_paths, rem_pid, ploidy)
            else:
                scaffold_paths = RemoveHaplotype(scaffold_paths, rem_pid, h)
        scaffold_paths = pd.concat([scaffold_paths, new_paths], ignore_index=True)
        # Clean up at the end
        scaffold_paths = ShiftHaplotypesToLowestPossible(scaffold_paths, ploidy) # Do not do this earlier because it invalides haplotypes stored in duplications
        scaffold_paths = CompressPaths(scaffold_paths, ploidy)
#
    if len(ends_in):
        if len(group_info):
            group_info = group_info[['pid']+[f'pid{h}' for h in range(ploidy)]].drop_duplicates()
        return scaffold_paths, group_info
    else:
        return scaffold_paths

def RequireDuplicationAtPathEnd(duplications, scaf_len, mcols):
    ends = duplications.groupby(mcols)[['apos','bpos']].agg(['min','max']).reset_index()
    for p in ['a','b']:
        ends[f'{p}max'] = ends[[f'{p}pid']].droplevel(1,axis=1).rename(columns={f'{p}pid':'pid'}).merge(scaf_len, on=['pid'], how='left')['pos'].values
    ends = ends.loc[((ends['apos','min'] == 0) | (ends['apos','max'] == ends['amax'])) & ((ends['bpos','min'] == 0) | (ends['bpos','max'] == ends['bmax'])), mcols].droplevel(1,axis=1)
    duplications = duplications.merge(ends, on=mcols, how='inner')
#
    return duplications

def GetDuplicatedPathEnds(scaffold_paths, ploidy):
    # Get duplications that contain a path end for both sides
    duplications = GetDuplications(scaffold_paths, ploidy)
    scaf_len = scaffold_paths.groupby(['pid'])['pos'].max().reset_index()
    duplications = RequireDuplicationAtPathEnd(duplications, scaf_len, ['apid','bpid'])
    # Check if they are on the same or opposite strand (alone it is useless, but it sets the requirements for the direction of change for the positions)
    duplications = AddStrandToDuplications(duplications, scaffold_paths, ploidy)
    duplications['samedir'] = duplications['astrand'] == duplications['bstrand']
    duplications.drop(columns=['astrand','bstrand'], inplace=True)
    # Extend the duplications with valid distances from each end
    duplications = SeparateDuplicationsByHaplotype(duplications, scaffold_paths, ploidy)
    ldups = ExtendDuplicationsFromEnd(duplications, scaffold_paths, 'l', scaf_len, ploidy)
    rdups = ExtendDuplicationsFromEnd(duplications, scaffold_paths, 'r', scaf_len, ploidy)
    rdups['did'] += 1 + ldups['did'].max()
    duplications = pd.concat([ldups,rdups], ignore_index=True)
#
    # Check at what end the duplications are
    ends = duplications.groupby(['did','apid','ahap','bpid','bhap'])[['apos','bpos']].agg(['min','max','size']).reset_index()
    ends.columns = [col[0]+col[1] for col in ends.columns]
    ends.rename(columns={'aposmin':'amin','aposmax':'amax','bposmin':'bmin','bposmax':'bmax','bpossize':'matches'}, inplace=True)
    ends.drop(columns=['apossize'], inplace=True)
    for p in ['a','b']:
        ends[f'{p}len'] = ends[[f'{p}pid']].rename(columns={f'{p}pid':'pid'}).merge(scaf_len, on=['pid'], how='left')['pos'].values
        ends[f'{p}left'] = ends[f'{p}min'] == 0
        ends[f'{p}right'] = ends[f'{p}max'] == ends[f'{p}len']
    # Filter the duplications that are either at no end or at both ends (both ends means one of the paths is fully covered by the other, so no extension of that one is possible. We keep the shorter one as a separate scaffold, because it might belong to an alternative haplotype)
    ends = ends[(ends['aleft'] != ends['aright']) & (ends['bleft'] != ends['bright'])].copy()
    ends['aside'] = np.where(ends['aleft'], 'l', 'r')
    ends['bside'] = np.where(ends['bleft'], 'l', 'r')
    ends.drop(columns=['aleft','aright','bleft','bright'], inplace=True)
    duplications = duplications.merge(ends[['did']], on=['did'], how='inner')
    ends['samedir'] = duplications.groupby(['did'])['samedir'].first().values
#
    return ends

def SelectingHaplotypeAndApplyingSideToPath(cpath, ploidy, p):
    for h in range(1,ploidy):
        cur = (cpath[f'{p}hap'] == h) & (cpath[f'phase{h}'] > 0)
        cpath.loc[cur, ['scaf0','strand0','dist0']] = cpath.loc[cur, [f'scaf{h}',f'strand{h}',f'dist{h}']].values
    cpath.drop(columns=[f'{n}{h}' for h in range(1,ploidy) for n in ['phase','scaf','strand','dist']], inplace=True)
    cpath = cpath[cpath['scaf0'] >= 0].copy()
    cpath['pos'] = cpath.groupby(['pid'], sort=False).cumcount()
    cpath = ReverseScaffolds(cpath, cpath[f'{p}side'] == 'r', 1)
    cpath.rename(columns={'scaf0':'scaf','strand0':'strand','dist0':'dist'}, inplace=True)
    cpath.drop(columns=['phase0'], inplace=True)
#
    return cpath

def GetPathAFromEnds(ends, scaffold_paths, ploidy):
    patha = ends[['apid','ahap','aside']].drop_duplicates()
    patha.sort_values(['apid','ahap','aside'])
    patha['pid'] = np.arange(len(patha))
    ends['opid'] = ends[['apid','ahap','aside']].merge(patha, on=['apid','ahap','aside'], how='left')['pid'].values
    patha = patha.merge(scaffold_paths.rename(columns={'pid':'apid'}), on=['apid'], how='left')
    patha = SelectingHaplotypeAndApplyingSideToPath(patha, ploidy, 'a')
    patha.drop(columns=['apid','ahap','aside'], inplace=True)
#
    return ends, patha

def GetPathBFromEnds(ends, scaffold_paths, ploidy):
    pathb = ends[['bpid','bhap','bside','bmin','bmax']].drop_duplicates()
    pathb['pid'] = np.arange(len(pathb)) # We need a pid that separates the different bhap and bside from the same bpid
    ends['epid'] = ends[['bpid','bhap','bside','bmin','bmax']].merge(pathb, on=['bpid','bhap','bside','bmin','bmax'], how='left')['pid'].values
    pathb = pathb.merge(scaffold_paths.rename(columns={'pid':'bpid'}), on=['bpid'], how='left').drop(columns=['bpid'])
    pathb['keep'] = (pathb['pos'] < pathb['bmin']) | (pathb['pos'] > pathb['bmax'])
    pathb.drop(columns=['bmin','bmax'], inplace=True)
    pathb = SelectingHaplotypeAndApplyingSideToPath(pathb, ploidy, 'b')
    pathb = pathb[pathb['keep']].drop(columns=['bhap','bside','keep']) # Only remove the overlapping scaffolds here, because we are interested in the distance at position 0
    pathb['pos'] = pathb.groupby(['pid'], sort=False).cumcount()
#
    return ends, pathb

def FilterInvalidConnections(ends, scaffold_paths, scaffold_graph, graph_ext, ploidy):
    if len(ends):
        # Find origins with full length matches. If we do not find one at the end go back along patha until we find a full length match.
        ends, patha = GetPathAFromEnds(ends, scaffold_paths, ploidy)
        patha['strand'] = np.where(patha['strand'] == '+', '-', '+') # origin has the opposite direction of patha, so fix strand in patha here and during the comparison take shifted distances from origin to match them (the order/positions of scaffolds are identical though)
        patha_len = patha.groupby(['pid'])['pos'].max().reset_index(name='len')
        patha_len['len'] += 1
        missing_pids = np.unique(patha['pid'].values)
        cut = 0
        org_storage = []
        while len(missing_pids):
            cur_org = graph_ext['org'][['olength','scaf0','strand0']].reset_index().rename(columns={'index':'oindex','olength':'length','scaf0':'scaf','strand0':'strand'}).merge(patha.loc[(patha['pos'] == cut) & np.isin(patha['pid'], missing_pids), ['pid','scaf','strand']], on=['scaf','strand'], how='inner').drop(columns=['scaf','strand'])
            cur_org['length'] = np.minimum(cur_org['length'], cur_org[['pid']].merge(patha_len, on=['pid'], how='left')['len'].values-cut)
            cur_org['matches'] = 1
            for s in range(1, cur_org['length'].max()):
                comp = (cur_org['matches'] == s) & (cur_org['length'] > s)
                cur_org.loc[comp,'matches'] += (graph_ext['org'].loc[cur_org.loc[comp, 'oindex'].values, [f'oscaf{s}',f'ostrand{s}',f'odist{s-1}']].values == cur_org.loc[comp, ['pid']].merge(patha[patha['pos'] == s+cut], on=['pid'], how='left')[['scaf','strand','dist']].values).all(axis=1).astype(int)
            cur_org = cur_org[cur_org['matches'].values == cur_org['length'].values].drop(columns=['length','matches'])
            missing_pids = np.setdiff1d(missing_pids, np.unique(cur_org['pid'].values))
            cur_org['cut'] = cut
            cut += 1
            org_storage.append(cur_org)
        org_storage = pd.concat(org_storage, ignore_index=True)
        # Propagate the origin forward to the end again
        pairs = graph_ext['pairs'].copy()
        pairs[['nscaf','nstrand','ndist']] = graph_ext['ext'].loc[pairs['eindex'].values, ['scaf1','strand1','dist1']].values
        has_valid_ext = pairs.drop(columns=['eindex']).drop_duplicates()
        patha['dist'] = patha['dist'].shift(-1, fill_value=0) # Shift the distances additionally to the previous strand flip to get the reverse direction
        cur_org = []
        for cut in range(org_storage['cut'].max(), 0, -1):
            # Start from the origins with the longest cut and add every round the ones that need to be additionally included
            if len(cur_org) == 0:
                cur_org = org_storage[org_storage['cut'] == cut].drop(columns=['cut'])
            else:
                cur_org = pd.concat([cur_org, org_storage[org_storage['cut'] == cut].drop(columns=['cut'])], ignore_index=True)
            # Only keep cur_org that have an extension that match at least one position
            cur_org[['nscaf','nstrand','ndist']] = cur_org[['pid']].merge(patha[patha['pos'] == cut-1], on=['pid'], how='left')[['scaf','strand','dist']].values
            cur_org = cur_org.merge(has_valid_ext, on=['oindex','nscaf','nstrand','ndist'], how='inner')
            # Prepare next round by stepping one scaffold forward to also cover branches that do not reach up to the end
            cur_org = cur_org.rename(columns={'oindex':'oindex1'}).merge(graph_ext['ocont'], on=['oindex1','nscaf','nstrand','ndist'], how='inner').drop(columns=['oindex1','nscaf','nstrand','ndist']).rename(columns={'oindex2':'oindex'})
            cur_org.drop_duplicates(inplace=True)
        if len(cur_org):
            cur_org = pd.concat([org_storage[org_storage['cut'] == 0].drop(columns=['cut']), cur_org], ignore_index=True)
        else:
            cur_org = org_storage.drop(columns=['cut'])
        patha['strand'] = np.where(patha['strand'] == '+', '-', '+') # Revert the strand flip for the previous comparison
        patha['dist'] = patha['dist'].shift(1, fill_value=0) # And also shift the distances back to the correct position
        # Get the path to which the extensions of the origins should match
        ends, pathb = GetPathBFromEnds(ends, scaffold_paths, ploidy)
        cur_org = cur_org.rename(columns={'pid':'opid'}).merge(ends[['opid','epid']].reset_index().rename(columns={'index':'endindex'}), on=['opid'], how='left')
        cur_org.drop(columns=['opid'], inplace=True)
        cur_org.rename(columns={'epid':'pid'}, inplace=True)
        # Get extensions that pair with the valid origins
        cur_org[['nscaf','nstrand','ndist']] = cur_org[['pid']].merge(pathb[pathb['pos'] == 0], on=['pid'], how='left')[['scaf','strand','dist']].values
        cur_ext = cur_org.merge(pairs, on=['oindex','nscaf','nstrand','ndist'], how='inner').drop(columns=['nscaf','nstrand','ndist']).drop_duplicates()
        valid_ends = []
        while len(cur_ext):
            # Check how long the extensions match pathb
            cur_ext['length'] = np.minimum(graph_ext['ext'].loc[cur_ext['eindex'].values, 'length'].values-1, cur_ext[['pid']].merge(pathb.groupby(['pid'])['pos'].max().reset_index(), on=['pid'], how='left')['pos'].values+1) # The +1 is because it is the max pos not one after it and the -1 for the extensions is because pathb has scaf0 from extension removed, so that the positions are shifted by one
            cur_ext['matches'] = 1
            for s in range(1, cur_ext['length'].max()):
                comp = (cur_ext['matches'] == s) & (cur_ext['length'] > s)
                cur_ext.loc[comp,'matches'] += (graph_ext['ext'].loc[cur_ext.loc[comp, 'eindex'].values, [f'scaf{s+1}',f'strand{s+1}',f'dist{s+1}']].values == cur_ext.loc[comp, ['pid']].merge(pathb[pathb['pos'] == s], on=['pid'], how='left')[['scaf','strand','dist']].values).all(axis=1).astype(int)
            # Handle valid ends, where we have a full length match between extension and pathb
            valid_ids = np.unique(cur_ext.loc[cur_ext['length'] == cur_ext['matches'], 'endindex'].values)
            valid_ends.append( ends.loc[valid_ids, ['apid','ahap','aside','bpid','bhap','bside']].copy() )
            cur_ext = cur_ext[np.isin(cur_ext['endindex'], valid_ids) == False].copy()
            cur_org = cur_org.merge(cur_ext[['oindex','endindex']].drop_duplicates(), on=['oindex','endindex'], how='inner') # Taking only the endindex still in cur_ext also filters the cur_org that do not have any matching extension with the allowed paths in end
            # Prepare next round by stepping one scaffold forward to also cover branches that do not reach up to the end
            pathb = pathb[pathb['pos'] > 0].copy()
            pathb['pos'] -= 1
            cur_org = cur_org.rename(columns={'oindex':'oindex1'}).merge(graph_ext['ocont'], on=['oindex1','nscaf','nstrand','ndist'], how='inner').drop(columns=['oindex1']).rename(columns={'oindex2':'oindex'})
            cur_org[['nscaf','nstrand','ndist']] = cur_org[['pid']].merge(pathb[pathb['pos'] == 0], on=['pid'], how='left')[['scaf','strand','dist']].values
            cur_org.drop_duplicates(inplace=True)
            cur_ext = cur_org.merge(pairs, on=['oindex','nscaf','nstrand','ndist'], how='inner').drop(columns=['nscaf','nstrand','ndist']).drop_duplicates()
        # Overwrite ends with valid_end
        if len(valid_ends):
            valid_ends = pd.concat(valid_ends, ignore_index=True)
            valid_ends.sort_values(['apid','bpid','ahap','bhap','aside','bside'], inplace=True)
            valid_ends = valid_ends.merge(valid_ends.rename(columns={'apid':'bpid','ahap':'bhap','aside':'bside','bpid':'apid','bhap':'ahap','bside':'aside'}), on=['apid','ahap','aside','bpid','bhap','bside'], how='outer', indicator=True)
            valid_ends.rename(columns={'_merge':'valid_path'}, inplace=True)
            valid_ends['valid_path'] = np.where(valid_ends['valid_path'] == "both", 'ab', np.where(valid_ends['valid_path'] == "left_only", 'a', 'b'))
            ends = ends.merge(valid_ends, on=[col for col in valid_ends.columns if col != 'valid_path'], how='inner').drop(columns=['opid','epid'])
        else:
            ends = []
#
    return ends

def GetDeduplicatedHaplotypesAtPathEnds(path_sides, scaffold_paths, scaffold_graph, ploidy):
    # Reverse the scaffolds, where we are interested on the right side, so that the position 0 is always the first on the interesting side
    cur_paths = scaffold_paths.merge(path_sides, on=['pid'], how='inner')
    rcur = cur_paths[cur_paths['side'] == 'r'].copy()
    rcur['reverse'] = True
    cur_paths = [ cur_paths[cur_paths['side'] == 'l'] ]
    for m in np.unique(rcur['matches']):
        cur_paths.append( ReverseScaffolds(rcur[rcur['matches'] == m].copy(), rcur.loc[rcur['matches'] == m, 'reverse'].values, ploidy).drop(columns=['reverse']) )
    cur_paths = pd.concat(cur_paths, ignore_index=True)
    cur_paths.sort_values(['pid','side','matches','pos'], inplace=True)
    # Get the deduplicated haplotypes
    haps = []
    dedup_haps = [ path_sides ] # The main paths will never be removed through deduplication
    dedup_haps[-1]['hap'] = 0
    for h in range(ploidy):
        # Get all haplotypes that differ from the main (and the main) without any deletions
        cur = cur_paths[np.isin(cur_paths['pid'], np.unique(cur_paths.loc[cur_paths[f'phase{h}'] > 0, 'pid'].values))].copy()
        if np.sum(cur[f'phase{h}'] < 0):
            cur.loc[cur[f'phase{h}'] < 0, [f'scaf{h}',f'strand{h}',f'dist{h}']] = cur.loc[cur[f'phase{h}'] < 0, ['scaf0','strand0','dist0']].values
        cur = cur.loc[cur[f'scaf{h}'] >= 0, ['pid','side','matches','pos',f'scaf{h}',f'strand{h}',f'dist{h}']].rename(columns={f'scaf{h}':'scaf',f'strand{h}':'strand',f'dist{h}':'dist'})
        cur['pos'] = cur.groupby(['pid','side','matches'], sort=False).cumcount()
        cur['hap'] = h
        # Compare to with all previous haplotypes if it differs
        if h == 0:
            haps.append(cur) # The main does not have any to compare
        else:
            cur_dedups = cur[['pid','side','matches','hap']].drop_duplicates()
            for hap in haps:
                # Get the first position from where it differs to the currently compared haplotype
                cur[['cscaf','cstrand','cdist']] = cur[['pid','side','matches','pos']].merge(hap, on=['pid','side','matches','pos'], how='left')[['scaf','strand','dist']].values
                cur['diff'] = np.isnan(cur['cscaf']) | (cur[['cscaf','cstrand','cdist']].values != cur[['scaf','strand','dist']].values).any(axis=1)
                cur['diff'] = cur.groupby(['pid','side','matches'], sort=False)['diff'].cummax()
                # Also get it the other way round
                chap = hap.merge(cur[['pid']].drop_duplicates(), on=['pid'], how='inner')
                chap[['cscaf','cstrand','cdist']] = chap[['pid','side','matches','pos']].merge(cur, on=['pid','side','matches','pos'], how='left')[['scaf','strand','dist']].values
                chap['diff'] = np.isnan(chap['cscaf']) | (chap[['cscaf','cstrand','cdist']].values != chap[['scaf','strand','dist']].values).any(axis=1)
                chap['diff'] = chap.groupby(['pid','side','matches'], sort=False)['diff'].cummax()
                # Do the next tests on both haplotypes combined and if one of them passes all the two haplotypes are no duplicates
                chap = pd.concat([chap,cur], ignore_index=True)
                # Keep paths up to first difference
                chap['diff'] = chap['diff'] & (chap['diff'].shift(1) | (chap[['pid','side','matches','hap']] != chap[['pid','side','matches','hap']].shift(1)).any(axis=1))
                chap = chap[chap['diff'] == False].drop(columns=['diff'])
                # Make it a vertical paths starting from the highest position
                vpaths = chap.groupby(['pid','side','matches','hap'])['pos'].max().reset_index()
                vpaths['length'] = 0
                s = 0
                while vpaths['length'].max() == s:
                    vpaths[[f'scaf{s}',f'strand{s}',f'dist{s+1}']] = vpaths[['pid','side','matches','hap','pos']].merge(chap, on=['pid','side','matches','hap','pos'], how='left')[['scaf','strand','dist']].values
                    if np.sum(np.isnan(vpaths[f'scaf{s}']) == False):
                        vpaths.loc[np.isnan(vpaths[f'scaf{s}']) == False, 'length'] += 1
                        vpaths.loc[np.isnan(vpaths[f'scaf{s}']) == False, f'strand{s}'] = np.where(vpaths.loc[np.isnan(vpaths[f'scaf{s}']) == False, f'strand{s}'] == '+', '-', '+') # We go in reverse order so we have to flip the strand
                    vpaths['pos'] -= 1
                    s += 1
                # Check scaffold_graph to see if the haplotype extends from its first unique position over the end
                vpaths = vpaths[vpaths['length'] < scaffold_graph['length'].max()].drop(columns=['pos']) # If a path is as long or longer than any in scaffold_graph, scaffold_graph cannot extend on this path
                vpaths.rename(columns={'scaf0':'from','strand0':'from_side'}, inplace=True)
                vpaths['from_side'] = np.where(vpaths['from_side'] == '+', 'r', 'l')
                vpaths['slen'] = -1
                for l in np.unique(vpaths['length']):
                    mcols = ['from','from_side']+[f'{n}{s}' for s in range(1,l) for n in ['scaf','strand','dist']]
                    vpaths.loc[vpaths['length'] == l, 'slen'] = vpaths.loc[vpaths['length'] == l, mcols].merge(scaffold_graph.groupby(mcols)['length'].max().reset_index(), on=mcols, how='left')['length'].values
                cur_dedups = cur_dedups.merge(vpaths.loc[vpaths['length'] < np.maximum(vpaths['slen'],vpaths['matches']+1), ['pid','side','matches']].drop_duplicates(), on=['pid','side','matches'], how='inner')
            # Store the haplotypes that are different to all other haplotypes
            dedup_haps.append(cur_dedups)
            # Store the haplotype to compare it to the next haplotypes in the loop
            haps.append(cur.drop(columns=['cscaf','cstrand','cdist','diff']))
    dedup_haps = pd.concat(dedup_haps, ignore_index=True)
#
    return dedup_haps

def RemoveHaplotypes(scaffold_paths, delete_haps, ploidy):
    for h in range(ploidy-1, -1, -1): # We need to go through in the inverse order, because we still need the main for the alternatives
        remove = np.isin(scaffold_paths['pid'], delete_haps.loc[delete_haps['hap'] == h, 'pid'].values)
        if np.sum(remove):
            if h==0:
                scaffold_paths = RemoveMainPath(scaffold_paths, remove, ploidy)
            else:
                scaffold_paths = RemoveHaplotype(scaffold_paths, remove, h)
#
    return scaffold_paths

def GetNumberOfHaplotypes(scaffold_paths, ploidy):
    return (((scaffold_paths.groupby(['pid'])[[f'phase{h}' for h in range(ploidy)]].max() >= 0)*[h for h in range(ploidy)]).max(axis=1)+1).reset_index(name='nhaps')

def SetDistanceAtFirstPositionToZero(scaffold_paths, ploidy):
    scaffold_paths.loc[scaffold_paths['pos'] == 0, [f'dist{h}' for h in range(ploidy)]] = 0
    # Handle first positions that are not at zero due to deletions in that haplotype
    del_start = []
    for h in range(ploidy):
        del_start.append(scaffold_paths.loc[(scaffold_paths['pos'] == 0) & ( ((scaffold_paths[f'scaf{h}'] < 0) & (scaffold_paths[f'phase{h}'] > 0)) |
                                                                             ((scaffold_paths['scaf0'] < 0) & (scaffold_paths[f'phase{h}'] < 0)) ), ['pid','pos']].copy())
        del_start[-1]['hap'] = h
    del_start = pd.concat(del_start, ignore_index=True)
    del_start['pos'] = 1
    while len(del_start):
        new_start = []
        for h in range(ploidy):
            scaffold_paths['hap'] = scaffold_paths[['pid','pos']].merge(del_start[del_start['hap'] == h], on=['pid','pos'], how='left')['hap'].values
            if h == 0:
                # When we set the main paths to zero, first store it in all the ones that are equal to the main paths
                for h2 in range(1,ploidy):
                    cur = (np.isnan(scaffold_paths['hap']) == False) & (scaffold_paths[f'phase{h2}'] < 0)
                    if np.sum(cur):
                        scaffold_paths.loc[cur, [f'scaf{h2}',f'strand{h2}',f'dist{h2}']] = scaffold_paths.loc[cur, ['scaf0','strand0','dist0']].values
                        scaffold_paths.loc[cur, f'phase{h2}'] = -scaffold_paths.loc[cur, f'phase{h2}']
            scaffold_paths.loc[np.isnan(scaffold_paths['hap']) == False, f'dist{h}'] = 0
            new_start.append( scaffold_paths.loc[(np.isnan(scaffold_paths['hap']) == False) & ( ((scaffold_paths[f'scaf{h}'] < 0) & (scaffold_paths[f'phase{h}'] > 0)) |
                                                                                                ((scaffold_paths['scaf0'] < 0) & (scaffold_paths[f'phase{h}'] < 0)) ), ['pid','pos','hap']].copy() )
        scaffold_paths.drop(columns=['hap'], inplace=True)
        del_start = pd.concat(new_start, ignore_index=True)
        del_start['pos'] += 1
    # Clean up    
    scaffold_paths = TrimAlternativesConsistentWithMain(scaffold_paths, ploidy)
#
    return scaffold_paths

def TurnHorizontalHaplotypeIntoVertical(haps):
    vhaps = haps[['pid']].drop_duplicates()
    vhaps['length'] = 0
    if len(haps):
        for p in range(haps['pos'].max()+1):
            vhaps[[f'scaf{p}',f'strand{p}',f'dist{p}']] = vhaps[['pid']].merge(haps[haps['pos'] == p], on=['pid'], how='left')[['scaf','strand','dist']].values
            vhaps.loc[np.isnan(vhaps[f'scaf{p}']) == False, 'length'] += 1
        vhaps.drop(columns=['dist0'], inplace=True)
#
    return vhaps

def FindInvalidOverlaps(ends, scaffold_paths, ploidy):
    # Check first position, where distance does not matter
    for p in ['a','b']:
        cur = (ends[f'{p}side'] == 'r') == (p == 'a')
        ends[f'{p}pos'] = np.where(cur, ends[f'{p}min'], ends[f'{p}max'])
        ends[f'{p}dir'] = np.where(cur, 1, -1)
        ends = GetPositionFromPaths(ends, scaffold_paths, ploidy, f'{p}scaf', 'scaf', f'{p}pid', f'{p}pos', f'{p}hap')
        ends = GetPositionFromPaths(ends, scaffold_paths, ploidy, f'{p}strand', 'strand', f'{p}pid', f'{p}pos', f'{p}hap')
    ends['valid_overlap'] = (ends['ascaf'] == ends['bscaf']) & ((ends['astrand'] == ends['bstrand']) == ends['samedir'])
    # Check following positions
    ends['unfinished'] = True
    switch_cols = {'apid':'bpid','ahap':'bhap','bpid':'apid','bhap':'ahap'}
    while np.sum(ends['unfinished']):
        for p in ['a','b']:
            ends.rename(columns=switch_cols, inplace=True)
            ends['mpos'] = ends[f'{p}pos']
            ends['dir'] = ends[f'{p}dir']
            ends = GetFullNextPositionInPathB(ends, scaffold_paths, ploidy)
            ends[f'{p}pos'] = ends['mpos']
            ends[f'{p}scaf'] = ends['next_scaf']
            ends[f'{p}strand'] = ends['next_strand']
            ends[f'{p}dist'] = ends['next_dist']
            ends[f'{p}unfinished'] = (ends[f'{p}min'] <= ends[f'{p}pos']) & (ends[f'{p}pos'] <= ends[f'{p}max'])
        ends['unfinished'] = ends['aunfinished'] | ends['bunfinished']
        ends.loc[ends['aunfinished'] != ends['bunfinished'], 'valid_overlap'] = False
        ends.loc[ends['unfinished'], 'valid_overlap'] = ends.loc[ends['unfinished'], 'valid_overlap'] & (ends.loc[ends['unfinished'], ['ascaf','astrand','adist']].values == ends.loc[ends['unfinished'], ['bscaf','bstrand','bdist']].values).all(axis=1)
    ends.drop(columns=['apos','adir','ascaf','astrand','aunfinished','bpos','bdir','bscaf','bstrand','bunfinished','unfinished','mpos','dir','opos','next_scaf','next_strand','dist_pos','next_dist','adist','bdist'], inplace=True)
    # If the haplotype is identical to a lower haplotype at the overlap, ignore this test (to be able to combine non-haploid overlaps)
    pids = pd.concat([ends.loc[ends['valid_overlap'] == False, ['apid','amin','amax']].rename(columns={'apid':'pid','amin':'min','amax':'max'}), ends.loc[ends['valid_overlap'] == False, ['bpid','bmin','bmax']].rename(columns={'bpid':'pid','bmin':'min','bmax':'max'})], ignore_index=True).drop_duplicates()
    pids = scaffold_paths.merge(pids, on=['pid'], how='inner')
    pids = pids[(pids['min'] <= pids['pos']) & (pids['pos'] <= pids['max'])].drop(columns=['min','max'])
    pids['pos'] = pids.groupby(['pid']).cumcount()
    pids = SetDistanceAtFirstPositionToZero(pids, ploidy)
    haps = []
    dedup_haps = []
    for h in range(1,ploidy):
        cur = pids.groupby(['pid'])[f'phase{h}'].max()
        dedup_haps.append(cur[cur < 0].reset_index()[['pid']])
        dedup_haps[-1]['hap'] = h
        cur = pids[np.isin(pids['pid'], cur[cur > 0].reset_index()['pid'].values)].copy()
        cur.loc[cur[f'phase{h}'] < 0, [f'scaf{h}',f'strand{h}',f'dist{h}']] = cur.loc[cur[f'phase{h}'] < 0, ['scaf0','strand0','dist0']].values
        cur = cur.loc[cur[f'scaf{h}'] >= 0, ['pid','pos',f'scaf{h}',f'strand{h}',f'dist{h}']].rename(columns={f'scaf{h}':'scaf',f'strand{h}':'strand',f'dist{h}':'dist'})
        cur['pos'] = cur.groupby(['pid']).cumcount()
        new_haps = TurnHorizontalHaplotypeIntoVertical(cur)
        if len(haps):
            cols = list(np.intersect1d(new_haps.columns, haps.columns))
            new_haps['dups'] = new_haps[cols].merge(haps[cols], on=cols, how='left', indicator=True)['_merge'].values == "both"
            dedup_haps.append(new_haps.loc[new_haps['dups'], ['pid']].copy())
            dedup_haps[-1]['hap'] = h
            new_haps = new_haps[new_haps['dups'] == False].drop(columns=['dups'])
            haps = pd.concat([haps, new_haps], ignore_index=True)
        else:
            haps = new_haps
    dedup_haps = pd.concat(dedup_haps, ignore_index=True)
    ends['dup_hap'] = ''
    for p in ['a','b']:
        ends.loc[ends[[f'{p}pid',f'{p}hap']].rename(columns={f'{p}pid':'pid',f'{p}hap':'hap'}).merge(dedup_haps, on=['pid','hap'], how='left', indicator=True)['_merge'].values == "both", 'dup_hap'] += p
#
    return ends

def SetConnectablePathsInMetaScaffold(meta_scaffolds, ends, connectable):
    meta_scaffolds['connectable'] = meta_scaffolds['connectable'] | (meta_scaffolds[['new_pid','bpid']].rename(columns={'new_pid':'apid'}).merge(connectable[['apid','bpid']], on=['apid','bpid'], how='left', indicator=True)['_merge'].values == "both")
    ends = ends[ends[['apid','bpid']].merge(connectable[['apid','bpid']], on=['apid','bpid'], how='left', indicator=True)['_merge'].values == "left_only"].copy()
#
    return meta_scaffolds, ends

def FillHaplotypes(scaffold_paths, fill, ploidy):
    for h in range(1,ploidy):
        need_fill = fill & (scaffold_paths[f'phase{h}'] < 0)
        scaffold_paths.loc[need_fill, f'phase{h}'] = -1*scaffold_paths.loc[need_fill, f'phase{h}']
        scaffold_paths.loc[need_fill, [f'scaf{h}',f'strand{h}',f'dist{h}']] = scaffold_paths.loc[need_fill, ['scaf0','strand0','dist0']].values
#
    return scaffold_paths

def SwitchHaplotypes(scaffold_paths, switches, ploidy):
     # Guarantee that all haplotypes are later still present
    missing = switches[['pid','hap2']].drop_duplicates()
    missing.rename(columns={'hap2':'hap1'}, inplace=True)
    missing = missing[missing.merge(switches[['pid','hap1']].drop_duplicates(), on=['pid','hap1'], how='left', indicator=True)['_merge'].values == "left_only"].copy()
    missing.sort_values(['pid','hap1'], inplace=True)
    free = switches[['pid','hap1']].drop_duplicates()
    free.rename(columns={'hap1':'hap2'}, inplace=True)
    free = free[free.merge(switches[['pid','hap2']].drop_duplicates(), on=['pid','hap2'], how='left', indicator=True)['_merge'].values == "left_only"].copy()
    free.sort_values(['pid','hap2'], inplace=True)
    missing['hap2'] = free['hap2'].values
    switches = pd.concat([switches, missing], ignore_index=True)
    # Apply switches
    scaffold_paths['change'] = np.isin(scaffold_paths['pid'], np.unique(switches['pid'].values))
    if np.sum(scaffold_paths['change']):
        org_paths = scaffold_paths[scaffold_paths['change']].copy()
        org_paths = FillHaplotypes(org_paths, org_paths['change'], ploidy)
        for h1 in range(ploidy):
            tmp_paths = org_paths[[f'phase{h1}',f'scaf{h1}',f'strand{h1}',f'dist{h1}']].copy()
            for h2 in range(ploidy):
                switch = np.isin(org_paths['pid'], switches.loc[(switches['hap1'] == h1) & (switches['hap2'] == h2), 'pid'].values)
                tmp_paths.loc[switch, [f'phase{h1}',f'scaf{h1}',f'strand{h1}',f'dist{h1}']] = org_paths.loc[switch, [f'phase{h2}',f'scaf{h2}',f'strand{h2}',f'dist{h2}']].values
            tmp_paths[[f'phase{h1}',f'scaf{h1}',f'dist{h1}']] = tmp_paths[[f'phase{h1}',f'scaf{h1}',f'dist{h1}']].astype(int)
            scaffold_paths.loc[scaffold_paths['change'], [f'phase{h1}',f'scaf{h1}',f'strand{h1}',f'dist{h1}']] = tmp_paths.values
    scaffold_paths = TrimAlternativesConsistentWithMain(scaffold_paths, ploidy)
    scaffold_paths.drop(columns=['change'], inplace=True)
#
    return scaffold_paths

def DuplicateHaplotypes(scaffold_paths, duplicate, ploidy):
    for h1 in range(1,ploidy):
        for h2 in range(h1+1, ploidy):
            copy = np.isin(scaffold_paths['pid'], duplicate.loc[(duplicate['hap1'] == h1) & (duplicate['hap2'] == h2), 'pid'].values)
            if np.sum(copy):
                scaffold_paths.loc[copy, [f'phase{h2}',f'scaf{h2}', f'strand{h2}', f'dist{h2}']] = scaffold_paths.loc[copy, [f'phase{h1}',f'scaf{h1}', f'strand{h1}', f'dist{h1}']].values
#
    return scaffold_paths

def GetHaplotypesThatDifferOnlyByDistance(scaffold_paths, check_pids, ploidy):
    # Find haplotypes that are identical except distance differences
    check_paths = scaffold_paths[np.isin(scaffold_paths['pid'], check_pids)].copy()
    dist_diff_only = []
    for h in range(1, ploidy):
        for h1 in range(h):
            if h1==0:
                check_paths['identical'] = (check_paths[f'phase{h}'] < 0) | ((check_paths[f'scaf{h}'] == check_paths['scaf0']) & (check_paths[f'strand{h}'] == check_paths['strand0']))
            else:
                check_paths['identical'] = ( ((check_paths[f'phase{h}'] < 0) & (check_paths[f'phase{h1}'] < 0)) |
                                             ((check_paths[f'phase{h}'] >= 0) & (check_paths[f'phase{h1}'] >= 0) & (check_paths[f'scaf{h}'] == check_paths[f'scaf{h1}']) & (check_paths[f'strand{h}'] == check_paths[f'strand{h1}'])) | # Here we need the phase checks to make sure we not consider a deletion the same as a strand equal to main
                                             ((check_paths[f'phase{h}'] < 0) & (check_paths[f'scaf{h1}'] == check_paths['scaf0']) & (check_paths[f'strand{h1}'] == check_paths['strand0'])) |
                                             ((check_paths[f'phase{h1}'] < 0) & (check_paths[f'scaf{h}'] == check_paths['scaf0']) & (check_paths[f'strand{h}'] == check_paths['strand0'])) )
            identical = check_paths.groupby(['pid'])['identical'].min().reset_index()
            identical = identical[identical['identical']].drop(columns=['identical'])
            identical['hap1'] = h1
            identical['hap2'] = h
            dist_diff_only.append(identical)
    # Enter haplotypes in both directions
    dist_diff_only = pd.concat(dist_diff_only, ignore_index=True)
    dist_diff_only = pd.concat([dist_diff_only, dist_diff_only.rename(columns={'hap1':'hap2','hap2':'hap1'})], ignore_index=True)
#
    return dist_diff_only

def CombinePathAccordingToMetaParts(scaffold_paths, meta_parts_in, conns, scaffold_graph, graph_ext, scaf_bridges, scaf_len, ploidy):
    # Combine scaffold_paths from lowest to highest position in meta scaffolds
    meta_scaffolds = meta_parts_in.loc[meta_parts_in['pos'] == 0].drop(columns=['pos'])
    scaffold_paths['reverse'] = scaffold_paths[['pid']].merge(meta_scaffolds[['pid','reverse']], on=['pid'], how='left')['reverse'].fillna(False).values.astype(bool)
    scaffold_paths = ReverseScaffolds(scaffold_paths, scaffold_paths['reverse'], ploidy)
    scaffold_paths.drop(columns=['reverse'], inplace=True)
    meta_scaffolds['start_pos'] = meta_scaffolds[['pid']].merge(scaf_len, on=['pid'], how='left')['pos'].values + 1
    meta_scaffolds.rename(columns={'pid':'new_pid'}, inplace=True)
    meta_scaffolds['apid'] = meta_scaffolds['new_pid']
    meta_scaffolds['aside'] = np.where(meta_scaffolds['reverse'], 'l', 'r')
    meta_scaffolds.drop(columns=['reverse'], inplace=True)
    meta_parts = meta_parts_in[meta_parts_in['pos'] > 0].copy()
    pos=1
    while len(meta_parts):
        # Get next connection
        meta_scaffolds = meta_scaffolds.merge(meta_parts.loc[meta_parts['pos'] == pos].drop(columns=['pos']), on=['meta'], how='inner')
        scaffold_paths['reverse'] = scaffold_paths[['pid']].merge(meta_scaffolds[['pid','reverse']], on=['pid'], how='left')['reverse'].fillna(False).values.astype(bool)
        scaffold_paths = ReverseScaffolds(scaffold_paths, scaffold_paths['reverse'], ploidy)
        scaffold_paths.drop(columns=['reverse'], inplace=True)
        meta_scaffolds.rename(columns={'pid':'bpid'}, inplace=True)
        meta_scaffolds['bside'] = np.where(meta_scaffolds['reverse'], 'r', 'l')
        meta_scaffolds[['aoverlap','boverlap']] = meta_scaffolds[['apid','aside','bpid','bside']].merge(conns, on=['apid','aside','bpid','bside'], how='left')[['aoverlap','boverlap']].values
        # Check which haplotypes are connectable (we cannot take the previous checks, because the scaffolds might be longer now)
        meta_scaffolds['connectable'] = False
        for iteration in range(4):
            nhaps = GetNumberOfHaplotypes(scaffold_paths, ploidy)
            meta_scaffolds['anhaps'] = meta_scaffolds[['new_pid']].rename(columns={'new_pid':'pid'}).merge(nhaps, on=['pid'], how='left')['nhaps'].values # apid does not exist anymore, because we merged it into new_pid
            meta_scaffolds['bnhaps'] = meta_scaffolds[['bpid']].rename(columns={'bpid':'pid'}).merge(nhaps, on=['pid'], how='left')['nhaps'].values
            ends = meta_scaffolds.loc[np.repeat(meta_scaffolds[meta_scaffolds['connectable'] == False].index.values, meta_scaffolds.loc[meta_scaffolds['connectable'] == False, 'anhaps'].values*meta_scaffolds.loc[meta_scaffolds['connectable'] == False, 'bnhaps'].values), ['new_pid','bpid','anhaps','bnhaps','aoverlap','boverlap','start_pos']].rename(columns={'new_pid':'apid'})
            meta_scaffolds.drop(columns=['anhaps','bnhaps'], inplace=True)
            ends.sort_values(['apid','bpid'], inplace=True)
            ends.reset_index(drop=True, inplace=True)
            ends['ahap'] = ends.groupby(['apid','bpid'], sort=False).cumcount()
            ends['bhap'] = ends['ahap'] % ends['bnhaps']
            ends['ahap'] = ends['ahap'] // ends['bnhaps']
            ends['aside'] = 'r'
            ends['bside'] = 'l'
            ends['amin'] = ends['start_pos'] - ends['aoverlap']
            ends['amax'] = ends['start_pos'] - 1
            ends['alen'] = ends['amax']
            ends['bmin'] = 0
            ends['bmax'] = ends['boverlap']-1
            ends['blen'] = ends[['bpid']].rename(columns={'bpid':'pid'}).merge(scaf_len, on=['pid'], how='left')['pos'].values
            ends.drop(columns=['anhaps','bnhaps','aoverlap','boverlap','start_pos'], inplace=True)
            cols = ['pid','hap','side','min','max','len']
            ends = pd.concat([ends, ends.rename(columns={**{f'a{n}':f'b{n}' for n in cols},**{f'b{n}':f'a{n}' for n in cols}})], ignore_index=True)
            ends = FilterInvalidConnections(ends, scaffold_paths, scaffold_graph, graph_ext, ploidy)
            if len(ends) == 0:
                break
            else:
                ends['samedir'] = True
                ends = FindInvalidOverlaps(ends, scaffold_paths, ploidy)
                ends = ends[['apid','ahap','bpid','bhap','valid_overlap','dup_hap','valid_path']].merge(meta_scaffolds[['new_pid','bpid']].rename(columns={'new_pid':'apid'}), on=['apid','bpid'], how='inner')
                ends['valid_overlap'] = ends['valid_overlap'] & (ends['valid_path'] == 'ab')
                ends.loc[ends['valid_path'] == 'a', 'dup_hap'] = np.where(np.isin(ends.loc[ends['valid_path'] == 'a', 'dup_hap'],['ab','b']), 'b', '')
                ends.loc[ends['valid_path'] == 'b', 'dup_hap'] = np.where(np.isin(ends.loc[ends['valid_path'] == 'b', 'dup_hap'],['ab','a']), 'a', '')
                for p in ['a','b']:
                    ends[f'{p}nhaps'] = ends[[f'{p}pid']].rename(columns={f'{p}pid':'pid'}).merge(nhaps, on=['pid'], how='left')['nhaps'].values
                ends['nhaps'] = np.maximum(ends['anhaps'], ends['bnhaps'])
                 # When all haplotypes either match to the corresponding haplotype or, if the corresponding haplotype does not exist, to the main, scaffolds are connectable
                connectable = ends.copy()
                connectable = ends[ ((ends['ahap'] == ends['bhap']) & (ends['valid_overlap'] | (ends['dup_hap'] != ''))) | 
                                    ((ends['ahap'] == 0) & (ends['bhap'] >= ends['anhaps']) & np.isin(ends['valid_path'],['ab','b'])) |
                                    ((ends['bhap'] == 0) & (ends['ahap'] >= ends['bnhaps']) & np.isin(ends['valid_path'],['ab','a'])) ].drop(columns=['valid_overlap','dup_hap','valid_path'])
                connectable = connectable.groupby(['apid','bpid','nhaps']).size().reset_index(name='matches')
                connectable = connectable[connectable['nhaps'] == connectable['matches']].copy()
                meta_scaffolds, ends = SetConnectablePathsInMetaScaffold(meta_scaffolds, ends, connectable)
                if 0 == iteration:
                    # The first step is to bring everything that is valid from both sides to the lower haplotypes
                    for p1, p2 in zip(['a','b'],['b','a']):
                        connectable = ends[ends['valid_overlap'] & (ends[f'{p1}nhaps'] > ends[f'{p2}nhaps'])].drop(columns=['valid_overlap','dup_hap','valid_path'])
                        connectable.sort_values(['apid','bpid',f'{p1}hap',f'{p2}hap'], inplace=True)
                        connectable['fixed'] = False
                        while np.sum(connectable['fixed'] == False):
                            # Lowest haplotypes of p2 are fixed for lowest haplotypes of p1
                            connectable.loc[(connectable['apid'] != connectable['apid'].shift(1)) | connectable['fixed'].shift(1), 'fixed'] = True
                            # Remove all other haplotypes of p2 for fixed haplotypes of p1
                            connectable['delete'] = False
                            connectable.loc[(connectable['apid'] == connectable['apid'].shift(1)) & (connectable[f'{p1}hap'] == connectable[f'{p1}hap'].shift(1)) & connectable['fixed'].shift(1), 'delete'] = True
                            while True:
                                old_ndels = np.sum(connectable['delete'])
                                connectable.loc[(connectable['apid'] == connectable['apid'].shift(1)) & (connectable[f'{p1}hap'] == connectable[f'{p1}hap'].shift(1)) & connectable['delete'].shift(1), 'delete'] = True
                                if old_ndels == np.sum(connectable['delete']):
                                    break
                            # Remove haplotypes of p2 that are fixed in a haplotype of p1 in all other haplotypes of p1
                            connectable.loc[connectable[['apid','bpid',f'{p2}hap']].merge(connectable.loc[connectable['fixed'], ['apid','bpid',f'{p2}hap']], on=['apid','bpid',f'{p2}hap'], how='left', indicator=True)['_merge'].values == "both", 'delete'] = True
                            connectable.loc[connectable['fixed'], 'delete'] = False
                            connectable = connectable[connectable['delete'] == False].copy()
                        switches = connectable.loc[connectable[f'{p1}hap'] != connectable[f'{p2}hap'], [f'{p1}pid',f'{p1}hap',f'{p2}hap']].rename(columns={f'{p1}pid':'pid',f'{p1}hap':'hap1',f'{p2}hap':'hap2'})
                        scaffold_paths = SwitchHaplotypes(scaffold_paths, switches, ploidy)
                elif 1 == iteration or 3 == iteration:
                    # When all haplotypes either match to the corresponding haplotype or, if the corresponding haplotype does not exist, to another haplotype, scaffolds can be made connectable by duplicating haplotypes on the paths with less haplotypes(p1)
                    for p1, p2 in zip(['a','b'],['b','a']):
                        connectable = ends[ (ends[f'{p2}hap'] >= ends[f'{p1}nhaps']) & np.isin(ends['valid_path'],['ab',p2])].drop(columns=['valid_overlap','dup_hap','valid_path'])
                        connectable.sort_values(['apid','bpid',f'{p2}hap',f'{p1}hap'], inplace=True)
                        connectable = connectable.groupby(['apid','bpid',f'{p2}hap']).first().reset_index() # Take the lowest haplotype that can be duplicated to fill the missing one
                        connectable = connectable.loc[connectable[f'{p1}hap'] > 0, [f'{p1}pid',f'{p1}hap',f'{p2}hap']].rename(columns={f'{p1}pid':'pid', f'{p1}hap':'hap1', f'{p2}hap':'hap2'}) # If the lowest is the main, we do not need to do anything
                        scaffold_paths = DuplicateHaplotypes(scaffold_paths, connectable, ploidy)
                    # Switching haplotypes might also help to make paths connectable
                    connectable = ends[ends['valid_overlap'] | (ends['dup_hap'] != '')].drop(columns=['valid_overlap','valid_path'])
                    connectable.rename(columns={'dup_hap':'switchcol'}, inplace=True)
                    connectable['switchcol'] = np.where(connectable['switchcol'] != 'b', 'b', 'a') # Switch the column that is not a duplicated haplotype, because switching a duplicated haplotype to a position lower than the duplicate would remove that status (as we always want to keep one version and chose the lowest haplotype with that version for it)
                    connectable['nhaps'] = np.minimum(connectable['anhaps'], connectable['bnhaps'])
                    connectable = connectable[(connectable['ahap'] < connectable['nhaps']) & (connectable['bhap'] < connectable['nhaps'])].drop(columns=['anhaps','bnhaps'])
                    connectable['nmatches'] = connectable[['apid','bpid']].merge(connectable[connectable['ahap'] == connectable['bhap']].groupby(['apid','bpid']).size().reset_index(name='matches'), on=['apid','bpid'], how='left')['matches'].fillna(0).values.astype(int)
                    connectable = connectable[connectable['nmatches'] < connectable['nhaps']].copy()
                    connectable['new_ahap'] = connectable['ahap']
                    connectable['new_bhap'] = connectable['bhap']
                    while len(connectable):
                        connectable['match'] = connectable['new_ahap'] == connectable['new_bhap']
                        connectable['amatch'] = connectable[['apid','bpid','new_ahap']].merge(connectable.groupby(['apid','bpid','new_ahap'])['match'].max().reset_index(), on=['apid','bpid','new_ahap'], how='left')['match'].values
                        connectable['bmatch'] = connectable[['apid','bpid','new_bhap']].merge(connectable.groupby(['apid','bpid','new_bhap'])['match'].max().reset_index(), on=['apid','bpid','new_bhap'], how='left')['match'].values
                        connectable['switchable'] = connectable[['apid','bpid','new_ahap','new_bhap']].merge( connectable[['apid','bpid','new_ahap','new_bhap']].rename(columns={'new_ahap':'new_bhap','new_bhap':'new_ahap'}), on=['apid','bpid','new_ahap','new_bhap'], how='left', indicator=True)['_merge'].values == "both"
                        for p1, p2 in zip(['a','b'],['b','a']):
                            switches = connectable.loc[(connectable[f'{p1}match'] == False) & (connectable['switchcol'] == p2) & ((connectable[f'{p2}match'] == False) | connectable['switchable']), ['apid','new_ahap','bpid','new_bhap','switchcol']].copy()
                            switches = switches.groupby(['apid','bpid']).first().reset_index() # Only one switch per meta_paths per round to avoid conflicts
                            switches.rename(columns={f'new_{p2}hap':f'{p2}hap',f'new_{p1}hap':f'new_{p2}hap'}, inplace=True)
                            switches = pd.concat([switches.rename(columns={f'new_{p2}hap':f'{p2}hap',f'{p2}hap':f'new_{p2}hap'}), switches], ignore_index=True)
                            switches = connectable[['apid','bpid',f'new_{p2}hap']].rename(columns={f'new_{p2}hap':f'{p2}hap'}).merge(switches, on=['apid','bpid',f'{p2}hap'], how='left')[f'new_{p2}hap'].values
                            connectable[f'new_{p2}hap'] = np.where(np.isnan(switches), connectable[f'new_{p2}hap'], switches).astype(int)
                        connectable['old_nmatches'] = connectable['nmatches']
                        connectable['nmatches'] = connectable[['apid','bpid']].merge(connectable[connectable['new_ahap'] == connectable['new_bhap']].groupby(['apid','bpid']).size().reset_index(name='matches'), on=['apid','bpid'], how='left')['matches'].fillna(0).values.astype(int)
                        improvable = (connectable['old_nmatches'] < connectable['nmatches']) & (connectable['nmatches'] < connectable['nhaps'])
                        for p in ['a','b']:
                            switches = connectable.loc[(improvable == False) & (connectable[f'{p}hap'] != connectable[f'new_{p}hap']), [f'{p}pid',f'{p}hap',f'new_{p}hap']].drop_duplicates()
                            switches.rename(columns={f'{p}pid':'pid',f'{p}hap':'hap1',f'new_{p}hap':'hap2'}, inplace=True)
                            scaffold_paths = SwitchHaplotypes(scaffold_paths, switches, ploidy)
                        connectable = connectable[improvable].copy()
                elif 2 == iteration:
                        # Remove haplotypes that block a connection if they differ only by distance from a valid haplotype
                        delete_haps = []
                        connectable_pids = []
                        for p in ['a','b']:
                            dist_diff_only = GetHaplotypesThatDifferOnlyByDistance(scaffold_paths, ends[f'{p}pid'].drop_duplicates().values, ploidy)
                            valid_haps = ends.loc[ends['valid_overlap'], [f'{p}pid',f'{p}hap']].drop_duplicates()
                            valid_haps.rename(columns={f'{p}pid':'pid',f'{p}hap':'hap'}, inplace=True)
                            dist_diff_only = dist_diff_only.merge(valid_haps.rename(columns={'hap':'hap1'}), on=['pid','hap1'], how='inner')
                            invalid_haps = ends[[f'{p}pid',f'{p}nhaps']].drop_duplicates()
                            invalid_haps.rename(columns={f'{p}pid':'pid'}, inplace=True)
                            invalid_haps = invalid_haps.loc[np.repeat(invalid_haps.index.values, invalid_haps[f'{p}nhaps'].values), ['pid']].copy()
                            invalid_haps['hap'] = invalid_haps.groupby(['pid']).cumcount()
                            invalid_haps = invalid_haps[invalid_haps.merge(valid_haps, on=['pid','hap'], how='left', indicator=True)['_merge'].values == "left_only"].copy()
                            invalid_haps['dist_diff_only'] = invalid_haps.merge(dist_diff_only[['pid','hap2']].rename(columns={'hap2':'hap'}), on=['pid','hap'], how='left', indicator=True)['_merge'].values == "both"
                            delete_haps.append(invalid_haps.loc[invalid_haps['dist_diff_only'], ['pid','hap']])
                            valid_haps['valid'] = True
                            valid_haps = pd.concat([valid_haps, invalid_haps.rename(columns={'dist_diff_only':'valid'})], ignore_index=True)
                            valid_haps = valid_haps.groupby(['pid'])['valid'].min().reset_index()
                            valid_haps = valid_haps.loc[valid_haps['valid'], ['pid']].rename(columns={'pid':f'{p}pid'})
                            valid_haps = valid_haps.merge(ends[['apid','bpid']].drop_duplicates(), on=[f'{p}pid'], how='left')
                            connectable_pids.append(valid_haps)
                        connectable_pids = connectable_pids[0].merge(connectable_pids[1], on=['apid','bpid'], how='inner')
                        delete_haps[0] = delete_haps[0][np.isin(delete_haps[0]['pid'], connectable_pids['apid'].values)].copy()
                        delete_haps[1] = delete_haps[1][np.isin(delete_haps[1]['pid'], connectable_pids['bpid'].values)].copy()
                        delete_haps = pd.concat(delete_haps, ignore_index=True)
                        scaffold_paths = RemoveHaplotypes(scaffold_paths, delete_haps, ploidy)
                        scaffold_paths = ShiftHaplotypesToLowestPossible(scaffold_paths, ploidy)
        meta_scaffolds.drop(columns=['bside'], inplace=True)
#
        # Since all the existing haplotypes must match take the paths with more haplotypes in the overlapping region (only relevant starting at the second position, since the first cannot have a variant or they would not have been merged)
        long_overlaps = meta_scaffolds.loc[meta_scaffolds['connectable'] & (np.maximum(meta_scaffolds['aoverlap'],meta_scaffolds['boverlap']) > 1), ['new_pid','bpid','aoverlap','boverlap','start_pos']].reset_index()
        overhaps = long_overlaps[['new_pid','aoverlap','start_pos','index']].rename(columns={'new_pid':'pid','aoverlap':'min_pos','start_pos':'max_pos'})
        overhaps['min_pos'] = overhaps['max_pos'] - overhaps['min_pos']
        overhaps['max_pos'] -= 1
        overhaps['path'] = 'a'
        overhaps = [overhaps]
        overhaps.append(long_overlaps[['bpid','boverlap','index']].rename(columns={'bpid':'pid','boverlap':'max_pos'}))
        overhaps[-1]['min_pos'] = 0
        overhaps[-1]['max_pos'] -= 1
        overhaps[-1]['path'] = 'b'
        overhaps = pd.concat(overhaps, ignore_index=True)
        overhaps['len'] = overhaps['max_pos'] - overhaps['min_pos'] + 1
        overhaps.sort_values(['index','path'], inplace=True)
        overhaps = overhaps.loc[np.repeat(overhaps.index.values, overhaps['len'].values), ['index','path','pid','min_pos']].reset_index(drop=True)
        overhaps['pos'] = overhaps['min_pos'] + overhaps.groupby(['index','path'], sort=False).cumcount()
        overhaps[[f'hap{h}' for h in range(ploidy)]] = overhaps[['pid','pos']].merge(scaffold_paths[['pid','pos']+[f'phase{h}' for h in range(ploidy)]], on=['pid','pos'], how='left')[[f'phase{h}' for h in range(ploidy)]] >= 0
        for h in range(1,ploidy):
            # Distance variants at the first position do not matter
            overhaps.loc[(overhaps['min_pos'] == overhaps['pos']), f'hap{h}'] = overhaps.loc[(overhaps['min_pos'] == overhaps['pos']), f'hap{h}'].values & (overhaps.loc[(overhaps['min_pos'] == overhaps['pos']), ['pid','pos']].merge(scaffold_paths[['pid','pos']+[f'scaf{h}',f'strand{h}']], on=['pid','pos'], how='left')[[f'scaf{h}',f'strand{h}']].values != overhaps.loc[(overhaps['min_pos'] == overhaps['pos']), ['pid','pos']].merge(scaffold_paths[['pid','pos']+['scaf0','strand0']], on=['pid','pos'], how='left')[['scaf0','strand0']].values).any(axis=1)
        overhaps = overhaps.groupby(['index','path'], sort=False)[[f'hap{h}' for h in range(ploidy)]].max().reset_index()
        overhaps['nhaps'] = overhaps[[f'hap{h}' for h in range(ploidy)]].sum(axis=1)
        meta_scaffolds[['ahaps','bhaps']] = 0
        for p in ['a','b']:
            meta_scaffolds.loc[ overhaps.loc[overhaps['path'] == p, 'index'].values, f'{p}haps'] = overhaps.loc[overhaps['path'] == p, 'nhaps'].values
        meta_scaffolds.loc[meta_scaffolds['ahaps'] < meta_scaffolds['bhaps'], 'boverlap'] = 1 # The first scaffold in pathb will also be removed, because it cannot have an alternative and does not contain the distance information
        meta_scaffolds.loc[meta_scaffolds['ahaps'] < meta_scaffolds['bhaps'], 'start_pos'] -= meta_scaffolds.loc[meta_scaffolds['ahaps'] < meta_scaffolds['bhaps'], 'aoverlap'] - 1
        meta_scaffolds.drop(columns=['aoverlap','ahaps','bhaps'], inplace=True)
        # Connect scaffolds
        scaffold_paths[['overlap','shift']] = scaffold_paths[['pid']].merge( meta_scaffolds.loc[meta_scaffolds['connectable'], ['bpid','boverlap','start_pos']].rename(columns={'bpid':'pid'}), on=['pid'], how='left')[['boverlap','start_pos']].fillna(0).values.astype(int)
        scaffold_paths['shift'] -= scaffold_paths['overlap']
        scaffold_paths = scaffold_paths[ scaffold_paths['pos'] >= scaffold_paths['overlap'] ].drop(columns=['overlap'])
        scaffold_paths['pos'] += scaffold_paths['shift']
        scaffold_paths.drop(columns=['shift'], inplace=True)
        scaffold_paths['trim'] = scaffold_paths[['pid']].merge( meta_scaffolds.loc[meta_scaffolds['connectable'], ['new_pid','start_pos']].rename(columns={'new_pid':'pid'}), on=['pid'], how='left')['start_pos'].fillna(sys.maxsize*0.9).values.astype(int) # sys.maxsize*0.9 to avoid variable overrun due to type conversion
        scaffold_paths = scaffold_paths[scaffold_paths['pos'] < scaffold_paths['trim']].drop(columns=['trim'])
        scaffold_paths['new_pid'] = scaffold_paths[['pid']].merge( meta_scaffolds.loc[meta_scaffolds['connectable'], ['bpid','new_pid']].rename(columns={'bpid':'pid'}), on=['pid'], how='left')['new_pid'].values
        scaffold_paths.loc[np.isnan(scaffold_paths['new_pid']) == False, 'pid'] = scaffold_paths.loc[np.isnan(scaffold_paths['new_pid']) == False, 'new_pid'].astype(int)
        scaffold_paths.drop(columns=['new_pid'], inplace=True)
        scaffold_paths.sort_values(['pid','pos'], inplace=True)
#
        # The unconnectable paths in meta_scaffolds might have had haplotypes duplicated in an attempt to make them connectable. Remove those duplications
        for h1 in range(1, ploidy):
            for h2 in range(h1+1, ploidy):
                scaffold_paths['remove'] =  ( (np.sign(scaffold_paths[f'phase{h1}']) == np.sign(scaffold_paths[f'phase{h2}'])) & (scaffold_paths[f'scaf{h1}'] == scaffold_paths[f'scaf{h2}']) &
                                              (scaffold_paths[f'strand{h1}'] == scaffold_paths[f'strand{h2}']) & (scaffold_paths[f'dist{h1}'] == scaffold_paths[f'dist{h2}']) )
                remove = scaffold_paths.groupby(['pid'])['remove'].min().reset_index()
                remove = remove.loc[remove['remove'], 'pid'].values
                scaffold_paths['remove'] = np.isin(scaffold_paths['pid'], remove)
                scaffold_paths = RemoveHaplotype(scaffold_paths, scaffold_paths['remove'], h2)
                scaffold_paths.drop(columns=['remove'], inplace=True)
#
        # Make sure the haplotypes are still sorted by highest support for bridges
        bsupp = pd.concat([ meta_scaffolds[['new_pid']].rename(columns={'new_pid':'pid'}), meta_scaffolds.loc[meta_scaffolds['connectable'] == False, ['bpid']].rename(columns={'bpid':'pid'})], ignore_index=True)
        bsupp.sort_values(['pid'], inplace=True)
        nhaps = GetNumberOfHaplotypes(scaffold_paths, ploidy)
        bsupp = bsupp.merge(nhaps, on=['pid'], how='left')
        bsupp = bsupp[bsupp['nhaps'] > 1].copy()
        bsupp = bsupp.loc[np.repeat(bsupp.index.values, bsupp['nhaps'].values)].drop(columns=['nhaps'])
        bsupp.reset_index(drop=True, inplace=True)
        bsupp['hap'] = bsupp.groupby(['pid'], sort=False).cumcount()
        bsupp['group'] = 0
        bsupp = GetBridgeSupport(bsupp, scaffold_paths, scaf_bridges, ploidy)
        if len(bsupp):
            bsupp.drop(columns=['group'], inplace=True)
            bsupp.sort_values(['pid','bplace'], inplace=True)
            bsupp['new_hap'] = bsupp.groupby(['pid']).cumcount()
            bsupp = bsupp.loc[bsupp['hap'] != bsupp['new_hap'], ['pid','hap','new_hap']].rename(columns={'hap':'hap1','new_hap':'hap2'})
            scaffold_paths = SwitchHaplotypes(scaffold_paths, bsupp, ploidy)
            scaffold_paths = ShiftHaplotypesToLowestPossible(scaffold_paths, ploidy)
#
        # Break unconnectable meta_scaffolds
        meta_scaffolds.loc[meta_scaffolds['connectable'] == False, 'new_pid'] = meta_scaffolds.loc[meta_scaffolds['connectable'] == False, 'bpid']
        meta_scaffolds.loc[meta_scaffolds['connectable'] == False, ['start_pos','boverlap']] = 0
#
        # Prepare next round
        meta_scaffolds['start_pos'] += meta_scaffolds[['bpid']].rename(columns={'bpid':'pid'}).merge(scaf_len, on=['pid'], how='left')['pos'].values + 1 - meta_scaffolds['boverlap']
        meta_scaffolds['apid'] = meta_scaffolds['bpid']
        meta_scaffolds['aside'] = np.where(meta_scaffolds['reverse'], 'l', 'r')
        meta_scaffolds.drop(columns=['bpid','reverse','boverlap'], inplace=True)
        meta_parts = meta_parts[meta_parts['pos'] > pos].copy()
        pos += 1
#
    # Check that positions are consistent in scaffold_paths
    inconsistent = scaffold_paths[(scaffold_paths['pos'] < 0) |
                                  ((scaffold_paths['pos'] == 0) & (scaffold_paths['pid'] == scaffold_paths['pid'].shift(1))) |
                                  ((scaffold_paths['pos'] > 0) & ((scaffold_paths['pid'] != scaffold_paths['pid'].shift(1)) | (scaffold_paths['pos'] != scaffold_paths['pos'].shift(1)+1)))].copy()
    if len(inconsistent):
        print("Warning: Scaffold paths is inconsistent after CombinePathAccordingToMetaParts.")
        print(inconsistent)
#
    return scaffold_paths

def CombinePathOnUniqueOverlap(scaffold_paths, scaffold_graph, graph_ext, scaf_bridges, ploidy):
    ends = GetDuplicatedPathEnds(scaffold_paths, ploidy)
    # If the paths continue in the same direction on the non-end side they are alternatives and not combinable
    ends = ends[ends['samedir'] == (ends['aside'] != ends['bside'])].copy()
#
    # Check that combining the paths does not violate scaffold_graph
    ends = FilterInvalidConnections(ends, scaffold_paths, scaffold_graph, graph_ext, ploidy)
    ends = ends[ends['valid_path'] == 'ab'].drop(columns=['valid_path'])
#
    if len(ends):
        # Combine all ends that describe the same connection (multiple haplotypes of same path and multiple mapping options): Take the lowest overlap (to not compress repeats) supported by most haplotypes (requiring all haplotypes to be present would remove connections, where one haplotype reaches further into the other scaffold than other haplotypes)
        for p in ['a','b']:
            ends[f'{p}overlap'] = np.where(ends[f'{p}side'] == 'l', ends[f'{p}max']+1, ends[f'{p}len']-ends[f'{p}min']+1)
        ends.drop(columns=['did','amin','amax','bmin','bmax','alen','blen'], inplace=True)
        conns = ends.groupby(['apid','aside','bpid','bside','aoverlap','boverlap'])['matches'].max().reset_index()
        matches = conns.groupby(['apid','aside','bpid','bside'])['matches'].agg(['max','size'])
        conns['max_matches'] = np.repeat(matches['max'].values, matches['size'].values)
        conns['anhaps'] = ends[['apid','aside','bpid','bside','aoverlap','boverlap','ahap']].drop_duplicates().groupby(['apid','aside','bpid','bside','aoverlap','boverlap']).size().values
        conns['bnhaps'] = ends[['apid','aside','bpid','bside','aoverlap','boverlap','bhap']].drop_duplicates().groupby(['apid','aside','bpid','bside','aoverlap','boverlap']).size().values
        conns['minhaps'] = np.minimum(conns['anhaps'], conns['bnhaps'])
        conns['maxhaps'] = np.maximum(conns['anhaps'], conns['bnhaps'])
        conns['maxoverlap'] = np.maximum(conns['aoverlap'], conns['boverlap'])
        conns['minoverlap'] = np.minimum(conns['aoverlap'], conns['boverlap'])
        conns['tiebreakhaps'] = np.where(conns['apid'] < conns['bpid'], conns['anhaps'], conns['bnhaps'])
        conns['tiebreakoverlap'] = np.where(conns['apid'] < conns['bpid'], conns['aoverlap'], conns['boverlap'])
        conns.sort_values(['apid','aside','bpid','bside','minhaps','maxhaps','maxoverlap','minoverlap','tiebreakhaps','tiebreakoverlap'], ascending=[True,True,True,True,False,False,True,True,False,True], inplace=True)
        conns = conns.groupby(['apid','aside','bpid','bside']).first().reset_index()
        conns.drop(columns=['minhaps','maxhaps','maxoverlap','minoverlap','tiebreakhaps','tiebreakoverlap'], inplace=True)
        # Get number of different haplotypes that are actually compared (because they differ close enough to the checked end) to not give an advantage to paths with more haplotypes in the upcoming filtering
        dedup_haps = GetDeduplicatedHaplotypesAtPathEnds(ends[['apid','aside','matches']].drop_duplicates().rename(columns={'apid':'pid','aside':'side'}), scaffold_paths, scaffold_graph, ploidy)
        dedup_haps = ends[['apid','aside','ahap','bpid','bside','bhap','matches']].merge(dedup_haps.rename(columns={col:f'a{col}' for col in dedup_haps.columns if col != 'matches'}), on=['apid','aside','matches','ahap'], how='inner')\
                                                                                  .merge(dedup_haps.rename(columns={col:f'b{col}' for col in dedup_haps.columns if col != 'matches'}), on=['bpid','bside','matches','bhap'], how='inner')
        dedup_haps.drop(columns=['matches'], inplace=True)
        for p in ['a','b']:
            conns[f'{p}nhaps'] = conns[['apid','aside','bpid','bside']].merge(dedup_haps[['apid','aside','bpid','bside',f'{p}hap']].drop_duplicates().groupby(['apid','aside','bpid','bside']).size().reset_index(name='nhaps'), on=['apid','aside','bpid','bside'], how='left')['nhaps'].astype(int).values
        conns['minhaps'] = np.minimum(conns['anhaps'], conns['bnhaps'])
        conns['maxhaps'] = np.maximum(conns['anhaps'], conns['bnhaps'])
        # Count alternatives and take only unique connections with preferences giving to more haplotypes and matches
        for p in ['a','b']:
            sort_cols = [f'{p}pid',f'{p}side','minhaps','maxhaps',f'{p}nhaps','max_matches']
            conns.sort_values(sort_cols, ascending=[True,True,False,False,False,False], inplace=True) # Do not use minhaps/maxhaps here, because they are between different scaffolds and this would give an advantage to scaffolds with more haplotypes
            conns[f'{p}alts'] = conns.groupby([f'{p}pid',f'{p}side'], sort=False).cumcount()+1
            equivalent = conns.groupby(sort_cols, sort=False)[f'{p}alts'].agg(['max','size'])
            conns[f'{p}alts'] = np.repeat(equivalent['max'].values, equivalent['size'].values)
        conns = conns.loc[(conns['aalts'] == 1) & (conns['balts'] == 1), ['apid','aside','bpid','bside','aoverlap','boverlap']].copy()
#
        # Assign all connected paths to a meta scaffold to define connection order
        conns.sort_values(['apid','aside','bpid','bside'], inplace=True)
        meta_scaffolds = pd.DataFrame({'meta':np.unique(conns['apid']), 'size':1, 'lcon':-1, 'lcon_side':'', 'rcon':-1, 'rcon_side':''})
        meta_scaffolds.index = meta_scaffolds['meta'].values
        meta_parts = pd.DataFrame({'scaffold':meta_scaffolds['meta'], 'meta':meta_scaffolds['meta'], 'pos':0, 'reverse':False})
        meta_parts.index = meta_parts['scaffold'].values
        meta_scaffolds.loc[conns.loc[conns['aside'] == 'l', 'apid'].values, ['lcon','lcon_side']] = conns.loc[conns['aside'] == 'l', ['bpid','bside']].values
        meta_scaffolds.loc[conns.loc[conns['aside'] == 'r', 'apid'].values, ['rcon','rcon_side']] = conns.loc[conns['aside'] == 'r', ['bpid','bside']].values
        # Rename some columns and create extra columns just to call the same function as used for the contig scaffolding on unique bridges
        meta_scaffolds['left'] = meta_scaffolds['meta']
        meta_scaffolds['lside'] = 'l'
        meta_scaffolds['right'] = meta_scaffolds['meta']
        meta_scaffolds['rside'] = 'r'
        meta_scaffolds['lextendible'] = True
        meta_scaffolds['rextendible'] = True
        meta_scaffolds['circular'] = False
        meta_scaffolds.rename(columns={'meta':'scaffold','lcon':'lscaf','lcon_side':'lscaf_side','rcon':'rscaf','rcon_side':'rscaf_side'}, inplace=True)
        meta_parts.rename(columns={'scaffold':'conpart','meta':'scaffold'}, inplace=True)
        meta_scaffolds, meta_parts = ScaffoldAlongGivenConnections(meta_scaffolds, meta_parts)
        meta_parts.rename(columns={'conpart':'pid','scaffold':'meta'}, inplace=True)
        meta_parts.sort_values(['meta','pos'], inplace=True)
#
        scaf_len = scaffold_paths.groupby(['pid'])['pos'].max().reset_index()
        scaffold_paths = CombinePathAccordingToMetaParts(scaffold_paths, meta_parts, conns, scaffold_graph, graph_ext, scaf_bridges, scaf_len, ploidy)

    return scaffold_paths

def GetExistingHaplotypesFromPaths(scaffold_paths, ploidy):
    haps = [scaffold_paths[['pid']].drop_duplicates()]
    haps[0]['hap'] = 0
    for h in range(1, ploidy):
        haps.append( scaffold_paths.loc[scaffold_paths[f'phase{h}'] > 0, ['pid']].drop_duplicates() )
        haps[-1]['hap'] = h
    haps = pd.concat(haps, ignore_index=True).sort_values(['pid','hap']).reset_index(drop=True)
#
    return haps

def GetEndPosForEachHaplotypes(scaffold_paths, ploidy):
    scaf_len = scaffold_paths.groupby(['pid'])['pos'].max().reset_index(name='max_pos')
    scaf_len['min_pos'] = 0
    scaf_len = GetExistingHaplotypesFromPaths(scaffold_paths, ploidy).merge(scaf_len[['pid','min_pos','max_pos']], on=['pid'], how='left')
    scaf_len['finished'] = False
    while np.sum(scaf_len['finished'] == False):
        for h in range(ploidy):
            cur = (scaf_len['finished'] == False) & (scaf_len['hap'] == h)
            scaf_len.loc[cur, ['phase','scaf','scaf0']] = scaf_len.loc[cur, ['pid','min_pos']].rename(columns={'min_pos':'pos'}).merge(scaffold_paths[['pid','pos',f'phase{h}',f'scaf{h}']+([] if h==0 else ['scaf0'])], on=['pid','pos'], how='left')[[f'phase{h}',f'scaf{h}','scaf0']].values
        found_del = (scaf_len['scaf'] < 0) & (scaf_len['phase'] > 0) | (scaf_len['scaf0'] < 0) & (scaf_len['phase'] < 0)
        scaf_len.loc[found_del, 'min_pos'] += 1
        scaf_len.loc[scaf_len['min_pos'] > scaf_len['max_pos'], 'finished'] = True
        scaf_len.loc[found_del == False, 'finished'] = True
    scaf_len['finished'] = scaf_len['max_pos'] < scaf_len['min_pos']
    while np.sum(scaf_len['finished'] == False):
        for h in range(ploidy):
            cur = (scaf_len['finished'] == False) & (scaf_len['hap'] == h)
            scaf_len.loc[cur, ['phase','scaf','scaf0']] = scaf_len.loc[cur, ['pid','max_pos']].rename(columns={'max_pos':'pos'}).merge(scaffold_paths[['pid','pos',f'phase{h}',f'scaf{h}']+([] if h==0 else ['scaf0'])], on=['pid','pos'], how='left')[[f'phase{h}',f'scaf{h}','scaf0']].values
        found_del = (scaf_len['scaf'] < 0) & (scaf_len['phase'] > 0) | (scaf_len['scaf0'] < 0) & (scaf_len['phase'] < 0)
        scaf_len.loc[found_del, 'max_pos'] -= 1
        scaf_len.loc[found_del == False, 'finished'] = True
    scaf_len.drop(columns=['finished','phase','scaf','scaf0'], inplace=True)
#
    return scaf_len

def ExtendDuplicationsFromEnd(duplications, scaffold_paths, end, scaf_len, ploidy):
    # Get duplications starting at end
    ext_dups = []
    if 'min_pos' in scaf_len.columns:
        edups = duplications[duplications['apos'] == (duplications[['apid','ahap']].rename(columns={'apid':'pid','ahap':'hap'}).merge(scaf_len, on=['pid','hap'], how='left')['min_pos' if end == 'l' else 'max_pos'].values)].copy()
    else:
        edups = duplications[duplications['apos'] == (0 if end == 'l' else duplications[['apid']].rename(columns={'apid':'pid'}).merge(scaf_len, on=['pid'], how='left')['pos'].values)].copy()
    edups['did'] = np.arange(len(edups))
    ext_dups.append(edups.copy())
    while len(edups):
        # Get next position to see if we can extend the duplication
        edups = GetPositionsBeforeDuplication(edups, scaffold_paths, ploidy, end == 'l')
        # Remove everything that does not fit
        edups = edups[(edups['adist'] == edups['bdist']) & (edups['aprev_pos'] >= 0) & (edups['bprev_pos'] >= 0)].drop(columns=['adist','bdist'])
        edups['apos'] = edups['aprev_pos']
        edups['bpos'] = edups['bprev_pos']
        # Check if we have a duplication at the new position
        edups = edups[['apid','apos','ahap','bpid','bpos','bhap','samedir','did']].merge(duplications, on=['apid','apos','ahap','bpid','bpos','bhap','samedir'], how='inner')
        # Insert the valid extensions
        ext_dups.append(edups.copy())
    ext_dups = pd.concat(ext_dups, ignore_index=True)
    ext_dups.sort_values(['did','apos'], inplace=True)
#
    return ext_dups

def RemoveDuplicates(scaffold_paths, remove_all, ploidy):
    # Get min/max position for each haplotype in paths (excluding deletions at ends)
    scaf_len = GetEndPosForEachHaplotypes(scaffold_paths, ploidy)
    # Remove haplotypes that are only deletions
    rem_paths = scaf_len.loc[scaf_len['max_pos'] < scaf_len['min_pos'], ['pid','hap']].copy()
    scaf_len = scaf_len[scaf_len['max_pos'] >= scaf_len['min_pos']].copy()
    # Remove haplotype that are a complete duplication of another haplotype
    scaf_len['len'] = scaf_len['max_pos'] - scaf_len['min_pos'] + 1
    check = scaf_len.drop(columns=['max_pos']).rename(columns={'hap':'hap1','len':'len1'}).merge(scaf_len[['pid','hap','len']].rename(columns={'hap':'hap2','len':'len2'}), on=['pid'], how='left')
    scaf_len.drop(columns=['len'], inplace=True)
    check = check[ (check['len1'] < check['len2']) | ((check['len1'] == check['len2']) & (check['hap1'] > check['hap2'])) ].drop(columns=['len2']) # Only the shorter one can be a (partial) duplicate of the other one and if they are the same length we want to keep one of the duplications, so chose the higher haplotype for potential removal
    if len(check):
        check['index'] = check.index.values
        check = check.loc[np.repeat(check['index'].values, check['len1'].values)].reset_index(drop=True)
        check['pos'] = check.groupby(['index'], sort=False).cumcount() + check['min_pos']
        check_paths = check[['pid','pos']].merge(scaffold_paths, on=['pid','pos'], how='left')
        for i in [1,2]:
            check[[f'scaf{i}',f'strand{i}',f'dist{i}']] = check_paths[['scaf0','strand0','dist0']].values
            for h in range(1,ploidy):
                cur = (check[f'hap{i}'] == h) & (check_paths[f'phase{h}'] > 0)
                check.loc[cur, [f'scaf{i}',f'strand{i}',f'dist{i}']] = check_paths.loc[cur, [f'scaf{h}',f'strand{h}',f'dist{h}']].values
        check['identical'] = (check['scaf1'] == check['scaf2']) & (check['strand1'] == check['strand2']) & ((check['dist1'] == check['dist2']) | (check['pos'] == check['min_pos']))
        check = check.groupby(['pid','hap1','hap2'])['identical'].min().reset_index()
        check = check[check['identical']].drop(columns=['identical','hap2']).drop_duplicates()
        rem_paths = pd.concat([rem_paths, check.rename(columns={'hap1':'hap'})], ignore_index=True)
    if len(rem_paths):
        scaffold_paths = RemoveHaplotypes(scaffold_paths, rem_paths, ploidy)
        scaf_len = GetEndPosForEachHaplotypes(scaffold_paths, ploidy)
#
    # Get duplications that contain both path ends for side a (we cannot go down to haplotype level here, because we have not separated the haplotypes yet, so they miss the duplications they share with main)
    duplications = GetDuplications(scaffold_paths, ploidy)
    ends = duplications.groupby(['apid','bpid'])['apos'].agg(['min','max']).reset_index()
    ends['amin'] = ends[['apid']].rename(columns={'apid':'pid'}).merge(scaf_len.groupby(['pid'])['min_pos'].max().reset_index(), on=['pid'], how='left')['min_pos'].values
    ends['amax'] = ends[['apid']].rename(columns={'apid':'pid'}).merge(scaf_len.groupby(['pid'])['max_pos'].min().reset_index(), on=['pid'], how='left')['max_pos'].values
    ends = ends.loc[(ends['min'] <= ends['amin']) & (ends['max'] >= ends['amax']), ['apid','bpid']].copy()
    duplications = duplications.merge(ends, on=['apid','bpid'], how='inner')
    # Add length of haplotype a
    duplications['alen']  = duplications[['apid','ahap']].rename(columns={'apid':'pid','ahap':'hap'}).merge(scaf_len[['pid','hap','max_pos']], on=['pid','hap'], how='left')['max_pos'].values
    # Check if they are on the same or opposite strand (alone it is useless, but it sets the requirements for the direction of change for the positions)
    duplications = AddStrandToDuplications(duplications, scaffold_paths, ploidy)
    duplications['samedir'] = duplications['astrand'] == duplications['bstrand']
    duplications.drop(columns=['astrand','bstrand'], inplace=True)
    # Extend the duplications with valid distances from start and only keep the ones that reach end
    duplications = SeparateDuplicationsByHaplotype(duplications, scaffold_paths, ploidy)
    duplications = ExtendDuplicationsFromEnd(duplications, scaffold_paths, 'l', scaf_len, ploidy)
    duplications = duplications.groupby(['apid','ahap','bpid','bhap','alen'])['apos'].max().reset_index(name='amax')
    duplications = duplications[duplications['amax'] == duplications['alen']].drop(columns=['amax','alen'])
    # Remove duplicated scaffolds (Depending on the setting only remove duplicates with the same length, because we need the other ends later for merging)
    rem_paths = duplications.merge(duplications.rename(columns={'apid':'bpid','ahap':'bhap','bpid':'apid','bhap':'ahap'}), on=['apid','ahap','bpid','bhap'], how='inner') # Paths that are exactly the same (required same length of the haplotype is not the same as same length of paths, thus we cannot use paths length)
    duplications = duplications.loc[duplications.merge(rem_paths, on=['apid','ahap','bpid','bhap'], how='left', indicator=True)['_merge'].values == "left_only", ['apid','ahap']].copy() # Paths that are part of a larger part
    rem_paths = rem_paths.loc[rem_paths['apid'] < rem_paths['bpid'], ['apid','ahap']].copy() # Only remove the lower pid (because we add merged path with new, larger pids at the end)
    if remove_all:
        rem_paths = pd.concat([rem_paths, duplications], ignore_index=True)
    rem_paths.drop_duplicates(inplace=True)
    rem_paths.rename(columns={'apid':'pid','ahap':'hap'}, inplace=True)
    scaffold_paths = RemoveHaplotypes(scaffold_paths, rem_paths, ploidy)
    scaffold_paths = CompressPaths(scaffold_paths, ploidy)
#
    return scaffold_paths

def PlacePathAInPathB(duplications, scaffold_paths, scaffold_graph, graph_ext, scaf_bridges, ploidy):
    includes = duplications.groupby(['ldid','rdid','apid','ahap','bpid','bhap'])['bpos'].agg(['min','max']).reset_index()
    scaf_len = scaffold_paths.groupby(['pid'])['pos'].max().reset_index()
    includes['blen'] = includes[['bpid']].rename(columns={'bpid':'pid'}).merge(scaf_len, on=['pid'], how='left')['pos'].values
    includes['reverse'] = includes[['ldid','rdid']].merge(duplications.groupby(['ldid','rdid'])['samedir'].first().reset_index(), on=['ldid','rdid'], how='left')['samedir'].values == False
    includes['tpid1'] = np.arange(0,3*len(includes),3)
    includes['tpid2'] = includes['tpid1'] + 1
    includes['tpid3'] = includes['tpid1'] + 2
    includes['group'] = np.arange(len(includes))
    # Merge middle section of path b (where path a is inserted) and path a
    test_paths = pd.concat([ includes[['apid','tpid1']].rename(columns={'apid':'pid','tpid1':'tpid'}), includes[['bpid','tpid2']].rename(columns={'bpid':'pid','tpid2':'tpid'}) ], ignore_index=True)
    test_paths = test_paths.merge(scaffold_paths, on=['pid'], how='left')
    test_paths['pid'] = test_paths['tpid']
    test_paths.drop(columns=['tpid'], inplace=True)
    test_paths.sort_values(['pid','pos'], inplace=True)
    test_paths['reverse'] = test_paths[['pid']].merge(includes[['tpid1','reverse']].rename(columns={'tpid1':'pid'}), on=['pid'], how='left')['reverse'].fillna(False).values.astype(bool)
    test_paths = ReverseScaffolds(test_paths, test_paths['reverse'], ploidy)
    test_paths.drop(columns=['reverse'], inplace=True)
    test_paths[['min','max']] = test_paths[['pid']].merge(includes[['min','max','tpid2']].rename(columns={'tpid2':'pid'}), on=['pid'], how='left')[['min','max']].values
    test_paths['min'] = test_paths['min'].fillna(0).astype(int)
    test_paths['max'] = test_paths['max'].fillna(scaf_len['pos'].max()).astype(int)
    test_paths = test_paths[(test_paths['min'] <= test_paths['pos']) & (test_paths['pos'] <= test_paths['max'])].drop(columns=['min','max'])
    test_paths['pos'] = test_paths.groupby(['pid'], sort=False).cumcount()
    ends = pd.concat([ includes[['tpid1','group']].rename(columns={'tpid1':'pid'}), includes[['tpid2','group']].rename(columns={'tpid2':'pid'}) ], ignore_index=True)
    if len(ends) == 0:
        includes['success'] = False
    else:
        test_paths, group_info = MergeHaplotypes(test_paths, scaf_bridges, ploidy, ends)
        pids = np.unique(test_paths['pid'])
        includes['success'] = (np.isin(includes['tpid1'], pids) == False) & (np.isin(includes['tpid2'], pids) == False) # If we still have the original pids they could not be merged
        group_info['new_pid'] = group_info[[f'pid{h}' for h in range(ploidy)]].max(axis=1) # We get here tpid2 from includes, which is the pid we want to assign to the now merged middle part
        test_paths['new_pid'] = test_paths[['pid']].merge(group_info[['pid','new_pid']], on=['pid'], how='left')['new_pid'].values
        test_paths = test_paths[np.isnan(test_paths['new_pid']) == False].copy()
        test_paths['pid'] = test_paths['new_pid'].astype(int)
        test_paths.drop(columns=['new_pid'], inplace=True)
        # Add part of path b before and after the merged region
        test_paths = [test_paths]
        spaths = scaffold_paths.merge(includes.loc[includes['success'] & (includes['min'] > 0), ['bpid','min','tpid1']].rename(columns={'bpid':'pid'}), on=['pid'], how='inner')
        spaths = spaths[spaths['pos'] <= spaths['min']].copy() # We need an overlap of 1 for the combining function to work properly
        spaths['pid'] = spaths['tpid1']
        spaths.drop(columns=['min','tpid1'], inplace=True)
        test_paths.append(spaths)
        epaths = scaffold_paths.merge(includes.loc[includes['success'] & (includes['max'] < includes['blen']), ['bpid','max','tpid3']].rename(columns={'bpid':'pid'}), on=['pid'], how='inner')
        epaths = epaths[epaths['pos'] >= epaths['max']].copy() # We need an overlap of 1 for the combining function to work properly
        epaths['pid'] = epaths['tpid3']
        epaths.drop(columns=['max','tpid3'], inplace=True)
        test_paths.append(epaths)
        test_paths = pd.concat(test_paths, ignore_index=True)
        test_paths.sort_values(['pid','pos'], inplace=True)
        test_paths.loc[test_paths['pos'] == 0, [f'dist{h}' for h in range(ploidy)]] = 0
        test_paths = TrimAlternativesConsistentWithMain(test_paths, ploidy)
        # Prepare additional information necessary for combining the individual parts in test_paths
        meta_parts = [ includes.loc[includes['success'] & (includes['min'] > 0), ['tpid1','tpid1']] ]
        meta_parts[-1].columns = ['pid','meta']
        meta_parts[-1]['pos'] = 0
        meta_parts.append( includes.loc[includes['success'], ['tpid2','tpid1']].rename(columns={'tpid2':'pid','tpid1':'meta'}) )
        meta_parts[-1]['pos'] = 1
        meta_parts.append( includes.loc[includes['success'] & (includes['max'] < includes['blen']), ['tpid3','tpid1']].rename(columns={'tpid3':'pid','tpid1':'meta'}) )
        meta_parts[-1]['pos'] = 2
        meta_parts = pd.concat(meta_parts, ignore_index=True)
        meta_parts.sort_values(['pid'], inplace=True)
        meta_parts.index = meta_parts['pid'].values
        meta_parts = meta_parts[(meta_parts['meta'] == meta_parts['meta'].shift(-1)) | (meta_parts['meta'] == meta_parts['meta'].shift(1))].copy() # If we do not have anything to combine we do not need to add it to meta_parts
        meta_parts['pos'] = meta_parts.groupby(['meta']).cumcount()
        meta_parts['reverse'] = False
        conns = meta_parts.drop(columns=['pos','reverse'])
        conns.rename(columns={'pid':'apid'}, inplace=True)
        conns['aside'] = 'r'
        conns['bpid'] = conns['apid'].shift(-1, fill_value=0)
        conns = conns[conns['meta'] == conns['meta'].shift(-1)].copy()
        conns['bside'] = 'l'
        conns.drop(columns=['meta'], inplace=True)
        conns = pd.concat([ conns, conns.rename(columns={'apid':'bpid','aside':'bside','bpid':'apid','bside':'aside'})], ignore_index=True)
        conns['aoverlap'] = 1
        conns['boverlap'] = 1
        scaf_len = test_paths.groupby(['pid'])['pos'].max().reset_index()
        test_paths = CombinePathAccordingToMetaParts(test_paths, meta_parts, conns, scaffold_graph, graph_ext, scaf_bridges, scaf_len, ploidy)
        pids = np.unique(test_paths['pid'])
        includes['success'] = (np.isin(includes[['tpid1','tpid2','tpid3']], pids).sum(axis=1) == 1)
        test_paths = test_paths.merge(pd.concat([ includes.loc[includes['success'], [f'tpid{i}','tpid3']].rename(columns={f'tpid{i}':'pid'}) for i in [1,2] ], ignore_index=True), on=['pid'], how='inner')
        test_paths['pid'] = test_paths['tpid3']
        test_paths.drop(columns=['tpid3'], inplace=True)
        test_paths.sort_values(['pid','pos'], inplace=True)
    includes = includes.loc[includes['success'], ['ldid','rdid','apid','bpid','tpid3']].rename(columns={'tpid3':'tpid'})
#
    return test_paths, includes

def PlaceUnambigouslyPlaceablePathsAsAlternativeHaplotypes(scaffold_paths, scaffold_graph, graph_ext, scaf_bridges, ploidy):
    duplications = GetDuplications(scaffold_paths, ploidy)
    # Only insert a maximum of one path a per path b per round to avoid conflicts between inserted path
    while len(duplications):
        # Get duplications that contain both path ends for side a
        scaf_len = scaffold_paths.groupby(['pid'])['pos'].max().reset_index()
        ends = duplications.groupby(['apid','bpid'])['apos'].agg(['min','max']).reset_index()
        ends['alen'] = ends[['apid']].rename(columns={'apid':'pid'}).merge(scaf_len, on=['pid'], how='left')['pos'].values
        ends = ends.loc[(ends['min'] == 0) & (ends['max'] == ends['alen']), ['apid','bpid','alen']].copy()
        duplications = duplications.merge(ends, on=['apid','bpid'], how='inner')
        # Only keep duplications, where path a is haploid
        nhaps = GetNumberOfHaplotypes(scaffold_paths, ploidy)
        duplications['anhaps'] = duplications[['apid']].rename(columns={'apid':'pid'}).merge(nhaps, on=['pid'], how='left')['nhaps'].values
        duplications = duplications[duplications['anhaps'] == 1].drop(columns=['anhaps'])
        # Check if they are on the same or opposite strand (alone it is useless, but it sets the requirements for the direction of change for the positions)
        duplications = AddStrandToDuplications(duplications, scaffold_paths, ploidy)
        duplications['samedir'] = duplications['astrand'] == duplications['bstrand']
        duplications.drop(columns=['astrand','bstrand'], inplace=True)
        # Extend the duplications with valid distances from both ends and only keep the ones that have both ends on the same path in the same direction
        duplications = SeparateDuplicationsByHaplotype(duplications, scaffold_paths, ploidy)
        ldups = ExtendDuplicationsFromEnd(duplications, scaffold_paths, 'l', scaf_len, ploidy)
        ldups.rename(columns={'did':'ldid'}, inplace=True)
        rdups = ExtendDuplicationsFromEnd(duplications, scaffold_paths, 'r', scaf_len, ploidy)
        rdups.rename(columns={'did':'rdid'}, inplace=True)
        mcols = ['apid','ahap','bpid','bhap']
        ldups = ldups.merge(rdups[mcols+['rdid']].drop_duplicates(), on=mcols, how='inner')
        rdups = rdups.merge(ldups[mcols+['ldid']].drop_duplicates(), on=mcols, how='inner')
        duplications = pd.concat([ldups,rdups], ignore_index=True)
        duplications.drop_duplicates(inplace=True)
        includes = duplications.groupby(['ldid','rdid'])['samedir'].agg(['min','max']).reset_index()
        includes = includes[includes['min'] == includes['max']].drop(columns=['min','max'])
        duplications = duplications.merge(includes, on=['ldid','rdid'], how='inner')
        # We cannot insert an alternative haplotype if the start and end map to the same position
        includes = duplications.groupby(['apid','ahap','bpid','bhap','ldid','rdid'])['bpos'].agg(['min','max']).reset_index()
        includes.sort_values(['apid','ahap','bpid','min','max','bhap','ldid','rdid'], inplace=True)
        includes = includes.groupby(['apid','ahap','bpid','min','max'], sort=False).first().reset_index() # We add it to a path not a single haplotype, so only take the lowest haplotype to have the corresponding duplications
        includes = includes[includes['min'] != includes['max']].copy()
        # Check that path b has a free haplotype to include path a
        includes = includes.loc[np.repeat(includes.index.values, includes['max']-includes['min'])].reset_index(drop=True)
        includes['pos'] = includes.groupby(['apid','ahap','bpid','min','max'], sort=False).cumcount() + includes['min'] + 1
        includes[[f'free{h}' for h in range(1,ploidy)]] = (includes[['bpid','pos']].rename(columns={'bpid':'pid'}).merge(scaffold_paths[['pid','pos']+[f'phase{h}' for h in range(1,ploidy)]], on=['pid','pos'], how='left')[[f'phase{h}' for h in range(1,ploidy)]].values < 0)
        includes = includes.groupby(['ldid','rdid','bpid','min','max'])[[f'free{h}' for h in range(1,ploidy)]].min().reset_index()
        includes = includes.merge(scaffold_paths.rename(columns={'pid':'bpid','pos':'min'}), on=['bpid','min'], how='left')
        for h in range(1,ploidy):
            includes[f'free{h}'] = includes[f'free{h}'] & ((includes[f'phase{h}'] < 0) | ((includes[f'scaf{h}'] == includes['scaf0']) & (includes[f'strand{h}'] == includes['strand0'])))
        includes['nfree'] = includes[[f'free{h}' for h in range(1,ploidy)]].sum(axis=1)
        includes = includes.loc[includes['nfree'] > 0, ['ldid','rdid','min','max']].copy()
        duplications = duplications.merge(includes, on=['ldid','rdid'], how='inner')
        # The positions for a and b must match samedir
        duplications['valid'] = (duplications['bpos'] != duplications['min']) | (duplications['samedir'] == (duplications['apos'] == 0)) # If samedir at lowest position of path b it must also be lowest for path a
        duplications['valid'] = duplications[['ldid','rdid']].merge(duplications.groupby(['ldid','rdid'])['valid'].min().reset_index(), on=['ldid','rdid'], how='left')['valid'].values
        duplications = duplications[duplications['valid']].copy()
        # Check that including path a as haplotype of path b does not violate scaffold_graph
        test_paths, includes = PlacePathAInPathB(duplications, scaffold_paths, scaffold_graph, graph_ext, scaf_bridges, ploidy)
        # Require that path a has a unique placement
        includes.sort_values(['apid'], inplace=True)
        includes = includes[(includes['apid'] != includes['apid'].shift(1)) & (includes['apid'] != includes['apid'].shift(-1))].copy()
        duplications = duplications.merge(includes[['ldid','rdid']], on=['ldid','rdid'], how='inner')
        # A path a cannot be at the same time a path b
        includes = includes[np.isin(includes['apid'],includes['bpid'].values) == False].copy()
        # Only one include per path b per round
        includes.sort_values(['bpid'], inplace=True)
        includes = includes[includes['bpid'] != includes['bpid'].shift(1)].copy()
        # Include those scaffolds
        if len(test_paths):
            test_paths = test_paths.merge(includes[['tpid','bpid']].rename(columns={'tpid':'pid'}), on=['pid'], how='inner')
            test_paths['pid'] = test_paths['bpid']
            test_paths.drop(columns=['bpid'], inplace=True)
            scaffold_paths = scaffold_paths[np.isin(scaffold_paths['pid'], np.concatenate([includes['apid'].values, includes['bpid'].values])) == False].copy()
            scaffold_paths = pd.concat([scaffold_paths, test_paths], ignore_index=True)
            scaffold_paths.sort_values(['pid','pos'], inplace=True)
        # Get the not yet used duplications and update them
        duplications = duplications[ duplications.merge(includes[['ldid','rdid']], on=['ldid','rdid'], how='left', indicator=True)['_merge'].values == "left_only"].copy()
        includes = duplications[['apid','bpid']].drop_duplicates()
        duplications = GetDuplications(scaffold_paths, ploidy)
        duplications = duplications.merge(includes, on=['apid','bpid'], how='inner')
#
    return scaffold_paths

def GetNextPositionInPathB(ends, scaffold_paths, ploidy):
    # Get next position (jumping over deletions)
    ends['opos'] = ends['mpos']
    update = np.repeat(True, len(ends))
    while np.sum(update):
        ends.loc[update, 'mpos'] += ends.loc[update, 'dir']
        ends = GetPositionFromPaths(ends, scaffold_paths, ploidy, 'next_scaf', 'scaf', 'bpid', 'mpos', 'bhap')
        update = ends['next_scaf'] < 0
#
    return ends

def GetFullNextPositionInPathB(ends, scaffold_paths, ploidy):
    ends = GetNextPositionInPathB(ends, scaffold_paths, ploidy)
    ends = GetPositionFromPaths(ends, scaffold_paths, ploidy, 'next_strand', 'strand', 'bpid', 'mpos', 'bhap')
    ends['next_strand'] = ends['next_strand'].fillna('')
    if np.sum(ends['dir'] == -1):
        ends.loc[ends['dir'] == -1, 'next_strand'] = np.where(ends.loc[ends['dir'] == -1, 'next_strand'] == '+', '-', '+')
    ends['dist_pos'] = np.where(ends['dir'] == 1, ends['mpos'], ends['opos'])
    ends = GetPositionFromPaths(ends, scaffold_paths, ploidy, 'next_dist', 'dist', 'bpid', 'dist_pos', 'bhap')
#
    return ends

def TrimAmbiguousOverlap(scaffold_paths, scaffold_graph, ploidy):
    ends = GetDuplicatedPathEnds(scaffold_paths, ploidy)
    # Make sure the duplications conflicting overlaps (coming from the same side) not connectable overlaps
    ends = ends[ends['samedir'] == (ends['aside'] == ends['bside'])].copy()
    # Get the maximum overlap for each path
    ends = pd.concat([ ends[ends['aside'] == 'l'].groupby(['apid','aside'])['amax'].max().reset_index(name='apos'), ends[ends['aside'] == 'r'].groupby(['apid','aside'])['amin'].min().reset_index(name='apos') ], ignore_index=True)
    ends['dir'] = np.where(ends['aside'] == 'l', -1, +1)
    ends.drop(columns=['aside'], inplace=True)
    # Separate the scaffolds by haplotype for following checks
    nhaps = GetNumberOfHaplotypes(scaffold_paths, ploidy)
    ends = ends.loc[np.repeat(ends.index.values, ends[['apid']].rename(columns={'apid':'pid'}).merge(nhaps, on=['pid'], how='left')['nhaps'].values)].reset_index(drop=True)
    ends['ahap'] = ends.groupby(['dir','apid'], sort=False).cumcount()
    # Check how much the scaffold_graph extends the first unambiguous position into the overlap (unambiguous support)
    ends['apos'] -= ends['dir'] # First go in the opposite direction to get first unambiguous position
    while True:
        ends = GetPositionFromPaths(ends, scaffold_paths, ploidy, 'from', 'scaf', 'apid', 'apos', 'ahap')
        if np.sum(ends['from'] < 0) == 0:
            break
        else:
            ends.loc[ends['from'] < 0, 'apos'] -= ends.loc[ends['from'] < 0, 'dir'] # Start after deletions
    ends = GetPositionFromPaths(ends, scaffold_paths, ploidy, 'from_side', 'strand', 'apid', 'apos', 'ahap')
    ends['from_side'] = np.where((ends['from_side'] == '+') == (ends['dir'] == -1), 'l', 'r')
    ends.rename(columns={'apid':'bpid','ahap':'bhap'}, inplace=True)
    ends['mpos'] = ends['apos']
    ends = GetFullNextPositionInPathB(ends, scaffold_paths, ploidy)
    pairs = ends[['from','from_side','next_scaf','next_strand','next_dist']].reset_index()
    pairs.rename(columns={'index':'dindex','next_scaf':'scaf1','next_strand':'strand1','next_dist':'dist1'}, inplace=True)
    pairs = pairs.merge(scaffold_graph[['from','from_side','scaf1','strand1','dist1']].reset_index().rename(columns={'index':'sindex'}), on=['from','from_side','scaf1','strand1','dist1'], how='inner')
    pairs.drop(columns=['from','from_side','scaf1','strand1','dist1'], inplace=True)
    s = 2
    while len(pairs):
        # If scaffold_graph extends further than the current position, we can store it
        pairs = pairs[scaffold_graph.loc[pairs['sindex'].values, 'length'].values > s].copy()
        if len(pairs):
            valid = np.unique(pairs['dindex'].values)
            ends.loc[valid, 'apos'] = ends.loc[valid, 'mpos']
            # Check next position
            ends = GetFullNextPositionInPathB(ends, scaffold_paths, ploidy)
            pairs = pairs[ (scaffold_graph.loc[pairs['sindex'].values, [f'scaf{s}',f'strand{s}',f'dist{s}']].values == ends.loc[pairs['dindex'].values, ['next_scaf','next_strand','next_dist']].values).all(axis=1) ].copy()
            s += 1
    ends.rename(columns={'bpid':'pid'}, inplace=True)
    ends = ends.groupby(['pid','dir'])['apos'].agg(['min','max']).reset_index()
    ends['pos'] = np.where(ends['dir'] == 1, ends['max'], ends['min'])
    ends.drop(columns=['min','max'], inplace=True)
    # In case the ambiguous overlap from both sides overlaps take the middle for the split
    ends['new_pos'] = ends['pos'] + (ends['pos'].shift(1, fill_value=0) - ends['pos'])//2
    ends.loc[(ends['pid'] != ends['pid'].shift(1)) | (ends['pos'] >= ends['pos'].shift(1)), 'new_pos'] = -1
    ends['new_pos'] = np.where( (ends['pid'] == ends['pid'].shift(-1)) & (ends['pos'] > ends['pos'].shift(-1)), ends['new_pos'].shift(-1, fill_value=0) + 1, ends['new_pos'] )
    ends.loc[ends['new_pos'] >= 0, 'pos'] = ends.loc[ends['new_pos'] >= 0, 'new_pos']
    # Separate the ambiguous overlap on both sides into their own paths and remove duplicates
    ends.rename(columns={'new_pos':'new_pid'}, inplace=True)
    ends['new_pid'] = np.arange(len(ends)) + 1 + scaffold_paths['pid'].max()
    max_len = scaffold_paths.groupby(['pid'])['pos'].max().max()
    scaffold_paths[['new_pid','end']] = scaffold_paths[['pid']].merge(ends.loc[ends['dir'] == -1, ['pid','new_pid','pos']], on=['pid'], how='left')[['new_pid','pos']].fillna(0).values.astype(int)
    scaffold_paths.loc[scaffold_paths['pos'] < scaffold_paths['end'], 'pid'] = scaffold_paths.loc[scaffold_paths['pos'] < scaffold_paths['end'], 'new_pid']
    scaffold_paths[['new_pid','end']] = scaffold_paths[['pid']].merge(ends.loc[ends['dir'] == 1, ['pid','new_pid','pos']], on=['pid'], how='left')[['new_pid','pos']].fillna(max_len).values.astype(int)
    scaffold_paths.loc[scaffold_paths['pos'] > scaffold_paths['end'], 'pid'] = scaffold_paths.loc[scaffold_paths['pos'] > scaffold_paths['end'], 'new_pid']
    scaffold_paths.drop(columns=['new_pid','end'], inplace=True)
    scaffold_paths.sort_values(['pid','pos'], inplace=True)
    scaffold_paths['pos'] = scaffold_paths.groupby(['pid']).cumcount()
    scaffold_paths = SetDistanceAtFirstPositionToZero(scaffold_paths, ploidy)
    scaffold_paths = RemoveDuplicates(scaffold_paths, True, ploidy)
#
    return scaffold_paths

def TrimCircularPaths(scaffold_paths, ploidy):
    # Find circular paths (Check where the first scaffold in paths is duplicated and see if the duplications reaches the path end from there)
    circular = [ pd.concat( [scaffold_paths.loc[(scaffold_paths['pos'] > 0) & (scaffold_paths[f'scaf{h1}'] >= 0), ['pid','pos',f'scaf{h1}',f'strand{h1}']].rename(columns={f'scaf{h1}':'scaf',f'strand{h1}':'strand'}).merge(scaffold_paths.loc[(scaffold_paths['pos'] == 0) & (scaffold_paths[f'scaf{h2}'] >= 0), ['pid',f'scaf{h2}',f'strand{h2}']].rename(columns={f'scaf{h2}':'scaf',f'strand{h2}':'strand'}), on=['pid','scaf','strand'], how='inner') for h1 in range(ploidy)], ignore_index=True ).drop_duplicates() for h2 in range(ploidy) ]
    circular = pd.concat(circular, ignore_index=True).groupby(['pid','pos']).size().reset_index(name='nhaps')
    circular['start_pos'] = circular['pos']
    scaf_len = scaffold_paths.groupby(['pid'])['pos'].max().reset_index()
    circular['len'] = circular[['pid']].merge(scaf_len, on=['pid'], how='left')['pos'].values
    circular['circular'] = False
    p = 0
    while True:
        # All haplotypes need to be duplicated
        circular = circular[circular['circular'] | (circular['nhaps'] == circular[['pid']].merge((scaffold_paths.loc[scaffold_paths['pos'] == p, ['pid']+[f'phase{h}' for h in range(ploidy)]].set_index(['pid']) >= 0).sum(axis=1).reset_index(name='nhaps'), on=['pid'], how='left')['nhaps'].values)].copy()
        # If we reach the end of path it is a valid circular duplication
        circular.loc[circular['pos'] == circular['len'], 'circular'] = True
        if np.sum(circular['circular'] == False):
            # Check next position
            circular['pos'] += 1
            p += 1
            circular['nhaps'] = 0
            for h2 in range(ploidy):
                cur_test = circular.loc[circular['circular'] == False, ['pid','pos']].reset_index().merge(scaffold_paths.loc[(scaffold_paths['pos'] == p) & (scaffold_paths[f'scaf{h2}'] >= 0), ['pid',f'scaf{h2}',f'strand{h2}',f'dist{h2}']].rename(columns={f'scaf{h2}':'scaf',f'strand{h2}':'strand',f'dist{h2}':'dist'}), on=['pid'], how='inner')
                circular.loc[pd.concat([cur_test.merge(scaffold_paths[['pid','pos',f'scaf{h1}',f'strand{h1}',f'dist{h1}']].rename(columns={f'scaf{h1}':'scaf',f'strand{h1}':'strand',f'dist{h1}':'dist'}), on=['pid','pos','scaf','strand','dist'], how='inner') for h1 in range(ploidy)], ignore_index=True).drop_duplicates()['index'].values, 'nhaps'] += 1
        else:
            break
    # Remove longest circular duplication for each path
    circular = circular.groupby(['pid'])['start_pos'].min().reset_index()
    scaffold_paths['trim'] = scaffold_paths[['pid']].merge(circular, on=['pid'], how='left')['start_pos'].fillna(-1).values.astype(int)
    scaffold_paths = scaffold_paths[(scaffold_paths['pos'] < scaffold_paths['trim']) | (scaffold_paths['trim'] == -1)].drop(columns=['trim'])
    # Check if the remainder is duplicated somewhere else
    scaffold_paths = RemoveDuplicates(scaffold_paths, True, ploidy)
#
    return scaffold_paths

def TraverseScaffoldingGraph(scaffolds, scaffold_graph, graph_ext, scaf_bridges, org_scaf_conns, ploidy, max_loop_units):
    # Get phased haplotypes
    knots = UnravelKnots(scaffold_graph, scaffolds)
    scaffold_paths = FollowUniquePathsThroughGraph(knots, scaffold_graph)
    scaffold_paths, handled_scaf_conns = AddPathThroughLoops(scaffold_paths, scaffold_graph, scaf_bridges, org_scaf_conns, ploidy, max_loop_units)
    scaffold_paths, handled_scaf_conns = AddPathThroughInvertedRepeats(scaffold_paths, handled_scaf_conns, scaffold_graph, scaf_bridges, ploidy)
    scaffold_paths = AddUntraversedConnectedPaths(scaffold_paths, knots, scaffold_graph, handled_scaf_conns)
    scaffold_paths = AddUnconnectedPaths(scaffold_paths, scaffolds, scaffold_graph)
#
    # Turn them into full path with ploidy
    scaffold_paths.insert(2, 'phase0', scaffold_paths['pid'].values+1) #+1, because phases must be larger than zero to be able to be positive and negative (negative means identical to main paths at that position)
    for h in range(1,ploidy):
        scaffold_paths[f'phase{h}'] = -scaffold_paths['phase0'].values
        scaffold_paths[f'scaf{h}'] = -1
        scaffold_paths[f'strand{h}'] = ''
        scaffold_paths[f'dist{h}'] = 0
    CheckScaffoldPathsConsistency(scaffold_paths)
    CheckIfScaffoldPathsFollowsValidBridges(scaffold_paths, scaf_bridges, ploidy)
#
    # Combine paths as much as possible
    print("Start")
    print(len(np.unique(scaffold_paths['pid'].values)))
    for i in range(3):
        old_nscaf = 0
        n = 1
        while old_nscaf != len(np.unique(scaffold_paths['pid'].values)):
            time0 = clock()
            print(f"Iteration {n}")
            n+=1
            old_nscaf = len(np.unique(scaffold_paths['pid'].values))
             # First Merge then Combine to not accidentially merge a haplotype, where the other haplotype of the paths is not compatible and thus joining wrong paths
            scaffold_paths = MergeHaplotypes(scaffold_paths, scaf_bridges, ploidy)
            time1 = clock()
            print(str(timedelta(seconds=time1-time0)), len(np.unique(scaffold_paths['pid'].values)))
            scaffold_paths = CombinePathOnUniqueOverlap(scaffold_paths, scaffold_graph, graph_ext, scaf_bridges, ploidy)
            print(str(timedelta(seconds=clock()-time1)), len(np.unique(scaffold_paths['pid'].values)))
            CheckScaffoldPathsConsistency(scaffold_paths)
            CheckIfScaffoldPathsFollowsValidBridges(scaffold_paths, scaf_bridges, ploidy)
        if i==0:
            print("RemoveDuplicates")
            scaffold_paths = RemoveDuplicates(scaffold_paths, True, ploidy)
        elif i==1:
            print("PlaceUnambigouslyPlaceables")
            scaffold_paths = PlaceUnambigouslyPlaceablePathsAsAlternativeHaplotypes(scaffold_paths, scaffold_graph, graph_ext, scaf_bridges, ploidy)
        if i != 2:
            print(len(np.unique(scaffold_paths['pid'].values)))
            CheckScaffoldPathsConsistency(scaffold_paths)
            CheckIfScaffoldPathsFollowsValidBridges(scaffold_paths, scaf_bridges, ploidy)
    print("TrimAmbiguousOverlap")
    scaffold_paths = TrimAmbiguousOverlap(scaffold_paths, scaffold_graph, ploidy)
    print(len(np.unique(scaffold_paths['pid'].values)))
    print("TrimCircularPaths")
    scaffold_paths = TrimCircularPaths(scaffold_paths, ploidy)
    print(len(np.unique(scaffold_paths['pid'].values)))
    CheckScaffoldPathsConsistency(scaffold_paths)
    CheckIfScaffoldPathsFollowsValidBridges(scaffold_paths, scaf_bridges, ploidy)

    return scaffold_paths

def AssignNewPhases(scaffold_paths, phase_change_in, ploidy):
    phase_change = phase_change_in.copy()
    phase_change['new_phase'] = phase_change['from_phase']
    while True:
        phase_change['gphase'] = phase_change[['new_phase']].rename(columns={'new_phase':'to_phase'}).merge(phase_change[['to_phase','new_phase']], on=['to_phase'], how='left')['new_phase'].values
        if np.sum(np.isnan(phase_change['gphase']) == False):
            phase_change.loc[np.isnan(phase_change['gphase']) == False, 'new_phase'] = phase_change.loc[np.isnan(phase_change['gphase']) == False, 'gphase'].astype(int)
        else:
            break
    phase_change.drop(columns=['from_phase','gphase'], inplace=True)
    phase_change = pd.concat([phase_change, phase_change*-1], ignore_index=True)
    for h in range(ploidy):
        scaffold_paths['new_phase'] = scaffold_paths[[f'phase{h}']].rename(columns={f'phase{h}':'to_phase'}).merge(phase_change, on=['to_phase'], how='left')['new_phase'].values
        scaffold_paths.loc[np.isnan(scaffold_paths['new_phase']) == False, f'phase{h}'] = scaffold_paths.loc[np.isnan(scaffold_paths['new_phase']) == False, 'new_phase'].astype(int).values
    scaffold_paths.drop(columns=['new_phase'], inplace=True)
#
    return scaffold_paths

def PhaseScaffoldsWithScafBridges(scaffold_paths, scaf_bridges, ploidy):
    ## Combine phases where scaf_bridges leaves only one option
    # First get all bridges combining phases (ignoring deletions)
    test_bridges = []
    for h2 in range(ploidy):
        cur = scaffold_paths[['pid','pos',f'phase{h2}',f'scaf{h2}',f'strand{h2}',f'dist{h2}']].rename(columns={'pos':'to_pos',f'phase{h2}':'to_phase',f'scaf{h2}':'to',f'strand{h2}':'to_side',f'dist{h2}':'mean_dist'})
        cur['to_side'] = np.where(cur['to_side'] == '+', 'l', 'r')
        cur['to_hap'] = h2
        cur['update'] = True
        cur['from_phase'] = 0
        cur['from'] = -1
        cur['from_side'] = ''
        cur['from_pos'] = -1
        s = 1
        while np.sum(cur['update']):
            cur['deletion'] = False
            for h1 in range(ploidy):
                cur['from_hap'] = h1
                cur['from_phase'] = np.where(scaffold_paths['pid'] == scaffold_paths['pid'].shift(s), scaffold_paths[f'phase{h1}'].shift(s, fill_value=0), 0)
                cur['from'] = np.where(cur['from_phase'] > 0, scaffold_paths[f'scaf{h1}'].shift(s, fill_value=-2), np.where(cur['from_phase'] == 0, -2, scaffold_paths['scaf0'].shift(s, fill_value=-2)))
                cur['from_side'] = np.where(scaffold_paths[f'strand{h1}'].shift(s, fill_value='') == '+', 'r', 'l')
                cur['from_pos'] = cur['to_pos']-s
                cur['deletion'] = cur['deletion'] | (cur['from'] == -1)
                test_bridges.append( cur.loc[cur['update'] & (cur['from_phase'] > 0) & (cur['to_phase'] > 0) & (cur['to'] >= 0) & (cur['from'] >= 0) & (cur['from_phase'] != cur['to_phase']), ['pid','from_pos','from_hap','from_phase','from','from_side','to_pos','to_hap','to_phase','to','to_side','mean_dist']].copy() )
            s += 1
            cur['update'] = cur['deletion']
    test_bridges = pd.concat(test_bridges, ignore_index=True)
    # Filter to have only valid bridges
    test_bridges = test_bridges.merge(scaf_bridges[['from','from_side','to','to_side','mean_dist']], on=['from','from_side','to','to_side','mean_dist'], how='inner')
    test_bridges.sort_values(['pid','from_pos','from_hap','to_pos','to_hap'], inplace=True)
    # Combine phases without alternatives
    alt_count = test_bridges.groupby(['pid','from_pos','from_hap'], sort=False).size().values
    test_bridges['from_alts'] = np.repeat(alt_count, alt_count)
    test_bridges['to_alts'] = test_bridges[['pid','to_pos','to_hap']].merge(test_bridges.groupby(['pid','to_pos','to_hap']).size().reset_index(name='alts'), on=['pid','to_pos','to_hap'], how='left')['alts'].values
    test_bridges = test_bridges.loc[(test_bridges['from_alts'] == 1) & (test_bridges['to_alts'] == 1), ['from_phase','to_phase']].copy()
    scaffold_paths = AssignNewPhases(scaffold_paths, test_bridges, ploidy)
    # Handle deletions by assigning them to the following phase if that phase cannot connect to an alternative to going through the deletion
    deletions = []
    for h in range(ploidy):
        scaffold_paths['deletion'] = np.where(scaffold_paths[f'phase{h}'] > 0, scaffold_paths[f'scaf{h}'], scaffold_paths['scaf0']) < 0
        scaffold_paths['add'] = (scaffold_paths['deletion'] == False) & (scaffold_paths[f'phase{h}'] > 0)
        s = 1
        while True:
            scaffold_paths['add'] = scaffold_paths['add'] & scaffold_paths['deletion'].shift(s) & (scaffold_paths['pid'] == scaffold_paths['pid'].shift(s))
            scaffold_paths['from_phase'] = np.abs(scaffold_paths[f'phase{h}'].shift(s, fill_value=0)) # We need the phase of the deletion later, not the one of the checked connection, since we are looking for no valid connections
            if np.sum(scaffold_paths['add']):
                for h2 in range(ploidy):
                    if h2 != h:
                        for s2 in range(1,s+1):
                            scaffold_paths['from'] = scaffold_paths[f'scaf{h2}'].shift(s2, fill_value=-1)
                            scaffold_paths['from_side'] = scaffold_paths[f'strand{h2}'].shift(s2, fill_value='')
                            deletions.append( scaffold_paths.loc[scaffold_paths['add'] & (scaffold_paths[f'phase{h2}'].shift(s2) > 0) & (scaffold_paths['from'] >= 0), ['pid','pos','from_phase','from','from_side',f'phase{h}',f'scaf{h}',f'strand{h}',f'dist{h}']].rename(columns={f'phase{h}':'to_phase',f'scaf{h}':'to',f'strand{h}':'to_side',f'dist{h}':'mean_dist'}) )
                s += 1
            else:
                break
    scaffold_paths.drop(columns=['deletion','add','from_phase','from','from_side'], inplace=True)
    deletions = pd.concat(deletions, ignore_index=True)
    deletions['from_side'] = np.where(deletions['from_side'] == '+', 'r', 'l')
    deletions['to_side'] = np.where(deletions['to_side'] == '+', 'l', 'r')
    deletions['valid'] = deletions[['from','from_side','to','to_side','mean_dist']].merge(scaf_bridges[['from','from_side','to','to_side','mean_dist']], on=['from','from_side','to','to_side','mean_dist'], how='left', indicator=True)['_merge'].values == "both"
    deletions.sort_values(['pid','pos','from_phase'], inplace=True)
    valid = deletions.groupby(['pid','pos','from_phase'], sort=False)['valid'].agg(['max','size'])
    deletions['valid'] = np.repeat(valid['max'].values, valid['size'].values)
    deletions = deletions.loc[deletions['valid'] == False, ['from_phase','to_phase']].drop_duplicates()
    deletions.rename(columns={'from_phase':'to_phase','to_phase':'from_phase'}, inplace=True) # We need to assign 'to_phase' to 'from_phase', because from_phase is unique, but to_phase not necessarily
    scaffold_paths = AssignNewPhases(scaffold_paths, deletions, ploidy)
    # Combine adjacent deletions if the previous phase has no connection into the deletion (to a position of the deletion that is not the first)
    deletions = []
    for h in range(ploidy):
        scaffold_paths['deletion'] = np.where(scaffold_paths[f'phase{h}'] > 0, scaffold_paths[f'scaf{h}'], scaffold_paths['scaf0']) < 0
        scaffold_paths['add'] = (scaffold_paths['deletion'] == False) & (scaffold_paths['deletion'].shift(-1) == True) & (scaffold_paths[f'phase{h}'] > 0)
        scaffold_paths['from_phase'] = scaffold_paths[f'phase{h}'].shift(-1, fill_value=0)
        s = 2
        while True:
            scaffold_paths['add'] = scaffold_paths['add'] & scaffold_paths['deletion'].shift(-s) & (scaffold_paths['pid'] == scaffold_paths['pid'].shift(-s))
            scaffold_paths['to_phase'] = np.abs(scaffold_paths[f'phase{h}'].shift(-s, fill_value=0)) # We need the phase of the deletion later, not the one of the checked connection, since we are looking for no valid connections
            if np.sum(scaffold_paths['add']):
                for h2 in range(ploidy):
                    if h2 != h:
                        for s2 in range(2,s+1):
                            scaffold_paths['to'] = scaffold_paths[f'scaf{h2}'].shift(-s2, fill_value=-1)
                            scaffold_paths['to_side'] = scaffold_paths[f'strand{h2}'].shift(-s2, fill_value='')
                            scaffold_paths['mean_dist'] = scaffold_paths[f'dist{h2}'].shift(-s2, fill_value=0)
                            deletions.append( scaffold_paths.loc[scaffold_paths['add'] & (scaffold_paths[f'phase{h2}'].shift(-s2) > 0) & (scaffold_paths['to'] >= 0), ['pid','pos','from_phase',f'scaf{h}',f'strand{h}','to_phase','to','to_side','mean_dist']].rename(columns={f'scaf{h}':'from',f'strand{h}':'from_side'}) )
                s += 1
            else:
                break
    scaffold_paths.drop(columns=['deletion','add','from_phase','to_phase','to','to_side','mean_dist'], inplace=True)
    deletions = pd.concat(deletions, ignore_index=True)
    deletions = deletions[deletions['from_phase'] != deletions['to_phase']].copy()
    deletions['from_side'] = np.where(deletions['from_side'] == '+', 'r', 'l')
    deletions['to_side'] = np.where(deletions['to_side'] == '+', 'l', 'r')
    deletions['valid'] = deletions[['from','from_side','to','to_side','mean_dist']].merge(scaf_bridges[['from','from_side','to','to_side','mean_dist']], on=['from','from_side','to','to_side','mean_dist'], how='left', indicator=True)['_merge'].values == "both"
    deletions.sort_values(['pid','pos','from_phase','to_phase'], inplace=True)
    valid = deletions.groupby(['pid','pos','from_phase','to_phase'], sort=False)['valid'].agg(['max','size'])
    deletions['valid'] = np.repeat(valid['max'].values, valid['size'].values)
    deletions = deletions.loc[deletions['valid'] == False, ['from_phase','to_phase']].drop_duplicates()
    scaffold_paths = AssignNewPhases(scaffold_paths, deletions, ploidy)
#
    return scaffold_paths

def FindNextValidPositionOnBothSidesOfTestConnection(test_conns, scaffold_paths, ploidy):
    for s1 in ['l','r']:
        for s2 in ['l','r']:
            test_conns[f'{s1}{s2}spos'] = test_conns['pos']
            while True:
                test_conns = GetPositionFromPaths(test_conns, scaffold_paths, ploidy, f'{s1}{s2}scaf', 'scaf', 'pid', f'{s1}{s2}spos', f'{s1}hap')
                if np.sum(test_conns[f'{s1}{s2}scaf'] == -1):
                    test_conns.loc[test_conns[f'{s1}{s2}scaf'] == -1, f'{s1}{s2}spos'] += -1 if s2 == 'l' else 1
                else:
                    break
            # Check if we have another scaffold between the first valid one and the end of the scaffold
            for d in [-1,+1]:
                test_conns['test_pos'] = test_conns[f'{s1}{s2}spos'] + d
                while True:
                    test_conns = GetPositionFromPaths(test_conns, scaffold_paths, ploidy, 'test', 'scaf', 'pid', 'test_pos', f'{s1}hap')
                    if np.sum(test_conns['test'] == -1):
                        test_conns.loc[test_conns['test'] == -1, 'test_pos'] += d
                    else:
                        break
                test_conns.loc[np.isnan(test_conns['test']), f'{s1}{s2}scaf'] = np.nan
    test_conns[['invalid','lspos','rspos']] = [True,-1,-1]
    for s1 in ['r','l']:
        for s2 in ['r','l']:
            valid = (test_conns[f'l{s1}scaf'] == test_conns[f'r{s2}scaf'])
            test_conns.loc[valid, 'invalid'] = False
            test_conns.loc[valid, ['lspos','rspos']] = test_conns.loc[valid, [f'l{s1}spos',f'r{s2}spos']].values
    test_conns.drop(columns=['llspos','llscaf','test_pos','test','lrspos','lrscaf','rlspos','rlscaf','rrspos','rrscaf'], inplace=True)
#
    return test_conns

def PhaseScaffoldsWithScaffoldGraph(scaffold_paths, scaffold_graph, graph_ext, ploidy):
    ## Combine phases where scaffold_graph leaves only one option
    # Get all remaining phase_breaks
    phase_breaks = []
    for h in range(ploidy):
        phase_breaks.append( scaffold_paths.loc[(scaffold_paths[f'phase{h}'] != scaffold_paths[f'phase{h}'].shift(1)) & (scaffold_paths['pid'] == scaffold_paths['pid'].shift(1)), ['pid','pos']].rename(columns={'pos':'rpos'}) )
        phase_breaks[-1]['hap'] = h
    phase_breaks = pd.concat(phase_breaks, ignore_index=True)
    phase_breaks['lpos'] = phase_breaks['rpos'] - 1
    test_conns = []
    for h in range(ploidy):
        test_conns.append(phase_breaks[['pid','lpos','rpos','hap']].rename(columns={'hap':'lhap'}))
        test_conns[-1]['rhap'] = h
        test_conns.append(phase_breaks[['pid','lpos','rpos','hap']].rename(columns={'hap':'rhap'}))
        test_conns[-1]['lhap'] = h
    test_conns = pd.concat(test_conns, ignore_index=True)
    test_conns.drop_duplicates(inplace=True)
    # Filter out phases breaks where we cannot create an overlap between left and right side, which is not at the path end (if we create an overlap at the paths end, we do not have two sides to check a connection anymore)
    test_conns['pos'] = test_conns['lpos']
    test_conns = FindNextValidPositionOnBothSidesOfTestConnection(test_conns, scaffold_paths, ploidy)
    test_conns.loc[test_conns['invalid'], 'pos'] = test_conns.loc[test_conns['invalid'], 'rpos']
    test_conns = FindNextValidPositionOnBothSidesOfTestConnection(test_conns, scaffold_paths, ploidy)
    phase_breaks = phase_breaks[ phase_breaks[['pid','lpos','rpos']].merge(test_conns.loc[test_conns['invalid'], ['pid','lpos','rpos']].drop_duplicates(), on=['pid','lpos','rpos'], how='left', indicator=True)['_merge'].values == "left_only" ].copy()
    test_conns.drop(columns=['invalid'], inplace=True)
    test_conns = test_conns.merge(phase_breaks[['pid','lpos','rpos']].drop_duplicates(), on=['pid','lpos','rpos'], how='inner')
    # Test connections
    test_conns.sort_values(['pid','pos','lspos','rspos'], inplace=True, ignore_index=True)
    test_conns['apid'] = (((test_conns['pid'] != test_conns['pid'].shift(1)) | (test_conns['pos'] != test_conns['pos'].shift(1)) |
                           (test_conns['lspos'] != test_conns['lspos'].shift(1)) | (test_conns['rspos'] != test_conns['rspos'].shift(1))).cumsum() - 1 ) * 2
    test_conns['bpid'] = test_conns['apid']+1
    test_paths = scaffold_paths.merge(test_conns[['pid','pos','apid','bpid']].drop_duplicates().rename(columns={'pos':'split_pos'}), on=['pid'], how='inner')
    test_paths = test_paths.loc[ np.repeat(test_paths.index.values, 1+(test_paths['pos'] == test_paths['split_pos']).values) ].reset_index()
    test_paths['pid'] = np.where((test_paths['pos'] < test_paths['split_pos']) | (test_paths['index'] == test_paths['index'].shift(-1)), test_paths['apid'], test_paths['bpid'])
    test_paths.drop(columns=['index','apid','bpid'], inplace=True)
    test_paths.sort_values(['pid','pos'], inplace=True, ignore_index=True)
    switch_pos = pd.concat( [test_conns.loc[test_conns['lspos'] != test_conns['pos'], ['apid','lspos','pid','lhap']].rename(columns={'apid':'pid','lspos':'switch_pos','pid':'opid','lhap':'hap'}), 
                             test_conns.loc[test_conns['rspos'] != test_conns['pos'], ['bpid','rspos','pid','rhap']].rename(columns={'bpid':'pid','rspos':'switch_pos','pid':'opid','rhap':'hap'})], ignore_index=True).drop_duplicates()
    if len(switch_pos):
        test_paths[['switch_pos','opid','hap']] = test_paths[['pid']].merge(switch_pos, on=['pid'], how='left')[['switch_pos','opid','hap']].fillna(-1).astype(int).values
        for h in range(ploidy):
            if np.sum(test_paths['hap'] == h):
                # Replace deletion with scaffold next to deletion (we need to take it from scaffold_paths, because it might be on the other side of the split)
                cur = (test_paths['hap'] == h) & (test_paths['pos'] == test_paths['split_pos'])
                cols = [f'scaf{h}',f'strand{h}',f'dist{h}']
                test_paths.loc[cur, cols] = test_paths.loc[cur, ['opid','switch_pos']].rename(columns={'opid':'pid','switch_pos':'pos'}).merge(scaffold_paths[['pid','pos']+cols], on=['pid','pos'], how='left')[cols].values
                # Make sure we do not affect the other haplotypes by changing the main path
                cur = (test_paths['hap'] == h) & (test_paths['pos'] == test_paths['switch_pos'])
                if h == 0:
                    for h1 in range(1,ploidy):
                        test_paths.loc[cur & (test_paths[f'phase{h1}'] < 0), [f'scaf{h1}',f'strand{h1}',f'dist{h1}']] = test_paths.loc[cur & (test_paths[f'phase{h1}'] < 0), ['scaf0','strand0','dist0']].values
                # Delete the scaffold we shifted
                test_paths.loc[cur, cols] = [-1,'',0]
        test_paths.drop(columns=['switch_pos','opid','hap'], inplace=True)
    test_paths.drop(columns=['split_pos'], inplace=True)  
    test_paths['pos'] = test_paths.groupby(['pid'], sort=False).cumcount()
    test_paths = SetDistanceAtFirstPositionToZero(test_paths, ploidy)
    test_conns.rename(columns={'lhap':'ahap','rhap':'bhap'}, inplace=True)
    test_conns['aside'] = 'r'
    test_conns['bside'] = 'l'
    test_conns['amin'] = test_conns['pos']
    test_conns['amax'] = test_conns['pos']
    test_conns['alen'] = test_conns['amax']
    test_conns['bmin'] = 0
    test_conns['bmax'] = 0
    scaf_len = scaffold_paths.groupby(['pid'])['pos'].max().reset_index()
    test_conns['blen'] = test_conns[['pid']].merge(scaf_len, on=['pid'], how='left')['pos'].values - test_conns['pos']
    ends = test_conns.drop(columns=['pid','lpos','rpos','pos','lspos','rspos']).drop_duplicates()
    cols = ['pid','hap','side','min','max','len']
    ends = pd.concat([ends, ends.rename(columns={**{f'a{n}':f'b{n}' for n in cols},**{f'b{n}':f'a{n}' for n in cols}})], ignore_index=True)
    ends = FilterInvalidConnections(ends, test_paths, scaffold_graph, graph_ext, ploidy)
    ends = ends[ends['valid_path'] == 'ab'].drop(columns=['valid_path'])
    if len(ends) == 0:
        path_errors = phase_breaks
        phase_breaks = []
    else:
        test_conns = test_conns.merge(ends, on=list(ends.columns), how='inner')
        test_conns = test_conns[['pid','lpos','rpos','ahap','bhap']].rename(columns={'ahap':'lhap','bhap':'rhap'})
        # Check if we have evidence for errors in paths (same haplotype is not a valid connection)
        phase_breaks['rhap'] = phase_breaks['hap']
        path_errors = phase_breaks[ phase_breaks.rename(columns={'hap':'lhap'}).merge(test_conns, on=['pid','lpos','rpos','lhap','rhap'], how='left', indicator=True)['_merge'].values == "left_only" ].copy()
    if len(path_errors):
        print("Warning: Found scaffold_paths that violate scaffold_graph.")
        print(path_errors)
    # Combine phases where we do not have alternative options
    if len(phase_breaks):
        phase_breaks = phase_breaks.merge(test_conns.rename(columns={'lhap':'hap'}), on=['pid','lpos','rpos','hap','rhap'], how='inner').drop(columns=['rhap'])
    if len(phase_breaks):
        for s in ['l','r']:
            cols = ['pid','lpos','rpos','hap']
            phase_breaks[f'{s}alts'] = phase_breaks[cols].merge(test_conns.rename(columns={f'{s}hap':'hap'}).groupby(cols).size().reset_index(name='alts'), on=cols, how='left')['alts'].values
        phase_breaks = phase_breaks[(phase_breaks['lalts'] == 1) & (phase_breaks['ralts'] == 1)].copy()
        phase_breaks[['from_phase','to_phase']] = 1
        for h in range(ploidy):
            phase_breaks.loc[phase_breaks['hap'] == h, 'from_phase'] = np.abs(phase_breaks.loc[phase_breaks['hap'] == h, ['pid','lpos']].rename(columns={'lpos':'pos'}).merge(scaffold_paths[['pid','pos',f'phase{h}']], on=['pid','pos'], how='left')[f'phase{h}'].values)
            phase_breaks.loc[phase_breaks['hap'] == h, 'to_phase'] = np.abs(phase_breaks.loc[phase_breaks['hap'] == h, ['pid','rpos']].rename(columns={'rpos':'pos'}).merge(scaffold_paths[['pid','pos',f'phase{h}']], on=['pid','pos'], how='left')[f'phase{h}'].values)
        phase_breaks = phase_breaks[['from_phase','to_phase']].drop_duplicates()
        scaffold_paths = AssignNewPhases(scaffold_paths, phase_breaks, ploidy)
#
    return scaffold_paths

def PhaseScaffolds(scaffold_paths, scaffold_graph, graph_ext, scaf_bridges, ploidy):
    ## Set initial phasing by giving every polyploid positions its own phase
    scaffold_paths['polyploid'] = (scaffold_paths[[f'phase{h}' for h in range(ploidy)]].values > 0).sum(axis=1) > 1
    scaffold_paths['polyploid'] = scaffold_paths['polyploid'] | (scaffold_paths['polyploid'].shift(1) & (scaffold_paths['pid'] == scaffold_paths['pid'].shift(1)))
    scaffold_paths['new_phase'] = scaffold_paths['polyploid'] | (scaffold_paths['pid'] != scaffold_paths['pid'].shift(1))
    scaffold_paths['new_phase'] = scaffold_paths['new_phase'].cumsum() - 1
    for h in range(ploidy):
        scaffold_paths[f'phase{h}'] = np.sign(scaffold_paths[f'phase{h}']) * (scaffold_paths['new_phase'] * ploidy + 1 + h)
    scaffold_paths.drop(columns=['polyploid','new_phase'], inplace=True)
#
    scaffold_paths = PhaseScaffoldsWithScafBridges(scaffold_paths, scaf_bridges, ploidy)
    scaffold_paths = PhaseScaffoldsWithScaffoldGraph(scaffold_paths, scaffold_graph, graph_ext, ploidy)
#
    # Merge phases, where no other phase has a break
    while True:
        for h in range(ploidy):
            scaffold_paths[f'from_phase{h}'] = np.abs(scaffold_paths[f'phase{h}'].shift(1, fill_value=0))
            scaffold_paths[f'to_phase{h}'] = np.abs(scaffold_paths[f'phase{h}'])
        scaffold_paths.loc[scaffold_paths['pid'] != scaffold_paths['pid'].shift(1), [f'from_phase{h}' for h in range(ploidy)]] = 0
        merge_phase = scaffold_paths.loc[(scaffold_paths['from_phase0'] != 0) & ((scaffold_paths[[f'from_phase{h}' for h in range(ploidy)]].values != scaffold_paths[[f'to_phase{h}' for h in range(ploidy)]].values).sum(axis=1) == 1), [f'{n}_phase{h}' for h in range(ploidy) for n in ['from','to']]].copy()
        scaffold_paths.drop(columns=[f'{n}_phase{h}' for h in range(ploidy) for n in ['from','to']], inplace=True)
        if len(merge_phase):
            merge_phase[['from_phase','to_phase']] = merge_phase[['from_phase0','to_phase0']].values
            for h in range(1, ploidy):
                merge_phase.loc[merge_phase[f'from_phase{h}'] != merge_phase[f'to_phase{h}'], ['from_phase','to_phase']] = merge_phase.loc[merge_phase[f'from_phase{h}'] != merge_phase[f'to_phase{h}'], [f'from_phase{h}',f'to_phase{h}']].values
            merge_phase = merge_phase[['from_phase','to_phase']].drop_duplicates()
            scaffold_paths = AssignNewPhases(scaffold_paths, merge_phase, ploidy)
        else:
            break
#
    return scaffold_paths

def ExpandScaffoldsWithContigs(scaffold_paths, scaffolds, scaffold_parts, ploidy):
    for h in range(0,ploidy):
        scaffold_paths = scaffold_paths.merge(scaffolds[['scaffold','size']].rename(columns={'scaffold':f'scaf{h}','size':f'size{h}'}), on=[f'scaf{h}'], how='left')
        scaffold_paths[f'size{h}'] = scaffold_paths[f'size{h}'].fillna(0).astype(int)
        
    scaffold_paths = scaffold_paths.loc[np.repeat(scaffold_paths.index.values, scaffold_paths[[f'size{h}' for h in range(ploidy)]].max(axis=1).values)]
    scaffold_paths['spos'] = scaffold_paths.groupby(['pid','pos'], sort=False).cumcount()
    for h in range(ploidy-1,-1,-1): # We need to go in the opposite direction, because we need the main to handle the alternatives
        scaffold_paths['mpos'] = np.where(scaffold_paths[f'strand{h}'] == '-', scaffold_paths[f'size{h}']-scaffold_paths['spos']-1, scaffold_paths['spos'])
        not_first = scaffold_paths['mpos'].values != np.where(scaffold_paths[f'strand{h}'] == '+', 0, scaffold_paths[f'size{h}']-1)
        if h>0:
            # Remove alternatives that are the same except for distance for all position except the first, where we need the distance
            scaffold_paths.loc[(scaffold_paths[f'scaf{h}'] == scaffold_paths['scaf0']) & (scaffold_paths[f'strand{h}'] == scaffold_paths['strand0']) & not_first, [f'phase{h}',f'scaf{h}',f'strand{h}',f'dist{h}']] = [-1,-1,'',0]
        scaffold_paths = scaffold_paths.merge(scaffold_parts.rename(columns={'scaffold':f'scaf{h}','pos':'mpos'}), on=[f'scaf{h}','mpos'], how='left')
        scaffold_paths.loc[np.isnan(scaffold_paths['conpart']), [f'scaf{h}',f'strand{h}',f'dist{h}']] = [-1,'',0] # Set positions that do not have a match (scaffold is shorter than for other haplotypes) to a deletion
        scaffold_paths.loc[np.isnan(scaffold_paths['conpart']) == False, f'scaf{h}'] = scaffold_paths.loc[np.isnan(scaffold_paths['conpart']) == False, 'conpart'].astype(int)
        scaffold_paths.rename(columns={f'scaf{h}':f'con{h}'}, inplace=True)
        apply_dist = (np.isnan(scaffold_paths['prev_dist']) == False) & not_first
        scaffold_paths.loc[apply_dist, f'dist{h}'] = np.where(scaffold_paths.loc[apply_dist, f'strand{h}'] == '-', scaffold_paths.loc[apply_dist, 'next_dist'], scaffold_paths.loc[apply_dist, 'prev_dist']).astype(int)
        scaffold_paths.loc[scaffold_paths['reverse'] == True, f'strand{h}'] = np.where(scaffold_paths.loc[scaffold_paths['reverse'] == True, f'strand{h}'] == '+', '-', '+')
        scaffold_paths.drop(columns=['conpart','reverse','next_dist','prev_dist'],inplace=True)
    scaffold_paths = scaffold_paths[['pid','pos'] + [f'{col}{h}' for h in range(ploidy) for col in ['phase','con','strand','dist']]].rename(columns={'pid':'scaf'})
    scaffold_paths['pos'] = scaffold_paths.groupby(['scaf']).cumcount()
#
    return scaffold_paths

def GetOriginalConnections(scaffold_paths, contig_parts, ploidy):
    # Find original connections in contig_parts
    org_cons = contig_parts.reset_index().loc[contig_parts['org_dist_right'] > 0, ['index','org_dist_right']].copy() # Do not include breaks (org_dist_right==0). If they could not be resealed, they should be separated for good
    org_cons.rename(columns={'index':'from','org_dist_right':'distance'}, inplace=True)
    org_cons['from_side'] = 'r'
    org_cons['to'] = org_cons['from']+1
    org_cons['to_side'] = 'l'

    # Lift connections to scaffolds while dropping connections that don't end at scaffold ends
    scaffold_ends_left = []
    scaffold_ends_right = []
    for h in range(ploidy):
        ends = scaffold_paths.loc[(scaffold_paths[f'phase{h}'] < 0) | (scaffold_paths[f'con{h}'] >= 0), :] # Remove deletions at this haplotype, because they cannot be an end
        lends = ends.groupby(['scaf'], sort=False).first().reset_index()
        lends.loc[lends[f'phase{h}'] < 0, [f'con{h}',f'strand{h}']] = lends.loc[lends[f'phase{h}'] < 0, ['con0','strand0']].values
        lends = lends[['scaf',f'con{h}',f'strand{h}']].rename(columns={'scaf':'scaffold',f'con{h}':'from',f'strand{h}':'fstrand'})
        lends['from_side'] = np.where(lends['fstrand'] == '+', 'l', 'r')
        lends['scaf_side'] = 'l'
        scaffold_ends_left.append( lends.drop(columns=['fstrand']) )
        rends = ends.groupby(['scaf'], sort=False).last().reset_index()
        rends.loc[rends[f'phase{h}'] < 0, [f'con{h}',f'strand{h}']] = rends.loc[rends[f'phase{h}'] < 0, ['con0','strand0']].values
        rends = rends[['scaf',f'con{h}',f'strand{h}']].rename(columns={'scaf':'scaffold',f'con{h}':'from',f'strand{h}':'fstrand'})
        rends['from_side'] = np.where(rends['fstrand'] == '+', 'r', 'l')
        rends['scaf_side'] = 'r'
        scaffold_ends_right.append( rends.drop(columns=['fstrand']) )
    scaffold_ends_left = pd.concat(scaffold_ends_left, ignore_index=True).drop_duplicates()
    scaffold_ends_right = pd.concat(scaffold_ends_right, ignore_index=True).drop_duplicates()
    scaffold_ends = pd.concat([ scaffold_ends_left, scaffold_ends_right ], ignore_index=True)
    org_cons = org_cons.merge(scaffold_ends, on=['from','from_side'], how='inner')
    org_cons.drop(columns=['from','from_side'], inplace=True)
    org_cons.rename(columns={'scaffold':'from','scaf_side':'from_side'}, inplace=True)
    org_cons = org_cons.merge(scaffold_ends.rename(columns={'from':'to', 'from_side':'to_side'}), on=['to','to_side'], how='inner')
    org_cons.drop(columns=['to','to_side'], inplace=True)
    org_cons.rename(columns={'scaffold':'to','scaf_side':'to_side'}, inplace=True)

    # Also insert reversed connections
    org_cons = pd.concat( [org_cons, org_cons.rename(columns={'from':'to','from_side':'to_side','to':'from','to_side':'from_side'})], ignore_index=True )

    # Remove scaffolding connections, where different haplotypes have different connections
    org_cons = org_cons.groupby(['from','from_side','to','to_side'])['distance'].mean().reset_index()
    fdups = org_cons.groupby(['from','from_side']).size()
    fdups = fdups[fdups > 1].reset_index().drop(columns=[0])
    tdups = org_cons.groupby(['to','to_side']).size()
    tdups = tdups[tdups > 1].reset_index().drop(columns=[0])
    org_cons = org_cons[ (org_cons.merge(fdups, on=['from','from_side'], how='left', indicator=True)['_merge'].values == "left_only") & 
                         (org_cons.merge(tdups, on=['to','to_side'], how='left', indicator=True)['_merge'].values == "left_only") ].copy()

    return org_cons

def OrderByUnbrokenOriginalScaffolds(scaffold_paths, contig_parts, ploidy):
    ## Bring scaffolds in order of unbroken original scaffolding and remove circularities
    # Every scaffold starts as its own metascaffold
    meta_scaffolds = pd.DataFrame({'meta':np.unique(scaffold_paths['scaf']), 'size':1, 'lcon':-1, 'lcon_side':'', 'rcon':-1, 'rcon_side':''})
    meta_scaffolds.index = meta_scaffolds['meta'].values
    meta_parts = pd.DataFrame({'scaffold':meta_scaffolds['meta'], 'meta':meta_scaffolds['meta'], 'pos':0, 'reverse':False})
    meta_parts.index = meta_parts['scaffold'].values
#
    # Prepare meta scaffolding
    org_cons = GetOriginalConnections(scaffold_paths, contig_parts, ploidy)
    meta_scaffolds.loc[org_cons.loc[org_cons['from_side'] == 'l', 'from'].values, 'lcon'] = org_cons.loc[org_cons['from_side'] == 'l', 'to'].values
    meta_scaffolds.loc[org_cons.loc[org_cons['from_side'] == 'l', 'from'].values, 'lcon_side'] = org_cons.loc[org_cons['from_side'] == 'l', 'to_side'].values
    meta_scaffolds.loc[org_cons.loc[org_cons['from_side'] == 'r', 'from'].values, 'rcon'] = org_cons.loc[org_cons['from_side'] == 'r', 'to'].values
    meta_scaffolds.loc[org_cons.loc[org_cons['from_side'] == 'r', 'from'].values, 'rcon_side'] = org_cons.loc[org_cons['from_side'] == 'r', 'to_side'].values
#
    # Rename some columns and create extra columns just to call the same function as used for the normal scaffolding
    meta_scaffolds['left'] = meta_scaffolds['meta']
    meta_scaffolds['lside'] = 'l'
    meta_scaffolds['right'] = meta_scaffolds['meta']
    meta_scaffolds['rside'] = 'r'
    meta_scaffolds['lextendible'] = True
    meta_scaffolds['rextendible'] = True
    meta_scaffolds['circular'] = False
    meta_scaffolds.rename(columns={'meta':'scaffold','lcon':'lscaf','lcon_side':'lscaf_side','rcon':'rscaf','rcon_side':'rscaf_side'}, inplace=True)
    meta_parts.rename(columns={'scaffold':'conpart','meta':'scaffold'}, inplace=True)
    meta_scaffolds, meta_parts = ScaffoldAlongGivenConnections(meta_scaffolds, meta_parts)
    meta_parts.rename(columns={'conpart':'scaffold','scaffold':'meta'}, inplace=True)
#
    # Set new continuous scaffold ids consistent with original scaffolding and apply reversions
    meta_parts.sort_values(['meta','pos'], inplace=True)
    meta_parts['new_scaf'] = range(len(meta_parts))
    scaffold_paths = scaffold_paths.merge(meta_parts[['scaffold','reverse','new_scaf']].rename(columns={'scaffold':'scaf'}), on=['scaf'], how='left')
    scaffold_paths['scaf'] = scaffold_paths['new_scaf']
    col_rename = {**{'scaf':'pid'}, **{f'con{h}':f'scaf{h}' for h in range(ploidy)}}
    scaffold_paths = ReverseScaffolds(scaffold_paths.rename(columns=col_rename), scaffold_paths['reverse'], ploidy).rename(columns={v: k for k, v in col_rename.items()})
    scaffold_paths.drop(columns=['reverse','new_scaf'], inplace=True)

    # Set org_dist_left/right based on scaffold (not the contig anymore)
    org_cons = GetOriginalConnections(scaffold_paths, contig_parts, ploidy)
    org_cons = org_cons.loc[(org_cons['from_side'] == 'l') & (org_cons['from']-1 == org_cons['to']), :]
    scaffold_paths['sdist_left'] = -1
    scaffold_paths.loc[np.isin(scaffold_paths['scaf'], org_cons['from'].values) & (scaffold_paths['pos'] == 0), 'sdist_left'] = org_cons['distance'].values
    scaffold_paths['sdist_right'] = scaffold_paths['sdist_left'].shift(-1, fill_value=-1)

    return scaffold_paths

def ScaffoldContigs(contig_parts, bridges, mappings, ploidy, max_loop_units):
    # Each contig starts with being its own scaffold
    scaffold_parts = pd.DataFrame({'conpart': contig_parts.index.values, 'scaffold': contig_parts.index.values, 'pos': 0, 'reverse': False})
    scaffolds = pd.DataFrame({'scaffold': contig_parts.index.values, 'left': contig_parts.index.values, 'lside':'l', 'right': contig_parts.index.values, 'rside':'r', 'lextendible':True, 'rextendible':True, 'circular':False, 'size':1})
#
    # Combine contigs into scaffolds on unique bridges
    unique_bridges = bridges.loc[(bridges['to_alt'] == 1) & (bridges['from_alt'] == 1), ['from','from_side','to','to_side']]
    scaffolds = scaffolds.merge(unique_bridges[unique_bridges['from_side']=='l'].drop(columns=['from_side']).rename(columns={'from':'scaffold', 'to':'lscaf', 'to_side':'lscaf_side'}), on=['scaffold'], how='left')
    scaffolds = scaffolds.merge(unique_bridges[unique_bridges['from_side']=='r'].drop(columns=['from_side']).rename(columns={'from':'scaffold', 'to':'rscaf', 'to_side':'rscaf_side'}), on=['scaffold'], how='left')
    scaffolds[['lscaf','rscaf']] = scaffolds[['lscaf','rscaf']].fillna(-1).astype(int)
    scaffolds[['lscaf_side','rscaf_side']] = scaffolds[['lscaf_side','rscaf_side']].fillna('')
    scaffolds, scaffold_parts = ScaffoldAlongGivenConnections(scaffolds, scaffold_parts)
    scaffolds.drop(columns=['lscaf','lscaf_side','rscaf','rscaf_side'], inplace = True)
    scaffold_parts = AddDistaneInformation(scaffold_parts, bridges[(bridges['to_alt'] == 1) & (bridges['from_alt'] == 1)]) # Do not use unique_bridges, because we require 'mean_dist' column
    scaf_bridges = LiftBridgesFromContigsToScaffolds(bridges, scaffolds)
    org_scaf_conns = GetOriginalScaffoldConnections(contig_parts, scaffolds)
#
    # Build scaffold graph to find unique bridges over scaffolds with alternative connections
    long_range_connections = GetLongRangeConnections(bridges, mappings)
    long_range_connections = TransformContigConnectionsToScaffoldConnections(long_range_connections, scaffold_parts)
    scaffold_graph = BuildScaffoldGraph(long_range_connections, scaf_bridges)
    graph_ext = FindValidExtensionsInScaffoldGraph(scaffold_graph)
    scaffold_paths = TraverseScaffoldingGraph(scaffolds, scaffold_graph, graph_ext, scaf_bridges, org_scaf_conns, ploidy, max_loop_units)
    scaffold_paths = PhaseScaffolds(scaffold_paths, scaffold_graph, graph_ext, scaf_bridges, ploidy)
    
    # Finish Scaffolding
    scaffold_paths = ExpandScaffoldsWithContigs(scaffold_paths, scaffolds, scaffold_parts, ploidy)
    scaffold_paths = OrderByUnbrokenOriginalScaffolds(scaffold_paths, contig_parts, ploidy)

    return scaffold_paths

def PrepareScaffoldPathForMapping(scaffold_paths, bridges, ploidy):
    # Prepare entries in scaffold_paths for mapping
    all_scafs = []
    for h in range(ploidy):
        hap_paths = scaffold_paths[['scaf','pos']+[f'phase{h}',f'con{h}',f'strand{h}',f'dist{h}']].rename(columns={f'phase{h}':'phase',f'con{h}':'conpart',f'strand{h}':'scaf_strand',f'dist{h}':'ldist'})
        if np.sum(hap_paths['phase'] < 0):
            hap_paths.loc[hap_paths['phase'] < 0, ['conpart','scaf_strand','ldist']] = scaffold_paths.loc[hap_paths['phase'].values < 0, ['con0','strand0','dist0']].values
        ## Get haplotype of connected contigs
        hap_paths['lhap'] = h # If the only difference is the distance, an alternative haplotype could only be alternative on one side
        hap_paths['rhap'] = h
        if 0 < h:
            hap_paths['lphase'] = hap_paths['phase'].shift(1, fill_value=-1)
            hap_paths.loc[hap_paths['scaf'] != hap_paths['scaf'].shift(1), 'lphase'] = -1
            hap_paths['rphase'] = hap_paths['phase'].shift(-1, fill_value=-1)
            hap_paths.loc[hap_paths['scaf'] != hap_paths['scaf'].shift(-1), 'rphase'] = -1
            # The important phase is the one storing the distance, but sometimes the distances to different contigs are the same and then we need the other phase
            hap_paths['main'] = (hap_paths['conpart'] == scaffold_paths['con0'].values) & (hap_paths['scaf_strand'] == scaffold_paths['strand0'].values) # Get the alternative haplotypes identical to main
            hap_paths['lphase'] = np.where((hap_paths['phase'] >= 0) | hap_paths['main'].shift(1, fill_value=True), hap_paths['phase'], hap_paths['lphase'])
            hap_paths['rphase'] = np.where((hap_paths['rphase'] >= 0) | hap_paths['main'], hap_paths['rphase'], hap_paths['phase'])
            # Get haplotypes based on phase
            hap_paths.loc[hap_paths['lphase'] < 0, 'lhap'] = 0
            hap_paths.loc[hap_paths['rphase'] < 0, 'rhap'] = 0
        else:
            hap_paths['main'] = False # While the main path is by definition identical to main, we do not want to remove it, thus set it to False
        ## Remove deletions (we needed them to get the proper phase, but now they are only making things complicated)
        hap_paths = hap_paths.loc[(hap_paths['conpart'] >= 0), :]
        ## Get connected contigs with strand and distance (we already have ldist)
        hap_paths['lcon'] = hap_paths['conpart'].shift(1, fill_value=-1)
        hap_paths['lstrand'] = hap_paths['scaf_strand'].shift(1, fill_value='')
        hap_paths['lpos'] = hap_paths['pos'].shift(1, fill_value=-1)
        hap_paths.loc[hap_paths['scaf'] != hap_paths['scaf'].shift(1), ['lcon','lstrand','lpos']] = [-1,'',-1]
        hap_paths['rcon'] = hap_paths['conpart'].shift(-1, fill_value=-1)
        hap_paths['rstrand'] = hap_paths['scaf_strand'].shift(-1, fill_value='')
        hap_paths['rdist'] = hap_paths['ldist'].shift(-1, fill_value=0)
        hap_paths['rpos'] = hap_paths['pos'].shift(-1, fill_value=-1)
        hap_paths.loc[hap_paths['scaf'] != hap_paths['scaf'].shift(-1), ['rcon','rstrand','rdist','rpos']] = [-1,'',0,-1]
        ## Filter the ones completely identical to main
        if 0 < h:
            hap_paths = hap_paths.loc[(hap_paths['lhap'] > 0) | (hap_paths['rhap'] > 0), :]
        all_scafs.append(hap_paths[['scaf','pos','conpart','scaf_strand','lpos','lhap','lcon','lstrand','ldist','rpos','rhap','rcon','rstrand','rdist','main']])
    all_scafs = pd.concat(all_scafs, ignore_index=True)
    # Get allowed distance range for contig bridges
    all_scafs['from_side'] = np.where(all_scafs['scaf_strand'] == '+', 'l', 'r')
    all_scafs['to_side'] = np.where(all_scafs['lstrand'] == '+', 'r', 'l')
    all_scafs = all_scafs.merge(bridges[['from','from_side','to','to_side','mean_dist','min_dist','max_dist']].rename(columns={'from':'conpart','to':'lcon','mean_dist':'ldist','min_dist':'ldmin','max_dist':'ldmax'}), on=['conpart','from_side','lcon','to_side','ldist'], how='left')
    all_scafs['from_side'] = np.where(all_scafs['scaf_strand'] == '+', 'r', 'l')
    all_scafs['to_side'] = np.where(all_scafs['rstrand'] == '+', 'l', 'r')
    all_scafs = all_scafs.merge(bridges[['from','from_side','to','to_side','mean_dist','min_dist','max_dist']].rename(columns={'from':'conpart','to':'rcon','mean_dist':'rdist','min_dist':'rdmin','max_dist':'rdmax'}), on=['conpart','from_side','rcon','to_side','rdist'], how='left')
    consitency_check = all_scafs[(np.isnan(all_scafs['ldmin']) & (all_scafs['lcon'] >= 0)) | (np.isnan(all_scafs['rdmin']) & (all_scafs['rcon'] >= 0))].copy()
    if len(consitency_check):
        print("Error: Scaffold paths contains connections that are not supported by a bridge.")
        print(consitency_check)
    all_scafs[['ldmin','rdmin']] = all_scafs[['ldmin','rdmin']].fillna(-sys.maxsize*0.99).values.astype(int) # Use *0.99 to avoid overflow through type convertion from float to int (should be the lcon/rcon -1, where we do not want to set any constraints)
    all_scafs[['ldmax','rdmax']] = all_scafs[['ldmax','rdmax']].fillna(sys.maxsize*0.99).values.astype(int)
    all_scafs.drop(columns=['from_side','to_side'], inplace=True)
#
    return all_scafs

def BasicMappingToScaffolds(mappings, all_scafs):
    # Mapping to matching contigs and reverse if the contig is reversed on the scaffold
    mappings = mappings.merge(all_scafs, on=['conpart'], how='inner')
    mappings.loc[mappings['scaf_strand'] == '-', 'strand'] = np.where(mappings.loc[mappings['scaf_strand'] == '-', 'strand'].values == '+', '-', '+')
    tmp = mappings.loc[mappings['scaf_strand'] == '-', ['right_con','right_con_dist','rmapq','rmatches']].values
    mappings.loc[mappings['scaf_strand'] == '-', ['right_con','right_con_dist','rmapq','rmatches']] = mappings.loc[mappings['scaf_strand'] == '-', ['left_con','left_con_dist','lmapq','lmatches']].values
    mappings.loc[mappings['scaf_strand'] == '-', ['left_con','left_con_dist','lmapq','lmatches']] = tmp
    tmp = mappings.loc[mappings['scaf_strand'] == '-', 'right_con_side'].values
    mappings.loc[mappings['scaf_strand'] == '-', 'right_con_side'] = mappings.loc[mappings['scaf_strand'] == '-', 'left_con_side'].values
    mappings.loc[mappings['scaf_strand'] == '-', 'left_con_side'] = tmp
    # Check that connections to left and right fit with the scaffold
    mappings = mappings[((mappings['left_con'] == mappings['lcon']) | (mappings['left_con'] < 0) | (mappings['lcon'] < 0)) & ((mappings['right_con'] == mappings['rcon']) | (mappings['right_con'] < 0) | (mappings['rcon'] < 0))].drop(columns=['rcon','lcon']).rename(columns={'left_con':'lcon','right_con':'rcon'})
    mappings = mappings[(((mappings['left_con_side'] == 'r') == (mappings['lstrand'] == '+')) | (mappings['left_con_side'] == '') | (mappings['lstrand'] == '')) & (((mappings['right_con_side'] == 'l') == (mappings['rstrand'] == '+')) | (mappings['right_con_side'] == '') | (mappings['rstrand'] == ''))].drop(columns=['left_con_side','right_con_side','lstrand','rstrand'])
    mappings = mappings[(((mappings['left_con_dist'] <= mappings['ldmax']) & (mappings['left_con_dist'] >= mappings['ldmin'])) | (mappings['lcon'] < 0)) & (((mappings['right_con_dist'] <= mappings['rdmax']) & (mappings['right_con_dist'] >= mappings['rdmin'])) | (mappings['rcon'] < 0))].drop(columns=['ldmin','ldmax','rdmin','rdmax','ldist','rdist']).rename(columns={'left_con_dist':'ldist','right_con_dist':'rdist','scaf_strand':'con_strand'})
    # Check for alternative haplotypes that the mapping is not only supporting a side that is identical to main
    mappings = mappings.loc[(mappings['lhap'] == mappings['rhap']) | ((mappings['lhap'] > 0) & (mappings['lcon'] >= 0)) | ((mappings['rhap'] > 0) & (mappings['rcon'] >= 0)),:]
    mappings = mappings[(mappings['main'] == False) | (mappings['lcon'] >= 0) | (mappings['rcon'] >= 0)].drop(columns=['main']) # Remove alternative haplotypes, where both sides, but not the contig itself, are distinct from the main path and the read does not reach either side
    # Check that we do not have another haplotype with a longer scaffold supported by the read
    mappings['nconns'] = (mappings[['lpos','rpos']] >= 0).sum(axis=1)
    mappings.sort_values(['read_name','read_start','read_pos','scaf','pos','lhap','rhap'], inplace=True)
    mappings = mappings[ ( (mappings[['read_name','read_start','read_pos','scaf','pos']] != mappings[['read_name','read_start','read_pos','scaf','pos']].shift(1)).any(axis=1) | (mappings['nconns'] >= mappings['nconns'].shift(1)) ) &
                         ( (mappings[['read_name','read_start','read_pos','scaf','pos']] != mappings[['read_name','read_start','read_pos','scaf','pos']].shift(-1)).any(axis=1) | (mappings['nconns'] >= mappings['nconns'].shift(-1)) ) ].copy()
    mappings.drop(columns=['nconns'], inplace=True)
#
    # Assign groups to separate parts of a read mapping to two separate (maybe overlapping) locations on the scaffold
    mappings['group'] = ( (mappings['read_name'] != mappings['read_name'].shift(1)) | (mappings['read_start'] != mappings['read_start'].shift(1)) | (mappings['read_pos'] != mappings['read_pos'].shift(1)+1) | # We do not follow the read
                          (mappings['scaf'] != mappings['scaf'].shift(1)) | (np.where(mappings['strand'] == '-', mappings['rpos'], mappings['lpos']) != mappings['pos'].shift(1)) | # We do not follow the scaffold
                          np.where(mappings['strand'] == '-', mappings['rhap'] != mappings['lhap'].shift(1), mappings['lhap'] != mappings['rhap'].shift(1)) ) # We do not follow the haplotype
    mappings['unigroup'] = ( mappings['group'] | ((mappings['read_name'] == mappings['read_name'].shift(2)) & (mappings['read_pos'] == mappings['read_pos'].shift(2)+1)) | 
                                                 ((mappings['read_name'] == mappings['read_name'].shift(-1)) & (mappings['read_pos'] == mappings['read_pos'].shift(-1))) ).cumsum() # We have more than one option (thus while we want to group them if everything is correct, we do not want to propagate errors through those branchings, because the other side might correctly continue along another branch)
    mappings['group'] = mappings['group'].cumsum()
    # Remove mappings, where reads do not continue, allthough they should
    old_len = 0
    while old_len != len(mappings):
        old_len = len(mappings)
        for d, d2 in zip(['l','r'],['r','l']):
            mappings['check_pos'] = mappings['read_pos'] + np.where(mappings['strand'] == '+', -1, 1) * (1 if d == 'l' else -1)
            invalid = mappings.loc[(mappings[f'{d}con'] >= 0) & (mappings[f'{d}pos'] >= 0), ['read_name','read_start','check_pos','scaf',f'{d}pos',f'{d}hap','unigroup']].merge(mappings[['read_name','read_start','read_pos','scaf','pos',f'{d2}hap']].rename(columns={'read_pos':'check_pos','pos':f'{d}pos',f'{d2}hap':f'{d}hap'}), on=['read_name','read_start','check_pos','scaf',f'{d}pos',f'{d}hap'], how='left', indicator=True)[['unigroup','_merge']]
            invalid = np.unique(invalid.loc[invalid['_merge'] == "left_only", 'unigroup'].values)
            mappings = mappings[np.isin(mappings['unigroup'], invalid) == False].copy()
    mappings.drop(columns=['unigroup'], inplace=True)
    # Assign the groups matching the left/right connection to combine groups that are not in neighbouring rows
    groups = []
    for d, d2 in zip(['l','r'],['r','l']):
        mappings['check_pos'] = mappings['read_pos'] + np.where(mappings['strand'] == '+', -1, 1) * (1 if d == 'l' else -1)
        new_groups = mappings[['read_name','read_start','check_pos','scaf',f'{d}pos',f'{d}hap','group']].merge(mappings[['read_name','read_start','read_pos','scaf','pos',f'{d2}hap','group']].rename(columns={'read_pos':'check_pos','pos':f'{d}pos',f'{d2}hap':f'{d}hap','group':f'{d}group'}), on=['read_name','read_start','check_pos','scaf',f'{d}pos',f'{d}hap'], how='inner')[['group',f'{d}group']] # We can have multiple matches here, when haplotypes split, we will handle that later
        new_groups = new_groups[new_groups['group'] != new_groups[f'{d}group']].drop_duplicates()
        if len(groups) == 0:
            groups = new_groups
        else:
            groups = groups.merge(new_groups, on=['group'], how='outer')
    mappings.drop(columns=['check_pos'], inplace=True)
    for d in ['l','r']:
        groups.loc[np.isnan(groups[f'{d}group']), f'{d}group'] = groups.loc[np.isnan(groups[f'{d}group']), 'group']
    groups = groups.astype(int)
    groups['new_group'] = groups.min(axis=1)
    while True:
        # Assign lowest connected group to all groups
        groups['check'] = groups['new_group']
        groups = groups.merge(groups[['group','new_group']].rename(columns={'group':'lgroup','new_group':'lngroup'}), on=['lgroup'], how='left')
        groups = groups.merge(groups[['group','new_group']].rename(columns={'group':'rgroup','new_group':'rngroup'}), on=['rgroup'], how='left')
        groups['new_group'] = groups[['new_group','lngroup','rngroup']].min(axis=1).astype(int)
        groups.drop(columns=['lngroup','rngroup'], inplace=True)
        groups.drop_duplicates(inplace=True)
        groups = groups.groupby(['group','lgroup','rgroup','check'])['new_group'].min().reset_index()
        tmp = groups.groupby(['group'], sort=False)['new_group'].agg(['min','size'])
        groups['new_group'] = np.repeat(tmp['min'].values, tmp['size'].values)
        if np.sum(groups['new_group'] != groups['check']) == 0:
            groups = groups.groupby(['group'])['new_group'].max().reset_index()
            break
    groups = groups.loc[groups['group'] != groups['new_group'], :]
    mappings = mappings.merge(groups, on=['group'], how='left')
    mappings.loc[np.isnan(mappings['new_group']) == False, 'group'] = mappings.loc[np.isnan(mappings['new_group']) == False, 'new_group'].astype(int)
    mappings.drop(columns=['new_group'], inplace=True)
    # If a read matches multiple haplotypes of the same read they might end up in the same group. To resolve this, duplicate the groups.
    merged_groups = mappings.groupby(['group','read_pos']).size()
    merged_groups = merged_groups[merged_groups > 1].reset_index()[['group']].drop_duplicates()
    if len(merged_groups):
        mappings['mindex'] = mappings.index.values
        old_groups = mappings.loc[np.isin(mappings['group'].values, merged_groups['group'].values), ['group','read_pos','pos','rhap','lhap','mindex','strand','rpos','lpos']].copy()
        # Start from the lowest read_pos in the group and expand+duplicate on valid connections
        merged_groups = merged_groups.merge(old_groups.groupby(['group'])['read_pos'].min().reset_index(), on=['group'], how='left')
        merged_groups['max_pos'] = merged_groups[['group']].merge(old_groups.groupby(['group'])['read_pos'].max().reset_index(), on=['group'], how='left')['read_pos'].values
        merged_groups['new_group'] = merged_groups['group']
        merged_groups = merged_groups.merge(old_groups, on=['group','read_pos'], how='left')
        start_group_id = mappings['group'].max() + 1
        split = merged_groups['new_group'] == merged_groups['new_group'].shift(1)
        nsplit = np.sum(split)
        if nsplit:
            merged_groups.loc[split, 'new_group'] = np.arange(start_group_id, start_group_id+nsplit)
            start_group_id += nsplit
        sep_groups = merged_groups[['group','new_group','mindex']].copy()
        merged_groups = merged_groups[merged_groups['read_pos'] < merged_groups['max_pos']].drop(columns=['mindex','strand','rpos','lpos'])
        while len(merged_groups):
            # Get the valid extensions for the next position
            merged_groups['read_pos'] += 1
            merged_groups.rename(columns={col:f'o{col}' for col in ['pos','rhap','lhap']}, inplace=True)
            merged_groups = merged_groups.merge(old_groups, on=['group','read_pos'], how='left')
            merged_groups = merged_groups[ (np.where(merged_groups['strand'] == '-', merged_groups['rpos'], merged_groups['lpos']) == merged_groups['opos']) &
                                           np.where(merged_groups['strand'] == '-', merged_groups['rhap'] == merged_groups['olhap'], merged_groups['lhap'] == merged_groups['orhap']) ].drop(columns=['opos','olhap','orhap'])
            # Split groups
            merged_groups.sort_values(['group','new_group'], inplace=True)
            split = merged_groups['new_group'] == merged_groups['new_group'].shift(1)
            nsplit = np.sum(split)
            if nsplit:
                merged_groups['new_group2'] =  merged_groups['new_group']
                merged_groups.loc[split, 'new_group2'] = np.arange(start_group_id, start_group_id+nsplit)
                start_group_id += nsplit
                sep_groups = sep_groups.merge(merged_groups[['new_group','new_group2']], on=['new_group'], how='left')
                sep_groups.loc[np.isnan(sep_groups['new_group2']) == False, 'new_group'] = sep_groups.loc[np.isnan(sep_groups['new_group2']) == False, 'new_group2'].astype(int) # If the new_group is no longer in merged_groups (because the group finished) we get NaNs
                sep_groups.drop(columns=['new_group2'], inplace=True)
                merged_groups['new_group'] = merged_groups['new_group2']
                merged_groups.drop(columns=['new_group2'], inplace=True)
            # Check if a new groups starts here (unconnected entry at this position for the group)
            new_groups = merged_groups[['group','read_pos','max_pos']].drop_duplicates().merge(old_groups, on=['group','read_pos'], how='left')
            new_groups = new_groups[np.isin(new_groups['mindex'].values, merged_groups['mindex'].values) == False].copy()
            if len(new_groups):
                new_groups['new_group'] = np.arange(start_group_id, start_group_id+len(new_groups))
                start_group_id += len(new_groups)
                merged_groups = pd.concat([merged_groups, sep_groups], ignore_index=True)
            # Store valid extensions and prepare next round
            sep_groups = pd.concat([sep_groups, merged_groups[['group','new_group','mindex']].copy()], ignore_index=True)
            merged_groups = merged_groups[merged_groups['read_pos'] < merged_groups['max_pos']].drop(columns=['mindex','strand','rpos','lpos'])
        sep_groups.sort_values(['mindex','group','new_group'], inplace=True)
        mappings = mappings.merge(sep_groups, on=['mindex','group'], how='left')
        mappings.loc[np.isnan(mappings['new_group']) == False, 'group'] = mappings.loc[np.isnan(mappings['new_group']) == False, 'new_group'].astype(int)
        mappings.drop(columns=['mindex','new_group'], inplace=True)
    # Check that groups are consistent now
    merged_groups = mappings.groupby(['group','read_pos']).size()
    merged_groups = merged_groups[merged_groups > 1].reset_index()[['group']].drop_duplicates()
    if len(merged_groups):
        print("Error: Groups have duplicated positions after BasicMappingToScaffolds.")
        print( mappings.loc[np.isin(mappings['group'].values, merged_groups['group'].values)] )
#
    return mappings

def MapReadsToScaffolds(mappings, scaffold_paths, bridges, ploidy):
    # Preserve some information on the neighbouring mappings that we need later
    mappings['read_pos'] = mappings.groupby(['read_name','read_start'], sort=False).cumcount()
    mappings[['lmapq','lmatches']] = mappings[['mapq','matches']].shift(1, fill_value=-1).values
    tmp =  mappings[['mapq','matches']].shift(-1, fill_value=-1)
    mappings['rmapq'] = np.where(mappings['strand'] == '+', tmp['mapq'].values, mappings['lmapq'].values)
    mappings['rmatches'] = np.where(mappings['strand'] == '+', tmp['matches'].values, mappings['lmatches'].values)
    mappings.loc[mappings['strand'] == '-', ['lmapq','lmatches']] = tmp[mappings['strand'].values == '-'].values
    mappings.loc[mappings['left_con'] < 0, ['lmapq','lmatches']] = -1
    mappings.loc[mappings['right_con'] < 0, ['rmapq','rmatches']] = -1
    mappings.drop(columns=['num_mappings'], inplace=True)
#
    # Store full mappings, such that we can later go back and check if we can improve scaffold path to fit the reads
    org_mappings = mappings.copy()
#
    while True:
        all_scafs = PrepareScaffoldPathForMapping(scaffold_paths, bridges, ploidy)
        mappings = BasicMappingToScaffolds(mappings, all_scafs)
#
        # Get coverage for connections
        conn_cov = mappings.loc[(mappings['lcon'] >= 0) & (mappings['lpos'] >= 0), ['scaf','pos','lpos','lhap']].rename(columns={'pos':'rpos','lhap':'hap'}).groupby(['scaf','lpos','rpos','hap']).size().reset_index(name='cov')
        conn_cov = all_scafs.loc[(all_scafs['lpos'] >= 0), ['scaf','lpos','pos','lhap']].drop_duplicates().rename(columns={'pos':'rpos','lhap':'hap'}).sort_values(['scaf','lpos','rpos','hap']).merge(conn_cov, on=['scaf','lpos','rpos','hap'], how='left')
        conn_cov['cov'] = conn_cov['cov'].fillna(0).astype(int)
#
        # Remove reads, where they map to multiple locations (keep them separately, so that we can restore them if it does leave a connection between contigs without reads)
        dups_maps = mappings[['read_name','read_start','read_pos','scaf','pos']].drop_duplicates().groupby(['read_name','read_start','read_pos'], sort=False).size().reset_index(name='count')
        mcols = ['read_name','read_start','read_pos']
        mappings['count'] = mappings[mcols].merge(dups_maps, on=mcols, how='left')['count'].values
        dups_maps = mappings[['group','count']].groupby(['group'])['count'].min().reset_index(name='gcount')
        mappings.rename(columns={'count':'gcount'}, inplace=True)
        mappings['gcount'] = mappings[['group']].merge(dups_maps, on=['group'], how='left')['gcount'].values
        dups_maps = mappings[mappings['gcount'] > 1].copy()
        mappings = mappings[mappings['gcount'] == 1].drop(columns=['gcount'])
        mappings.rename(columns={'group':'mapid'}, inplace=True)
        conn_cov = conn_cov.merge(mappings.loc[(mappings['lcon'] >= 0) & (mappings['lpos'] >= 0), ['scaf','pos','lpos','lhap']].rename(columns={'pos':'rpos','lhap':'hap'}).groupby(['scaf','lpos','rpos','hap']).size().reset_index(name='ucov'), on=['scaf','lpos','rpos','hap'], how='left')
        conn_cov['ucov'] = conn_cov['ucov'].fillna(0).astype(int)
#
        # Try to fix scaffold_paths, where no reads support the connection
        # Start by getting mappings that have both sides in them
        unsupp_conns = conn_cov[conn_cov['ucov'] == 0].copy()
        if len(unsupp_conns):
            unsupp_conns = unsupp_conns.merge(scaffold_paths[['scaf','pos']+[f'con{h}' for h in range(ploidy)]].rename(columns={'pos':'lpos','con0':'lcon'}), on=['scaf','lpos'], how='left')
            for h in range(1,ploidy):
                sel = (h == unsupp_conns['hap']) & (unsupp_conns[f'con{h}'] >= 0)
                unsupp_conns.loc[sel, 'lcon'] = unsupp_conns.loc[sel, f'con{h}'] 
            unsupp_conns.drop(columns=[f'con{h}' for h in range(1,ploidy)], inplace=True)
            unsupp_conns = unsupp_conns.merge(scaffold_paths[['scaf','pos']+[f'con{h}' for h in range(ploidy)]].rename(columns={'pos':'rpos','con0':'rcon'}), on=['scaf','rpos'], how='left')
            for h in range(1,ploidy):
                sel = (h == unsupp_conns['hap']) & (unsupp_conns[f'con{h}'] >= 0)
                unsupp_conns.loc[sel, 'rcon'] = unsupp_conns.loc[sel, f'con{h}'] 
            unsupp_conns.drop(columns=[f'con{h}' for h in range(1,ploidy)], inplace=True)
            lreads = unsupp_conns[['lcon']].reset_index().merge(org_mappings[['conpart','read_name']].rename(columns={'conpart':'lcon'}), on=['lcon'], how='inner')
            rreads = unsupp_conns[['rcon']].reset_index().merge(org_mappings[['conpart','read_name']].rename(columns={'conpart':'rcon'}), on=['rcon'], how='inner')
            supp_reads = lreads.drop(columns=['lcon']).merge(rreads.drop(columns=['rcon']), on=['index','read_name'], how='inner')[['read_name']].drop_duplicates()
            # Remove reads that already have a valid mapping to scaffold_paths
            supp_reads = supp_reads.loc[supp_reads.merge(mappings[['read_name']].drop_duplicates(), on=['read_name'], how='left', indicator=True)['_merge'].values == "left_only", :]
            supp_reads = supp_reads.loc[supp_reads.merge(dups_maps[['read_name']].drop_duplicates(), on=['read_name'], how='left', indicator=True)['_merge'].values == "left_only", :]
            supp_reads = supp_reads.merge(org_mappings, on=['read_name'], how='inner')
            supp_reads.sort_values(['read_name','read_start','read_pos'], inplace=True)
            # Remove the unsupported connections from all_scafs and try mapping supp_reads again
            all_scafs.loc[all_scafs[['scaf','pos','rhap']].merge(unsupp_conns[['scaf','lpos','hap']].rename(columns={'lpos':'pos','hap':'rhap'}), on=['scaf','pos','rhap'], how='left', indicator=True)['_merge'].values == "both", ['rpos','rcon','rstrand','rdist','rdmin','rdmax']] = [-1,-1,'',0,-int(sys.maxsize*0.99),int(sys.maxsize)*0.99]
            all_scafs.loc[all_scafs[['scaf','pos','lhap']].merge(unsupp_conns[['scaf','rpos','hap']].rename(columns={'rpos':'pos','hap':'lhap'}), on=['scaf','pos','lhap'], how='left', indicator=True)['_merge'].values == "both", ['lpos','lcon','lstrand','ldist','ldmin','ldmax']] = [-1,-1,'',0,-int(sys.maxsize*0.99),int(sys.maxsize)*0.99]
            supp_reads = BasicMappingToScaffolds(supp_reads, all_scafs)
            # Only keep supp_reads that still map to both sides of a the unsupported connection
            lreads = unsupp_conns[['lcon']].reset_index().merge(supp_reads[['conpart','read_name']].rename(columns={'conpart':'lcon'}), on=['lcon'], how='inner')
            rreads = unsupp_conns[['rcon']].reset_index().merge(supp_reads[['conpart','read_name']].rename(columns={'conpart':'rcon'}), on=['rcon'], how='inner')
            supp_reads = lreads.drop(columns=['lcon']).merge(rreads.drop(columns=['rcon']), on=['index','read_name'], how='inner')[['read_name']].drop_duplicates().merge(supp_reads, on=['read_name'], how='inner')
            supp_reads.sort_values(['read_name','read_start','read_pos'], inplace=True)
            # Remove supp_reads, where read_pos have been filtered out between both sides of the unsupported connection
            
            # supp_reads[supp_reads['scaf'] == 214].drop(columns=['read_start','read_end','read_from','read_to','con_from','con_to','matches','lmatches','rmatches'])
            # unsupp_conns[unsupp_conns['scaf'] == 214]
            # scaffold_paths[scaffold_paths['scaf'] == 214].head(30).tail(10)
            
            # Do something about connected alternatives and read truncation
            
            # supp_reads[supp_reads['scaf'] == 59].drop(columns=['read_start','read_end','read_from','read_to','con_from','con_to','matches','lmatches','rmatches'])
            # unsupp_conns[unsupp_conns['scaf'] == 59]
            # scaffold_paths[scaffold_paths['scaf'] == 59].head(15).tail(15)
            
            # Do something about connected haplotypes, where only one is bad
            
            # supp_reads[supp_reads['scaf'] == 3043].drop(columns=['read_start','read_end','read_from','read_to','con_from','con_to','matches','lmatches','rmatches'])
            # unsupp_conns[unsupp_conns['scaf'] == 3043]
            # scaffold_paths[scaffold_paths['scaf'] == 3043].head(10)
            
            # Remove unsupp_conns, where the difference between haplotypes is only the distance and one is supported
            
            # supp_reads[supp_reads['scaf'] == 266].drop(columns=['read_start','read_end','read_from','read_to','con_from','con_to','matches','lmatches','rmatches'])
            # unsupp_conns[unsupp_conns['scaf'] == 266]
            # scaffold_paths[scaffold_paths['scaf'] == 266].head(10)
        
        # Put back duplicated reads, where otherwise no read support exists for the connection
        dcons = conn_cov[(conn_cov['ucov'] == 0) & (conn_cov['cov'] > 0)].drop(columns=['cov','ucov']).reset_index(drop=True).reset_index().rename(columns={'index':'ci'})
        dups_maps['lci'] = dups_maps[['scaf','pos','lhap']].merge(dcons[['scaf','rpos','hap','ci']].rename(columns={'rpos':'pos','hap':'lhap'}), on=['scaf','pos','lhap'], how='left')['ci'].fillna(-1).astype(int).values
        dups_maps['rci'] = dups_maps[['scaf','pos','rhap']].merge(dcons[['scaf','lpos','hap','ci']].rename(columns={'lpos':'pos','hap':'rhap'}), on=['scaf','pos','rhap'], how='left')['ci'].fillna(-1).astype(int).values
        dups_maps.loc[dups_maps['lcon'] < 0, 'lci'] = -1
        dups_maps.loc[dups_maps['rcon'] < 0, 'rci'] = -1
        dups_maps = dups_maps.loc[(dups_maps['lci'] >= 0) | (dups_maps['rci'] >= 0), :]
        # Duplicate entries that have a left and a right connection that can only be filled with a duplicated read
        dups_maps.sort_values(['read_name','read_start','group','read_pos','scaf','pos','lhap','rhap'], inplace=True)
        dups_maps.reset_index(inplace=True)
        dups_maps = dups_maps.loc[np.repeat(dups_maps.index.values, (dups_maps['lci'] >= 0).values.astype(int) + (dups_maps['rci'] >= 0).values.astype(int))].reset_index(drop=True)
        dups_maps.loc[dups_maps['index'] == dups_maps['index'].shift(1), 'lci'] = -1
        dups_maps.loc[dups_maps['index'] == dups_maps['index'].shift(-1), 'rci'] = -1
        dups_maps.drop(columns=['index'], inplace=True)
        # Only keep the groups for a ci that covers both sides
        dups_maps = dups_maps[ (dups_maps['lci'] < 0) | (dups_maps[['group','lci']].rename(columns={'lci':'rci'}).merge(dups_maps[['group','rci']].drop_duplicates(), on=['group','rci'], how='left', indicator=True)['_merge'].values == "both") ].copy()
        dups_maps = dups_maps[ (dups_maps['rci'] < 0) | (dups_maps[['group','rci']].rename(columns={'rci':'lci'}).merge(dups_maps[['group','lci']].drop_duplicates(), on=['group','lci'], how='left', indicator=True)['_merge'].values == "both") ].copy()
        # Choose the reads with the lowest amount of duplication for the connections
        for col in ['lci','rci']:
            dups_maps.sort_values([col], inplace=True)
            mcount = dups_maps.groupby([col], sort=False)['gcount'].agg(['min','size'])
            dups_maps.loc[dups_maps['gcount'] > np.repeat(mcount['min'].values, mcount['size'].values), col] = -1
        dups_maps = dups_maps[(dups_maps['lci'] >= 0) | (dups_maps['rci'] >= 0)].drop(columns=['gcount'])
        # Remove the unused connections, such that we do not use them for other connections
        dups_maps.loc[dups_maps['lci'] < 0, ['lcon','ldist','lmapq','lmatches']] = [-1,0,-1,-1]
        dups_maps.loc[dups_maps['rci'] < 0, ['rcon','rdist','rmapq','rmatches']] = [-1,0,-1,-1]
        # In rare cases a read can map on the forward and reverse strand, chose one of those arbitrarily
        dups_maps['ci'] = dups_maps[['lci','rci']].max(axis=1) # Only one is != -1 at this point
        dups_maps.drop(columns=['lci','rci'], inplace=True)
        dups_maps.sort_values(['read_name','read_start','group','ci'], inplace=True)
        tmp = dups_maps.groupby(['read_name','read_start','group','ci'], sort=False).size().reset_index(name='mappings')
        if np.sum(tmp['mappings'] != 2):
            print("Error: A connection between two mappings of a read has more or less than two mappings associated.")
            print(tmp[tmp['mappings'] != 2])
        tmp = tmp.drop(columns=['mappings']).groupby(['read_name','read_start','ci'], sort=False).first().reset_index() # Take here the group with the lower id to resolve the strand conflict
        dups_maps = dups_maps.merge(tmp, on=['read_name','read_start','group','ci'], how='inner')
        # Trim the length of the duplicated reads, such taht we do not use them for extending
        tmp = dups_maps.groupby(['read_name','read_start','group','ci'], sort=False).agg({'read_from':['min'], 'read_to':['max']})
        dups_maps['read_start'] = np.repeat( tmp['read_from','min'].values, 2 )
        dups_maps['read_end'] = np.repeat( tmp['read_to','max'].values, 2 )
        # Assing a map id to uniquely identify the individual mapping groups
        dups_maps.sort_values(['group','ci'], inplace=True)
        dups_maps['mapid'] = ((dups_maps['group'] != dups_maps['group'].shift(1)) | (dups_maps['ci'] != dups_maps['ci'].shift(1))).cumsum() + mappings['mapid'].max()
        dups_maps.drop(columns=['group','ci'], inplace=True)
        # Update read positions
        dups_maps.sort_values(['read_name','read_start','mapid','read_pos'], inplace=True)
        dups_maps['read_pos'] = dups_maps.groupby(['read_name','read_start','mapid'], sort=False).cumcount()
        # Merge unique and duplicated mappings
        mappings = pd.concat([mappings, dups_maps], ignore_index=True).sort_values(['read_name','read_start','mapid','read_pos'], ignore_index=True)
        mappings['mapid'] = (mappings['mapid'] != mappings['mapid'].shift(1)).cumsum()-1
#
        # Break connections where they are not supported by reads even with multi mapping reads and after fixing attemps(should never happen, so give a warning)
        if len(conn_cov[conn_cov['cov'] == 0]) == 0:
            break
        else:
            print( len(conn_cov[conn_cov['cov'] == 0]), "gaps were created for which no read for filling can be found. The connections will be broken up again.")
            # Start with alternative paths
            for h in range(1,ploidy):
                rem = conn_cov.loc[(conn_cov['cov'] == 0) & (conn_cov['hap'] == h), ['scaf','rpos']].rename(columns={'rpos':'pos'}).merge(scaffold_paths[['scaf','pos',f'phase{h}']], on=['scaf','pos'], how='inner')[f'phase{h}'].values
                rem = np.isin(scaffold_paths[f'phase{h}'], rem)
                scaffold_paths.loc[rem, [f'con{h}',f'strand{h}',f'dist{h}']] = [-1,'',0]
                scaffold_paths.loc[rem, f'phase{h}'] = -scaffold_paths.loc[rem, f'phase{h}']
            # Continue with main paths that has alternatives
            rem_conns = conn_cov[(conn_cov['cov'] == 0) & (conn_cov['hap'] == 0)].copy()
            for h in range(1,ploidy):
                rem = rem_conns[['scaf','rpos']].reset_index().rename(columns={'rpos':'pos'}).merge(scaffold_paths.loc[scaffold_paths[f'phase{h}'] >= 0, ['scaf','pos',f'phase{h}']], on=['scaf','pos'], how='inner')
                rem_conns.drop(rem['index'].values, inplace=True)
                rem = np.isin(scaffold_paths[f'phase{h}'], rem[f'phase{h}'].values)
                scaffold_paths.loc[rem, ['phase0','con0','strand0','dist0']] = scaffold_paths.loc[rem, [f'phase{h}',f'con{h}',f'strand{h}',f'dist{h}']].values
                scaffold_paths.loc[rem, [f'con{h}',f'strand{h}',f'dist{h}']] = [-1,'',0]
                scaffold_paths.loc[rem, f'phase{h}'] = -scaffold_paths.loc[rem, f'phase{h}']
            # Finally split contig, where we do not have any valid connection
            scaffold_paths['split'] = scaffold_paths[['scaf','pos']].merge(rem_conns[['scaf','rpos']].rename(columns={'rpos':'pos'}), on=['scaf','pos'], how='left', indicator=True)['_merge'].values == "both"
            scaffold_paths['scaf'] = ((scaffold_paths['scaf'] != scaffold_paths['scaf'].shift(1)) | scaffold_paths['split']).cumsum()-1
            scaffold_paths.drop(columns=['split'], inplace=True)
            # Clean up
            scaffold_paths = scaffold_paths[(scaffold_paths[[f'con{h}' for h in range(ploidy)]] == -1).all(axis=1) == False].copy()
            scaffold_paths['pos'] = scaffold_paths.groupby(['scaf'], sort=False).cumcount()
            scaffold_paths.loc[scaffold_paths['pos'] == 0, 'dist0'] = 0
            mappings = org_mappings.copy()
#
    ## Handle circular scaffolds
    mcols = [f'con{h}' for h in range(ploidy)]+[f'strand{h}' for h in range(ploidy)]
    mappings[mcols] = mappings[['scaf']].merge(scaffold_paths.loc[scaffold_paths['pos'] == 0, ['scaf']+mcols], on=['scaf'], how='left')[mcols].values
    mappings.loc[(mappings['rpos'] >= 0) | (mappings['rcon'] < 0), [f'con{h}' for h in range(ploidy)]] = -1
    # Check if the read continues at the beginning of the scaffold
    mappings['check_pos'] = mappings['read_pos'] + np.where(mappings['strand'] == '+', 1, -1)
    mappings['rmapid'] = mappings[['read_name','read_start','scaf','strand','check_pos']].merge(mappings.loc[mappings['pos'] == 0, ['read_name','read_start','scaf','strand','read_pos','mapid']].rename(columns={'read_pos':'check_pos'}).drop_duplicates(), on=['read_name','read_start','scaf','strand','check_pos'], how='left')['mapid'].values
    mappings.loc[np.isnan(mappings['rmapid']), [f'con{h}' for h in range(ploidy)]] = -1
    mappings['rmapid'] = mappings['rmapid'].fillna(-1).astype(int)
    mappings.drop(columns=['check_pos'], inplace=True)
    # Check available bridges
    mappings['index'] = mappings.index.values
    for h in range(ploidy):
        mappings.loc[(mappings[f'con{h}'] != mappings['rcon']), f'con{h}'] = -1
        mappings = mappings.merge(bridges[['from','from_side','to','to_side','min_dist','max_dist','mean_dist']].rename(columns={'from':'conpart','to':f'con{h}','mean_dist':f'rmdist{h}'}), on=['conpart',f'con{h}'], how='left')
        mappings.loc[mappings['from_side'].isnull() | ((mappings['from_side'] == 'r') != (mappings['con_strand'] == '+')) | ((mappings['to_side'] == 'l') != (mappings[f'strand{h}'] == '+')), f'con{h}'] = -1
        mappings.loc[(mappings['rdist'] < mappings['min_dist']) | (mappings['rdist'] > mappings['max_dist']), f'con{h}'] = -1
        mappings.drop(columns=[f'strand{h}','from_side','to_side','min_dist','max_dist'], inplace=True)
        mappings[f'rmdist{h}'] = mappings[f'rmdist{h}'].fillna(0).astype(int)
        mappings.sort_values(['index',f'con{h}',f'rmdist{h}'], inplace=True)
        mappings = mappings.groupby(['index'], sort=False).last().reset_index()
    mappings.drop(columns=['index'], inplace=True)
    mappings.sort_values(['mapid','read_pos','scaf','pos','lhap','rhap'], inplace=True)
    # Summarize the circular connections and filter
    circular = pd.concat([mappings.loc[mappings[f'con{h}'] >= 0, ['scaf','pos','rhap',f'rmdist{h}']].rename(columns={f'rmdist{h}':'rmdist'}) for h in range(ploidy)], ignore_index=True)
    circular['lhap'] = np.repeat([h for h in range(ploidy)], [np.sum(mappings[f'con{h}'] >= 0) for h in range(ploidy)])
    circular = circular.groupby(['scaf','pos','rhap','lhap','rmdist']).size().reset_index(name='count')
    tmp = circular.groupby(['scaf'])['count'].agg(['sum','size'])
    circular['tot'] = np.repeat(tmp['sum'].values, tmp['size'].values)
    # Filter circular connections with more than ploidy options
    circular = circular[np.repeat(tmp['size'].values<=ploidy, tmp['size'].values)].copy()
    # Filter scaffolds where non circular options are dominant
    circular = circular.merge(circular[['scaf','pos']].drop_duplicates().merge(mappings.loc[(mappings['rcon'] >= 0) & (mappings[[f'con{h}' for h in range(ploidy)]] < 0).all(axis=1), ['scaf','pos']], on=['scaf','pos'], how='inner').groupby(['scaf','pos']).size().reset_index(name='veto'), on=['scaf','pos'], how='left')
    circular = circular[circular['count'] > circular['veto']].copy()
    # If we have multiple haplotypes take the one with more counts (at some point I should use both, so print a warning to find a proper dataset for testing)
    circular.sort_values(['scaf','pos','rhap','count'], inplace=True)
    old_len = len(circular)
    circular = circular.groupby(['scaf','pos','rhap']).last().reset_index()
    if old_len != len(circular):
        print("Warning:", old_len-len(circular), "alternative haplotypes have been removed for the connection of circular scaffolds. This case still needs proper handling.")
    # Add information to accepted circular mappings
    mappings[['hap','rmdist']] = mappings[['scaf','pos','rhap']].merge(circular[['scaf','pos','rhap','lhap','rmdist']].rename(columns={'lhap':'hap'}), on=['scaf','pos','rhap'], how='left')[['hap','rmdist']].values
    mappings['circular'] = False
    for h in range(ploidy):
        mappings.loc[(mappings['hap'] == h) & (mappings[f'rmdist{h}'] == mappings['rmdist']) & (mappings[f'con{h}'] >= 0), 'circular'] = True
    mappings.loc[mappings['circular'], 'rpos'] = 0
    mappings.loc[mappings['circular'] == False, 'rmapid'] = mappings.loc[mappings['circular'] == False, 'mapid']
    mappings.drop(columns=[f'con{h}' for h in range(ploidy)] + [f'rmdist{h}' for h in range(ploidy)] + ['hap','rmdist','circular'], inplace=True)
    mappings['check_pos'] = mappings['read_pos'] + np.where(mappings['strand'] == '+', 1, -1)
    mappings['circular'] = mappings[['read_name','read_start','read_pos','scaf','pos','strand','conpart','lcon','ldist','mapid']].merge(mappings.loc[mappings['rpos'] == 0, ['read_name','read_start','scaf','strand','check_pos','rpos','conpart','rcon','rdist','rmapid','pos']].rename(columns={'check_pos':'read_pos','rpos':'pos','conpart':'lcon','rcon':'conpart','rdist':'ldist','rmapid':'mapid','pos':'circular'}), on=['read_name','read_start','read_pos','scaf','pos','strand','conpart','lcon','ldist','mapid'], how='left')['circular'].values
    mappings.loc[np.isnan(mappings['circular']) == False, 'lpos'] = mappings.loc[np.isnan(mappings['circular']) == False, 'circular'].astype(int)
    mappings.drop(columns=['check_pos','circular'], inplace=True)
#
    return mappings, scaffold_paths

def CountResealedBreaks(result_info, scaffold_paths, contig_parts, ploidy):
    ## Get resealed breaks (Number of total contig_parts - minimum number of scaffold chunks needed to have them all included in the proper order)
    broken_contigs = contig_parts.reset_index().rename(columns={'index':'conpart'}).loc[(contig_parts['part'] > 0) | (contig_parts['part'].shift(-1) > 0), ['contig','part','conpart']]
    # For this treat haplotypes as different scaffolds
    resealed_breaks = scaffold_paths.loc[np.repeat(scaffold_paths.index.values, ploidy), ['scaf','pos']+[f'{n}{h}' for h in range(ploidy) for n in ['phase','con','strand']]]
    resealed_breaks['hap'] = list(range(ploidy))*len(scaffold_paths)
    resealed_breaks.rename(columns={'con0':'con','strand0':'strand'}, inplace=True)
    for h in range(1,ploidy):
        change = (resealed_breaks['hap'] == h) & (resealed_breaks[f'phase{h}'] >= 0)
        resealed_breaks.loc[change, ['con','strand']] = resealed_breaks.loc[change, [f'con{h}',f'strand{h}']].values
    resealed_breaks['scaf'] = resealed_breaks['scaf']*ploidy + resealed_breaks['hap']
    resealed_breaks.drop(columns=['hap','phase0'] + [f'{n}{h}' for h in range(1,ploidy) for n in ['phase','con','strand']], inplace=True)
    resealed_breaks = resealed_breaks[resealed_breaks['con'] >= 0].copy()
    resealed_breaks.sort_values(['scaf','pos'], inplace=True)
    resealed_breaks['pos'] = resealed_breaks.groupby(['scaf']).cumcount()
    # Reduce scaffolds to the chunks of connected broken contig parts
    resealed_breaks = resealed_breaks[np.isin(resealed_breaks['con'].values, broken_contigs['conpart'].values)].copy()
    resealed_breaks['chunk'] = ( (resealed_breaks['scaf'] != resealed_breaks['scaf'].shift(1)) | (resealed_breaks['strand'] != resealed_breaks['strand'].shift(1)) |
                                 (resealed_breaks['pos'] != resealed_breaks['pos'].shift(1)+1) | (resealed_breaks['con'] != resealed_breaks['con'].shift(1)+np.where(resealed_breaks['strand'] == '-', -1, 1))).cumsum()
    resealed_breaks = resealed_breaks.groupby(['chunk'], sort=False)['con'].agg(['min','max']).reset_index(drop=True)
    resealed_breaks = resealed_breaks[resealed_breaks['min'] < resealed_breaks['max']].sort_values(['min','max']).drop_duplicates()
    resealed_breaks['group'] = (resealed_breaks['min'] > resealed_breaks['max'].shift(1)).cumsum()
    resealed_breaks['length'] = resealed_breaks['max'] - resealed_breaks['min'] + 1
    # Count resealed parts
    resealed_parts = resealed_breaks.loc[np.repeat(resealed_breaks.index.values, resealed_breaks['length'].values)].reset_index()[['index','min']]
    resealed_parts['conpart'] = resealed_parts['min'] + resealed_parts.groupby(['index'], sort=False).cumcount()
    # num_resealed = Number of total contig_parts - number of contig_parts not sealed at all - number of scaffold chunks uniquely sealing contigs
    num_resealed = len(broken_contigs) - len(np.setdiff1d(broken_contigs['conpart'].values, resealed_parts['conpart'].values)) - np.sum((resealed_breaks['group'] != resealed_breaks['group'].shift(1)) & (resealed_breaks['group'] != resealed_breaks['group'].shift(-1)))

    # The non-unqiue sealing events still have to be processed
    resealed_breaks = resealed_breaks[(resealed_breaks['group'] == resealed_breaks['group'].shift(1)) | (resealed_breaks['group'] == resealed_breaks['group'].shift(-1))].copy()
    resealed_breaks = resealed_breaks.reset_index(drop=True).reset_index()
    resealed_breaks['keep'] = False
    while np.sum(False == resealed_breaks['keep']):
        resealed_parts = resealed_breaks.loc[np.repeat(resealed_breaks.index.values, resealed_breaks['length'].values)].reset_index()[['index','min']]
        resealed_parts['conpart'] = resealed_parts['min'] + resealed_parts.groupby(['index'], sort=False).cumcount()
        resealed_parts = resealed_parts.groupby('conpart')['index'].agg(['first','size'])
        resealed_breaks['keep'] = np.isin(resealed_breaks['index'], resealed_parts.loc[resealed_parts['size'] == 1, ['first']].reset_index()['first'].values) # Keep all chunks with unique contig parts
        resealed_breaks.sort_values(['group','keep','length'], ascending=[True,False,False], inplace=True)
        resealed_breaks = resealed_breaks[resealed_breaks['keep'] | (resealed_breaks['group'] == resealed_breaks['group'].shift(-1))].copy() # Remove shortest non-unqiue chunk in each group

    result_info['breaks']['resealed'] = num_resealed - len(resealed_breaks)

    return result_info

def FillGapsWithReads(scaffold_paths, mappings, contig_parts, ploidy, max_dist_contig_end, min_extension, min_num_reads, pdf):
    ## Find best reads to fill into gaps
    possible_reads = mappings.loc[(mappings['rcon'] >= 0) & (mappings['rpos'] >= 0), ['scaf','pos','rhap','rpos','read_pos','read_name','read_start','read_from','read_to','strand','rdist','mapq','rmapq','matches','rmatches','con_from','con_to','rmapid']].sort_values(['scaf','pos','rhap'])
#
    # First take the one with the highest mapping qualities on both sides
    possible_reads['cmapq'] = np.minimum(possible_reads['mapq'], possible_reads['rmapq'])*1000 + np.maximum(possible_reads['mapq'], possible_reads['rmapq'])
    tmp = possible_reads.groupby(['scaf','pos','rhap'], sort=False)['cmapq'].agg(['max','size'])
    possible_reads = possible_reads[possible_reads['cmapq'] == np.repeat(tmp['max'].values, tmp['size'].values)].copy()
    possible_reads.drop(columns=['cmapq','mapq','rmapq'], inplace=True)
#
    # Then take the one the closest to the mean distance to get the highest chance of the other reads mapping to it later for the consensus
    tmp = possible_reads.groupby(['scaf','pos','rhap'], sort=False)['rdist'].agg(['mean','size'])
    possible_reads['dist_diff'] = np.abs(possible_reads['rdist'] - np.repeat(tmp['mean'].values, tmp['size'].values))
    tmp = possible_reads.groupby(['scaf','pos','rhap'], sort=False)['dist_diff'].agg(['min','size'])
    possible_reads = possible_reads[possible_reads['dist_diff'] == np.repeat(tmp['min'].values, tmp['size'].values)].copy()
    possible_reads.drop(columns=['dist_diff'], inplace=True)
#
    # Use matches as a final tie-breaker
    possible_reads['cmatches'] = np.minimum(possible_reads['matches'], possible_reads['rmatches'])
    tmp = possible_reads.groupby(['scaf','pos','rhap'], sort=False)['cmatches'].agg(['max','size'])
    possible_reads = possible_reads[possible_reads['cmatches'] == np.repeat(tmp['max'].values, tmp['size'].values)].copy()
    possible_reads['cmatches'] = np.maximum(possible_reads['matches'], possible_reads['rmatches'])
    tmp = possible_reads.groupby(['scaf','pos','rhap'], sort=False)['cmatches'].agg(['max','size'])
    possible_reads = possible_reads[possible_reads['cmatches'] == np.repeat(tmp['max'].values, tmp['size'].values)].copy()
    possible_reads.drop(columns=['cmatches','matches','rmatches'], inplace=True)
#
    # If everything is equally good just take the first
    possible_reads = possible_reads.groupby(['scaf','pos','rhap'], sort=False).first().reset_index()
#
    # Get the read in the gap instead of the read on the mapping
    possible_reads.loc[possible_reads['strand'] == '+', 'read_from'] = possible_reads.loc[possible_reads['strand'] == '+', 'read_to']
    possible_reads.loc[possible_reads['strand'] == '+', 'read_to'] = possible_reads.loc[possible_reads['strand'] == '+', 'read_from'] + possible_reads.loc[possible_reads['strand'] == '+', 'rdist']
    possible_reads.loc[possible_reads['strand'] == '-', 'read_to'] = possible_reads.loc[possible_reads['strand'] == '-', 'read_from']
    possible_reads.loc[possible_reads['strand'] == '-', 'read_from'] = possible_reads.loc[possible_reads['strand'] == '-', 'read_to'] - possible_reads.loc[possible_reads['strand'] == '-', 'rdist']
    possible_reads.loc[possible_reads['rdist'] <= 0, ['read_from','read_to']] = 0
    possible_reads.drop(columns=['rdist'], inplace=True)
#
    # Get the mapping information for the other side of the gap (chap is the haplotype that it connects to on the other side, for circular scaffolds it is not guaranteed to be identical with rhap)
    possible_reads['read_pos'] += np.where(possible_reads['strand'] == '+', 1, -1)
    possible_reads[['rcon_from','rcon_to','chap']] = possible_reads[['read_name','read_start','read_pos','scaf','pos','rpos','rmapid']].merge(mappings[['read_name','read_start','read_pos','scaf','pos','lpos','con_from','con_to','lhap','mapid']].rename(columns={'pos':'rpos','lpos':'pos','con_from':'rcon_from','con_to':'rcon_to','lhap':'chap','mapid':'rmapid'}), on=['read_name','read_start','read_pos','scaf','pos','rpos','rmapid'], how='left')[['rcon_from','rcon_to','chap']].astype(int).values
#
    ## Fill scaffold_paths with read and contig information
    for h in range(ploidy):
        # Insert contig information
        scaffold_paths = scaffold_paths.merge(contig_parts[['name','start','end']].reset_index().rename(columns={'index':f'con{h}','name':f'name{h}','start':f'start{h}','end':f'end{h}'}), on=[f'con{h}'], how='left')
        # Insert read information
        scaffold_paths = scaffold_paths.merge(possible_reads.loc[possible_reads['rhap'] == h, ['scaf','pos','read_name','read_from','read_to','strand','con_from','con_to']].rename(columns={'read_name':f'read_name{h}','read_from':f'read_from{h}','read_to':f'read_to{h}','strand':f'read_strand{h}','con_from':f'con_from{h}','con_to':f'con_to{h}'}), on=['scaf','pos'], how='left')
        scaffold_paths = scaffold_paths.merge(possible_reads.loc[possible_reads['chap'] == h, ['scaf','rpos','rcon_from','rcon_to']].rename(columns={'rpos':'pos','rcon_from':f'rcon_from{h}','rcon_to':f'rcon_to{h}'}), on=['scaf','pos'], how='left')
    # Split into contig and read part and apply information
    scaffold_paths = scaffold_paths.loc[np.repeat(scaffold_paths.index.values, 1+(scaffold_paths[[f'read_name{h}' for h in range(ploidy)]].isnull().all(axis=1) == False) )].reset_index()
    scaffold_paths['type'] = np.where(scaffold_paths.groupby(['index'], sort=False).cumcount() == 0, 'contig','read')
    scaffold_paths.drop(columns=['index'],inplace=True)
    for h in range(ploidy):
        scaffold_paths.loc[scaffold_paths['type'] == 'read', [f'name{h}',f'start{h}',f'end{h}',f'strand{h}']] = scaffold_paths.loc[scaffold_paths['type'] == 'read', [f'read_name{h}',f'read_from{h}',f'read_to{h}',f'read_strand{h}']].values
        scaffold_paths.drop(columns=[f'con{h}',f'dist{h}',f'read_name{h}',f'read_from{h}',f'read_to{h}',f'read_strand{h}'], inplace=True)
        scaffold_paths[[f'name{h}',f'strand{h}']] = scaffold_paths[[f'name{h}',f'strand{h}']].fillna('')
        scaffold_paths[[f'start{h}',f'end{h}']] = scaffold_paths[[f'start{h}',f'end{h}']].fillna(0).astype(int)
        # Trim contigs, where the reads stop mapping on both sides of the gap
        for side in ['','r']:
            trim = (scaffold_paths['type'] == 'contig') & (scaffold_paths[f'{side}con_from{h}'].isnull() == False)
            get_main = trim & (scaffold_paths[f'phase{h}'] < 0)
            if np.sum(get_main):
                scaffold_paths.loc[get_main, [f'name{h}',f'start{h}',f'end{h}',f'strand{h}']] = scaffold_paths.loc[get_main, ['name0','start0','end0','strand0']].values
                scaffold_paths.loc[get_main, f'phase{h}'] = -scaffold_paths.loc[get_main, f'phase{h}']
            trim_strand = trim & (scaffold_paths[f'strand{h}'] == ('-' if side=='r' else '+'))
            scaffold_paths.loc[trim_strand, f'end{h}'] = scaffold_paths.loc[trim_strand, f'{side}con_to{h}'].astype(int)
            trim_strand = trim & (scaffold_paths[f'strand{h}'] == ('+' if side=='r' else '-'))
            scaffold_paths.loc[trim_strand, f'start{h}'] = scaffold_paths.loc[trim_strand, f'{side}con_from{h}'].astype(int)
            scaffold_paths.drop(columns=[f'{side}con_to{h}',f'{side}con_from{h}'],inplace=True)
    # Remove reads that fill a negative gap length (so do not appear)
    for h in range(ploidy):
        scaffold_paths.loc[scaffold_paths[f'start{h}'] >= scaffold_paths[f'end{h}'], [f'name{h}',f'strand{h}']] = ''
    scaffold_paths = scaffold_paths.loc[(scaffold_paths[[f'name{h}' for h in range(ploidy)]] != '').any(axis=1), ['scaf','pos','type']+[f'{n}{h}' for h in range(ploidy) for n in ['phase','name','start','end','strand']]+['sdist_left','sdist_right']].copy()
    # Reads get the phase of the next contig
    for h in range(ploidy):
        scaffold_paths[f'phase{h}'] = np.where((scaffold_paths['type'] == 'read') & (scaffold_paths['scaf'] == scaffold_paths['scaf'].shift(-1)), scaffold_paths[f'phase{h}'].shift(-1, fill_value=0), scaffold_paths[f'phase{h}'])
        scaffold_paths.loc[scaffold_paths['type'] == 'read', f'phase{h}'] = np.abs(scaffold_paths.loc[scaffold_paths['type'] == 'read', f'phase{h}'])
    # Split alternative contigs that are identical to main into the overlapping part and the non-overlapping part
    scaffold_paths['smin'] = scaffold_paths['start0']
    scaffold_paths['smax'] = scaffold_paths['start0']
    scaffold_paths['emin'] = scaffold_paths['end0']
    scaffold_paths['emax'] = scaffold_paths['end0']
    for h in range(1,ploidy):
        identical = (scaffold_paths[f'name{h}'] == scaffold_paths['name0']) & (scaffold_paths[f'strand{h}'] == scaffold_paths['strand0']) & (scaffold_paths[f'phase{h}'] >= 0)
        if np.sum(identical):
            scaffold_paths.loc[identical, 'smin'] = scaffold_paths.loc[identical, ['smin',f'start{h}']].min(axis=1)
            scaffold_paths.loc[identical, 'smax'] = scaffold_paths.loc[identical, ['smax',f'start{h}']].max(axis=1)
            scaffold_paths.loc[identical, 'emin'] = scaffold_paths.loc[identical, ['emin',f'end{h}']].min(axis=1)
            scaffold_paths.loc[identical, 'emax'] = scaffold_paths.loc[identical, ['emax',f'end{h}']].max(axis=1)
    scaffold_paths.loc[(scaffold_paths['type'] == 'read') | (scaffold_paths['smax']>scaffold_paths['emin']), ['smin','smax','emin','emax']] = -1 # If we do not have an overlap in the middle we cannot use the split approach and for the reads we should not have identical reads for different haplotypes
    scaffold_paths['bsplit'] = np.where(scaffold_paths['strand0'] == '+', scaffold_paths['smin'] != scaffold_paths['smax'], scaffold_paths['emin'] != scaffold_paths['emax']) # Split before
    scaffold_paths['asplit'] = np.where(scaffold_paths['strand0'] == '+', scaffold_paths['emin'] != scaffold_paths['emax'], scaffold_paths['smin'] != scaffold_paths['smax']) # Split after
    scaffold_paths = scaffold_paths.loc[np.repeat(scaffold_paths.index.values,1+scaffold_paths['bsplit']+scaffold_paths['asplit'])].reset_index()
    scaffold_paths['split'] = scaffold_paths.groupby(['index'], sort=False).cumcount()+(scaffold_paths['bsplit'] == False)
    for h in range(ploidy-1,-1,-1):
        identical = (scaffold_paths[f'name{h}'] == scaffold_paths['name0']) & (scaffold_paths[f'strand{h}'] == scaffold_paths['strand0']) & (scaffold_paths[f'phase{h}'] >= 0)
        scaffold_paths.loc[(identical == False) & (scaffold_paths['split'] != 1), [f'name{h}',f'start{h}',f'end{h}',f'strand{h}']] = ['',0,0,''] # For the ones that we do not split, only keep the center
        mod = identical & (scaffold_paths['split'] == 0) & (scaffold_paths['strand0'] == '+')
        scaffold_paths.loc[mod, f'end{h}'] = scaffold_paths.loc[mod, 'smax']
        mod = identical & (scaffold_paths['split'] == 0) & (scaffold_paths['strand0'] == '-')
        scaffold_paths.loc[mod, f'start{h}'] = scaffold_paths.loc[mod, 'emin']
        mod = identical & (scaffold_paths['split'] == 1) & (scaffold_paths['smin'] != scaffold_paths['smax'])
        scaffold_paths.loc[mod, f'start{h}'] = scaffold_paths.loc[mod, 'smax']
        mod = identical & (scaffold_paths['split'] == 1) & (scaffold_paths['emin'] != scaffold_paths['emax'])
        scaffold_paths.loc[mod, f'end{h}'] = scaffold_paths.loc[mod, 'emin']
        mod = identical & (scaffold_paths['split'] == 2) & (scaffold_paths['strand0'] == '+')
        scaffold_paths.loc[mod, f'start{h}'] = scaffold_paths.loc[mod, 'emin']
        mod = identical & (scaffold_paths['split'] == 2) & (scaffold_paths['strand0'] == '-')
        scaffold_paths.loc[mod, f'end{h}'] = scaffold_paths.loc[mod, 'smax']
        scaffold_paths.loc[scaffold_paths[f'start{h}'] >= scaffold_paths[f'end{h}'], [f'name{h}',f'start{h}',f'end{h}',f'strand{h}']] = ['',0,0,'']
        # The end of the split contig gets the phase of the read or the next contig if the read was removed
        scaffold_paths[f'phase{h}'] = np.where((scaffold_paths['split'] == 2) & (scaffold_paths['scaf'] == scaffold_paths['scaf'].shift(-1)), np.abs(scaffold_paths[f'phase{h}'].shift(-1, fill_value=0)), scaffold_paths[f'phase{h}'])
    scaffold_paths.drop(columns=['index','smin','smax','emin','emax','bsplit','asplit','split'], inplace=True)
    # Remove alternative contigs, where they are identical to main
    for h in range(1,ploidy):
        identical = (scaffold_paths[[f'name{h}',f'start{h}',f'end{h}',f'strand{h}']] == scaffold_paths[['name0','start0','end0','strand0']].values).all(axis=1) & (scaffold_paths[f'phase{h}'] >= 0)
        scaffold_paths.loc[identical, [f'name{h}',f'start{h}',f'end{h}',f'strand{h}']] = ['',0,0,'']
        scaffold_paths.loc[identical, f'phase{h}'] = -scaffold_paths.loc[identical, f'phase{h}']
    # Update positions in scaffold to accomodate the reads and split contigs
    scaffold_paths['pos'] = scaffold_paths.groupby(['scaf'], sort=False).cumcount()
    # Move sdist_left and sdist_right to first or last entry of a scaffold only
    tmp = scaffold_paths[['scaf','sdist_left']].groupby(['scaf']).max().values
    scaffold_paths['sdist_left'] = -1
    scaffold_paths.loc[scaffold_paths['pos'] == 0, 'sdist_left'] = tmp
    tmp = scaffold_paths[['scaf','sdist_right']].groupby(['scaf']).max().values
    scaffold_paths['sdist_right'] = -1
    scaffold_paths.loc[scaffold_paths['scaf'] != scaffold_paths['scaf'].shift(-1), 'sdist_right'] = tmp
#
    ## Remove mappings that are not needed for scaffold extensions into gaps and update the rest
    mappings.drop(columns=['lmapq','rmapq','lmatches','rmatches','read_pos','pos'],inplace=True)
    circular = np.unique(mappings.loc[mappings['rpos'] == 0, 'scaf'].values) # Get them now, because the next step removes evidence, but remove them later to use the expensive np.isin() step on less entries
    mappings = mappings[(mappings['rpos'] < 0) | (mappings['lpos'] < 0)].copy() # Only keep mappings to ends of scaffolds
    mappings = mappings[np.isin(mappings['scaf'], circular) == False].copy() # We do not extend circular scaffolds
    # Check how much we extend in both directions (if extensions are negative set them to 0, such that min_extension==0 extends everything)
    mappings['con_start'] = contig_parts.iloc[mappings['conpart'].values, contig_parts.columns.get_loc('start')].values
    mappings['con_end'] = contig_parts.iloc[mappings['conpart'].values, contig_parts.columns.get_loc('end')].values
    mappings['ltrim'] = np.where('+' == mappings['con_strand'], mappings['con_from']-mappings['con_start'], mappings['con_end']-mappings['con_to'])
    mappings['rtrim'] = np.where('-' == mappings['con_strand'], mappings['con_from']-mappings['con_start'], mappings['con_end']-mappings['con_to'])
    mappings.drop(columns=['conpart','con_start','con_end','con_from','con_to','con_strand'],inplace=True)
    mappings['lext'] = np.maximum(0, np.where('+' == mappings['strand'], mappings['read_from']-mappings['read_start'], mappings['read_end']-mappings['read_to']) - mappings['ltrim'])
    mappings['rext'] = np.maximum(0, np.where('-' == mappings['strand'], mappings['read_from']-mappings['read_start'], mappings['read_end']-mappings['read_to']) - mappings['rtrim'])
    # We do not extend in the direction the scaffold continues
    mappings.loc[mappings['lpos'] >= 0, 'lext'] = -1
    mappings.loc[mappings['rpos'] >= 0, 'rext'] = -1
    mappings.drop(columns=['lpos','rpos'],inplace=True)
    # Set extensions to zero that are too far away from the contig_end
    mappings.loc[mappings['ltrim'] > max_dist_contig_end, 'lext'] = -1
    mappings.loc[mappings['rtrim'] > max_dist_contig_end, 'rext'] = -1
    # Only keep long enough extensions
    if pdf:
        extension_lengths = np.concatenate([mappings.loc[0<=mappings['lext'],'lext'], mappings.loc[0<=mappings['rext'],'rext']])
        if len(extension_lengths):
            if np.sum(extension_lengths < 10*min_extension):
                PlotHist(pdf, "Extension length", "# Extensions", np.extract(extension_lengths < 10*min_extension, extension_lengths), threshold=min_extension)
            PlotHist(pdf, "Extension length", "# Extensions", extension_lengths, threshold=min_extension, logx=True)
        del extension_lengths
    mappings = mappings[(mappings['lext'] >= min_extension) | (mappings['rext'] >= min_extension)].copy()
    # Separate mappings by side
    mappings = mappings.loc[np.repeat(mappings.index.values, 2)].reset_index(drop=True)
    mappings['side'] = np.tile(['l','r'], len(mappings)//2)
    mappings[['hap','ext','trim']] = mappings[['rhap','rext','rtrim']].values
    mappings.loc[mappings['side'] == 'l', ['hap','ext','trim']] = mappings.loc[mappings['side'] == 'l', ['lhap','lext','ltrim']].values
    mappings.drop(columns=['lhap','lext','ltrim','rhap','rext','rtrim'],inplace=True)
    mappings = mappings[mappings['ext'] >= min_extension].copy()
    # Get unmapped extension (This part is worth to add as a new scaffold if we do not have a unique extension, because it is not in the genome yet)
    mappings['unmap_ext'] = np.maximum(0,np.where(mappings['side'] == 'l', mappings['ldist'], mappings['rdist'])-mappings['trim'])
    no_connection = np.where(mappings['side'] == 'l', mappings['lcon'], mappings['rcon']) < 0
    mappings.loc[no_connection, 'unmap_ext'] = mappings.loc[no_connection, 'ext']
    mappings = mappings[['scaf','side','hap','ext','trim','unmap_ext','read_name','read_start','read_end','read_from','read_to','strand','mapq','matches']].copy()
    # Only keep extension, when there are enough of them
    mappings.sort_values(['scaf','side','hap'], inplace=True)
    count = mappings.groupby(['scaf','side','hap'], sort=False).size().values
    mappings = mappings[np.repeat(count >= min_num_reads, count)].copy()
#
    return scaffold_paths, mappings

def GetOutputInfo(result_info, scaffold_paths):
    # Calculate lengths
    con_len = scaffold_paths[['scaf', 'start0', 'end0']].copy()
    con_len['length'] = con_len['end0'] - con_len['start0']
    con_len = con_len.groupby('scaf', sort=False)['length'].sum().values

    scaf_len = scaffold_paths[['scaf','sdist_left']].groupby(['scaf'], sort=False)['sdist_left'].max().reset_index()
    scaf_len['length'] = con_len
    scaf_len.loc[scaf_len['sdist_left'] >= 0, 'length'] += scaf_len.loc[scaf_len['sdist_left'] >= 0, 'sdist_left']
    scaf_len['meta'] = (scaf_len['sdist_left'] == -1).cumsum()
    scaf_len = scaf_len.groupby('meta', sort=False)['length'].sum().values

    # Calculate N50
    con_len.sort()
    con_N50, con_total = calculateN50(con_len)

    scaf_len.sort()
    scaf_N50, scaf_total = calculateN50(scaf_len)

    # Store output stats
    result_info['output'] = {}
    result_info['output']['contigs'] = {}
    result_info['output']['contigs']['num'] = len(con_len)
    result_info['output']['contigs']['total'] = con_total
    result_info['output']['contigs']['min'] = con_len[0]
    result_info['output']['contigs']['max'] = con_len[-1]
    result_info['output']['contigs']['N50'] = con_N50

    result_info['output']['scaffolds'] = {}
    result_info['output']['scaffolds']['num'] = len(scaf_len)
    result_info['output']['scaffolds']['total'] = scaf_total
    result_info['output']['scaffolds']['min'] = scaf_len[0]
    result_info['output']['scaffolds']['max'] = scaf_len[-1]
    result_info['output']['scaffolds']['N50'] = scaf_N50

    return result_info

def PrintStats(result_info):
    print("Input assembly:  {:,.0f} contigs   (Total sequence: {:,.0f} Min: {:,.0f} Max: {:,.0f} N50: {:,.0f})".format(result_info['input']['contigs']['num'], result_info['input']['contigs']['total'], result_info['input']['contigs']['min'], result_info['input']['contigs']['max'], result_info['input']['contigs']['N50']))
    print("                 {:,.0f} scaffolds (Total sequence: {:,.0f} Min: {:,.0f} Max: {:,.0f} N50: {:,.0f})".format(result_info['input']['scaffolds']['num'], result_info['input']['scaffolds']['total'], result_info['input']['scaffolds']['min'], result_info['input']['scaffolds']['max'], result_info['input']['scaffolds']['N50']))
    print("Removed          {:,.0f} contigs   (Total sequence: {:,.0f} Min: {:,.0f} Max: {:,.0f} Mean: {:,.0f})".format(result_info['removed']['num'], result_info['removed']['total'], result_info['removed']['min'], result_info['removed']['max'], result_info['removed']['mean']))
    print("Introduced {:,.0f} breaks of which {:,.0f} have been resealed".format(result_info['breaks']['opened'], result_info['breaks']['resealed']))
    print("Output assembly: {:,.0f} contigs   (Total sequence: {:,.0f} Min: {:,.0f} Max: {:,.0f} N50: {:,.0f})".format(result_info['output']['contigs']['num'], result_info['output']['contigs']['total'], result_info['output']['contigs']['min'], result_info['output']['contigs']['max'], result_info['output']['contigs']['N50']))
    print("                 {:,.0f} scaffolds (Total sequence: {:,.0f} Min: {:,.0f} Max: {:,.0f} N50: {:,.0f})".format(result_info['output']['scaffolds']['num'], result_info['output']['scaffolds']['total'], result_info['output']['scaffolds']['min'], result_info['output']['scaffolds']['max'], result_info['output']['scaffolds']['N50']))

def MiniGapScaffold(assembly_file, mapping_file, repeat_file, min_mapq, min_mapping_length, min_length_contig_break, prefix=False, stats=None):
    # Put in default parameters if nothing was specified
    if False == prefix:
        if ".gz" == assembly_file[-3:len(assembly_file)]:
            prefix = assembly_file.rsplit('.',2)[0]
        else:
            prefix = assembly_file.rsplit('.',1)[0]

    keep_all_subreads = False
    alignment_precision = 100
#
    max_repeat_extension = 1000 # Expected to be bigger than or equal to min_mapping_length
    min_len_repeat_connection = 5000
    repeat_len_factor_unique = 10
    remove_duplicated_contigs = True
#
    remove_zero_hit_contigs = True
    remove_short_contigs = True
    min_extension = 500
    max_dist_contig_end = 2000
    max_break_point_distance = 200
    merge_block_length = 10000
#
    min_num_reads = 2
    borderline_removal = False
    allow_same_contig_breaks = True
    min_factor_alternatives = 1.1
    cov_bin_fraction = 0.01
    num_read_len_groups = 10
    prob_factor = 10
    min_distance_tolerance = 20
    rel_distance_tolerance = 0.2
    ploidy = 2
    org_scaffold_trust = "basic" # blind: If there is a read that supports it use the org scaffold; Do not break contigs
                                 # full: If there is no confirmed other option use the org scaffold
                                 # basic: If there is no alternative bridge use the org scaffold
                                 # no: Do not give preference to org scaffolds
    max_loop_units = 10

    # Guarantee that outdir exists
    outdir = os.path.dirname(prefix)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    del outdir
    # Remove output files, such that we do not accidentially use an old one after a crash
    for f in [f"{prefix}_extensions.csv", f"{prefix}_scaffold_paths.csv", f"{prefix}_extending_reads.lst"]:
        if os.path.exists(f):
            os.remove(f)

    pdf = None
    if stats:
        pdf = PdfPages(stats)
        plt.ioff()

    print( str(timedelta(seconds=clock())), "Reading in original assembly")
    contigs, contig_ids = ReadContigs(assembly_file)
    result_info = {}
    result_info = GetInputInfo(result_info, contigs)
#
    print( str(timedelta(seconds=clock())), "Processing repeats")
#    contigs, center_repeats = MaskRepeatEnds(contigs, repeat_file, contig_ids, max_repeat_extension, min_len_repeat_connection, repeat_len_factor_unique, remove_duplicated_contigs, pdf)
#
    print( str(timedelta(seconds=clock())), "Filtering mappings")
    mappings, cov_counts, cov_probs = ReadMappings(mapping_file, contig_ids, min_mapq, keep_all_subreads, alignment_precision, min_num_reads, cov_bin_fraction, num_read_len_groups, pdf)
    contigs = RemoveUnmappedContigs(contigs, mappings, remove_zero_hit_contigs)
#    mappings = RemoveUnanchoredMappings(mappings, contigs, center_repeats, min_mapping_length, pdf, max_dist_contig_end)
#    del center_repeats
#
    print( str(timedelta(seconds=clock())), "Account for left-over adapters")
    mappings = BreakReadsAtAdapters(mappings, alignment_precision, keep_all_subreads)
#
    print( str(timedelta(seconds=clock())), "Search for possible break points")
    if "blind" == org_scaffold_trust:
        # Do not break contigs
        break_groups, spurious_break_indexes, non_informative_mappings = CallAllBreaksSpurious(mappings, contigs, max_dist_contig_end, min_length_contig_break, min_extension, pdf)
    else:
        break_groups, spurious_break_indexes, non_informative_mappings, unconnected_breaks = FindBreakPoints(mappings, contigs, max_dist_contig_end, min_mapping_length, min_length_contig_break, max_break_point_distance, min_num_reads, min_extension, merge_block_length, org_scaffold_trust, cov_probs, prob_factor, allow_same_contig_breaks, pdf)
    mappings.drop(np.concatenate([np.unique(spurious_break_indexes['map_index'].values), non_informative_mappings]), inplace=True) # Remove not-accepted breaks from mappings and mappings that do not contain any information (mappings inside of contigs that do not overlap with breaks)
    #SplitReadsAtSpuriousBreakIndexes(mappings, spurious_break_indexes)
    del spurious_break_indexes, non_informative_mappings
#
    contig_parts, contigs = GetContigParts(contigs, break_groups, remove_short_contigs, min_mapping_length, alignment_precision, pdf)
    result_info = GetBreakAndRemovalInfo(result_info, contigs, contig_parts)
    mappings = UpdateMappingsToContigParts(mappings, contig_parts, min_mapping_length, max_dist_contig_end, min_extension)
    del break_groups, contigs, contig_ids
#
    print( str(timedelta(seconds=clock())), "Search for possible bridges")
    bridges = GetBridges(mappings, borderline_removal, min_factor_alternatives, min_num_reads, org_scaffold_trust, contig_parts, cov_probs, prob_factor, min_mapping_length, min_distance_tolerance, rel_distance_tolerance, pdf)
#
    print( str(timedelta(seconds=clock())), "Scaffold the contigs")
    scaffold_paths = ScaffoldContigs(contig_parts, bridges, mappings, ploidy, max_loop_units)
#
    print( str(timedelta(seconds=clock())), "Fill gaps")
    mappings, scaffold_paths = MapReadsToScaffolds(mappings, scaffold_paths, bridges, ploidy) # Might break apart scaffolds again, if we cannot find a mapping read for a connection
    result_info = CountResealedBreaks(result_info, scaffold_paths, contig_parts, ploidy)
    scaffold_paths, mappings = FillGapsWithReads(scaffold_paths, mappings, contig_parts, ploidy, max_dist_contig_end, min_extension, min_num_reads, pdf) # Mappings are prepared for scaffold extensions
    result_info = GetOutputInfo(result_info, scaffold_paths)
#
    if pdf:
        pdf.close()

    print( str(timedelta(seconds=clock())), "Writing output")
    mappings.to_csv(f"{prefix}_extensions.csv", index=False)
    scaffold_paths.to_csv(f"{prefix}_scaffold_paths.csv", index=False)
    np.savetxt(f"{prefix}_extending_reads.lst", np.unique(mappings['read_name']), fmt='%s')

    print( str(timedelta(seconds=clock())), "Finished")
    PrintStats(result_info)

def GetPloidyFromPaths(scaffold_paths):
    return len([col for col in scaffold_paths.columns if col[:4] == 'name'])

def LoadExtensions(prefix, min_extension):
    # Load extending mappings
    mappings = pd.read_csv(prefix+"_extensions.csv")

    # Drop sides that do not fullfil min_extension
    mappings = mappings[min_extension <= mappings['ext']].reset_index(drop=True)
    
    return mappings

def RemoveDuplicatedReadMappings(extensions):
    # Minimap2 has sometimes two overlapping mappings: Remove the shorter mapping
    extensions.sort_values(['scaf','side','hap','q_index','t_index'], inplace=True)
    extensions['min_len'] = np.minimum(extensions['q_end']-extensions['q_start'], extensions['t_end']-extensions['t_start'])
    duplicates = extensions.groupby(['scaf','side','hap','q_index','t_index'], sort=False)['min_len'].agg(['max','size'])
    extensions['max_len'] = np.repeat(duplicates['max'].values,duplicates['size'].values)
    extensions = extensions[extensions['min_len'] == extensions['max_len']].copy()

    # Remove the more unequal one
    extensions['max_len'] = np.maximum(extensions['q_end']-extensions['q_start'], extensions['t_end']-extensions['t_start'])
    duplicates = extensions.groupby(['scaf','side','hap','q_index','t_index'], sort=False)['max_len'].agg(['min','size'])
    extensions['min_len'] = np.repeat(duplicates['min'].values,duplicates['size'].values)
    extensions = extensions[extensions['min_len'] == extensions['max_len']].copy()
    extensions.drop(columns=['min_len','max_len'], inplace=True)

    # Otherwise simply take the first
    extensions = extensions.groupby(['scaf','side','hap','q_index','t_index'], sort=False).first().reset_index()

    return extensions

def LoadReads(all_vs_all_mapping_file, mappings, min_length_contig_break):
    # Load all vs. all mappings for extending reads
    reads = ReadPaf(all_vs_all_mapping_file)
    reads.drop(columns=['matches','alignment_length','mapq'], inplace=True) # We don't need those columns
#
    # Get valid pairings on a smaller dataframe
    extensions = reads[['q_name','t_name']].drop_duplicates()
    extensions = extensions.merge(mappings[['read_name','read_start','scaf','side']].reset_index().rename(columns={'index':'q_index', 'read_name':'q_name', 'read_start':'q_read_start'}), on=['q_name'], how='inner')
    extensions = extensions.merge(mappings[['read_name','read_start','scaf','side']].reset_index().rename(columns={'index':'t_index', 'read_name':'t_name', 'read_start':'t_read_start'}), on=['t_name','scaf','side'], how='inner')
    extensions.drop(extensions.index[(extensions['q_name'] == extensions['t_name']) & (extensions['q_read_start'] == extensions['t_read_start'])].values, inplace=True) # remove reads that map to itself
#
    # Keep only reads that are part of a valid pairings
    extensions.drop(columns=['scaf','side','q_read_start', 't_read_start'], inplace=True)
    reads = reads.merge(extensions, on=['q_name','t_name'], how='inner')
    extensions = reads
    del reads
    extensions.drop(columns=['q_name','t_name'], inplace=True)
#
    # Add scaffold and side to which the query reads belong to
    extensions.reset_index(drop=True, inplace=True)
    extensions = pd.concat([extensions, mappings.loc[extensions['q_index'].values, ['read_start','read_end','read_from','read_to','scaf','side','hap','strand']].reset_index(drop=True).rename(columns={'read_from':'q_read_from', 'read_to':'q_read_to', 'hap':'q_hap', 'strand':'q_strand'})], axis=1)
#
    # Remove reads where the all vs. all mapping is not in the gap for the scaffold belonging to the query
    extensions.drop(extensions.index[np.where(np.logical_xor('+' == extensions['q_strand'], 'l' == extensions['side']), extensions['q_end'] <= extensions['q_read_to'], extensions['q_start'] >= extensions['q_read_from']) | (extensions['q_read_from'] >= extensions['read_end']) | (extensions['q_read_to'] <= extensions['read_start'])].values, inplace=True)
    extensions.drop(columns=['read_start','read_end'], inplace=True)
#
    # Repeat the last two steps for the target reads
    extensions.reset_index(drop=True, inplace=True)
    extensions = pd.concat([extensions, mappings.loc[extensions['t_index'].values, ['read_start','read_end','read_from','read_to','hap','strand']].reset_index(drop=True).rename(columns={'read_from':'t_read_from', 'read_to':'t_read_to', 'hap':'t_hap', 'strand':'t_strand'})], axis=1)
    extensions.drop(extensions.index[np.where(np.logical_xor('+' == extensions['t_strand'], 'l' == extensions['side']), extensions['t_end'] <= extensions['t_read_to'], extensions['t_start'] >= extensions['t_read_from']) | (extensions['t_read_from'] >= extensions['read_end']) | (extensions['t_read_to'] <= extensions['read_start'])].values, inplace=True)
    extensions.drop(columns=['read_start','read_end'], inplace=True)
#
    # Extensions: Remove reads that somehow do not fullfil strand rules and remove superfluous strand column
    extensions.drop(extensions.index[np.where(extensions['q_strand'] == extensions['t_strand'], '+', '-') != extensions['strand']].values, inplace=True)
    extensions.drop(columns=['strand'], inplace=True)
#
    # Split off hap_merger 
    hap_merger = extensions[extensions['q_hap'] != extensions['t_hap']].copy()
    extensions.drop(extensions.index[extensions['q_hap'] != extensions['t_hap']].values, inplace=True)
    extensions.drop(columns=['t_hap'], inplace=True)
    extensions.rename(columns={'q_hap':'hap'}, inplace=True)

    # Filter extensions where the all vs. all mapping does not touch the the contig mapping of query and target or the reads diverge more than min_length_contig_break of their mapping length within the contig mapping
    extensions['q_max_divergence'] = np.minimum(min_length_contig_break, np.where(np.logical_xor('+' == extensions['q_strand'], 'l' == extensions['side']), extensions['q_read_to']-extensions['q_start'], extensions['q_end']-extensions['q_read_from']))
    extensions['t_max_divergence'] = np.minimum(min_length_contig_break, np.where(np.logical_xor('+' == extensions['t_strand'], 'l' == extensions['side']), extensions['t_read_to']-extensions['t_start'], extensions['t_end']-extensions['t_read_from']))
    extensions = extensions[(0 < extensions['q_max_divergence']) & (0 < extensions['t_max_divergence'])].copy() # all vs. all mappings do not touch contig mapping
    extensions['read_divergence'] = np.minimum(np.where(np.logical_xor('+' == extensions['q_strand'], 'l' == extensions['side']), extensions['q_start'], extensions['q_len']-extensions['q_end']), np.where(np.logical_xor('+' == extensions['t_strand'], 'l' == extensions['side']), extensions['t_start'], extensions['t_len']-extensions['t_end']))
    extensions = extensions[( (0 < np.where(np.logical_xor('+' == extensions['q_strand'], 'l' == extensions['side']), extensions['q_read_from']-extensions['q_start'], extensions['q_end']-extensions['q_read_to']) + extensions['q_max_divergence']) |
                              (extensions['read_divergence'] < extensions['q_max_divergence']) ) &
                            ( (0 < np.where(np.logical_xor('+' == extensions['t_strand'], 'l' == extensions['side']), extensions['t_read_from']-extensions['t_start'], extensions['t_end']-extensions['t_read_to']) + extensions['t_max_divergence']) |
                              (extensions['read_divergence'] < extensions['t_max_divergence']) ) ].copy()
    extensions.drop(columns=['q_max_divergence','t_max_divergence','read_divergence'], inplace=True)

    # Add the flipped entries between query and strand
    extensions = pd.concat([extensions[['scaf','side','hap','t_index','t_strand','t_read_from','t_read_to','t_len','t_start','t_end','q_index','q_strand','q_read_from','q_read_to','q_len','q_start','q_end']].rename(columns={'t_index':'q_index','t_strand':'q_strand','t_read_from':'q_read_from','t_read_to':'q_read_to','t_len':'q_len','t_start':'q_start','t_end':'q_end','q_index':'t_index','q_strand':'t_strand','q_read_from':'t_read_from','q_read_to':'t_read_to','q_len':'t_len','q_start':'t_start','q_end':'t_end'}),
                            extensions[['scaf','side','hap','q_index','q_strand','q_read_from','q_read_to','q_len','q_start','q_end','t_index','t_strand','t_read_from','t_read_to','t_len','t_start','t_end']] ], ignore_index=True)
    hap_merger = pd.concat([hap_merger[['scaf','side','t_hap','t_index','t_strand','t_read_from','t_read_to','t_len','t_start','t_end','q_hap','q_index','q_strand','q_read_from','q_read_to','q_len','q_start','q_end']].rename(columns={'t_hap':'q_hap','t_index':'q_index','t_strand':'q_strand','t_read_from':'q_read_from','t_read_to':'q_read_to','t_len':'q_len','t_start':'q_start','t_end':'q_end','q_hap':'t_hap','q_index':'t_index','q_strand':'t_strand','q_read_from':'t_read_from','q_read_to':'t_read_to','q_len':'t_len','q_start':'t_start','q_end':'t_end'}),
                            hap_merger[['scaf','side','q_hap','q_index','q_strand','q_read_from','q_read_to','q_len','q_start','q_end','t_hap','t_index','t_strand','t_read_from','t_read_to','t_len','t_start','t_end']] ], ignore_index=True)

    # Haplotype merger are the first mapping of a read with a given read from the main haplotype (We can stop extending the alternative haplotype there, because it is identical to the main again)
    hap_merger = hap_merger[hap_merger['t_hap'] == 0].copy()
    hap_merger['max_ext'] = np.where((hap_merger['q_strand'] == '+') == (hap_merger['side'] == 'r'), hap_merger['q_start'] - hap_merger['q_read_to'], hap_merger['q_read_from'] - hap_merger['q_end'])
    hap_merger = hap_merger[['scaf','side','q_hap','q_index','t_index','max_ext']].copy()
    hap_merger['abs_est'] = np.abs(hap_merger['max_ext'])
    hap_merger.sort_values(['scaf','side','q_hap','q_index','t_index','abs_est'], inplace=True)
    hap_merger = hap_merger.groupby(['scaf','side','q_hap','q_index','t_index']).first().reset_index()
    hap_merger.drop(columns=['abs_est'], inplace=True)
    hap_merger.rename(columns={'q_hap':'hap'}, inplace=True)
    
    # Minimap2 has sometimes two overlapping mappings
    extensions = RemoveDuplicatedReadMappings(extensions)

    return extensions, hap_merger

def ClusterExtension(extensions, mappings, min_num_reads, min_scaf_len):
    # Remove indexes that cannot fulfill min_num_reads (have less than min_num_reads-1 mappings to other reads)
    extensions.sort_values(['scaf','side','hap','q_index'], inplace=True)
    org_len = len(extensions)+1
    while len(extensions) < org_len:
        org_len = len(extensions)
        num_mappings = extensions.groupby(['scaf','side','hap','q_index'], sort=False).size().values
        extensions = extensions[ np.minimum( extensions[['scaf','side','hap','t_index']].merge(extensions.groupby(['scaf','side','hap','t_index']).size().reset_index(name='num_mappings'), on=['scaf','side','hap','t_index'], how='left')['num_mappings'].values,
                                             np.repeat(num_mappings, num_mappings) ) >= min_num_reads-1 ].copy()
#
    # Cluster reads that share a mapping (A read in a cluster maps at least to one other read in this cluster)
    clusters = extensions.groupby(['scaf','side','hap','q_index'], sort=False).size().reset_index(name='size')
    clusters['cluster'] = np.arange(len(clusters))
    extensions['q_cluster_id'] = np.repeat(clusters.index.values, clusters['size'].values)
    extensions = extensions.merge(clusters[['scaf','side','hap','q_index']].reset_index().rename(columns={'q_index':'t_index','index':'t_cluster_id'}), on=['scaf','side','hap','t_index'], how='left')
#
    cluster_col = clusters.columns.get_loc('cluster')
    extensions['cluster'] = np.minimum(clusters.iloc[extensions['q_cluster_id'].values, cluster_col].values, clusters.iloc[extensions['t_cluster_id'].values, cluster_col].values)
    clusters['new_cluster'] = np.minimum( extensions.groupby('q_cluster_id')['cluster'].min().values, extensions.groupby('t_cluster_id')['cluster'].min().values )
    while np.sum(clusters['new_cluster'] != clusters['cluster']):
        clusters['cluster'] = clusters['new_cluster']
        extensions['cluster'] = np.minimum(clusters.iloc[extensions['q_cluster_id'].values, cluster_col].values, clusters.iloc[extensions['t_cluster_id'].values, cluster_col].values)
        clusters['new_cluster'] = np.minimum( extensions.groupby('q_cluster_id')['cluster'].min().values, extensions.groupby('t_cluster_id')['cluster'].min().values )
    extensions['cluster'] = clusters.iloc[extensions['q_cluster_id'].values, cluster_col].values
    clusters.drop(columns=['new_cluster'], inplace=True)
#
    # Remove sides with alternative clusters
    clusters.sort_values(['cluster','size','q_index'], ascending=[True,False,True], inplace=True)
    alternatives = clusters[['scaf','side','hap','cluster']].drop_duplicates().groupby(['scaf','side','hap']).size().reset_index(name='alternatives')
    clusters['cluster_id'] = clusters.index # We have to save it, because 'q_cluster_id' and 't_cluster_id' use them and merging removes the index
    clusters = clusters.merge(alternatives, on=['scaf','side','hap'], how='left')
    clusters.index = clusters['cluster_id'].values
    extensions['alternatives'] = clusters.loc[ extensions['q_cluster_id'].values, 'alternatives' ].values
    new_scaffolds = extensions.loc[ extensions['alternatives'] > 1, ['scaf','side','hap','cluster','q_index','q_strand','q_read_from','q_read_to','q_start','q_end']].copy()
    clusters = clusters[ clusters['alternatives'] == 1 ].copy()
    extensions = extensions[ extensions['alternatives'] == 1 ].copy()
    clusters.drop(columns=['alternatives'], inplace=True)
    extensions.drop(columns=['alternatives'], inplace=True)
#
    # Add how long the query agrees with the target in the gap
    extensions['q_agree'] = np.where( np.logical_xor('+' == extensions['q_strand'], 'l' == extensions['side']), extensions['q_end']-extensions['q_read_to'], extensions['q_read_from']-extensions['q_start'] )
    extensions.drop(columns=['cluster','q_cluster_id','t_cluster_id'], inplace=True)
#
    # Select the part of the removed clusters that is not in the assembly to add it as separate scaffolds that can later be connected in another round of scaffolding
    new_scaffolds['val_ext'] = np.where((new_scaffolds['q_strand'] == '+') == (new_scaffolds['side'] == 'r'), new_scaffolds['q_end'] - new_scaffolds['q_read_to'], new_scaffolds['q_read_from'] - new_scaffolds['q_start'])
    new_scaffolds.sort_values(['scaf','side','hap','cluster','q_index','val_ext'], ascending=[True,True,True,True,True,False], inplace=True)
    new_scaffolds['cov_sup'] = new_scaffolds.groupby(['scaf','side','hap','cluster','q_index']).cumcount()+1
    new_scaffolds = new_scaffolds[new_scaffolds['cov_sup'] == min_num_reads].copy()
    new_scaffolds.drop(columns=['cov_sup'], inplace=True)
    new_scaffolds.reset_index(drop=True, inplace=True)
    new_scaffolds = pd.concat([new_scaffolds, mappings.loc[new_scaffolds['q_index'],['ext','trim','unmap_ext','read_name','mapq','matches']].reset_index(drop=True)], axis=1)
    new_scaffolds['val_ext'] -= new_scaffolds['trim']
    new_scaffolds['gap_covered'] = np.minimum(new_scaffolds['val_ext']/new_scaffolds['unmap_ext'], 1.0)
    new_scaffolds['crosses_gap'] = new_scaffolds['unmap_ext'] < new_scaffolds['ext']
    new_scaffolds.loc[new_scaffolds['crosses_gap'] == False, 'unmap_ext'] *= -1
    new_scaffolds.sort_values(['scaf','side','hap','cluster','mapq','crosses_gap','unmap_ext','matches','gap_covered'], ascending=[True,True,True,True,False,False,True,False,False], inplace=True)
    new_scaffolds = new_scaffolds.groupby(['scaf','side','hap','cluster']).first().reset_index()
    new_scaffolds['unmap_ext'] = np.minimum(np.abs(new_scaffolds['unmap_ext']), new_scaffolds['val_ext']) + new_scaffolds['trim']
    new_scaffolds = new_scaffolds[new_scaffolds['unmap_ext'] >= min_scaf_len].copy()
    new_scaffolds['q_start'] = np.where((new_scaffolds['q_strand'] == '+') == (new_scaffolds['side'] == 'r'), new_scaffolds['q_read_to'], new_scaffolds['q_read_from']-new_scaffolds['unmap_ext'])
    new_scaffolds['q_end'] = np.where((new_scaffolds['q_strand'] == '+') == (new_scaffolds['side'] == 'r'), new_scaffolds['q_read_to']+new_scaffolds['unmap_ext'], new_scaffolds['q_read_from'])
    new_scaffolds = new_scaffolds[['read_name','q_start','q_end']].rename(columns={'read_name':'name0','q_start':'start0','q_end':'end0'})
#
    return extensions, new_scaffolds

def ExtendScaffolds(scaffold_paths, extensions, hap_merger, new_scaffolds, mappings, min_num_reads, max_mapping_uncertainty, min_scaf_len, ploidy):
    extension_info = {}
    extension_info['count'] = 0
    extension_info['new'] = 0

    if len(extensions) and len(mappings):
        # Create table on how long mappings agree in the gap with at least min_num_reads-1 (-1 because they always agree with themselves)
        len_agree = extensions[['scaf','side','hap','q_index','q_agree']].sort_values(['scaf','side','hap','q_index','q_agree'], ascending=[True,True,True,True,False])
        len_agree['n_longest'] = len_agree.groupby(['scaf','side','hap','q_index'], sort=False).cumcount()+1
        len_agree = len_agree[len_agree['n_longest'] == max(1,min_num_reads-1)].copy()
        len_agree.drop(columns=['n_longest'], inplace=True)
        len_mappings = mappings.iloc[len_agree['q_index'].values]
        len_agree['q_ext_len'] = np.where( np.logical_xor('+' == len_mappings['strand'], 'l' == len_mappings['side']), len_mappings['read_end']-len_mappings['read_to'], len_mappings['read_from'] )
        
        # Take the read that has the longest agreement with at least min_num_reads-1 and see if and at what position another bundle of min_num_reads diverge from it (extend until that position or as long as min_num_reads-1 agree)
        len_agree.sort_values(['scaf','side','hap','q_agree'], ascending=[True,True,True,False], inplace=True)
        extending_reads = len_agree.groupby(['scaf','side','hap'], sort=False).first().reset_index()
        len_agree = len_agree.merge(extending_reads[['scaf','side','hap','q_index']].rename(columns={'q_index':'t_index'}), on=['scaf','side','hap'], how='inner')
        len_agree = len_agree[len_agree['q_index'] != len_agree['t_index']].copy()
        len_agree = len_agree.merge(extensions[['scaf','side','hap','q_index','t_index','q_agree']].rename(columns={'q_agree':'qt_agree'}), on=['scaf','side','hap','q_index','t_index'], how='left')
        len_agree['qt_agree'].fillna(0, inplace=True)
        len_agree = len_agree[ len_agree['q_agree'] > len_agree['qt_agree']+max_mapping_uncertainty].copy()
        len_agree = len_agree.merge(extensions[['scaf','side','hap','q_index','t_index','q_agree']].rename(columns={'q_index':'t_index','t_index':'q_index','q_agree':'tq_agree'}), on=['scaf','side','hap','q_index','t_index'], how='left')
        len_agree['tq_agree'].fillna(0, inplace=True)
        
        len_agree.sort_values(['scaf','side','hap','tq_agree'], inplace=True)
        len_agree['n_disagree'] = len_agree.groupby(['scaf','side','hap'], sort=False).cumcount() + 1
        len_agree = len_agree[ len_agree['n_disagree'] == min_num_reads ].copy()
        extending_reads = extending_reads.merge(len_agree[['scaf','side','hap','tq_agree']].rename(columns={'tq_agree':'valid_ext'}), on=['scaf','side','hap'], how='left')
        extending_reads.loc[np.isnan(extending_reads['valid_ext']),'valid_ext'] = extending_reads.loc[np.isnan(extending_reads['valid_ext']),'q_agree'].values
        extending_reads['valid_ext'] = extending_reads['valid_ext'].astype(int)
        
        # Get max_ext to later trim extension of alternative haplotypes, when they get identical to main again
        extending_reads['t_hap'] = 0
        extending_reads = extending_reads.merge(extending_reads[['scaf','side','hap','q_index']].rename(columns={'hap':'t_hap','q_index':'t_index'}), on=['scaf','side','t_hap'], how='left')
        extending_reads['t_index'] = extending_reads['t_index'].fillna(-1).astype(int)
        extending_reads.drop(columns=['t_hap'], inplace=True)
        extending_reads = extending_reads.merge(hap_merger, on=['scaf','side','hap','q_index','t_index'], how='left')
        
        # Add the rest of the two reads after the split as new_scaffolds, up to unmap_ext (except if they are a trimmed alternative haplotype)
        add_scaffolds = []
        if len(extending_reads):
            extending_reads = pd.concat([extending_reads.reset_index(drop=True), mappings.loc[extending_reads['q_index'].values, ['trim','unmap_ext','read_name','read_from','read_to','strand']].reset_index(drop=True)], axis=1)
            add_scaffolds.append(extending_reads.loc[extending_reads['unmap_ext'] > 0, ['side','q_agree','valid_ext','max_ext','trim','unmap_ext','read_name','read_from','read_to','strand']].copy())
        if len(len_agree):
            len_agree.drop(columns=['q_ext_len','tq_agree','n_disagree','t_index'], inplace=True)
            len_agree.rename(columns={'qt_agree':'valid_ext'}, inplace=True)
            len_agree['valid_ext'] = len_agree['valid_ext'].astype(int)
            # Add max_ext
            len_agree['t_hap'] = 0
            len_agree = len_agree.merge(extending_reads[['scaf','side','hap','q_index']].rename(columns={'hap':'t_hap','q_index':'t_index'}), on=['scaf','side','t_hap'], how='left')
            len_agree['t_index'] = len_agree['t_index'].fillna(-1).astype(int)
            len_agree.drop(columns=['t_hap'], inplace=True)
            len_agree = len_agree.merge(hap_merger, on=['scaf','side','hap','q_index','t_index'], how='left')
            # Add mapping information
            len_agree = pd.concat([len_agree.reset_index(drop=True), mappings.loc[len_agree['q_index'].values, ['trim','unmap_ext','read_name','read_from','read_to','strand']].reset_index(drop=True)], axis=1)
            add_scaffolds.append(len_agree.loc[len_agree['unmap_ext'] > 0, ['side','q_agree','valid_ext','max_ext','trim','unmap_ext','read_name','read_from','read_to','strand']].copy())
        add_scaffolds = pd.concat(add_scaffolds, ignore_index=True)
        if len(add_scaffolds) == 0:
            gap_scaffolds = new_scaffolds
        else:
            add_scaffolds['new_scaf'] = np.minimum(add_scaffolds['unmap_ext'],add_scaffolds['q_agree'])+add_scaffolds['trim']
            add_scaffolds.loc[np.isnan(add_scaffolds['max_ext']) == False, 'new_scaf'] = np.minimum(add_scaffolds.loc[np.isnan(add_scaffolds['max_ext']) == False, 'new_scaf'], add_scaffolds.loc[np.isnan(add_scaffolds['max_ext']) == False, 'max_ext']).astype(int)
            add_scaffolds['new_scaf'] -= add_scaffolds['valid_ext']
            add_scaffolds = add_scaffolds[add_scaffolds['new_scaf'] >= min_scaf_len].copy()
            add_scaffolds['start0'] = np.where((add_scaffolds['strand'] == '+') == (add_scaffolds['side'] == 'r'), add_scaffolds['read_to'] + add_scaffolds['valid_ext'], add_scaffolds['read_from'] - add_scaffolds['valid_ext'] - add_scaffolds['new_scaf'])
            add_scaffolds['end0'] = np.where((add_scaffolds['strand'] == '+') == (add_scaffolds['side'] == 'r'), add_scaffolds['read_to'] + add_scaffolds['valid_ext'] + add_scaffolds['new_scaf'], add_scaffolds['read_from'] - add_scaffolds['valid_ext'])
            add_scaffolds = add_scaffolds[['read_name','start0','end0']].rename(columns={'read_name':'name0'})
            gap_scaffolds = pd.concat([new_scaffolds, add_scaffolds], ignore_index=True)

        # Trim the part that does not match the read independent of whether we can later extend the scaffold or not
        if len(extending_reads):
            # Trim left side (Do sides separately, because a contig can be both sides of a scaffold)
            scaffold_paths['side'] = np.where(0 == scaffold_paths['pos'], 'l', '')
            for h in range(ploidy):
                scaffold_paths = scaffold_paths.merge(extending_reads.loc[extending_reads['hap'] == h, ['scaf','side','trim']], on=['scaf','side'], how='left')
                scaffold_paths['trim'] = scaffold_paths['trim'].fillna(0).astype(int)
                scaffold_paths.loc[scaffold_paths[f'strand{h}'] == '+', f'start{h}'] += scaffold_paths.loc[scaffold_paths[f'strand{h}'] == '+', 'trim']
                scaffold_paths.loc[scaffold_paths[f'strand{h}'] == '-', f'end{h}'] -= scaffold_paths.loc[scaffold_paths[f'strand{h}'] == '-', 'trim']
                scaffold_paths.drop(columns=['trim'], inplace=True)
            # Trim right side
            scaffold_paths['side'] = np.where(scaffold_paths['scaf'] != scaffold_paths['scaf'].shift(-1), 'r', '')
            for h in range(ploidy):
                scaffold_paths = scaffold_paths.merge(extending_reads.loc[extending_reads['hap'] == h, ['scaf','side','trim']], on=['scaf','side'], how='left')
                scaffold_paths['trim'] = scaffold_paths['trim'].fillna(0).astype(int)
                scaffold_paths.loc[scaffold_paths[f'strand{h}'] == '+', f'end{h}'] -= scaffold_paths.loc[scaffold_paths[f'strand{h}'] == '+', 'trim']
                scaffold_paths.loc[scaffold_paths[f'strand{h}'] == '-', f'start{h}'] += scaffold_paths.loc[scaffold_paths[f'strand{h}'] == '-', 'trim']
                scaffold_paths.drop(columns=['trim'], inplace=True)
            # Get the last position in the scaffold (the right side extension gets last_pos+1)
            ext_pos = scaffold_paths.loc[scaffold_paths['side'] == 'r', ['scaf','pos']].copy()
            scaffold_paths.drop(columns=['side'], inplace=True)
        
        # Extend scaffolds
        extending_reads.loc[np.isnan(extending_reads['max_ext']) == False, 'valid_ext'] = np.minimum(extending_reads.loc[np.isnan(extending_reads['max_ext']) == False, 'valid_ext'], extending_reads.loc[np.isnan(extending_reads['max_ext']) == False, 'max_ext']).astype(int)
        extending_reads = extending_reads[extending_reads['valid_ext'] > 0].copy()
        if len(extending_reads):
            # Fill extension info
            extension_info['count'] = len(extending_reads)
            extension_info['left'] = len(extending_reads[extending_reads['side'] == 'l'])
            extension_info['right'] = len(extending_reads[extending_reads['side'] == 'r'])
            extension_info['mean'] = int(round(np.mean(extending_reads['valid_ext'])))
            extension_info['min'] = np.min(extending_reads['valid_ext'])
            extension_info['max'] = np.max(extending_reads['valid_ext'])
            
            # Get start and end of extending read
            extending_reads['start0'] = np.where((extending_reads['strand'] == '+') == (extending_reads['side'] == 'r'), extending_reads['read_to'], extending_reads['read_from']-extending_reads['valid_ext'])
            extending_reads['end0'] = np.where((extending_reads['strand'] == '+') == (extending_reads['side'] == 'r'), extending_reads['read_to']+extending_reads['valid_ext'], extending_reads['read_from'])

            # Change structure of extending_reads to the same as scaffold_paths
            extending_reads = extending_reads[['scaf','side','hap','read_name','start0','end0','strand']].rename(columns={'read_name':'name0','strand':'strand0'})
            for h in range(1,ploidy):
                extending_reads = extending_reads.merge(extending_reads.loc[extending_reads['hap'] == h, ['scaf','side','name0','start0','end0','strand0']].rename(columns={'name0':f'name{h}','start0':f'start{h}','end0':f'end{h}','strand0':f'strand{h}'}), on=['scaf','side'], how='left')
                extending_reads[[f'name{h}',f'strand{h}']] = extending_reads[[f'name{h}',f'strand{h}']].fillna('')
                extending_reads[[f'start{h}',f'end{h}']] = extending_reads[[f'start{h}',f'end{h}']].fillna(0).astype(int)
                extending_reads.loc[extending_reads['hap'] == h, ['name0','start0','end0','strand0']] = ['',0,0,'']
            extending_reads.sort_values(['scaf','side','hap'], inplace=True)
            extending_reads = extending_reads[(extending_reads['scaf'] != extending_reads['scaf'].shift(1)) | (extending_reads['side'] != extending_reads['side'].shift(1))].copy()
            extending_reads.drop(columns=['hap'], inplace=True)
            extending_reads = extending_reads.merge(ext_pos, on='scaf', how='left')
            extending_reads.loc[extending_reads['side'] == 'l', 'pos'] = 0
            extending_reads = extending_reads.merge(scaffold_paths[['scaf','pos']+[f'phase{h}' for h in range(ploidy)]+['sdist_left','sdist_right']], on=['scaf','pos'], how='left')
            for h in range(1,ploidy):
                extending_reads.loc[extending_reads[f'name{h}'] != '', f'phase{h}'] = np.abs(extending_reads.loc[extending_reads[f'name{h}'] != '', f'phase{h}'])
            extending_reads['pos'] += 1
            extending_reads.loc[extending_reads['side'] == 'l', 'pos'] = -1
            extending_reads['type'] = 'read'

            # Add extending_reads to scaffold_table, sort again and make sure that sdist_left and sdist_right are only set at the new ends
            scaffold_paths = scaffold_paths.append( extending_reads[['scaf','pos','type']+[f'{n}{h}' for h in range(ploidy) for n in ['phase','name','start','end','strand']]+['sdist_left','sdist_right']] )
            scaffold_paths.sort_values(['scaf','pos'], inplace=True)
            scaffold_paths['pos'] = scaffold_paths.groupby(['scaf'], sort=False).cumcount()
            scaffold_paths.loc[scaffold_paths['pos'] > 0, 'sdist_left'] = -1
            scaffold_paths.loc[scaffold_paths['scaf'] == scaffold_paths['scaf'].shift(-1), 'sdist_right'] = -1
            
        if len(gap_scaffolds):
            # Fill extension info
            gap_scaffolds['len'] = gap_scaffolds['end0'] - gap_scaffolds['start0']
            extension_info['new'] = len(gap_scaffolds)
            extension_info['new_mean'] = int(round(np.mean(gap_scaffolds['len'])))
            extension_info['new_min'] = np.min(gap_scaffolds['len'])
            extension_info['new_max'] = np.max(gap_scaffolds['len'])
            
            # Change structure of gap_scaffolds to the same as scaffold_paths
            gap_scaffolds['scaf'] = np.arange(len(gap_scaffolds)) + 1 + scaffold_paths['scaf'].max()
            gap_scaffolds['pos'] = 0
            gap_scaffolds['type'] = 'read'
            gap_scaffolds['phase0'] = np.arange(len(gap_scaffolds)) + 1 + scaffold_paths[[f'phase{h}' for h in range(ploidy)]].abs().max().max()
            gap_scaffolds['strand0'] = '+'
            for h in range(1,ploidy):
                gap_scaffolds[f'phase{h}'] = -gap_scaffolds['phase0']
                gap_scaffolds[[f'name{h}',f'strand{h}']] = ''
                gap_scaffolds[[f'start{h}',f'end{h}']] = 0
            gap_scaffolds[['sdist_left','sdist_right']] = -1
            scaffold_paths = scaffold_paths.append( gap_scaffolds[['scaf','pos','type']+[f'{n}{h}' for h in range(ploidy) for n in ['phase','name','start','end','strand']]+['sdist_left','sdist_right']] )

    if extension_info['count'] == 0:
        extension_info['left'] = 0
        extension_info['right'] = 0
        extension_info['mean'] = 0
        extension_info['min'] = 0
        extension_info['max'] = 0
        
    if extension_info['new'] == 0:
        extension_info['new_mean'] = 0
        extension_info['new_min'] = 0
        extension_info['new_max'] = 0
        
    return scaffold_paths, extension_info

def MiniGapExtend(all_vs_all_mapping_file, prefix, min_length_contig_break):
    # Define parameters
    min_extension = 500    
    min_num_reads = 3
    max_mapping_uncertainty = 200
    min_scaf_len = 600
    
    # Remove output files, such that we do not accidentially use an old one after a crash
    for f in [f"{prefix}_extended_scaffold_paths.csv", f"{prefix}_used_reads.lst"]:
        if os.path.exists(f):
            os.remove(f)
    
    print( str(timedelta(seconds=clock())), "Preparing data from files")
    scaffold_paths = pd.read_csv(prefix+"_scaffold_paths.csv").fillna('')
    ploidy = GetPloidyFromPaths(scaffold_paths)
    mappings = LoadExtensions(prefix, min_extension)
    extensions, hap_merger = LoadReads(all_vs_all_mapping_file, mappings, min_length_contig_break)
    
    print( str(timedelta(seconds=clock())), "Searching for extensions")
    extensions, new_scaffolds = ClusterExtension(extensions, mappings, min_num_reads, min_scaf_len)
    scaffold_paths, extension_info = ExtendScaffolds(scaffold_paths, extensions, hap_merger, new_scaffolds, mappings, min_num_reads, max_mapping_uncertainty, min_scaf_len, ploidy)

    print( str(timedelta(seconds=clock())), "Writing output")
    scaffold_paths.to_csv(f"{prefix}_extended_scaffold_paths.csv", index=False)
    np.savetxt(f"{prefix}_used_reads.lst", np.unique(pd.concat([scaffold_paths.loc[('read' == scaffold_paths['type']) & ('' != scaffold_paths[f'name{h}']), f'name{h}'] for h in range(ploidy)], ignore_index=True)), fmt='%s')
    
    print( str(timedelta(seconds=clock())), "Finished")
    print( "Extended {} scaffolds (left: {}, right:{}).".format(extension_info['count'], extension_info['left'], extension_info['right']) )
    print( "The extensions ranged from {} to {} bases and had a mean length of {}.".format(extension_info['min'], extension_info['max'], extension_info['mean']) )
    print( "Added {} scaffolds in the gaps with a length ranging from {} to {} bases and a mean of {}.".format(extension_info['new'],extension_info['new_min'], extension_info['new_max'], extension_info['new_mean']) )

def SwitchBubbleWithMain(outpaths, hap, switch):
    swi = switch & (outpaths[f'phase{hap}'] >= 0)
    tmp = outpaths.loc[swi, ['name0','start0','end0','strand0']].values
    outpaths.loc[swi, ['name0','start0','end0','strand0']] = outpaths.loc[swi, [f'name{hap}',f'start{hap}',f'end{hap}',f'strand{hap}']].values
    outpaths.loc[swi, [f'name{hap}',f'start{hap}',f'end{hap}',f'strand{hap}']] = tmp
#
    return outpaths

def RemoveBubbleHaplotype(outpaths, hap, remove):
    rem = remove & (outpaths[f'phase{hap}'] >= 0)
    outpaths.loc[rem, [f'name{hap}',f'start{hap}',f'end{hap}',f'strand{hap}']] = ['',0,0,'']
    outpaths.loc[rem, f'phase{hap}'] = -outpaths.loc[rem, f'phase{hap}']
#
    return outpaths

def MiniGapFinish(assembly_file, read_file, read_format, scaffold_file, output_files, haplotypes):
    min_length = 600 # Minimum length for a contig or alternative haplotype to be in the output
    skip_length = 50 # Haplotypes with a length shorter than this get lowest priority for being included in the main scaffold for mixed mode
    merge_dist = 1000 # Alternative haplotypes separated by at max merge_dist of common sequence are merged and output as a whole alternative contig in mixed mode
#
    if False == output_files[0]:
        if ".gz" == assembly_file[-3:len(assembly_file)]:
            output_files[0] = assembly_file.rsplit('.',2)[0]+"_minigap.fa"
        else:
            output_files[0] = assembly_file.rsplit('.',1)[0]+"_minigap.fa"
#
    # Remove output files, such that we do not accidentially use an old one after a crash
    for f in output_files:
        if os.path.exists(f):
            os.remove(f)
#
    print( str(timedelta(seconds=clock())), "Loading scaffold info from: {}".format(scaffold_file))
    scaffold_paths = pd.read_csv(scaffold_file).fillna('')
    ploidy = GetPloidyFromPaths(scaffold_paths)
    for i in range(len(haplotypes)):
        if haplotypes[i] == -1:
            haplotypes[i] = -1 if ploidy > 1 else 0
        elif haplotypes[i] >= ploidy:
            print(f"The highest haplotype in scaffold file is {ploidy-1}, but {haplotypes[i]} was specified")
            sys.exit(1)
#
    print( str(timedelta(seconds=clock())), "Loading assembly from: {}".format(assembly_file))
    contigs = {}
    with gzip.open(assembly_file, 'rb') if 'gz' == assembly_file.rsplit('.',1)[-1] else open(assembly_file, 'rU') as fin:
        for record in SeqIO.parse(fin, "fasta"):
            contigs[ record.description.split(' ', 1)[0] ] = record.seq
#
    print( str(timedelta(seconds=clock())), "Loading reads from: {}".format(read_file))
    reads = {}
    if read_format == False:
        lread_file = read_file.lower()
        if (lread_file[-3:] == ".fa") or (lread_file[-6:] == ".fasta") or (lread_file[-6:] == ".fa.gz") or (lread_file[-9:] == ".fasta.gz"):
            read_format = "fasta"
        else:
            read_format = "fastq"
    with gzip.open(read_file, 'rb') if 'gz' == read_file.rsplit('.',1)[-1] else open(read_file, 'rU') as fin:
        for record in SeqIO.parse(fin, read_format):
            reads[ record.description.split(' ', 1)[0] ] = record.seq
    
    for outfile, hap in zip(output_files, haplotypes):
        outpaths = scaffold_paths.copy()
        outpaths['meta'] = ((outpaths['scaf'] != outpaths['scaf'].shift(1)) & (outpaths['sdist_left'] < 0)).cumsum()
        # Select the paths based on haplotype
        if hap < 0:
            print( str(timedelta(seconds=clock())), "Writing mixed assembly to: {}".format(outfile))
            outpaths['bubble'] = ((outpaths[[f'phase{h}' for h in range(1,ploidy)]] < 0).all(axis=1) | (outpaths['scaf'] != outpaths['scaf'].shift(1))).cumsum()
            for h in range(ploidy):
                outpaths[f'len{h}'] = outpaths[f'end{h}']-outpaths[f'start{h}']
            # Merge bubbles if they are separeted by merge_dist or less
            bubbles = outpaths.groupby(['scaf','bubble'])[[f'len{h}' for h in range(ploidy)]].sum().reset_index()
            bubbles['mblock'] = bubbles['len0'].cumsum()
            bubbles = bubbles[(bubbles[[f'len{h}' for h in range(1,ploidy)]] > 0).any(axis=1)].copy()
            bubbles['mblock'] = bubbles['mblock'] - bubbles['mblock'].shift(1, fill_value=0) - bubbles['len0'] + bubbles[['bubble']].merge(outpaths[['bubble','len0']].groupby(['bubble']).first().reset_index(), on=['bubble'], how='left')['len0'].values # We also need to account for the first entry in every bubble that is identical to main (The definition of a bubble start)
            bubbles['new_bubble'] = bubbles['bubble'].shift(1, fill_value=-1)
            bubbles = bubbles[(bubbles['scaf'] == bubbles['scaf'].shift(1)) & (bubbles['mblock'] <= merge_dist)].copy()
            if len(bubbles):
                while True:
                    bubbles = bubbles[['bubble','new_bubble']].copy()
                    bubbles = bubbles.merge(bubbles.rename(columns={'bubble':'new_bubble','new_bubble':'min_bubble'}), on=['new_bubble'], how='left')
                    if np.sum(np.isnan(bubbles['min_bubble']) == False) == 0:
                        break
                    else:
                        bubbles.loc[np.isnan(bubbles['min_bubble']) == False, 'new_bubble'] = bubbles.loc[np.isnan(bubbles['min_bubble']) == False, 'min_bubble'].astype(int)
                        bubbles.drop(columns=['min_bubble'], inplace=True)
                outpaths = outpaths.merge(bubbles[['bubble','new_bubble']], on=['bubble'], how='left')
                outpaths.loc[np.isnan(outpaths['new_bubble']) == False, 'bubble'] = outpaths.loc[np.isnan(outpaths['new_bubble']) == False, 'new_bubble'].astype(int)
                outpaths.drop(columns=['new_bubble'], inplace=True)
            # Remove haplotypes that are inversions or identical, but shorter than alternatives (skip_length dufferences are tolerated to still count as identical)
            #!!!! Still need to handle the comparison between alternative haplotypes
            for h in range(1,ploidy):
                # find the haplotypes to remove
                outpaths['same'] = (outpaths[f'phase{h}'] < 0)
                outpaths['len0'] = outpaths['end0']-outpaths['start0']
                outpaths[f'len{h}'] = np.where(outpaths[f'phase{h}'] < 0, outpaths['len0'], outpaths[f'end{h}']-outpaths[f'start{h}'])
                outpaths['identical'] = (outpaths[f'phase{h}'] < 0) | (outpaths[f'len{h}'] <= skip_length) | ((outpaths[f'name{h}'] == outpaths['name0']) & (outpaths[f'start{h}'] >= outpaths['start0']-skip_length) & (outpaths[f'end{h}'] <= outpaths['end0']+skip_length)) # Inversions also count as identical here
                outpaths['identical2'] = (outpaths[f'phase{h}'] < 0) | (outpaths['len0'] <= skip_length) | ((outpaths[f'name{h}'] == outpaths['name0']) & (outpaths[f'start{h}']-skip_length <= outpaths['start0']) & (outpaths[f'end{h}']+skip_length >= outpaths['end0'])) # Inversions also count as identical here
                bubbles = outpaths.groupby(['bubble'])[['same','identical','identical2']].min().reset_index()
                bubbles = bubbles[(bubbles['same'] == False) & bubbles[['identical','identical2']].any(axis=1)].copy()
                bubbles = bubbles.merge(outpaths.groupby(['bubble'])[['len0',f'len{h}']].sum().reset_index(), on=['bubble'], how='left')
                bubbles.loc[bubbles['identical'] & (bubbles['len0'] >= bubbles[f'len{h}']), 'identical2'] = False
                bubbles.loc[bubbles['identical2'] & (bubbles['len0'] < bubbles[f'len{h}']), 'identical'] = False
                # Remove haplotypes
                outpaths = outpaths.drop(columns=['same','identical','identical2']).merge(bubbles[['bubble','identical','identical2']], on=['bubble'], how='left').fillna(False)
                outpaths = SwitchBubbleWithMain(outpaths, h, outpaths['identical2'])
                outpaths = RemoveBubbleHaplotype(outpaths, h, outpaths['identical'] | outpaths['identical2'])
            outpaths.drop(columns=['identical','identical2'], inplace=True)
            # Take the worst to map (shortest) haplotype as main, except if we are going to remove the alternative, because it is shorter min_length: than keep longest by putting it in main (bubble pathes shorter than skip_length are only kept by putting it into main if no longer path exist, because they can be easily recreated from reads)
            for h in range(ploidy):
                outpaths[f'len{h}'] = np.where(outpaths[f'phase{h}'] < 0, outpaths['len0'], outpaths[f'end{h}']-outpaths[f'start{h}'])
            bubbles = outpaths.groupby(['bubble'])[[f'len{h}' for h in range(ploidy)]].sum().reset_index()
            bubbles = bubbles.merge(outpaths.groupby(['bubble'])[[f'phase{h}' for h in range(1,ploidy)]].max().reset_index(), on=['bubble'], how='left')
            bubbles = bubbles.loc[(bubbles[[f'phase{h}' for h in range(1,ploidy)]] > 0).any(axis=1), ['bubble']+[f'len{h}' for h in range(ploidy)]].melt(id_vars='bubble',var_name='hap',value_name='len')
            bubbles['hap'] = bubbles['hap'].str[3:].astype(int)
            bubbles['prio'] = np.where(bubbles['len'] >= min_length, 2, np.where(bubbles['len'] < skip_length, 3, 1))
            bubbles.loc[bubbles['len'] < min_length, 'len'] = -bubbles.loc[bubbles['len'] < min_length, 'len'] # For everything shorter than min_length we want the longest not the shortest
            bubbles.sort_values(['bubble','prio','len','hap'], inplace=True) # Sort by hap to keep main if no reason to switch
            bubbles = bubbles.groupby(['bubble'], sort=False).first().reset_index()
            bubbles = bubbles[bubbles['hap'] > 0].copy()
            outpaths = outpaths.merge(bubbles[['bubble','hap']].rename(columns={'hap':'switch'}), on=['bubble'], how='left')
            for h in range(1,ploidy):
                outpaths = SwitchBubbleWithMain(outpaths, h, outpaths['switch']==h)
            outpaths.drop(columns=['len0','len1','switch'], inplace=True)
            # Separate every alternative from a bubble into its own haplotype
            outpaths.rename(columns={'name0':'name','start0':'start','end0':'end','strand0':'strand'}, inplace=True)
            alternatives = []
            max_scaf = outpaths['scaf'].max()
            for h in range(1,ploidy):
                bubbles = outpaths.loc[outpaths[f'phase{h}'] >= 0, ['meta','bubble']].groupby(['meta','bubble']).size().reset_index(name='count').drop(columns='count')
                if len(bubbles):
                    bubbles['alt'] = bubbles.groupby(['meta']).cumcount() + 1
                    outpaths = outpaths.merge(bubbles[['bubble','alt']], on=['bubble'], how='left')
                    bubbles = outpaths.loc[np.isnan(outpaths['alt']) == False, ['meta','alt','pos','bubble','type',f'phase{h}',f'name{h}',f'start{h}',f'end{h}',f'strand{h}']].rename(columns={f'phase{h}':'phase',f'name{h}':'name',f'start{h}':'start',f'end{h}':'end',f'strand{h}':'strand'})
                    # Trim parts that are identical to main from both sides
                    while True:
                        old_len = len(bubbles)
                        bubbles = bubbles[(bubbles['phase'] >= 0) | ((bubbles['bubble'] == bubbles['bubble'].shift(1)) & (bubbles['bubble'] == bubbles['bubble'].shift(-1)))].copy()
                        if len(bubbles) == old_len:
                            break
                    outpaths.drop(columns=[f'phase{h}',f'name{h}',f'start{h}',f'end{h}',f'strand{h}','alt'], inplace=True)
                if len(bubbles):
                    # Prepare all necessary columns
                    bubbles['alt'] = bubbles['alt'].astype(int)
                    bubbles['scaf'] = (bubbles['bubble'] != bubbles['bubble'].shift(1)).cumsum() + max_scaf
                    max_scaf = bubbles['scaf'].max()
                    bubbles['pos'] = bubbles.groupby(['bubble']).cumcount()
                    bubbles['sdist_right'] = -1
                    bubbles['hap'] = h
                    alternatives.append(bubbles[['meta','hap','alt','scaf','pos','type','name','start','end','strand','sdist_right']])
            outpaths['alt'] = 0
            outpaths['hap'] = 0
            outpaths = pd.concat([outpaths[['meta','hap','alt','scaf','pos','type','name','start','end','strand','sdist_right']]] + alternatives, ignore_index=True).sort_values(['meta','alt','pos'])
            outpaths['meta'] = np.repeat("scaffold", len(outpaths)) + outpaths['meta'].astype(str) + np.where(outpaths['hap'] == 0, "", "_hap" + str(h) + "_alt" + outpaths['alt'].astype('str'))
        else:
            print( str(timedelta(seconds=clock())), "Writing haplotype {} of assembly to: {}".format(hap,outfile))
            outpaths['meta'] = np.repeat("scaffold", len(outpaths)) + outpaths['meta'].astype(str)
            outpaths.rename(columns={'name0':'name','start0':'start','end0':'end','strand0':'strand'}, inplace=True)
            if hap > 0:
                outpaths.loc[outpaths[f'phase{hap}'] >= 0, ['name','start','end','strand']] = outpaths.loc[outpaths[f'phase{hap}'] >= 0, [f'name{hap}',f'start{hap}',f'end{hap}',f'strand{hap}']].values
        # Remove scaffolds that are shorter than min_length
        outpaths = outpaths.loc[outpaths['name'] != '', ['meta','scaf','pos','type','name','start','end','strand','sdist_right']].copy()
        outpaths['length'] = outpaths['end'] - outpaths['start']
        scaf_len = outpaths.groupby(['scaf'])['length'].agg(['sum','size'])
        outpaths = outpaths[np.repeat(scaf_len['sum'].values >= min_length, scaf_len['size'].values)].copy()
        outpaths.loc[(outpaths['meta'] != outpaths['meta'].shift(-1)), 'sdist_right'] = -1
        # Write to file
        with gzip.open(outfile, 'wb') if 'gz' == outfile.rsplit('.',1)[-1] else open(outfile, 'w') as fout:
            meta = ''
            seq = []
            for row in outpaths.itertuples(index=False):
                # Check if scaffold changed
                if meta != row.meta:
                    if meta != '':
                        # Write scaffold to disc
                        fout.write('>')
                        fout.write(meta)
                        fout.write('\n')
                        fout.write(''.join(seq))
                        fout.write('\n')
                    # Start new scaffold
                    meta = row.meta
                    seq = []
#
                # Add sequences to scaffold
                if 'contig' == row.type:
                    if row.strand == '-':
                        seq.append(str(contigs[row.name][row.start:row.end].reverse_complement()))
                    else:
                        seq.append(str(contigs[row.name][row.start:row.end]))
                elif 'read' == row.type:
                    if row.strand == '-':
                        seq.append(str(reads[row.name][row.start:row.end].reverse_complement()))
                    else:
                        seq.append(str(reads[row.name][row.start:row.end]))
#
                # Add N's to separate contigs
                if 0 <= row.sdist_right:
                    seq.append( 'N' * row.sdist_right )
#
            # Write out last scaffold
            fout.write('>')
            fout.write(meta)
            fout.write('\n')
            fout.write(''.join(seq))
            fout.write('\n')
        
        print( str(timedelta(seconds=clock())), "Finished" )

def GetMappingsInRegion(mapping_file, regions, min_mapq, min_mapping_length, keep_all_subreads, alignment_precision, min_length_contig_break, max_dist_contig_end):
    mappings = ReadPaf(mapping_file)
    
    # Filter mappings
    mappings = mappings[(min_mapq <= mappings['mapq']) & (min_mapping_length <= mappings['t_end'] - mappings['t_start'])].copy()
    
    # Remove all but the best subread
    if not keep_all_subreads:
        mappings = GetBestSubreads(mappings, alignment_precision)
    
    # Find mappings inside defined regions
    mappings['in_region'] = False
    for reg in regions:
         mappings.loc[ (reg['scaffold'] == mappings['t_name']) & (reg['start'] < mappings['t_end']) & (reg['end'] >= mappings['t_start']) , 'in_region'] = True
         
    # Only keep mappings that are in defined regions or are the previous or next mappings of a read in the region
    mappings.sort_values(['q_name','q_start','q_end'], inplace=True)
    mappings = mappings[ mappings['in_region'] | (mappings['in_region'].shift(-1, fill_value=False) & (mappings['q_name'] == mappings['q_name'].shift(-1, fill_value=''))) | (mappings['in_region'].shift(1, fill_value=False) & (mappings['q_name'] == mappings['q_name'].shift(1, fill_value=''))) ].copy()
    
    # Get left and right scaffold of read (first ignore strand and later switch next and prev if necessary)
    mappings['left_scaf'] = np.where(mappings['q_name'] != mappings['q_name'].shift(1, fill_value=''), '', mappings['t_name'].shift(1, fill_value=''))
    mappings['right_scaf'] = np.where(mappings['q_name'] != mappings['q_name'].shift(-1, fill_value=''), '', mappings['t_name'].shift(-1, fill_value=''))
    # Get distance to next mapping or distance to read end if no next mapping exists
    mappings['left_dist'] = np.where('' == mappings['left_scaf'], mappings['q_start'], mappings['q_start']-mappings['q_end'].shift(1, fill_value=0))
    mappings['right_dist'] = np.where('' == mappings['right_scaf'], mappings['q_len']-mappings['q_end'], mappings['q_start'].shift(-1, fill_value=0)-mappings['q_end'])
    # Get length of continued read after mapping end
    mappings['left_len'] = mappings['q_start']
    mappings['right_len'] = mappings['q_len']-mappings['q_end']

    # Switch left and right for negative strand
    tmp = mappings.loc['-' == mappings['strand'], 'left_scaf']
    mappings.loc['-' == mappings['strand'], 'left_scaf'] = mappings.loc['-' == mappings['strand'], 'right_scaf']
    mappings.loc['-' == mappings['strand'], 'right_scaf'] = tmp
    tmp = mappings.loc['-' == mappings['strand'], 'left_dist']
    mappings.loc['-' == mappings['strand'], 'left_dist'] = mappings.loc['-' == mappings['strand'], 'right_dist']
    mappings.loc['-' == mappings['strand'], 'right_dist'] = tmp
    tmp = mappings.loc['-' == mappings['strand'], 'left_len']
    mappings.loc['-' == mappings['strand'], 'left_len'] = mappings.loc['-' == mappings['strand'], 'right_len']
    mappings.loc['-' == mappings['strand'], 'right_len'] = tmp
    mappings.reset_index(drop=True, inplace=True)

    # Assign region the mapping belongs to, duplicate mappings if they belong to multiple regions
    if 1 == len(regions):
        # With only one region it is simple
        mappings['region'] = np.where(mappings['in_region'], 0, -1)
        mappings['start'] = regions[0]['start']
        mappings['end'] = regions[0]['end']
        mappings['scaffold'] = regions[0]['scaffold']
    else:
        region_order = []
        same_scaffold = []
        for i, reg in enumerate(regions):
            if len(same_scaffold) and reg['scaffold'] != regions[i-1]['scaffold']:
                region_order.append(same_scaffold)
                same_scaffold = []
                
            same_scaffold.append( (reg['start'], i) )

        region_order.append(same_scaffold)

        for i in range(len(region_order)):
            region_order[i].sort()

        inverted_order = [reg[1] for scaf in region_order for reg in scaf[::-1]]
        region_order = [reg[1] for scaf in region_order for reg in scaf]
    
        # Duplicate all entries and remove the incorrect ones
        mappings = mappings.loc[np.repeat(mappings.index, len(region_order))].copy()
        mappings['region'] = np.where(mappings['strand'] == '+', region_order*(int(len(mappings)/len(region_order))), inverted_order*(int(len(mappings)/len(region_order))))
        mappings = mappings[mappings['in_region'] | (mappings['region'] == 0)].copy() # The mappings not in any region do not need to be duplicated
        mappings.loc[mappings['in_region']==False, 'region'] = -1
        
        mappings = mappings.merge(pd.DataFrame.from_dict(regions).reset_index().rename(columns={'index':'region'}), on=['region'], how='left')
        mappings = mappings[(mappings['region'] == -1) | ((mappings['t_name'] == mappings['scaffold']) & (mappings['t_end'] > mappings['start']) & (mappings['t_start'] <= mappings['end']))].copy()
        
        
    # If the mapping reaches out of the region of interest, ignore left/right scaffold
    mappings['overrun_left'] = mappings['start'] > mappings['t_start']
    mappings.loc[mappings['overrun_left'], 'left_scaf'] = ''
    mappings.loc[mappings['overrun_left'], 'left_dist'] = 0
    mappings.loc[mappings['overrun_left'], 'left_len'] = 0
    mappings['overrun_right'] = mappings['end'] < mappings['t_end']-1
    mappings.loc[mappings['overrun_right'], 'right_scaf'] = ''
    mappings.loc[mappings['overrun_right'], 'right_dist'] = 0
    mappings.loc[mappings['overrun_right'], 'right_len'] = 0
    
    # Remove mappings that are less than min_mapping_length in the region, except if they belong to a read with multiple mappings
    mappings = mappings[ ((min_mapping_length <= mappings['t_end'] - mappings['start']) & (min_mapping_length <= mappings['end'] - mappings['t_start'])) | (mappings['q_name'] == mappings['q_name'].shift(1, fill_value="")) | (mappings['q_name'] == mappings['q_name'].shift(-1, fill_value="")) ].copy()
    
    mappings.drop(columns=['in_region','start','end','scaffold'], inplace=True)
        
    # Check if reads connect regions and if so in what way
    mappings['con_prev_reg'] = 'no'
    if 1 < len(regions):
        # Find indirect connections, where the previous region has a mapping somewhere before the current mapping
        for s in range(2,np.max(mappings.groupby(['q_name'], sort=False).size())+1):
            mappings.loc[('+' == mappings['strand']) & (mappings['q_name'] == mappings['q_name'].shift(s, fill_value="")) & (0 < mappings['region']) & (mappings['region']-1 == mappings['region'].shift(s, fill_value=-1)), 'con_prev_reg'] = 'indirect'
            mappings.loc[('-' == mappings['strand']) & (mappings['q_name'] == mappings['q_name'].shift(-s, fill_value="")) & (0 < mappings['region']) & (mappings['region']-1 == mappings['region'].shift(-s, fill_value=-1)), 'con_prev_reg'] = 'indirect'
        
        # Find direct connections
        mappings.loc[('+' == mappings['strand']) & (mappings['q_name'] == mappings['q_name'].shift(1, fill_value="")) & ('+' == mappings['strand'].shift(1, fill_value="")) &\
                     (0 < mappings['region']) & (mappings['region']-1 == mappings['region'].shift(1, fill_value=-1)), 'con_prev_reg'] = 'direct'
        mappings.loc[('-' == mappings['strand']) & (mappings['q_name'] == mappings['q_name'].shift(-1, fill_value="")) & ('-' == mappings['strand'].shift(-1, fill_value="")) &\
                 (0 < mappings['region']) & (mappings['region']-1 == mappings['region'].shift(-1, fill_value=-1)), 'con_prev_reg'] = 'direct'
        # Clear scaffold connection info for direct connections
        mappings.loc['direct' == mappings['con_prev_reg'], 'left_scaf'] = ''
        mappings.loc[('+' == mappings['strand']) & ('direct' == mappings['con_prev_reg'].shift(-1, fill_value='')), 'right_scaf'] = ''
        mappings.loc[('-' == mappings['strand']) & ('direct' == mappings['con_prev_reg'].shift(1, fill_value='')), 'right_scaf'] = ''
        
        # Find same mappings connecting reads (full connection)
        mappings.loc[('+' == mappings['strand']) & ('direct' == mappings['con_prev_reg']) &  (mappings['q_start'] == mappings['q_start'].shift(1, fill_value="")) &  (mappings['q_end'] == mappings['q_end'].shift(1, fill_value="")), 'con_prev_reg'] = 'full'
        mappings.loc[('-' == mappings['strand']) & ('direct' == mappings['con_prev_reg']) &  (mappings['q_start'] == mappings['q_start'].shift(-1, fill_value="")) &  (mappings['q_end'] == mappings['q_end'].shift(-1, fill_value="")), 'con_prev_reg'] = 'full'
        
    # Remove mappings that are not in a region
    mappings = mappings[0 <= mappings['region']].copy()
    
    # Assign read ids to the mappings sorted from left to right by appearance on screen to give position in drawing
    tmp = mappings.groupby(['q_name'], sort=False)['region'].agg(['size','min'])
    mappings['min_region'] = np.repeat(tmp['min'].values, tmp['size'].values)
    mappings['t_min_start'] = np.where(mappings['min_region'] == mappings['region'], mappings['t_start'], sys.maxsize)
    mappings['t_max_end'] = np.where(mappings['min_region'] == mappings['region'], mappings['t_end'], 0)
    tmp = mappings.groupby(['q_name'], sort=False)[['t_min_start','t_max_end']].agg({'t_min_start':['size','min'], 't_max_end':['max']})
    mappings['t_min_start'] = np.repeat(tmp[('t_min_start','min')].values, tmp[('t_min_start','size')].values)
    mappings['t_max_end'] = np.repeat(tmp[('t_max_end','max')].values, tmp[('t_min_start','size')].values)
    mappings.sort_values(['min_region','t_min_start','t_max_end','q_name','region','t_start','t_end'], ascending=[True,True,True,True,False,False,False], inplace=True)
    mappings['read_id'] = (mappings['q_name'] != mappings['q_name'].shift(1, fill_value="")).cumsum()
    mappings.drop(columns=['min_region','t_min_start','t_max_end'], inplace=True)
    
    # Check whether the reads support a contig break within the region (on left or right side of the read)
    mappings['left_break'] = ((mappings['t_start'] > max_dist_contig_end) &
                             ( (mappings['left_scaf'] != '') | (np.where('+' == mappings['strand'], mappings['q_start'], mappings['q_len']-mappings['q_end']) > min_length_contig_break) ) &
                             (mappings['overrun_left'] == False))

    mappings['right_break'] = ((mappings['t_len']-mappings['t_end'] > max_dist_contig_end) &
                              ( (mappings['right_scaf'] != '') | (np.where('+' == mappings['strand'], mappings['q_len']-mappings['q_end'], mappings['q_start']) > min_length_contig_break) ) &
                              (mappings['overrun_right'] == False))
    
    return mappings

def DrawRegions(draw, fnt, regions, total_x, region_part_height, read_section_height):
    region_bar_height = 10
    region_bar_read_separation = 5
    border_width_x = 50
    region_separation_width = 50
    pixels_separating_ticks = 100
    tick_width = 2
    tick_length = 2
    region_text_dist = 18

    region_bar_y_bottom = region_part_height - region_bar_read_separation
    region_bar_y_top = region_bar_y_bottom - region_bar_height
    length_to_pixels = (total_x - 2*border_width_x - (len(regions)-1)*region_separation_width) / sum( [reg['end']-reg['start'] for reg in regions] )
    
    cur_length = 0
    for i, reg in enumerate(regions):
        regions[i]['start_x'] = border_width_x + i*region_separation_width + int(round(cur_length * length_to_pixels))
        cur_length += reg['end']-reg['start']
        regions[i]['end_x'] = border_width_x + i*region_separation_width + int(round(cur_length * length_to_pixels))

        draw.rectangle((regions[i]['start_x'], region_bar_y_top, regions[i]['end_x'], region_bar_y_bottom), fill=(0, 0, 0), outline=(0, 0, 0))
        w, h = draw.textsize(regions[i]['scaffold'], font=fnt)
        draw.text(((regions[i]['start_x'] + regions[i]['end_x'] - w)/2, region_bar_y_top-2*region_text_dist), regions[i]['scaffold'], font=fnt, fill=(0,0,0))
        
        ticks = []
        ticks.append( (regions[i]['start_x'], str(regions[i]['start'])) )
        tick_num = max(0, int( (regions[i]['end_x']-regions[i]['start_x']) / pixels_separating_ticks ) - 1 )
        for n in range(1, tick_num+1):
            # First find ideal x (without regions[i]['start_x'] yet)
            x = int( (regions[i]['end_x']-regions[i]['start_x']) / (tick_num+1) * n )
            # Then calculate pos accordingly, round it to the next int and set x according to rounded pos
            pos = regions[i]['start']+x/length_to_pixels
            x = regions[i]['start_x'] + int(x + ( int( round(pos) ) - pos)*length_to_pixels )
            ticks.append( (x, str(int( round(pos) ))) )
        ticks.append( (regions[i]['end_x']-1, str(regions[i]['end'])) )
        
        for x, pos in ticks:
            draw.line(((x, region_bar_y_bottom+1), (x, region_bar_y_bottom+read_section_height)), fill=(175, 175, 175), width=tick_width)
            draw.line(((x, region_bar_y_top-tick_length), (x, region_bar_y_top)), fill=(0, 0, 0), width=tick_width)
            w, h = draw.textsize(pos, font=fnt)
            draw.text((x-w/2,region_bar_y_top-region_text_dist), pos, font=fnt, fill=(0,0,0))

    return regions, length_to_pixels

def GetMapQColor(mapq):
    if mapq < 10:
        return (255, 255, 255)
    else:
        return tuple([ int(round(255*c)) for c in sns.color_palette("Blues")[min(5,int(mapq/10)-1)] ])

def DrawArrow(draw, x, y, width, color, direction, head_len):
    if '+' == direction:
        draw.polygon(((x[1], y), (x[1]-head_len, y+width), (x[0], y+width), (x[0]+head_len, y), (x[0], y-width), (x[1]-head_len, y-width)), fill=color, outline=GetMapQColor(60))
    else:
        draw.polygon(((x[0], y), (x[0]+head_len, y-width), (x[1], y-width), (x[1]-head_len, y), (x[1], y+width), (x[0]+head_len, y+width)), fill=color, outline=GetMapQColor(60))
        
def DrawCross(draw, x, y, size, color):
    draw.line(((x-size, y-size), (x+size, y+size)), fill=color, width=4)
    draw.line(((x+size, y-size), (x-size, y+size)), fill=color, width=4)

def DrawMappings(draw, fnt, regions, mappings, region_part_height, pixel_per_read, length_to_pixels):
    arrow_width = 3
    arrow_head_len = 10
    last_read_id = -1
    for row in mappings.itertuples(index=False):
        # Draw line every 10 reads
        y = region_part_height + row.read_id*pixel_per_read
        if last_read_id != row.read_id and 0 == row.read_id % 10:
            for reg in regions:
                draw.line(((reg['start_x'], y), (reg['end_x'], y)), fill=(175, 175, 174), width=1)
        
        # Get positions
        x1 = max(regions[row.region]['start_x'] - (arrow_head_len if row.overrun_left else 0), regions[row.region]['start_x'] + (row.t_start-regions[row.region]['start'])*length_to_pixels)
        x2 = min(regions[row.region]['end_x'] + (arrow_head_len if row.overrun_right else 0), regions[row.region]['start_x'] + (row.t_end-regions[row.region]['start'])*length_to_pixels)
        y = y - int(pixel_per_read/2)
        x0 = max(regions[row.region]['start_x'] - arrow_head_len, x1 - row.left_len*length_to_pixels)
        x3 = min(regions[row.region]['end_x'] + arrow_head_len, x2 + row.right_len*length_to_pixels)

        # Draw read continuation
        if last_read_id != row.read_id and x2 != x3:
            draw.line(((x2 if row.strand == '+' else x2-arrow_head_len, y), (x3, y)), fill=(0, 0, 0), width=1)
        if x0 != x1:
            draw.line(((x0, y), (x1 if row.strand == '-' else x1+arrow_head_len, y)), fill=(0, 0, 0), width=1)

        # Draw mapping
        DrawArrow(draw, (x1, x2), y, arrow_width, GetMapQColor(row.mapq), row.strand, arrow_head_len)
        
        # Draw breaks
        if row.left_break:
            DrawCross(draw, x1, y, 6, (217,173,60))
        if row.right_break:
            DrawCross(draw, x2, y, 6, (217,173,60))
        
        last_read_id = row.read_id


def MiniGapVisualize(region_defs, mapping_file, output, min_mapq, min_mapping_length, min_length_contig_break, keep_all_subreads):
    alignment_precision = 100
    max_dist_contig_end = 2000
    
    regions = []
    for reg in region_defs:
        split = reg.split(':')
        if 2 != len(split):
            print("Incorrect region definition: ", reg)
            sys.exit(1)
        else:
            scaf = split[0]
            split = split[1].split('-')
            if 2 != len(split):
                print("Incorrect region definition: ", reg)
                sys.exit(1)
            else:
                regions.append( {'scaffold':scaf, 'start':int(split[0]), 'end':int(split[1])} )
                
    for i, reg in enumerate(regions):
        for reg2 in regions[:i]:
            if reg['scaffold'] == reg2['scaffold'] and reg['start'] <= reg2['end'] and reg['end'] >= reg2['start']:
                print("Overlapping regions are not permitted: ", reg, " ", reg[2])
                sys.exit(1)
            
    # Prepare mappings
    mappings = GetMappingsInRegion(mapping_file, regions, min_mapq, min_mapping_length, keep_all_subreads, alignment_precision, min_length_contig_break, max_dist_contig_end)
    
    # Prepare drawing
    total_x = 1000
    region_part_height = 55
    pixel_per_read = 13
    
    if len(mappings):
        read_section_height = np.max(mappings['read_id'])*pixel_per_read
    else:
        read_section_height = 0
        
    img = Image.new('RGB', (total_x,region_part_height + read_section_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 12)
    
    # Draw
    regions, length_to_pixels = DrawRegions(draw, fnt, regions, total_x, region_part_height, read_section_height)
    DrawMappings(draw, fnt, regions, mappings, region_part_height, pixel_per_read, length_to_pixels)

    # Save drawing
    img.save(output)

def TestDuplicationConflictResolution():
    # Define test cases
    test_dups = pd.concat([
        #abab#
        #||  #
        #ab--#
        pd.DataFrame({'apid':0,'apos':[0,1,2,3],'ahap':0,'bpid':1,'bpos':[0,1,0,1],'bhap':0,'samedir':True, 'correct':[True,True,False,False]}),
        #aab#
        # ||#
        #-ab#
        pd.DataFrame({'apid':2,'apos':[0,1,2],'ahap':0,'bpid':3,'bpos':[0,0,1],'bhap':0,'samedir':True, 'correct':[False,True,True]}),
        #abbc#
        #||||#
        #abbc#
        pd.DataFrame({'apid':4,'apos':[0,1,1,2,2,3],'ahap':0,'bpid':5,'bpos':[0,1,2,1,2,3],'bhap':0,'samedir':True, 'correct':[True,True,False,False,True,True]}),
        #aabb#
        # || #
        #-ab-#
        pd.DataFrame({'apid':6,'apos':[0,1,2,3],'ahap':0,'bpid':7,'bpos':[0,0,1,1],'bhap':0,'samedir':True, 'correct':[False,True,True,False]}),
        #abcde#
        #| | |#
        #adcbe#
        pd.DataFrame({'apid':8,'apos':[0,1,2,3,4],'ahap':0,'bpid':9,'bpos':[0,3,2,1,4],'bhap':0,'samedir':True, 'correct':[True,False,True,False,True]}),
        #afghibcde#
        #|    |  |#
        #a--dcb--e#
        pd.DataFrame({'apid':10,'apos':[0,5,6,7,8],'ahap':0,'bpid':11,'bpos':[0,3,2,1,4],'bhap':0,'samedir':True, 'correct':[True,True,False,False,True]}),
        #abcb#
        #||| #
        #abc-#
        pd.DataFrame({'apid':12,'apos':[0,1,2,3],'ahap':0,'bpid':13,'bpos':[0,1,2,1],'bhap':0,'samedir':True, 'correct':[True,True,True,False]}),
        #abcb#
        #| ||#
        #a-cb#
        pd.DataFrame({'apid':14,'apos':[0,1,2,3],'ahap':0,'bpid':15,'bpos':[0,2,1,2],'bhap':0,'samedir':True, 'correct':[True,False,True,True]}),
        #aa#
        #||#
        #aa#
        pd.DataFrame({'apid':16,'apos':[0,0,1,1],'ahap':0,'bpid':17,'bpos':[0,1,0,1],'bhap':0,'samedir':True, 'correct':[True,False,False,True]}),
        #abbac#
        #  |||#
        #--bac#
        pd.DataFrame({'apid':18,'apos':[0,1,2,3,4],'ahap':0,'bpid':19,'bpos':[1,0,0,1,2],'bhap':0,'samedir':True, 'correct':[False,False,True,True,True]}),
        #abbc#
        #| ||#
        #adbc#
        pd.DataFrame({'apid':20,'apos':[0,1,2,3],'ahap':0,'bpid':21,'bpos':[0,2,2,3],'bhap':0,'samedir':True, 'correct':[True,False,True,True]}),
        #abcadc#
        #|  | |#
        #adeafc#
        pd.DataFrame({'apid':22,'apos':[0,0,3,3,5],'ahap':0,'bpid':23,'bpos':[0,3,0,3,5],'bhap':0,'samedir':True, 'correct':[True,False,False,True,True]}),
        #ab-cdaefca#
        #|  |||    #
        #agecda----#
        pd.DataFrame({'apid':24,'apos':[0,0,2,3,4,4,5,7,8,8],'ahap':0,'bpid':25,'bpos':[0,5,3,4,0,5,2,3,0,5],'bhap':0,'samedir':True, 'correct':[True,False,True,True,False,True,False,False,False,False]}),
        #a-bcd#
        #| | |#
        #acb-d#
        pd.DataFrame({'apid':26,'apos':[0,1,2,3],'ahap':0,'bpid':27,'bpos':[0,2,1,3],'bhap':0,'samedir':True, 'correct':[True,True,False,True]}),
        #aabcdbd#
        # || |  #
        #-ab-d--#
        pd.DataFrame({'apid':28,'apos':[0,1,2,4,5,6],'ahap':0,'bpid':29,'bpos':[0,0,1,2,1,2],'bhap':0,'samedir':True, 'correct':[False,True,True,True,False,False]}),
        #abc#
        #| |#
        #aac#
        pd.DataFrame({'apid':30,'apos':[0,0,2],'ahap':0,'bpid':31,'bpos':[0,1,2],'bhap':0,'samedir':True, 'correct':[True,False,True]}),
        #abc#
        #| |#
        #acc#
        pd.DataFrame({'apid':32,'apos':[0,2,2],'ahap':0,'bpid':33,'bpos':[0,1,2],'bhap':0,'samedir':True, 'correct':[True,False,True]}),
        #abc-#
        #| | #
        #aacc#
        pd.DataFrame({'apid':34,'apos':[0,0,2,2],'ahap':0,'bpid':35,'bpos':[0,1,2,3],'bhap':0,'samedir':True, 'correct':[True,False,True,False]}),
        #abcadd#
        #||  | #
        #ab--d-#
        pd.DataFrame({'apid':36,'apos':[0,1,3,4,5],'ahap':0,'bpid':37,'bpos':[0,1,0,2,2],'bhap':0,'samedir':True, 'correct':[True,True,False,True,False]})
        ], ignore_index=True)
#
    # Invert the test cases to check that they are consistent for both options
    test_dups = pd.concat([test_dups, test_dups.rename(columns={'apid':'bpid','apos':'bpos','ahap':'bhap','bpid':'apid','bpos':'apos','bhap':'ahap'})], ignore_index=True)
#
    # Run function
    duplications = test_dups.drop(columns=['correct'])
    duplications = RequireContinuousDirectionForDuplications(duplications)
#
    # Compare and report failed tests
    cols = [col for col in test_dups.columns if col != 'correct']
    test_dups['chosen'] = test_dups.merge(duplications[cols].drop_duplicates(), on=cols, how='left', indicator=True)['_merge'].values == "both"
    test_dups = test_dups[np.isin(test_dups['apid'].values, test_dups.loc[test_dups['correct'] != test_dups['chosen'], 'apid'].values)].copy()
#
    if len(test_dups) == 0:
        return False # Success
    else:
        print("TestDuplicationConflictResolution failed in", len(np.unique(test_dups['apid'].values)), "cases:")
        print(test_dups)
        return True

def FillMissingHaplotypes(scaffold_paths, ploidy):
    for i in range(len(scaffold_paths)):
        for h in range(1, ploidy):
            if f'scaf{h}' not in scaffold_paths[i].columns:
                scaffold_paths[i][f'phase{h}'] = -(scaffold_paths[i]['pid']*10 + h)
                scaffold_paths[i][[f'scaf{h}',f'strand{h}',f'dist{h}']] = [-1,'',0]
#
    return scaffold_paths

def CheckConsistencyOfScaffoldGraph(scaffold_graph):
    inconsistent = []
    for l in np.unique(scaffold_graph['length']):
        if f'scaf{l-1}' not in scaffold_graph.columns:
            inconsistent.append( scaffold_graph[scaffold_graph['length'] == l].copy() ) # If the column does not exist all entries with this length are wrong
        else:
            inconsistent.append( scaffold_graph[(scaffold_graph['length'] == l) & np.isnan(scaffold_graph[f'scaf{l-1}'])].copy() )
        if f'scaf{l}' in scaffold_graph.columns:
            inconsistent.append( scaffold_graph[(scaffold_graph['length'] == l) & (np.isnan(scaffold_graph[f'scaf{l}']) == False)].copy() )
    if len(inconsistent):
        inconsistent = pd.concat(inconsistent, ignore_index=False)
    if len(inconsistent):
        print("Warning: Inconsistent entries in scaffold_graph for test:")
        print(inconsistent)

def TestFilterInvalidConnections():
    # Define test cases
    ploidy = 4
    scaffold_paths = []
    scaffold_graph = []
    splits = []
    man_patha = []
    man_pathb = []
    man_ends = []
    # Case 1
    scaffold_paths.append( pd.DataFrame({
    'pid':      [    1,    1,    1,    1,    1],
    'pos':      [    0,    1,    2,    3,    4],
    'phase0':   [   10,   10,   10,   10,   10],
    'scaf0':    [  101,  102,  105,  106,  109],
    'strand0':  [  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0],
    'phase1':   [  -11,   11,  -11,   11,  -11],
    'scaf1':    [   -1,  103,   -1,  107,   -1],
    'strand1':  [   '',  '+',   '',  '+',   ''],
    'dist1':    [    0,    0,    0,    0,    0],
    'phase2':   [  -12,   12,  -12,   12,  -12],
    'scaf2':    [   -1,  104,   -1,  108,   -1],
    'strand2':  [   '',  '+',   '',  '+',   ''],
    'dist2':    [    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from':  101, 'from_side': 'r', 'length':  5, 'scaf1':  102, 'strand1': '+', 'dist1':    0, 'scaf2':  105, 'strand2': '+', 'dist2':    0, 'scaf3':  106, 'strand3': '+', 'dist3':    0, 'scaf4':  109, 'strand4': '+', 'dist4':    0},
        {'from':  101, 'from_side': 'r', 'length':  5, 'scaf1':  103, 'strand1': '+', 'dist1':    0, 'scaf2':  105, 'strand2': '+', 'dist2':    0, 'scaf3':  107, 'strand3': '+', 'dist3':    0, 'scaf4':  109, 'strand4': '+', 'dist4':    0},
        {'from':  101, 'from_side': 'r', 'length':  3, 'scaf1':  104, 'strand1': '+', 'dist1':    0, 'scaf2':  105, 'strand2': '+', 'dist2':    0},
        {'from':  105, 'from_side': 'r', 'length':  3, 'scaf1':  108, 'strand1': '+', 'dist1':    0, 'scaf2':  109, 'strand2': '+', 'dist2':    0}
        ]) )
    for p in range(1,4):
        splits.append(pd.DataFrame({'pid': 1, 'pos': p, 'ahap':[0,0,1,1,2,2,2], 'bhap':[0,2,1,2,0,1,2], 'valid_path':'ab'}))
    splits.append(pd.DataFrame({'pid': 1, 'pos': 1, 'ahap':[0,1], 'bhap':[1,0], 'valid_path':'b'}))
    splits.append(pd.DataFrame({'pid': 1, 'pos': 3, 'ahap':[0,1], 'bhap':[1,0], 'valid_path':'a'}))
    # Case 2
    scaffold_paths.append( pd.DataFrame({
    'pid':      [    2,    2,    2,    2,    2,    2],
    'pos':      [    0,    1,    2,    3,    4,    5],
    'phase0':   [   20,   20,   20,   20,   20,   20],
    'scaf0':    [  201,  202,  205,  206,  207,  210],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0],
    'phase1':   [  -21,   21,  -21,  -21,   21,  -21],
    'scaf1':    [   -1,  203,   -1,   -1,  208,   -1],
    'strand1':  [   '',  '+',   '',   '',  '+',   ''],
    'dist1':    [    0,    0,    0,    0,    0,    0],
    'phase2':   [  -22,   22,  -22,  -22,   22,  -22],
    'scaf2':    [   -1,  204,   -1,   -1,  209,   -1],
    'strand2':  [   '',  '+',   '',   '',  '+',   ''],
    'dist2':    [    0,    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from':  201, 'from_side': 'r', 'length':  6, 'scaf1':  202, 'strand1': '+', 'dist1':    0, 'scaf2':  205, 'strand2': '+', 'dist2':    0, 'scaf3':  206, 'strand3': '+', 'dist3':    0, 'scaf4':  207, 'strand4': '+', 'dist4':    0, 'scaf5':  210, 'strand5': '+', 'dist5':    0},
        {'from':  201, 'from_side': 'r', 'length':  6, 'scaf1':  203, 'strand1': '+', 'dist1':    0, 'scaf2':  205, 'strand2': '+', 'dist2':    0, 'scaf3':  206, 'strand3': '+', 'dist3':    0, 'scaf4':  208, 'strand4': '+', 'dist4':    0, 'scaf5':  210, 'strand5': '+', 'dist5':    0},
        {'from':  201, 'from_side': 'r', 'length':  3, 'scaf1':  204, 'strand1': '+', 'dist1':    0, 'scaf2':  205, 'strand2': '+', 'dist2':    0},
        {'from':  206, 'from_side': 'r', 'length':  3, 'scaf1':  209, 'strand1': '+', 'dist1':    0, 'scaf2':  210, 'strand2': '+', 'dist2':    0}
        ]) )
    for p in range(1,5):
        splits.append(pd.DataFrame({'pid': 2, 'pos': p, 'ahap':[0,0,1,1,2,2,2], 'bhap':[0,2,1,2,0,1,2], 'valid_path':'ab'}))
    splits.append(pd.DataFrame({'pid': 2, 'pos': 1, 'ahap':[0,1], 'bhap':[1,0], 'valid_path':'b'}))
    splits.append(pd.DataFrame({'pid': 2, 'pos': 4, 'ahap':[0,1], 'bhap':[1,0], 'valid_path':'a'}))
    # Case 3
    scaffold_paths.append( pd.DataFrame({
    'pid':      [    3,    3,    3,    3,    3,    3,    3],
    'pos':      [    0,    1,    2,    3,    4,    5,    6],
    'phase0':   [   30,   30,   30,   30,   30,   30,   30],
    'scaf0':    [  301,  302,  305,  306,  307,  308,  311],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0,    0],
    'phase1':   [  -31,   31,  -31,  -31,  -31,   31,  -31],
    'scaf1':    [   -1,  303,   -1,   -1,   -1,  309,   -1],
    'strand1':  [   '',  '+',   '',   '',   '',  '+',   ''],
    'dist1':    [    0,    0,    0,    0,    0,    0,    0],
    'phase2':   [  -32,   32,  -32,  -32,  -32,   32,  -32],
    'scaf2':    [   -1,  304,   -1,   -1,   -1,  310,   -1],
    'strand2':  [   '',  '+',   '',   '',   '',  '+',   ''],
    'dist2':    [    0,    0,    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from':  301, 'from_side': 'r', 'length':  7, 'scaf1':  302, 'strand1': '+', 'dist1':    0, 'scaf2':  305, 'strand2': '+', 'dist2':    0, 'scaf3':  306, 'strand3': '+', 'dist3':    0, 'scaf4':  307, 'strand4': '+', 'dist4':    0, 'scaf5':  308, 'strand5': '+', 'dist5':    0, 'scaf6':  311, 'strand6': '+', 'dist6':    0},
        {'from':  301, 'from_side': 'r', 'length':  7, 'scaf1':  303, 'strand1': '+', 'dist1':    0, 'scaf2':  305, 'strand2': '+', 'dist2':    0, 'scaf3':  306, 'strand3': '+', 'dist3':    0, 'scaf4':  307, 'strand4': '+', 'dist4':    0, 'scaf5':  309, 'strand5': '+', 'dist5':    0, 'scaf6':  311, 'strand6': '+', 'dist6':    0},
        {'from':  301, 'from_side': 'r', 'length':  3, 'scaf1':  304, 'strand1': '+', 'dist1':    0, 'scaf2':  305, 'strand2': '+', 'dist2':    0},
        {'from':  307, 'from_side': 'r', 'length':  3, 'scaf1':  310, 'strand1': '+', 'dist1':    0, 'scaf2':  311, 'strand2': '+', 'dist2':    0}
        ]) )
    for p in range(1,6):
        splits.append(pd.DataFrame({'pid': 3, 'pos': p, 'ahap':[0,0,1,1,2,2,2], 'bhap':[0,2,1,2,0,1,2], 'valid_path':'ab'}))
    splits.append(pd.DataFrame({'pid': 3, 'pos': 1, 'ahap':[0,1], 'bhap':[1,0], 'valid_path':'b'}))
    splits.append(pd.DataFrame({'pid': 3, 'pos': 5, 'ahap':[0,1], 'bhap':[1,0], 'valid_path':'a'}))
    # Case 4
    scaffold_paths.append( pd.DataFrame({
    'pid':      [    4,    4,    4,    4,    4,    4],
    'pos':      [    0,    1,    2,    3,    4,    5],
    'phase0':   [   40,   40,   40,   40,   40,   40],
    'scaf0':    [  401,  402,  404,  405,  406,  408],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0],
    'phase1':   [  -41,   41,  -41,  -41,   41,  -41],
    'scaf1':    [   -1,  403,   -1,   -1,  407,   -1],
    'strand1':  [   '',  '+',   '',   '',  '+',   ''],
    'dist1':    [    0,    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from':  401, 'from_side': 'r', 'length':  3, 'scaf1':  402, 'strand1': '+', 'dist1':    0, 'scaf2':  404, 'strand2': '+', 'dist2':    0},
        {'from':  401, 'from_side': 'r', 'length':  3, 'scaf1':  403, 'strand1': '+', 'dist1':    0, 'scaf2':  404, 'strand2': '+', 'dist2':    0},
        {'from':  404, 'from_side': 'r', 'length':  2, 'scaf1':  405, 'strand1': '+', 'dist1':    0},
        {'from':  405, 'from_side': 'r', 'length':  3, 'scaf1':  406, 'strand1': '+', 'dist1':    0, 'scaf2':  408, 'strand2': '+', 'dist2':    0},
        {'from':  405, 'from_side': 'r', 'length':  3, 'scaf1':  407, 'strand1': '+', 'dist1':    0, 'scaf2':  408, 'strand2': '+', 'dist2':    0}
        ]) )
    for p in range(1,5):
        splits.append(pd.DataFrame({'pid': 4, 'pos': p, 'ahap':[0,0,1,1], 'bhap':[0,1,0,1], 'valid_path':'ab'}))
    # Case 5
    scaffold_paths.append( pd.DataFrame({
    'pid':      [    5,    5,    5,    5,    5,    5],
    'pos':      [    0,    1,    2,    3,    4,    5],
    'phase0':   [   50,   50,   50,   50,   50,   50],
    'scaf0':    [  501,  502,  504,  505,  506,  508],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0],
    'phase1':   [  -51,   51,  -51,  -51,   51,  -51],
    'scaf1':    [   -1,  503,   -1,   -1,  507,   -1],
    'strand1':  [   '',  '+',   '',   '',  '+',   ''],
    'dist1':    [    0,    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from':  501, 'from_side': 'r', 'length':  4, 'scaf1':  502, 'strand1': '+', 'dist1':    0, 'scaf2':  504, 'strand2': '+', 'dist2':    0, 'scaf3':  505, 'strand3': '+', 'dist3':    0},
        {'from':  501, 'from_side': 'r', 'length':  3, 'scaf1':  503, 'strand1': '+', 'dist1':    0, 'scaf2':  504, 'strand2': '+', 'dist2':    0},
        {'from':  505, 'from_side': 'r', 'length':  3, 'scaf1':  506, 'strand1': '+', 'dist1':    0, 'scaf2':  508, 'strand2': '+', 'dist2':    0},
        {'from':  505, 'from_side': 'r', 'length':  3, 'scaf1':  507, 'strand1': '+', 'dist1':    0, 'scaf2':  508, 'strand2': '+', 'dist2':    0}
        ]) )
    for p in range(1,5):
        splits.append(pd.DataFrame({'pid': 5, 'pos': p, 'ahap':[0,0,1,1], 'bhap':[0,1,0,1], 'valid_path':'ab'}))
    # Case 6
    scaffold_paths.append( pd.DataFrame({
    'pid':      [    6,    6,    6,    6,    6,    6],
    'pos':      [    0,    1,    2,    3,    4,    5],
    'phase0':   [   60,   60,   60,   60,   60,   60],
    'scaf0':    [  601,  602,  604,  605,  606,  607],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0],
    'phase1':   [  -61,   61,  -61,  -61,   61,  -61],
    'scaf1':    [   -1,  603,   -1,   -1,  606,   -1],
    'strand1':  [   '',  '+',   '',   '',  '+',   ''],
    'dist1':    [    0,    0,    0,    0,  100,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from':  601, 'from_side': 'r', 'length':  4, 'scaf1':  602, 'strand1': '+', 'dist1':    0, 'scaf2':  604, 'strand2': '+', 'dist2':    0, 'scaf3':  605, 'strand3': '+', 'dist3':    0},
        {'from':  601, 'from_side': 'r', 'length':  3, 'scaf1':  603, 'strand1': '+', 'dist1':    0, 'scaf2':  604, 'strand2': '+', 'dist2':    0},
        {'from':  604, 'from_side': 'r', 'length':  4, 'scaf1':  605, 'strand1': '+', 'dist1':    0, 'scaf2':  606, 'strand2': '+', 'dist2':  100, 'scaf3':  607, 'strand3': '+', 'dist3':    0},
        {'from':  605, 'from_side': 'r', 'length':  3, 'scaf1':  606, 'strand1': '+', 'dist1':    0, 'scaf2':  607, 'strand2': '+', 'dist2':    0}
        ]) )
    for p in range(1,5):
        splits.append(pd.DataFrame({'pid': 6, 'pos': p, 'ahap':[0,0,1,1], 'bhap':[0,1,0,1], 'valid_path':'ab'}))
    # Case 7
    scaffold_paths.append( pd.DataFrame({
    'pid':      [    7,    7,    7,    7,    7],
    'pos':      [    0,    1,    2,    3,    4],
    'phase0':   [   70,   70,   70,   70,   70],
    'scaf0':    [  701,  702,  704,  705,  707],
    'strand0':  [  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0],
    'phase1':   [  -71,   71,  -71,   71,  -71],
    'scaf1':    [   -1,  703,   -1,  706,   -1],
    'strand1':  [   '',  '+',   '',  '+',   ''],
    'dist1':    [    0,    0,    0,    0,    0],
    'phase2':   [  -72,   72,  -72,   72,  -72],
    'scaf2':    [   -1,   -1,   -1,   -1,   -1],
    'strand2':  [   '',   '',   '',   '',   ''],
    'dist2':    [    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from':  701, 'from_side': 'r', 'length':  5, 'scaf1':  702, 'strand1': '+', 'dist1':    0, 'scaf2':  704, 'strand2': '+', 'dist2':    0, 'scaf3':  705, 'strand3': '+', 'dist3':    0, 'scaf4':  707, 'strand4': '+', 'dist4':    0},
        {'from':  701, 'from_side': 'r', 'length':  5, 'scaf1':  703, 'strand1': '+', 'dist1':    0, 'scaf2':  704, 'strand2': '+', 'dist2':    0, 'scaf3':  706, 'strand3': '+', 'dist3':    0, 'scaf4':  707, 'strand4': '+', 'dist4':    0},
        {'from':  701, 'from_side': 'r', 'length':  4, 'scaf1':  703, 'strand1': '+', 'dist1':    0, 'scaf2':  704, 'strand2': '+', 'dist2':    0, 'scaf3':  707, 'strand3': '+', 'dist3':    0},
        {'from':  701, 'from_side': 'r', 'length':  4, 'scaf1':  704, 'strand1': '+', 'dist1':    0, 'scaf2':  706, 'strand2': '+', 'dist2':    0, 'scaf3':  707, 'strand3': '+', 'dist3':    0},
        {'from':  701, 'from_side': 'r', 'length':  3, 'scaf1':  704, 'strand1': '+', 'dist1':    0, 'scaf2':  707, 'strand2': '+', 'dist2':    0}
        ]) )
    splits.append(pd.DataFrame({'pid': 7, 'pos': 2, 'ahap':[0,1,1,2,2], 'bhap':[0,1,2,1,2], 'valid_path':'ab'}))
    # Case 8
    scaffold_paths.append( pd.DataFrame({
    'pid':      [    8,    8,    8,    8,    8],
    'pos':      [    0,    1,    2,    3,    4],
    'phase0':   [   80,   80,   80,   80,   80],
    'scaf0':    [  801,  802,  804,  805,  806],
    'strand0':  [  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0],
    'phase1':   [  -81,   81,  -81,  -81,  -81],
    'scaf1':    [   -1,  803,   -1,   -1,   -1],
    'strand1':  [   '',  '+',   '',   '',   ''],
    'dist1':    [    0,    0,    0,    0,    0],
    'phase2':   [  -82,   82,  -82,  -82,  -82],
    'scaf2':    [   -1,   -1,   -1,   -1,   -1],
    'strand2':  [   '',   '',   '',   '',   ''],
    'dist2':    [    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from':  801, 'from_side': 'r', 'length':  5, 'scaf1':  802, 'strand1': '+', 'dist1':    0, 'scaf2':  804, 'strand2': '+', 'dist2':    0, 'scaf3':  805, 'strand3': '+', 'dist3':    0, 'scaf4':  806, 'strand4': '+', 'dist4':    0},
        {'from':  801, 'from_side': 'r', 'length':  5, 'scaf1':  803, 'strand1': '+', 'dist1':    0, 'scaf2':  804, 'strand2': '+', 'dist2':    0, 'scaf3':  805, 'strand3': '+', 'dist3':    0, 'scaf4':  806, 'strand4': '+', 'dist4':    0},
        {'from':  801, 'from_side': 'r', 'length':  2, 'scaf1':  804, 'strand1': '+', 'dist1':    0}
        ]) )
    for p in range(2,4):
        splits.append(pd.DataFrame({'pid': 8, 'pos': p, 'ahap':[0,0,0,1,1,1,2,2,2], 'bhap':[0,1,2,0,1,2,0,1,2], 'valid_path':'ab'}))
    # Case 9
    scaffold_paths.append( pd.DataFrame({
    'pid':      [    9,    9,    9,    9,    9,    9,    9,    9],
    'pos':      [    0,    1,    2,    3,    4,    5,    6,    7],
    'phase0':   [   90,   90,   90,   90,   90,   90,   90,   90],
    'scaf0':    [  901,  902,  904,  905,  906,  905,  906,  907],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0,    0,    0],
    'phase1':   [  -91,   91,  -91,  -91,  -91,  -91,  -91,  -91],
    'scaf1':    [   -1,  903,   -1,   -1,   -1,   -1,   -1,   -1],
    'strand1':  [   '',  '+',   '',   '',   '',   '',   '',   ''],
    'dist1':    [    0,    0,    0,    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from':  901, 'from_side': 'r', 'length':  4, 'scaf1':  902, 'strand1': '+', 'dist1':    0, 'scaf2':  904, 'strand2': '+', 'dist2':    0, 'scaf3':  905, 'strand3': '+', 'dist3':    0},
        {'from':  901, 'from_side': 'r', 'length':  3, 'scaf1':  903, 'strand1': '+', 'dist1':    0, 'scaf2':  904, 'strand2': '+', 'dist2':    0},
        {'from':  904, 'from_side': 'r', 'length':  6, 'scaf1':  905, 'strand1': '+', 'dist1':    0, 'scaf2':  906, 'strand2': '+', 'dist2':    0, 'scaf3':  905, 'strand3': '+', 'dist3':    0, 'scaf4':  906, 'strand4': '+', 'dist4':    0, 'scaf5':  907, 'strand5': '+', 'dist5':    0}
        ]) )
    for p in range(1,7):
        splits.append(pd.DataFrame({'pid': 9, 'pos': p, 'ahap':[0,0,1,1], 'bhap':[0,1,0,1], 'valid_path':'ab'}))
    # Case 10
    scaffold_paths.append( pd.DataFrame({
    'pid':      [   10,   10,   10,   10,   10,   10],
    'pos':      [    0,    1,    2,    3,    4,    5],
    'phase0':   [  100,  100,  100,  100,  100,  100],
    'scaf0':    [ 1001, 1002, 1004, 1005, 1006, 1007],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0],
    'phase1':   [ -101,  101, -101, -101, -101, -101],
    'scaf1':    [   -1, 1003,   -1,   -1,   -1,   -1],
    'strand1':  [   '',  '+',   '',   '',   '',   ''],
    'dist1':    [    0,    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 1001, 'from_side': 'r', 'length':  4, 'scaf1': 1002, 'strand1': '+', 'dist1':    0, 'scaf2': 1004, 'strand2': '+', 'dist2':    0, 'scaf3': 1005, 'strand3': '+', 'dist3':    0},
        {'from': 1001, 'from_side': 'r', 'length':  3, 'scaf1': 1003, 'strand1': '+', 'dist1':    0, 'scaf2': 1004, 'strand2': '+', 'dist2':    0},
        {'from': 1004, 'from_side': 'r', 'length':  6, 'scaf1': 1005, 'strand1': '+', 'dist1':    0, 'scaf2': 1006, 'strand2': '+', 'dist2':    0, 'scaf3': 1005, 'strand3': '+', 'dist3':    0, 'scaf4': 1006, 'strand4': '+', 'dist4':    0, 'scaf5': 1007, 'strand5': '+', 'dist5':    0}
        ]) )
    splits.append(pd.DataFrame({'pid': 10, 'pos': [3,4], 'ahap':-1, 'bhap':-1, 'valid_path':''}))
    # Case 11
    scaffold_paths.append( pd.DataFrame({
    'pid':      [   11,   11,   11,   11,   11,   11,   11,   11],
    'pos':      [    0,    1,    2,    3,    4,    5,    6,    7],
    'phase0':   [  110,  110,  110,  110,  110,  110,  110,  110],
    'scaf0':    [ 1101, 1102, 1104, 1105, 1106, 1105, 1106, 1107],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0,    0,    0],
    'phase1':   [ -111,  111, -111, -111, -111, -111, -111, -111],
    'scaf1':    [   -1, 1103,   -1,   -1,   -1,   -1,   -1,   -1],
    'strand1':  [   '',  '+',   '',   '',   '',   '',   '',   ''],
    'dist1':    [    0,    0,    0,    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 1101, 'from_side': 'r', 'length':  4, 'scaf1': 1102, 'strand1': '+', 'dist1':    0, 'scaf2': 1104, 'strand2': '+', 'dist2':    0, 'scaf3': 1105, 'strand3': '+', 'dist3':    0},
        {'from': 1101, 'from_side': 'r', 'length':  3, 'scaf1': 1103, 'strand1': '+', 'dist1':    0, 'scaf2': 1104, 'strand2': '+', 'dist2':    0},
        {'from': 1105, 'from_side': 'r', 'length':  5, 'scaf1': 1106, 'strand1': '+', 'dist1':    0, 'scaf2': 1105, 'strand2': '+', 'dist2':    0, 'scaf3': 1106, 'strand3': '+', 'dist3':    0, 'scaf4': 1107, 'strand4': '+', 'dist4':    0}
        ]) )
    for p in range(1,7):
        splits.append(pd.DataFrame({'pid': 11, 'pos': p, 'ahap':[0,0,1,1], 'bhap':[0,1,0,1], 'valid_path':'ab'}))
    # Case 12
    scaffold_paths.append( pd.DataFrame({
    'pid':      [   12,   12,   12,   12,   12,   12],
    'pos':      [    0,    1,    2,    3,    4,    5],
    'phase0':   [  120,  120,  120,  120,  120,  120],
    'scaf0':    [ 1201, 1202, 1204, 1205, 1206, 1207],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0],
    'phase1':   [ -121,  121, -121, -121, -121, -121],
    'scaf1':    [   -1, 1203,   -1,   -1,   -1,   -1],
    'strand1':  [   '',  '+',   '',   '',   '',   ''],
    'dist1':    [    0,    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 1201, 'from_side': 'r', 'length':  4, 'scaf1': 1202, 'strand1': '+', 'dist1':    0, 'scaf2': 1204, 'strand2': '+', 'dist2':    0, 'scaf3': 1205, 'strand3': '+', 'dist3':    0},
        {'from': 1201, 'from_side': 'r', 'length':  3, 'scaf1': 1203, 'strand1': '+', 'dist1':    0, 'scaf2': 1204, 'strand2': '+', 'dist2':    0},
        {'from': 1205, 'from_side': 'r', 'length':  5, 'scaf1': 1206, 'strand1': '+', 'dist1':    0, 'scaf2': 1205, 'strand2': '+', 'dist2':    0, 'scaf3': 1206, 'strand3': '+', 'dist3':    0, 'scaf4': 1207, 'strand4': '+', 'dist4':    0}
        ]) )
    for p in [3,4]:
        splits.append(pd.DataFrame({'pid': 12, 'pos': p, 'ahap':[0,0,1,1], 'bhap':[0,1,0,1], 'valid_path':'ab'})) # Except for bridged repeats, over or underrepresented repeats are not something this function can take care of (Instead we allow everything and later block due to ambiguity, so that we do not make errors and other functions can handle it)
    # Case 13
    scaffold_paths.append( pd.DataFrame({
    'pid':      [   13,   13,   13,   13,   13,   13,   13,   13],
    'pos':      [    0,    1,    2,    3,    4,    5,    6,    7],
    'phase0':   [  130,  130,  130,  130,  130,  130,  130,  130],
    'scaf0':    [ 1301, 1302, 1304, 1305, 1306, 1305, 1306, 1307],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0,    0,    0],
    'phase1':   [  -91,  131,  -91,  -91,  -91,  -91,  -91,  -91],
    'scaf1':    [   -1, 1303,   -1,   -1,   -1,   -1,   -1,   -1],
    'strand1':  [   '',  '+',   '',   '',   '',   '',   '',   ''],
    'dist1':    [    0,    0,    0,    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 1301, 'from_side': 'r', 'length':  4, 'scaf1': 1302, 'strand1': '+', 'dist1':    0, 'scaf2': 1304, 'strand2': '+', 'dist2':    0, 'scaf3': 1305, 'strand3': '+', 'dist3':    0},
        {'from': 1301, 'from_side': 'r', 'length':  3, 'scaf1': 1303, 'strand1': '+', 'dist1':    0, 'scaf2': 1304, 'strand2': '+', 'dist2':    0},
        {'from': 1306, 'from_side': 'r', 'length':  4, 'scaf1': 1305, 'strand1': '+', 'dist1':    0, 'scaf2': 1306, 'strand2': '+', 'dist2':    0, 'scaf3': 1307, 'strand3': '+', 'dist3':    0}
        ]) )
    for p in range(1,7):
        splits.append(pd.DataFrame({'pid': 13, 'pos': p, 'ahap':[0,0,1,1], 'bhap':[0,1,0,1], 'valid_path':'ab'}))
    # Case 14
    scaffold_paths.append( pd.DataFrame({
    'pid':      [   14,   14,   14,   14,   14,   14],
    'pos':      [    0,    1,    2,    3,    4,    5],
    'phase0':   [  140,  140,  140,  140,  140,  140],
    'scaf0':    [ 1401, 1402, 1404, 1405, 1406, 1407],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0],
    'phase1':   [ -141,  141, -141, -141, -141, -141],
    'scaf1':    [   -1, 1403,   -1,   -1,   -1,   -1],
    'strand1':  [   '',  '+',   '',   '',   '',   ''],
    'dist1':    [    0,    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 1401, 'from_side': 'r', 'length':  4, 'scaf1': 1402, 'strand1': '+', 'dist1':    0, 'scaf2': 1404, 'strand2': '+', 'dist2':    0, 'scaf3': 1405, 'strand3': '+', 'dist3':    0},
        {'from': 1401, 'from_side': 'r', 'length':  3, 'scaf1': 1403, 'strand1': '+', 'dist1':    0, 'scaf2': 1404, 'strand2': '+', 'dist2':    0},
        {'from': 1406, 'from_side': 'r', 'length':  4, 'scaf1': 1405, 'strand1': '+', 'dist1':    0, 'scaf2': 1406, 'strand2': '+', 'dist2':    0, 'scaf3': 1407, 'strand3': '+', 'dist3':    0}
        ]) )
    for p in [3,4]:
        splits.append(pd.DataFrame({'pid': 14, 'pos': p, 'ahap':[0,0,1,1], 'bhap':[0,1,0,1], 'valid_path':'ab'})) # Except for bridged repeats, over or underrepresented repeats are not something this function can take care of (Instead we allow everything and later block due to ambiguity, so that we do not make errors and other functions can handle it)
    # Case 15
    scaffold_paths.append( pd.DataFrame({
    'pid':      [   15,   15,   15,   15,   15,   15,   15],
    'pos':      [    0,    1,    2,    3,    4,    5,    6],
    'phase0':   [  150,  150,  150,  150,  150,  150,  150],
    'scaf0':    [ 1501, 1502, 1505, 1506, 1507, 1509, 1511],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0,    0],
    'phase1':   [ -151,  151, -151, -151,  151,  151, -151],
    'scaf1':    [   -1, 1503,   -1,   -1, 1508, 1510,   -1],
    'strand1':  [   '',  '+',   '',   '',  '+',  '+',   ''],
    'dist1':    [    0,    0,    0,    0,    0,    0,    0],
    'phase2':   [ -152,  152, -152, -152, -152,  152, -152],
    'scaf2':    [   -1, 1504,   -1,   -1,   -1, 1510,   -1],
    'strand2':  [   '',  '+',   '',   '',   '',  '+',   ''],
    'dist2':    [    0,    0,    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 1501, 'from_side': 'r', 'length':  7, 'scaf1': 1502, 'strand1': '+', 'dist1':    0, 'scaf2': 1505, 'strand2': '+', 'dist2':    0, 'scaf3': 1506, 'strand3': '+', 'dist3':    0, 'scaf4': 1507, 'strand4': '+', 'dist4':    0, 'scaf5': 1509, 'strand5': '+', 'dist5':    0, 'scaf6': 1511, 'strand6': '+', 'dist6':    0},
        {'from': 1501, 'from_side': 'r', 'length':  3, 'scaf1': 1503, 'strand1': '+', 'dist1':    0, 'scaf2': 1505, 'strand2': '+', 'dist2':    0},
        {'from': 1501, 'from_side': 'r', 'length':  3, 'scaf1': 1504, 'strand1': '+', 'dist1':    0, 'scaf2': 1505, 'strand2': '+', 'dist2':    0},
        {'from': 1506, 'from_side': 'r', 'length':  4, 'scaf1': 1508, 'strand1': '+', 'dist1':    0, 'scaf2': 1510, 'strand2': '+', 'dist2':    0, 'scaf3': 1511, 'strand3': '+', 'dist3':    0},
        {'from': 1507, 'from_side': 'r', 'length':  3, 'scaf1': 1510, 'strand1': '+', 'dist1':    0, 'scaf2': 1511, 'strand2': '+', 'dist2':    0}
        ]) )
    for p in range(1,4):
        splits.append(pd.DataFrame({'pid': 15, 'pos': p, 'ahap':[0,0,0,1,1,1,2,2,2], 'bhap':[0,1,2,0,1,2,0,1,2], 'valid_path':'ab'}))
    for p in [4,5]:
        splits.append(pd.DataFrame({'pid': 15, 'pos': p, 'ahap':[0,0,0,1,1,2,2,2], 'bhap':[0,1,2,1,2,0,1,2], 'valid_path':'ab'}))
    splits.append(pd.DataFrame({'pid': 15, 'pos': [4,5], 'ahap':[1,1], 'bhap':[0,0], 'valid_path':['b','a']}))
    # Case 16
    scaffold_paths.append( pd.DataFrame({
    'pid':      [   16,   16,   16,   16,   16,   16,   16],
    'pos':      [    0,    1,    2,    3,    4,    5,    6],
    'phase0':   [  160,  160,  160,  160,  160,  160,  160],
    'scaf0':    [ 1601, 1602, 1603, 1604, 1605, 1606, 1607],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0,    0],
    'phase1':   [ -161, -161, -161,  161, -161,  161, -161],
    'scaf1':    [   -1,   -1,   -1,   -1,   -1, 1606,   -1],
    'strand1':  [   '',   '',   '',   '',   '',  '-',   ''],
    'dist1':    [    0,    0,    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 1601, 'from_side': 'r', 'length':  5, 'scaf1': 1602, 'strand1': '+', 'dist1':    0, 'scaf2': 1603, 'strand2': '+', 'dist2':    0, 'scaf3': 1604, 'strand3': '+', 'dist3':    0, 'scaf4': 1605, 'strand4': '+', 'dist4':    0},
        {'from': 1602, 'from_side': 'r', 'length':  6, 'scaf1': 1603, 'strand1': '+', 'dist1':    0, 'scaf2': 1604, 'strand2': '+', 'dist2':    0, 'scaf3': 1605, 'strand3': '+', 'dist3':    0, 'scaf4': 1606, 'strand4': '+', 'dist4':    0, 'scaf5': 1607, 'strand5': '+', 'dist5':    0},
        {'from': 1603, 'from_side': 'r', 'length':  4, 'scaf1': 1605, 'strand1': '+', 'dist1':    0, 'scaf2': 1606, 'strand2': '-', 'dist2':    0, 'scaf3': 1607, 'strand3': '+', 'dist3':    0}
        ]) )
    for p in [1,2]:
        splits.append(pd.DataFrame({'pid': 16, 'pos': p, 'ahap':[0,0,1,1], 'bhap':[0,1,0,1], 'valid_path':'ab'}))
    splits.append(pd.DataFrame({'pid': 16, 'pos': 4, 'ahap':[0,1], 'bhap':[0,1], 'valid_path':'ab'}))
    # Case 17
    man_patha.append( pd.DataFrame({
    'pid':      [   17,   17,   17,   17,   17],
    'pos':      [    0,    1,    2,    3,    4],
    'phase0':   [  170,  170,  170,  170,  170],
    'scaf0':    [ 1701, 1702, 1703, 1704, 1705],
    'strand0':  [  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0]
    }) )
    man_pathb.append( pd.DataFrame({
    'pid':      [   17,   17,   17,   17,   17],
    'pos':      [    0,    1,    2,    3,    4],
    'phase0':   [  170,  170,  170,  170,  170],
    'scaf0':    [ 1703, 1704, 1705, 1706, 1707],
    'strand0':  [  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0],
    'phase1':   [ -171,  171, -171,  171, -171],
    'scaf1':    [   -1,   -1,   -1, 1706,   -1],
    'strand1':  [   '',   '',   '',  '-',   ''],
    'dist1':    [    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 1701, 'from_side': 'r', 'length':  5, 'scaf1': 1702, 'strand1': '+', 'dist1':    0, 'scaf2': 1703, 'strand2': '+', 'dist2':    0, 'scaf3': 1704, 'strand3': '+', 'dist3':    0, 'scaf4': 1705, 'strand4': '+', 'dist4':    0},
        {'from': 1702, 'from_side': 'r', 'length':  6, 'scaf1': 1703, 'strand1': '+', 'dist1':    0, 'scaf2': 1704, 'strand2': '+', 'dist2':    0, 'scaf3': 1705, 'strand3': '+', 'dist3':    0, 'scaf4': 1706, 'strand4': '+', 'dist4':    0, 'scaf5': 1707, 'strand5': '+', 'dist5':    0},
        {'from': 1703, 'from_side': 'r', 'length':  4, 'scaf1': 1705, 'strand1': '+', 'dist1':    0, 'scaf2': 1706, 'strand2': '-', 'dist2':    0, 'scaf3': 1707, 'strand3': '+', 'dist3':    0}
    ]) )
    man_ends.append( pd.DataFrame([ # Differences in the overlapping region need to be ignored and need separate hanlding to be able to combine these two paths
        {'apid': 17, 'ahap': 0, 'bpid': 17, 'bhap': 0, 'amin':  2, 'amax':  4, 'bmin':  0, 'bmax':  2, 'matches':  3, 'alen':  4, 'blen':  4, 'aside': 'r', 'bside': 'l', 'valid_path':'ab'},
        {'apid': 17, 'ahap': 0, 'bpid': 17, 'bhap': 1, 'amin':  2, 'amax':  4, 'bmin':  0, 'bmax':  2, 'matches':  3, 'alen':  4, 'blen':  4, 'aside': 'r', 'bside': 'l', 'valid_path':'b'},
        ]) )
    # Case 18
    scaffold_paths.append( pd.DataFrame({
    'pid':      [   18,   18,   18,   18],
    'pos':      [    0,    1,    2,    3],
    'phase0':   [  180,  180,  180,  180],
    'scaf0':    [ 1801, 1802, 1803, 1804],
    'strand0':  [  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0],
    'phase1':   [ -181,  181, -181, -181],
    'scaf1':    [   -1, 1802,   -1,   -1],
    'strand1':  [   '',  '+',   '',   ''],
    'dist1':    [    0,  100,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 1801, 'from_side': 'r', 'length':  4, 'scaf1': 1802, 'strand1': '+', 'dist1':    0, 'scaf2': 1803, 'strand2': '+', 'dist2':    0, 'scaf3': 1805, 'strand3': '+', 'dist3':    0},
        {'from': 1801, 'from_side': 'r', 'length':  4, 'scaf1': 1802, 'strand1': '+', 'dist1':  100, 'scaf2': 1803, 'strand2': '+', 'dist2':    0, 'scaf3': 1804, 'strand3': '+', 'dist3':    0},
        {'from': 1806, 'from_side': 'r', 'length':  3, 'scaf1': 1801, 'strand1': '+', 'dist1':    0, 'scaf2': 1802, 'strand2': '+', 'dist2':    0},
        {'from': 1807, 'from_side': 'r', 'length':  4, 'scaf1': 1801, 'strand1': '+', 'dist1':    0, 'scaf2': 1802, 'strand2': '+', 'dist2':    0, 'scaf3': 1803, 'strand3': '+', 'dist3':    0}
        ]) )
    splits.append(pd.DataFrame({'pid': 18, 'pos': 1, 'ahap':1, 'bhap':[0,1], 'valid_path':'ab'}))
    # Case 19
    scaffold_paths.append( pd.DataFrame({
    'pid':      [   19,   19,   19,   19,   19,   19],
    'pos':      [    0,    1,    2,    3,    4,    5],
    'phase0':   [  190,  190,  190,  190,  190,  190],
    'scaf0':    [ 1901, 1902, 1904, 1905, 1906, 1907],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0],
    'phase1':   [ -191,  191, -191, -191, -191, -191],
    'scaf1':    [   -1, 1903,   -1,   -1,   -1,   -1],
    'strand1':  [   '',  '+',   '',   '',   '',   ''],
    'dist1':    [    0,    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 1901, 'from_side': 'r', 'length':  6, 'scaf1': 1902, 'strand1': '+', 'dist1':    0, 'scaf2': 1904, 'strand2': '+', 'dist2':    0, 'scaf3': 1905, 'strand3': '+', 'dist3':    0, 'scaf4': 1906, 'strand4': '+', 'dist4':    0, 'scaf5': 1907, 'strand5': '+', 'dist5':    0},
        {'from': 1901, 'from_side': 'r', 'length':  5, 'scaf1': 1903, 'strand1': '+', 'dist1':    0, 'scaf2': 1904, 'strand2': '+', 'dist2':    0, 'scaf3': 1905, 'strand3': '+', 'dist3':    0, 'scaf4': 1908, 'strand4': '+', 'dist4':    0}
        ]) )
    splits.append(pd.DataFrame({'pid': 19, 'pos': 1, 'ahap':[0,0,1], 'bhap':[0,1,0], 'valid_path':['ab','a','b']}))
    for p in range(2,5):
        splits.append(pd.DataFrame({'pid': 19, 'pos': p, 'ahap':[0,0], 'bhap':[0,1], 'valid_path':'ab'}))
    # Case 20
    scaffold_paths.append( pd.DataFrame({
    'pid':      [   20,   20,   20,   20,   20,   20,   20,   20,   20],
    'pos':      [    0,    1,    2,    3,    4,    5,    6,    7,    8],
    'phase0':   [  200,  200,  200,  200,  200,  200,  200,  200,  200],
    'scaf0':    [ 2001, 2002, 2004, 2005, 2006, 2007, 2008, 2009, 2011],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0,    0,    0,    0],
    'phase1':   [ -201,  201, -201, -201,  201, -201, -201,  201, -201],
    'scaf1':    [   -1, 2003,   -1,   -1,   -1,   -1,   -1, 2010,   -1],
    'strand1':  [   '',  '+',   '',   '',   '',   '',   '',  '+',   ''],
    'dist1':    [    0,    0,    0,    0,    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 2001, 'from_side': 'r', 'length':  9, 'scaf1': 2002, 'strand1': '+', 'dist1':    0, 'scaf2': 2004, 'strand2': '+', 'dist2':    0, 'scaf3': 2005, 'strand3': '+', 'dist3':    0, 'scaf4': 2006, 'strand4': '+', 'dist4':    0, 'scaf5': 2007, 'strand5': '+', 'dist5':    0, 'scaf6': 2008, 'strand6': '+', 'dist6':    0, 'scaf7': 2009, 'strand7': '+', 'dist7':    0, 'scaf8': 2011, 'strand8': '+', 'dist8':    0},
        {'from': 2001, 'from_side': 'r', 'length':  3, 'scaf1': 2003, 'strand1': '+', 'dist1':    0, 'scaf2': 2004, 'strand2': '+', 'dist2':    0},
        {'from': 2005, 'from_side': 'r', 'length':  2, 'scaf1': 2007, 'strand1': '+', 'dist1':    0},
        {'from': 2008, 'from_side': 'r', 'length':  3, 'scaf1': 2010, 'strand1': '+', 'dist1':    0, 'scaf2': 2011, 'strand2': '+', 'dist2':    0}
        ]) )
    for p in [2,3,5,6]:
        splits.append(pd.DataFrame({'pid': 20, 'pos': p, 'ahap':[0,0,1,1], 'bhap':[0,1,0,1], 'valid_path':'ab'}))
    # Case 21
    scaffold_paths.append( pd.DataFrame({
    'pid':      [   21,   21,   21,   21,   21,   21],
    'pos':      [    0,    1,    2,    3,    4,    5],
    'phase0':   [  210,  210,  210,  210,  210,  210],
    'scaf0':    [ 2101, 2102,   -1, 2103, 2104, 2106],
    'strand0':  [  '+',  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0,    0],
    'phase1':   [ -211, -211,  211, -211,  211, -211],
    'scaf1':    [   -1,   -1, 2102,   -1, 2105,   -1],
    'strand1':  [   '',   '',  '+',   '',  '+',   ''],
    'dist1':    [    0,    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 2101, 'from_side': 'r', 'length':  4, 'scaf1': 2102, 'strand1': '+', 'dist1':    0, 'scaf2': 2102, 'strand2': '+', 'dist2':    0, 'scaf3': 2103, 'strand3': '+', 'dist3':    0},
        {'from': 2101, 'from_side': 'r', 'length':  3, 'scaf1': 2102, 'strand1': '+', 'dist1':    0, 'scaf2': 2103, 'strand2': '+', 'dist2':    0},
        {'from': 2103, 'from_side': 'r', 'length':  3, 'scaf1': 2104, 'strand1': '+', 'dist1':    0, 'scaf2': 2106, 'strand2': '+', 'dist2':    0},
        {'from': 2103, 'from_side': 'r', 'length':  3, 'scaf1': 2105, 'strand1': '+', 'dist1':    0, 'scaf2': 2106, 'strand2': '+', 'dist2':    0}
        ]) )
    for p in [1,3]:
        splits.append(pd.DataFrame({'pid': 21, 'pos': p, 'ahap':[0,0,1,1], 'bhap':[0,1,0,1], 'valid_path':'ab'}))
    # Case 22
    man_patha.append( pd.DataFrame({
    'pid':      [   22,   22,   22],
    'pos':      [    0,    1,    2],
    'phase0':   [  220,  220,  220],
    'scaf0':    [ 2201, 2202,   -1],
    'strand0':  [  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0],
    'phase1':   [ -221, -221,  221],
    'scaf1':    [   -1,   -1, 2202],
    'strand1':  [   '',   '',  '+'],
    'dist1':    [    0,    0,    0]
    }) )
    man_pathb.append( pd.DataFrame({
    'pid':      [   22,   22,   22,   22],
    'pos':      [    0,    1,    2,    3],
    'phase0':   [  220,  220,  220,  220],
    'scaf0':    [   -1, 2203, 2204, 2206],
    'strand0':  [  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0],
    'phase1':   [  221, -221,  221, -221],
    'scaf1':    [ 2202,   -1, 2205,   -1],
    'strand1':  [  '+',   '',  '+',   ''],
    'dist1':    [    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 2201, 'from_side': 'r', 'length':  4, 'scaf1': 2202, 'strand1': '+', 'dist1':    0, 'scaf2': 2202, 'strand2': '+', 'dist2':    0, 'scaf3': 2203, 'strand3': '+', 'dist3':    0},
        {'from': 2201, 'from_side': 'r', 'length':  3, 'scaf1': 2202, 'strand1': '+', 'dist1':    0, 'scaf2': 2203, 'strand2': '+', 'dist2':    0},
        {'from': 2203, 'from_side': 'r', 'length':  3, 'scaf1': 2204, 'strand1': '+', 'dist1':    0, 'scaf2': 2206, 'strand2': '+', 'dist2':    0},
        {'from': 2203, 'from_side': 'r', 'length':  3, 'scaf1': 2205, 'strand1': '+', 'dist1':    0, 'scaf2': 2206, 'strand2': '+', 'dist2':    0}
    ]) )
    man_ends.append( pd.DataFrame([ # Differences in the overlapping region need to be ignored and need separate hanlding to be able to combine these two paths
        {'apid': 22, 'ahap': 1, 'bpid': 22, 'bhap': 1, 'amin':  2, 'amax':  2, 'bmin':  0, 'bmax':  0, 'matches':  1, 'alen':  2, 'blen':  3, 'aside': 'r', 'bside': 'l', 'valid_path':'ab'}
        ]) )
    # Case 23
    man_patha.append( pd.DataFrame({
    'pid':      [   23,   23,   23,   23,   23],
    'pos':      [    0,    1,    2,    3,    4],
    'phase0':   [  230,  230,  230,  230,  230],
    'scaf0':    [ 2301, 2302, 2303, 2304, 2305],
    'strand0':  [  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0]
    }) )
    man_pathb.append( pd.DataFrame({
    'pid':      [   23,   23,   23,   23,   23],
    'pos':      [    0,    1,    2,    3,    4],
    'phase0':   [  230,  230,  230,  230,  230],
    'scaf0':    [ 2303, 2304, 2305, 2306, 2307],
    'strand0':  [  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0],
    'phase1':   [ -231,  231, -231, -231, -231],
    'scaf1':    [   -1,   -1,   -1,   -1,   -1],
    'strand1':  [   '',   '',   '',   '',   ''],
    'dist1':    [    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 2301, 'from_side': 'r', 'length':  5, 'scaf1': 2302, 'strand1': '+', 'dist1':    0, 'scaf2': 2303, 'strand2': '+', 'dist2':    0, 'scaf3': 2304, 'strand3': '+', 'dist3':    0, 'scaf4': 2305, 'strand4': '+', 'dist4':    0},
        {'from': 2302, 'from_side': 'r', 'length':  6, 'scaf1': 2303, 'strand1': '+', 'dist1':    0, 'scaf2': 2304, 'strand2': '+', 'dist2':    0, 'scaf3': 2305, 'strand3': '+', 'dist3':    0, 'scaf4': 2306, 'strand4': '+', 'dist4':    0, 'scaf5': 2307, 'strand5': '+', 'dist5':    0},
        {'from': 2308, 'from_side': 'r', 'length':  5, 'scaf1': 2303, 'strand1': '+', 'dist1':    0, 'scaf2': 2305, 'strand2': '+', 'dist2':    0, 'scaf3': 2306, 'strand3': '+', 'dist3':    0, 'scaf4': 2307, 'strand4': '+', 'dist4':    0}
    ]) )
    man_ends.append( pd.DataFrame([ # Differences in the overlapping region need to be ignored and need separate hanlding to be able to combine these two paths
        {'apid': 23, 'ahap': 0, 'bpid': 23, 'bhap': 0, 'amin':  2, 'amax':  4, 'bmin':  0, 'bmax':  2, 'matches':  3, 'alen':  4, 'blen':  4, 'aside': 'r', 'bside': 'l', 'valid_path':'ab'},
        {'apid': 23, 'ahap': 0, 'bpid': 23, 'bhap': 1, 'amin':  2, 'amax':  4, 'bmin':  0, 'bmax':  2, 'matches':  3, 'alen':  4, 'blen':  4, 'aside': 'r', 'bside': 'l', 'valid_path':'a'},
        ]) )
    # Case 24
    man_patha.append( pd.DataFrame({
    'pid':      [   24,   24,   24,   24],
    'pos':      [    0,    1,    2,    3],
    'phase0':   [  240,  240,  240,  240],
    'scaf0':    [ 2401, 2402, 2403, 2404],
    'strand0':  [  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0]
    }) )
    man_pathb.append( pd.DataFrame({
    'pid':      [   24,   24,   24,   24,   24],
    'pos':      [    0,    1,    2,    3,    4],
    'phase0':   [  240,  240,  240,  240,  240],
    'scaf0':    [ 2403, 2404, 2406, 2407, 2408],
    'strand0':  [  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0],
    'phase1':   [ -241,  241, -241, -241, -241],
    'scaf1':    [   -1, 2405,   -1,   -1,   -1],
    'strand1':  [   '',   '',   '',   '',   ''],
    'dist1':    [    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 2401, 'from_side': 'r', 'length':  5, 'scaf1': 2402, 'strand1': '+', 'dist1':    0, 'scaf2': 2403, 'strand2': '+', 'dist2':    0, 'scaf3': 2404, 'strand3': '+', 'dist3':    0, 'scaf4': 2406, 'strand4': '+', 'dist4':    0},
        {'from': 2402, 'from_side': 'r', 'length':  6, 'scaf1': 2403, 'strand1': '+', 'dist1':    0, 'scaf2': 2404, 'strand2': '+', 'dist2':    0, 'scaf3': 2406, 'strand3': '+', 'dist3':    0, 'scaf4': 2407, 'strand4': '+', 'dist4':    0, 'scaf5': 2408, 'strand5': '+', 'dist5':    0},
        {'from': 2409, 'from_side': 'r', 'length':  6, 'scaf1': 2403, 'strand1': '+', 'dist1':    0, 'scaf2': 2405, 'strand2': '+', 'dist2':    0, 'scaf3': 2406, 'strand3': '+', 'dist3':    0, 'scaf4': 2407, 'strand4': '+', 'dist4':    0, 'scaf5': 2408, 'strand5': '+', 'dist5':    0}
    ]) )
    man_ends.append( pd.DataFrame([ # Differences in the overlapping region need to be ignored and need separate hanlding to be able to combine these two paths
        {'apid': 24, 'ahap': 0, 'bpid': 24, 'bhap': 0, 'amin':  2, 'amax':  3, 'bmin':  0, 'bmax':  1, 'matches':  2, 'alen':  3, 'blen':  4, 'aside': 'r', 'bside': 'l', 'valid_path':'ab'},
        {'apid': 24, 'ahap': 0, 'bpid': 24, 'bhap': 1, 'amin':  2, 'amax':  3, 'bmin':  0, 'bmax':  1, 'matches':  2, 'alen':  3, 'blen':  4, 'aside': 'r', 'bside': 'l', 'valid_path':'a'},
        ]) )
    # Case 25
    scaffold_paths.append( pd.DataFrame({
    'pid':      [   25,   25,   25,   25,   25],
    'pos':      [    0,    1,    2,    3,    4],
    'phase0':   [  250,  250,  250,  250,  250],
    'scaf0':    [ 2501, 2502, 2503, 2504, 2505],
    'strand0':  [  '+',  '+',  '+',  '+',  '+'],
    'dist0':    [    0,    0,    0,    0,    0]
    }) )
    scaffold_graph.append( pd.DataFrame([
        {'from': 2501, 'from_side': 'r', 'length':  4, 'scaf1': 2502, 'strand1': '+', 'dist1':    0, 'scaf2': 2503, 'strand2': '+', 'dist2':    0, 'scaf3': 2504, 'strand3': '+', 'dist3':    0},
        {'from': 2506, 'from_side': 'r', 'length':  5, 'scaf1': 2502, 'strand1': '+', 'dist1':    0, 'scaf2': 2503, 'strand2': '+', 'dist2':    0, 'scaf3': 2504, 'strand3': '+', 'dist3':    0, 'scaf4': 2505, 'strand4': '+', 'dist4':    0},
        {'from': 2506, 'from_side': 'r', 'length':  5, 'scaf1': 2502, 'strand1': '+', 'dist1':    0, 'scaf2': 2503, 'strand2': '+', 'dist2':    0, 'scaf3': 2504, 'strand3': '+', 'dist3':    0, 'scaf4': 2509, 'strand4': '+', 'dist4':    0},
        {'from': 2510, 'from_side': 'r', 'length':  5, 'scaf1': 2501, 'strand1': '+', 'dist1':    0, 'scaf2': 2502, 'strand2': '+', 'dist2':    0, 'scaf3': 2503, 'strand3': '+', 'dist3':    0, 'scaf4': 2508, 'strand4': '+', 'dist4':    0},
        {'from': 2507, 'from_side': 'r', 'length':  5, 'scaf1': 2501, 'strand1': '+', 'dist1':    0, 'scaf2': 2502, 'strand2': '+', 'dist2':    0, 'scaf3': 2503, 'strand3': '+', 'dist3':    0, 'scaf4': 2508, 'strand4': '+', 'dist4':    0}
        ]) )
    splits.append(pd.DataFrame({'pid': 25, 'pos': [1,2,3], 'ahap':0, 'bhap':0, 'valid_path':'ab'}))
#
    # Fill missing haplotype with empty ones and combine scaffold_paths
    scaffold_paths = pd.concat(FillMissingHaplotypes(scaffold_paths, ploidy), ignore_index=True)
    man_patha = pd.concat(FillMissingHaplotypes(man_patha, ploidy), ignore_index=True)
    man_pathb = pd.concat(FillMissingHaplotypes(man_pathb, ploidy), ignore_index=True)
    # Combine scaffold_graph and add the shorter and reverse paths
    scaffold_graph = pd.concat(scaffold_graph, ignore_index=True)
    CheckConsistencyOfScaffoldGraph(scaffold_graph)
    reverse_graph = scaffold_graph.rename(columns={'from':'scaf0','from_side':'strand0'})
    reverse_graph['strand0'] = np.where(reverse_graph['strand0'] == 'r', '+', '-')
    reverse_graph = ReverseVerticalPaths(reverse_graph).drop(columns=['lindex'])
    reverse_graph.rename(columns={'scaf0':'from','strand0':'from_side'}, inplace=True)
    reverse_graph['from_side'] = np.where(reverse_graph['from_side'] == '+', 'r', 'l')
    scaffold_graph = pd.concat([scaffold_graph,reverse_graph], ignore_index=True)
    tmp_graph = [ scaffold_graph.copy() ]
    while len(scaffold_graph):
        scaffold_graph = scaffold_graph[scaffold_graph['length'] > 2].copy()
        if len(scaffold_graph):
            scaffold_graph.drop(columns=['from','from_side','dist1'], inplace=True)
            scaffold_graph.rename(columns={**{'scaf1':'from', 'strand1':'from_side'}, **{f'{n}{s}':f'{n}{s-1}' for s in range(2,scaffold_graph['length'].max()) for n in ['scaf','strand','dist']}}, inplace=True)
            scaffold_graph['from_side'] = np.where(scaffold_graph['from_side'] == '+', 'r', 'l')
            scaffold_graph['length'] -= 1
            tmp_graph.append( scaffold_graph.copy() )
    scaffold_graph = pd.concat(tmp_graph, ignore_index=True)
    scaffold_graph[['from','scaf1','dist1']] = scaffold_graph[['from','scaf1','dist1']].astype(int).values
    scaffold_graph.sort_values(['from','from_side'] + [f'{n}{s}' for s in range(1,scaffold_graph['length'].max()) for n in ['scaf','strand','dist']], inplace=True)
    scaffold_graph = RemoveRedundantEntriesInScaffoldGraph(scaffold_graph)
    # Prepare ends
    splits = pd.concat(splits, ignore_index=True)
    split_pos = splits[['pid','pos']].drop_duplicates()
    splits = splits[splits['ahap'] >= 0].copy()
    split_pos['apid'] = np.arange(len(split_pos))*2
    split_pos['bpid'] = split_pos['apid'] + 1
    splits[['apid','bpid']] = splits[['pid','pos']].merge(split_pos, on=['pid','pos'], how='left')[['apid','bpid']].values
    ends = split_pos.copy()
    ends['len'] = ends[['pid']].merge(scaffold_paths.groupby(['pid'])['pos'].max().reset_index(), on=['pid'], how='left')['pos'].values
    ends['nhaps'] = ends[['pid']].merge(GetNumberOfHaplotypes(scaffold_paths, ploidy), on=['pid'], how='left')['nhaps'].values
    ends['index'] = ends.index.values
    ends = ends.loc[np.repeat(ends.index.values, ends['nhaps'].values ** 2)].copy()
    ends['ahap'] = ends.groupby(['index'], sort=False).cumcount() // ends['nhaps'].values
    ends['bhap'] = ends.groupby(['index'], sort=False).cumcount() % ends['nhaps'].values
    ends['amin'] = ends['pos']
    ends['amax'] = ends['amin']
    ends[['bmin','bmax']] = 0
    ends['matches'] = 1
    ends['alen'] = ends['amax']
    ends['blen'] = ends['len'] - ends['pos']
    ends['aside'] = 'r'
    ends['bside'] = 'l'
    ends = ends[['apid','ahap','bpid','bhap','amin','amax','bmin','bmax','matches','alen','blen','aside','bside']].copy()
    # Split scaffold_path according to splits(split_pos)
    apath = scaffold_paths.merge(split_pos[['pid','apid','pos']].rename(columns={'pos':'mpos'}), on=['pid'], how='left')
    apath['pid'] = apath['apid']
    apath = apath[apath['pos'] <= apath['mpos']].drop(columns=['apid','mpos'])
    bpath = scaffold_paths.merge(split_pos[['pid','bpid','pos']].rename(columns={'pos':'mpos'}), on=['pid'], how='left')
    bpath['pid'] = bpath['bpid']
    bpath = bpath[bpath['pos'] >= bpath['mpos']].drop(columns=['bpid','mpos'])
    bpath['pos'] = bpath.groupby(['pid'], sort=False).cumcount()
    # Add manual entries
    man_ends = pd.concat(man_ends, ignore_index=True)
    man_test_pids = man_ends[['apid']].drop_duplicates()
    man_test_pids.rename(columns={'apid':'pid'}, inplace=True)
    man_test_pids['apid'] = np.arange(len(man_test_pids))*2 + split_pos['apid'].max() + 2
    man_test_pids['bpid'] = man_test_pids['apid'] + 1
    split_pos = pd.concat([split_pos, man_test_pids], ignore_index=True)
    man_ends[['apid','bpid']] = man_ends[['apid']].rename(columns={'apid':'pid'}).merge(man_test_pids, on=['pid'], how='left')[['apid','bpid']].values
    splits = pd.concat([splits, man_ends.loc[man_ends['valid_path'] != '', ['apid','bpid','ahap','bhap','valid_path']]], ignore_index=True)
    ends = pd.concat([ends, man_ends.drop(columns=['valid_path'])], ignore_index=True)
    man_patha['pid'] = man_patha[['pid']].merge(man_test_pids[['pid','apid']], on=['pid'], how='left')['apid'].values
    man_pathb['pid'] = man_pathb[['pid']].merge(man_test_pids[['pid','bpid']], on=['pid'], how='left')['bpid'].values
    scaffold_paths = pd.concat([apath, bpath, man_patha, man_pathb], ignore_index=True)
    scaffold_paths.sort_values(['pid','pos'], inplace=True)
    scaffold_paths = SetDistanceAtFirstPositionToZero(scaffold_paths, ploidy)
    # Make ends symmetrical
    ends = pd.concat([ends, ends.rename(columns={**{col:f'b{col[1:]}' for col in ends.columns if col[0] == "a"}, **{col:f'a{col[1:]}' for col in ends.columns if col[0] == "b"}})], ignore_index=True)
#
    # Run function
    graph_ext = FindValidExtensionsInScaffoldGraph(scaffold_graph)
    #print(graph_ext['org'][np.isin(graph_ext['org']['scaf0'], [1802])])
    #print(graph_ext['ext'][np.isin(graph_ext['ext']['scaf0'], [1802])])
    #print(graph_ext['pairs'][np.isin(graph_ext['pairs'][['oindex','eindex']], list(range(348,351))).any(axis=1)])
    #print(ends[np.isin(ends[['apid','bpid']].values, [118,119]).all(axis=1)].copy())
    #print(ends[np.isin(ends[['apid','bpid']], [118,119]).any(axis=1)])
    #print(scaffold_graph[(scaffold_graph['from'] == 1801)])
    #print(scaffold_graph[(scaffold_graph['from'] == 1802)])
    #print(scaffold_graph[(scaffold_graph['from'] == 1803)])
    #print(scaffold_graph[(scaffold_graph['from'] == 1804)])
    #print(scaffold_graph[(scaffold_graph['from'] == 1805)])
    ends = FilterInvalidConnections(ends, scaffold_paths, scaffold_graph, graph_ext, ploidy)
#
    # Compare and report failed tests
    splits['correct'] = True
    if len(ends):
        ends = ends[ends['apid'] < ends['bpid']].copy()
        ends['obtained'] = True
        splits = splits.merge(ends[['ahap','bhap','apid','bpid','valid_path','obtained']], on=['ahap','bhap','apid','bpid','valid_path'], how='outer')
        splits[['pid','pos']] = splits[['apid']].merge(split_pos[['apid','pid','pos']], on=['apid'], how='left')[['pid','pos']].values
        splits['pid'] = splits['pid'].astype(int)
        splits[['correct','obtained']] = splits[['correct','obtained']].fillna(False).values
    else:
        splits['obtained'] = False
    splits = splits.loc[splits['correct'] != splits['obtained'], ['pid','pos','apid','bpid','ahap','bhap','correct','valid_path','obtained']].copy()
    if len(splits) == 0:
        # Everything as it should be
        return False
    else:
        print("TestFilterInvalidConnections failed:")
        splits.sort_values(['pid','apid','bpid','ahap','bhap'], inplace=True)
        print(splits)
        return True

def TestTraverseScaffoldingGraph():
    # Define test cases
    ploidy = 2
    max_loop_units = 10
    scaffolds = [] # (test==new scaffold_graph): ', '.join(np.unique(test['from']).astype(str))
    scaffold_graph = [] # print(",\n".join([str(" "*8+"{") + ', '.join(["'{}': {:>6}".format(k,v) for k,v in zip(entry[entry.isnull() == False].index.values,[val[:-2] if val[-2:] == ".0" else (f"'{val}'" if val in ['-','+','l','r'] else val) for val in entry[entry.isnull() == False].astype(str).values])]) + "}" for entry in [scaffold_graph.loc[i, ['from','from_side','length']+[f'{n}{s}' for s in range(1,scaffold_graph['length'].max()) for n in ['scaf','strand','dist']]] for i in scaffold_graph[scaffold_graph['from'] == x].index.values]]))
    scaf_bridges = [] # tmp = test[['from','from_side','scaf1','strand1','dist1']].rename(columns={'scaf1':'to','strand1':'to_side','dist1':'mean_dist'}); tmp['to_side'] = np.where(tmp['to_side'] == '+', 'l', 'r'); tmp=tmp.merge(scaf_bridges, on=list(tmp.columns), how='inner').drop_duplicates(); print(",\n".join([str(" "*8+"{") + ', '.join(["'{}': {:>6}".format(k,v) for k,v in zip(entry[entry.isnull() == False].index.values,["{:8.6f}".format(float(val)) if val[:2] == "0." else (f"'{val}'" if val in ['-','+','l','r'] else val) for val in entry[entry.isnull() == False].astype(str).values])]) + "}" for entry in [tmp.loc[i] for i in tmp.index.values]]))
    org_scaf_conns = [pd.DataFrame({'from':[],'from_side':[],'to':[],'to_side':[],'distance':[]})]
    result_paths = []
#
    # Test 1
    scaffolds.append( pd.DataFrame({'case':1, 'scaffold':[44, 69, 114, 115, 372, 674, 929, 1306, 2722, 2725, 2799, 2885, 9344, 10723, 11659, 12896, 12910, 13029, 13434, 13452, 13455, 13591, 14096, 15177, 15812, 20727, 26855, 30179, 30214, 31749, 31756, 32229, 33144, 33994, 40554, 41636, 47404, 47516, 49093, 51660, 53480, 56740, 56987, 58443, 70951, 71091, 76860, 96716, 99004, 99341, 99342, 101215, 101373, 107483, 107484, 107485, 107486, 109207, 110827, 112333, 115803, 117303, 117304, 118890, 118892]}) )
    scaffold_graph.append( pd.DataFrame([
        {'from':     44, 'from_side':    'l', 'length':      5, 'scaf1':  99342, 'strand1':    '-', 'dist1':    -44, 'scaf2':  99341, 'strand2':    '-', 'dist2':      0, 'scaf3':   9344, 'strand3':    '+', 'dist3':      0, 'scaf4': 118890, 'strand4':    '-', 'dist4':    -13},
        {'from':     44, 'from_side':    'r', 'length':      3, 'scaf1':   1306, 'strand1':    '-', 'dist1':    -43, 'scaf2':  71091, 'strand2':    '+', 'dist2':    -45},
        {'from':     44, 'from_side':    'r', 'length':      2, 'scaf1':  71091, 'strand1':    '+', 'dist1':      1},
        {'from':     69, 'from_side':    'l', 'length':      3, 'scaf1':  47404, 'strand1':    '-', 'dist1':    -43, 'scaf2':  30179, 'strand2':    '-', 'dist2':    -32},
        {'from':     69, 'from_side':    'l', 'length':      2, 'scaf1': 110827, 'strand1':    '-', 'dist1':    -43},
        {'from':     69, 'from_side':    'r', 'length':      5, 'scaf1':  13434, 'strand1':    '-', 'dist1':    -41, 'scaf2': 112333, 'strand2':    '+', 'dist2':     38, 'scaf3':    114, 'strand3':    '+', 'dist3':   -819, 'scaf4':    115, 'strand4':    '+', 'dist4':   1131},
        {'from':     69, 'from_side':    'r', 'length':      5, 'scaf1':  13455, 'strand1':    '-', 'dist1':    -43, 'scaf2': 112333, 'strand2':    '+', 'dist2':    -43, 'scaf3':  41636, 'strand3':    '-', 'dist3':   -710, 'scaf4':    115, 'strand4':    '+', 'dist4':   1932},
        {'from':    114, 'from_side':    'l', 'length':      4, 'scaf1': 112333, 'strand1':    '-', 'dist1':   -819, 'scaf2':  13434, 'strand2':    '+', 'dist2':     38, 'scaf3':     69, 'strand3':    '-', 'dist3':    -41},
        {'from':    114, 'from_side':    'r', 'length':      3, 'scaf1':    115, 'strand1':    '+', 'dist1':   1131, 'scaf2': 115803, 'strand2':    '+', 'dist2':    885},
        {'from':    115, 'from_side':    'l', 'length':      5, 'scaf1':    114, 'strand1':    '-', 'dist1':   1131, 'scaf2': 112333, 'strand2':    '-', 'dist2':   -819, 'scaf3':  13434, 'strand3':    '+', 'dist3':     38, 'scaf4':     69, 'strand4':    '-', 'dist4':    -41},
        {'from':    115, 'from_side':    'l', 'length':      5, 'scaf1':  41636, 'strand1':    '+', 'dist1':   1932, 'scaf2': 112333, 'strand2':    '-', 'dist2':   -710, 'scaf3':  13455, 'strand3':    '+', 'dist3':    -43, 'scaf4':     69, 'strand4':    '-', 'dist4':    -43},
        {'from':    115, 'from_side':    'r', 'length':      3, 'scaf1':  15177, 'strand1':    '+', 'dist1':    -42, 'scaf2': 115803, 'strand2':    '+', 'dist2':      5},
        {'from':    115, 'from_side':    'r', 'length':      2, 'scaf1': 115803, 'strand1':    '+', 'dist1':    885},
        {'from':    372, 'from_side':    'r', 'length':      5, 'scaf1': 107485, 'strand1':    '+', 'dist1':  -1436, 'scaf2': 107486, 'strand2':    '+', 'dist2':      0, 'scaf3': 101373, 'strand3':    '-', 'dist3':  -1922, 'scaf4':  70951, 'strand4':    '+', 'dist4':      0},
        {'from':    674, 'from_side':    'l', 'length':      2, 'scaf1':   2799, 'strand1':    '+', 'dist1':     60},
        {'from':    674, 'from_side':    'r', 'length':      2, 'scaf1':  20727, 'strand1':    '-', 'dist1':    -68},
        {'from':    674, 'from_side':    'r', 'length':      2, 'scaf1':  20727, 'strand1':    '-', 'dist1':    -18},
        {'from':    929, 'from_side':    'l', 'length':      6, 'scaf1': 117303, 'strand1':    '-', 'dist1':   2544, 'scaf2': 107484, 'strand2':    '+', 'dist2':   -888, 'scaf3': 107485, 'strand3':    '+', 'dist3':      0, 'scaf4': 107486, 'strand4':    '+', 'dist4':      0, 'scaf5':  33994, 'strand5':    '+', 'dist5':    -44},
        {'from':    929, 'from_side':    'l', 'length':      2, 'scaf1': 117304, 'strand1':    '-', 'dist1':   2489},
        {'from':    929, 'from_side':    'r', 'length':      2, 'scaf1':   2725, 'strand1':    '+', 'dist1':   1205},
        {'from':    929, 'from_side':    'r', 'length':      3, 'scaf1':  14096, 'strand1':    '+', 'dist1':    -44, 'scaf2':   2725, 'strand2':    '+', 'dist2':    642},
        {'from':   1306, 'from_side':    'l', 'length':      2, 'scaf1':  71091, 'strand1':    '+', 'dist1':    -45},
        {'from':   1306, 'from_side':    'r', 'length':      2, 'scaf1':     44, 'strand1':    '-', 'dist1':    -43},
        {'from':   2722, 'from_side':    'l', 'length':      3, 'scaf1':  53480, 'strand1':    '+', 'dist1':      0, 'scaf2':  40554, 'strand2':    '-', 'dist2':      0},
        {'from':   2725, 'from_side':    'l', 'length':      2, 'scaf1':    929, 'strand1':    '-', 'dist1':   1205},
        {'from':   2725, 'from_side':    'l', 'length':      3, 'scaf1':  14096, 'strand1':    '-', 'dist1':    642, 'scaf2':    929, 'strand2':    '-', 'dist2':    -44},
        {'from':   2725, 'from_side':    'r', 'length':      3, 'scaf1':  11659, 'strand1':    '-', 'dist1':    136, 'scaf2':  13591, 'strand2':    '-', 'dist2':    -43},
        {'from':   2725, 'from_side':    'r', 'length':      3, 'scaf1':  13029, 'strand1':    '+', 'dist1':    143, 'scaf2':  13591, 'strand2':    '-', 'dist2':    -43},
        {'from':   2725, 'from_side':    'r', 'length':      2, 'scaf1':  13591, 'strand1':    '-', 'dist1':    734},
        {'from':   2799, 'from_side':    'l', 'length':      2, 'scaf1':    674, 'strand1':    '+', 'dist1':     60},
        {'from':   2885, 'from_side':    'l', 'length':      2, 'scaf1':  20727, 'strand1':    '+', 'dist1':    -44},
        {'from':   2885, 'from_side':    'l', 'length':      2, 'scaf1':  20727, 'strand1':    '+', 'dist1':    245},
        {'from':   2885, 'from_side':    'r', 'length':      3, 'scaf1':  13452, 'strand1':    '-', 'dist1':    -45, 'scaf2':  10723, 'strand2':    '-', 'dist2':    -41},
        {'from':   2885, 'from_side':    'r', 'length':      3, 'scaf1':  15812, 'strand1':    '+', 'dist1':    -44, 'scaf2':  10723, 'strand2':    '-', 'dist2':    -43},
        {'from':   9344, 'from_side':    'l', 'length':      4, 'scaf1':  99341, 'strand1':    '+', 'dist1':      0, 'scaf2':  99342, 'strand2':    '+', 'dist2':      0, 'scaf3':     44, 'strand3':    '+', 'dist3':    -44},
        {'from':   9344, 'from_side':    'l', 'length':      7, 'scaf1':  99342, 'strand1':    '+', 'dist1':     73, 'scaf2': 107483, 'strand2':    '+', 'dist2':    -44, 'scaf3': 107484, 'strand3':    '+', 'dist3':      0, 'scaf4': 107485, 'strand4':    '+', 'dist4':      0, 'scaf5': 107486, 'strand5':    '+', 'dist5':      0, 'scaf6':  99004, 'strand6':    '-', 'dist6':    838},
        {'from':   9344, 'from_side':    'r', 'length':      2, 'scaf1': 118890, 'strand1':    '-', 'dist1':    -13},
        {'from':   9344, 'from_side':    'r', 'length':      2, 'scaf1': 118892, 'strand1':    '-', 'dist1':      8},
        {'from':  10723, 'from_side':    'l', 'length':      3, 'scaf1':  12896, 'strand1':    '-', 'dist1':    -43, 'scaf2':  76860, 'strand2':    '-', 'dist2':    -46},
        {'from':  10723, 'from_side':    'l', 'length':      3, 'scaf1':  12910, 'strand1':    '-', 'dist1':    -43, 'scaf2':  76860, 'strand2':    '-', 'dist2':    -42},
        {'from':  10723, 'from_side':    'r', 'length':      3, 'scaf1':  13452, 'strand1':    '+', 'dist1':    -41, 'scaf2':   2885, 'strand2':    '-', 'dist2':    -45},
        {'from':  10723, 'from_side':    'r', 'length':      3, 'scaf1':  15812, 'strand1':    '-', 'dist1':    -43, 'scaf2':   2885, 'strand2':    '-', 'dist2':    -44},
        {'from':  11659, 'from_side':    'l', 'length':      2, 'scaf1':  13591, 'strand1':    '-', 'dist1':    -43},
        {'from':  11659, 'from_side':    'r', 'length':      2, 'scaf1':   2725, 'strand1':    '-', 'dist1':    136},
        {'from':  12896, 'from_side':    'l', 'length':      2, 'scaf1':  76860, 'strand1':    '-', 'dist1':    -46},
        {'from':  12896, 'from_side':    'r', 'length':      2, 'scaf1':  10723, 'strand1':    '+', 'dist1':    -43},
        {'from':  12910, 'from_side':    'l', 'length':      2, 'scaf1':  76860, 'strand1':    '-', 'dist1':    -42},
        {'from':  12910, 'from_side':    'r', 'length':      2, 'scaf1':  10723, 'strand1':    '+', 'dist1':    -43},
        {'from':  13029, 'from_side':    'l', 'length':      2, 'scaf1':   2725, 'strand1':    '-', 'dist1':    143},
        {'from':  13029, 'from_side':    'r', 'length':      2, 'scaf1':  13591, 'strand1':    '-', 'dist1':    -43},
        {'from':  13434, 'from_side':    'l', 'length':      4, 'scaf1': 112333, 'strand1':    '+', 'dist1':     38, 'scaf2':    114, 'strand2':    '+', 'dist2':   -819, 'scaf3':    115, 'strand3':    '+', 'dist3':   1131},
        {'from':  13434, 'from_side':    'r', 'length':      2, 'scaf1':     69, 'strand1':    '-', 'dist1':    -41},
        {'from':  13452, 'from_side':    'l', 'length':      2, 'scaf1':  10723, 'strand1':    '-', 'dist1':    -41},
        {'from':  13452, 'from_side':    'r', 'length':      2, 'scaf1':   2885, 'strand1':    '-', 'dist1':    -45},
        {'from':  13455, 'from_side':    'l', 'length':      4, 'scaf1': 112333, 'strand1':    '+', 'dist1':    -43, 'scaf2':  41636, 'strand2':    '-', 'dist2':   -710, 'scaf3':    115, 'strand3':    '+', 'dist3':   1932},
        {'from':  13455, 'from_side':    'r', 'length':      2, 'scaf1':     69, 'strand1':    '-', 'dist1':    -43},
        {'from':  13591, 'from_side':    'r', 'length':      2, 'scaf1':   2725, 'strand1':    '-', 'dist1':    734},
        {'from':  13591, 'from_side':    'r', 'length':      3, 'scaf1':  11659, 'strand1':    '+', 'dist1':    -43, 'scaf2':   2725, 'strand2':    '-', 'dist2':    136},
        {'from':  13591, 'from_side':    'r', 'length':      3, 'scaf1':  13029, 'strand1':    '-', 'dist1':    -43, 'scaf2':   2725, 'strand2':    '-', 'dist2':    143},
        {'from':  14096, 'from_side':    'l', 'length':      2, 'scaf1':    929, 'strand1':    '-', 'dist1':    -44},
        {'from':  14096, 'from_side':    'r', 'length':      2, 'scaf1':   2725, 'strand1':    '+', 'dist1':    642},
        {'from':  15177, 'from_side':    'l', 'length':      2, 'scaf1':    115, 'strand1':    '-', 'dist1':    -42},
        {'from':  15177, 'from_side':    'r', 'length':      2, 'scaf1': 115803, 'strand1':    '+', 'dist1':      5},
        {'from':  15812, 'from_side':    'l', 'length':      2, 'scaf1':   2885, 'strand1':    '-', 'dist1':    -44},
        {'from':  15812, 'from_side':    'r', 'length':      2, 'scaf1':  10723, 'strand1':    '-', 'dist1':    -43},
        {'from':  20727, 'from_side':    'l', 'length':      2, 'scaf1':   2885, 'strand1':    '+', 'dist1':    -44},
        {'from':  20727, 'from_side':    'l', 'length':      2, 'scaf1':   2885, 'strand1':    '+', 'dist1':    245},
        {'from':  20727, 'from_side':    'r', 'length':      2, 'scaf1':    674, 'strand1':    '-', 'dist1':    -68},
        {'from':  20727, 'from_side':    'r', 'length':      2, 'scaf1':    674, 'strand1':    '-', 'dist1':    -18},
        {'from':  26855, 'from_side':    'l', 'length':      3, 'scaf1':  33144, 'strand1':    '-', 'dist1':    -40, 'scaf2':  47404, 'strand2':    '+', 'dist2':  -4062},
        {'from':  26855, 'from_side':    'r', 'length':      2, 'scaf1': 109207, 'strand1':    '-', 'dist1':    -43},
        {'from':  30179, 'from_side':    'l', 'length':      2, 'scaf1': 109207, 'strand1':    '-', 'dist1':    -40},
        {'from':  30179, 'from_side':    'r', 'length':      3, 'scaf1':  47404, 'strand1':    '+', 'dist1':    -32, 'scaf2':     69, 'strand2':    '+', 'dist2':    -43},
        {'from':  30214, 'from_side':    'l', 'length':      2, 'scaf1': 109207, 'strand1':    '+', 'dist1':     -6},
        {'from':  30214, 'from_side':    'r', 'length':      4, 'scaf1':  49093, 'strand1':    '-', 'dist1':     17, 'scaf2':  51660, 'strand2':    '+', 'dist2':    -14, 'scaf3': 101215, 'strand3':    '+', 'dist3':      5},
        {'from':  31749, 'from_side':    'l', 'length':      2, 'scaf1':  31756, 'strand1':    '-', 'dist1':     -1},
        {'from':  31749, 'from_side':    'l', 'length':      3, 'scaf1':  53480, 'strand1':    '-', 'dist1':   -646, 'scaf2': 101373, 'strand2':    '-', 'dist2':  -1383},
        {'from':  31749, 'from_side':    'r', 'length':      4, 'scaf1': 101373, 'strand1':    '+', 'dist1':   -350, 'scaf2': 107486, 'strand2':    '-', 'dist2':  -1922, 'scaf3': 107485, 'strand3':    '-', 'dist3':      0},
        {'from':  31756, 'from_side':    'r', 'length':      3, 'scaf1':  31749, 'strand1':    '+', 'dist1':     -1, 'scaf2': 101373, 'strand2':    '+', 'dist2':   -350},
        {'from':  32229, 'from_side':    'l', 'length':      2, 'scaf1':  76860, 'strand1':    '+', 'dist1':   -456},
        {'from':  32229, 'from_side':    'r', 'length':      2, 'scaf1':  71091, 'strand1':    '-', 'dist1':      3},
        {'from':  33144, 'from_side':    'l', 'length':      2, 'scaf1':  47404, 'strand1':    '+', 'dist1':  -4062},
        {'from':  33144, 'from_side':    'r', 'length':      3, 'scaf1':  26855, 'strand1':    '+', 'dist1':    -40, 'scaf2': 109207, 'strand2':    '-', 'dist2':    -43},
        {'from':  33994, 'from_side':    'l', 'length':      6, 'scaf1': 107486, 'strand1':    '-', 'dist1':    -44, 'scaf2': 107485, 'strand2':    '-', 'dist2':      0, 'scaf3': 107484, 'strand3':    '-', 'dist3':      0, 'scaf4': 117303, 'strand4':    '+', 'dist4':   -888, 'scaf5':    929, 'strand5':    '+', 'dist5':   2544},
        {'from':  40554, 'from_side':    'r', 'length':      3, 'scaf1':  53480, 'strand1':    '-', 'dist1':      0, 'scaf2':   2722, 'strand2':    '+', 'dist2':      0},
        {'from':  41636, 'from_side':    'l', 'length':      2, 'scaf1':    115, 'strand1':    '+', 'dist1':   1932},
        {'from':  41636, 'from_side':    'r', 'length':      4, 'scaf1': 112333, 'strand1':    '-', 'dist1':   -710, 'scaf2':  13455, 'strand2':    '+', 'dist2':    -43, 'scaf3':     69, 'strand3':    '-', 'dist3':    -43},        {'from':  47404, 'from_side':    'l', 'length':      3, 'scaf1':  30179, 'strand1':    '-', 'dist1':    -32, 'scaf2': 109207, 'strand2':    '-', 'dist2':    -40},
        {'from':  47404, 'from_side':    'l', 'length':      3, 'scaf1':  33144, 'strand1':    '+', 'dist1':  -4062, 'scaf2':  26855, 'strand2':    '+', 'dist2':    -40},
        {'from':  47404, 'from_side':    'r', 'length':      2, 'scaf1':     69, 'strand1':    '+', 'dist1':    -43},
        {'from':  49093, 'from_side':    'l', 'length':      3, 'scaf1':  51660, 'strand1':    '+', 'dist1':    -14, 'scaf2': 101215, 'strand2':    '+', 'dist2':      5},
        {'from':  49093, 'from_side':    'l', 'length':      5, 'scaf1':  56987, 'strand1':    '+', 'dist1':      4, 'scaf2': 101215, 'strand2':    '+', 'dist2':    -10, 'scaf3':  56740, 'strand3':    '+', 'dist3':    469, 'scaf4':  96716, 'strand4':    '-', 'dist4':    -45},
        {'from':  49093, 'from_side':    'r', 'length':      3, 'scaf1':  30214, 'strand1':    '-', 'dist1':     17, 'scaf2': 109207, 'strand2':    '+', 'dist2':     -6},
        {'from':  49093, 'from_side':    'r', 'length':      2, 'scaf1': 109207, 'strand1':    '+', 'dist1':    780},
        {'from':  47516, 'from_side':    'l', 'length':      3, 'scaf1': 101215, 'strand1':    '-', 'dist1':    453, 'scaf2':  51660, 'strand2':    '-', 'dist2':      5},
        {'from':  47516, 'from_side':    'r', 'length':      3, 'scaf1':  96716, 'strand1':    '-', 'dist1':    -45, 'scaf2': 118892, 'strand2':    '+', 'dist2':    -38},
        {'from':  51660, 'from_side':    'l', 'length':      4, 'scaf1':  49093, 'strand1':    '+', 'dist1':    -14, 'scaf2':  30214, 'strand2':    '-', 'dist2':     17, 'scaf3': 109207, 'strand3':    '+', 'dist3':     -6},
        {'from':  51660, 'from_side':    'r', 'length':      5, 'scaf1': 101215, 'strand1':    '+', 'dist1':      5, 'scaf2':  47516, 'strand2':    '+', 'dist2':    453, 'scaf3':  96716, 'strand3':    '-', 'dist3':    -45, 'scaf4': 118892, 'strand4':    '+', 'dist4':    -38},
        {'from':  53480, 'from_side':    'l', 'length':      2, 'scaf1':   2722, 'strand1':    '+', 'dist1':      0},
        {'from':  53480, 'from_side':    'l', 'length':      2, 'scaf1': 101373, 'strand1':    '-', 'dist1':  -1383},
        {'from':  53480, 'from_side':    'r', 'length':      3, 'scaf1':  31749, 'strand1':    '+', 'dist1':   -646, 'scaf2': 101373, 'strand2':    '+', 'dist2':   -350},
        {'from':  53480, 'from_side':    'r', 'length':      2, 'scaf1':  40554, 'strand1':    '-', 'dist1':      0},
        {'from':  56740, 'from_side':    'l', 'length':      5, 'scaf1': 101215, 'strand1':    '-', 'dist1':    469, 'scaf2':  56987, 'strand2':    '-', 'dist2':    -10, 'scaf3':  49093, 'strand3':    '+', 'dist3':      4, 'scaf4': 109207, 'strand4':    '+', 'dist4':    780},
        {'from':  56740, 'from_side':    'r', 'length':      3, 'scaf1':  96716, 'strand1':    '-', 'dist1':    -45, 'scaf2': 118890, 'strand2':    '+', 'dist2':     -3},
        {'from':  56987, 'from_side':    'l', 'length':      3, 'scaf1':  49093, 'strand1':    '+', 'dist1':      4, 'scaf2': 109207, 'strand2':    '+', 'dist2':    780},
        {'from':  56987, 'from_side':    'r', 'length':      4, 'scaf1': 101215, 'strand1':    '+', 'dist1':    -10, 'scaf2':  56740, 'strand2':    '+', 'dist2':    469, 'scaf3':  96716, 'strand3':    '-', 'dist3':    -45},
        {'from':  58443, 'from_side':    'r', 'length':      2, 'scaf1': 107486, 'strand1':    '+', 'dist1':    -83},
        {'from':  70951, 'from_side':    'l', 'length':      5, 'scaf1': 101373, 'strand1':    '+', 'dist1':      0, 'scaf2': 107486, 'strand2':    '-', 'dist2':  -1922, 'scaf3': 107485, 'strand3':    '-', 'dist3':      0, 'scaf4':    372, 'strand4':    '-', 'dist4':  -1436},
        {'from':  71091, 'from_side':    'l', 'length':      2, 'scaf1':     44, 'strand1':    '-', 'dist1':      1},
        {'from':  71091, 'from_side':    'l', 'length':      3, 'scaf1':   1306, 'strand1':    '+', 'dist1':    -45, 'scaf2':     44, 'strand2':    '-', 'dist2':    -43},
        {'from':  71091, 'from_side':    'r', 'length':      3, 'scaf1':  32229, 'strand1':    '-', 'dist1':      3, 'scaf2':  76860, 'strand2':    '+', 'dist2':   -456},
        {'from':  71091, 'from_side':    'r', 'length':      2, 'scaf1':  76860, 'strand1':    '+', 'dist1':   2134},
        {'from':  76860, 'from_side':    'l', 'length':      3, 'scaf1':  32229, 'strand1':    '+', 'dist1':   -456, 'scaf2':  71091, 'strand2':    '-', 'dist2':      3},
        {'from':  76860, 'from_side':    'l', 'length':      2, 'scaf1':  71091, 'strand1':    '-', 'dist1':   2134},
        {'from':  76860, 'from_side':    'r', 'length':      3, 'scaf1':  12896, 'strand1':    '+', 'dist1':    -46, 'scaf2':  10723, 'strand2':    '+', 'dist2':    -43},
        {'from':  76860, 'from_side':    'r', 'length':      3, 'scaf1':  12910, 'strand1':    '+', 'dist1':    -42, 'scaf2':  10723, 'strand2':    '+', 'dist2':    -43},
        {'from':  96716, 'from_side':    'l', 'length':      2, 'scaf1': 118890, 'strand1':    '+', 'dist1':     -3},
        {'from':  96716, 'from_side':    'l', 'length':      2, 'scaf1': 118892, 'strand1':    '+', 'dist1':    -38},
        {'from':  96716, 'from_side':    'r', 'length':      4, 'scaf1':  47516, 'strand1':    '-', 'dist1':    -45, 'scaf2': 101215, 'strand2':    '-', 'dist2':    453, 'scaf3':  51660, 'strand3':    '-', 'dist3':      5},
        {'from':  96716, 'from_side':    'r', 'length':      5, 'scaf1':  56740, 'strand1':    '-', 'dist1':    -45, 'scaf2': 101215, 'strand2':    '-', 'dist2':    469, 'scaf3':  56987, 'strand3':    '-', 'dist3':    -10, 'scaf4':  49093, 'strand4':    '+', 'dist4':      4},
        {'from':  99004, 'from_side':    'r', 'length':      7, 'scaf1': 107486, 'strand1':    '-', 'dist1':    838, 'scaf2': 107485, 'strand2':    '-', 'dist2':      0, 'scaf3': 107484, 'strand3':    '-', 'dist3':      0, 'scaf4': 107483, 'strand4':    '-', 'dist4':      0, 'scaf5':  99342, 'strand5':    '-', 'dist5':    -44, 'scaf6':   9344, 'strand6':    '+', 'dist6':     73},
        {'from':  99341, 'from_side':    'l', 'length':      3, 'scaf1':   9344, 'strand1':    '+', 'dist1':      0, 'scaf2': 118890, 'strand2':    '-', 'dist2':    -13},
        {'from':  99341, 'from_side':    'r', 'length':      3, 'scaf1':  99342, 'strand1':    '+', 'dist1':      0, 'scaf2':     44, 'strand2':    '+', 'dist2':    -44},
        {'from':  99342, 'from_side':    'l', 'length':      2, 'scaf1':   9344, 'strand1':    '+', 'dist1':     73},
        {'from':  99342, 'from_side':    'l', 'length':      4, 'scaf1':  99341, 'strand1':    '-', 'dist1':      0, 'scaf2':   9344, 'strand2':    '+', 'dist2':      0, 'scaf3': 118890, 'strand3':    '-', 'dist3':    -13},
        {'from':  99342, 'from_side':    'r', 'length':      2, 'scaf1':     44, 'strand1':    '+', 'dist1':    -44},
        {'from':  99342, 'from_side':    'r', 'length':      6, 'scaf1': 107483, 'strand1':    '+', 'dist1':    -44, 'scaf2': 107484, 'strand2':    '+', 'dist2':      0, 'scaf3': 107485, 'strand3':    '+', 'dist3':      0, 'scaf4': 107486, 'strand4':    '+', 'dist4':      0, 'scaf5':  99004, 'strand5':    '-', 'dist5':    838},
        {'from': 101215, 'from_side':    'l', 'length':      5, 'scaf1':  51660, 'strand1':    '-', 'dist1':      5, 'scaf2':  49093, 'strand2':    '+', 'dist2':    -14, 'scaf3':  30214, 'strand3':    '-', 'dist3':     17, 'scaf4': 109207, 'strand4':    '+', 'dist4':     -6},
        {'from': 101215, 'from_side':    'l', 'length':      4, 'scaf1':  56987, 'strand1':    '-', 'dist1':    -10, 'scaf2':  49093, 'strand2':    '+', 'dist2':      4, 'scaf3': 109207, 'strand3':    '+', 'dist3':    780},
        {'from': 101215, 'from_side':    'r', 'length':      4, 'scaf1':  47516, 'strand1':    '+', 'dist1':    453, 'scaf2':  96716, 'strand2':    '-', 'dist2':    -45, 'scaf3': 118892, 'strand3':    '+', 'dist3':    -38},
        {'from': 101215, 'from_side':    'r', 'length':      4, 'scaf1':  56740, 'strand1':    '+', 'dist1':    469, 'scaf2':  96716, 'strand2':    '-', 'dist2':    -45, 'scaf3': 118890, 'strand3':    '+', 'dist3':     -3},
        {'from': 101373, 'from_side':    'l', 'length':      3, 'scaf1':  31749, 'strand1':    '-', 'dist1':   -350, 'scaf2':  31756, 'strand2':    '-', 'dist2':     -1},
        {'from': 101373, 'from_side':    'l', 'length':      4, 'scaf1':  31749, 'strand1':    '-', 'dist1':   -350, 'scaf2':  53480, 'strand2':    '-', 'dist2':   -646, 'scaf3': 101373, 'strand3':    '-', 'dist3':  -1383},
        {'from': 101373, 'from_side':    'l', 'length':      2, 'scaf1':  70951, 'strand1':    '+', 'dist1':      0},
        {'from': 101373, 'from_side':    'r', 'length':      4, 'scaf1':  53480, 'strand1':    '+', 'dist1':  -1383, 'scaf2':  31749, 'strand2':    '+', 'dist2':   -646, 'scaf3': 101373, 'strand3':    '+', 'dist3':   -350},
        {'from': 101373, 'from_side':    'r', 'length':      4, 'scaf1': 107486, 'strand1':    '-', 'dist1':  -1922, 'scaf2': 107485, 'strand2':    '-', 'dist2':      0, 'scaf3':    372, 'strand3':    '-', 'dist3':  -1436},
        {'from': 101373, 'from_side':    'r', 'length':      5, 'scaf1': 107486, 'strand1':    '-', 'dist1':  -1922, 'scaf2': 107485, 'strand2':    '-', 'dist2':      0, 'scaf3': 107484, 'strand3':    '-', 'dist3':      0, 'scaf4': 107483, 'strand4':    '-', 'dist4':      0},
        {'from': 107483, 'from_side':    'l', 'length':      3, 'scaf1':  99342, 'strand1':    '-', 'dist1':    -44, 'scaf2':   9344, 'strand2':    '+', 'dist2':     73},
        {'from': 107483, 'from_side':    'r', 'length':      5, 'scaf1': 107484, 'strand1':    '+', 'dist1':      0, 'scaf2': 107485, 'strand2':    '+', 'dist2':      0, 'scaf3': 107486, 'strand3':    '+', 'dist3':      0, 'scaf4':  99004, 'strand4':    '-', 'dist4':    838},
        {'from': 107483, 'from_side':    'r', 'length':      5, 'scaf1': 107484, 'strand1':    '+', 'dist1':      0, 'scaf2': 107485, 'strand2':    '+', 'dist2':      0, 'scaf3': 107486, 'strand3':    '+', 'dist3':      0, 'scaf4': 101373, 'strand4':    '-', 'dist4':  -1922},
        {'from': 107484, 'from_side':    'l', 'length':      4, 'scaf1': 107483, 'strand1':    '-', 'dist1':      0, 'scaf2':  99342, 'strand2':    '-', 'dist2':    -44, 'scaf3':   9344, 'strand3':    '+', 'dist3':     73},
        {'from': 107484, 'from_side':    'l', 'length':      3, 'scaf1': 117303, 'strand1':    '+', 'dist1':   -888, 'scaf2':    929, 'strand2':    '+', 'dist2':   2544},
        {'from': 107484, 'from_side':    'r', 'length':      4, 'scaf1': 107485, 'strand1':    '+', 'dist1':      0, 'scaf2': 107486, 'strand2':    '+', 'dist2':      0, 'scaf3':  33994, 'strand3':    '+', 'dist3':    -44},
        {'from': 107484, 'from_side':    'r', 'length':      4, 'scaf1': 107485, 'strand1':    '+', 'dist1':      0, 'scaf2': 107486, 'strand2':    '+', 'dist2':      0, 'scaf3':  99004, 'strand3':    '-', 'dist3':    838},
        {'from': 107484, 'from_side':    'r', 'length':      4, 'scaf1': 107485, 'strand1':    '+', 'dist1':      0, 'scaf2': 107486, 'strand2':    '+', 'dist2':      0, 'scaf3': 101373, 'strand3':    '-', 'dist3':  -1922},
        {'from': 107485, 'from_side':    'l', 'length':      2, 'scaf1':    372, 'strand1':    '-', 'dist1':  -1436},
        {'from': 107485, 'from_side':    'l', 'length':      5, 'scaf1': 107484, 'strand1':    '-', 'dist1':      0, 'scaf2': 107483, 'strand2':    '-', 'dist2':      0, 'scaf3':  99342, 'strand3':    '-', 'dist3':    -44, 'scaf4':   9344, 'strand4':    '+', 'dist4':     73},
        {'from': 107485, 'from_side':    'l', 'length':      4, 'scaf1': 107484, 'strand1':    '-', 'dist1':      0, 'scaf2': 117303, 'strand2':    '+', 'dist2':   -888, 'scaf3':    929, 'strand3':    '+', 'dist3':   2544},
        {'from': 107485, 'from_side':    'r', 'length':      3, 'scaf1': 107486, 'strand1':    '+', 'dist1':      0, 'scaf2':  33994, 'strand2':    '+', 'dist2':    -44},
        {'from': 107485, 'from_side':    'r', 'length':      3, 'scaf1': 107486, 'strand1':    '+', 'dist1':      0, 'scaf2':  99004, 'strand2':    '-', 'dist2':    838},
        {'from': 107485, 'from_side':    'r', 'length':      4, 'scaf1': 107486, 'strand1':    '+', 'dist1':      0, 'scaf2': 101373, 'strand2':    '-', 'dist2':  -1922, 'scaf3':  31749, 'strand3':    '-', 'dist3':   -350},
        {'from': 107485, 'from_side':    'r', 'length':      4, 'scaf1': 107486, 'strand1':    '+', 'dist1':      0, 'scaf2': 101373, 'strand2':    '-', 'dist2':  -1922, 'scaf3':  70951, 'strand3':    '+', 'dist3':      0},
        {'from': 107486, 'from_side':    'l', 'length':      2, 'scaf1':  58443, 'strand1':    '-', 'dist1':    -83},
        {'from': 107486, 'from_side':    'l', 'length':      3, 'scaf1': 107485, 'strand1':    '-', 'dist1':      0, 'scaf2':    372, 'strand2':    '-', 'dist2':  -1436},
        {'from': 107486, 'from_side':    'l', 'length':      6, 'scaf1': 107485, 'strand1':    '-', 'dist1':      0, 'scaf2': 107484, 'strand2':    '-', 'dist2':      0, 'scaf3': 107483, 'strand3':    '-', 'dist3':      0, 'scaf4':  99342, 'strand4':    '-', 'dist4':    -44, 'scaf5':   9344, 'strand5':    '+', 'dist5':     73},
        {'from': 107486, 'from_side':    'l', 'length':      5, 'scaf1': 107485, 'strand1':    '-', 'dist1':      0, 'scaf2': 107484, 'strand2':    '-', 'dist2':      0, 'scaf3': 117303, 'strand3':    '+', 'dist3':   -888, 'scaf4':    929, 'strand4':    '+', 'dist4':   2544},
        {'from': 107486, 'from_side':    'r', 'length':      2, 'scaf1':  33994, 'strand1':    '+', 'dist1':    -44},
        {'from': 107486, 'from_side':    'r', 'length':      2, 'scaf1':  99004, 'strand1':    '-', 'dist1':    838},
        {'from': 107486, 'from_side':    'r', 'length':      3, 'scaf1': 101373, 'strand1':    '-', 'dist1':  -1922, 'scaf2':  31749, 'strand2':    '-', 'dist2':   -350},
        {'from': 107486, 'from_side':    'r', 'length':      3, 'scaf1': 101373, 'strand1':    '-', 'dist1':  -1922, 'scaf2':  70951, 'strand2':    '+', 'dist2':      0},
        {'from': 109207, 'from_side':    'l', 'length':      5, 'scaf1':  30214, 'strand1':    '+', 'dist1':     -6, 'scaf2':  49093, 'strand2':    '-', 'dist2':     17, 'scaf3':  51660, 'strand3':    '+', 'dist3':    -14, 'scaf4': 101215, 'strand4':    '+', 'dist4':      5},
        {'from': 109207, 'from_side':    'l', 'length':      5, 'scaf1':  49093, 'strand1':    '-', 'dist1':    780, 'scaf2':  56987, 'strand2':    '+', 'dist2':      4, 'scaf3': 101215, 'strand3':    '+', 'dist3':    -10, 'scaf4':  56740, 'strand4':    '+', 'dist4':    469},
        {'from': 109207, 'from_side':    'r', 'length':      3, 'scaf1':  26855, 'strand1':    '-', 'dist1':    -43, 'scaf2':  33144, 'strand2':    '-', 'dist2':    -40},
        {'from': 109207, 'from_side':    'r', 'length':      3, 'scaf1':  30179, 'strand1':    '+', 'dist1':    -40, 'scaf2':  47404, 'strand2':    '+', 'dist2':    -32},
        {'from': 110827, 'from_side':    'r', 'length':      2, 'scaf1':     69, 'strand1':    '+', 'dist1':    -43},
        {'from': 112333, 'from_side':    'l', 'length':      3, 'scaf1':  13434, 'strand1':    '+', 'dist1':     38, 'scaf2':     69, 'strand2':    '-', 'dist2':    -41},
        {'from': 112333, 'from_side':    'l', 'length':      3, 'scaf1':  13455, 'strand1':    '+', 'dist1':    -43, 'scaf2':     69, 'strand2':    '-', 'dist2':    -43},
        {'from': 112333, 'from_side':    'r', 'length':      4, 'scaf1':    114, 'strand1':    '+', 'dist1':   -819, 'scaf2':    115, 'strand2':    '+', 'dist2':   1131, 'scaf3': 115803, 'strand3':    '+', 'dist3':    885},
        {'from': 112333, 'from_side':    'r', 'length':      3, 'scaf1':  41636, 'strand1':    '-', 'dist1':   -710, 'scaf2':    115, 'strand2':    '+', 'dist2':   1932},
        {'from': 115803, 'from_side':    'l', 'length':      4, 'scaf1':    115, 'strand1':    '-', 'dist1':    885, 'scaf2':    114, 'strand2':    '-', 'dist2':   1131, 'scaf3': 112333, 'strand3':    '-', 'dist3':   -819},
        {'from': 115803, 'from_side':    'l', 'length':      3, 'scaf1':  15177, 'strand1':    '-', 'dist1':      5, 'scaf2':    115, 'strand2':    '-', 'dist2':    -42},
        {'from': 117303, 'from_side':    'l', 'length':      5, 'scaf1': 107484, 'strand1':    '+', 'dist1':   -888, 'scaf2': 107485, 'strand2':    '+', 'dist2':      0, 'scaf3': 107486, 'strand3':    '+', 'dist3':      0, 'scaf4':  33994, 'strand4':    '+', 'dist4':    -44},
        {'from': 117303, 'from_side':    'r', 'length':      2, 'scaf1':    929, 'strand1':    '+', 'dist1':   2544},
        {'from': 117304, 'from_side':    'r', 'length':      2, 'scaf1':    929, 'strand1':    '+', 'dist1':   2489},
        {'from': 118890, 'from_side':    'l', 'length':      4, 'scaf1':  96716, 'strand1':    '+', 'dist1':     -3, 'scaf2':  56740, 'strand2':    '-', 'dist2':    -45, 'scaf3': 101215, 'strand3':    '-', 'dist3':    469},
        {'from': 118890, 'from_side':    'r', 'length':      5, 'scaf1':   9344, 'strand1':    '-', 'dist1':    -13, 'scaf2':  99341, 'strand2':    '+', 'dist2':      0, 'scaf3':  99342, 'strand3':    '+', 'dist3':      0, 'scaf4':     44, 'strand4':    '+', 'dist4':    -44},
        {'from': 118892, 'from_side':    'l', 'length':      5, 'scaf1':  96716, 'strand1':    '+', 'dist1':    -38, 'scaf2':  47516, 'strand2':    '-', 'dist2':    -45, 'scaf3': 101215, 'strand3':    '-', 'dist3':    453, 'scaf4':  51660, 'strand4':    '-', 'dist4':      5},
        {'from': 118892, 'from_side':    'r', 'length':      2, 'scaf1':   9344, 'strand1':    '-', 'dist1':      8}
        ]) )
    scaf_bridges.append( pd.DataFrame([
        {'from':     44, 'from_side':    'l', 'to':  99342, 'to_side':    'r', 'mean_dist':    -44, 'mapq':  60060, 'bcount':     19, 'min_dist':    -49, 'max_dist':    -33, 'probability': 0.273725, 'to_alt':      2, 'from_alt':      1},
        {'from':     44, 'from_side':    'r', 'to':   1306, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     12, 'min_dist':    -46, 'max_dist':    -39, 'probability': 0.064232, 'to_alt':      1, 'from_alt':      2},
        {'from':     44, 'from_side':    'r', 'to':  71091, 'to_side':    'l', 'mean_dist':      1, 'mapq':  60060, 'bcount':      5, 'min_dist':     -1, 'max_dist':      6, 'probability': 0.007368, 'to_alt':      2, 'from_alt':      2},
        {'from':     69, 'from_side':    'l', 'to':  47404, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     15, 'min_dist':    -49, 'max_dist':    -38, 'probability': 0.129976, 'to_alt':      1, 'from_alt':      2},
        {'from':     69, 'from_side':    'l', 'to': 110827, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     24, 'min_dist':    -50, 'max_dist':    -16, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from':     69, 'from_side':    'r', 'to':  13434, 'to_side':    'r', 'mean_dist':    -41, 'mapq':  60060, 'bcount':     18, 'min_dist':    -51, 'max_dist':    -31, 'probability': 0.231835, 'to_alt':      1, 'from_alt':      2},
        {'from':     69, 'from_side':    'r', 'to':  13455, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     27, 'min_dist':    -60, 'max_dist':    -17, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from':    114, 'from_side':    'l', 'to': 112333, 'to_side':    'r', 'mean_dist':   -819, 'mapq':  60060, 'bcount':     22, 'min_dist':   -931, 'max_dist':   -785, 'probability': 0.417654, 'to_alt':      2, 'from_alt':      1},
        {'from':    114, 'from_side':    'r', 'to':    115, 'to_side':    'l', 'mean_dist':   1131, 'mapq':  60060, 'bcount':     21, 'min_dist':   1080, 'max_dist':   1263, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':    115, 'from_side':    'l', 'to':    114, 'to_side':    'r', 'mean_dist':   1131, 'mapq':  60060, 'bcount':     21, 'min_dist':   1080, 'max_dist':   1263, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from':    115, 'from_side':    'l', 'to':  41636, 'to_side':    'l', 'mean_dist':   1932, 'mapq':  60060, 'bcount':     18, 'min_dist':   1811, 'max_dist':   2087, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from':    115, 'from_side':    'r', 'to':  15177, 'to_side':    'l', 'mean_dist':    -42, 'mapq':  60060, 'bcount':     14, 'min_dist':    -52, 'max_dist':    -20, 'probability': 0.104244, 'to_alt':      1, 'from_alt':      2},
        {'from':    115, 'from_side':    'r', 'to': 115803, 'to_side':    'l', 'mean_dist':    885, 'mapq':  60060, 'bcount':     15, 'min_dist':    818, 'max_dist':   1120, 'probability': 0.209365, 'to_alt':      2, 'from_alt':      2},
        {'from':    372, 'from_side':    'r', 'to': 107485, 'to_side':    'l', 'mean_dist':  -1436, 'mapq':  60060, 'bcount':     14, 'min_dist':  -1605, 'max_dist':  -1350, 'probability': 0.169655, 'to_alt':      2, 'from_alt':      1},
        {'from':    674, 'from_side':    'l', 'to':   2799, 'to_side':    'l', 'mean_dist':     60, 'mapq':  60060, 'bcount':     19, 'min_dist':     43, 'max_dist':    107, 'probability': 0.273725, 'to_alt':      2, 'from_alt':      1},
        {'from':    674, 'from_side':    'r', 'to':  20727, 'to_side':    'r', 'mean_dist':    -68, 'mapq':  60060, 'bcount':     10, 'min_dist':    -79, 'max_dist':    -63, 'probability': 0.037322, 'to_alt':      2, 'from_alt':      2},
        {'from':    674, 'from_side':    'r', 'to':  20727, 'to_side':    'r', 'mean_dist':    -18, 'mapq':  60060, 'bcount':      8, 'min_dist':    -22, 'max_dist':    -11, 'probability': 0.020422, 'to_alt':      2, 'from_alt':      2},
        {'from':    929, 'from_side':    'l', 'to': 117303, 'to_side':    'r', 'mean_dist':   2544, 'mapq':  60060, 'bcount':      7, 'min_dist':   2451, 'max_dist':   2631, 'probability': 0.076700, 'to_alt':      1, 'from_alt':      2},
        {'from':    929, 'from_side':    'l', 'to': 117304, 'to_side':    'r', 'mean_dist':   2489, 'mapq':  60060, 'bcount':      2, 'min_dist':   2450, 'max_dist':   2534, 'probability': 0.008611, 'to_alt':      1, 'from_alt':      2},
        {'from':    929, 'from_side':    'r', 'to':   2725, 'to_side':    'l', 'mean_dist':   1205, 'mapq':  60060, 'bcount':      6, 'min_dist':   1175, 'max_dist':   1229, 'probability': 0.016554, 'to_alt':      2, 'from_alt':      2},
        {'from':    929, 'from_side':    'r', 'to':  14096, 'to_side':    'l', 'mean_dist':    -44, 'mapq':  60060, 'bcount':      9, 'min_dist':    -49, 'max_dist':    -34, 'probability': 0.027818, 'to_alt':      1, 'from_alt':      2},
        {'from':   1306, 'from_side':    'l', 'to':  71091, 'to_side':    'l', 'mean_dist':    -45, 'mapq':  60060, 'bcount':      8, 'min_dist':    -49, 'max_dist':    -40, 'probability': 0.020422, 'to_alt':      2, 'from_alt':      1},
        {'from':   1306, 'from_side':    'r', 'to':     44, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     12, 'min_dist':    -46, 'max_dist':    -39, 'probability': 0.064232, 'to_alt':      2, 'from_alt':      1},
        {'from':   2722, 'from_side':    'l', 'to':  53480, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     16, 'min_dist':      0, 'max_dist':      0, 'probability': 0.159802, 'to_alt':      2, 'from_alt':      1},
        {'from':   2725, 'from_side':    'l', 'to':    929, 'to_side':    'r', 'mean_dist':   1205, 'mapq':  60060, 'bcount':      6, 'min_dist':   1175, 'max_dist':   1229, 'probability': 0.016554, 'to_alt':      2, 'from_alt':      2},
        {'from':   2725, 'from_side':    'l', 'to':  14096, 'to_side':    'r', 'mean_dist':    642, 'mapq':  60060, 'bcount':      8, 'min_dist':    586, 'max_dist':    773, 'probability': 0.033108, 'to_alt':      1, 'from_alt':      2},
        {'from':   2725, 'from_side':    'r', 'to':  11659, 'to_side':    'r', 'mean_dist':    136, 'mapq':  60060, 'bcount':      7, 'min_dist':    129, 'max_dist':    146, 'probability': 0.014765, 'to_alt':      1, 'from_alt':      3},
        {'from':   2725, 'from_side':    'r', 'to':  13029, 'to_side':    'l', 'mean_dist':    143, 'mapq':  60060, 'bcount':      3, 'min_dist':    126, 'max_dist':    201, 'probability': 0.003454, 'to_alt':      1, 'from_alt':      3},
        {'from':   2725, 'from_side':    'r', 'to':  13591, 'to_side':    'r', 'mean_dist':    734, 'mapq':  60060, 'bcount':      5, 'min_dist':    684, 'max_dist':    797, 'probability': 0.011373, 'to_alt':      3, 'from_alt':      3},
        {'from':   2799, 'from_side':    'l', 'to':    674, 'to_side':    'l', 'mean_dist':     60, 'mapq':  60060, 'bcount':     19, 'min_dist':     43, 'max_dist':    107, 'probability': 0.273725, 'to_alt':      1, 'from_alt':      2},
        {'from':   2885, 'from_side':    'l', 'to':  20727, 'to_side':    'l', 'mean_dist':    -44, 'mapq':  60060, 'bcount':     14, 'min_dist':    -55, 'max_dist':    -35, 'probability': 0.104244, 'to_alt':      2, 'from_alt':      2},
        {'from':   2885, 'from_side':    'l', 'to':  20727, 'to_side':    'l', 'mean_dist':    245, 'mapq':  60060, 'bcount':     12, 'min_dist':    220, 'max_dist':    324, 'probability': 0.064232, 'to_alt':      2, 'from_alt':      2},
        {'from':   2885, 'from_side':    'r', 'to':  13452, 'to_side':    'r', 'mean_dist':    -45, 'mapq':  60060, 'bcount':      6, 'min_dist':    -47, 'max_dist':    -43, 'probability': 0.010512, 'to_alt':      1, 'from_alt':      2},
        {'from':   2885, 'from_side':    'r', 'to':  15812, 'to_side':    'l', 'mean_dist':    -44, 'mapq':  60060, 'bcount':      7, 'min_dist':    -47, 'max_dist':    -38, 'probability': 0.014765, 'to_alt':      1, 'from_alt':      2},
        {'from':   9344, 'from_side':    'l', 'to':  99341, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     20, 'min_dist':      0, 'max_dist':      0, 'probability': 0.319050, 'to_alt':      1, 'from_alt':      2},
        {'from':   9344, 'from_side':    'l', 'to':  99342, 'to_side':    'l', 'mean_dist':     73, 'mapq':  60060, 'bcount':     12, 'min_dist':     64, 'max_dist':     80, 'probability': 0.064232, 'to_alt':      2, 'from_alt':      2},
        {'from':   9344, 'from_side':    'r', 'to': 118890, 'to_side':    'r', 'mean_dist':    -13, 'mapq':  60060, 'bcount':     25, 'min_dist':    -20, 'max_dist':     39, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from':   9344, 'from_side':    'r', 'to': 118892, 'to_side':    'r', 'mean_dist':      8, 'mapq':  60060, 'bcount':     20, 'min_dist':      0, 'max_dist':     35, 'probability': 0.319050, 'to_alt':      1, 'from_alt':      2},
        {'from':  10723, 'from_side':    'l', 'to':  12896, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':      7, 'min_dist':    -47, 'max_dist':    -24, 'probability': 0.014765, 'to_alt':      1, 'from_alt':      2},
        {'from':  10723, 'from_side':    'l', 'to':  12910, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':      7, 'min_dist':    -47, 'max_dist':    -36, 'probability': 0.014765, 'to_alt':      1, 'from_alt':      2},
        {'from':  10723, 'from_side':    'r', 'to':  13452, 'to_side':    'l', 'mean_dist':    -41, 'mapq':  60060, 'bcount':      5, 'min_dist':    -46, 'max_dist':    -30, 'probability': 0.007368, 'to_alt':      1, 'from_alt':      2},
        {'from':  10723, 'from_side':    'r', 'to':  15812, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     10, 'min_dist':    -47, 'max_dist':    -33, 'probability': 0.037322, 'to_alt':      1, 'from_alt':      2},
        {'from':  11659, 'from_side':    'l', 'to':  13591, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':      7, 'min_dist':    -47, 'max_dist':    -36, 'probability': 0.014765, 'to_alt':      3, 'from_alt':      1},
        {'from':  11659, 'from_side':    'r', 'to':   2725, 'to_side':    'r', 'mean_dist':    136, 'mapq':  60060, 'bcount':      7, 'min_dist':    129, 'max_dist':    146, 'probability': 0.014765, 'to_alt':      3, 'from_alt':      1},
        {'from':  12896, 'from_side':    'l', 'to':  76860, 'to_side':    'r', 'mean_dist':    -46, 'mapq':  60060, 'bcount':      6, 'min_dist':    -51, 'max_dist':    -43, 'probability': 0.010512, 'to_alt':      2, 'from_alt':      1},
        {'from':  12896, 'from_side':    'r', 'to':  10723, 'to_side':    'l', 'mean_dist':    -43, 'mapq':  60060, 'bcount':      7, 'min_dist':    -47, 'max_dist':    -24, 'probability': 0.014765, 'to_alt':      2, 'from_alt':      1},
        {'from':  12910, 'from_side':    'l', 'to':  76860, 'to_side':    'r', 'mean_dist':    -42, 'mapq':  60060, 'bcount':      5, 'min_dist':    -52, 'max_dist':    -30, 'probability': 0.007368, 'to_alt':      2, 'from_alt':      1},
        {'from':  12910, 'from_side':    'r', 'to':  10723, 'to_side':    'l', 'mean_dist':    -43, 'mapq':  60060, 'bcount':      7, 'min_dist':    -47, 'max_dist':    -36, 'probability': 0.014765, 'to_alt':      2, 'from_alt':      1},
        {'from':  13029, 'from_side':    'l', 'to':   2725, 'to_side':    'r', 'mean_dist':    143, 'mapq':  60060, 'bcount':      3, 'min_dist':    126, 'max_dist':    201, 'probability': 0.003454, 'to_alt':      3, 'from_alt':      1},
        {'from':  13029, 'from_side':    'r', 'to':  13591, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':      3, 'min_dist':    -47, 'max_dist':    -39, 'probability': 0.003454, 'to_alt':      3, 'from_alt':      1},
        {'from':  13434, 'from_side':    'l', 'to': 112333, 'to_side':    'l', 'mean_dist':     38, 'mapq':  60060, 'bcount':     18, 'min_dist':     32, 'max_dist':     52, 'probability': 0.231835, 'to_alt':      2, 'from_alt':      1},
        {'from':  13434, 'from_side':    'r', 'to':     69, 'to_side':    'r', 'mean_dist':    -41, 'mapq':  60060, 'bcount':     18, 'min_dist':    -51, 'max_dist':    -31, 'probability': 0.231835, 'to_alt':      2, 'from_alt':      1},
        {'from':  13452, 'from_side':    'l', 'to':  10723, 'to_side':    'r', 'mean_dist':    -41, 'mapq':  60060, 'bcount':      5, 'min_dist':    -46, 'max_dist':    -30, 'probability': 0.007368, 'to_alt':      2, 'from_alt':      1},
        {'from':  13452, 'from_side':    'r', 'to':   2885, 'to_side':    'r', 'mean_dist':    -45, 'mapq':  60060, 'bcount':      6, 'min_dist':    -47, 'max_dist':    -43, 'probability': 0.010512, 'to_alt':      2, 'from_alt':      1},
        {'from':  13455, 'from_side':    'l', 'to': 112333, 'to_side':    'l', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     29, 'min_dist':    -49, 'max_dist':    -34, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  13455, 'from_side':    'r', 'to':     69, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     27, 'min_dist':    -60, 'max_dist':    -17, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  13591, 'from_side':    'r', 'to':   2725, 'to_side':    'r', 'mean_dist':    734, 'mapq':  60060, 'bcount':      5, 'min_dist':    684, 'max_dist':    797, 'probability': 0.011373, 'to_alt':      3, 'from_alt':      3},
        {'from':  13591, 'from_side':    'r', 'to':  11659, 'to_side':    'l', 'mean_dist':    -43, 'mapq':  60060, 'bcount':      7, 'min_dist':    -47, 'max_dist':    -36, 'probability': 0.014765, 'to_alt':      1, 'from_alt':      3},
        {'from':  13591, 'from_side':    'r', 'to':  13029, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':      3, 'min_dist':    -47, 'max_dist':    -39, 'probability': 0.003454, 'to_alt':      1, 'from_alt':      3},
        {'from':  14096, 'from_side':    'l', 'to':    929, 'to_side':    'r', 'mean_dist':    -44, 'mapq':  60060, 'bcount':      9, 'min_dist':    -49, 'max_dist':    -34, 'probability': 0.027818, 'to_alt':      2, 'from_alt':      1},
        {'from':  14096, 'from_side':    'r', 'to':   2725, 'to_side':    'l', 'mean_dist':    642, 'mapq':  60060, 'bcount':      8, 'min_dist':    586, 'max_dist':    773, 'probability': 0.033108, 'to_alt':      2, 'from_alt':      1},
        {'from':  15177, 'from_side':    'l', 'to':    115, 'to_side':    'r', 'mean_dist':    -42, 'mapq':  60060, 'bcount':     14, 'min_dist':    -52, 'max_dist':    -20, 'probability': 0.104244, 'to_alt':      2, 'from_alt':      1},
        {'from':  15177, 'from_side':    'r', 'to': 115803, 'to_side':    'l', 'mean_dist':      5, 'mapq':  60060, 'bcount':     12, 'min_dist':      1, 'max_dist':     15, 'probability': 0.064232, 'to_alt':      2, 'from_alt':      1},
        {'from':  15812, 'from_side':    'l', 'to':   2885, 'to_side':    'r', 'mean_dist':    -44, 'mapq':  60060, 'bcount':      7, 'min_dist':    -47, 'max_dist':    -38, 'probability': 0.014765, 'to_alt':      2, 'from_alt':      1},
        {'from':  15812, 'from_side':    'r', 'to':  10723, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     10, 'min_dist':    -47, 'max_dist':    -33, 'probability': 0.037322, 'to_alt':      2, 'from_alt':      1},
        {'from':  20727, 'from_side':    'l', 'to':   2885, 'to_side':    'l', 'mean_dist':    -44, 'mapq':  60060, 'bcount':     14, 'min_dist':    -55, 'max_dist':    -35, 'probability': 0.104244, 'to_alt':      2, 'from_alt':      2},
        {'from':  20727, 'from_side':    'l', 'to':   2885, 'to_side':    'l', 'mean_dist':    245, 'mapq':  60060, 'bcount':     12, 'min_dist':    220, 'max_dist':    324, 'probability': 0.064232, 'to_alt':      2, 'from_alt':      2},
        {'from':  20727, 'from_side':    'r', 'to':    674, 'to_side':    'r', 'mean_dist':    -68, 'mapq':  60060, 'bcount':     10, 'min_dist':    -79, 'max_dist':    -63, 'probability': 0.037322, 'to_alt':      2, 'from_alt':      2},
        {'from':  20727, 'from_side':    'r', 'to':    674, 'to_side':    'r', 'mean_dist':    -18, 'mapq':  60060, 'bcount':      8, 'min_dist':    -22, 'max_dist':    -11, 'probability': 0.020422, 'to_alt':      2, 'from_alt':      2},
        {'from':  26855, 'from_side':    'l', 'to':  33144, 'to_side':    'r', 'mean_dist':    -40, 'mapq':  60060, 'bcount':      4, 'min_dist':    -46, 'max_dist':    -31, 'probability': 0.005085, 'to_alt':      1, 'from_alt':      2},
        {'from':  26855, 'from_side':    'r', 'to': 109207, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     13, 'min_dist':    -51, 'max_dist':    -14, 'probability': 0.082422, 'to_alt':      2, 'from_alt':      1},
        {'from':  30179, 'from_side':    'l', 'to': 109207, 'to_side':    'r', 'mean_dist':    -40, 'mapq':  60060, 'bcount':      8, 'min_dist':    -47, 'max_dist':    -23, 'probability': 0.020422, 'to_alt':      2, 'from_alt':      1},
        {'from':  30179, 'from_side':    'r', 'to':  47404, 'to_side':    'l', 'mean_dist':    -32, 'mapq':  60060, 'bcount':     13, 'min_dist':    -39, 'max_dist':    -18, 'probability': 0.082422, 'to_alt':      2, 'from_alt':      1},
        {'from':  30214, 'from_side':    'l', 'to': 109207, 'to_side':    'l', 'mean_dist':     -6, 'mapq':  60060, 'bcount':     13, 'min_dist':    -10, 'max_dist':     21, 'probability': 0.082422, 'to_alt':      2, 'from_alt':      1},
        {'from':  30214, 'from_side':    'r', 'to':  49093, 'to_side':    'r', 'mean_dist':     17, 'mapq':  60060, 'bcount':     12, 'min_dist':     13, 'max_dist':     25, 'probability': 0.064232, 'to_alt':      2, 'from_alt':      1},
        {'from':  31749, 'from_side':    'l', 'to':  31756, 'to_side':    'r', 'mean_dist':     -1, 'mapq':  60060, 'bcount':     12, 'min_dist':     -8, 'max_dist':     26, 'probability': 0.064232, 'to_alt':      1, 'from_alt':      2},
        {'from':  31749, 'from_side':    'l', 'to':  53480, 'to_side':    'r', 'mean_dist':   -646, 'mapq':  60060, 'bcount':     18, 'min_dist':   -716, 'max_dist':   -614, 'probability': 0.231835, 'to_alt':      2, 'from_alt':      2},
        {'from':  31749, 'from_side':    'r', 'to': 101373, 'to_side':    'l', 'mean_dist':   -350, 'mapq':  60060, 'bcount':    109, 'min_dist':   -433, 'max_dist':   -288, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  31756, 'from_side':    'r', 'to':  31749, 'to_side':    'l', 'mean_dist':     -1, 'mapq':  60060, 'bcount':     12, 'min_dist':     -8, 'max_dist':     26, 'probability': 0.064232, 'to_alt':      2, 'from_alt':      1},
        {'from':  32229, 'from_side':    'l', 'to':  76860, 'to_side':    'l', 'mean_dist':   -456, 'mapq':  60060, 'bcount':      7, 'min_dist':   -527, 'max_dist':   -412, 'probability': 0.014765, 'to_alt':      2, 'from_alt':      1},
        {'from':  32229, 'from_side':    'r', 'to':  71091, 'to_side':    'r', 'mean_dist':      3, 'mapq':  60060, 'bcount':     12, 'min_dist':      1, 'max_dist':      7, 'probability': 0.064232, 'to_alt':      2, 'from_alt':      1},
        {'from':  33144, 'from_side':    'l', 'to':  47404, 'to_side':    'l', 'mean_dist':  -4062, 'mapq':  60060, 'bcount':      2, 'min_dist':  -4070, 'max_dist':  -4055, 'probability': 0.008611, 'to_alt':      2, 'from_alt':      1},
        {'from':  33144, 'from_side':    'r', 'to':  26855, 'to_side':    'l', 'mean_dist':    -40, 'mapq':  60060, 'bcount':      4, 'min_dist':    -46, 'max_dist':    -31, 'probability': 0.005085, 'to_alt':      2, 'from_alt':      1},
        {'from':  33994, 'from_side':    'l', 'to': 107486, 'to_side':    'r', 'mean_dist':    -44, 'mapq':  60060, 'bcount':     25, 'min_dist':    -52, 'max_dist':    -15, 'probability': 0.500000, 'to_alt':      3, 'from_alt':      1},
        {'from':  40554, 'from_side':    'r', 'to':  53480, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':     19, 'min_dist':      0, 'max_dist':      0, 'probability': 0.273725, 'to_alt':      2, 'from_alt':      1},
        {'from':  41636, 'from_side':    'l', 'to':    115, 'to_side':    'l', 'mean_dist':   1932, 'mapq':  60060, 'bcount':     18, 'min_dist':   1811, 'max_dist':   2087, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  41636, 'from_side':    'r', 'to': 112333, 'to_side':    'r', 'mean_dist':   -710, 'mapq':  60060, 'bcount':     20, 'min_dist':   -775, 'max_dist':   -662, 'probability': 0.319050, 'to_alt':      2, 'from_alt':      1},
        {'from':  47404, 'from_side':    'l', 'to':  30179, 'to_side':    'r', 'mean_dist':    -32, 'mapq':  60060, 'bcount':     13, 'min_dist':    -39, 'max_dist':    -18, 'probability': 0.082422, 'to_alt':      1, 'from_alt':      2},
        {'from':  47404, 'from_side':    'l', 'to':  33144, 'to_side':    'l', 'mean_dist':  -4062, 'mapq':  60060, 'bcount':      2, 'min_dist':  -4070, 'max_dist':  -4055, 'probability': 0.008611, 'to_alt':      1, 'from_alt':      2},
        {'from':  47404, 'from_side':    'r', 'to':     69, 'to_side':    'l', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     15, 'min_dist':    -49, 'max_dist':    -38, 'probability': 0.129976, 'to_alt':      2, 'from_alt':      1},
        {'from':  49093, 'from_side':    'l', 'to':  51660, 'to_side':    'l', 'mean_dist':    -14, 'mapq':  60060, 'bcount':     18, 'min_dist':    -19, 'max_dist':     14, 'probability': 0.231835, 'to_alt':      1, 'from_alt':      2},
        {'from':  49093, 'from_side':    'l', 'to':  56987, 'to_side':    'l', 'mean_dist':      4, 'mapq':  60060, 'bcount':     21, 'min_dist':      0, 'max_dist':     15, 'probability': 0.367256, 'to_alt':      1, 'from_alt':      2},
        {'from':  49093, 'from_side':    'r', 'to':  30214, 'to_side':    'r', 'mean_dist':     17, 'mapq':  60060, 'bcount':     12, 'min_dist':     13, 'max_dist':     25, 'probability': 0.064232, 'to_alt':      1, 'from_alt':      2},
        {'from':  49093, 'from_side':    'r', 'to': 109207, 'to_side':    'l', 'mean_dist':    780, 'mapq':  60060, 'bcount':     11, 'min_dist':    722, 'max_dist':   1043, 'probability': 0.081320, 'to_alt':      2, 'from_alt':      2},
        {'from':  47516, 'from_side':    'l', 'to': 101215, 'to_side':    'r', 'mean_dist':    453, 'mapq':  60060, 'bcount':     13, 'min_dist':    422, 'max_dist':    595, 'probability': 0.135136, 'to_alt':      2, 'from_alt':      1},
        {'from':  47516, 'from_side':    'r', 'to':  96716, 'to_side':    'r', 'mean_dist':    -45, 'mapq':  60060, 'bcount':     16, 'min_dist':    -49, 'max_dist':    -36, 'probability': 0.159802, 'to_alt':      2, 'from_alt':      1},
        {'from':  51660, 'from_side':    'l', 'to':  49093, 'to_side':    'l', 'mean_dist':    -14, 'mapq':  60060, 'bcount':     18, 'min_dist':    -19, 'max_dist':     14, 'probability': 0.231835, 'to_alt':      2, 'from_alt':      1},
        {'from':  51660, 'from_side':    'r', 'to': 101215, 'to_side':    'l', 'mean_dist':      5, 'mapq':  60060, 'bcount':     17, 'min_dist':      0, 'max_dist':     18, 'probability': 0.193782, 'to_alt':      2, 'from_alt':      1},
        {'from':  53480, 'from_side':    'l', 'to':   2722, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     16, 'min_dist':      0, 'max_dist':      0, 'probability': 0.159802, 'to_alt':      1, 'from_alt':      2},
        {'from':  53480, 'from_side':    'l', 'to': 101373, 'to_side':    'r', 'mean_dist':  -1383, 'mapq':  60060, 'bcount':     12, 'min_dist':  -1547, 'max_dist':  -1281, 'probability': 0.105770, 'to_alt':      2, 'from_alt':      2},
        {'from':  53480, 'from_side':    'r', 'to':  31749, 'to_side':    'l', 'mean_dist':   -646, 'mapq':  60060, 'bcount':     18, 'min_dist':   -716, 'max_dist':   -614, 'probability': 0.231835, 'to_alt':      2, 'from_alt':      2},
        {'from':  53480, 'from_side':    'r', 'to':  40554, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':     19, 'min_dist':      0, 'max_dist':      0, 'probability': 0.273725, 'to_alt':      1, 'from_alt':      2},
        {'from':  56740, 'from_side':    'l', 'to': 101215, 'to_side':    'r', 'mean_dist':    469, 'mapq':  60060, 'bcount':     15, 'min_dist':    418, 'max_dist':    704, 'probability': 0.209365, 'to_alt':      2, 'from_alt':      1},
        {'from':  56740, 'from_side':    'r', 'to':  96716, 'to_side':    'r', 'mean_dist':    -45, 'mapq':  60060, 'bcount':     14, 'min_dist':    -51, 'max_dist':    -23, 'probability': 0.104244, 'to_alt':      2, 'from_alt':      1},
        {'from':  56987, 'from_side':    'l', 'to':  49093, 'to_side':    'l', 'mean_dist':      4, 'mapq':  60060, 'bcount':     21, 'min_dist':      0, 'max_dist':     15, 'probability': 0.367256, 'to_alt':      2, 'from_alt':      1},
        {'from':  56987, 'from_side':    'r', 'to': 101215, 'to_side':    'l', 'mean_dist':    -10, 'mapq':  60060, 'bcount':     19, 'min_dist':    -12, 'max_dist':     -7, 'probability': 0.273725, 'to_alt':      2, 'from_alt':      1},
        {'from':  58443, 'from_side':    'r', 'to': 107486, 'to_side':    'l', 'mean_dist':    -83, 'mapq':  60060, 'bcount':     15, 'min_dist':   -103, 'max_dist':    -76, 'probability': 0.129976, 'to_alt':      2, 'from_alt':      1},
        {'from':  70951, 'from_side':    'l', 'to': 101373, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     29, 'min_dist':      0, 'max_dist':      0, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  71091, 'from_side':    'l', 'to':     44, 'to_side':    'r', 'mean_dist':      1, 'mapq':  60060, 'bcount':      5, 'min_dist':     -1, 'max_dist':      6, 'probability': 0.007368, 'to_alt':      2, 'from_alt':      2},
        {'from':  71091, 'from_side':    'l', 'to':   1306, 'to_side':    'l', 'mean_dist':    -45, 'mapq':  60060, 'bcount':      8, 'min_dist':    -49, 'max_dist':    -40, 'probability': 0.020422, 'to_alt':      1, 'from_alt':      2},
        {'from':  71091, 'from_side':    'r', 'to':  32229, 'to_side':    'r', 'mean_dist':      3, 'mapq':  60060, 'bcount':     12, 'min_dist':      1, 'max_dist':      7, 'probability': 0.064232, 'to_alt':      1, 'from_alt':      2},
        {'from':  71091, 'from_side':    'r', 'to':  76860, 'to_side':    'l', 'mean_dist':   2134, 'mapq':  60060, 'bcount':      4, 'min_dist':   2026, 'max_dist':   2303, 'probability': 0.012554, 'to_alt':      2, 'from_alt':      2},
        {'from':  76860, 'from_side':    'l', 'to':  32229, 'to_side':    'l', 'mean_dist':   -456, 'mapq':  60060, 'bcount':      7, 'min_dist':   -527, 'max_dist':   -412, 'probability': 0.014765, 'to_alt':      1, 'from_alt':      2},
        {'from':  76860, 'from_side':    'l', 'to':  71091, 'to_side':    'r', 'mean_dist':   2134, 'mapq':  60060, 'bcount':      4, 'min_dist':   2026, 'max_dist':   2303, 'probability': 0.012554, 'to_alt':      2, 'from_alt':      2},
        {'from':  76860, 'from_side':    'r', 'to':  12896, 'to_side':    'l', 'mean_dist':    -46, 'mapq':  60060, 'bcount':      6, 'min_dist':    -51, 'max_dist':    -43, 'probability': 0.010512, 'to_alt':      1, 'from_alt':      2},
        {'from':  76860, 'from_side':    'r', 'to':  12910, 'to_side':    'l', 'mean_dist':    -42, 'mapq':  60060, 'bcount':      5, 'min_dist':    -52, 'max_dist':    -30, 'probability': 0.007368, 'to_alt':      1, 'from_alt':      2},
        {'from':  96716, 'from_side':    'l', 'to': 118890, 'to_side':    'l', 'mean_dist':     -3, 'mapq':  60060, 'bcount':     16, 'min_dist':     -6, 'max_dist':     16, 'probability': 0.159802, 'to_alt':      1, 'from_alt':      2},
        {'from':  96716, 'from_side':    'l', 'to': 118892, 'to_side':    'l', 'mean_dist':    -38, 'mapq':  60060, 'bcount':     17, 'min_dist':    -43, 'max_dist':    -27, 'probability': 0.193782, 'to_alt':      1, 'from_alt':      2},
        {'from':  96716, 'from_side':    'r', 'to':  47516, 'to_side':    'r', 'mean_dist':    -45, 'mapq':  60060, 'bcount':     16, 'min_dist':    -49, 'max_dist':    -36, 'probability': 0.159802, 'to_alt':      1, 'from_alt':      2},
        {'from':  96716, 'from_side':    'r', 'to':  56740, 'to_side':    'r', 'mean_dist':    -45, 'mapq':  60060, 'bcount':     14, 'min_dist':    -51, 'max_dist':    -23, 'probability': 0.104244, 'to_alt':      1, 'from_alt':      2},
        {'from':  99004, 'from_side':    'r', 'to': 107486, 'to_side':    'r', 'mean_dist':    838, 'mapq':  60060, 'bcount':     14, 'min_dist':    787, 'max_dist':    885, 'probability': 0.169655, 'to_alt':      3, 'from_alt':      1},
        {'from':  99341, 'from_side':    'l', 'to':   9344, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     20, 'min_dist':      0, 'max_dist':      0, 'probability': 0.319050, 'to_alt':      2, 'from_alt':      1},
        {'from':  99341, 'from_side':    'r', 'to':  99342, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     21, 'min_dist':      0, 'max_dist':      0, 'probability': 0.367256, 'to_alt':      2, 'from_alt':      1},
        {'from':  99342, 'from_side':    'l', 'to':   9344, 'to_side':    'l', 'mean_dist':     73, 'mapq':  60060, 'bcount':     12, 'min_dist':     64, 'max_dist':     80, 'probability': 0.064232, 'to_alt':      2, 'from_alt':      2},
        {'from':  99342, 'from_side':    'l', 'to':  99341, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':     21, 'min_dist':      0, 'max_dist':      0, 'probability': 0.367256, 'to_alt':      1, 'from_alt':      2},
        {'from':  99342, 'from_side':    'r', 'to':     44, 'to_side':    'l', 'mean_dist':    -44, 'mapq':  60060, 'bcount':     19, 'min_dist':    -49, 'max_dist':    -33, 'probability': 0.273725, 'to_alt':      1, 'from_alt':      2},
        {'from':  99342, 'from_side':    'r', 'to': 107483, 'to_side':    'l', 'mean_dist':    -44, 'mapq':  60060, 'bcount':     11, 'min_dist':    -49, 'max_dist':    -37, 'probability': 0.049326, 'to_alt':      1, 'from_alt':      2},
        {'from': 101215, 'from_side':    'l', 'to':  51660, 'to_side':    'r', 'mean_dist':      5, 'mapq':  60060, 'bcount':     17, 'min_dist':      0, 'max_dist':     18, 'probability': 0.193782, 'to_alt':      1, 'from_alt':      2},
        {'from': 101215, 'from_side':    'l', 'to':  56987, 'to_side':    'r', 'mean_dist':    -10, 'mapq':  60060, 'bcount':     19, 'min_dist':    -12, 'max_dist':     -7, 'probability': 0.273725, 'to_alt':      1, 'from_alt':      2},
        {'from': 101215, 'from_side':    'r', 'to':  47516, 'to_side':    'l', 'mean_dist':    453, 'mapq':  60060, 'bcount':     13, 'min_dist':    422, 'max_dist':    595, 'probability': 0.135136, 'to_alt':      1, 'from_alt':      2},
        {'from': 101215, 'from_side':    'r', 'to':  56740, 'to_side':    'l', 'mean_dist':    469, 'mapq':  60060, 'bcount':     15, 'min_dist':    418, 'max_dist':    704, 'probability': 0.209365, 'to_alt':      1, 'from_alt':      2},
        {'from': 101373, 'from_side':    'l', 'to':  31749, 'to_side':    'r', 'mean_dist':   -350, 'mapq':  60060, 'bcount':    109, 'min_dist':   -433, 'max_dist':   -288, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from': 101373, 'from_side':    'l', 'to':  70951, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     29, 'min_dist':      0, 'max_dist':      0, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from': 101373, 'from_side':    'r', 'to':  53480, 'to_side':    'l', 'mean_dist':  -1383, 'mapq':  60060, 'bcount':     12, 'min_dist':  -1547, 'max_dist':  -1281, 'probability': 0.105770, 'to_alt':      2, 'from_alt':      2},
        {'from': 101373, 'from_side':    'r', 'to': 107486, 'to_side':    'r', 'mean_dist':  -1922, 'mapq':  60060, 'bcount':     26, 'min_dist':  -2024, 'max_dist':  -1836, 'probability': 0.500000, 'to_alt':      3, 'from_alt':      2},
        {'from': 107483, 'from_side':    'l', 'to':  99342, 'to_side':    'r', 'mean_dist':    -44, 'mapq':  60060, 'bcount':     11, 'min_dist':    -49, 'max_dist':    -37, 'probability': 0.049326, 'to_alt':      2, 'from_alt':      1},
        {'from': 107483, 'from_side':    'r', 'to': 107484, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     19, 'min_dist':      0, 'max_dist':      0, 'probability': 0.273725, 'to_alt':      2, 'from_alt':      1},
        {'from': 107484, 'from_side':    'l', 'to': 107483, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':     19, 'min_dist':      0, 'max_dist':      0, 'probability': 0.273725, 'to_alt':      1, 'from_alt':      2},
        {'from': 107484, 'from_side':    'l', 'to': 117303, 'to_side':    'l', 'mean_dist':   -888, 'mapq':  60060, 'bcount':      8, 'min_dist':   -912, 'max_dist':   -865, 'probability': 0.033108, 'to_alt':      1, 'from_alt':      2},
        {'from': 107484, 'from_side':    'r', 'to': 107485, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     49, 'min_dist':      0, 'max_dist':      0, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from': 107485, 'from_side':    'l', 'to':    372, 'to_side':    'r', 'mean_dist':  -1436, 'mapq':  60060, 'bcount':     14, 'min_dist':  -1605, 'max_dist':  -1350, 'probability': 0.169655, 'to_alt':      1, 'from_alt':      2},
        {'from': 107485, 'from_side':    'l', 'to': 107484, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':     49, 'min_dist':      0, 'max_dist':      0, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from': 107485, 'from_side':    'r', 'to': 107486, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':    106, 'min_dist':      0, 'max_dist':      0, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from': 107486, 'from_side':    'l', 'to':  58443, 'to_side':    'r', 'mean_dist':    -83, 'mapq':  60060, 'bcount':     15, 'min_dist':   -103, 'max_dist':    -76, 'probability': 0.129976, 'to_alt':      1, 'from_alt':      2},
        {'from': 107486, 'from_side':    'l', 'to': 107485, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':    106, 'min_dist':      0, 'max_dist':      0, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from': 107486, 'from_side':    'r', 'to':  33994, 'to_side':    'l', 'mean_dist':    -44, 'mapq':  60060, 'bcount':     25, 'min_dist':    -52, 'max_dist':    -15, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      3},
        {'from': 107486, 'from_side':    'r', 'to':  99004, 'to_side':    'r', 'mean_dist':    838, 'mapq':  60060, 'bcount':     14, 'min_dist':    787, 'max_dist':    885, 'probability': 0.169655, 'to_alt':      1, 'from_alt':      3},
        {'from': 107486, 'from_side':    'r', 'to': 101373, 'to_side':    'r', 'mean_dist':  -1922, 'mapq':  60060, 'bcount':     26, 'min_dist':  -2024, 'max_dist':  -1836, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      3},
        {'from': 109207, 'from_side':    'l', 'to':  30214, 'to_side':    'l', 'mean_dist':     -6, 'mapq':  60060, 'bcount':     13, 'min_dist':    -10, 'max_dist':     21, 'probability': 0.082422, 'to_alt':      1, 'from_alt':      2},
        {'from': 109207, 'from_side':    'l', 'to':  49093, 'to_side':    'r', 'mean_dist':    780, 'mapq':  60060, 'bcount':     11, 'min_dist':    722, 'max_dist':   1043, 'probability': 0.081320, 'to_alt':      2, 'from_alt':      2},
        {'from': 109207, 'from_side':    'r', 'to':  26855, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     13, 'min_dist':    -51, 'max_dist':    -14, 'probability': 0.082422, 'to_alt':      1, 'from_alt':      2},
        {'from': 109207, 'from_side':    'r', 'to':  30179, 'to_side':    'l', 'mean_dist':    -40, 'mapq':  60060, 'bcount':      8, 'min_dist':    -47, 'max_dist':    -23, 'probability': 0.020422, 'to_alt':      1, 'from_alt':      2},
        {'from': 110827, 'from_side':    'r', 'to':     69, 'to_side':    'l', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     24, 'min_dist':    -50, 'max_dist':    -16, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from': 112333, 'from_side':    'l', 'to':  13434, 'to_side':    'l', 'mean_dist':     38, 'mapq':  60060, 'bcount':     18, 'min_dist':     32, 'max_dist':     52, 'probability': 0.231835, 'to_alt':      1, 'from_alt':      2},
        {'from': 112333, 'from_side':    'l', 'to':  13455, 'to_side':    'l', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     29, 'min_dist':    -49, 'max_dist':    -34, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from': 112333, 'from_side':    'r', 'to':    114, 'to_side':    'l', 'mean_dist':   -819, 'mapq':  60060, 'bcount':     22, 'min_dist':   -931, 'max_dist':   -785, 'probability': 0.417654, 'to_alt':      1, 'from_alt':      2},
        {'from': 112333, 'from_side':    'r', 'to':  41636, 'to_side':    'r', 'mean_dist':   -710, 'mapq':  60060, 'bcount':     20, 'min_dist':   -775, 'max_dist':   -662, 'probability': 0.319050, 'to_alt':      1, 'from_alt':      2},
        {'from': 115803, 'from_side':    'l', 'to':    115, 'to_side':    'r', 'mean_dist':    885, 'mapq':  60060, 'bcount':     15, 'min_dist':    818, 'max_dist':   1120, 'probability': 0.209365, 'to_alt':      2, 'from_alt':      2},
        {'from': 115803, 'from_side':    'l', 'to':  15177, 'to_side':    'r', 'mean_dist':      5, 'mapq':  60060, 'bcount':     12, 'min_dist':      1, 'max_dist':     15, 'probability': 0.064232, 'to_alt':      1, 'from_alt':      2},
        {'from': 117303, 'from_side':    'l', 'to': 107484, 'to_side':    'l', 'mean_dist':   -888, 'mapq':  60060, 'bcount':      8, 'min_dist':   -912, 'max_dist':   -865, 'probability': 0.033108, 'to_alt':      2, 'from_alt':      1},
        {'from': 117303, 'from_side':    'r', 'to':    929, 'to_side':    'l', 'mean_dist':   2544, 'mapq':  60060, 'bcount':      7, 'min_dist':   2451, 'max_dist':   2631, 'probability': 0.076700, 'to_alt':      2, 'from_alt':      1},
        {'from': 117304, 'from_side':    'r', 'to':    929, 'to_side':    'l', 'mean_dist':   2489, 'mapq':  60060, 'bcount':      2, 'min_dist':   2450, 'max_dist':   2534, 'probability': 0.008611, 'to_alt':      2, 'from_alt':      1},
        {'from': 118890, 'from_side':    'l', 'to':  96716, 'to_side':    'l', 'mean_dist':     -3, 'mapq':  60060, 'bcount':     16, 'min_dist':     -6, 'max_dist':     16, 'probability': 0.159802, 'to_alt':      2, 'from_alt':      1},
        {'from': 118890, 'from_side':    'r', 'to':   9344, 'to_side':    'r', 'mean_dist':    -13, 'mapq':  60060, 'bcount':     25, 'min_dist':    -20, 'max_dist':     39, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from': 118892, 'from_side':    'l', 'to':  96716, 'to_side':    'l', 'mean_dist':    -38, 'mapq':  60060, 'bcount':     17, 'min_dist':    -43, 'max_dist':    -27, 'probability': 0.193782, 'to_alt':      2, 'from_alt':      1},
        {'from': 118892, 'from_side':    'r', 'to':   9344, 'to_side':    'r', 'mean_dist':      8, 'mapq':  60060, 'bcount':     20, 'min_dist':      0, 'max_dist':     35, 'probability': 0.319050, 'to_alt':      2, 'from_alt':      1}
        ]) )
    result_paths.append( pd.DataFrame({
        'pid':      [    100,    100,    100,    100,    100,    100,    100,    100,    100,    100,    100,    100,    100,    100,    100,    100,    100,    100,    100,    100,    100],
        'pos':      [      0,      1,      2,      3,      4,      5,      6,      7,      8,      9,     10,     11,     12,     13,     14,     15,     16,     17,     18,     19,     20],
        'phase0':   [    101,    101,    101,    101,    101,    101,    101,    101,    101,    101,    101,    101,    101,    101,    101,    101,    101,    101,    101,    101,    101],
        'scaf0':    [   2799,    674,  20727,   2885,  15812,  10723,  12896,  76860,  32229,  71091,   1306,     44,  99342,  99341,   9344, 118890,  96716,  56740, 101215,  56987,  49093],
        'strand0':  [    '-',    '+',    '-',    '+',    '+',    '-',    '-',    '-',    '+',    '-',    '+',    '-',    '-',    '-',    '+',    '-',    '+',    '-',    '-',    '-',    '+'],
        'dist0':    [      0,     60,    -68,    -44,    -44,    -43,    -43,    -46,   -456,      3,    -45,    -43,    -44,      0,      0,    -13,     -3,    -45,    469,    -10,      4],
        'phase1':   [   -102,   -102,    102,    102,    102,    102,    102,    102,    102,    102,    102,    102,   -102,   -102,   -102,   -102,   -102,   -102,   -102,   -102,   -102],
        'scaf1':    [     -1,     -1,  20727,   2885,  13452,  10723,  12910,  76860,     -1,  71091,     -1,     44,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1],
        'strand1':  [     '',     '',    '-',    '+',    '-',    '-',    '-',    '-',     '',    '-',     '',    '-',     '',     '',     '',     '',     '',     '',     '',     '',     ''],
        'dist1':    [      0,      0,    -18,    245,    -45,    -41,    -43,    -42,      0,   2134,      0,      1,      0,      0,      0,      0,      0,      0,      0,      0,      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    101,    101,    101,    101],
        'pos':      [      0,      1,      2,      3],
        'phase0':   [    103,    103,    103,    103],
        'scaf0':    [ 109207,  30179,     -1,  47404],
        'strand0':  [    '+',    '+',     '',    '+'],
        'dist0':    [      0,    -40,      0,    -32],
        'phase1':   [   -104,    104,    104,    104],
        'scaf1':    [     -1,  26855,  33144,  47404],
        'strand1':  [     '',    '-',    '-',    '+'],
        'dist1':    [      0,    -43,    -40,  -4062]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    102,    102,    102,    102,    102,    102,    102,    102,    102,    102,    102,    102,    102],
        'pos':      [      0,      1,      2,      3,      4,      5,      6,      7,      8,      9,     10,     11,     12],
        'phase0':   [    105,    105,    105,    105,    105,    105,    105,    105,    105,    105,    105,    105,    105],
        'scaf0':    [  30214,  49093,  51660, 101215,  47516,  96716, 118892,   9344,  99342, 107483, 107484, 107485, 107486],
        'strand0':  [    '+',    '-',    '+',    '+',    '+',    '-',    '+',    '-',    '+',    '+',    '+',    '+',    '+'],
        'dist0':    [      0,     17,    -14,      5,    453,    -45,    -38,      8,     73,    -44,      0,      0,      0],
        'phase1':   [   -106,   -106,   -106,   -106,   -106,   -106,   -106,   -106,   -106,   -106,   -106,   -106,   -106],
        'scaf1':    [     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1],
        'strand1':  [     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''],
        'dist1':    [      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    103,    103,    103,    103,    103],
        'pos':      [      0,      1,      2,      3,      4],
        'phase0':   [    107,    107,    107,    107,    107],
        'scaf0':    [  13591,  11659,   2725,  14096,    929],
        'strand0':  [    '+',    '+',    '-',    '-',    '-'],
        'dist0':    [      0,    -43,    136,    642,    -44],
        'phase1':   [   -108,    108,    108,    108,    108],
        'scaf1':    [     -1,  13029,   2725,     -1,    929],
        'strand1':  [     '',    '-',    '-',     '',    '-'],
        'dist1':    [      0,    -43,    143,      0,   1205]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    104],
        'pos':      [      0],
        'phase0':   [    109],
        'scaf0':    [ 117304],
        'strand0':  [    '+'],
        'dist0':    [      0],
        'phase1':   [   -110],
        'scaf1':    [     -1],
        'strand1':  [     ''],
        'dist1':    [      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    105,    105,    105,    105,    105],
        'pos':      [      0,      1,      2,      3,      4],
        'phase0':   [    111,    111,    111,    111,    111],
        'scaf0':    [ 117303, 107484, 107485, 107486,  33994],
        'strand0':  [    '-',    '+',    '+',    '+',    '+'],
        'dist0':    [      0,   -888,      0,      0,    -44],
        'phase1':   [   -112,   -112,   -112,   -112,   -112],
        'scaf1':    [     -1,     -1,     -1,     -1,     -1],
        'strand1':  [     '',     '',     '',     '',     ''],
        'dist1':    [      0,      0,      0,      0,      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    106],
        'pos':      [      0],
        'phase0':   [    113],
        'scaf0':    [  58443],
        'strand0':  [    '+'],
        'dist0':    [      0],
        'phase1':   [   -114],
        'scaf1':    [     -1],
        'strand1':  [     ''],
        'dist1':    [      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    107,    107],
        'pos':      [      0,      1],
        'phase0':   [    115,    115],
        'scaf0':    [  99004, 107486],
        'strand0':  [    '+',    '-'],
        'dist0':    [      0,    838],
        'phase1':   [   -116,   -116],
        'scaf1':    [     -1,     -1],
        'strand1':  [     '',     ''],
        'dist1':    [      0,      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    108],
        'pos':      [      0],
        'phase0':   [    117],
        'scaf0':    [ 110827],
        'strand0':  [    '+'],
        'dist0':    [      0],
        'phase1':   [   -118],
        'scaf1':    [     -1],
        'strand1':  [     ''],
        'dist1':    [      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    109,    109,    109,    109,    109,    109,    109],
        'pos':      [      0,      1,      2,      3,      4,      5,      6],
        'phase0':   [    119,    119,    119,    119,    119,    119,    119],
        'scaf0':    [     69,  13434, 112333,    114,    115,     -1, 115803],
        'strand0':  [    '+',    '-',    '+',    '+',    '+',    '+',    '+'],
        'dist0':    [      0,    -41,     38,   -819,   1131,      0,    885],
        'phase1':   [   -120,    120,    120,    120,    120,    120,    120],
        'scaf1':    [     -1,  13455, 112333,  41636,    115,  15177, 115803],
        'strand1':  [     '',    '-',    '+',    '-',    '+',    '+',    '+'],
        'dist1':    [      0,    -43,    -43,   -710,   1932,    -42,      5]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    110,    110,    110,    110,    110],
        'pos':      [      0,      1,      2,      3,      4],
        'phase0':   [    121,    121,    121,    121,    121],
        'scaf0':    [    372, 107485, 107486, 101373,  70951],
        'strand0':  [    '+',    '+',    '+',    '-',    '+'],
        'dist0':    [      0,  -1436,      0,  -1922,      0],
        'phase1':   [   -122,   -122,   -122,   -122,   -122],
        'scaf1':    [     -1,     -1,     -1,     -1,     -1],
        'strand1':  [     '',     '',     '',     '',     ''],
        'dist1':    [      0,      0,      0,      0,      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    111,    111,    111,    111,    111,    111,    111],
        'pos':      [      0,      1,      2,      3,      4,      5,      6],
        'phase0':   [    123,    123,    123,    123,    123,    123,    123],
        'scaf0':    [ 107486, 101373,  31749,  53480, 101373,  31749,  31756],
        'strand0':  [    '+',    '-',    '-',    '-',    '-',    '-',    '-'],
        'dist0':    [      0,  -1922,   -350,   -646,  -1383,   -350,     -1],
        'phase1':   [   -124,   -124,   -124,   -124,   -124,   -124,   -124],
        'scaf1':    [     -1,     -1,     -1,     -1,     -1,     -1,     -1],
        'strand1':  [     '',     '',     '',     '',     '',     '',     ''],
        'dist1':    [      0,      0,      0,      0,      0,      0,      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    112,    112,    112],
        'pos':      [      0,      1,      2],
        'phase0':   [    125,    125,    125],
        'scaf0':    [  40554,  53480,   2722],
        'strand0':  [    '+',    '-',    '+'],
        'dist0':    [      0,      0,      0],
        'phase1':   [   -126,   -126,   -126],
        'scaf1':    [     -1,     -1,     -1],
        'strand1':  [     '',     '',     ''],
        'dist1':    [      0,      0,      0]
        }) )
#
    # Test 2
    scaffolds.append( pd.DataFrame({'case':2, 'scaffold':[19382, 66038 ,115947, 115948, 115949, 115950, 115951, 115952]}) )
    scaffold_graph.append( pd.DataFrame([
        {'from':  19382, 'from_side':    'l', 'length':      4, 'scaf1': 115947, 'strand1':    '+', 'dist1':    -41, 'scaf2': 115949, 'strand2':    '+', 'dist2':    515, 'scaf3': 115951, 'strand3':    '+', 'dist3':   9325},
        {'from':  66038, 'from_side':    'r', 'length':      3, 'scaf1': 115947, 'strand1':    '+', 'dist1':    381, 'scaf2': 115948, 'strand2':    '+', 'dist2':      0},
        {'from': 115947, 'from_side':    'l', 'length':      2, 'scaf1':  19382, 'strand1':    '+', 'dist1':    -41},
        {'from': 115947, 'from_side':    'l', 'length':      2, 'scaf1':  66038, 'strand1':    '-', 'dist1':    381},
        {'from': 115947, 'from_side':    'r', 'length':      2, 'scaf1': 115948, 'strand1':    '+', 'dist1':      0},
        {'from': 115947, 'from_side':    'r', 'length':      4, 'scaf1': 115949, 'strand1':    '+', 'dist1':    515, 'scaf2': 115951, 'strand2':    '+', 'dist2':   9325, 'scaf3': 115952, 'strand3':    '-', 'dist3':    -43},
        {'from': 115948, 'from_side':    'l', 'length':      3, 'scaf1': 115947, 'strand1':    '-', 'dist1':      0, 'scaf2':  66038, 'strand2':    '-', 'dist2':    381},
        {'from': 115948, 'from_side':    'r', 'length':      5, 'scaf1': 115949, 'strand1':    '+', 'dist1':      0, 'scaf2': 115950, 'strand2':    '+', 'dist2':      0, 'scaf3': 115951, 'strand3':    '+', 'dist3':      0, 'scaf4': 115952, 'strand4':    '+', 'dist4':    -42},
        {'from': 115949, 'from_side':    'l', 'length':      3, 'scaf1': 115947, 'strand1':    '-', 'dist1':    515, 'scaf2':  19382, 'strand2':    '+', 'dist2':    -41},
        {'from': 115949, 'from_side':    'l', 'length':      2, 'scaf1': 115948, 'strand1':    '-', 'dist1':      0},
        {'from': 115949, 'from_side':    'r', 'length':      4, 'scaf1': 115950, 'strand1':    '+', 'dist1':      0, 'scaf2': 115951, 'strand2':    '+', 'dist2':      0, 'scaf3': 115952, 'strand3':    '+', 'dist3':    -42},
        {'from': 115949, 'from_side':    'r', 'length':      3, 'scaf1': 115951, 'strand1':    '+', 'dist1':   9325, 'scaf2': 115952, 'strand2':    '-', 'dist2':    -43},
        {'from': 115950, 'from_side':    'l', 'length':      3, 'scaf1': 115949, 'strand1':    '-', 'dist1':      0, 'scaf2': 115948, 'strand2':    '-', 'dist2':      0},
        {'from': 115950, 'from_side':    'r', 'length':      3, 'scaf1': 115951, 'strand1':    '+', 'dist1':      0, 'scaf2': 115952, 'strand2':    '+', 'dist2':    -42},
        {'from': 115951, 'from_side':    'l', 'length':      4, 'scaf1': 115949, 'strand1':    '-', 'dist1':   9325, 'scaf2': 115947, 'strand2':    '-', 'dist2':    515, 'scaf3':  19382, 'strand3':    '+', 'dist3':    -41},
        {'from': 115951, 'from_side':    'l', 'length':      4, 'scaf1': 115950, 'strand1':    '-', 'dist1':      0, 'scaf2': 115949, 'strand2':    '-', 'dist2':      0, 'scaf3': 115948, 'strand3':    '-', 'dist3':      0},
        {'from': 115951, 'from_side':    'r', 'length':      2, 'scaf1': 115952, 'strand1':    '+', 'dist1':    -42},
        {'from': 115951, 'from_side':    'r', 'length':      2, 'scaf1': 115952, 'strand1':    '-', 'dist1':    -43},
        {'from': 115952, 'from_side':    'l', 'length':      5, 'scaf1': 115951, 'strand1':    '-', 'dist1':    -42, 'scaf2': 115950, 'strand2':    '-', 'dist2':      0, 'scaf3': 115949, 'strand3':    '-', 'dist3':      0, 'scaf4': 115948, 'strand4':    '-', 'dist4':      0},
        {'from': 115952, 'from_side':    'r', 'length':      4, 'scaf1': 115951, 'strand1':    '-', 'dist1':    -43, 'scaf2': 115949, 'strand2':    '-', 'dist2':   9325, 'scaf3': 115947, 'strand3':    '-', 'dist3':    515}
        ]) )
    scaf_bridges.append( pd.DataFrame([
        {'from':  19382, 'from_side':    'l', 'to': 115947, 'to_side':    'l', 'mean_dist':    -41, 'mapq':  60060, 'bcount':     22, 'min_dist':    -48, 'max_dist':    -25, 'probability': 0.417654, 'to_alt':      2, 'from_alt':      1},
        {'from':  66038, 'from_side':    'r', 'to': 115947, 'to_side':    'l', 'mean_dist':    381, 'mapq':  60060, 'bcount':     22, 'min_dist':    354, 'max_dist':    485, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from': 115947, 'from_side':    'l', 'to':  19382, 'to_side':    'l', 'mean_dist':    -41, 'mapq':  60060, 'bcount':     22, 'min_dist':    -48, 'max_dist':    -25, 'probability': 0.417654, 'to_alt':      1, 'from_alt':      2},
        {'from': 115947, 'from_side':    'l', 'to':  66038, 'to_side':    'r', 'mean_dist':    381, 'mapq':  60060, 'bcount':     22, 'min_dist':    354, 'max_dist':    485, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from': 115947, 'from_side':    'r', 'to': 115948, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     23, 'min_dist':      0, 'max_dist':      0, 'probability': 0.469443, 'to_alt':      1, 'from_alt':      2},
        {'from': 115947, 'from_side':    'r', 'to': 115949, 'to_side':    'l', 'mean_dist':    515, 'mapq':  60060, 'bcount':     20, 'min_dist':    473, 'max_dist':    642, 'probability': 0.470466, 'to_alt':      2, 'from_alt':      2},
        {'from': 115948, 'from_side':    'l', 'to': 115947, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':     23, 'min_dist':      0, 'max_dist':      0, 'probability': 0.469443, 'to_alt':      2, 'from_alt':      1},
        {'from': 115948, 'from_side':    'r', 'to': 115949, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     17, 'min_dist':      0, 'max_dist':      0, 'probability': 0.193782, 'to_alt':      2, 'from_alt':      1},
        {'from': 115949, 'from_side':    'l', 'to': 115947, 'to_side':    'r', 'mean_dist':    515, 'mapq':  60060, 'bcount':     20, 'min_dist':    473, 'max_dist':    642, 'probability': 0.470466, 'to_alt':      2, 'from_alt':      2},
        {'from': 115949, 'from_side':    'l', 'to': 115948, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':     17, 'min_dist':      0, 'max_dist':      0, 'probability': 0.193782, 'to_alt':      1, 'from_alt':      2},
        {'from': 115949, 'from_side':    'r', 'to': 115950, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     16, 'min_dist':      0, 'max_dist':      0, 'probability': 0.159802, 'to_alt':      1, 'from_alt':      2},
        {'from': 115949, 'from_side':    'r', 'to': 115951, 'to_side':    'l', 'mean_dist':   9325, 'mapq':  60060, 'bcount':      6, 'min_dist':   8844, 'max_dist':  10272, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      2},
        {'from': 115950, 'from_side':    'l', 'to': 115949, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':     16, 'min_dist':      0, 'max_dist':      0, 'probability': 0.159802, 'to_alt':      2, 'from_alt':      1},
        {'from': 115950, 'from_side':    'r', 'to': 115951, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     13, 'min_dist':      0, 'max_dist':      0, 'probability': 0.082422, 'to_alt':      2, 'from_alt':      1},
        {'from': 115951, 'from_side':    'l', 'to': 115949, 'to_side':    'r', 'mean_dist':   9325, 'mapq':  60060, 'bcount':      6, 'min_dist':   8844, 'max_dist':  10272, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      2},
        {'from': 115951, 'from_side':    'l', 'to': 115950, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':     13, 'min_dist':      0, 'max_dist':      0, 'probability': 0.082422, 'to_alt':      1, 'from_alt':      2},
        {'from': 115951, 'from_side':    'r', 'to': 115952, 'to_side':    'l', 'mean_dist':    -42, 'mapq':  60060, 'bcount':     10, 'min_dist':    -49, 'max_dist':    -37, 'probability': 0.037322, 'to_alt':      1, 'from_alt':      2},
        {'from': 115951, 'from_side':    'r', 'to': 115952, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     20, 'min_dist':    -49, 'max_dist':    -28, 'probability': 0.319050, 'to_alt':      1, 'from_alt':      2},
        {'from': 115952, 'from_side':    'l', 'to': 115951, 'to_side':    'r', 'mean_dist':    -42, 'mapq':  60060, 'bcount':     10, 'min_dist':    -49, 'max_dist':    -37, 'probability': 0.037322, 'to_alt':      2, 'from_alt':      1},
        {'from': 115952, 'from_side':    'r', 'to': 115951, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     20, 'min_dist':    -49, 'max_dist':    -28, 'probability': 0.319050, 'to_alt':      2, 'from_alt':      1}
        ]) )
    result_paths.append( pd.DataFrame({
        'pid':      [    200,    200,    200,    200,    200,    200,    200,    200,    200,    200,    200],
        'pos':      [      0,      1,      2,      3,      4,      5,      6,      7,      8,      9,     10],
        'phase0':   [    201,    201,    201,    201,    201,    201,    201,    201,    201,    201,    201],
        'scaf0':    [  66038, 115947, 115948, 115949, 115950, 115951, 115952, 115951, 115949, 115947,  19382],
        'strand0':  [    '+',    '+',    '+',    '+',    '+',    '+',    '+',    '-',    '-',    '-',    '+'],
        'dist0':    [      0,    381,      0,      0,      0,      0,    -42,    -43,   9325,    515,    -41],
        'phase1':   [   -202,   -202,   -202,   -202,   -202,   -202,   -202,   -202,   -202,   -202,   -202],
        'scaf1':    [     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1],
        'strand1':  [     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''],
        'dist1':    [      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0]
        }) )
#
    # Test 3
    scaffolds.append( pd.DataFrame({'case':3, 'scaffold':[30387,  95786, 108403, 110072]}) )
    scaffold_graph.append( pd.DataFrame([
        {'from':  30387, 'from_side':    'l', 'length':      3, 'scaf1': 110072, 'strand1':    '+', 'dist1':    758, 'scaf2': 108403, 'strand2':    '+', 'dist2':    -30},
        {'from':  95786, 'from_side':    'l', 'length':      3, 'scaf1': 110072, 'strand1':    '-', 'dist1':    -29, 'scaf2': 108403, 'strand2':    '-', 'dist2':    813},
        {'from': 108403, 'from_side':    'l', 'length':      3, 'scaf1': 110072, 'strand1':    '-', 'dist1':    -30, 'scaf2':  30387, 'strand2':    '+', 'dist2':    758},
        {'from': 108403, 'from_side':    'l', 'length':      3, 'scaf1': 110072, 'strand1':    '-', 'dist1':    -30, 'scaf2': 108403, 'strand2':    '-', 'dist2':    813},
        {'from': 108403, 'from_side':    'r', 'length':      3, 'scaf1': 110072, 'strand1':    '+', 'dist1':    813, 'scaf2':  95786, 'strand2':    '+', 'dist2':    -29},
        {'from': 108403, 'from_side':    'r', 'length':      3, 'scaf1': 110072, 'strand1':    '+', 'dist1':    813, 'scaf2': 108403, 'strand2':    '+', 'dist2':    -30},
        {'from': 110072, 'from_side':    'l', 'length':      2, 'scaf1':  30387, 'strand1':    '+', 'dist1':    758},
        {'from': 110072, 'from_side':    'l', 'length':      3, 'scaf1': 108403, 'strand1':    '-', 'dist1':    813, 'scaf2': 110072, 'strand2':    '-', 'dist2':    -30},
        {'from': 110072, 'from_side':    'r', 'length':      2, 'scaf1':  95786, 'strand1':    '+', 'dist1':    -29},
        {'from': 110072, 'from_side':    'r', 'length':      3, 'scaf1': 108403, 'strand1':    '+', 'dist1':    -30, 'scaf2': 110072, 'strand2':    '+', 'dist2':    813}
        ]) )
    scaf_bridges.append( pd.DataFrame([
        {'from':  30387, 'from_side':    'l', 'to': 110072, 'to_side':    'l', 'mean_dist':    758, 'mapq':  60060, 'bcount':     17, 'min_dist':    711, 'max_dist':    992, 'probability': 0.303341, 'to_alt':      2, 'from_alt':      1},
        {'from':  95786, 'from_side':    'l', 'to': 110072, 'to_side':    'r', 'mean_dist':    -29, 'mapq':  60060, 'bcount':     24, 'min_dist':    -35, 'max_dist':    -20, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from': 108403, 'from_side':    'l', 'to': 110072, 'to_side':    'r', 'mean_dist':    -30, 'mapq':  60060, 'bcount':     34, 'min_dist':    -38, 'max_dist':    -24, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from': 108403, 'from_side':    'r', 'to': 110072, 'to_side':    'l', 'mean_dist':    813, 'mapq':  60060, 'bcount':     31, 'min_dist':    745, 'max_dist':   1052, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from': 110072, 'from_side':    'l', 'to':  30387, 'to_side':    'l', 'mean_dist':    758, 'mapq':  60060, 'bcount':     17, 'min_dist':    711, 'max_dist':    992, 'probability': 0.303341, 'to_alt':      1, 'from_alt':      2},
        {'from': 110072, 'from_side':    'l', 'to': 108403, 'to_side':    'r', 'mean_dist':    813, 'mapq':  60060, 'bcount':     31, 'min_dist':    745, 'max_dist':   1052, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from': 110072, 'from_side':    'r', 'to':  95786, 'to_side':    'l', 'mean_dist':    -29, 'mapq':  60060, 'bcount':     24, 'min_dist':    -35, 'max_dist':    -20, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from': 110072, 'from_side':    'r', 'to': 108403, 'to_side':    'l', 'mean_dist':    -30, 'mapq':  60060, 'bcount':     34, 'min_dist':    -38, 'max_dist':    -24, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2}
        ]) )
    result_paths.append( pd.DataFrame({
        'pid':      [    300,    300,    300,    300,    300,    300,    300],
        'pos':      [      0,      1,      2,      3,      4,      5,      6],
        'phase0':   [    301,    301,    301,    301,    301,    301,    301],
        'scaf0':    [  30387, 110072, 108403, 110072, 108403, 110072,  95786],
        'strand0':  [    '-',    '+',    '+',    '+',    '+',    '+',    '+'],
        'dist0':    [      0,    758,    -30,    813,    -30,    813,    -29],
        'phase1':   [   -302,   -302,   -302,   -302,   -302,   -302,   -302],
        'scaf1':    [     -1,     -1,     -1,     -1,     -1,     -1,     -1],
        'strand1':  [     '',     '',     '',     '',     '',     '',     ''],
        'dist1':    [      0,      0,      0,      0,      0,      0,      0]
        }) )
#
    # Test 4
    scaffolds.append( pd.DataFrame({'case':4, 'scaffold':[928, 9067, 12976, 13100, 20542, 45222, 80469]}) )
    scaffold_graph.append( pd.DataFrame([
        {'from':    928, 'from_side':    'r', 'length':      2, 'scaf1':  45222, 'strand1':    '+', 'dist1':    -44},
        {'from':    928, 'from_side':    'r', 'length':      2, 'scaf1':  45222, 'strand1':    '+', 'dist1':     20},
        {'from':   9067, 'from_side':    'l', 'length':      3, 'scaf1':   9067, 'strand1':    '-', 'dist1':   2982, 'scaf2':  80469, 'strand2':    '-', 'dist2':      4},
        {'from':   9067, 'from_side':    'l', 'length':      2, 'scaf1':  80469, 'strand1':    '-', 'dist1':      4},
        {'from':   9067, 'from_side':    'r', 'length':      3, 'scaf1':   9067, 'strand1':    '+', 'dist1':   2982, 'scaf2':  45222, 'strand2':    '-', 'dist2':    397},
        {'from':   9067, 'from_side':    'r', 'length':      2, 'scaf1':  45222, 'strand1':    '-', 'dist1':    397},
        {'from':  12976, 'from_side':    'l', 'length':      2, 'scaf1':  80469, 'strand1':    '+', 'dist1':    -44},
        {'from':  12976, 'from_side':    'r', 'length':      2, 'scaf1':  20542, 'strand1':    '+', 'dist1':    -46},
        {'from':  13100, 'from_side':    'l', 'length':      2, 'scaf1':  20542, 'strand1':    '+', 'dist1':    -43},
        {'from':  13100, 'from_side':    'r', 'length':      2, 'scaf1':  80469, 'strand1':    '+', 'dist1':    -42},
        {'from':  20542, 'from_side':    'l', 'length':      3, 'scaf1':  12976, 'strand1':    '-', 'dist1':    -46, 'scaf2':  80469, 'strand2':    '+', 'dist2':    -44},
        {'from':  20542, 'from_side':    'l', 'length':      3, 'scaf1':  13100, 'strand1':    '+', 'dist1':    -43, 'scaf2':  80469, 'strand2':    '+', 'dist2':    -42},
        {'from':  45222, 'from_side':    'l', 'length':      2, 'scaf1':    928, 'strand1':    '-', 'dist1':    -44},
        {'from':  45222, 'from_side':    'l', 'length':      2, 'scaf1':    928, 'strand1':    '-', 'dist1':     20},
        {'from':  45222, 'from_side':    'r', 'length':      4, 'scaf1':   9067, 'strand1':    '-', 'dist1':    397, 'scaf2':   9067, 'strand2':    '-', 'dist2':   2982, 'scaf3':  80469, 'strand3':    '-', 'dist3':      4},
        {'from':  45222, 'from_side':    'r', 'length':      3, 'scaf1':   9067, 'strand1':    '-', 'dist1':    397, 'scaf2':  80469, 'strand2':    '-', 'dist2':      4},
        {'from':  80469, 'from_side':    'l', 'length':      3, 'scaf1':  12976, 'strand1':    '+', 'dist1':    -44, 'scaf2':  20542, 'strand2':    '+', 'dist2':    -46},
        {'from':  80469, 'from_side':    'l', 'length':      3, 'scaf1':  13100, 'strand1':    '-', 'dist1':    -42, 'scaf2':  20542, 'strand2':    '+', 'dist2':    -43},
        {'from':  80469, 'from_side':    'r', 'length':      4, 'scaf1':   9067, 'strand1':    '+', 'dist1':      4, 'scaf2':   9067, 'strand2':    '+', 'dist2':   2982, 'scaf3':  45222, 'strand3':    '-', 'dist3':    397},
        {'from':  80469, 'from_side':    'r', 'length':      3, 'scaf1':   9067, 'strand1':    '+', 'dist1':      4, 'scaf2':  45222, 'strand2':    '-', 'dist2':    397}
        ]) )
    scaf_bridges.append( pd.DataFrame([
        {'from':    928, 'from_side':    'r', 'to':  45222, 'to_side':    'l', 'mean_dist':    -44, 'mapq':  60060, 'bcount':     10, 'min_dist':    -49, 'max_dist':    -39, 'probability': 0.037322, 'to_alt':      2, 'from_alt':      2},
        {'from':    928, 'from_side':    'r', 'to':  45222, 'to_side':    'l', 'mean_dist':     20, 'mapq':  60060, 'bcount':     10, 'min_dist':      9, 'max_dist':     39, 'probability': 0.037322, 'to_alt':      2, 'from_alt':      2},
        {'from':   9067, 'from_side':    'l', 'to':   9067, 'to_side':    'r', 'mean_dist':   2982, 'mapq':  60060, 'bcount':      7, 'min_dist':   2857, 'max_dist':   3235, 'probability': 0.076700, 'to_alt':      2, 'from_alt':      2},
        {'from':   9067, 'from_side':    'l', 'to':  80469, 'to_side':    'r', 'mean_dist':      4, 'mapq':  60060, 'bcount':     19, 'min_dist':      1, 'max_dist':     19, 'probability': 0.273725, 'to_alt':      1, 'from_alt':      2},
        {'from':   9067, 'from_side':    'r', 'to':   9067, 'to_side':    'l', 'mean_dist':   2982, 'mapq':  60060, 'bcount':      7, 'min_dist':   2857, 'max_dist':   3235, 'probability': 0.076700, 'to_alt':      2, 'from_alt':      2},
        {'from':   9067, 'from_side':    'r', 'to':  45222, 'to_side':    'r', 'mean_dist':    397, 'mapq':  60060, 'bcount':     24, 'min_dist':    379, 'max_dist':    424, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from':  12976, 'from_side':    'l', 'to':  80469, 'to_side':    'l', 'mean_dist':    -44, 'mapq':  60060, 'bcount':      9, 'min_dist':    -49, 'max_dist':    -37, 'probability': 0.027818, 'to_alt':      2, 'from_alt':      1},
        {'from':  12976, 'from_side':    'r', 'to':  20542, 'to_side':    'l', 'mean_dist':    -46, 'mapq':  60060, 'bcount':      5, 'min_dist':    -55, 'max_dist':    -35, 'probability': 0.007368, 'to_alt':      2, 'from_alt':      1},
        {'from':  13100, 'from_side':    'l', 'to':  20542, 'to_side':    'l', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     10, 'min_dist':    -54, 'max_dist':    -29, 'probability': 0.037322, 'to_alt':      2, 'from_alt':      1},
        {'from':  13100, 'from_side':    'r', 'to':  80469, 'to_side':    'l', 'mean_dist':    -42, 'mapq':  60060, 'bcount':      9, 'min_dist':    -47, 'max_dist':    -27, 'probability': 0.027818, 'to_alt':      2, 'from_alt':      1},
        {'from':  20542, 'from_side':    'l', 'to':  12976, 'to_side':    'r', 'mean_dist':    -46, 'mapq':  60060, 'bcount':      5, 'min_dist':    -55, 'max_dist':    -35, 'probability': 0.007368, 'to_alt':      1, 'from_alt':      2},
        {'from':  20542, 'from_side':    'l', 'to':  13100, 'to_side':    'l', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     10, 'min_dist':    -54, 'max_dist':    -29, 'probability': 0.037322, 'to_alt':      1, 'from_alt':      2},
        {'from':  45222, 'from_side':    'l', 'to':    928, 'to_side':    'r', 'mean_dist':    -44, 'mapq':  60060, 'bcount':     10, 'min_dist':    -49, 'max_dist':    -39, 'probability': 0.037322, 'to_alt':      2, 'from_alt':      2},
        {'from':  45222, 'from_side':    'l', 'to':    928, 'to_side':    'r', 'mean_dist':     20, 'mapq':  60060, 'bcount':     10, 'min_dist':      9, 'max_dist':     39, 'probability': 0.037322, 'to_alt':      2, 'from_alt':      2},
        {'from':  45222, 'from_side':    'r', 'to':   9067, 'to_side':    'r', 'mean_dist':    397, 'mapq':  60060, 'bcount':     24, 'min_dist':    379, 'max_dist':    424, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  80469, 'from_side':    'l', 'to':  12976, 'to_side':    'l', 'mean_dist':    -44, 'mapq':  60060, 'bcount':      9, 'min_dist':    -49, 'max_dist':    -37, 'probability': 0.027818, 'to_alt':      1, 'from_alt':      2},
        {'from':  80469, 'from_side':    'l', 'to':  13100, 'to_side':    'r', 'mean_dist':    -42, 'mapq':  60060, 'bcount':      9, 'min_dist':    -47, 'max_dist':    -27, 'probability': 0.027818, 'to_alt':      1, 'from_alt':      2},
        {'from':  80469, 'from_side':    'r', 'to':   9067, 'to_side':    'l', 'mean_dist':      4, 'mapq':  60060, 'bcount':     19, 'min_dist':      1, 'max_dist':     19, 'probability': 0.273725, 'to_alt':      2, 'from_alt':      1}
        ]) )
    result_paths.append( pd.DataFrame({
        'pid':      [    400,    400,    400,    400,    400,    400,    400],
        'pos':      [      0,      1,      2,      3,      4,      5,      6],
        'phase0':   [    401,    401,    401,    401,    401,    401,    401],
        'scaf0':    [    928,  45222,   9067,     -1,  80469,  13100,  20542],
        'strand0':  [    '+',    '+',    '-',     '',    '-',    '-',    '+'],
        'dist0':    [      0,     20,    397,      0,      4,    -42,    -43],
        'phase1':   [   -402,    402,   -402,    402,   -402,    402,    402],
        'scaf1':    [     -1,  45222,     -1,   9067,     -1,  12976,  20542],
        'strand1':  [     '',    '+',     '',    '-',     '',    '+',    '+'],
        'dist1':    [      0,    -44,      0,   2982,      0,    -44,    -46]
        }) )
#
    # Test 5
    scaffolds.append( pd.DataFrame({'case':5, 'scaffold':[7, 1440, 7349, 10945, 11769, 23515, 29100, 30446, 31108, 31729, 31737, 31758, 32135, 32420, 45782, 45783, 47750, 49372, 54753, 74998, 76037, 86633, 93920, 95291, 105853, 110006, 113898]}) )
    scaffold_graph.append( pd.DataFrame([
        {'from':      7, 'from_side':    'l', 'length':      4, 'scaf1':  11769, 'strand1':    '+', 'dist1':     76, 'scaf2':   1440, 'strand2':    '+', 'dist2':    -45, 'scaf3': 110006, 'strand3':    '-', 'dist3':    406},
        {'from':      7, 'from_side':    'l', 'length':      2, 'scaf1':  93920, 'strand1':    '+', 'dist1':    383},
        {'from':      7, 'from_side':    'r', 'length':      6, 'scaf1':  45782, 'strand1':    '+', 'dist1':   -230, 'scaf2':  45783, 'strand2':    '+', 'dist2':      0, 'scaf3':  31737, 'strand3':    '+', 'dist3':   1044, 'scaf4':  31758, 'strand4':    '+', 'dist4':     86, 'scaf5':  47750, 'strand5':    '+', 'dist5':    -42},
        {'from':      7, 'from_side':    'r', 'length':      5, 'scaf1':  74998, 'strand1':    '+', 'dist1':   -558, 'scaf2':  32135, 'strand2':    '-', 'dist2':    739, 'scaf3':  45782, 'strand3':    '+', 'dist3':   -502, 'scaf4':  45783, 'strand4':    '+', 'dist4':      0},
        {'from':      7, 'from_side':    'r', 'length':      6, 'scaf1':  74998, 'strand1':    '+', 'dist1':   -558, 'scaf2':  32135, 'strand2':    '-', 'dist2':    739, 'scaf3':  45782, 'strand3':    '+', 'dist3':   -502, 'scaf4':  49372, 'strand4':    '-', 'dist4':    334, 'scaf5':  30446, 'strand5':    '+', 'dist5':   9158},
        {'from':      7, 'from_side':    'r', 'length':      6, 'scaf1':  74998, 'strand1':    '+', 'dist1':   -558, 'scaf2':  32135, 'strand2':    '-', 'dist2':    739, 'scaf3':  45782, 'strand3':    '+', 'dist3':   -502, 'scaf4':  49372, 'strand4':    '-', 'dist4':    334, 'scaf5':  86633, 'strand5':    '-', 'dist5':    432},
        {'from':   1440, 'from_side':    'l', 'length':      5, 'scaf1':  11769, 'strand1':    '-', 'dist1':    -45, 'scaf2':      7, 'strand2':    '+', 'dist2':     76, 'scaf3':  74998, 'strand3':    '+', 'dist3':   -558, 'scaf4':  32135, 'strand4':    '-', 'dist4':    739},
        {'from':   1440, 'from_side':    'r', 'length':      2, 'scaf1': 110006, 'strand1':    '-', 'dist1':    406},
        {'from':   7349, 'from_side':    'l', 'length':      2, 'scaf1': 110006, 'strand1':    '-', 'dist1':    387},
        {'from':   7349, 'from_side':    'r', 'length':      2, 'scaf1':  11769, 'strand1':    '-', 'dist1':    -45},
        {'from':  10945, 'from_side':    'l', 'length':      4, 'scaf1':  95291, 'strand1':    '-', 'dist1':    -45, 'scaf2':  54753, 'strand2':    '-', 'dist2':    -18, 'scaf3':  31108, 'strand3':    '+', 'dist3':      0},
        {'from':  10945, 'from_side':    'r', 'length':      7, 'scaf1': 105853, 'strand1':    '-', 'dist1':    -43, 'scaf2':  32135, 'strand2':    '-', 'dist2':    -45, 'scaf3':  45782, 'strand3':    '+', 'dist3':   -502, 'scaf4':  49372, 'strand4':    '-', 'dist4':    334, 'scaf5':  86633, 'strand5':    '-', 'dist5':    432, 'scaf6':  30446, 'strand6':    '+', 'dist6':    -43},
        {'from':  11769, 'from_side':    'l', 'length':      7, 'scaf1':      7, 'strand1':    '+', 'dist1':     76, 'scaf2':  74998, 'strand2':    '+', 'dist2':   -558, 'scaf3':  32135, 'strand3':    '-', 'dist3':    739, 'scaf4':  45782, 'strand4':    '+', 'dist4':   -502, 'scaf5':  49372, 'strand5':    '-', 'dist5':    334, 'scaf6':  86633, 'strand6':    '-', 'dist6':    432},
        {'from':  11769, 'from_side':    'r', 'length':      3, 'scaf1':   1440, 'strand1':    '+', 'dist1':    -45, 'scaf2': 110006, 'strand2':    '-', 'dist2':    406},
        {'from':  11769, 'from_side':    'r', 'length':      3, 'scaf1':   7349, 'strand1':    '-', 'dist1':    -45, 'scaf2': 110006, 'strand2':    '-', 'dist2':    387},
        {'from':  23515, 'from_side':    'r', 'length':      5, 'scaf1':  95291, 'strand1':    '+', 'dist1':      3, 'scaf2':  29100, 'strand2':    '-', 'dist2':    -43, 'scaf3': 105853, 'strand3':    '-', 'dist3':    -44, 'scaf4':  32420, 'strand4':    '-', 'dist4':    -44},
        {'from':  29100, 'from_side':    'l', 'length':      3, 'scaf1': 105853, 'strand1':    '-', 'dist1':    -44, 'scaf2':  32420, 'strand2':    '-', 'dist2':    -44},
        {'from':  29100, 'from_side':    'r', 'length':      3, 'scaf1':  95291, 'strand1':    '-', 'dist1':    -43, 'scaf2':  23515, 'strand2':    '-', 'dist2':      3},
        {'from':  30446, 'from_side':    'l', 'length':      6, 'scaf1':  49372, 'strand1':    '+', 'dist1':   9158, 'scaf2':  45782, 'strand2':    '-', 'dist2':    334, 'scaf3':  32135, 'strand3':    '+', 'dist3':   -502, 'scaf4':  74998, 'strand4':    '-', 'dist4':    739, 'scaf5':      7, 'strand5':    '-', 'dist5':   -558},
        {'from':  30446, 'from_side':    'l', 'length':      8, 'scaf1':  86633, 'strand1':    '+', 'dist1':    -43, 'scaf2':  49372, 'strand2':    '+', 'dist2':    432, 'scaf3':  45782, 'strand3':    '-', 'dist3':    334, 'scaf4':  32135, 'strand4':    '+', 'dist4':   -502, 'scaf5': 105853, 'strand5':    '+', 'dist5':    -45, 'scaf6':  10945, 'strand6':    '-', 'dist6':    -43, 'scaf7':  95291, 'strand7':    '-', 'dist7':    -45},
        {'from':  30446, 'from_side':    'r', 'length':      4, 'scaf1':  31729, 'strand1':    '+', 'dist1':    630, 'scaf2':  31758, 'strand2':    '+', 'dist2':   -399, 'scaf3':  47750, 'strand3':    '+', 'dist3':    -42},
        {'from':  30446, 'from_side':    'r', 'length':      4, 'scaf1':  31729, 'strand1':    '+', 'dist1':    630, 'scaf2':  31758, 'strand2':    '+', 'dist2':   -399, 'scaf3': 113898, 'strand3':    '+', 'dist3':    686},
        {'from':  30446, 'from_side':    'r', 'length':      5, 'scaf1':  76037, 'strand1':    '-', 'dist1':      0, 'scaf2':  31729, 'strand2':    '+', 'dist2':   -162, 'scaf3':  31758, 'strand3':    '+', 'dist3':   -399, 'scaf4':  47750, 'strand4':    '+', 'dist4':    -42},
        {'from':  30446, 'from_side':    'r', 'length':      5, 'scaf1':  76037, 'strand1':    '-', 'dist1':      0, 'scaf2':  31729, 'strand2':    '+', 'dist2':   -162, 'scaf3':  31758, 'strand3':    '+', 'dist3':   -399, 'scaf4': 113898, 'strand4':    '+', 'dist4':    686},
        {'from':  30446, 'from_side':    'r', 'length':      3, 'scaf1':  76037, 'strand1':    '-', 'dist1':      0, 'scaf2':  31737, 'strand2':    '+', 'dist2':   -168},
        {'from':  31108, 'from_side':    'l', 'length':      8, 'scaf1':  54753, 'strand1':    '+', 'dist1':      0, 'scaf2':  95291, 'strand2':    '+', 'dist2':    -18, 'scaf3':  10945, 'strand3':    '+', 'dist3':    -45, 'scaf4': 105853, 'strand4':    '-', 'dist4':    -43, 'scaf5':  32135, 'strand5':    '-', 'dist5':    -45, 'scaf6':  45782, 'strand6':    '+', 'dist6':   -502, 'scaf7':  49372, 'strand7':    '-', 'dist7':    334},
        {'from':  31108, 'from_side':    'l', 'length':      2, 'scaf1':  93920, 'strand1':    '-', 'dist1':  -4681},
        {'from':  31729, 'from_side':    'l', 'length':      6, 'scaf1':  30446, 'strand1':    '-', 'dist1':    630, 'scaf2':  86633, 'strand2':    '+', 'dist2':    -43, 'scaf3':  49372, 'strand3':    '+', 'dist3':    432, 'scaf4':  45782, 'strand4':    '-', 'dist4':    334, 'scaf5':  32135, 'strand5':    '+', 'dist5':   -502},
        {'from':  31729, 'from_side':    'l', 'length':      5, 'scaf1':  76037, 'strand1':    '+', 'dist1':   -162, 'scaf2':  30446, 'strand2':    '-', 'dist2':      0, 'scaf3':  86633, 'strand3':    '+', 'dist3':    -43, 'scaf4':  49372, 'strand4':    '+', 'dist4':    432},
        {'from':  31729, 'from_side':    'r', 'length':      3, 'scaf1':  31758, 'strand1':    '+', 'dist1':   -399, 'scaf2':  47750, 'strand2':    '+', 'dist2':    -42},
        {'from':  31729, 'from_side':    'r', 'length':      3, 'scaf1':  31758, 'strand1':    '+', 'dist1':   -399, 'scaf2': 113898, 'strand2':    '+', 'dist2':    686},
        {'from':  31737, 'from_side':    'l', 'length':      5, 'scaf1':  45783, 'strand1':    '-', 'dist1':   1044, 'scaf2':  45782, 'strand2':    '-', 'dist2':      0, 'scaf3':      7, 'strand3':    '-', 'dist3':   -230, 'scaf4':  93920, 'strand4':    '+', 'dist4':    383},
        {'from':  31737, 'from_side':    'l', 'length':      3, 'scaf1':  76037, 'strand1':    '+', 'dist1':   -168, 'scaf2':  30446, 'strand2':    '-', 'dist2':      0},
        {'from':  31737, 'from_side':    'r', 'length':      4, 'scaf1':  31758, 'strand1':    '+', 'dist1':     86, 'scaf2':  47750, 'strand2':    '+', 'dist2':    -42, 'scaf3': 113898, 'strand3':    '+', 'dist3':    -10},
        {'from':  31758, 'from_side':    'l', 'length':      7, 'scaf1':  31729, 'strand1':    '-', 'dist1':   -399, 'scaf2':  30446, 'strand2':    '-', 'dist2':    630, 'scaf3':  86633, 'strand3':    '+', 'dist3':    -43, 'scaf4':  49372, 'strand4':    '+', 'dist4':    432, 'scaf5':  45782, 'strand5':    '-', 'dist5':    334, 'scaf6':  32135, 'strand6':    '+', 'dist6':   -502},
        {'from':  31758, 'from_side':    'l', 'length':      6, 'scaf1':  31729, 'strand1':    '-', 'dist1':   -399, 'scaf2':  76037, 'strand2':    '+', 'dist2':   -162, 'scaf3':  30446, 'strand3':    '-', 'dist3':      0, 'scaf4':  86633, 'strand4':    '+', 'dist4':    -43, 'scaf5':  49372, 'strand5':    '+', 'dist5':    432},
        {'from':  31758, 'from_side':    'l', 'length':      6, 'scaf1':  31737, 'strand1':    '-', 'dist1':     86, 'scaf2':  45783, 'strand2':    '-', 'dist2':   1044, 'scaf3':  45782, 'strand3':    '-', 'dist3':      0, 'scaf4':      7, 'strand4':    '-', 'dist4':   -230, 'scaf5':  93920, 'strand5':    '+', 'dist5':    383},
        {'from':  31758, 'from_side':    'l', 'length':      3, 'scaf1':  31737, 'strand1':    '-', 'dist1':     86, 'scaf2':  76037, 'strand2':    '+', 'dist2':   -168},
        {'from':  31758, 'from_side':    'r', 'length':      3, 'scaf1':  47750, 'strand1':    '+', 'dist1':    -42, 'scaf2': 113898, 'strand2':    '+', 'dist2':    -10},
        {'from':  31758, 'from_side':    'r', 'length':      2, 'scaf1': 113898, 'strand1':    '+', 'dist1':    686},
        {'from':  32135, 'from_side':    'l', 'length':      3, 'scaf1':  45782, 'strand1':    '+', 'dist1':   -502, 'scaf2':  45783, 'strand2':    '+', 'dist2':      0},
        {'from':  32135, 'from_side':    'l', 'length':      4, 'scaf1':  45782, 'strand1':    '+', 'dist1':   -502, 'scaf2':  49372, 'strand2':    '-', 'dist2':    334, 'scaf3':  30446, 'strand3':    '+', 'dist3':   9158},
        {'from':  32135, 'from_side':    'l', 'length':      8, 'scaf1':  45782, 'strand1':    '+', 'dist1':   -502, 'scaf2':  49372, 'strand2':    '-', 'dist2':    334, 'scaf3':  86633, 'strand3':    '-', 'dist3':    432, 'scaf4':  30446, 'strand4':    '+', 'dist4':    -43, 'scaf5':  31729, 'strand5':    '+', 'dist5':    630, 'scaf6':  31758, 'strand6':    '+', 'dist6':   -399, 'scaf7': 113898, 'strand7':    '+', 'dist7':    686},
        {'from':  32135, 'from_side':    'r', 'length':      6, 'scaf1':  74998, 'strand1':    '-', 'dist1':    739, 'scaf2':      7, 'strand2':    '-', 'dist2':   -558, 'scaf3':  11769, 'strand3':    '+', 'dist3':     76, 'scaf4':   1440, 'strand4':    '+', 'dist4':    -45, 'scaf5': 110006, 'strand5':    '-', 'dist5':    406},
        {'from':  32135, 'from_side':    'r', 'length':      4, 'scaf1':  74998, 'strand1':    '-', 'dist1':    739, 'scaf2':      7, 'strand2':    '-', 'dist2':   -558, 'scaf3':  93920, 'strand3':    '+', 'dist3':    383},
        {'from':  32135, 'from_side':    'r', 'length':      6, 'scaf1': 105853, 'strand1':    '+', 'dist1':    -45, 'scaf2':  10945, 'strand2':    '-', 'dist2':    -43, 'scaf3':  95291, 'strand3':    '-', 'dist3':    -45, 'scaf4':  54753, 'strand4':    '-', 'dist4':    -18, 'scaf5':  31108, 'strand5':    '+', 'dist5':      0},
        {'from':  32420, 'from_side':    'r', 'length':      5, 'scaf1': 105853, 'strand1':    '+', 'dist1':    -44, 'scaf2':  29100, 'strand2':    '+', 'dist2':    -44, 'scaf3':  95291, 'strand3':    '-', 'dist3':    -43, 'scaf4':  23515, 'strand4':    '-', 'dist4':      3},
        {'from':  45782, 'from_side':    'l', 'length':      3, 'scaf1':      7, 'strand1':    '-', 'dist1':   -230, 'scaf2':  93920, 'strand2':    '+', 'dist2':    383},
        {'from':  45782, 'from_side':    'l', 'length':      5, 'scaf1':  32135, 'strand1':    '+', 'dist1':   -502, 'scaf2':  74998, 'strand2':    '-', 'dist2':    739, 'scaf3':      7, 'strand3':    '-', 'dist3':   -558, 'scaf4':  11769, 'strand4':    '+', 'dist4':     76},
        {'from':  45782, 'from_side':    'l', 'length':      5, 'scaf1':  32135, 'strand1':    '+', 'dist1':   -502, 'scaf2':  74998, 'strand2':    '-', 'dist2':    739, 'scaf3':      7, 'strand3':    '-', 'dist3':   -558, 'scaf4':  93920, 'strand4':    '+', 'dist4':    383},
        {'from':  45782, 'from_side':    'l', 'length':      7, 'scaf1':  32135, 'strand1':    '+', 'dist1':   -502, 'scaf2': 105853, 'strand2':    '+', 'dist2':    -45, 'scaf3':  10945, 'strand3':    '-', 'dist3':    -43, 'scaf4':  95291, 'strand4':    '-', 'dist4':    -45, 'scaf5':  54753, 'strand5':    '-', 'dist5':    -18, 'scaf6':  31108, 'strand6':    '+', 'dist6':      0},
        {'from':  45782, 'from_side':    'r', 'length':      5, 'scaf1':  45783, 'strand1':    '+', 'dist1':      0, 'scaf2':  31737, 'strand2':    '+', 'dist2':   1044, 'scaf3':  31758, 'strand3':    '+', 'dist3':     86, 'scaf4':  47750, 'strand4':    '+', 'dist4':    -42},
        {'from':  45782, 'from_side':    'r', 'length':      3, 'scaf1':  49372, 'strand1':    '-', 'dist1':    334, 'scaf2':  30446, 'strand2':    '+', 'dist2':   9158},
        {'from':  45782, 'from_side':    'r', 'length':      7, 'scaf1':  49372, 'strand1':    '-', 'dist1':    334, 'scaf2':  86633, 'strand2':    '-', 'dist2':    432, 'scaf3':  30446, 'strand3':    '+', 'dist3':    -43, 'scaf4':  31729, 'strand4':    '+', 'dist4':    630, 'scaf5':  31758, 'strand5':    '+', 'dist5':   -399, 'scaf6': 113898, 'strand6':    '+', 'dist6':    686},
        {'from':  45783, 'from_side':    'l', 'length':      4, 'scaf1':  45782, 'strand1':    '-', 'dist1':      0, 'scaf2':      7, 'strand2':    '-', 'dist2':   -230, 'scaf3':  93920, 'strand3':    '+', 'dist3':    383},
        {'from':  45783, 'from_side':    'l', 'length':      6, 'scaf1':  45782, 'strand1':    '-', 'dist1':      0, 'scaf2':  32135, 'strand2':    '+', 'dist2':   -502, 'scaf3':  74998, 'strand3':    '-', 'dist3':    739, 'scaf4':      7, 'strand4':    '-', 'dist4':   -558, 'scaf5':  93920, 'strand5':    '+', 'dist5':    383},
        {'from':  45783, 'from_side':    'r', 'length':      5, 'scaf1':  31737, 'strand1':    '+', 'dist1':   1044, 'scaf2':  31758, 'strand2':    '+', 'dist2':     86, 'scaf3':  47750, 'strand3':    '+', 'dist3':    -42, 'scaf4': 113898, 'strand4':    '+', 'dist4':    -10},
        {'from':  47750, 'from_side':    'l', 'length':      4, 'scaf1':  31758, 'strand1':    '-', 'dist1':    -42, 'scaf2':  31729, 'strand2':    '-', 'dist2':   -399, 'scaf3':  30446, 'strand3':    '-', 'dist3':    630},
        {'from':  47750, 'from_side':    'l', 'length':      5, 'scaf1':  31758, 'strand1':    '-', 'dist1':    -42, 'scaf2':  31729, 'strand2':    '-', 'dist2':   -399, 'scaf3':  76037, 'strand3':    '+', 'dist3':   -162, 'scaf4':  30446, 'strand4':    '-', 'dist4':      0},
        {'from':  47750, 'from_side':    'l', 'length':      7, 'scaf1':  31758, 'strand1':    '-', 'dist1':    -42, 'scaf2':  31737, 'strand2':    '-', 'dist2':     86, 'scaf3':  45783, 'strand3':    '-', 'dist3':   1044, 'scaf4':  45782, 'strand4':    '-', 'dist4':      0, 'scaf5':      7, 'strand5':    '-', 'dist5':   -230, 'scaf6':  93920, 'strand6':    '+', 'dist6':    383},
        {'from':  47750, 'from_side':    'l', 'length':      4, 'scaf1':  31758, 'strand1':    '-', 'dist1':    -42, 'scaf2':  31737, 'strand2':    '-', 'dist2':     86, 'scaf3':  76037, 'strand3':    '+', 'dist3':   -168},
        {'from':  47750, 'from_side':    'r', 'length':      2, 'scaf1': 113898, 'strand1':    '+', 'dist1':    -10},
        {'from':  49372, 'from_side':    'l', 'length':      2, 'scaf1':  30446, 'strand1':    '+', 'dist1':   9158},
        {'from':  49372, 'from_side':    'l', 'length':      6, 'scaf1':  86633, 'strand1':    '-', 'dist1':    432, 'scaf2':  30446, 'strand2':    '+', 'dist2':    -43, 'scaf3':  31729, 'strand3':    '+', 'dist3':    630, 'scaf4':  31758, 'strand4':    '+', 'dist4':   -399, 'scaf5': 113898, 'strand5':    '+', 'dist5':    686},
        {'from':  49372, 'from_side':    'l', 'length':      6, 'scaf1':  86633, 'strand1':    '-', 'dist1':    432, 'scaf2':  30446, 'strand2':    '+', 'dist2':    -43, 'scaf3':  76037, 'strand3':    '-', 'dist3':      0, 'scaf4':  31729, 'strand4':    '+', 'dist4':   -162, 'scaf5':  31758, 'strand5':    '+', 'dist5':   -399},
        {'from':  49372, 'from_side':    'r', 'length':      6, 'scaf1':  45782, 'strand1':    '-', 'dist1':    334, 'scaf2':  32135, 'strand2':    '+', 'dist2':   -502, 'scaf3':  74998, 'strand3':    '-', 'dist3':    739, 'scaf4':      7, 'strand4':    '-', 'dist4':   -558, 'scaf5':  11769, 'strand5':    '+', 'dist5':     76},
        {'from':  49372, 'from_side':    'r', 'length':      8, 'scaf1':  45782, 'strand1':    '-', 'dist1':    334, 'scaf2':  32135, 'strand2':    '+', 'dist2':   -502, 'scaf3': 105853, 'strand3':    '+', 'dist3':    -45, 'scaf4':  10945, 'strand4':    '-', 'dist4':    -43, 'scaf5':  95291, 'strand5':    '-', 'dist5':    -45, 'scaf6':  54753, 'strand6':    '-', 'dist6':    -18, 'scaf7':  31108, 'strand7':    '+', 'dist7':      0},
        {'from':  54753, 'from_side':    'l', 'length':      2, 'scaf1':  31108, 'strand1':    '+', 'dist1':      0},
        {'from':  54753, 'from_side':    'r', 'length':      7, 'scaf1':  95291, 'strand1':    '+', 'dist1':    -18, 'scaf2':  10945, 'strand2':    '+', 'dist2':    -45, 'scaf3': 105853, 'strand3':    '-', 'dist3':    -43, 'scaf4':  32135, 'strand4':    '-', 'dist4':    -45, 'scaf5':  45782, 'strand5':    '+', 'dist5':   -502, 'scaf6':  49372, 'strand6':    '-', 'dist6':    334},
        {'from':  74998, 'from_side':    'l', 'length':      5, 'scaf1':      7, 'strand1':    '-', 'dist1':   -558, 'scaf2':  11769, 'strand2':    '+', 'dist2':     76, 'scaf3':   1440, 'strand3':    '+', 'dist3':    -45, 'scaf4': 110006, 'strand4':    '-', 'dist4':    406},
        {'from':  74998, 'from_side':    'l', 'length':      3, 'scaf1':      7, 'strand1':    '-', 'dist1':   -558, 'scaf2':  93920, 'strand2':    '+', 'dist2':    383},
        {'from':  74998, 'from_side':    'r', 'length':      4, 'scaf1':  32135, 'strand1':    '-', 'dist1':    739, 'scaf2':  45782, 'strand2':    '+', 'dist2':   -502, 'scaf3':  45783, 'strand3':    '+', 'dist3':      0},
        {'from':  74998, 'from_side':    'r', 'length':      5, 'scaf1':  32135, 'strand1':    '-', 'dist1':    739, 'scaf2':  45782, 'strand2':    '+', 'dist2':   -502, 'scaf3':  49372, 'strand3':    '-', 'dist3':    334, 'scaf4':  30446, 'strand4':    '+', 'dist4':   9158},
        {'from':  74998, 'from_side':    'r', 'length':      5, 'scaf1':  32135, 'strand1':    '-', 'dist1':    739, 'scaf2':  45782, 'strand2':    '+', 'dist2':   -502, 'scaf3':  49372, 'strand3':    '-', 'dist3':    334, 'scaf4':  86633, 'strand4':    '-', 'dist4':    432},
        {'from':  76037, 'from_side':    'l', 'length':      4, 'scaf1':  31729, 'strand1':    '+', 'dist1':   -162, 'scaf2':  31758, 'strand2':    '+', 'dist2':   -399, 'scaf3':  47750, 'strand3':    '+', 'dist3':    -42},
        {'from':  76037, 'from_side':    'l', 'length':      4, 'scaf1':  31729, 'strand1':    '+', 'dist1':   -162, 'scaf2':  31758, 'strand2':    '+', 'dist2':   -399, 'scaf3': 113898, 'strand3':    '+', 'dist3':    686},
        {'from':  76037, 'from_side':    'l', 'length':      5, 'scaf1':  31737, 'strand1':    '+', 'dist1':   -168, 'scaf2':  31758, 'strand2':    '+', 'dist2':     86, 'scaf3':  47750, 'strand3':    '+', 'dist3':    -42, 'scaf4': 113898, 'strand4':    '+', 'dist4':    -10},
        {'from':  76037, 'from_side':    'r', 'length':      4, 'scaf1':  30446, 'strand1':    '-', 'dist1':      0, 'scaf2':  86633, 'strand2':    '+', 'dist2':    -43, 'scaf3':  49372, 'strand3':    '+', 'dist3':    432},
        {'from':  86633, 'from_side':    'l', 'length':      5, 'scaf1':  30446, 'strand1':    '+', 'dist1':    -43, 'scaf2':  31729, 'strand2':    '+', 'dist2':    630, 'scaf3':  31758, 'strand3':    '+', 'dist3':   -399, 'scaf4': 113898, 'strand4':    '+', 'dist4':    686},
        {'from':  86633, 'from_side':    'l', 'length':      6, 'scaf1':  30446, 'strand1':    '+', 'dist1':    -43, 'scaf2':  76037, 'strand2':    '-', 'dist2':      0, 'scaf3':  31729, 'strand3':    '+', 'dist3':   -162, 'scaf4':  31758, 'strand4':    '+', 'dist4':   -399, 'scaf5': 113898, 'strand5':    '+', 'dist5':    686},
        {'from':  86633, 'from_side':    'r', 'length':      7, 'scaf1':  49372, 'strand1':    '+', 'dist1':    432, 'scaf2':  45782, 'strand2':    '-', 'dist2':    334, 'scaf3':  32135, 'strand3':    '+', 'dist3':   -502, 'scaf4':  74998, 'strand4':    '-', 'dist4':    739, 'scaf5':      7, 'strand5':    '-', 'dist5':   -558, 'scaf6':  11769, 'strand6':    '+', 'dist6':     76},
        {'from':  86633, 'from_side':    'r', 'length':      7, 'scaf1':  49372, 'strand1':    '+', 'dist1':    432, 'scaf2':  45782, 'strand2':    '-', 'dist2':    334, 'scaf3':  32135, 'strand3':    '+', 'dist3':   -502, 'scaf4': 105853, 'strand4':    '+', 'dist4':    -45, 'scaf5':  10945, 'strand5':    '-', 'dist5':    -43, 'scaf6':  95291, 'strand6':    '-', 'dist6':    -45},
        {'from':  93920, 'from_side':    'l', 'length':      7, 'scaf1':      7, 'strand1':    '+', 'dist1':    383, 'scaf2':  45782, 'strand2':    '+', 'dist2':   -230, 'scaf3':  45783, 'strand3':    '+', 'dist3':      0, 'scaf4':  31737, 'strand4':    '+', 'dist4':   1044, 'scaf5':  31758, 'strand5':    '+', 'dist5':     86, 'scaf6':  47750, 'strand6':    '+', 'dist6':    -42},
        {'from':  93920, 'from_side':    'l', 'length':      6, 'scaf1':      7, 'strand1':    '+', 'dist1':    383, 'scaf2':  74998, 'strand2':    '+', 'dist2':   -558, 'scaf3':  32135, 'strand3':    '-', 'dist3':    739, 'scaf4':  45782, 'strand4':    '+', 'dist4':   -502, 'scaf5':  45783, 'strand5':    '+', 'dist5':      0},
        {'from':  93920, 'from_side':    'r', 'length':      2, 'scaf1':  31108, 'strand1':    '+', 'dist1':  -4681},
        {'from':  95291, 'from_side':    'l', 'length':      2, 'scaf1':  23515, 'strand1':    '-', 'dist1':      3},
        {'from':  95291, 'from_side':    'l', 'length':      3, 'scaf1':  54753, 'strand1':    '-', 'dist1':    -18, 'scaf2':  31108, 'strand2':    '+', 'dist2':      0},
        {'from':  95291, 'from_side':    'r', 'length':      8, 'scaf1':  10945, 'strand1':    '+', 'dist1':    -45, 'scaf2': 105853, 'strand2':    '-', 'dist2':    -43, 'scaf3':  32135, 'strand3':    '-', 'dist3':    -45, 'scaf4':  45782, 'strand4':    '+', 'dist4':   -502, 'scaf5':  49372, 'strand5':    '-', 'dist5':    334, 'scaf6':  86633, 'strand6':    '-', 'dist6':    432, 'scaf7':  30446, 'strand7':    '+', 'dist7':    -43},
        {'from':  95291, 'from_side':    'r', 'length':      4, 'scaf1':  29100, 'strand1':    '-', 'dist1':    -43, 'scaf2': 105853, 'strand2':    '-', 'dist2':    -44, 'scaf3':  32420, 'strand3':    '-', 'dist3':    -44},
        {'from': 105853, 'from_side':    'l', 'length':      6, 'scaf1':  32135, 'strand1':    '-', 'dist1':    -45, 'scaf2':  45782, 'strand2':    '+', 'dist2':   -502, 'scaf3':  49372, 'strand3':    '-', 'dist3':    334, 'scaf4':  86633, 'strand4':    '-', 'dist4':    432, 'scaf5':  30446, 'strand5':    '+', 'dist5':    -43},
        {'from': 105853, 'from_side':    'l', 'length':      2, 'scaf1':  32420, 'strand1':    '-', 'dist1':    -44},
        {'from': 105853, 'from_side':    'r', 'length':      5, 'scaf1':  10945, 'strand1':    '-', 'dist1':    -43, 'scaf2':  95291, 'strand2':    '-', 'dist2':    -45, 'scaf3':  54753, 'strand3':    '-', 'dist3':    -18, 'scaf4':  31108, 'strand4':    '+', 'dist4':      0},
        {'from': 105853, 'from_side':    'r', 'length':      4, 'scaf1':  29100, 'strand1':    '+', 'dist1':    -44, 'scaf2':  95291, 'strand2':    '-', 'dist2':    -43, 'scaf3':  23515, 'strand3':    '-', 'dist3':      3},
        {'from': 110006, 'from_side':    'r', 'length':      6, 'scaf1':   1440, 'strand1':    '-', 'dist1':    406, 'scaf2':  11769, 'strand2':    '-', 'dist2':    -45, 'scaf3':      7, 'strand3':    '+', 'dist3':     76, 'scaf4':  74998, 'strand4':    '+', 'dist4':   -558, 'scaf5':  32135, 'strand5':    '-', 'dist5':    739},
        {'from': 110006, 'from_side':    'r', 'length':      3, 'scaf1':   7349, 'strand1':    '+', 'dist1':    387, 'scaf2':  11769, 'strand2':    '-', 'dist2':    -45},
        {'from': 113898, 'from_side':    'l', 'length':      8, 'scaf1':  31758, 'strand1':    '-', 'dist1':    686, 'scaf2':  31729, 'strand2':    '-', 'dist2':   -399, 'scaf3':  30446, 'strand3':    '-', 'dist3':    630, 'scaf4':  86633, 'strand4':    '+', 'dist4':    -43, 'scaf5':  49372, 'strand5':    '+', 'dist5':    432, 'scaf6':  45782, 'strand6':    '-', 'dist6':    334, 'scaf7':  32135, 'strand7':    '+', 'dist7':   -502},
        {'from': 113898, 'from_side':    'l', 'length':      6, 'scaf1':  31758, 'strand1':    '-', 'dist1':    686, 'scaf2':  31729, 'strand2':    '-', 'dist2':   -399, 'scaf3':  76037, 'strand3':    '+', 'dist3':   -162, 'scaf4':  30446, 'strand4':    '-', 'dist4':      0, 'scaf5':  86633, 'strand5':    '+', 'dist5':    -43},
        {'from': 113898, 'from_side':    'l', 'length':      5, 'scaf1':  47750, 'strand1':    '-', 'dist1':    -10, 'scaf2':  31758, 'strand2':    '-', 'dist2':    -42, 'scaf3':  31737, 'strand3':    '-', 'dist3':     86, 'scaf4':  45783, 'strand4':    '-', 'dist4':   1044},
        {'from': 113898, 'from_side':    'l', 'length':      5, 'scaf1':  47750, 'strand1':    '-', 'dist1':    -10, 'scaf2':  31758, 'strand2':    '-', 'dist2':    -42, 'scaf3':  31737, 'strand3':    '-', 'dist3':     86, 'scaf4':  76037, 'strand4':    '+', 'dist4':   -168},
        ]) )
    scaf_bridges.append( pd.DataFrame([
        {'from':      7, 'from_side':    'l', 'to':  11769, 'to_side':    'l', 'mean_dist':     76, 'mapq':  60060, 'bcount':     16, 'min_dist':     67, 'max_dist':     90, 'probability': 0.159802, 'to_alt':      1, 'from_alt':      2},
        {'from':      7, 'from_side':    'l', 'to':  93920, 'to_side':    'l', 'mean_dist':    383, 'mapq':  60060, 'bcount':     38, 'min_dist':    364, 'max_dist':    425, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from':      7, 'from_side':    'r', 'to':  45782, 'to_side':    'l', 'mean_dist':   -230, 'mapq':  60060, 'bcount':     42, 'min_dist':   -253, 'max_dist':   -155, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      2},
        {'from':      7, 'from_side':    'r', 'to':  74998, 'to_side':    'l', 'mean_dist':   -558, 'mapq':  60060, 'bcount':     40, 'min_dist':   -659, 'max_dist':   -401, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from':   1440, 'from_side':    'l', 'to':  11769, 'to_side':    'r', 'mean_dist':    -45, 'mapq':  60060, 'bcount':     13, 'min_dist':    -54, 'max_dist':    -37, 'probability': 0.082422, 'to_alt':      2, 'from_alt':      1},
        {'from':   1440, 'from_side':    'r', 'to': 110006, 'to_side':    'r', 'mean_dist':    406, 'mapq':  60060, 'bcount':     11, 'min_dist':    386, 'max_dist':    433, 'probability': 0.081320, 'to_alt':      2, 'from_alt':      1},
        {'from':   7349, 'from_side':    'l', 'to': 110006, 'to_side':    'r', 'mean_dist':    387, 'mapq':  60060, 'bcount':      6, 'min_dist':    371, 'max_dist':    408, 'probability': 0.016554, 'to_alt':      2, 'from_alt':      1},
        {'from':   7349, 'from_side':    'r', 'to':  11769, 'to_side':    'r', 'mean_dist':    -45, 'mapq':  60060, 'bcount':      8, 'min_dist':    -51, 'max_dist':    -39, 'probability': 0.020422, 'to_alt':      2, 'from_alt':      1},
        {'from':  10945, 'from_side':    'l', 'to':  95291, 'to_side':    'r', 'mean_dist':    -45, 'mapq':  60060, 'bcount':     16, 'min_dist':    -49, 'max_dist':    -40, 'probability': 0.159802, 'to_alt':      2, 'from_alt':      1},
        {'from':  10945, 'from_side':    'r', 'to': 105853, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     15, 'min_dist':    -49, 'max_dist':    -15, 'probability': 0.129976, 'to_alt':      2, 'from_alt':      1},
        {'from':  11769, 'from_side':    'l', 'to':      7, 'to_side':    'l', 'mean_dist':     76, 'mapq':  60060, 'bcount':     16, 'min_dist':     67, 'max_dist':     90, 'probability': 0.159802, 'to_alt':      2, 'from_alt':      1},
        {'from':  11769, 'from_side':    'r', 'to':   1440, 'to_side':    'l', 'mean_dist':    -45, 'mapq':  60060, 'bcount':     13, 'min_dist':    -54, 'max_dist':    -37, 'probability': 0.082422, 'to_alt':      1, 'from_alt':      2},
        {'from':  11769, 'from_side':    'r', 'to':   7349, 'to_side':    'r', 'mean_dist':    -45, 'mapq':  60060, 'bcount':      8, 'min_dist':    -51, 'max_dist':    -39, 'probability': 0.020422, 'to_alt':      1, 'from_alt':      2},
        {'from':  23515, 'from_side':    'r', 'to':  95291, 'to_side':    'l', 'mean_dist':      3, 'mapq':  60060, 'bcount':     13, 'min_dist':      1, 'max_dist':      8, 'probability': 0.082422, 'to_alt':      2, 'from_alt':      1},
        {'from':  29100, 'from_side':    'l', 'to': 105853, 'to_side':    'r', 'mean_dist':    -44, 'mapq':  60060, 'bcount':     17, 'min_dist':    -48, 'max_dist':    -38, 'probability': 0.193782, 'to_alt':      2, 'from_alt':      1},
        {'from':  29100, 'from_side':    'r', 'to':  95291, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     14, 'min_dist':    -50, 'max_dist':    -34, 'probability': 0.104244, 'to_alt':      2, 'from_alt':      1},
        {'from':  30446, 'from_side':    'l', 'to':  49372, 'to_side':    'l', 'mean_dist':   9158, 'mapq':  60060, 'bcount':      2, 'min_dist':   8995, 'max_dist':   9322, 'probability': 0.096053, 'to_alt':      3, 'from_alt':      2},
        {'from':  30446, 'from_side':    'l', 'to':  86633, 'to_side':    'l', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     30, 'min_dist':    -50, 'max_dist':    -32, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from':  30446, 'from_side':    'r', 'to':  31729, 'to_side':    'l', 'mean_dist':    630, 'mapq':  60060, 'bcount':     17, 'min_dist':    586, 'max_dist':    882, 'probability': 0.303341, 'to_alt':      2, 'from_alt':      2},
        {'from':  30446, 'from_side':    'r', 'to':  76037, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':     27, 'min_dist':      0, 'max_dist':      0, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from':  31108, 'from_side':    'l', 'to':  54753, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     14, 'min_dist':      0, 'max_dist':      0, 'probability': 0.104244, 'to_alt':      2, 'from_alt':      1},
        {'from':  31108, 'from_side':    'l', 'to':  93920, 'to_side':    'r', 'mean_dist':  -4681, 'mapq':  60060, 'bcount':      6, 'min_dist':  -4716, 'max_dist':  -4577, 'probability': 0.108994, 'to_alt':      2, 'from_alt':      1},
        {'from':  31729, 'from_side':    'l', 'to':  30446, 'to_side':    'r', 'mean_dist':    630, 'mapq':  60060, 'bcount':     17, 'min_dist':    586, 'max_dist':    882, 'probability': 0.303341, 'to_alt':      2, 'from_alt':      2},
        {'from':  31729, 'from_side':    'l', 'to':  76037, 'to_side':    'l', 'mean_dist':   -162, 'mapq':  60060, 'bcount':     14, 'min_dist':   -174, 'max_dist':   -147, 'probability': 0.104244, 'to_alt':      2, 'from_alt':      2},
        {'from':  31729, 'from_side':    'r', 'to':  31758, 'to_side':    'l', 'mean_dist':   -399, 'mapq':  60060, 'bcount':     21, 'min_dist':   -463, 'max_dist':   -335, 'probability': 0.367256, 'to_alt':      2, 'from_alt':      1},
        {'from':  31737, 'from_side':    'l', 'to':  45783, 'to_side':    'r', 'mean_dist':   1044, 'mapq':  60060, 'bcount':      9, 'min_dist':    996, 'max_dist':   1138, 'probability': 0.045509, 'to_alt':      1, 'from_alt':      2},
        {'from':  31737, 'from_side':    'l', 'to':  76037, 'to_side':    'l', 'mean_dist':   -168, 'mapq':  60060, 'bcount':      9, 'min_dist':   -186, 'max_dist':   -143, 'probability': 0.027818, 'to_alt':      2, 'from_alt':      2},
        {'from':  31737, 'from_side':    'r', 'to':  31758, 'to_side':    'l', 'mean_dist':     86, 'mapq':  60060, 'bcount':     18, 'min_dist':     69, 'max_dist':    120, 'probability': 0.231835, 'to_alt':      2, 'from_alt':      1},
        {'from':  31758, 'from_side':    'l', 'to':  31729, 'to_side':    'r', 'mean_dist':   -399, 'mapq':  60060, 'bcount':     21, 'min_dist':   -463, 'max_dist':   -335, 'probability': 0.367256, 'to_alt':      1, 'from_alt':      2},
        {'from':  31758, 'from_side':    'l', 'to':  31737, 'to_side':    'r', 'mean_dist':     86, 'mapq':  60060, 'bcount':     18, 'min_dist':     69, 'max_dist':    120, 'probability': 0.231835, 'to_alt':      1, 'from_alt':      2},
        {'from':  31758, 'from_side':    'r', 'to':  47750, 'to_side':    'l', 'mean_dist':    -42, 'mapq':  60060, 'bcount':     47, 'min_dist':    -49, 'max_dist':    -17, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from':  31758, 'from_side':    'r', 'to': 113898, 'to_side':    'l', 'mean_dist':    686, 'mapq':  60060, 'bcount':     20, 'min_dist':    648, 'max_dist':    763, 'probability': 0.470466, 'to_alt':      2, 'from_alt':      2},
        {'from':  32135, 'from_side':    'l', 'to':  45782, 'to_side':    'l', 'mean_dist':   -502, 'mapq':  60060, 'bcount':     42, 'min_dist':   -560, 'max_dist':   -395, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  32135, 'from_side':    'r', 'to':  74998, 'to_side':    'r', 'mean_dist':    739, 'mapq':  60060, 'bcount':     24, 'min_dist':    660, 'max_dist':    984, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from':  32135, 'from_side':    'r', 'to': 105853, 'to_side':    'l', 'mean_dist':    -45, 'mapq':  60060, 'bcount':     21, 'min_dist':    -50, 'max_dist':    -34, 'probability': 0.367256, 'to_alt':      2, 'from_alt':      2},
        {'from':  32420, 'from_side':    'r', 'to': 105853, 'to_side':    'l', 'mean_dist':    -44, 'mapq':  60060, 'bcount':     18, 'min_dist':    -50, 'max_dist':    -32, 'probability': 0.231835, 'to_alt':      2, 'from_alt':      1},
        {'from':  45782, 'from_side':    'l', 'to':      7, 'to_side':    'r', 'mean_dist':   -230, 'mapq':  60060, 'bcount':     42, 'min_dist':   -253, 'max_dist':   -155, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      2},
        {'from':  45782, 'from_side':    'l', 'to':  32135, 'to_side':    'l', 'mean_dist':   -502, 'mapq':  60060, 'bcount':     42, 'min_dist':   -560, 'max_dist':   -395, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from':  45782, 'from_side':    'r', 'to':  45783, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     43, 'min_dist':      0, 'max_dist':      0, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from':  45782, 'from_side':    'r', 'to':  49372, 'to_side':    'r', 'mean_dist':    334, 'mapq':  60060, 'bcount':     35, 'min_dist':    308, 'max_dist':    376, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2},
        {'from':  45783, 'from_side':    'l', 'to':  45782, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':     43, 'min_dist':      0, 'max_dist':      0, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  45783, 'from_side':    'r', 'to':  31737, 'to_side':    'l', 'mean_dist':   1044, 'mapq':  60060, 'bcount':      9, 'min_dist':    996, 'max_dist':   1138, 'probability': 0.045509, 'to_alt':      2, 'from_alt':      1},
        {'from':  47750, 'from_side':    'l', 'to':  31758, 'to_side':    'r', 'mean_dist':    -42, 'mapq':  60060, 'bcount':     47, 'min_dist':    -49, 'max_dist':    -17, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  47750, 'from_side':    'r', 'to': 113898, 'to_side':    'l', 'mean_dist':    -10, 'mapq':  60060, 'bcount':     31, 'min_dist':    -33, 'max_dist':     40, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  49372, 'from_side':    'l', 'to':  30446, 'to_side':    'l', 'mean_dist':   9158, 'mapq':  60060, 'bcount':      2, 'min_dist':   8995, 'max_dist':   9322, 'probability': 0.096053, 'to_alt':      2, 'from_alt':      3},
        {'from':  49372, 'from_side':    'l', 'to':  86633, 'to_side':    'r', 'mean_dist':    432, 'mapq':  60060, 'bcount':     18, 'min_dist':    409, 'max_dist':    461, 'probability': 0.356470, 'to_alt':      1, 'from_alt':      3},
        {'from':  49372, 'from_side':    'r', 'to':  45782, 'to_side':    'r', 'mean_dist':    334, 'mapq':  60060, 'bcount':     35, 'min_dist':    308, 'max_dist':    376, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  54753, 'from_side':    'l', 'to':  31108, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     14, 'min_dist':      0, 'max_dist':      0, 'probability': 0.104244, 'to_alt':      2, 'from_alt':      1},
        {'from':  54753, 'from_side':    'r', 'to':  95291, 'to_side':    'l', 'mean_dist':    -18, 'mapq':  60060, 'bcount':     16, 'min_dist':    -22, 'max_dist':    -14, 'probability': 0.159802, 'to_alt':      2, 'from_alt':      1},
        {'from':  74998, 'from_side':    'l', 'to':      7, 'to_side':    'r', 'mean_dist':   -558, 'mapq':  60060, 'bcount':     40, 'min_dist':   -659, 'max_dist':   -401, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  74998, 'from_side':    'r', 'to':  32135, 'to_side':    'r', 'mean_dist':    739, 'mapq':  60060, 'bcount':     24, 'min_dist':    660, 'max_dist':    984, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  76037, 'from_side':    'l', 'to':  31729, 'to_side':    'l', 'mean_dist':   -162, 'mapq':  60060, 'bcount':     14, 'min_dist':   -174, 'max_dist':   -147, 'probability': 0.104244, 'to_alt':      2, 'from_alt':      2},
        {'from':  76037, 'from_side':    'l', 'to':  31737, 'to_side':    'l', 'mean_dist':   -168, 'mapq':  60060, 'bcount':      9, 'min_dist':   -186, 'max_dist':   -143, 'probability': 0.027818, 'to_alt':      2, 'from_alt':      2},
        {'from':  76037, 'from_side':    'r', 'to':  30446, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':     27, 'min_dist':      0, 'max_dist':      0, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  86633, 'from_side':    'l', 'to':  30446, 'to_side':    'l', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     30, 'min_dist':    -50, 'max_dist':    -32, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  86633, 'from_side':    'r', 'to':  49372, 'to_side':    'l', 'mean_dist':    432, 'mapq':  60060, 'bcount':     18, 'min_dist':    409, 'max_dist':    461, 'probability': 0.356470, 'to_alt':      3, 'from_alt':      1},
        {'from':  93920, 'from_side':    'l', 'to':      7, 'to_side':    'l', 'mean_dist':    383, 'mapq':  60060, 'bcount':     38, 'min_dist':    364, 'max_dist':    425, 'probability': 0.500000, 'to_alt':      2, 'from_alt':      1},
        {'from':  93920, 'from_side':    'r', 'to':  31108, 'to_side':    'l', 'mean_dist':  -4681, 'mapq':  60060, 'bcount':      6, 'min_dist':  -4716, 'max_dist':  -4577, 'probability': 0.108994, 'to_alt':      2, 'from_alt':      1},
        {'from':  95291, 'from_side':    'l', 'to':  23515, 'to_side':    'r', 'mean_dist':      3, 'mapq':  60060, 'bcount':     13, 'min_dist':      1, 'max_dist':      8, 'probability': 0.082422, 'to_alt':      1, 'from_alt':      2},
        {'from':  95291, 'from_side':    'l', 'to':  54753, 'to_side':    'r', 'mean_dist':    -18, 'mapq':  60060, 'bcount':     16, 'min_dist':    -22, 'max_dist':    -14, 'probability': 0.159802, 'to_alt':      1, 'from_alt':      2},
        {'from':  95291, 'from_side':    'r', 'to':  10945, 'to_side':    'l', 'mean_dist':    -45, 'mapq':  60060, 'bcount':     16, 'min_dist':    -49, 'max_dist':    -40, 'probability': 0.159802, 'to_alt':      1, 'from_alt':      2},
        {'from':  95291, 'from_side':    'r', 'to':  29100, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     14, 'min_dist':    -50, 'max_dist':    -34, 'probability': 0.104244, 'to_alt':      1, 'from_alt':      2},
        {'from': 105853, 'from_side':    'l', 'to':  32135, 'to_side':    'r', 'mean_dist':    -45, 'mapq':  60060, 'bcount':     21, 'min_dist':    -50, 'max_dist':    -34, 'probability': 0.367256, 'to_alt':      2, 'from_alt':      2},
        {'from': 105853, 'from_side':    'l', 'to':  32420, 'to_side':    'r', 'mean_dist':    -44, 'mapq':  60060, 'bcount':     18, 'min_dist':    -50, 'max_dist':    -32, 'probability': 0.231835, 'to_alt':      1, 'from_alt':      2},
        {'from': 105853, 'from_side':    'r', 'to':  10945, 'to_side':    'r', 'mean_dist':    -43, 'mapq':  60060, 'bcount':     15, 'min_dist':    -49, 'max_dist':    -15, 'probability': 0.129976, 'to_alt':      1, 'from_alt':      2},
        {'from': 105853, 'from_side':    'r', 'to':  29100, 'to_side':    'l', 'mean_dist':    -44, 'mapq':  60060, 'bcount':     17, 'min_dist':    -48, 'max_dist':    -38, 'probability': 0.193782, 'to_alt':      1, 'from_alt':      2},
        {'from': 110006, 'from_side':    'r', 'to':   1440, 'to_side':    'r', 'mean_dist':    406, 'mapq':  60060, 'bcount':     11, 'min_dist':    386, 'max_dist':    433, 'probability': 0.081320, 'to_alt':      1, 'from_alt':      2},
        {'from': 110006, 'from_side':    'r', 'to':   7349, 'to_side':    'l', 'mean_dist':    387, 'mapq':  60060, 'bcount':      6, 'min_dist':    371, 'max_dist':    408, 'probability': 0.016554, 'to_alt':      1, 'from_alt':      2},
        {'from': 113898, 'from_side':    'l', 'to':  31758, 'to_side':    'r', 'mean_dist':    686, 'mapq':  60060, 'bcount':     20, 'min_dist':    648, 'max_dist':    763, 'probability': 0.470466, 'to_alt':      2, 'from_alt':      2},
        {'from': 113898, 'from_side':    'l', 'to':  47750, 'to_side':    'r', 'mean_dist':    -10, 'mapq':  60060, 'bcount':     31, 'min_dist':    -33, 'max_dist':     40, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      2}
        ]) )
    result_paths.append( pd.DataFrame({
        'pid':      [    500,    500,    500,    500,    500,    500,    500,    500,    500,    500],
        'pos':      [      0,      1,      2,      3,      4,      5,      6,      7,      8,      9],
        'phase0':   [    501,    501,    501,    501,    501,    501,    501,    501,    501,    501],
        'scaf0':    [ 110006,   1440,  11769,      7,  74998,  32135,  45782,  49372,  86633,  30446],
        'strand0':  [    '+',    '-',    '-',    '+',    '+',    '-',    '+',    '-',    '-',    '+'],
        'dist0':    [      0,    406,    -45,     76,   -558,    739,   -502,    334,    432,    -43],
        'phase1':   [   -502,    502,   -502,   -502,   -502,   -502,   -502,   -502,    502,    502],
        'scaf1':    [     -1,   7349,     -1,     -1,     -1,     -1,     -1,     -1,     -1,  30446],
        'strand1':  [     '',    '+',     '',     '',     '',     '',     '',     '',     '',    '+'],
        'dist1':    [      0,    387,      0,      0,      0,      0,      0,      0,      0,   9158]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    501,    501,    501,    501,    501,    501,    501,    501],
        'pos':      [      0,      1,      2,      3,      4,      5,      6,      7],
        'phase0':   [    503,    503,    503,    503,    503,    503,    503,    503],
        'scaf0':    [  93920,      7,     -1,     -1,  45782,  45783,  31737,  31758],
        'strand0':  [    '-',    '+',     '',     '',    '+',    '+',    '+',    '+'],
        'dist0':    [      0,    383,      0,      0,   -230,      0,   1044,     86],
        'phase1':   [   -504,   -504,    504,    504,    504,   -504,   -504,   -504],
        'scaf1':    [     -1,     -1,  74998,  32135,  45782,     -1,     -1,     -1],
        'strand1':  [     '',     '',    '+',    '-',    '+',     '',     '',     ''],
        'dist1':    [      0,      0,   -558,    739,   -502,      0,      0,      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    502,    502,    502,    502,    502],
        'pos':      [      0,      1,      2,      3,      4],
        'phase0':   [    505,    505,    505,    505,    505],
        'scaf0':    [  54753,  95291,  10945, 105853,  32135],
        'strand0':  [    '+',    '+',    '+',    '-',    '-'],
        'dist0':    [      0,    -18,    -45,    -43,    -45],
        'phase1':   [   -506,   -506,   -506,   -506,   -506],
        'scaf1':    [     -1,     -1,     -1,     -1,     -1],
        'strand1':  [     '',     '',     '',     '',     ''],
        'dist1':    [      0,      0,      0,      0,      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    503,    503,    503,    503,    503],
        'pos':      [      0,      1,      2,      3,      4],
        'phase0':   [    507,    507,    507,    507,    507],
        'scaf0':    [  23515,  95291,  29100, 105853,  32420],
        'strand0':  [    '+',    '+',    '-',    '-',    '-'],
        'dist0':    [      0,      3,    -43,    -44,    -44],
        'phase1':   [   -508,   -508,   -508,   -508,   -508],
        'scaf1':    [     -1,     -1,     -1,     -1,     -1],
        'strand1':  [     '',     '',     '',     '',     ''],
        'dist1':    [      0,      0,      0,      0,      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    504,    504,    504,    504],
        'pos':      [      0,      1,      2,      3],
        'phase0':   [    509,    509,    509,    509],
        'scaf0':    [  31729,  31758,  47750, 113898],
        'strand0':  [    '+',    '+',    '+',    '+'],
        'dist0':    [      0,   -399,    -42,    -10],
        'phase1':   [   -510,   -510,    510,    510],
        'scaf1':    [     -1,     -1,     -1, 113898],
        'strand1':  [     '',     '',     '',    '+'],
        'dist1':    [      0,      0,      0,    686]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    506,    506],
        'pos':      [      0,      1],
        'phase0':   [    513,    513],
        'scaf0':    [  76037,  30446],
        'strand0':  [    '+',    '-'],
        'dist0':    [      0,      0],
        'phase1':   [   -514,   -514],
        'scaf1':    [     -1,     -1],
        'strand1':  [     '',     ''],
        'dist1':    [      0,      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    507,    507],
        'pos':      [      0,      1],
        'phase0':   [    515,    515],
        'scaf0':    [  31729,  30446],
        'strand0':  [    '-',    '-'],
        'dist0':    [      0,    630],
        'phase1':   [   -516,   -516],
        'scaf1':    [     -1,     -1],
        'strand1':  [     '',     ''],
        'dist1':    [      0,      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    508],
        'pos':      [      0],
        'phase0':   [    517],
        'scaf0':    [  31108],
        'strand0':  [    '+'],
        'dist0':    [      0],
        'phase1':   [   -518],
        'scaf1':    [     -1],
        'strand1':  [     ''],
        'dist1':    [      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    509,    509],
        'pos':      [      0,      1],
        'phase0':   [    519,    519],
        'scaf0':    [  31729,  76037],
        'strand0':  [    '-',    '+'],
        'dist0':    [      0,   -162],
        'phase1':   [   -520,   -520],
        'scaf1':    [     -1,     -1],
        'strand1':  [     '',     ''],
        'dist1':    [      0,      0]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    511,    511],
        'pos':      [      0,      1],
        'phase0':   [    523,    523],
        'scaf0':    [  76037,  31737],
        'strand0':  [    '-',    '+'],
        'dist0':    [      0,   -168],
        'phase1':   [   -524,   -524],
        'scaf1':    [     -1,     -1],
        'strand1':  [     '',     ''],
        'dist1':    [      0,      0]
        }) )
#
    # Test 6
    scaffolds.append( pd.DataFrame({'case':6, 'scaffold':[9064, 41269, 51925, 67414, 123224, 123225, 123226, 123227, 123228, 123229, 123230, 123231, 123236, 123237, 123238]}) )
    scaffold_graph.append( pd.DataFrame([
        {'from':   9064, 'from_side':    'l', 'length':      7, 'scaf1': 123226, 'strand1':    '+', 'dist1':     -5, 'scaf2': 123238, 'strand2':    '+', 'dist2':      3, 'scaf3': 123225, 'strand3':    '+', 'dist3':    -42, 'scaf4': 123231, 'strand4':    '+', 'dist4':    -42, 'scaf5': 123227, 'strand5':    '+', 'dist5':    549, 'scaf6': 123236, 'strand6':    '+', 'dist6':    224},
        {'from':   9064, 'from_side':    'l', 'length':      6, 'scaf1': 123226, 'strand1':    '+', 'dist1':     -5, 'scaf2': 123238, 'strand2':    '+', 'dist2':    400, 'scaf3': 123225, 'strand3':    '+', 'dist3':    -42, 'scaf4': 123231, 'strand4':    '+', 'dist4':    -42, 'scaf5': 123227, 'strand5':    '+', 'dist5':    549},
        {'from':  41269, 'from_side':    'l', 'length':      5, 'scaf1': 123224, 'strand1':    '+', 'dist1':   -519, 'scaf2': 123228, 'strand2':    '+', 'dist2':     43, 'scaf3':  51925, 'strand3':    '+', 'dist3':     55, 'scaf4':  67414, 'strand4':    '+', 'dist4':     87},
        {'from':  41269, 'from_side':    'r', 'length':      7, 'scaf1':  51925, 'strand1':    '-', 'dist1':      4, 'scaf2': 123224, 'strand2':    '-', 'dist2':    965, 'scaf3': 123229, 'strand3':    '-', 'dist3':    179, 'scaf4': 123225, 'strand4':    '-', 'dist4':    -42, 'scaf5': 123230, 'strand5':    '-', 'dist5':    -40, 'scaf6': 123227, 'strand6':    '-', 'dist6':    196},
        {'from':  41269, 'from_side':    'r', 'length':      5, 'scaf1':  51925, 'strand1':    '-', 'dist1':      4, 'scaf2': 123224, 'strand2':    '-', 'dist2':    965, 'scaf3': 123229, 'strand3':    '-', 'dist3':    179, 'scaf4': 123230, 'strand4':    '-', 'dist4':    590},
        {'from':  51925, 'from_side':    'l', 'length':      6, 'scaf1': 123224, 'strand1':    '-', 'dist1':    965, 'scaf2': 123229, 'strand2':    '-', 'dist2':    179, 'scaf3': 123225, 'strand3':    '-', 'dist3':    -42, 'scaf4': 123230, 'strand4':    '-', 'dist4':    -40, 'scaf5': 123227, 'strand5':    '-', 'dist5':    196},
        {'from':  51925, 'from_side':    'l', 'length':      4, 'scaf1': 123224, 'strand1':    '-', 'dist1':    965, 'scaf2': 123229, 'strand2':    '-', 'dist2':    179, 'scaf3': 123230, 'strand3':    '-', 'dist3':    590},
        {'from':  51925, 'from_side':    'l', 'length':      4, 'scaf1': 123228, 'strand1':    '-', 'dist1':     55, 'scaf2': 123224, 'strand2':    '-', 'dist2':     43, 'scaf3':  41269, 'strand3':    '+', 'dist3':   -519},
        {'from':  51925, 'from_side':    'r', 'length':      2, 'scaf1':  41269, 'strand1':    '-', 'dist1':      4},
        {'from':  51925, 'from_side':    'r', 'length':      2, 'scaf1':  67414, 'strand1':    '+', 'dist1':     87},
        {'from':  67414, 'from_side':    'l', 'length':      5, 'scaf1':  51925, 'strand1':    '-', 'dist1':     87, 'scaf2': 123228, 'strand2':    '-', 'dist2':     55, 'scaf3': 123224, 'strand3':    '-', 'dist3':     43, 'scaf4':  41269, 'strand4':    '+', 'dist4':   -519},
        {'from': 123224, 'from_side':    'l', 'length':      2, 'scaf1':  41269, 'strand1':    '+', 'dist1':   -519},
        {'from': 123224, 'from_side':    'l', 'length':      6, 'scaf1': 123229, 'strand1':    '-', 'dist1':    179, 'scaf2': 123225, 'strand2':    '-', 'dist2':    -42, 'scaf3': 123230, 'strand3':    '-', 'dist3':    -40, 'scaf4': 123227, 'strand4':    '-', 'dist4':    196, 'scaf5': 123237, 'strand5':    '-', 'dist5':    542},
        {'from': 123224, 'from_side':    'l', 'length':      3, 'scaf1': 123229, 'strand1':    '-', 'dist1':    179, 'scaf2': 123230, 'strand2':    '-', 'dist2':    590},
        {'from': 123224, 'from_side':    'r', 'length':      3, 'scaf1':  51925, 'strand1':    '+', 'dist1':    965, 'scaf2':  41269, 'strand2':    '-', 'dist2':      4},
        {'from': 123224, 'from_side':    'r', 'length':      4, 'scaf1': 123228, 'strand1':    '+', 'dist1':     43, 'scaf2':  51925, 'strand2':    '+', 'dist2':     55, 'scaf3':  67414, 'strand3':    '+', 'dist3':     87},
        {'from': 123225, 'from_side':    'l', 'length':      4, 'scaf1': 123230, 'strand1':    '-', 'dist1':    -40, 'scaf2': 123227, 'strand2':    '-', 'dist2':    196, 'scaf3': 123237, 'strand3':    '-', 'dist3':    542},
        {'from': 123225, 'from_side':    'l', 'length':      4, 'scaf1': 123238, 'strand1':    '-', 'dist1':    -42, 'scaf2': 123226, 'strand2':    '-', 'dist2':      3, 'scaf3':   9064, 'strand3':    '+', 'dist3':     -5},
        {'from': 123225, 'from_side':    'l', 'length':      4, 'scaf1': 123238, 'strand1':    '-', 'dist1':    -42, 'scaf2': 123226, 'strand2':    '-', 'dist2':    400, 'scaf3':   9064, 'strand3':    '+', 'dist3':     -5},
        {'from': 123225, 'from_side':    'r', 'length':      5, 'scaf1': 123229, 'strand1':    '+', 'dist1':    -42, 'scaf2': 123224, 'strand2':    '+', 'dist2':    179, 'scaf3':  51925, 'strand3':    '+', 'dist3':    965, 'scaf4':  41269, 'strand4':    '-', 'dist4':      4},
        {'from': 123225, 'from_side':    'r', 'length':      4, 'scaf1': 123231, 'strand1':    '+', 'dist1':    -42, 'scaf2': 123227, 'strand2':    '+', 'dist2':    549, 'scaf3': 123236, 'strand3':    '+', 'dist3':    224},
        {'from': 123226, 'from_side':    'l', 'length':      2, 'scaf1':   9064, 'strand1':    '+', 'dist1':     -5},
        {'from': 123226, 'from_side':    'l', 'length':      4, 'scaf1': 123236, 'strand1':    '-', 'dist1':      2, 'scaf2': 123227, 'strand2':    '-', 'dist2':    224, 'scaf3': 123231, 'strand3':    '-', 'dist3':    549},
        {'from': 123226, 'from_side':    'r', 'length':      4, 'scaf1': 123237, 'strand1':    '+', 'dist1':      0, 'scaf2': 123227, 'strand2':    '+', 'dist2':    542, 'scaf3': 123230, 'strand3':    '+', 'dist3':    196},
        {'from': 123226, 'from_side':    'r', 'length':      6, 'scaf1': 123238, 'strand1':    '+', 'dist1':      3, 'scaf2': 123225, 'strand2':    '+', 'dist2':    -42, 'scaf3': 123231, 'strand3':    '+', 'dist3':    -42, 'scaf4': 123227, 'strand4':    '+', 'dist4':    549, 'scaf5': 123236, 'strand5':    '+', 'dist5':    224},
        {'from': 123226, 'from_side':    'r', 'length':      4, 'scaf1': 123238, 'strand1':    '+', 'dist1':      3, 'scaf2': 123231, 'strand2':    '+', 'dist2':    564, 'scaf3': 123227, 'strand3':    '+', 'dist3':    549},
        {'from': 123226, 'from_side':    'r', 'length':      5, 'scaf1': 123238, 'strand1':    '+', 'dist1':    400, 'scaf2': 123225, 'strand2':    '+', 'dist2':    -42, 'scaf3': 123231, 'strand3':    '+', 'dist3':    -42, 'scaf4': 123227, 'strand4':    '+', 'dist4':    549},
        {'from': 123227, 'from_side':    'l', 'length':      6, 'scaf1': 123231, 'strand1':    '-', 'dist1':    549, 'scaf2': 123225, 'strand2':    '-', 'dist2':    -42, 'scaf3': 123238, 'strand3':    '-', 'dist3':    -42, 'scaf4': 123226, 'strand4':    '-', 'dist4':      3, 'scaf5':   9064, 'strand5':    '+', 'dist5':     -5},
        {'from': 123227, 'from_side':    'l', 'length':      6, 'scaf1': 123231, 'strand1':    '-', 'dist1':    549, 'scaf2': 123225, 'strand2':    '-', 'dist2':    -42, 'scaf3': 123238, 'strand3':    '-', 'dist3':    -42, 'scaf4': 123226, 'strand4':    '-', 'dist4':    400, 'scaf5':   9064, 'strand5':    '+', 'dist5':     -5},
        {'from': 123227, 'from_side':    'l', 'length':      4, 'scaf1': 123231, 'strand1':    '-', 'dist1':    549, 'scaf2': 123238, 'strand2':    '-', 'dist2':    564, 'scaf3': 123226, 'strand3':    '-', 'dist3':      3},
        {'from': 123227, 'from_side':    'l', 'length':      5, 'scaf1': 123237, 'strand1':    '-', 'dist1':    542, 'scaf2': 123226, 'strand2':    '-', 'dist2':      0, 'scaf3': 123236, 'strand3':    '-', 'dist3':      2, 'scaf4': 123227, 'strand4':    '-', 'dist4':    224},
        {'from': 123227, 'from_side':    'r', 'length':      7, 'scaf1': 123230, 'strand1':    '+', 'dist1':    196, 'scaf2': 123225, 'strand2':    '+', 'dist2':    -40, 'scaf3': 123229, 'strand3':    '+', 'dist3':    -42, 'scaf4': 123224, 'strand4':    '+', 'dist4':    179, 'scaf5':  51925, 'strand5':    '+', 'dist5':    965, 'scaf6':  41269, 'strand6':    '-', 'dist6':      4},
        {'from': 123227, 'from_side':    'r', 'length':      5, 'scaf1': 123236, 'strand1':    '+', 'dist1':    224, 'scaf2': 123226, 'strand2':    '+', 'dist2':      2, 'scaf3': 123237, 'strand3':    '+', 'dist3':      0, 'scaf4': 123227, 'strand4':    '+', 'dist4':    542},
        {'from': 123228, 'from_side':    'l', 'length':      3, 'scaf1': 123224, 'strand1':    '-', 'dist1':     43, 'scaf2':  41269, 'strand2':    '+', 'dist2':   -519},
        {'from': 123228, 'from_side':    'r', 'length':      3, 'scaf1':  51925, 'strand1':    '+', 'dist1':     55, 'scaf2':  67414, 'strand2':    '+', 'dist2':     87},
        {'from': 123229, 'from_side':    'l', 'length':      5, 'scaf1': 123225, 'strand1':    '-', 'dist1':    -42, 'scaf2': 123230, 'strand2':    '-', 'dist2':    -40, 'scaf3': 123227, 'strand3':    '-', 'dist3':    196, 'scaf4': 123237, 'strand4':    '-', 'dist4':    542},
        {'from': 123229, 'from_side':    'l', 'length':      2, 'scaf1': 123230, 'strand1':    '-', 'dist1':    590},
        {'from': 123229, 'from_side':    'r', 'length':      4, 'scaf1': 123224, 'strand1':    '+', 'dist1':    179, 'scaf2':  51925, 'strand2':    '+', 'dist2':    965, 'scaf3':  41269, 'strand3':    '-', 'dist3':      4},
        {'from': 123230, 'from_side':    'l', 'length':      4, 'scaf1': 123227, 'strand1':    '-', 'dist1':    196, 'scaf2': 123237, 'strand2':    '-', 'dist2':    542, 'scaf3': 123226, 'strand3':    '-', 'dist3':      0},
        {'from': 123230, 'from_side':    'r', 'length':      6, 'scaf1': 123225, 'strand1':    '+', 'dist1':    -40, 'scaf2': 123229, 'strand2':    '+', 'dist2':    -42, 'scaf3': 123224, 'strand3':    '+', 'dist3':    179, 'scaf4':  51925, 'strand4':    '+', 'dist4':    965, 'scaf5':  41269, 'strand5':    '-', 'dist5':      4},
        {'from': 123230, 'from_side':    'r', 'length':      5, 'scaf1': 123229, 'strand1':    '+', 'dist1':    590, 'scaf2': 123224, 'strand2':    '+', 'dist2':    179, 'scaf3':  51925, 'strand3':    '+', 'dist3':    965, 'scaf4':  41269, 'strand4':    '-', 'dist4':      4},
        {'from': 123231, 'from_side':    'l', 'length':      5, 'scaf1': 123225, 'strand1':    '-', 'dist1':    -42, 'scaf2': 123238, 'strand2':    '-', 'dist2':    -42, 'scaf3': 123226, 'strand3':    '-', 'dist3':      3, 'scaf4':   9064, 'strand4':    '+', 'dist4':     -5},
        {'from': 123231, 'from_side':    'l', 'length':      5, 'scaf1': 123225, 'strand1':    '-', 'dist1':    -42, 'scaf2': 123238, 'strand2':    '-', 'dist2':    -42, 'scaf3': 123226, 'strand3':    '-', 'dist3':    400, 'scaf4':   9064, 'strand4':    '+', 'dist4':     -5},
        {'from': 123231, 'from_side':    'l', 'length':      3, 'scaf1': 123238, 'strand1':    '-', 'dist1':    564, 'scaf2': 123226, 'strand2':    '-', 'dist2':      3},
        {'from': 123231, 'from_side':    'r', 'length':      4, 'scaf1': 123227, 'strand1':    '+', 'dist1':    549, 'scaf2': 123236, 'strand2':    '+', 'dist2':    224, 'scaf3': 123226, 'strand3':    '+', 'dist3':      2},
        {'from': 123236, 'from_side':    'l', 'length':      7, 'scaf1': 123227, 'strand1':    '-', 'dist1':    224, 'scaf2': 123231, 'strand2':    '-', 'dist2':    549, 'scaf3': 123225, 'strand3':    '-', 'dist3':    -42, 'scaf4': 123238, 'strand4':    '-', 'dist4':    -42, 'scaf5': 123226, 'strand5':    '-', 'dist5':      3, 'scaf6':   9064, 'strand6':    '+', 'dist6':     -5},
        {'from': 123236, 'from_side':    'l', 'length':      4, 'scaf1': 123227, 'strand1':    '-', 'dist1':    224, 'scaf2': 123231, 'strand2':    '-', 'dist2':    549, 'scaf3': 123238, 'strand3':    '-', 'dist3':    564},
        {'from': 123236, 'from_side':    'r', 'length':      4, 'scaf1': 123226, 'strand1':    '+', 'dist1':      2, 'scaf2': 123237, 'strand2':    '+', 'dist2':      0, 'scaf3': 123227, 'strand3':    '+', 'dist3':    542},
        {'from': 123237, 'from_side':    'l', 'length':      4, 'scaf1': 123226, 'strand1':    '-', 'dist1':      0, 'scaf2': 123236, 'strand2':    '-', 'dist2':      2, 'scaf3': 123227, 'strand3':    '-', 'dist3':    224},
        {'from': 123237, 'from_side':    'r', 'length':      6, 'scaf1': 123227, 'strand1':    '+', 'dist1':    542, 'scaf2': 123230, 'strand2':    '+', 'dist2':    196, 'scaf3': 123225, 'strand3':    '+', 'dist3':    -40, 'scaf4': 123229, 'strand4':    '+', 'dist4':    -42, 'scaf5': 123224, 'strand5':    '+', 'dist5':    179},
        {'from': 123238, 'from_side':    'l', 'length':      3, 'scaf1': 123226, 'strand1':    '-', 'dist1':      3, 'scaf2':   9064, 'strand2':    '+', 'dist2':     -5},
        {'from': 123238, 'from_side':    'l', 'length':      3, 'scaf1': 123226, 'strand1':    '-', 'dist1':    400, 'scaf2':   9064, 'strand2':    '+', 'dist2':     -5},
        {'from': 123238, 'from_side':    'r', 'length':      5, 'scaf1': 123225, 'strand1':    '+', 'dist1':    -42, 'scaf2': 123231, 'strand2':    '+', 'dist2':    -42, 'scaf3': 123227, 'strand3':    '+', 'dist3':    549, 'scaf4': 123236, 'strand4':    '+', 'dist4':    224},
        {'from': 123238, 'from_side':    'r', 'length':      4, 'scaf1': 123231, 'strand1':    '+', 'dist1':    564, 'scaf2': 123227, 'strand2':    '+', 'dist2':    549, 'scaf3': 123236, 'strand3':    '+', 'dist3':    224}
        ]) )
    scaf_bridges.append( pd.DataFrame([
        {'from':   9064, 'from_side':    'l', 'to': 123226, 'to_side':    'l', 'mean_dist':     -5, 'mapq':  60060, 'bcount':     16, 'min_dist':    -10, 'max_dist':      4, 'probability': 0.159802, 'to_alt':      2, 'from_alt':      1},
        {'from':  41269, 'from_side':    'l', 'to': 123224, 'to_side':    'l', 'mean_dist':   -519, 'mapq':  60060, 'bcount':      9, 'min_dist':   -553, 'max_dist':   -501, 'probability': 0.027818, 'to_alt':      2, 'from_alt':      1},
        {'from':  41269, 'from_side':    'r', 'to':  51925, 'to_side':    'r', 'mean_dist':      4, 'mapq':  60060, 'bcount':     14, 'min_dist':      1, 'max_dist':      9, 'probability': 0.104244, 'to_alt':      2, 'from_alt':      1},
        {'from':  51925, 'from_side':    'l', 'to': 123224, 'to_side':    'r', 'mean_dist':    965, 'mapq':  60060, 'bcount':     12, 'min_dist':    917, 'max_dist':   1099, 'probability': 0.105770, 'to_alt':      2, 'from_alt':      2},
        {'from':  51925, 'from_side':    'l', 'to': 123228, 'to_side':    'r', 'mean_dist':     55, 'mapq':  60060, 'bcount':      7, 'min_dist':     44, 'max_dist':     84, 'probability': 0.014765, 'to_alt':      1, 'from_alt':      2},
        {'from':  51925, 'from_side':    'r', 'to':  41269, 'to_side':    'r', 'mean_dist':      4, 'mapq':  60060, 'bcount':     14, 'min_dist':      1, 'max_dist':      9, 'probability': 0.104244, 'to_alt':      1, 'from_alt':      2},
        {'from':  51925, 'from_side':    'r', 'to':  67414, 'to_side':    'l', 'mean_dist':     87, 'mapq':  60060, 'bcount':      6, 'min_dist':     52, 'max_dist':    117, 'probability': 0.010512, 'to_alt':      1, 'from_alt':      2},
        {'from':  67414, 'from_side':    'l', 'to':  51925, 'to_side':    'r', 'mean_dist':     87, 'mapq':  60060, 'bcount':      6, 'min_dist':     52, 'max_dist':    117, 'probability': 0.010512, 'to_alt':      2, 'from_alt':      1},
        {'from': 123224, 'from_side':    'l', 'to':  41269, 'to_side':    'l', 'mean_dist':   -519, 'mapq':  60060, 'bcount':      9, 'min_dist':   -553, 'max_dist':   -501, 'probability': 0.027818, 'to_alt':      1, 'from_alt':      2},
        {'from': 123224, 'from_side':    'l', 'to': 123229, 'to_side':    'r', 'mean_dist':    179, 'mapq':  60060, 'bcount':      9, 'min_dist':    162, 'max_dist':    207, 'probability': 0.027818, 'to_alt':      1, 'from_alt':      2},
        {'from': 123224, 'from_side':    'r', 'to':  51925, 'to_side':    'l', 'mean_dist':    965, 'mapq':  60060, 'bcount':     12, 'min_dist':    917, 'max_dist':   1099, 'probability': 0.105770, 'to_alt':      2, 'from_alt':      2},
        {'from': 123224, 'from_side':    'r', 'to': 123228, 'to_side':    'l', 'mean_dist':     43, 'mapq':  60060, 'bcount':     10, 'min_dist':     32, 'max_dist':     89, 'probability': 0.037322, 'to_alt':      1, 'from_alt':      2},
        {'from': 123225, 'from_side':    'l', 'to': 123230, 'to_side':    'r', 'mean_dist':    -40, 'mapq':  60060, 'bcount':      8, 'min_dist':    -49, 'max_dist':    -20, 'probability': 0.020422, 'to_alt':      2, 'from_alt':      2},
        {'from': 123225, 'from_side':    'l', 'to': 123238, 'to_side':    'r', 'mean_dist':    -42, 'mapq':  60060, 'bcount':      9, 'min_dist':    -48, 'max_dist':    -32, 'probability': 0.027818, 'to_alt':      2, 'from_alt':      2},
        {'from': 123225, 'from_side':    'r', 'to': 123229, 'to_side':    'l', 'mean_dist':    -42, 'mapq':  60060, 'bcount':      7, 'min_dist':    -49, 'max_dist':    -36, 'probability': 0.014765, 'to_alt':      2, 'from_alt':      2},
        {'from': 123225, 'from_side':    'r', 'to': 123231, 'to_side':    'l', 'mean_dist':    -42, 'mapq':  60060, 'bcount':      7, 'min_dist':    -47, 'max_dist':    -37, 'probability': 0.014765, 'to_alt':      2, 'from_alt':      2},
        {'from': 123226, 'from_side':    'l', 'to':   9064, 'to_side':    'l', 'mean_dist':     -5, 'mapq':  60060, 'bcount':     16, 'min_dist':    -10, 'max_dist':      4, 'probability': 0.159802, 'to_alt':      1, 'from_alt':      2},
        {'from': 123226, 'from_side':    'l', 'to': 123236, 'to_side':    'r', 'mean_dist':      2, 'mapq':  60060, 'bcount':      9, 'min_dist':      0, 'max_dist':      4, 'probability': 0.027818, 'to_alt':      1, 'from_alt':      2},
        {'from': 123226, 'from_side':    'r', 'to': 123237, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':      6, 'min_dist':     -3, 'max_dist':      9, 'probability': 0.010512, 'to_alt':      1, 'from_alt':      3},
        {'from': 123226, 'from_side':    'r', 'to': 123238, 'to_side':    'l', 'mean_dist':      3, 'mapq':  60060, 'bcount':      8, 'min_dist':      1, 'max_dist':      9, 'probability': 0.020422, 'to_alt':      2, 'from_alt':      3},
        {'from': 123226, 'from_side':    'r', 'to': 123238, 'to_side':    'l', 'mean_dist':    400, 'mapq':  60060, 'bcount':      7, 'min_dist':    381, 'max_dist':    425, 'probability': 0.023635, 'to_alt':      2, 'from_alt':      3},
        {'from': 123227, 'from_side':    'l', 'to': 123231, 'to_side':    'r', 'mean_dist':    549, 'mapq':  60060, 'bcount':     13, 'min_dist':    488, 'max_dist':    605, 'probability': 0.135136, 'to_alt':      1, 'from_alt':      2},
        {'from': 123227, 'from_side':    'l', 'to': 123237, 'to_side':    'r', 'mean_dist':    542, 'mapq':  60060, 'bcount':      8, 'min_dist':    523, 'max_dist':    560, 'probability': 0.033108, 'to_alt':      1, 'from_alt':      2},
        {'from': 123227, 'from_side':    'r', 'to': 123230, 'to_side':    'l', 'mean_dist':    196, 'mapq':  60060, 'bcount':      5, 'min_dist':    185, 'max_dist':    202, 'probability': 0.007368, 'to_alt':      1, 'from_alt':      2},
        {'from': 123227, 'from_side':    'r', 'to': 123236, 'to_side':    'l', 'mean_dist':    224, 'mapq':  60060, 'bcount':     12, 'min_dist':    191, 'max_dist':    374, 'probability': 0.064232, 'to_alt':      1, 'from_alt':      2},
        {'from': 123228, 'from_side':    'l', 'to': 123224, 'to_side':    'r', 'mean_dist':     43, 'mapq':  60060, 'bcount':     10, 'min_dist':     32, 'max_dist':     89, 'probability': 0.037322, 'to_alt':      2, 'from_alt':      1},
        {'from': 123228, 'from_side':    'r', 'to':  51925, 'to_side':    'l', 'mean_dist':     55, 'mapq':  60060, 'bcount':      7, 'min_dist':     44, 'max_dist':     84, 'probability': 0.014765, 'to_alt':      2, 'from_alt':      1},
        {'from': 123229, 'from_side':    'l', 'to': 123225, 'to_side':    'r', 'mean_dist':    -42, 'mapq':  60060, 'bcount':      7, 'min_dist':    -49, 'max_dist':    -36, 'probability': 0.014765, 'to_alt':      2, 'from_alt':      2},
        {'from': 123229, 'from_side':    'l', 'to': 123230, 'to_side':    'r', 'mean_dist':    590, 'mapq':  60060, 'bcount':      3, 'min_dist':    526, 'max_dist':    628, 'probability': 0.005063, 'to_alt':      2, 'from_alt':      2},
        {'from': 123229, 'from_side':    'r', 'to': 123224, 'to_side':    'l', 'mean_dist':    179, 'mapq':  60060, 'bcount':      9, 'min_dist':    162, 'max_dist':    207, 'probability': 0.027818, 'to_alt':      2, 'from_alt':      1},
        {'from': 123230, 'from_side':    'l', 'to': 123227, 'to_side':    'r', 'mean_dist':    196, 'mapq':  60060, 'bcount':      5, 'min_dist':    185, 'max_dist':    202, 'probability': 0.007368, 'to_alt':      2, 'from_alt':      1},
        {'from': 123230, 'from_side':    'r', 'to': 123225, 'to_side':    'l', 'mean_dist':    -40, 'mapq':  60060, 'bcount':      8, 'min_dist':    -49, 'max_dist':    -20, 'probability': 0.020422, 'to_alt':      2, 'from_alt':      2},
        {'from': 123230, 'from_side':    'r', 'to': 123229, 'to_side':    'l', 'mean_dist':    590, 'mapq':  60060, 'bcount':      3, 'min_dist':    526, 'max_dist':    628, 'probability': 0.005063, 'to_alt':      2, 'from_alt':      2},
        {'from': 123231, 'from_side':    'l', 'to': 123225, 'to_side':    'r', 'mean_dist':    -42, 'mapq':  60060, 'bcount':      7, 'min_dist':    -47, 'max_dist':    -37, 'probability': 0.014765, 'to_alt':      2, 'from_alt':      2},
        {'from': 123231, 'from_side':    'l', 'to': 123238, 'to_side':    'r', 'mean_dist':    564, 'mapq':  60060, 'bcount':      2, 'min_dist':    552, 'max_dist':    575, 'probability': 0.003280, 'to_alt':      2, 'from_alt':      2},
        {'from': 123231, 'from_side':    'r', 'to': 123227, 'to_side':    'l', 'mean_dist':    549, 'mapq':  60060, 'bcount':     13, 'min_dist':    488, 'max_dist':    605, 'probability': 0.135136, 'to_alt':      2, 'from_alt':      1},
        {'from': 123236, 'from_side':    'l', 'to': 123227, 'to_side':    'r', 'mean_dist':    224, 'mapq':  60060, 'bcount':     12, 'min_dist':    191, 'max_dist':    374, 'probability': 0.064232, 'to_alt':      2, 'from_alt':      1},
        {'from': 123236, 'from_side':    'r', 'to': 123226, 'to_side':    'l', 'mean_dist':      2, 'mapq':  60060, 'bcount':      9, 'min_dist':      0, 'max_dist':      4, 'probability': 0.027818, 'to_alt':      2, 'from_alt':      1},
        {'from': 123237, 'from_side':    'l', 'to': 123226, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':      6, 'min_dist':     -3, 'max_dist':      9, 'probability': 0.010512, 'to_alt':      3, 'from_alt':      1},
        {'from': 123237, 'from_side':    'r', 'to': 123227, 'to_side':    'l', 'mean_dist':    542, 'mapq':  60060, 'bcount':      8, 'min_dist':    523, 'max_dist':    560, 'probability': 0.033108, 'to_alt':      2, 'from_alt':      1},
        {'from': 123238, 'from_side':    'l', 'to': 123226, 'to_side':    'r', 'mean_dist':      3, 'mapq':  60060, 'bcount':      8, 'min_dist':      1, 'max_dist':      9, 'probability': 0.020422, 'to_alt':      3, 'from_alt':      2},
        {'from': 123238, 'from_side':    'l', 'to': 123226, 'to_side':    'r', 'mean_dist':    400, 'mapq':  60060, 'bcount':      7, 'min_dist':    381, 'max_dist':    425, 'probability': 0.023635, 'to_alt':      3, 'from_alt':      2},
        {'from': 123238, 'from_side':    'r', 'to': 123225, 'to_side':    'l', 'mean_dist':    -42, 'mapq':  60060, 'bcount':      9, 'min_dist':    -48, 'max_dist':    -32, 'probability': 0.027818, 'to_alt':      2, 'from_alt':      2},
        {'from': 123238, 'from_side':    'r', 'to': 123231, 'to_side':    'l', 'mean_dist':    564, 'mapq':  60060, 'bcount':      2, 'min_dist':    552, 'max_dist':    575, 'probability': 0.003280, 'to_alt':      2, 'from_alt':      2}
        ]) )
    result_paths.append( pd.DataFrame({
        'pid':      [    600,    600,    600,    600,    600,    600,    600,    600,    600,    600,    600,    600,    600,    600,    600,    600,    600,    600,    600,    600],
        'pos':      [      0,      1,      2,      3,      4,      5,      6,      7,      8,      9,     10,     11,     12,     13,     14,     15,     16,     17,     18,     19],
        'phase0':   [    601,    601,    601,    601,    601,    601,    601,    601,    601,    601,    601,    601,    601,    601,    601,    601,    601,    601,    601,    601],
        'scaf0':    [   9064, 123226, 123238, 123225, 123231, 123227, 123236, 123226, 123237, 123227, 123230, 123225, 123229, 123224,  51925,  41269, 123224, 123228,  51925,  67414],
        'strand0':  [    '-',    '+',    '+',    '+',    '+',    '+',    '+',    '+',    '+',    '+',    '+',    '+',    '+',    '+',    '+',    '-',    '+',    '+',    '+',    '+'],
        'dist0':    [      0,     -5,    400,    -42,    -42,    549,    224,      2,      0,    542,    196,    -40,    -42,    179,    965,      4,   -519,     43,     55,     87],
        'phase1':   [   -602,   -602,    602,    602,    602,   -602,   -602,   -602,   -602,   -602,   -602,    602,    602,   -602,   -602,   -602,   -602,   -602,   -602,   -602],
        'scaf1':    [     -1,     -1, 123238,     -1, 123231,     -1,     -1,     -1,     -1,     -1,     -1,     -1, 123229,     -1,     -1,     -1,     -1,     -1,     -1,     -1],
        'strand1':  [     '',     '',    '+',     '',    '+',     '',     '',     '',     '',     '',     '',     '',    '+',     '',     '',     '',     '',     '',     '',     ''],
        'dist1':    [      0,      0,      3,      0,    564,      0,      0,      0,      0,      0,      0,      0,    590,      0,      0,      0,      0,      0,      0,      0]
        }) )
#
    # Test 7
    scaffolds.append( pd.DataFrame({'case':7, 'scaffold':[17428, 48692, 123617]}) )
    scaffold_graph.append( pd.DataFrame([
        {'from':  17428, 'from_side':    'l', 'length':      2, 'scaf1':  17428, 'strand1':    '+', 'dist1':  28194},
        {'from':  17428, 'from_side':    'r', 'length':      2, 'scaf1':  48692, 'strand1':    '-', 'dist1':   1916},
        {'from':  17428, 'from_side':    'r', 'length':      3, 'scaf1': 123617, 'strand1':    '-', 'dist1':    -44, 'scaf2':  48692, 'strand2':    '-', 'dist2':    -38},
        {'from':  48692, 'from_side':    'r', 'length':      2, 'scaf1':  17428, 'strand1':    '-', 'dist1':   1916},
        {'from':  48692, 'from_side':    'r', 'length':      3, 'scaf1': 123617, 'strand1':    '+', 'dist1':    -38, 'scaf2':  17428, 'strand2':    '-', 'dist2':    -44},
        {'from': 123617, 'from_side':    'l', 'length':      2, 'scaf1':  48692, 'strand1':    '-', 'dist1':    -38},
        {'from': 123617, 'from_side':    'r', 'length':      2, 'scaf1':  17428, 'strand1':    '-', 'dist1':    -44}
        ]) )
    scaf_bridges.append( pd.DataFrame([
        {'from':  17428, 'from_side':    'l', 'to':  17428, 'to_side':    'l', 'mean_dist':  28194, 'mapq':  60060, 'bcount':      2, 'min_dist':  28194, 'max_dist':  28194, 'probability': 0.500000, 'to_alt':      1, 'from_alt':      1},
        {'from':  17428, 'from_side':    'r', 'to':  48692, 'to_side':    'r', 'mean_dist':   1916, 'mapq':  60060, 'bcount':      1, 'min_dist':   1916, 'max_dist':   1916, 'probability': 0.003098, 'to_alt':      2, 'from_alt':      2},
        {'from':  17428, 'from_side':    'r', 'to': 123617, 'to_side':    'r', 'mean_dist':    -44, 'mapq':  60060, 'bcount':      6, 'min_dist':    -50, 'max_dist':    -28, 'probability': 0.010512, 'to_alt':      1, 'from_alt':      2},
        {'from':  48692, 'from_side':    'r', 'to':  17428, 'to_side':    'r', 'mean_dist':   1916, 'mapq':  60060, 'bcount':      1, 'min_dist':   1916, 'max_dist':   1916, 'probability': 0.003098, 'to_alt':      2, 'from_alt':      2},
        {'from':  48692, 'from_side':    'r', 'to': 123617, 'to_side':    'l', 'mean_dist':    -38, 'mapq':  60060, 'bcount':      5, 'min_dist':    -48, 'max_dist':    -17, 'probability': 0.007368, 'to_alt':      1, 'from_alt':      2},
        {'from': 123617, 'from_side':    'l', 'to':  48692, 'to_side':    'r', 'mean_dist':    -38, 'mapq':  60060, 'bcount':      5, 'min_dist':    -48, 'max_dist':    -17, 'probability': 0.007368, 'to_alt':      2, 'from_alt':      1},
        {'from': 123617, 'from_side':    'r', 'to':  17428, 'to_side':    'r', 'mean_dist':    -44, 'mapq':  60060, 'bcount':      6, 'min_dist':    -50, 'max_dist':    -28, 'probability': 0.010512, 'to_alt':      2, 'from_alt':      1}
        ]) )
    result_paths.append( pd.DataFrame({
        'pid':      [    700,    700,    700],
        'pos':      [      0,      1,      2],
        'phase0':   [    701,    701,    701],
        'scaf0':    [  48692, 123617,  17428],
        'strand0':  [    '+',    '+',    '-'],
        'dist0':    [      0,    -38,    -44],
        'phase1':   [   -702,    702,    702],
        'scaf1':    [     -1,     -1,  17428],
        'strand1':  [     '',     '',    '-'],
        'dist1':    [      0,      0,   1916]
        }) )
    result_paths.append( pd.DataFrame({
        'pid':      [    701,    701],
        'pos':      [      0,      1],
        'phase0':   [    703,    703],
        'scaf0':    [  17428,  17428],
        'strand0':  [    '-',    '+'],
        'dist0':    [      0,  28194],
        'phase1':   [   -704,   -704],
        'scaf1':    [     -1,     -1],
        'strand1':  [     '',     ''],
        'dist1':    [      0,      0]
        }) )
#
    # Test 8
    scaffolds.append( pd.DataFrame({'case':8, 'scaffold':[197, 198, 199, 39830]}) )
    scaffold_graph.append( pd.DataFrame([
        {'from':    197, 'from_side':    'r', 'length':      4, 'scaf1':    198, 'strand1':    '+', 'dist1':      0, 'scaf2':    199, 'strand2':    '+', 'dist2':   -616, 'scaf3':    198, 'strand3':    '+', 'dist3':      3},
        {'from':    198, 'from_side':    'l', 'length':      2, 'scaf1':    197, 'strand1':    '-', 'dist1':      0},
        {'from':    198, 'from_side':    'l', 'length':      4, 'scaf1':    199, 'strand1':    '-', 'dist1':      3, 'scaf2':    198, 'strand2':    '-', 'dist2':   -616, 'scaf3':    197, 'strand3':    '-', 'dist3':      0},
        {'from':    198, 'from_side':    'r', 'length':      3, 'scaf1':    199, 'strand1':    '+', 'dist1':   -616, 'scaf2':    198, 'strand2':    '+', 'dist2':      3},
        {'from':    198, 'from_side':    'r', 'length':      2, 'scaf1':  39830, 'strand1':    '-', 'dist1':      0},
        {'from':    199, 'from_side':    'l', 'length':      3, 'scaf1':    198, 'strand1':    '-', 'dist1':   -616, 'scaf2':    197, 'strand2':    '-', 'dist2':      0},
        {'from':    199, 'from_side':    'r', 'length':      3, 'scaf1':    198, 'strand1':    '+', 'dist1':      3, 'scaf2':  39830, 'strand2':    '-', 'dist2':      0},
        {'from':  39830, 'from_side':    'r', 'length':      3, 'scaf1':    198, 'strand1':    '-', 'dist1':      0, 'scaf2':    199, 'strand2':    '-', 'dist2':      3}
        ]) )
    scaf_bridges.append( pd.DataFrame([
        {'from':    197, 'from_side':    'r', 'to':    198, 'to_side':    'l', 'mean_dist':      0, 'mapq':  60060, 'bcount':     15, 'min_dist':      0, 'max_dist':      0, 'probability': 0.129976, 'to_alt':      2, 'from_alt':      1},
        {'from':    198, 'from_side':    'l', 'to':    197, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':     15, 'min_dist':      0, 'max_dist':      0, 'probability': 0.129976, 'to_alt':      1, 'from_alt':      2},
        {'from':    198, 'from_side':    'l', 'to':    199, 'to_side':    'r', 'mean_dist':      3, 'mapq':  60060, 'bcount':     13, 'min_dist':     -1, 'max_dist':      5, 'probability': 0.082422, 'to_alt':      1, 'from_alt':      2},
        {'from':    198, 'from_side':    'r', 'to':    199, 'to_side':    'l', 'mean_dist':   -616, 'mapq':  60060, 'bcount':     12, 'min_dist':   -713, 'max_dist':   -568, 'probability': 0.064232, 'to_alt':      1, 'from_alt':      2},
        {'from':    198, 'from_side':    'r', 'to':  39830, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':     13, 'min_dist':      0, 'max_dist':      0, 'probability': 0.082422, 'to_alt':      1, 'from_alt':      2},
        {'from':    199, 'from_side':    'l', 'to':    198, 'to_side':    'r', 'mean_dist':   -616, 'mapq':  60060, 'bcount':     12, 'min_dist':   -713, 'max_dist':   -568, 'probability': 0.064232, 'to_alt':      2, 'from_alt':      1},
        {'from':    199, 'from_side':    'r', 'to':    198, 'to_side':    'l', 'mean_dist':      3, 'mapq':  60060, 'bcount':     13, 'min_dist':     -1, 'max_dist':      5, 'probability': 0.082422, 'to_alt':      2, 'from_alt':      1},
        {'from':  39830, 'from_side':    'r', 'to':    198, 'to_side':    'r', 'mean_dist':      0, 'mapq':  60060, 'bcount':     13, 'min_dist':      0, 'max_dist':      0, 'probability': 0.082422, 'to_alt':      2, 'from_alt':      1}
        ]) )
    result_paths.append( pd.DataFrame({
        'pid':      [    800,    800,    800,    800,    800],
        'pos':      [      0,      1,      2,      3,      4],
        'phase0':   [    801,    801,    801,    801,    801],
        'scaf0':    [    197,    198,    199,    198,  39830],
        'strand0':  [    '+',    '+',    '+',    '+',    '-'],
        'dist0':    [      0,      0,   -616,      3,      0],
        'phase1':   [   -802,   -802,   -802,   -802,   -802],
        'scaf1':    [     -1,     -1,     -1,     -1,     -1],
        'strand1':  [     '',     '',     '',     '',     ''],
        'dist1':    [      0,      0,      0,      0,      0]
        }) )
#
    # Combine tests
    scaffolds = pd.concat(scaffolds, ignore_index=True)
    scaffolds.index = scaffolds['scaffold'].values
    scaffolds['left'] = scaffolds['scaffold']
    scaffolds['right'] = scaffolds['scaffold']
    scaffolds[['lside','rside','lextendible','rextendible','circular','size']] = ['l','r',True,True,False,1]
    scaffold_graph = pd.concat(scaffold_graph, ignore_index=True)
    scaf_bridges = pd.concat(scaf_bridges, ignore_index=True)
    org_scaf_conns = pd.concat(org_scaf_conns, ignore_index=True)
    result_paths = pd.concat(result_paths, ignore_index=True)
#
    # Consistency tests
    if len(scaffolds.drop_duplicates()) != len(scaffolds):
        check = scaffolds.groupby(['scaffold']).size()
        check = check[check > 1].reset_index()['scaffold'].values
        print("Warning: Double usage of scaffolds in test: {check}")
    CheckConsistencyOfScaffoldGraph(scaffold_graph)
#
    # Run function
    graph_ext = FindValidExtensionsInScaffoldGraph(scaffold_graph)
    scaffold_paths = TraverseScaffoldingGraph(scaffolds.drop(columns=['case']), scaffold_graph, graph_ext, scaf_bridges, org_scaf_conns, ploidy, max_loop_units)
#
    # Compare and report failed tests
    failed = False
    for t in np.unique(scaffolds['case']):
        correct_paths = result_paths[np.isin(result_paths['pid'], np.unique(result_paths.loc[np.isin(result_paths['scaf0'], scaffolds.loc[scaffolds['case'] == t, 'scaffold'].values),'pid'].values))].copy()
        reversed_paths = correct_paths.copy()
        correct_paths['reverse'] = False
        reversed_paths['reverse'] = True
        reversed_paths = ReverseScaffolds(reversed_paths, reversed_paths['reverse'] , ploidy)
        tmp_paths = pd.concat([correct_paths, reversed_paths], ignore_index=True)
        correct_haps = []
        for h in range(ploidy):
            tmp_paths.loc[tmp_paths[f'phase{h}'] < 0, [f'scaf{h}',f'strand{h}',f'dist{h}']] = tmp_paths.loc[tmp_paths[f'phase{h}'] < 0, ['scaf0','strand0','dist0']].values
            correct_haps.append( tmp_paths.loc[tmp_paths[f'scaf{h}'] >= 0, ['pid','reverse',f'scaf{h}',f'strand{h}',f'dist{h}']].rename(columns={'pid':'cpid',f'scaf{h}':'scaf',f'strand{h}':'strand',f'dist{h}':'dist'}) )
            correct_haps[-1]['hap'] = h
        correct_haps = pd.concat(correct_haps, ignore_index=True)
        correct_haps['pos'] = correct_haps.groupby(['cpid','hap','reverse'], sort=False).cumcount()
        obtained_paths = scaffold_paths[np.isin(scaffold_paths['pid'], np.unique(scaffold_paths.loc[np.isin(scaffold_paths['scaf0'], scaffolds.loc[scaffolds['case'] == t, 'scaffold'].values),'pid'].values))].copy()
        tmp_paths = obtained_paths.copy()
        obtained_haps = []
        for h in range(ploidy):
            tmp_paths.loc[tmp_paths[f'phase{h}'] < 0, [f'scaf{h}',f'strand{h}',f'dist{h}']] = tmp_paths.loc[tmp_paths[f'phase{h}'] < 0, ['scaf0','strand0','dist0']].values
            obtained_haps.append( tmp_paths.loc[tmp_paths[f'scaf{h}'] >= 0, ['pid',f'scaf{h}',f'strand{h}',f'dist{h}']].rename(columns={'pid':'opid',f'scaf{h}':'scaf',f'strand{h}':'strand',f'dist{h}':'dist'}) )
            obtained_haps[-1]['hap'] = h
        obtained_haps = pd.concat(obtained_haps, ignore_index=True)
        obtained_haps['pos'] = obtained_haps.groupby(['opid','hap'], sort=False).cumcount()
        comp = correct_haps.merge(obtained_haps, on=['hap','pos','scaf','strand','dist'], how='inner')
        comp = comp.groupby(['cpid','opid','hap','reverse']).size().reset_index(name='bcount').groupby(['cpid','opid','hap'])[['bcount']].max().reset_index()
        comp = correct_haps[correct_haps['reverse']].groupby(['cpid','hap']).size().reset_index(name='ccount').merge(comp, on=['cpid','hap'], how='inner')
        comp = comp[comp['ccount'] == comp['bcount']].merge( obtained_haps.groupby(['opid','hap']).size().reset_index(name='ocount'), on=['opid','hap'], how='left')
        comp = comp[comp['ccount'] == comp['ocount']].groupby(['cpid','hap']).first().reset_index()
        comp = comp.groupby(['cpid','opid']).size().reset_index(name='nhaps')
        comp = comp[comp['nhaps'] == ploidy].copy()
        correct_paths = correct_paths[np.isin(correct_paths['pid'], comp['cpid']) == False].drop(columns=['reverse'])
        obtained_paths = obtained_paths[np.isin(obtained_paths['pid'], comp['opid']) == False].copy()
        if len(correct_paths) | len(obtained_paths):
            print(f"TestTraverseScaffoldingGraph: Test case {t} failed.")
            print("Unmatched correct paths:")
            print(correct_paths)
            print("Unmatched obtained paths:")
            print(obtained_paths)
            failed = True
#
    return failed

def MiniGapTest():
    failed_tests = 0
    failed_tests += TestDuplicationConflictResolution()
    failed_tests += TestFilterInvalidConnections()
    failed_tests += TestTraverseScaffoldingGraph()
    
    if failed_tests == 0:
        print("All tests succeeded.")
    else:
        print(failed_tests, "tests failed.")
    
def Usage(module=""):
    if "" == module:
        print("Program: minigap")
        print("Version: 0.1")
        print("Contact: Stephan Schmeing <stephan.schmeing@uzh.ch>")
        print()
        print("Usage:  minigap.py <module> [options]")
        print("Modules:")
        print("split         Splits scaffolds into contigs")
        print("scaffold      Scaffolds contigs and assigns reads to gaps")
        print("extend        Extend scaffold ends")
        print("finish        Create new fasta assembly file")
        print("visualize     Visualizes regions to manually inspect breaks or joins")
        print("test          Short test")
    elif "split" == module:
        print("Usage: minigap.py split [OPTIONS] {assembly}.fa")
        print("Splits scaffolds into contigs.")
        print("  -h, --help                Display this help and exit")
        print("  -n, --minN [int]          Minimum number of N's to split at that position (1)")
        print("  -o, --output FILE.fa      File to which the split sequences should be written to ({assembly}_split.fa)")
    elif "scaffold" == module:
        print("Usage: minigap.py scaffold [OPTIONS] {assembly}.fa {mapping}.paf {repeat}.paf")
        print("Scaffolds contigs and assigns reads to gaps.")
        print("  -h, --help                Display this help and exit")
        print("  -p, --prefix FILE         Prefix for output files ({assembly})")
        print("  -s, --stats FILE.pdf      Output file for plots with statistics regarding input parameters (deactivated)")
        print("      --minLenBreak INT     Minimum length for a read to diverge from a contig to consider a contig break (600)")
        print("      --minMapLength INT    Minimum length of individual mappings of reads (400)")
        print("      --minMapQ INT         Minimum mapping quality of reads (20)")
    elif "extend" == module:
        print("Usage: minigap.py extend -p {prefix} {all_vs_all}.paf")
        print("Extend scaffold ends with reads reaching over the ends.")
        print("  -h, --help                Display this help and exit")
        print("  -p, --prefix FILE         Prefix for output files of scaffolding step (mandatory)")
        print("      --minLenBreak INT     Minimum length for two reads to diverge to consider them incompatible for this contig (1000)")
    elif "finish" == module:
        print("Usage: minigap.py finish [OPTIONS] -s {scaffolds}.csv {assembly}.fa {reads}.fq")
        print("Creates previously defined scaffolds. Only providing necessary reads increases speed and substantially reduces memory requirements.")
        print("  -h, --help                Display this help and exit")
        print("  -f, --format FORMAT       Format of {reads}.fq (fasta/fastq) (Default: fastq if not determinable from read ending)")
        print("  -H, --hap INT             Haplotype starting from 0 written to {output} (default: optimal mix)")
        print("  --hap[1-9] INT            Haplotypes starting from 0 written to {out[1-9]} (default: optimal mix)")
        print("  -o, --output FILE.fa      Output file for modified assembly ({assembly}_minigap.fa)")
        print("  --out[1-9] FILE.fa        Additional output files for modified assembly (deactivated)")
        print("  -s, --scaffolds FILE.csv  Csv file from previous steps describing the scaffolding (mandatory)")
    elif "visualize" == module:
        print("Usage: minigap.py visualize [OPTIONS] -o {output}.pdf {mapping}.paf {scaffold}:{start}-{end} [{scaffold}:{start}-{end} ...]")
        print("Visualizes specified regions to manually inspect breaks or joins.")
        print("  -h, --help                Display this help and exit")
        print("  -o, --output FILE.pdf     Output file for visualization (mandatory)")
        print("      --keepAllSubreads     Shows all subreads instead of only the best")
        print("      --minLenBreak INT     Minimum length for a read to diverge from a contig to consider a contig break (600)")
        print("      --minMapLength INT    Minimum length of individual mappings of reads (400)")
        print("      --minMapQ INT         Minimum mapping quality of reads (20)")
        
def main(argv):
    if 0 == len(argv):
        Usage()
        sys.exit()
    
    module = argv[0]
    argv.pop(0)
    if "-h" == module or "--help" == module:
        Usage()
        sys.exit()
    if "split" == module:
        try:
            optlist, args = getopt.getopt(argv, 'hn:o:', ['help','minN=','output='])
        except getopt.GetoptError:
            print("Unknown option")
            Usage(module)
            sys.exit(1)
    
        o_file = False
        min_n = False
        for opt, par in optlist:
            if opt in ("-h", "--help"):
                Usage(module)
                sys.exit()
            elif opt in ("-n", "--minN"):
                try:
                    min_n = int(par)
                except ValueError:
                    print("-n,--minN option only accepts integers")
                    sys.exit(1)
            elif opt in ("-o", "--output"):
                o_file = par
    
        if 1 != len(args):
            print("Wrong number of files. Exactly one file is required.")
            Usage(module)
            sys.exit(1)
    
        MiniGapSplit(args[0],o_file,min_n)
    elif "scaffold" == module:
        try:
            optlist, args = getopt.getopt(argv, 'hp:s:', ['help','prefix=','stats=','--minLenBreak=','minMapLength=','minMapQ='])
        except getopt.GetoptError:
            print("Unknown option\n")
            Usage(module)
            sys.exit(1)
            
        prefix = False
        stats = None
        min_length_contig_break = 700
        min_mapping_length = 500
        min_mapq = 20
        for opt, par in optlist:
            if opt in ("-h", "--help"):
                Usage(module)
                sys.exit()
            elif opt in ("-p", "--prefix"):
                prefix = par
            elif opt in ("-s", "--stats"):
                stats = par
                if stats[-4:] != ".pdf":
                    print("stats argument needs to end on .pdf")
                    Usage(module)
                    sys.exit(1)
            elif opt == "--minLenBreak":
                try:
                    min_length_contig_break = int(par)
                except ValueError:
                    print("--minLenBreak option only accepts integers")
                    sys.exit(1)
            elif opt == "--minMapLength":
                try:
                    min_mapping_length = int(par)
                except ValueError:
                    print("--minMapLength option only accepts integers")
                    sys.exit(1)
            elif opt == "--minMapQ":
                try:
                    min_mapq = int(par)
                except ValueError:
                    print("--minMapQ option only accepts integers")
                    sys.exit(1)

                    
        if 3 != len(args):
            print("Wrong number of files. Exactly three files are required.\n")
            Usage(module)
            sys.exit(2)

        MiniGapScaffold(args[0], args[1], args[2], min_mapq, min_mapping_length, min_length_contig_break, prefix, stats)
    elif "extend" == module:
        try:
            optlist, args = getopt.getopt(argv, 'hp:', ['help','prefix=','--minLenBreak'])
        except getopt.GetoptError:
            print("Unknown option\n")
            Usage(module)
            sys.exit(1)
            
        prefix = False
        min_length_contig_break = 1200
        for opt, par in optlist:
            if opt in ("-h", "--help"):
                Usage(module)
                sys.exit()
            elif opt in ("-p", "--prefix"):
                prefix = par
            elif opt == "--minLenBreak":
                try:
                    min_length_contig_break = int(par)
                except ValueError:
                    print("--minLenBreak option only accepts integers")
                    sys.exit(1)
                    
        if 1 != len(args):
            print("Wrong number of files. Exactly one file is required.\n")
            Usage(module)
            sys.exit(2)
            
        if False == prefix:
            print("prefix argument is mandatory")
            Usage(module)
            sys.exit(1)

        MiniGapExtend(args[0], prefix, min_length_contig_break)
    elif "finish" == module:
        num_slots = 10
        try:
            optlist, args = getopt.getopt(argv, 'hf:H:o:s:', ['help','format=','hap=','output=','scaffolds=']+[f'hap{i}=' for i in range(1,num_slots)]+[f'out{i}=' for i in range(1,num_slots)])
        except getopt.GetoptError:
            print("Unknown option\n")
            Usage(module)
            sys.exit(1)
        
        read_format = False
        output = [False]*num_slots
        haplotypes = [-1]*num_slots
        scaffolds = False
        for opt, par in optlist:
            if opt in ("-h", "--help"):
                Usage(module)
                sys.exit()
            elif opt in ("-f", "--format"):
                if 'fasta' == par.lower() or 'fa' == par.lower():
                    read_format = 'fasta'
                elif 'fastq' == par.lower() or 'fq' == par.lower():
                    pass # default
                else:
                    print("Unsupported read format: {}.".format(par))
                    Usage(module)
                    sys.exit(1)
            elif opt in ("-H", "--hap"):
                haplotypes[0] = int(par)
            elif opt in [f'--hap{i}' for i in range(1,num_slots)]:
                haplotypes[int(opt[5:])] = int(par)
            elif opt in ("-o", "--output"):
                output[0] = par
            elif opt in [f'--out{i}' for i in range(1,num_slots)]:
                output[int(opt[5:])] = par
            elif opt in ("-s", "--scaffolds"):
                scaffolds = par
        
        if 2 != len(args):
            print("Wrong number of files. Exactly two files are required.")
            Usage(module)
            sys.exit(2)
            
        if False == scaffolds:
            print("scaffolds argument is mandatory")
            Usage(module)
            sys.exit(1)
            
        selected_output = [output[0]]
        selected_haplotypes = [haplotypes[0]]
        for i in range(1,num_slots):
            if output[i] == False:
                if haplotypes[i] != -1:
                    print(f"--hap{i} was specified without --out{i}")
                    Usage(module)
                    sys.exit(1)
            else:
                selected_output.append(output[i])
                selected_haplotypes.append(haplotypes[i])

        MiniGapFinish(args[0], args[1], read_format, scaffolds, selected_output, selected_haplotypes)
    elif "visualize" == module:
        try:
            optlist, args = getopt.getopt(argv, 'ho:', ['help','output=','--keepAllSubreads','--minLenBreak=','minMapLength=','minMapQ='])
        except getopt.GetoptError:
            print("Unknown option\n")
            Usage(module)
            sys.exit(1)
            
        output = False
        keep_all_subreads = False
        min_length_contig_break = 700
        min_mapping_length = 500
        min_mapq = 20
        for opt, par in optlist:
            if opt in ("-h", "--help"):
                Usage(module)
                sys.exit()
            elif opt in ("-o", "--output"):
                output = par
            elif opt == "--keepAllSubreads":
                keep_all_subreads = True
            elif opt == "--minLenBreak":
                try:
                    min_length_contig_break = int(par)
                except ValueError:
                    print("--minLenBreak option only accepts integers")
                    sys.exit(1)
            elif opt == "--minMapLength":
                try:
                    min_mapping_length = int(par)
                except ValueError:
                    print("--minMapLength option only accepts integers")
                    sys.exit(1)
            elif opt == "--minMapQ":
                try:
                    min_mapq = int(par)
                except ValueError:
                    print("--minMapQ option only accepts integers")
                    sys.exit(1)
            
        if 2 > len(args):
            print("Wrong number of arguments. The mapping file and at least one region definition are needed.\n")
            Usage(module)
            sys.exit(2)
            
        if False == output:
            print("output argument is mandatory")
            Usage(module)
            sys.exit(1)
            
        MiniGapVisualize(args[1:], args[0], output, min_mapq, min_mapping_length, min_length_contig_break, keep_all_subreads)
    elif "test" == module:
        MiniGapTest()
    else:
        print("Unknown module: {}.".format(module))
        Usage()
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])
