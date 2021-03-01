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

def CallAllBreaksSpurious(mappings, contigs, max_dist_contig_end, min_length_contig_break, min_extension, pdf):
    left_breaks, right_breaks = GetBrokenMappings(mappings, max_dist_contig_end, min_length_contig_break, pdf)

    break_groups = []
    spurious_break_indexes = np.unique(np.concatenate([left_breaks.index, right_breaks.index]))
    non_informative_mappings = GetNonInformativeMappings(mappings, contigs, min_extension, break_groups, left_breaks, right_breaks)

    return break_groups, spurious_break_indexes, non_informative_mappings

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

def NormCDFCapped(x, mu, sigma):
    return np.minimum(0.5, NormCDF(x, mu, sigma)) # Cap it at the mean so that repeats do not get a bonus

def GetConProb(cov_probs, req_length, counts):
    probs = pd.DataFrame({'length':req_length, 'counts':counts, 'mu':0.0, 'sigma':1.0})
    for plen, mu, sigma in zip(reversed(cov_probs['length']), reversed(cov_probs['mu']), reversed(cov_probs['sigma'])):
        probs.loc[plen >= probs['length'], 'mu'] = mu
        probs.loc[plen >= probs['length'], 'sigma'] = sigma

    return NormCDFCapped(probs['counts'],probs['mu'], probs['sigma'])      

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

def FindBreakPoints(mappings, contigs, max_dist_contig_end, min_mapping_length, min_length_contig_break, max_break_point_distance, min_num_reads, min_extension, merge_block_length, org_scaffold_trust, cov_probs, prob_factor, pdf):
    if pdf:
        loose_reads_ends = mappings[(mappings['t_start'] > max_dist_contig_end) & np.where('+' == mappings['strand'], -1 == mappings['prev_con'], -1 == mappings['next_con'])]
        loose_reads_ends_length = np.where('+' == loose_reads_ends['strand'], loose_reads_ends['q_start']-loose_reads_ends['read_start'], loose_reads_ends['read_end']-loose_reads_ends['q_end'])
        loose_reads_ends = mappings[(mappings['t_len']-mappings['t_end'] > max_dist_contig_end) & np.where('+' == mappings['strand'], -1 == mappings['next_con'], -1 == mappings['prev_con'])]
        loose_reads_ends_length = np.concatenate([loose_reads_ends_length, np.where('+' == loose_reads_ends['strand'], loose_reads_ends['read_end']-loose_reads_ends['q_end'], loose_reads_ends['q_start']-loose_reads_ends['read_start'])])
        if len(loose_reads_ends_length):
            if np.sum(loose_reads_ends_length < 10*min_length_contig_break):
                PlotHist(pdf, "Loose end length", "# Ends", np.extract(loose_reads_ends_length < 10*min_length_contig_break, loose_reads_ends_length), threshold=min_length_contig_break, logy=True)
            PlotHist(pdf, "Loose end length", "# Ends", loose_reads_ends_length, threshold=min_length_contig_break, logx=True)

    pot_breaks = GetBrokenMappings(mappings, max_dist_contig_end, min_length_contig_break, pdf)
    break_points = pot_breaks[['contig_id','side','position','mapq','map_len']].copy()
    break_points['connected'] = 0 <= pot_breaks['con']
    break_points = break_points[ break_points['map_len'] >= min_mapping_length+max_break_point_distance ].copy() # require half the mapping length for breaks as we will later require for continously mapping reads that veto breaks, since breaking reads only map on one side
    break_points.drop(columns=['map_len'], inplace=True)
    break_points.sort_values(['contig_id','position'], inplace=True)
    if min_num_reads > 1:
        break_points = break_points[ ((break_points['contig_id'] == break_points['contig_id'].shift(-1, fill_value=-1)) & (break_points['position']+max_break_point_distance >= break_points['position'].shift(-1, fill_value=-1))) | ((break_points['contig_id'] == break_points['contig_id'].shift(1, fill_value=-1)) & (break_points['position']-max_break_point_distance <= break_points['position'].shift(1, fill_value=-1))) ].copy() # require at least one supporting break within +- max_break_point_distance

    if pdf:
        break_point_dist = (break_points['position'] - break_points['position'].shift(1, fill_value=0))[break_points['contig_id'] == break_points['contig_id'].shift(1, fill_value=-1)]
        if len(break_point_dist):
            if np.sum(break_point_dist <= 10*max_break_point_distance):
                PlotHist(pdf, "Break point distance", "# Break point pairs", break_point_dist[break_point_dist <= 10*max_break_point_distance], threshold=max_break_point_distance )
            PlotHist(pdf, "Break point distance", "# Break point pairs", break_point_dist, threshold=max_break_point_distance, logx=True )

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

        break_points = pd.concat([break_points, break_supp], ignore_index=True).groupby(['contig_id','position','mapq']).sum().reset_index()
        break_points.sort_values(['contig_id','position','mapq'], ascending=[True,True,False], inplace=True)
        # Add support from higher mapping qualities to lower mapping qualities
        break_points['support'] = break_points.groupby(['contig_id','position'])['support'].cumsum()
        break_points['con_supp'] = break_points.groupby(['contig_id','position'])['con_supp'].cumsum()
        # Require a support of at least min_num_reads at some mapping quality level
        max_support = break_points.groupby(['contig_id','position'], sort=False)['support'].agg(['max','size'])
        break_points = break_points[ np.repeat(max_support['max'].values, max_support['size'].values) >= min_num_reads].copy()

    if len(break_points):
        # Check how many reads veto a break (continously map over the break region position +- (min_mapping_length+max_break_point_distance))
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

        break_points = break_points.merge(pd.concat(break_list, ignore_index=True), on=['contig_id','position','mapq'], how='outer').fillna(0)
        break_points.sort_values(['contig_id','position','mapq'], ascending=[True,True,False], inplace=True)
        break_points.reset_index(inplace=True, drop=True)
        break_points[['support', 'con_supp']] = break_points.groupby(['contig_id','position'], sort=False)[['support', 'con_supp']].cummax().astype(int)
        break_points['vetos'] = break_points.groupby(['contig_id','position'], sort=False)['vetos'].cumsum().astype(int)

        # Remove breaks where the vetos are much more likely than the breaks
        break_points['break_prob'] = GetConProb(cov_probs, min_mapping_length+max_break_point_distance+min_length_contig_break, break_points['support'])
        break_points['veto_prob'] = GetConProb(cov_probs, 2*(min_mapping_length+max_break_point_distance), break_points['vetos'])
        break_points['vetoed'] = (break_points['vetos'] >= min_num_reads) & (break_points['veto_prob'] >= prob_factor*break_points['break_prob']) # Always make sure that vetos reach min_num_reads, before removing something based on their probability
        break_points['vetoed'] = break_points.groupby(['contig_id','position'], sort=False)['vetoed'].cumsum().astype(bool) # Propagate vetos to lower mapping qualities
        break_points = break_points[break_points['vetoed'] == False].copy()

        # Remove break points that will reconnected anyways
        break_points['vetoed'] = (break_points['vetos'] >= min_num_reads) & (break_points['con_supp'] < min_num_reads) # If we have enough support, but not enough con_supp the bridge finding step would anyways reseal the break with a unique bridge, so don't even break it
        break_points['vetoed'] = break_points.groupby(['contig_id','position'], sort=False)['vetoed'].cummax() # Propagate vetos to lower mapping qualities
        unconnected_break_points = [break_points[break_points['vetoed'] == True].copy()] # Store unconnected break points to later check if we can insert additional sequence to provide a connection in a second run
        break_points = break_points[break_points['vetoed'] == False].copy()

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

        # Remove break points with not enough consistent breaks to be likely enough
        break_pos['con_prob'] = GetConProb(cov_probs, np.where(0 <= break_pos['mean_dist'], break_pos['mean_dist']+2*min_mapping_length, min_mapping_length-break_pos['mean_dist']), break_pos['count'])
        break_points = break_points.merge(break_pos.groupby(['contig_id','position','mapq'])['con_prob'].max().reset_index(name='con_prob'), on=['contig_id','position','mapq'], how='left')
        break_points['con_prob'] =  break_points.fillna(0.0).groupby(['contig_id','position'], sort=False)['con_prob'].cummax()
        break_points['vetoed'] = (break_points['vetos'] >= min_num_reads) & (break_points['veto_prob'] >= prob_factor*break_points['con_prob']) # Always make sure that vetos reach min_num_reads, before removing something based on their probability
        break_points['vetoed'] = break_points.groupby(['contig_id','position'], sort=False)['vetoed'].cummax() # Propagate vetos to lower mapping qualities
        unconnected_break_points.append( break_points[break_points['vetoed'] == True].drop(columns=['con_prob']) )
        break_points = break_points[break_points['vetoed'] == False].copy()

        # Remove breaks that do not fullfil requirements and reduce accepted ones to a single entry per break_id
        break_points = break_points[break_points['support'] >= min_num_reads].copy()
        break_points = break_points.groupby(['contig_id','position'], sort=False).first().reset_index()

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

        break_groups.drop(columns=['group','size'], inplace=True)
        break_groups.rename(columns={'min':'start', 'max':'end'}, inplace=True)
        break_groups['end'] += 1 # Set end to one position after the last included one to be consistent with other ends (like q_end, t_end)
        break_groups['pos'] = (break_groups['end']+break_groups['start'])//2
        break_groups['num'] = break_groups['contig_id'] == break_groups['contig_id'].shift(1, fill_value=-1)
        break_groups['num'] = break_groups.groupby('contig_id')['num'].cumsum()
    else:
        break_groups = []

    if len(break_groups):
        # Find mappings belonging to accepted breaks
        pot_breaks['keep'] = False
        for i in range(break_groups['num'].max()+1):
            breaks = pd.DataFrame({'contig':np.arange(len(contigs)), 'start':sys.maxsize, 'end':-1})
            breaks.iloc[break_groups.loc[i==break_groups['num'], 'contig_id'], [breaks.columns.get_loc('start'), breaks.columns.get_loc('end')]] = break_groups.loc[i==break_groups['num'], ['start', 'end']].values
            breaks = breaks.iloc[pot_breaks['contig_id']]
            pot_breaks['keep'] = pot_breaks['keep'] | np.where('l' == pot_breaks['side'], (breaks['start'].values <= pot_breaks['t_end'].values) & (pot_breaks['t_end'].values <= breaks['end'].values),
                                                                                          (breaks['start'].values <= pot_breaks['t_start'].values) & (pot_breaks['t_start'].values <= breaks['end'].values))

        # Find reads where all mappings are either connected to itself or belong to accepted breaks
        pot_breaks['self'] = pot_breaks['keep'] | (pot_breaks['contig_id'] == pot_breaks['con'])
        self_con = pot_breaks.groupby(['q_name','contig_id'])['self'].agg(['sum','size']).reset_index()
        self_con = self_con.loc[self_con['sum'] == self_con['size'], ['q_name','contig_id']].drop_duplicates()
        pot_breaks['keep'] = pot_breaks['keep'] | (pot_breaks.merge(self_con, on=['q_name','contig_id'], how='left', indicator=True)['_merge'] == "both").values

        # Mappings not belonging to accepted breaks or connecting to itself are spurious
        spurious_break_indexes = np.unique(pot_breaks.loc[pot_breaks['keep'] == False, 'map_index'].values)
    else:
        # We don't have breaks, so all are spurious
        spurious_break_indexes = np.unique(pot_breaks['map_index'].values)

    non_informative_mappings = GetNonInformativeMappings(mappings, contigs, min_extension, break_groups, pot_breaks)

    # Filter unconnected break points that overlap a break_group (probably from vetoed low mapping qualities, where the high mapping quality was not vetoed)
    unconnected_break_points = pd.concat(unconnected_break_points, ignore_index=True)[['contig_id','position']].sort_values(['contig_id','position']).drop_duplicates()
    unconnected_break_points['remove'] = False
    for i in range(break_groups['num'].max()+1):
        breaks = pd.DataFrame({'contig':np.arange(len(contigs)), 'start':sys.maxsize, 'end':-1})
        breaks.iloc[break_groups.loc[i==break_groups['num'], 'contig_id'], [breaks.columns.get_loc('start'), breaks.columns.get_loc('end')]] = break_groups.loc[i==break_groups['num'], ['start', 'end']].values
        breaks = breaks.iloc[unconnected_break_points['contig_id']]
        unconnected_break_points['remove'] = unconnected_break_points['remove'] | ( (breaks['start'].values-max_break_point_distance <= unconnected_break_points['position'].values) & (unconnected_break_points['position'].values <= breaks['end'].values+max_break_point_distance) )
    unconnected_break_points = unconnected_break_points[unconnected_break_points['remove'] == False].drop(columns=['remove'])

    # Handle unconnected break points
    unconnected_break_points['dist'] = np.where( unconnected_break_points['contig_id'] != unconnected_break_points['contig_id'].shift(1, fill_value=-1), -1, unconnected_break_points['position'] - unconnected_break_points['position'].shift(1, fill_value=0) )
    unconnected_break_points['group'] = ((unconnected_break_points['dist'] > max_break_point_distance) | (-1 == unconnected_break_points['dist'])).cumsum()
    unconnected_break_points = unconnected_break_points.groupby(['contig_id','group'])['position'].agg(['min','max']).reset_index().rename(columns={'min':'bfrom','max':'bto'}).drop(columns=['group'])
    unconnected_break_points = unconnected_break_points.merge(pot_breaks, on=['contig_id'], how='left')
    unconnected_break_points = unconnected_break_points[(unconnected_break_points['position'] >= unconnected_break_points['bfrom'] - max_break_point_distance) & (unconnected_break_points['position'] <= unconnected_break_points['bto'] + max_break_point_distance)].copy()
    
    # If the unconnected break points are connected to another contig on the other side, the extension will likely take care of it and we don't need to do anything
    unconnected_break_points.sort_values(['contig_id','bfrom','bto','side','opos_con'], inplace=True)
    count_breaks = unconnected_break_points.groupby(['contig_id','bfrom','bto','side','opos_con'], sort=False).size()
    unconnected_break_points['ocon_count'] = np.where(unconnected_break_points['opos_con'] < 0, 0, np.repeat(count_breaks.values, count_breaks.values))
    count_breaks = unconnected_break_points.groupby(['contig_id','bfrom','bto','side'], sort=False)['ocon_count'].agg(['max','size'])
    unconnected_break_points['ocon_count'] = np.repeat(count_breaks['max'].values, count_breaks['size'].values)
    unconnected_break_points = unconnected_break_points[(unconnected_break_points['ocon_count'] < min_num_reads) & (np.repeat(count_breaks['size'].values, count_breaks['size'].values) >= min_num_reads)].copy()

    return break_groups, spurious_break_indexes, non_informative_mappings, unconnected_break_points

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
    long_range_connections[['scaffold','scaf_pos','reverse']] = scaffold_parts.loc[long_range_connections['conpart'].values,['scaffold','pos','reverse']].values
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

    # Then add scaf_bridges with alternatives
    short_bridges = scaf_bridges.loc[(scaf_bridges['from_alt'] > 1) | (scaf_bridges['to_alt'] > 1), ['from','from_side','to','to_side','mean_dist']].copy()
    short_bridges.rename(columns={'to':'scaf1','to_side':'strand1','mean_dist':'dist1'}, inplace=True)
    short_bridges['strand1'] = np.where(short_bridges['strand1'] == 'l', '+', '-')
    short_bridges['size'] = 2
    short_bridges['org_pos'] = 0
    scaffold_graph = pd.concat([scaffold_graph, short_bridges[['from','from_side','size','org_pos','scaf1','strand1','dist1']]], ignore_index=True)

    # Now remove all the paths that overlap a longer one (for equally long ones just take one of them)
    scaffold_graph['length'] = scaffold_graph['size'] - scaffold_graph['org_pos']
    scaffold_graph.sort_values(['from','from_side']+[col for sublist in [['scaf'+str(s),'strand'+str(s),'dist'+str(s)] for s in range(1,scaffold_graph['size'].max())] for col in sublist], inplace=True)
    scaffold_graph['redundant'] = (scaffold_graph['from'] == scaffold_graph['from'].shift(1, fill_value=-1)) & (scaffold_graph['from_side'] == scaffold_graph['from_side'].shift(1, fill_value=''))
    for s in range(1,scaffold_graph['size'].max()):
        scaffold_graph['redundant'] = scaffold_graph['redundant'] & ( np.isnan(scaffold_graph['scaf'+str(s)]) | ( (scaffold_graph['scaf'+str(s)] == scaffold_graph['scaf'+str(s)].shift(1, fill_value=-1)) & (scaffold_graph['strand'+str(s)] == scaffold_graph['strand'+str(s)].shift(1, fill_value='')) & (scaffold_graph['dist'+str(s)] == scaffold_graph['dist'+str(s)].shift(1, fill_value='')))) 
    scaffold_graph = scaffold_graph[ scaffold_graph['redundant'] == False ].copy()
    scaffold_graph.drop(columns=['redundant','size','org_pos'],inplace=True)

    # Remove overlapping paths with repeated starting scaffold
    scaffold_graph.reset_index(inplace=True, drop=True)
    scaffold_graph['rep_len'] = -1
    reps = []
    for s in range(1,scaffold_graph['length'].max()):
        scaffold_graph.loc[(scaffold_graph['scaf'+str(s)] == scaffold_graph['from']) & (scaffold_graph['strand'+str(s)] == np.where(scaffold_graph['from_side'] == 'r', '+', '-')), 'rep_len'] = s
        reps.append( scaffold_graph[(scaffold_graph['rep_len'] == s) & (scaffold_graph['rep_len']+1 < scaffold_graph['length'])].copy() )
    reps = pd.concat(reps, ignore_index=True)
    if len(reps):
        reps = reps.merge(scaffold_graph.reset_index()[['from','from_side','index']].rename(columns={'index':'red_ind'}), on=['from','from_side'], how='inner') # Add all possibly redundant indexes in scaffold_graph
        reps = reps[scaffold_graph.loc[reps['red_ind'], 'length'].values == reps['length'] - reps['rep_len']].copy() # Filter out all non-redundant entries
        for s in range(1, (reps['length']-reps['rep_len']).max()):
            for rl in range(1, reps['rep_len'].max()):
                if s+rl < reps['length'].max():
                    reps = reps[(reps['rep_len'] != rl) | (reps['length']-rl-1 < s) | ((scaffold_graph.loc[reps['red_ind'], 'scaf'+str(s)].values == reps['scaf'+str(s+rl)]) & (scaffold_graph.loc[reps['red_ind'], 'strand'+str(s)].values == reps['strand'+str(s+rl)]) & (scaffold_graph.loc[reps['red_ind'], 'dist'+str(s)].values == reps['dist'+str(s+rl)]))].copy()
        scaffold_graph.drop(index=np.unique(reps['red_ind'].values), inplace=True)
        scaffold_graph.drop(columns=['rep_len'], inplace=True)

    return scaffold_graph

def FindUniqueExtensions(unique_extensions, potential_paths):
    unique_path = potential_paths.copy()
    if len(unique_path):
        unique_path['unique_len'] = 0
        unique_path['unique'] = True
        unique_path.sort_values(['from','from_side']+[col for sublist in [['scaf'+str(s),'strand'+str(s),'dist'+str(s)] for s in range(1,unique_path['length'].max())] for col in sublist], inplace=True)

        for s in range(1,unique_path['length'].max()):
            # Check if path with s+1 scaffolds is unique for the given direction
            unique_path['unique'] = unique_path['unique'] & (np.isnan(unique_path['scaf'+str(s)]) == False) & ((unique_path['from'] != unique_path['from'].shift(1)) | (unique_path['from_side'] != unique_path['from_side'].shift(1)) |
                                                                                                              ((unique_path['scaf'+str(s)] == unique_path['scaf'+str(s)].shift(1)) & (unique_path['strand'+str(s)] == unique_path['strand'+str(s)].shift(1)) & (unique_path['dist'+str(s)] == unique_path['dist'+str(s)].shift(1))))
            uniqueness = unique_path.groupby(['from','from_side'], sort=False)[['unique','scaf'+str(s),'strand'+str(s),'dist'+str(s)]].agg({'unique':['sum','size'], ('scaf'+str(s)):['first'], ('strand'+str(s)):['first'], ('dist'+str(s)):['first']}).reset_index()
            uniqueness['identical'] = uniqueness[('unique','sum')]
            uniqueness = uniqueness.drop(columns=('unique','sum')).droplevel(axis='columns',level=1).rename(columns={'unique':'size',('scaf'+str(s)):'to', ('strand'+str(s)):'to_side', ('strand'+str(s)):'to_side', ('dist'+str(s)):'dist'})
            uniqueness['to_side'] = np.where(uniqueness['to_side'] == '+', 'l', 'r')
            uniqueness['unique'] = uniqueness['size'] == uniqueness['identical']
            # Update unique_path for the next s (if the path is already not unique at length s+1 it cannot be at length s+2)
            unique_path['unique'] = np.repeat(uniqueness['unique'].values, uniqueness['size'].values)
            unique_path.loc[unique_path['unique'], 'unique_len'] += 1

        unique_path = unique_path[unique_path['unique_len'] > 0].copy()

    if len(unique_path):
        unique_path = unique_path.groupby(['from','from_side'], sort=False).first().reset_index() # All path have the same values for the unique length, so just take the first path
        unique_path = unique_path[['from','from_side','unique_len'] + [col for sublist in [['scaf'+str(s),'strand'+str(s),'dist'+str(s)] for s in range(1,unique_path['unique_len'].max()+1)] for col in sublist]].copy()
        for s in range(1,unique_path['unique_len'].max()+1):
            unique_path.loc[unique_path['unique_len'] < s, ['scaf'+str(s),'strand'+str(s),'dist'+str(s)]] = np.nan
        unique_path = unique_path[(unique_path['unique_len'] > 1) | (unique_path['scaf1'] != unique_path['from'])].copy() # Remove simple repeats
        unique_extensions.append(unique_path.copy())

    return unique_extensions

def FindBubbles(scaffold_graph, ploidy):
    all_paths = scaffold_graph.copy() # All path that have not been handled or dismissed already
    bubbles = []
    loops = []
    unique_extensions = []
    repeated_scaffolds = []
#
    cs = 1 # Current scaffold position for extension
    while len(all_paths):
        # Store and remove loops (so that we find the start and end for unique loops)
        all_paths['loop_len'] = -1
        for s in range(1,all_paths['length'].max()):
            all_paths.loc[(all_paths['scaf'+str(s)] == all_paths['from']) & (all_paths['strand'+str(s)] == np.where(all_paths['from_side'] == 'r', '+', '-')), 'loop_len'] = s-1
        loop_path = all_paths[all_paths['loop_len'] >= 0].copy()
        if len(loop_path):
            loop_path = loop_path[['from','from_side','loop_len']+[col for sublist in [['scaf'+str(s),'strand'+str(s),'dist'+str(s)] for s in range(1,loop_path['loop_len'].max()+2)] for col in sublist]].copy()
            for s in range(1,loop_path['loop_len'].max()+2):
                loop_path.loc[loop_path['loop_len']+1 < s, ['scaf'+str(s),'strand'+str(s),'dist'+str(s)]] = np.nan # Keep the repeated scaffold on both sides to have the full distance information
            loop_path.drop_duplicates(inplace=True)
            loop_path = loop_path[loop_path['from_side'] == 'r'].copy()
            loops.append(loop_path.copy())
            repeated_scaffolds = np.unique(np.concatenate([repeated_scaffolds, loop_path['from'].values]).astype(int))
            all_paths = all_paths[all_paths['loop_len'] == -1].copy()
            all_paths.drop(columns=['loop_len'], inplace=True)
#
        # Find scaffolds that appear in all path from a given start contig (requirement for unique path and bubble)
        all_paths.sort_values(['from','from_side'], inplace=True)
        all_paths['path_id'] = range(len(all_paths))
        alternatives = all_paths.groupby(['from','from_side']).size()
        all_paths['alts'] = np.repeat(alternatives.values, alternatives.values)
#
        path_long_format = []
        if len(all_paths):
            for s in range(1,all_paths['length'].max()):
                path_long_format.append(all_paths.loc[np.isnan(all_paths['scaf'+str(s)]) == False, ['from','from_side','path_id','alts','scaf'+str(s),'strand'+str(s)]].rename(columns={'scaf'+str(s):'to', 'strand'+str(s):'to_side'}))
                path_long_format[-1]['pos'] = s
            path_long_format = pd.concat(path_long_format, ignore_index=True)
            path_long_format['to'] = path_long_format['to'].astype(int)
            path_long_format['to_side'] = np.where(path_long_format['to_side'] == '+', 'l', 'r')
            # In case the scaffold is circular only take the first appearance of each subscaffold (repeats will be filtered at a later stage)
            path_long_format = path_long_format.groupby(['from','from_side','alts','to','to_side','path_id'])['pos'].min().reset_index(name='pos')
#
        if len(path_long_format):
            bubble_candidates = path_long_format.groupby(['from','from_side','alts','to','to_side']).size().reset_index(name='count')
             # All paths have this scaffold in this orientation ('to', 'to_side') in it
            bubble_candidates = bubble_candidates[(bubble_candidates['count'] == bubble_candidates['alts']) & (np.isin(bubble_candidates['from'], repeated_scaffolds) == False) & (np.isin(bubble_candidates['to'], repeated_scaffolds) == False)].copy()
        else:
            bubble_candidates = []
#
        if len(bubble_candidates):
            bubble_candidates.drop(columns=['count','alts'], inplace=True)
            # Check that the connection exists in both directions
            bubble_candidates = bubble_candidates.merge(bubble_candidates.rename(columns={'from':'to','from_side':'to_side','to':'from','to_side':'from_side'}), on=['from','from_side','to','to_side'], how='inner')
#
        if len(bubble_candidates):
            # Add path information
            bubble_candidates = bubble_candidates.merge(path_long_format.drop(columns=['alts']), on=['from','from_side','to','to_side'], how='left')
            bubble_candidates = bubble_candidates.merge(all_paths.drop(columns=['length','alts']), on=['from','from_side','path_id'], how='left')
            bubble_candidates['last_dist'] = 0
            for s in range(1,all_paths['length'].max()):
                bubble_candidates.loc[s == bubble_candidates['pos'],'last_dist'] = bubble_candidates['dist'+str(s)]
                bubble_candidates.loc[bubble_candidates['pos'] <= s, ['scaf'+str(s),'strand'+str(s),'dist'+str(s)]] = np.nan
                if len(bubble_candidates) == np.sum(np.isnan(bubble_candidates['scaf'+str(s)])):
                    bubble_candidates.drop(columns=[col for sublist in [['scaf'+str(s),'strand'+str(s),'dist'+str(s)] for s in range(s,all_paths['length'].max())] for col in sublist], inplace=True)
                    break
            bubble_candidates['last_dist'] = bubble_candidates['last_dist'].astype(int)
            # Remove conflicting bubbles (bubbles that have different orders depending on the path taken)
            bpaths = path_long_format[np.isin(path_long_format['path_id'], bubble_candidates['path_id'])].drop(columns=['alts']) # Get full path for all bubbles
            bpaths = bpaths.merge(bubble_candidates[['from','from_side','to','to_side']].drop_duplicates(), on=['from','from_side','to','to_side'], how='inner').sort_values(['from','from_side','path_id','pos']) # Only take the bubble ends (but also in the other paths, where they aren't the end)
            conflicts = [-1]
            while len(conflicts):
                tmp_candidates = bpaths.groupby(['from','from_side','path_id'], sort=False).first().reset_index().drop(columns=['path_id','pos']).drop_duplicates() # Get first bubble in every path
                conflicts = tmp_candidates.groupby(['from','from_side'], sort=False).size().values
                tmp_candidates['conflicts'] = np.repeat(conflicts, conflicts) - 1 # Conflicts are the number of endings - 1, because a single ending is consistent with itself
                conflicts = tmp_candidates[tmp_candidates['conflicts'] > 0].copy()
                bpaths = bpaths[bpaths.merge(conflicts[['from','from_side','to','to_side']].drop_duplicates(), on=['from','from_side','to','to_side'], how='left', indicator=True)['_merge'].values == 'left_only'] # Remove conflicts
                bpaths = bpaths[bpaths.merge(conflicts[['from','from_side','to','to_side']].drop_duplicates().rename(columns={'from':'to','from_side':'to_side','to':'from','to_side':'from_side'}), on=['from','from_side','to','to_side'], how='left', indicator=True)['_merge'].values == 'left_only'] # Remove conflicts
            # Remove the conflicts from bubble_candidates
            bubble_candidates = bubble_candidates.merge(bpaths[['from','from_side','to','to_side']].drop_duplicates(), on=['from','from_side','to','to_side'], how='inner')
#
        if len(bubble_candidates):
            # Remove duplicates
            bubble_candidates.drop(columns=['path_id'], inplace=True)
            bubble_candidates.drop_duplicates(inplace=True)
            # Remove candidates that are only present in one direction (Should not happen, but if it happens we don't introduce missassemblies, because it still is a bubble)
            bubble_candidates = bubble_candidates.merge(bubble_candidates[['from','from_side','to','to_side','pos']].drop_duplicates().rename(columns={'from':'to','from_side':'to_side','to':'from','to_side':'from_side'}), on=['from','from_side','to','to_side','pos'], how='inner')
            if len(bubble_candidates):
                bubble_candidates['bid'] = np.arange(len(bubble_candidates)) # Assign an id to identical connections independent of the direction
                rev_bubbles = bubble_candidates.rename(columns={'bid':'bid2'})
                rev_bubbles[['from','from_side','to','to_side']] = bubble_candidates[['to','to_side','from','from_side']]
                rev_bubbles.loc[1<rev_bubbles['pos'],'last_dist'] = bubble_candidates.loc[1<bubble_candidates['pos'],'dist1'].astype(int)
                for cpos in range(2,rev_bubbles['pos'].max()+1):
                    for s in range(1,cpos):
                        rev_bubbles.loc[rev_bubbles['pos'] == cpos, ['scaf'+str(s),'strand'+str(s),'dist'+str(s)]] = bubble_candidates.drop(columns=['dist'+str(cpos)], errors='ignore').rename(columns={'last_dist':('dist'+str(cpos))}).loc[bubble_candidates['pos'] == cpos, ['scaf'+str(cpos-s),'strand'+str(cpos-s),'dist'+str(cpos-s+1)]].values
                        rev_bubbles.loc[rev_bubbles['pos'] == cpos, 'strand'+str(s)] = np.where(rev_bubbles.loc[rev_bubbles['pos'] == cpos, 'strand'+str(s)] == '+', '-', '+')
                bubble_candidates = bubble_candidates.merge(rev_bubbles, on=bubble_candidates.columns.tolist()[:-1], how='inner')
                bubble_candidates['bid'] = np.minimum(bubble_candidates['bid'],bubble_candidates['bid2'])
                bubble_candidates.drop(columns=['bid2'], inplace=True)
#
        if len(bubble_candidates):
            # Only take the first bubble (position wise) to have less duplicates (we can later stich them back together)
            bubble_candidates.sort_values(['from','from_side','to','to_side'], inplace=True)
            min_pos = bubble_candidates.groupby(['from','from_side'], sort=False)['pos'].agg(['min','size'])
            bubble_candidates['min_pos_group'] = np.repeat(min_pos['min'].values, min_pos['size'].values)
            min_pos = bubble_candidates.groupby(['from','from_side','to','to_side'], sort=False)['pos'].agg(['min','size'])
            bubble_candidates['min_pos_bubble'] = np.repeat(min_pos['min'].values, min_pos['size'].values)
            bubble_candidates = bubble_candidates[bubble_candidates['min_pos_group'] == bubble_candidates['min_pos_bubble']].drop(columns=['min_pos_group','min_pos_bubble'])
            # Remove bubbles which lost their counterpart (Can happen when long reads form a bubble simply by being longer than other reads and therefore no alternative path starting at the same scaffold exists)
            remove_bubbles = bubble_candidates.groupby(['bid']).size()
            remove_bubbles = bubble_candidates.merge(remove_bubbles[remove_bubbles != 2].reset_index()[['bid']], on=['bid'], how='inner')
            bubble_candidates = bubble_candidates[np.isin(bubble_candidates['bid'],remove_bubbles.loc[remove_bubbles['from'] != remove_bubbles['to'],'bid'].values) == False].copy()
            # Get true amount of alternatives in bubble (not overall amount of alternatives for that start scaffold)
            alternatives = bubble_candidates.groupby(['from','from_side','to','to_side'], sort=False).size().values
            bubble_candidates['alts'] = np.repeat(alternatives, alternatives)
#
        # Store detected bubbles
        if len(bubble_candidates):
            bubble_candidates = bubble_candidates[['from','from_side','to','to_side','last_dist','pos','alts'] + [col for sublist in [['scaf'+str(s),'strand'+str(s),'dist'+str(s)] for s in range(1,bubble_candidates['pos'].max())] for col in sublist]].rename(columns={'pos':'length'})
            bubble_candidates['length'] -= 1
            bubbles.append(bubble_candidates.copy())
#
            # Remove paths where we already found a bubble
            all_paths = all_paths[ (all_paths[['from','from_side']].merge(pd.concat([bubble_candidates[['from','from_side']], bubble_candidates[['to','to_side']].rename(columns={'to':'from','to_side':'from_side'})], ignore_index=True).drop_duplicates(), on=['from','from_side'], how='left', indicator=True)['_merge'] == "left_only").values ].copy()
#
        # Dismiss paths that have more alternatives than ploidy
        if len(all_paths):
            all_paths['remove'] = all_paths['alts'] > ploidy
            unique_extensions = FindUniqueExtensions(unique_extensions, all_paths[all_paths['remove']])
            all_paths = all_paths[all_paths['remove'] == False].copy()
#
        # Only extend paths if the current scaffold (cs) hasn't been in the path before (otherwise we would run into loops)
        if len(all_paths):
            all_paths['mscaf'] = all_paths['scaf'+str(cs)].fillna(-1).astype(int)
            for s in range(1, cs):
                all_paths.loc[(all_paths['scaf'+str(cs)] == all_paths['scaf'+str(s)]) & (all_paths['strand'+str(cs)] == all_paths['strand'+str(s)]), 'mscaf'] = -2
            all_paths['mside'] = np.where(all_paths['strand'+str(cs)] == '+', 'r', 'l')
            all_paths = all_paths.merge(scaffold_graph[['from','from_side','length'] + [col for sublist in [['scaf'+str(s),'strand'+str(s),'dist'+str(s)] for s in range(1,scaffold_graph['length'].max())] for col in sublist]].rename(columns={**{'from':'mscaf', 'from_side':'mside', 'length':'new_len'}, **{('scaf'+str(s)):('cscaf'+str(s+cs)) for s in range(scaffold_graph['length'].max())}, **{('strand'+str(s)):('cstrand'+str(s+cs)) for s in range(scaffold_graph['length'].max())}, **{('dist'+str(s)):('cdist'+str(s+cs)) for s in range(scaffold_graph['length'].max())}}), on=['mscaf','mside'], how='left')
            all_paths['ext_id'] = range(len(all_paths))
#
            if np.isnan(all_paths['new_len'].max()):
                all_paths['remove'] = True
                all_paths.drop(columns=[col for col in all_paths.columns if (col[:5] == 'cscaf') or (col[:7] == 'cstrand') or (col[:5] == 'cdist')], inplace=True)
                all_paths.drop(columns=['ext_id'], inplace=True)
            else:
                # If the current scaffold is repeated later on, try the extension for all and take the first that works
                rep_extensions = [all_paths[['ext_id']].copy()]
                rep_extensions[0]['shift'] = 0
                for s in range(cs+1, all_paths['length'].max()):
                    rep_extensions.append( all_paths.loc[(all_paths['mscaf'] == all_paths['scaf'+str(s)]) & (all_paths['strand'+str(cs)] == all_paths['strand'+str(s)]), ['ext_id']].copy() )
                    rep_extensions[-1]['shift'] = s-cs
                rep_extensions = pd.concat(rep_extensions, ignore_index=True)
                all_paths = all_paths.merge(rep_extensions, on=['ext_id'], how='left')
#
                # Create additional columns if we need them
                all_paths['new_len'] += cs
                for s in range(all_paths['length'].max(), (all_paths['new_len']+all_paths['shift']).max().astype(int)):
                    all_paths['scaf'+str(s)] = np.nan 
                    all_paths['strand'+str(s)] = np.nan
                    all_paths['dist'+str(s)] = np.nan
#
                all_paths.loc[np.isnan(all_paths['new_len']), 'remove'] = True # If we don't have an extension remove the path
                for s in range(cs+1,all_paths['new_len'].max().astype(int)):
                    for shift in range(all_paths['shift'].max()+1):
                        if s+shift < (all_paths['new_len']+all_paths['shift']).max().astype(int):
                            # If the extension diverges from the path on the overlap remove it
                            all_paths.loc[(all_paths['shift'] == shift) & (np.isnan(all_paths['scaf'+str(s+shift)]) == False) & (np.isnan(all_paths['cscaf'+str(s)]) == False) & ((all_paths['scaf'+str(s+shift)] != all_paths['cscaf'+str(s)]) | (all_paths['strand'+str(s+shift)] != all_paths['cstrand'+str(s)]) | (all_paths['dist'+str(s+shift)] != all_paths['cdist'+str(s)])), 'remove'] = True
                            # Update extended values
                            all_paths.loc[(all_paths['shift'] == shift) & (all_paths['remove'] == False) & np.isnan(all_paths['scaf'+str(s+shift)]) & (np.isnan(all_paths['cscaf'+str(s)]) == False), ['strand'+str(s+shift),'scaf'+str(s+shift),'dist'+str(s+shift)]] = all_paths.loc[(all_paths['shift'] == shift) & (all_paths['remove'] == False) & np.isnan(all_paths['scaf'+str(s+shift)]) & (np.isnan(all_paths['cscaf'+str(s)]) == False), ['cstrand'+str(s), 'cscaf'+str(s), 'cdist'+str(s)]].values
#
                all_paths['new_len'] += all_paths['shift'] # Add this only after the loop so that we don't increase the maximum for s in the loop, because we separately add the shift there
                all_paths.drop(columns=[col for col in all_paths.columns if (col[:5] == 'cscaf') or (col[:7] == 'cstrand') or (col[:5] == 'cdist')], inplace=True)
#
                # Take for every repeated scaffold the first that does not diverge from the path or the last if all diverge
                all_paths['keep'] = all_paths['remove'] == False
                all_paths['keep'] = all_paths.groupby(['ext_id'])['keep'].cumsum()-all_paths['keep']
                all_paths = all_paths[all_paths['keep'] == 0].copy()
                all_paths = all_paths.groupby(['ext_id']).last().reset_index(drop=True)
                all_paths.drop(columns=['shift','keep'], inplace=True)
#
                # Finish the removal selection
                all_paths.loc[all_paths['new_len'] <= all_paths['length'], 'remove'] = True # If we don't gain length it is not an extension
            all_paths['extended'] = all_paths['remove'] == False
            all_paths.loc[(all_paths['length'] > cs+1), 'remove'] = False # If we have possibilities for extensions later on, we don't remove the path
            all_paths.drop(columns=['mscaf','mside'], inplace=True)
            all_paths.drop_duplicates(inplace=True)
            removals = all_paths.groupby(['from','from_side'], sort=False)['remove'].agg(['sum','size'])
            all_paths['remove'] = np.repeat( removals['sum'].values == removals['size'].values, removals['size'].values ) # Only remove a path if all path from this starting scaffold are removed
            cs += 1
#
            # Remove non-extendable paths
            unique_extensions = FindUniqueExtensions(unique_extensions, all_paths[all_paths['remove']])
            all_paths = all_paths[all_paths['remove'] == False].copy()
            all_paths.loc[all_paths['extended'], 'length'] = all_paths.loc[all_paths['extended'], 'new_len'].astype(int)
            ext_count = all_paths.groupby(['path_id'])['extended'].agg(['sum','size'])
            all_paths = all_paths[all_paths['extended'] | np.repeat(ext_count['sum'].values == 0, ext_count['size'].values)].copy() # If we have extensions for a given path, drop the failed extension attempts 
            all_paths.drop(columns=['path_id','alts','remove','new_len','extended'], inplace=True)
            all_paths.drop_duplicates(inplace=True)
#
    bubbles = pd.concat(bubbles, ignore_index=True)
    bubbles.sort_values(['from','from_side','to','to_side'], inplace=True)
    bubbles.drop_duplicates(inplace=True)
#
    unique_extensions = pd.concat(unique_extensions, ignore_index=True)
    unique_extensions.sort_values(['from','from_side'], inplace=True)
#
    # Insert reversed versions of loops
    loops = pd.concat(loops, ignore_index=True)
    loops['from_side'] = 'r'
    rev_path = loops.copy()
    rev_path['from_side'] = 'l'
    for l in range(1, rev_path['loop_len'].max()+1):
        rev_path.loc[rev_path['loop_len'] == l, 'dist'+str(l+1)] = loops.loc[loops['loop_len'] == l, 'dist1']
        for s in range(1, l+1):
            rev_path.loc[rev_path['loop_len'] == l, 'scaf'+str(s)] = loops.loc[loops['loop_len'] == l, 'scaf'+str(l+1-s)]
            rev_path.loc[rev_path['loop_len'] == l, 'strand'+str(s)] = np.where(loops.loc[loops['loop_len'] == l, 'strand'+str(l+1-s)] == '+', '-', '+')
            rev_path.loc[rev_path['loop_len'] == l, 'dist'+str(s)] = loops.loc[loops['loop_len'] == l, 'dist'+str(l+2-s)]
    loops = pd.concat([loops, rev_path], ignore_index=True)
    loops.sort_values(['from','from_side'], inplace=True)
    loops.drop_duplicates(inplace=True)
#
    return bubbles, loops, unique_extensions

def ResolveLoops(loops, unique_paths, bubbles, unique_extensions):
    ## Check if we can traverse loops and update unique_paths, bubbles and unique_extensions accordingly
    # Group connected loops
    conns = []
    conns.append(loops[['id', 'from']].rename(columns={'id':'conn', 'from':'scaffold'}))
    for s in range(1,loops['loop_len'].max()+1):
        conns.append(loops.loc[np.isnan(loops['scaf'+str(s)]) == False, ['id', 'scaf'+str(s)]].rename(columns={'id':'conn', ('scaf'+str(s)):'scaffold'}))
        conns[-1]['scaffold'] = conns[-1]['scaffold'].astype(int)
    conns = pd.concat(conns, ignore_index=True).drop_duplicates()
    conns.sort_values(['conn'], inplace=True)
    loops['loop'] = 0
    loops['new_loop'] = range(len(loops))
    while np.sum(loops['loop'] != loops['new_loop']):
        loops['loop'] = loops['new_loop']
        loops.drop(columns=['new_loop'], inplace=True)
        tmp_conns = conns.rename(columns={'scaffold':'from'}).merge(loops[['from','loop']], on=['from'], how='left')
        min_loop = tmp_conns.groupby(['conn'], sort=False)['loop'].agg(['min','size'])
        tmp_conns['loop'] = np.repeat(min_loop['min'].values.astype(int), min_loop['size'].values)
        loops = loops.merge(tmp_conns.groupby(['from'])['loop'].min().reset_index(name='new_loop'), on=['from'], how='left')
    
    loops.drop(columns=['new_loop'], inplace=True)
    loops.sort_values(['loop','from','from_side'], inplace=True)
    loop_scaffolds = tmp_conns[['loop','from']].drop_duplicates().sort_values(['loop','from']).rename(columns={'from':'scaffold'})
    unique_loop_exit = [ loop_scaffolds.merge(unique_paths.reset_index(), left_on=['scaffold'], right_on=['from'], how='inner') ]
    for s in range(1,unique_paths['length'].max()+1):
        unique_loop_exit.append( loop_scaffolds.merge(unique_paths.reset_index(), left_on=['scaffold'], right_on=['scaf'+str(s)], how='inner') )
    unique_loop_exit = pd.concat(unique_loop_exit, ignore_index=True).drop(columns=['scaffold']).sort_values(['loop','from','from_side','to','to_side']).drop_duplicates()
    loops_with_unique_exit = np.unique(unique_loop_exit['loop'].values)
        
    bubble_loop_exit = [ loop_scaffolds.merge(bubbles.reset_index(), left_on=['scaffold'], right_on=['from'], how='inner') ]
    for s in range(1,bubbles['length'].max()+1):
        bubble_loop_exit.append( loop_scaffolds.merge(bubbles.reset_index(), left_on=['scaffold'], right_on=['scaf'+str(s)], how='inner') )
    bubble_loop_exit = pd.concat(bubble_loop_exit, ignore_index=True).drop(columns=['scaffold']).sort_values(['loop','from','from_side','to','to_side']).drop_duplicates()
    loops_with_bubble_exit = np.unique(bubble_loop_exit['loop'].values)
    
    unique_loop_exit['has_bubble'] = np.isin(unique_loop_exit['loop'], loops_with_bubble_exit)
    unique_loop_exit['from_in_loop']  = (unique_loop_exit[['loop','from']].merge(loop_scaffolds.rename(columns={'scaffold':'from'}), on=['loop','from'], how='left', indicator=True)['_merge'] == "both").values
    unique_loop_exit['to_in_loop']  = (unique_loop_exit[['loop','to']].merge(loop_scaffolds.rename(columns={'scaffold':'to'}), on=['loop','to'], how='left', indicator=True)['_merge'] == "both").values

    return unique_paths, bubbles, unique_extensions

def GroupConnectedScaffoldsIntoKnots(bubbles, scaffolds, scaffold_parts, contig_parts):
    # Create copies, so that we don't modify the original
    cbubbles = bubbles.copy()

    # Group all connected scaffolds
    conns = []
    if len(cbubbles):
        cbubbles['id'] = range(len(cbubbles))
        conns.append(cbubbles[['id', 'from']].rename(columns={'id':'conn', 'from':'scaffold'}))
        conns.append(cbubbles[['id', 'to']].rename(columns={'id':'conn', 'to':'scaffold'}))
        for s in range(1,cbubbles['length'].max()+1):
            conns.append(cbubbles.loc[np.isnan(cbubbles['scaf'+str(s)]) == False, ['id', 'scaf'+str(s)]].rename(columns={'id':'conn', ('scaf'+str(s)):'scaffold'}))
            conns[-1]['scaffold'] = conns[-1]['scaffold'].astype(int)

    knots = scaffolds[['scaffold']].copy()
    if len(conns) == 0:
        knots['knot'] = range(len(knots))
    else:
        conns = pd.concat(conns, ignore_index=True)
        conns.sort_values(['conn'], inplace=True)
        knots['knot'] = 0
        knots['new_knot'] = range(len(knots))
        while np.sum((np.isnan(knots['new_knot']) == False) & (knots['knot'] != knots['new_knot'])):
            knots.loc[np.isnan(knots['new_knot']) == False, 'knot'] = knots.loc[np.isnan(knots['new_knot']) == False, 'new_knot'].astype(int)
            knots.drop(columns=['new_knot'], inplace=True)
            tmp_conns = conns.merge(knots, on=['scaffold'], how='left')
            min_knot = tmp_conns.groupby(['conn'], sort=False)['knot'].agg(['min','size'])
            tmp_conns['knot'] = np.repeat(min_knot['min'].values, min_knot['size'].values)
            knots = knots.merge(tmp_conns.groupby(['scaffold'])['knot'].min().reset_index(name='new_knot'), on=['scaffold'], how='left')
        knots.drop(columns=['new_knot'], inplace=True)
        
    # Get scaffold sizes (in nucleotides)
    conlen = pd.DataFrame({'conpart':range(len(contig_parts)), 'length':(contig_parts['end'].values - contig_parts['start'].values)})
    knots = knots.merge(scaffold_parts[['conpart','scaffold']].merge(conlen.rename(columns={'contig':'conpart'}), on=['conpart'], how='left').groupby(['scaffold'])['length'].sum().reset_index(name='scaf_len'), on=['scaffold'], how='left')
    knots.sort_values(['knot','scaf_len'], ascending=[True,False], inplace=True)

    return knots

def UnreelBubbles(bubbles):
    # Prepare bubbles for better extension (strands are defined '+' if the right side of the scaffold points in the direction of the extension)
    bubbles.drop(columns=['alts'], inplace=True)
    bubbles.rename(columns={'from_side':'from_strand','to_side':'to_strand'}, inplace=True)
    bubbles['from_strand'] = np.where(bubbles['from_strand'] == 'r', '+', '-')
    bubbles['to_strand'] = np.where(bubbles['to_strand'] == 'l', '+', '-')
    phasing_connections = bubbles[['from','from_strand','to','to_strand']].drop_duplicates()
    # Add 'to' to the end of the path
    max_len=bubbles['length'].max()+1
    bubbles[['scaf'+str(max_len),'strand'+str(max_len),'dist'+str(max_len)]] = np.nan # Make sure we have enough columns
    bubbles['length'] += 1
    for cl in range(bubbles['length'].min(), max_len+1):
        bubbles.loc[bubbles['length'] == cl, 'scaf'+str(cl)] = bubbles.loc[bubbles['length'] == cl, 'to']
        bubbles.loc[bubbles['length'] == cl, 'strand'+str(cl)] = bubbles.loc[bubbles['length'] == cl, 'to_strand']
        bubbles.loc[bubbles['length'] == cl, 'dist'+str(cl)] = bubbles.loc[bubbles['length'] == cl, 'last_dist']
    bubbles.drop(columns=['to','to_strand','last_dist'], inplace=True)
#
    return bubbles, phasing_connections

def TrimConsistentFromPath(new_alters):
    # Trim scaffolds still in the starting alternative from path
    for c in range(new_alters['consistent'].min(), new_alters['consistent'].max()+1):
        max_len = new_alters.loc[new_alters['consistent'] == c, 'length'].max()
        if np.isnan(max_len) == False:
            for s in range(1, int(max_len)+1):
                if s+c > max_len:
                    new_alters.loc[new_alters['consistent'] == c, ['scaf'+str(s),'strand'+str(s),'dist'+str(s)]] = np.nan
                else:
                    new_alters.loc[new_alters['consistent'] == c, ['scaf'+str(s),'strand'+str(s),'dist'+str(s)]] = new_alters.loc[new_alters['consistent'] == c, ['scaf'+str(s+c),'strand'+str(s+c),'dist'+str(s+c)]].values
    new_alters['length'] -= new_alters['consistent']

    return new_alters

def CombineBubblesIntoPaths(bubbles, phasing_connections, knots, scaf_bridges, ploidy):
    # Start from the longest scaffold in each knot and extend it to both sides as much as possible and repeat this until all scaffolds have been addressed
    cknots = knots.copy()
    final_paths = []
    final_alternatives = []
    included_scaffolds = np.array([], dtype=int)
    next_free_alt_id = 0
    while len(cknots):
        cur_start = cknots.groupby(['knot']).first().reset_index()
        cur_paths = pd.DataFrame({'center':cur_start['scaffold'].values, 'pos':0, 'scaffold':cur_start['scaffold'].values, 'strand':'+', 'distance':0})
        cur_alters = []
#
        # Extend to both sides
        for direction in ["left","right"]:
            ext = pd.DataFrame({'center':cur_start['scaffold'].values, 'pos':0, 'len':1, 'solid':1}).sort_values(['center'])
            while len(ext):
                # Store current values for next iteration before we change them
                new_paths = [cur_paths.copy()]
                if len(cur_alters):
                    new_alters = [cur_alters.copy()]
                else:
                    new_alters = []
                # Update values so that they are behaving the same no matter what the direction is
                if direction == "left":
                    cur_paths['pos'] *= -1
                    cur_paths['strand'] = np.where(cur_paths['strand'] == '+', '-', '+')
                    if len(cur_alters):
                        cur_alters[['start_pos','end_pos']] *= -1
                        cur_alters['strand'] = np.where(cur_alters['strand'] == '+', '-', '+')
                if len(cur_alters):
                    cur_alters['alt_pos'] = np.abs(cur_alters['alt_pos']) # Now it behaves the same in the way that alt_pos.max is the last scaffold and overlapping with the main path, but if we would want to check scaffolds along the path, the alt_pos.max is not consistently closest to the current end of the path anymore
                # Insert values for this round
                ext = ext.merge(cur_paths.drop(columns=['distance']), on=['center','pos'], how='left')
#
                ## Check if we can extend with a bubble
                bext = ext.merge(bubbles.rename(columns={'from':'scaffold', 'from_strand':'strand'}), on=['scaffold','strand'], how='inner')
                if len(bext):
                    bext['consistent'] = True # Create this column for later to store if the alternatives are consistent with the path or not (information helps phasing)
                    # Remove bubbles that do not fit the current path
                    bext['new_len'] = bext['pos'] + bext['length'] + 1
                    bext['cur_pos'] = bext['pos']
                    alt_ext = [] # Before removing bubbles that do not fit the current path store them to check if they are from an alternative path
                    for s in range(1,bext['length'].max()+1):
                        bext['cur_pos'] += 1
                        bext = bext.merge(cur_paths[['center', 'pos', 'scaffold', 'strand', 'distance']].rename(columns={'pos':'cur_pos', 'scaffold':'pscaf', 'strand':'pstrand', 'distance':'pdist'}), on=['center','cur_pos'], how='left')
                        valid = np.isnan(bext['pscaf']) | (bext['length'] < s) | ((bext['pscaf'] == bext['scaf'+str(s)]) & (bext['pstrand'] == bext['strand'+str(s)]) & (bext['pdist'] == bext['dist'+str(s)]))
                        bext.drop(columns=['pscaf','pstrand','pdist'], inplace=True)
                        alt_ext.append( bext[valid == False].copy() )
                        bext = bext[valid].copy()
                    bext.drop(columns=['cur_pos'], inplace=True)
                    # Remove alternatives that are inconsistent with current solids
                    alt_ext = pd.concat(alt_ext, ignore_index=True)
                    alt_ext['consistent'] = False
                    alt_ext = alt_ext[alt_ext['cur_pos'] >= alt_ext['solid']].copy() # Remove alternatives that are not consistent with the solid scaffolds (Divergence must be after the last solid scaffold, >= equal solid length)
                    alt_ext = alt_ext.merge(bext[['center']].drop_duplicates(), on=['center'], how='inner') # Remove cases where all possibilities are inconsistent with the main path
                    # Prepare alt_ext for the next section
                    if len(alt_ext):
                        alt_ext = [alt_ext.drop(columns=['cur_pos'])]
                    else:
                        alt_ext = []
                if len(bext):
                    # If a consistent path through a bubble connects with a previous alternative, make it again an alternative (except if all path through a bubble do that)
                    if len(cur_alters):
                        bconnections = []
                        for s in range(1, bext['length'].max()): # Do not use last/to scaffold as this is identical to main path
                            pcons = bext.loc[bext['length'] > s, ['center','scaf'+str(s),'strand'+str(s)]].rename(columns={('scaf'+str(s)):'to', ('strand'+str(s)):'to_strand'}).reset_index()
                            pcons['to'] = pcons['to'].astype(int)
                            pcons = pcons.merge(phasing_connections, on=['to','to_strand'], how='inner')
                            bconnections.append( pcons.merge(cur_alters[['center','scaffold','strand']].rename(columns={'scaffold':'from', 'strand':'from_strand'}), on=['center','from','from_strand'], how='inner')['index'].values )
                        if len(bconnections):
                            bext['alternative'] = False
                            bext.loc[np.unique(np.concatenate(bconnections)), 'alternative'] = True
                            galts = bext.groupby(['center'], sort=False)['alternative'].agg(['min','size'])
                            chosen = (bext['alternative'] == False) | np.repeat(galts['min'].values, galts['size'].values) # Take it as main if it is not connected to an alternative or all paths are connected to an alternative
                            bext.drop(columns=['alternative'], inplace=True)
                            alt_ext.append(bext[chosen == False].copy())
                            bext = bext[chosen].copy()
                    # If some paths through a bubble have a connection pointing forward prefer them over other paths
                    bconnections = []
                    for s in range(1, bext['length'].max()): # Do not use last/to scaffold as this is identical to main path
                        pcons = bext.loc[bext['length'] > s, ['center','scaf'+str(s),'strand'+str(s)]].rename(columns={('scaf'+str(s)):'from', ('strand'+str(s)):'from_strand'}).reset_index()
                        pcons['from'] = pcons['from'].astype(int)
                        bconnections.append( pcons.merge(phasing_connections, on=['from','from_strand'], how='inner')['index'].values )
                    if len(bconnections):
                        bext['connection'] = False
                        bext.loc[np.unique(np.concatenate(bconnections)), 'connection'] = True
                        gcons = bext.groupby(['center'], sort=False)['connection'].agg(['max','size'])
                        chosen = bext['connection'] | np.repeat(gcons['max'].values == False, gcons['size'].values) # Take it as main if it is connected or no paths is connected
                        bext.drop(columns=['connection'], inplace=True)
                        alt_ext.append(bext[chosen == False].copy())
                        bext = bext[chosen].copy()
                    # In the cases, where connections cannot decide, pick the one with more reads supporting the bridge as main path (first bridge, last bridge)
                    bext[['from','to','mean_dist']] = bext[['scaffold','scaf1','dist1']].values.astype(int)
                    bext['from_side'] = np.where(bext['strand'] == '+', 'r', 'l')
                    bext['to_side'] = np.where(bext['strand1'] == '+', 'l', 'r')
                    bext = bext.merge(scaf_bridges[['from','from_side','to','to_side','mean_dist','bcount']], on=['from','from_side','to','to_side','mean_dist'], how='left')
                    bcounts = bext.groupby(['center'], sort=False)['bcount'].agg(['max','size'])
                    chosen = bext['bcount'] == np.repeat(bcounts['max'].values, bcounts['size'].values)
                    bext.drop(columns=['bcount'], inplace=True)
                    alt_ext.append(bext[chosen == False].drop(columns=['from','from_side','to','to_side','mean_dist']))
                    bext = bext[chosen].copy()
                    for s in range(2,bext['length'].max()+1): # If length == 1 from and to have not changed compared to the last comparison
                        bext.loc[s == bext['length'], ['from','to','mean_dist']] = bext.loc[s == bext['length'], ['scaf'+str(s-1),'scaf'+str(s),'dist'+str(s)]].values.astype(int)
                        bext.loc[s == bext['length'], 'from_side'] = np.where(bext.loc[s == bext['length'], 'strand'+str(s-1)] == '+', 'r', 'l')
                        bext.loc[s == bext['length'], 'to_side'] = np.where(bext.loc[s == bext['length'], 'strand'+str(s)] == '+', 'l', 'r')
                    bext = bext.merge(scaf_bridges[['from','from_side','to','to_side','mean_dist','bcount']], on=['from','from_side','to','to_side','mean_dist'], how='left')
                    bcounts = bext.groupby(['center'], sort=False)['bcount'].agg(['max','size'])
                    chosen = bext['bcount'] == np.repeat(bcounts['max'].values, bcounts['size'].values)
                    bext.drop(columns=['from','from_side','to','to_side','mean_dist','bcount'], inplace=True)
                    alt_ext.append(bext[chosen == False].copy())
                    bext = bext[chosen].copy()
                    # If bridges are equal take the ones including more scaffolds
                    max_len = bext.groupby(['center'], sort=False)['length'].agg(['max','size'])
                    chosen = bext['length'] == np.repeat(max_len['max'].values, max_len['size'].values)
                    alt_ext.append(bext[chosen == False].copy())
                    bext = bext[chosen].copy()
                    # If there are still multiple options just take the first
                    chosen = 0 == bext.groupby(['center'], sort=False).cumcount()
                    alt_ext.append(bext[chosen == False].copy())
                    bext = bext[chosen].copy()
                    # Check if the extension is already included (so that we do not get stuck in a loop)
                    bext['included'] = (bext['new_len'] > bext['len']) # Keep the once that are shorter than len (which are necessarily already included)
                    for s in range(1,bext['length'].max()+1):
                        bext.loc[bext['included'] & (bext['length'] >= s), 'included'] = bext.loc[bext['included'] & (bext['length'] >= s), ['center','scaf'+str(s),'strand'+str(s)]].rename(columns={('scaf'+str(s)):'scaffold', ('strand'+str(s)):'strand'}).merge(cur_paths[['center','scaffold','strand']].drop_duplicates(), on=['center','scaffold','strand'], how='left', indicator=True)['_merge'].values == "both"
                    bext = bext[ bext['included'] == False ].copy()
                    # Join alternative extensions and get end_pos from the main alternative
                    alt_ext = pd.concat(alt_ext, ignore_index=True).sort_values(['center'])
                    alt_ext = alt_ext.merge(bext[['center','new_len']].rename(columns={'new_len':'end_pos'}), on=['center'], how='inner')
                if len(bext):
                    # Turn cur_path solid up to end of bubbles if they do not start within a bubble themselves
                    pot_solids = bext[['center','new_len','pos']].copy()
                    if len(cur_alters):
                        # Filter out the unique extensions starting within a bubble
                        pot_solids = pot_solids.merge(cur_alters[['center','end_pos']].drop_duplicates(), on=['center'], how='left')
                        pot_solids.sort_values(['center','end_pos'], inplace=True)
                        pot_solids = pot_solids.groupby(['center']).last().reset_index() # Only take the last bubbles for comparison
                        pot_solids = pot_solids[(pot_solids['pos'] >= pot_solids['end_pos']-1) | np.isnan(pot_solids['end_pos'])].copy() # The last scaffold of an alternative is the same as the main just with a different distance to the previous one, thus a solid extension is allowed to start from there              
                    ext = ext.merge(pot_solids[['center','new_len']], on=['center'], how='left')
                    ext.loc[np.isnan(ext['new_len']) == False, 'solid'] = np.maximum(ext.loc[np.isnan(ext['new_len']) == False, 'solid'], ext.loc[np.isnan(ext['new_len']) == False, 'new_len'].astype(int))
                    ext.drop(columns=['new_len'], inplace=True)
                    # Insert new scaffolds into path
                    bext = bext[ bext['new_len'] > bext['len'] ].copy() # Remove extensions that are shorter or equally long as the current path (don't extend) (Do not do this earlier, because it may lead to inconsistencies when having multiple alternatives, such that we have both, consistent alternatives that are shorter and longer than path )
                    if len(bext):
                        for s in range(1,bext['length'].max()+1):
                            bext['pos'] += 1
                            new_paths.append(bext.loc[(bext['length'] >= s) & (bext['pos'] >= bext['len']), ['center','pos','scaf'+str(s),'strand'+str(s),'dist'+str(s)]].rename(columns={('scaf'+str(s)):'scaffold', ('strand'+str(s)):'strand', ('dist'+str(s)):'distance'}))
                            if len(new_paths[-1]) == 0:
                                del new_paths[-1]
                            else:
                                new_paths[-1][['scaffold','distance']] = new_paths[-1][['scaffold','distance']].values.astype(int)
                                if direction == "left":
                                    new_paths[-1]['strand'] = np.where(new_paths[-1]['strand'] == '+', '-', '+')
                                    new_paths[-1]['pos'] *= -1
                        ext = ext.merge(bext[['center','new_len']], on=['center'], how='left')
                        ext.loc[np.isnan(ext['new_len']) == False, 'len'] = ext.loc[np.isnan(ext['new_len']) == False, 'new_len'].astype(int)
                        ext.drop(columns=['new_len'], inplace=True)
                    # Handle alternatives (must be in the "if len(bext)", because otherwise alt_ext is not well-defined)
                    if len(alt_ext):
                        # Since bubbles are not allowed to start in another bubble turn everything solid until end of alternative (in case we start in a bubble, but shorten the alternative so that it is not in the bubble anymore)
                        ext = ext.merge(alt_ext[['center','end_pos']], on=['center'], how='left')
                        ext.loc[np.isnan(ext['end_pos']) == False, 'solid'] = np.maximum(ext.loc[np.isnan(ext['end_pos']) == False, 'solid'], ext.loc[np.isnan(ext['end_pos']) == False, 'end_pos'].astype(int))
                        ext.drop(columns=['end_pos'], inplace=True)
                        # Shorten bubbles if they overlap with the solid part (They must be consistent to end up here)
                        alt_ext['start_pos'] = alt_ext['pos'] + 1
                        alt_ext.drop(columns=['pos','scaffold','strand'], inplace=True) # Remove the columns we do not need anymore and do not correct in case of overlaps
                        if np.sum(alt_ext['start_pos'] < alt_ext['solid']):
                            alt_ext['overlap'] = np.maximum(0, (alt_ext['solid']-alt_ext['start_pos']).values )
                            for o in range(alt_ext.loc[alt_ext['overlap'] > 0, 'overlap'].min(), alt_ext['overlap'].max()+1):
                                max_len = alt_ext.loc[alt_ext['overlap'] == o, 'length'].max()
                                for s in range(1, max_len+1):
                                    if s+o > max_len:
                                        alt_ext.loc[alt_ext['overlap'] == o, ['scaf'+str(s),'strand'+str(s),'dist'+str(s)]] = np.nan
                                    else:
                                        alt_ext.loc[alt_ext['overlap'] == o, ['scaf'+str(s),'strand'+str(s),'dist'+str(s)]] = alt_ext.loc[alt_ext['overlap'] == o, ['scaf'+str(s+o),'strand'+str(s+o),'dist'+str(s+o)]].values
                            alt_ext['start_pos'] += alt_ext['overlap']
                            alt_ext['length'] -= alt_ext['overlap']
                            alt_ext.drop(columns=['overlap'], inplace=True)
                        # Assign alt_id. Keeps track of phased bubbles and separates bubbles if we have multiple alternatives (polyploid case)
                        alt_ext['alt_id'] = np.arange(0,len(alt_ext)) + next_free_alt_id
                        next_free_alt_id += len(alt_ext)
                        if len(cur_alters):
                            # If a connection to a previous alternative exists use this alt_id
                            phasing = []
                            for s in range(1,alt_ext['length'].max()): # Do not use last/to scaffold as this is identical to main path
                                phasing.append( alt_ext.loc[alt_ext['length'] > s, ['center','alt_id','scaf'+str(s),'strand'+str(s)]].rename(columns={('scaf'+str(s)):'to', ('strand'+str(s)):'to_strand'}) )
                            if len(phasing):
                                phasing = pd.concat(phasing, ignore_index=True).merge(phasing_connections, on=['to','to_strand'], how='inner')
                                phasing = phasing.merge(cur_alters[['center','scaffold','strand','alt_id','alt_pos','start_pos']].rename(columns={'scaffold':'from','strand':'from_strand','alt_id':'new_id'}), on=['center','from','from_strand'], how='inner')
                                # Remove last scaffold in each alternative, because it is identical to the main path (except for dist, which we do not check here)
                                phasing = phasing.merge(cur_alters[['center','alt_id','alt_pos','start_pos']].groupby(['center','alt_id','start_pos']).max().reset_index().rename(columns={'alt_id':'new_id','alt_pos':'max_pos'}), on=['center','new_id','start_pos'], how='left' )
                                phasing = phasing[phasing['alt_pos'] != phasing['max_pos']].copy()
                            if len(phasing):
                                # Remove unconsistent phases
                                phasing = phasing[['alt_id','new_id']].drop_duplicates()
                                phasing.sort_values(['alt_id'], inplace=True)
                                new_id_possibilities = phasing.groupby(['alt_id'], sort=False).size().values
                                phasing = phasing[1 == np.repeat(new_id_possibilities, new_id_possibilities)].copy()
                                alt_ext = alt_ext.merge(phasing, on=['alt_id'], how='left')
                                # Merge the two alternative ids with the same phase
                                alt_ext.loc[np.isnan(alt_ext['new_id']) == False, 'alt_id'] = alt_ext.loc[np.isnan(alt_ext['new_id']) == False, 'new_id'].astype(int)
                                alt_ext.drop(columns=['new_id'],inplace=True)
                            # If we have an alternative inconsistent with the main path, the main path must be phased, thus in the diploid case we can also phase the alternatives
                            if 2 == ploidy and np.sum(alt_ext['consistent']) != len(alt_ext):
                                phasing = alt_ext[(alt_ext['consistent'] == False) & (False == np.isin(alt_ext['alt_id'].values,cur_alters['alt_id'].values))].reset_index() # Only phase what has not been already phased
                                phasing = phasing.merge(cur_alters[['center','end_pos','alt_id']].drop_duplicates().rename(columns={'end_pos':'alt_end','alt_id':'new_id'}), on=['center'], how='inner')
                                phasing.sort_values(['center','alt_end'], inplace=True)
                                phasing = phasing.groupby(['center']).last().reset_index() # Take the alt_id from the last bubbles
                                alt_ext.loc[phasing['index'].values, 'alt_id'] = phasing['new_id'].values
                        # Insert alternative paths
                        if direction == "left":
                            alt_ext[['start_pos','end_pos']] *= -1
                        alt_ext['alt_pos'] = 0
                        for s in range(1,alt_ext['length'].max()+1):
                            new_alters.append(alt_ext.loc[alt_ext['length'] >= s, ['center','start_pos','end_pos','alt_id','alt_pos','scaf'+str(s),'strand'+str(s),'dist'+str(s)]].rename(columns={('scaf'+str(s)):'scaffold', ('strand'+str(s)):'strand', ('dist'+str(s)):'distance'}))
                            if direction == "left":
                                new_alters[-1]['strand'] = np.where(new_alters[-1]['strand'] == '+', '-', '+')
                                alt_ext['alt_pos'] -= 1
                            else:
                                alt_ext['alt_pos'] += 1
                            if len(new_alters[-1]) == 0:
                                del new_alters[-1]
                            else:
                                new_alters[-1][['scaffold','distance']] = new_alters[-1][['scaffold','distance']].values.astype(int)
#
                ## Remove finished extensions
                ext['pos'] += 1
                ext = ext[ext['pos'] < ext['len']].copy()
                ext.drop(columns=['scaffold','strand'], inplace=True)
                cur_paths = pd.concat(new_paths, ignore_index=True)
                if len(new_alters):
                    cur_alters = pd.concat(new_alters, ignore_index=True)
#
            # Check if we have missed an alternative at the end of path, because there is no bubble anymore
            if len(cur_alters):
                new_alters = cur_alters.drop(columns=['distance']).sort_values(['center','start_pos','alt_id'])
                if direction == "left":
                    # Similarize things for both directions
                    new_alters[['start_pos','end_pos','alt_pos']] *= -1
                    new_alters['strand'] = np.where(new_alters['strand'] == '+', '-', '+')
                else:
                    # Position 0 is not allowed to be a bubble, so we do not take alternatives into account that cross it
                    new_alters = new_alters[new_alters['start_pos'] >= 0].copy()
                # The last scaffold is also part of the main path so ignore connections from it
                alt_size = new_alters.groupby(['center','start_pos','alt_id'], sort=False)['alt_pos'].agg(['max','size'])
                new_alters = new_alters[new_alters['alt_pos'] != np.repeat(alt_size['max'].values, alt_size['size'].values)].copy()
                # Only take the last alternative in a path to search for potential extensions
                alt_size = new_alters.groupby(['center'], sort=False)['start_pos'].agg(['max','size'])
                new_alters = new_alters[new_alters['start_pos'] == np.repeat(alt_size['max'].values, alt_size['size'].values)].copy()
                # Merge bubbles to see if we have an extension of the alternative
                new_alters = new_alters.rename(columns={'scaffold':'from','strand':'from_strand'}).merge(bubbles, on=['from','from_strand'], how='inner')
                new_alters.drop(columns=['from','from_strand'], inplace=True)
                if len(new_alters):
                    # Go back to direction aware state (for simpler merging with cur_alters and cur_paths)
                    if direction == "left":
                        new_alters[['start_pos','end_pos','alt_pos']] *= -1
                        cols = [col for col in new_alters.columns if 'strand' in col]
                        new_alters[cols] = np.where(new_alters[cols] == '+', '-', np.where(new_alters[cols] == '-', '+', np.nan))
                    # Check if rest of alternative matches extending path
                    new_alters['cscaf'] = -1
                    new_alters['consistent'] = 0
                    s=1
                    while np.sum(np.isnan(new_alters['cscaf'])) != len(new_alters):
                        new_alters.drop(columns=['cscaf'], inplace=True)
                        new_alters['alt_pos'] += -1 if direction == "left" else 1
                        new_alters = new_alters.merge(cur_alters.rename(columns={'scaffold':'cscaf','strand':'cstrand','distance':'cdist'}), on=['center','start_pos','end_pos','alt_id','alt_pos'], how='left')
                        valid = new_alters['length'] >= s
                        valid[valid] = valid[valid] & ( (new_alters.loc[valid, 'scaf'+str(s)] == new_alters.loc[valid, 'cscaf']) & (new_alters.loc[valid, 'dist'+str(s)] == new_alters.loc[valid, 'cdist']) &
                                                        (new_alters.loc[valid, 'strand'+str(s)] == new_alters.loc[valid, 'cstrand']) )
                        new_alters = new_alters[np.isnan(new_alters['cscaf']) | valid].copy()
                        new_alters.loc[np.isnan(new_alters['cscaf']) == False, 'consistent'] += 1
                        s+=1
                        new_alters.drop(columns=['cstrand','cdist'], inplace=True)
                    new_alters.drop(columns=['cscaf','alt_pos','start_pos'], inplace=True)
                if len(new_alters):
                    # Trim scaffolds still in the starting alternative from path
                    new_alters = TrimConsistentFromPath(new_alters)
                    # Check where the new alternatives diverge from the path
                    new_alters['consistent'] = 0
                    new_alters.rename(columns={'end_pos':'pos'}, inplace=True)
                    s=1
                    divergent = []
                    while len(new_alters) and s <= new_alters['length'].max():
                        new_alters = new_alters.merge(cur_paths.rename(columns={'scaffold':'cscaf','strand':'cstrand','distance':'cdist'}), on=['center','pos'], how='left')
                        new_alters = new_alters[(np.isnan(new_alters['scaf'+str(s)]) == False)].copy() # Remove alternatives that are consistent until end
                        # Check if it is consistent with main path
                        new_alters['divergent'] = ( (new_alters['scaf'+str(s)] != new_alters['cscaf']) | (new_alters['dist'+str(s)] != new_alters['cdist']) |
                                                    (new_alters['strand'+str(s)] != new_alters['cstrand']) )
                        new_alters.drop(columns=['cscaf','cstrand','cdist'], inplace=True)
                        # Check if it is consistent with alternative path
                        new_alters = new_alters.merge(cur_alters[['center','start_pos','end_pos','scaffold','strand','distance']].rename(columns={'start_pos':'pos','scaffold':'cscaf','strand':'cstrand','distance':'cdist'}), on=['center','pos'], how='left')
                        alt_divergent = ( (new_alters['scaf'+str(s)] != new_alters['cscaf']) | (new_alters['dist'+str(s)] != new_alters['cdist']) |
                                          (new_alters['strand'+str(s)] != new_alters['cstrand']) )
                        new_alters.loc[alt_divergent == False, 'pos'] = new_alters.loc[alt_divergent == False, 'end_pos'].astype(int) - (-1 if direction == "left" else 1)
                        new_alters.drop(columns=['end_pos','cscaf','cstrand','cdist'], inplace=True)
                        new_alters['divergent'] = new_alters['divergent'] & alt_divergent
                        divergent.append(new_alters[new_alters['divergent']].copy())
                        new_alters = new_alters[new_alters['divergent'] == False].copy()
                        new_alters['consistent'] += 1
                        new_alters['pos'] += -1 if direction == "left" else 1
                        s += 1
                    if len(divergent):
                        new_alters = pd.concat(divergent, ignore_index=True).drop(columns=['divergent'])
                if len(new_alters):
                    # Check that we do not diverge before the last bubble
                    new_alters = new_alters.merge(cur_alters[['center','end_pos']].drop_duplicates(), on=['center'], how='left')
                    new_alters.sort_values(['center','end_pos','pos','alt_id'], inplace=True)
                    if direction == "left":
                        last_alt = new_alters.groupby(['center'], sort=False)['end_pos'].agg(['min','size']).rename(columns={'min':'last'})
                    else:
                        last_alt = new_alters.groupby(['center'], sort=False)['end_pos'].agg(['max','size']).rename(columns={'max':'last'})
                    new_alters = new_alters[new_alters['end_pos'] == np.repeat(last_alt['last'].values, last_alt['size'].values)].copy()
                    new_alters = new_alters[np.abs(new_alters['end_pos']) <= np.abs(new_alters['pos'])].drop(columns=['end_pos'])
                    # Check that we do not have more than ploidy-1 (-1 is the main path) alternatives from the same center
                    alt_size = new_alters.groupby(['center'], sort=False).size().values
                    new_alters = new_alters[np.repeat(alt_size <= ploidy-1, alt_size)].copy()
                if len(new_alters):
                    # Go to direction unaware state to handle both directions in the same way
                    if direction == "left":
                        new_alters['pos'] *= -1
                        cols = [col for col in new_alters.columns if 'strand' in col]
                        new_alters[cols] = np.where(new_alters[cols] == '+', '-', np.where(new_alters[cols] == '-', '+', np.nan))
                    # Unify start position of the extending alternatives from the same center
                    alt_size = new_alters.groupby(['center'], sort=False)['pos'].agg(['min','size']).rename(columns={'min':'start_pos'})
                    new_alters['start_pos'] = np.repeat(alt_size['start_pos'].values, alt_size['size'].values)
                    new_alters['consistent'] -= new_alters['pos'] - new_alters['start_pos']
                    # Trim path that is consistent with main
                    new_alters = TrimConsistentFromPath(new_alters)
                    new_alters.drop(columns=['pos','consistent'], inplace=True)
                    # Extend alternatives as far as possible with consistent bubbles; removing extensions if they get over ploidy-1 possibilities from the same center
                    new_alters.columns = new_alters.columns.str.replace('scaf', 'ascaf')
                    new_alters.columns = new_alters.columns.str.replace('strand', 'astrand')
                    new_alters.columns = new_alters.columns.str.replace('dist', 'adist')
                    new_alters.sort_values(['center','alt_id'], inplace=True)
                    s=1
                    while s <= new_alters['length'].max():
                        # Create merge columns without NaNs
                        new_alters['from'] = np.where(s <= new_alters['length'], new_alters['ascaf'+str(s)], -1).astype(int)
                        new_alters['from_strand'] = np.where(new_alters['astrand'+str(s)] == '+', '+', '-') # If this was a none the from column prevents merging
                        # Merge potential extensions
                        new_alters = new_alters.merge(bubbles.rename(columns={'length':'new_len'}), on=['from','from_strand'], how='left')
                        new_alters['new_len'] += s
                        maxlen = new_alters['new_len'].max()
                        if np.isnan(maxlen) == False:
                            # Remove inconsistent extensions and insert extending ones
                            oldlen = new_alters['length'].max()
                            for c in range(s+1, int(maxlen)+1):
                                if c > oldlen:
                                    new_alters[['ascaf'+str(c),'astrand'+str(c),'adist'+str(c)]] = new_alters[['scaf'+str(c-s),'strand'+str(c-s),'dist'+str(c-s)]].values
                                else:
                                    new_alters = new_alters[np.isnan(new_alters['ascaf'+str(c)]) | ( (new_alters['ascaf'+str(c)] == new_alters['scaf'+str(c-s)]) &
                                                                                                     (new_alters['astrand'+str(c)] == new_alters['strand'+str(c-s)]) &
                                                                                                     (new_alters['adist'+str(c)] == new_alters['dist'+str(c-s)]) )].copy()
                                    new_alters.loc[np.isnan(new_alters['ascaf'+str(c)]), ['ascaf'+str(c),'astrand'+str(c),'adist'+str(c)]] = new_alters.loc[np.isnan(new_alters['ascaf'+str(c)]), ['scaf'+str(c-s),'strand'+str(c-s),'dist'+str(c-s)]].values
                            # Update length
                            new_alters.loc[np.isnan(new_alters['new_len']) == False, 'length'] = np.maximum(new_alters.loc[np.isnan(new_alters['new_len']) == False, 'length'], new_alters.loc[np.isnan(new_alters['new_len']) == False, 'new_len'].astype(int))
                        new_alters.drop( columns=new_alters.columns[new_alters.columns.str.match(pat = '^scaf|^strand|^dist')].values, inplace=True)
                        new_alters.drop(columns=['new_len'], inplace=True)
                        # Check ploidy
                        alt_size = new_alters.groupby(['center','alt_id'], sort=False).size().values
                        new_alters = new_alters[np.repeat(alt_size < ploidy, alt_size)].copy()
                        s+=1
                    new_alters.drop(columns=['from','from_strand'], inplace=True)
                    # Add the new alternatives
                    if len(new_alters):
                        if direction == "left":
                            new_alters['start_pos'] *= -1
                            new_alters = new_alters.merge(cur_paths.groupby(['center'])['pos'].min().reset_index().rename(columns={'pos':'end_pos'}), on=['center'], how='left')
                            new_alters['end_pos'] -= 1
                        else:
                            new_alters = new_alters.merge(cur_paths.groupby(['center'])['pos'].max().reset_index().rename(columns={'pos':'end_pos'}), on=['center'], how='left')
                            new_alters['end_pos'] += 1
                        new_alters['alt_pos'] = 0
                        cur_alters = [cur_alters]
                        for s in range(1,new_alters['length'].max()+1):
                            cur_alters.append(new_alters.loc[new_alters['length'] >= s, ['center','start_pos','end_pos','alt_id','alt_pos','ascaf'+str(s),'astrand'+str(s),'adist'+str(s)]].rename(columns={('ascaf'+str(s)):'scaffold', ('astrand'+str(s)):'strand', ('adist'+str(s)):'distance'}))
                            if direction == "left":
                                cur_alters[-1]['strand'] = np.where(cur_alters[-1]['strand'] == '+', '-', '+')
                                new_alters['alt_pos'] -= 1
                            else:
                                new_alters['alt_pos'] += 1
                            if len(cur_alters[-1]) == 0:
                                del cur_alters[-1]
                            else:
                                cur_alters[-1][['scaffold','distance']] = cur_alters[-1][['scaffold','distance']].values.astype(int)
                        cur_alters = pd.concat(cur_alters, ignore_index=True)
#
        # Book keeping
        final_paths.append(cur_paths)
        if len(cur_alters):
            final_alternatives.append(cur_alters)
            included_scaffolds = np.unique(np.concatenate([included_scaffolds, cur_paths['scaffold'].values, cur_alters['scaffold'].values]))
        else:
            included_scaffolds = np.unique(np.concatenate([included_scaffolds, cur_paths['scaffold'].values]))
        cknots = cknots[np.isin(cknots['scaffold'], included_scaffolds) == False].copy()
#
    return final_paths, final_alternatives

def TrimAlternativesConsistentWithMainOld(final_paths, ploidy):
    for h in range(1, ploidy):
        final_paths.loc[(final_paths['scaffold'] == final_paths[f'alt_scaf{h}']) & (final_paths['strand'] == final_paths[f'alt_strand{h}']) & (final_paths['distance'] == final_paths[f'alt_dist{h}']), [f'alt_id{h}', f'alt_scaf{h}', f'alt_dist{h}', f'alt_strand{h}']] = [-1,-1,0,'']
#
    return final_paths

def ShiftDistancesDueToDirectionChange(final_paths, change, direction, ploidy):
    for h in range(1, ploidy):
        # Shift distances in alternatives
        final_paths.loc[change & (final_paths[f'alt_id{h}'] < 0), f'alt_dist{h}'] = final_paths.loc[change & (final_paths[f'alt_id{h}'] < 0), 'distance']
        final_paths[f'alt_dist{h}'] = np.where(change, final_paths[f'alt_dist{h}'].shift(direction, fill_value=0), final_paths[f'alt_dist{h}'])
        final_paths.loc[change & (final_paths['center'] != final_paths['center'].shift(direction)) | ((final_paths[f'alt_id{h}'].shift(direction) < 0) & (final_paths[f'alt_id{h}'] < 0)), f'alt_dist{h}'] = 0
        while True:
            final_paths['jumped'] = (final_paths[f'alt_id{h}'] >= 0) & (final_paths[f'alt_scaf{h}'] < 0) & (final_paths[f'alt_dist{h}'] != 0) # Jump deletions
            if 0 == np.sum(final_paths['jumped']):
                break
            else:
                final_paths.loc[final_paths['jumped'].shift(direction, fill_value=False), f'alt_dist{h}'] = final_paths.loc[final_paths['jumped'], f'alt_dist{h}'].values
                final_paths.loc[final_paths['jumped'], f'alt_dist{h}'] = 0
        # We also need to shift the alt_id, because the distance variation at the end of the bubble is on the other side now
        final_paths['new_id'] = np.where(change & (final_paths[f'alt_id{h}'].shift(direction, fill_value=-1) >= 0) & (final_paths['center'] == final_paths['center'].shift(direction)), final_paths[f'alt_id{h}'].shift(direction, fill_value=-1), final_paths[f'alt_id{h}'])
        missing_information = (final_paths[f'alt_id{h}'] < 0) & (final_paths['new_id'] >= 0)
        final_paths.loc[missing_information, [f'alt_scaf{h}', f'alt_strand{h}']] = final_paths.loc[missing_information, ['scaffold','strand']].values
        final_paths[f'alt_id{h}'] = final_paths['new_id']
    final_paths.drop(columns=['new_id'], inplace=True)
    # Shift main distances, wait until after the alternatives, because we need the main alternatives during the alternative shift
    final_paths['distance'] = np.where(change, final_paths['distance'].shift(direction, fill_value=0), final_paths['distance'])
    final_paths.loc[change & (final_paths['center'] != final_paths['center'].shift(direction)), 'distance'] = 0
    while True:
        final_paths['jumped'] = (final_paths['scaffold'] == -1) & (final_paths['distance'] != 0) # Jump deletions
        if 0 == np.sum(final_paths['jumped']):
            break
        else:
            final_paths.loc[final_paths['jumped'].shift(direction, fill_value=False), 'distance'] = final_paths.loc[final_paths['jumped'], 'distance'].values
            final_paths.loc[final_paths['jumped'], 'distance'] = 0
    final_paths.drop(columns=['jumped'], inplace=True)
#
    # Trim alternatives, where they are consistent with main path
    final_paths = TrimAlternativesConsistentWithMainOld(final_paths, ploidy)
#
    return final_paths

def IntegrateAlternativesIntoPaths(final_paths, final_alternatives, ploidy):
    if len(final_paths):
        final_paths = pd.concat(final_paths, ignore_index=True)
        final_paths[['scaffold','distance']] = final_paths[['scaffold','distance']].values.astype(int)
#
        ## Merge final_alternatives with final_paths
        if len(final_alternatives):
            final_alternatives = pd.concat(final_alternatives, ignore_index=True)
            # Prepare final_alternatives
            final_alternatives.sort_values(['center','start_pos','alt_id'], inplace=True)
            gsize = final_alternatives.groupby(['center','start_pos','alt_id'], sort=False).size().values
            final_alternatives['hap'] = np.repeat( final_alternatives[['center','start_pos','alt_id']].drop_duplicates().groupby(['center','start_pos']).cumcount().values, gsize ) + 1
            final_alternatives['alt_size'] = np.repeat( gsize, gsize )
            final_alternatives['alt_pos'] = np.abs(final_alternatives['alt_pos'])
            final_alternatives['main_size'] = np.abs(final_alternatives['end_pos'] - final_alternatives['start_pos'])
            # Assure that we do not have any haplotype switches for alternatives with the same id
            hap_switches = final_alternatives.groupby(['center','alt_id'])['hap'].agg(['min','max']).reset_index()
            hap_switches = hap_switches[hap_switches['min'] != hap_switches['max']].copy()
            if len(hap_switches):
                print("Uncorrected haplotype switches. Please create an issue on github.")
            # Add positions in the paths, where only alternatives exist
            path_ext = final_alternatives.loc[final_alternatives['main_size'] == 0, ['center','start_pos']].drop_duplicates().rename(columns={'start_pos':'pos'})
            path_ext['scaffold'] = -1
            path_ext['strand'] = ''
            path_ext['distance'] = 0
            final_paths = pd.concat([final_paths, path_ext], ignore_index=True)
            final_alternatives.loc[final_alternatives['main_size'] == 0, 'end_pos'] += np.where(final_alternatives.loc[final_alternatives['main_size'] == 0, 'end_pos'] < 0, -1, 1)
            final_alternatives.loc[final_alternatives['main_size'] == 0, 'main_size'] = 1
            # Add merge ids, to identify positions after shifts
            merge_ids = final_alternatives[['center','start_pos','end_pos']].drop_duplicates()
            merge_ids['mid'] = np.arange(len(merge_ids))
            final_alternatives = final_alternatives.merge(merge_ids, on=['center','start_pos','end_pos'], how='left')
            final_alternatives['mpos'] = 0
            merge_ids.reset_index(drop=True, inplace=True)
            merge_ids = merge_ids.loc[np.repeat(merge_ids.index.values, np.abs(merge_ids['end_pos'] - merge_ids['start_pos']))]
            merge_ids['mpos'] = merge_ids.groupby(['center','start_pos','end_pos'], sort=False).cumcount()
            merge_ids['pos'] = merge_ids['start_pos'] + merge_ids['mpos'] * np.where(merge_ids['start_pos'] < 0, -1, 1)
            final_paths = final_paths.merge(merge_ids[['center','pos','mid','mpos']], on=['center','pos'], how='left')
            final_paths[['mid','mpos']] = final_paths[['mid','mpos']].fillna(-1).astype(int)
            final_alternatives.drop(columns=['start_pos','end_pos'], inplace=True)
#
        # Prepare final_paths for merging
        final_paths.sort_values(['center','pos'], inplace=True)
        for h in range(1, ploidy):
            final_paths[['alt_id'+str(h),'alt_scaf'+str(h),'alt_dist'+str(h)]] = [-1,-1,0]
            final_paths['alt_strand'+str(h)] = ''
#
        # Start merging
        while len(final_alternatives):
            for h in range(1, ploidy):
                #### alt_size < main_size:  (scaffold == ascaf)   True  -> Apply alternative
                ####                                              False -> Mark as deletion (setting alt_id) and move to next mpos
                #### alt_size == main_size:                                Apply alternative
                #### alt_size > main_size:  (scaffold == ascaf)   True  -> Apply alternative   (main_size == 1) True -> Add rest of alternative after the scaffold
                ####                                              False -> Add alternative before scaffold
                # Apply alternatives where directly possible
                final_paths = final_paths.merge(final_alternatives.loc[(final_alternatives['hap'] == h) & (final_alternatives['alt_pos'] == 0), ['mid','mpos','alt_id','scaffold','strand','distance','alt_size','main_size']].rename(columns={'scaffold':'ascaf','strand':'astrand','distance':'adist'}), on=['mid','mpos'], how='left')
                final_paths['accepted'] = (final_paths['alt_size'] == final_paths['main_size']) | ((final_paths['ascaf'] == final_paths['scaffold']) & (final_paths['astrand'] == final_paths['strand'])) # Do not check for distance, because that one basically never merges (It is not an alternative if it does)
                final_paths.loc[final_paths['accepted'], ['alt_id'+str(h),'alt_scaf'+str(h),'alt_dist'+str(h)]] = final_paths.loc[final_paths['accepted'], ['alt_id','ascaf','adist']].values.astype(int)
                final_paths.loc[final_paths['accepted'], 'alt_strand'+str(h)] = final_paths.loc[final_paths['accepted'], 'astrand']
                # Mark deletions in final_paths (Do not filter out the accepted once, where we now simply overwrite alt_id with the same value again)
                final_paths.loc[(final_paths['alt_size'] < final_paths['main_size']), 'alt_id'+str(h)] =  final_paths.loc[(final_paths['alt_size'] < final_paths['main_size']), 'alt_id'].astype(int)
                # Check which alternatives could be applied
                final_alternatives = final_alternatives.merge(final_paths.loc[final_paths['accepted'], ['mid','accepted']], on=['mid'], how='left')
                final_alternatives['accepted'] = (final_alternatives['accepted'] == True) & (h == final_alternatives['hap']) # Remove NaN and other haplotypes
                # Remove scaffolds that have been applied or will be added before scaffold
                final_alternatives = final_alternatives[((final_alternatives['accepted'] == False) & (final_alternatives['alt_size'] <= final_alternatives['main_size'])) | ( 0 != final_alternatives['alt_pos']) | (h != final_alternatives['hap'])].copy()
                # Clone scaffolds to insert before or after them
                final_paths = final_paths.loc[np.repeat(final_paths.index.values, np.where((final_paths['alt_size'].values <= final_paths['main_size'].values) | np.isnan(final_paths['alt_size'].values), 1, np.where(final_paths['accepted'].values, np.where(final_paths['main_size'].values==1, (final_paths['alt_size'].values - final_paths['main_size'].values).astype(int), 1), 2)))]
                final_paths.reset_index(inplace=True)
                final_paths['ipos'] = final_paths.groupby('index').cumcount() 
                # Flip ipos in case position is negative, so that 0 is always the closest to the center
                gsize = final_paths.groupby(['index'], sort=False).size().values
                final_paths['ipos'] = np.where(final_paths['pos'] >= 0, final_paths['ipos'], np.repeat(gsize-1, gsize)-final_paths['ipos'])
                # Handle the insertions before
                added_before = (final_paths['alt_size'].values > final_paths['main_size'].values) & (final_paths['accepted'] == False) & (final_paths['ipos'] == 0)
                if np.sum(added_before):
                    final_paths.loc[added_before, ['scaffold','distance','mid','mpos']] = [-1,0,-1,-1] # We never merge with insertions. Instead, we later merge insertion that are next to each other and on other haplotypes
                    final_paths.loc[added_before, 'strand'] = ''
                    final_paths.loc[added_before, ['alt_id'+str(hap) for hap in range(1,ploidy)]] = -1
                    final_paths.loc[added_before, ['alt_scaf'+str(hap) for hap in range(1,ploidy)]] = -1
                    final_paths.loc[added_before, ['alt_dist'+str(hap) for hap in range(1,ploidy)]] = 0
                    final_paths.loc[added_before, ['alt_strand'+str(hap) for hap in range(1,ploidy)]] = ''
                    final_paths.loc[added_before, ['alt_id'+str(h),'alt_scaf'+str(h),'alt_dist'+str(h)]] = final_paths.loc[added_before, ['alt_id','ascaf','adist']].values.astype(int)
                    final_paths.loc[added_before, 'alt_strand'+str(h)] = final_paths.loc[added_before, 'astrand']
                final_paths.drop(columns=['alt_id','ascaf','astrand','adist'], inplace=True)
                # Handle the insertions afterwards
                added_after = (final_paths['alt_size'].values > final_paths['main_size'].values) & final_paths['accepted'] & (final_paths['main_size'] == 1) & (final_paths['ipos'] > 0)
                if np.sum(added_after):
                    final_paths.loc[added_after, ['scaffold','distance']] = [-1,0]
                    final_paths.loc[added_after, 'strand'] = ''
                    final_paths.loc[added_after, ['alt_id'+str(hap) for hap in range(1,ploidy)]] = -1
                    final_paths.loc[added_after, ['alt_scaf'+str(hap) for hap in range(1,ploidy)]] = -1
                    final_paths.loc[added_after, ['alt_dist'+str(hap) for hap in range(1,ploidy)]] = 0
                    final_paths.loc[added_after, ['alt_strand'+str(hap) for hap in range(1,ploidy)]] = ''
                    final_paths.loc[added_after, 'mpos'] += final_paths.loc[added_after, 'ipos']
                    added_after_alt = (final_alternatives['alt_size'].values > final_alternatives['main_size'].values) & final_alternatives['accepted'] & (final_alternatives['main_size'] == 1)
                    final_alternatives.loc[added_after_alt, 'mpos'] = final_alternatives.loc[added_after_alt, 'alt_pos']
                    final_paths = final_paths.merge(final_alternatives.loc[added_after_alt, ['mid','mpos','alt_id','scaffold','strand','distance']].rename(columns={'scaffold':'ascaf','strand':'astrand','distance':'adist'}), on=['mid','mpos'], how='left')
                    final_paths.loc[added_after, ['alt_id'+str(h),'alt_scaf'+str(h),'alt_dist'+str(h)]] = final_paths.loc[added_after, ['alt_id','ascaf','adist']].values.astype(int)
                    final_paths.loc[added_after, 'alt_strand'+str(h)] = final_paths.loc[added_after, 'astrand'] 
                    final_alternatives = final_alternatives[added_after_alt == False].copy()
                    final_paths.loc[added_after, ['mid','mpos']] = -1 # Prevent other haplotypes to merge with this insertions. Instead, we later merge insertion that are next to each other and on other haplotypes
                    final_paths.drop(columns=['alt_id','ascaf','astrand','adist'], inplace=True)
                # Clean up and preparation for next iteration (store boolean arrays in variable before applying, because the values change when applied)
                final_paths.drop(columns=['index','alt_size','main_size','accepted','ipos'], inplace=True)
                red_alt = final_alternatives['accepted'] | (final_alternatives['alt_size'] > final_alternatives['main_size'])
                red_main = final_alternatives['accepted'] | (final_alternatives['alt_size'] < final_alternatives['main_size'])
                final_alternatives.loc[ red_alt, ['alt_pos','alt_size']] -= 1
                final_alternatives.loc[ red_main, ['mpos','main_size']] += [1, -1]
                final_alternatives.drop(columns=['accepted'], inplace=True)
        final_paths.drop(columns=['mid','mpos'], inplace=True)

        # Merge insertions from different haplotypes
        if ploidy > 2:
            while True:
                final_paths['mergeable'] = (final_paths['center'] == final_paths['center'].shift(1)) & (final_paths['scaffold'] == -1) & (final_paths['scaffold'].shift(1) == -1) # Tthe entry and the one before are from the same center and both insertions
                for h in range(1, ploidy):
                    final_paths['mergeable'] = final_paths['mergeable'] & ((final_paths['alt_id'+str(h)] == -1) | (final_paths['alt_id'+str(h)].shift(1) == -1)) # Every haplotype has at max one valid entry in the two entries
                final_paths['mergeable'] = final_paths['mergeable'] & (final_paths['mergeable'].shift(1) == False) # If the one before will be merged, do not merge this one
                if 0 == np.sum(final_paths['mergeable']):
                    break
                else:
                    for h in range(1, ploidy):
                        merge_dest = final_paths['mergeable'].shift(-1, fill_value=False) & (final_paths['alt_id'+str(h)] == -1) # The one before the mergeable if it does not contain information already (in which case the merge_org does not contain information)
                        merge_org = final_paths['mergeable'] & (final_paths['alt_id'+str(h)].shift(1) == -1)
                        final_paths.loc[merge_dest, ['alt_id'+str(h), 'alt_scaf'+str(h), 'alt_dist'+str(h), 'alt_strand'+str(h)]] = final_paths.loc[merge_org, ['alt_id'+str(h), 'alt_scaf'+str(h), 'alt_dist'+str(h), 'alt_strand'+str(h)]].values
                        final_paths.loc[merge_dest, ] = final_paths.loc[merge_org, ].values
                final_paths = final_paths[final_paths['mergeable'] == False].copy()
            final_paths.drop(columns=['mergeable'], inplace=True)

        # Reset positions starting from 0 and shift distances so they point to start of scaffold not to center of it
        final_paths['before_center'] = final_paths['pos'] <= 0
        final_paths['pos'] = final_paths.groupby('center').cumcount()
        final_paths = ShiftDistancesDueToDirectionChange(final_paths, final_paths['before_center'], 1, ploidy)
        final_paths.drop(columns=['before_center'], inplace=True)

    return final_paths, final_alternatives

def FillBridgePathInfo(tmp_paths, final_paths, knots, ploidy):
    tmp_paths = tmp_paths.merge(final_paths, on=['center','pos'], how='left')
    tmp_paths = tmp_paths.merge(knots[['scaffold','scaf_len']], on=['scaffold'], how='left')
    tmp_paths.loc[tmp_paths['scaffold'] == -1, 'scaf_len'] = 0
    tmp_paths['scaf_len'] = tmp_paths['scaf_len'].astype(int)
    tmp_paths['nhaps'] = 1 # We always have the main path
    tmp_paths['ndels'] = (tmp_paths['scaffold'] < 0).astype(int)
    for h in range(1, ploidy):
        tmp_paths.loc[(tmp_paths['alt_id'+str(h)] >= 0) & ((tmp_paths['alt_scaf'+str(h)] != tmp_paths['scaffold']) | (tmp_paths['alt_strand'+str(h)] != tmp_paths['strand'])), 'nhaps'] += 1
        tmp_paths.loc[(tmp_paths['alt_id'+str(h)] >= 0) & (tmp_paths['alt_scaf'+str(h)] < 0), 'ndels'] += 1
        tmp_paths.loc[(tmp_paths['alt_id'+str(h)] >= 0) & (tmp_paths['alt_scaf'+str(h)] < 0), 'scaf_len'] = 0
    tmp_paths['nhaps'] -= tmp_paths['ndels']
    tmp_paths['ndels'] = np.minimum(1, tmp_paths['ndels'])
    tmp_paths = tmp_paths.groupby(['index']).agg({'scaf_len':['sum'],'nhaps':['max'],'ndels':['sum']}).droplevel(1,axis=1).reset_index()
#
    return tmp_paths

def RemoveUnsupportedDuplications(duplications, final_paths, unsupp, ploidy):
    for p2, q2 in zip(['a','b'], ['b','a']):
        remdups = duplications[[f'center{p2}',f'pos{p2}']].rename(columns={f'center{p2}':'center',f'pos{p2}':'pos'}).merge(final_paths.loc[unsupp,['center','pos']], on=['center','pos'], how='left', indicator=True)['_merge'].values == "both"
        if np.sum(remdups):
            duplications.loc[remdups, f'nhaps{p2}'] -= 1
            dupsp = final_paths.merge(duplications.loc[remdups, [f'center{p2}',f'pos{p2}']].rename(columns={f'center{p2}':'center',f'pos{p2}':'pos'}), on=['center','pos'], how='right').rename(columns={'scaffold':'alt_scaf0','strand':'alt_strand0','distance':'alt_dist0'})
            dupsp['alt_id0'] = 0
            dupsq = final_paths.merge(duplications.loc[remdups, [f'center{q2}',f'pos{q2}']].rename(columns={f'center{q2}':'center',f'pos{q2}':'pos'}), on=['center','pos'], how='right').rename(columns={'scaffold':'alt_scaf0','strand':'alt_strand0','distance':'alt_dist0'})
            dupsq['alt_id0'] = 0
            for h2 in range(0,ploidy):
                dupsp[f'match{h2}'] = False
                dupsq[f'match{h2}'] = False
            for hp in range(0,ploidy):
                for hq in range(0,ploidy):
                    dmatch = (dupsp[f'alt_id{hp}'].values >= 0) & (dupsq[f'alt_id{hq}'].values >= 0) & (dupsp[f'alt_scaf{hp}'].values == dupsq[f'alt_scaf{hq}'].values) & (np.where(dupsp[f'alt_scaf{hp}'].values < 0, True, duplications.loc[remdups, 'samedir'].values) == (dupsp[f'alt_strand{hp}'].values == dupsq[f'alt_strand{hq}'].values))
                    dupsp.loc[dmatch, f'match{hp}'] = True
                    dupsq.loc[dmatch, f'match{hq}'] = True
            dupsp['nmatches'] = dupsp[[f'match{h2}' for h2 in range(0,ploidy)]].sum(axis=1)
            dupsq['nmatches'] = dupsq[[f'match{h2}' for h2 in range(0,ploidy)]].sum(axis=1)
            duplications.loc[remdups, f'full{p2}'] = dupsp['nmatches'].values == duplications.loc[remdups, f'nhaps{p2}'].values
            duplications.loc[remdups, f'full{q2}'] = dupsq['nmatches'].values == duplications.loc[remdups, f'nhaps{q2}'].values
            duplications.drop(duplications[remdups][dupsp['nmatches'].values == 0].index.values, inplace=True)

    return duplications

def RemoveUnsupportedDuplicationsAndPath(final_paths, duplications, fixed_errors, unsupp, h, ploidy):
    final_paths.loc[unsupp, [f'alt_id{h}',f'alt_scaf{h}',f'alt_dist{h}',f'alt_strand{h}']] = [-1,-1,0,'']
    fixed_errors += np.sum(unsupp)

    duplications = RemoveUnsupportedDuplications(duplications, final_paths, unsupp, ploidy)

    return final_paths, duplications, fixed_errors

def SwitchPathWithMain(final_paths, switch, h):
    tmp = final_paths.loc[switch, [f'alt_scaf{h}',f'alt_dist{h}',f'alt_strand{h}']].values
    final_paths.loc[switch, [f'alt_scaf{h}',f'alt_dist{h}',f'alt_strand{h}']] = final_paths.loc[switch, ['scaffold','distance','strand']].values
    final_paths.loc[switch, ['scaffold','distance','strand']] = tmp

    return final_paths

def RemoveHaplotypeFromPath(final_paths, rem_tracker, alt_ids, h):
    removal = np.isin(final_paths[f'alt_id{h}'].values, alt_ids)
    final_paths.loc[removal, [f'alt_id{h}',f'alt_scaf{h}',f'alt_dist{h}',f'alt_strand{h}']] = [-1,-1,0,'']
    rem_tracker.append(final_paths.loc[removal, ['center','pos']])
#
    return final_paths, rem_tracker

def TrimPath(final_paths, rem_tracker, rem_df, remdir, ploidy):
    unsupp_pos = rem_df.groupby(['center'])['pos'].agg(['min'] if remdir=='r' else ['max']).reset_index().rename(columns={('min' if remdir=='r' else 'max'):'unsupp_pos'})
    unsupp_pos = final_paths[['center']].merge(unsupp_pos, on=['center'], how='left')
    if remdir == 'l':
        removal = final_paths['pos'] <= unsupp_pos['unsupp_pos'].values
    else:
        removal = final_paths['pos'] >= unsupp_pos['unsupp_pos'].values
    final_paths.loc[removal, ['scaffold','strand','distance']] = [-1,'',0] # Only set everything to a deletion to keep positions. Remove them after the duplication check is fully done
    for h in range(1,ploidy):
        final_paths.loc[removal, [f'alt_id{h}',f'alt_scaf{h}',f'alt_dist{h}',f'alt_strand{h}']] = [-1,-1,0,'']
    rem_tracker.append(final_paths.loc[removal, ['center','pos']])
#
    return final_paths, rem_tracker

def RemoveMainFromPath(final_paths, rem_tracker, rem_path_info, removal, ploidy, remdir):
    # Replace main with alternative haplotype
    for h in range(1,ploidy):
        if np.sum(removal):
            switch = np.isin(final_paths[f'alt_id{h}'].values, rem_path_info.loc[removal & (rem_path_info[f'alt_id{h}'] >= 0), f'alt_id{h}'].values)
            if np.sum(switch):
                final_paths = SwitchPathWithMain(final_paths, switch, h)
                hap_rem = final_paths[['center','pos']].merge(rem_path_info.loc[removal & (rem_path_info[f'alt_id{h}'] >= 0),['center','pos']], on=['center','pos'], how='left', indicator=True)['_merge'].values == "both"
                hap_rem = switch & (hap_rem | ( (final_paths[f'alt_scaf{h}'] >= 0) & (final_paths['scaffold'] == final_paths[f'alt_scaf{h}']) & (final_paths['strand'] == final_paths[f'alt_strand{h}']) &
                                                (final_paths['center'] == final_paths['center'].shift(1)) & np.concatenate([[False], hap_rem[:-1]]) )) # Extend the unsupported with the distance variant at the end of the bubble
                final_paths.loc[hap_rem, [f'alt_id{h}',f'alt_scaf{h}',f'alt_dist{h}',f'alt_strand{h}']] = [-1,-1,0,'']
                rem_tracker.append(final_paths.loc[hap_rem, ['center','pos']])
            removal = removal & (rem_path_info[f'alt_id{h}'] < 0)
    # If no alternative haplotype exists remove everything in scaffold in removal direction (remdir)
    if np.sum(removal) and (remdir != ''):
        TrimPath(final_paths, rem_tracker, rem_path_info.loc[removal], remdir, ploidy)
#
    return final_paths, rem_tracker

def RemoveEmptyColumns(df):
    # Clean up columns with all NaN
    cols = df.count()
    df.drop(columns=cols[cols == 0].index.values,inplace=True)
    return df

def MergeNextPos(connection, final_paths, ploidy):
    connection['dels'] = True
    while True:
        # Get corresponding scaffold from final_paths
        connection = connection.merge(pd.concat([final_paths[['center','pos','scaffold','strand','distance']]]+[final_paths.loc[final_paths[f'alt_id{h}'] >= 0, ['center','pos',f'alt_scaf{h}',f'alt_strand{h}',f'alt_dist{h}']].rename(columns={f'alt_scaf{h}':'scaffold',f'alt_strand{h}':'strand',f'alt_dist{h}':'distance'}) for h in range(1,ploidy)], ignore_index=True), on=['center','pos'], how='left')
        # Ensure that we do not have deletions as starting scaffold
        connection = connection[(connection['scaffold'] >= 0) | ((np.isnan(connection['scaffold']) == False) & connection['dels'])].copy()
        if np.sum(connection['scaffold'] < 0) == 0:
            break
        else:
            connection['dels'] = False
            # Take the next position, where we have a deletion (only for the deletion, the other scaffold can start there)
            connection.loc[connection['scaffold'] < 0, 'pos'] += connection.loc[connection['scaffold'] < 0, 'dir']
            connection.loc[connection['scaffold'] < 0, 'dels'] = True
            connection.drop(columns=['scaffold','strand','distance'], inplace=True)
            connection.drop_duplicates(inplace=True)
    connection.drop(columns=['dels'], inplace=True)
#
    return connection

def HandlePathDuplications(final_paths, knots, scaffold_graph, ploidy):
    # Search for duplicated scaffolds
    duplications = [final_paths[['scaffold','center','pos']]]
    for h in range(1, ploidy):
        duplications.append(final_paths.loc[final_paths['alt_id'+str(h)] >= 0,['alt_scaf'+str(h),'center','pos']].rename(columns={'alt_scaf'+str(h):'scaffold'}))
    duplications = pd.concat(duplications, ignore_index=True)
    duplications = duplications[duplications['scaffold'] >= 0].copy()
    duplications = duplications.rename(columns={'center':'centera','pos':'posa'}).merge(duplications.rename(columns={'center':'centerb','pos':'posb'}), on=['scaffold'], how='inner')
    duplications = duplications[duplications['centera'] < duplications['centerb']].copy() # Remove duplications on the same scaffold and keep the others only once
    duplications.sort_values(['centera','centerb','posb','posa'], inplace=True)
    duplications.rename(columns={'scaffold':'did'}, inplace=True)
    duplications['did'] = ((duplications['centera'] != duplications['centera'].shift(1)) | (duplications['centerb'] != duplications['centerb'].shift(1))).cumsum()
    duplications.drop_duplicates(inplace=True)
#
    # If we have multiple duplications with same centera, centerb and posa/posb, do not try to solve this rare and very complex condition, but arbitrarily take the first and hope for the best
    duplications = duplications[(duplications['did'] != duplications['did'].shift(1)) | (duplications['posb'] != duplications['posb'].shift(1))].copy()
    duplications.sort_values(['centera','centerb','posa','posb'], inplace=True)
    duplications = duplications[(duplications['did'] != duplications['did'].shift(1)) | (duplications['posa'] != duplications['posa'].shift(1))].copy()
#
    # Handle single duplication entries (either an unconnected scaffold is involved and we can remove that one or we discard them)
    sidups = duplications[(duplications['did'] != duplications['did'].shift(1)) & (duplications['did'] != duplications['did'].shift(-1))].copy()
    duplications = duplications[(duplications['did'] == duplications['did'].shift(1)) | (duplications['did'] == duplications['did'].shift(-1))].copy()
    path_len = final_paths.groupby(['center']).size().reset_index(name='size')
    sidups = sidups.merge(path_len.rename(columns={'center':'centera','size':'sizea'}), on=['centera'], how='left')
    sidups = sidups.merge(path_len.rename(columns={'center':'centerb','size':'sizeb'}), on=['centerb'], how='left')
    rem_paths = [sidups.loc[sidups['sizea'] == 1,'centera'].values]
    rem_paths.append( sidups.loc[sidups['sizeb'] == 1,'centerb'].values )
    del sidups
#
    # Break duplications connected with the same id, where the direction changes
    duplications['dira'] = np.where(duplications['did'] != duplications['did'].shift(-1), 0, duplications['posa'].shift(-1, fill_value=0)-duplications['posa'])
    duplications['dirb'] = np.where(duplications['did'] != duplications['did'].shift(-1), 0, duplications['posb'].shift(-1, fill_value=0)-duplications['posb'])
    duplications['samedir'] = (duplications['dira'] < 0) == (duplications['dirb'] < 0)
    duplications['flip'] = np.where((0 == duplications['dira']) | (duplications['did'] != duplications['did'].shift(1)), False, duplications['samedir'] != duplications['samedir'].shift(1) )
    # Add scaffold where the flip happens to both sides
    duplications.reset_index(drop=True, inplace=True)
    duplications.reset_index(inplace=True)
    duplications = duplications.loc[np.repeat(duplications.index.values, duplications['flip'].values+1)].copy()
    duplications.reset_index(drop=True, inplace=True)
    duplications.loc[duplications['index'] == duplications['index'].shift(-1), ['dira','dirb']] = 0
    duplications.loc[duplications['index'] == duplications['index'].shift(-1), 'flip'] = False
    duplications['did'] = ((duplications['did'] != duplications['did'].shift(1)) | duplications['flip']).cumsum()
    duplications.drop(columns=['index','flip'], inplace=True)
    # For the last connected duplicate we do not have direction information, so fill in samedir with the previous duplicate
    duplications.loc[(duplications['did'] == duplications['did'].shift(1)) & (0 == duplications['dira']), 'samedir'] = duplications['samedir'].shift(1).loc[(duplications['did'] == duplications['did'].shift(1)) & (0 == duplications['dira'])]
    duplications.drop(columns=['dira','dirb'], inplace=True)
#
    # Verify strand for matches 
    dupa = duplications[['centera','posa']].merge(final_paths.rename(columns={'center':'centera','pos':'posa'}), on=['centera','posa'], how='left').rename(columns={'scaffold':'alt_scaf0','strand':'alt_strand0'})
    dupb = duplications[['centerb','posb']].merge(final_paths.rename(columns={'center':'centerb','pos':'posb'}), on=['centerb','posb'], how='left').rename(columns={'scaffold':'alt_scaf0','strand':'alt_strand0'})
    dupa['alt_id0'] = 0 # The value does not matter as long as it is not smaller than zero
    dupb['alt_id0'] = 0 # The value does not matter as long as it is not smaller than zero
    matches = []
    for ha in range(0, ploidy):
        for hb in range(0, ploidy):
            matches.append( duplications.reset_index().loc[(dupa['alt_id'+str(ha)] >= 0) & (dupb['alt_id'+str(hb)] >= 0) & (dupa['alt_scaf'+str(ha)] == dupb['alt_scaf'+str(hb)]) & (np.where(dupa['alt_scaf'+str(ha)] == -1, True, duplications['samedir']) == (dupa['alt_strand'+str(ha)] == dupb['alt_strand'+str(hb)])), ['index','did','centera','posa','centerb','posb']])
            if 0 == len(matches[-1]):
                del matches[-1]
            else:
                matches[-1]['hapa'] = ha
                matches[-1]['hapb'] = hb
#
    if len(matches):
        # Get deletions (Deletions that do not have an alternative are otherwise not considered)
        tmp_paths = final_paths.loc[np.isin(final_paths['center'], np.unique(np.concatenate([duplications['centera'].values, duplications['centerb'].values])))].copy()
        deletions = [final_paths.loc[final_paths['scaffold'] == -1, ['center','pos']]]
        deletions[-1]['hap'] = 0
        for h in range(1, ploidy):
            deletions.append(final_paths.loc[(final_paths[f'alt_scaf{h}'] == -1) & (final_paths[f'alt_id{h}'] != -1), ['center','pos']])
            deletions[-1]['hap'] = h
        deletions = pd.concat(deletions, ignore_index=True)
        deletions['index'] = -1
        # Find positions within a single bubble to later verify that we do not have haplotype switches in a single bubble
        tmp_paths['merger'] = True # All path are the same, thus bubbles end here
        for h in range(1, ploidy):
            tmp_paths.loc[(tmp_paths['alt_id'+str(h)] >= 0) & ((tmp_paths['scaffold'] != tmp_paths['alt_scaf'+str(h)]) | (tmp_paths['strand'] != tmp_paths['alt_strand'+str(h)])), 'merger'] = False
        tmp_paths['bid'] = tmp_paths['merger'].cumsum()
        tmp_paths = tmp_paths.loc[tmp_paths['merger'] == False, ['center','pos','bid']]
        tmp_paths = tmp_paths[(tmp_paths['bid'] == tmp_paths['bid'].shift(1)) | (tmp_paths['bid'] == tmp_paths['bid'].shift(-1))].copy() # If only a single scaffold is in the bubble we do not care
        tmp_paths = tmp_paths.merge(pd.concat([dupa[['centera','posa']].rename(columns={'centera':'center','posa':'pos'}), dupb[['centerb','posb']].rename(columns={'centerb':'center','posb':'pos'})], ignore_index=True).drop_duplicates(), on=['center','pos'], how='inner') # If the positions are not in matches they are not interesting
        tmp_paths.sort_values(['bid'], inplace=True)
        bid_size = tmp_paths.groupby(['bid']).size().values
        tmp_paths['size'] = np.repeat(bid_size, bid_size)
        tmp_paths = tmp_paths[tmp_paths['size'] > 1].copy() # If only a single scaffold is in the bubble we do not care
#
        # Verify that we do not have haplotype switches in a single bubble
        matches = pd.concat(matches, ignore_index=True).reset_index(drop=False)
        while True:
            rejected = pd.concat([matches[['did','centera','posa','hapa']].rename(columns={'centera':'center','posa':'pos','hapa':'hap'}).reset_index(), matches[['did','centerb','posb','hapb']].rename(columns={'centerb':'center','posb':'pos','hapb':'hap'}).reset_index()], ignore_index=True)
            rejected = pd.concat([rejected, deletions.merge(rejected[['did','center','hap']].drop_duplicates(), on=['center','hap'], how='inner')[['index','did','center','pos','hap']]], ignore_index=True).sort_values(['center','did','hap','pos','index']).groupby(['center','did','hap','pos'], sort=False).last().reset_index() # Add deletions for all did of that center,hap if this position does not have a match already
            rejected = tmp_paths.merge(rejected, on=['center','pos'], how='left')
            rejected.sort_values(['center','did','hap'], inplace=True)
            bid_size = rejected.groupby(['center','did','hap'], sort=False).size().values
            rejected = rejected[rejected['size'] > np.repeat(bid_size,bid_size)].copy()
            if 0 == len(rejected):
                break
            else:
                matches.drop(rejected.loc[rejected['index'] >= 0, 'index'].values, inplace=True)
        deletions.drop(columns=['index'], inplace=True)
#
    if len(matches) == 0:
        duplications = [] # All duplications were rejected
    else:
        # Before filtering get the number of haplotypes at each position from dupa and dupb, such that we do not need them anymore
        dupa['nhaps'] = 1 # We always have the main path
        dupb['nhaps'] = 1
        for h in range(1, ploidy):
            dupa.loc[dupa['alt_id'+str(h)] >= 0, 'nhaps'] += 1
            dupb.loc[dupb['alt_id'+str(h)] >= 0, 'nhaps'] += 1
        duplications['nhapsa'] = dupa['nhaps']
        duplications['nhapsb'] = dupb['nhaps']
        # Check if duplications match for all haplotypes and count deletions that do not match
        duplications['matchesa'] = 0
        duplications['matchesb'] = 0
        duplications['delsa'] = 0
        duplications['delsb'] = 0
        for h in range(0, ploidy):
                duplications['match'] = False
                duplications.loc[np.unique(matches.loc[matches['hapa'] == h, 'index'].values), 'match'] = True
                duplications.loc[duplications['match'], 'matchesa'] += 1
                duplications.loc[(duplications['match'] == False) & (dupa[f'alt_id{h}'] >= 0) & (dupa[f'alt_scaf{h}'] < 0), 'delsa'] += 1
                duplications['match'] = False
                duplications.loc[np.unique(matches.loc[matches['hapb'] == h, 'index'].values), 'match'] = True
                duplications.loc[duplications['match'], 'matchesb'] += 1
                duplications.loc[(duplications['match'] == False) & (dupb[f'alt_id{h}'] >= 0) & (dupb[f'alt_scaf{h}'] < 0), 'delsb'] += 1
        duplications.drop(columns=['match'], inplace=True)
        # Keep only duplicates that truely match
        duplications = duplications.loc[np.unique(matches['index'].values)] # We never had deletions in, so we cannot remove them here, thus we do not need to handle them for the filter
#
    if len(duplications):
        # Remove scaffolds that are fully duplicated (ignore if deletions are not duplicated)
        duplications['fulla'] = duplications['nhapsa'] == duplications['matchesa'] + duplications['delsa']
        duplications['fullb'] = duplications['nhapsb'] == duplications['matchesb'] + duplications['delsb']
        dsize = duplications.groupby(['did'], sort=False).agg({'fulla':['size','sum'], 'fullb':['sum']})
        duplications['ndups'] = np.repeat(dsize[('fulla','size')].values, dsize[('fulla','size')].values)
        duplications['nfulla'] = np.repeat(dsize[('fulla','sum')].values, dsize[('fulla','size')].values)
        duplications['nfullb'] = np.repeat(dsize[('fullb','sum')].values, dsize[('fulla','size')].values)
        duplications = duplications.merge(path_len.rename(columns={'center':'centera','size':'lena'}), on=['centera'], how='left')
        duplications = duplications.merge(path_len.rename(columns={'center':'centerb','size':'lenb'}), on=['centerb'], how='left')
        rem_paths.append( np.unique(duplications.loc[duplications['lena'] == duplications['nfulla'], 'centera'].values) )
        rem_paths.append( np.unique(duplications.loc[duplications['lenb'] == duplications['nfullb'], 'centerb'].values) )
        matches = matches[(np.isin(matches['centera'], rem_paths[-2]) == False) & (np.isin(matches['centerb'], rem_paths[-1]) == False)].copy()
        duplications = duplications[(duplications['lena'] != duplications['nfulla']) & (duplications['lenb'] != duplications['nfullb'])].copy()
        # For later we want the true full duplications (including deletions)
        duplications['fulla'] = duplications['nhapsa'] == duplications['matchesa']
        duplications['fullb'] = duplications['nhapsb'] == duplications['matchesb']
        # Remove scaffolds that are fully duplicated, but within multiple other scaffolds
        while True:
            tmp_dups = []
            for p in ['a','b']:
                tmp_dups.append(duplications[['did',f'center{p}',f'pos{p}',f'nhaps{p}',f'len{p}']].rename(columns={f'center{p}':'center',f'pos{p}':'pos',f'nhaps{p}':'nhaps',f'len{p}':'len'}))
            tmp_dups = pd.concat(tmp_dups, ignore_index=True).drop_duplicates().drop(columns=['did']).groupby(['center','pos','nhaps','len']).size().reset_index(name='count')
            center_count = tmp_dups.groupby(['center'],sort=False)['count'].agg(['max','size'])
            tmp_dups = tmp_dups[np.repeat(center_count['max'].values > 1, center_count['size'].values)].copy() # Only take centers that have duplications with multiple other centers
            for h in range(0, ploidy):
                tmp_dups[f'match{h}'] = ( (tmp_dups.merge(matches.loc[matches['hapa'] == h, ['centera','posa']].rename(columns={'centera':'center','posa':'pos'}).drop_duplicates(), on=['center','pos'], how='left', indicator=True)['_merge'].values == "both") | 
                                          (tmp_dups.merge(matches.loc[matches['hapb'] == h, ['centerb','posb']].rename(columns={'centerb':'center','posb':'pos'}).drop_duplicates(), on=['center','pos'], how='left', indicator=True)['_merge'].values == "both") )
            # Also take duplicated deletions into account for scaffold removal
            dup_dels = deletions.merge(tmp_dups[['center']].drop_duplicates(), on=['center'], how='inner').merge(pd.concat([duplications[['centera','centerb']].rename(columns={'centera':'center','centerb':'center2'}), duplications[['centerb','centera']].rename(columns={'centerb':'center','centera':'center2'})], ignore_index=True).drop_duplicates(), on=['center'], how='left')
            dup_dels['pnext'] = dup_dels['pos']+1
            while True:
                chain_dels = dup_dels.merge(dup_dels[['center','center2','pos']].rename(columns={'pos':'pnext'}), on=['center','center2','pnext'], how='left', indicator=True)['_merge'].values == "both"
                if np.sum(chain_dels):
                    dup_dels.loc[chain_dels, 'pnext'] += 1
                else:
                    break
            dup_dels['pprev'] = dup_dels['pos']-1
            while True:
                chain_dels = dup_dels.merge(dup_dels[['center','center2','pos']].rename(columns={'pos':'pprev'}), on=['center','center2','pprev'], how='left', indicator=True)['_merge'].values == "both"
                if np.sum(chain_dels):
                    dup_dels.loc[chain_dels, 'pprev'] -= 1
                else:
                    break
            dup_dels = dup_dels.merge(pd.concat([duplications[['centerb','posb','centera','posa']].rename(columns={'centerb':'center','posb':'pnext','centera':'center2','posa':'pnext2'}), duplications[['centera','posa','centerb','posb']].rename(columns={'centera':'center','posa':'pnext','centerb':'center2','posb':'pnext2'})], ignore_index=True), on=['center','center2','pnext'], how='inner')
            dup_dels = dup_dels.merge(pd.concat([duplications[['centerb','posb','centera','posa']].rename(columns={'centerb':'center','posb':'pprev','centera':'center2','posa':'pprev2'}), duplications[['centera','posa','centerb','posb']].rename(columns={'centera':'center','posa':'pprev','centerb':'center2','posb':'pprev2'})], ignore_index=True), on=['center','center2','pprev'], how='inner')
            dup_dels = dup_dels[np.abs(dup_dels['pnext2']-dup_dels['pprev2']) == 1].copy() # If the other path has no scaffolds in between it is truely a missed deletion (if the other path has a deletion alternative it is captured before already)
            for h in range(0, ploidy):
                tmp_dups.loc[tmp_dups.merge(dup_dels.loc[dup_dels['hap'] == h, ['center','pos']].drop_duplicates(), on=['center','pos'], how='left', indicator=True)['_merge'].values == "both", f'match{h}'] = True
            tmp_dups['nmatches'] = 0
            for h in range(0, ploidy):
                tmp_dups.loc[tmp_dups[f'match{h}'], 'nmatches'] += 1
            tmp_dups = tmp_dups[tmp_dups['nmatches'] == tmp_dups['nhaps']].groupby(['center','len']).size().reset_index(name='nfull')
            tmp_dups = tmp_dups[tmp_dups['len'] == tmp_dups['nfull']].copy()
            # Only remove at max one from each duplication cluster to make sure that we do not remove all occurances of a scaffold
            dup_clust = pd.concat([duplications[['centerb','centera']].rename(columns={'centerb':'center','centera':'center2'}), duplications[['centera','centerb']].rename(columns={'centera':'center','centerb':'center2'})], ignore_index=True).drop_duplicates().sort_values(['center'])
            dup_clust['cid'] = (dup_clust['center'] != dup_clust['center'].shift(1)).cumsum()
            while True:
                dup_clust = dup_clust.merge(dup_clust[['center','cid']].drop_duplicates().rename(columns={'center':'center2','cid':'cid2'}), on=['center2'], how='left')
                cid2 = dup_clust.groupby(['center'])['cid2'].agg(['min','size'])
                cid2 = np.repeat(cid2['min'].values, cid2['size'].values)
                dup_clust.drop(columns=['cid2'], inplace=True)
                if 0 == np.sum(dup_clust['cid'] != cid2):
                    break
                else:
                    dup_clust['cid'] = np.minimum(dup_clust['cid'], cid2)
            tmp_dups = tmp_dups.merge(dup_clust[['center','cid']].drop_duplicates(), on=['center'], how='left')
            full_length = len(tmp_dups)
            tmp_dups = tmp_dups.groupby(['cid']).first().reset_index()
            rem_paths.append( tmp_dups['center'].values )
            matches = matches[(np.isin(matches['centera'], rem_paths[-1]) == False) & (np.isin(matches['centerb'], rem_paths[-1]) == False)].copy()
            duplications = duplications[(np.isin(duplications['centera'], rem_paths[-1]) == False) & (np.isin(duplications['centerb'], rem_paths[-1]) == False)].copy()
            if full_length == len(tmp_dups):
                break
        duplications.drop(columns=['matchesa','matchesb','ndups','nfulla','nfullb','delsa','delsb'], inplace=True)
#
    if len(duplications):
        # Check in scaffold_graph if all haplotypes in final_paths are valid, if we have a duplication with multiple haplotypes
        fixed_errors = 1
        while fixed_errors:
            fixed_errors = 0
            unsupported_dups = [] # Store duplications that have no support for any of the haplotypes to remove the alternatives that have support in the other path/center of the duplication
            for p in ['a','b']:
                pos_errors = duplications.loc[1 < duplications[f'nhaps{p}'], ['did',f'center{p}',f'pos{p}']].rename(columns={f'center{p}':'center',f'pos{p}':'pos'})
                for h in range(0,ploidy):
                    pos_errors[f'supp{h}'] = False
                for d in ['l','r']:
                    if 'pindex' in pos_errors.columns:
                        # For the second round we already have the index for final_paths
                        pos_errors = pd.concat([pos_errors, final_paths.loc[pos_errors['pindex'].values].drop(columns=['center','pos']).rename(columns={'scaffold':'alt_scaf0','strand':'alt_strand0','distance':'alt_dist0'}).reset_index(drop=True)], axis=1)
                    else:
                        pos_errors = pos_errors.merge(final_paths.reset_index().rename(columns={'index':'pindex','scaffold':'alt_scaf0','strand':'alt_strand0','distance':'alt_dist0'}), on=['center','pos'], how='left')
                    pos_errors['alt_id0'] = 0 # Anything as long as it is not below 0
                    # Replace deletions with the next scaffold, which should be found at this position instead
                    for h in range(0,ploidy):
                        pos_errors['npos'] = pos_errors['pos']
                        while np.sum((pos_errors[f'alt_scaf{h}'] == -1) & (pos_errors[f'alt_id{h}'] >= 0)):
                            pos_errors['npos'] += (1 if d == 'l' else -1)
                            if 0 == h:
                                pos_errors = pos_errors.merge(final_paths[['center','pos','scaffold','strand','distance']].rename(columns={'pos':'npos'}), on=['center','npos'], how='left')
                            else:
                                pos_errors = pos_errors.merge(final_paths[['center','pos','scaffold','strand','distance',f'alt_id{h}',f'alt_scaf{h}',f'alt_strand{h}',f'alt_dist{h}']].rename(columns={'pos':'npos',f'alt_id{h}':'nid',f'alt_scaf{h}':'nscaf',f'alt_strand{h}':'nstrand',f'alt_dist{h}':'ndist'}), on=['center','npos'], how='left')
                            pos_errors.loc[(pos_errors[f'alt_scaf{h}'] == -1) & (pos_errors[f'alt_id{h}'] >= 0) & np.isnan(pos_errors['scaffold']), f'alt_scaf{h}'] = -2 # Mark that we have a deletion at the end that cannot be verified
                            if 0 == h:
                                selection = pos_errors['alt_scaf0'] == -1
                                pos_errors.loc[selection, 'alt_strand0'] = pos_errors.loc[selection, 'strand'].values
                                pos_errors.loc[selection, ['alt_scaf0','alt_dist0']] = pos_errors.loc[selection, ['scaffold','distance']].values.astype(int)
                            else:
                                selection = (pos_errors[f'alt_scaf{h}'] == -1) & (pos_errors[f'alt_id{h}'] >= 0)
                                pos_errors.loc[selection, f'alt_strand{h}'] = np.where(pos_errors.loc[selection, 'nid'] < 0, pos_errors.loc[selection, 'strand'].values, pos_errors.loc[selection, 'nstrand'].values)
                                pos_errors.loc[selection, f'alt_dist{h}'] = np.where(pos_errors.loc[selection, 'nid'] < 0, pos_errors.loc[selection, 'distance'].values.astype(int), pos_errors.loc[selection, 'ndist'].values.astype(int))
                                pos_errors.loc[selection, f'alt_scaf{h}'] = np.where(pos_errors.loc[selection, 'nid'] < 0, pos_errors.loc[selection, 'scaffold'].values.astype(int), pos_errors.loc[selection, 'nscaf'].values.astype(int))
                                pos_errors.drop(columns=['nid','nscaf','nstrand','ndist'], inplace=True)
                            pos_errors.drop(columns=['scaffold','strand','distance'], inplace=True)
                    pos_errors.drop(columns=['npos'], inplace=True)
                    # Find positions that are not duplicated and not a deletion to have proper anchors
                    pos_errors['apos'] = np.where(pos_errors[[f'supp{h}' for h in range(0,ploidy)]].min(axis=1), -1, pos_errors['pos'] + (-1 if d == 'l' else 1)) # If we already have full support from previous direction, we do not need to test them anymore
                    pos_errors['adels'] = True
                    while True:
                        # Ensure apos is not a duplication
                        while True:
                            not_unique = pos_errors[['did','center','apos']].rename(columns={'center':f'center{p}','apos':f'pos{p}'}).merge(duplications[['did',f'center{p}',f'pos{p}']], on=['did',f'center{p}',f'pos{p}'], how='left', indicator=True)['_merge'].values == "both"
                            if np.sum(not_unique):
                                pos_errors.loc[not_unique,'apos'] += (-1 if d == 'l' else 1)
                            else:
                                break
                         # Get corresponding anchored connections from scaffold_graph
                        pos_errors = pos_errors.merge(pd.concat([final_paths[['center','pos','scaffold','strand','distance']].rename(columns={'pos':'apos','scaffold':'ascaf','strand':'astrand','distance':'adist'})]+[final_paths.loc[final_paths[f'alt_id{h}'] >= 0, ['center','pos',f'alt_scaf{h}',f'alt_strand{h}',f'alt_dist{h}']].rename(columns={'pos':'apos',f'alt_scaf{h}':'ascaf',f'alt_strand{h}':'astrand',f'alt_dist{h}':'adist'}) for h in range(1,ploidy)], ignore_index=True), on=['center','apos'], how='left')
                        # Ensure that we do not have deletions as ascaf
                        pos_errors = pos_errors[np.isnan(pos_errors['ascaf']) | (pos_errors['ascaf'] >= 0) | pos_errors['adels']].copy()
                        if np.sum(pos_errors['ascaf'] < 0) == 0:
                            break
                        else:
                            pos_errors['adels'] = False
                            # Take the next position, where we have a deletion (only for the deletion, the other scaffold can start there)
                            pos_errors.loc[pos_errors['ascaf'] < 0, 'apos'] += (-1 if d == 'l' else 1)
                            pos_errors.loc[pos_errors['ascaf'] < 0, 'adels'] = True
                            pos_errors.drop(columns=['ascaf','astrand','adist'], inplace=True)
                            pos_errors.drop_duplicates(inplace=True)
                    pos_errors.drop(columns=['adels'], inplace=True)
                    # Check if we have support
                    if np.sum(np.isnan(pos_errors['ascaf'])) < len(pos_errors):
                        pos_errors.loc[np.isnan(pos_errors['ascaf']), 'apos'] = -1 # If the anchor is after the end of the path also set it to -1
                        pos_errors['ascaf'] = pos_errors['ascaf'].fillna(-1).astype(int)
                        if d == 'l':
                            pos_errors['aside'] = np.where(pos_errors['astrand'] == '+', 'r', 'l')
                            pos_errors.drop(columns=['adist'], inplace=True) # For left direction adist is outside of the compared region
                        else:
                            pos_errors['aside'] = np.where(pos_errors['astrand'] == '+', 'l', 'r')
                            pos_errors.rename(columns={'adist':'dist1'}, inplace=True) # For right direction we do the comparison directly in the merge
                            pos_errors['dist1'] = pos_errors['dist1'].fillna(0).astype(int)
                        pos_errors['cpos'] = np.where(pos_errors['apos'] == -1, -1, pos_errors['apos'] + (1 if d == 'l' else -1))
                        pos_errors = pos_errors.merge(scaffold_graph.rename(columns={'from':'ascaf','from_side':'aside'}), on=(['ascaf','aside'] if d=='l' else ['ascaf','aside','dist1']), how='left')
                        pos_errors.drop(columns=['ascaf','astrand','aside'], inplace=True)
                        pos_errors = RemoveEmptyColumns(pos_errors)
                        if np.sum(np.isnan(pos_errors['length'])) < len(pos_errors):
                            pos_errors.loc[np.isnan(pos_errors['length']),'cpos'] = -1
                            max_len = pos_errors['length'].max().astype(int)
                            for s in range(1, max_len):
                                if np.sum(pos_errors['cpos'] < 0) < len(pos_errors):
                                    pos_errors['cdels'] = True
                                    while True:
                                        # Check if we have support where we are at pos
                                        for h in range(0,ploidy):
                                            if d == 'l':
                                                pos_errors.loc[(pos_errors['pos'] == pos_errors['cpos']) & (pos_errors[f'alt_scaf{h}'] == pos_errors[f'scaf{s}']) &
                                                               (pos_errors[f'alt_strand{h}'] == pos_errors[f'strand{s}']) & (pos_errors[f'alt_dist{h}'] == pos_errors[f'dist{s}']), f'supp{h}'] = True
                                            else:
                                                if s+1 != max_len:
                                                    pos_errors.loc[(pos_errors['pos'] == pos_errors['cpos']) & (pos_errors[f'alt_scaf{h}'] == pos_errors[f'scaf{s}']) & (pos_errors[f'alt_strand{h}'] != pos_errors[f'strand{s}']) & (pos_errors[f'alt_dist{h}'] == pos_errors[f'dist{s+1}']), f'supp{h}'] = True
                                        if d == 'r':
                                            # Make another round where we do not require to have the same distance if we otherwise cannot decide (to at least remove scaffolds that do not fit, even if the distances cannot be checked)
                                            nosupp = (pos_errors['pos'] == pos_errors['cpos']) & (pos_errors[[f'supp{h}' for h in range(0,ploidy)]].max(axis=1) == False)
                                            for h in range(0,ploidy):
                                                pos_errors.loc[nosupp & (pos_errors[f'alt_scaf{h}'] == pos_errors[f'scaf{s}']) & (pos_errors[f'alt_strand{h}'] != pos_errors[f'strand{s}']), f'supp{h}'] = True
                                        pos_errors.loc[pos_errors['pos'] == pos_errors['cpos'], 'cpos'] = -1
                                        # Get the scaffold in final_paths corresponding to current scaffold s in scaffold_graph
                                        pos_errors = pos_errors.merge(pd.concat([final_paths[['center','pos','scaffold','strand','distance']].rename(columns={'pos':'cpos','scaffold':'cscaf','strand':'cstrand','distance':'cdist'})]+[final_paths.loc[final_paths[f'alt_id{h}'] >= 0, ['center','pos',f'alt_scaf{h}',f'alt_strand{h}',f'alt_dist{h}']].rename(columns={'pos':'cpos',f'alt_scaf{h}':'cscaf',f'alt_strand{h}':'cstrand',f'alt_dist{h}':'cdist'}) for h in range(1,ploidy)], ignore_index=True), on=['center','cpos'], how='left')
                                        pos_errors = pos_errors[np.isnan(pos_errors['cscaf']) | (pos_errors['cscaf'] >= 0) | pos_errors['cdels']].copy()
                                        if np.sum(pos_errors['cscaf'] < 0) == 0:
                                            break
                                        else:
                                            pos_errors['cdels'] = False
                                            # Skip deletions
                                            pos_errors.loc[pos_errors['cscaf'] < 0, 'cpos'] += (1 if d == 'l' else -1)
                                            pos_errors.loc[pos_errors['cscaf'] < 0, 'cdels'] = True
                                            pos_errors.drop(columns=['cscaf','cstrand','cdist'], inplace=True)
                                            pos_errors.drop_duplicates(inplace=True)
                                    pos_errors.drop(columns=['cdels'], inplace=True)
                                    # Check that the connection is valid where we are not yet at pos
                                    pos_errors.loc[pos_errors['length'] <= s+1,'cpos'] = -1 # If this is the last scaffold in the connection from scaffold_graph, we have not chance of reaching pos (we already handled deletions)
                                    pos_errors.loc[(pos_errors[f'scaf{s}'] != pos_errors['cscaf']) | ((pos_errors[f'strand{s}'] != pos_errors['cstrand']) if d=='l' else (pos_errors[f'strand{s}'] == pos_errors['cstrand'])),'cpos'] = -1
                                    if np.sum(pos_errors['cpos'] < 0) < len(pos_errors):
                                        # If we have connections longer than this we can safely check this also for the right direction, where we need to check the next dist
                                        if d == 'l':
                                            pos_errors.loc[(pos_errors[f'dist{s}'] != pos_errors['cdist']),'cpos'] = -1
                                        else:
                                            pos_errors.loc[(pos_errors[f'dist{s+1}'] != pos_errors['cdist']),'cpos'] = -1
                                    pos_errors.loc[pos_errors['cpos'] >= 0,'cpos'] += (1 if d == 'l' else -1)
                                    pos_errors.drop(columns=['cscaf','cstrand','cdist'], inplace=True)
                                pos_errors.drop(columns=[f'scaf{s}',f'strand{s}',f'dist{s}'], inplace=True)
                                pos_errors.drop_duplicates(inplace=True)
                    # For the haplotypes identical to main assign support from main (only after doing both directions)
                    for h in range(1,ploidy):
                        pos_errors.loc[pos_errors[f'alt_id{h}'] < 0, f'supp{h}'] = pos_errors.loc[pos_errors[f'alt_id{h}'] < 0, 'supp0']
                    pos_errors = pos_errors.groupby(['did','center','pos','pindex'])[[f'supp{h}' for h in range(0,ploidy)]].max().reset_index()
                # Store completely unsupported duplications
                completely_unsupported = pos_errors[[f'supp{h}' for h in range(0,ploidy)]].max(axis=1) == False
                unsupported_dups.append(pos_errors[completely_unsupported].drop(columns=[f'supp{h}' for h in range(0,ploidy)]))
                # Drop pos_errors where nothing has to be done, since they are either completely supported or completely unsupported
                pos_errors = pos_errors[(completely_unsupported == False) & (pos_errors[[f'supp{h}' for h in range(0,ploidy)]].min(axis=1) == False)].copy()
                # Remove unsupported alternative paths
                for h in range(1,ploidy):
                    unsupp = np.isin(final_paths[f'alt_id{h}'], final_paths.loc[pos_errors.loc[pos_errors[f'supp{h}'] == False, 'pindex'].values, f'alt_id{h}'].values)
                    final_paths, duplications, fixed_errors = RemoveUnsupportedDuplicationsAndPath(final_paths, duplications, fixed_errors, unsupp, h, ploidy)
                # Remove unsupported main
                pos_errors = pos_errors.loc[pos_errors['supp0'] == False, :]
                for h in range(1,ploidy):
                    # First switch the haplotype h where it is supported with main
                    switch = np.isin(final_paths[f'alt_id{h}'], final_paths.loc[pos_errors.loc[pos_errors[f'supp{h}'], 'pindex'].values, f'alt_id{h}'].values)
                    if np.sum(switch):
                        final_paths = SwitchPathWithMain(final_paths, switch, h)
                        unsupp = np.isin(final_paths.index.values, pos_errors.loc[pos_errors[f'supp{h}'], 'pindex'].values)
                        unsupp = switch & (unsupp | ( (final_paths[f'alt_scaf{h}'] >= 0) & (final_paths['scaffold'] == final_paths[f'alt_scaf{h}']) & (final_paths['strand'] == final_paths[f'alt_strand{h}']) &
                                                      (final_paths['center'] == final_paths['center'].shift(1)) & np.concatenate([[False], unsupp[:-1]]) )) # Extend the unsupported with the distance variant at the end of the bubble
                        final_paths, duplications, fixed_errors = RemoveUnsupportedDuplicationsAndPath(final_paths, duplications, fixed_errors, unsupp, h, ploidy)
                    pos_errors = pos_errors.loc[pos_errors[f'supp{h}'] == False, :]
        # In case we have multiple haplotypes partiallly duplicated drop the duplicated ones if they are not supported
        unsupp = []
        for p, q in zip(['a','b'], ['b','a']):
            unsupported = (duplications[f'full{p}'] == False) & (duplications[f'nhaps{q}'] == 1) & (duplications[['did',f'center{p}',f'pos{p}']].rename(columns={f'center{p}':'center',f'pos{p}':'pos'}).merge(unsupported_dups[0 if p=='a' else 1][['did','center','pos']], on=['did','center','pos'], how='left', indicator=True)['_merge'].values == "both")
            dupsp = final_paths.merge(duplications.loc[unsupported, [f'center{p}',f'pos{p}']].rename(columns={f'center{p}':'center',f'pos{p}':'pos'}), on=['center','pos'], how='right')
            dupsq = final_paths.merge(duplications.loc[unsupported, [f'center{q}',f'pos{q}']].rename(columns={f'center{q}':'center',f'pos{q}':'pos'}), on=['center','pos'], how='right')
            # Remove unsupported alternative paths
            for h in range(1,ploidy):
                removal = (dupsp[f'alt_id{h}'].values >= 0) & (dupsp[f'alt_scaf{h}'].values == dupsq['scaffold'].values) & (np.where(dupsp[f'alt_scaf{h}'].values < 0, True, duplications.loc[unsupported, 'samedir'].values) == (dupsp[f'alt_strand{h}'].values == dupsq['strand'].values))
                if np.sum(removal):
                    final_paths, unsupp = RemoveHaplotypeFromPath(final_paths, unsupp, dupsp.loc[removal, f'alt_id{h}'].values, h)
                    dupsp.loc[removal, f'alt_id{h}'] = -1 # Ensure that the unsupported alternative is not used to replace unsupported main paths
            # Remove unsupported main
            removal = (dupsp['scaffold'].values == dupsq['scaffold'].values) & (np.where(dupsp['scaffold'].values < 0, True, duplications.loc[unsupported, 'samedir'].values) == (dupsp['strand'].values == dupsq['strand'].values))
            final_paths, unsupp = RemoveMainFromPath(final_paths, unsupp, dupsp, removal, ploidy, '')
        duplications = RemoveUnsupportedDuplications(duplications, final_paths, pd.concat(unsupp, ignore_index=False).drop_duplicates().index.values, ploidy)
        rem_scafs = []
        # Get unique scaffolds (we will end the check if we hit one, because afterwards should be ok)
        unique_scaffolds, unique_counts = np.unique(np.concatenate([final_paths.loc[final_paths['scaffold'] >= 0, 'scaffold'].values] + [final_paths.loc[final_paths[f'alt_scaf{h}'] >= 0, f'alt_scaf{h}'].values for h in range(1,ploidy)]), return_counts=True)
        unique_scaffolds = unique_scaffolds[unique_counts == 1]
        # Check if we lost support for other scaffolds by removing the unsupported ones (only up to the first unique, which should guarantee that the ones after it are supported)
        for d in ['l','r']:
            unsupported = pd.concat(unsupp, ignore_index=True).sort_values(['center','pos']).drop_duplicates()
            # Take the last one in the direction we are going (since it reaches the furthest)
            shift = (1 if d=='l' else -1)
            unsupported = unsupported.loc[(unsupported['center'] != unsupported['center'].shift(shift)) | (unsupported['pos'] != unsupported['pos'].shift(shift)+shift), :]
            if d == 'r':
                # If this scaffold is duplicated shift position to include the distance to the scaffold
                unsupported.loc[unsupported.merge(pd.concat([duplications[[f'center{p}',f'pos{p}']].rename(columns={f'center{p}':'center',f'pos{p}':'pos'}) for p in ['a','b']], ignore_index=True).drop_duplicates(), on=['center','pos'], how='left', indicator=True)['_merge'].values == "both", 'pos'] -= 1
            # Skip deletions
            unsupported['dels'] = True
            while True:
                unsupported = unsupported[['center','pos','dels']].drop_duplicates().merge(final_paths, on=['center','pos'], how='inner')
                unsupported = pd.concat([unsupported[['center','pos','scaffold','strand','distance','dels']]] + [unsupported.loc[unsupported[f'alt_id{h}'] >= 0, ['center','pos',f'alt_scaf{h}',f'alt_strand{h}',f'alt_dist{h}','dels']].rename(columns={f'alt_scaf{h}':'scaffold',f'alt_strand{h}':'strand',f'alt_dist{h}':'distance'}) for h in range(1,ploidy)], ignore_index=True)
                unsupported = unsupported.loc[(unsupported['scaffold'] >= 0) | unsupported['dels'], :]
                if np.sum(unsupported['scaffold'] < 0) == 0:
                    break
                else:
                    unsupported.loc[unsupported['scaffold'] >= 0, 'dels'] = False
                    unsupported.loc[unsupported['scaffold'] < 0, 'pos'] += shift
            unsupported.drop(columns=['dels'], inplace=True)
            unsupported.sort_values(['center','pos'], inplace=True)
            # Check scaffold graph
            if len(unsupported):
                unsupported = unsupported.merge(final_paths.groupby(['center'])['pos'].max().reset_index(name='last_pos'), on=['center'], how='left')
                if d == 'l':
                    unsupported['side'] = np.where(unsupported['strand'] == '+', 'l', 'r')
                    unsupported.rename(columns={'distance':'dist1'}, inplace=True) # For right direction we do the comparison directly in the merge
                else:
                    unsupported['side'] = np.where(unsupported['strand'] == '+', 'r', 'l')
                    unsupported.drop(columns=['distance'], inplace=True) # For left direction adist is outside of the compared region
                unsupported['cpos'] = unsupported['pos'] - shift
                unsupported = unsupported.merge(scaffold_graph.rename(columns={'from':'scaffold','from_side':'side'}), on=(['scaffold','side'] if d=='r' else ['scaffold','side','dist1']), how='left')
                unsupported['cscaf'] = 1
                unsupported = RemoveEmptyColumns(unsupported)
                # Remove final_paths, where we do not have any connection (for left direction it happens automatically with the direction check)
                if d == 'r':
                    if np.sum(np.isnan(unsupported['length'])):
                        tmp_paths = unsupported.loc[np.isnan(unsupported['length']), ['center','pos']].merge(final_paths, on=['center','pos'], how='left')
                        for h in range(1,ploidy):
                            tmp_paths['hrem'] = (tmp_paths[f'alt_scaf{h}'].values == unsupported['scaffold'].values) & (tmp_paths[f'alt_strand{h}'].values == unsupported['strand'].values) & (tmp_paths[f'alt_id{h}'] >= 0)
                            if np.sum(tmp_paths['hrem']):
                                final_paths, rem_scafs = RemoveHaplotypeFromPath(final_paths, rem_scafs, tmp_paths.loc[tmp_paths['hrem'], f'alt_id{h}'].values, h)
                                tmp_paths.loc[tmp_paths['hrem'], f'alt_id{h}'] = -1
                        tmp_paths['hrem'] = (tmp_paths['scaffold'].values == unsupported['scaffold'].values) & (tmp_paths['strand'].values == unsupported['strand'].values)
                        if np.sum(tmp_paths['hrem']):
                            final_paths, rem_scafs = RemoveMainFromPath(final_paths, rem_scafs, tmp_paths, tmp_paths['hrem'], ploidy, 'r')
                    unsupported = unsupported.loc[np.isnan(unsupported['length']) == False, :]
                # Check first distance if left direction
                else:
                    unsupported = unsupported.loc[np.isnan(unsupported['length']) == False, :]
                    tmp_paths = unsupported[['center','pos']].drop_duplicates().merge(final_paths, on=['center','pos'], how='left')
                    # Check alternatives
                    for h in range(1,ploidy):
                        tmp_paths['dist1'] = tmp_paths[f'alt_dist{h}']
                        tmp_paths['hsupp'] = (tmp_paths.merge(unsupported[['center','pos','dist1']].drop_duplicates(), on=['center','pos','dist1'], how='left', indicator=True)['_merge'].values == "both") | (tmp_paths[f'alt_scaf{h}'] < 0)
                        if np.sum(tmp_paths['hsupp'] == False):
                            final_paths, rem_scafs = RemoveHaplotypeFromPath(final_paths, rem_scafs, tmp_paths.loc[tmp_paths['hsupp'] == False, f'alt_id{h}'].values, h)
                            tmp_paths.loc[tmp_paths['hsupp'] == False, f'alt_id{h}'] = -1 # Ensure that the unsupported alternative is not used to replace unsupported main paths
                    # Check main
                    tmp_paths['dist1'] = tmp_paths['distance']
                    tmp_paths['hsupp'] = (tmp_paths.merge(unsupported[['center','pos','dist1']].drop_duplicates(), on=['center','pos','dist1'], how='left', indicator=True)['_merge'].values == "both") | (tmp_paths['scaffold'] < 0)
                    if np.sum(tmp_paths['hsupp'] == False):
                        final_paths, rem_scafs = RemoveMainFromPath(final_paths, rem_scafs, tmp_paths, tmp_paths['hsupp'] == False, ploidy, 'l')
                unsupported.drop(columns=['scaffold','strand','side'], inplace=True)
                # Remove connections, where we cannot find anything anymore
                unsupported = unsupported.loc[(unsupported['cscaf'] < unsupported['length']) & (unsupported['cpos'] >= 0) & (unsupported['cpos'] <= unsupported['last_pos']), :]
                while len(unsupported):
                    # Always look at only one position per center (so ignore positions that are ahead of the others)
                    mcpos = unsupported.groupby(['center'])['cpos'].agg(['size']+(['max'] if d == 'l' else ['min']))
                    mcpos = np.repeat(mcpos['max' if d == 'l' else 'min'].values, mcpos['size'].values)
                    unsupported['mpos'] = np.where(mcpos == unsupported['cpos'], unsupported['cpos'], -1)
                    # Get the path information to compare the scaffold graph connection to
                    comp_paths = unsupported[['center','cpos','mpos']].merge(final_paths.rename(columns={'pos':'mpos'}), on=['center','mpos'], how='left')
                    unsupported.reset_index(drop=True, inplace=True) # To be able to compare comp_paths and unsupported without always adding .values
                    # Skip deletions
                    comp_paths['dels0'] = False
                    for h in range(1,ploidy):
                        comp_paths[f'dels{h}'] = False
                    while True:
                        comp_paths['has_dels'] = comp_paths['scaffold'] < 0
                        for h in range(1,ploidy):
                            comp_paths.loc[(comp_paths[f'alt_id{h}'] >= 0) & (comp_paths[f'alt_scaf{h}'] < 0),'has_dels'] = True
                        comp_paths.loc[(comp_paths['mpos'] <= 0) & (comp_paths['mpos'] >= unsupported['last_pos']),'has_dels'] = False # If we nothing is following the deletion there is nothing we can do
                        if np.sum(comp_paths['has_dels']) == 0:
                            break
                        else:
                            comp_paths.loc[comp_paths['has_dels'], 'mpos'] -= shift
                            tmp_paths = comp_paths[['center','mpos']].merge(final_paths.rename(columns={'pos':'mpos'}), on=['center','mpos'], how='left')
                            dels = comp_paths['has_dels'] & (comp_paths['scaffold'] < 0)
                            comp_paths.loc[dels, 'dels0'] = True
                            comp_paths.loc[dels, ['scaffold','strand','distance']] = tmp_paths.loc[dels, ['scaffold','strand','distance']].values
                            for h in range(1,ploidy):
                                dels = comp_paths['has_dels'] & (comp_paths[f'alt_id{h}'] >= 0) & (comp_paths[f'alt_scaf{h}'] < 0)
                                if np.sum(dels):
                                    comp_paths.loc[dels, f'dels{h}'] = True
                                    comp_paths.loc[dels, [f'alt_id{h}',f'alt_scaf{h}',f'alt_strand{h}',f'alt_dist{h}']] = tmp_paths.loc[dels, [f'alt_id{h}',f'alt_scaf{h}',f'alt_strand{h}',f'alt_dist{h}']].values
                    comp_paths.drop(columns=['has_dels'], inplace=True)
                    comp_paths.rename(columns={'scaffold':'alt_scaf0', 'strand':'alt_strand0', 'distance':'alt_dist0'}, inplace=True)
                    # Check the haplotypes at cmpos one after another
                    unsupported['delsupp'] = False # If a deletion is supported, we cannot go to the next cscaf afterwards, because we still need it to support the position after the deletion
                    for h in range(0,ploidy):
                        unsupported[f'hsupp{h}'] = False
                        unsupported[f'hsupp_full{h}'] = False
                        for s in range(unsupported['cscaf'].min(),unsupported['cscaf'].max()+1):
                            cur = s == unsupported['cscaf']
                            unsupported.loc[cur, f'hsupp{h}'] = (unsupported.loc[cur,f'scaf{s}'] == comp_paths.loc[cur,f'alt_scaf{h}']) & ((False if d=='l' else True) == (unsupported.loc[cur,f'strand{s}'] == comp_paths.loc[cur,f'alt_strand{h}']))
                            if d == 'l':
                                unsupported.loc[cur & unsupported[f'hsupp{h}'] & (comp_paths[f'dels{h}'] | (unsupported['cpos'] == 0)), f'hsupp_full{0}'] = True # Also make the last scaffold in the path automatically fully supported, because you cannot have full support there and this would remove all alternatives to deletions otherwise
                                if s+1 < unsupported['length'].max():
                                    unsupported.loc[cur & unsupported[f'hsupp{h}'] & (unsupported[f'scaf{s+1}'] == comp_paths[f'alt_scaf{h}']), f'hsupp_full{0}'] = True
                            else:
                                unsupported.loc[cur, f'hsupp{h}'] = unsupported.loc[cur, f'hsupp{h}'] & (unsupported.loc[cur,f'dist{s}'] == comp_paths.loc[cur,f'alt_dist{h}'])
                        if d == 'r': # For right direction the distance is always included   
                            unsupported[f'hsupp_full{h}'] = unsupported[f'hsupp{h}']
                        unsupported.loc[unsupported[f'hsupp{h}'] & comp_paths[f'dels{h}'], 'delsupp'] = True
                    # Combine support from all connections with the same center,cpos
                    support = unsupported[unsupported['mpos'] == unsupported['cpos']].groupby(['center','cpos'])[[f'hsupp{h}' for h in range(0,ploidy)]+[f'hsupp_full{h}' for h in range(0,ploidy)]].max().reset_index()
                    # If one haplotype has full support do not accept less than full support for the others with the same center,cpos
                    support['req_full'] = support[[f'hsupp_full{h}' for h in range(0,ploidy)]].max(axis=1)
                    unsupported = unsupported.merge(support[['center','cpos','req_full']], on=['center','cpos'], how='left')
                    unsupported['req_full'] = unsupported['req_full'].fillna(False)
                    for h in range(0,ploidy):
                        support.loc[support['req_full'], f'hsupp{h}'] = support.loc[support['req_full'], f'hsupp_full{h}']
                        unsupported.loc[unsupported['req_full'], f'hsupp{h}'] = unsupported.loc[unsupported['req_full'], f'hsupp_full{h}']
                    unsupported.drop(columns=['req_full'], inplace=True)
                    # Remove haplotypes without support
                    comp_paths = comp_paths.drop(columns=[f'dels{h}' for h in range(0,ploidy)]).drop_duplicates().sort_values(['center','cpos'])
                    for h in range(1,ploidy):
                        removal = (support[f'hsupp{h}'].values == False) & (comp_paths[f'alt_id{h}'].values >= 0)
                        if np.sum(removal):
                            final_paths, rem_scafs = RemoveHaplotypeFromPath(final_paths, rem_scafs, comp_paths.loc[removal, f'alt_id{h}'].values, h)
                            comp_paths.loc[removal, f'alt_id{h}'] = -1 # Ensure that the unsupported alternative is not used to replace unsupported main paths
                    removal = (support['hsupp0'].values == False)
                    if np.sum(removal):
                        final_paths, rem_scafs = RemoveMainFromPath(final_paths, rem_scafs, comp_paths.rename(columns={'cpos':'pos'}), removal, ploidy, d)
                    # Remove connections without support
                    unsupported = unsupported.loc[unsupported[[f'hsupp{h}' for h in range(0,ploidy)]].max(axis=1), :]
                    if len(unsupported):
                        # End check if we find a unique scaffold
                        for s in range(unsupported['cscaf'].min(),unsupported['cscaf'].max()+1):
                            unsupported = unsupported.loc[(s != unsupported['cscaf']) | (np.isin(unsupported[f'scaf{s}'], unique_scaffolds) == False), :]
                        # Update cscaf and cpos
                        unsupported.loc[(unsupported['delsupp'] == False) & (unsupported['mpos'] == unsupported['cpos']), 'cscaf'] += 1
                        unsupported.loc[unsupported['mpos'] == unsupported['cpos'], 'cpos'] -= shift
                        # Remove connections, where we cannot find anything anymore
                        unsupported = unsupported.loc[(unsupported['cscaf'] < unsupported['length']) & (unsupported['cpos'] >= 0) & (unsupported['cpos'] <= unsupported['last_pos']), :]
        if len(rem_scafs):
            duplications = RemoveUnsupportedDuplications(duplications, final_paths, pd.concat(rem_scafs, ignore_index=False).drop_duplicates().index.values, ploidy)
        duplications.drop(columns=['nhapsa','nhapsb'], inplace=True)
#
    if len(duplications):
        ## Split duplications by assigning different duplication ids, where we do not want to connect the duplications later and keep track if they are connected to the start or end of read
        duplications['starta'] = False
        duplications.loc[(duplications['centera'] != duplications['centera'].shift(1)) & (duplications['centerb'] != duplications['centerb'].shift(1)), 'starta'] = True
        duplications['startb'] = duplications['starta']
        duplications['enda'] = False
        duplications.loc[(duplications['centera'] != duplications['centera'].shift(-1)) & (duplications['centerb'] != duplications['centerb'].shift(-1)), 'enda'] = True
        duplications['endb'] = duplications['enda']
#
        # Start by spliting duplications where the unique path in between the duplications contains alternative haplotypes
        for d in ['next','prev']:
            if d == 'next':
                duplications['bridgea'] = np.where(duplications['did'] != duplications['did'].shift(-1), duplications['lena']-duplications['posa']-1, duplications['posa'].shift(-1, fill_value=0)-duplications['posa']-1)
                duplications['bridgeb'] = np.where(duplications['did'] != duplications['did'].shift(-1), np.where(duplications['samedir'],duplications['lenb']-duplications['posb']-1,-duplications['posb']), duplications['posb'].shift(-1, fill_value=0)-duplications['posb']-np.where(duplications['samedir'],1,-1))
            else:
                duplications['bridgea'] = np.where(duplications['starta'], -duplications['posa'], 0)
                duplications['bridgeb'] = np.where(duplications['startb'], np.where(duplications['samedir'],-duplications['posb'],duplications['lenb']-duplications['posb']-1), 0)
#
            for p,q in zip(['a','b'], ['b','a']):
                tmp_paths = duplications.loc[np.repeat(duplications.index.values, np.abs(duplications[f'bridge{p}'].values)), [f'center{p}',f'pos{p}',f'bridge{p}']].rename(columns={f'center{p}':'center',f'pos{p}':'pos',f'bridge{p}':'dir'}).reset_index()
                tmp_paths['dir'] = np.where(tmp_paths['dir'] < 0, -1, 1) # values with 0 do not exist in tmp_paths['dir']
                tmp_paths['pos'] += tmp_paths.groupby(['index'], sort=False)['dir'].cumsum()
                tmp_paths = FillBridgePathInfo(tmp_paths, final_paths, knots, ploidy)
                duplications[f'{d}_dist{p}'] = 0
                duplications.loc[tmp_paths['index'].values, f'{d}_dist{p}'] = tmp_paths['scaf_len'].values
                duplications[f'{d}_dels{p}'] = 0
                duplications.loc[tmp_paths['index'].values, f'{d}_dels{p}'] = tmp_paths['ndels'].values
                duplications[f'{d}_haps{p}'] = 0
                duplications.loc[tmp_paths['index'].values, f'{d}_haps{p}'] = tmp_paths['nhaps'].values
                if d == 'next':
                    split_dup = (1 < duplications[f'next_haps{p}']) | ((0 < duplications[f'next_dels{p}']) & (duplications[f'end{p}'] | (np.abs(duplications[f'bridge{p}'])-np.abs(duplications[f'bridge{q}'])!=duplications[f'next_dels{p}']))) 
                    duplications.loc[split_dup, f'end{p}'] = False
                    duplications['did'] = ((duplications['did'] != duplications['did'].shift(1)) | split_dup.shift(1)).cumsum()
                else:
                    duplications.loc[(1 < duplications[f'prev_haps{p}']) | (0 < duplications[f'prev_dels{p}']), f'start{p}'] = False
        duplications.drop(columns=['bridgea','bridgeb','next_hapsa','next_hapsb','prev_hapsa','prev_hapsb','next_delsa','next_delsb','prev_delsa','prev_delsb'], inplace=True)
#
        # Remove positions and split paths where the path (a or b) with the lowest amount of not full duplicates has not full duplicates
        dsize = duplications.groupby(['did'], sort=False).agg({'fulla':['size','sum'], 'fullb':['sum']})
        duplications['nfulla'] = np.repeat(dsize[('fulla','sum')].values, dsize[('fulla','size')].values)
        duplications['nfullb'] = np.repeat(dsize[('fullb','sum')].values, dsize[('fulla','size')].values)
        split_dup = ((duplications['nfulla'] > duplications['nfullb']) & (duplications['fulla'] == False)) | ((duplications['nfulla'] <= duplications['nfullb']) & (duplications['fullb'] == False))
        duplications['did'] = ((duplications['did'] != duplications['did'].shift(1)) | split_dup.shift(1)).cumsum()
        duplications.drop(columns=['nfulla','nfullb'], inplace=True)
        duplications = duplications[split_dup == False].copy()
#
        # Update dsitances where we split or removed ends
        for p in ['a','b']:
            duplications.loc[duplications[f'start{p}'] == False, f'prev_dist{p}'] = 0
            duplications.loc[(duplications[f'end{p}'] == False) & (duplications['did'] != duplications['did'].shift(-1)), f'next_dist{p}'] = 0
        # Split duplications at longest separation until the duplicated part is larger than the unique part
        duplications = duplications.merge(final_paths[['center','pos','scaffold']].rename(columns={'center':'centera','pos':'posa'}), on=['centera','posa'], how='left')
        duplications = duplications.merge(knots[['scaffold','scaf_len']], on=['scaffold'], how='left')
        duplications['scaf_len'] = np.where(duplications['scaffold'] == -1, 0, duplications['scaf_len']).astype(int)
        duplications.drop(columns=['scaffold'], inplace=True)
        nsplit = 1
        while nsplit:
            # Filter duplications not reaching an end (since we already removed complete duplicates, they are not interesting)
            duplications['enddup'] = duplications['starta'] | duplications['startb'] | duplications['enda'] | duplications['endb']
            enddups = duplications.groupby(['did'], sort=False)['enddup'].agg(['size','sum'])
            duplications = duplications[np.repeat(enddups['sum'].values.astype(bool), enddups['size'].values)].copy()
            # Split path
            nsplit = 0
            for p in ["a","b"]:
                duplications['uniquelen'] = duplications[f'next_dist{p}'] + duplications[f'prev_dist{p}']
                duplen = duplications.groupby(['did'], sort=False).agg({'uniquelen':['size','sum'],'scaf_len':['sum'],f'next_dist{p}':['max'],f'prev_dist{p}':['max']})
                duplen['split'] = duplen['uniquelen','sum'] > duplen['scaf_len','sum']
                duplen['splitnext'] = duplen['split'] & (duplen[f'next_dist{p}','max'] >= duplen[f'prev_dist{p}','max'])
                duplen['splitprev'] = duplen['split'] & (duplen['splitnext'] == False)
                duplications.loc[np.repeat(duplen['splitprev'].values, duplen['uniquelen','size'].values), f'start{p}'] = False
                duplications.loc[duplications[f'start{p}'] == False, f'prev_dist{p}'] = 0
                split_dup = np.repeat(duplen['splitnext'].values, duplen['uniquelen','size'].values) & (np.repeat(duplen[f'next_dist{p}','max'].values, duplen['uniquelen','size'].values) == duplications[f'next_dist{p}'].values)
                nsplit += np.sum(split_dup)
                duplications.loc[split_dup, f'next_dist{p}'] = 0
                duplications.loc[split_dup, f'end{p}'] = False
                duplications['did'] = ((duplications['did'] != duplications['did'].shift(1)) | np.concatenate([[False],split_dup[:-1]])).cumsum() # Shift(1) manually done because it is an array not a pd.Series
        duplications.drop(columns=['next_dista','next_distb','prev_dista','prev_distb','scaf_len','enddup','uniquelen'], inplace=True)
        
        # Require that both paths have start or end
        duplications['start'] = duplications['starta'] & duplications['startb']
        duplications['end'] = duplications['enda'] & duplications['endb']
        duplications.drop(columns=['starta','startb','enda','endb'], inplace=True)
        ends = duplications.groupby(['did'], sort=False).agg({'start':['max','size'], 'end':['max']})
        duplications['start'] = np.repeat(ends['start','max'].values, ends['start','size'].values)
        duplications['end'] = np.repeat(ends['end','max'].values, ends['start','size'].values)
        duplications = duplications.loc[duplications['start'] | duplications['end'], :]

    if len(duplications):
        # Merge duplications if both paths reach both ends and do not have unique alternatives
        mergers = duplications[duplications['start'] & duplications['end']].drop(columns=['start','end'])
        tmp_paths = final_paths.merge(pd.concat([mergers[['did',f'center{p}']].rename(columns={f'center{p}':'center'}) for p in ['a','b']], ignore_index=True).drop_duplicates(), on=['center'], how='inner') # Get all positions in path related to the potential mergers
        tmp_paths = tmp_paths.loc[tmp_paths.merge(pd.concat([mergers.loc[mergers[f'full{p}'], ['did',f'center{p}',f'pos{p}']].rename(columns={f'center{p}':'center',f'pos{p}':'pos'}) for p in ['a','b']], ignore_index=True).drop_duplicates(), on=['did','center','pos'], how='left', indicator=True).values == "left_only", :] # Keep only the unique positions
        tmp_paths = tmp_paths.loc[tmp_paths[[f'alt_id{h}' for h in range(1,ploidy)]].max(axis=1) >= 0, :] # Keep only positions with alternative haplotypes
        mergers.drop(columns=['fulla','fullb'], inplace=True)
        if len(tmp_paths):
            mergers = mergers.loc[np.isin(mergers['did'].values, np.unique(tmp_paths['did'].values)) == False, :]
        if len(mergers):
            # We do not need to handle the mergers afterwards anymore (remove from duplications)
            merged_centers = np.unique(np.concatenate([mergers[f'center{p}'].values for p in ['a','b']]))
            duplications = duplications.loc[(np.isin(duplications['centera'].values, merged_centers) == False) & (np.isin(duplications['centerb'].values, merged_centers) == False), :]
            # If we have multiple mergers (polyploid case) remove all centera that also appear in centerb, since we later will always merge to a and thus merge all others to the lowest center
            mergers = mergers.loc[np.isin(mergers['centera'].values, np.unique(mergers['centerb'].values)) == False, :]
            # Add positions to centera, where centerb has more, such that we can merge b into a
            mergers['add_before'] = np.maximum(0, np.where(mergers['did'] == mergers['did'].shift(1), 0, np.where(mergers['samedir'], mergers['posb']-mergers['posa'], (mergers['lenb']-mergers['posb']-1)-mergers['posa'])))
            mergers['add_after'] = np.maximum(0, np.where(mergers['did'] == mergers['did'].shift(-1), np.abs(mergers['posb'].shift(-1, fill_value=0)-mergers['posb'])-(mergers['posa'].shift(-1, fill_value=0)-mergers['posa']), np.where(mergers['samedir'], (mergers['lenb']-mergers['posb'])-(mergers['lena']-mergers['posa']), mergers['posb']-(mergers['lena']-mergers['posa']-1))))
            final_paths = final_paths.merge(mergers.groupby(['centera','posa'])[['add_before','add_after']].max().reset_index().rename(columns={'centera':'center','posa':'pos'}), on=['center','pos'], how='left')
            final_paths[['add_before','add_after']] = final_paths[['add_before','add_after']].fillna(0).astype(int)
            final_paths.reset_index(inplace=True)
            final_paths = final_paths.loc[np.repeat(final_paths.index.values, final_paths['add_before'].values+final_paths['add_after'].values+1)].reset_index(drop=True)
            final_paths['ipos'] = final_paths.groupby(['index'],sort=False).cumcount()
            added = final_paths['ipos'] != final_paths['add_before']
            final_paths.loc[added, ['scaffold','strand','distance']] = [-1,'',0]
            for h in range(1,ploidy):
                final_paths.loc[added, [f'alt_id{h}',f'alt_scaf{h}',f'alt_strand{h}',f'alt_dist{h}']] = [-1,-1,'',0]
            final_paths['newpos'] = final_paths.groupby(['center'],sort=False).cumcount()
            mergers = mergers.merge(final_paths.loc[added==False,['center','pos','newpos']].rename(columns={'center':'centera','pos':'posa'}), on=['centera','posa'], how='left')
            final_paths['pos'] = final_paths['newpos']
            final_paths.drop(columns=['index','add_before','add_after','ipos','newpos'], inplace=True)
            mergers['posa'] = mergers['newpos']
            mergers.drop(columns=['add_before','add_after','newpos'], inplace=True)
            # Check to which haplotype we are adding the stuff
            haps = mergers[['did','centera']].drop_duplicates()
            haps['hap'] = haps.groupby(['centera'], sort=False).cumcount() + 1
            haps['alt_id'] = np.arange(len(haps))+1 + final_paths[[f'alt_id{h}' for h in range(1,ploidy)]].max().max()
            mergers = mergers.loc[np.isin(mergers['centera'].values, np.unique(haps.loc[haps['hap'] >= ploidy, 'centera'].values)) == False, :] # Remove the merge if we have more options than possible haplotypes
            mergers = mergers.merge(haps, on=['did','centera'], how='left')
            # Prepare merging (Get all the position from centerb that we want to add to centera)
            patha = mergers[['did','centera','posa','centerb','posb']].merge(final_paths.rename(columns={'center':'centera','pos':'posa'}), on=['centera','posa'], how='left')
            patha['nalts'] = (patha[[f'alt_id{h}' for h in range(1,ploidy)]] >= 0).sum(axis=1)
            pathb = final_paths.merge(mergers[['centerb','samedir','did','hap']].drop_duplicates().rename(columns={'centerb':'center'}), on=['center'], how='inner')
            pathb = ShiftDistancesDueToDirectionChange(pathb, pathb['samedir'] == False, -1, ploidy)
            pathb.loc[pathb['samedir'] == False, 'strand'] = np.where(pathb.loc[pathb['samedir'] == False, 'strand'] == '+','-','+')
            for h in range(1,ploidy):
                pathb.loc[(pathb['samedir'] == False) & (pathb[f'alt_id{h}'] >= 0), f'alt_strand{h}'] = np.where(pathb.loc[(pathb['samedir'] == False) & (pathb[f'alt_id{h}'] >= 0), f'alt_strand{h}'] == '+','-','+')
            pathb = pathb.rename(columns={'center':'centerb','pos':'posb'}).merge(patha[['did','centera','posa','centerb','posb','distance','nalts']].rename(columns={'distance':'dista'}), on=['did','centerb','posb'], how='left')
            minposa = pathb.groupby(['centerb', 'did'], sort=False)[['posa','centera']].agg(['min','size'])
            pathb['firsta'] = np.repeat(minposa['posa','min'].values, minposa['posa','size'].values) == pathb['posa']
            pathb['centera'] = np.repeat(minposa['centera','min'].values, minposa['centera','size'].values).astype(int)
            while np.sum(np.isnan(pathb['posa'])):
                pathb.loc[pathb['samedir'] & np.isnan(pathb['posa']) & (np.isnan(pathb['posa'].shift(1)) == False) & (pathb['did'].shift(1) == pathb['did']), 'posa'] = pathb.loc[pathb['samedir'] & (np.isnan(pathb['posa']) == False) & np.isnan(pathb['posa'].shift(-1, fill_value=0)) & (pathb['did'].shift(-1) == pathb['did']), 'posa'].values+1
                pathb.loc[pathb['samedir'] & np.isnan(pathb['posa']) & pathb['firsta'].shift(-1, fill_value=False) & (pathb['did'].shift(-1) == pathb['did']), 'posa'] = pathb.loc[pathb['samedir'] & pathb['firsta'] & np.isnan(pathb['posa'].shift(1, fill_value=0)) & (pathb['did'].shift(1) == pathb['did']), 'posa'].values-1
                pathb.loc[pathb['samedir'] & pathb['firsta'].shift(-1, fill_value=False) & (pathb['did'].shift(-1) == pathb['did']), 'firsta'] = True
                pathb.loc[(pathb['samedir'] == False) & np.isnan(pathb['posa']) & (np.isnan(pathb['posa'].shift(-1)) == False) & (pathb['did'].shift(-1) == pathb['did']), 'posa'] = pathb.loc[(pathb['samedir'] == False) & (np.isnan(pathb['posa']) == False) & np.isnan(pathb['posa'].shift(1, fill_value=0)) & (pathb['did'].shift(1) == pathb['did']), 'posa'].values+1
                pathb.loc[(pathb['samedir'] == False) & np.isnan(pathb['posa']) & pathb['firsta'].shift(1, fill_value=False) & (pathb['did'].shift(1) == pathb['did']), 'posa'] = pathb.loc[(pathb['samedir'] == False) & pathb['firsta'] & np.isnan(pathb['posa'].shift(-1, fill_value=0)) & (pathb['did'].shift(-1) == pathb['did']), 'posa'].values-1
                pathb.loc[(pathb['samedir'] == False) & pathb['firsta'].shift(1, fill_value=False) & (pathb['did'].shift(1) == pathb['did']), 'firsta'] = True
            pathb['posa'] = pathb['posa'].astype(int)
            pathb = pathb.loc[((pathb[[f'alt_id{h}' for h in range(1,ploidy)]] >= 0).sum(axis=1) == 0) & (np.isnan(pathb['nalts']) | ((pathb['nalts'] == 0) & (pathb['dista'] != pathb['distance']))),:] # Positions with alternatives and positions where the main is identical we do not copy
            mergers['add_self'] = mergers.merge(pathb[['did','centera','posa','centerb','posb']], on=['did','centera','posa','centerb','posb'], how='left', indicator=True)['_merge'].values == "both"
            # Start merging
            for h in range(1,ploidy):
                # Assign new alt_id to all unique scaffolds of path a (Meaning we have a deletion if we do not fill in a scaffold later and all unique scaffolds of path b are phased)
                final_paths = final_paths.merge(mergers.loc[mergers['hap'] == h, ['centera','alt_id']].drop_duplicates().rename(columns={'centera':'center'}), on=['center'], how='left')
                final_paths.loc[final_paths[['center','pos']].merge(mergers.loc[(mergers['hap'] == h) & (mergers['add_self'] == False), ['centera','posa']].rename(columns={'centera':'center','posa':'pos'}), on=['center','pos'], how='left', indicator=True)['_merge'].values == "both", 'alt_id'] = np.nan # Do not assign new alt_id for duplications (except the ones we add due to distance variants)
                final_paths.loc[np.isnan(final_paths['alt_id']) == False, f'alt_id{h}'] = final_paths.loc[np.isnan(final_paths['alt_id']) == False, 'alt_id'].astype(int)
                final_paths.drop(columns=['alt_id'], inplace=True)
                # Add unique scaffolds from path b to path a
                final_paths = final_paths.merge(pathb[['centera','posa','scaffold','strand','distance']].rename(columns={'centera':'center','posa':'pos','scaffold':'nscaf','strand':'nstrand','distance':'ndist'}), on=['center','pos'], how='left')
                final_paths.loc[np.isnan(final_paths['nscaf']) == False, [f'alt_scaf{h}',f'alt_dist{h}']] = final_paths.loc[np.isnan(final_paths['nscaf']) == False, ['nscaf','ndist']].values.astype(int)
                final_paths.loc[final_paths['nstrand'].isnull() == False, f'alt_strand{h}'] = final_paths.loc[final_paths['nstrand'].isnull() == False, 'nstrand'].values
                final_paths.drop(columns=['nscaf','nstrand','ndist'], inplace=True)
            # Remove the paths (b) we merged
            rem_paths.append(np.unique(mergers['centerb'].values))

    # Delete duplicated paths
    final_paths = final_paths[np.isin(final_paths['center'], np.concatenate(rem_paths)) == False].copy()
    
    # Split off duplicated ends if necessary (no connections spanning the duplication for both path)
    final_paths['pid'] = (final_paths['center'] != final_paths['center'].shift(1)).cumsum()
    if len(duplications):
        req = ((duplications['posb'] == 0) | (duplications['posb'] == duplications['lenb']-1)) & duplications['fulla'] & duplications['fullb']
        duplications['start'] = (duplications['posa'] == 0) & req
        duplications['end'] = (duplications['posa'] == duplications['lena']-1) & req
        for side in ['start','end']:
            trim = duplications.groupby(['did'])[side].agg(['max','size'])
            trim = duplications[np.repeat(trim['max'].values,trim['size'].values)].copy()
            if len(trim):
                # Check if a connection in scaffold graph spans the duplications
                start = trim[trim[side]].drop(columns=['fulla','fullb','start','end'])
                shift = 1 if side == 'start' else -1
                for p in ['a','b']:
                    if p == 'b':
                        shift *= np.where(start['samedir'], 1, -1)
                    start[f'pos{p}'] += shift
                    while True:
                        duplicated = start.merge(duplications[[f'center{p}',f'pos{p}']], on=[f'center{p}',f'pos{p}'], how='left', indicator=True)['_merge'].values == "both"
                        if np.sum(duplicated):
                            start.loc[duplicated, f'pos{p}'] += shift if p=='a' else shift[duplicated]
                        else:
                            break
                start['dira'] = -1 if side == 'start' else 1
                start['dirb'] = start['dira'] * np.where(start['samedir'], 1, -1)
                start.drop(columns=['samedir'], inplace=True)
                extensions = pd.concat([start[['did',f'center{p}',f'pos{p}',f'dir{p}',f'len{p}']].rename(columns={f'center{p}':'center',f'pos{p}':'pos',f'dir{p}':'dir',f'len{p}':'len'}) for p in ['a','b']], ignore_index=True)
                extensions['dir'] *= -1 # We want to skip deletions in the opposite direction than the one we later use to search an extension of the paths
                extensions = MergeNextPos(extensions, final_paths, ploidy)
                extensions['dir'] *= -1
                extensions['side'] = np.where( (extensions['strand'] == '+') == (extensions['dir'] == 1), 'r', 'l')
                extensions = extensions.merge(scaffold_graph.rename(columns={'from':'scaffold','from_side':'side'}), on=['scaffold','side'], how='inner')
                extensions = extensions.loc[(extensions['dir'] == 1) | (extensions['dist1'] == extensions['distance']), :]
                extendible = []
                if len(extensions):
                    extensions.drop(columns=['scaffold','strand','distance','side'], inplace=True)
                    extensions = RemoveEmptyColumns(extensions)
                    extensions['pos'] += extensions['dir']
                    s = 1 # current scaffold
                    while len(extensions):
                        extensions = MergeNextPos(extensions, final_paths, ploidy)
                        extensions = extensions.loc[(extensions['length'] > s+1) & (extensions[f'scaf{s}'] == extensions['scaffold']) & ((extensions['dir'] == 1) == (extensions[f'strand{s}'] == extensions['strand'])), :]
                        if len(extensions):
                            extensions = extensions.loc[((extensions['dir'] == 1) & (extensions[f'dist{s}'] == extensions['distance'])) | ((extensions['dir'] == -1) & ((extensions[f'dist{s+1}'] == extensions['distance']) | (extensions['pos'] <= 0))), :]
                            extensions.drop(columns=['scaffold','strand','distance',f'scaf{s}',f'strand{s}',f'dist{s}'], inplace=True)
                            extensions['pos'] += extensions['dir']
                            extendible.append(np.unique(extensions.loc[(extensions['pos'] < 0) | (extensions['pos'] >= extensions['len']), 'did'].values))
                            extensions = extensions.loc[(extensions['pos'] >= 0) & (extensions['pos'] < extensions['len']), :]
                        s += 1
                # Split off the parts that do not have a connection that spans the duplications
                if len(extendible):
                    extendible = np.unique(np.concatenate(extendible))
                start = start.loc[np.isin(start['did'], extendible) == False, :]
                if len(trim):
                    # Check if we have a not fully duplicated positions before start to reduce the trimmed length (since it is a duplicate and thus in both path, it is enough to handle path a)
                    start = start.merge(trim.loc[(trim['fulla'] == False) | (trim['fullb'] == False), ['did','posa','posb']].rename(columns={'posa':'partiala', 'posb':'partialb'}), on=['did'], how='left')
                    partial_before = ((start['dira'] == -1) & (start['partiala'] < start['posa'])) | ((start['dira'] == 1) & (start['partiala'] > start['posa']))
                    if np.sum(partial_before):
                        start.loc[partial_before, ['posa','posb']] = start.loc[partial_before, ['partiala','partialb']].astype(int)
                        tmp = start.groupby(['did'])['posa'].agg(['min','max','size'])
                        start = start.loc[np.where(start['dira'] == 1, np.repeat(tmp['max'].values, tmp['size'].values), np.repeat(tmp['min'].values, tmp['size'].values)) == start['posa'], :]
                    start.drop(columns=['partiala','partialb'], inplace=True)
                    # Split path a
                    start.loc[start['dira'] == 1, 'posa'] += 1 # posa marks the start of the new path, thus add one if duplication is at the end
                    final_paths['split'] = final_paths.merge(start[['centera','posa']].rename(columns={'centera':'center','posa':'pos'}), on=['center','pos'], how='left', indicator=True)['_merge'].values == "both"
                    final_paths['pid'] = (final_paths['split'] | (final_paths['pid'] != final_paths['pid'].shift(1))).cumsum()
                    final_paths.drop(columns=['split'], inplace=True)
                    start.drop(columns=['did','centera','posa','lena','dira'], inplace=True)
                    # Remove duplicated part of path b
                    start['posb'] += start['dirb']
                    start.reset_index(inplace=True)
                    start = start.loc[np.repeat(start.index.values, np.where(start['dirb'] == 1, start['lenb'].values-start['posb'], start['posb']+1))]
                    start['posb'] += start.groupby(['index']).cumcount()*start['dirb']
                    final_paths['remove'] = final_paths.merge(start[['centerb','posb']].rename(columns={'centerb':'center','posb':'pos'}), on=['center','pos'], how='left', indicator=True)['_merge'].values == "both"
                    final_paths = final_paths.loc[final_paths['remove'] == False, :]
                    final_paths.drop(columns=['remove'], inplace=True)
    # Remove deletions on all haplotypes and reassign positions
    final_paths = final_paths.loc[final_paths[['scaffold']+[f'alt_scaf{h}' for h in range(1,ploidy)]].max(axis=1) >= 0, :]
    final_paths['pos'] = final_paths.groupby(['pid']).cumcount()
    # Set distances at pos 0 to zero (Migth be different due to split paths)
    final_paths.loc[final_paths['pos'] == 0, ['distance']+[f'alt_dist{h}' for h in range(1,ploidy)]] = 0
    final_paths = TrimAlternativesConsistentWithMainOld(final_paths, ploidy)

    return final_paths

def InsertUniquelyPlaceableBubbles(final_paths, bubbles, ploidy):
    if len(bubbles):
        # Get end of bubble in to column again
        ibubbles = bubbles.copy()
        ibubbles[['to','to_strand']] = [-1,'']
        for s in range(ibubbles['length'].min(), ibubbles['length'].max()+1):
            ibubbles.loc[ibubbles['length'] == s, 'to'] = ibubbles.loc[ibubbles['length'] == s, f'scaf{s}'].values.astype(int)
            ibubbles.loc[ibubbles['length'] == s, 'to_strand'] = ibubbles.loc[ibubbles['length'] == s, f'strand{s}'].values
        ibubbles = RemoveEmptyColumns(ibubbles)
        # Check on which path the two ends of the bubble fall
        tmp_path = pd.concat([final_paths.loc[final_paths['scaffold'] >= 0, ['pid','pos','scaffold','strand']]]+[final_paths.loc[final_paths[f'alt_scaf{h}'] >= 0, ['pid','pos',f'alt_scaf{h}',f'alt_strand{h}']].rename(columns={f'alt_scaf{h}':'scaffold',f'alt_strand{h}':'strand'}) for h in range(1,ploidy)], ignore_index=True).drop_duplicates()
        ibubbles = ibubbles.merge(tmp_path.rename(columns={'pid':'from_pid','pos':'from_pos','scaffold':'from','strand':'from_strand'}), on=['from','from_strand'], how='inner')
        ibubbles = ibubbles.merge(tmp_path.rename(columns={'pid':'to_pid','pos':'to_pos','scaffold':'to','strand':'to_strand'}), on=['to','to_strand'], how='inner')
        ibubbles = ibubbles[(ibubbles['from_pid'] == ibubbles['to_pid']) & (ibubbles['from_pos'] < ibubbles['to_pos'])].rename(columns={'from_pid':'pid'}).drop(columns=['to_pid'])
        ## Only keep uniquely placeable within ploidy range
        ibubbles.sort_values(['from','from_strand'], inplace=True)
        ibubbles['bid'] = ((ibubbles['from'] != ibubbles['from'].shift(1)) | (ibubbles['from_strand'] != ibubbles['from_strand'].shift(1))).cumsum()
        # If the reversed bubble is also placed, it cannot be unique
        revbubble = ibubbles[['to','to_strand','bid']].rename(columns={'to':'from','to_strand':'from_strand','bid':'rid'}).drop_duplicates()
        revbubble['from_strand']= np.where(revbubble['from_strand'] == '+', '-', '+')
        ibubbles = ibubbles.merge(revbubble, on=['from','from_strand'], how='left')
        ibubbles = ibubbles[np.isnan(ibubbles['rid'])].copy()
        ibubbles.drop(columns=['to','to_strand','rid'], inplace=True)
        # Count possible positions options
        poscount = ibubbles[['bid','pid','from_pos','to_pos']].groupby(['bid','pid','from_pos','to_pos'], sort=False).size()
        poscount = poscount[(poscount > 1) & (poscount <= ploidy)].reset_index(name='size').drop(columns=['size']) # Filter on ploidy of individual bubbles (additionally filter out unique connections, because they must have been already included in final_paths)
        poscount = poscount.groupby(['bid'], sort=False).size()
        poscount = poscount[poscount == 1].reset_index(name='size').drop(columns=['size']) # Keep only uniquely placeable bubbles
        ibubbles = ibubbles.merge(poscount, on=['bid'], how='inner')
        # Remove bubble paths that are already in final_paths
        for h in range(0, ploidy):
            if h==0:
                tmp_path = final_paths[['pid','pos','scaffold','strand','distance']]
            else:
                tmp_path = final_paths[['pid','pos',f'alt_scaf{h}',f'alt_strand{h}',f'alt_dist{h}']].rename(columns={f'alt_scaf{h}':'scaffold',f'alt_strand{h}':'strand',f'alt_dist{h}':'distance'})
                tmp_path.loc[final_paths[f'alt_id{h}'] < 0, ['scaffold','strand','distance']] = final_paths.loc[final_paths[f'alt_id{h}'] < 0, ['scaffold','strand','distance']].values
            ibubbles['pos'] = ibubbles['from_pos']
            ibubbles['identical'] = True
            s = 1 # current scaffold
            while np.sum(ibubbles['identical']):
                if s > ibubbles['length'].max():
                    ibubbles['identical'] = False
                else:
                    ibubbles['pos'] += 1
                    while True:
                        ibubbles = ibubbles.merge(tmp_path, on=['pid','pos'], how='left')
                        if np.sum(ibubbles['scaffold'] < 0):
                            ibubbles.loc[ibubbles['scaffold'] < 0, 'pos'] += 1
                            ibubbles.drop(columns=['scaffold','strand','distance'], inplace=True)
                        else:
                            break
                    ibubbles.loc[(ibubbles['scaffold'] != ibubbles[f'scaf{s}']) | (ibubbles['strand'] != ibubbles[f'strand{s}']) | (ibubbles['distance'] != ibubbles[f'dist{s}']), 'identical'] = False
                    ibubbles = ibubbles[(ibubbles['identical'] == False) | (ibubbles['to_pos'] > ibubbles['pos'])].drop(columns=['scaffold','strand','distance'])
                    s += 1
            ibubbles.drop(columns=['identical','pos'], inplace=True)
        
    return final_paths

def TraverseScaffoldingGraph(scaffolds, scaffold_parts, scaffold_graph, contig_parts, scaf_bridges, ploidy):
    bubbles, loops, unique_extensions = FindBubbles(scaffold_graph, ploidy)
    #unique_paths, bubbles, unique_extensions = ResolveLoops(loops, unique_paths, bubbles, unique_extensions)
    # Do here: Handle unique extensions
    knots = GroupConnectedScaffoldsIntoKnots(bubbles, scaffolds, scaffold_parts, contig_parts)
#
    # Separate some stuff from bubbles
    inversions = bubbles[bubbles['from'] == bubbles['to']].copy()
    bubbles = bubbles[bubbles['from'] != bubbles['to']].copy()
    repeat_bubbles = bubbles[bubbles['alts'] > ploidy].copy()
    bubbles = bubbles[bubbles['alts'] <= ploidy].copy()
#
    bubbles, phasing_connections = UnreelBubbles(bubbles)
    final_paths, final_alternatives = CombineBubblesIntoPaths(bubbles, phasing_connections, knots, scaf_bridges, ploidy)
    final_paths, final_alternatives = IntegrateAlternativesIntoPaths(final_paths, final_alternatives, ploidy)
#
    final_paths = HandlePathDuplications(final_paths, knots, scaffold_graph, ploidy)
    #final_paths = InsertUniquelyPlaceableBubbles(final_paths, bubbles, ploidy)

    # Extend scaffolds according to final_paths
    final_paths.rename(columns={'scaffold':'scaf0','strand':'strand0','distance':'dist0'}, inplace=True)
    final_paths['phase0'] = 0
    for h in range(1,ploidy):
        final_paths.rename(columns={f'alt_id{h}':f'phase{h}',f'alt_scaf{h}':f'scaf{h}',f'alt_strand{h}':f'strand{h}',f'alt_dist{h}':f'dist{h}'}, inplace=True)
    for h in range(0,ploidy):
        final_paths = final_paths.merge(scaffolds[['scaffold','size']].rename(columns={'scaffold':f'scaf{h}','size':f'size{h}'}), on=[f'scaf{h}'], how='left')
        final_paths[f'size{h}'] = final_paths[f'size{h}'].fillna(0).astype(int)
        
    scaffold_paths = final_paths.drop(columns=['center']).loc[np.repeat(final_paths.index.values, final_paths[[f'size{h}' for h in range(ploidy)]].max(axis=1).values)]
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
    
    next_phase = scaffold_paths[[f'phase{h}' for h in range(ploidy)]].max().max()+1
    for h in range(1,ploidy):
        scaffold_paths.loc[scaffold_paths[f'phase{h}'] == 0, f'phase{h}'] = next_phase
        next_phase += 1
        new_phases = np.sum(scaffold_paths[f'phase{h}'] < 0)
        scaffold_paths.loc[scaffold_paths[f'phase{h}'] < 0, f'phase{h}'] = -(np.arange(new_phases)+next_phase)
        next_phase += new_phases
    
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

def TrimAlternativesConsistentWithMain(scaffold_paths, ploidy):
    for h in range(1, ploidy):
        consistent = (scaffold_paths['alt_scaf0'] == scaffold_paths[f'alt_scaf{h}']) & (scaffold_paths['alt_strand0'] == scaffold_paths[f'alt_strand{h}']) & (scaffold_paths['alt_dist0'] == scaffold_paths[f'alt_dist{h}'])
        scaffold_paths.loc[consistent, [f'alt_scaf{h}', f'alt_dist{h}', f'alt_strand{h}']] = [-1,0,'']
        scaffold_paths.loc[consistent, f'phase{h}'] = -scaffold_paths.loc[consistent, f'phase{h}']
#
    return scaffold_paths

def ReverseScaffolds(scaffold_paths, reverse, ploidy):
    # Reverse Strand
    for h in range(ploidy):
        sreverse = reverse & (scaffold_paths[f'alt_strand{h}'] != '')
        scaffold_paths.loc[sreverse, f'alt_strand{h}'] = np.where(scaffold_paths.loc[sreverse, f'alt_strand{h}'] == '+', '-', '+')
    # Reverse distance and phase
    for h in range(ploidy-1,-1,-1):
        # Shift distances in alternatives
        missing_information = reverse & (scaffold_paths['pid'] == scaffold_paths['pid'].shift(-1)) & (scaffold_paths[f'phase{h}'] < 0) & (scaffold_paths[f'phase{h}'].shift(-1) >= 0)
        if np.sum(missing_information):
            scaffold_paths.loc[missing_information, [f'alt_scaf{h}', f'alt_strand{h}', f'alt_dist{h}']] = scaffold_paths.loc[missing_information, ['alt_scaf0','alt_strand0','alt_dist0']].values
            scaffold_paths.loc[missing_information, [f'phase{h}']] = -scaffold_paths.loc[missing_information, [f'phase{h}']]
        scaffold_paths.loc[reverse & (scaffold_paths[f'phase{h}'] < 0), f'alt_dist{h}'] = scaffold_paths.loc[reverse & (scaffold_paths[f'phase{h}'] < 0), 'alt_dist0']
        scaffold_paths[f'alt_dist{h}'] = np.where(reverse, scaffold_paths[f'alt_dist{h}'].shift(-1, fill_value=0), scaffold_paths[f'alt_dist{h}'])
        scaffold_paths.loc[reverse & (scaffold_paths['pid'] != scaffold_paths['pid'].shift(-1)) | (scaffold_paths[f'phase{h}'] < 0), f'alt_dist{h}'] = 0
        while True:
            scaffold_paths['jumped'] = (scaffold_paths[f'phase{h}'] >= 0) & (scaffold_paths[f'alt_scaf{h}'] < 0) & (scaffold_paths[f'alt_dist{h}'] != 0) # Jump deletions
            if 0 == np.sum(scaffold_paths['jumped']):
                break
            else:
                scaffold_paths.loc[scaffold_paths['jumped'].shift(-1, fill_value=False), f'alt_dist{h}'] = scaffold_paths.loc[scaffold_paths['jumped'], f'alt_dist{h}'].values
                scaffold_paths.loc[scaffold_paths['jumped'], f'alt_dist{h}'] = 0
        # We also need to shift the phase, because the distance variation at the end of the bubble is on the other side now
        scaffold_paths[f'phase{h}'] = np.where(reverse & (scaffold_paths['pid'] == scaffold_paths['pid'].shift(-1)), np.sign(scaffold_paths[f'phase{h}'])*np.abs(scaffold_paths[f'phase{h}'].shift(-1, fill_value=0)), scaffold_paths[f'phase{h}'])
    scaffold_paths.drop(columns=['jumped'], inplace=True)
    TrimAlternativesConsistentWithMain(scaffold_paths, ploidy)
    # Reverse ordering
    scaffold_paths.loc[reverse, 'pos'] *= -1
    scaffold_paths.sort_values(['pid','pos'], inplace=True)
    scaffold_paths['pos'] = scaffold_paths.groupby(['pid'], sort=False).cumcount()

    return scaffold_paths

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
    # Set new continous scaffold ids consistent with original scaffolding and apply reversions
    meta_parts.sort_values(['meta','pos'], inplace=True)
    meta_parts['new_scaf'] = range(len(meta_parts))
    scaffold_paths = scaffold_paths.merge(meta_parts[['scaffold','reverse','new_scaf']].rename(columns={'scaffold':'scaf'}), on=['scaf'], how='left')
    scaffold_paths['scaf'] = scaffold_paths['new_scaf']
    col_rename = {**{'scaf':'pid'}, **{f'{n1}{h}':f'{n2}{h}' for h in range(ploidy) for n1,n2 in [('con','alt_scaf'),('strand','alt_strand'),('dist','alt_dist')]}}
    scaffold_paths = ReverseScaffolds(scaffold_paths.rename(columns=col_rename), scaffold_paths['reverse'], ploidy).rename(columns={v: k for k, v in col_rename.items()})
    scaffold_paths.drop(columns=['reverse','new_scaf'], inplace=True)

    # Set org_dist_left/right based on scaffold (not the contig anymore)
    org_cons = GetOriginalConnections(scaffold_paths, contig_parts, ploidy)
    org_cons = org_cons.loc[(org_cons['from_side'] == 'l') & (org_cons['from']-1 == org_cons['to']), :]
    scaffold_paths['sdist_left'] = -1
    scaffold_paths.loc[np.isin(scaffold_paths['scaf'], org_cons['from'].values) & (scaffold_paths['pos'] == 0), 'sdist_left'] = org_cons['distance'].values
    scaffold_paths['sdist_right'] = scaffold_paths['sdist_left'].shift(-1, fill_value=-1)

    return scaffold_paths

def ScaffoldContigs(contig_parts, bridges, mappings, ploidy):
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
#
    # Build scaffold graph to find unique bridges over scaffolds with alternative connections
    long_range_connections = GetLongRangeConnections(bridges, mappings)
    long_range_connections = TransformContigConnectionsToScaffoldConnections(long_range_connections, scaffold_parts)
    scaffold_graph = BuildScaffoldGraph(long_range_connections, scaf_bridges)
    scaffold_paths = TraverseScaffoldingGraph(scaffolds, scaffold_parts, scaffold_graph, contig_parts, scaf_bridges, ploidy)

    # Finish Scaffolding
    scaffold_paths = OrderByUnbrokenOriginalScaffolds(scaffold_paths, contig_parts, ploidy)

    return scaffold_paths

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
    # Check that the read connects to the right positions in the scaffold and assign groups to separate parts of a read mapping to two separate (maybe overlapping) locations on the scaffold
    mappings['group'] = ( (mappings['read_name'] != mappings['read_name'].shift(1)) | (mappings['read_start'] != mappings['read_start'].shift(1)) | (mappings['read_pos'] != mappings['read_pos'].shift(1)+1) | # We do not follow the read
                          (mappings['scaf'] != mappings['scaf'].shift(1)) | (np.where(mappings['strand'] == '-', mappings['rpos'], mappings['lpos']) != mappings['pos'].shift(1)) | # We do not follow the scaffold
                          np.where(mappings['strand'] == '-', mappings['rhap'] != mappings['lhap'].shift(1), mappings['lhap'] != mappings['rhap'].shift(1)) | # We do not follow the haplotype
                          ((mappings['read_name'] == mappings['read_name'].shift(2)) & (mappings['read_pos'] == mappings['read_pos'].shift(2)+1)) | 
                          ((mappings['read_name'] == mappings['read_name'].shift(-1)) & (mappings['read_pos'] == mappings['read_pos'].shift(-1))) ).cumsum() # We have more than one option (and only one of them can be valid, so do not allow it to group with one, because it might be the wrong one)
    for d, d2 in zip(['l','r'],['r','l']):
        mappings['check_pos'] = mappings['read_pos'] + np.where(mappings['strand'] == '+', -1, 1) * (1 if d == 'l' else -1)
        mappings = mappings.merge(mappings[['read_name','read_start','read_pos','scaf','pos',f'{d2}hap','group']].drop_duplicates().rename(columns={'read_pos':'check_pos','pos':f'{d}pos',f'{d2}hap':f'{d}hap','group':f'{d}group'}), on=['read_name','read_start','check_pos','scaf',f'{d}pos',f'{d}hap'], how='left')
        mappings[f'{d}group'] = mappings[f'{d}group'].fillna(-1).astype(int)
        nocon = ((mappings[f'{d}con'] < 0) | (mappings[f'{d}pos'] < 0)) & (mappings[f'{d}group'] < 0)
        mappings.loc[nocon, f'{d}group'] = mappings.loc[nocon, 'group']
    groups = mappings.loc[(mappings['group'] != mappings['lgroup']) | (mappings['group'] != mappings['rgroup']), ['group','lgroup','rgroup']].drop_duplicates()
    mappings.drop(columns=['check_pos','lgroup','rgroup'], inplace=True)
    mappings.drop_duplicates(inplace=True)
    groups['new_group'] = groups.min(axis=1)
    while True:
        groups['check'] = groups['new_group']
        groups = groups.merge(groups[['group','new_group']].rename(columns={'group':'lgroup','new_group':'lngroup'}), on=['lgroup'], how='left')
        groups = groups.merge(groups[['group','new_group']].rename(columns={'group':'rgroup','new_group':'rngroup'}), on=['rgroup'], how='left')
        groups['new_group'] = groups[['new_group','lngroup','rngroup']].fillna(-1).min(axis=1).astype(int)
        groups.drop(columns=['lngroup','rngroup'], inplace=True)
        groups.drop_duplicates(inplace=True)
        groups = groups.groupby(['group','lgroup','rgroup','check'])['new_group'].min().reset_index()
        tmp = groups.groupby(['group'], sort=False)['new_group'].agg(['max','size']) # Take the maximum here, to not delete both path extensions if one is valid
        groups['new_group'] = np.repeat(tmp['max'].values, tmp['size'].values)
        if np.sum(groups['new_group'] != groups['check']) == 0:
            groups = groups.groupby(['group'])['new_group'].max().reset_index()
            break
    groups = groups.loc[groups['group'] != groups['new_group'], :]
    mappings = mappings.merge(groups, on=['group'], how='left')
    mappings.loc[np.isnan(mappings['new_group']) == False, 'group'] = mappings.loc[np.isnan(mappings['new_group']) == False, 'new_group'].astype(int)
    mappings.drop(columns=['new_group'], inplace=True)
    mappings = mappings.loc[(mappings['group'] >= 0), :]
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
        all_scafs[['ldmin','rdmin']] = all_scafs[['ldmin','rdmin']].fillna(-sys.maxsize*0.99).values.astype(int) # Use *0.99 to avoid overflow through type convertion from float to int
        all_scafs[['ldmax','rdmax']] = all_scafs[['ldmax','rdmax']].fillna(sys.maxsize*0.99).values.astype(int)
        all_scafs.drop(columns=['from_side','to_side'], inplace=True)
#
        mappings = BasicMappingToScaffolds(mappings, all_scafs)
#
        # Get coverage for connections
        conn_cov = mappings.loc[(mappings['lcon'] >= 0) & (mappings['lpos'] >= 0), ['scaf','pos','lpos','lhap']].rename(columns={'pos':'rpos','lhap':'hap'}).groupby(['scaf','lpos','rpos','hap']).size().reset_index(name='cov')
        conn_cov = all_scafs.loc[(all_scafs['lpos'] >= 0), ['scaf','lpos','pos','lhap']].rename(columns={'pos':'rpos','lhap':'hap'}).sort_values(['scaf','lpos','rpos','hap']).merge(conn_cov, on=['scaf','lpos','rpos','hap'], how='left')
        conn_cov['cov'] = conn_cov['cov'].fillna(0).astype(int)
#
        # Remove reads, where they map to multiple locations (keep them separately, so that we can restore them if it does leave a connection between contigs without reads)
        dups_maps = mappings[['read_name','read_start','read_pos','scaf','pos']].drop_duplicates().groupby(['read_name','read_start','read_pos'], sort=False).size().reset_index(name='count')
        mappings = mappings.merge(dups_maps, on=['read_name','read_start','read_pos'], how='left')
        dups_maps = mappings[['group','count']].groupby(['group'])['count'].min().reset_index(name='gcount')
        mappings = mappings.drop(columns=['count']).merge(dups_maps, on=['group'], how='left')
        dups_maps = mappings[mappings['gcount'] > 1].copy()
        mappings = mappings[mappings['gcount'] == 1].drop(columns=['gcount','group'])
        conn_cov = conn_cov.merge(mappings.loc[(mappings['lcon'] >= 0) & (mappings['lpos'] >= 0), ['scaf','pos','lpos','lhap']].rename(columns={'pos':'rpos','lhap':'hap'}).groupby(['scaf','lpos','rpos','hap']).size().reset_index(name='ucov'), on=['scaf','lpos','rpos','hap'], how='left')
        conn_cov['ucov'] = conn_cov['ucov'].fillna(0).astype(int)
#
        # Try to fix scaffold_paths, where no reads support the connection
        # Start by getting mappings that have both sides in them
        unsupp_conns = conn_cov[conn_cov['ucov'] == 0].copy()
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
        dups_maps = dups_maps.merge(dcons[['scaf','rpos','hap','ci']].rename(columns={'rpos':'pos','hap':'lhap','ci':'lci'}), on=['scaf','pos','lhap'], how='left')
        dups_maps = dups_maps.merge(dcons[['scaf','lpos','hap','ci']].rename(columns={'lpos':'pos','hap':'rhap','ci':'rci'}), on=['scaf','pos','rhap'], how='left')
        dups_maps = dups_maps.loc[((dups_maps['lcon'] >= 0) & (np.isnan(dups_maps['lci']) == False)) | ((dups_maps['rcon'] >= 0) & (np.isnan(dups_maps['rci']) == False)), :]
        dups_maps[['lci','rci']] = dups_maps[['lci','rci']].fillna(-1).astype(int)
        dups_maps.loc[dups_maps['lcon'] < 0, 'lci'] = -1
        dups_maps.loc[dups_maps['rcon'] < 0, 'rci'] = -1
        # Choose the reads with the lowest amount of duplication for the connections
        dups_maps.sort_values(['lci'], inplace=True)
        mcount = dups_maps.groupby(['lci'], sort=False)['gcount'].agg(['min','size'])
        dups_maps.loc[dups_maps['gcount'] > np.repeat(mcount['min'].values, mcount['size'].values), 'lci'] = -1
        dups_maps.sort_values(['rci'], inplace=True)
        mcount = dups_maps.groupby(['rci'], sort=False)['gcount'].agg(['min','size'])
        dups_maps.loc[dups_maps['gcount'] > np.repeat(mcount['min'].values, mcount['size'].values), 'rci'] = -1
        dups_maps.drop(columns=['group','gcount'], inplace=True)
        # Duplicate entries that have a left and a right connection that can only be filled with a duplicated read
        dups_maps.sort_values(['read_name','read_start','read_pos','scaf','pos','lhap','rhap'], inplace=True)
        dups_maps.reset_index(inplace=True)
        dups_maps = dups_maps.loc[np.repeat(dups_maps.index.values, (dups_maps['lci'] >= 0).values.astype(int) + (dups_maps['rci'] >= 0).values.astype(int))]
        dups_maps.loc[dups_maps['index'] == dups_maps['index'].shift(1), 'lci'] = -1
        dups_maps.loc[dups_maps['index'] == dups_maps['index'].shift(-1), 'rci'] = -1
        dups_maps.drop(columns=['index'], inplace=True)
        # Remove the unused connections and trim the length of the duplicated reads, such that we do not use them for other connections or extending
        dups_maps.loc[dups_maps['lci'] < 0, ['lcon','ldist','lmapq','lmatches']] = [-1,0,-1,-1]
        dups_maps.loc[dups_maps['rci'] < 0, ['rcon','rdist','rmapq','rmatches']] = [-1,0,-1,-1]
        dups_maps['ci'] = dups_maps[['lci','rci']].max(axis=1)
        dups_maps.drop(columns=['lci','rci'], inplace=True)
        dups_maps.sort_values(['read_name','read_start','ci'], inplace=True)
        tmp = dups_maps.groupby(['read_name','read_start','ci'], sort=False).agg({'read_from':['min'], 'read_to':['max']})
        dups_maps['read_start'] = np.repeat( tmp['read_from','min'].values, 2 )
        dups_maps['read_end'] = np.repeat( tmp['read_to','max'].values, 2 )
        dups_maps.sort_values(['read_name','read_start','read_pos'], inplace=True)
        dups_maps['read_pos'] = dups_maps.groupby(['read_name','read_start'], sort=False).cumcount()
        dups_maps.drop(columns=['ci'], inplace=True)
        mappings = pd.concat([mappings, dups_maps]).sort_values(['read_name','read_start','read_pos'], ignore_index=True)
#
        # Break connections where they are not supported by reads even with multi mapping reads and after fixing attemps(should never happen, so give a warning)
        if len(conn_cov[conn_cov['cov'] == 0]) == 0:
            break
        else:
            print( "Warning:", len(conn_cov[conn_cov['cov'] == 0]), "gaps were created for which no read for filling can be found. The connections will be broken up again, but this should never happen and something weird is going on.")
            # Start with alternative paths
            for h in range(1,ploidy):
                rem = conn_cov.loc[(conn_cov['cov'] == 0) & (conn_cov['hap'] == h), ['scaf','rpos']].rename(columns={'rpos':'pos'}).merge(scaffold_paths[['scaf','pos',f'phase{h}']], on=['scaf','pos'], how='inner')[f'phase{h}'].values
                rem = np.isin(scaffold_paths[f'phase{h}'], rem)
                scaffold_paths.loc[rem, [f'con{h}',f'strand{h}',f'dist{h}']] = [-1,'',0]
                scaffold_paths.loc[rem, f'phase{h}'] = -scaffold_paths.loc[rem, f'phase{h}']
            # Continue with main paths that has alternatives
            rem_conns = conn_cov[(conn_cov['cov'] == 0) & (conn_cov['hap'] == 0)].copy()
            for h in range(1,ploidy):
                #!!!! needs update when main phase is not zero anymore: switch phases for main and alternative
                rem = rem_conns[['scaf','rpos']].reset_index().rename(columns={'rpos':'pos'}).merge(scaffold_paths.loc[scaffold_paths[f'phase{h}'] >= 0, ['scaf','pos',f'phase{h}']], on=['scaf','pos'], how='inner')
                rem_conns.drop(rem['index'].values, inplace=True)
                rem = np.isin(scaffold_paths[f'phase{h}'], rem[f'phase{h}'].values)
                scaffold_paths.loc[rem, ['con0','strand0','dist0']] = scaffold_paths.loc[rem, [f'con{h}',f'strand{h}',f'dist{h}']].values
                scaffold_paths.loc[rem, [f'con{h}',f'strand{h}',f'dist{h}']] = [-1,'',0]
                scaffold_paths.loc[rem, f'phase{h}'] = -scaffold_paths.loc[rem, f'phase{h}']
            # Finally split contig, where we do not have any valid connection
            scaffold_paths['split'] = scaffold_paths[['scaf','pos']].merge(rem_conns[['scaf','rpos']].rename(columns={'rpos':'pos'}), on=['scaf','pos'], how='left', indicator=True)['_merge'].values == "both"
            scaffold_paths['scaf'] = ((scaffold_paths['scaf'] != scaffold_paths['scaf'].shift(1)) | scaffold_paths['split']).cumsum()-1
            scaffold_paths['pos'] = scaffold_paths.groupby(['scaf'], sort=False).cumcount()
            scaffold_paths.drop(columns=['split'], inplace=True)
            mappings = org_mappings.copy()
#
    ## Handle circular scaffolds
    mappings = mappings.merge(scaffold_paths.loc[scaffold_paths['pos'] == 0, ['scaf']+[f'con{h}' for h in range(ploidy)]+[f'strand{h}' for h in range(ploidy)]], on=['scaf'], how='left')
    mappings.loc[(mappings['rpos'] >= 0) | (mappings['rcon'] < 0), [f'con{h}' for h in range(ploidy)]] = -1
    # Check if the read continues at the beginning of the scaffold
    mappings['check_pos'] = mappings['read_pos'] + np.where(mappings['strand'] == '+', 1, -1)
    mappings.loc[ mappings[['read_name','read_start','scaf','strand','check_pos']].merge(mappings.loc[mappings['pos'] == 0, ['read_name','read_start','scaf','strand','read_pos']].rename(columns={'read_pos':'check_pos'}).drop_duplicates(), on=['read_name','read_start','scaf','strand','check_pos'], how='left', indicator=True)['_merge'].values == "left_only", [f'con{h}' for h in range(ploidy)]] = -1
    mappings.drop(columns=['check_pos'], inplace=True)
    # Check available bridges
    for h in range(ploidy):
        mappings.loc[(mappings[f'con{h}'] != mappings['rcon']), f'con{h}'] = -1
        mappings = mappings.merge(bridges[['from','from_side','to','to_side','min_dist','max_dist','mean_dist']].rename(columns={'from':'conpart','to':f'con{h}','mean_dist':f'rmdist{h}'}), on=['conpart',f'con{h}'], how='left')
        mappings.loc[mappings['from_side'].isnull() | ((mappings['from_side'] == 'r') != (mappings['con_strand'] == '+')) | ((mappings['to_side'] == 'l') != (mappings[f'strand{h}'] == '+')), f'con{h}'] = -1
        mappings.loc[(mappings['rdist'] < mappings['min_dist']) | (mappings['rdist'] > mappings['max_dist']), f'con{h}'] = -1
        mappings.drop(columns=[f'strand{h}','from_side','to_side','min_dist','max_dist'], inplace=True)
        mappings[f'rmdist{h}'] = mappings[f'rmdist{h}'].fillna(0).astype(int)
        mappings.sort_values([col for col in mappings.columns if col not in [f'con{h}',f'rmdist{h}']] + [f'con{h}',f'rmdist{h}'], inplace=True)
        mappings = mappings.groupby([col for col in mappings.columns if col not in [f'con{h}',f'rmdist{h}']], sort=False).last().reset_index()
    mappings.sort_values(['read_name','read_start','read_pos','scaf','pos','lhap','rhap'], inplace=True)
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
    mappings = mappings.merge(circular[['scaf','pos','rhap','lhap','rmdist']].rename(columns={'lhap':'hap'}), on=['scaf','pos','rhap'], how='left')
    mappings['circular'] = False
    for h in range(ploidy):
        mappings.loc[(mappings['hap'] == h) & (mappings[f'rmdist{h}'] == mappings['rmdist']) & (mappings[f'con{h}'] >= 0), 'circular'] = True
    mappings.loc[mappings['circular'], 'rpos'] = 0
    mappings.drop(columns=[f'con{h}' for h in range(ploidy)] + [f'rmdist{h}' for h in range(ploidy)] + ['hap','rmdist','circular'], inplace=True)
    mappings['check_pos'] = mappings['read_pos'] + np.where(mappings['strand'] == '+', 1, -1)
    mappings = mappings.merge(mappings.loc[mappings['rpos'] == 0, ['read_name','read_start','scaf','strand','check_pos','rpos','conpart','rcon','rdist','pos']].rename(columns={'check_pos':'read_pos','rpos':'pos','conpart':'lcon','rcon':'conpart','rdist':'ldist','pos':'circular'}), on=['read_name','read_start','read_pos','scaf','pos','strand','conpart','lcon','ldist'], how='left')
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
    possible_reads = mappings.loc[(mappings['rcon'] >= 0) & (mappings['rpos'] >= 0), ['scaf','pos','rhap','rpos','read_pos','read_name','read_start','read_from','read_to','strand','rdist','mapq','rmapq','matches','rmatches','con_from','con_to']].sort_values(['scaf','pos','rhap'])
#
    # First take the one with the highest mapping qualities on both sides
    possible_reads['cmapq'] = np.minimum(possible_reads['mapq'], possible_reads['rmapq'])*1000 + np.maximum(possible_reads['mapq'], possible_reads['rmapq'])
    tmp = possible_reads.groupby(['scaf','pos','rhap'], sort=False)['cmapq'].agg(['max','size'])
    possible_reads = possible_reads[possible_reads['cmapq'] == np.repeat(tmp['max'].values, tmp['size'].values)].copy()
    possible_reads.drop(columns=['cmapq','mapq','rmapq'], inplace=True)
#
    # First take the one the closest to the mean distance to get the highest chance of the other reads mapping to it later for the consensus
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
    possible_reads = possible_reads.merge(mappings[['read_name','read_start','read_pos','scaf','pos','lpos','con_from','con_to','lhap']].rename(columns={'pos':'rpos','lpos':'pos','con_from':'rcon_from','con_to':'rcon_to','lhap':'chap'}), on=['read_name','read_start','read_pos','scaf','pos','rpos'], how='left')
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

def StoreExtendingReadsSublists(prefix, mappings):
    if not os.path.exists(f"{prefix}_extending_reads"):
        os.makedirs(f"{prefix}_extending_reads")
    else:
        for filename in glob.glob(f"{prefix}_extending_reads/extending_reads_*.lst"):
            os.remove(filename)
    if len(mappings):
        mappings['i'] = (mappings[['scaf','side','hap']] != mappings[['scaf','side','hap']].shift(1)).any(axis=1).cumsum()-1
        # Join groups which share a mapping
        while True:
            mappings = mappings.merge(mappings[['read_name','i']].drop_duplicates().groupby(['read_name'])['i'].min().reset_index(name='imin'), on=['read_name'], how='left')
            if np.sum(mappings['i'] != mappings['imin']) == 0:
                break
            else:
                mappings = mappings.merge(mappings.groupby(['i'])['imin'].min().reset_index().rename(columns={'imin':'newi'}), on=['i'], how='left')
                mappings['i'] = mappings['newi']
                mappings.drop(columns=['imin','newi'], inplace=True)
        # Output one file per group
        for i in np.unique(mappings['i']):
            np.savetxt(f"{prefix}_extending_reads/extending_reads_{i}.lst", np.unique(mappings.loc[mappings['i'] == i, 'read_name']), fmt='%s')

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

    # Guarantee that outdir exists
    outdir = os.path.dirname(prefix)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    del outdir

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
        break_groups, spurious_break_indexes, non_informative_mappings, unconnected_breaks = FindBreakPoints(mappings, contigs, max_dist_contig_end, min_mapping_length, min_length_contig_break, max_break_point_distance, min_num_reads, min_extension, merge_block_length, org_scaffold_trust, cov_probs, prob_factor, pdf)
    mappings.drop(np.concatenate([spurious_break_indexes, non_informative_mappings]), inplace=True) # Remove not-accepted breaks from mappings and mappings that do not contain any information (mappings inside of contigs that do not overlap with breaks)
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
    scaffold_paths = ScaffoldContigs(contig_parts, bridges, mappings, ploidy)
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
    #StoreExtendingReadsSublists(prefix, mappings)

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
    
    print( str(timedelta(seconds=clock())), "Preparing data from files")
    scaffold_paths = pd.read_csv(prefix+"_scaffold_paths.csv").fillna('')
    ploidy = GetPloidyFromPaths(scaffold_paths)
    mappings = LoadExtensions(prefix, min_extension)
    extensions, hap_merger = LoadReads(all_vs_all_mapping_file, mappings, min_length_contig_break)
    
    print( str(timedelta(seconds=clock())), "Searching for extensions")
    extensions, new_scaffolds = ClusterExtension(extensions, mappings, min_num_reads, min_scaf_len)
    scaffold_paths, extension_info = ExtendScaffolds(scaffold_paths, extensions, hap_merger, new_scaffolds, mappings, min_num_reads, max_mapping_uncertainty, min_scaf_len, ploidy)

    print( str(timedelta(seconds=clock())), "Writing output")
    scaffold_paths.to_csv(prefix+"_extended_scaffold_paths.csv", index=False)
    np.savetxt(prefix+'_used_reads.lst', np.unique(pd.concat([scaffold_paths.loc[('read' == scaffold_paths['type']) & ('' != scaffold_paths[f'name{h}']), f'name{h}'] for h in range(ploidy)], ignore_index=True)), fmt='%s')
    
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
    print( str(timedelta(seconds=clock())), "Loading scaffold info from: {}".format(scaffold_file))
    scaffold_paths = pd.read_csv(scaffold_file).fillna('')
    ploidy = GetPloidyFromPaths(scaffold_paths)
    for i in range(len(haplotypes)):
        if haplotypes[i] == False:
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

def MiniGapTest():
    # contigs, contig_ids = ReadContigs(assembly_file)
    # contigs, center_repeats = MaskRepeatEnds(contigs, repeat_file, contig_ids, max_repeat_extension, min_len_repeat_connection, repeat_len_factor_unique, remove_duplicated_contigs, pdf)
    
    pass
    
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
        haplotypes = [False]*num_slots
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
                if haplotypes[i] != False:
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
