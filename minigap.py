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
        break_groups['num'] = break_groups.groupby(['contig_id'], sort=False).cumcount()
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
#
    # Then add scaf_bridges with alternatives
    short_bridges = scaf_bridges.loc[(scaf_bridges['from_alt'] > 1) | (scaf_bridges['to_alt'] > 1), ['from','from_side','to','to_side','mean_dist']].copy()
    short_bridges.rename(columns={'to':'scaf1','to_side':'strand1','mean_dist':'dist1'}, inplace=True)
    short_bridges['strand1'] = np.where(short_bridges['strand1'] == 'l', '+', '-')
    short_bridges['size'] = 2
    short_bridges['org_pos'] = 0
    scaffold_graph = pd.concat([scaffold_graph, short_bridges[['from','from_side','size','org_pos','scaf1','strand1','dist1']]], ignore_index=True, sort=False)
#
    # Now remove all the paths that overlap a longer one (for equally long ones just take one of them)
    scaffold_graph['length'] = scaffold_graph['size'] - scaffold_graph['org_pos']
    scaffold_graph.sort_values(['from','from_side']+[col for sublist in [['scaf'+str(s),'strand'+str(s),'dist'+str(s)] for s in range(1,scaffold_graph['size'].max())] for col in sublist], inplace=True)
    scaffold_graph['redundant'] = (scaffold_graph['from'] == scaffold_graph['from'].shift(1, fill_value=-1)) & (scaffold_graph['from_side'] == scaffold_graph['from_side'].shift(1, fill_value=''))
    for s in range(1,scaffold_graph['size'].max()):
        scaffold_graph['redundant'] = scaffold_graph['redundant'] & ( np.isnan(scaffold_graph['scaf'+str(s)]) | ( (scaffold_graph['scaf'+str(s)] == scaffold_graph['scaf'+str(s)].shift(1, fill_value=-1)) & (scaffold_graph['strand'+str(s)] == scaffold_graph['strand'+str(s)].shift(1, fill_value='')) & (scaffold_graph['dist'+str(s)] == scaffold_graph['dist'+str(s)].shift(1, fill_value='')))) 
    scaffold_graph = scaffold_graph[ scaffold_graph['redundant'] == False ].copy()
    scaffold_graph.drop(columns=['redundant','size','org_pos'],inplace=True)

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

def RemoveEmptyColumns(df):
    # Clean up columns with all NaN
    cols = df.count()
    df.drop(columns=cols[cols == 0].index.values,inplace=True)
    return df

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

def AddPathTroughRepeats(scaffold_paths, scaffold_graph):
    # Repeats need to be handled separately
    repeated_scaffolds = []
    for s in range(1, scaffold_graph['length'].max()):
        repeated_scaffolds.append(np.unique(scaffold_graph.loc[scaffold_graph['from'] == scaffold_graph[f'scaf{s}'], 'from']))
    repeated_scaffolds = np.unique(np.concatenate(repeated_scaffolds))
#
    # Finally handle repeats
    repeat_graph = scaffold_graph[np.isin(scaffold_graph['from'], repeated_scaffolds)].copy()
    ravels = []
    for s in range(1, repeat_graph['length'].max()):
        ravels.append(repeat_graph.loc[np.isnan(repeat_graph[f'scaf{s}']) == False, ['from',f'scaf{s}']].rename(columns={'from':'ravel',f'scaf{s}':'scaf'}))
    ravels = pd.concat(ravels, ignore_index=True).drop_duplicates()
    ravels['scaf'] = ravels['scaf'].astype(int)
    ravels['repeated'] = np.isin(ravels['scaf'], repeated_scaffolds)
    while True:
        ravels['min_ravel'] = ravels[['scaf']].merge(ravels.loc[ravels['repeated'], ['scaf','ravel']].groupby(['scaf']).min().reset_index(), on=['scaf'], how='left')['ravel'].values
        ravels['min_ravel'] = ravels[['ravel']].merge(ravels.loc[ravels['repeated'], ['ravel','min_ravel']].groupby(['ravel']).min().reset_index(), on=['ravel'], how='left')['min_ravel'].values
        if np.sum(ravels['min_ravel'] != ravels['ravel']):
            break
        else:
            ravels['ravel'] = ravels['min_ravel'].astype(int)
    ravels.drop(columns=['min_ravel'], inplace=True)
    ravels.drop_duplicates(inplace=True)
    
    
    
    
    exit_graph = scaffold_graph[np.isin(scaffold_graph['from'], ravels.loc[ravels['repeated'] == False, 'scaf'].values)].copy()
    
    return scaffold_paths

def AddUntraversedConnectedPaths(scaffold_paths, knots, scaffold_graph):
    # Get untraversed connections to neighbouring scaffolds in graph
    conn_path = scaffold_graph.loc[np.isin(scaffold_graph.index.values, knots['oindex'].values) == False, ['from','from_side','scaf1','strand1','dist1']].drop_duplicates()
    # Require that it is untraversed in both directions, otherwise we would duplicate the end of a phased path
    conn_path['to_side'] = np.where(conn_path['strand1'] == '+', 'l', 'r')
    conn_path = conn_path.merge(conn_path[['from','from_side','scaf1','to_side','dist1']].rename(columns={'from':'scaf1','from_side':'to_side','scaf1':'from','to_side':'from_side'}), on=['from','from_side','scaf1','to_side','dist1'], how='inner')
    # Only use one of the two directions for the path
    conn_path = conn_path[(conn_path['from'] < conn_path['scaf1']) | ((conn_path['from'] == conn_path['scaf1']) & ((conn_path['from_side'] == conn_path['to_side']) | (conn_path['from_side'] == 'r')))].drop(columns=['to_side'])
    # Turn into proper format for scaffold_paths
    conn_path['strand0'] = np.where(conn_path['from_side'] == 'r', '+', '-')
    conn_path['pid'] = np.arange(len(conn_path)) + 1 + scaffold_paths['pid'].max()
    conn_path['dist0'] = 0
    conn_path['pos'] = 0
    conn_path['pos1'] = 1

    scaffold_paths = pd.concat([scaffold_paths, conn_path[['pid','pos','from','strand0','dist0']].rename(columns={'from':'scaf0'}), conn_path[['pid','pos1','scaf1','strand1','dist1']].rename(columns={'pos1':'pos','scaf1':'scaf0','strand1':'strand0','dist1':'dist0'})], ignore_index=True)
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

def GetDuplications(scaffold_paths, ploidy):
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
    duplications = duplications.rename(columns={col:f'a{col}' for col in duplications.columns if col != "scaf"}).merge(duplications.rename(columns={col:f'b{col}' for col in duplications.columns if col != "scaf"}), on=['scaf'], how='left') # 'left' keeps the order and we always have at least the self mapping, thus 'inner' does not reduce the size
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
    if ploidy == 1:
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

def ExtendDuplicationsFromEnd(duplications, scaffold_paths, end, scaf_len, ploidy):
    # Get duplications starting at end
    ext_dups = []
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

def RemoveDuplicates(scaffold_paths, remove_all, ploidy):
    # Get duplications that contain both path ends for side a
    duplications = GetDuplications(scaffold_paths, ploidy)
    scaf_len = scaffold_paths.groupby(['pid'])['pos'].max().reset_index()
    ends = duplications.groupby(['apid','bpid'])['apos'].agg(['min','max']).reset_index()
    ends['alen'] = ends[['apid']].rename(columns={'apid':'pid'}).merge(scaf_len, on=['pid'], how='left')['pos'].values
    ends = ends.loc[(ends['min'] == 0) & (ends['max'] == ends['alen']), ['apid','bpid','alen']].copy()
    duplications = duplications.merge(ends, on=['apid','bpid'], how='inner')
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
    rem_paths.rename(columns={'apid':'pid','ahap':'hap'}, inplace=True)
    scaffold_paths = RemoveHaplotypes(scaffold_paths, rem_paths, ploidy)
    scaffold_paths = CompressPaths(scaffold_paths, ploidy)
#
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

def GetNextPositionInPathA(ends, scaffold_paths, ploidy):
    col_dict = {'dir':'bdir','adir':'dir','bpid':'apid','apid':'bpid','ahap':'bhap','bhap':'ahap'}
    ends['adir'] *= -1 # adir steps away from path b, but for this we want to step towards path b
    ends.rename(columns=col_dict, inplace=True)
    ends = GetNextPositionInPathB(ends, scaffold_paths, ploidy)
    ends.rename(columns={v:k for k,v in col_dict.items()}, inplace=True)
    ends['adir'] *= -1
    ends = GetPositionFromPaths(ends, scaffold_paths, ploidy, 'next_strand', 'strand', 'apid', 'mpos', 'ahap')
    ends['next_strand'] = ends['next_strand'].fillna('')
    if np.sum(ends['adir'] == 1):
        ends.loc[ends['adir'] == 1, 'next_strand'] = np.where(ends.loc[ends['adir'] == 1, 'next_strand'] == '+', '-', '+')
    ends['dist_pos'] = np.where(ends['adir'] == -1, ends['mpos'], ends['opos'])
    ends = GetPositionFromPaths(ends, scaffold_paths, ploidy, 'next_dist', 'dist', 'apid', 'dist_pos', 'ahap')
#
    return ends

def FilterInvalidConnections(ends, scaffold_paths, scaffold_graph, ploidy, symmetrical):
    ends['short'] = False # If path b is shorter than the scaffold_graph coming from path a
    if len(ends):
        ends['apos'] = np.where(ends['aside'] == 'l', ends['amax'], ends['amin'])
        ends['adir'] = np.where(ends['aside'] == 'l', 1, -1)
        ends['dir'] = np.where(ends['bside'] == 'l', 1, -1)
        # Get the possible branches for this scaffold to later see where the alternative scaffolds branch of and check for consistency only at the branch points to limit the advantage of longer reads (otherwise the direction where the read connecting most scaffolds is going is always chosen)
        ends['mpos'] = np.where(ends['aside'] == 'l', 0, ends['alen'])
        ends[['from','from_side']] = ends[['apid','mpos']].rename(columns={'apid':'pid','mpos':'pos'}).merge(scaffold_paths[['pid','pos','scaf0','strand0']], on=['pid','pos'], how='left')[['scaf0','strand0']].values
        ends['from_side'] = np.where((ends['from_side'] == '+') == (ends['aside'] == 'r'), 'l', 'r')
        ends['nbranches'] = ends[['from','from_side']].merge(scaffold_graph.groupby(['from','from_side']).size().reset_index(name='nbranches'), on=['from','from_side'], how='left')['nbranches'].values
        col_dict = {'dir':'bdir','adir':'dir','bpid':'apid','apid':'bpid','ahap':'bhap','bhap':'ahap'}
        ends.rename(columns=col_dict, inplace=True)
        ends = GetNextPositionInPathB(ends, scaffold_paths, ploidy)
        ends.rename(columns={v:k for k,v in col_dict.items()}, inplace=True) # Go back to original names
        branches = ends[['from','from_side','next_scaf']].reset_index().rename(columns={'index':'eindex','next_scaf':'scaf1'}).merge(scaffold_graph[['from','from_side','scaf1']].reset_index().rename(columns={'index':'sindex'}), on=['from','from_side','scaf1'], how='inner') # Already use scaf1 here to drastically limit the number of branches, but we have to be careful to not use the reduced number to calculate nbranches until we truely checked for the first position
        branches.drop(columns=['from','from_side','scaf1'], inplace=True)
        branches['invalid'] = False
        ends.drop(columns=['from','from_side'], inplace=True)
        ends['spos'] = 1
        spos = 1
        while True:
            # If amin != amax we need to follow the branches until we reach the end of the overlap
            ends = GetPositionFromPaths(ends, scaffold_paths, ploidy, 'next_strand', 'strand', 'apid', 'mpos', 'ahap')
            ends['next_strand'] = ends['next_strand'].fillna('')
            if np.sum(ends['adir'] == -1):
                ends.loc[ends['adir'] == -1, 'next_strand'] = np.where(ends.loc[ends['adir'] == -1, 'next_strand'] == '+', '-', '+')
            ends['dist_pos'] = np.where(ends['adir'] == 1, ends['mpos'], ends['opos'])
            ends = GetPositionFromPaths(ends, scaffold_paths, ploidy, 'next_dist', 'dist', 'apid', 'dist_pos', 'ahap')
            ends['test'] = ends['mpos']*ends['adir'] <= ends['apos']*ends['adir']
            if np.sum(ends['test']):
                # Test
                test = ends.loc[branches['eindex'].values, 'test'].values
                branches.loc[test, 'invalid'] = (scaffold_graph.loc[branches.loc[test, 'sindex'].values, [f'scaf{spos}',f'strand{spos}',f'dist{spos}']].values != ends.loc[branches.loc[test, 'eindex'].values, ['next_scaf','next_strand','next_dist']].values).any(axis=1)
                branches = branches[branches['invalid'] == False].copy()
                ends.loc[ends['test'], 'spos'] += 1
                spos += 1
                # Update nbranches
                nbranches = branches.groupby(['eindex']).size().reset_index(name='nbranches')
                ends.loc[nbranches['eindex'].values, 'nbranches'] = np.where(ends.loc[nbranches['eindex'].values, 'test'], nbranches['nbranches'].values, ends.loc[nbranches['eindex'].values, 'nbranches'])
                # Go to next position
                ends.rename(columns=col_dict, inplace=True)
                ends = GetNextPositionInPathB(ends, scaffold_paths, ploidy)
                ends.rename(columns={v:k for k,v in col_dict.items()}, inplace=True)
            else:
                break
        # Start main loop to see if ends are valid
        smin = 2
        while True:
            # Get the next position in path a
            ends['mpos'] = ends['apos']
            ends.rename(columns=col_dict, inplace=True)
            ends = GetNextPositionInPathB(ends, scaffold_paths, ploidy)
            ends.rename(columns={v:k for k,v in col_dict.items()}, inplace=True) # Go back to original names
            ends['apos'] = ends['mpos']
            ends = GetPositionFromPaths(ends, scaffold_paths, ploidy, 'next_strand', 'strand', 'apid', 'apos', 'ahap')
            ends['next_strand'] = ends['next_strand'].fillna('')
            if np.sum(ends['adir'] == -1):
                ends.loc[ends['adir'] == -1, 'next_strand'] = np.where(ends.loc[ends['adir'] == -1, 'next_strand'] == '+', '-', '+')
            ends['dist_pos'] = np.where(ends['adir'] == 1, ends['mpos'], ends['opos'])
            ends = GetPositionFromPaths(ends, scaffold_paths, ploidy, 'next_dist', 'dist', 'apid', 'dist_pos', 'ahap')
            # Check branches
            ends.loc[np.isnan(ends['next_scaf']) | (ends['nbranches'] == 1), 'nbranches'] = 0 # If we are at the end of the path or do not have any branch points anymore, we do not need further checks on this end (ends['nbranches'] == 1 must be checked on the values from last round, because there the path is already validated, which is not the case for this round)
            branches = branches[ ends.loc[branches['eindex'].values, 'nbranches'].values > 0 ].copy()
            ends['valid'] = (ends['nbranches'] == 0) # If the path does not have any branch points anymore it is automatically valid
            for spos in range(ends['spos'].min(), ends['spos'].max()+1):
                ends['test'] = (ends['nbranches'] > 0) & (ends['spos'] == spos)
                test = ends.loc[branches['eindex'].values, 'test'].values
                branches.loc[test, 'invalid'] = (scaffold_graph.loc[branches.loc[test, 'sindex'].values, [f'scaf{spos}',f'strand{spos}',f'dist{spos}']].values != ends.loc[branches.loc[test, 'eindex'].values, ['next_scaf','next_strand','next_dist']].values).any(axis=1)
            ends['spos'] += 1
            branches = branches[branches['invalid'] == False].copy()
            nbranches = branches.groupby(['eindex']).size().reset_index(name='nbranches')
            nbranches['old_number'] = ends.loc[nbranches['eindex'].values, 'nbranches'].values
            ends.loc[nbranches['eindex'].values, 'nbranches'] = nbranches['nbranches'].values
            ends.loc[nbranches.loc[nbranches['old_number'] == nbranches['nbranches'], 'eindex'].values, 'valid'] = True # If we do not have a break point those ends are automatically valid and we do not test them this round
            # Get the corresponding 'from' 'from_side' for scaffold_graph
            pairs = ends.loc[ends['valid'] == False, ['next_scaf','next_strand']].reset_index()
            if len(pairs):
                pairs.rename(columns={'index':'eindex','next_scaf':'from','next_strand':'from_side'}, inplace=True)
                pairs['from'] = pairs['from'].astype(int)
                pairs['from_side'] = np.where(pairs['from_side'] == '+', 'l', 'r')
                # Include all the requirements that the scaffold_graph entry matches path a up to path b (in path b we do not need to check, because we later do for path b and in the duplicated region they must be identical)
                for s in range(1,smin):
                    ends = GetNextPositionInPathA(ends, scaffold_paths, ploidy)
                    pairs[f'scaf{s}'] = ends.loc[pairs['eindex'].values, 'next_scaf'].fillna(-1).values.astype(int)
                    pairs[f'strand{s}'] = ends.loc[pairs['eindex'].values, 'next_strand'].fillna('').values
                    pairs[f'dist{s}'] = ends.loc[pairs['eindex'].values, 'next_dist'].fillna(0).values.astype(int)
                mcols = ['from','from_side']+[f'{n}{s}' for s in range(1,smin) for n in ['scaf','strand','dist']]
                pairs = pairs.merge(scaffold_graph[mcols].reset_index(), on=mcols, how='inner')
                pairs.drop(columns=mcols, inplace=True)
                pairs.rename(columns={'index':'sindex'}, inplace=True)
                # If path in scaffold_graph does not reach path b it is automatically valid
                ends['reachesb'] = False
                ends.loc[pairs.loc[scaffold_graph.loc[pairs['sindex'].values, 'length'].values > smin, 'eindex'].drop_duplicates().values, 'reachesb'] = True
                ends.loc[(ends['reachesb'] == False) & (ends['valid'] == False), 'nbranches'] = 0 # If we did check, but do not have a pair that reaches the unique part of path b, we do not need to check anymore
                ends.loc[ends['reachesb'] == False, 'valid'] = True
                ends.drop(columns=['reachesb'], inplace=True)
                pairs = pairs[ends.loc[pairs['eindex'], 'valid'].values == False].copy()
            if len(pairs):
                s = smin
                ends['mpos'] = np.where(ends['bside'] == 'l', ends['bmin'], ends['bmax'])
                ends = GetNextPositionInPathB(ends, scaffold_paths, ploidy)
                while len(pairs):
                    # Get strand and distance between opos and mpos
                    ends = GetPositionFromPaths(ends, scaffold_paths, ploidy, 'next_strand', 'strand', 'bpid', 'mpos', 'bhap')
                    ends.loc[ends['dir'] == -1, 'next_strand'] = np.where(ends.loc[ends['dir'] == -1, 'next_strand'] == '+', '-', '+')
                    ends['dist_pos'] = np.where(ends['dir'] == 1, ends['mpos'], ends['opos'])
                    ends = GetPositionFromPaths(ends, scaffold_paths, ploidy, 'next_dist', 'dist', 'bpid', 'dist_pos', 'bhap')
                    # Check if this matches with scaffold_graph
                    pairs = pairs[ (scaffold_graph.loc[pairs['sindex'].values, [f'scaf{s}',f'strand{s}',f'dist{s}']].values == ends.loc[pairs['eindex'].values, ['next_scaf','next_strand','next_dist']].values).all(axis=1) ].copy()
                    s += 1
                    # Mark the ends that reached the end of entry in scaffold_graph as valid
                    ends.loc[pairs.loc[scaffold_graph.loc[pairs['sindex'].values, 'length'].values == s, 'eindex'].values, 'valid'] = True
                    if len(pairs):
                        # If we are at the other end of path b set it as valid but short (we do not delete anything here, because the deletion operation takes a lot of time)
                        ends = GetNextPositionInPathB(ends, scaffold_paths, ploidy)
                        ends['active'] = False
                        ends.loc[pairs['eindex'].drop_duplicates().values, 'active'] = True
                        ends.loc[ends['active'] & (ends['valid'] == False) & ((ends['mpos'] < 0) | (ends['mpos'] > ends['blen'])), ['valid','short']] = True
                        ends.drop(columns=['active'], inplace=True)
                        pairs = pairs[ends.loc[pairs['eindex'].values, 'valid'].values == False].copy()
            ends = ends[ends['valid']].copy()
            if symmetrical:
                mcols = [f'{p}{n}' for p in ['a','b'] for n in ['pid','hap','min','max']]
                ends = ends[ends[mcols].rename(columns={**{f'a{n}':f'b{n}' for n in ['pid','hap','min','max']}, **{f'b{n}':f'a{n}' for n in ['pid','hap','min','max']}}).merge(ends[mcols], on=mcols, how='left', indicator=True).values == "both"].copy()
            branches = branches[np.isin(branches['eindex'], ends.index.values)].copy()
            smin += 1
            if np.sum(ends['nbranches'] > 0) == 0: # No entries in scaffold_graph reach path b anymore
                break
        ends.drop(columns=['apos','adir','dir','mpos','nbranches','opos','next_scaf','next_strand','valid','dist_pos','next_dist','spos','test'], inplace=True)
#
    return ends

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

def SetConnectablePathsInMetaScaffold(meta_scaffolds, ends, connectable):
    meta_scaffolds['connectable'] = meta_scaffolds['connectable'] | (meta_scaffolds[['new_pid','bpid']].rename(columns={'new_pid':'apid'}).merge(connectable[['apid','bpid']], on=['apid','bpid'], how='left', indicator=True)['_merge'].values == "both")
    ends = ends[ends[['apid','bpid']].merge(connectable[['apid','bpid']], on=['apid','bpid'], how='left', indicator=True)['_merge'].values == "left_only"].copy()
#
    return meta_scaffolds, ends

def GetNumberOfHaplotypes(scaffold_paths, ploidy):
    return (((scaffold_paths.groupby(['pid'])[[f'phase{h}' for h in range(ploidy)]].max() >= 0)*[h for h in range(ploidy)]).max(axis=1)+1).reset_index(name='nhaps')

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
    bsupp = bsupp.groupby(['group','pid','hap'])['bcount'].agg(['min','median','max']).reset_index()
#
    return bsupp

def CombinePathAccordingToMetaParts(scaffold_paths, meta_parts, conns, scaffold_graph, scaf_bridges, scaf_len, ploidy):
    # Combine scaffold_paths from lowest to highest position in meta scaffolds
    meta_scaffolds = meta_parts.loc[meta_parts['pos'] == 0].drop(columns=['pos'])
    scaffold_paths['reverse'] = scaffold_paths[['pid']].merge(meta_scaffolds[['pid','reverse']], on=['pid'], how='left')['reverse'].fillna(False).values.astype(bool)
    scaffold_paths = ReverseScaffolds(scaffold_paths, scaffold_paths['reverse'], ploidy)
    scaffold_paths.drop(columns=['reverse'], inplace=True)
    meta_scaffolds['start_pos'] = meta_scaffolds[['pid']].merge(scaf_len, on=['pid'], how='left')['pos'].values + 1
    meta_scaffolds.rename(columns={'pid':'new_pid'}, inplace=True)
    meta_scaffolds['apid'] = meta_scaffolds['new_pid']
    meta_scaffolds['aside'] = np.where(meta_scaffolds['reverse'], 'l', 'r')
    meta_scaffolds.drop(columns=['reverse'], inplace=True)
    meta_parts = meta_parts[meta_parts['pos'] > 0].copy()
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
        for iteration in range(5):
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
            ends = FilterInvalidConnections(ends, scaffold_paths, scaffold_graph, ploidy, False)
            if len(ends) == 0:
                break
            else:
                ends = ends[['apid','ahap','bpid','bhap']].merge(ends[['apid','ahap','bpid','bhap']].rename(columns={'apid':'bpid','ahap':'bhap','bpid':'apid','bhap':'ahap'}), on=['apid','ahap','bpid','bhap'], how='outer', indicator=True)
                ends = ends.merge(meta_scaffolds[['new_pid','bpid']].rename(columns={'new_pid':'apid'}), on=['apid','bpid'], how='inner')
#
                for p in ['a','b']:
                    ends[f'{p}nhaps'] = ends[[f'{p}pid']].rename(columns={f'{p}pid':'pid'}).merge(nhaps, on=['pid'], how='left')['nhaps'].values
                ends['nhaps'] = np.maximum(ends['anhaps'], ends['bnhaps'])
                 # When all haplotypes either match to the corresponding haplotype or, if the corresponding haplotype does not exist, to the main, scaffolds are connectable
                connectable = ends[ends['_merge'] == "both"].drop(columns=['_merge'])
                connectable = connectable[(connectable['ahap'] == connectable['bhap']) | ((connectable['ahap'] == 0) & (connectable['bhap'] >= connectable['anhaps'])) | 
                                                                                         ((connectable['bhap'] == 0) & (connectable['ahap'] >= connectable['bnhaps']))].copy()
                connectable = connectable.groupby(['apid','bpid','nhaps']).size().reset_index(name='matches')
                connectable = connectable[connectable['nhaps'] == connectable['matches']].copy()
                meta_scaffolds, ends = SetConnectablePathsInMetaScaffold(meta_scaffolds, ends, connectable)
                if 0 == iteration:
                    # The first step is to bring everything that is valid from both sides to the lower haplotypes
                    for p1, p2 in zip(['a','b'],['b','a']):
                        connectable = ends[(ends['_merge'] == "both") & (ends[f'{p1}nhaps'] > ends[f'{p2}nhaps'])].drop(columns=['_merge'])
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
                        connectable = ends[ends['_merge'] == "both"].drop(columns=['_merge'])
                        connectable = connectable[ (connectable[f'{p2}hap'] >= connectable[f'{p1}nhaps']) ].copy()
                        connectable.sort_values(['apid','bpid',f'{p2}hap',f'{p1}hap'], inplace=True)
                        connectable = connectable.groupby(['apid','bpid',f'{p2}hap']).first().reset_index() # Take the lowest haplotype that can be duplicated to fill the missing one
                        connectable = connectable.loc[connectable[f'{p1}hap'] > 0, [f'{p1}pid',f'{p1}hap',f'{p2}hap']].rename(columns={f'{p1}pid':'pid', f'{p1}hap':'hap1', f'{p2}hap':'hap2'}) # If the lowest is the main, we do not need to do anything
                        scaffold_paths = DuplicateHaplotypes(scaffold_paths, connectable, ploidy)
                    # Switching haplotypes might also help to make paths connectable
                    connectable = ends[ends['_merge'] == "both"].drop(columns=['_merge'])
                    connectable['nhaps'] = np.minimum(connectable['anhaps'], connectable['bnhaps'])
                    connectable = connectable[(connectable['ahap'] < connectable['nhaps']) & (connectable['bhap'] < connectable['nhaps'])].drop(columns=['anhaps','bnhaps'])
                    connectable['nmatches'] = connectable[['apid','bpid']].merge(connectable[connectable['ahap'] == connectable['bhap']].groupby(['apid','bpid']).size().reset_index(name='matches'), on=['apid','bpid'], how='left')['matches'].fillna(0).values.astype(int)
                    connectable = connectable[connectable['nmatches'] < connectable['nhaps']].copy()
                    connectable['new_bhap'] = connectable['bhap']
                    while len(connectable):
                        connectable['match'] = connectable['ahap'] == connectable['new_bhap']
                        connectable['amatch'] = connectable[['apid','bpid','ahap']].merge(connectable.groupby(['apid','bpid','ahap'])['match'].max().reset_index(), on=['apid','bpid','ahap'], how='left')['match'].values
                        connectable['bmatch'] = connectable[['apid','bpid','new_bhap']].merge(connectable.groupby(['apid','bpid','new_bhap'])['match'].max().reset_index(), on=['apid','bpid','new_bhap'], how='left')['match'].values
                        connectable['switchable'] = connectable[['apid','bpid','ahap','new_bhap']].merge( connectable[['apid','bpid','ahap','new_bhap']].rename(columns={'ahap':'new_bhap','new_bhap':'ahap'}), on=['apid','bpid','ahap','new_bhap'], how='left', indicator=True)['_merge'].values == "both"
                        switches = connectable.loc[(connectable['amatch'] == False) & ((connectable['bmatch'] == False) | connectable['switchable']), ['apid','ahap','bpid','new_bhap']].copy()
                        switches = switches.groupby(['apid','bpid']).first().reset_index() # Only one switch per meta_paths per round to avoid conflicts
                        switches.rename(columns={'new_bhap':'bhap','ahap':'new_bhap'}, inplace=True)
                        switches = pd.concat([switches.rename(columns={'new_bhap':'bhap','bhap':'new_bhap'}), switches], ignore_index=True)
                        switches = connectable[['apid','bpid','new_bhap']].rename(columns={'new_bhap':'bhap'}).merge(switches, on=['apid','bpid','bhap'], how='left')['new_bhap'].values
                        connectable['new_bhap'] = np.where(np.isnan(switches), connectable['new_bhap'], switches).astype(int)
                        connectable['old_nmatches'] = connectable['nmatches']
                        connectable['nmatches'] = connectable[['apid','bpid']].merge(connectable[connectable['ahap'] == connectable['new_bhap']].groupby(['apid','bpid']).size().reset_index(name='matches'), on=['apid','bpid'], how='left')['matches'].fillna(0).values.astype(int)
                        improvable = (connectable['old_nmatches'] < connectable['nmatches']) & (connectable['nmatches'] < connectable['nhaps'])
                        switches = connectable.loc[(improvable == False) & (connectable['bhap'] != connectable['new_bhap']), ['bpid','bhap','new_bhap']].drop_duplicates()
                        switches.rename(columns={'bpid':'pid','bhap':'hap1','new_bhap':'hap2'}, inplace=True)
                        scaffold_paths = SwitchHaplotypes(scaffold_paths, switches, ploidy)
                        connectable = connectable[improvable].copy()
                elif 2 == iteration or 4 == iteration:
                    # If all haplotype of the path with fewer haplotypes are continued in the one with more haplotypes and the non-continued from the one with more haplotypes do not reach the unqiue part of the other path and can be incorporated as new haplotype in the other path, the connection is also valid
                    ends['aconsistent'] = ends['_merge'] == "left_only"
                    ends['bconsistent'] = ends['_merge'] == "right_only"
                    ends.loc[ends['_merge'] == "both", ['aconsistent','bconsistent']] = True
                    for p in ['a','b']:
                        ends[f'{p}hascon'] = ends[['apid','bpid',f'{p}hap']].merge(ends.groupby(['apid','bpid',f'{p}hap'])[f'{p}consistent'].max().reset_index(), on=['apid','bpid',f'{p}hap'], how='left')[f'{p}consistent'].values
                        ends[f'{p}ncon'] = ends[['apid','bpid']].merge(ends[['apid','bpid',f'{p}hap',f'{p}hascon']].drop_duplicates().groupby(['apid','bpid'])[f'{p}hascon'].sum().reset_index(), on=['apid','bpid'], how='left')[f'{p}hascon'].values
                    connectable = ends[(ends['anhaps'] == ends['ancon']) & (ends['bnhaps'] == ends['bncon'])].copy()
                    connectable = connectable[ ((connectable['ahap'] == connectable['bhap']) & (connectable['_merge'] == "both")) | # Because we require both sides to be consistent for everything with existing haplotypes, we cannot proceed exactly like in the case, where we only considered the both case, because adding the haplotypes invalidades this and next round they would be sorted out
                                               ((connectable['bhap'] >= connectable['anhaps']) & connectable['bconsistent']) | 
                                               ((connectable['ahap'] >= connectable['bnhaps']) & connectable['aconsistent']) ].copy()
                    # Keep only lowest possible haplotypes in other path to match haplotypes that exceed the number of haplotypes in the other path
                    for p1, p2 in zip(['a','b'],['b','a']):
                        connectable.sort_values(['apid','bpid',f'{p2}hap',f'{p1}hap'], inplace=True)
                        connectable = connectable[(connectable[f'{p1}nhaps'] >= connectable[f'{p2}nhaps']) | (connectable['apid'] != connectable['apid'].shift(1)) | (connectable[f'{p2}hap'] != connectable[f'{p2}hap'].shift(1))].copy()
                    # Check that all haplotypes have a match
                    connectable['matches'] = connectable[['apid','bpid']].merge(connectable.groupby(['apid','bpid']).size().reset_index(name='matches'), on=['apid','bpid'], how='left')['matches'].values
                    connectable = connectable[connectable['nhaps'] == connectable['matches']].copy()
                    # Copy missing haplotypes in scaffold_paths
                    for p1, p2 in zip(['a','b'],['b','a']):
                        duplicate = connectable.loc[(connectable[f'{p2}hap'] >= connectable[f'{p1}nhaps']) & (connectable[f'{p1}hap'] > 0), [f'{p1}pid',f'{p1}hap',f'{p2}hap']].rename(columns={f'{p1}pid':'pid', f'{p1}hap':'hap1', f'{p2}hap':'hap2'})
                        scaffold_paths = DuplicateHaplotypes(scaffold_paths, duplicate, ploidy)
                    # Register valid connections
                    meta_scaffolds, ends = SetConnectablePathsInMetaScaffold(meta_scaffolds, ends, connectable[['apid','bpid']].drop_duplicates())
                    if 2 == iteration:
                        # Remove haplotypes that block a connection if they differ only by distance from a valid haplotype
                        delete_haps = []
                        connectable_pids = []
                        for p in ['a','b']:
                            dist_diff_only = GetHaplotypesThatDifferOnlyByDistance(scaffold_paths, ends[f'{p}pid'].drop_duplicates().values, ploidy)
                            valid_haps = ends.loc[ends['_merge'] == "both", [f'{p}pid',f'{p}hap']].drop_duplicates()
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
#
        # Connect scaffolds
        scaffold_paths[['overlap','shift']] = scaffold_paths[['pid']].merge( meta_scaffolds.loc[meta_scaffolds['connectable'], ['bpid','boverlap','start_pos']].rename(columns={'bpid':'pid'}), on=['pid'], how='left')[['boverlap','start_pos']].fillna(0).values.astype(int)
        scaffold_paths['shift'] -= scaffold_paths['overlap']
        scaffold_paths = scaffold_paths[ scaffold_paths['pos'] >= scaffold_paths['overlap'] ].drop(columns=['overlap'])
        scaffold_paths['pos'] += scaffold_paths['shift']
        scaffold_paths.drop(columns=['shift'], inplace=True)
        scaffold_paths['new_pid'] = scaffold_paths[['pid']].merge( meta_scaffolds.loc[meta_scaffolds['connectable'], ['bpid','new_pid']].rename(columns={'bpid':'pid'}), on=['pid'], how='left')['new_pid'].values
        scaffold_paths.loc[np.isnan(scaffold_paths['new_pid']) == False, 'pid'] = scaffold_paths.loc[np.isnan(scaffold_paths['new_pid']) == False, 'new_pid'].astype(int)
        scaffold_paths.drop(columns=['new_pid'], inplace=True)
        scaffold_paths.sort_values(['pid','pos'], inplace=True)
#
        # The unconnectable paths in meta_scaffolds might have had haplotypes duplicated in an attempt to make them connectable. Remove those duplications
        for h1 in range(1, ploidy):
            for h2 in range(h1+1, ploidy):
                remove =  ( (np.sign(scaffold_paths[f'phase{h1}']) == np.sign(scaffold_paths[f'phase{h2}'])) & (scaffold_paths[f'scaf{h1}'] == scaffold_paths[f'scaf{h2}']) &
                            (scaffold_paths[f'strand{h1}'] == scaffold_paths[f'strand{h2}']) & (scaffold_paths[f'dist{h1}'] == scaffold_paths[f'dist{h2}']) )
                scaffold_paths = RemoveHaplotype(scaffold_paths, remove, h2)
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
        bsupp.drop(columns=['group'], inplace=True)
        bsupp.sort_values(['pid','min','median','max'], ascending=[True,False,False,False], inplace=True)
        bsupp['new_hap'] = bsupp.groupby(['pid']).cumcount()
        bsupp = bsupp.loc[bsupp['hap'] != bsupp['new_hap'], ['pid','hap','new_hap']].rename(columns={'hap':'hap1','new_hap':'hap2'})
        scaffold_paths = SwitchHaplotypes(scaffold_paths, bsupp, ploidy)
        scaffold_paths = ShiftHaplotypesToLowestPossible(scaffold_paths, ploidy)
#
        # Break unconnectable meta_scaffolds
        meta_scaffolds.loc[meta_scaffolds['connectable'] == False, 'new_pid'] = meta_scaffolds.loc[meta_scaffolds['connectable'] == False, 'bpid']
        meta_scaffolds.loc[meta_scaffolds['connectable'] == False, 'start_pos'] = 0
#
        # Prepare next round
        meta_scaffolds['start_pos'] += meta_scaffolds[['bpid']].rename(columns={'bpid':'pid'}).merge(scaf_len, on=['pid'], how='left')['pos'].values + 1
        meta_scaffolds['apid'] = meta_scaffolds['bpid']
        meta_scaffolds['aside'] = np.where(meta_scaffolds['reverse'], 'l', 'r')
        meta_scaffolds.drop(columns=['bpid','reverse','bside','aoverlap','boverlap'], inplace=True)
        meta_parts = meta_parts[meta_parts['pos'] > pos].copy()
        pos += 1
#
    return scaffold_paths

def CombinePathOnUniqueOverlap(scaffold_paths, scaffold_graph, scaf_bridges, ploidy):
    ends = GetDuplicatedPathEnds(scaffold_paths, ploidy)
    # If the paths continue in the same direction on the non-end side they are alternatives and not combinable
    ends = ends[ends['samedir'] == (ends['aside'] != ends['bside'])].copy()
#
    # Check that combining the paths does not violate scaffold_graph
    ends = FilterInvalidConnections(ends, scaffold_paths, scaffold_graph, ploidy, True)
#
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
    # Count alternatives and take only unique connections with preferences giving to more haplotypes and matches
    for p in ['a','b']:
        conns.sort_values([f'{p}pid',f'{p}side',f'{p}nhaps','max_matches'], ascending=[True,True,False,False], inplace=True) # Do not use minhaps/maxhaps here, because they are between different scaffolds and this would give an advantage to scaffolds with more haplotypes
        conns[f'{p}alts'] = conns.groupby([f'{p}pid',f'{p}side'], sort=False).cumcount()+1
        equivalent = conns.groupby([f'{p}pid',f'{p}side',f'{p}nhaps','max_matches'], sort=False)[f'{p}alts'].agg(['max','size'])
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
    scaffold_paths = CombinePathAccordingToMetaParts(scaffold_paths, meta_parts, conns, scaffold_graph, scaf_bridges, scaf_len, ploidy)

    return scaffold_paths

def RemoveSelectedDuplicationsToSolveConflicts(conflicts, rem_index):
    nworse = pd.concat([ conflicts.loc[conflicts[f'worse{i}'], [f'index{i}']].rename(columns={f'index{i}':'index'}) for i in [1,2] ], ignore_index=True).groupby(['index']).size().reset_index(name='nworse')
    for i in [1,2]:
        conflicts[f'nworse{i}'] = conflicts[[f'index{i}']].rename(columns={f'index{i}':'index'}).merge(nworse, on=['index'], how='left')['nworse'].fillna(0).values.astype(int)
    remove = pd.concat([ conflicts.loc[(conflicts[f'nworse{i}'] > 0) & (conflicts[f'nworse{j}'] == 0), [f'index{i}','did']].rename(columns={f'index{i}':'index'}) for i,j in zip([1,2],[2,1]) ], ignore_index=True)
    rem_index.append(np.unique(remove['index']))
    conflicts = conflicts[np.isin(conflicts['did'], np.unique(remove['did'])) == False].copy()
#
    return conflicts, rem_index

def SolveDuplicationConflicts(duplications):
    old_len = 0
    while old_len != len(duplications):
        ## Find conflicts
        old_len = len(duplications)
        duplications.reset_index(drop=True, inplace=True) # Make sure the index is continuous with no missing numbers
        duplications['lcons'] = 0
        duplications['rcons'] = 0
        duplications['conflicts'] = 0
        s = 1
        old_conflicts = 1
        conflicts = []
        while old_conflicts != np.sum(duplications['conflicts']):
            old_conflicts = np.sum(duplications['conflicts'])
            duplications['dist'] = duplications['bpos'] - duplications['bpos'].shift(s)
            duplications.loc[((duplications['did'] == duplications['did'].shift(s)) & ((duplications['apos'] == duplications['apos'].shift(s)) | (duplications['dist'] == 0) | ((duplications['dist'] < 0) == duplications['samedir']))), 'lcons'] += 1
            duplications['dist'] = duplications['bpos'].shift(-s) - duplications['bpos']
            conflict_found = ((duplications['did'] == duplications['did'].shift(-s)) & ((duplications['apos'] == duplications['apos'].shift(-s)) | (duplications['dist'] == 0) | ((duplications['dist'] < 0) == duplications['samedir'])))
            duplications.loc[conflict_found, 'rcons'] += 1
            duplications['conflicts'] = duplications['lcons']+duplications['rcons']
            conflicts.append(pd.DataFrame({'index1':duplications[conflict_found].index.values, 'index2':duplications[conflict_found].index.values + s}))
            s += 1
        conflicts = pd.concat(conflicts, ignore_index=True)
        conflicts['did'] = duplications.loc[conflicts['index1'].values, 'did'].values
        ## Remove the duplication of a conflict that is at least in one of all of its conflicts the worse(based on differnt criteria), while the other never is the worse (This makes sure that we do not accidentally remove both sides of a conflict, where one would be enough to solve all conflicts)
        rem_index = []
        # Critetia 1: Total number of conflicts
        conflicts['conflicts1'] = duplications.loc[conflicts['index1'].values, 'conflicts'].values
        conflicts['conflicts2'] = duplications.loc[conflicts['index2'].values, 'conflicts'].values
        conflicts['worse1'] = conflicts['conflicts1'] > conflicts['conflicts2']
        conflicts['worse2'] = conflicts['conflicts2'] > conflicts['conflicts1']
        conflicts, rem_index = RemoveSelectedDuplicationsToSolveConflicts(conflicts, rem_index)
        # Critetia 2: Max number of conflicts on one side
        conflicts['conflicts1'] = duplications.loc[conflicts['index1'].values, ['lcons','rcons']].max(axis=1).values
        conflicts['conflicts2'] = duplications.loc[conflicts['index2'].values, ['lcons','rcons']].max(axis=1).values
        conflicts['worse1'] = conflicts['conflicts1'] > conflicts['conflicts2']
        conflicts['worse2'] = conflicts['conflicts2'] > conflicts['conflicts1']
        conflicts, rem_index = RemoveSelectedDuplicationsToSolveConflicts(conflicts, rem_index)
        conflicts.drop(columns=['conflicts1','conflicts2'], inplace=True)
        # Critetia 3: Distance between positions
        for i in [1,2]:
            conflicts[f'apos{i}'] = duplications.loc[conflicts[f'index{i}'].values, 'apos'].values
            conflicts[f'bpos{i}'] = duplications.loc[conflicts[f'index{i}'].values, 'bpos'].values
            conflicts[f'dist{i}'] = np.abs(conflicts[f'apos{i}']-conflicts[f'bpos{i}'])
        conflicts['worse1'] = conflicts['dist1'] > conflicts['dist2']
        conflicts['worse2'] = conflicts['dist2'] > conflicts['dist1']
        conflicts, rem_index = RemoveSelectedDuplicationsToSolveConflicts(conflicts, rem_index)
        conflicts.drop(columns=['dist1','dist2'], inplace=True)
        # Critetia 4: The highest positions
        for i in [1,2]:
            conflicts[f'maxpos{i}'] = np.maximum(conflicts[f'apos{i}'],conflicts[f'bpos{i}'])
        conflicts['worse1'] = conflicts['maxpos1'] > conflicts['maxpos2']
        conflicts['worse2'] = conflicts['maxpos2'] > conflicts['maxpos1']
        conflicts, rem_index = RemoveSelectedDuplicationsToSolveConflicts(conflicts, rem_index)
        # Critetia 5: The highest positions in the path with the higher pid
        for i in [1,2]:
            conflicts[f'apid{i}'] = duplications.loc[conflicts[f'index{i}'].values, 'apid'].values
            conflicts[f'bpid{i}'] = duplications.loc[conflicts[f'index{i}'].values, 'bpid'].values
            conflicts[f'maxpos{i}'] = np.where(conflicts[f'apid{i}'] > conflicts[f'bpid{i}'], conflicts[f'apos{i}'], conflicts[f'bpos{i}'])
        conflicts['worse1'] = conflicts['maxpos1'] > conflicts['maxpos2']
        conflicts['worse2'] = conflicts['maxpos2'] > conflicts['maxpos1']
        conflicts, rem_index = RemoveSelectedDuplicationsToSolveConflicts(conflicts, rem_index)
        # Delete duplications
        duplications.drop(np.concatenate(rem_index), inplace=True)
    if old_len:
        duplications.drop(columns=['lcons','rcons','conflicts','dist'], inplace=True)
#
    return duplications

def RequireContinuousDirectionForDuplications(duplications):
    # Sort them in the direction the positions should change
    duplications.loc[duplications['samedir'] == False, 'bpos'] *= -1
    duplications.sort_values(['apid','ahap','bpid','bhap','apos','bpos'], inplace=True)
    duplications.loc[duplications['samedir'] == False, 'bpos'] *= -1
    duplications['did'] = ((duplications['apid'] != duplications['apid'].shift(1)) | (duplications['ahap'] != duplications['ahap'].shift(1)) | (duplications['bpid'] != duplications['bpid'].shift(1)) | (duplications['bhap'] != duplications['bhap'].shift(1))).cumsum()
    duplications.reset_index(drop=True, inplace=True)
    # Store the duplications to try alternative mappings later
    alt_duplications = duplications.copy() 
    # Remove conflicts
    duplications = SolveDuplicationConflicts(duplications)
    dup_count = duplications.groupby(['did']).size().reset_index(name='full')
    # Try to shorten the sequence at duplicated starts/ends to improve mapping
    for p in ['a','b']:
        for d in ['e','s']: # Start with trimming at the end
            while True:
                ## See which ones to keep for this and the next rounds and how much to trim in this round
                # p='b', d='s' (we do it in reverse order of the loop to overwrite 'trim', but still keep 'keep')
                alt_duplications['trim'] = alt_duplications[['did']].merge(alt_duplications.loc[(alt_duplications['did'] == alt_duplications['did'].shift(1)) & (alt_duplications['apos'] == alt_duplications['apos'].shift(1)), ['did','bpos']].groupby(['did']).min().reset_index(), on=['did'], how='left')['bpos'].values
                alt_duplications['keep'] = np.isnan(alt_duplications['trim']) == False
                if p == 'a' or d == 'e':
                    # p='b', d='e'
                    alt_duplications['trim'] = alt_duplications[['did']].merge(alt_duplications.loc[(alt_duplications['did'] == alt_duplications['did'].shift(-1)) & (alt_duplications['apos'] == alt_duplications['apos'].shift(-1)), ['did','bpos']].groupby(['did']).max().reset_index(), on=['did'], how='left')['bpos'].values
                    alt_duplications['keep'] = alt_duplications['keep'] | (np.isnan(alt_duplications['trim']) == False)
                if p == 'a':
                    # p='a', d='s'
                    alt_duplications['capos'] = alt_duplications[['did','bpos']].merge(alt_duplications.groupby(['did','bpos'])['apos'].min().reset_index(), on=['did','bpos'], how='left')['apos'].values
                    alt_duplications['trim'] = alt_duplications[['did']].merge(alt_duplications.loc[alt_duplications['apos'] != alt_duplications['capos'], ['did','apos']].groupby(['did']).min().reset_index(), on=['did'], how='left')['apos'].values
                    alt_duplications['keep'] = alt_duplications['keep'] | (np.isnan(alt_duplications['trim']) == False)
                    if d == 'e':
                        # p='a', d='e'
                        alt_duplications['capos'] = alt_duplications[['did','bpos']].merge(alt_duplications.groupby(['did','bpos'])['apos'].max().reset_index(), on=['did','bpos'], how='left')['apos'].values
                        alt_duplications['trim'] = alt_duplications[['did']].merge(alt_duplications.loc[alt_duplications['apos'] != alt_duplications['capos'], ['did','apos']].groupby(['did']).max().reset_index(), on=['did'], how='left')['apos'].values
                        alt_duplications['keep'] = alt_duplications['keep'] | (np.isnan(alt_duplications['trim']) == False)
                alt_duplications = alt_duplications[alt_duplications['keep']].copy()
                ## See if the trimmed version can keep the same number of duplications
                test_dups = alt_duplications[(np.isnan(alt_duplications['trim']) == False)].drop(columns=['keep','capos'])
                if p=='b':
                    if d=='s':
                        test_dups = test_dups[test_dups['bpos'] >= test_dups['trim']].copy()
                    else:
                        test_dups = test_dups[test_dups['bpos'] <= test_dups['trim']].copy()
                else:
                    if d=='s':
                        test_dups = test_dups[test_dups['apos'] >= test_dups['trim']].copy()
                    else:
                        test_dups = test_dups[test_dups['apos'] <= test_dups['trim']].copy()
                test_dups = SolveDuplicationConflicts(test_dups)
                test_count = dup_count.merge(test_dups.groupby(['did']).size().reset_index(name='test'), on=['did'], how='inner')
                accepted = test_count.loc[test_count['test'] == test_count['full'], 'did'].values
                if len(accepted):
                    duplications = pd.concat([ duplications[np.isin(duplications['did'], accepted) == False], test_dups[np.isin(test_dups['did'], accepted)].drop(columns=['trim']) ])
                    if p=='b':
                        if d=='s':
                            alt_duplications = alt_duplications[(alt_duplications['bpos'] >= alt_duplications['trim']) | (np.isin(alt_duplications['did'], accepted) == False)].copy()
                        else:
                            alt_duplications = alt_duplications[(alt_duplications['bpos'] <= alt_duplications['trim']) | (np.isin(alt_duplications['did'], accepted) == False)].copy()
                    else:
                        if d=='s':
                            alt_duplications = alt_duplications[(alt_duplications['apos'] >= alt_duplications['trim']) | (np.isin(alt_duplications['did'], accepted) == False)].copy()
                        else:
                            alt_duplications = alt_duplications[(alt_duplications['apos'] <= alt_duplications['trim']) | (np.isin(alt_duplications['did'], accepted) == False)].copy()
                else:
                    break
#
    # We need to sort again, since we destroyed the order by concatenating (but this time we have only one bpos per apos, which makes it easier)
    duplications.sort_values(['did','apos'], inplace=True)
#
    return duplications

def AssignLowestHaplotypeToMain(duplications):
    # This becomes necessary when the main paths was removed
    min_hap = duplications[['apid','ahap']].drop_duplicates().groupby(['apid']).min().reset_index()
    min_hap = min_hap[min_hap['ahap'] > 0].copy()
    duplications.loc[duplications[['apid','ahap']].merge(min_hap, on=['apid','ahap'], how='left', indicator=True)['_merge'].values == "both", 'ahap'] = 0
    min_hap.rename(columns={'apid':'bpid','ahap':'bhap'}, inplace=True)
    duplications.loc[duplications[['bpid','bhap']].merge(min_hap, on=['bpid','bhap'], how='left', indicator=True)['_merge'].values == "both", 'bhap'] = 0
#
    return duplications

def GetDuplicationDifferences(duplications):
    return duplications.groupby(['group','did','apid','ahap','bpid','bhap'])[['scaf_diff','dist_diff']].sum().reset_index()

def RemoveDuplicatedHaplotypesWithLowestSupport(scaffold_paths, duplications, rem_haps, bsupp, ploidy):
    # Find haplotype with lowest support in each group
    rem_haps = rem_haps.merge(bsupp, on=['group','pid','hap'], how='left')
    rem_haps.sort_values(['group','min','median','max'], ascending=[True,True,True,True], inplace=True)
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

def CompressPaths(scaffold_paths, ploidy):
    # Remove positions with only deletions
    scaffold_paths = scaffold_paths[(scaffold_paths[[f'scaf{h}' for h in range(ploidy)]] >= 0).any(axis=1)].copy()
    scaffold_paths['pos'] = scaffold_paths.groupby(['pid'], sort=False).cumcount()
    scaffold_paths.reset_index(drop=True, inplace=True)
    # Compress paths where we have alternating deletions
    while True:
        shifts = scaffold_paths.loc[ ((np.where(scaffold_paths[[f'phase{h}' for h in range(ploidy)]].values < 0, scaffold_paths[['scaf0' for h in range(ploidy)]].values, scaffold_paths[[f'scaf{h}' for h in range(ploidy)]].values) < 0) |
                                      (np.where(scaffold_paths[[f'phase{h}' for h in range(ploidy)]].shift(1).values < 0, scaffold_paths[['scaf0' for h in range(ploidy)]].shift(1).values, scaffold_paths[[f'scaf{h}' for h in range(ploidy)]].shift(1).values) < 0)).all(axis=1), ['pid'] ]
        if len(shifts) == 0:
            break
        else:
            shifts['index'] = shifts.index.values
            shifts = shifts.groupby(['pid'], sort=False).first() # We can only take the first in each path, because otherwise we might block the optimal solution
            shifts['new_index'] = shifts['index']-2
            while True:
                further = ((np.where(scaffold_paths.loc[shifts['index'].values, [f'phase{h}' for h in range(ploidy)]].values < 0, scaffold_paths.loc[shifts['index'].values, ['scaf0' for h in range(ploidy)]].values, scaffold_paths.loc[shifts['index'].values, [f'scaf{h}' for h in range(ploidy)]].values) < 0) |
                           (np.where(scaffold_paths.loc[shifts['new_index'].values, [f'phase{h}' for h in range(ploidy)]].values < 0, scaffold_paths.loc[shifts['new_index'].values, ['scaf0' for h in range(ploidy)]].values, scaffold_paths.loc[shifts['new_index'].values, [f'scaf{h}' for h in range(ploidy)]].values) < 0)).all(axis=1)
                if np.sum(further) == 0: # We do not need to worry to go into another group, because the first position in every group must be a duplication for all included haplotypes, thus blocks continuation
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
    duplications = GetDuplications(scaffold_paths, ploidy)
    duplications['agroup'] = duplications[['apid']].rename(columns={'apid':'pid'}).merge(ends[['pid','group']], on=['pid'], how='left')['group'].values
    duplications['bgroup'] = duplications[['bpid']].rename(columns={'bpid':'pid'}).merge(ends[['pid','group']], on=['pid'], how='left')['group'].values
    duplications = duplications[duplications['agroup'] == duplications['bgroup']].drop(columns=['bgroup'])
    duplications.rename(columns={'agroup':'group'}, inplace=True)
    duplications['group'] = duplications['group'].astype(int)
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
    duplications['agroup'] = duplications['apid']
    duplications['bgroup'] = duplications['bpid']
    while True:
        matches = duplications.groupby(['agroup','apid','ahap','bgroup','bpid','bhap']).size().reset_index(name='matches')
        for p in ['a','b']:
            matches[f'{p}size'] = matches[[f'{p}group']].merge(matches[[f'{p}group',f'{p}pid',f'{p}hap']].drop_duplicates().groupby([f'{p}group']).size().reset_index(name='size'), on=[f'{p}group'], how='left')['size'].values
        matches = matches[matches['agroup'] != matches['bgroup']].copy()
        groups = matches.groupby(['agroup','bgroup','asize','bsize'])['matches'].agg(['size','min','median','max']).reset_index()
        delete = groups.loc[groups['size'] != groups['asize']*groups['bsize'], ['agroup','bgroup']].copy()
        if len(delete):
            duplications = duplications[duplications[['agroup','bgroup']].merge(delete, on=['agroup','bgroup'], how='left', indicator=True)['_merge'].values == "left_only"].copy()
        groups = groups[groups['size'] == groups['asize']*groups['bsize']].drop(columns=['size','asize','bsize'])
        if len(groups):
            groups.sort_values(['agroup','min','median','max','bgroup'], ascending=[True,False,False,False,True], inplace=True)
            groups = groups.groupby(['agroup'], sort=False).first().reset_index()
            groups.drop(columns=['min','median','max'], inplace=True)
            groups = groups.merge(groups.rename(columns={'agroup':'bgroup','bgroup':'agroup'}), on=['agroup','bgroup'], how='inner')
            groups = groups[groups['agroup'] < groups['bgroup']].rename(columns={'bgroup':'group','agroup':'new_group'})
            for p in ['a','b']:
                duplications['new_group'] = duplications[[f'{p}group']].rename(columns={f'{p}group':'group'}).merge(groups, on=['group'], how='left')['new_group'].values
                duplications.loc[np.isnan(duplications['new_group']) == False, f'{p}group'] = duplications.loc[np.isnan(duplications['new_group']) == False, 'new_group'].astype(int)
        else:
            break
    duplications['group'] = duplications['agroup']
    duplications.drop(columns=['agroup','bgroup','new_group'], inplace=True)
#
    # Get minimum difference to another haplotype in group
    duplications = GetPositionsBeforeDuplication(duplications, scaffold_paths, ploidy, False)
    duplications['scaf_diff'] = (duplications['aprev_pos'] != duplications['apos'].shift(1)) | (duplications['bprev_pos'] != duplications['bpos'].shift(1))
    duplications['dist_diff'] = (duplications['scaf_diff'] == False) & (duplications['adist'] != duplications['bdist'])
    duplications.loc[duplications['apos'] == 0, 'scaf_diff'] = False
    differences = GetDuplicationDifferences(duplications)
#
    # Remove all except one version of haplotypes with no differences
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
    # Get bridge counts for different path/haplotypes to base decisions on it
    bsupp = duplications[['group','apid','ahap']].drop_duplicates()
    bsupp.rename(columns={'apid':'pid','ahap':'hap'}, inplace=True)
    bsupp = GetBridgeSupport(bsupp, scaffold_paths, scaf_bridges, ploidy)
#
    # Remove distance only variants with worst bridge support as long as we are above ploidy haplotypes
    while True:
        groups = duplications.groupby(['group','apid','ahap']).size().groupby(['group']).size().reset_index(name='nhaps')
        rem_groups = groups[groups['nhaps'] > ploidy].copy()
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
        groups = duplications.groupby(['group','apid','ahap']).size().groupby(['group']).size().reset_index(name='nhaps')
        rem_groups = groups[groups['nhaps'] > ploidy].copy()
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
    # Merge all groups that do not have more than ploidy haplotypes
    groups = duplications.groupby(['group','apid','ahap']).size().groupby(['group']).size().reset_index(name='nhaps')
    groups = groups[groups['nhaps'] <= ploidy].copy()
    duplications = duplications[np.isin(duplications['group'], groups['group'].values)].copy()
    # Define insertion order by amount of bridge support (highest support == lowest new haplotype)
    bsupp = bsupp.merge(duplications[['group','apid','ahap']].drop_duplicates().rename(columns={'apid':'pid','ahap':'hap'}), on=['group','pid','hap'], how='inner')
    bsupp.sort_values(['group','min','median','max'], ascending=[True,True,True,True], inplace=True)
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
        group_info = group_info[['pid']+[f'pid{h}' for h in range(ploidy)]].drop_duplicates()
        return scaffold_paths, group_info
    else:
        return scaffold_paths

def PlacePathAInPathB(duplications, scaffold_paths, scaffold_graph, scaf_bridges, ploidy):
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
    test_paths = CombinePathAccordingToMetaParts(test_paths, meta_parts, conns, scaffold_graph, scaf_bridges, scaf_len, ploidy)
    pids = np.unique(test_paths['pid'])
    includes['success'] = (np.isin(includes[['tpid1','tpid2','tpid3']], pids).sum(axis=1) == 1)
    test_paths = test_paths.merge(pd.concat([ includes.loc[includes['success'], [f'tpid{i}','tpid3']].rename(columns={f'tpid{i}':'pid'}) for i in [1,2] ], ignore_index=True), on=['pid'], how='inner')
    test_paths['pid'] = test_paths['tpid3']
    test_paths.drop(columns=['tpid3'], inplace=True)
    test_paths.sort_values(['pid','pos'], inplace=True)
    includes = includes.loc[includes['success'], ['ldid','rdid','apid','bpid','tpid3']].rename(columns={'tpid3':'tpid'})
#
    return test_paths, includes

def PlaceUnambigouslyPlaceablePathsAsAlternativeHaplotypes(scaffold_paths, scaffold_graph, scaf_bridges, ploidy):
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
        test_paths, includes = PlacePathAInPathB(duplications, scaffold_paths, scaffold_graph, scaf_bridges, ploidy)
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
    scaffold_paths.loc[scaffold_paths['pos'] == 0, [f'dist{h}' for h in range(ploidy)]] = 0
    scaffold_paths = TrimAlternativesConsistentWithMain(scaffold_paths, ploidy)
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

def TraverseScaffoldingGraph(scaffolds, scaffold_graph, scaf_bridges, ploidy):
    # Get phased haplotypes
    knots = UnravelKnots(scaffold_graph, scaffolds)
    scaffold_paths = FollowUniquePathsThroughGraph(knots, scaffold_graph)
    # scaffold_paths = AddPathTroughRepeats(scaffold_paths, scaffold_graph)
    scaffold_paths = AddUntraversedConnectedPaths(scaffold_paths, knots, scaffold_graph)
    scaffold_paths = AddUnconnectedPaths(scaffold_paths, scaffolds, scaffold_graph)
#
    # Turn them into full path with ploidy
    scaffold_paths.insert(2, 'phase0', scaffold_paths['pid'].values+1) #+1, because phases must be larger than zero to be able to be positive and negative (negative means identical to main paths at that position)
    for h in range(1,ploidy):
        scaffold_paths[f'phase{h}'] = -scaffold_paths['phase0'].values
        scaffold_paths[f'scaf{h}'] = -1
        scaffold_paths[f'strand{h}'] = ''
        scaffold_paths[f'dist{h}'] = 0
#
    # Combine paths as much as possible
    print("Start")
    print(len(np.unique(scaffold_paths['pid'].values)))
    #scaffold_paths = RemoveDuplicates(scaffold_paths, False, ploidy)
    #print(len(np.unique(scaffold_paths['pid'].values)))
    for i in range(3):
        old_nscaf = 0
        n = 1
        while old_nscaf != len(np.unique(scaffold_paths['pid'].values)):
            print(f"Iteration {n}")
            n+=1
            old_nscaf = len(np.unique(scaffold_paths['pid'].values))
             # First Merge then Combine to not accidentially merge a haplotype, where the other haplotype of the paths is not compatible and thus joining wrong paths
            scaffold_paths = MergeHaplotypes(scaffold_paths, scaf_bridges, ploidy)
            print(len(np.unique(scaffold_paths['pid'].values)))
            scaffold_paths = CombinePathOnUniqueOverlap(scaffold_paths, scaffold_graph, scaf_bridges, ploidy)
            print(len(np.unique(scaffold_paths['pid'].values)))
        if i==0:
            print("RemoveDuplicates")
            scaffold_paths = RemoveDuplicates(scaffold_paths, True, ploidy)
        elif i==1:
            print("PlaceUnambigouslyPlaceables")
            scaffold_paths = PlaceUnambigouslyPlaceablePathsAsAlternativeHaplotypes(scaffold_paths, scaffold_graph, scaf_bridges, ploidy)
        print(len(np.unique(scaffold_paths['pid'].values)))
    print("TrimAmbiguousOverlap")
    scaffold_paths = TrimAmbiguousOverlap(scaffold_paths, scaffold_graph, ploidy)
    print(len(np.unique(scaffold_paths['pid'].values)))
    print("TrimCircularPaths")
    scaffold_paths = TrimCircularPaths(scaffold_paths, ploidy)
    print(len(np.unique(scaffold_paths['pid'].values)))

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
    # Set new continous scaffold ids consistent with original scaffolding and apply reversions
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
    scaffold_paths = TraverseScaffoldingGraph(scaffolds, scaffold_graph, scaf_bridges, ploidy)
    
    # Finish Scaffolding
    scaffold_paths = ExpandScaffoldsWithContigs(scaffold_paths, scaffolds, scaffold_parts, ploidy)
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
        conn_cov = all_scafs.loc[(all_scafs['lpos'] >= 0), ['scaf','lpos','pos','lhap']].drop_duplicates().rename(columns={'pos':'rpos','lhap':'hap'}).sort_values(['scaf','lpos','rpos','hap']).merge(conn_cov, on=['scaf','lpos','rpos','hap'], how='left')
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
        dups_maps = dups_maps.groupby(['read_name','read_start','read_pos','ci']).first().reset_index() # In rare cases a read can map on the forward and reverse strand, chose one of those arbitrarily
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
            #print( "Warning:", len(conn_cov[conn_cov['cov'] == 0]), "gaps were created for which no read for filling can be found. The connections will be broken up again, but this should never happen and something weird is going on.")
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
