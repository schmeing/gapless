#!/usr/bin/env python3
from Bio import SeqIO

from datetime import timedelta
import getopt
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

    return pd.concat(cov_counts)

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

    pot_breaks = pd.concat([left_breaks, right_breaks])
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
        break_supp = pd.concat(break_supp).drop_duplicates()
        break_supp = break_supp[break_supp['position'] != break_supp['supp_pos']].copy() # If we are at the same position we would multicount the values, when we do a cumsum later
        break_supp['position'] = break_supp['supp_pos']
        break_supp.drop(columns=['supp_pos'], inplace=True)

        break_points = pd.concat([break_points, break_supp]).groupby(['contig_id','position','mapq']).sum().reset_index()
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

        break_points = break_points.merge(pd.concat(break_list), on=['contig_id','position','mapq'], how='outer').fillna(0)
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
    unconnected_break_points = pd.concat(unconnected_break_points)[['contig_id','position']].sort_values(['contig_id','position']).drop_duplicates()
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

def UpdateMappingsToContigParts(mappings, contig_parts, min_mapping_length):
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

    # Get distance to connected contigparts
    mappings['next_dist'] = mappings['read_from'].shift(-1, fill_value=0) - mappings['read_to']
    mappings['prev_dist'] = mappings['read_from'] - mappings['read_to'].shift(1, fill_value=0)
    mappings['left_con_dist'] = np.where('+' == mappings['strand'], mappings['prev_dist'], mappings['next_dist'])
    mappings.loc[-1 == mappings['left_con'], 'left_con_dist'] = 0
    mappings['right_con_dist'] = np.where('-' == mappings['strand'], mappings['prev_dist'], mappings['next_dist'])
    mappings.loc[-1 == mappings['right_con'], 'right_con_dist'] = 0

    # Select the columns we still need
    mappings.rename(columns={'q_name':'read_name'}, inplace=True)
    mappings = mappings[['read_name', 'read_start', 'read_end', 'read_from', 'read_to', 'strand', 'conpart', 'con_from', 'con_to', 'left_con', 'left_con_side', 'left_con_dist', 'right_con', 'right_con_side', 'right_con_dist', 'mapq', 'matches']].copy()

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
        long_range_connections = long_range_mappings[['conpart','strand','left_con_dist','right_con_dist']].copy()
        long_range_connections['conn_id'] = ((long_range_mappings['read_name'] != long_range_mappings['read_name'].shift(1, fill_value='')) | (long_range_mappings['read_start'] != long_range_mappings['read_start'].shift(1, fill_value=-1))).cumsum()
    else:
        long_range_connections = []

    if len(long_range_connections):
        # Break long_range_mappings when they go through invalid bridges and get number of prev alternatives (how many alternative to the connection of a mapping with its previous mapping exist)
        long_range_connections['from'] = np.where(long_range_connections['conn_id'] != long_range_connections['conn_id'].shift(1, fill_value=-1), -1, long_range_connections['conpart'].shift(1, fill_value=-1))
        long_range_connections['from_side'] = np.where(long_range_connections['strand'].shift(1, fill_value='') == '+', 'r', 'l') # When the previous entry does not belong to same conn_id this is garbage, but 'from' is already preventing the merge happening next, so it does not matter
        long_range_connections['to'] = long_range_connections['conpart']
        long_range_connections['to_side'] = np.where(long_range_connections['strand'] == '+', 'l', 'r')
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
    scaffold_graph = pd.concat([scaffold_graph, short_bridges[['from','from_side','size','org_pos','scaf1','strand1','dist1']]])

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
    reps = pd.concat(reps)
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

def FindBubblesAndUniquePaths(scaffolds, scaffold_parts, scaffold_graph, ploidy):
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
            path_long_format = pd.concat(path_long_format)
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
            all_paths = all_paths[ (all_paths[['from','from_side']].merge(pd.concat([bubble_candidates[['from','from_side']], bubble_candidates[['to','to_side']].rename(columns={'to':'from','to_side':'from_side'})]).drop_duplicates(), on=['from','from_side'], how='left', indicator=True)['_merge'] == "left_only").values ].copy()
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
                rep_extensions = pd.concat(rep_extensions)
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
    bubbles = pd.concat(bubbles)
    bubbles.sort_values(['from','from_side','to','to_side'], inplace=True)
    bubbles.drop_duplicates(inplace=True)
#
    unique_extensions = pd.concat(unique_extensions)
    unique_extensions.sort_values(['from','from_side'], inplace=True)
#
    # Insert reversed versions of loops
    loops = pd.concat(loops)
    loops['from_side'] = 'r'
    rev_path = loops.copy()
    rev_path['from_side'] = 'l'
    for l in range(1, rev_path['loop_len'].max()+1):
        rev_path.loc[rev_path['loop_len'] == l, 'dist'+str(l+1)] = loops.loc[loops['loop_len'] == l, 'dist1']
        for s in range(1, l+1):
            rev_path.loc[rev_path['loop_len'] == l, 'scaf'+str(s)] = loops.loc[loops['loop_len'] == l, 'scaf'+str(l+1-s)]
            rev_path.loc[rev_path['loop_len'] == l, 'strand'+str(s)] = np.where(loops.loc[loops['loop_len'] == l, 'strand'+str(l+1-s)] == '+', '-', '+')
            rev_path.loc[rev_path['loop_len'] == l, 'dist'+str(s)] = loops.loc[loops['loop_len'] == l, 'dist'+str(l+2-s)]
    loops = pd.concat([loops, rev_path])
    loops.sort_values(['from','from_side'], inplace=True)
    loops.drop_duplicates(inplace=True)
#
    return bubbles, loops, unique_extensions
    
def GroupConnectedScaffoldsIntoKnots(bubbles, loops, unique_extensions, scaffolds, scaffold_parts, contig_parts):
    # Create copies, so that we don't modify the original
    cbubbles = bubbles.copy()
    cloops = loops.copy()
    cunique_extensions = unique_extensions.copy()

    # Group all connected scaffolds
    conns = []
    if len(cbubbles):
        cbubbles['id'] = range(len(cbubbles))
        conns.append(cbubbles[['id', 'from']].rename(columns={'id':'conn', 'from':'scaffold'}))
        conns.append(cbubbles[['id', 'to']].rename(columns={'id':'conn', 'to':'scaffold'}))
        for s in range(1,cbubbles['length'].max()+1):
            conns.append(cbubbles.loc[np.isnan(cbubbles['scaf'+str(s)]) == False, ['id', 'scaf'+str(s)]].rename(columns={'id':'conn', ('scaf'+str(s)):'scaffold'}))
            conns[-1]['scaffold'] = conns[-1]['scaffold'].astype(int)

    if len(cloops):
        cloops['id'] = range(len(cbubbles), len(cbubbles)+len(cloops))
        conns.append(cloops[['id', 'from']].rename(columns={'id':'conn', 'from':'scaffold'}))
        for s in range(1,cloops['loop_len'].max()+1):
            conns.append(cloops.loc[np.isnan(cloops['scaf'+str(s)]) == False, ['id', 'scaf'+str(s)]].rename(columns={'id':'conn', ('scaf'+str(s)):'scaffold'}))
            conns[-1]['scaffold'] = conns[-1]['scaffold'].astype(int)

    if len(cunique_extensions):
        cunique_extensions['id'] = range(len(cbubbles)+len(cloops), len(cbubbles)+len(cloops)+len(cunique_extensions))
        conns.append(cunique_extensions[['id', 'from']].rename(columns={'id':'conn', 'from':'scaffold'}))
        for s in range(1,cunique_extensions['unique_len'].max()+1):
            conns.append(cunique_extensions.loc[np.isnan(cunique_extensions['scaf'+str(s)]) == False, ['id', 'scaf'+str(s)]].rename(columns={'id':'conn', ('scaf'+str(s)):'scaffold'}))
            conns[-1]['scaffold'] = conns[-1]['scaffold'].astype(int)

    if len(conns) == 0:
        knots = []
    else:
        conns = pd.concat(conns)
        conns.sort_values(['conn'], inplace=True)
        knots = scaffolds[['scaffold']].copy()
        knots['knot'] = 0
        knots['new_knot'] = range(len(knots))
        while np.sum((np.isnan(knots['new_knot']) == False) & (knots['knot'] != knots['new_knot'])):
            knots.loc[np.isnan(knots['new_knot']) == False, 'knot'] = knots.loc[np.isnan(knots['new_knot']) == False, 'new_knot'].astype(int)
            knots.drop(columns=['new_knot'], inplace=True)
            tmp_conns = conns.merge(knots, on=['scaffold'], how='left')
            min_knot = tmp_conns.groupby(['conn'], sort=False)['knot'].agg(['min','size'])
            tmp_conns['knot'] = np.repeat(min_knot['min'].values, min_knot['size'].values)
            knots = knots.merge(tmp_conns.groupby(['scaffold'])['knot'].min().reset_index(name='new_knot'), on=['scaffold'], how='left')

        # Remove all knots that only contain a single scaffold
        knots.drop(columns=['new_knot'], inplace=True)
        knots.sort_values(['knot'], inplace=True)
        knot_size = knots.groupby(['knot'], sort=False).size()
        knots = knots[np.repeat(knot_size.values > 1, knot_size.values)].copy()

        # Get scaffold sizes (in nucleotides)
        conlen = pd.DataFrame({'conpart':range(len(contig_parts)), 'length':(contig_parts['end'].values - contig_parts['start'].values)})
        knots = knots.merge(scaffold_parts[['conpart','scaffold']].merge(conlen.rename(columns={'contig':'conpart'}), on=['conpart'], how='left').groupby(['scaffold'])['length'].sum().reset_index(name='scaf_len'), on=['scaffold'], how='left')
        knots.sort_values(['knot','scaf_len'], ascending=[True,False], inplace=True)

    return knots

def ResolveLoops(loops, unique_paths, bubbles, unique_extensions):
    ## Check if we can traverse loops and update unique_paths, bubbles and unique_extensions accordingly
    # Group connected loops
    conns = []
    conns.append(loops[['id', 'from']].rename(columns={'id':'conn', 'from':'scaffold'}))
    for s in range(1,loops['loop_len'].max()+1):
        conns.append(loops.loc[np.isnan(loops['scaf'+str(s)]) == False, ['id', 'scaf'+str(s)]].rename(columns={'id':'conn', ('scaf'+str(s)):'scaffold'}))
        conns[-1]['scaffold'] = conns[-1]['scaffold'].astype(int)
    conns = pd.concat(conns).drop_duplicates()
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
    unique_loop_exit = pd.concat(unique_loop_exit).drop(columns=['scaffold']).sort_values(['loop','from','from_side','to','to_side']).drop_duplicates()
    loops_with_unique_exit = np.unique(unique_loop_exit['loop'].values)
        
    bubble_loop_exit = [ loop_scaffolds.merge(bubbles.reset_index(), left_on=['scaffold'], right_on=['from'], how='inner') ]
    for s in range(1,bubbles['length'].max()+1):
        bubble_loop_exit.append( loop_scaffolds.merge(bubbles.reset_index(), left_on=['scaffold'], right_on=['scaf'+str(s)], how='inner') )
    bubble_loop_exit = pd.concat(bubble_loop_exit).drop(columns=['scaffold']).sort_values(['loop','from','from_side','to','to_side']).drop_duplicates()
    loops_with_bubble_exit = np.unique(bubble_loop_exit['loop'].values)
    
    unique_loop_exit['has_bubble'] = np.isin(unique_loop_exit['loop'], loops_with_bubble_exit)
    unique_loop_exit['from_in_loop']  = (unique_loop_exit[['loop','from']].merge(loop_scaffolds.rename(columns={'scaffold':'from'}), on=['loop','from'], how='left', indicator=True)['_merge'] == "both").values
    unique_loop_exit['to_in_loop']  = (unique_loop_exit[['loop','to']].merge(loop_scaffolds.rename(columns={'scaffold':'to'}), on=['loop','to'], how='left', indicator=True)['_merge'] == "both").values
    
    return unique_paths, bubbles, unique_extensions

def TraverseScaffoldingGraph(scaffolds, scaffold_parts, scaffold_graph, contig_parts, scaf_bridges, ploidy):
    bubbles, loops, unique_extensions = FindBubblesAndUniquePaths(scaffolds, scaffold_parts, scaffold_graph, ploidy)
    knots = GroupConnectedScaffoldsIntoKnots(bubbles, loops, unique_extensions, scaffolds, scaffold_parts, contig_parts)
#
    # Separate some stuff from bubbles
    inversions = bubbles[bubbles['from'] == bubbles['to']].copy()
    bubbles = bubbles[bubbles['from'] != bubbles['to']].copy()
    unique_paths = bubbles[1 == bubbles['alts']].drop(columns=['alts'])
    repeat_bubbles = bubbles[bubbles['alts'] > ploidy].copy()
    bubbles = bubbles[(1 < bubbles['alts']) & (bubbles['alts'] <= ploidy)].copy()
#
    #unique_paths, bubbles, unique_extensions = ResolveLoops(loops, unique_paths, bubbles, unique_extensions)
    # Do here: Handle unique extensions
#
    # Start from the longest scaffold in each knot and extend it to both sides as much as possible and repeat this until all scaffolds have been addressed
    final_paths = []
    final_alternatives = []
    duplicated_scaffolds = []
    while len(knots):
        cur_start = knots.groupby(['knot']).first().reset_index()
        cur_paths = pd.DataFrame({'center':cur_start['scaffold'].values, 'pos':0, 'scaffold':cur_start['scaffold'].values, 'strand':'+', 'distance':0})
        cur_alters = []
        included_scaffolds = []
#
        # Extend to both sides
        for direction in ["left","right"]:
            ext = pd.DataFrame({'center':cur_start['scaffold'].values, 'pos':0, 'len':1})
            while len(ext):
                new_paths = [cur_paths.copy()]
                if len(cur_alters):
                    new_alters = [cur_alters]
                else:
                    new_alters = []
                if direction == "right":
                    ext = ext.merge(cur_paths, on=['center','pos'], how='left')
                    ext['side'] = np.where(ext['strand'] == '+', 'r', 'l')
                else:
                    cur_paths['rev_strand'] = np.where(cur_paths['strand'] == '+', '-', '+')
                    cur_paths['rev_pos'] = cur_paths['pos'] * (-1)
                    ext = ext.merge(cur_paths[['center', 'rev_pos', 'scaffold', 'strand', 'distance']].rename(columns={'rev_pos':'pos'}), on=['center','pos'], how='left')
                    ext['side'] = np.where(ext['strand'] == '+', 'l', 'r')
#
                ## Check if we can extend with a unique path
                uni_ext = ext.merge(unique_paths.rename(columns={'from':'scaffold', 'from_side':'side'}), on=['scaffold','side'], how='inner')
                uni_ext['new_len'] = uni_ext['pos'] + uni_ext['length'] + 2
                uni_ext = uni_ext[ uni_ext['new_len'] > uni_ext['len'] ].copy() # Remove extensions that are shorter or equally long as the current path (don't extend)
                if len(uni_ext):
                    # Check if the extension is already included (so that we do not get stuck in a loop)
                    if direction == "right":
                        uni_ext['to_strand'] = np.where(uni_ext['to_side'] == 'l', '+', '-')
                    else:
                        uni_ext['to_strand'] = np.where(uni_ext['to_side'] == 'l', '-', '+')
                    uni_ext['included'] = (uni_ext[['center','to','to_strand']].rename(columns={'to':'scaffold', 'to_strand':'strand'}).merge(cur_paths[['center','scaffold','strand']], on=['center','scaffold','strand'], how='left', indicator=True).drop_duplicates()['_merge'] == "both").values
                    for s in range(1,uni_ext['length'].max()+1):
                        uni_ext.loc[uni_ext['included'] & (uni_ext['length'] >= s), 'included'] = (uni_ext.loc[uni_ext['included'] & (uni_ext['length'] >= s), ['center','scaf'+str(s),'strand'+str(s)]].rename(columns={('scaf'+str(s)):'scaffold', ('strand'+str(s)):'strand'}).merge( (cur_paths[['center','scaffold','strand']] if direction == "right" else cur_paths[['center','scaffold','rev_strand']].rename(columns={'rev_strand':'strand'})), on=['center','scaffold','strand'], how='left', indicator=True).drop_duplicates()['_merge'] == "both").values
                    uni_ext = uni_ext[ uni_ext['included'] == False ].copy()
                if len(uni_ext):
                    # Add newly included scaffolds to included_scaffolds
                    included_scaffolds.append(uni_ext['to'].values)
                    for s in range(1,uni_ext['length'].max()+1):
                        included_scaffolds.append(uni_ext.loc[uni_ext['length'] >= s, 'scaf'+str(s)].values.astype(int))
                    # Insert new scaffolds into path
                    uni_ext['pos'] += 1
                    for s in range(1,uni_ext['length'].max()+1):
                        new_paths.append(uni_ext.loc[(uni_ext['length'] >= s) & (uni_ext['pos'] >= uni_ext['len']), ['center','pos','scaf'+str(s),'strand'+str(s),'dist'+str(s)]].rename(columns={('scaf'+str(s)):'scaffold', ('strand'+str(s)):'strand', ('dist'+str(s)):'distance'}))
                        new_paths[-1]['scaffold'] = new_paths[-1]['scaffold'].astype(int)
                        new_paths[-1]['distance'] = new_paths[-1]['distance'].astype(int)
                        if direction == "left":
                            new_paths[-1]['strand'] = np.where(new_paths[-1]['strand'] == '+', '-', '+')
                            new_paths[-1]['pos'] = new_paths[-1]['pos'] * (-1)
                        uni_ext['pos'] += 1
                    uni_ext['pos'] = uni_ext['new_len']-1
                    new_paths.append(uni_ext[['center','pos','to','to_strand','last_dist']].rename(columns={'to':'scaffold', 'to_strand':'strand', 'last_dist':'distance'}))
                    if direction == "left":
                        new_paths[-1]['pos'] = new_paths[-1]['pos'] * (-1)
                    ext = ext.merge(uni_ext[['center','new_len']], on=['center'], how='left')
                    ext.loc[np.isnan(ext['new_len']) == False, 'len'] = ext.loc[np.isnan(ext['new_len']) == False, 'new_len'].astype(int)
                    ext.drop(columns=['new_len'], inplace=True)
#
                ## Check if we can extend with a bubble
                bext = ext.merge(bubbles.rename(columns={'from':'scaffold', 'from_side':'side'}), on=['scaffold','side'], how='inner')
                if len(bext):
                    # Add all scaffolds from the bubbles to included_scaffolds (also if we don't include them, because then they are another haplotype from the bubble (or a repeat, but in that case we should include them coming from a different scaffold))
                    included_scaffolds.append(bext['to'].values)
                    for s in range(1,bext['length'].max()+1):
                        included_scaffolds.append(bext.loc[bext['length'] >= s, 'scaf'+str(s)].values.astype(int))
                    # Remove bubbles that do not fit the current path
                    bext['new_len'] = bext['pos'] + bext['length'] + 2
                    bext = bext[ bext['new_len'] > bext['len'] ].copy() # Remove extensions that are shorter or equally long as the current path (don't extend)
                    if len(bext):
                        bext['cur_pos'] = bext['pos'] + 1
                        if direction == "right":
                            bext['to_strand'] = np.where(bext['to_side'] == 'l', '+', '-')
                        else:
                            bext['to_strand'] = np.where(bext['to_side'] == 'l', '-', '+')
                        alt_ext = [] # Before removing bubbles that do not fit the current path store them to check if they are from an alternative path
                        for s in range(1,bext['length'].max()+1):
                            bext = bext.merge(cur_paths[['center', ('pos' if direction == "right" else 'rev_pos'), 'scaffold', ('strand' if direction == "right" else 'rev_strand'), 'distance']].rename(columns={('pos' if direction == "right" else 'rev_pos'):'cur_pos', 'scaffold':'pscaf', ('strand' if direction == "right" else 'rev_strand'):'pstrand', 'distance':'pdist'}), on=['center','cur_pos'], how='left')
                            alt_ext.append( bext[(np.isnan(bext['pscaf']) == False) & ((bext['pscaf'] != bext['scaf'+str(s)]) | (bext['pstrand'] != bext['strand'+str(s)]) | (bext['pdist'] != bext['dist'+str(s)]))].drop(columns=['pscaf','pstrand','pdist']) )
                            bext = bext[np.isnan(bext['pscaf']) | ((bext['pscaf'] == bext['scaf'+str(s)]) & (bext['pstrand'] == bext['strand'+str(s)]) & (bext['pdist'] == bext['dist'+str(s)]))].copy()
                            bext.drop(columns=['pscaf','pstrand','pdist'], inplace=True)
                            bext['cur_pos'] += 1
                        bext['cur_pos'] = bext['new_len']-1
                        bext = bext.merge(cur_paths[['center', ('pos' if direction == "right" else 'rev_pos'), 'scaffold', 'strand', 'distance']].rename(columns={('pos' if direction == "right" else 'rev_pos'):'cur_pos', 'scaffold':'pscaf', 'strand':'pstrand', 'distance':'pdist'}), on=['center','cur_pos'], how='left')
                        alt_ext.append( bext[(np.isnan(bext['pscaf']) == False) & ((bext['pscaf'] != bext['to']) | (bext['pstrand'] != bext['to_strand']) | (bext['pdist'] != bext['last_dist']))].drop(columns=['pscaf','pstrand','pdist']) )
                        bext = bext[np.isnan(bext['pscaf']) | ((bext['pscaf'] == bext['to']) & (bext['pstrand'] == bext['to_strand']) & (bext['pdist'] == bext['last_dist']))].copy() # 'to_strand' we do not need to reverse, because we directly created it correctly based on direction
                        bext.drop(columns=['pscaf','pstrand','pdist','cur_pos'], inplace=True)
                        if 0 == len(cur_alters):
                            alt_ext = []
                        else:
                            # Remove alternatives that do not belong to one that is consistent with the main path
                            alt_ext = pd.concat(alt_ext)
                            alt_ext = alt_ext.merge(bext[['center']].drop_duplicates(), on=['center'], how='inner')
                            # Remove alternatives where we are currently in a bubble or a bubble comes afterwards
                            alt_ext = alt_ext.merge(cur_alters.rename(columns={'scaffold':'alt_scaf','strand':'alt_strand','distance':'alt_dist'}), on=['center'], how='inner') # If there is no previous alternative it cannot connect to one
                            if len(alt_ext) and direction == "right":
                                # Ignore the alternatives from the other direction
                                alt_ext = alt_ext[alt_ext['end_pos'] > 0].copy()
                            if len(alt_ext):
                                alt_ext.sort_values(['center','alt_id','alt_pos'], inplace=True)
                                last_alt = alt_ext.groupby(['center'], sort=False)['start_pos'].agg(['min' if direction == "left" else 'max', 'size'])
                                alt_ext['last_alt'] = np.repeat(last_alt['min' if direction == "left" else 'max'].values, last_alt['size'].values)
                                alt_ext = alt_ext[alt_ext['start_pos'] == alt_ext['last_alt']].drop(columns=['last_alt'])
                                alt_ext = alt_ext[alt_ext['pos'] >= np.abs(alt_ext['end_pos'])-1].copy()
                            if len(alt_ext):
                                alt_ext['alt_dist'] = alt_ext['alt_dist'].shift(1 if direction == "left" else -1, fill_value=0)
                                # Store alternatives that cannot connect (because they are a distance variant or a deletion or they are supposed to connect to a deletion)
                                alt_size = alt_ext.groupby(['center'], sort=False).size().values
                                alt_ext['alt_size'] = np.repeat(alt_size, alt_size)
                                non_connectable = (0 == alt_ext['length']) | ((1 == alt_ext['alt_size']) & (1 < np.abs(alt_ext['end_pos']-alt_ext['start_pos'])))
                                accepted_alts = alt_ext[non_connectable].copy()
                                accepted_alts['alt_id'] = np.nan # Remove alternative id, because we cannot phase it without connection
                                alt_ext = alt_ext[(False == non_connectable) & (np.abs(alt_ext['alt_pos'])+1 != alt_ext['alt_size'])].copy() # Additionally remove the 'to' scaffold from the alternatives, because it is also part of the main path
                                # Remove alternatives that do not connect with a previous alternative (the main connects to the previous main otherwise it would be handled in the alt_ext cases in the next section)
                                if len(alt_ext):
                                    alt_path = []
                                    for s in range(1, (alt_ext['cur_pos']-alt_ext['pos']).max()+1):
                                        alt_path.append( alt_ext.loc[alt_ext['cur_pos']-alt_ext['pos'] >= s, ['scaf'+str(s),'strand'+str(s),'alt_scaf','alt_strand','alt_dist']].rename(columns={('scaf'+str(s)):'from', ('strand'+str(s)):'from_strand', 'alt_scaf':'to', 'alt_strand':'to_strand','alt_dist':'last_dist'}).reset_index() )
                                        alt_path[-1]['bpos'] = s
                                    alt_path = pd.concat(alt_path)
                                    alt_path['from'] = alt_path['from'].astype(int)
                                    alt_path['from_side'] = np.where(alt_path['from_strand'] == '+', 'l', 'r')
                                    if direction == "left": 
                                        alt_path['to_side'] = np.where(alt_path['to_strand'] == '+', 'l', 'r')
                                    else:
                                        alt_path['to_side'] = np.where(alt_path['to_strand'] == '+', 'r', 'l')
                                    alt_path.drop(columns=['from_strand','to_strand'], inplace=True)
                                    alt_path = pd.concat([alt_path.merge(unique_paths, on=['from','from_side','to','to_side','last_dist'], how='inner'), alt_path.merge(bubbles.drop(columns=['alts']), on=['from','from_side','to','to_side','last_dist'], how='inner')])
                                    if len(alt_path):
                                        ## Check that the paths are consistent
                                        # First check that the paths have the right length
                                        alt_path = alt_path[ alt_path['length']+2 == alt_path['bpos'] + (alt_ext.loc[alt_path['index'].values, 'pos'].values - np.abs(alt_ext.loc[alt_path['index'].values, 'end_pos'].values) + 1) + (alt_ext.loc[alt_path['index'].values, 'alt_size'].values - np.abs(alt_ext.loc[alt_path['index'].values, 'alt_pos'].values))].copy()
                                        # Then check all the scaffolds within the current bubble before the connecting scaffold
                                        for cbpos in range(alt_path['bpos'].min(), alt_path['bpos'].max()+1):
                                            # The last one in the bubble was already checked by merging except for the distance
                                            alt_path = alt_path[ (alt_path['bpos'] != cbpos) | ( alt_path['dist1'].values == alt_ext.loc[alt_path['index'].values, 'dist'+str(cbpos)].values) ].copy()
                                        alt_path['bpos'] -= 1
                                        alt_path['check'] = 1 # The next scaffold to check
                                        cbpos = alt_path['bpos'].max()
                                        while 0 < cbpos:
                                            for ccheck in range(alt_path.loc[alt_path['bpos'] == cbpos, 'check'].min(), alt_path.loc[alt_path['bpos'] == cbpos, 'check'].max()+1):
                                                alt_path = alt_path[ (alt_path['bpos'] != cbpos) | (alt_path['check'] != ccheck) |
                                                                     ( (alt_path['scaf'+str(ccheck)].values == alt_ext.loc[alt_path['index'].values, 'scaf'+str(cbpos)].values) &
                                                                       (alt_path['strand'+str(ccheck)].values != alt_ext.loc[alt_path['index'].values, 'strand'+str(cbpos)].values) & # Strand must be different because we go in different directions
                                                                       (alt_path['dist'+str(ccheck+1)].values == alt_ext.loc[alt_path['index'].values, 'dist'+str(cbpos)].values) ) ].copy() # We have to take the next distance because we go in different directions
                                            alt_path.loc[alt_path['bpos'] == cbpos, 'check'] += 1
                                            alt_path.loc[alt_path['bpos'] == cbpos, 'bpos'] -= 1
                                            cbpos = alt_path['bpos'].max()
                                        # Check all scaffolds on the main path in between the current bubble and the connected alternative path
                                        if len(alt_path):
                                            alt_path.rename(columns={'bpos':'cpos'}, inplace=True)
                                            alt_path['cpos'] = alt_ext.loc[alt_path['index'].values, 'pos'].values # Start of the bubble
                                            alt_path['apos'] = np.abs(alt_ext.loc[alt_path['index'].values, 'end_pos'].values)-1 # Last position of the alternative path
                                            alt_path['center'] = alt_ext.loc[alt_path['index'].values, 'center'].values
                                            tmp_path = cur_paths.merge(alt_path.loc[alt_path['cpos'] > alt_path['apos'], ['center']], on=['center'], how='inner')
                                            if direction == "left":
                                                tmp_path.drop(columns=['pos','strand'], inplace=True)
                                                tmp_path.rename(columns={'rev_pos':'cpos', 'scaffold':'cscaf', 'rev_strand':'cstrand', 'distance':'cdist'}, inplace=True)
                                            else:
                                                tmp_path.rename(columns={'pos':'cpos', 'scaffold':'cscaf', 'strand':'cstrand', 'distance':'cdist'}, inplace=True)
                                            tmp_path = tmp_path[tmp_path['cpos'] > 1].copy() # Position 0 cannot have an alternative, thus pos 1 is the first possible alternative and therefore we never compare it to the main path
                                            while len(tmp_path):
                                                alt_path = alt_path.merge(tmp_path, on=['center','cpos'], how='left')
                                                for ccheck in range(alt_path.loc[alt_path['cpos'] > alt_path['apos'], 'check'].min(), alt_path.loc[alt_path['cpos'] > alt_path['apos'], 'check'].max()+1):
                                                    alt_path = alt_path[ (alt_path['cpos'] <= alt_path['apos']) | (alt_path['check'] != ccheck) |
                                                                         ( (alt_path['scaf'+str(ccheck)] == alt_path['cscaf']) &
                                                                           (alt_path['strand'+str(ccheck)] != alt_path['cstrand']) & # Strand must be different because we go in different directions
                                                                           (alt_path['dist'+str(ccheck+1)].values == alt_path['cdist']) ) ].copy() # We have to take the next distance because we go in different directions
                                                alt_path.loc[alt_path['cpos'] > alt_path['apos'], 'check'] += 1
                                                alt_path.loc[alt_path['cpos'] > alt_path['apos'], 'cpos'] -= 1
                                                alt_path.drop(columns=['cscaf','cstrand','cdist'], inplace=True)
                                                tmp_path = tmp_path.merge(alt_ext.loc[alt_path[alt_path['cpos'] > alt_path['apos']].index.values, ['center']], on=['center'], how='inner')
                                        # Check all scaffolds in the connected alternative after the connected scaffold
                                        if len(alt_path):
                                            alt_path.rename(columns={'cpos':'epos'}, inplace=True)
                                            alt_path['epos'] = np.abs(alt_ext.loc[alt_path['index'].values, 'alt_pos'].values) # Connected position on the alternative path (end position: Has already been tested by the original merge)
                                            alt_path['apos'] = alt_ext.loc[alt_path['index'].values, 'alt_size'].values - 1 # Last position of the alternative path
                                            alt_path['alt_id'] = alt_ext.loc[alt_path['index'].values, 'alt_id'].values
                                            alt_path['dist'+str(alt_path['length'].max()+1)] = np.nan # Make sure this dist exists, because we might read one over the existing ones for distances
                                            tmp_alts = cur_alters.merge(alt_path.loc[alt_path['apos'] > alt_path['epos'], ['center','alt_id']], on=['center','alt_id'], how='inner')
                                            tmp_alts.drop(columns=['start_pos','end_pos'], inplace=True)
                                            tmp_alts.rename(columns={'alt_pos':'apos', 'scaffold':'cscaf', 'strand':'cstrand', 'distance':'cdist'}, inplace=True)
                                            tmp_alts['apos'] = np.abs(tmp_alts['apos'])
                                            if direction == "left":
                                                tmp_alts['cstrand'] = np.where(tmp_alts['cstrand'] == '+', '-', '+')
                                            while len(tmp_alts):
                                                alt_path = alt_path.merge(tmp_alts, on=['center','alt_id','apos'], how='left')
                                                for ccheck in range(alt_path.loc[alt_path['apos'] > alt_path['epos'], 'check'].min(), alt_path.loc[alt_path['apos'] > alt_path['epos'], 'check'].max()+1):
                                                    alt_path = alt_path[ (alt_path['apos'] <= alt_path['epos']) | (alt_path['check'] != ccheck) |
                                                                         ( (alt_path['scaf'+str(ccheck)] == alt_path['cscaf']) &
                                                                           (alt_path['strand'+str(ccheck)] != alt_path['cstrand']) & # Strand must be different because we go in different directions
                                                                           ((alt_path['dist'+str(ccheck+1)].values == alt_path['cdist']) | (1 == alt_path['apos'] - alt_path['epos'])) ) ].copy() # We have to take the next distance because we go in different directions and if we check the last scaffold the distance is stored in last_dist, which we already checked
                                                alt_path.loc[alt_path['apos'] > alt_path['epos'], 'check'] += 1
                                                alt_path.loc[alt_path['apos'] > alt_path['epos'], 'apos'] -= 1
                                                alt_path.drop(columns=['cscaf','cstrand','cdist'], inplace=True)
                                                tmp_alts = tmp_alts.merge(alt_path.loc[alt_path['apos'] > alt_path['epos'], ['center','alt_id']], on=['center','alt_id'], how='inner')
                                    # Only keep extensions that still have a connecting path
                                    if len(alt_path):
                                        alt_ext = alt_ext.loc[np.unique(alt_path['index'].values)]
                                    else:
                                        alt_ext = []
                                # Recombine with the non_connectable
                                if len(alt_ext):
                                    alt_ext = pd.concat([alt_ext, accepted_alts])
                                else:
                                    alt_ext = accepted_alts
                            # Prepare alt_ext for the next section
                            if len(alt_ext):
                                alt_ext = [alt_ext.drop(columns=['cur_pos','start_pos','end_pos','alt_pos','alt_scaf','alt_strand','alt_dist','alt_size'])]
                            else:
                                alt_ext = []
                if len(bext):
                    # From all consistent paths through a bubble pick the one with more reads supporting the bridge as main path
                    bext['first'] = np.where(bext['length'] == 0, bext['to'], bext['scaf1']).astype(int)
                    bext['first_side'] = np.where(bext['length'] == 0, bext['to_side'], np.where(bext['strand1'] == '+', 'l', 'r'))
                    bext['first_dist'] = np.where(bext['length'] == 0, bext['last_dist'], bext['dist1']).astype(int)
                    bext = bext.merge(scaf_bridges[['from','from_side','to','to_side','mean_dist','bcount']].rename(columns={'from':'scaffold', 'from_side':'side', 'to':'first', 'to_side':'first_side', 'mean_dist':'first_dist'}), on=['scaffold','side','first','first_side','first_dist'], how='left')
                    bext.sort_values(['center'], inplace=True)
                    bcounts = bext.groupby(['center'], sort=False)['bcount'].agg(['max','size'])
                    alt_ext.append(bext[bext['bcount'] != np.repeat(bcounts['max'].values, bcounts['size'].values)].drop(columns=['first','first_side','first_dist','bcount']))
                    bext = bext[bext['bcount'] == np.repeat(bcounts['max'].values, bcounts['size'].values)].copy()
                    bext.drop(columns=['first','first_side','first_dist','bcount'], inplace=True)
                    bext['last'] = bext['scaffold']
                    bext['last_side'] = bext['side']
                    for s in range(1,bext['length'].max()+1):
                        bext.loc[bext['length'] == s, 'last'] = bext.loc[bext['length'] == s, 'scaf'+str(s)].astype(int)
                        bext.loc[bext['length'] == s, 'last_side'] = np.where(bext.loc[bext['length'] == s, 'strand'+str(s)] == '+', 'r', 'l')
                    bext = bext.merge(scaf_bridges[['from','from_side','to','to_side','mean_dist','bcount']].rename(columns={'from':'last', 'from_side':'last_side', 'mean_dist':'last_dist'}), on=['last','last_side','to','to_side','last_dist'], how='left')
                    bcounts = bext.groupby(['center'], sort=False)['bcount'].agg(['max','size'])
                    alt_ext.append(bext[bext['bcount'] != np.repeat(bcounts['max'].values, bcounts['size'].values)].drop(columns=['last','last_side','bcount']))
                    bext = bext[bext['bcount'] == np.repeat(bcounts['max'].values, bcounts['size'].values)].copy()
                    bext.drop(columns=['last','last_side','bcount'], inplace=True)
                    # If bridges are equal take the ones including more scaffolds
                    max_len = bext.groupby(['center'], sort=False)['length'].agg(['max','size'])
                    alt_ext.append(bext[bext['length'] != np.repeat(max_len['max'].values, max_len['size'].values)].copy())
                    bext = bext[bext['length'] == np.repeat(max_len['max'].values, max_len['size'].values)].copy()
                    # If there are still multiple options just take the first
                    bext['option'] = bext.groupby(['center'], sort=False).cumcount()
                    alt_ext.append(bext[0 < bext['option']].drop(columns=['option']))
                    bext = bext[0 == bext['option']].drop(columns=['option'])
                    # Check if the extension is already included (so that we do not get stuck in a loop)
                    bext['included'] = (bext[['center','to','to_strand']].rename(columns={'to':'scaffold', 'to_strand':'strand'}).merge(cur_paths[['center','scaffold','strand']], on=['center','scaffold','strand'], how='left', indicator=True).drop_duplicates()['_merge'] == "both").values
                    for s in range(1,bext['length'].max()+1):
                        bext.loc[bext['included'] & (bext['length'] >= s), 'included'] = (bext.loc[bext['included'] & (bext['length'] >= s), ['center','scaf'+str(s),'strand'+str(s)]].rename(columns={('scaf'+str(s)):'scaffold', ('strand'+str(s)):'strand'}).merge((cur_paths[['center','scaffold','strand']] if direction == "right" else cur_paths[['center','scaffold','rev_strand']].rename(columns={'rev_strand':'strand'})), on=['center','scaffold','strand'], how='left', indicator=True).drop_duplicates()['_merge'] == "both").values
                    bext = bext[ bext['included'] == False ].copy()
                    # Join alternative extensions
                    alt_ext = pd.concat(alt_ext).sort_values(['center'])
                    if len(alt_ext):
                        alt_ext = alt_ext[np.isin(alt_ext['center'], bext['center'].values)].copy()
                if len(bext):
                    # Insert new scaffolds into path
                    bext['pos'] += 1
                    for s in range(1,bext['length'].max()+1):
                        new_paths.append(bext.loc[(bext['length'] >= s) & (bext['pos'] >= bext['len']), ['center','pos','scaf'+str(s),'strand'+str(s),'dist'+str(s)]].rename(columns={('scaf'+str(s)):'scaffold', ('strand'+str(s)):'strand', ('dist'+str(s)):'distance'}))
                        new_paths[-1]['scaffold'] = new_paths[-1]['scaffold'].astype(int)
                        new_paths[-1]['distance'] = new_paths[-1]['distance'].astype(int)
                        if direction == "left":
                            new_paths[-1]['strand'] = np.where(new_paths[-1]['strand'] == '+', '-', '+')
                            new_paths[-1]['pos'] *= -1
                        bext['pos'] += 1
                    bext['pos'] = bext['new_len']-1
                    new_paths.append(bext[['center','pos','to','to_strand','last_dist']].rename(columns={'to':'scaffold', 'to_strand':'strand', 'last_dist':'distance'}))
                    if direction == "left":
                        new_paths[-1]['pos'] *= -1
                    ext = ext.merge(bext[['center','new_len']], on=['center'], how='left')
                    ext.loc[np.isnan(ext['new_len']) == False, 'len'] = ext.loc[np.isnan(ext['new_len']) == False, 'new_len'].astype(int)
                    ext.drop(columns=['new_len'], inplace=True)
                    # Handle alternatives (must be in the "if len(bext)", because otherwise alt_ext is not well-defined)
                    if len(alt_ext):
                        # Assign alt_id, for the ones that don't have one yet. Keeps track of phased bubbles and separates bubbles if we have multiple alternatives (polyploid case)
                        if 0 == len(cur_alters):
                            if 0 == len(final_alternatives):
                                alt_ext['alt_id'] = np.arange(0,len(alt_ext))
                            else:
                                alt_ext['alt_id'] = np.arange(0,len(alt_ext)) + final_alternatives[-1]['alt_id'].max()+1
                        else:
                            if 'alt_id' not in list(alt_ext.columns):
                                alt_ext['alt_id'] = np.arange(0,len(alt_ext)) + cur_alters['alt_id'].max()+1
                            else:
                                alt_ext.loc[np.isnan(alt_ext['alt_id']), 'alt_id'] = np.arange(0,np.sum(np.isnan(alt_ext['alt_id']))) + cur_alters['alt_id'].max()+1
                                alt_ext['alt_id'] = alt_ext['alt_id'].astype(int)
                        # Insert alternative paths
                        alt_ext = alt_ext.merge(bext[['center','pos']].rename(columns={'pos':'end_pos'}), on=['center'], how='left')
                        alt_ext['end_pos'] += 1 # Before it was the last pos, now it is one after
                        alt_ext['start_pos'] = alt_ext['pos'] + 1
                        if direction == "left":
                            alt_ext['start_pos'] *= -1
                            alt_ext['end_pos'] *= -1
                        alt_ext['alt_pos'] = 0
                        for s in range(1,alt_ext['length'].max()+1):
                            new_alters.append(alt_ext.loc[alt_ext['length'] >= s, ['center','start_pos','end_pos','alt_id','alt_pos','scaf'+str(s),'strand'+str(s),'dist'+str(s)]].rename(columns={('scaf'+str(s)):'scaffold', ('strand'+str(s)):'strand', ('dist'+str(s)):'distance'}))
                            new_alters[-1]['scaffold'] = new_alters[-1]['scaffold'].astype(int)
                            new_alters[-1]['distance'] = new_alters[-1]['distance'].astype(int)
                            if direction == "left":
                                new_alters[-1]['strand'] = np.where(new_alters[-1]['strand'] == '+', '-', '+')
                                alt_ext['alt_pos'] -= 1
                            else:
                                alt_ext['alt_pos'] += 1
                        if direction == "left":
                            alt_ext['alt_pos'] = -alt_ext['length']
                        else:
                            alt_ext['alt_pos'] = alt_ext['length']
                        new_alters.append(alt_ext[['center','start_pos','end_pos','alt_id','alt_pos','to','to_strand','last_dist']].rename(columns={'to':'scaffold', 'to_strand':'strand', 'last_dist':'distance'}))
#
                ## Remove finished extensions
                ext['pos'] += 1
                ext = ext[ext['pos'] < ext['len']].copy()
                ext.drop(columns=['scaffold','strand','side','distance'], inplace=True)
                cur_paths = pd.concat(new_paths)
                if len(new_alters):
                    cur_alters = pd.concat(new_alters)
#
        # Book keeping
        final_paths.append(cur_paths)
        if len(cur_alters):
            final_alternatives.append(cur_alters)
        if len(included_scaffolds):
            included_scaffolds = np.unique(np.concatenate(included_scaffolds))
            duplicated_scaffolds.append(included_scaffolds)
            included_scaffolds = np.unique(np.concatenate([included_scaffolds,cur_start['scaffold'].values]))
        else:
            included_scaffolds = cur_start['scaffold'].values
        knots = knots[np.isin(knots['scaffold'], included_scaffolds) == False]

    if len(duplicated_scaffolds):
        duplicated_scaffolds = np.unique(np.concatenate(duplicated_scaffolds))
    if len(final_paths):
        final_paths = pd.concat(final_paths)
#
        # Extend scaffolds according to final_paths
        ext_scaffolds = scaffolds.drop(duplicated_scaffolds)
        ext_scaffold_parts = [scaffold_parts[np.isin(scaffold_parts['scaffold'], duplicated_scaffolds) == False].copy()]
        for p in range(1, final_paths['pos'].max()+1):
            ext_scaffolds = ext_scaffolds.merge(final_paths.loc[final_paths['pos'] == p, ['center','scaffold','strand']].rename(columns={'center':'scaffold', 'scaffold':'escaf', 'strand':'estrand'}), on=['scaffold'], how='left')
            if np.sum((np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '+')):
                ext_scaffolds.loc[(np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '+'), ['right','rside','rextendible']] = ext_scaffolds.loc[(np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '+'), ['escaf']].merge(scaffolds[['scaffold','right','rside','rextendible']].rename(columns={'scaffold':'escaf'}), on=['escaf'], how='left')[['right','rside','rextendible']].values
                # Add scaffold_parts
                added_parts = scaffold_parts.merge(ext_scaffolds.loc[(np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '+'), ['escaf','scaffold','size']].rename(columns={'escaf':'scaffold', 'scaffold':'new_scaf'}), on=['scaffold'], how='inner')
                added_parts['pos'] += added_parts['size']
                added_parts['scaffold'] = added_parts['new_scaf']
                ext_scaffold_parts.append( added_parts.drop(columns=['size','new_scaf']) )
            ext_scaffolds.loc[np.isnan(ext_scaffolds['escaf']) == False, 'size'] += ext_scaffolds.loc[np.isnan(ext_scaffolds['escaf']) == False, ['escaf']].merge(scaffolds[['scaffold','size']].rename(columns={'scaffold':'escaf'}), on=['escaf'], how='left')['size'].values
            if np.sum((np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '-')):
                ext_scaffolds.loc[(np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '-'), ['right','rside','rextendible']] = ext_scaffolds.loc[(np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '-'), ['escaf']].merge(scaffolds[['scaffold','left','lside','lextendible']].rename(columns={'scaffold':'escaf'}), on=['escaf'], how='left')[['left','lside','lextendible']].values
                # Add scaffold_parts
                added_parts = scaffold_parts.merge(ext_scaffolds.loc[(np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '-'), ['escaf','scaffold','size']].rename(columns={'escaf':'scaffold', 'scaffold':'new_scaf'}), on=['scaffold'], how='inner')
                added_parts['pos'] = added_parts['size'] - added_parts['pos'] - 1
                added_parts['scaffold'] = added_parts['new_scaf']
                added_parts['reverse'] = added_parts['reverse'] == False
                ext_scaffold_parts.append( added_parts.drop(columns=['size','new_scaf']) )
            ext_scaffolds.drop(columns=['escaf','estrand'], inplace=True)
#
        ext_scaffolds['neg_size'] = 1
        for p in range(-1, final_paths['pos'].min()-1, -1):
            ext_scaffolds = ext_scaffolds.merge(final_paths.loc[final_paths['pos'] == p, ['center','scaffold','strand']].rename(columns={'center':'scaffold', 'scaffold':'escaf', 'strand':'estrand'}), on=['scaffold'], how='left')
            if np.sum((np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '-')):
                ext_scaffolds.loc[(np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '-'), ['left','lside','lextendible']] = ext_scaffolds.loc[(np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '-'), ['escaf']].merge(scaffolds[['scaffold','right','rside','rextendible']].rename(columns={'scaffold':'escaf'}), on=['escaf'], how='left')[['right','rside','rextendible']].values
                # Add scaffold_parts
                added_parts = scaffold_parts.merge(ext_scaffolds.loc[(np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '-'), ['escaf','scaffold','neg_size']].rename(columns={'escaf':'scaffold', 'scaffold':'new_scaf'}), on=['scaffold'], how='inner')
                added_parts['pos'] = (added_parts['pos'] + added_parts['neg_size']) * (-1)
                added_parts['scaffold'] = added_parts['new_scaf']
                added_parts['reverse'] = added_parts['reverse'] == False
                ext_scaffold_parts.append( added_parts.drop(columns=['neg_size','new_scaf']) )
            ext_scaffolds.loc[np.isnan(ext_scaffolds['escaf']) == False, 'neg_size'] += ext_scaffolds.loc[np.isnan(ext_scaffolds['escaf']) == False, ['escaf']].merge(scaffolds[['scaffold','size']].rename(columns={'scaffold':'escaf'}), on=['escaf'], how='left')['size'].values
            if np.sum((np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '+')):
                ext_scaffolds.loc[(np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '+'), ['left','lside','lextendible']] = ext_scaffolds.loc[(np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '+'), ['escaf']].merge(scaffolds[['scaffold','left','lside','lextendible']].rename(columns={'scaffold':'escaf'}), on=['escaf'], how='left')[['left','lside','lextendible']].values
                # Add scaffold_parts
                added_parts = scaffold_parts.merge(ext_scaffolds.loc[(np.isnan(ext_scaffolds['escaf']) == False) & (ext_scaffolds['estrand'] == '+'), ['escaf','scaffold','neg_size']].rename(columns={'escaf':'scaffold', 'scaffold':'new_scaf'}), on=['scaffold'], how='inner')
                added_parts['pos'] = (added_parts['neg_size'] - added_parts['pos'] - 1) * (-1)
                added_parts['scaffold'] = added_parts['new_scaf']
                ext_scaffold_parts.append( added_parts.drop(columns=['neg_size','new_scaf']) )
            ext_scaffolds.drop(columns=['escaf','estrand'], inplace=True)
        ext_scaffolds['size'] += ext_scaffolds['neg_size'] - 1
        ext_scaffolds.drop(columns=['neg_size'], inplace=True)
#
        # Set positions starting from 0
        ext_scaffold_parts = pd.concat(ext_scaffold_parts)
        ext_scaffold_parts.sort_values(['scaffold','pos'], inplace=True)
        ext_scaffold_parts['pos'] = ext_scaffold_parts.groupby(['scaffold'], sort=False).cumcount()
#
        scaffolds = ext_scaffolds
        scaffolds.index = scaffolds['scaffold'].values
        scaffold_parts = ext_scaffold_parts

    return scaffolds, scaffold_parts

def GetOriginalConnections(scaffolds, contig_parts):
    # Find original connections in contig_parts
    org_cons = contig_parts.reset_index().loc[contig_parts['org_dist_right'] > 0, ['index','org_dist_right']].copy() # Do not include breaks (org_dist_right==0). If they could not be resealed, they should be separated for good
    org_cons.rename(columns={'index':'from','org_dist_right':'distance'}, inplace=True)
    org_cons['from_side'] = 'r'
    org_cons['to'] = org_cons['from']+1
    org_cons['to_side'] = 'l'
    
    # Lift connections to scaffolds while dropping connections that don't end at scaffold ends
    scaffold_ends_left = scaffolds[['scaffold','left','lside']].rename(columns={'left':'from', 'lside':'from_side'})
    scaffold_ends_left['scaf_side'] = 'l'
    scaffold_ends_right = scaffolds[['scaffold','right','rside']].rename(columns={'right':'from', 'rside':'from_side'})
    scaffold_ends_right['scaf_side'] = 'r'
    scaffold_ends = pd.concat([ scaffold_ends_left, scaffold_ends_right ])
    org_cons = org_cons.merge(scaffold_ends, on=['from','from_side'], how='inner')
    org_cons.drop(columns=['from','from_side'], inplace=True)
    org_cons.rename(columns={'scaffold':'from','scaf_side':'from_side'}, inplace=True)
    org_cons = org_cons.merge(scaffold_ends.rename(columns={'from':'to', 'from_side':'to_side'}), on=['to','to_side'], how='inner')
    org_cons.drop(columns=['to','to_side'], inplace=True)
    org_cons.rename(columns={'scaffold':'to','scaf_side':'to_side'}, inplace=True)
    
    # Also insert reversed connections
    org_cons = pd.concat( [org_cons, org_cons.rename(columns={'from':'to','from_side':'to_side','to':'from','to_side':'from_side'})] )
    
    return org_cons

def OrderByUnbrokenOriginalScaffolds(scaffolds, scaffold_parts, contig_parts):
    ## Bring scaffolds in order of unbroken original scaffolding and remove circularities
    # Every scaffold starts as its own metascaffold
    meta_scaffolds = pd.DataFrame({'meta':scaffolds['scaffold'], 'size':1, 'lcon':-1, 'lcon_side':'', 'rcon':-1, 'rcon_side':''})
    meta_parts = pd.DataFrame({'scaffold':scaffolds['scaffold'], 'meta':scaffolds['scaffold'], 'pos':0, 'reverse':False})

    # Prepare meta scaffolding
    org_cons = GetOriginalConnections(scaffolds, contig_parts)
    meta_scaffolds.loc[org_cons.loc[org_cons['from_side'] == 'l', 'from'].values, 'lcon'] = org_cons.loc[org_cons['from_side'] == 'l', 'to'].values
    meta_scaffolds.loc[org_cons.loc[org_cons['from_side'] == 'l', 'from'].values, 'lcon_side'] = org_cons.loc[org_cons['from_side'] == 'l', 'to_side'].values
    meta_scaffolds.loc[org_cons.loc[org_cons['from_side'] == 'r', 'from'].values, 'rcon'] = org_cons.loc[org_cons['from_side'] == 'r', 'to'].values
    meta_scaffolds.loc[org_cons.loc[org_cons['from_side'] == 'r', 'from'].values, 'rcon_side'] = org_cons.loc[org_cons['from_side'] == 'r', 'to_side'].values

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

    # Inverse meta scaffolds if that reduces the number of inversions for the single contigs
    meta_parts = meta_parts.merge(scaffold_parts.groupby(['scaffold'])['reverse'].agg(['sum','size']).reset_index().rename(columns={'sum':'reversed'}), on=['scaffold'], how='left')
    meta_parts.loc[meta_parts['reverse'], 'reversed'] = meta_parts.loc[meta_parts['reverse'], 'size'] - meta_parts.loc[meta_parts['reverse'], 'reversed']
    meta_parts.sort_values(['meta','pos'], inplace=True)
    meta_reversed = meta_parts.groupby(['meta'], sort=False)[['reversed','size']].sum().reset_index()
    meta_reversed['forward'] = meta_reversed['size'] - meta_reversed['reversed']
    meta_reversed = meta_reversed.loc[meta_reversed['reversed'] > meta_reversed['forward'], 'meta'].values
    meta_reversed = np.isin(meta_parts['meta'], meta_reversed)
    meta_parts.loc[meta_reversed, 'reverse'] = meta_parts.loc[meta_reversed, 'reverse'] == False
    meta_parts.loc[meta_reversed, 'pos'] = meta_parts.loc[meta_reversed, 'pos']*-1
    meta_parts.sort_values(['meta','pos'], inplace=True)
    meta_parts['pos'] = meta_parts.groupby(['meta'], sort=False).cumcount()
    
    # Set new continous scaffold ids consistent with original scaffolding and apply reversions
    meta_parts['new_scaf'] = range(len(meta_parts))
    scaffold_parts = scaffold_parts.merge(meta_parts[['scaffold','reverse','new_scaf']].rename(columns={'reverse':'metareverse'}), on=['scaffold'], how='left')
    scaffold_parts['scaffold'] = scaffold_parts['new_scaf']
    scaffold_parts.loc[scaffold_parts['metareverse'], 'reverse'] = scaffold_parts.loc[scaffold_parts['metareverse'], 'reverse'] == False
    scaffold_parts.loc[scaffold_parts['metareverse'], 'pos'] = scaffold_parts.loc[scaffold_parts['metareverse'], 'pos']*-1
    scaffold_parts.drop(columns=['metareverse','new_scaf'], inplace=True)
    scaffold_parts.sort_values(['scaffold','pos'], inplace=True)
    scaffold_parts['pos'] = scaffold_parts.groupby(['scaffold'], sort=False).cumcount()
    
    toreverse = meta_parts.loc[meta_parts['reverse'], 'scaffold'].values
    tmp = scaffolds.loc[toreverse, 'left']
    scaffolds.loc[toreverse, 'left'] = scaffolds.loc[toreverse, 'right']
    scaffolds.loc[toreverse, 'right'] = tmp
    tmp = scaffolds.loc[toreverse, 'lside']
    scaffolds.loc[toreverse, 'lside'] = scaffolds.loc[toreverse, 'rside']
    scaffolds.loc[toreverse, 'rside'] = tmp
    tmp = scaffolds.loc[toreverse, 'lextendible']
    scaffolds.loc[toreverse, 'lextendible'] = scaffolds.loc[toreverse, 'rextendible']
    scaffolds.loc[toreverse, 'rextendible'] = tmp
    scaffolds.loc[meta_parts['scaffold'].values, 'scaffold'] = meta_parts['new_scaf'].values
    scaffolds.index = scaffolds['scaffold'].values # Make sure we can access scaffolds with .loc[scaffold]
    scaffolds.sort_values(['scaffold'], inplace=True)

    # Set org_dist_left/right based on scaffold (not the contig anymore)
    org_cons = GetOriginalConnections(scaffolds, contig_parts)
    scaffolds['org_dist_left'] = -1
    scaffolds['org_dist_right'] = -1
    scaffolds.loc[org_cons.loc[org_cons['from_side'] == 'l', 'from'].values, 'org_dist_left'] = org_cons.loc[org_cons['from_side'] == 'l', 'distance'].values
    scaffolds.loc[org_cons.loc[org_cons['from_side'] == 'r', 'from'].values, 'org_dist_right'] = org_cons.loc[org_cons['from_side'] == 'r', 'distance'].values

    return scaffolds, scaffold_parts

def MapReadsToScaffolds(mappings, scaffold_table):
    # Start by duplicating the reads for all possible path that reads can take through the scaffolds
    possible_reads = mappings[mappings['num_mappings'] > 1].copy() # Without connections we cannot use them to fill gaps
    possible_reads.drop(columns=['left_con','left_con_side','right_con','right_con_side'], inplace=True)
    possible_reads['read_pos'] = possible_reads.groupby(['read_name','read_start'], sort=False).cumcount()
    possible_reads = possible_reads.merge(scaffold_table[['conpart','scaffold','pos','reverse','scaf_size']], on=['conpart'], how='inner') # Get all possible scaffolds a mapping could map to (the scaffolds the contig part is included)
    possible_reads.loc[possible_reads['reverse'], 'strand'] = np.where(possible_reads.loc[possible_reads['reverse'], 'strand'] == '+', '-', '+') # Update strand to scaffold
    possible_path = possible_reads[['read_name','read_start']].drop_duplicates()
    for p in range(possible_reads['num_mappings'].max()):
        possible_path = possible_path.merge(possible_reads.loc[possible_reads['read_pos'] == p, ['read_name','read_start','scaffold', 'pos', 'strand']].rename(columns={'scaffold':'scaf'+str(p), 'pos':'pos'+str(p), 'strand':'strand'+str(p)}), on=['read_name','read_start'], how='left')
    possible_path = possible_path.reset_index().rename(columns={'index':'path'})
    possible_path = possible_path.loc[np.repeat(possible_path.index.values, possible_reads['num_mappings'].max())]
    possible_path['read_pos'] = possible_path.groupby(['path'], sort=False).cumcount()
    possible_path['scaffold'] = np.nan
    possible_path['pos'] = np.nan
    possible_path['strand'] = np.nan
    for p in range(possible_reads['num_mappings'].max()):
        possible_path.loc[possible_path['read_pos'] == p, 'scaffold'] = possible_path.loc[possible_path['read_pos'] == p, 'scaf'+str(p)]
        possible_path.loc[possible_path['read_pos'] == p, 'pos'] = possible_path.loc[possible_path['read_pos'] == p, 'pos'+str(p)]
        possible_path.loc[possible_path['read_pos'] == p, 'strand'] = possible_path.loc[possible_path['read_pos'] == p, 'strand'+str(p)]
    possible_path.drop(columns=[col for sublist in [['scaf'+str(p),'pos'+str(p),'strand'+str(p)] for p in range(possible_reads['num_mappings'].max())] for col in sublist], inplace=True)
    possible_path = possible_path[np.isnan(possible_path['scaffold']) == False].copy()
    possible_path['scaffold'] = possible_path['scaffold'].astype(int)
    possible_path['pos'] = possible_path['pos'].astype(int)
    possible_reads = possible_reads.merge(possible_path, on=['read_name','read_start','read_pos','scaffold','pos','strand'], how='left')
    possible_reads.sort_values(['path','read_pos'], inplace=True)

    # Filter paths that violate scaffolds for those scaffolds
    possible_reads['spath'] = ((possible_reads['scaffold'] != possible_reads['scaffold'].shift(1)) | (possible_reads['path'] != possible_reads['path'].shift(1))).cumsum() - 1
    # Check if reads are consistent within scaffolds
    possible_reads['remove'] = ( (possible_reads['spath'] == possible_reads['spath'].shift(1)) & # If we don't compare the same read path and scaffold we don't have a reason to remove the read
                                 ( ((possible_reads['strand'] == '+') & ((possible_reads['strand'].shift(1) != '+') | (possible_reads['pos']-1 != possible_reads['pos'].shift(1)))) | # Strands must be consistent and positions must increment for '+' strand
                                   ((possible_reads['strand'] == '-') & ((possible_reads['strand'].shift(1) != '-') | (possible_reads['pos']+1 != possible_reads['pos'].shift(1)))) ) ) # Strands must be consistent and positions must decrement for '-' strand
    # Check if reads experience sudden jumps between scaffolds
    possible_reads['remove'] = ( possible_reads['remove'] | ( (possible_reads['path'] == possible_reads['path'].shift(1)) & (possible_reads['scaffold'] != possible_reads['scaffold'].shift(1)) & # If we don't have a scaffold change within the same read path we don't have a reason to remove the read
                                 ( ((possible_reads['strand'] == '+') & (possible_reads['pos'] != 0)) |
                                   ((possible_reads['strand'] == '-') & (possible_reads['pos'] != possible_reads['scaf_size']-1)) ) ))
    possible_reads['remove'] = ( possible_reads['remove'] | ( (possible_reads['path'] == possible_reads['path'].shift(-1)) & (possible_reads['scaffold'] != possible_reads['scaffold'].shift(-1)) & # If we don't have a scaffold change within the same read path we don't have a reason to remove the read
                                 ( ((possible_reads['strand'] == '-') & (possible_reads['pos'] != 0)) |
                                   ((possible_reads['strand'] == '+') & (possible_reads['pos'] != possible_reads['scaf_size']-1)) ) ))
    # Remove everything from the read on the scaffold it violates
    removals = possible_reads.groupby(['spath'], sort=False)['remove'].agg(['sum','size']).reset_index()
    removals.loc[removals['size'] == 1, 'sum'] += 1 # Also remove the scaffolds, where we only have a single mapping (those are the ones that violated another scaffold, but when they are size 1 cannot be violated itself)
    removals = removals.loc[removals['sum'] > 0, 'spath'].values
    possible_reads = possible_reads[np.isin(possible_reads['spath'], removals) == False].copy()
    possible_reads.drop(columns=['path','remove'], inplace=True)

    return possible_reads

def FillGapsWithReads(scaffolds, scaffold_parts, mappings, contig_parts):
    scaffold_table = scaffold_parts.copy()
    
    # Insert contig and scaffold information
    scaffold_table = scaffold_table.merge(contig_parts[['start','end','name']].reset_index().rename(columns={'index':'conpart'}), on=['conpart'], how='left')
    scaffold_table = scaffold_table.merge(scaffolds[['scaffold','size','org_dist_left','org_dist_right']].rename(columns={'size':'scaf_size'}), on=['scaffold'], how='left')
    scaffold_table.loc[scaffold_table['pos'] != 0, 'org_dist_left'] = -1 
    scaffold_table.loc[scaffold_table['pos'] != scaffold_table['scaf_size']-1, 'org_dist_right'] = -1 
    
    # Find best reads to fill into gaps
    possible_reads = MapReadsToScaffolds(mappings, scaffold_table)
    possible_reads.drop(columns=['read_start','read_end','num_mappings','read_pos','scaf_size','conpart'], inplace=True)
    possible_reads.sort_values(['spath','pos'], inplace=True)
    scaf_hits = possible_reads.groupby(['spath'], sort=False).size().reset_index(name='size') # Get number of times the read maps to a scaffold
    possible_reads['scaf_hits'] = np.repeat(scaf_hits['size'].values, scaf_hits['size'].values)
    possible_reads['scaf_pos'] = possible_reads.groupby(['spath'], sort=False).cumcount()
    possible_reads['rhits'] = possible_reads['scaf_hits'] - possible_reads['scaf_pos'] - 1
    possible_reads['min_hits'] = np.minimum(possible_reads['scaf_pos'], possible_reads['rhits'])
    possible_reads.drop(columns=['scaf_pos'], inplace=True)
    possible_reads['rmapq'] = np.where(possible_reads['rhits'] == 0, -1, possible_reads['mapq'].shift(-1, fill_value=-1))
    possible_reads['rmatches'] = np.where(possible_reads['rhits'] == 0, -1, possible_reads['matches'].shift(-1, fill_value=-1))
    # Filter on mappings to the scaffold (more mappings give us more confidence that the mapping truely belongs here)
    best_reads = possible_reads.loc[possible_reads['rhits'] > 0, ['spath','scaffold','pos','min_hits','scaf_hits','rhits','mapq','rmapq','matches','rmatches']].copy() # Keep all reads in possible_reads, so that we later still have the mapping on the other side of the gap matching the spath selected for the gap filling
    best_reads.sort_values(['scaffold','pos'], inplace=True)
    best = best_reads.groupby(['scaffold','pos'])['min_hits'].agg(['max','size'])
    best_reads = best_reads[best_reads['min_hits'] == np.repeat(best['max'].values, best['size'].values)].copy()
    best = best_reads.groupby(['scaffold','pos'])['scaf_hits'].agg(['max','size'])
    best_reads = best_reads[best_reads['scaf_hits'] == np.repeat(best['max'].values, best['size'].values)].copy()
    best = best_reads.groupby(['scaffold','pos'])['rhits'].agg(['max','size']) # We use rhits as last resort of hit number comparison, because the one with more hits on the right side would win 'min_hits' check if we would take the mapping on the other side of the gap
    best_reads = best_reads[best_reads['rhits'] == np.repeat(best['max'].values, best['size'].values)].copy()
    best_reads.drop(columns=['min_hits','scaf_hits','rhits'], inplace=True)
    # Filter on mapping quality and matches (Get read that has presumably the best quality in the gap)
    best_reads['min_mapq'] = np.minimum(best_reads['mapq'], best_reads['rmapq'])
    best = best_reads.groupby(['scaffold','pos'])['min_mapq'].agg(['max','size'])
    best_reads = best_reads[best_reads['min_mapq'] == np.repeat(best['max'].values, best['size'].values)].copy()
    best_reads['max_mapq'] = np.maximum(best_reads['mapq'], best_reads['rmapq'])
    best = best_reads.groupby(['scaffold','pos'])['max_mapq'].agg(['max','size'])
    best_reads = best_reads[best_reads['max_mapq'] == np.repeat(best['max'].values, best['size'].values)].copy()
    best_reads['min_matches'] = np.minimum(best_reads['matches'], best_reads['rmatches'])
    best = best_reads.groupby(['scaffold','pos'])['min_matches'].agg(['max','size'])
    best_reads = best_reads[best_reads['min_matches'] == np.repeat(best['max'].values, best['size'].values)].copy()
    best_reads['max_matches'] = np.maximum(best_reads['matches'], best_reads['rmatches'])
    best = best_reads.groupby(['scaffold','pos'])['max_matches'].agg(['max','size'])
    best_reads = best_reads[best_reads['max_matches'] == np.repeat(best['max'].values, best['size'].values)].copy()
    # If everything is equally good just take the first
    best_reads = best_reads.groupby(['scaffold','pos']).first().reset_index()
    
    # Get complete information for filling the gap for both sides of it
    best_reads = best_reads[['scaffold','pos', 'spath']].copy()
    best_reads_left = best_reads.merge(possible_reads[['scaffold','pos','spath','read_from','read_to','strand','con_from','con_to','read_name']], on=['scaffold','pos', 'spath'], how='left')
    best_reads_left.drop(columns=['spath'], inplace=True)
    best_reads['pos'] += 1
    best_reads_right = best_reads.merge(possible_reads[['scaffold','pos','spath','read_from','read_to','strand','con_from','con_to']], on=['scaffold','pos', 'spath'], how='left')
    best_reads_right.drop(columns=['spath'], inplace=True)
    
    # Add read information to scaffold table and adjust start and end of contigs
    scaffold_table = scaffold_table.merge(best_reads_left, on=['scaffold','pos'], how='left')
    gapinfo = np.isnan(scaffold_table['read_from']) == False
    scaffold_table.loc[gapinfo & (scaffold_table['reverse'] == False), 'end'] = scaffold_table.loc[gapinfo & (scaffold_table['reverse'] == False), 'con_to'].astype(int)
    scaffold_table.loc[gapinfo & (scaffold_table['reverse']), 'start'] = scaffold_table.loc[gapinfo & (scaffold_table['reverse']), 'con_from'].astype(int)
    scaffold_table['read_name'].fillna('', inplace=True)
    scaffold_table['read_start'] = -1
    scaffold_table['read_end'] = -1
    scaffold_table.loc[gapinfo & (scaffold_table['strand'] == '+'), 'read_start'] = scaffold_table.loc[gapinfo & (scaffold_table['strand'] == '+'), 'read_to'].astype(int)
    scaffold_table.loc[gapinfo & (scaffold_table['strand'] == '-'), 'read_end'] = scaffold_table.loc[gapinfo & (scaffold_table['strand'] == '-'), 'read_from'].astype(int)
    scaffold_table['read_reverse'] = scaffold_table['strand'] == '-'
    scaffold_table.drop(columns=['read_from','read_to','strand','con_from','con_to'], inplace=True)
    
    scaffold_table = scaffold_table.merge(best_reads_right, on=['scaffold','pos'], how='left')
    gapinfo = np.isnan(scaffold_table['read_from']) == False
    scaffold_table.loc[gapinfo & (scaffold_table['reverse'] == False), 'start'] = scaffold_table.loc[gapinfo & (scaffold_table['reverse'] == False), 'con_from'].astype(int)
    scaffold_table.loc[gapinfo & (scaffold_table['reverse']), 'end'] = scaffold_table.loc[gapinfo & (scaffold_table['reverse']), 'con_to'].astype(int)
    scaffold_table.loc[gapinfo.shift(-1, fill_value=False) & (scaffold_table['strand'].shift(-1) == '+'), 'read_end'] = scaffold_table.loc[gapinfo & (scaffold_table['strand'] == '+'), 'read_from'].astype(int).values
    scaffold_table.loc[gapinfo.shift(-1, fill_value=False) & (scaffold_table['strand'].shift(-1) == '-'), 'read_start'] = scaffold_table.loc[gapinfo & (scaffold_table['strand'] == '-'), 'read_to'].astype(int).values
    scaffold_table['break'] = np.isnan(scaffold_table['read_from']) & (scaffold_table['pos'] > 0) # Store if we could not find a read to fill the gap to break the scaffold later
    scaffold_table.drop(columns=['read_from','read_to','strand','con_from','con_to'], inplace=True)
    
    # Break scaffolds if no read was found to close the gap that does not break the scaffold (Trying to rescue the scaffolding, but admit it: we messed up before)
    breaks = scaffold_table.loc[scaffold_table['break'], ['scaffold','conpart','reverse']].rename(columns={'conpart':'new_left','reverse':'new_lside'})
    breaks['new_lside'] = np.where(breaks['new_lside'], 'r', 'l')
    breaks.reset_index(inplace=True,drop=True)
    breaks = breaks.loc[np.repeat(breaks.index, (breaks['scaffold'] != breaks['scaffold'].shift(1))+1).values] # Duplicate the first entry of every scaffold, so that we have as many entries as scaffold parts
    breaks.loc[breaks['scaffold'] != breaks['scaffold'].shift(1), 'new_left'] = -1 # The first scaffold part keeps the left from the unbroken scaffold
    breaks.loc[breaks['scaffold'] != breaks['scaffold'].shift(1), 'new_lside'] = ''
    breaks['new_right'] = -1
    breaks['new_rside'] = ''
    breaks.loc[breaks['scaffold'] == breaks['scaffold'].shift(-1), ['new_right','new_rside']] = scaffold_table.loc[scaffold_table['break'].shift(-1, fill_value=False), ['conpart','reverse']].values # All except the last scaffold part get a new right
    breaks.loc[breaks['new_rside'] != '', 'new_rside'] = np.where(breaks.loc[breaks['new_rside'] != '', 'new_rside']==False, 'r', 'l')
    scaffolds = scaffolds.merge(breaks, on=['scaffold'], how='left')
    scaffolds['new_left'].fillna(-1, inplace=True)
    scaffolds['new_lside'].fillna('', inplace=True)
    scaffolds['new_right'].fillna(-1, inplace=True)
    scaffolds['new_rside'].fillna('', inplace=True)
    scaffolds.loc[scaffolds['new_left'] != -1, 'left'] = scaffolds.loc[scaffolds['new_left'] != -1, 'new_left'].astype(int)
    scaffolds.loc[scaffolds['new_left'] != -1, 'lside'] = scaffolds.loc[scaffolds['new_left'] != -1, 'new_lside']
    scaffolds.loc[scaffolds['new_left'] != -1, 'lextendible'] = True
    scaffolds.loc[scaffolds['new_left'] != -1, 'org_dist_left'] = -1
    scaffolds.loc[scaffolds['new_right'] != -1, 'right'] = scaffolds.loc[scaffolds['new_right'] != -1, 'new_right'].astype(int)
    scaffolds.loc[scaffolds['new_right'] != -1, 'rside'] = scaffolds.loc[scaffolds['new_right'] != -1, 'new_rside']
    scaffolds.loc[scaffolds['new_right'] != -1, 'rextendible'] = True
    scaffolds.loc[scaffolds['new_right'] != -1, 'org_dist_right'] = -1
    scaffolds.drop(columns=['new_left','new_lside','new_right','new_rside'], inplace=True)
    scaffolds['scaffold'] = range(len(scaffolds))
    scaffold_table['scaffold'] = (scaffold_table['break'] | (scaffold_table['scaffold'] != scaffold_table['scaffold'].shift(1))).cumsum()-1
    scaf_size = scaffold_table.groupby('scaffold', sort=False).size().reset_index(name='size')
    scaffolds['size'] = scaf_size['size'].values
    scaffold_table['scaf_size'] = np.repeat(scaf_size['size'].values, scaf_size['size'].values)
    scaffold_parts = scaffold_table[['conpart','scaffold','pos','reverse']].copy()
    scaffold_table.drop(columns=['break'], inplace=True)
    
    # Handle gaps with negative length
    scaffold_table.loc[(scaffold_table['read_start'] != -1) & (scaffold_table['read_start'] >= scaffold_table['read_end']), 'read_name'] = ''
    trim = (scaffold_table['read_start'] != -1) & (scaffold_table['read_start'] >= scaffold_table['read_end']) & (scaffold_table['reverse'] == False)
    scaffold_table.loc[trim, 'end'] -= scaffold_table.loc[trim, 'read_start'] - scaffold_table.loc[trim, 'read_end']
    trim = (scaffold_table['read_start'] != -1) & (scaffold_table['read_start'] >= scaffold_table['read_end']) & (scaffold_table['reverse'])
    scaffold_table.loc[trim, 'start'] += scaffold_table.loc[trim, 'read_start'] - scaffold_table.loc[trim, 'read_end']
    scaffold_table.loc[(scaffold_table['read_start'] != -1) & (scaffold_table['read_start'] >= scaffold_table['read_end']), 'read_end'] = -1
    scaffold_table.loc[(scaffold_table['read_start'] != -1) & (scaffold_table['read_start'] >= scaffold_table['read_end']), 'read_start'] = -1
    scaffold_table.loc[scaffold_table['start'] >= scaffold_table['end'], 'end'] = 0 # This should not happen, but better safe than sorry
    scaffold_table.loc[scaffold_table['start'] >= scaffold_table['end'], 'start'] = 0
    
    # Insert reads between contigs
    scaffold_table.drop(columns=['conpart'], inplace=True)
    scaffold_table = scaffold_table.loc[np.repeat(scaffold_table.index.values, 2)].copy()
    scaffold_table['type'] = ['contig','read']*len(scaffold_parts)
    scaffold_table = scaffold_table[ (scaffold_table['type'] == 'contig') | (scaffold_table['read_name'] != '') ].copy()
    scaffold_table.loc[scaffold_table['type'] == 'read', 'name'] = scaffold_table.loc[scaffold_table['type'] == 'read', 'read_name']
    scaffold_table.loc[scaffold_table['type'] == 'read', 'start'] = scaffold_table.loc[scaffold_table['type'] == 'read', 'read_start']
    scaffold_table.loc[scaffold_table['type'] == 'read', 'end'] = scaffold_table.loc[scaffold_table['type'] == 'read', 'read_end']
    scaffold_table.loc[scaffold_table['type'] == 'read', 'reverse'] = scaffold_table.loc[scaffold_table['type'] == 'read', 'read_reverse']
    scaf_size = scaffold_table.groupby('scaffold', sort=False).size().reset_index(name='size')
    scaffold_table['scaf_size'] = np.repeat(scaf_size['size'].values, scaf_size['size'].values)
    scaffold_table['pos'] = scaffold_table.groupby(['scaffold']).cumcount()
    
    # Reorder and select columns
    scaffold_table = scaffold_table[['scaffold','pos','scaf_size','type','name','start','end','reverse','org_dist_left','org_dist_right']].reset_index(drop=True).copy()
    
    return scaffold_table, scaffolds, scaffold_parts

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
    scaffolds, scaffold_parts = TraverseScaffoldingGraph(scaffolds, scaffold_parts, scaffold_graph, contig_parts, scaf_bridges, ploidy)

    # Finish Scaffolding
    scaffolds, scaffold_parts = OrderByUnbrokenOriginalScaffolds(scaffolds, scaffold_parts, contig_parts)
    scaffold_table, scaffolds, scaffold_parts = FillGapsWithReads(scaffolds, scaffold_parts, mappings, contig_parts) # Might break apart scaffolds again, if we cannot find a gap filling read

    return scaffolds, scaffold_parts, scaffold_table

def GetOutputInfo(result_info, scaffolds, scaffold_parts, scaffold_table, contig_parts):
    # Calculate lengths
    con_len = scaffold_table[['scaffold', 'start', 'end']].copy()
    con_len['length'] = con_len['end'] - con_len['start']
    con_len = con_len.groupby('scaffold', sort=False)['length'].sum().values
    
    scaf_len = scaffolds[['org_dist_left']].copy()
    scaf_len['length'] = con_len
    scaf_len.loc[scaf_len['org_dist_left'] >= 0, 'length'] += scaf_len.loc[scaf_len['org_dist_left'] >= 0, 'org_dist_left']
    scaf_len['meta'] = (scaf_len['org_dist_left'] == -1).cumsum()
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
    
    # Get resealed breaks (Number of total contig_parts - minimum number of scaffold chunks needed to have them all included in the proper order)
    broken_contigs = contig_parts.loc[(contig_parts['part'] > 0) | (contig_parts['part'].shift(-1) > 0), ['contig','part','conpart']]
    resealed_breaks = scaffold_parts[np.isin(scaffold_parts['conpart'].values, broken_contigs['conpart'].values)].sort_values(['scaffold','pos'])
    resealed_breaks['chunk'] = ( (resealed_breaks['scaffold'] != resealed_breaks['scaffold'].shift(1)) | (resealed_breaks['reverse'] != resealed_breaks['reverse'].shift(1)) |
                                 (resealed_breaks['pos'] != resealed_breaks['pos'].shift(1)+1) | (resealed_breaks['conpart'] != resealed_breaks['conpart'].shift(1)+np.where(resealed_breaks['reverse'], -1, 1))).cumsum()
    resealed_breaks = resealed_breaks.groupby(['chunk'], sort=False)['conpart'].agg(['min','max']).reset_index(drop=True)
    resealed_breaks = resealed_breaks[resealed_breaks['min'] < resealed_breaks['max']].sort_values(['min','max']).drop_duplicates()
    resealed_breaks['group'] = (resealed_breaks['min'] > resealed_breaks['max'].shift(1)).cumsum()
    resealed_breaks['length'] = resealed_breaks['max'] - resealed_breaks['min'] + 1

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

def GetScaffoldExtendingMappings(mappings, contig_parts, scaffolds, max_dist_contig_end, min_extension, min_num_reads, pdf):
    mappings.drop(columns=['left_con','left_con_side' ,'right_con','right_con_side','num_mappings'], inplace=True)

    # Only reads that map to the outter contigs in a scaffold are relevant
    mappings = mappings[ np.isin(mappings['conpart'], pd.concat([scaffolds.loc[scaffolds['lextendible'], 'left'], scaffolds.loc[scaffolds['rextendible'], 'right']]).values) ].copy()

    # Reads need to extend the contig in the right direction
    mappings['con_start'] = contig_parts.iloc[mappings['conpart'].values, contig_parts.columns.get_loc('start')].values
    mappings['con_end'] = contig_parts.iloc[mappings['conpart'].values, contig_parts.columns.get_loc('end')].values
    mappings['left_ext'] = np.where('+' == mappings['strand'], mappings['read_from']-mappings['read_start']-(mappings['con_from']-mappings['con_start']), (mappings['read_end']-mappings['read_to'])-(mappings['con_from']-mappings['con_start']))
    mappings['right_ext'] = np.where('-' == mappings['strand'], mappings['read_from']-mappings['read_start']-(mappings['con_end']-mappings['con_to']), (mappings['read_end']-mappings['read_to'])-(mappings['con_end']-mappings['con_to']))

    # Set extensions to zero that are in the wrong direction
    mappings.loc[ np.isin(mappings['conpart'], pd.concat([scaffolds.loc[scaffolds['lextendible'] & (scaffolds['lside'] == 'l'), 'left'], scaffolds.loc[scaffolds['rextendible'] & (scaffolds['rside'] == 'l'), 'right']]).values) == False, 'left_ext'] = 0
    mappings.loc[ np.isin(mappings['conpart'], pd.concat([scaffolds.loc[scaffolds['lextendible'] & (scaffolds['lside'] == 'r'), 'left'], scaffolds.loc[scaffolds['rextendible'] & (scaffolds['rside'] == 'r'), 'right']]).values) == False, 'right_ext'] = 0

    # Set extensions to zero that are too far away from the contig_end
    mappings.loc[mappings['con_from']-mappings['con_start'] > max_dist_contig_end, 'left_ext'] = 0
    mappings.loc[mappings['con_end']-mappings['con_to'] > max_dist_contig_end, 'right_ext'] = 0
    
    mappings = mappings[(mappings['left_ext'] > 0) | (mappings['right_ext'] > 0)].copy()
    mappings['left_ext'] = np.maximum(0, mappings['left_ext'])
    mappings['right_ext'] = np.maximum(0, mappings['right_ext'])
    
    # Lift mappings from contigs to scaffolds
    left_exts = scaffolds.loc[scaffolds['lextendible'], ['scaffold','left']].rename(columns={'left':'conpart'})
    left_exts['reverse'] = scaffolds.loc[scaffolds['lextendible'], 'lside'].values == 'r'
    right_exts = scaffolds.loc[scaffolds['rextendible'], ['scaffold','right']].rename(columns={'right':'conpart'})
    right_exts['reverse'] = scaffolds.loc[scaffolds['rextendible'], 'rside'].values == 'l'
    mappings = mappings.merge(pd.concat([left_exts,right_exts]).drop_duplicates(), on=['conpart'])
    mappings.loc[mappings['reverse'], 'strand'] = np.where(mappings.loc[mappings['reverse'], 'strand'] == '+', '-', '+')
    tmp_ext = mappings.loc[mappings['reverse'], 'left_ext'].values
    mappings.loc[mappings['reverse'], 'left_ext'] = mappings.loc[mappings['reverse'], 'right_ext']
    mappings.loc[mappings['reverse'], 'right_ext'] = tmp_ext

    mappings['left_dist_start'] = np.where(mappings['reverse'], mappings['con_end']-mappings['con_to'], mappings['con_from']-mappings['con_start'])
    mappings['left_dist_end'] = np.where(mappings['reverse'], mappings['con_end']-mappings['con_from'], mappings['con_to']-mappings['con_start'])
    mappings['right_dist_start'] = np.where(mappings['reverse'], mappings['con_from']-mappings['con_start'], mappings['con_end']-mappings['con_to'])
    mappings['right_dist_end'] = np.where(mappings['reverse'], mappings['con_to']-mappings['con_start'], mappings['con_end']-mappings['con_from'])

    mappings.loc[mappings['left_ext']==0, 'left_dist_start'] = -1
    mappings.loc[mappings['left_ext']==0, 'left_dist_end'] = -1
    mappings.loc[mappings['right_ext']==0, 'right_dist_start'] = -1
    mappings.loc[mappings['right_ext']==0, 'right_dist_end'] = -1

    mappings = mappings[['read_name','read_start','read_end','read_from','read_to', 'scaffold','left_dist_start','left_dist_end','right_dist_end','right_dist_start', 'strand','mapq','matches', 'left_ext','right_ext']].copy()

    # Only keep long enough extensions
    if pdf:
        extension_lengths = np.concatenate([mappings.loc[0<mappings['left_ext'],'left_ext'], mappings.loc[0<mappings['right_ext'],'right_ext']])
        if len(extension_lengths):
            if np.sum(extension_lengths < 10*min_extension):
                PlotHist(pdf, "Extension length", "# Extensions", np.extract(extension_lengths < 10*min_extension, extension_lengths), threshold=min_extension)
            PlotHist(pdf, "Extension length", "# Extensions", extension_lengths, threshold=min_extension, logx=True)
        del extension_lengths
    
    mappings = mappings[(min_extension<=mappings['left_ext']) | (min_extension<=mappings['right_ext'])].copy()
    
    # Only keep extension, when there are enough of them
    num_left_extensions = mappings[mappings['left_ext']>0].groupby(['scaffold']).size().reset_index(name='counts')
    num_right_extensions = mappings[mappings['right_ext']>0].groupby(['scaffold']).size().reset_index(name='counts')
    mappings = mappings[((mappings['left_ext']>0) & np.isin(mappings['scaffold'], num_left_extensions.loc[num_left_extensions['counts']>=min_num_reads, 'scaffold'].values)) |
                        ((mappings['right_ext']>0) & np.isin(mappings['scaffold'], num_right_extensions.loc[num_right_extensions['counts']>=min_num_reads, 'scaffold'].values))].copy()

    return mappings

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
    mappings = UpdateMappingsToContigParts(mappings, contig_parts, min_mapping_length)
    del break_groups, contigs, contig_ids
#
    print( str(timedelta(seconds=clock())), "Search for possible bridges")
    bridges = GetBridges(mappings, borderline_removal, min_factor_alternatives, min_num_reads, org_scaffold_trust, contig_parts, cov_probs, prob_factor, min_mapping_length, min_distance_tolerance, rel_distance_tolerance, pdf)
#
    print( str(timedelta(seconds=clock())), "Scaffold the contigs")
    scaffolds, scaffold_parts, scaffold_table = ScaffoldContigs(contig_parts, bridges, mappings, ploidy)
    result_info = GetOutputInfo(result_info, scaffolds, scaffold_parts, scaffold_table, contig_parts)

    print( str(timedelta(seconds=clock())), "Search for possible extensions")
    mappings = GetScaffoldExtendingMappings(mappings, contig_parts, scaffolds, max_dist_contig_end, min_extension, min_num_reads, pdf)

    if pdf:
        pdf.close()

    print( str(timedelta(seconds=clock())), "Writing output")
    mappings.to_csv(prefix+"_extensions.csv", index=False)
    scaffold_table.to_csv(prefix+"_scaffold_table.csv", index=False)
    np.savetxt(prefix+'_extending_reads.lst', np.unique(mappings['read_name']), fmt='%s')

    print( str(timedelta(seconds=clock())), "Finished")
    PrintStats(result_info)

def LoadExtensions(prefix, min_extension):
    # Load extending mappings
    mappings = pd.read_csv(prefix+"_extensions.csv")

    # Split mappings by side
    mappings = mappings.iloc[np.repeat(mappings.index, 2)].reset_index(drop=True).copy()
    mappings['side'] = np.tile(['l','r'], int(len(mappings)/2))
    mappings['dist_start'] = np.where('l' == mappings['side'], mappings['left_dist_start'], mappings['right_dist_start'])
    mappings['dist_end'] = np.where('l' == mappings['side'], mappings['left_dist_end'], mappings['right_dist_end'])
    mappings['extension'] = np.where('l' == mappings['side'], mappings['left_ext'], mappings['right_ext'])
    mappings.drop(columns = ['left_dist_start', 'right_dist_start', 'left_dist_end', 'right_dist_end', 'left_ext', 'right_ext'], inplace=True)
    
    # Drop sides that do not fullfil min_extension
    mappings = mappings[min_extension <= mappings['extension']].reset_index(drop=True).copy()
    
    return mappings

def RemoveDuplicatedReadMappings(reads):
    # Minimap2 has sometimes two overlapping mappings: Remove the shorter mapping
    reads.sort_values(['q_scaffold','q_side','t_scaffold','t_side','q_index','t_index'], inplace=True)
    reads['min_len'] = np.minimum(reads['q_end']-reads['q_start'], reads['t_end']-reads['t_start'])
    duplicates = reads.groupby(['q_scaffold','q_side','t_scaffold','t_side','q_index','t_index'])['min_len'].agg(['max','size'])
    reads['max_len'] = np.repeat(duplicates['max'].values,duplicates['size'].values)
    reads = reads[reads['min_len'] == reads['max_len']].copy()

    # Remove the more unequal one
    reads['max_len'] = np.maximum(reads['q_end']-reads['q_start'], reads['t_end']-reads['t_start'])
    duplicates = reads.groupby(['q_scaffold','q_side','t_scaffold','t_side','q_index','t_index'])['max_len'].agg(['min','size'])
    reads['min_len'] = np.repeat(duplicates['min'].values,duplicates['size'].values)
    reads = reads[reads['min_len'] == reads['max_len']].copy()
    reads.drop(columns=['min_len','max_len'], inplace=True)

    # Otherwise remove the second one (with the higher original id)
    duplicates = reads.groupby(['q_scaffold','q_side','t_scaffold','t_side','q_index','t_index'])['org_id'].agg(['min','size'])
    reads['min_id'] = np.repeat(duplicates['min'].values,duplicates['size'].values)
    reads = reads[reads['org_id'] == reads['min_id']].copy()
    reads.drop(columns=['min_id','org_id'], inplace=True)

    return reads

def LoadReads(all_vs_all_mapping_file, mappings, min_length_contig_break):
    # Load all vs. all mappings for extending reads
    reads = ReadPaf(all_vs_all_mapping_file)
    reads.drop(columns=['matches','alignment_length','mapq'], inplace=True) # We don't need those columns

    # Add scaffold and side to which the query reads belong to
    reads = reads.merge(mappings[['read_name','read_start','read_from','read_to','scaffold','side','strand']].reset_index().rename(columns={'index':'q_index', 'read_name':'q_name', 'read_start':'q_read_start', 'read_from':'q_read_from', 'read_to':'q_read_to', 'scaffold':'q_scaffold', 'side':'q_side', 'strand':'q_strand'}), on=['q_name'], how='inner')

    # Remove reads where the all vs. all mapping is not in the gap for the scaffold belonging to the query
    reads = reads[np.where(np.logical_xor('+' == reads['q_strand'], 'l' == reads['q_side']), reads['q_end'] > reads['q_read_to'], reads['q_start'] < reads['q_read_from'])].copy()

    # Repeat the last two steps for the target reads
    reads = reads.merge(mappings[['read_name','read_start','read_from','read_to','scaffold','side','strand']].reset_index().rename(columns={'index':'t_index','read_name':'t_name', 'read_start':'t_read_start', 'read_from':'t_read_from', 'read_to':'t_read_to', 'scaffold':'t_scaffold', 'side':'t_side', 'strand':'t_strand'}), on=['t_name'], how='inner')
    reads = reads[np.where(np.logical_xor('+' == reads['t_strand'], 'l' == reads['t_side']), reads['t_end'] > reads['t_read_to'], reads['t_start'] < reads['t_read_from'])].copy()

    # Remove reads that map to itself, drop superfluous columns and reorder remaining columns
    reads = reads.loc[(reads['q_name'] != reads['t_name']) | (reads['q_read_start'] != reads['t_read_start']), ['q_scaffold','q_side','q_index','q_read_from','q_read_to','q_start','q_end','q_len','q_strand','t_scaffold','t_side','t_index','t_read_from','t_read_to','t_start','t_end','t_len','t_strand','strand'] ]

    # Add query-target reversed copy to reads, so that all relevant read mappings show up if we search for example for q_scaffold
    reads['org_id'] = reads.index # We need this one to properly remove duplicated mappings later
    reads = pd.concat([reads, reads.rename(columns={'q_scaffold':'t_scaffold', 't_scaffold':'q_scaffold', 'q_side':'t_side', 't_side':'q_side', 'q_index':'t_index', 't_index':'q_index',
                                                    'q_read_from':'t_read_from', 't_read_from':'q_read_from', 'q_read_to':'t_read_to', 't_read_to':'q_read_to', 
                                                    'q_start':'t_start', 't_start':'q_start', 'q_end':'t_end', 't_end':'q_end', 'q_len':'t_len', 't_len':'q_len',
                                                    'q_strand':'t_strand', 't_strand':'q_strand'})], ignore_index=True, sort=False)

    # Minimap2 has sometimes two overlapping mappings
    reads = RemoveDuplicatedReadMappings(reads)

    # Split into extending read mappings(share the same scaffold) and connecting read mappings(do not share the same scaffold)
    extensions = reads[(reads['q_scaffold'] == reads['t_scaffold']) & (reads['q_side'] == reads['t_side'])].drop(columns=['t_scaffold','t_side']).rename(columns={'q_scaffold':'scaffold','q_side':'side'}).copy()
    connections = reads[(reads['q_scaffold'] != reads['t_scaffold']) | (reads['q_side'] != reads['t_side'])].copy()

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

    # Extensions: Remove reads that somehow do not fullfil strand rules and remove superfluous strand column
    extensions = extensions[ np.where(extensions['q_strand'] == extensions['t_strand'], '+', '-') == extensions['strand'] ].copy()
    extensions.drop(columns=['strand'], inplace=True)

    return extensions, connections

def ClusterExtension(extensions, min_num_reads):
    # Remove indexes that cannot fulfill min_num_reads (have less than min_num_reads-1 mappings to other reads)
    extensions.sort_values(['scaffold','side','q_index'], inplace=True)
    org_len = len(extensions)+1
    while len(extensions) < org_len:
        num_mappings = extensions.groupby(['scaffold','side','q_index'], sort=False).size().values
        extensions['q_mappings'] = np.repeat(num_mappings, num_mappings)
        extensions.sort_values(['scaffold','side','t_index'], inplace=True)
        
        num_mappings = extensions.groupby(['scaffold','side','t_index'], sort=False).size().values
        extensions['t_mappings'] = np.repeat(num_mappings, num_mappings)
        
        org_len = len(extensions)
        extensions = extensions[ np.minimum(extensions['q_mappings'], extensions['t_mappings']) >= min_num_reads-1 ].copy()
        if len(extensions) < org_len:
            num_mappings = extensions.groupby(['scaffold','side','t_index'], sort=False).size().values
            extensions['t_mappings'] = np.repeat(num_mappings, num_mappings)
            
            extensions.sort_values(['scaffold','side','q_index'], inplace=True)
            num_mappings = extensions.groupby(['scaffold','side','q_index'], sort=False).size().values
            extensions['q_mappings'] = np.repeat(num_mappings, num_mappings)
            
            org_len = len(extensions)
            extensions = extensions[ np.minimum(extensions['q_mappings'], extensions['t_mappings']) >= min_num_reads-1 ].copy()
            
    extensions.drop(columns=['q_mappings','t_mappings'], inplace=True)
    
    # Cluster reads that share a mapping (A read in a cluster maps at least to one other read in this cluster)
    extensions.sort_values(['scaffold','side','q_index'], inplace=True)
    clusters = extensions.groupby(['scaffold','side','q_index'], sort=False).size().reset_index(name='size')
    clusters['cluster'] = np.arange(len(clusters))
    extensions['q_cluster_id'] = np.repeat(clusters.index.values, clusters['size'].values)
    extensions = extensions.merge(clusters[['scaffold','side','q_index']].reset_index().rename(columns={'q_index':'t_index','index':'t_cluster_id'}), on=['scaffold','side','t_index'], how='left')
    
    cluster_col = clusters.columns.get_loc('cluster')
    extensions['cluster'] = np.minimum(clusters.iloc[extensions['q_cluster_id'].values, cluster_col].values, clusters.iloc[extensions['t_cluster_id'].values, cluster_col].values)
    clusters['new_cluster'] = np.minimum( extensions.groupby('q_cluster_id')['cluster'].min().values, extensions.groupby('t_cluster_id')['cluster'].min().values )
    while np.sum(clusters['new_cluster'] != clusters['cluster']):
        clusters['cluster'] = clusters['new_cluster']
        extensions['cluster'] = np.minimum(clusters.iloc[extensions['q_cluster_id'].values, cluster_col].values, clusters.iloc[extensions['t_cluster_id'].values, cluster_col].values)
        clusters['new_cluster'] = np.minimum( extensions.groupby('q_cluster_id')['cluster'].min().values, extensions.groupby('t_cluster_id')['cluster'].min().values )
    
    extensions['cluster'] = clusters.iloc[extensions['q_cluster_id'].values, cluster_col].values
    clusters.drop(columns=['new_cluster'], inplace=True)
    
    # Remove sides with two alternative clusters
    clusters.sort_values(['cluster','size','q_index'], ascending=[True,False,True], inplace=True)
    alternatives = clusters[['scaffold','side','cluster']].drop_duplicates().groupby(['scaffold','side']).size().reset_index(name='alternatives')
    clusters['cluster_id'] = clusters.index # We have to save it, because 'q_cluster_id' and 't_cluster_id' use them and merging removes the index
    clusters = clusters.merge(alternatives, on=['scaffold','side'], how='left')
    clusters.index = clusters['cluster_id'].values
    extensions['alternatives'] = clusters.loc[ extensions['q_cluster_id'].values, 'alternatives' ].values
    clusters = clusters[ clusters['alternatives'] == 1 ].copy()
    extensions = extensions[ extensions['alternatives'] == 1 ].copy()
    clusters.drop(columns=['alternatives'], inplace=True)
    extensions.drop(columns=['alternatives'], inplace=True)
    
    # Add how long the query agrees with the target in the gap
    extensions['q_agree'] = np.where( np.logical_xor('+' == extensions['q_strand'], 'l' == extensions['side']), extensions['q_end']-extensions['q_read_to'], extensions['q_read_from']-extensions['q_start'] )
    
    extensions.drop(columns=['cluster','q_cluster_id','t_cluster_id'], inplace=True)
    return extensions

def ExtendScaffolds(scaffold_table, extensions, mappings, min_num_reads, max_mapping_uncertainty):
    if len(extensions) and len(mappings):
        # Create table on how long mappings agree in the gap with at least min_num_reads-1 (-1 because they always agree with themselves)
        len_agree = extensions[['scaffold','side','q_index','q_agree']].sort_values(['scaffold','side','q_index','q_agree'], ascending=[True,True,True,False])
        len_agree['n_longest'] = 1
        len_agree['n_longest'] = len_agree[['scaffold','side','q_index','n_longest']].groupby(['scaffold','side','q_index'], sort=False).cumsum()
        len_agree = len_agree[len_agree['n_longest'] == max(1,min_num_reads-1)]
        len_agree.drop(columns=['n_longest'], inplace=True)
        len_mappings = mappings.iloc[len_agree['q_index'].values]
        len_agree['q_ext_len'] = np.where( np.logical_xor('+' == len_mappings['strand'], 'l' == len_mappings['side']), len_mappings['read_end']-len_mappings['read_to'], len_mappings['read_from'] )
        
        # Take the read that has the longest agreement with at least min_num_reads-1 and see if and at what position another bundle of min_num_reads diverge from it (extend until that position or as long as min_num_reads-1 agree)
        len_agree.sort_values(['scaffold','side','q_agree'], ascending=[True,True,False], inplace=True)
        extending_reads = len_agree.groupby(['scaffold','side'], sort=False).first().reset_index()
        len_agree = len_agree.merge(extending_reads[['scaffold','side','q_index']].rename(columns={'q_index':'t_index'}), on=['scaffold','side'], how='inner')
        len_agree = len_agree[len_agree['q_index'] != len_agree['t_index']].copy()
        len_agree = len_agree.merge(extensions[['scaffold','side','q_index','t_index','q_agree']].rename(columns={'q_agree':'qt_agree'}), on=['scaffold','side','q_index','t_index'], how='left')
        len_agree['qt_agree'].fillna(0, inplace=True)
        len_agree = len_agree[ len_agree['q_agree'] > len_agree['qt_agree']+max_mapping_uncertainty].copy()
        len_agree = len_agree.merge(extensions[['scaffold','side','q_index','t_index','q_agree']].rename(columns={'q_index':'t_index','t_index':'q_index','q_agree':'tq_agree'}), on=['scaffold','side','q_index','t_index'], how='left')
        len_agree['tq_agree'].fillna(0, inplace=True)
        
        len_agree.sort_values(['scaffold','side','tq_agree'], inplace=True)
        len_agree['n_disagree'] = 1
        len_agree['n_disagree'] = len_agree[['scaffold','side','n_disagree']].groupby(['scaffold','side'], sort=False).cumsum()
        len_agree = len_agree[ len_agree['n_disagree'] == min_num_reads ].copy()
        extending_reads = extending_reads.merge(len_agree[['scaffold','side','tq_agree']].rename(columns={'tq_agree':'valid_ext'}), on=['scaffold','side'], how='left')
        extending_reads.loc[np.isnan(extending_reads['valid_ext']),'valid_ext'] = extending_reads.loc[np.isnan(extending_reads['valid_ext']),'q_agree'].values
        extending_reads = extending_reads[extending_reads['valid_ext'] > 0.0].copy()
        
        if len(extending_reads):
            # Change structure of extending_reads to the same as scaffold_table
            # Start by adding read information
            ext_mappings = mappings.iloc[extending_reads['q_index'].values]
            extending_reads['name'] = ext_mappings['read_name'].values
            extending_reads['start'] = np.where( np.logical_xor('+' == ext_mappings['strand'], 'l' == ext_mappings['side']), ext_mappings['read_to'].values, ext_mappings['read_from'].values-extending_reads['valid_ext'].values.astype(int) )
            extending_reads['end'] = np.where( np.logical_xor('+' == ext_mappings['strand'], 'l' == ext_mappings['side']), ext_mappings['read_to'].values+extending_reads['valid_ext'].values.astype(int), ext_mappings['read_from'].values )
            extending_reads['reverse'] = ('-' == ext_mappings['strand'].values)
            
            # Cut contigs where it is required by the extensions
            scaffold_table['side'] = np.where(0 == scaffold_table['pos'], 'l', '')
            scaffold_table = scaffold_table.merge(ext_mappings[['scaffold','side','dist_start']].rename(columns={'dist_start':'left_start'}), on=['scaffold','side'], how='left')
            scaffold_table['side'] = np.where(scaffold_table['scaf_size'] == scaffold_table['pos']+1, 'r', '')
            scaffold_table = scaffold_table.merge(ext_mappings[['scaffold','side','dist_start']].rename(columns={'dist_start':'right_start'}), on=['scaffold','side'], how='left')
            scaffold_table.loc[(False == scaffold_table['reverse']) & (False == np.isnan(scaffold_table['left_start'])), 'start'] += scaffold_table.loc[(False == scaffold_table['reverse']) & (False == np.isnan(scaffold_table['left_start'])), 'left_start']
            scaffold_table.loc[scaffold_table['reverse'] & (False == np.isnan(scaffold_table['left_start'])), 'end'] -= scaffold_table.loc[scaffold_table['reverse'] & (False == np.isnan(scaffold_table['left_start'])), 'left_start']
            scaffold_table.loc[(False == scaffold_table['reverse']) & (False == np.isnan(scaffold_table['right_start'])), 'end'] -= scaffold_table.loc[(False == scaffold_table['reverse']) & (False == np.isnan(scaffold_table['right_start'])), 'right_start']
            scaffold_table.loc[scaffold_table['reverse'] & (False == np.isnan(scaffold_table['right_start'])), 'start'] += scaffold_table.loc[scaffold_table['reverse'] & (False == np.isnan(scaffold_table['right_start'])), 'right_start']
            scaffold_table['start'] = scaffold_table['start'].astype(int)
            scaffold_table['end'] = scaffold_table['end'].astype(int)
            
            # Update scaffold information in scaffold_table and add it to extending_reads
            ext_count = scaffold_table.groupby(['scaffold'], sort=False).agg({'left_start':['size','count'], 'right_start':['count']})
            scaffold_table['scaf_size'] += np.repeat(ext_count['left_start','count'].values + ext_count['right_start','count'].values, ext_count['left_start','size'])
            scaffold_table['pos'] += np.repeat(ext_count['left_start','count'].values, ext_count['left_start','size'])
            
            extending_reads = extending_reads.merge(scaffold_table[['scaffold','scaf_size']].groupby(['scaffold'], sort=False).first(), on=['scaffold'], how='left')
            extending_reads['pos'] = np.where(extending_reads['side'] == 'l', 0, extending_reads['scaf_size']-1)
            
            extending_reads = extending_reads.merge(scaffold_table[['scaffold','side','org_dist_right']], on=['scaffold','side'], how='left')
            extending_reads['org_dist_right'] = extending_reads['org_dist_right'].fillna(-1).astype(int)
            scaffold_table['side'] = np.where(1 == scaffold_table['pos'], 'l', '') # Position of left contig with extension is now 1, because we already shifted (For non-extended scaffold that is not the left-most part, but we don't care, because those scaffolds are not in extending_reads)
            extending_reads = extending_reads.merge(scaffold_table[['scaffold','side','org_dist_left']], on=['scaffold','side'], how='left')
            extending_reads['org_dist_left'] = extending_reads['org_dist_left'].fillna(-1).astype(int)
            
            scaffold_table.loc[False == np.isnan(scaffold_table['left_start']), 'org_dist_left'] = -1
            scaffold_table.loc[False == np.isnan(scaffold_table['right_start']), 'org_dist_right'] = -1
            scaffold_table.drop(columns=['side','left_start','right_start'], inplace=True)
        
            # Add extending_reads to scaffold_table and sort again
            extending_reads['type'] = 'read'
            scaffold_table = scaffold_table.append( extending_reads[['scaffold','pos','scaf_size','type','name','start','end','reverse','org_dist_left','org_dist_right']] )
            scaffold_table.sort_values(['scaffold','pos'], inplace=True)
        
            extension_info = {}
            extension_info['count'] = len(extending_reads)
            extension_info['left'] = len(extending_reads[extending_reads['side'] == 'l'])
            extension_info['right'] = len(extending_reads[extending_reads['side'] == 'r'])
            extension_info['mean'] = int(round(np.mean(extending_reads['valid_ext'])))
            extension_info['min'] = int(np.min(extending_reads['valid_ext']))
            extension_info['max'] = int(np.max(extending_reads['valid_ext']))
        else:
            extension_info = {}
            extension_info['count'] = 0
            extension_info['left'] = 0
            extension_info['right'] = 0
            extension_info['mean'] = 0
            extension_info['min'] = 0
            extension_info['max'] = 0
    else:
        extension_info = {}
        extension_info['count'] = 0
        extension_info['left'] = 0
        extension_info['right'] = 0
        extension_info['mean'] = 0
        extension_info['min'] = 0
        extension_info['max'] = 0
        
    return scaffold_table, extension_info

def MiniGapExtend(all_vs_all_mapping_file, prefix, min_length_contig_break):
    # Define parameters
    min_extension = 500    
    min_num_reads = 3
    max_mapping_uncertainty = 200
    
    print( str(timedelta(seconds=clock())), "Preparing data from files")
    scaffold_table = pd.read_csv(prefix+"_scaffold_table.csv")
    mappings = LoadExtensions(prefix, min_extension)
    extensions, connections = LoadReads(all_vs_all_mapping_file, mappings, min_length_contig_break)
    
    print( str(timedelta(seconds=clock())), "Searching for extensions")
    extensions = ClusterExtension(extensions, min_num_reads)
    scaffold_table, extension_info = ExtendScaffolds(scaffold_table, extensions, mappings, min_num_reads, max_mapping_uncertainty)

    print( str(timedelta(seconds=clock())), "Writing output")
    scaffold_table.to_csv(prefix+"_extended_scaffold_table.csv", index=False)
    np.savetxt(prefix+'_used_reads.lst', np.unique(scaffold_table.loc['read' == scaffold_table['type'], 'name']), fmt='%s')
    
    print( str(timedelta(seconds=clock())), "Finished")
    print( "Extended {} scaffolds (left: {}, right:{}).".format(extension_info['count'], extension_info['left'], extension_info['right']) )
    print( "The extensions ranged from {} to {} bases and had a mean length of {}.".format(extension_info['min'], extension_info['max'], extension_info['mean']) )

def MiniGapFinish(assembly_file, read_file, read_format, scaffold_file, output_file):
    if False == output_file:
        if ".gz" == assembly_file[-3:len(assembly_file)]:
            output_file = assembly_file.rsplit('.',2)[0]+"_minigap.fa"
        else:
            output_file = assembly_file.rsplit('.',1)[0]+"_minigap.fa"
        pass
    
    print( str(timedelta(seconds=clock())), "Loading assembly from: {}".format(assembly_file))
    contigs = {}
    with gzip.open(assembly_file, 'rb') if 'gz' == assembly_file.rsplit('.',1)[-1] else open(assembly_file, 'rU') as fin:
        for record in SeqIO.parse(fin, "fasta"):
            contigs[ record.description.split(' ', 1)[0] ] = record.seq
    
    print( str(timedelta(seconds=clock())), "Loading reads from: {}".format(read_file))
    reads = {}
    with gzip.open(read_file, 'rb') if 'gz' == read_file.rsplit('.',1)[-1] else open(read_file, 'rU') as fin:
        for record in SeqIO.parse(fin, read_format):
            reads[ record.description.split(' ', 1)[0] ] = record.seq
            
    print( str(timedelta(seconds=clock())), "Loading scaffold info from: {}".format(scaffold_file))
    scaffold_table = pd.read_csv(scaffold_file)
    
    print( str(timedelta(seconds=clock())), "Writing modified assembly to: {}".format(output_file))
    with gzip.open(output_file, 'wb') if 'gz' == output_file.rsplit('.',1)[-1] else open(output_file, 'w') as fout:
        cur_scaffold = 0
        name = "miniscaffold1"
        out_count = 2
        seq = []
        for row in scaffold_table.itertuples(index=False):
            # Check if scaffold changed
            if cur_scaffold != row.scaffold:
                if cur_scaffold < row.scaffold:
                    # Write scaffold to disc
                    fout.write('>')
                    fout.write(name)
                    fout.write('\n')
                    fout.write(''.join(seq))
                    fout.write('\n')

                    # Start new scaffold
                    cur_scaffold = row.scaffold
                    name = "miniscaffold{}".format(out_count)
                    out_count += 1
                    seq = []
                else:
                    print("Encountered scaffold {} while handling scaffold {}. The loaded scaffold table is invalid.".format(row.scaffold, cur_scaffold) )
            
            # Add sequences to scaffold
            if 'contig' == row.type:
                if row.reverse:
                    seq.append(str(contigs[row.name][row.start:row.end].reverse_complement()))
                else:
                    seq.append(str(contigs[row.name][row.start:row.end]))
            else:
                if row.reverse:
                    seq.append(str(reads[row.name][row.start:row.end].reverse_complement()))
                else:
                    seq.append(str(reads[row.name][row.start:row.end]))
                
            # Add N's to keep original scaffolds
            if 0 <= row.org_dist_right:
                seq.append( 'N' * row.org_dist_right )
                cur_scaffold += 1 # Combine this scaffold with the next
            
        # Write out last scaffold
        fout.write('>')
        fout.write(name)
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
        print("  -f, --format FORMAT       Format of {reads}.fq (fasta/fastq) (Default: fastq)")
        print("  -o, --output FILE.fa      Output file for modified assembly ({assembly}_minigap.fa)")
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
        min_length_contig_break = 600
        min_mapping_length = 400
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
        min_length_contig_break = 1000
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
        try:
            optlist, args = getopt.getopt(argv, 'hf:o:s:', ['help','format=','output=','scaffolds='])
        except getopt.GetoptError:
            print("Unknown option\n")
            Usage(module)
            sys.exit(1)
            
        read_format = 'fastq'
        output = False
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
            elif opt in ("-o", "--output"):
                output = par
            elif opt in ("-s", "--scaffolds"):
                scaffolds = par
        
        if 2 != len(args):
            print("Wrong number of files. Exactly two files are required.\n")
            Usage(module)
            sys.exit(2)
            
        if False == scaffolds:
            print("scaffolds argument is mandatory")
            Usage(module)
            sys.exit(1)

        MiniGapFinish(args[0], args[1], read_format, scaffolds, output)
    elif "visualize" == module:
        try:
            optlist, args = getopt.getopt(argv, 'ho:', ['help','output=','--keepAllSubreads','--minLenBreak=','minMapLength=','minMapQ='])
        except getopt.GetoptError:
            print("Unknown option\n")
            Usage(module)
            sys.exit(1)
            
        output = False
        keep_all_subreads = False
        min_length_contig_break = 600
        min_mapping_length = 400
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
