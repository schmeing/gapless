#!/usr/bin/python

from Bio import SeqIO
from Bio.SeqIO.QualityIO import FastqGeneralIterator
from Bio.SeqIO.FastaIO import SimpleFastaParser

from datetime import timedelta
from functools import reduce
import getopt
import gzip
#from itertools import izip as zip
from matplotlib import use as mpl_use
mpl_use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import os
import pandas as pd
import seaborn as sns
import sys
from time import clock

def GetReadFormat(read_file):
    if 'gz' == read_file.rsplit('.',1)[-1]:
        with gzip.open(read_file, 'rb') as fin:
            start = fin.peek(0).decode("utf-8")[0]
    else:
        with open(read_file, 'rU') as fin:
            start = fin.read(1)
    
    read_format = ''
    if '@' == start:
        read_format='fastq'
    elif '>' == start:
        read_format='fasta'
            
    return read_format

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
                    if 0 < contig_end and chunk[0] == last_scaffold:
                        distances_next.append(int(coords[0]) - contig_end - 1)
                    elif 0 <= contig_end:
                        distances_next.append(-1)

                    contig_end = int(coords[1])
                    last_scaffold = chunk[0]
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

def ReadPaf(file_name):
    return pd.read_csv(file_name, sep='\t', header=None, usecols=range(12), names=['q_name','q_len','q_start','q_end','strand','t_name','t_len','t_start','t_end','matches','alignment_length','mapq'], dtype={'q_len':np.int32, 'q_start':np.int32, 'q_end':np.int32, 't_len':np.int32, 't_start':np.int32, 't_end':np.int32, 'matches':np.int32, 'alignment_length':np.int32, 'mapq':np.int16})

def stackhist(x, y, **kws):
    grouped = pd.groupby(x, y)
    data = [d for _, d in grouped]
    labels = [l for l, _ in grouped]
    plt.hist(data, histtype="barstacked", label=labels)

import matplotlib.patches as patches

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
        ax.set_yscale('log', nonposy='clip')
    else:
        # Set at what range of exponents they are not plotted in exponential format for example (-3,4): [0.001-1000[
        ax.get_yaxis().get_major_formatter().set_powerlimits((-3,4))

    pdf.savefig()

    return

def PlotXY(pdf, xtitle, ytitle, x, y, category=[], logx=False):
    plt.close()

    if logx:    
        y = np.extract(x>0, y)
        if len(category):
            category = np.extract(x>0, category)           
        x = np.log10(np.extract(x>0,x))

    if len(category):
        ax = sns.scatterplot(x, y, hue=category, linewidth=0, alpha = 0.7);
    else:
        ax = sns.scatterplot(x, y, linewidth=0, alpha = 0.7);
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

def MaskRepeatEnds(contigs, repeat_file, contig_ids, max_repeat_extension, min_len_repeat_connection, repeat_len_factor_unique, pdf):
    # Read in minimap 2 repeat table (.paf)
    repeat_table = ReadPaf(repeat_file)
    
    repeat_table = repeat_table[(repeat_table['q_name'] != repeat_table['t_name']) | (repeat_table['q_start'] != repeat_table['t_start'])].copy()  # Ignore repeats on same contig and same position
     
    if pdf:
        tmp_dist = np.minimum(repeat_table['q_start'], repeat_table['q_len']-repeat_table['q_end']) # Repeat distance to contig end
        PlotHist(pdf, "Repeat distance to contig end", "# Repeat mappings", tmp_dist, logx=True, threshold=max_repeat_extension)
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
    con_search = con_search.merge(second_longest, on=['q_id','left_repeat','right_repeat'], how='left').fillna(0)
    con_search['accepted'] = (con_search['rep_len'] >= min_len_repeat_connection) & (con_search['len_second']*repeat_len_factor_unique < con_search['rep_len'])
    if pdf:
        PlotHist(pdf, "Connection length", "# Potential repeat connections", con_search['rep_len'], logx=True, threshold=min_len_repeat_connection)
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
    contigs.iloc[con_search.loc[con_search['left_repeat'], 'q_id'], contigs.columns.get_loc('rep_con_left')] = con_search.loc[con_search['left_repeat'], 't_id'].values
    contigs.iloc[con_search.loc[con_search['left_repeat'], 'q_id'], contigs.columns.get_loc('rep_con_side_left')] = np.where(con_search.loc[con_search['left_repeat'], 'con_left'].values, 'l', 'r')
    contigs.iloc[con_search.loc[con_search['right_repeat'], 'q_id'], contigs.columns.get_loc('rep_con_right')] = con_search.loc[con_search['right_repeat'], 't_id'].values
    contigs.iloc[con_search.loc[con_search['right_repeat'], 'q_id'], contigs.columns.get_loc('rep_con_side_right')] = np.where(con_search.loc[con_search['right_repeat'], 'con_left'].values, 'l', 'r')

    ## Mask repeat ends
    # Keep only repeat positions and sort repeats by contig and start position to merge everything that is closer than max_repeat_extension
    repeat_table = repeat_table[['q_id','q_len','q_start','q_end']].copy()
    repeat_table.sort_values(['q_id','q_start','q_end'], inplace=True)
    repeat_table['group'] = (repeat_table['q_id'] != repeat_table['q_id'].shift(1, fill_value=-1)) | (repeat_table['q_start'] > repeat_table['q_end'].shift(1, fill_value=-1) + max_repeat_extension)
    while sum(repeat_table['group']) < len(repeat_table):
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

        PlotHist(pdf, "% of bases masked", "# Contigs", masked, category=category, catname='length', logy=True)

    return contigs, repeat_table

def ReadMappings(mapping_file, contig_ids, min_mapq, pdf):
    mappings = ReadPaf(mapping_file)

    # Filter low mapping qualities
    if pdf:
        PlotHist(pdf, "Mapping quality", "# Mappings", mappings['mapq'], threshold=min_mapq, logy=True)
    mappings = mappings[min_mapq <= mappings['mapq']].copy()

    mappings['t_id'] = itemgetter(*mappings['t_name'])(contig_ids)

    return mappings

def RemoveUnmappedContigs(contigs, mappings):
    # Schedule contigs for removal that don't have any high quality reads mapping to it
    mapped_reads = np.bincount(mappings['t_id'], minlength=len(contigs))
    contigs.loc[0==mapped_reads, 'remove'] = True

    org_contig_info = {}
    org_contig_info['num'] = {}
    org_contig_info['num']['total'] = len(contigs)
    org_contig_info['num']['removed_total'] = sum(contigs['remove'])
    org_contig_info['num']['removed_no_mapping'] = sum(0==mapped_reads)
    org_contig_info['num']['removed_duplicates'] = org_contig_info['num']['removed_total'] - org_contig_info['num']['removed_no_mapping']
    org_contig_info['num']['masked'] = sum(np.logical_not(contigs['remove']) & (contigs['repeat_mask_right'] == 0))
    org_contig_info['len'] = {}
    org_contig_info['len']['total'] = sum(contigs['length'])
    org_contig_info['len']['removed_total'] = sum(contigs.loc[contigs['remove'], 'length'])
    org_contig_info['len']['removed_no_mapping'] = sum(contigs.loc[0==mapped_reads, 'length'])
    org_contig_info['len']['removed_duplicates'] = org_contig_info['len']['removed_total'] - org_contig_info['len']['removed_no_mapping']
    org_contig_info['len']['masked'] = sum(contigs.loc[np.logical_not(contigs['remove']) & (contigs['repeat_mask_right'] == 0), 'length'])
    org_contig_info['max'] = {}
    org_contig_info['max']['total'] = max(contigs['length'])
    org_contig_info['max']['removed_total'] = max(contigs.loc[contigs['remove'], 'length'])
    org_contig_info['max']['removed_no_mapping'] = max(contigs.loc[0==mapped_reads, 'length'])
    org_contig_info['max']['removed_duplicates'] = max(contigs.loc[contigs['remove'] & (0<mapped_reads), 'length'])
    org_contig_info['max']['masked'] = max(contigs.loc[np.logical_not(contigs['remove']) & (contigs['repeat_mask_right'] == 0), 'length'])

    return contigs, org_contig_info

def RemoveUnanchoredMappings(mappings, contigs, center_repeats, min_mapping_length, pdf, max_dist_contig_end):
    # Filter reads only mapping to repeat ends
    mapping_lengths = np.minimum(
                mappings['t_end'] - np.maximum(mappings['t_start'], contigs['repeat_mask_left'].iloc[mappings['t_id']]),
                np.minimum(mappings['t_end'], contigs['repeat_mask_right'].iloc[mappings['t_id']]) - mappings['t_start'] )
    mappings = mappings[min_mapping_length <= mapping_lengths].copy()

    if pdf:
        mapping_lengths[ 0 > mapping_lengths ] = 0 # Set negative values to zero (Only mapping to repeat)
        PlotHist(pdf, "Mapping length", "# Mappings", np.extract(mapping_lengths < 10*min_mapping_length, mapping_lengths), threshold=min_mapping_length)
        PlotHist(pdf, "Mapping length", "# Mappings", mapping_lengths, threshold=min_mapping_length, logx=True)
                 
        # Plot distance of mappings from contig ends, where the continuing read is longer than the continuing contig
        dists_contig_ends = mappings.loc[mappings['t_start'] < np.where('+' == mappings['strand'], mappings['q_start'], mappings['q_len']-mappings['q_end']), 't_start']
        dists_contig_ends = np.concatenate([dists_contig_ends, 
                                            np.extract( mappings['t_len']-mappings['t_end'] < np.where('+' == mappings['strand'], mappings['q_len']-mappings['q_end'], mappings['q_start']), mappings['t_len']-mappings['t_end'] )] )
        PlotHist(pdf, "Distance to contig end", "# Reads reaching over contig ends", np.extract(dists_contig_ends < 10*max_dist_contig_end, dists_contig_ends), threshold=max_dist_contig_end, logy=True)
        PlotHist(pdf, "Distance to contig end", "# Reads reaching over contig ends", dists_contig_ends, threshold=max_dist_contig_end, logx=True)

    # Find and remove reads only mapping to center repeats (Distance to merge repeats should be larger than min_mapping_length, so we can consider here one repeat at a time)
    repeats = center_repeats.merge(mappings, left_on=['con_id'], right_on=['t_id'], how='left', indicator=True)
    repeats = repeats[repeats['_merge'] == "both"].copy()
    repeats = repeats[ (repeats['start']-min_mapping_length < repeats['t_start']) & (repeats['end']+min_mapping_length > repeats['t_end']) ].copy()
    mappings = mappings[(mappings.merge(repeats[['q_name','q_start']].drop_duplicates(), on=['q_name','q_start'], how='left', indicator=True)['_merge'] == "left_only").values].copy()

    return mappings

def BreakReadsAtAdapters(mappings, adapter_signal_max_dist, pdf):
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

    # Account for left-over adapters
    mappings['read_start'] = 0
    mappings['read_end'] = mappings['q_len']

    adapter = (mappings['next_con'] == mappings['t_id']) & (mappings['next_strand'] != mappings['strand'])
    location_shift = np.abs(np.where('+' == mappings['strand'], mappings['t_end'] - mappings['t_end'].shift(-1, fill_value=0), mappings['t_start'] - mappings['t_start'].shift(-1, fill_value=0)))

    if pdf:
        potential_adapter_dist = location_shift[adapter]
        PlotHist(pdf, "Distance for potential adapter signal", "# Signals", np.extract(potential_adapter_dist < 10*adapter_signal_max_dist, potential_adapter_dist), threshold=adapter_signal_max_dist, logy=True)
        PlotHist(pdf, "Distance for potential adapter signal", "# Signals", potential_adapter_dist, threshold=adapter_signal_max_dist, logx=True)

    adapter = adapter & (location_shift <= adapter_signal_max_dist)

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

def FindBreakPoints(mappings, contigs, max_dist_contig_end, min_mapping_length, min_length_contig_break, max_break_point_distance, min_num_reads, min_factor_alternatives, pdf):
    if pdf:
        loose_reads_ends = mappings[(mappings['t_start'] > max_dist_contig_end) & np.where('+' == mappings['strand'], -1 == mappings['prev_con'], -1 == mappings['next_con'])]
        loose_reads_ends_length = np.where('+' == loose_reads_ends['strand'], loose_reads_ends['q_start']-loose_reads_ends['read_start'], loose_reads_ends['read_end']-loose_reads_ends['q_end'])
        loose_reads_ends = mappings[(mappings['t_len']-mappings['t_end'] > max_dist_contig_end) & np.where('+' == mappings['strand'], -1 == mappings['next_con'], -1 == mappings['prev_con'])]
        loose_reads_ends_length = np.concatenate([loose_reads_ends_length, np.where('+' == loose_reads_ends['strand'], loose_reads_ends['read_end']-loose_reads_ends['q_end'], loose_reads_ends['q_start']-loose_reads_ends['read_start'])])
        PlotHist(pdf, "Loose end length", "# Ends", np.extract(loose_reads_ends_length < 10*min_length_contig_break, loose_reads_ends_length), threshold=min_length_contig_break, logy=True)
        PlotHist(pdf, "Loose end length", "# Ends", loose_reads_ends_length, threshold=min_length_contig_break, logx=True)

    left_breaks = mappings[ (mappings['t_len']-mappings['t_end'] > max_dist_contig_end) &
                             np.where('+' == mappings['strand'],
                                      (0 <= mappings['next_con']) | (mappings['read_end']-mappings['q_end'] > min_length_contig_break),
                                      (0 <= mappings['prev_con']) | (mappings['q_start']-mappings['read_start'] > min_length_contig_break)) ]

    right_breaks = mappings[ (mappings['t_start'] > max_dist_contig_end) &
                              np.where('+' == mappings['strand'],
                                       (0 <= mappings['prev_con']) | (mappings['q_start']-mappings['read_start'] > min_length_contig_break),
                                       (0 <= mappings['next_con']) | (mappings['read_end']-mappings['q_end'] > min_length_contig_break)) ]

    if pdf:
        dists_contig_ends = np.concatenate([left_breaks['t_len']-left_breaks['t_end'], right_breaks['t_start'] ])
        PlotHist(pdf, "Distance to contig end", "# Potential contig breaks", np.extract(dists_contig_ends < 10*max_dist_contig_end, dists_contig_ends), threshold=max_dist_contig_end)
        PlotHist(pdf, "Distance to contig end", "# Potential contig breaks", dists_contig_ends, threshold=max_dist_contig_end, logx=True)

    break_points = pd.DataFrame({ 'contig_id': np.concatenate([ left_breaks['t_id'], right_breaks['t_id'] ]),
                                  'side': np.concatenate([ np.full(len(left_breaks), 'l'), np.full(len(right_breaks), 'r') ]),
                                  'position': np.concatenate([ left_breaks['t_end'], right_breaks['t_start'] ])})
    break_points.sort_values(['contig_id','position'], inplace=True)

    if pdf:
        break_point_dist = (break_points['position'] - break_points['position'].shift(1, fill_value=0))[break_points['contig_id'] == break_points['contig_id'].shift(1, fill_value=-1)]
        PlotHist(pdf, "Break point distance", "# Break point pairs", break_point_dist[break_point_dist <= 10*max_break_point_distance], threshold=max_break_point_distance )
        PlotHist(pdf, "Break point distance", "# Break point pairs", break_point_dist, threshold=max_break_point_distance, logx=True )

    break_points['group'] = ( (break_points['position'] - break_points['position'].shift(1, fill_value=0) > max_break_point_distance) |
                              (break_points['contig_id'] != break_points['contig_id'].shift(1, fill_value=-1)) ).cumsum()

    break_groups = break_points.groupby(['contig_id','group','side']).agg(['size','min','max'])['position'].unstack(fill_value=0)
    break_groups.columns = break_groups.columns.map('_'.join)
    break_groups = break_groups.rename(columns={'size_l':'left_breaks', 'size_r':'right_breaks'})

    break_groups.loc[0==break_groups['left_breaks'],'min_l'] = break_groups.loc[0==break_groups['left_breaks'],'min_r']
    break_groups.loc[0==break_groups['left_breaks'],'max_l'] = break_groups.loc[0==break_groups['left_breaks'],'max_r']
    break_groups.loc[0==break_groups['right_breaks'],'min_r'] = break_groups.loc[0==break_groups['right_breaks'],'min_l']
    break_groups.loc[0==break_groups['right_breaks'],'max_r'] = break_groups.loc[0==break_groups['right_breaks'],'max_l']
    break_groups['start_pos'] = break_groups[['min_l','min_r']].min(axis=1)
    break_groups['end_pos'] = break_groups[['max_l','max_r']].max(axis=1)

    break_groups = break_groups.reset_index().drop(columns=['group','min_l','max_l','min_r','max_r'])

    if pdf:
        PlotHist(pdf, "# Breaking reads", "# Break points", np.concatenate([break_groups.loc[break_groups['left_breaks'] < max(100,10*min_num_reads), 'left_breaks'],break_groups.loc[break_groups['right_breaks'] < max(100,10*min_num_reads), 'right_breaks']]), threshold=min_num_reads, logy=True )
        #PlotHist(pdf, "# Breaking reads", "# Break points", np.concatenate([break_groups['left_breaks'],break_groups['right_breaks']]), threshold=min_num_reads, logx=True )

    break_groups = break_groups[ (min_num_reads <= break_groups['left_breaks']) | (min_num_reads <= break_groups['right_breaks']) ]

    break_groups['tmp'] = break_groups['contig_id'] == break_groups['contig_id'].shift(1, fill_value=-1)
    break_groups['num'] = break_groups[['contig_id', 'tmp']].groupby('contig_id').cumsum().astype(int)
    break_groups.drop(columns=['tmp'], inplace=True)

    break_groups['continuing'] = 0
    for i in range(break_groups['num'].max()+1):
        continues = pd.DataFrame({'contig':np.arange(len(contigs)), 'start':0, 'end':sys.maxsize//2, 'continuing':0})
        continues.iloc[break_groups.loc[i==break_groups['num'], 'contig_id'], [continues.columns.get_loc('start'), continues.columns.get_loc('end')]] = break_groups.loc[i==break_groups['num'], ['start_pos', 'end_pos']].values
        breaks = continues.iloc[mappings['t_id']]
        breaks = breaks[(mappings['t_start'].values <= breaks['start'] - min_mapping_length) & (mappings['t_end'].values >= breaks['end'] + min_mapping_length)]
        breaks = breaks.groupby('contig').size().reset_index(name='counts')
        continues.iloc[breaks['contig'], continues.columns.get_loc('continuing')] = breaks['counts'].values
        break_groups.loc[i==break_groups['num'], 'continuing'] = continues.iloc[break_groups.loc[i==break_groups['num'], 'contig_id'].values, continues.columns.get_loc('continuing')].values

    if pdf:
        PlotXY(pdf, "Continuous read", "Breaking reads", break_groups['continuing'].values, break_groups[['left_breaks','right_breaks']].max(axis=1), category=np.where(break_groups['continuing'].values*min_factor_alternatives <= break_groups[['left_breaks','right_breaks']].max(axis=1), "accepted", "declined") )
        #PlotXY(pdf, "Continuous read", "Breaking reads", break_groups.loc[break_groups['continuing'] < max(100,10*min_num_reads),'continuing'].values, break_groups.loc[break_groups['continuing'] < max(100,10*min_num_reads),['left_breaks','right_breaks']].max(axis=1) )
        #PlotXY(pdf, "Continuous read", "Breaking reads", break_groups['continuing'].values, break_groups[['left_breaks','right_breaks']].max(axis=1), logx=True )

    break_groups = break_groups[ (break_groups['continuing']*min_factor_alternatives <= break_groups['left_breaks']) | (break_groups['continuing']*min_factor_alternatives <= break_groups['right_breaks']) ]

    if pdf:
        break_groups['contig_length'] = contigs.iloc[break_groups['contig_id'], contigs.columns.get_loc('length')].values
        dists_contig_ends = np.concatenate([ break_groups.loc[break_groups['left_breaks']*min_factor_alternatives <= break_groups['right_breaks'], 'end_pos'],
                                            (break_groups['contig_length']-break_groups['start_pos'])[break_groups['left_breaks'] >= break_groups['right_breaks']*min_factor_alternatives] ])
        PlotHist(pdf, "Distance to contig end", "# Single sided breaks", np.extract(dists_contig_ends < 10*max_dist_contig_end, dists_contig_ends), threshold=max_dist_contig_end)
        PlotHist(pdf, "Distance to contig end", "# Single sided breaks", dists_contig_ends, threshold=max_dist_contig_end, logx=True)

    break_groups['num'] = break_groups['contig_id'] == break_groups['contig_id'].shift(1, fill_value=-1)
    break_groups['num'] = break_groups[['contig_id', 'num']].groupby('contig_id').cumsum()
    break_groups['pos'] = break_groups['start_pos'] + (break_groups['end_pos']-break_groups['start_pos']+1)//2

    # Information about not-accepted breaks
    keep_row_left = np.full(len(left_breaks),False)
    keep_row_right = np.full(len(right_breaks),False)
    for i in range(break_groups['num'].max()+1):
        breaks = pd.DataFrame({'contig':np.arange(len(contigs)), 'start':sys.maxsize, 'end':-1})
        breaks.iloc[break_groups.loc[i==break_groups['num'], 'contig_id'], [breaks.columns.get_loc('start'), breaks.columns.get_loc('end')]] = break_groups.loc[i==break_groups['num'], ['start_pos', 'end_pos']].values
        lbreaks = breaks.iloc[left_breaks['t_id']]
        rbreaks = breaks.iloc[right_breaks['t_id']]
        keep_row_left = keep_row_left | ((lbreaks['start'].values <= left_breaks['t_end'].values) & (left_breaks['t_end'].values <= lbreaks['end'].values))
        keep_row_right = keep_row_right | ((rbreaks['start'].values <= right_breaks['t_start'].values) & (right_breaks['t_start'].values <= rbreaks['end'].values))

    spurious_break_indexes = np.unique(np.concatenate([left_breaks[np.logical_not(keep_row_left)].index, right_breaks[np.logical_not(keep_row_right)].index]))

    return break_groups, spurious_break_indexes

def GetContigParts(contigs, break_groups, pdf):
    # Create a dataframe with the contigs being split into parts and contigs scheduled for removal not included
    contig_parts = break_groups.groupby('contig_id').size().reset_index(name='counts')
    contigs['parts'] = 0
    contigs.iloc[contig_parts['contig_id'], contigs.columns.get_loc('parts')] = contig_parts['counts'].values
    contigs['parts'] += 1
    contigs.loc[contigs['remove'], 'parts'] = 0
    
    if pdf:
        category = np.array(["deleted no coverage"]*len(contigs))
        category[:] = "used"
        category[ contigs['parts']>1 ] = "split"
        category[ contigs['repeat_mask_right'] == 0 ] = "masked"
        category[ (contigs['repeat_mask_right'] == 0) & contigs['remove'] ] = "deleted duplicate"
        category[ (contigs['repeat_mask_right'] > 0) & contigs['remove'] ] = "deleted no coverage"

        PlotHist(pdf, "Original contig length", "# Contigs", contigs['length'], category=category, catname="type", logx=True)
        
    contig_parts = pd.DataFrame({'contig':np.repeat(np.arange(len(contigs)), contigs['parts']), 'part':1, 'start':0})
    contig_parts['part'] = contig_parts.groupby('contig').cumsum()['part']-1
    for i in range(break_groups['num'].max()+1):
        contig_parts.loc[contig_parts['part']==i+1, 'start'] = break_groups.loc[break_groups['num']==i,'pos'].values
    contig_parts['end'] = contig_parts['start'].shift(-1,fill_value=0)
    contig_parts.loc[contig_parts['end']==0, 'end'] = contigs.iloc[contig_parts.loc[contig_parts['end']==0, 'contig'], contigs.columns.get_loc('length')].values
    contig_parts['name'] = contigs.iloc[contig_parts['contig'], contigs.columns.get_loc('name')].values

    contigs['first_part'] = -1
    contigs.loc[contigs['remove']==False, 'first_part'] = contig_parts[contig_parts['part']==0].index
    contigs['last_part'] = contigs['first_part'] + contigs['parts'] - 1

    # Assign scaffold info from contigs to contig_parts
    contig_parts['org_dist_left'] = -1
    tmp_contigs = contigs[(contigs['remove']==False) & (contigs['remove'].shift(1, fill_value=True)==False)]
    contig_parts.iloc[tmp_contigs['first_part'].values, contig_parts.columns.get_loc('org_dist_left')] = tmp_contigs['org_dist_left'].values
    contig_parts['org_dist_right'] = -1
    tmp_contigs = contigs[(contigs['remove']==False) & (contigs['remove'].shift(-1, fill_value=True)==False)]
    contig_parts.iloc[tmp_contigs['last_part'].values, contig_parts.columns.get_loc('org_dist_right')] = tmp_contigs['org_dist_right'].values

    # Rescue scaffolds broken by removed contigs
    tmp_contigs = contigs[contigs['remove'] & (contigs['org_dist_right'] != -1) & (contigs['org_dist_left'] != -1)].copy()
    tmp_contigs.reset_index(inplace=True)
    tmp_contigs['group'] = (tmp_contigs['index'] != tmp_contigs['index'].shift(1, fill_value=-1)).cumsum()
    tmp_contigs = tmp_contigs.groupby('group', sort=False)['index','length','org_dist_right','org_dist_left'].agg(['first','last','sum'])
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
    contig_parts['rep_con_left'] = -1
    contig_parts['rep_con_side_left'] = ''
    contig_parts['rep_con_right'] = -1
    contig_parts['rep_con_side_right'] = ''
    tmp_contigs = contigs[(contigs['remove']==False)]
    contig_parts.iloc[tmp_contigs['first_part'].values, contig_parts.columns.get_loc('rep_con_left')] = np.where(tmp_contigs['rep_con_side_left'] == 'l',
                                                          contigs.iloc[tmp_contigs['rep_con_left'].values, contigs.columns.get_loc('first_part')].values,
                                                          contigs.iloc[tmp_contigs['rep_con_left'].values, contigs.columns.get_loc('last_part')].values)
    contig_parts.iloc[tmp_contigs['first_part'].values, contig_parts.columns.get_loc('rep_con_side_left')] = tmp_contigs['rep_con_side_left'].values
    contig_parts.iloc[tmp_contigs['last_part'].values, contig_parts.columns.get_loc('rep_con_right')] = np.where(tmp_contigs['rep_con_side_right'] == 'l',
                                                          contigs.iloc[tmp_contigs['rep_con_right'].values, contigs.columns.get_loc('first_part')].values,
                                                          contigs.iloc[tmp_contigs['rep_con_right'].values, contigs.columns.get_loc('last_part')].values)
    contig_parts.iloc[tmp_contigs['last_part'].values, contig_parts.columns.get_loc('rep_con_side_right')] = tmp_contigs['rep_con_side_right'].values

    # Prepare connection variables
    contig_parts['left_con'] = -1
    contig_parts['left_con_side'] = ''
    contig_parts['left_con_read_name'] = ''
    contig_parts['left_con_read_strand'] = ''
    contig_parts['left_con_read_from'] = -1
    contig_parts['left_con_read_to'] = -1
    contig_parts['right_con'] = -1
    contig_parts['right_con_side'] = ''
    contig_parts['right_con_read_name'] = ''
    contig_parts['right_con_read_strand'] = ''
    contig_parts['right_con_read_from'] = -1
    contig_parts['right_con_read_to'] = -1

    # Mark contig parts that are from contigs which are completely masked due to repeats, but not removed (Cannot be broken as we don't allow reads to map to them)
    contig_parts.iloc[contigs.loc[(contigs['repeat_mask_right'] == 0) & (contigs['remove'] == False), 'first_part'].values, contig_parts.columns.get_loc('left_con')] = -2
    contig_parts.iloc[contigs.loc[(contigs['repeat_mask_right'] == 0) & (contigs['remove'] == False), 'first_part'].values, contig_parts.columns.get_loc('right_con')] = -2

    return contig_parts, contigs

def UpdateMappingsToContigParts(mappings, contigs, contig_parts, break_groups, min_mapping_length):
    # Insert information on contig parts
    mappings['left_part'] = contigs.iloc[mappings['t_id'], contigs.columns.get_loc('first_part')].values
    for i in range(break_groups['num'].max()+1):
        break_points = np.full(len(contigs), sys.maxsize)
        break_points[ break_groups.loc[break_groups['num']==i,'contig_id'] ] = break_groups.loc[break_groups['num']==i,'start_pos'] - min_mapping_length
        break_points = break_points[mappings['t_id']]
        mappings.loc[mappings['t_start'] > break_points , 'left_part'] += 1
    mappings['right_part'] = contigs.iloc[mappings['t_id'], contigs.columns.get_loc('last_part')].values
    for i in range(break_groups['num'].max()+1):
        break_points = np.full(len(contigs), -1)
        break_points[ break_groups.loc[break_groups['num']==i,'contig_id'] ] = break_groups.loc[break_groups['num']==i,'end_pos'] + min_mapping_length
        break_points = break_points[mappings['t_id']]
        mappings.loc[mappings['t_end'] < break_points , 'right_part'] -= 1
        
    # Remove super short mappings right over the break points that cannot be attributed to parts properly
    mappings = mappings[ mappings['left_part'] <= mappings['right_part'] ].copy()

    # Update mappings after the removals
    mappings['next_strand'] = mappings['strand'].shift(-1, fill_value='')
    mappings['next_con'] = np.where('+' == mappings['next_strand'], mappings['left_part'].shift(-1, fill_value=-1), mappings['right_part'].shift(-1, fill_value=-1))
    mappings.loc[(mappings['read_end'].shift(-1, fill_value=-1) != mappings['read_end']) | (mappings['q_name'].shift(-1, fill_value='') != mappings['q_name']), 'next_con'] = -1
    mappings['prev_strand'] = mappings['strand'].shift(1, fill_value='')
    mappings['prev_con'] = np.where('-' == mappings['prev_strand'], mappings['left_part'].shift(1, fill_value=-1), mappings['right_part'].shift(1, fill_value=-1))
    mappings.loc[(mappings['read_end'].shift(1, fill_value=-1) != mappings['read_end']) | (mappings['q_name'].shift(1, fill_value='') != mappings['q_name']), 'prev_con'] = -1
    
    mappings['left_con'] = np.where('+' == mappings['strand'], mappings['prev_con'], mappings['next_con'])
    mappings['right_con'] = np.where('-' == mappings['strand'], mappings['prev_con'], mappings['next_con'])
    
    mappings['next_side'] = np.where('+' == mappings['next_strand'], 'l', 'r')
    mappings['prev_side'] = np.where('-' == mappings['prev_strand'], 'l', 'r')
    mappings['left_con_side'] = np.where('+' == mappings['strand'], mappings['prev_side'], mappings['next_side'])
    mappings.loc[-1 == mappings['left_con'], 'left_con_side'] = ''
    mappings['right_con_side'] = np.where('-' == mappings['strand'], mappings['prev_side'], mappings['next_side'])
    mappings.loc[-1 == mappings['right_con'], 'right_con_side'] = ''

    mappings.drop(columns=['next_con','prev_con','next_strand','prev_strand','next_side','prev_side'], inplace=True)

    # Break mappings at break points and restructure 
    mappings['parts'] = mappings['right_part'] - mappings['left_part'] + 1
    mappings = mappings.loc[np.repeat(mappings.index,mappings['parts'])].copy()
    mappings['num_part'] = 1
    mappings['num_part'] = mappings.groupby(level=0)['num_part'].cumsum()-1
    mappings.loc[mappings['strand'] == '-', 'num_part'] = mappings.loc[mappings['strand'] == '-', 'parts'] - mappings.loc[mappings['strand'] == '-', 'num_part'] - 1 # Invert the order in case read is on the opposite strand
    mappings['conpart'] = mappings['left_part'] + mappings['num_part']
    mappings.loc[0 < mappings['num_part'], 'left_con'] = mappings.loc[0 < mappings['num_part'], 'conpart'] - 1
    mappings.loc[0 < mappings['num_part'], 'left_con_side'] = 'r'
    mappings.loc[mappings['parts'] - mappings['num_part'] > 1, 'right_con'] = mappings.loc[mappings['parts'] - mappings['num_part'] > 1, 'conpart'] + 1
    mappings.loc[mappings['parts'] - mappings['num_part'] > 1, 'right_con_side'] = 'l'
    mappings['con_from'] = mappings['t_start']
    mappings['con_to'] = mappings['t_end']
    mappings.loc[0 < mappings['num_part'], 'con_from'] = contig_parts.iloc[mappings.loc[0 < mappings['num_part'], 'conpart'], contig_parts.columns.get_loc('start')].values
    mappings.loc[mappings['parts'] - mappings['num_part'] > 1, 'con_to'] = contig_parts.iloc[mappings.loc[mappings['parts'] - mappings['num_part'] > 1, 'conpart'], contig_parts.columns.get_loc('end')].values
    mappings['read_from'] = mappings['q_start']
    mappings['read_to'] = mappings['q_end']
    
    # Rough estimate of the split for the reads, better not use the new information, but maybe not always avoidable (it's important that read_to and read_from of the next entry match, so that using this mappings avoids the rough estimate by not adding or removing any sequences)
    # Forward strand
    multi_maps = mappings[(0 < mappings['num_part']) & (mappings['strand'] == '+')]
    mappings.loc[(0 < mappings['num_part']) & (mappings['strand'] == '+'), 'read_from'] = np.round(multi_maps['q_end'] - (multi_maps['q_end']-multi_maps['q_start'])/(multi_maps['t_end']-multi_maps['t_start'])*(multi_maps['t_end']-multi_maps['con_from'])).astype(int)
    mappings['read_to'] = np.where((mappings['parts'] - mappings['num_part'] > 1) & (mappings['strand'] == '+'), mappings['read_from'].shift(-1, fill_value = 0), mappings['read_to'])
    # Reverse strand
    multi_maps = mappings[(0 < mappings['num_part']) & (mappings['strand'] == '-')]
    mappings.loc[(0 < mappings['num_part']) & (mappings['strand'] == '-'), 'read_to'] = np.round(multi_maps['q_start'] + (multi_maps['q_end']-multi_maps['q_start'])/(multi_maps['t_end']-multi_maps['t_start'])*(multi_maps['t_end']-multi_maps['con_from'])).astype(int)
    mappings['read_from'] = np.where((mappings['parts'] - mappings['num_part'] > 1) & (mappings['strand'] == '-'), mappings['read_to'].shift(1, fill_value = 0), mappings['read_from'])

    mappings.reset_index(inplace=True)
    mappings['identity'] = mappings['matches'].astype(float)/mappings['alignment_length']
    mappings.rename(columns={'q_name':'read_name'}, inplace=True)
    mappings = mappings[['read_name', 'read_start', 'read_end', 'read_from', 'read_to', 'strand', 'conpart', 'con_from', 'con_to', 'left_con', 'left_con_side', 'right_con', 'right_con_side', 'mapq', 'identity']].copy()

    # Count how many mappings each read has
    mappings['num_mappings'] = 1
    num_mappings = mappings.groupby(['read_name','read_start'], sort=False)['num_mappings'].size().values
    mappings['num_mappings'] = np.repeat(num_mappings, num_mappings)

    return mappings

def CreateBridges(left_bridge, right_bridge):
    bridges = pd.concat([left_bridge[['from', 'from_side', 'to', 'to_side', 'from_count', 'to_count','mapq']],
                         right_bridge[['from', 'from_side', 'to', 'to_side', 'from_count', 'to_count','mapq']]], ignore_index=True, sort=False)

    bridges.sort_values(['from','from_side','to','to_side','mapq'], ascending=[True, True, True, True, False], inplace=True)
    bridges = bridges.groupby(['from','from_side','to','to_side','mapq'], sort=False).sum().reset_index()
    bridges['from_cumcount'] = 0
    bridges['to_cumcount'] = 0
    bridges.loc[:,['from_cumcount', 'to_cumcount']] = bridges.groupby(['from','from_side','to','to_side'], sort=False)['from_count','to_count'].cumsum().values
    # Duplicate bridges and switch from to, so that both sortings have all relevant bridges
    bridges = pd.concat([bridges[['from', 'from_side', 'to', 'to_side', 'from_cumcount', 'to_cumcount', 'mapq']],
                         bridges.rename(columns={'from':'to', 'to':'from', 'from_side':'to_side', 'to_side':'from_side', 'from_cumcount':'to_cumcount', 'to_cumcount':'from_cumcount'})[['from', 'from_side', 'to', 'to_side', 'from_cumcount', 'to_cumcount', 'mapq']]], ignore_index=True, sort=False)

    return bridges

def FilterBridges(bridges, min_factor_alternatives, pdf=None):
    # Remove bridges being lower than min_factor_alternatives compared to the maximum bridge for that contig side
    bridges.sort_values(['from','from_side','mapq','to','to_side'], ascending=[True, True, False, True, True], inplace=True)
    max_count = bridges.groupby(['from','from_side','mapq'], sort=False)['from_cumcount'].agg(['max','size']).reset_index()
    bridges['from_maxcount'] = np.repeat(max_count['max'], max_count['size']).values
    bridges['from_maxcount'] = bridges.groupby(['from','from_side'], sort=False)['from_maxcount'].cummax()

    bridges.sort_values(['to','to_side','mapq','from','from_side'], ascending=[True, True, False, True, True], inplace=True)
    max_count = bridges.groupby(['to','to_side','mapq'], sort=False)['to_cumcount'].agg(['max','size']).reset_index()
    bridges['to_maxcount'] = np.repeat(max_count['max'], max_count['size']).values
    bridges['to_maxcount'] = bridges.groupby(['to','to_side'], sort=False)['to_maxcount'].cummax()

    if pdf:
        alternative_bridges = bridges[ bridges['from_maxcount'] > bridges['from_cumcount'] ]
        alt_ratio = alternative_bridges['from_maxcount'] / alternative_bridges['from_cumcount']
        PlotXY(pdf,  "# Reads main connection", "# Reads alternative connection", alternative_bridges['from_maxcount'], alternative_bridges['from_cumcount'], category=np.where(alt_ratio >= min_factor_alternatives, "accepted", "declined"))

    bridges = bridges[ (bridges['from_cumcount']*min_factor_alternatives > bridges['from_maxcount']) &
                       (bridges['to_cumcount']*min_factor_alternatives > bridges['to_maxcount']) ].copy()

    # Remove bridges where bridges with higher mapping quality exist for the same contig side
    bridges = bridges[['from', 'from_side', 'to', 'to_side', 'from_cumcount', 'to_cumcount', 'mapq']].groupby(['to','to_side','from','from_side']).max().reset_index()
    max_mapq = bridges.groupby(['to','to_side'], sort=False)['mapq'].agg(['max','size']).reset_index()
    bridges['from_max_mapq'] = np.repeat(max_mapq['max'], max_mapq['size']).values

    bridges.sort_values(['from','from_side','to', 'to_side'], inplace=True)
    max_mapq = bridges.groupby(['from','from_side'], sort=False)['mapq'].agg(['max','size']).reset_index()
    bridges['to_max_mapq'] = np.repeat(max_mapq['max'], max_mapq['size']).values
    bridges = bridges[(bridges['mapq'] == bridges['from_max_mapq']) & (bridges['mapq'] == bridges['to_max_mapq'])].copy()

    bridges.drop(columns=['from_max_mapq','to_max_mapq'], inplace=True)
    return bridges

def GetBridges(mappings, min_factor_alternatives, min_num_reads, pdf):
    # Get bridges
    left_bridge = mappings.loc[mappings['left_con'] >= 0, ['conpart','left_con','left_con_side','mapq']].groupby(['conpart','left_con','left_con_side','mapq']).size().reset_index(name='count')
    left_bridge['from'] = left_bridge['conpart']
    left_bridge['from_side'] = 'l'
    left_bridge['to'] = left_bridge['left_con']
    left_bridge['to_side'] = left_bridge['left_con_side']
    left_bridge['from_count'] = left_bridge['count']
    left_bridge['to_count'] = 0
    left_bridge.loc[left_bridge['conpart'] > left_bridge['left_con'], ['from', 'from_side', 'to', 'to_side', 'from_count', 'to_count']] = left_bridge.loc[left_bridge['conpart'] > left_bridge['left_con'], ['to', 'to_side', 'from', 'from_side', 'to_count', 'from_count']].values

    right_bridge = mappings.loc[mappings['right_con'] >= 0, ['conpart','right_con','right_con_side','mapq']].groupby(['conpart','right_con','right_con_side','mapq']).size().reset_index(name='count')
    right_bridge['from'] = right_bridge['conpart']
    right_bridge['from_side'] = 'r'
    right_bridge['to'] = right_bridge['right_con']
    right_bridge['to_side'] = right_bridge['right_con_side']
    right_bridge['from_count'] = right_bridge['count']
    right_bridge['to_count'] = 0
    right_bridge.loc[right_bridge['conpart'] >= right_bridge['right_con'], ['from', 'from_side', 'to', 'to_side', 'from_count', 'to_count']] = right_bridge.loc[right_bridge['conpart'] >= right_bridge['right_con'], ['to', 'to_side', 'from', 'from_side', 'to_count', 'from_count']].values # >= here so that contig loops are also sorted properly

    bridges = CreateBridges(left_bridge, right_bridge)
    bridges = FilterBridges(bridges, min_factor_alternatives)

    # Get contigs with loops (contigs bridging to itself)
    loop_contigs = np.unique(bridges.loc[(bridges['from']==bridges['to']) & ((bridges['from_cumcount'] >= min_num_reads) | (bridges['to_cumcount'] >= min_num_reads)), ['from']])

    # Repeat bridge creation and filtering but this time without the loops
    bridges = CreateBridges(left_bridge, right_bridge)
    bridges = bridges[bridges['from'] != bridges['to']]
    bridges = FilterBridges(bridges, min_factor_alternatives, pdf)

    # Count alternative bridges for each contig site
    alternatives = bridges.groupby(['from','from_side'], sort=False).size().values
    bridges['from_alt'] = np.repeat(alternatives, alternatives)
    bridges.sort_values(['to', 'to_side','from','from_side'], inplace=True)
    alternatives = bridges.groupby(['to','to_side'], sort=False).size().values
    bridges['to_alt'] = np.repeat(alternatives, alternatives)

    if pdf:
        PlotHist(pdf, "# Supporting reads", "# Unambigouos connections", bridges.loc[ bridges['from_alt'] == 1, 'from_cumcount' ], threshold=min_num_reads, logx=True)
        PlotHist(pdf, "# Supporting reads", "# Unambigouos connections", bridges.loc[ (bridges['from_alt'] == 1) & (bridges['from_cumcount'] <= 10*min_num_reads), 'from_cumcount' ], threshold=min_num_reads)
        PlotHist(pdf, "# Alternative connections", "# Connections", bridges['from_alt'], logy=True)

    return bridges, loop_contigs

def GetLoopMappings(mappings, bridges, loop_contigs_with_exit, invalid_anchors):
    # Searching for reads that cross the whole repeat section
    loop_mappings = mappings[mappings['num_mappings']>=3].copy()
    loop_mappings['repeat'] = np.isin(loop_mappings['conpart'], loop_contigs_with_exit)
    has_repeat = loop_mappings.groupby(['read_name','read_start'], sort=False)['repeat'].agg(['max','size']).reset_index()
    loop_mappings = loop_mappings[np.repeat(has_repeat['max'], has_repeat['size']).astype(bool).values].copy()

    # Remove reads that have not at least two valid anchors
    loop_mappings['anchor'] = np.logical_not(np.isin(loop_mappings['conpart'], invalid_anchors))
    anchor_totcount = loop_mappings.groupby(['read_name','read_start'], sort=False)['anchor'].agg(['sum','size']).reset_index()
    loop_mappings = loop_mappings[np.repeat(anchor_totcount['sum'], anchor_totcount['size']).values > 1]

    # Remove repeats that are not in between anchors
    anchor_totcount = loop_mappings.groupby(['read_name','read_start'], sort=False)['anchor'].agg(['sum','size']).reset_index()
    anchor_totcount = np.repeat(anchor_totcount['sum'], anchor_totcount['size']).values
    anchor_cumcount = loop_mappings.groupby(['read_name','read_start'], sort=False)['anchor'].cumsum()
    loop_mappings = loop_mappings[loop_mappings['anchor'] | (anchor_cumcount > 0) & (anchor_cumcount < anchor_totcount)].copy()

    # Remove anchors that are in direct contant only to other anchors
    # Duplicate anchors that connect to two repeats and assign groups going from anchor to anchor
    # Use !anchor instead of repeat, because repeats without exit are not in repeat here, but certainly aren't an anchor
    connected_repeats = (2 - loop_mappings.groupby(['read_name','read_start'], sort=False)['anchor'].shift(-1, fill_value=True) -
                             loop_mappings.groupby(['read_name','read_start'], sort=False)['anchor'].shift(1, fill_value=True))
    connected_repeats[np.logical_not(loop_mappings['anchor'])] = 1
    loop_mappings = loop_mappings.loc[np.repeat(loop_mappings.index, connected_repeats)].copy()
    loop_mappings['group'] = (loop_mappings['anchor'].cumsum() + 1)//2-1

    # Remove anchored groups that don't have a repeat with exit in it, but only one without exit
    has_repeat = loop_mappings.groupby(['group'], sort=False)['repeat'].agg(['max','size']).reset_index()
    loop_mappings = loop_mappings[np.repeat(has_repeat['max'], has_repeat['size']).astype(bool).values].copy()

    # Remove anchored groups with invalid connections (connections between repeats cannot be checked: Check from start anchor to first repeat and from end anchor to last repeat)
    start_anchor = loop_mappings[loop_mappings.groupby('group', sort=False)['repeat'].cumsum() == 0].copy()
    start_anchor['from'] = start_anchor['conpart']
    start_anchor['from_side'] = np.where(start_anchor['strand'] == '+', 'r', 'l')
    start_anchor['to'] = np.where(start_anchor['strand'] == '+', start_anchor['right_con'], start_anchor['left_con'])
    start_anchor['to_side'] = np.where(start_anchor['strand'] == '+', start_anchor['right_con_side'], start_anchor['left_con_side'])
    start_anchor = start_anchor[['group', 'from', 'from_side', 'to', 'to_side']].copy()
    start_anchor['valid'] = (start_anchor.merge(bridges[['from','from_side','to','to_side']], on=['from','from_side','to','to_side'], how='left', indicator=True)['_merge'] != "left_only").values
    start_anchor = start_anchor.groupby('group')['valid'].min().reset_index(name='valid')

    end_anchor = loop_mappings.groupby('group', sort=False)['repeat'].agg(['sum','size'])
    end_anchor = loop_mappings[np.logical_not(loop_mappings['repeat']) & (loop_mappings.groupby('group', sort=False)['repeat'].cumsum().values == np.repeat(end_anchor['sum'].values,end_anchor['size']))].copy()
    end_anchor['from'] = end_anchor['conpart']
    end_anchor['from_side'] = np.where(end_anchor['strand'] == '-', 'r', 'l')
    end_anchor['to'] = np.where(end_anchor['strand'] == '-', end_anchor['right_con'], end_anchor['left_con'])
    end_anchor['to_side'] = np.where(end_anchor['strand'] == '-', end_anchor['right_con_side'], end_anchor['left_con_side'])
    end_anchor = end_anchor[['group', 'from', 'from_side', 'to', 'to_side']].copy()
    end_anchor['valid'] = (end_anchor.merge(bridges[['from','from_side','to','to_side']], on=['from','from_side','to','to_side'], how='left', indicator=True)['_merge'] != "left_only").values
    end_anchor = end_anchor.groupby('group')['valid'].min().reset_index(name='valid')

    loop_mappings = loop_mappings[np.repeat(start_anchor['valid'].values & end_anchor['valid'].values, loop_mappings.groupby('group', sort=False).size().values)].copy()

    return loop_mappings

def CollectConnections(con_mappings):
    # Restructure con_mappings to see the possible ways
    # Copy and revert mappings + order as connections are always in both directions and not just the one mapping is directed in
    connections = con_mappings[['group','conpart','strand']].copy()

    connections_reversed = con_mappings.sort_values(['group','read_from'], ascending=[True,False])[['group','conpart','strand']]
    connections_reversed['strand'] = np.where(connections_reversed['strand'] == '+', '-', np.where(connections_reversed['strand'] == '-', '+', ''))
    connections_reversed['group'] += connections['group'].max() + 1

    connections = pd.concat([connections, connections_reversed])

    # Reduce repeats
    connections['pos'] = ((connections['group'] != connections['group'].shift(1, fill_value=-1)) |
                             (connections['conpart'] != connections['conpart'].shift(1, fill_value=-1)) |
                             (connections['strand'] != connections['strand'].shift(1, fill_value='')))
    connections['pos'] = connections['pos'].cumsum()
    connections = connections.groupby(connections.columns.to_list(), sort=False).size().reset_index(name='repeats')

    # Reshape connections
    connections['pos'] = 1
    connections['pos'] = connections.groupby('group', sort=False)['pos'].cumsum()
    connections.loc[connections['pos'] == 1,'pos'] = 0 # Start anker is position 0
    # End anker is position 1
    connections.loc[connections['pos'] == np.repeat(connections.groupby('group', sort=False)['pos'].max().values, connections.groupby('group', sort=False)['pos'].size().values),'pos'] = 1
    connections = connections.pivot(index='group', columns='pos').fillna(-1)
    connections.columns = [x[0]+str(x[1]-1) for x in connections.columns.values]

    connections.rename(columns={'conpart-1':'conpart_s', 'conpart0':'conpart_e', 'strand-1':'strand_s', 'strand0':'strand_e'}, inplace=True)
    connections.drop(columns=['repeats-1', 'repeats0'], inplace=True)

    connections = connections.groupby(connections.columns.tolist()).size().reset_index(name='count')

    # Get number of alternative connections from start to end anker
    connections.sort_values(['conpart_s','strand_s'], inplace=True)
    alternatives = connections.groupby(['conpart_s','strand_s'], sort=False).size().values
    connections['alternatives_s'] = np.repeat(alternatives, alternatives)
    connections.sort_values(['conpart_e','strand_e'], inplace=True)
    alternatives = connections.groupby(['conpart_e','strand_e'], sort=False).size().values
    connections['alternatives_e'] = np.repeat(alternatives, alternatives)

    return connections

def ExtendMappingInfo(selected_mappings):
    # Get minimum identity for groups (sorting criteria: If multiple reads support a connection take the one with the highest minimum identity)
    min_identity = selected_mappings.groupby('group', sort=False)['identity'].agg(['min','size'])
    selected_mappings['min_identity'] = np.repeat(min_identity['min'].values, min_identity['size'].values)

    # Get whether a group is reverse from bridge order (bridge order: conpart_s < conpart_e, if equal '+' for strand_s)
    tmp_mappings = selected_mappings.groupby('group', sort=False)['conpart','strand'].agg(['first','last','size']).reset_index()
    tmp_mappings['size'] = tmp_mappings[('conpart','size')]
    tmp_mappings['reverse'] = (tmp_mappings[('conpart','first')] > tmp_mappings[('conpart','last')]) | (tmp_mappings[('conpart','first')] == tmp_mappings[('conpart','last')]) & (tmp_mappings[('strand','first')] == '+')
    selected_mappings['reverse'] = np.repeat(tmp_mappings['reverse'].values, tmp_mappings['size'].values)

    # Get if the mapping is the first or the last of the group (in the bridge order, so potentially reverse it)
    selected_mappings['first'] = (selected_mappings['group'] != selected_mappings['group'].shift(1, fill_value = -1))
    selected_mappings['last'] = (selected_mappings['group'] != selected_mappings['group'].shift(-1, fill_value = -1))
    tmp = np.where(selected_mappings['reverse'], selected_mappings['last'], selected_mappings['first'])
    selected_mappings['last'] = np.where(selected_mappings['reverse'], selected_mappings['first'], selected_mappings['last'])
    selected_mappings['first'] = tmp

    return selected_mappings

def HandleNegativeGaps(conns):
    # Handle negative gaps (overlapping contig parts): Cut contigs and remove read_information
    conns['con_end'] = np.where(conns['gap_from'] < conns['gap_to'], conns['con_end'], conns['con_end'] + np.where(conns['side'] == 'l', 1, -1) * (conns['gap_from'] - conns['gap_to'])//2)
    conns['read_name'] = np.where(conns['gap_from'] < conns['gap_to'], conns['read_name'], '')
    conns['strand'] = np.where(conns['gap_from'] < conns['gap_to'], conns['strand'], '')
    conns['gap_to'] = np.where(conns['gap_from'] < conns['gap_to'], conns['gap_to'], 0)
    conns['gap_from'] = np.where(conns['gap_from'] < conns['gap_to'], conns['gap_from'], 0)

    return conns

def GetReadInfoForBridges(selected_bridges, selected_mappings):
    # Find best read matching the connection/bridge for each direction
    bridging_reads = selected_mappings[np.isin(selected_mappings['conpart'],selected_bridges['conpart_s'])].groupby('group')[['conpart','strand','min_identity']].agg(['first','size'])[[('conpart','first'), ('strand','first'), ('conpart','size'), ('min_identity','first')]].reset_index()
    bridging_reads.columns = ['group','conpart','strand','size','min_identity']
    bridging_reads = bridging_reads.sort_values(['conpart','strand','size','min_identity']).groupby(['conpart','strand'], sort=False).last().reset_index()
    # Select the better read from the two directions
    bridge_to = {(f,s): t for f, s, t in zip(selected_bridges['conpart_s'], selected_bridges['strand_s'], selected_bridges['conpart_e'])}
    bridging_reads['other_conpart'] = itemgetter(*zip(bridging_reads['conpart'],bridging_reads['strand']))(bridge_to)
    bridge_to = {(f,s): t for f, s, t in zip(selected_bridges['conpart_s'], selected_bridges['strand_s'], selected_bridges['strand_e'])}
    bridging_reads['other_strand'] = itemgetter(*zip(bridging_reads['conpart'],bridging_reads['strand']))(bridge_to)
    bridging_reads['other_strand'] = np.where(bridging_reads['other_strand'] == '+', '-', '+') # Invert the strand as the read from the other direction has the opposite strand and we want them to be equal if they belong to the same connection
    bridging_reads['low_conpart'] = np.minimum(bridging_reads['conpart'], bridging_reads['other_conpart'])
    bridging_reads['low_strand'] = np.where(bridging_reads['conpart'] < bridging_reads['other_conpart'], bridging_reads['strand'], bridging_reads['other_strand'])
    bridging_reads = bridging_reads.sort_values(['low_conpart','low_strand','size','min_identity']).groupby(['low_conpart','low_strand'], sort=False).last().reset_index()
    
    selected_bridges = selected_bridges.merge(bridging_reads[['group', 'conpart', 'strand']], left_on=['conpart_s','strand_s'], right_on=['conpart','strand'], how='inner').drop(columns=['conpart','strand'])
    
    # Get the read information
    selected_bridges['id'] = selected_bridges.index
    selected_bridges.rename(columns={'conpart_s':'conpart0', 'conpart_e':'conpart'+str(max([int(col[7:]) for col in selected_bridges if col.startswith('conpart') and not col.startswith('conpart_')])+1)}, inplace=True)
    selected_bridges = selected_bridges.melt(id_vars=['id','group'], value_vars=[col for col in selected_bridges if col.startswith('conpart')], var_name='pos', value_name='conpart')
    selected_bridges = selected_bridges[selected_bridges['conpart'] != -1.0]
    selected_bridges['pos'] = selected_bridges['pos'].str[7:].astype(int)
    
    selected_bridges = selected_bridges.merge(selected_mappings[['group','read_name','read_from', 'read_to', 'strand', 'conpart', 'con_from','con_to','identity','reverse','first','last']], on=['group','conpart'])
    
    # Remove the duplicated entries, when a group maps to the contig part multiple times (Keep the ones that really represent the correct read order)
    selected_bridges.sort_values(['group','conpart','pos'], inplace=True)
    selected_bridges['bridge_order'] = selected_bridges.groupby(['group','conpart'], sort=False)['pos'].shift(1, fill_value=-1) != selected_bridges['pos']
    selected_bridges['bridge_order'] = selected_bridges.groupby(['group','conpart'], sort=False)['bridge_order'].cumsum()
    selected_bridges.sort_values(['group','conpart','read_from'], inplace=True)
    selected_bridges['read_order'] = selected_bridges.groupby(['group','conpart'], sort=False)['read_from'].shift(1, fill_value=-1) != selected_bridges['read_from']
    selected_bridges['read_order'] = selected_bridges.groupby(['group','conpart'], sort=False)['read_order'].cumsum()
    selected_bridges.sort_values(['group','conpart','read_from'], ascending=[True,True,False], inplace=True)
    selected_bridges['read_order_rev'] = selected_bridges.groupby(['group','conpart'], sort=False)['read_from'].shift(1, fill_value=-1) != selected_bridges['read_from']
    selected_bridges['read_order_rev'] = selected_bridges.groupby(['group','conpart'], sort=False)['read_order_rev'].cumsum()
    
    selected_bridges = selected_bridges[np.where(selected_bridges['reverse'], selected_bridges['bridge_order'] == selected_bridges['read_order_rev'], selected_bridges['bridge_order'] == selected_bridges['read_order'])].copy()

    # If a contig part is in multiple connections drop all except the one with the best identity (always keep first and last to properly handle loops)
    selected_bridges['protector'] = (selected_bridges['first'] | selected_bridges['last']).cumsum()
    selected_bridges.loc[np.logical_not(selected_bridges['first'] | selected_bridges['last']), 'protector'] = 0
    anchored_contig_parts = np.unique(selected_bridges.loc[selected_bridges['protector']==0,'conpart']).astype(int)
    selected_bridges = selected_bridges.sort_values(['conpart','protector','identity']).groupby(['conpart','protector'], sort=False).last().reset_index()
    selected_bridges.sort_values(['group','read_from'], inplace=True)
    
    selected_bridges = selected_bridges[['id','conpart','read_name','read_from','read_to','strand','con_from','con_to']].copy()
    
    # Restructure to contig part info
    selected_bridges['conpart'] = selected_bridges['conpart'].astype(int)
    selected_bridges['prev_conpart'] = selected_bridges.groupby('id')['conpart'].shift(1,fill_value=-1)
    selected_bridges['next_conpart'] = selected_bridges.groupby('id')['conpart'].shift(-1,fill_value=-1)
    selected_bridges['prev_from'] = selected_bridges.groupby('id')['read_from'].shift(1,fill_value=-1)
    selected_bridges['next_from'] = selected_bridges.groupby('id')['read_from'].shift(-1,fill_value=-1)
    selected_bridges['prev_to'] = selected_bridges.groupby('id')['read_to'].shift(1,fill_value=-1)
    selected_bridges['next_to'] = selected_bridges.groupby('id')['read_to'].shift(-1,fill_value=-1)
    selected_bridges['prev_strand'] = selected_bridges.groupby('id')['strand'].shift(1,fill_value=-1)
    selected_bridges['next_strand'] = selected_bridges.groupby('id')['strand'].shift(-1,fill_value=-1)
    
    selected_bridges = selected_bridges.loc[np.repeat(selected_bridges.index,2)].reset_index().drop(columns=['index','id'])
    selected_bridges['type'] = ['prev','next']*(len(selected_bridges)//2)
    selected_bridges['side'] = np.where(selected_bridges['type'] == 'prev', np.where(selected_bridges['strand'] == '+', 'l', 'r'), np.where(selected_bridges['strand'] == '+', 'r', 'l'))
    selected_bridges['con_end'] = np.where(selected_bridges['side'] == 'l', selected_bridges['con_from'], selected_bridges['con_to'])
    selected_bridges['to_conpart'] = np.where(selected_bridges['type'] == 'prev', selected_bridges['prev_conpart'], selected_bridges['next_conpart'])
    selected_bridges['to_side'] = np.where(selected_bridges['type'] == 'prev', np.where(selected_bridges['prev_strand'] == '+', 'r', 'l'), np.where(selected_bridges['next_strand'] == '+', 'l', 'r'))
    selected_bridges['gap_from'] = np.where(selected_bridges['type'] == 'prev', selected_bridges['prev_to'], selected_bridges['read_to'])
    selected_bridges['gap_to'] = np.where(selected_bridges['type'] == 'prev', selected_bridges['read_from'], selected_bridges['next_from'])
    selected_bridges = selected_bridges.loc[selected_bridges['to_conpart'] != -1, ['conpart','side','con_end','to_conpart','to_side','read_name','gap_from','gap_to','strand']].copy()
    
    selected_bridges = HandleNegativeGaps(selected_bridges)

    return selected_bridges, anchored_contig_parts

def InsertBridges(contig_parts, selected_bridges):
    # Left sides
    contig_parts.iloc[selected_bridges.loc[selected_bridges['side']=='l','conpart'], contig_parts.columns.get_loc('left_con')] = selected_bridges.loc[selected_bridges['side']=='l', 'to_conpart'].values
    contig_parts.iloc[selected_bridges.loc[selected_bridges['side']=='l','conpart'], contig_parts.columns.get_loc('left_con_side')] = selected_bridges.loc[selected_bridges['side']=='l', 'to_side'].values
    contig_parts.iloc[selected_bridges.loc[selected_bridges['side']=='l','conpart'], contig_parts.columns.get_loc('start')] = selected_bridges.loc[selected_bridges['side']=='l', 'con_end'].values
    contig_parts.iloc[selected_bridges.loc[selected_bridges['side']=='l','conpart'], contig_parts.columns.get_loc('left_con_read_name')] = selected_bridges.loc[selected_bridges['side']=='l', 'read_name'].values
    contig_parts.iloc[selected_bridges.loc[selected_bridges['side']=='l','conpart'], contig_parts.columns.get_loc('left_con_read_strand')] = selected_bridges.loc[selected_bridges['side']=='l', 'strand'].values
    contig_parts.iloc[selected_bridges.loc[selected_bridges['side']=='l','conpart'], contig_parts.columns.get_loc('left_con_read_from')] = selected_bridges.loc[selected_bridges['side']=='l', 'gap_from'].values
    contig_parts.iloc[selected_bridges.loc[selected_bridges['side']=='l','conpart'], contig_parts.columns.get_loc('left_con_read_to')] = selected_bridges.loc[selected_bridges['side']=='l', 'gap_to'].values

    # Right sides
    contig_parts.iloc[selected_bridges.loc[selected_bridges['side']=='r','conpart'], contig_parts.columns.get_loc('right_con')] = selected_bridges.loc[selected_bridges['side']=='r', 'to_conpart'].values
    contig_parts.iloc[selected_bridges.loc[selected_bridges['side']=='r','conpart'], contig_parts.columns.get_loc('right_con_side')] = selected_bridges.loc[selected_bridges['side']=='r', 'to_side'].values
    contig_parts.iloc[selected_bridges.loc[selected_bridges['side']=='r','conpart'], contig_parts.columns.get_loc('end')] = selected_bridges.loc[selected_bridges['side']=='r', 'con_end'].values
    contig_parts.iloc[selected_bridges.loc[selected_bridges['side']=='r','conpart'], contig_parts.columns.get_loc('right_con_read_name')] = selected_bridges.loc[selected_bridges['side']=='r', 'read_name'].values
    contig_parts.iloc[selected_bridges.loc[selected_bridges['side']=='r','conpart'], contig_parts.columns.get_loc('right_con_read_strand')] = selected_bridges.loc[selected_bridges['side']=='r', 'strand'].values
    contig_parts.iloc[selected_bridges.loc[selected_bridges['side']=='r','conpart'], contig_parts.columns.get_loc('right_con_read_from')] = selected_bridges.loc[selected_bridges['side']=='r', 'gap_from'].values
    contig_parts.iloc[selected_bridges.loc[selected_bridges['side']=='r','conpart'], contig_parts.columns.get_loc('right_con_read_to')] = selected_bridges.loc[selected_bridges['side']=='r', 'gap_to'].values

    return contig_parts

def HandleSimpleLoops(contig_parts, bridges, loop_contigs, mappings, max_length_diff_loop_extension, min_factor_alternatives, min_num_reads, pdf):
    # Handle simple loops (contigs bridging to itself)
    loop_contigs_with_exit = reduce(np.intersect1d, (loop_contigs, bridges.loc[bridges['from_side'] == 'l', 'from'], bridges.loc[bridges['from_side'] == 'r', 'from']))
    invalid_anchors = loop_contigs
    
    # Add contig parts that have at least two alternative bridges to invalid_anchors as they must be repeated somehow in the genome until we have only unique connections left
    last_len_invalid_anchors = 0
    while last_len_invalid_anchors < len(invalid_anchors):
        last_len_invalid_anchors = len(invalid_anchors)
        loop_mappings = GetLoopMappings(mappings, bridges, loop_contigs_with_exit, invalid_anchors)
        repeat_bridges = CollectConnections(loop_mappings)
        # We only need to check alternative_s as we have all connections in both directions in there
        invalid_anchors = np.unique(np.concatenate([invalid_anchors, repeat_bridges.loc[repeat_bridges['alternatives_s'] > 1, 'conpart_s'].astype(int)]))
    
    # Deactivate extensions for all invalid_anchors (could be circular molecules without exit and otherwise without an anchor repeat number cannot be properly resolved)
    contig_parts.iloc[invalid_anchors, contig_parts.columns.get_loc('left_con')] = -2
    contig_parts.iloc[invalid_anchors, contig_parts.columns.get_loc('right_con')] = -2
    
    # Set connections for repeat_bridges (Partially overwriting the invalid_anchors from before, if we found a read spanning from a valid anchors to another through those invalid_anchors)
    loop_mappings = ExtendMappingInfo(loop_mappings)
    repeat_bridges, anchored_contig_parts = GetReadInfoForBridges(repeat_bridges, loop_mappings)
    contig_parts = InsertBridges(contig_parts, repeat_bridges)

    # Fill the gap in the loop for contigs which do not connect to others (in case they are circular molecules)
    loop_contigs_without_exit = np.setdiff1d(loop_contigs, loop_contigs_with_exit)
    circular_mappings = mappings[np.isin(mappings['conpart'], loop_contigs_without_exit)].copy()
    circular_mappings = circular_mappings[ (circular_mappings['left_con'] == circular_mappings['conpart']) | (circular_mappings['right_con'] == circular_mappings['conpart'])].copy()
    circular_mappings.drop(columns=['left_con','left_con_side','right_con','right_con_side','num_mappings'], inplace=True)
    
    circular_mappings = pd.concat([circular_mappings.iloc[:-1].reset_index(), circular_mappings.iloc[1:].rename(columns={c:c+'2' for c in circular_mappings.columns}).reset_index()], axis=1)    
    circular_mappings = circular_mappings[(circular_mappings['read_name'] == circular_mappings['read_name2']) &
                                          (circular_mappings['read_start'] == circular_mappings['read_start2']) &
                                          (circular_mappings['strand'] == circular_mappings['strand2']) &
                                          (circular_mappings['conpart'] == circular_mappings['conpart2'])].copy()
    circular_mappings['mapq'] = np.minimum(circular_mappings['mapq'], circular_mappings['mapq2'])
    circular_mappings['identity'] = np.minimum(circular_mappings['identity'], circular_mappings['identity2'])
    circular_mappings['read_from'] = circular_mappings['read_to'] # The gap is of interest now
    circular_mappings['read_to'] = circular_mappings['read_from2'] # The gap is of interest now
    circular_mappings['con_from'] = np.where(circular_mappings['strand'] == '+', circular_mappings['con_from2'], circular_mappings['con_from'])
    circular_mappings['con_to'] = np.where(circular_mappings['strand'] == '+', circular_mappings['con_to'], circular_mappings['con_to2'])
    circular_mappings.drop(columns=['index','read_name2','read_start','read_start2','read_end','read_end2','read_from2','read_to2','strand2','conpart2','con_from2','con_to2','mapq2','identity2'], inplace=True)

    # Cluster potential fillings for the loop by length (reduce allowed distance in length between different loops until all loops in a cluster do not vary by more than max_length_diff_loop_extension from each other)
    circular_mappings['len'] = circular_mappings['read_to']-circular_mappings['read_from']
    circular_mappings.sort_values(['conpart','len'], inplace=True)
    circular_mappings['dist_prev'] = circular_mappings['len'] - circular_mappings['len'].shift(1,fill_value=np.inf)
    circular_mappings.loc[circular_mappings['conpart'] != circular_mappings['conpart'].shift(1, fill_value=-1), 'dist_prev'] = np.inf # Split contig parts
    circular_mappings.loc[circular_mappings['dist_prev'] > max_length_diff_loop_extension, 'dist_prev'] = np.inf # Split connections that are alone already larger than max_length_diff_loop_extension
    
    if pdf:
        tmp = circular_mappings.groupby('conpart')['len'].agg(['min','max']).reset_index()
        PlotHist(pdf, "#contigs", "Max difference loop length", tmp['max']-tmp['min'], threshold=max_length_diff_loop_extension)
    
    too_big = [1]
    while len(too_big):
        circular_mappings['group'] = np.isinf(circular_mappings['dist_prev']).cumsum()
        tmp = circular_mappings.groupby('group')['len'].agg(['min','max']).reset_index()
        too_big = tmp.loc[tmp['max']-tmp['min'] > max_length_diff_loop_extension, 'group'].values
        if len(too_big):
            tmp = circular_mappings[np.isin(circular_mappings['group'], too_big) & (False == np.isinf(circular_mappings['dist_prev']))].groupby('group')['dist_prev'].max().reset_index()
            circular_mappings['cut'] = np.inf
            new_cuts = {g:c for g,c in zip(tmp['group'],tmp['dist_prev'])}
            circular_mappings.loc[np.isin(circular_mappings['group'], too_big), 'cut'] = itemgetter(*circular_mappings.loc[np.isin(circular_mappings['group'], too_big), 'group'])(new_cuts)
            circular_mappings.loc[circular_mappings['dist_prev'] == circular_mappings['cut'], 'dist_prev'] = np.inf
            
    # Select valid groups for loop gap fillings
    circular_mappings.sort_values(['conpart','group','mapq','identity','strand'], ascending=[True,True,False,False,True], inplace=True)
    circular_mappings = circular_mappings[['conpart','group','con_from','con_to','read_name','read_from','read_to','strand']].groupby(['conpart','group'], sort=False).agg(['first','size'])
    circular_mappings['size'] = circular_mappings[('con_from','size')]
    circular_mappings = circular_mappings[[('size',''),('con_from','first'),('con_to','first'),('read_name','first'),('read_from','first'),('read_to','first'),('strand','first')]].droplevel(-1, axis=1).reset_index()
    circular_mappings.sort_values(['conpart','size'],ascending=[True,False], inplace=True)
    circular_mappings = circular_mappings[(circular_mappings['conpart'] != circular_mappings['conpart'].shift(1, fill_value=-1)) & ((circular_mappings['conpart'] != circular_mappings['conpart'].shift(-1, fill_value=-1)) | (circular_mappings['size'] >= min_factor_alternatives*circular_mappings['size'].shift(-1, fill_value=0)))].copy()
    circular_mappings = circular_mappings[ circular_mappings['size'] >= min_num_reads ].copy()
    
    # Fill in extension info for loop gaps
    circular_mappings.loc[circular_mappings['read_from'] >= circular_mappings['read_to'], 'read_name'] = ''
    circular_mappings.loc[circular_mappings['read_from'] >= circular_mappings['read_to'], 'strand'] = ''
    circular_mappings.loc[circular_mappings['read_from'] >= circular_mappings['read_to'], 'read_to'] = -1
    circular_mappings.loc[circular_mappings['read_from'] >= circular_mappings['read_to'], 'read_from'] = -1
    contig_parts.iloc[circular_mappings['conpart'].values, contig_parts.columns.get_loc('start')] = circular_mappings['con_from'].values
    contig_parts.iloc[circular_mappings['conpart'].values, contig_parts.columns.get_loc('end')] = circular_mappings['con_to'].values
    contig_parts.iloc[circular_mappings['conpart'].values, contig_parts.columns.get_loc('right_con_read_name')] = circular_mappings['read_name'].values
    contig_parts.iloc[circular_mappings['conpart'].values, contig_parts.columns.get_loc('right_con_read_strand')] = circular_mappings['strand'].values
    contig_parts.iloc[circular_mappings['conpart'].values, contig_parts.columns.get_loc('right_con_read_to')] = circular_mappings['read_to'].values
    contig_parts.iloc[circular_mappings['conpart'].values, contig_parts.columns.get_loc('right_con_read_from')] = circular_mappings['read_from'].values
    
    return contig_parts, invalid_anchors

def GetLongRangeMappings(mappings, bridges, invalid_anchors):
    # Searching for reads that anchor an invalid anchors on both sides
    long_range_mappings = mappings[mappings['num_mappings']>=3].copy()
    long_range_mappings['anchor'] = np.logical_not(np.isin(long_range_mappings['conpart'], invalid_anchors))
    anchor_totcount = long_range_mappings.groupby(['read_name','read_start'], sort=False)['anchor'].agg(['sum','size']).reset_index()
    anchor_totcount['keep'] = (anchor_totcount['sum'] >= 2) & (anchor_totcount['sum'] < anchor_totcount['size']) # At least two anchors and one invalid
    long_range_mappings = long_range_mappings[np.repeat(anchor_totcount['keep'], anchor_totcount['size']).astype(bool).values].copy()

    # Remove invalid anchors that are not in between anchors
    anchor_totcount = long_range_mappings.groupby(['read_name','read_start'], sort=False)['anchor'].agg(['sum','size']).reset_index()
    anchor_totcount = np.repeat(anchor_totcount['sum'], anchor_totcount['size']).values
    anchor_cumcount = long_range_mappings.groupby(['read_name','read_start'], sort=False)['anchor'].cumsum()
    long_range_mappings = long_range_mappings[long_range_mappings['anchor'] | (anchor_cumcount > 0) & (anchor_cumcount < anchor_totcount)].copy()

    # Remove anchors that are in direct contant only to other anchors
    # Duplicate anchors that connect to two invalid anchors and assign groups going from anchor to anchor
    connected_invalids = (2 - long_range_mappings.groupby(['read_name','read_start'], sort=False)['anchor'].shift(-1, fill_value=True) -
                             long_range_mappings.groupby(['read_name','read_start'], sort=False)['anchor'].shift(1, fill_value=True))
    connected_invalids[np.logical_not(long_range_mappings['anchor'])] = 1
    long_range_mappings = long_range_mappings.loc[np.repeat(long_range_mappings.index, connected_invalids)].copy()
    long_range_mappings['group'] = (long_range_mappings['anchor'].cumsum() + 1)//2-1

    # Remove anchored groups with invalid connections (we only check in one direction as bridges are bidirectional)
    check = long_range_mappings.copy()
    check['from'] = check['conpart']
    check['from_side'] = np.where(check['strand'] == '+', 'r', 'l')
    check['to'] = np.where(check['strand'] == '+', check['right_con'], check['left_con'])
    check['to_side'] = np.where(check['strand'] == '+', check['right_con_side'], check['left_con_side'])
    check = check[['group', 'from', 'from_side', 'to', 'to_side']].copy()
    check['valid'] = (check.merge(bridges[['from','from_side','to','to_side']], on=['from','from_side','to','to_side'], how='left', indicator=True)['_merge'] != "left_only").values
    check.loc[check['group'] != check['group'].shift(-1, fill_value=-1), 'valid'] = True
    check = check.groupby('group')['valid'].min().reset_index(name='valid')

    long_range_mappings = long_range_mappings[np.repeat(check['valid'].values, long_range_mappings.groupby('group', sort=False).size().values)].copy()

    return long_range_mappings

def HandleAlternativeConnections(contig_parts, bridges, mappings, min_num_reads):
    valid_bridges = bridges[(bridges['from_cumcount'] >= min_num_reads) & (bridges['to_cumcount'] >= min_num_reads)].copy()
    invalid_anchors = np.unique(np.concatenate([valid_bridges.loc[valid_bridges['from_alt']>1, 'from'], valid_bridges.loc[valid_bridges['to_alt']>1, 'to']]))
    
    # Add contig parts that have at least two alternative bridges to invalid_anchors as they must be repeated somehow in the genome until we have only unique connections left
    last_len_invalid_anchors = 0
    while last_len_invalid_anchors < len(invalid_anchors):
        last_len_invalid_anchors = len(invalid_anchors)
        long_range_mappings = GetLongRangeMappings(mappings, bridges, invalid_anchors)
        long_range_connections = CollectConnections(long_range_mappings)
        # We only need to check alternative_s as we have all connections in both directions in there
        invalid_anchors = np.unique(np.concatenate([invalid_anchors, long_range_connections.loc[long_range_connections['alternatives_s'] > 1, 'conpart_s'].astype(int)]))

    # Deactivate extensions for all invalid_anchors (could be circular molecules without exit and otherwise without an anchor repeat number cannot be properly resolved)
    # Unique bridges between invalid_anchors are still applied later, to merge two contigs that are repeated together
    contig_parts.iloc[invalid_anchors, contig_parts.columns.get_loc('left_con')] = -2
    contig_parts.iloc[invalid_anchors, contig_parts.columns.get_loc('right_con')] = -2
    
    # Set connections for repeat_bridges (Partially overwriting the invalid_anchors from before, if we found a read spanning from a valid anchors to another through those invalid_anchors)
    long_range_mappings = ExtendMappingInfo(long_range_mappings)
    long_range_connections, anchored_contig_parts = GetReadInfoForBridges(long_range_connections, long_range_mappings)
    contig_parts = InsertBridges(contig_parts, long_range_connections)

    return contig_parts, anchored_contig_parts

def GetShortRangeConnections(valid_bridges, mappings):
    # Mappings that connect two contig parts of interest
    short_mappings = mappings[mappings['num_mappings']>=2].copy()
    short_mappings = short_mappings[ np.isin(short_mappings['conpart'], np.unique(valid_bridges['from'])) ].copy()

    # Mappings that connect on the left side to the correct contig part
    left_mappings = short_mappings[ (0 <= short_mappings['left_con']) & np.isin(short_mappings['conpart'], np.unique(valid_bridges.loc[valid_bridges['from_side']=='l','from'])) ].copy()
    bridge_to = {f: t for f, t in zip(valid_bridges.loc[valid_bridges['from_side']=='l','from'], valid_bridges.loc[valid_bridges['from_side']=='l','to'])}
    left_mappings = left_mappings[ left_mappings['left_con'].values == itemgetter(*left_mappings['conpart'])(bridge_to)].copy()
    bridge_to = {(f, t): s for f, t, s in zip(valid_bridges.loc[valid_bridges['from_side']=='l','from'], valid_bridges.loc[valid_bridges['from_side']=='l','to'], valid_bridges.loc[valid_bridges['from_side']=='l','to_side'])}
    left_mappings = left_mappings[ left_mappings['left_con_side'].values == itemgetter(*zip(left_mappings['conpart'], left_mappings['left_con']))(bridge_to)].copy()

    # Mappings that connect on the right side to the correct contig part
    right_mappings = short_mappings[ (0 <= short_mappings['right_con']) & np.isin(short_mappings['conpart'], np.unique(valid_bridges.loc[valid_bridges['from_side']=='r','from'])) ].copy()
    bridge_to = {f: t for f, t in zip(valid_bridges.loc[valid_bridges['from_side']=='r','from'], valid_bridges.loc[valid_bridges['from_side']=='r','to'])}
    right_mappings = right_mappings[ right_mappings['right_con'].values == itemgetter(*right_mappings['conpart'])(bridge_to)].copy()
    bridge_to = {(f, t): s for f, t, s in zip(valid_bridges.loc[valid_bridges['from_side']=='r','from'], valid_bridges.loc[valid_bridges['from_side']=='r','to'], valid_bridges.loc[valid_bridges['from_side']=='r','to_side'])}
    right_mappings = right_mappings[ right_mappings['right_con_side'].values == itemgetter(*zip(right_mappings['conpart'], right_mappings['right_con']))(bridge_to)].copy()

    short_mappings = pd.concat([left_mappings, right_mappings])
    short_mappings.sort_values(['read_name','read_start','read_from'], inplace=True)

    # Merge the two mappings that belong to one connection
    short_mappings = short_mappings[['read_name', 'read_from', 'read_to', 'strand', 'conpart', 'con_from', 'con_to', 'mapq', 'identity']].copy()
    short_mappings['first'] = [True, False] * (len(short_mappings)//2)
    short1 = short_mappings[short_mappings['first']].reset_index().drop(columns=['index','first'])
    short1.columns = [x+str(1) for x in short1.columns]
    short2 = short_mappings[short_mappings['first']==False].reset_index().drop(columns=['index','first'])
    short2.columns = [x+str(2) for x in short2.columns]

    connections = pd.concat([short1, short2], axis=1)
    connections.rename(columns={'read_name1':'read_name', 'read_to1':'gap_from', 'read_from2':'gap_to'}, inplace=True)
    connections.drop(columns=['read_name2','read_from1','read_to2'], inplace=True)
    connections['mapq'] = np.minimum(connections['mapq1'], connections['mapq2'])
    connections['identity'] = np.minimum(connections['identity1'], connections['identity2'])
    connections.drop(columns=['mapq1','mapq2','identity1','identity2'], inplace=True)
    connections['from'] = np.minimum(connections['conpart1'], connections['conpart2'])
    connections['to'] = np.maximum(connections['conpart1'], connections['conpart2'])
    connections['side1'] = np.where(connections['strand1'] == '+', 'r', 'l')
    connections['side2'] = np.where(connections['strand2'] == '+', 'l', 'r')
    connections['from_side'] = np.where(connections['conpart1'] < connections['conpart2'], connections['side1'], connections['side2'])
    connections['to_side'] = np.where(connections['conpart1'] > connections['conpart2'], connections['side1'], connections['side2'])

    # Select best mappings for each connection
    connections = connections.sort_values(['from','from_side','to','to_side','mapq','identity']).groupby(['from','from_side','to','to_side']).last().reset_index()
    connections = connections.loc[np.repeat(connections.index, 2)].reset_index().drop(columns=['index', 'from','from_side','to','to_side'])
    connections['type'] = [1, 2] * (len(connections)//2)
    connections['conpart'] = np.where(connections['type']==1, connections['conpart1'], connections['conpart2'])
    connections['side'] = np.where(connections['type']==1, connections['side1'], connections['side2'])
    connections['con_end'] = np.where(connections['type']==1, np.where(connections['side']=='l', connections['con_from1'], connections['con_to1']), np.where(connections['side']=='l', connections['con_from2'], connections['con_to2']))
    connections['to_conpart'] = np.where(connections['type']==2, connections['conpart1'], connections['conpart2'])
    connections['to_side'] = np.where(connections['type']==2, connections['side1'], connections['side2'])
    connections['strand'] = np.where(connections['type']==1, connections['strand1'], connections['strand2'])
    connections = connections[['conpart','side','con_end','to_conpart','to_side','read_name','gap_from','gap_to','strand']].copy()

    connections = HandleNegativeGaps(connections)

    return connections

def HandleUniqueBridges(contig_parts, bridges, mappings, min_num_reads):
    valid_bridges = bridges[(bridges['from_cumcount'] >= min_num_reads) & (bridges['to_cumcount'] >= min_num_reads)].copy()
    connections = GetShortRangeConnections(valid_bridges, mappings)
    contig_parts = InsertBridges(contig_parts, connections)
    
    # Get low quality connections
    lowq_bridges = bridges[(bridges['from_cumcount'] < min_num_reads) | (bridges['to_cumcount'] < min_num_reads)].copy()
    lowq_connections = GetShortRangeConnections(lowq_bridges, mappings)
    
    contig_parts['left_lowq'] = np.isin(range(len(contig_parts)), lowq_connections.loc[lowq_connections['side']=='l', 'conpart'])
    contig_parts['right_lowq'] = np.isin(range(len(contig_parts)), lowq_connections.loc[lowq_connections['side']=='r', 'conpart'])
    
    # Original scaffold connections
    org_connections = lowq_connections[(lowq_connections['conpart']+1 == lowq_connections['to_conpart']) | (lowq_connections['conpart']-1 == lowq_connections['to_conpart'])].copy()
    org_connections = org_connections[ (org_connections['conpart'] < org_connections['to_conpart']) & (org_connections['side'] == 'r') & (org_connections['to_side'] == 'l') |
                                       (org_connections['conpart'] > org_connections['to_conpart']) & (org_connections['side'] == 'l') & (org_connections['to_side'] == 'r')].copy()
    org_connections = org_connections[np.where(org_connections['conpart'] < org_connections['to_conpart'], contig_parts.iloc[org_connections['conpart']]['org_dist_right'].values, contig_parts.iloc[org_connections['conpart']]['org_dist_left'].values) > 0].copy()
    contig_parts = InsertBridges(contig_parts, org_connections)
    
    # Repeat connections
    lowq_bridges = lowq_bridges[['from','from_side','to','to_side']].copy()
    tmp_parts = contig_parts.iloc[lowq_bridges['from'].values]
    lowq_bridges['from_rep_con'] = np.where(lowq_bridges['from_side'] == 'l', tmp_parts['rep_con_left'].values, tmp_parts['rep_con_right'].values)
    lowq_bridges['from_rep_con_side'] = np.where(lowq_bridges['from_side'] == 'l', tmp_parts['rep_con_side_left'].values, tmp_parts['rep_con_side_right'].values)
    tmp_parts = contig_parts.iloc[lowq_bridges['to'].values]
    lowq_bridges['to_rep_con'] = np.where(lowq_bridges['to_side'] == 'l', tmp_parts['rep_con_left'].values, tmp_parts['rep_con_right'].values)
    lowq_bridges['to_rep_con_side'] = np.where(lowq_bridges['to_side'] == 'l', tmp_parts['rep_con_side_left'].values, tmp_parts['rep_con_side_right'].values)
    
    lowq_bridges = lowq_bridges[(lowq_bridges['from_rep_con'] == lowq_bridges['to']) & (lowq_bridges['from_rep_con_side'] == lowq_bridges['to_side']) &
                                (lowq_bridges['to_rep_con'] == lowq_bridges['from']) & (lowq_bridges['to_rep_con_side'] == lowq_bridges['from_side']) ].copy()
    if len(lowq_bridges):
        lowq_connections = GetShortRangeConnections(lowq_bridges, mappings)
        contig_parts = InsertBridges(contig_parts, org_connections)
    
    return contig_parts

def MergeCircularityGroups(circular):
    mergeable = [True]
    while sum(mergeable):
        circular = [ list(np.unique(x)) for x in circular ]
        circular.sort()
        mergeable = [False] + [ x[0] == y[0] for x,y in zip(circular[1:],circular[:-1]) ]
        n_groups = len(mergeable)-sum(mergeable)
        group_ids = np.cumsum(np.logical_not(mergeable))-1
        new_circular = [ [] for i in range(n_groups) ]
        for i, gr in enumerate(group_ids):
            new_circular[gr] += circular[i]
        circular = new_circular
        
    return new_circular

def MakeScaffoldIdsContinuous(scaffolds_col):
    scaffold_ids = np.unique(scaffolds_col)
    scaffold_ids = {o:n for o,n in zip(scaffold_ids, range(len(scaffold_ids)))}
    scaffolds_col = itemgetter(*scaffolds_col)(scaffold_ids)

    return scaffolds_col

def CombineContigsOnLeftConnections(scaffold_result, contig_parts):
    ## Merge from right to left
    # Follow the left connections and ignore left to left connections until no connection exists anymore or we are back at the original contig part (circularity)
    scaffold_result['left_end_con'] =  np.where(scaffold_result['left_con_side']=='r', scaffold_result['left_con'], -1)
    scaffold_result['left_end_dist'] = np.where(scaffold_result['left_con_side']=='r', 1, 0)
    scaffold_result['left_end_con_side'] = ''
    scaffold_result.loc[scaffold_result['left_end_con']>=0,'left_end_con_side'] = scaffold_result.iloc[scaffold_result.loc[scaffold_result['left_end_con']>=0,'left_end_con'].values, scaffold_result.columns.get_loc('left_con_side')].values

    while sum(scaffold_result['left_end_con_side'] == 'r'):
        # Make next step
        scaffold_result.loc[scaffold_result['left_end_con_side']=='r','left_end_con'] = scaffold_result.iloc[scaffold_result.loc[scaffold_result['left_end_con_side']=='r','left_end_con'].values, scaffold_result.columns.get_loc('left_con')].values # Do not use left_con_end column (would make it faster, but potentially miss circularity and end up in inifinit loop)
        scaffold_result.loc[scaffold_result['left_end_con_side']=='r','left_end_dist'] += 1

        # Check for circularities
        circular = scaffold_result[scaffold_result['left_end_con'] == scaffold_result['part_id']]
        if len(circular):
            # Find circular groups
            circular = list(zip(circular['left_con'],circular['part_id'],circular['right_con']))
            circular = MergeCircularityGroups(circular)

            # Break circularities at left connection for contig with lowest id
            break_ids = [ x[0] for x in circular ]
            scaffold_result.iloc[ scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('left_con')].values, scaffold_result.columns.get_loc('right_con')] = -2
            scaffold_result.iloc[ scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('left_con')].values, scaffold_result.columns.get_loc('right_con_side')] = ''
            scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('left_con')] = -2
            scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('left_con_side')] = ''
            # Remove read info only on one side so it will be added to the other to complete the circular scaffold
            scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('left_con_read_name')] = ''
            scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('left_con_read_strand')] = ''
            scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('left_con_read_from')] = -1
            scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('left_con_read_to')] = -1

            # Repeat the same for contig_parts
            contig_parts.iloc[ contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con')].values, contig_parts.columns.get_loc('right_con')] = -2
            contig_parts.iloc[ contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con')].values, contig_parts.columns.get_loc('right_con_side')] = ''
            contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con')] = -2
            contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con_side')] = ''
            contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con_read_name')] = ''
            contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con_read_strand')] = ''
            contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con_read_from')] = -1
            contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con_read_to')] = -1

            # Reset left end for contigs in circle
            circular = np.concatenate(circular)
            scaffold_result.iloc[circular, scaffold_result.columns.get_loc('left_end_con')] = scaffold_result.iloc[circular, scaffold_result.columns.get_loc('left_con')].values
            scaffold_result.iloc[circular, scaffold_result.columns.get_loc('left_end_dist')] = 1

            # Don't search for the end contig where we broke the circularity
            scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('left_end_con')] = -1
            scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('left_end_con_side')] = ''
            scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('left_end_dist')] = 0

        # Get side of left_end_con
        scaffold_result.loc[scaffold_result['left_end_con']>=0,'left_end_con_side'] = scaffold_result.iloc[scaffold_result.loc[scaffold_result['left_end_con']>=0,'left_end_con'].values, scaffold_result.columns.get_loc('left_con_side')].values

    scaffold_result['scaffold'] = np.where(scaffold_result['left_end_con']<0, scaffold_result.index, scaffold_result['left_end_con'])
    scaffold_result.rename(columns={'left_end_dist':'pos'}, inplace=True)
    scaffold_result.drop(columns=['left_end_con', 'left_end_con_side'], inplace=True)

    ## Merge scaffolds which have a left-left connection
    # Reverse the scaffolds with the higher id
    scaffold_result['scaf_left_con'] = scaffold_result.iloc[ scaffold_result['scaffold'].values, scaffold_result.columns.get_loc('left_con') ].values
    scaffold_result['reverse'] = (scaffold_result['scaf_left_con'] < scaffold_result['scaffold']) & (scaffold_result['scaf_left_con'] >= 0)
    scaf_id, scaf_size = np.unique(scaffold_result['scaffold'], return_counts=True)
    scaf_size = {s:l for s,l in zip(scaf_id,scaf_size)}
    scaffold_result['scaf_size'] = itemgetter(*scaffold_result['scaffold'])(scaf_size)
    scaffold_result['pos'] = np.where(scaffold_result['reverse'], scaffold_result['scaf_size']-scaffold_result['pos']-1, scaffold_result['pos'])
    # Increase the positions of the scaffold with the lower id
    scaffold_result['scaf_left_size'] = np.where(scaffold_result['scaf_left_con'].values < 0, 0, scaffold_result.iloc[ scaffold_result['scaf_left_con'].values, scaffold_result.columns.get_loc('scaf_size') ].values)
    scaffold_result['pos'] = np.where(scaffold_result['reverse'], scaffold_result['pos'], scaffold_result['pos']+scaffold_result['scaf_left_size'])
    # Assign lower scaffold id to both merged scaffolds and then assign new ids from 0 to number of scaffolds-1
    scaffold_result['scaffold'] = np.where(scaffold_result['reverse'], scaffold_result['scaf_left_con'], scaffold_result['scaffold'])
    scaffold_result['scaf_size'] += scaffold_result['scaf_left_size']

    scaffold_result['scaffold'] = MakeScaffoldIdsContinuous(scaffold_result['scaffold'])
    scaffold_result.drop(columns=['scaf_left_con','scaf_left_size'], inplace=True)

    return scaffold_result, contig_parts

def GetScaffolds(scaffold_result):
    scaffolds = scaffold_result[['scaffold','pos','scaf_size','reverse','right_con']].sort_values(['scaffold','pos']).drop(columns=['pos']).groupby('scaffold', sort=False).agg(['first','last'])
    scaffolds.columns = ['_'.join(x) for x in scaffolds.columns.tolist()]
    scaffolds.rename(columns={'scaf_size_first':'scaf_size', 'right_con_first':'left_con', 'right_con_last':'right_con'}, inplace=True)

    # Only right contig connections haven't been handled yet and depending on the contig being reversed or not they point inward or outward for the scaffold (only keep outward)
    scaffolds['left_con'] = np.where(scaffolds['reverse_first'], scaffolds['left_con'], -1)
    scaffolds['right_con'] = np.where(scaffolds['reverse_last'], -1, scaffolds['right_con'])
    scaffolds.drop(columns=['scaf_size_last','reverse_first','reverse_last'], inplace=True)
    scaffolds.reset_index(inplace=True)

    # Convert the connections from contig ids to scaffold ids
    scaffolds['left_con'] = np.where(scaffolds['left_con']<0, scaffolds['left_con'], scaffold_result.iloc[scaffolds['left_con'].values, scaffold_result.columns.get_loc('scaffold')].values)
    scaffolds['right_con'] = np.where(scaffolds['right_con']<0, scaffolds['right_con'], scaffold_result.iloc[scaffolds['right_con'].values, scaffold_result.columns.get_loc('scaffold')].values)

    # Get the sides to where the connections connect
    scaffolds['left_con_side'] = ''
    scaffolds.loc[scaffolds['left_con']>=0,'left_con_side'] = np.where( scaffolds.iloc[scaffolds.loc[scaffolds['left_con']>=0,'left_con'].values, scaffolds.columns.get_loc('left_con')].values == scaffolds.loc[scaffolds['left_con']>=0,'scaffold'].values, 'l', 'r')
    scaffolds['right_con_side'] = ''
    scaffolds.loc[scaffolds['right_con']>=0,'right_con_side'] = np.where( scaffolds.iloc[scaffolds.loc[scaffolds['right_con']>=0,'right_con'].values, scaffolds.columns.get_loc('right_con')].values == scaffolds.loc[scaffolds['right_con']>=0,'scaffold'].values, 'r', 'l')

    return scaffolds

def HandleCircularScaffolds(scaffolds, scaffold_result, contig_parts):
    circular = scaffolds.loc[scaffolds['right_con'] == scaffolds['scaffold'], 'scaffold'].values

    scaffolds.iloc[circular, scaffolds.columns.get_loc('left_con')] = -1
    scaffolds.iloc[circular, scaffolds.columns.get_loc('right_con')] = -1
    scaffolds.iloc[circular, scaffolds.columns.get_loc('left_con_side')] = ''
    scaffolds.iloc[circular, scaffolds.columns.get_loc('right_con_side')] = ''

    # Break circularities in scaffold_result
    circular_contigs = scaffold_result.loc[np.isin(scaffold_result['scaffold'], circular) & ((scaffold_result['pos'] == 0) | (scaffold_result['pos'] == scaffold_result['scaf_size']-1)), 'part_id'].values
    scaffold_result.iloc[circular_contigs, scaffold_result.columns.get_loc('right_con')] = -2
    scaffold_result.iloc[circular_contigs, scaffold_result.columns.get_loc('right_con_side')] = ''
    # Remove read info only on one contig(scaffold side) so it will be added to the other to complete the circular scaffold
    circular_contigs_first = scaffold_result.iloc[circular_contigs]
    circular_contigs_first = circular_contigs_first.loc[circular_contigs_first['pos'] == 0, 'part_id'].values
    scaffold_result.iloc[circular_contigs_first, scaffold_result.columns.get_loc('right_con_read_name')] = ''
    scaffold_result.iloc[circular_contigs_first, scaffold_result.columns.get_loc('right_con_read_strand')] = ''
    scaffold_result.iloc[circular_contigs_first, scaffold_result.columns.get_loc('right_con_read_from')] = -1
    scaffold_result.iloc[circular_contigs_first, scaffold_result.columns.get_loc('right_con_read_to')] = -1

    # Break circularities in contig_parts doing exactly the same
    contig_parts.iloc[circular_contigs, contig_parts.columns.get_loc('right_con')] = -2
    contig_parts.iloc[circular_contigs, contig_parts.columns.get_loc('right_con_side')] = ''
    contig_parts.iloc[circular_contigs_first, contig_parts.columns.get_loc('right_con_read_name')] = ''
    contig_parts.iloc[circular_contigs_first, contig_parts.columns.get_loc('right_con_read_strand')] = ''
    contig_parts.iloc[circular_contigs_first, contig_parts.columns.get_loc('right_con_read_from')] = -1
    contig_parts.iloc[circular_contigs_first, contig_parts.columns.get_loc('right_con_read_to')] = -1

    return scaffolds, scaffold_result, contig_parts

def ApplyScaffoldMerges(scaffold_result, scaffolds):
    # Apply the scaffold merges to scaffold_result
    scaffold_result = scaffold_result.merge(scaffolds[['scaffold','reverse_scaf','pos_shift','new_scaf','new_size']], on='scaffold', how='left')
    scaffold_result['pos'] = np.where(scaffold_result['reverse_scaf'], scaffold_result['scaf_size']-scaffold_result['pos']-1, scaffold_result['pos'])
    scaffold_result['reverse'] = np.where(scaffold_result['reverse_scaf'], np.logical_not(scaffold_result['reverse']), scaffold_result['reverse'])
    scaffold_result['pos'] += scaffold_result['pos_shift']
    scaffold_result['scaffold'] = scaffold_result['new_scaf']
    scaffold_result['scaf_size'] = scaffold_result['new_size']
    scaffold_result.drop(columns=['reverse_scaf','pos_shift','new_scaf','new_size'], inplace=True)

    return scaffold_result

def HandleRightRightScaffoldConnections(scaffold_result, scaffolds):
    ## Merge scaffolds which have a right-right connection
    scaffolds = scaffolds.copy() # Do not change original
    # Reverse the scaffolds with the higher id and increase its positions
    scaffolds['reverse_scaf'] = (scaffolds['right_con_side'] == 'r') & (scaffolds['right_con'] < scaffolds['scaffold'])
    scaffolds['pos_shift'] = np.where(scaffolds['reverse_scaf'].values, scaffolds.iloc[scaffolds['right_con'], scaffolds.columns.get_loc('scaf_size')], 0)
    # Assign lower scaffold id to both merged scaffolds and then assign new ids from 0 to number of scaffolds-1
    scaffolds['new_scaf'] = np.where(scaffolds['reverse_scaf'].values, scaffolds['right_con'], scaffolds['scaffold'])
    scaffolds['new_size'] = np.where(scaffolds['right_con_side'] == 'r', scaffolds['scaf_size']+scaffolds.iloc[scaffolds['right_con'], scaffolds.columns.get_loc('scaf_size')].values, scaffolds['scaf_size'])

    scaffolds['new_scaf'] = MakeScaffoldIdsContinuous(scaffolds['new_scaf'])
    scaffold_result = ApplyScaffoldMerges(scaffold_result, scaffolds)

    return scaffold_result

def HandleLeftRightScaffoldConnections(scaffold_result, contig_parts, scaffolds):
    ## Merge from right to left
    scaffolds = scaffolds.copy() # Do not change original
    # Follow the left connections and ignore left to left connections until no connection exists anymore or we are back at the original scaffold (circularity)
    scaffolds['left_end_scaf'] =  np.where(scaffolds['left_con_side']=='r', scaffolds['left_con'], -1)
    scaffolds['left_end_dist'] = np.where(scaffolds['left_con_side']=='r', scaffolds.iloc[scaffolds['left_con'].values, scaffolds.columns.get_loc('scaf_size')], 0)
    scaffolds['left_end_con_side'] = ''
    scaffolds.loc[scaffolds['left_end_scaf']>=0,'left_end_con_side'] = scaffolds.iloc[scaffolds.loc[scaffolds['left_end_scaf']>=0,'left_end_scaf'].values, scaffolds.columns.get_loc('left_con_side')].values

    while sum(scaffolds['left_end_con_side'] == 'r'):
        # Make next step
        scaffolds.loc[scaffolds['left_end_con_side']=='r','left_end_scaf'] = scaffolds.iloc[scaffolds.loc[scaffolds['left_end_con_side']=='r','left_end_scaf'].values, scaffolds.columns.get_loc('left_con')].values # Do not use left_con_end column (would make it faster, but potentially miss circularity and end up in inifinit loop)
        scaffolds.loc[scaffolds['left_end_con_side']=='r','left_end_dist'] += scaffolds.iloc[scaffolds.loc[scaffolds['left_end_con_side']=='r','left_end_scaf'].values, scaffolds.columns.get_loc('scaf_size')].values

        # Check for circularities
        circular = scaffolds[scaffolds['left_end_scaf'] == scaffolds['scaffold']]
        if len(circular):
            # Find circular groups
            circular = list(zip(circular['left_con'],circular['scaffold'],circular['right_con']))
            circular = MergeCircularityGroups(circular)

            # Break circularities at left connection for scaffolds with lowest id (It is a right contig connection)
            break_ids = [ x[0] for x in circular ]
            scaffolds.iloc[ scaffolds.iloc[break_ids, scaffolds.columns.get_loc('left_con')].values, scaffolds.columns.get_loc('right_con')] = -2
            scaffolds.iloc[ scaffolds.iloc[break_ids, scaffolds.columns.get_loc('left_con')].values, scaffolds.columns.get_loc('right_con_side')] = ''
            scaffolds.iloc[break_ids, scaffolds.columns.get_loc('left_con')] = -2
            scaffolds.iloc[break_ids, scaffolds.columns.get_loc('left_con_side')] = ''

            # Reset left end for contigs in circle
            circular = np.concatenate(circular)
            scaffolds.iloc[circular, scaffolds.columns.get_loc('left_end_con')] = scaffolds.iloc[circular, scaffolds.columns.get_loc('left_con')].values
            scaffolds.iloc[circular, scaffolds.columns.get_loc('left_end_dist')] = scaffolds.iloc[ scaffolds.iloc[circular, scaffolds.columns.get_loc('left_con')].values, scaffolds.columns.get_loc('scaf_size')]

            # Don't search for the end scaffold where we broke the circularity
            scaffolds.iloc[break_ids, scaffolds.columns.get_loc('left_end_con')] = -1
            scaffolds.iloc[break_ids, scaffolds.columns.get_loc('left_end_con_side')] = ''
            scaffolds.iloc[break_ids, scaffolds.columns.get_loc('left_end_dist')] = 0

            # Apply break also to scaffold_result
            break_ids = scaffold_result.loc[ np.isin(scaffold_result['scaffold'], break_ids) & (scaffold_result['pos'] == 0) , 'part_id'].values # Convert break_ids to contig ids
            scaffold_result.iloc[ scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('right_con')].values, scaffold_result.columns.get_loc('right_con')] = -2
            scaffold_result.iloc[ scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('right_con')].values, scaffold_result.columns.get_loc('right_con_side')] = ''
            scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('right_con')] = -2
            scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('right_con_side')] = ''
            # Remove read info only on one side so it will be added to the other to complete the circular scaffold
            scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('right_con_read_name')] = ''
            scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('right_con_read_strand')] = ''
            scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('right_con_read_from')] = -1
            scaffold_result.iloc[break_ids, scaffold_result.columns.get_loc('right_con_read_to')] = -1

            # Apply break also to contig_parts
            contig_parts.iloc[ contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con')].values, contig_parts.columns.get_loc('right_con')] = -2
            contig_parts.iloc[ contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con')].values, contig_parts.columns.get_loc('right_con_side')] = ''
            contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con')] = -2
            contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con_side')] = ''
            contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con_read_name')] = ''
            contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con_read_strand')] = ''
            contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con_read_from')] = -1
            contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con_read_to')] = -1

        # Get side of left_end_con
        scaffolds.loc[scaffolds['left_end_scaf']>=0,'left_end_con_side'] = scaffolds.iloc[scaffolds.loc[scaffolds['left_end_scaf']>=0,'left_end_scaf'].values, scaffolds.columns.get_loc('left_con_side')].values

    scaffolds['new_scaf'] = np.where(scaffolds['left_end_scaf']<0, scaffolds['scaffold'], scaffolds['left_end_scaf'])
    scaffolds.rename(columns={'left_end_dist':'pos_shift'}, inplace=True)
    scaffolds['reverse_scaf'] = False
    scaf_size = scaffolds.groupby('new_scaf')['scaf_size'].sum().reset_index(name='new_size')
    scaffolds = scaffolds.merge(scaf_size, on='new_scaf', how='left')

    scaffolds['new_scaf'] = MakeScaffoldIdsContinuous(scaffolds['new_scaf'])
    scaffold_result = ApplyScaffoldMerges(scaffold_result, scaffolds)

    return scaffold_result, contig_parts

def HandleLeftLeftScaffoldConnections(scaffold_result, scaffolds):
    ## Merge scaffolds which have a right-right connection
    scaffolds = scaffolds.copy() # Do not change original
    # Reverse the scaffolds with the higher id and increase the position of the one with the lower id
    scaffolds['reverse_scaf'] = (scaffolds['left_con_side'] == 'l') & (scaffolds['left_con'] < scaffolds['scaffold'])
    scaffolds['pos_shift'] = np.where((scaffolds['left_con_side'] == 'l') & (scaffolds['left_con'] > scaffolds['scaffold']), scaffolds.iloc[scaffolds['left_con'], scaffolds.columns.get_loc('scaf_size')], 0)
    # Assign lower scaffold id to both merged scaffolds and then assign new ids from 0 to number of scaffolds-1
    scaffolds['new_scaf'] = np.where(scaffolds['reverse_scaf'].values, scaffolds['left_con'], scaffolds['scaffold'])
    scaffolds['new_size'] = np.where(scaffolds['left_con_side'] == 'l', scaffolds['scaf_size']+scaffolds.iloc[scaffolds['left_con'], scaffolds.columns.get_loc('scaf_size')].values, scaffolds['scaf_size'])

    scaffolds['new_scaf'] = MakeScaffoldIdsContinuous(scaffolds['new_scaf'])
    scaffold_result = ApplyScaffoldMerges(scaffold_result, scaffolds)

    return scaffold_result

def OrderByUnbrokenOriginalScaffolds(scaffold_result, contig_parts):
    ## Bring scaffolds in order of unbroken original scaffolding and remove circularities
    # Get connected scaffolds
    scaffold_order = scaffold_result.loc[(scaffold_result['org_dist_right'] > 0), ['scaffold','reverse']].rename(columns={'scaffold':'left_scaf', 'reverse':'left_rev'})
    scaffold_order['right_scaf'] = scaffold_result.loc[(scaffold_result['org_dist_left'] > 0), 'scaffold'].values
    scaffold_order['right_rev'] = scaffold_result.loc[(scaffold_result['org_dist_left'] > 0), 'reverse'].values
    
    # All scaffolds that are already in correct order and are not changed due to other pairs can be removed from ordering
    complex_scaffolds, scaf_counts = np.unique(np.concatenate( [scaffold_order['left_scaf'], scaffold_order['right_scaf']] ), return_counts=True)
    complex_scaffolds[scaf_counts > 1]
    complex_scaffolds = complex_scaffolds[scaf_counts > 1]
    scaffold_order = scaffold_order[ np.isin(scaffold_order['left_scaf'], complex_scaffolds) | np.isin(scaffold_order['right_scaf'], complex_scaffolds) | # Scaffolds involved in multiple orderings
                                     scaffold_order['left_rev'] | scaffold_order['right_rev'] | # Scaffolds need to be reversed or left/right has to be updated
                                     (scaffold_order['left_scaf']+1 != scaffold_order['right_scaf']) ] # Scaffold ids need to be changed
    
    # Break scaffolds that are connected to itself
    break_ids = scaffold_order.loc[scaffold_order['left_scaf'] == scaffold_order['right_scaf'], 'left_scaf'].values
    break_left = scaffold_result.loc[(np.isin(scaffold_result['scaffold'], break_ids)) & (scaffold_result['org_dist_left'] > 0), 'part_id'].values
    break_right = scaffold_result.loc[(np.isin(scaffold_result['scaffold'], break_ids)) & (scaffold_result['org_dist_right'] > 0), 'part_id'].values

    scaffold_result.iloc[break_left, scaffold_result.columns.get_loc('org_dist_left')] = -2
    scaffold_result.iloc[break_right, scaffold_result.columns.get_loc('org_dist_right')] = -2
    contig_parts.iloc[break_left, contig_parts.columns.get_loc('org_dist_left')] = -2
    contig_parts.iloc[break_right, contig_parts.columns.get_loc('org_dist_right')] = -2
    
    scaffold_order = scaffold_order[scaffold_order['left_scaf'] !=  scaffold_order['right_scaf']].copy()
    
    # Group all scaffolds by the lowest id that is involved with it directly and over other scaffolds
    scaffold_order['min_scaf'] = np.minimum(scaffold_order['left_scaf'], scaffold_order['right_scaf'])
    scaffold_order['max_scaf'] = np.maximum(scaffold_order['left_scaf'], scaffold_order['right_scaf'])
    
    # Merge by min-max connections (min-min is already a single group)
    ungrouped = np.isin(scaffold_order['min_scaf'], scaffold_order['max_scaf'])
    while sum(ungrouped):
        lowest_scaf = {h:l for h,l in zip(scaffold_order['max_scaf'], scaffold_order['min_scaf'])}
        scaffold_order.loc[ungrouped, 'min_scaf'] = itemgetter(*scaffold_order.loc[ungrouped, 'min_scaf'])(lowest_scaf)
        ungrouped = np.isin(scaffold_order['min_scaf'], scaffold_order['max_scaf'])
    
    # Merge by max-max connections
    group_merges = [1]
    while len(group_merges):
        group_merges = scaffold_order[['min_scaf','max_scaf']].groupby('max_scaf').agg(['min','max','size'])
        group_merges.columns = group_merges.columns.droplevel()
        group_merges = group_merges[group_merges['min'] != group_merges['max']].copy()
        if len(group_merges):
            new_groups = {o:n for o,n in zip(group_merges['max'],group_merges['min'])}
            scaffold_order.loc[ np.isin(scaffold_order['min_scaf'],group_merges['max']), 'min_scaf'] = itemgetter(*scaffold_order.loc[np.isin(scaffold_order['min_scaf'],group_merges['max']), 'min_scaf'])(new_groups)

    
    # Find circularities (unique scaffolds in group are less than connections+1)
    scaffolds = pd.DataFrame({'scaffold':np.unique(scaffold_result['scaffold'])})
    space_needed = pd.concat([ scaffold_order[['left_scaf','min_scaf']].rename(columns={'left_scaf':'scaf'}), scaffold_order[['right_scaf','min_scaf']].rename(columns={'right_scaf':'scaf'}) ]).groupby(['min_scaf','scaf']).first().reset_index().groupby('min_scaf').size()
    space_needed2 = (scaffold_order.groupby('min_scaf').size()+1)
    
    break_ids = space_needed[space_needed != space_needed2].index.values
    if len(break_ids):
        # Remove first scaffold_order entry to break circularity
        break_ids = scaffold_order.reset_index().loc[ np.isin(scaffold_order['min_scaf'], break_ids), 'min_scaf'].drop_duplicates().index.values
        
        # Get corresponding scaffolds
        break_left = scaffold_order.iloc[ break_ids, scaffold_order.columns.get_loc('right_scaf') ].values
        break_right = scaffold_order.iloc[ break_ids, scaffold_order.columns.get_loc('left_scaf') ].values
        
        # Get corresponding contigs
        break_left = scaffold_result.loc[(np.isin(scaffold_result['scaffold'], break_left)) & (scaffold_result['org_dist_left'] > 0), 'part_id'].values
        break_right = scaffold_result.loc[(np.isin(scaffold_result['scaffold'], break_right)) & (scaffold_result['org_dist_right'] > 0), 'part_id'].values
        
        # Break
        scaffold_order = scaffold_order[ np.isin(np.arange(len(scaffold_order)), break_ids) == False ].copy()
        
        scaffold_result.iloc[break_left, scaffold_result.columns.get_loc('org_dist_left')] = -2
        scaffold_result.iloc[break_right, scaffold_result.columns.get_loc('org_dist_right')] = -2
        contig_parts.iloc[break_left, contig_parts.columns.get_loc('org_dist_left')] = -2
        contig_parts.iloc[break_right, contig_parts.columns.get_loc('org_dist_right')] = -2

    # Shift scaffold ids so that groups are together
    scaffolds['shift'] = 0
    scaffolds.iloc[space_needed.index+1, scaffolds.columns.get_loc('shift')] += space_needed.values
    scaffolds.iloc[np.unique(np.concatenate( [scaffold_order['left_scaf'], scaffold_order['right_scaf']] ))+1, scaffolds.columns.get_loc('shift')] -= 1
    scaffolds['shift'] = scaffolds['shift'].cumsum()
    scaffolds['new_scaf'] = scaffolds['scaffold'] + scaffolds['shift']
    new_scafs = {o:n for o,n in zip(scaffolds['scaffold'],scaffolds['new_scaf'])}
    scaffold_order['min_scaf'] = itemgetter(*scaffold_order['min_scaf'])(new_scafs)
    
    # Get left and right scaffold original connections
    scaffolds['left_con'] = -1
    scaffolds['left_con_side'] = ''
    scaffolds['right_con'] = -1
    scaffolds['right_con_side'] = ''
    scaffolds['group_min_scaf'] = scaffolds['new_scaf']
    
    scaffolds.iloc[ scaffold_order.loc[scaffold_order['right_rev']==False, 'right_scaf'].values, scaffolds.columns.get_loc('left_con') ] = scaffold_order.loc[scaffold_order['right_rev']==False, 'left_scaf'].values
    scaffolds.iloc[ scaffold_order.loc[scaffold_order['right_rev'], 'right_scaf'].values, scaffolds.columns.get_loc('right_con') ] = scaffold_order.loc[scaffold_order['right_rev'], 'left_scaf'].values
    scaffolds.iloc[ scaffold_order.loc[scaffold_order['left_rev']==False, 'left_scaf'].values, scaffolds.columns.get_loc('right_con') ] = scaffold_order.loc[scaffold_order['left_rev']==False, 'right_scaf'].values
    scaffolds.iloc[ scaffold_order.loc[scaffold_order['left_rev'], 'left_scaf'].values, scaffolds.columns.get_loc('left_con') ] = scaffold_order.loc[scaffold_order['left_rev'], 'right_scaf'].values
    
    scaffolds.loc[ scaffolds['left_con'] >= 0, 'left_con_side'] = np.where(scaffolds.loc[ scaffolds['left_con'] >= 0, 'scaffold'].values == scaffolds.iloc[scaffolds.loc[ scaffolds['left_con'] >= 0, 'left_con'].values, scaffolds.columns.get_loc('left_con')].values, 'l', 'r')
    scaffolds.loc[ scaffolds['right_con'] >= 0, 'right_con_side'] = np.where(scaffolds.loc[ scaffolds['right_con'] >= 0, 'scaffold'].values == scaffolds.iloc[scaffolds.loc[ scaffolds['right_con'] >= 0, 'right_con'].values, scaffolds.columns.get_loc('right_con')].values, 'r', 'l')
    
    scaffolds.iloc[ scaffold_order['right_scaf'].values, scaffolds.columns.get_loc('group_min_scaf') ] = scaffold_order['min_scaf'].values
    scaffolds.iloc[ scaffold_order['left_scaf'].values, scaffolds.columns.get_loc('group_min_scaf') ] = scaffold_order['min_scaf'].values
    
    # Assign position and reverse starting from a arbitrary scaffold with only one connection in the meta-scaffold
    scaffolds['pos'] = -1
    scaffolds['reverse'] = False
    
    scaffolds.iloc[ scaffolds.loc[(scaffolds['left_con'] < 0) | (scaffolds['right_con'] < 0), ['scaffold','group_min_scaf']].groupby('group_min_scaf').first().values, scaffolds.columns.get_loc('pos')] = 0
    scaffolds.loc[scaffolds['pos'] == 0, 'reverse'] = scaffolds.loc[scaffolds['pos'] == 0, 'left_con'].values > 0
    
    cur_scaffolds = scaffolds.loc[(scaffolds['pos'] == 0) & ((scaffolds['left_con'] >= 0) | (scaffolds['right_con'] >= 0))]
    cur_pos = 1
    while len(cur_scaffolds):
        scaffolds.iloc[np.where(cur_scaffolds['reverse'], cur_scaffolds['left_con'], cur_scaffolds['right_con']), scaffolds.columns.get_loc('pos')] = cur_pos
        scaffolds.iloc[np.where(cur_scaffolds['reverse'], cur_scaffolds['left_con'], cur_scaffolds['right_con']), scaffolds.columns.get_loc('reverse')] = np.where(cur_scaffolds['reverse'], cur_scaffolds['left_con_side'], cur_scaffolds['right_con_side']) == 'r'
        
        cur_scaffolds = scaffolds.loc[(scaffolds['pos'] == cur_pos) & (scaffolds['left_con'] >= 0) & (scaffolds['right_con'] >= 0)]
        cur_pos += 1
    
    # Inverse meta-scaffolds if that reduces the number of inversions for the single scaffolds
    scaffolds['scaf_size'] = scaffold_result.groupby('scaffold').size().values
    scaffolds = scaffolds.merge( scaffolds.groupby(['group_min_scaf','reverse'])['scaf_size'].sum().reset_index(name='size').pivot(index='group_min_scaf',columns='reverse',values='size').fillna(0).reset_index().rename(columns={False:'forward_count',True:'reverse_count'}), on='group_min_scaf', how='left')
    
    scaffolds.loc[scaffolds['forward_count'] < scaffolds['reverse_count'], 'reverse'] = np.logical_not(scaffolds.loc[scaffolds['forward_count'] < scaffolds['reverse_count'], 'reverse'].values)
    scaffolds = scaffolds.merge( scaffolds.groupby('group_min_scaf')['scaf_size'].size().reset_index(name='group_size'), on='group_min_scaf', how='left')
    scaffolds.loc[scaffolds['forward_count'] < scaffolds['reverse_count'], 'pos'] = scaffolds.loc[scaffolds['forward_count'] < scaffolds['reverse_count'], 'group_size'].values - scaffolds.loc[scaffolds['forward_count'] < scaffolds['reverse_count'], 'pos'].values - 1
    
    # Apply changes
    scaffolds['new_scaf'] = scaffolds['group_min_scaf'] + scaffolds['pos']
    
    rev_scafs = np.isin(scaffold_result['scaffold'], scaffolds.loc[scaffolds['reverse'], 'scaffold'].values)
    scaffold_result.loc[rev_scafs, 'reverse'] = np.logical_not(scaffold_result.loc[rev_scafs, 'reverse'])
    scaffold_result.loc[rev_scafs, 'pos'] = scaffold_result.loc[rev_scafs, 'scaf_size'] - scaffold_result.loc[rev_scafs, 'pos'] - 1
    
    new_scafs = {o:n for o,n in zip(scaffolds['scaffold'],scaffolds['new_scaf'])}
    scaffold_result['scaffold'] = itemgetter(*scaffold_result['scaffold'])(new_scafs)
    
    # Set org_dist_left/right based on scaffold (not the contig anymore): Flip left/right if contig is reversed
    tmp_dist = scaffold_result.loc[scaffold_result['reverse'],'org_dist_left'].copy().values
    scaffold_result.loc[scaffold_result['reverse'],'org_dist_left'] = scaffold_result.loc[scaffold_result['reverse'],'org_dist_right'].values
    scaffold_result.loc[scaffold_result['reverse'],'org_dist_right'] = tmp_dist
    
    return scaffold_result, contig_parts

def CreateNewScaffolds(contig_parts):
    scaffold_result = contig_parts[['contig', 'start', 'end', 'org_dist_left', 'org_dist_right',
                                    'left_con', 'left_con_side', 'left_con_read_name', 'left_con_read_strand', 'left_con_read_from', 'left_con_read_to',
                                    'right_con', 'right_con_side', 'right_con_read_name', 'right_con_read_strand', 'right_con_read_from', 'right_con_read_to']].copy()
    
    # Remove original connections if gap is not completely untouched and if we introduced breaks(with org_dist 0)
    scaffold_result['org_dist_right'] = np.where((scaffold_result['org_dist_right'] > 0) & (scaffold_result['right_con'] == -1) & (scaffold_result['left_con'].shift(-1, fill_value=0) == -1), scaffold_result['org_dist_right'], -1)
    scaffold_result['org_dist_left'] = np.where((contig_parts['org_dist_left'] > 0) & (contig_parts['left_con'] == -1) & (contig_parts['right_con'].shift(1, fill_value=0) == -1), scaffold_result['org_dist_left'], -1)
    
    scaffold_result['part_id'] = scaffold_result.index
    scaffold_result, contig_parts = CombineContigsOnLeftConnections(scaffold_result, contig_parts)
    
    # Now contig wise all left connections have been handled, but the scaffolds can still have a left connection in case it starts with a reversed contig
    num_scaffolds_old = max(scaffold_result['scaffold'])+1
    num_scaffolds = 0
    while num_scaffolds_old != num_scaffolds:
        # Handle right-right scaffold connections
        scaffolds = GetScaffolds(scaffold_result)
        scaffolds, scaffold_result, contig_parts = HandleCircularScaffolds(scaffolds, scaffold_result, contig_parts)
        scaffold_result = HandleRightRightScaffoldConnections(scaffold_result, scaffolds)
    
        # Handle left-right scaffold connections
        scaffolds = GetScaffolds(scaffold_result)
        scaffolds, scaffold_result, contig_parts = HandleCircularScaffolds(scaffolds, scaffold_result, contig_parts)
        scaffold_result, contig_parts = HandleLeftRightScaffoldConnections(scaffold_result, contig_parts, scaffolds)
    
        # Handle left-left scaffold connections
        scaffolds = GetScaffolds(scaffold_result)
        scaffold_result = HandleLeftLeftScaffoldConnections(scaffold_result, scaffolds)
    
        num_scaffolds_old = num_scaffolds
        num_scaffolds = max(scaffold_result['scaffold'])+1

    scaffold_result, contig_parts = OrderByUnbrokenOriginalScaffolds(scaffold_result, contig_parts)
    
    # Create scaffold_info which later is used to lift mappings from contig based to scaffold based
    scaffold_info = scaffold_result[['part_id','pos','scaffold','reverse','scaf_size']].copy()
    
    # Create info to create scaffolds
    scaffold_table = scaffold_result[['contig','start','end','reverse','left_con_read_name','left_con_read_strand','left_con_read_from','left_con_read_to','right_con_read_name','right_con_read_strand','right_con_read_from','right_con_read_to','org_dist_left','org_dist_right','pos','scaffold','scaf_size']].copy()
    scaffold_table.sort_values(['scaffold','pos'], inplace=True)
    scaffold_table = scaffold_table.loc[np.repeat(scaffold_table.index, np.where(0==scaffold_table['pos'], 3, 2))].copy()
    scaffold_table['type'] = 1
    scaffold_table['type'] = scaffold_table.groupby(['scaffold','pos'], sort=False)['type'].cumsum()
    scaffold_table.loc[0==scaffold_table['pos'], 'type'] -= 1
    
    scaffold_table['name'] = np.where(scaffold_table['type']==1, contig_parts.iloc[scaffold_table.index, contig_parts.columns.get_loc('name')].values, np.where(scaffold_table['type']==0, np.where(scaffold_table['reverse'], scaffold_table['right_con_read_name'], scaffold_table['left_con_read_name']), np.where(scaffold_table['reverse'], scaffold_table['left_con_read_name'], scaffold_table['right_con_read_name']) ) )
    scaffold_table = scaffold_table[scaffold_table['name'] != ''].copy()
    scaffold_table['start'] = np.where(scaffold_table['type']==1, scaffold_table['start'], np.where(scaffold_table['type']==0, np.where(scaffold_table['reverse'], scaffold_table['right_con_read_from'], scaffold_table['left_con_read_from']), np.where(scaffold_table['reverse'], scaffold_table['left_con_read_from'], scaffold_table['right_con_read_from']) ) )
    scaffold_table['end'] = np.where(scaffold_table['type']==1, scaffold_table['end'], np.where(scaffold_table['type']==0, np.where(scaffold_table['reverse'], scaffold_table['right_con_read_to'], scaffold_table['left_con_read_to']), np.where(scaffold_table['reverse'], scaffold_table['left_con_read_to'], scaffold_table['right_con_read_to']) ) )
    scaffold_table['reverse'] = np.where(scaffold_table['type']==1, scaffold_table['reverse'], np.where(scaffold_table['type']==0, np.where(scaffold_table['reverse'], scaffold_table['right_con_read_strand']=='+', scaffold_table['left_con_read_strand']=='-'), np.where(scaffold_table['reverse'], scaffold_table['left_con_read_strand']=='+', scaffold_table['right_con_read_strand']=='-') ) )
    
    scaffold_table = scaffold_table[['scaffold','pos','scaf_size','type','name','start','end','reverse','org_dist_left','org_dist_right']].copy()
    scaffold_table['pos'] = scaffold_table.groupby(['scaffold'], sort=False).cumcount()
    scaf_size = scaffold_table.groupby('scaffold', sort=False).size().values
    scaffold_table['scaf_size'] = np.repeat(scaf_size,scaf_size)
    
    # Remove org_dist from reads (Cannot be on an outward facing read as extensions haven't been done generally yet, only for circular scaffolds, where org_dist is anyway invalid)
    scaffold_table.loc[scaffold_table['type'] != 1, 'org_dist_left'] = -1
    scaffold_table.loc[scaffold_table['type'] != 1, 'org_dist_right'] = -1
    
    scaffold_table['type'] = np.where(scaffold_table['type'] == 1, "contig", "read")

    return scaffold_table, scaffold_info, contig_parts

def GetScaffoldExtendingMappings(mappings, contig_parts, scaffold_info, max_dist_contig_end, min_extension, min_num_reads, pdf):
    mappings.drop(columns=['left_con','left_con_side' ,'right_con','right_con_side','num_mappings'], inplace=True)
    
    # Only reads that map to the outter contigs in a scaffold are relevant
    mappings = mappings[ np.isin(mappings['conpart'], scaffold_info.loc[(scaffold_info['pos'] == 0) | (scaffold_info['pos'] == scaffold_info['scaf_size']-1), 'part_id'].values) ].copy()
    
    # Reads need to extend the contig in the right direction
    mappings['con_len'] = contig_parts.iloc[mappings['conpart'].values, contig_parts.columns.get_loc('end')].values
    mappings['left_ext'] = np.where('+' == mappings['strand'], mappings['read_from']-mappings['read_start']-mappings['con_from'], (mappings['read_end']-mappings['read_to'])-mappings['con_from'])
    mappings['right_ext'] = np.where('-' == mappings['strand'], mappings['read_from']-mappings['read_start']-(mappings['con_len']-mappings['con_to']), (mappings['read_end']-mappings['read_to'])-(mappings['con_len']-mappings['con_to']))
    
    # Set extensions to zero that are in the wrong direction
    tmp_info = scaffold_info.iloc[mappings['conpart'].values]
    mappings.loc[ np.where(tmp_info['reverse'], tmp_info['scaf_size']-1 != tmp_info['pos'], tmp_info['pos'] > 0), 'left_ext'] = 0
    mappings.loc[ np.where(tmp_info['reverse'], tmp_info['pos'] > 0, tmp_info['scaf_size']-1 != tmp_info['pos']), 'right_ext'] = 0
    
    # Set extensions to zero for scaffold ends which are circular and should not be extended
    tmp_info2 = contig_parts.iloc[mappings['conpart'].values]
    mappings.loc[ (tmp_info2['left_con'] == -2).values, 'left_ext'] = 0
    mappings.loc[ (tmp_info2['right_con'] == -2).values, 'right_ext'] = 0
    
    # Set extensions to zero that are too far away from the contig_end
    mappings.loc[mappings['con_from'] > max_dist_contig_end, 'left_ext'] = 0
    mappings.loc[mappings['con_len']-mappings['con_to'] > max_dist_contig_end, 'right_ext'] = 0

    mappings = mappings[(mappings['left_ext'] > 0) | (mappings['right_ext'] > 0)].copy()
    mappings['left_ext'] = np.maximum(0, mappings['left_ext'])
    mappings['right_ext'] = np.maximum(0, mappings['right_ext'])

    # Lift mappings from contigs to scaffolds
    mappings['scaffold'] = scaffold_info.iloc[mappings['conpart'].values, scaffold_info.columns.get_loc('scaffold')].values
    is_reversed = scaffold_info.iloc[mappings['conpart'].values]['reverse'].values
    mappings['strand'] = np.where(mappings['strand']=='+', np.where(is_reversed, '-', '+'), np.where(is_reversed, '+', '-'))
    tmp_ext = mappings['left_ext'].copy().values
    mappings['left_ext'] = np.where(is_reversed, mappings['right_ext'], mappings['left_ext'])
    mappings['right_ext'] = np.where(is_reversed, tmp_ext, mappings['right_ext'])

    mappings['left_dist_start'] = np.where(is_reversed, mappings['con_len']-mappings['con_to'], mappings['con_from'])
    mappings['left_dist_end'] = np.where(is_reversed, mappings['con_len']-mappings['con_from'], mappings['con_to'])
    mappings['right_dist_start'] = np.where(is_reversed, mappings['con_from'], mappings['con_len']-mappings['con_to'])
    mappings['right_dist_end'] = np.where(is_reversed, mappings['con_to'], mappings['con_len']-mappings['con_from'])
    
    mappings.loc[mappings['left_ext']==0, 'left_dist_start'] = -1
    mappings.loc[mappings['left_ext']==0, 'left_dist_end'] = -1
    mappings.loc[mappings['right_ext']==0, 'right_dist_start'] = -1
    mappings.loc[mappings['right_ext']==0, 'right_dist_end'] = -1
    
    mappings = mappings[['read_name','read_start','read_end','read_from','read_to', 'scaffold','left_dist_start','left_dist_end','right_dist_end','right_dist_start', 'strand','mapq','identity', 'left_ext','right_ext', 'conpart']].copy()

    # Only keep long enough extensions
    if pdf:
        extension_lengths = np.concatenate([mappings.loc[0<mappings['left_ext'],'left_ext'], mappings.loc[0<mappings['right_ext'],'right_ext']])
        PlotHist(pdf, "Extension length", "# Extensions", np.extract(extension_lengths < 10*min_extension, extension_lengths), threshold=min_extension)
        PlotHist(pdf, "Extension length", "# Extensions", extension_lengths, threshold=min_extension, logx=True)
        del extension_lengths
    
    mappings = mappings[(min_extension<=mappings['left_ext']) | (min_extension<=mappings['right_ext'])].copy()

    # Only keep extension, when there are enough of them
    num_left_extensions = mappings[mappings['left_ext']>0].groupby(['scaffold','conpart']).size().reset_index(name='counts')
    num_right_extensions = mappings[mappings['right_ext']>0].groupby(['scaffold','conpart']).size().reset_index(name='counts')
    num_left_extensions['reverse'] = scaffold_info.iloc[num_left_extensions['conpart'].values, scaffold_info.columns.get_loc('reverse')].values
    num_right_extensions['reverse'] = scaffold_info.iloc[num_right_extensions['conpart'].values, scaffold_info.columns.get_loc('reverse')].values
    
    contig_parts['left_ext'] = 0
    contig_parts['right_ext'] = 0
    
    contig_parts.iloc[num_left_extensions.loc[np.logical_not(num_left_extensions['reverse']), 'conpart'].values, contig_parts.columns.get_loc('left_ext')] = num_left_extensions.loc[np.logical_not(num_left_extensions['reverse']), 'counts'].values
    contig_parts.iloc[num_right_extensions.loc[num_right_extensions['reverse'], 'conpart'].values, contig_parts.columns.get_loc('left_ext')] = num_right_extensions.loc[num_right_extensions['reverse'], 'counts'].values
    contig_parts.iloc[num_right_extensions.loc[np.logical_not(num_right_extensions['reverse']), 'conpart'].values, contig_parts.columns.get_loc('right_ext')] = num_right_extensions.loc[np.logical_not(num_right_extensions['reverse']), 'counts'].values
    contig_parts.iloc[num_left_extensions.loc[num_left_extensions['reverse'], 'conpart'].values, contig_parts.columns.get_loc('right_ext')] = num_left_extensions.loc[num_left_extensions['reverse'], 'counts'].values
    mappings.drop(columns=['conpart'], inplace=True)

    mappings = mappings[((mappings['left_ext']>0) & np.isin(mappings['scaffold'], num_left_extensions.loc[num_left_extensions['counts']>=min_num_reads, 'scaffold'].values)) |
                        ((mappings['right_ext']>0) & np.isin(mappings['scaffold'], num_right_extensions.loc[num_right_extensions['counts']>=min_num_reads, 'scaffold'].values))]

    return mappings, contig_parts

def FilterExtendingReads(read_file, read_format, mappings, prefix):
    extending_reads = set(np.unique(mappings['read_name']))
    
    if "fastq" == read_format:
        with open(prefix+'_extending_reads.fq', "w") as fout:
            with gzip.open(read_file, 'rt') if 'gz' == read_file.rsplit('.',1)[-1] else open(read_file, 'r') as fin:
                for (title, seq, qual) in FastqGeneralIterator(fin):
                    if title.split(' ')[0] in extending_reads:
                        tmp = fout.write("@{}\n{}\n+\n{}\n".format(title, seq, qual))
    elif "fasta" == read_format:
        with open(prefix+'_extending_reads.fa', "w") as fout:
            with gzip.open(read_file, 'rt') if 'gz' == read_file.rsplit('.',1)[-1] else open(read_file, 'r') as fin:
                for (title, seq) in SimpleFastaParser(fin):
                    if title.split(' ')[0] in extending_reads:
                        tmp = fout.write(">{}\n{}\n".format(title, seq))

def SplitPrintByExtensionStatus(category, min_num_reads):
    category['ext'] = {}
    category['ext']['left'] = category['left'][ category['left']['left_ext']>=min_num_reads ].copy()
    category['ext']['right'] = category['right'][ category['right']['right_ext']>=min_num_reads ].copy()

    category['noext'] = {}
    category['noext']['left'] = category['left'][ category['left']['left_ext']<=0].copy()
    category['noext']['right'] = category['right'][ category['right']['right_ext']<=0].copy()

    category['lowcov'] = {}
    category['lowcov']['left'] = category['left'][ (category['left']['left_ext']>0) & (category['left']['left_ext']<min_num_reads) ].copy()
    category['lowcov']['right'] = category['right'][ (category['right']['right_ext']>0) & (category['right']['right_ext']<min_num_reads) ].copy()

    return category

def SplitPrintByQualityStatus(category, min_num_reads):
    category['lowq'] = {}
    category['lowq']['left'] = category['left'][ category['left']['left_lowq'] ].copy()
    category['lowq']['right'] = category['right'][ category['right']['right_lowq'] ].copy()
    category['lowq'] = SplitPrintByExtensionStatus(category['lowq'], min_num_reads)

    category['nolowq'] = {}
    category['nolowq']['left'] = category['left'][ category['left']['left_lowq'] == False ].copy()
    category['nolowq']['right'] = category['right'][ category['right']['right_lowq'] == False ].copy()
    category['nolowq'] = SplitPrintByExtensionStatus(category['nolowq'], min_num_reads)

    return category

def SplitPrintByConnectionStatus(category, min_num_reads):
    category['connected'] = {}
    category['connected']['left'] = category['left'][ category['left']['left_con'] >= 0 ].copy()
    category['connected']['right'] = category['right'][ category['right']['right_con'] >= 0 ].copy()
    category['connected'] = SplitPrintByQualityStatus(category['connected'], min_num_reads)

    category['unconnected'] = {}
    category['unconnected']['left'] = category['left'][ category['left']['left_con'] == -1 ].copy()
    category['unconnected']['right'] = category['right'][ category['right']['right_con'] == -1 ].copy()
    category['unconnected'] = SplitPrintByQualityStatus(category['unconnected'], min_num_reads)

    category['circular'] = {}
    category['circular']['left'] = category['left'][ category['left']['left_con'] == -2 ].copy()
    category['circular']['right'] = category['right'][ category['right']['right_con'] == -2 ].copy()

    return category

def SplitPrintByModificationStatus(category, min_num_reads):
    tmpl = category['left'].copy()
    tmpr = category['right'].copy()

    category['closed'] = {}
    category['closed']['left'] = tmpl[(tmpl['left_con_side'] == 'r') & (tmpl['left_con'] == tmpl.index-1)].copy()
    category['closed']['right'] = tmpr[(tmpr['right_con_side'] == 'l') & (tmpr['right_con'] == tmpr.index+1)].copy()
    category['closed'] = SplitPrintByQualityStatus(category['closed'], min_num_reads)

    tmpl = tmpl[(tmpl['left_con_side'] != 'r') | (tmpl['left_con'] != tmpl.index-1)].copy()
    tmpr = tmpr[(tmpr['right_con_side'] != 'l') | (tmpr['right_con'] != tmpr.index+1)].copy()

    category['open'] = {}
    category['open']['left'] = tmpl[(tmpl['left_con'].values == -1) & (tmpr['right_con'].values == -1)].copy()
    category['open']['right'] = tmpr[(tmpl['left_con'].values == -1) & (tmpr['right_con'].values == -1)].copy()
    category['open'] = SplitPrintByQualityStatus(category['open'], min_num_reads)

    category['broken'] = {}
    category['broken']['left'] = tmpl[(tmpl['left_con'].values != -1) | (tmpr['right_con'].values != -1)].copy()
    category['broken']['right'] = tmpr[(tmpl['left_con'].values != -1) | (tmpr['right_con'].values != -1)].copy()
    category['broken'] = SplitPrintByConnectionStatus(category['broken'], min_num_reads)

    return category

def GetPrintCategories(contig_parts, min_num_reads):
    gaps = {}
    gaps['left'] = contig_parts[contig_parts['org_dist_left'] > 0].copy()
    gaps['right'] = contig_parts[contig_parts['org_dist_right'] > 0].copy()
    gaps = SplitPrintByModificationStatus(gaps, min_num_reads)

    breaks = {}
    breaks['left'] = contig_parts[contig_parts['org_dist_left'] == 0].copy()
    breaks['right'] = contig_parts[contig_parts['org_dist_right'] == 0].copy()
    breaks = SplitPrintByModificationStatus(breaks, min_num_reads)

    scaffolds = {}
    scaffolds['left'] = contig_parts[contig_parts['org_dist_left'] < 0].copy()
    scaffolds['right'] = contig_parts[contig_parts['org_dist_right'] < 0].copy()
    scaffolds = SplitPrintByConnectionStatus(scaffolds, min_num_reads)    

    return gaps, breaks, scaffolds

def PrintStatsRow(level, title, value, percent=0, sides=False):
    if percent>0:
        perc = ' ({:.2f}%)'.format(value/percent*100)
    else:
        perc = ''

    if sides:
        sid = ' [{} sides]'.format(2*value)
    else:
        sid = ''

    print('{}{}: {}{}{}'.format(' '*2*level, title, value, perc, sid))
    

def PrintUnconnectedExtensionStats(category, total, level):
    PrintStatsRow(level,"Extendable",len(category['ext']['right']) + len(category['ext']['left']),total)
    PrintStatsRow(level,"Low coverage",len(category['lowcov']['right']) + len(category['lowcov']['left']),total)
    PrintStatsRow(level,"No coverage",len(category['noext']['right']) + len(category['noext']['left']),total)
    
def PrintUnconnectedStats(category, total, level=2):
    lowq = len(category['lowq']['right']) + len(category['lowq']['left'])
    PrintStatsRow(level,"Sides without accepted connections",lowq,total)
    PrintUnconnectedExtensionStats(category['lowq'], lowq, level+1)
    nocon = len(category['nolowq']['right']) + len(category['nolowq']['left'])
    PrintStatsRow(level,"Sides without any unambiguous connections",nocon,total)
    PrintUnconnectedExtensionStats(category['nolowq'], nocon, level+1) 

def PrintStats(contig_parts, org_contig_info, min_num_reads):
    break_info = contig_parts.groupby('contig').size()
    
    print("Contigs: {} ({} bases, largest: {} bases)".format(org_contig_info['num']['total'], org_contig_info['len']['total'], org_contig_info['max']['total']))
    print("  Removed: {} ({} bases, largest: {} bases)".format(org_contig_info['num']['removed_total'], org_contig_info['len']['removed_total'], org_contig_info['max']['removed_total']))
    print("    No long reads mapping: {} ({} bases, largest: {} bases)".format(org_contig_info['num']['removed_no_mapping'], org_contig_info['len']['removed_no_mapping'], org_contig_info['max']['removed_no_mapping']))
    print("    Complete duplications: {} ({} bases, largest: {} bases)".format(org_contig_info['num']['removed_duplicates'], org_contig_info['len']['removed_duplicates'], org_contig_info['max']['removed_duplicates']))
    print("  Masked: {} ({} bases, largest: {} bases)".format(org_contig_info['num']['masked'], org_contig_info['len']['masked'], org_contig_info['max']['masked']))
    print("  Broken: {} into {} contigs".format(sum(break_info>1), sum(break_info[break_info>1])))
    
    gaps, breaks, scaffolds = GetPrintCategories(contig_parts, min_num_reads)
    
    total_num_gaps = len(gaps['right'])
    PrintStatsRow(0,"Number of gaps",total_num_gaps)
    closed_gaps = len(gaps['closed']['right'])
    PrintStatsRow(1,"Closed gaps",closed_gaps,total_num_gaps)
    PrintStatsRow(2,"Hiqh quality closing",len(gaps['closed']['nolowq']['right']),closed_gaps)
    PrintStatsRow(2,"Low quality closing",len(gaps['closed']['lowq']['right']),closed_gaps)
    PrintStatsRow(1,"Untouched gaps",len(gaps['open']['right']),total_num_gaps,True)
    PrintUnconnectedStats(gaps['open'], 2*len(gaps['open']['right']))
    PrintStatsRow(1,"Broken gaps",len(gaps['broken']['right']),total_num_gaps,True)
    broken_gaps_sides = 2*len(gaps['broken']['right'])
    PrintStatsRow(2,"Sides connected to other contigs",len(gaps['broken']['connected']['right'])+len(gaps['broken']['connected']['left']),broken_gaps_sides)
    PrintStatsRow(2,"Sides being circular or repetitive",len(gaps['broken']['circular']['right'])+len(gaps['broken']['circular']['left']),broken_gaps_sides)
    PrintUnconnectedStats(gaps['broken']['unconnected'], broken_gaps_sides)

    total_num_contig_breaks = len(breaks['right'])
    PrintStatsRow(0,"Number of contig breaks",total_num_contig_breaks)
    PrintStatsRow(1,"Resealed breaks",len(breaks['closed']['nolowq']['right']),total_num_contig_breaks)
    PrintStatsRow(1,"Open breaks",len(breaks['open']['right']),total_num_contig_breaks,True)
    PrintUnconnectedStats(breaks['open'], 2*len(breaks['open']['right']))
    PrintStatsRow(1,"Modified breaks",len(breaks['broken']['right']),total_num_contig_breaks,True)
    modified_breaks_sides = 2*len(breaks['broken']['right'])
    PrintStatsRow(2,"Sides connected to other contigs",len(breaks['broken']['connected']['right'])+len(breaks['broken']['connected']['left']),modified_breaks_sides)
    PrintStatsRow(2,"Sides being circular or repetitive",len(breaks['broken']['circular']['right'])+len(breaks['broken']['circular']['left']),modified_breaks_sides)
    PrintUnconnectedStats(breaks['broken']['unconnected'], modified_breaks_sides)
    
    total_num_scaffold_sides = len(scaffolds['right'])*2
    PrintStatsRow(0,"Number of scaffolds",total_num_scaffold_sides//2,sides=True)
    PrintStatsRow(1,"Sides connected with hiqh quality",len(scaffolds['connected']['nolowq']['right']) + len(scaffolds['connected']['nolowq']['left']),total_num_scaffold_sides)
    PrintStatsRow(1,"Sides connected through low quality repeat connections",len(scaffolds['connected']['lowq']['right']) + len(scaffolds['connected']['lowq']['left']),total_num_scaffold_sides)
    PrintStatsRow(1,"Sides being circular or repetitive",len(scaffolds['circular']['right']) + len(scaffolds['circular']['left']),total_num_scaffold_sides)
    PrintUnconnectedStats(scaffolds['unconnected'], total_num_scaffold_sides, 1)

def MiniGapScaffold(assembly_file, mapping_file, repeat_file, read_file, prefix=False, stats=None):
    # Put in default parameters if nothing was specified
    if False == prefix:
        if ".gz" == assembly_file[-3:len(assembly_file)]:
            prefix = assembly_file.rsplit('.',2)[0]
        else:
            prefix = assembly_file.rsplit('.',1)[0]
            
    max_repeat_extension = 1000 # Expected to be bigger than or equal to min_mapping_length
    min_len_repeat_connection = 5000
    repeat_len_factor_unique = 10
    
    adapter_signal_max_dist = 3000
    min_mapq = 20
    min_mapping_length = 500
    min_extension = 500
    max_dist_contig_end = 2000
    min_length_contig_break = 1000
    max_break_point_distance = 2000
    
    min_num_reads = 3
    min_factor_alternatives = 2
    
    max_length_diff_loop_extension = 500
    
    # Guarantee that outdir exists
    outdir = os.path.dirname(prefix)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    
    if not os.path.exists(read_file):
        print(read_file, "does not exist or no read permission.")
        return
    else:
        read_format = GetReadFormat(read_file)
        if read_format != "fastq" and read_format != "fasta":
            print(read_file, "is not in one of the supported formats: fastq, fasta, gz")
            return

    pdf = None
    if stats:
        pdf = PdfPages(stats)
        plt.ioff()
        
    print( str(timedelta(seconds=clock())), "Reading in original assembly")
    contigs, contig_ids = ReadContigs(assembly_file)
    
    print( str(timedelta(seconds=clock())), "Processing repeats")
    contigs, center_repeats = MaskRepeatEnds(contigs, repeat_file, contig_ids, max_repeat_extension, min_len_repeat_connection, repeat_len_factor_unique, pdf)
    
    print( str(timedelta(seconds=clock())), "Filtering mappings")
    mappings = ReadMappings(mapping_file, contig_ids, min_mapq, pdf)
    contigs, org_contig_info = RemoveUnmappedContigs(contigs, mappings)
    mappings = RemoveUnanchoredMappings(mappings, contigs, center_repeats, min_mapping_length, pdf, max_dist_contig_end)
    del center_repeats
    
    print( str(timedelta(seconds=clock())), "Account for left-over adapters")
    mappings = BreakReadsAtAdapters(mappings, adapter_signal_max_dist, pdf)
    
    print( str(timedelta(seconds=clock())), "Search for possible break points")
    break_groups, spurious_break_indexes = FindBreakPoints(mappings, contigs, max_dist_contig_end, min_mapping_length, min_length_contig_break, max_break_point_distance, min_num_reads, min_factor_alternatives, pdf)
    mappings.drop(spurious_break_indexes, inplace=True) # Remove not-accepted breaks from mappings
    del spurious_break_indexes
    
    contig_parts, contigs = GetContigParts(contigs, break_groups, pdf)
    mappings = UpdateMappingsToContigParts(mappings, contigs, contig_parts, break_groups, min_mapping_length)
    del break_groups, contigs, contig_ids
    
    print( str(timedelta(seconds=clock())), "Search for possible bridges")
    bridges, loop_contigs = GetBridges(mappings, min_factor_alternatives, min_num_reads, pdf)
    
    contig_parts, invalid_anchors = HandleSimpleLoops(contig_parts, bridges, loop_contigs, mappings, max_length_diff_loop_extension, min_factor_alternatives, min_num_reads, pdf)
    bridges = bridges[np.logical_not(np.isin(bridges['from'],invalid_anchors) | np.isin(bridges['to'],invalid_anchors))].copy()
    del loop_contigs, invalid_anchors

    contig_parts, anchored_contig_parts = HandleAlternativeConnections(contig_parts, bridges, mappings, min_num_reads)
    bridges = bridges[(bridges['from_alt'] == 1) & (bridges['to_alt'] == 1)].copy()
    bridges = bridges[np.isin(bridges['from'], anchored_contig_parts) == False].copy()
    bridges = bridges[np.isin(bridges['to'], anchored_contig_parts) == False].copy()
    del anchored_contig_parts
    
    contig_parts = HandleUniqueBridges(contig_parts, bridges, mappings, min_num_reads)
    
    print( str(timedelta(seconds=clock())), "Scaffold the contigs")
    scaffold_table, scaffold_info, contig_parts = CreateNewScaffolds(contig_parts)
    
    print( str(timedelta(seconds=clock())), "Search for possible extensions")
    mappings, contig_parts = GetScaffoldExtendingMappings(mappings, contig_parts, scaffold_info, max_dist_contig_end, min_extension, min_num_reads, pdf)

    if pdf:
        pdf.close()
        
    print( str(timedelta(seconds=clock())), "Writing output")
    mappings.to_csv(prefix+"_extensions.csv", index=False)
    scaffold_table.to_csv(prefix+"_scaffold_table.csv", index=False)
    FilterExtendingReads(read_file, read_format, mappings, prefix)

    print( str(timedelta(seconds=clock())), "Finished")
    PrintStats(contig_parts, org_contig_info, min_num_reads)
        
def usage():
    print("Usage: minigap.py [OPTIONS] {assembly}.fa {mapping}.paf {repeat}.paf {reads}.[fa/fq]")
    print("Scaffolding and gap filling with long reads.")
    print("  -h, --help            display this help and exit")
    print("  -p, --prefix FILE     prefix for output files ({assembly})")
    print("  -s, --stats FILE.pdf  Output file for plots with statistics regarding input parameters (deactivated)")

def main(argv):
    try:
        optlist, args = getopt.getopt(argv, 'hp:s:', ['help','prefix=','stats='])
    except getopt.GetoptError:
        print("Unknown option\n")
        usage()
        sys.exit(2)

    prefix = False
    stats = None
    for opt, par in optlist:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-p", "--prefix"):
            prefix = par
        elif opt in ("-s", "--stats"):
            stats = par
            if(stats[-4:] != ".pdf"):
                print("stats argument needs to end on .pdf")
                usage()
                sys.exit(2)

    if 4 != len(args):
        print("Wrong number of files. Exactly four file are required.\n")
        usage()
        sys.exit(2)

    MiniGapScaffold(args[0], args[1], args[2], args[3], prefix, stats)

if __name__ == "__main__":
    main(sys.argv[1:])
