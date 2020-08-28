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

def PlotXY(pdf, xtitle, ytitle, x, y, category=[], count=[], logx=False):
    plt.close()

    if logx:    
        y = np.extract(x>0, y)
        if len(category):
            category = np.extract(x>0, category)           
        x = np.log10(np.extract(x>0,x))

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

def GetBestSubreads(mappings, subread_alignment_precision):
    # Return if we do not have subreads

    
    # Identify groups of subread mappings (of which only one is required)
    mappings['subread'] = [ re.sub(r'^(.*?/.*?)/.+', r'\1', qname) for qname in mappings['q_name'].values ]
    mappings.sort_values(['subread','t_name','t_start','t_end'], inplace=True)
    mappings['group'] = (( (mappings['subread'] != mappings['subread'].shift(1, fill_value="")) | (mappings['t_name'] != mappings['t_name'].shift(1, fill_value="")) | ((mappings['t_start']-mappings['t_start'].shift(1, fill_value=0) > subread_alignment_precision) & (np.abs(mappings['t_end'] - mappings['t_end'].shift(1, fill_value=0)) > subread_alignment_precision)) )).cumsum()
    
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

def ReadMappings(mapping_file, contig_ids, min_mapq, keep_all_subreads, subread_alignment_precision, pdf):
    mappings = ReadPaf(mapping_file)

    # Filter low mapping qualities
    if pdf:
        PlotHist(pdf, "Mapping quality", "# Mappings", mappings['mapq'], threshold=min_mapq, logy=True)
    mappings = mappings[min_mapq <= mappings['mapq']].copy()

    # Remove all but the best subread
    if not keep_all_subreads:
        mappings = GetBestSubreads(mappings, subread_alignment_precision)

    mappings['t_id'] = itemgetter(*mappings['t_name'])(contig_ids)

    return mappings

def RemoveUnmappedContigs(contigs, mappings, remove_zero_hit_contigs):
    # Schedule contigs for removal that don't have any high quality reads mapping to it
    mapped_reads = np.bincount(mappings['t_id'], minlength=len(contigs))
    if remove_zero_hit_contigs:
        contigs.loc[0==mapped_reads, 'remove'] = True

    org_contig_info = {}
    org_contig_info['num'] = {}
    org_contig_info['num']['total'] = len(contigs)
    org_contig_info['num']['removed_total'] = np.sum(contigs['remove'])
    org_contig_info['num']['removed_no_mapping'] = np.sum(0==mapped_reads)
    org_contig_info['num']['removed_duplicates'] = org_contig_info['num']['removed_total'] - org_contig_info['num']['removed_no_mapping']
    org_contig_info['num']['masked'] = np.sum(np.logical_not(contigs['remove']) & (contigs['repeat_mask_right'] == 0))
    org_contig_info['len'] = {}
    org_contig_info['len']['total'] = np.sum(contigs['length'])
    org_contig_info['len']['removed_total'] = np.sum(contigs.loc[contigs['remove'], 'length'])
    org_contig_info['len']['removed_no_mapping'] = np.sum(contigs.loc[0==mapped_reads, 'length'])
    org_contig_info['len']['removed_duplicates'] = org_contig_info['len']['removed_total'] - org_contig_info['len']['removed_no_mapping']
    org_contig_info['len']['masked'] = np.sum(contigs.loc[np.logical_not(contigs['remove']) & (contigs['repeat_mask_right'] == 0), 'length'])
    org_contig_info['max'] = {}
    org_contig_info['max']['total'] = max(contigs['length'])
    if 0 < org_contig_info['num']['removed_total']:
        org_contig_info['max']['removed_total'] = max(contigs.loc[contigs['remove'], 'length'])
    else:
        org_contig_info['max']['removed_total'] = 0
    if 0 < org_contig_info['num']['removed_no_mapping']:
        org_contig_info['max']['removed_no_mapping'] = max(contigs.loc[0==mapped_reads, 'length'])
    else:
        org_contig_info['max']['removed_no_mapping'] = 0
    if 0 < org_contig_info['num']['removed_duplicates']:
        org_contig_info['max']['removed_duplicates'] = max(contigs.loc[contigs['remove'] & (0<mapped_reads), 'length'])
    else:
        org_contig_info['max']['removed_duplicates'] = 0
    if 0 < org_contig_info['num']['masked']:
        org_contig_info['max']['masked'] = max(contigs.loc[np.logical_not(contigs['remove']) & (contigs['repeat_mask_right'] == 0), 'length'])
    else:
        org_contig_info['max']['masked'] = 0

    return contigs, org_contig_info

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
    repeats = center_repeats.merge(mappings, left_on=['con_id'], right_on=['t_id'], how='left', indicator=True)
    repeats = repeats[repeats['_merge'] == "both"].copy()
    repeats = repeats[ (repeats['start']-min_mapping_length < repeats['t_start']) & (repeats['end']+min_mapping_length > repeats['t_end']) ].copy()
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
        if len(dists_contig_ends):
            if np.sum(dists_contig_ends < 10*max_dist_contig_end):
                PlotHist(pdf, "Distance to contig end", "# Potential contig breaks", np.extract(dists_contig_ends < 10*max_dist_contig_end, dists_contig_ends), threshold=max_dist_contig_end)
            PlotHist(pdf, "Distance to contig end", "# Potential contig breaks", dists_contig_ends, threshold=max_dist_contig_end, logx=True)

    return left_breaks, right_breaks

def CallAllBreaksSpurious(mappings, max_dist_contig_end, min_length_contig_break, pdf):
    left_breaks, right_breaks = GetBrokenMappings(mappings, max_dist_contig_end, min_length_contig_break, pdf)

    spurious_break_indexes = np.unique(np.concatenate([left_breaks.index, right_breaks.index]))

    return spurious_break_indexes

def FindBreakPoints(mappings, contigs, max_dist_contig_end, min_mapping_length, min_length_contig_break, max_break_point_distance, min_num_reads, min_factor_alternatives, min_extension, pdf):
    if pdf:
        loose_reads_ends = mappings[(mappings['t_start'] > max_dist_contig_end) & np.where('+' == mappings['strand'], -1 == mappings['prev_con'], -1 == mappings['next_con'])]
        loose_reads_ends_length = np.where('+' == loose_reads_ends['strand'], loose_reads_ends['q_start']-loose_reads_ends['read_start'], loose_reads_ends['read_end']-loose_reads_ends['q_end'])
        loose_reads_ends = mappings[(mappings['t_len']-mappings['t_end'] > max_dist_contig_end) & np.where('+' == mappings['strand'], -1 == mappings['next_con'], -1 == mappings['prev_con'])]
        loose_reads_ends_length = np.concatenate([loose_reads_ends_length, np.where('+' == loose_reads_ends['strand'], loose_reads_ends['read_end']-loose_reads_ends['q_end'], loose_reads_ends['q_start']-loose_reads_ends['read_start'])])
        if len(loose_reads_ends_length):
            if np.sum(loose_reads_ends_length < 10*min_length_contig_break):
                PlotHist(pdf, "Loose end length", "# Ends", np.extract(loose_reads_ends_length < 10*min_length_contig_break, loose_reads_ends_length), threshold=min_length_contig_break, logy=True)
            PlotHist(pdf, "Loose end length", "# Ends", loose_reads_ends_length, threshold=min_length_contig_break, logx=True)

    left_breaks, right_breaks = GetBrokenMappings(mappings, max_dist_contig_end, min_length_contig_break, pdf)
    
    break_points = pd.DataFrame({ 'contig_id': np.concatenate([ left_breaks['t_id'], right_breaks['t_id'] ]),
                                  'side': np.concatenate([ np.full(len(left_breaks), 'l'), np.full(len(right_breaks), 'r') ]),
                                  'position': np.concatenate([ left_breaks['t_end'], right_breaks['t_start'] ]),
                                  'mapq': np.concatenate([ left_breaks['mapq'], right_breaks['mapq'] ]),
                                  'read_start': np.concatenate([ left_breaks['t_start'], right_breaks['t_end'] ])})
    break_points.sort_values(['contig_id','position'], inplace=True)

    if pdf:
        break_point_dist = (break_points['position'] - break_points['position'].shift(1, fill_value=0))[break_points['contig_id'] == break_points['contig_id'].shift(1, fill_value=-1)]
        if len(break_point_dist):
            if np.sum(break_point_dist <= 10*max_break_point_distance):
                PlotHist(pdf, "Break point distance", "# Break point pairs", break_point_dist[break_point_dist <= 10*max_break_point_distance], threshold=max_break_point_distance )
            PlotHist(pdf, "Break point distance", "# Break point pairs", break_point_dist, threshold=max_break_point_distance, logx=True )

    # Cluster break_points into groups
    break_points['group'] = ( (break_points['position'] - break_points['position'].shift(1, fill_value=0) > max_break_point_distance) |
                              (break_points['contig_id'] != break_points['contig_id'].shift(1, fill_value=-1)) ).cumsum()
    
    break_groups = break_points.groupby(['contig_id','group','side'])['position'].agg(['size','min','max']).unstack(fill_value=0)
    break_groups.columns = break_groups.columns.map('_'.join)
    break_groups = break_groups.rename(columns={'size_l':'left_breaks', 'size_r':'right_breaks'})
    
    break_groups.loc[0==break_groups['left_breaks'],'min_l'] = break_groups.loc[0==break_groups['left_breaks'],'min_r']
    break_groups.loc[0==break_groups['left_breaks'],'max_l'] = break_groups.loc[0==break_groups['left_breaks'],'max_r']
    break_groups.loc[0==break_groups['right_breaks'],'min_r'] = break_groups.loc[0==break_groups['right_breaks'],'min_l']
    break_groups.loc[0==break_groups['right_breaks'],'max_r'] = break_groups.loc[0==break_groups['right_breaks'],'max_l']
    break_groups['start_pos'] = break_groups[['min_l','min_r']].min(axis=1)
    break_groups['end_pos'] = break_groups[['max_l','max_r']].max(axis=1)
    
    break_groups = break_groups.reset_index().drop(columns=['min_l','max_l','min_r','max_r'])

    if pdf:
        plot_breaks = np.concatenate([break_groups.loc[break_groups['left_breaks'] < max(100,10*min_num_reads), 'left_breaks'],break_groups.loc[break_groups['right_breaks'] < max(100,10*min_num_reads), 'right_breaks']])
        if len(plot_breaks):
            PlotHist(pdf, "# Breaking reads", "# Break points", plot_breaks, threshold=min_num_reads, logy=True )
        #PlotHist(pdf, "# Breaking reads", "# Break points", np.concatenate([break_groups['left_breaks'],break_groups['right_breaks']]), threshold=min_num_reads, logx=True )

    # Find continuous reads crossing the break
    break_groups['tmp'] = break_groups['contig_id'] == break_groups['contig_id'].shift(1, fill_value=-1)
    break_groups['num'] = break_groups[['contig_id', 'tmp']].groupby('contig_id').cumsum().astype(int)
    break_groups.drop(columns=['tmp'], inplace=True)
    
    cont_list = [ break_points[['contig_id','side','read_start','mapq','group']] ]
    cont_list[0]['break'] = True
    
    for i in range(break_groups['num'].max()+1):
        breaks = pd.DataFrame({'contig':np.arange(len(contigs)), 'start':0, 'end':sys.maxsize//2, 'group':-1})
        breaks.iloc[break_groups.loc[i==break_groups['num'], 'contig_id'], [breaks.columns.get_loc('start'), breaks.columns.get_loc('end'), breaks.columns.get_loc('group')]] = break_groups.loc[i==break_groups['num'], ['start_pos', 'end_pos', 'group']].values
        continues = breaks.iloc[mappings['t_id']]
        continues = mappings[['t_id','t_start','t_end','mapq']][(mappings['t_start'].values <= continues['start'].values - min_mapping_length) & (mappings['t_end'].values >= continues['end'].values + min_mapping_length)]
        continues['group'] = breaks.iloc[continues['t_id'].values, breaks.columns.get_loc('group')].values
        continues['break'] = False
        cont_list.append( continues.rename(columns={'t_id':'contig_id', 't_start':'l', 't_end':'r'}).melt(id_vars=['contig_id','mapq','group','break'], value_vars=['l', 'r'], var_name='side', value_name='read_start') )

    # Order continues and breaking reads by trust level (mapping quality, length)
    break_points = pd.concat(cont_list, sort=True)
    break_points.loc[break_points['side']=='l','read_start'] = break_points.loc[break_points['side']=='l','read_start']*-1 # Multiply left side start by -1, so that for both sides the higher value is better now
    break_points.sort_values(['contig_id','group','side','mapq','read_start'], ascending=[True,True,True,False,False], inplace=True)
    break_points = break_points.groupby(['contig_id','group','side','mapq','read_start'], sort=False).agg(['size','sum'])
    break_points.columns = break_points.columns.droplevel()
    break_points.rename(columns={'size':'continues', 'sum':'break'}, inplace=True)
    break_points['break'] = break_points['break'].astype(int)
    break_points['continues'] -= break_points['break']
    
    # Use cumulative counts (support for this trust level and higher) and decided by first trust level (from highest to lowest) that reaches min_factor_alternatives for either side
    break_points = break_points.groupby(['contig_id','group','side'], sort=False).cumsum()
    break_points['valid_break'] = (break_points['continues']*min_factor_alternatives <= break_points['break']) & (break_points['break'] >= min_num_reads)
    break_points['valid_break'] = break_points.groupby(['contig_id','group','side'], sort=False)['valid_break'].cummax().values
    break_points['cont_veto'] = (break_points['continues'] >= break_points['break']*min_factor_alternatives) & (break_points['continues'] >= min_num_reads)
    break_points['cont_veto'] = break_points.groupby(['contig_id','group','side'], sort=False)['cont_veto'].cummax().values
    break_points.loc[break_points['valid_break'] & break_points['cont_veto'], ['valid_break','cont_veto']] = (False, False)

    if pdf:
        break_points['ratio'] = break_points['break']/break_points['continues']
        plot_valid = break_points[break_points['valid_break']].copy()
        plot_valid = plot_valid[plot_valid.groupby(['contig_id','group','side'], sort=False)['ratio'].transform(max) == plot_valid['ratio']].copy()
        plot_valid = plot_valid.groupby(['contig_id','group','side'], sort=False).last()
        plot_valid = plot_valid.groupby(['continues','break']).size().reset_index(name='count')
        plot_valid['status'] = 'accepted'
        
        plot_reject = break_points[break_points['cont_veto']].copy()
        plot_reject = plot_reject[plot_reject.groupby(['contig_id','group','side'], sort=False)['ratio'].transform(min) == plot_reject['ratio']].copy()
        plot_reject = plot_reject.groupby(['contig_id','group','side'], sort=False).last()
        plot_reject = plot_reject.groupby(['continues','break']).size().reset_index(name='count')
        plot_reject['status'] = 'rejected'
        
        break_points['decision'] = break_points['valid_break'] | break_points['cont_veto']
        plot_undecided = break_points[break_points.groupby(['contig_id','group','side'], sort=False)['decision'].transform(max) == False].copy()
        plot_undecided_lowcov = plot_undecided.groupby(['contig_id','group','side'], sort=False).last()
        plot_undecided_lowcov = plot_undecided_lowcov[plot_undecided_lowcov['continues'] + plot_undecided_lowcov['break'] < min_num_reads].copy()
        plot_undecided = plot_undecided[plot_undecided['continues'] + plot_undecided['break'] >= min_num_reads].copy()
        plot_undecided.loc[plot_undecided['ratio'] < 1.0, 'ratio'] = 1.0/plot_undecided.loc[plot_undecided['ratio'] < 1.0, 'ratio'].values
        plot_undecided = plot_undecided[plot_undecided.groupby(['contig_id','group','side'], sort=False)['ratio'].transform(max) == plot_undecided['ratio']].copy()
        plot_undecided = plot_undecided.groupby(['contig_id','group','side'], sort=False).last()
        plot_undecided = pd.concat([plot_undecided,plot_undecided_lowcov] ).groupby(['continues','break']).size().reset_index(name='count')    
        plot_undecided['status'] = 'not-accepted'

        plot_df = pd.concat([plot_valid, plot_undecided, plot_reject])
        if len(plot_df):
            PlotXY(pdf, "Continuous read", "Breaking reads", plot_df['continues'].values, plot_df['break'].values, category=plot_df['status'].values, count=plot_df['count'].values )

    break_groups = break_groups[ np.isin(break_groups['group'], np.unique(break_points.reset_index().loc[break_points['valid_break'].values,'group'])) ].copy()

    if len(break_groups):
        if pdf:
            break_groups['contig_length'] = contigs.iloc[break_groups['contig_id'], contigs.columns.get_loc('length')].values
            dists_contig_ends = np.concatenate([ break_groups.loc[break_groups['left_breaks']*min_factor_alternatives <= break_groups['right_breaks'], 'end_pos'],
                                                (break_groups['contig_length']-break_groups['start_pos'])[break_groups['left_breaks'] >= break_groups['right_breaks']*min_factor_alternatives] ])
            if len(dists_contig_ends):
                if np.sum(dists_contig_ends < 10*max_dist_contig_end):
                    PlotHist(pdf, "Distance to contig end", "# Single sided breaks", np.extract(dists_contig_ends < 10*max_dist_contig_end, dists_contig_ends), threshold=max_dist_contig_end)
                PlotHist(pdf, "Distance to contig end", "# Single sided breaks", dists_contig_ends, threshold=max_dist_contig_end, logx=True)
    
        break_groups['num'] = break_groups['contig_id'] == break_groups['contig_id'].shift(1, fill_value=-1)
        break_groups['num'] = break_groups[['contig_id', 'num']].groupby('contig_id').cumsum().astype(int)
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
    else:
        # We don't have breaks, so all are spurious
        spurious_break_indexes = np.unique(np.concatenate([left_breaks.index, right_breaks.index]))

    # All reads that are not extending contigs enough and do not have multiple mappings are non informative (except they overlap breaks)
    non_informative_mappings = mappings[(min_extension > mappings['q_start']) & (min_extension > mappings['q_len']-mappings['q_end']) & (mappings['q_name'].shift(1, fill_value='') != mappings['q_name']) & (mappings['q_name'].shift(-1, fill_value='') != mappings['q_name'])].index
    non_informative_mappings = np.setdiff1d(non_informative_mappings,np.concatenate([left_breaks.index, right_breaks.index])) # Remove breaking reads
    non_informative_mappings = mappings.loc[non_informative_mappings, ['t_id', 't_start', 't_end']]

    if len(break_groups):
        # Remove continues reads overlapping breaks from non_informative_mappings
        touching_breaks = []
        for i in range(break_groups['num'].max()+1):
            # Set breaks for every mapping, depending on their contig_id
            breaks = pd.DataFrame({'contig':np.arange(len(contigs)), 'start':sys.maxsize//2, 'end':0, 'group':-1})
            breaks.iloc[break_groups.loc[i==break_groups['num'], 'contig_id'], [breaks.columns.get_loc('start'), breaks.columns.get_loc('end')]] = break_groups.loc[i==break_groups['num'], ['start_pos', 'end_pos']].values
            breaks = breaks.iloc[non_informative_mappings['t_id']]
            # Find mappings that touch the previously set breaks (we could filter more, but this is the maximum set, so we don't declare informative mappings as non_informative_mappings, and the numbers we could filter more don't really matter speed wise)
            touching_breaks.append(non_informative_mappings[(non_informative_mappings['t_start'] <= breaks['end'].values) & (non_informative_mappings['t_end'] >= breaks['start'].values)].index)

        non_informative_mappings = np.setdiff1d(non_informative_mappings.index,np.concatenate(touching_breaks))
    else:
        # Without breaks no reads can overlap them
        non_informative_mappings = non_informative_mappings.index

    return break_groups, spurious_break_indexes, non_informative_mappings

def GetContigParts(contigs, break_groups, pdf):
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
    if len(break_groups):
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
    if len(break_groups):
        for i in range(break_groups['num'].max()+1):
            break_points = np.full(len(contigs), sys.maxsize)
            break_points[ break_groups.loc[break_groups['num']==i,'contig_id'] ] = break_groups.loc[break_groups['num']==i,'start_pos'] - min_mapping_length
            break_points = break_points[mappings['t_id']]
            mappings.loc[mappings['t_start'] > break_points , 'left_part'] += 1
    mappings['right_part'] = contigs.iloc[mappings['t_id'], contigs.columns.get_loc('last_part')].values
    if len(break_groups):
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
    mappings['con_from'] = np.maximum(mappings['t_start'].values, contig_parts.iloc[mappings['conpart'], contig_parts.columns.get_loc('start')].values)  # We cannot work with 0 < mappings['num_part'], because of reads that start before mappings['con_start'] but do not cross the break region and therefore don't have a mapping before
    mappings['con_to'] = np.minimum(mappings['t_end'].values, contig_parts.iloc[mappings['conpart'], contig_parts.columns.get_loc('end')].values)
    mappings['read_from'] = mappings['q_start']
    mappings['read_to'] = mappings['q_end']
    
    # Rough estimate of the split for the reads, better not use the new information, but not always avoidable (it's important that read_to and read_from of the next entry match, so that using this mappings avoids the rough estimate by not adding or removing any sequences)
    # Forward strand
    multi_maps = mappings[(mappings['con_from'] > mappings['t_start']) & (mappings['strand'] == '+')]
    mappings.loc[(mappings['con_from'] > mappings['t_start']) & (mappings['strand'] == '+'), 'read_from'] = np.round(multi_maps['q_end'] - (multi_maps['q_end']-multi_maps['q_start'])/(multi_maps['t_end']-multi_maps['t_start'])*(multi_maps['t_end']-multi_maps['con_from'])).astype(int)
    mappings['read_to'] = np.where((mappings['parts'] - mappings['num_part'] > 1) & (mappings['strand'] == '+'), mappings['read_from'].shift(-1, fill_value = 0), mappings['read_to'])
    multi_maps = mappings[(mappings['con_to'] < mappings['t_end']) & (mappings['strand'] == '+') & (mappings['parts'] - mappings['num_part'] == 1)]
    mappings.loc[(mappings['con_to'] < mappings['t_end']) & (mappings['strand'] == '+') & (mappings['parts'] - mappings['num_part'] == 1), 'read_to'] = np.round(multi_maps['q_start'] + (multi_maps['q_end']-multi_maps['q_start'])/(multi_maps['t_end']-multi_maps['t_start'])*(multi_maps['con_to']-multi_maps['t_start'])).astype(int)
    # Reverse strand
    multi_maps = mappings[(mappings['con_from'] > mappings['t_start']) & (mappings['strand'] == '-')]
    mappings.loc[(mappings['con_from'] > mappings['t_start']) & (mappings['strand'] == '-'), 'read_to'] = np.round(multi_maps['q_start'] + (multi_maps['q_end']-multi_maps['q_start'])/(multi_maps['t_end']-multi_maps['t_start'])*(multi_maps['t_end']-multi_maps['con_from'])).astype(int)
    mappings['read_from'] = np.where((mappings['parts'] - mappings['num_part'] > 1) & (mappings['strand'] == '-'), mappings['read_to'].shift(1, fill_value = 0), mappings['read_from'])
    multi_maps = mappings[(mappings['con_to'] < mappings['t_end']) & (mappings['strand'] == '-') & (mappings['parts'] - mappings['num_part'] == 1)]
    mappings.loc[(mappings['con_to'] < mappings['t_end']) & (mappings['strand'] == '-') & (mappings['parts'] - mappings['num_part'] == 1), 'read_from'] = np.round(multi_maps['q_end'] - (multi_maps['q_end']-multi_maps['q_start'])/(multi_maps['t_end']-multi_maps['t_start'])*(multi_maps['con_to']-multi_maps['t_start'])).astype(int)

    mappings.reset_index(inplace=True)
    mappings['matches'] = np.round(mappings['matches']/(mappings['t_end']-mappings['t_start'])*(mappings['con_to']-mappings['con_from'])).astype(int) # Rough estimate use with care!
    mappings.rename(columns={'q_name':'read_name'}, inplace=True)
    mappings = mappings[['read_name', 'read_start', 'read_end', 'read_from', 'read_to', 'strand', 'conpart', 'con_from', 'con_to', 'left_con', 'left_con_side', 'right_con', 'right_con_side', 'mapq', 'matches']].copy()
    
    # Count how many mappings each read has
    mappings['num_mappings'] = 1
    num_mappings = mappings.groupby(['read_name','read_start'], sort=False)['num_mappings'].size().values
    mappings['num_mappings'] = np.repeat(num_mappings, num_mappings)

    return mappings

def CreateBridges(left_bridge, right_bridge):
    bridges = pd.concat([left_bridge,right_bridge], ignore_index=True, sort=False)

    # Duplicate bridges and switch from to, so that both sortings have all relevant bridges
    bridges = pd.concat([bridges[['from','from_side','to','to_side','min_mapq']],
                         bridges.rename(columns={'from':'to', 'to':'from', 'from_side':'to_side', 'to_side':'from_side'})[['from','from_side','to','to_side','min_mapq']]], ignore_index=True, sort=False)

    # Bundle identical bridges
    bridges = bridges.groupby(['from','from_side','to','to_side','min_mapq']).size().reset_index(name='count')

    # Get cumulative counts (counts for this trust level and higher)
    bridges.sort_values(['from','from_side','to','to_side','min_mapq'], ascending=[True, True, True, True, False], inplace=True)
    bridges['cumcount'] = bridges.groupby(['from','from_side','to','to_side'], sort=False)['count'].cumsum().values

    bridges.drop(columns=['count'], inplace=True)

    return bridges

def MarkOrgScaffoldBridges(bridges, contig_parts, requirement):
    # requirement (-1: org scaffold, 0: unbroken org scaffold)
    bridges['org_scaffold'] = False
    bridges.loc[(bridges['from']+1 == bridges['to']) & ('r' == bridges['from_side']) & ('l' == bridges['to_side']) & (requirement < contig_parts.iloc[bridges['from'].values, contig_parts.columns.get_loc('org_dist_right')].values), 'org_scaffold'] = True
    bridges.loc[(bridges['from']-1 == bridges['to']) & ('l' == bridges['from_side']) & ('r' == bridges['to_side']) & (requirement < contig_parts.iloc[bridges['from'].values, contig_parts.columns.get_loc('org_dist_left')].values), 'org_scaffold'] = True

    return bridges

def CountAlternatives(bridges):
    bridges.sort_values(['to', 'to_side','from','from_side'], inplace=True)
    alternatives = bridges.groupby(['to','to_side'], sort=False)['high_q'].agg(['size','sum'])
    bridges['to_alt'] = np.repeat(alternatives['sum'].values.astype(int), alternatives['size'].values)
    bridges.sort_values(['from','from_side','to','to_side'], inplace=True)
    alternatives = bridges.groupby(['from','from_side'], sort=False)['high_q'].agg(['size','sum'])
    bridges['from_alt'] = np.repeat(alternatives['sum'].values.astype(int), alternatives['size'].values)

    return bridges

def FilterBridges(bridges, min_factor_alternatives, min_num_reads, org_scaffold_trust, contig_parts, pdf=None):
    if "blind" == org_scaffold_trust:
        # Make sure the org scaffolds are used if any read is there and do not use any other connection there even if we don't have a read
        bridges = MarkOrgScaffoldBridges(bridges, contig_parts, -1)

        bridges['org_scaf_conflict'] = False
        bridges.loc[('r' == bridges['from_side']) & (-1 != contig_parts.iloc[bridges['from'].values, contig_parts.columns.get_loc('org_dist_right')].values), 'org_scaf_conflict'] = True
        bridges.loc[('l' == bridges['from_side']) & (-1 != contig_parts.iloc[bridges['from'].values, contig_parts.columns.get_loc('org_dist_left')].values), 'org_scaf_conflict'] = True
        bridges.loc[('r' == bridges['to_side']) & (-1 != contig_parts.iloc[bridges['to'].values, contig_parts.columns.get_loc('org_dist_right')].values), 'org_scaf_conflict'] = True
        bridges.loc[('l' == bridges['to_side']) & (-1 != contig_parts.iloc[bridges['to'].values, contig_parts.columns.get_loc('org_dist_left')].values), 'org_scaf_conflict'] = True

        bridges = bridges[(False == bridges['org_scaf_conflict']) | bridges['org_scaffold']]

        bridges.drop(columns=['org_scaffold', 'org_scaf_conflict'], inplace=True)

    # Set lowq flags for all bridges that don't fulfill min_num_reads
    bridges['high_q'] = np.where(bridges['cumcount'] < min_num_reads, False, True)
    
    #  Set lowq flags for all bridges that are below min_factor_alternatives compared to other bridges
    bridges.sort_values(['from','from_side','min_mapq','cumcount'], ascending=[True, True, False, False], inplace=True)
    bridges['maxcount'] = np.where(bridges['from'] == bridges['to'], 0, bridges['cumcount'])
    bridges['maxcount'] = bridges.groupby(['from','from_side'], sort=False)['maxcount'].cummax()
    bridges.loc[bridges['maxcount'] >= bridges['cumcount']*min_factor_alternatives, 'high_q'] = False
    bridges.sort_values(['to','to_side','min_mapq','cumcount'], ascending=[True, True, False, False], inplace=True)
    bridges['maxcount'] = np.where(bridges['from'] == bridges['to'], 0, bridges['cumcount'])
    bridges['maxcount'] = bridges.groupby(['to','to_side'], sort=False)['maxcount'].cummax()
    bridges.loc[bridges['maxcount'] >= bridges['cumcount']*min_factor_alternatives, 'high_q'] = False
    bridges.drop(columns=['maxcount','cumcount'], inplace=True)

    #  Set lowq flags for all bridges that compete with a high-quality bridge with a higher trust level
    bridges.sort_values(['from','from_side','high_q','min_mapq'], ascending=[True, True, False,False], inplace=True)
    bridges['max_highq_mapq'] = np.where(bridges['high_q'], bridges['min_mapq'], 0)
    bridges['max_highq_mapq'] = bridges.groupby(['from','from_side'], sort=False)['max_highq_mapq'].cummax()
    bridges.sort_values(['to','to_side','high_q','min_mapq'], ascending=[True, True, False,False], inplace=True)
    bridges['max_highq_mapq2'] = np.where(bridges['high_q'], bridges['min_mapq'], 0)
    bridges['max_highq_mapq2'] = bridges.groupby(['to','to_side'], sort=False)['max_highq_mapq2'].cummax()
    bridges.loc[np.maximum(bridges['max_highq_mapq'], bridges['max_highq_mapq2']) > bridges['min_mapq'], 'high_q'] = False
    bridges.drop(columns=['min_mapq','max_highq_mapq'], inplace=True)
    
    # Reduce all identical bridges to one and count how many high quality alternatives there are
    bridges = bridges.groupby(['from','from_side','to','to_side']).max().reset_index()
    bridges = CountAlternatives(bridges)

    # Remnove low quality bridges, where high quality bridges exist
    bridges = bridges[ bridges['high_q'] | ((0 == bridges['from_alt']) & (0 == bridges['to_alt'])) ]
    
    # Store the real quality status in low_q, because high_q is potentially overwritten by original scaffold information
    bridges['low_q'] = (False == bridges['high_q'])

    if org_scaffold_trust in ["blind", "full", "basic"]:
        if "basic" == org_scaffold_trust:
            bridges = MarkOrgScaffoldBridges(bridges, contig_parts, 0) # Do not connect previously broken contigs trhough loew quality reads
        else:
            bridges = MarkOrgScaffoldBridges(bridges, contig_parts, -1)

        # Set low quality bridges to high_q if they are an original scaffold (all low quality bridges with other confirmed options are already removed)
        bridges.loc[bridges['org_scaffold'], 'high_q'] = True

        if "full" == org_scaffold_trust:
            # Set ambiguous bridges to low quality if they compeat with the original scaffold
            org_scaffolds = bridges.groupby(['from','from_side'], sort=False)['org_scaffold'].agg(['size','sum'])
            bridges.loc[np.repeat(org_scaffolds['sum'].values, org_scaffolds['size'].values) & (False == bridges['org_scaffold']), 'high_q'] = False
            bridges.sort_values(['to', 'to_side','from','from_side'], inplace=True)
            org_scaffolds = bridges.groupby(['to','to_side'], sort=False)['org_scaffold'].agg(['size','sum'])
            bridges.loc[np.repeat(org_scaffolds['sum'].values, org_scaffolds['size'].values) & (False == bridges['org_scaffold']), 'high_q'] = False

        # Remnove new low quality bridges after the original scaffold overwrite
        bridges = CountAlternatives(bridges)
        bridges = bridges[ bridges['high_q'] | ((0 == bridges['from_alt']) & (0 == bridges['to_alt'])) ]


    #if pdf:
    #    PlotHist(pdf, "# Supporting reads", "# Unambigouos connections", bridges.loc[ bridges['from_alt'] == 1, 'from_cumcount' ], threshold=min_num_reads, logx=True)
    #    PlotHist(pdf, "# Supporting reads", "# Unambigouos connections", bridges.loc[ (bridges['from_alt'] == 1) & (bridges['from_cumcount'] <= 10*min_num_reads), 'from_cumcount' ], threshold=min_num_reads)
    #    PlotHist(pdf, "# Alternative connections", "# Connections", bridges['from_alt'], logy=True)

    #if pdf:
    #    alternative_bridges = bridges[ bridges['from_maxcount'] > bridges['from_cumcount'] ]
    #    alt_ratio = alternative_bridges['from_maxcount'] / alternative_bridges['from_cumcount']
    #    PlotXY(pdf,  "# Reads main connection", "# Reads alternative connection", alternative_bridges['from_maxcount'], alternative_bridges['from_cumcount'], category=np.where(alt_ratio >= min_factor_alternatives, "accepted", "declined"))

    # Mark low quality bridges that don't have alternatives
    lowq_bridges = bridges.loc[bridges['low_q'], ['from', 'from_side', 'to', 'to_side']] # Low quality bridges (some of them are valid due to org_scaffold_trust level)
    lowq_bridges['high_q'] = True # We only need this for CountAlternatives
    lowq_bridges = CountAlternatives(lowq_bridges)
    lowq_bridges = lowq_bridges.loc[ (1 == lowq_bridges['from_alt']) & (1 == lowq_bridges['to_alt']), ['from', 'from_side', 'to', 'to_side'] ].copy()
    
    # Return only valid bridges
    bridges = bridges[bridges['high_q']].copy()
    
    bridges.drop(columns=['high_q','low_q','org_scaffold'], inplace=True)

    return bridges, lowq_bridges

def GetBridges(mappings, min_factor_alternatives, min_num_reads, org_scaffold_trust, contig_parts, pdf):
    # Get bridges
    left_bridge = mappings.loc[mappings['left_con'] >= 0, ['conpart','left_con','left_con_side','mapq']]
    left_bridge.rename(columns={'conpart':'from','left_con':'to','left_con_side':'to_side','mapq':'from_mapq'}, inplace=True)
    left_bridge['from_side'] = 'l'
    left_bridge['to_mapq'] = np.where('+' == mappings['strand'], mappings['mapq'].shift(1, fill_value = -1), mappings['mapq'].shift(-1, fill_value = -1))[mappings['left_con'] >= 0]
    left_bridge['min_mapq'] = np.where(left_bridge['to_mapq'] < left_bridge['from_mapq'], left_bridge['to_mapq'], left_bridge['from_mapq'])
    left_bridge.drop(columns=['from_mapq','to_mapq'], inplace=True)
    left_bridge.loc[left_bridge['from'] > left_bridge['to'], ['from', 'from_side', 'to', 'to_side', 'min_mapq']] = left_bridge.loc[left_bridge['from'] > left_bridge['to'], ['to', 'to_side', 'from', 'from_side', 'min_mapq']].values
    left_bridge['min_mapq'] = left_bridge['min_mapq'].astype(int)

    right_bridge = mappings.loc[mappings['right_con'] >= 0, ['conpart','right_con','right_con_side','mapq']]
    right_bridge.rename(columns={'conpart':'from','right_con':'to','right_con_side':'to_side','mapq':'from_mapq'}, inplace=True)
    right_bridge['from_side'] = 'r'
    right_bridge['to_mapq'] = np.where('-' == mappings['strand'], mappings['mapq'].shift(1, fill_value = -1), mappings['mapq'].shift(-1, fill_value = -1))[mappings['right_con'] >= 0]
    right_bridge['min_mapq'] = np.where(right_bridge['to_mapq'] < right_bridge['from_mapq'], right_bridge['to_mapq'], right_bridge['from_mapq'])
    right_bridge.drop(columns=['from_mapq','to_mapq'], inplace=True)
    right_bridge.loc[right_bridge['from'] >= right_bridge['to'], ['from', 'from_side', 'to', 'to_side', 'min_mapq']] = right_bridge.loc[right_bridge['from'] >= right_bridge['to'], ['to', 'to_side', 'from', 'from_side', 'min_mapq']].values # >= here so that contig loops are also sorted properly
    right_bridge['min_mapq'] = right_bridge['min_mapq'].astype(int)

    bridges = CreateBridges(left_bridge, right_bridge)
    bridges, lowq_bridges = FilterBridges(bridges, min_factor_alternatives, min_num_reads, org_scaffold_trust, contig_parts)

    return bridges, lowq_bridges

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
    # Get minimum matches for groups (sorting criteria: If multiple reads support a connection take the one with the highest minimum matches)
    min_matches = selected_mappings.groupby('group', sort=False)['matches'].agg(['min','size'])
    selected_mappings['min_matches'] = np.repeat(min_matches['min'].values, min_matches['size'].values)

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
    bridging_reads = selected_mappings[np.isin(selected_mappings['conpart'],selected_bridges['conpart_s'])].groupby('group')[['conpart','strand','min_matches']].agg(['first','size'])[[('conpart','first'), ('strand','first'), ('conpart','size'), ('min_matches','first')]].reset_index()
    bridging_reads.columns = ['group','conpart','strand','size','min_matches']
    bridging_reads = bridging_reads.sort_values(['conpart','strand','size','min_matches']).groupby(['conpart','strand'], sort=False).last().reset_index()
    # Select the better read from the two directions
    bridge_to = {(f,s): t for f, s, t in zip(selected_bridges['conpart_s'], selected_bridges['strand_s'], selected_bridges['conpart_e'])}
    bridging_reads['other_conpart'] = itemgetter(*zip(bridging_reads['conpart'],bridging_reads['strand']))(bridge_to)
    bridge_to = {(f,s): t for f, s, t in zip(selected_bridges['conpart_s'], selected_bridges['strand_s'], selected_bridges['strand_e'])}
    bridging_reads['other_strand'] = itemgetter(*zip(bridging_reads['conpart'],bridging_reads['strand']))(bridge_to)
    bridging_reads['other_strand'] = np.where(bridging_reads['other_strand'] == '+', '-', '+') # Invert the strand as the read from the other direction has the opposite strand and we want them to be equal if they belong to the same connection
    bridging_reads['low_conpart'] = np.minimum(bridging_reads['conpart'], bridging_reads['other_conpart'])
    bridging_reads['low_strand'] = np.where(bridging_reads['conpart'] < bridging_reads['other_conpart'], bridging_reads['strand'], bridging_reads['other_strand'])
    bridging_reads = bridging_reads.sort_values(['low_conpart','low_strand','size','min_matches']).groupby(['low_conpart','low_strand'], sort=False).last().reset_index()
    
    selected_bridges = selected_bridges.merge(bridging_reads[['group', 'conpart', 'strand']], left_on=['conpart_s','strand_s'], right_on=['conpart','strand'], how='inner').drop(columns=['conpart','strand'])
    
    # Get the read information
    selected_bridges['id'] = selected_bridges.index
    selected_bridges.rename(columns={'conpart_s':'conpart0', 'conpart_e':'conpart'+str(max([int(col[7:]) for col in selected_bridges if col.startswith('conpart') and not col.startswith('conpart_')])+1)}, inplace=True)
    selected_bridges = selected_bridges.melt(id_vars=['id','group'], value_vars=[col for col in selected_bridges if col.startswith('conpart')], var_name='pos', value_name='conpart')
    selected_bridges = selected_bridges[selected_bridges['conpart'] != -1.0]
    selected_bridges['pos'] = selected_bridges['pos'].str[7:].astype(int)
    
    selected_bridges = selected_bridges.merge(selected_mappings[['group','read_name','read_from', 'read_to', 'strand', 'conpart', 'con_from','con_to','matches','reverse','first','last']], on=['group','conpart'])
    
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

    # If a contig part is in multiple connections drop all except the one with the best matches (always keep first and last to properly handle loops)
    selected_bridges['protector'] = (selected_bridges['first'] | selected_bridges['last']).cumsum()
    selected_bridges.loc[np.logical_not(selected_bridges['first'] | selected_bridges['last']), 'protector'] = 0
    anchored_contig_parts = np.unique(selected_bridges.loc[selected_bridges['protector']==0,'conpart']).astype(int)
    selected_bridges = selected_bridges.sort_values(['conpart','protector','matches']).groupby(['conpart','protector'], sort=False).last().reset_index()
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
    connected_invalids = 2 - long_range_mappings['anchor'].shift(-1, fill_value=True) - long_range_mappings['anchor'].shift(1, fill_value=True) # We don't need to account for out of group shifts, because the groups always end at an anchor (which is the fill value we use)
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

def HandleAlternativeConnections(contig_parts, bridges, mappings):
    invalid_anchors = np.unique(np.concatenate([bridges.loc[bridges['from_alt']>1, 'from'], bridges.loc[bridges['to_alt']>1, 'to']]))

    if len(invalid_anchors):
        # Add contig parts that have at least two alternative bridges to invalid_anchors as they must be repeated somehow in the genome until we have only unique connections left
        last_len_invalid_anchors = 0
        while last_len_invalid_anchors < len(invalid_anchors):
            last_len_invalid_anchors = len(invalid_anchors)
            long_range_mappings = GetLongRangeMappings(mappings, bridges, invalid_anchors)
            if len(long_range_mappings):
                long_range_connections = CollectConnections(long_range_mappings)
                # We only need to check alternative_s as we have all connections in both directions in there
                invalid_anchors = np.unique(np.concatenate([invalid_anchors, long_range_connections.loc[long_range_connections['alternatives_s'] > 1, 'conpart_s'].astype(int)]))
    
        # Deactivate extensions for all invalid_anchors (could be circular molecules without exit and otherwise without an anchor repeat number cannot be properly resolved)
        # Unique bridges between invalid_anchors are still applied later, to merge two contigs that are repeated together
        contig_parts.iloc[invalid_anchors, contig_parts.columns.get_loc('left_con')] = -2
        contig_parts.iloc[invalid_anchors, contig_parts.columns.get_loc('right_con')] = -2
    else:
        long_range_mappings = []
    
    # Set connections for repeat_bridges (Partially overwriting the invalid_anchors from before, if we found a read spanning from a valid anchors to another through those invalid_anchors)
    if len(long_range_mappings):
        long_range_mappings = ExtendMappingInfo(long_range_mappings)
        long_range_connections, anchored_contig_parts = GetReadInfoForBridges(long_range_connections, long_range_mappings)
        contig_parts = InsertBridges(contig_parts, long_range_connections)
    else:
        anchored_contig_parts = []

    return contig_parts, anchored_contig_parts

def GetShortRangeConnections(bridges, mappings):
    # Mappings that connect two contig parts of interest
    short_mappings = mappings[mappings['num_mappings']>=2].copy()
    short_mappings = short_mappings[ np.isin(short_mappings['conpart'], np.unique(bridges['from'])) ].copy()

    # Mappings that connect on the left side to the correct contig part
    left_mappings = short_mappings[ (0 <= short_mappings['left_con']) & np.isin(short_mappings['conpart'], np.unique(bridges.loc[bridges['from_side']=='l','from'])) ].copy()
    bridge_to = {f: t for f, t in zip(bridges.loc[bridges['from_side']=='l','from'], bridges.loc[bridges['from_side']=='l','to'])}
    left_mappings = left_mappings[ left_mappings['left_con'].values == itemgetter(*left_mappings['conpart'])(bridge_to)].copy()
    bridge_to = {(f, t): s for f, t, s in zip(bridges.loc[bridges['from_side']=='l','from'], bridges.loc[bridges['from_side']=='l','to'], bridges.loc[bridges['from_side']=='l','to_side'])}
    left_mappings = left_mappings[ left_mappings['left_con_side'].values == itemgetter(*zip(left_mappings['conpart'], left_mappings['left_con']))(bridge_to)].copy()

    # Mappings that connect on the right side to the correct contig part
    right_mappings = short_mappings[ (0 <= short_mappings['right_con']) & np.isin(short_mappings['conpart'], np.unique(bridges.loc[bridges['from_side']=='r','from'])) ].copy()
    bridge_to = {f: t for f, t in zip(bridges.loc[bridges['from_side']=='r','from'], bridges.loc[bridges['from_side']=='r','to'])}
    right_mappings = right_mappings[ right_mappings['right_con'].values == itemgetter(*right_mappings['conpart'])(bridge_to)].copy()
    bridge_to = {(f, t): s for f, t, s in zip(bridges.loc[bridges['from_side']=='r','from'], bridges.loc[bridges['from_side']=='r','to'], bridges.loc[bridges['from_side']=='r','to_side'])}
    right_mappings = right_mappings[ right_mappings['right_con_side'].values == itemgetter(*zip(right_mappings['conpart'], right_mappings['right_con']))(bridge_to)].copy()

    short_mappings = pd.concat([left_mappings, right_mappings])
    short_mappings.sort_values(['read_name','read_start','read_from'], inplace=True)

    # Merge the two mappings that belong to one connection
    short_mappings = short_mappings[['read_name', 'read_from', 'read_to', 'strand', 'conpart', 'con_from', 'con_to', 'mapq', 'matches']].copy()
    short_mappings['first'] = [True, False] * (len(short_mappings)//2)
    short1 = short_mappings[short_mappings['first']].reset_index().drop(columns=['index','first'])
    short1.columns = [x+str(1) for x in short1.columns]
    short2 = short_mappings[short_mappings['first']==False].reset_index().drop(columns=['index','first'])
    short2.columns = [x+str(2) for x in short2.columns]

    connections = pd.concat([short1, short2], axis=1)
    connections.rename(columns={'read_name1':'read_name', 'read_to1':'gap_from', 'read_from2':'gap_to'}, inplace=True)
    connections.drop(columns=['read_name2','read_from1','read_to2'], inplace=True)
    connections['mapq'] = np.minimum(connections['mapq1'], connections['mapq2'])
    connections['matches'] = np.minimum(connections['matches1'], connections['matches2'])
    connections.drop(columns=['mapq1','mapq2','matches1','matches2'], inplace=True)
    connections['from'] = np.minimum(connections['conpart1'], connections['conpart2'])
    connections['to'] = np.maximum(connections['conpart1'], connections['conpart2'])
    connections['side1'] = np.where(connections['strand1'] == '+', 'r', 'l')
    connections['side2'] = np.where(connections['strand2'] == '+', 'l', 'r')
    connections['from_side'] = np.where(connections['conpart1'] < connections['conpart2'], connections['side1'], connections['side2'])
    connections['to_side'] = np.where(connections['conpart1'] > connections['conpart2'], connections['side1'], connections['side2'])

    # Select best mappings for each connection
    connections = connections.sort_values(['from','from_side','to','to_side','mapq','matches']).groupby(['from','from_side','to','to_side']).last().reset_index()
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

def HandleUniqueBridges(contig_parts, bridges, mappings, lowq_bridges):
    if len(bridges):
        connections = GetShortRangeConnections(bridges, mappings)
        contig_parts = InsertBridges(contig_parts, connections)

    
    # Mark low_quality bridges whether they are used or not (low quality bridges with alternatives have already been filtered out)
    contig_parts['right_lowq'] = False
    contig_parts['left_lowq'] = False
    if len(lowq_bridges):
        contig_parts.iloc[ lowq_bridges.loc['r' == lowq_bridges['from_side'], 'from'].values, contig_parts.columns.get_loc('right_lowq')] = True
        contig_parts.iloc[ lowq_bridges.loc['l' == lowq_bridges['from_side'], 'from'].values, contig_parts.columns.get_loc('left_lowq')] = True

    # Remove contig connections that loop to an inverted version of itself
    # (This is necessary to account for inverse repeats that are not handled by the scaffolding later on, which only stops for circular scaffolds, so left-right connections)
    # We keep the read info in the gap to complete at least what we can of the scaffold (We only have one side that is involved, so only one read info)
    # For repeated, circular contigs (left-right connections), we would need to remove the read info on one side, but that is done during scaffolding already, so we don't handle those here
    connection_break = (contig_parts['left_con'].values == contig_parts.index) & ('l' == contig_parts['left_con_side'])
    contig_parts.loc[connection_break, 'left_con_side'] = ''
    contig_parts.loc[connection_break, 'left_con'] = -2
    connection_break = (contig_parts['right_con'].values == contig_parts.index) & ('r' == contig_parts['right_con_side'])
    contig_parts.loc[connection_break, 'right_con_side'] = ''
    contig_parts.loc[connection_break, 'right_con'] = -2

    return contig_parts

def MergeCircularityGroups(circular):
    mergeable = [True]
    while np.sum(mergeable):
        circular = [ list(np.unique(x)) for x in circular ]
        circular.sort()
        mergeable = [False] + [ x[0] == y[0] for x,y in zip(circular[1:],circular[:-1]) ]
        n_groups = len(mergeable)-np.sum(mergeable)
        group_ids = np.cumsum(np.logical_not(mergeable))-1
        new_circular = [ [] for i in range(n_groups) ]
        for i, gr in enumerate(group_ids):
            new_circular[gr] += circular[i]
        circular = new_circular
        
    return new_circular

def TerminateScaffoldsForLeftContigSide(contig_parts, break_ids):
    # Terminate the scaffolds to end at the left side of break_ids
    contig_parts.iloc[ contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con')].values, contig_parts.columns.get_loc('right_con')] = -2
    contig_parts.iloc[ contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con')].values, contig_parts.columns.get_loc('right_con_side')] = ''
    contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con')] = -2
    contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con_side')] = ''
    
    # Remove read info only on one side so it will be added to the other to complete the circular scaffold
    contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con_read_name')] = ''
    contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con_read_strand')] = ''
    contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con_read_from')] = -1
    contig_parts.iloc[break_ids, contig_parts.columns.get_loc('left_con_read_to')] = -1
    
    return contig_parts

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

    while np.sum(scaffold_result['left_end_con_side'] == 'r'):
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
            scaffold_result = TerminateScaffoldsForLeftContigSide(scaffold_result, break_ids)

            # Repeat the same for contig_parts
            contig_parts = TerminateScaffoldsForLeftContigSide(contig_parts, break_ids)
            
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

def TerminateScaffoldsForRightContigSides(contig_parts, break_ids):
    # Terminate the scaffolds (idenpendent of the type of scaffold connection, we only have right-right contig connections left. All left contig connections have already been handled before, therefore we break both contigs on right side )
    contig_parts.iloc[ contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con')].values, contig_parts.columns.get_loc('right_con')] = -2
    contig_parts.iloc[ contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con')].values, contig_parts.columns.get_loc('right_con_side')] = ''
    contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con')] = -2
    contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con_side')] = ''
    
    # Remove read info only on one side so it will be added to the other to complete the circular scaffold
    contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con_read_name')] = ''
    contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con_read_strand')] = ''
    contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con_read_from')] = -1
    contig_parts.iloc[break_ids, contig_parts.columns.get_loc('right_con_read_to')] = -1
    
    return contig_parts

def HandleCircularScaffolds(scaffolds, scaffold_result, contig_parts):
    circular = scaffolds.loc[scaffolds['right_con'] == scaffolds['scaffold'], 'scaffold'].values

    scaffolds.iloc[circular, scaffolds.columns.get_loc('left_con')] = -1
    scaffolds.iloc[circular, scaffolds.columns.get_loc('right_con')] = -1
    scaffolds.iloc[circular, scaffolds.columns.get_loc('left_con_side')] = ''
    scaffolds.iloc[circular, scaffolds.columns.get_loc('right_con_side')] = ''

    # Break circularities in scaffold_result
    circular_contigs = scaffold_result.loc[np.isin(scaffold_result['scaffold'], circular) & (scaffold_result['pos'] == 0), 'part_id'].values
    scaffold_result = TerminateScaffoldsForRightContigSides(scaffold_result, circular_contigs)

    # Break circularities in contig_parts doing exactly the same
    contig_parts = TerminateScaffoldsForRightContigSides(contig_parts, circular_contigs)

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

    while np.sum(scaffolds['left_end_con_side'] == 'r'):
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
            scaffold_result = TerminateScaffoldsForRightContigSides(scaffold_result, break_ids)

            # Apply break also to contig_parts
            contig_parts = TerminateScaffoldsForRightContigSides(contig_parts, break_ids)

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
    if len(scaffold_order):
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
        while np.sum(ungrouped):
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
    
    # Remove original connections if gap is not completely untouched and if we introduced breaks(with org_dist 0) (This is only used for the scaffolding without supporting reads, so where the results are scaffolds not contigs)
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
    
        # Handle left-left scaffold connections (We don't need to handle circular scaffolds here, because that is done already in HandleLeftRightScaffoldConnections as it is needed there due to a loop)
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
    mappings['con_start'] = contig_parts.iloc[mappings['conpart'].values, contig_parts.columns.get_loc('start')].values
    mappings['con_end'] = contig_parts.iloc[mappings['conpart'].values, contig_parts.columns.get_loc('end')].values
    mappings['left_ext'] = np.where('+' == mappings['strand'], mappings['read_from']-mappings['read_start']-(mappings['con_from']-mappings['con_start']), (mappings['read_end']-mappings['read_to'])-(mappings['con_from']-mappings['con_start']))
    mappings['right_ext'] = np.where('-' == mappings['strand'], mappings['read_from']-mappings['read_start']-(mappings['con_end']-mappings['con_to']), (mappings['read_end']-mappings['read_to'])-(mappings['con_end']-mappings['con_to']))
    
    # Set extensions to zero that are in the wrong direction
    tmp_info = scaffold_info.iloc[mappings['conpart'].values]
    mappings.loc[ np.where(tmp_info['reverse'], tmp_info['scaf_size']-1 != tmp_info['pos'], tmp_info['pos'] > 0), 'left_ext'] = 0
    mappings.loc[ np.where(tmp_info['reverse'], tmp_info['pos'] > 0, tmp_info['scaf_size']-1 != tmp_info['pos']), 'right_ext'] = 0
    
    # Set extensions to zero for scaffold ends which are circular and should not be extended
    tmp_info2 = contig_parts.iloc[mappings['conpart'].values]
    mappings.loc[ (tmp_info2['left_con'] == -2).values, 'left_ext'] = 0
    mappings.loc[ (tmp_info2['right_con'] == -2).values, 'right_ext'] = 0
    
    # Set extensions to zero that are too far away from the contig_end
    mappings.loc[mappings['con_from']-mappings['con_start'] > max_dist_contig_end, 'left_ext'] = 0
    mappings.loc[mappings['con_end']-mappings['con_to'] > max_dist_contig_end, 'right_ext'] = 0
    
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
    
    mappings['left_dist_start'] = np.where(is_reversed, mappings['con_end']-mappings['con_to'], mappings['con_from']-mappings['con_start'])
    mappings['left_dist_end'] = np.where(is_reversed, mappings['con_end']-mappings['con_from'], mappings['con_to']-mappings['con_start'])
    mappings['right_dist_start'] = np.where(is_reversed, mappings['con_from']-mappings['con_start'], mappings['con_end']-mappings['con_to'])
    mappings['right_dist_end'] = np.where(is_reversed, mappings['con_to']-mappings['con_start'], mappings['con_end']-mappings['con_from'])
    
    mappings.loc[mappings['left_ext']==0, 'left_dist_start'] = -1
    mappings.loc[mappings['left_ext']==0, 'left_dist_end'] = -1
    mappings.loc[mappings['right_ext']==0, 'right_dist_start'] = -1
    mappings.loc[mappings['right_ext']==0, 'right_dist_end'] = -1
    
    mappings = mappings[['read_name','read_start','read_end','read_from','read_to', 'scaffold','left_dist_start','left_dist_end','right_dist_end','right_dist_start', 'strand','mapq','matches', 'left_ext','right_ext', 'conpart']].copy()

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
    print("  Broken: {} into {} contigs".format(np.sum(break_info>1), np.sum(break_info[break_info>1])))
    
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

def MiniGapScaffold(assembly_file, mapping_file, repeat_file, min_mapq, min_mapping_length, min_length_contig_break, prefix=False, stats=None):
    # Put in default parameters if nothing was specified
    if False == prefix:
        if ".gz" == assembly_file[-3:len(assembly_file)]:
            prefix = assembly_file.rsplit('.',2)[0]
        else:
            prefix = assembly_file.rsplit('.',1)[0]

    max_repeat_extension = 1000 # Expected to be bigger than or equal to min_mapping_length
    min_len_repeat_connection = 5000
    repeat_len_factor_unique = 10
    remove_duplicated_contigs = True
    
    remove_zero_hit_contigs = True
    min_extension = 500
    max_dist_contig_end = 2000
    max_break_point_distance = 2000
    
    min_num_reads = 3
    min_factor_alternatives = 2
    keep_all_subreads = False
    subread_alignment_precision = 100
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
    
    print( str(timedelta(seconds=clock())), "Processing repeats")
    contigs, center_repeats = MaskRepeatEnds(contigs, repeat_file, contig_ids, max_repeat_extension, min_len_repeat_connection, repeat_len_factor_unique, remove_duplicated_contigs, pdf)
    
    print( str(timedelta(seconds=clock())), "Filtering mappings")
    mappings = ReadMappings(mapping_file, contig_ids, min_mapq, keep_all_subreads, subread_alignment_precision, pdf)
    contigs, org_contig_info = RemoveUnmappedContigs(contigs, mappings, remove_zero_hit_contigs)
    mappings = RemoveUnanchoredMappings(mappings, contigs, center_repeats, min_mapping_length, pdf, max_dist_contig_end)
    del center_repeats
    
    print( str(timedelta(seconds=clock())), "Account for left-over adapters")
    mappings = BreakReadsAtAdapters(mappings, subread_alignment_precision, keep_all_subreads)
    
    print( str(timedelta(seconds=clock())), "Search for possible break points")
    if "blind" == org_scaffold_trust:
        # Do not break contigs
        break_groups = []
        spurious_break_indexes = CallAllBreaksSpurious(mappings, max_dist_contig_end, min_length_contig_break, pdf)
    else:
        break_groups, spurious_break_indexes, non_informative_mappings = FindBreakPoints(mappings, contigs, max_dist_contig_end, min_mapping_length, min_length_contig_break, max_break_point_distance, min_num_reads, min_factor_alternatives, min_extension, pdf)
    mappings.drop(np.concatenate([spurious_break_indexes, non_informative_mappings]), inplace=True) # Remove not-accepted breaks from mappings and mappings that do not contain any information (mappings inside of contigs that do not overlap with breaks)
    del spurious_break_indexes, non_informative_mappings
    
    contig_parts, contigs = GetContigParts(contigs, break_groups, pdf)
    mappings = UpdateMappingsToContigParts(mappings, contigs, contig_parts, break_groups, min_mapping_length)
    del break_groups, contigs, contig_ids
    
    print( str(timedelta(seconds=clock())), "Search for possible bridges")
    bridges, lowq_bridges = GetBridges(mappings, min_factor_alternatives, min_num_reads, org_scaffold_trust, contig_parts, pdf)
    
    contig_parts, anchored_contig_parts = HandleAlternativeConnections(contig_parts, bridges, mappings)
    bridges = bridges[(bridges['from_alt'] == 1) & (bridges['to_alt'] == 1)].copy()
    bridges = bridges[np.isin(bridges['from'], anchored_contig_parts) == False].copy()
    bridges = bridges[np.isin(bridges['to'], anchored_contig_parts) == False].copy()
    del anchored_contig_parts
    
    contig_parts = HandleUniqueBridges(contig_parts, bridges, mappings, lowq_bridges)
    del bridges, lowq_bridges
    
    print( str(timedelta(seconds=clock())), "Scaffold the contigs")
    scaffold_table, scaffold_info, contig_parts = CreateNewScaffolds(contig_parts)
    
    print( str(timedelta(seconds=clock())), "Search for possible extensions")
    mappings, contig_parts = GetScaffoldExtendingMappings(mappings, contig_parts, scaffold_info, max_dist_contig_end, min_extension, min_num_reads, pdf)

    if pdf:
        pdf.close()
        
    print( str(timedelta(seconds=clock())), "Writing output")
    mappings.to_csv(prefix+"_extensions.csv", index=False)
    scaffold_table.to_csv(prefix+"_scaffold_table.csv", index=False)
    np.savetxt(prefix+'_extending_reads.lst', np.unique(mappings['read_name']), fmt='%s')
    
    print( str(timedelta(seconds=clock())), "Finished")
    PrintStats(contig_parts, org_contig_info, min_num_reads)

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

    # Add query-target reverted copy to reads, so that all relevant read mappings show up if we search for example for q_scaffold
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
    
    # Add _partX to the contig names if they appear multiple times
    scaffold_table.sort_values(['type','name','start','end'], inplace=True)
    times_used = scaffold_table.groupby(['type','name'], sort=False).size().reset_index(name='size')
    scaffold_table['times_used'] = np.repeat(times_used['size'].values, times_used['size'].values)
    scaffold_table['part'] = 1
    scaffold_table['part'] = scaffold_table.groupby(['type','name'])['part'].cumsum()
    scaffold_table['outname'] = np.where('contig' != scaffold_table['type'], "", np.where(1 == scaffold_table['times_used'], scaffold_table['name'], scaffold_table['name']+"_part"+scaffold_table['part'].astype(str)))
    scaffold_table.sort_values(['scaffold','pos'], inplace=True)
    
    print( str(timedelta(seconds=clock())), "Writing modified assembly to: {}".format(output_file))
    with gzip.open(output_file, 'wb') if 'gz' == output_file.rsplit('.',1)[-1] else open(output_file, 'w') as fout:
        cur_scaffold = 0
        name = []
        seq = []
        for row in scaffold_table.itertuples(index=False):
            # Check if scaffold changed
            if cur_scaffold != row.scaffold:
                if cur_scaffold < row.scaffold:
                    # Write scaffold to disc
                    fout.write('>')
                    fout.write('_'.join(name))
                    fout.write('\n')
                    fout.write(''.join(seq))
                    fout.write('\n')

                    # Start new scaffold
                    cur_scaffold = row.scaffold
                    name = []
                    seq = []
                else:
                    print("Encountered scaffold {} while handling scaffold {}. The loaded scaffold table is invalid.".format(row.scaffold, cur_scaffold) )
            
            # Add sequences to scaffold
            if 'contig' == row.type:
                name.append(row.outname)
                
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
        fout.write('_'.join(name))
        fout.write('\n')
        fout.write(''.join(seq))
        fout.write('\n')
        
        print( str(timedelta(seconds=clock())), "Finished" )

def GetMappingsInRegion(mapping_file, regions, min_mapq, min_mapping_length, keep_all_subreads, subread_alignment_precision, min_length_contig_break, max_dist_contig_end):
    mappings = ReadPaf(mapping_file)
    
    # Filter mappings
    mappings = mappings[(min_mapq <= mappings['mapq']) & (min_mapping_length <= mappings['t_end'] - mappings['t_start'])].copy()
    
    # Remove all but the best subread
    if not keep_all_subreads:
        mappings = GetBestSubreads(mappings, subread_alignment_precision)
    
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
    
    # Check whether the reads support a contig break within the region
    mappings['left_break'] = ((mappings['t_len']-mappings['t_end'] > max_dist_contig_end) &
                             ( (mappings['left_scaf'] != '') | (np.where('+' == mappings['strand'], mappings['q_start'], mappings['q_len']-mappings['q_end']) > min_length_contig_break) ) &
                             (mappings['overrun_left'] == False))

    mappings['right_break'] = ((mappings['t_start'] > max_dist_contig_end) &
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
    subread_alignment_precision = 100
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
    mappings = GetMappingsInRegion(mapping_file, regions, min_mapq, min_mapping_length, keep_all_subreads, subread_alignment_precision, min_length_contig_break, max_dist_contig_end)
    
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
        print("      --minLenBreak INT     Minimum length for a read to diverge from a contig to consider a contig break")
        print("      --minMapLength INT    Minimum length of individual mappings of reads (500)")
        print("      --minMapQ INT         Minimum mapping quality of reads (20)")
    elif "extend" == module:
        print("Usage: minigap.py extend -p {prefix} {all_vs_all}.paf")
        print("Extend scaffold ends with reads reaching over the ends.")
        print("  -h, --help                Display this help and exit")
        print("  -p, --prefix FILE         Prefix for output files of scaffolding step (mandatory)")
        print("      --minLenBreak INT     Minimum length for two reads to diverge to consider them incompatible for this contig")
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
        print("      --minLenBreak INT     Minimum length for a read to diverge from a contig to consider a contig break")
        print("      --minMapLength INT    Minimum length of individual mappings of reads (500)")
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
        min_length_contig_break = 1000
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
        min_length_contig_break = 1000
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
