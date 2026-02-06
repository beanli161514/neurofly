import os
import sys
import networkx as nx
import csv

from neurofly.neurodb.neurodb_sqlite import NeurodbSQLite
from neurofly.backend.neuron_graph import NeuroGraph
from neurofly.util.data_conversion import graph2swc
from neurofly.util.length import cal_length_from_swc_interp, cal_length_from_swc_noInterp
from neurofly.util.file import Tee


def db_summary(db_path:str, save_path:str, save_swc:bool=True,  *, len_threshold:int=20, prefix:str="", angle_threshold:float=90):
    '''
    args:
        db_path: the path of Database to be summarized
        save_path: where to save summary results
        save_swc: whether to save SWC files
        len_threshold: point quantities, those below the threshold will be filtered
        prefix: the prefix of saved files
        angle_threshold: the angle threshold for removing abnormal edges while calculating length
    '''
    # check save path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    save_path = os.path.abspath(save_path)

    # logger
    log_path = os.path.join(save_path, f'{prefix}_logs.txt')
    sys.stdout = Tee(log_path, keep_stdout=True, keep_stderr=False)

    # swc dir
    if save_swc:
        swc_dir = os.path.join(save_path, f'{prefix}_SWC')
        if not os.path.exists(swc_dir):
            os.makedirs(swc_dir, exist_ok=True)

    # load DB and Graph
    DB = NeurodbSQLite(db_path)
    NODES = DB.read_nodes(nids='*')
    EDGES = DB.read_edges(nids='*')
    NeuronG = NeuroGraph(nodes=NODES, edges=EDGES)
    G = NeuronG.graph

    # get soma nids
    SOMA_NIDS = {_nid for _nid, _t in G.nodes(data='type') if _t==1}
    CC = sorted((cc for cc in nx.connected_components(G) if len(cc) >= len_threshold), key=len, reverse=True)

    # summarize
    print(f'Summary: {db_path}\n')
    SUMMARY = []
    for idx, cc in enumerate(CC):
        print(f'=== Summrizing CC [{idx}/{len(CC)}] ===')
        # get connected components from database
        sub_G:nx.Graph = nx.subgraph(G, cc)
        cc_soma_nids = set(cc).intersection(SOMA_NIDS)
        cc_soma_coords = [sub_G.nodes[_nid]['coord'] for _nid in cc_soma_nids]
        soma_num = len(cc_soma_nids)
        
        # graph to swc
        SWC, _, cycle_tag = graph2swc(sub_G)
        # calculate length
        length_total_interp, length_logger = cal_length_from_swc_interp(SWC, angle_threshold=angle_threshold, return_log=True)
        length_total_no_interp = cal_length_from_swc_noInterp(SWC)

        # logs
        print(f'{soma_num} soma detected.')
        print(f'Cycle detected.' if cycle_tag else f'No Cycle.')
        for _log_line in length_logger:
            print(f'[Length Log]: {_log_line}')
        print(f'Length: {length_total_interp:.3f}um')

        if save_swc:
            swc_file_prefix = f'soma{list(cc_soma_nids)[0]}' if soma_num>0 else f'cc{str(idx).zfill(4)}_nosoma'
            swc_file_name = f'{swc_file_prefix}_len({length_total_interp:.0f}um).swc'
            swc_filepath = os.path.join(swc_dir, swc_file_name)
            with open(swc_filepath, 'w') as f:
                f.writelines(SWC)
            print(f"Exported SWC to {swc_filepath}")
        
        summary = {
            'cc_idx': idx,
            'soma_num': soma_num,
            'cycle_tag': cycle_tag,
            'length': length_total_interp,
            'length_no_interp': length_total_no_interp,
            'length_logger': length_logger,
            'soma_nids': cc_soma_nids,
            'soma_coords': cc_soma_coords,
            'SWC': SWC,
        }
        SUMMARY.append(summary)
        print('Down\n')
    
    # save summary
    summary_save_path = os.path.join(save_path, f'{prefix}_summary.csv')
    with open(summary_save_path, 'w', newline="") as f:
        keys_keepd = ['cc_idx', 'soma_num', 'soma_nids', 'soma_coords', 'cycle_tag', 'length', 'length_no_interp']
        writer = csv.DictWriter(f, fieldnames=keys_keepd, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(SUMMARY)
    print(f'Summary Saved into {summary_save_path}')

    return SUMMARY