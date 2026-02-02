import os
import networkx as nx

from neurofly.neurodb.neurodb_sqlite import NeurodbSQLite
from neurofly.backend.neuron_graph import NeuroGraph
from neurofly.util.data_conversion import graph2swc
from neurofly.util.length import cal_length_from_swc_interp, cal_length_from_swc_noInterp


def export_swc_from_db(db_path:str, save_path:str, *, contain_soma:bool=True, len_threshold:int=20, mode:str='normal'):
    LOGGER = []
    LOG_INDEX = 0
    def __print_log__():
        nonlocal LOG_INDEX 
        start_idx, end_index = LOG_INDEX, len(LOGGER)
        for idx in range(start_idx, end_index):
            _log:str = LOGGER[idx]
            print(_log.strip('\n'))
        LOG_INDEX = len(LOGGER)

    assert mode in ['normal', 'forced'], 'Invalid mode in export_swc_from_db()'
    DB = NeurodbSQLite(db_path)
    NODES = DB.read_nodes(nids='*')
    EDGES = DB.read_edges(nids='*')
    NeuronG = NeuroGraph(nodes=NODES, edges=EDGES)
    G = NeuronG.graph

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    save_path = os.path.abspath(save_path)

    SOMA_NIDS = {_nid for _nid, _t in G.nodes(data='type') if _t==1}
    if contain_soma:
        CC = sorted((cc for cc in nx.connected_components(G) if cc & SOMA_NIDS), key=len, reverse=True)
    else:
        CC = sorted((cc for cc in nx.connected_components(G) if len(cc) >= len_threshold), key=len, reverse=True)

    FLAG_COUNTS = {
        'success': [],
        'warning': [],
        'error': []
    }
    LENGTH_INTERP_COUNTS = []
    LOGGER.append(f'=== NeuroFly: Export SWC from Database in "{mode}" mode ===\n')
    LOGGER.append('-' * 10+ '\n')
    for idx, cc in enumerate(CC):
        LOGGER.append(f'\n--- Progress [{idx+1}/{len(CC)}] ---\n')
        # get connected components from database
        sub_G:nx.Graph = nx.subgraph(G, cc)
        cc_soma_nids = set(cc).intersection(SOMA_NIDS)
        cc_soma_coords = [sub_G.nodes[_nid]['coord'] for _nid in cc_soma_nids]
        if len(cc_soma_nids) > 0:
            LOGGER.append(f'Soma nid: {cc_soma_nids}; Soma Coord: {cc_soma_coords}\n')
            swc_file_prefix = f'soma{list(cc_soma_nids)[0]}'
        else:
            LOGGER.append(f'No Soma\n')
            swc_file_prefix = f'cc{str(idx).zfill(4)}_nosoma'
        LOGGER.append(f'Nodes count: {len(G.nodes())}, Edges count: {len(G.edges())}\n')
        __print_log__()
        
        # graph to swc
        SWC, flag, swc_logger = graph2swc(sub_G)
        LOGGER.append(f'[Result]: !!!{flag}!!! {swc_logger}\n')
        FLAG_COUNTS[flag].append(cc)
        if flag == 'error':
            skip_error = False
            if mode == 'forced':
                LOGGER.append('!!!Error detected. Forced Mode: Continue Execution!!!\n')
                skip_error = True
            elif mode == 'normal':
                LOGGER.append('Skip this soma nid due to error.\n')
                skip_error = False            
            else:
                LOGGER.append('Skip this soma nid due to error.\n')
                skip_error = False
            if not skip_error:
                continue
        __print_log__()

        # calculate length
        length_total_interp, length_logger = cal_length_from_swc_interp(SWC, return_log=True)
        LENGTH_INTERP_COUNTS.append(length_total_interp)
        length_total_no_interp = cal_length_from_swc_noInterp(SWC)
        for _line in length_logger:
            LOGGER.append(f'[Log]: {_line}\n')
        LOGGER.append(f'[Total length]: {length_total_interp:.3f} um\n')
        LOGGER.append(f'[Total length without interpolation]: {length_total_no_interp:.3f} um\n')
        __print_log__()
        
        # add soma id and length to file name
        swc_file_name = f'{swc_file_prefix}_len({length_total_interp:.0f}um).swc'
        swc_filepath = os.path.join(save_path, swc_file_name)
        with open(swc_filepath, 'w') as f:
            f.writelines(SWC)
        LOGGER.append(f"Exported SWC to {swc_filepath} ({flag}: {swc_logger})\n")
        LOGGER.append('-' * 10+ '\n')
        __print_log__()
    
    # final log
    LOGGER.append(f'\n')
    LOGGER.append(f'Total: {len(CC)}; Success: {len(FLAG_COUNTS['success'])}; Warning: {len(FLAG_COUNTS['warning'])}; Error: {len(FLAG_COUNTS['error'])}.\n')
    if mode == 'forced':
        LOGGER.append(f'Forced Mode: Stored all connected components\n')
    LOGGER.append(f'Total Length: {sum(LENGTH_INTERP_COUNTS):.3f}um\n')
    log_file_path = os.path.join(save_path, 'export_swc_log.txt')
    LOGGER.append(f"Log file saved to {log_file_path}\n")
    __print_log__()
    # dump to file
    with open(log_file_path, 'w') as log_file:
        log_file.writelines(LOGGER)