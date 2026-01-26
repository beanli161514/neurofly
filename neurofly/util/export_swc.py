import os
from tqdm import tqdm

from neurofly.neurodb.neurodb_sqlite import NeurodbSQLite
from neurofly.backend.neuron_graph import NeuroGraph
from neurofly.util.data_conversion import CC_from_db_to_graph, graph2swc
from neurofly.util.length import cal_length_from_swc_interp, cal_length_from_swc_noInterp


def get_soma_nodes(DB:NeurodbSQLite):
    soma_nodes = DB.read_nodes(ntype=1)
    return soma_nodes

def export_swc_from_db(db_path:str, save_path:str, mode:str='normal'):
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
    soma_nodes = get_soma_nodes(DB)
    # print(soma_nodes)
    if not soma_nodes:
        print("No soma node found in the database.")
        return
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    save_path = os.path.abspath(save_path)

    SOMA_NIDS_LIST = list(soma_nodes.keys())
    LOGGER.append(f'=== NeuroFly: Export SWC from Database in "{mode}" mode ===\n')
    LOGGER.append(f'Soma nids as starting points: {SOMA_NIDS_LIST}\n')
    LOGGER.append('-' * 10+ '\n')

    FLAG_COUNTS = {
        'success': [],
        'warning': [],
        'error': []
    }
    LENGTH_INTERP_COUNTS = []
    for idx, soma_nid in enumerate(SOMA_NIDS_LIST):
        LOGGER.append(f'\n--- Progress [{idx}/{len(SOMA_NIDS_LIST)}] ---\n')
        # get connected components from database
        G:NeuroGraph = CC_from_db_to_graph(DB, [soma_nid])[0]
        soma_coord = soma_nodes[soma_nid]['coord']
        # print soma nid and coord
        LOGGER.append(f'Soma nid: [{soma_nid}]; Soma Coord: {soma_coord}\n')
        # print nodes count and edges count
        LOGGER.append(f'Nodes count: {len(G.nodes())}, Edges count: {len(G.edges())}\n')
        __print_log__()
        
        # graph to swc
        SWC, flag, swc_logger = graph2swc(G)
        LOGGER.append(f'[Result]: !!!{flag}!!! {swc_logger}\n')
        FLAG_COUNTS[flag].append(soma_nid)
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
        swc_file_name = f'soma{soma_nid}_len({length_total_interp:.3f}um).swc'
        swc_filepath = os.path.join(save_path, swc_file_name)
        with open(swc_filepath, 'w') as f:
            f.writelines(SWC)
        LOGGER.append(f"Exported SWC for soma nid {soma_nid} to {swc_filepath} ({flag}: {swc_logger})\n")
        LOGGER.append('-' * 10+ '\n')
        __print_log__()
    
    # final log
    LOGGER.append(f'\n')
    LOGGER.append(f'Total: {len(SOMA_NIDS_LIST)}; Success: {len(FLAG_COUNTS['success'])}; Warning: {len(FLAG_COUNTS['warning'])}; Error: {len(FLAG_COUNTS['error'])}.\n')
    LOGGER.append(f'Total Length: {sum(LENGTH_INTERP_COUNTS):.3f}um\n')
    log_file_path = os.path.join(save_path, 'export_swc_log.txt')
    LOGGER.append(f"Log file saved to {log_file_path}\n")
    __print_log__()
    # dump to file
    with open(log_file_path, 'w') as log_file:
        log_file.writelines(LOGGER)
