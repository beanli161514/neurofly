import os

from neurofly.neurodb.neurodb_sqlite import NeurodbSQLite
from neurofly.backend.neuron_graph import NeuroGraph
from neurofly.util.data_conversion import CC_from_db_to_graph, graph2swc
from neurofly.util.length import cal_length_from_swc_interp, cal_length_from_swc_noInterp


def get_soma_nodes(DB:NeurodbSQLite):
    soma_nodes = DB.read_nodes(ntype=1)
    return soma_nodes

def export_swc_from_db(db_path, save_path):
    def __print_log__(logger:list, start_idx, end_index):
        for idx in range(start_idx, end_index):
            _log:str = logger[idx]
            print(_log.strip('\n'))
        
    DB = NeurodbSQLite(db_path)
    soma_nodes = get_soma_nodes(DB)
    print(soma_nodes)
    if not soma_nodes:
        print("No soma node found in the database.")
        return
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    save_path = os.path.abspath(save_path)

    soma_nids_list = list(soma_nodes.keys())
    logger = []
    logger.append(f'Soma nids as starting points: {soma_nids_list}\n')
    logger.append('-' * 10+ '\n')
    log_index = 0
    for soma_nid in soma_nids_list:
        # get connected components from database
        G:NeuroGraph = CC_from_db_to_graph(DB, [soma_nid])[0]
        soma_coord = soma_nodes[soma_nid]['coord']
        # print soma nid and coord
        logger.append(f'Soma nid: [{soma_nid}]; Soma Coord: {soma_coord}\n')

        # print nodes count and edges count
        logger.append(f'Nodes count: {len(G.nodes())}, Edges count: {len(G.edges())}\n')
        
        SWC, flag, swc_logger = graph2swc(G)
        logger.append(f'[Result]: {flag}, {swc_logger}\n')
        if flag == 'error':
            logger.append('Skip this soma nid due to error.\n')
            __print_log__(logger, log_index, len(logger))
            log_index = len(logger) - 1
            continue
        
        length_total_interp, length_logger = cal_length_from_swc_interp(SWC, return_log=True)
        length_total_no_interp = cal_length_from_swc_noInterp(SWC)
        for _line in length_logger:
            logger.append(f'[Log]: {_line}\n')
        logger.append(f'[Total length]: {length_total_interp:.5f} um\n')
        logger.append(f'[Total length without interpolation]: {length_total_no_interp:.5f} um\n')
        
        # add soma id and length to file name
        swc_file_name = f'soma{soma_nid}_len({length_total_interp:.5f}um).swc'
        swc_filepath = os.path.join(save_path, swc_file_name)
        with open(swc_filepath, 'w') as f:
            f.writelines(SWC)
        logger.append(f"Exported SWC for soma nid {soma_nid} to {swc_filepath} ({flag}: {swc_logger})\n")
        logger.append('-' * 10+ '\n')

        __print_log__(logger, log_index, len(logger))
        log_index = len(logger) - 1
    
    log_file_path = os.path.join(save_path, 'export_swc_log.txt')
    logger.append(f"Log file saved to {log_file_path}\n")
    print(logger[-1])
    with open(log_file_path, 'w') as log_file:
        log_file.writelines(logger)
