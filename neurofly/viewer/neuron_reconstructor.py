from magicgui import widgets
import napari
import networkx as nx
import numpy as np

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.getcwd())
    from neurofly.viewer.simple_viewer import SimpleViewer
    from neurofly.viewer.widget.database_finder import DatabaseFinder
    from neurofly.neurodb.neurodb_sqlite import NeurodbSQLite

else:
    from .simple_viewer import SimpleViewer
    from .widget.database_finder import DatabaseFinder
    from ..neurodb.neurodb_sqlite import NeurodbSQLite

class NeuroGraph():
    def __init__(self, nodes:dict=None, edges:list=None):
        self.graph = nx.Graph()
        self.init_graph(nodes, edges)

    def init_graph(self, nodes:dict, edges:list):
        self.graph.clear()
        if nodes is not None:
            for nid, coord in nodes.items():
                self.graph.add_node(nid, coord=coord)
        if edges is not None:
            for src, dst in edges:
                if src in self.graph and dst in self.graph:
                    self.graph.add_edge(src, dst)

    def get_connected_components(self):
        cc_nids = list(nx.connected_components(self.graph))
        nodes = []
        edges = []
        colors = []
        for index, nids in enumerate(cc_nids):
            color = index/len(cc_nids)
            colors += [color] * len(nids)
            coords = [self.graph.nodes[nid]['coord'] for nid in nids]
            nodes += coords
        for src, dst in self.graph.edges:
            src_coord = np.asarray(self.graph.nodes[src]['coord'])
            dst_coord = np.asarray(self.graph.nodes[dst]['coord'])
            edges.append([src_coord, dst_coord-src_coord])
        nodes = np.asarray(nodes)
        edges = np.asarray(edges)
        colors = np.asarray(colors)
        return nodes, edges, colors
        
class NeuronReconstructor(SimpleViewer):
    def __init__(self, napari_viewer:napari.Viewer):
        super().__init__(napari_viewer)
        self.nodes_layer = self.viewer.add_points(ndim=3, size=1, shading='spherical', name='nodes')
        self.edges_layer = self.viewer.add_vectors(ndim=3, vector_style='line', edge_width=0.2, opacity=0.5, edge_color='orange', name='edges')

        self.DB = None
        self.G = NeuroGraph()
        # self.add_callback()
    
    def add_callback(self):
        """Add callbacks and widgets to the viewer."""
        super().add_callback()
        self.database_finder = DatabaseFinder()

        self.extend([self.database_finder])

        self.database_finder.reset_database_path_widget_callbacks(self.on_db_loading)
        self.database_finder.reset_render_button_callback(self.render)
    
    def on_db_loading(self):
        db_path = self.database_finder.get_database_path()
        self.DB = NeurodbSQLite(db_path)
        self.render()
    
    def refresh(self):
        self.render()
        super().refresh()
    
    def render(self):
        if self.resolution_level != 0 or not self.DB:
            self.nodes_layer.data = np.zeros((0, 3))
            self.edges_layer.data = np.zeros((0, 3))
            return
        center, size = self.roi_selector.get_roi()
        roi = [center[i]-size[i]//2 for i in range(3)] + size
        nodes, edges = self.DB.read_nodes_edges_within_roi(roi)
        self.G.init_graph(nodes, edges)
        nodes, edges, colors = self.G.get_connected_components()
        self.nodes_layer.data = nodes
        self.nodes_layer.properties = {'colors': colors}
        self.nodes_layer.face_colormap = 'hsl'
        self.nodes_layer.face_color = 'colors'
        self.edges_layer.data = edges

def main():
    """Main function to run the NeuroReconstructor."""
    viewer = napari.Viewer()
    reconstructor = NeuronReconstructor(viewer)
    viewer.window.add_dock_widget(reconstructor, name='Neuron Reconstructor')
    napari.run()

if __name__ == "__main__":
    main()


        
        


