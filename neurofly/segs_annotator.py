import numpy as np
import networkx as nx
import napari
import random
import shutil
from neurofly.dbio import read_edges, read_nodes, add_nodes, add_edges, check_node, uncheck_nodes, change_type, delete_nodes
from magicgui import widgets
from neurofly.image_reader import wrap_image
from napari.utils.notifications import show_info
from rtree import index
from neurofly.common import *

# use PushButton as a recorder of history
class PushButton(widgets.PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Annotator(widgets.Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        # --------- GUI ---------
        self.viewer = viewer
        self.viewer.dims.ndisplay = 3
        self.viewer.layers.clear()
        self.viewer.window.remove_dock_widget('all')
        # panorama mode
        self.panorama_image = self.viewer.add_image(np.ones((64, 64, 64), dtype=np.uint16), name='panorama image', metadata={'loaded': False},visible=False)
        self.panorama_points = self.viewer.add_points(None,ndim=3,size=None,shading='spherical',border_width=0,properties=None,face_colormap='hsl',name='panorama view',blending='additive',visible=True)
        self.panorama_points.click_get_value = self.panorama_points.get_value
        self.panorama_points.get_value = lambda position, view_direction=None, dims_displayed=None, world=False: None
        # labeling mode
        self.image_layer = self.viewer.add_image(np.ones((64, 64, 64), dtype=np.uint16),name='image',visible=False)
        self.point_layer = self.viewer.add_points(None,ndim=3,size=None,shading='spherical',border_width=0,properties=None,face_colormap='hsl',name='points',visible=False)
        self.edge_layer = self.viewer.add_vectors(None,ndim=3,name='added edges',vector_style='triangle',visible=False, edge_color='orange',opacity=1.0)
        self.ex_edge_layer = self.viewer.add_vectors(None,ndim=3,name='existing edges',vector_style='line',visible=False,edge_width=0.3,opacity=0.15)
        # ------------------------

        # path extension model
        self.dis_predictor = PosPredictor(default_transformer_weight_path)

        # node type notations according to http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
        self.node_types = [
            "undefined",         # 0
            "soma",              # 1
            "axon",              # 2
            "(basal) dendrite",  # 3
            "apical dendrite",   # 4
            "fork point",        # 5
            "end point",         # 6
            "ambiguous"          # 7
        ]

        self.num_branches = 10

        self.add_control()

        # --------- data structure ---------
        self.image = None
        self.G = None # networkx graph
        self.rtree = None
        self.deleted = {
            'nodes': [],
            'edges': []
        }
        self.added = {
            'nodes': [],
            'edges': []
        }
        # ----------------------------------


    def add_control(self):
        # ----- napari bindings -----
        self.viewer.bind_key('q', self.switch_mode, overwrite=True)
        self.viewer.bind_key('r', self.recover, overwrite=True)
        self.viewer.bind_key('g', self.switch_layer, overwrite=True)
        self.viewer.bind_key('d', self.refresh, overwrite=True)
        self.viewer.bind_key('i', self.deconvolve, overwrite=True)
        self.point_layer.bind_key('f', self.submit_result, overwrite=True)
        self.point_layer.bind_key('w', self.connect_one_nearest, overwrite=True)
        self.point_layer.bind_key('e', self.connect_two_nearest, overwrite=True)
        self.point_layer.bind_key('d', self.predict_displacement, overwrite=True)
        self.point_layer.bind_key('b', self.last_task, overwrite=True)
        self.point_layer.bind_key('n', self.get_next_task, overwrite=True)
        self.point_layer.bind_key('c', self.purge, overwrite=True)
        self.point_layer.bind_key('Up', self.show_more_branches , overwrite=True)
        self.point_layer.bind_key('Down', self.show_less_branches , overwrite=True)

        self.point_layer.bind_key('0', self.label_undefined, overwrite=True)
        self.point_layer.bind_key('1', self.label_soma, overwrite=False)
        self.point_layer.bind_key('8', self.label_ambiguous, overwrite=True)

        self.panorama_points.mouse_drag_callbacks.append(self.node_selection)
        self.point_layer.mouse_drag_callbacks.append(self.node_operations)
        self.image_layer.mouse_drag_callbacks.append(self.put_point)
        # ---------------------------

        # ----- widgets -----
        self.user_name = widgets.LineEdit(label="user name", value='tester')
        self.image_path = widgets.FileEdit(label="image path", mode='r')
        self.db_path = widgets.FileEdit(label="database path",filter='*.db')

        self.deconv_path = widgets.FileEdit(label="Deconv model weight")
        # self.channel = widgets.LineEdit(label="channel", value=0)
        self.channel = widgets.ComboBox(
            choices=[], 
            label='Resolution Level',
            tooltip='Select resolution level'
        )
        # self.level = widgets.LineEdit(label="resolution level", value=0)
        self.level = widgets.ComboBox(
            choices=[], 
            label='Resolution Level',
            tooltip='Select resolution level'
        )
        self.image_switch = widgets.CheckBox(value=False,text='show panorama image')
        self.segs_switch = widgets.CheckBox(value=True,text='show/hide long segments')
        self.refresh_panorama_button = widgets.PushButton(text="refresh panorama")

        self.min_length = widgets.Slider(label="short segs filter", value=10, min=0, max=200)
        self.len_thres = widgets.Slider(label="length thres", value=20, min=0, max=9999)
        self.point_size = widgets.Slider(label="point size", value=3, min=1, max=10)

        self.mode_switch = PushButton(text="switch mode (q)")
        self.mode_switch.mode = 'panorama'
        self.selected_node = widgets.LineEdit(label="node selection", value=0)
        self.total_length = widgets.LineEdit(label="total length", value=0)
        self.total_nodes_left = widgets.LineEdit(label="total nodes left", value=0)
        self.nodes_left = widgets.LineEdit(label="nodes left", value=0)
        self.image_size = widgets.Slider(label="block size", value=64, min=64, max=1024)
        self.refresh_button = widgets.PushButton(text="refresh (d)")
        self.deconv_button = widgets.PushButton(text="deconvolve (i)")
        self.return_button = widgets.PushButton(text="last task (b)")
        self.recover_button = widgets.PushButton(text="recover (r)")
        self.submit_button = PushButton(text="submit (f)")
        self.submit_button.history = []
        self.proofreading_switch = widgets.CheckBox(value=False,text='Proofreading')
        # next task is just ask for new task without submitting
        self.next_task_button = widgets.PushButton(text="get next task (n)")
        self.purge_button = widgets.PushButton(text="purge block (c)")
        self.export_swc_button = widgets.PushButton(text="export swc files")
        self.node_type_dropdown = widgets.ComboBox(
            choices=self.node_types,
            label="node type",
        )
        # ---------------------------

        # ----- widgets bindings -----
        self.proofreading_switch.changed.connect(self.on_proofreading_mode_change)
        self.submit_button.clicked.connect(self.submit_result)
        self.refresh_panorama_button.clicked.connect(self.refresh_panorama)
        self.mode_switch.clicked.connect(self.switch_mode)
        self.image_size.changed.connect(self.clip_value)
        self.refresh_button.clicked.connect(lambda: self.refresh(self.viewer,keep_image=False))
        self.recover_button.clicked.connect(self.recover)
        self.return_button.clicked.connect(self.last_task)
        self.next_task_button.clicked.connect(self.get_next_task)
        self.deconv_path.changed.connect(self.load_deconver)
        self.image_path.changed.connect(self.on_reading_image)
        self.db_path.changed.connect(self.on_reading_db)
        self.deconv_button.clicked.connect(self.deconvolve)
        self.purge_button.clicked.connect(self.purge)
        self.export_swc_button.clicked.connect(self.export_swc)
        self.node_type_dropdown.changed.connect(self.on_changing_type)
        self.channel.changed.connect(self.on_changing_channel)
        self.level.changed.connect(self.on_changing_level)
        # ---------------------------

        # ------load default model weights-----
        if os.path.exists(default_dec_weight_path):
            self.deconv_path.value = default_dec_weight_path
        if os.path.exists(default_dec_weight_path):
            self.deconv_path.value = default_dec_weight_path
        # ---------------------------

        self.extend([
            self.user_name,
            self.image_path,
            self.db_path,
            self.channel,
            self.level,
            self.export_swc_button,
            self.image_switch,
            self.segs_switch,
            self.min_length,
            self.len_thres,
            self.point_size,
            self.refresh_panorama_button,
            self.mode_switch,
            self.selected_node,
            self.total_length,
            self.nodes_left,
            self.total_nodes_left,
            self.proofreading_switch,
            self.image_size,
            self.node_type_dropdown,
            self.refresh_button,
            self.return_button,
            self.recover_button,
            self.deconv_button,
            self.purge_button,
            self.next_task_button,
            self.submit_button
        ])


    def on_changing_channel(self,viewer):
        self.panorama_image.metadata['loaded'] = False
        self.refresh_panorama()
    

    def on_changing_level(self,viewer):
        self.panorama_image.metadata['loaded'] = False
        self.refresh_panorama()


    def label_undefined(self, viewer):
        if self.mode_switch.mode == 'panorama':
            show_info("switch to labeling mode")
            return
        node_id = int(self.selected_node.value)
        type_idx = 0
        self.node_type_dropdown.changed.disconnect(self.on_changing_type)
        self.node_type_dropdown.value = self.node_types[type_idx]
        self.node_type_dropdown.changed.connect(self.on_changing_type)
        change_type(str(self.db_path.value),node_id,type_idx)
        self.G.nodes[node_id]['type'] = type_idx
        show_info(f"{node_id} labeled as {str(self.node_type_dropdown.value)}")
        self.refresh(self.viewer)


    def label_soma(self, viewer):
        if self.mode_switch.mode == 'panorama':
            show_info("switch to labeling mode")
            return
        node_id = int(self.selected_node.value)
        type_idx = 1
        self.node_type_dropdown.changed.disconnect(self.on_changing_type)
        self.node_type_dropdown.value = self.node_types[type_idx]
        self.node_type_dropdown.changed.connect(self.on_changing_type)
        change_type(str(self.db_path.value),node_id,type_idx)
        self.G.nodes[node_id]['type'] = type_idx
        show_info(f"{node_id} labeled as {str(self.node_type_dropdown.value)}")
        self.refresh(self.viewer)


    def label_ambiguous(self, viewer):
        if self.mode_switch.mode == 'panorama':
            show_info("switch to labeling mode")
            return
        node_id = int(self.selected_node.value)
        type_idx = 7
        self.node_type_dropdown.changed.disconnect(self.on_changing_type)
        self.node_type_dropdown.value = self.node_types[type_idx]
        self.node_type_dropdown.changed.connect(self.on_changing_type)
        change_type(str(self.db_path.value),node_id,type_idx)
        self.G.nodes[node_id]['type'] = type_idx
        show_info(f"{node_id} labeled as {str(self.node_type_dropdown.value)}")
        self.refresh(self.viewer)


    def on_reading_image(self):
        self.image = wrap_image(str(self.image_path.value))
        resolution_levels = self.image.resolution_levels
        channels = self.image.channels
        self.channel.changed.disconnect(self.on_changing_channel)
        self.channel.choices = channels
        self.channel.value = channels[0]
        self.channel.changed.connect(self.on_changing_channel)
        # self.level.changed.disconnect(self.on_resolution_change)
        self.level.choices = resolution_levels
        self.level.value = resolution_levels[0]
        # self.level.changed.connect(self.on_resolution_change)


    def on_reading_db(self):
        self.G = None
        self.refresh_panorama()


    def on_proofreading_mode_change(self):
        self.recover(self.viewer)
        self.refresh(self.viewer,keep_image=True)
    

    def load_deconver(self):
        self.deconver = Deconver(str(self.deconv_path.value))
        show_info("Deconvolution model loaded")
    

    def deconvolve(self,viewer):
        size = list(self.image_layer.data.shape)
        if (np.array(size)<=np.array([128,128,128])).all():
            sr_img = self.deconver.process_one(self.image_layer.data)
            self.image_layer.data = sr_img
            self.refresh(self.viewer,keep_image=True)
        else:
            show_info("this image is too large, try a smaller one")


    def purge(self, viewer):
        if self.added['edges'] != []:
            show_info("please submit the result first")
            return
        selection = int(self.selected_node.value)
        c_coord = self.G.nodes[selection]['coord']

        h_size = self.image_size.value//2
        # query a larger block
        query_box = (c_coord[0]-h_size-10,c_coord[1]-h_size-10,c_coord[2]-h_size-10,c_coord[0]+h_size+20,c_coord[1]+h_size+20,c_coord[2]+h_size+20)
        nbrs = list(self.rtree.intersection(query_box, objects=False))
        # nbrs = nbrs + self.added['nodes']
        sub_g = self.G.subgraph(nbrs)
        connected_components = list(nx.connected_components(sub_g))
        nodes_to_remove = []
        for cc in connected_components:
            if len(cc) <= 4 and selection not in cc:
                nodes_to_remove += cc
        for node in nodes_to_remove:
            self.deleted['nodes'].append(self.G.nodes[node])
            for nbr in self.G.neighbors(node):
                self.deleted['edges'].append([node,nbr])
            self.rtree.delete(node, tuple(self.G.nodes[node]['coord']+self.G.nodes[node]['coord']))
            self.G.remove_node(node)
        self.refresh(self.viewer, keep_image=True)


    def get_next_task(self,viewer):
        # find the largest unchecked component, set one of its endings selected node.
        if self.mode_switch.value == 'panorama':
            show_info("switch to labeling mode first")
            return
        nodes_left = [
            node for node in self.G.nodes
            if (self.G.nodes[node]['checked'] == -1) or (self.G.degree(node) == 1 and self.G.nodes[node]['checked'] == 0)
        ]
        self.total_nodes_left.value = len(nodes_left)

        connected_components = list(nx.connected_components(self.G))
        connected_components.sort(key=len)

        unchecked_nodes = []
        for cc in connected_components[::-1]:
            unchecked_nodes = []
            for node in cc:
                if ((self.G.degree(node) == 1 and self.G.nodes[node]['checked'] == 0)) or self.G.nodes[node]['checked'] == -1:
                    unchecked_nodes.append(node)
            if len(unchecked_nodes)>0:
                break
        
        if len(unchecked_nodes)==0:
            show_info("all nodes checked")
            return

        self.selected_node.value = str(unchecked_nodes[0])
        self.deleted = {
            'nodes': [],
            'edges': []
        }
        self.added = {
            'nodes': [],
            'edges': []
        }
        self.refresh_edge_layer()
        self.refresh(self.viewer,keep_image=False)
    

    def show_more_branches(self, viewer):
        if self.proofreading_switch.value == False:
            return
        self.num_branches += 1
        self.refresh(self.viewer)


    def show_less_branches(self, viewer):
        if self.proofreading_switch.value == False:
            return
        if self.num_branches >= 2:
            self.num_branches -= 1
        self.refresh(self.viewer)


    def on_changing_type(self):
        if self.mode_switch.mode == 'panorama':
            show_info("switch to labeling mode")
            return
        node_id = int(self.selected_node.value)
        type_idx = self.node_types.index(str(self.node_type_dropdown.value))
        change_type(str(self.db_path.value),node_id,type_idx)
        self.G.nodes[node_id]['type'] = type_idx
        show_info(f"{node_id} labeled as {str(self.node_type_dropdown.value)}")
        self.refresh(self.viewer)


    def connect_one_nearest(self,viewer):
        # find one closest neighbour point, connect it with center point
        selection = int(self.selected_node.value)
        c_coord = self.G.nodes[selection]['coord']

        h_size = self.image_size.value//2
        query_box = (c_coord[0]-h_size,c_coord[1]-h_size,c_coord[2]-h_size,c_coord[0]+h_size,c_coord[1]+h_size,c_coord[2]+h_size)
        nbrs = list(self.rtree.intersection(query_box, objects=False))

        cc = list(nx.node_connected_component(self.G,selection))
        nbrs = [i for i in nbrs if i not in cc]
        if len(nbrs)>0:
            nbr_coords = np.array([self.G.nodes[nid]['coord'] for nid in nbrs])
            distances = np.linalg.norm(nbr_coords - np.array(c_coord), axis=1)
            closest_indices = [nbrs[i] for i in np.argsort(distances)[:1]]
            self.added['edges'] = [[selection, int(closest_indices[0])]]
        self.refresh_edge_layer()
        self.refresh(self.viewer)


    def connect_two_nearest(self,viewer):
        # find one closest neighbour point, connect them with center point
        selection = int(self.selected_node.value)
        c_coord = self.G.nodes[selection]['coord']
        h_size = self.image_size.value//2
        query_box = (c_coord[0]-h_size,c_coord[1]-h_size,c_coord[2]-h_size,c_coord[0]+h_size,c_coord[1]+h_size,c_coord[2]+h_size)
        nbrs = list(self.rtree.intersection(query_box, objects=False))
        cc = nx.node_connected_component(self.G,selection)
        nbrs = [i for i in nbrs if i not in cc]
        if len(nbrs)>1:
            # sort nbr according to distance
            nbr_coords = np.array([self.G.nodes[nid]['coord'] for nid in nbrs])
            distances = np.linalg.norm(nbr_coords - np.array(c_coord), axis=1)
            closest_indices = [nbrs[i] for i in np.argsort(distances)[:2]]
            self.added['edges'] = [[selection, int(closest_indices[0])],
                                   [selection, int(closest_indices[1])]]
        self.refresh_edge_layer()
        self.refresh(self.viewer)
    

    def predict_displacement(self,viewer):
        # predict displacement of current node
        # first check if current node is the head of a segment longer than 5 points
        # then crop the image around the node and predict the displacement
        # add predicted point to the graph and update visual
        
        selection = int(self.selected_node.value)

        if self.G.degree(selection) != 1:
            show_info("current node is not the head of a valid segment")
            return
        
        traj_len = 5
        img_size = 32
        current = selection
        traj = [current]
        
        for _ in range(traj_len - 1):
            neighbors = [n for n in self.G.neighbors(current) if n not in traj]
            if len(neighbors) != 1:
                return None  # The path should be exactly one node wide
            current = neighbors[0]
            traj.append(current)
        
        traj.reverse()
        traj_coords = [self.G.nodes[n]['coord'] for n in traj]
        [x,y,z] = traj_coords[-1]
        channel = str(self.channel.value)
        resolution_level = str(self.level.value)
        img = self.image.from_roi([x-img_size//2,y-img_size//2,z-img_size//2,img_size, img_size, img_size], resolution_level, channel)

        displacement = self.dis_predictor.predict_displacement(traj_coords,img)

        new_coord = [i+j for i,j in zip(traj_coords[-1],displacement)]

        new_id = len(self.G)
        while self.G.has_node(new_id):
            new_id+=1
        
        # add node, add edge, submit result, get next task
        self.G.add_node(new_id, nid = new_id, coord = new_coord, type = 0, checked = 0, creator = self.user_name.value)
        self.G.add_edge(new_id, selection, creator = self.user_name.value)
        self.rtree.insert(new_id, tuple(new_coord + new_coord))
        self.added['nodes'].append(new_id)
        self.added['edges'].append([selection, new_id])
        self.viewer.layers.selection.active = self.point_layer
        self.submit_result(self.viewer)

    
    def clip_value(self):
        # image size should be multiples of 64
        self.image_size.value = (self.image_size.value//64)*64


    def switch_mode(self,viewer):
        if self.mode_switch.mode == 'panorama' and int(self.selected_node.value) in self.G.nodes:
            self.mode_switch.mode = 'labeling'
            self.panorama_points.visible = False
            self.panorama_image.visible = False
            self.point_layer.visible = True
            self.image_layer.visible = True
            self.edge_layer.visible = True
            self.ex_edge_layer.visible = True
            self.viewer.camera.zoom = 5
            self.refresh(self.viewer,keep_image=False)
        elif self.mode_switch.mode == 'labeling':
            self.recover(self.viewer)
            self.mode_switch.mode = 'panorama'
            self.deleted = {
                'nodes': [],
                'edges': []
            }
            self.added = {
                'nodes': [],
                'edges': []
            }
            self.refresh_panorama()
        else:
            show_info("select a node before switching to labeling mode")


    def switch_layer(self,viewer):
        if self.viewer.layers.selection.active == self.point_layer:
            self.viewer.layers.selection.active = self.image_layer
        elif self.viewer.layers.selection.active == self.image_layer:
            self.viewer.layers.selection.active = self.point_layer
    

    def last_task(self,viewer):
        # discard all changes and rollback to last task
        # TODO: solve confilicts
        if len(self.submit_button.history)>0:
            last_node = self.submit_button.history[-1]
            if last_node not in self.G.nodes:
                show_info("No history recorded")
                return
            self.recover(self.viewer)
            self.G.nodes[last_node]['checked'] = -1
            self.selected_node.value = str(last_node)
            self.submit_button.history.remove(last_node)
            self.refresh_edge_layer()
            self.refresh(self.viewer, keep_image=False)
        else:
            show_info("No history recorded")


    def refresh(self, viewer, keep_image=True):
        # update canvas according to center and size
        # it only needs one node id to generate one task
        # 1. choose one unchecked node from CC as center node
        # 2. query nodes in roi from rtree
        # 3. assign properties for nodes to identify different segments and center point, add existing edges to vector layer
        # 4. load image
        if int(self.selected_node.value) not in self.G.nodes:
            show_info("select a node first")
        connected_component = nx.node_connected_component(self.G, int(self.selected_node.value))
        unchecked_nodes = []
        for node in connected_component:
            if (self.G.degree(node) == 1 and self.G.nodes[node]['checked'] == 0) or self.G.nodes[node]['checked'] == -1:
                unchecked_nodes.append(node)
        # sort unchecked nodes according to distance between selected node
        dis = []
        for nid in unchecked_nodes:
            dis.append(nx.shortest_path_length(self.G, source=int(self.selected_node.value), target=nid))
        unchecked_nodes = [x for _ ,x in sorted(zip(dis,unchecked_nodes))]
        
        self.update_meter(len(connected_component),len(unchecked_nodes))

        if len(unchecked_nodes)==0 and self.proofreading_switch.value == False:
            show_info('all nodes checked, proofread or get next task')
            unchecked_nodes.append(int(self.selected_node.value))

        if self.proofreading_switch.value == False:
            selection = unchecked_nodes[0]
            self.selected_node.value = str(selection)
            c_coord = self.G.nodes[selection]['coord']

            h_size = self.image_size.value//2
            query_box = (c_coord[0]-h_size,c_coord[1]-h_size,c_coord[2]-h_size,c_coord[0]+h_size,c_coord[1]+h_size,c_coord[2]+h_size)
            nbrs = list(self.rtree.intersection(query_box, objects=False))
            # nbrs = nbrs + self.added['nodes']
            sub_g = self.G.subgraph(nbrs)
            connected_components = list(nx.connected_components(sub_g))
        else:
            selection = int(self.selected_node.value)
            c_coord = self.G.nodes[selection]['coord']
            self.selected_node.value = str(selection)
            connected_components = [list(nx.node_connected_component(self.G, selection))]
            nbrs = connected_components[0]

        coords = []
        sizes = []
        colors = []
        nids = []
        edges = []

        if self.proofreading_switch.value == False:
            for cc in connected_components:
                color = random.random()
                nodes = [self.G.nodes[i] for i in cc if self.G.has_node(i)]
                for node in nodes:
                    coords.append(node['coord'])
                    nids.append(node['nid'])
                    colors.append(color)
                    if node['nid']==selection:
                        sizes.append(2)
                    else:
                        sizes.append(1)
                    if node['type']==1:
                        sizes.pop(-1)
                        sizes.append(8)

            for c_node in nbrs:
                if not self.G.has_node(c_node):
                    continue
                p1 = self.G.nodes[c_node]['coord']
                for pid in list(self.G.neighbors(c_node)):
                    p2 = self.G.nodes[pid]['coord']
                    v = [j-i for i,j in zip(p1,p2)]
                    edges.append([p1,v])

        else:
            # visualize branches of the selected neuron
            branch_points = {node for node in self.G.nodes if self.G.degree[node] > 2}
            distances = {node: float('inf') for node in self.G.nodes}
            distances[selection] = 0
            
            # BFS queue (node, number of branch points encountered)
            queue = [(selection, 0)]
            visited = set([selection])
            
            while queue:
                current_node, branch_count = queue.pop(0)
                
                # If current node is a branch point, increment the branch count
                if current_node in branch_points and current_node != selection:
                    branch_count += 1
                
                for neighbor in self.G.neighbors(current_node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = branch_count
                        queue.append((neighbor, branch_count))
            

            for cc in connected_components:
                color = random.random()
                nodes = [self.G.nodes[i] for i in cc if self.G.has_node(i)]
                for node in nodes:
                    if distances[node['nid']]>=self.num_branches:
                        continue
                    coords.append(node['coord'])
                    nids.append(node['nid'])
                    if node['checked']==-1:
                        colors.append(5)
                    else:
                        colors.append(distances[node['nid']])
                    if node['nid']==selection:
                        sizes.append(2)
                    else:
                        sizes.append(1)
                    if node['type']==1:
                        sizes.pop(-1)
                        sizes.append(8)




        colors = np.array(colors)
        if np.max(colors) != np.min(colors) and len(colors)!=0:
            colors = (colors-np.min(colors))/(np.max(colors)-np.min(colors))

        properties = {
            'colors': colors,
            'nids': np.array(nids)
        }


        if keep_image == False:
            channel = str(self.channel.value)
            resolution_level = str(self.level.value)
            image = self.image.from_roi([i-self.image_size.value//2 for i in c_coord]+[self.image_size.value,self.image_size.value,self.image_size.value], level=resolution_level, channel=channel)
            translate = [int(i)-self.image_size.value//2 for i in c_coord]
            local_coords = np.array(coords) - np.array(translate)
            mask = np.all((local_coords >= np.array([0,0,0])) & (local_coords < np.array([self.image_size.value, self.image_size.value, self.image_size.value])), axis=1)
            local_coords = local_coords[mask]
            local_coords = local_coords.astype(int)
            # adjust contrast limits according to intensity distribution of foreground points
            intensities = image[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]]
            mean_value = np.mean(intensities)
            std_value = np.std(intensities)
            self.image_layer.data = image
            self.image_layer.reset_contrast_limits()
            self.image_layer.contrast_limits = [min(mean_value//2,200),mean_value+std_value]
            self.image_layer.translate = translate
            self.viewer.camera.center = c_coord


        self.node_type_dropdown.changed.disconnect(self.on_changing_type)
        self.node_type_dropdown.value = self.node_types[self.G.nodes[selection]['type']]
        self.node_type_dropdown.changed.connect(self.on_changing_type)


        self.point_layer.data = np.array(coords)
        self.point_layer.properties = properties
        self.point_layer.face_colormap = 'hsl'
        self.point_layer.face_color = 'colors'
        self.point_layer.size = sizes
        self.point_layer.selected_data = []
        self.ex_edge_layer.data = np.array(edges)
        self.viewer.layers.selection.active = self.point_layer


    def recover(self, viewer):
        # recover the preserved deleted nodes if exists
        # TODO: solve confilicts
        for node in self.deleted['nodes']:
            self.G.add_node(node['nid'], nid = node['nid'],coord = node['coord'], type = node['type'], checked = 0, creator = self.user_name.value)
            self.rtree.insert(node['nid'], tuple(node['coord']+node['coord']))
        for edge in self.deleted['edges']:
            self.G.add_edge(edge[0],edge[1],creator = self.user_name.value)

        if len(self.added['nodes'])!=0:
            for nid in self.added['nodes']:
                self.rtree.delete(nid,tuple(self.G.nodes[nid]['coord']+self.G.nodes[nid]['coord']))
            self.G.remove_nodes_from(self.added['nodes'])

        self.deleted = {
            'nodes': [],
            'edges': []
        }

        self.added = {
            'nodes': [],
            'edges': []
        }
        self.refresh_edge_layer()
        self.refresh(self.viewer)


    def submit_result(self,viewer):
        if self.mode_switch.mode != 'labeling':
            show_info('switch to labeling mode first')
            return
        self.submit_button.history.append(int(self.selected_node.value))
        self.update_database()
        self.update_local()


    def update_local(self):
        # label the center node of current task as checked in self.G
        # update canvas and local graph
        # run refresh to updata canvas
        for edge in self.added['edges']:
            self.G.add_edge(edge[0], edge[1], creator = self.user_name.value)

        self.G.nodes[int(self.selected_node.value)]['checked']+=1

        self.deleted = {
            'nodes': [],
            'edges': []
        }

        self.added = {
            'nodes': [],
            'edges': []
        }
        self.edge_layer.data = None
        self.refresh(self.viewer, keep_image=False)


    def update_database(self):
        # update database according to self.delected and self.added
        # deleted['nodes']: [{'nid','coord','creator','type','checked'}]
        # deleted['edges']: [[src,tar]]
        # added['nodes]: [{'nid','coord','creator','type','checked'}]
        # added['edges']: [[src,tar]] 
        # 1. remove nodes and edges in deleted_nodes
        # 2. add new nodes to database
        # 3. add new edges to database
        path = str(self.db_path.value)
        if self.proofreading_switch.value == True:
            # uncheck nodes from connected components
            nids = []
            for nid in nx.node_connected_component(self.G,int(self.selected_node.value)):
                node = self.G.nodes[nid]
                if node['checked'] == -1:
                    nids.append(node['nid'])
            uncheck_nodes(path,nids)
        else:
            deleted_nodes = []
            for node in self.deleted['nodes']:
                deleted_nodes.append(node['nid'])

            if len(deleted_nodes)>0:
                delete_nodes(path,deleted_nodes)

            added_nodes = []
            for nid in self.added['nodes']:
                added_nodes.append(self.G.nodes[nid])

            if len(added_nodes)>0:
                add_nodes(path,added_nodes)

            added_edges = []
            for edge in self.added['edges']:
                if edge[0] not in deleted_nodes and edge[1] not in deleted_nodes:
                    added_edges.append(edge)

            if len(added_edges)>0:
                add_edges(path, added_edges, self.user_name.value)

            check_node(path,int(self.selected_node.value))



    def update_meter(self,total_len,n_nodes):
        self.total_length.value = int(total_len)
        self.nodes_left.value = int(n_nodes)


    def refresh_panorama(self):
        if self.mode_switch.mode == 'labeling':
            show_info('switch to panorama mode first')
            return
        if self.G is None:
            p = index.Property(dimension=3)
            # load graph and rtree from database
            nodes = read_nodes(self.db_path.value)
            edges = read_edges(self.db_path.value)
            self.G = nx.Graph()
            print("loading nodes")
            rtree_data = []
            for node in nodes:
                self.G.add_node(node['nid'], nid = node['nid'], coord = node['coord'], type = node['type'], checked = node['checked'], creator = node['creator'])
                rtree_data.append((node['nid'], tuple(node['coord']+node['coord']),None))
            self.rtree = index.Index(rtree_data, properties=p)
            for edge in edges:
                self.G.add_edge(edge['src'],edge['des'],creator = edge['creator'])

            # read image
            self.image = wrap_image(str(self.image_path.value))
        
        if ('ims' in str(self.image_path.value) or 'zarr.zip' in str(self.image_path.value)) and self.panorama_image.metadata['loaded'] == False and self.image_switch.value == True:
            # iterate levels, find one with proper size
            level = 0
            for i, roi in enumerate(self.image.rois):
                if (np.array(roi[3:])<np.array([1000,1000,1000])).all():
                    level = i
                    break
            # calculate scale
            resolution_level = self.image.resolution_levels.index(str(self.level.value))
            hr_image_size = self.image.info[resolution_level]['image_size']
            lr_image_size = self.image.info[level]['image_size']
            scale = [i/j for i,j in zip(hr_image_size,lr_image_size)]

            image = self.image.from_roi(roi, level=level, channel=str(self.channel.value))
            origin = self.image.info[resolution_level]['origin']
            self.panorama_image.data = image
            # TODO: for anisotropic image
            self.panorama_image.scale = scale
            self.panorama_image.translate = origin
            self.panorama_image.visible = True
            self.panorama_image.reset_contrast_limits()
            self.panorama_image.metadata['loaded'] = True
        
        # load full image if it's tiff format
        if '.tif' in str(self.image_path.value) and self.image_switch.value == True: 
            image = self.image.from_roi(self.image.roi) 
            self.panorama_image.data = image
            self.panorama_image.visible = True
            self.panorama_image.reset_contrast_limits()


        # show the number of unlabeled nodes
        # TODO: This is a costly operation ,consider using a more efficient method
        nodes_left = [
            node for node in self.G.nodes
            if (self.G.nodes[node]['checked'] == -1) or (self.G.degree(node) == 1 and self.G.nodes[node]['checked'] == 0)
        ]
        self.total_nodes_left.value = len(nodes_left)

        connected_components = list(nx.connected_components(self.G))

        coords = []
        sizes = []
        colors = []
        nids = []


        for cc in connected_components:
            if (len(cc)<int(self.len_thres.value) and self.segs_switch.value == True) or len(cc) <= self.min_length.value:
                continue
            if (len(cc)>=int(self.len_thres.value) and self.segs_switch.value == False) or len(cc) <= self.min_length.value:
                continue
            color = random.random()
            # check empty nodes
            nodes = [self.G.nodes[i] for i in cc]
            for nid, node in zip(list(cc),nodes):
                if node == {}:
                    try:
                        delete_nodes(str(self.db_path.value),[nid])
                    except:
                        continue
                    self.G.remove_node(nid)
                    continue
                coords.append(node['coord'])
                nids.append(node['nid'])
                colors.append(color)
                sizes.append(self.point_size.value)

        if len(sizes)==0:
            show_info("No segment in range")
            return

        colors = np.array(colors)
        colors = (colors-np.min(colors))/(np.max(colors)-np.min(colors))
        colors = np.nan_to_num(colors, nan=0.5)
        sizes = np.array(sizes)

        properties = {
            'colors': colors,
            'nids': np.array(nids)
        }
        
        resolution_level = self.image.resolution_levels.index(str(self.level.value))
        camera_center  = [i + j//2 for i,j in zip(self.image.rois[resolution_level][0:3],self.image.rois[resolution_level][3:])]

        self.panorama_points.visible = True
        self.panorama_points.data = np.array(coords)
        self.panorama_points.properties = properties
        self.panorama_points.face_colormap = 'hsl'
        self.panorama_points.face_color = 'colors'
        self.panorama_points.size = sizes
        self.panorama_points.selected_data = []


        if self.image_switch.value == True:
            self.panorama_image.visible = True
        else:
            self.panorama_image.visible = False

        self.point_layer.visible = False 
        self.image_layer.visible = False
        self.edge_layer.visible = False
        self.ex_edge_layer.visible = False

        self.viewer.reset_view()
        self.viewer.camera.center = camera_center
        self.viewer.layers.selection.active = self.panorama_points


    def node_selection(self, layer, event):
        # this is appended to panorama_points layer
        if event.button == 1:
            # remove all connected points
            index = layer.click_get_value(
                event.position,
                view_direction = event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
            if index is not None:
                self.selected_node.value = str(layer.properties['nids'][index])


    def refresh_edge_layer(self):
        '''
        refresh edge layer according to self.added['edges']
        '''
        vectors = []
        p1 = self.G.nodes[int(self.selected_node.value)]['coord']
        for [_, pid] in self.added['edges']:
            p2 = self.G.nodes[pid]['coord']
            v = [j-i for i,j in zip(p1,p2)]
            vectors.append([p1,v])
        self.edge_layer.data = np.array(vectors)



    def export_swc(self):
        if self.mode_switch.mode != 'panorama':
            show_info("Switch to panorama mode first")
            return
        # create a folder alongside database file to hold swc files
        try:
            directory = os.path.dirname(self.db_path.value)
            new_dir = os.path.join(directory, 'swc_files')
            if os.path.exists(new_dir) and os.path.isdir(new_dir):
                shutil.rmtree(new_dir)
            os.makedirs(new_dir, exist_ok=True)
            print(f"Directory '{new_dir}' has been reset successfully.")
        except Exception as e:
            print(f"Error while resetting directory '{new_dir}': {e}")
        

        connected_components = list(nx.connected_components(self.G))

        total_length = 0
        for cc in connected_components:
            type = 'unknown_'
            if (len(cc)<int(self.len_thres.value) and self.segs_switch.value == True) or len(cc) <= self.min_length.value:
                continue
            if (len(cc)>=int(self.len_thres.value) and self.segs_switch.value == False) or len(cc) <= self.min_length.value:
                continue

            total_length += len(cc)*3
            subgraph = self.G.subgraph(cc)
            somas = [n for n, attr in subgraph.nodes(data=True) if attr.get('type') == 1]
            if not somas:
                somas = list(cc)[:1]
                type = 'no_soma_'
            elif len(somas) == 1:
                type = 'one_soma_'
            elif len(somas) > 1:
                type = 'many_soma_'
            soma = somas[0]
            
            # Step 2: Perform DFS to establish parent-child relationships
            parent_dict = {soma: -1}  # Soma has no parent
            stack = [soma]
            visited = set([soma])
            
            while stack:
                current = stack.pop()
                for neighbor in subgraph.neighbors(current):
                    if neighbor not in visited:
                        parent_dict[neighbor] = current
                        visited.add(neighbor)
                        stack.append(neighbor)
            
            # Check for cycles or disconnected nodes
            if len(parent_dict) != subgraph.number_of_nodes():
                print(f"Warning: Detected cycles or disconnected nodes in neuron {soma}. Exporting tree structure derived from DFS.")
            
            # Step 3: Prepare SWC content
            swc_lines = ["# Generated by NeuroFly\n"]
            for node, parent in parent_dict.items():
                attr = subgraph.nodes[node]
                coord = attr.get('coord', [0.0, 0.0, 0.0])  # Default coordinates if not provided
                node_type = attr.get('type', 0)            # Default type if not provided
                radius = attr.get('radius', 1.0)          # Default radius if not provided
                
                # Format parent ID (-1 for soma)
                parent_id = parent if parent in subgraph else -1
                
                swc_line = f"{node} {node_type} {coord[0]} {coord[1]} {coord[2]} {radius} {parent_id}\n"
                swc_lines.append(swc_line)
            
            # Step 4: Define SWC filename using soma's node ID
            swc_filename = os.path.join(new_dir, type + f"neuron_{soma}.swc")
            
            # Step 5: Write SWC file
            try:
                with open(swc_filename, 'w') as f:
                    f.writelines(swc_lines)
                print(f"Exported SWC file: {swc_filename}")
            except Exception as e:
                print(f"Error writing SWC file {swc_filename}: {e}")
        show_info(f"total length {total_length} um")


    def node_operations(self, layer, event):
        '''
        this is appended to point_layer
        node operations:
            In proofreading:
                mouse 1: switch center node
                mouse 2: label node as unchecked
            In labeling mode:
                mouse 1: add/remove edge
                mouse 2: remove node and its edges
                shift + mouse1: switch center node
        One operation contains (click type, mode, modifier)
        '''
        index = layer.get_value(
            event.position,
            view_direction = event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        if index is None:
            return
        else:
            node_id = int(self.point_layer.properties['nids'][index])
            # some ugly code
            if 'Shift' in event.modifiers:
                modifier = 'Shift'
            else:
                modifier = None

            mode = 'proofreading' if self.proofreading_switch.value == True else 'labeling'
        
        operation = (event.button, mode, modifier)
        c_node = int(self.selected_node.value)
        connected_nbrs = [edge[1] for edge in self.added['edges']] 


        if operation == (1, 'proofreading', None): # switch center node
            self.selected_node.value = str(node_id)
            self.refresh(self.viewer,keep_image=False)

        elif operation == (2, 'proofreading', None): # label node as unchecked
            if self.G.nodes[node_id]['checked'] == -1:
                self.G.nodes[node_id]['checked'] = 0
            else:
                self.G.nodes[node_id]['checked'] = -1
            self.refresh(self.viewer)

        elif operation == (1, 'labeling', None): # add/remove edge
            if node_id not in connected_nbrs:
                self.added['edges'].append([c_node, node_id])
            else:
                self.added['edges'].remove([c_node, node_id])
            # refresh edge layer
            self.refresh_edge_layer()

        elif operation == (2, 'labeling', None): # remove node and its edges
            current_cc = nx.node_connected_component(self.G, c_node)
            if len(current_cc)==1:
                # for isolated point
                if node_id in self.added['nodes']:
                    self.added['nodes'].remove(node_id)
                else:
                    self.deleted['nodes'].append(self.G.nodes[node_id])

                self.rtree.delete(node_id, tuple(self.G.nodes[node_id]['coord']+self.G.nodes[node_id]['coord']))
                self.G.remove_node(node_id)
                if node_id in connected_nbrs:
                    self.added['edges'].remove([c_node, node_id])

                self.get_next_task(self.viewer)
                return
            if node_id not in current_cc:
                # preserve the deleted node, until next submit
                if node_id in self.added['nodes']:
                    self.added['nodes'].remove(node_id)
                else:
                    self.deleted['nodes'].append(self.G.nodes[node_id])
                for nbr in self.G.neighbors(node_id):
                    self.deleted['edges'].append([node_id,nbr])
                    # after removing, label its neighbors as unchecked
                    self.G.nodes[nbr]['checked'] = 0

                self.rtree.delete(node_id, tuple(self.G.nodes[node_id]['coord']+self.G.nodes[node_id]['coord']))
                self.G.remove_node(node_id)

                if node_id in connected_nbrs:
                    self.added['edges'].remove([c_node, node_id])
                
                self.refresh_edge_layer()
                self.refresh(self.viewer)

            else:
                # cut current_cc, select the largest subgraph
                self.deleted['nodes'].append(self.G.nodes[node_id])
                # center node is not removed, keep it unchecked
                self.G.nodes[node_id]['checked']-=1
                nbrs = list(self.G.neighbors(node_id))
                for nbr in nbrs:
                    self.deleted['edges'].append([node_id,nbr])
                    self.G.nodes[nbr]['checked'] = 0
                self.rtree.delete(node_id,tuple(self.G.nodes[node_id]['coord']+self.G.nodes[node_id]['coord']))
                self.G.remove_node(node_id)
                if node_id in connected_nbrs:
                    self.added['edges'].remove([c_node, node_id])

                # select neighbor with largest connected component as new center node
                l_size = 0
                for nbr in nbrs:
                    length = len(nx.node_connected_component(self.G,nbr))
                    if length>l_size:
                        self.selected_node.value = str(nbr)
                        l_size = length

                self.refresh_edge_layer()
                self.refresh(self.viewer)


        elif operation == (1, 'labeling', 'Shift'): 
            # discard all changes, then switch center node
            self.recover(self.viewer)
            self.selected_node.value = str(node_id)
            if self.G.nodes[node_id]['checked'] >= 0:
                self.G.nodes[node_id]['checked'] = -1
            self.refresh_edge_layer()
            self.refresh(self.viewer,keep_image=False)
        else:
            show_info("operation not supported")


    def put_point(self,layer,event):
        # add new node to self.G and self.added['nodes']
        if(event.button==2 and self.proofreading_switch.value == False):
            near_point, far_point = layer.get_ray_intersections(
                event.position,
                event.view_direction,
                event.dims_displayed
            )
            sample_ray = far_point - near_point
            length_sample_vector = np.linalg.norm(sample_ray)
            increment_vector = sample_ray / (2 * length_sample_vector)
            n_iterations = int(2 * length_sample_vector)
            bbox = np.array([
                [0, layer.data.shape[0]-1],
                [0, layer.data.shape[1]-1],
                [0, layer.data.shape[2]-1]
            ])
            sample_points = []
            values = []
            for i in range(n_iterations):
                sample_point = np.asarray(near_point + i * increment_vector, dtype=int)
                sample_point = np.clip(sample_point, bbox[:, 0], bbox[:, 1])
                value = layer.data[sample_point[0], sample_point[1], sample_point[2]]
                sample_points.append(sample_point)
                values.append(value)
            max_point_index = values.index(max(values))
            max_point = sample_points[max_point_index]
            max_point = [i+int(j) for i,j in zip(max_point,self.image_layer.translate)]

            # get new node id
            new_id = len(self.G)
            while self.G.has_node(new_id):
                new_id+=10
            
            self.G.add_node(new_id, nid = new_id, coord = max_point, type = 0, checked = 0, creator = self.user_name.value)
            self.rtree.insert(new_id, tuple(max_point+max_point))
            self.added['nodes'].append(new_id)

            self.refresh(self.viewer)
            self.viewer.layers.selection.active = self.image_layer

