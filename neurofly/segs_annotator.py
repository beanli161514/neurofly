import numpy as np
import networkx as nx
import napari
import random
import shutil
import threading
from qtpy.QtCore import QTimer
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
        # shift 拖动图像层移动场景时，其余层（节点/切面板）必须跟随同一世
        # 界偏移，否则板拖走了节点停在原地。记录各层最近 translate，用户拖
        # 动产生 delta 时同步到其它层；我们自己写 translate 都走
        # _set_layer_translate（blocked + 更新记录），不会误触发
        self._scene_translate_last = {}
        for lyr in (self.panorama_image, self.panorama_points):
            self._scene_translate_last[id(lyr)] = np.asarray(lyr.translate, dtype=float)
            lyr.events.translate.connect(self._on_scene_translate)
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
        self.slice_enabled = widgets.CheckBox(label='slice plane', value=False)
        self.slice_azimuth = widgets.Slider(label='slice azimuth', value=90, min=0, max=360)
        self.slice_elevation = widgets.Slider(label='slice elevation', value=0, min=-90, max=90)
        self.slice_thickness = widgets.Slider(label='slice thickness (µm)', value=300, min=20, max=2000)
        self.slice_offset = widgets.Slider(label='slice offset %', value=50, min=0, max=100)
        self.slice_src_level = widgets.ComboBox(label='slice level', choices=[])
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
        self.image_size.changed.connect(self._on_image_size_changed)
        for w in (self.slice_enabled, self.slice_azimuth, self.slice_elevation,
                  self.slice_thickness, self.slice_offset, self.slice_src_level):
            w.changed.connect(self.update_slice_plane)
        self._recrop_timer = QTimer()
        self._recrop_timer.setSingleShot(True)
        self._recrop_timer.setInterval(500)
        self._recrop_timer.timeout.connect(self._recrop_current)
        self.slice_nodes = None
        self.slice_volume_layer = None
        self._slicer_was_on = False
        self._slice_pivot = None
        self._slice_pivot_offset = None
        self._slice_level_request = None
        # 切面清晰度：只后台读切面穿过的板状区域（世界系轴对齐包围盒，
        # 按存储块粒度实际只触碰相交块），读完把 slice volume 层换成该板
        self._slice_slab_level = None    # 已装载板的层级
        self._slice_slab_box = None      # 已装载板的 (lo, hi) level-voxel 界（hi 独占）
        self._slice_slab_loading = None  # 正在后台读的 (level_idx, lo, hi)
        self._slice_slab_pending = None  # 读好的 (level_idx, lo, hi, data)，主线程收
        self._slab_need = None           # 防抖期间记下的 (level_idx, lo, hi)
        self._slab_refused_key = None    # 超限提示只弹一次
        self._slab_read_timer = QTimer()  # 出界防抖：滑杆停稳 500ms 才发起补读
        self._slab_read_timer.setSingleShot(True)
        self._slab_read_timer.setInterval(500)
        self._slab_read_timer.timeout.connect(self._start_slab_read)
        self._slice_load_timer = QTimer()
        self._slice_load_timer.setInterval(400)
        self._slice_load_timer.timeout.connect(self._poll_slice_load)
        self._slice_load_timer.start()
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
            self.slice_enabled,
            self.slice_azimuth,
            self.slice_elevation,
            self.slice_thickness,
            self.slice_offset,
            self.slice_src_level,
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
        # hi-res slice source levels (default = the coarse panorama level;
        # user can pick a finer level for sharper slices)
        self.slice_src_level.choices = resolution_levels
        pano_level = 0
        for i, roi in enumerate(self.image.rois):
            if (np.array(roi[3:]) < np.array([1000, 1000, 1000])).all():
                pano_level = i
                break
        self.slice_src_level.value = resolution_levels[pano_level]
        # combo-heal falls back to the coarse panorama level, never to L0
        self._last_level_idx = pano_level


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
        
        if self.added['nodes'] != [] or self.added['edges'] != []:
            show_info("please submit the result first")
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
            if getattr(self, 'slice_nodes', None) is not None:
                self.slice_nodes.visible = False
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
            self._crop_block(c_coord, coords)

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


    def _heal_level_channel(self):
        """napari clears magicgui ComboBox choices whenever layers are added or
        removed; neurofly builds its choices once so they die after that
        (value becomes None and lookups like resolution_levels.index crash).
        Rebuild the level/channel choices from self.image in a blocked psygnal
        context (no handler side effects). Returns the current resolution
        level index, or None if there is no image loaded."""
        if self.image is None:
            return None
        levels = list(self.image.resolution_levels)
        if len(list(self.level.choices)) != len(levels) or self.level.value not in levels:
            prev = self.level.value if self.level.value in levels else None
            with self.level.changed.blocked():
                self.level.choices = levels
                if prev is None:
                    prev = levels[min(getattr(self, '_last_level_idx', 0), len(levels) - 1)]
                self.level.value = prev
        self._last_level_idx = levels.index(self.level.value)
        channels = list(self.image.channels)
        if len(list(self.channel.choices)) != len(channels) or self.channel.value not in channels:
            chan = self.channel.value if self.channel.value in channels else None
            with self.channel.changed.blocked():
                self.channel.choices = channels
                self.channel.value = chan if chan is not None else channels[0]
        return levels.index(self.level.value)


    def _crop_block(self, c_coord, coords):
        """Load the 64..1024³ block around c_coord and sync every coordinate
        layer to the physical (µm) world. Shared by refresh (task switch) and
        the debounced block-size re-crop."""
        if self._heal_level_channel() is None:
            return
        channel = str(self.channel.value)
        resolution_level = str(self.level.value)
        image = self.image.from_roi([i-self.image_size.value//2 for i in c_coord]+[self.image_size.value,self.image_size.value,self.image_size.value], level=resolution_level, channel=channel)
        translate = [int(i)-self.image_size.value//2 for i in c_coord]
        local_coords = np.array(coords) - np.array(translate)
        mask = np.all((local_coords >= np.array([0,0,0])) & (local_coords < np.array([self.image_size.value, self.image_size.value, self.image_size.value])), axis=1)
        local_coords = local_coords[mask]
        local_coords = local_coords.astype(int)
        # adjust contrast limits according to intensity distribution of foreground points
        if len(local_coords) == 0:
            local_coords = np.array([[self.image_size.value//2]*3])
        intensities = image[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]]
        mean_value = np.mean(intensities)
        std_value = np.std(intensities)
        self.image_layer.data = image
        self.image_layer.reset_contrast_limits()
        self.image_layer.contrast_limits = [min(mean_value//2,200),mean_value+std_value]
        # anisotropic datasets: use per-axis physical spacing (µm/voxel) for the
        # layer scale so the view keeps true proportions; node/edge layers carry
        # level-voxel absolute coordinates and follow the same spacing with
        # zero translate. The block origin is kept in voxels for put_point.
        spacing = self.image.info[self.image.resolution_levels.index(resolution_level)]['spacing']
        self._image_origin = translate
        self.image_layer.scale = spacing
        self.image_layer.translate = [i*j for i,j in zip(translate,spacing)]
        for attr in ('point_layer','edge_layer','ex_edge_layer'):
            coord_layer = getattr(self, attr, None)
            if coord_layer is not None:
                coord_layer.scale = spacing
                coord_layer.translate = [0.0,0.0,0.0]
        self.viewer.camera.center = [i*j for i,j in zip(c_coord,spacing)]


    def _on_image_size_changed(self):
        """Debounced live re-crop: the block reloads around the current
        selection without re-picking a task."""
        self._recrop_timer.start()


    def _recrop_current(self):
        if self.mode_switch.mode != 'labeling' or self.image is None or self.G is None:
            return
        try:
            nid = int(self.selected_node.value)
        except ValueError:
            return
        if nid not in self.G.nodes:
            return
        c_coord = self.G.nodes[nid]['coord']
        self._crop_block(c_coord, [c_coord])


    def _slice_plane_normal(self):
        az = np.deg2rad(self.slice_azimuth.value)
        el = np.deg2rad(self.slice_elevation.value)
        return np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])


    def _ensure_slice_nodes(self):
        if getattr(self, 'slice_nodes', None) is None:
            self.slice_nodes = self.viewer.add_points(
                name='slice nodes', face_color='#ff9f1c', size=4,
                shading='spherical', ndim=3)
            # click a node on the slice = select it (same as panorama view)
            self.slice_nodes.mouse_drag_callbacks.append(self.slice_node_selection)
        self.slice_nodes.visible = True
        return self.slice_nodes


    def slice_node_selection(self, layer, event):
        if event.button == 1:
            # napari 0.7 removed click_get_value; _get_value_ is what the
            # Points select binding itself uses
            index = layer._get_value_(
                position=event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
            if index is not None:
                self.selected_node.value = str(layer.properties['nids'][index])


    def update_slice_plane(self):
        """Slicer: a slicing plane through the whole panorama volume (coarse
        level, instant navigation). A finer 'slice level' loads that level's
        full volume in the background and swaps it into the slice layer, so
        the plane depiction samples finer voxels and the slice gets sharper."""
        if self.image is None or self.panorama_image is None or self.panorama_image.data is None \
                or not self.slice_enabled.value:
            if self.panorama_image is not None:
                # 恢复体渲染：plane depiction 下整层只画切面，整脑会"消失"
                self.panorama_image.depiction = 'volume'
                self.panorama_image.visible = True
            if getattr(self, 'slice_volume_layer', None) is not None:
                self.slice_volume_layer.visible = False
            if getattr(self, 'slice_nodes', None) is not None:
                self.slice_nodes.visible = False
            self._slicer_was_on = False
            return
        resolution_level = self._heal_level_channel()
        if resolution_level is None:
            return
        levels = list(self.image.resolution_levels)
        spacing_p = np.asarray(self.image.info[pano_idx := self._panorama_level_idx()]['spacing'], dtype=float)
        normal = self._slice_plane_normal()
        # world normal = data normal / spacing (per axis), normalized - an
        # axis-aligned normal is invariant under this, a diagonal one is not
        # (elevation tilts were ~2.8x off on anisotropic data)
        normal_world = normal / spacing_p
        normal_world /= np.linalg.norm(normal_world)
        # napari layer events clear our combo too — heal it with the panorama
        # level as fallback (never a fine level we cannot afford)
        # 用户的层级选择保存在 _slice_level_request（napari 事件清空 combo 后
        # 由这里恢复），清除/无效时才回退全景粗层
        cur = self.slice_src_level.value
        if cur in levels:
            self._slice_level_request = levels.index(cur) if levels.index(cur) != pano_idx else None
        else:
            request_idx = getattr(self, '_slice_level_request', None)
            if request_idx is None:
                request_idx = pano_idx
            request_idx = min(request_idx, len(levels) - 1)
            with self.slice_src_level.changed.blocked():
                self.slice_src_level.choices = levels
                self.slice_src_level.value = levels[request_idx]
        # 切面清晰度：请求层 ≠ 全景层 → 只读切面穿过的板状区域（见
        # _request_slice_slab，在算好切面几何后调用），层越细切面越清晰
        requested = str(self.slice_src_level.value)
        requested_idx = levels.index(requested) if requested in levels else pano_idx
        spacing_p = np.asarray(self.image.info[pano_idx]['spacing'], dtype=float)
        normal = self._slice_plane_normal()
        # world normal = data normal / spacing (per axis), normalized - an
        # axis-aligned normal is invariant under this, a diagonal one is not
        # (elevation tilts were ~2.8x off on anisotropic data)
        normal_world = normal / spacing_p
        normal_world /= np.linalg.norm(normal_world)

        # plane center (world µm): rotate around a fixed pivot; only the
        # offset slider moves the pivot (along the current world normal) -
        # otherwise rotating the normal would swing the plane center on a
        # circle and the slab would sweep the node cloud far faster than the
        # rotation itself. Pivot is clamped to the volume.
        roi = self.image.rois[pano_idx]
        volume_center = (np.asarray(roi[0:3], dtype=float) + np.asarray(roi[3:], dtype=float) / 2.0) * spacing_p
        roi_lo_w = np.asarray(roi[0:3], dtype=float) * spacing_p
        roi_hi_w = (np.asarray(roi[0:3], dtype=float) + np.asarray(roi[3:], dtype=float)) * spacing_p
        shift_max = 0.5 * float(np.linalg.norm(np.asarray(roi[3:], dtype=float) * spacing_p * normal))
        if getattr(self, '_slice_pivot', None) is None or self._slice_pivot_offset != self.slice_offset.value:
            shift = (self.slice_offset.value - 50) / 50.0 * shift_max
            self._slice_pivot = np.clip(volume_center + shift * normal_world, roi_lo_w, roi_hi_w)
            self._slice_pivot_offset = self.slice_offset.value
        P_world = np.clip(self._slice_pivot, roi_lo_w, roi_hi_w)

        # 切割平面画在复制的 slice volume 层上（depiction='plane' 只渲染切面，
        # 会遮住整层体数据）；panorama 保持 volume 渲染 = 整体体积始终可见
        pano = self.panorama_image
        pano.visible = True
        sv = self.slice_volume_layer
        if sv is None:
            self.slice_volume_layer = self.viewer.add_image(
                pano.data, name='slice volume', scale=tuple(float(v) for v in pano.scale),
                translate=tuple(float(v) for v in pano.translate),
                colormap=str(pano.colormap.name), blending='additive')
            sv = self.slice_volume_layer
            sv.depiction = 'plane'
            if id(sv) not in self._scene_translate_last:
                self._scene_translate_last[id(sv)] = np.asarray(sv.translate, dtype=float)
                sv.events.translate.connect(self._on_scene_translate)
        if self._slice_slab_level != requested_idx:
            # 目标层数据还没读到（或就是全景层）：贴全景数据兜底
            sv.data = pano.data
            sv.scale = tuple(float(v) for v in pano.scale)
            self._set_layer_translate(sv, pano.translate)
        sv.visible = True
        sv.depiction = 'plane'
        # 场景 shift 拖动的偏移 t_scene：板**读取**用真实世界坐标（P_world，
        # 与视图无关），**显示**时再加 t_scene（板跟着拖动的场景走），切面
        # 位置两边抵消恒等于真实帧 —— 之前把 t_scene 混进读取坐标，板被裁
        # 剪到体积边缘，切面落在板外，表现为切面层"消失"
        t_scene = np.asarray(self.panorama_points.translate, dtype=float)
        P_scene = P_world + t_scene
        if requested_idx != pano_idx:
            self._request_slice_slab(requested_idx, P_world, normal_world)
        sv.plane.position = (P_scene - np.asarray(sv.translate, dtype=float)) / np.asarray(sv.scale, dtype=float)
        n_d_unit = normal / np.linalg.norm(normal)
        sv.plane.normal = n_d_unit
        # 物理板厚 = thickness_data × |spacing·n̂_d|；按 sv 自己的 scale 换算
        # （sv 可能装的是细层板数据，spacing ≠ 全景层）。vispy 要求 ≥1.0 数
        # 据单位——粗层 × 小厚度下取 1.0 下限，只影响渲染板厚，节点过滤
        # 仍按真实 µm 精确执行。之前小于 1.0 时 vispy 抛异常，把本函数后半
        # 段（节点过滤）整个中断，表现为"调小 thickness 节点数不变"
        world_per_data = float(np.linalg.norm(np.asarray(sv.scale, dtype=float) * n_d_unit))
        sv.plane.thickness = max(1.0, float(self.slice_thickness.value) / world_per_data)

        # db nodes inside the slab (physical, world µm): hr-voxel coords -> world，
        # 加上场景 shift 拖动偏移（节点层 translate），与切面显示位置一致
        hr_spacing = np.asarray(self.image.info[resolution_level]['spacing'], dtype=float)
        pts = self.panorama_points.data
        props = self.panorama_points.properties
        if pts is None or len(pts) == 0:
            pts = np.array([self.G.nodes[n]['coord'] for n in self.G.nodes])
            nids_all = np.asarray(list(self.G.nodes))
        else:
            nids_all = np.asarray(props['nids']) if props and 'nids' in props else np.arange(len(pts))
        if len(pts) == 0:
            return
        t_scene = np.asarray(self.panorama_points.translate, dtype=float)
        node_world = np.asarray(pts, dtype=float) * hr_spacing + t_scene
        dist = (node_world - (P_world + t_scene)) @ normal_world
        inside = np.abs(dist) <= float(self.slice_thickness.value) / 2.0
        sl = self._ensure_slice_nodes()
        # slice-nodes carry the same hr-voxel coords as panorama_points；
        # translate 不重置——用户 shift 拖动场景后的偏移要保留
        sl.scale = hr_spacing
        sl.data = np.asarray(pts)[inside]
        sl.properties = {'nids': nids_all[inside]}
        if not self._slicer_was_on:
            # 勾选 slicer 即进入选点交互（点击节点=选中，q 进 labeling）
            self.viewer.layers.selection.active = sl
        self._slicer_was_on = True


    def _set_layer_translate(self, lyr, value):
        """我们自己的 translate 写入：blocked（不触发拖动同步）+ 更新记录"""
        value = tuple(float(v) for v in value)
        with lyr.events.blocker():
            lyr.translate = value
        self._scene_translate_last[id(lyr)] = np.asarray(value, dtype=float)


    def _on_scene_translate(self, event):
        """用户 shift 拖动任意场景层（pano/切面板/节点层）：把拖动产生的世
        界偏移 delta 同步到其余层，整场景一起动。我们自己的写入走
        _set_layer_translate（blocked），这里只响应用户拖动。"""
        src = getattr(event, 'source', None)
        if src is None:
            return
        cur = np.asarray(src.translate, dtype=float)
        prev = self._scene_translate_last.get(id(src))
        if prev is None:
            self._scene_translate_last[id(src)] = cur
            return
        if np.allclose(prev, cur):
            return
        delta = cur - prev
        self._scene_translate_last[id(src)] = cur
        for lyr in (self.panorama_image, self.panorama_points,
                    self.slice_volume_layer, self.slice_nodes):
            if lyr is None or lyr is src:
                continue
            key = id(lyr)
            base = self._scene_translate_last.get(key)
            base = np.asarray(base if base is not None else lyr.translate, dtype=float)
            new = base + delta
            self._scene_translate_last[key] = new
            with lyr.events.blocker():
                lyr.translate = tuple(float(v) for v in new)


    def _panorama_level_idx(self):
        """The coarse level refresh_panorama renders (first roi with all
        dims < 1000)."""
        for i, roi in enumerate(self.image.rois):
            if (np.array(roi[3:]) < np.array([1000, 1000, 1000])).all():
                return i
        return 0


    _SLICE_SLAB_MAX_GB = 16.0   # 单块板读取门槛（超过大概率是接近 45° 的斜切）
    _SLAB_MARGIN_VOX = 32       # 板外扩余量（体素）：小调 offset/角度不必马上补读


    @staticmethod
    def _slab_covers(box, need):
        """已装载板 box=(lo,hi) 是否完全覆盖需求 need（level-voxel 界，hi 独占）"""
        if box is None:
            return False
        lo_b, hi_b, lo_n, hi_n = (np.asarray(v) for v in (*box, *need))
        return bool(((lo_b <= lo_n).all() and (hi_n <= hi_b).all()))


    def _request_slice_slab(self, level_idx, P_world, normal_world):
        """选了比全景层细的层：只读切面 ±T/2 板在世界系的轴对齐包围盒。
        板厚沿轴 i = T/|n_i|（µm）；法线零分量轴 = 板与该轴平行，取全范围。
        已装载/在读的板盖住需求时不重读；出界则防抖后补读。读之前切面
        继续贴全景数据（或旧板）导航。"""
        levels = list(self.image.resolution_levels)
        spacing_s = np.asarray(self.image.info[level_idx]['spacing'], dtype=float)
        roi = np.asarray(self.image.rois[level_idx], dtype=float)
        size_w = roi[3:] * spacing_s
        thick = float(self.slice_thickness.value)
        abs_n = np.abs(normal_world)
        # 板厚沿轴 i = (T/2 + Σ_{j≠i}|n_j|·D_j/2)/|n_i|：除板厚外，平面在
        # 其它轴上扫过的整个范围也要计入（否则倾斜切面会被包围盒截断）；
        # 法线零分量轴 = 板与该轴平行，取全范围
        half_w = np.array([
            (thick / 2.0 + float(np.dot(np.delete(abs_n, i), (np.delete(size_w, i) / 2.0)))) / abs_n[i]
            if abs_n[i] > 1e-9 else size_w[i] / 2.0
            for i in range(3)])
        lo_w = np.clip(P_world - half_w, roi[0:3] * spacing_s, (roi[0:3] + roi[3:]) * spacing_s)
        hi_w = np.clip(P_world + half_w, roi[0:3] * spacing_s, (roi[0:3] + roi[3:]) * spacing_s)
        margin_w = self._SLAB_MARGIN_VOX * spacing_s
        lo = np.floor((lo_w - margin_w - roi[0:3] * spacing_s) / spacing_s).astype(np.int64)
        hi = np.floor((hi_w + margin_w - roi[0:3] * spacing_s) / spacing_s).astype(np.int64) + 1
        lo = np.clip(lo, 0, roi[3:].astype(np.int64))
        hi = np.clip(hi, lo + 1, roi[3:].astype(np.int64))
        gb = float(np.prod(hi - lo)) * 2.0 / 1e9
        if gb > self._SLICE_SLAB_MAX_GB:
            key = (level_idx, round(gb))
            if key != self._slab_refused_key:
                self._slab_refused_key = key
                show_info(f'{levels[level_idx]} slab ~{gb:.1f} GB — too big for a '
                          f'sharp slice (limit {self._SLICE_SLAB_MAX_GB:.0f} GB)')
            return
        need = (level_idx, tuple(lo.tolist()), tuple(hi.tolist()))
        if self._slice_slab_level == level_idx and self._slab_covers(self._slice_slab_box, need[1:]):
            return  # 已装载的板盖住需求
        if self._slice_slab_loading is not None and self._slab_covers(self._slice_slab_loading[1:], need[1:]):
            return  # 在读的板盖住需求
        self._slab_need = need
        self._slab_read_timer.start()  # 滑杆连续动时反复重置，停稳才真读


    def _start_slab_read(self):
        need = self._slab_need
        if need is None:
            return
        if self._slice_slab_loading is not None:
            # 已有一个读在进行：不并发开第二个（同一 HDF5/数据句柄并发读会
            # 撞锁卡死），500ms 后重试当前需求
            self._slab_read_timer.start()
            return
        self._slab_need = None
        level_idx, lo, hi = need
        lo, hi = np.asarray(lo), np.asarray(hi)
        if self._slice_slab_level == level_idx and self._slab_covers(self._slice_slab_box, (lo, hi)):
            return
        roi0 = np.asarray(self.image.rois[level_idx][0:3], dtype=np.int64)
        read_roi = [int(v) for v in (roi0 + lo)] + [int(v) for v in (hi - lo)]
        image, channel = self.image, str(self.channel.value)
        self._slice_slab_loading = (level_idx, lo, hi)
        show_info(f'loading {list(self.image.resolution_levels)[level_idx]} slab '
                  f'({float(np.prod(np.asarray(hi) - np.asarray(lo))) * 2.0 / 1e6:.0f} MB)…')

        def read():
            try:
                data = image.from_roi(read_roi, level=level_idx, channel=channel)
            except Exception as exc:  # noqa: BLE001
                print(f'slice slab read failed: {exc}', flush=True)
                data = None
            self._slice_slab_pending = (level_idx, lo, hi, data)

        threading.Thread(target=read, daemon=True).start()


    def _poll_slice_load(self):
        pending = self._slice_slab_pending
        if pending is None:
            return
        self._slice_slab_pending = None
        level_idx, lo, hi, data = pending
        if self._slice_slab_loading is not None and self._slice_slab_loading[:3] == (level_idx, lo, hi):
            self._slice_slab_loading = None
        if data is None or self.slice_volume_layer is None:
            return  # 读失败，或 slicer 已关（下次勾选由 update_slice_plane 建层）
        spacing_s = np.asarray(self.image.info[level_idx]['spacing'], dtype=float)
        roi0 = np.asarray(self.image.rois[level_idx][0:3], dtype=float)
        sv = self.slice_volume_layer
        if self._slice_slab_level == level_idx and self._slab_covers(self._slice_slab_box, (lo, hi)):
            return  # 已有更大的板盖住这块晚到的（比如刚补读了一块更全的）
        sv.data = data
        sv.scale = tuple(float(v) for v in spacing_s)
        # 显示位置 = 板的真实世界位置 + 场景 shift 拖动偏移（与切面/节点同帧）
        t_scene = np.asarray(self.panorama_points.translate, dtype=float)
        self._set_layer_translate(sv, (roi0 + np.asarray(lo, dtype=float)) * spacing_s + t_scene)
        try:
            sv.reset_contrast_limits()
        except Exception:
            pass
        self._slice_slab_level = level_idx
        self._slice_slab_box = (np.asarray(lo), np.asarray(hi))
        self.update_slice_plane()  # 换板后按新 translate 重算切面
        show_info(f'{list(self.image.resolution_levels)[level_idx]} slice loaded '
                  f'({float(np.prod(np.asarray(hi) - np.asarray(lo))) * 2.0 / 1e6:.0f} MB)')


    def _heal_hr_spacing(self):
        """Annotator 当前层的 spacing（节点 db 坐标的口径）。

        纯读：不触发下拉框自愈（那会从后台线程改 Qt 控件，必死锁）。
        """
        levels = list(self.image.resolution_levels)
        val = self.level.value
        idx = levels.index(val) if val in levels else 0
        return np.asarray(self.image.info[idx]['spacing'], dtype=float)

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


    def refresh_panorama(self, *args):
        # *args: magicgui PushButton.changed emits the button value (False);
        # a plain (self) signature crashed with "takes 1 positional argument
        # but 2 were given" when wired to that signal
        if self.mode_switch.mode == 'labeling':
            show_info('switch to panorama mode first')
            return
        if str(self.db_path.value) == '.':
            show_info('please select a database file')
            return
        if str(self.image_path.value) == '.':
            # e.g. reading the db first: loading the graph needs the image too,
            # retrying after the image is selected gives the complete state
            show_info('please select an image file')
            return
        resolution_level = self._heal_level_channel()
        if resolution_level is None:
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
            # per-axis physical spacing (µm/voxel) instead of size ratios,
            # so anisotropic datasets keep true proportions (resolves the TODO below)
            spacing = self.image.info[resolution_level]['spacing']
            pano_spacing = self.image.info[level]['spacing']

            image = self.image.from_roi(roi, level=level, channel=str(self.channel.value))
            origin = self.image.info[resolution_level]['origin']
            self.panorama_image.data = image
            self.panorama_image.scale = pano_spacing
            self._set_layer_translate(self.panorama_image, [i*j for i,j in zip(origin,spacing)])
            self.panorama_image.visible = True
            self.panorama_image.reset_contrast_limits()
            self.panorama_image.metadata['loaded'] = True
        
        # load full image if it's tiff format
        if '.tif' in str(self.image_path.value) and self.image_switch.value == True:
            image = self.image.from_roi(self.image.roi)
            self.panorama_image.data = image
            self.panorama_image.scale = self.image.info[0]['spacing']
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

        spacing = self.image.info[resolution_level]['spacing']
        camera_center  = [(i + j//2)*s for i,j,s in zip(self.image.rois[resolution_level][0:3],self.image.rois[resolution_level][3:],spacing)]

        self.panorama_points.visible = True
        self.panorama_points.data = np.array(coords)
        self.panorama_points.properties = properties
        self.panorama_points.face_colormap = 'hsl'
        self.panorama_points.face_color = 'colors'
        self.panorama_points.size = sizes
        self.panorama_points.selected_data = []

        # panorama point/view layers carry hr-level absolute voxel coordinates
        for attr in ('panorama_points','panorama_view'):
            coord_layer = getattr(self, attr, None)
            if coord_layer is not None:
                coord_layer.scale = spacing
                coord_layer.translate = [0.0,0.0,0.0]


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
        self.update_slice_plane()


    def node_selection(self, layer, event):
        # this is appended to panorama_points layer
        if event.button == 1:
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
            if near_point is None or far_point is None:
                # the click ray does not pass through the image volume
                # (e.g. clicking outside the block in 3D view)
                show_info("click is outside the image volume")
                return
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
            max_point = [i+int(j) for i,j in zip(max_point,getattr(self,'_image_origin',[0,0,0]))]

            # get new node id
            new_id = len(self.G)
            while self.G.has_node(new_id):
                new_id+=10
            
            self.G.add_node(new_id, nid = new_id, coord = max_point, type = 0, checked = 0, creator = self.user_name.value)
            self.rtree.insert(new_id, tuple(max_point+max_point))
            self.added['nodes'].append(new_id)

            self.refresh(self.viewer)
            self.viewer.layers.selection.active = self.image_layer
