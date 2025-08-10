import napari
import napari.layers
import napari.utils
import napari.utils.notifications
import numpy as np

from brightest_path_lib.algorithm import NBAStarSearch

from .simple_viewer import SimpleViewer
from .widget.rec_interactor import RecWidgets
from ..model import Deconver, default_dec_weight_path
from ..backend.task_manager import TaskManager
from ..backend.action import Action


class NeuronReconstructor(SimpleViewer):
    def __init__(self, napari_viewer:napari.Viewer):
        super().__init__(napari_viewer)
        self.viewer.__dict__['neurofly']['reconstructor'] = self
        self.nodes_layer = self.viewer.add_points(ndim=3, size=1, shading='spherical', name='nodes')
        self.edges_layer = self.viewer.add_vectors(ndim=3, vector_style='line', edge_color='orange', edge_width=0.3, name='edges')
        self.viewer.layers.selection.active = self.image_layer

        self.init_attributes()
        self.init_model()
        self.add_callback()
        self.add_shortcut()
    
    def init_attributes(self):
        """Initialize attributes for the NeuronReconstructor."""
        super().init_attributes()

        self.TaskManager = None
        self.task_node = None

        self.RecWidgets = RecWidgets()
        self.extend([self.RecWidgets])

    def init_model(self):
        self.Deconver = Deconver(default_dec_weight_path)
    
    def add_callback(self):
        """Add callbacks to the widgets and layers."""
        super().add_callback()
        self.RecWidgets.reset_database_path_widget_callback(self.on_db_loading)
        self.RecWidgets.reset_deconv_button_callback(self.deconv)
        self.RecWidgets.reset_revoke_button_callback(self.revoke)
        self.RecWidgets.reset_next_task_button_callback(self.next_task)
        self.RecWidgets.reset_last_task_button_callback(self.last_task)
        self.RecWidgets.reset_submit_button_callback(self.submit)
        self.RecWidgets.reset_proofreading_checkbox_callback(self.proofread)

        self.nodes_layer.click_get_value = self.nodes_layer.get_value
        self.nodes_layer.get_value = lambda position, view_direction=None, dims_displayed=None, world=False: None
        self.nodes_layer.mouse_drag_callbacks.append(self.viewer_node_operation)
        self.nodes_layer.mouse_double_click_callbacks.append(self.viewer_select_task_node)

        self.edges_layer.click_get_value = self.edges_layer.get_value
        self.edges_layer.get_value = lambda position, view_direction=None, dims_displayed=None, world=False: None
        self.edges_layer.mouse_drag_callbacks.append(self.viewer_edge_operation)
    
    def add_shortcut(self):
        """Add keyboard shortcuts for the viewer."""
        deconv_shortcut_fn = lambda _self, _evnet=None: self.deconv()
        self.viewer.bind_key('d', deconv_shortcut_fn, overwrite=True)
        check_shortcut_fn = lambda _self, _event=None: self.RecWidgets.on_check_button_clicked()
        self.viewer.bind_key('c', check_shortcut_fn, overwrite=True)
        revoke_shortcut_fn = lambda _self, _event=None: self.revoke()
        self.viewer.bind_key('r', revoke_shortcut_fn, overwrite=True)
        submit_shortcut_fn = lambda _self, _event=None: self.submit()
        self.viewer.bind_key('s', submit_shortcut_fn, overwrite=True)
    
    def on_db_loading(self):
        db_path = self.RecWidgets.get_database_path()
        self.TaskManager = TaskManager(db_path)
        self.task_node = self.TaskManager.task_node
        self.action_node = self.TaskManager.action_node
        self.refresh()

    def deconv(self):
        size = list(self.image_layer.data.shape)
        if (np.array(size)<=np.array([192,]*3)).all():
            sr_img = self.Deconver.process_one(self.image_layer.data)
            self.image_layer.data = sr_img
        else:
            napari.utils.notifications.show_info("this image is too large, try a smaller one")
    
    def get_task_info(self):
        task_nid = self.TaskManager.task_node['nid'] if self.TaskManager.task_node else None
        status = self.TaskManager.get_task_status()
        unchecked_length = status['unchecked_length']
        checked_length = status['checked_length']
        dynamic_length = status['dynamic_length']
        info = f"Task Node ID: {task_nid}\n"
        info += f"Dynamic Tasks: {dynamic_length}\n"
        info += f"Unchecked Tasks: {unchecked_length}\n"
        info += f"Checked Tasks: {checked_length}\n"
        return info

    def get_info(self):
        """Get the information to be displayed in the ImageFinder widget."""
        image_info = self.get_image_info()
        task_info = self.get_task_info()
        info = f"{image_info}\n--------\n{task_info}"
        return info
    
    def refresh(self):
        # update graph
        self.render(init_graph=True)
        # then update image
        if self.TaskManager:
            self.TaskManager.action_stack_clear()
            super().refresh(self.get_info())
        else:
            super().refresh()
        self.update_contrast()
    
    def render(self, *, init_graph:bool):
        # clear layers
        if self.resolution_level != 0 or self.TaskManager is None:
            self.nodes_layer.data = np.zeros((0, 3))
            self.edges_layer.data = np.empty((0, 2, 3))
            return
        
        # logic: update graph in TaskManager
        if init_graph:
            center, size = self.ROISelector.get_roi()
            roi = [center[i]-size[i]//2 for i in range(3)] + size
            self.TaskManager.init_graph(roi)
        
        # visual: update layers
        self.task_node = self.TaskManager.task_node if self.TaskManager.task_node else None
        if self.RecWidgets.get_proofreading_mode():
            self.TaskManager.init_graph_prof()
            nodes_coords, nodes_properties, edges_coords, edges_properties = self.TaskManager.G_prof.get_render_data(self.task_node)
        else:
            nodes_coords, nodes_properties, edges_coords, edges_properties = self.TaskManager.G.get_render_data(self.task_node)
        self.nodes_layer.data = nodes_coords
        self.nodes_layer.properties = {
            'nids': nodes_properties['nids'], 
        }
        self.nodes_layer.size = nodes_properties['sizes']
        if len(nodes_properties['colors']) > 0:
            self.nodes_layer.face_color = nodes_properties['colors']

        self.edges_layer.data = edges_coords
        self.edges_layer.properties = edges_properties

        # control: update widgets
        self.RecWidgets.set_node_type_idx(self.task_node['type'] if self.task_node else 0)
    
    def update_contrast(self):
        nodes_coords = self.nodes_layer.data
        if len(nodes_coords) > 0:
            # nodes layer is not empty, calculate mean and std intensity
            nodes_coords = nodes_coords - self.image_layer.translate
            valid_idx = (np.all((nodes_coords >= 0) & (nodes_coords < self.image_layer.data.shape), axis=1))
            nodes_coords = nodes_coords[valid_idx].astype(np.int16)
            intensities:np.ndarray = self.image_layer.data[tuple(nodes_coords.T)]
            mean_intensity = intensities.mean()
            std_intensity = intensities.std()
            self.image_layer.reset_contrast_limits()
            self.image_layer.contrast_limits = [min(mean_intensity//2, 150), mean_intensity+std_intensity]
        else:
            super().update_contrast()

    def proofread(self):
        if self.TaskManager is None or self.task_node is None:
            napari.utils.notifications.show_info("Please load a database and select a task node first.")
            self.RecWidgets.set_proofreading_mode(False)
            return
        self.render(init_graph=False)
    
    def next_task(self):
        task_node = self.TaskManager.get_next_task()
        if task_node is None:
            status = self.TaskManager.get_task_status()
            if status['unchecked_length']>0:
                napari.utils.notifications.show_info("Dynamic task stack is empty, click again to get the next task from global unchecked tasks.")
                self.TaskManager.reset_task_status()
                self.render(init_graph=True)
            else:
                napari.utils.notifications.show_info("No more tasks available.")
            return
        self.ROISelector.set_center(task_node['coord'])
        self.refresh()
    
    def last_task(self):
        task_node = self.TaskManager.get_last_task()
        if task_node is None:
            napari.utils.notifications.show_info("No last task available.")
            return
        self.ROISelector.set_center(task_node['coord'])
        self.refresh()
        
    def trace_astar(self, src_node, dst_node, interval=3, *, new_dst_node:bool=True):
        # sample function to sample the path at a given interval
        def __sample_path__(path, interval, with_end):
            sample_idxs = list(range(interval, len(path)-1, interval))
            if sample_idxs and sample_idxs[-1] == len(path)-2:
                sample_idxs.pop()
            if with_end:
                sample_idxs.append(len(path)-1)
            return [path[i] for i in sample_idxs]
        
        nodes = {}
        edges = {}
        src_nid, src_coord = src_node['nid'], np.asarray(src_node['coord'])
        dst_nid, dst_coord = dst_node['nid'], np.asarray(dst_node['coord'])
        center = (src_coord + dst_coord) // 2
        distance = np.linalg.norm(src_coord - dst_coord)
        # if distance is small, just connect the nodes directly
        if distance <= 5:
            edges[(src_nid, dst_nid)] = {}
        # if distance is large, use A* algorithm to find the path
        else:
            size = round(distance) + 8
            offset = center - size // 2
            roi = [int(offset[i]) for i in range(3)] + [size, size, size]
            img = self.IMAGE.from_roi(roi)
            tracer = NBAStarSearch(img, start_point=src_coord-offset, goal_point=dst_coord-offset)
            path_coord = tracer.search()
            sampled_path = __sample_path__(path_coord, interval, with_end=new_dst_node)
            if len(sampled_path) == 0:
                edges[(src_nid, dst_nid)] = {}
            else:
                neg_temp_nids = []
                neg_temp_max_nid = self.TaskManager.G.get_neg_temp_max_nid()
                for i, coord in enumerate(sampled_path):
                    neg_temp_max_nid += 1
                    _neg_temp_nid = -(neg_temp_max_nid)
                    neg_temp_nids.append(_neg_temp_nid)
                    nodes[neg_temp_nids[-1]] = {
                        'coord': (coord+offset).tolist(),
                        'checked': -1 if new_dst_node and i==len(sampled_path)-1 else 0,
                    }
                self.TaskManager.G.set_neg_temp_max_nid(neg_temp_max_nid)
                if new_dst_node:
                    neg_temp_nids = [src_nid] + neg_temp_nids
                else:
                    neg_temp_nids = [src_nid] + neg_temp_nids + [dst_nid]
                for _src, _dst in zip(neg_temp_nids[:-1], neg_temp_nids[1:]):
                    edges[(_src, _dst)] = {}
        return nodes, edges
    
    def action_select_task_node(self, nid:int):
        self.TaskManager.reset_task_status()
        self.task_node = self.TaskManager.set_task_node(nid)
        self.ROISelector.set_center(self.task_node['coord'])
        return True

    def action_add_path(self, task_node:dict, action_node:dict, new_action_node:bool, use_astar:bool):
        if use_astar:
            path_nodes, path_edges = self.trace_astar(
                src_node={'nid': task_node['nid'], 'coord': task_node['coord']},
                dst_node={'nid': action_node['nid'], 'coord': action_node['coord']},
                new_dst_node=new_action_node
            )
        else:
            path_nodes, path_edges = {}, {(task_node['nid'], action_node['nid']): {}}
        action = Action(self.RecWidgets.get_username(), task_node, 'add_path', 
                        action_node=action_node, action_edge=None,
                        path_nodes=path_nodes, path_edges=path_edges)
        is_successed = self.TaskManager.action_stack_push(action)
        return is_successed
    
    def action_delete_node(self, task_node:dict, action_node:dict):
        action = Action(self.RecWidgets.get_username(), task_node, 'delete_node', 
                        action_node=action_node, action_edge=None)
        is_successed = self.TaskManager.action_stack_push(action)
        return is_successed

    def action_delete_edge(self, task_node:dict, action_node:dict, action_edge:dict):
        action = Action(self.RecWidgets.get_username(), task_node, 'delete_edge', 
                        action_node=action_node, action_edge=action_edge)
        is_successed = self.TaskManager.action_stack_push(action)
        return is_successed

    def global_viewer_status_check(self, *, check_task_node:bool, check_prof_mode:bool):
        passed = True
        username = self.RecWidgets.get_username()
        info = ''
        if username is None or username == '':
            info += "Please set a username for the database.\n"
            passed = False
        if self.TaskManager is None:
            info += "Please load a database first.\n"
            passed = False
        if check_task_node and (self.task_node is None or self.task_node['nid'] not in self.TaskManager.G.nodes):
            info += "Please select a task node first.\n"
            self.TaskManager.reset_task_status()
            passed = False
        if check_prof_mode and self.RecWidgets.get_proofreading_mode():
            info = "Proofreading mode is enabled, please disable it to perform actions."
        if not passed:
            napari.utils.notifications.show_info(info)
        return passed
    
    def viewer_select_task_node(self, layer:napari.layers.Points, event):
        global_check = self.global_viewer_status_check(check_task_node=False, check_prof_mode=False)
        if not global_check:
            return
        index = layer.click_get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        if index is not None:
            if event.button == 1:
                nid = int(layer.properties['nids'][index])
                is_successed = self.action_select_task_node(nid)
            if is_successed:
                print(f'[Action] select task node: {nid}')
                self.refresh()
                print(f'refresh')

    def viewer_node_operation(self, layer:napari.layers.Points, event):
        global_check = self.global_viewer_status_check(check_task_node=True, check_prof_mode=True)
        if not global_check:
            return
        index = layer.click_get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        # if clicked on a existing node
        if index is not None:
            nid = int(layer.properties['nids'][index])
            # if performing an action, check if a task node is in current ROI
            is_successed = False
            self.task_node = self.TaskManager.task_node
            self.action_node = self.TaskManager.set_action_node(nid)
            # add path
            if 'Shift' in event.modifiers and event.button == 1:
                is_successed = self.action_add_path(self.task_node, self.action_node, new_action_node=False, use_astar=True)
                print(f'[Action] add path from task node ({self.task_node["nid"]}) to action node ({self.action_node["nid"]})')
            # add path without a_star
            if 'Control' in event.modifiers and event.button == 1:
                is_successed = self.action_add_path(self.task_node, self.action_node, new_action_node=False, use_astar=False)
                print(f'[Action] add path from task node ({self.task_node["nid"]}) to action node ({self.action_node["nid"]}) directly')
            # delete node
            elif event.button == 2:
                is_successed = self.action_delete_node(self.task_node, self.action_node)
                print(f'[Action] delete node: {self.action_node["nid"]}')

            # refresh
            if is_successed:
                self.render(init_graph=False)
                print(f'render')

        # if clicked on empty space
        elif index is None:
            is_successed = False
            # if user wants to create a new path to a new node
            if 'Shift' in event.modifiers and event.button == 2:
                coord = self.get_click_position(self.image_layer, event)
                self.task_node = self.TaskManager.task_node
                action_node = {
                    'nid': 0,
                    'coord': coord,
                }
                is_successed = self.action_add_path(self.task_node, action_node, new_action_node=True, use_astar=True)
                print(f'[Action] add path from task node ({self.task_node["nid"]}) to new action node ({action_node["nid"]})')
            if is_successed:
                self.render(init_graph=False)
                print(f'render')
    
    def viewer_edge_operation(self, layer: napari.layers.Vectors, event, threshold=2.0):
        if not self.global_viewer_status_check(check_task_node=True, check_prof_mode=True):
            return
        if event.button == 2:
            # delete edge
            edges_data = self.edges_layer.data
            src = edges_data[:, 0, :]
            src2dst = edges_data[:, 1, :]
            click = self.get_click_position(self.image_layer, event)
            
            src2click = click - src
            dstVec_dot_dstVec = np.sum(src2dst * src2dst, axis=1)
            clickVec_dot_dstVec = np.sum(src2click * src2dst, axis=1)
            dstVec_dot_dstVec[dstVec_dot_dstVec == 0] = 1e-12

            clickVec_proj_dstVec = np.clip(clickVec_dot_dstVec/dstVec_dot_dstVec, 0, 1)
            proj_point = src + src2dst * clickVec_proj_dstVec[:, None]
            distance = np.linalg.norm(proj_point - click, axis=1)

            min_idx = np.argmin(distance)
            min_distance = distance[min_idx]
            if min_distance < threshold:
                src_nid = int(layer.properties['src'][min_idx])
                dst_nid = int(layer.properties['dst'][min_idx])
                if src_nid > dst_nid:
                    src_nid, dst_nid = dst_nid, src_nid
                self.task_node = self.TaskManager.task_node
                if self.task_node['nid'] not in (src_nid, dst_nid):
                    return
                else:
                    self.task_node = self.TaskManager.task_node
                    self.action_node = self.TaskManager.set_action_node(src_nid if self.task_node['nid'] == dst_nid else dst_nid)
                    self.action_edge = self.TaskManager.set_action_edge(src_nid, dst_nid)
                    print(f'edge selected: {src_nid} -> {dst_nid}, distance to click: {min_distance}')
                    is_successed = self.action_delete_edge(self.task_node, self.action_node, self.action_edge)
                    if is_successed:
                        self.render(init_graph=False)
    
    def revoke(self):
        self.TaskManager.action_stack_pop()
        self.render(init_graph=False)

    def submit(self):
        if not self.global_viewer_status_check(check_task_node=True, check_prof_mode=True):
            self.RecWidgets.set_check_button_status('Check')
            return
        self.task_node = self.TaskManager.task_node
        action_node = self.task_node
        action_node.update({
            'creator': self.RecWidgets.get_username(),
            'type': self.RecWidgets.get_node_type_idx(),
        })
        action = Action(self.RecWidgets.get_username(), self.task_node, 'update_node',
                        action_node=action_node)
        self.TaskManager.action_stack_push(action)
        self.TaskManager.submit()
        self.next_task()
        self.RecWidgets.set_check_button_status('Check')
        self.RecWidgets.set_node_type_idx(0)


def main():
    """Main function to run the NeuroReconstructor."""
    viewer = napari.Viewer()
    reconstructor = NeuronReconstructor(viewer)
    viewer.window.add_dock_widget(reconstructor, name='Neuron Reconstructor')
    napari.run()

if __name__ == "__main__":
    main()
