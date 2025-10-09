import os
from datetime import datetime
import tifffile as tiff
import numpy as np
import napari
from brightest_path_lib.algorithm import NBAStarSearch

from .viewer import NeuronViewer
from .widget.skel_widgets import SkelWidget


class NeuronSkeletonizer(NeuronViewer):
    def __init__(self, napari_viewer:napari.Viewer):
        super().__init__(napari_viewer)
        self.viewer.__dict__['neurofly']['skeletonizer'] = self
        self.skeleton_layer = self.viewer.add_image(np.zeros((64, 64, 64), dtype=bool), colormap='magma', name='skeleton')
        self.src_node_layer = self.viewer.add_points(ndim=3, size=2, face_color='red', shading='spherical', name='start_node')
        self.dst_node_layer = self.viewer.add_points(ndim=3, size=2, face_color='blue', shading='spherical', name='end_node')
        self.viewer.layers.selection.active = self.image_layer

        self.init_attributes()
        self.add_callback()
        self.add_shortcut()
    
    def init_attributes(self):
        super().init_attributes()
        self.reset_state()

        self.SkelWidget = SkelWidget()
        self.extend([self.SkelWidget])

    def add_callback(self):
        super().add_callback()
        self.SkelWidget.reset_freeze_button_callback(self.on_freeze_button_clicked)
        self.SkelWidget.reset_trace_button_callback(self.on_trace_button_clicked)
        self.SkelWidget.reset_delete_button_callback(self.on_delete_button_clicked)

    def add_shortcut(self):
        freeze_shortcut_fn = lambda _self, _event=None: self.on_freeze_button_clicked()
        self.viewer.bind_key('f', freeze_shortcut_fn)

    def refresh(self):
        super().refresh()
        super().update_contrast()
        self.reset_state()
    
    def reset_state(self):
        self.Skeletons = []
        self.skeleton_layer.data = np.zeros((64, 64, 64), dtype=bool)
        self.src_node_layer.data = np.zeros((0, 3))
        self.dst_node_layer.data = np.zeros((0, 3))

    def render(self):
        # clear layers
        if self.resolution_level != 0:
            self.reset_state()

        skeleton_mask = np.zeros(self.image_layer.data.shape, dtype=bool)
        for skel in self.Skeletons:
            x, y, z = skel[:, 0], skel[:, 1], skel[:, 2]
            skeleton_mask[x, y, z] = True
        center, size = self.ROISelector.get_roi()
        offset = [_c - _s//2 for _c, _s in zip(center, size)]
        self.skeleton_layer.data = skeleton_mask
        self.skeleton_layer.translate = offset

    def freeze_view(self):
        self.SkelWidget.freeze_button.text = 'Unfreeze View (F)'
        self.SkelWidget.trace_button.enabled = True
        self.SkelWidget.delete_button.enabled = True
        self.ResChSelector.enabled = False
        self.ROISelector.enabled = False

        self.image_layer.mouse_double_click_callbacks.clear()
        self.image_layer.mouse_drag_callbacks.append(self.set_src_node)
        self.image_layer.mouse_drag_callbacks.append(self.set_dst_node)
        self.goal_layer.visible = False

        trace_shortcut_fn = lambda _self, _event=None: self.on_trace_button_clicked()
        self.image_layer.bind_key('t', trace_shortcut_fn)
        delete_shortcut_fn = lambda _self, _event=None: self.on_delete_button_clicked()
        self.image_layer.bind_key('d', delete_shortcut_fn)
    
    def unfreeze_view(self):
        self.SkelWidget.freeze_button.text = 'Freeze View (F)'
        self.SkelWidget.trace_button.enabled = False
        self.SkelWidget.delete_button.enabled = False
        self.ResChSelector.enabled = True
        self.ROISelector.enabled = True

        self.image_layer.mouse_drag_callbacks.clear()
        self.image_layer.mouse_double_click_callbacks.append(self.on_double_click)
        self.goal_layer.visible = True

        self.image_layer.keymap.clear()
    
    def on_freeze_button_clicked(self):
        if self.SkelWidget.freeze_button.text.startswith('Freeze'):
            self.freeze_view()
        else:
            self.unfreeze_view()

    def trace_astar(self, src_coord, dst_coord):
        src_coord = np.asarray(src_coord, dtype=int)
        dst_coord = np.asarray(dst_coord, dtype=int)
        img = self.image_layer.data
        tracer = NBAStarSearch(img, start_point=src_coord, goal_point=dst_coord)
        path_coord = tracer.search()
        return path_coord
    
    def on_trace_button_clicked(self):
        if len(self.src_node_layer.data) == 0 or len(self.dst_node_layer.data) == 0:
            napari.utils.notifications.show_info('Please set both start and end points.')
            return
        offset = self.image_layer.translate
        src_coord = self.src_node_layer.data[0] - offset
        dst_coord = self.dst_node_layer.data[0] - offset
        path_coord = self.trace_astar(src_coord, dst_coord)
        if path_coord is None:
            napari.utils.notifications.show_info('No path found.')
            return
        self.Skeletons.append(np.asarray(path_coord))
        self.render()

    def on_delete_button_clicked(self):
        if len(self.Skeletons) == 0:
            napari.utils.notifications.show_info('No skeleton to delete.')
            return
        self.Skeletons.pop()
        self.render()
    
    def set_src_node(self, layer:napari.layers.Points, event):
        if 'Control' in event.modifiers and event.button == 1:
            max_point = self.get_click_position(layer, event)
            self.src_node_layer.data = np.array([max_point])
    
    def set_dst_node(self, layer:napari.layers.Points, event):
        if 'Control' in event.modifiers and event.button == 2:
            max_point = self.get_click_position(layer, event)
            self.dst_node_layer.data = np.array([max_point])
    
    def on_save_clicked(self):
        save_dir = self.ImageFinder.get_save_path()
        time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
        
        image_save_path = os.path.join(save_dir, f"img_{time_stamp}.tif")
        img = np.asarray(self.image_layer.data)
        tiff.imwrite(image_save_path, img.astype(np.uint16))

        center, size = self.ROISelector.get_roi()
        meta_data = {
            'center': center,
            'size': size,
            'resolution_level': self.resolution_level,
            'channel': self.channel,
            'offset': self.image_layer.translate,
        }
        np.savez(os.path.join(save_dir, f'mask_{time_stamp}.npz'), meta=meta_data, skeletons=np.array(self.Skeletons, dtype=object))
    

def main():
    viewer = napari.Viewer()
    skeletonizer = NeuronSkeletonizer(viewer)
    viewer.window.add_dock_widget(skeletonizer, name='Neuron Skeletonizer')
    napari.run()

if __name__ == "__main__":
    main()
