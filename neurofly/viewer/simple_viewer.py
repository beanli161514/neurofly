import os
import numpy as np
import tifffile as tiff
from datetime import datetime

from magicgui import widgets
import napari

from .widget.roi_selector import ROISelector
from .widget.res_ch_selector import ResChSelector
from .widget.image_finder import ImageFinder
from ..neurodb.image_reader import ImageReader


class SimpleViewer(widgets.Container):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        # viewer
        self.viewer = napari_viewer
        self.viewer.__dict__['neurofly'] = {}
        self.viewer.__dict__['neurofly']['simple_viewer'] = self
        self.viewer.dims.ndisplay = 3
        self.viewer.layers.clear()
        self.viewer.window.remove_dock_widget('all')
        self.image_layer = self.viewer.add_image(np.zeros((64, 64, 64), dtype=np.uint16), name='image')
        self.goal_layer = self.viewer.add_points(ndim=3, symbol='cross', face_color='red', size=2, shading='spherical',name='goal')        
        self.viewer.layers.selection.active = self.image_layer

        if type(self) is SimpleViewer:
            self.init_attributes()
            self.add_callback()

    def init_attributes(self):
        """Initialize attributes for the SimpleViewer."""
        self.IMAGE = None
        self.resolution_level = 0
        self.channel = 0

         # widgets
        self.ImageFinder = ImageFinder()
        self.ResChSelector = ResChSelector()
        self.ROISelector = ROISelector()
        self.extend([
            self.ImageFinder,
            self.ResChSelector,
            self.ROISelector,
        ])

    def add_callback(self):
        """Add callbacks to the widgets and layers."""
        self.ImageFinder.reset_image_path_widget_callback(self.on_image_reading)
        self.ImageFinder.reset_save_button_callback(self.on_save_clicked)
        self.ResChSelector.reset_button_callbacks(self.level_up, self.level_down, self.refresh)
        self.image_layer.mouse_double_click_callbacks.append(self.on_double_click)

        # remove double_click_to_zoom function
        self.viewer.mouse_double_click_callbacks.clear()

    def on_image_reading(self):
        """Callback for when the image path is changed."""
        image_path = self.ImageFinder.get_image_path()
        self.IMAGE = ImageReader(image_path)

        OFFSET, SIZE = self.IMAGE.rois[0][:3], self.IMAGE.rois[0][3:]
        CENTER = [OFFSET[i] + SIZE[i] // 2 for i in range(3)]
        self.ROISelector.set_roi(CENTER, [64]*3)

        resolution_choices = self.IMAGE.resolution_levels
        channels_choices = self.IMAGE.channels

        self.ResChSelector.reset_combobox_choices(resolution_choices, channels_choices)
        self.ResChSelector.reset_combobox_callbacks(self.on_resolution_change, self.on_channel_change)
        self.refresh()
    
    def on_save_clicked(self):
        save_dir = self.ImageFinder.get_save_path()
        save_name = datetime.now().strftime("%y%m%d%H%M%S")
        save_path = os.path.join(save_dir, f"{save_name}.tif")
        img = np.asarray(self.image_layer.data)
        tiff.imwrite(save_path, img.astype(np.uint16))
        print(f'Save current ROI image to ({save_path})')
        napari.utils.notifications.show_info(f'Save current ROI image to ({save_path})')
    
    def on_resolution_change(self, event=None):
        """Callback for when the resolution is changed."""
        resolution = self.ResChSelector.get_resolution()
        if resolution is not None and self.IMAGE is not None:
            center, size = self.ROISelector.get_roi()

            level_current = self.resolution_level
            level_target = self.IMAGE.resolution_levels.index(resolution)
            self.resolution_level = level_target

            spacing_current = self.IMAGE.info[level_current]['spacing']
            spacing_target = self.IMAGE.info[level_target]['spacing']
            scale = [i/j for i,j in zip(spacing_current, spacing_target)]

            center_target = [int(center[i] * scale[i]) for i in range(3)]
            self.ROISelector.set_center(center_target)
            self.refresh()

    def on_channel_change(self, event=None):
        """Callback for when the channel is changed."""
        channel = self.ResChSelector.get_channel()
        if channel is not None and self.IMAGE is not None:
            self.channel = self.IMAGE.channels.index(channel)
            self.refresh()

    def level_up(self):
        """Callback for when the resolution level is increased."""
        center, size = self.ROISelector.get_roi()
        level_current = self.IMAGE.resolution_levels.index(self.ResChSelector.get_resolution())

        if level_current == len(self.IMAGE.rois)-1:
            return
        
        level_target = level_current + 1
        self.resolution_level = level_target

        spaceing_current = self.IMAGE.info[level_current]['spacing']
        spacing_target = self.IMAGE.info[level_target]['spacing']
        scale = [i/j for i,j in zip(spaceing_current,spacing_target)]

        self.ResChSelector.set_resolution(self.IMAGE.resolution_levels[level_target])

        center_target = [int(center[i] * scale[i]) for i in range(3)]
        self.ROISelector.set_center(center_target)
        self.refresh()
    
    def level_down(self):
        """Callback for when the resolution level is decreased."""
        center, size = self.ROISelector.get_roi()
        level_current = self.IMAGE.resolution_levels.index(self.ResChSelector.get_resolution())

        if level_current == 0:
            return
        
        level_target = level_current - 1
        self.resolution_level = level_target

        spaceing_current = self.IMAGE.info[level_current]['spacing']
        spacing_target = self.IMAGE.info[level_target]['spacing']
        scale = [i/j for i,j in zip(spaceing_current,spacing_target)]

        self.ResChSelector.set_resolution(self.IMAGE.resolution_levels[level_target])

        center_target = [int(center[i] * scale[i]) for i in range(3)]
        self.ROISelector.set_center(center_target)
        self.refresh()
    
    def get_click_position(self, layer:napari.layers.Layer=None, event=None):
        if layer is None:
            layer = self.image_layer
        #based on ray casting
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
        max_point = [int(i+j) for i,j in zip(max_point, self.image_layer.translate)]
        return max_point

    def on_double_click(self, layer:napari.layers.Layer, event):
        """Callback for double-click events on the image layer."""
        if event.button==1:
            max_point = self.get_click_position(layer, event)
            print(f'move ROI center to {max_point}')
            self.ROISelector.set_center(max_point)
            self.refresh()
    
    def get_image_info(self):
        info = "\n".join(f"{key}: {value}" for key, value in self.IMAGE.info[self.resolution_level].items())
        return info

    def update_info(self, info:str):
        """Update the information displayed in the ImageFinder widget."""
        self.ImageFinder.update_info(info)

    def refresh(self, info:str=None):
        """Refresh the image layer with the current ROI and resolution."""
        if self.IMAGE is None:
            return
        # Get the ROI, resolution, and channel from the selectors
        center, size = self.ROISelector.get_roi()
        roi = [int(center[i] - size[i] // 2) for i in range(3)] + [size[i] for i in range(3)]

        # update the image layer with the new ROI, resolution, and channel
        self.goal_layer.data = center
        self.image_layer.data = self.IMAGE.from_roi(roi, self.resolution_level, self.channel)
        self.image_layer.translate = roi[:3]

        camera_angles = self.viewer.camera.angles
        camera_zoom = self.viewer.camera.zoom
        self.viewer.reset_view()
        self.viewer.camera.angles = camera_angles
        self.viewer.camera.zoom = camera_zoom
        self.viewer.camera.center = np.array(center)
        if type(self) is SimpleViewer:
            self.update_contrast()
        
        if info is None:
            info = self.get_image_info()
        self.update_info(info)
    
    def update_contrast(self):
        img:np.ndarray = self.image_layer.data
        max_intensity = img.max()
        min_intensity = img.min()
        mean_intensity = img.mean()
        std_intensity = img.std()
        self.image_layer.reset_contrast_limits()
        contrast_min = min(mean_intensity//2, min_intensity)
        scale = np.log2(max_intensity//mean_intensity)
        contrast_max = int((mean_intensity+2*std_intensity) * scale)
        self.image_layer.contrast_limits = [contrast_min, contrast_max]
        

def main():
    viewer = napari.Viewer()
    simple_viewer_container = SimpleViewer(viewer)
    viewer.window.add_dock_widget(simple_viewer_container, name="Simple Viewer")
    napari.run()

if __name__ == "__main__":
    main()