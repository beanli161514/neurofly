import napari.layers
import numpy as np

from magicgui import widgets
import napari

if __name__ == "__main__":
    import sys,os
    sys.path.insert(0, os.getcwd())

    from neurofly.viewer.widget.roi_selector import ROI3DSelector
    from neurofly.viewer.widget.resolution_channel_selector import ResolutionChannelSelector
    from neurofly.viewer.widget.image_finder import ImageFinder
    from neurofly.neurodb.image_reader import ImageReader
else:
    from .widget.roi_selector import ROI3DSelector
    from .widget.resolution_channel_selector import ResolutionChannelSelector
    from .widget.image_finder import ImageFinder
    from ..neurodb.image_reader import ImageReader

class SimpleViewer(widgets.Container):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        # viewer
        self.viewer = napari_viewer
        self.viewer.dims.ndisplay = 3
        self.viewer.layers.clear()
        self.viewer.window.remove_dock_widget('all')
        self.image_layer = self.viewer.add_image(np.zeros((64, 64, 64), dtype=np.uint16), name='image')
        self.goal_layer = self.viewer.add_points(ndim=3, face_color='red', size=1, shading='spherical',name='goal')

        # image reader
        self.IMAGE = None
        self.resolution_level = 0
        self.channel = 0

        self.add_callback()

    def add_callback(self):
        """Add callbacks and widgets to the viewer."""
        # widgets
        self.image_finder = ImageFinder()
        self.resolution_channel_selector = ResolutionChannelSelector()
        self.roi_selector = ROI3DSelector()
        self.extend([
            self.image_finder,
            self.resolution_channel_selector,
            self.roi_selector,
        ])

        # callbacks
        self.image_finder.reset_image_path_widget_callbacks(self.on_image_reading)
        self.resolution_channel_selector.reset_button_callbacks(self.level_up, self.level_down, self.refresh)
        self.image_layer.mouse_double_click_callbacks.append(self.on_double_click)

    def on_image_reading(self):
        """Callback for when the image path is changed."""
        image_path = self.image_finder.get_image_path()
        self.IMAGE = ImageReader(image_path)

        OFFSET, SIZE = self.IMAGE.rois[0][:3], self.IMAGE.rois[0][3:]
        CENTER = [OFFSET[i] + SIZE[i] // 2 for i in range(3)]
        self.roi_selector.set_roi(CENTER, (64,64,64))

        resolution_choices = self.IMAGE.resolution_levels
        channels_choices = self.IMAGE.channels

        self.resolution_channel_selector.reset_combobox_choices(resolution_choices, channels_choices)
        self.resolution_channel_selector.reset_combobox_callbacks(self.on_resolution_change, self.on_channel_change)
        self.refresh()
    
    def on_resolution_change(self, event=None):
        """Callback for when the resolution is changed."""
        resolution = self.resolution_channel_selector.get_resolution()
        if resolution is not None and self.IMAGE is not None:
            center, size = self.roi_selector.get_roi()

            level_current = self.resolution_level
            level_target = self.IMAGE.resolution_levels.index(resolution)
            self.resolution_level = level_target

            spacing_current = self.IMAGE.info[level_current]['spacing']
            spacing_target = self.IMAGE.info[level_target]['spacing']
            scale = [i/j for i,j in zip(spacing_current, spacing_target)]

            center_target = [int(center[i] * scale[i]) for i in range(3)]
            self.roi_selector.set_center(center_target)
            self.refresh()

    def on_channel_change(self, event=None):
        """Callback for when the channel is changed."""
        channel = self.resolution_channel_selector.get_channel()
        if channel is not None and self.IMAGE is not None:
            self.channel = self.IMAGE.channels.index(channel)
            self.refresh()

    def level_up(self):
        """Callback for when the resolution level is increased."""
        center, size = self.roi_selector.get_roi()
        level_current = self.IMAGE.resolution_levels.index(self.resolution_channel_selector.get_resolution())

        if level_current == len(self.IMAGE.rois)-1:
            return
        
        level_target = level_current + 1
        self.resolution_level = level_target

        spaceing_current = self.IMAGE.info[level_current]['spacing']
        spacing_target = self.IMAGE.info[level_target]['spacing']
        scale = [i/j for i,j in zip(spaceing_current,spacing_target)]

        self.resolution_channel_selector.set_resolution(self.IMAGE.resolution_levels[level_target])

        center_target = [int(center[i] * scale[i]) for i in range(3)]
        self.roi_selector.set_center(center_target)
        self.refresh()
    
    def level_down(self):
        """Callback for when the resolution level is decreased."""
        center, size = self.roi_selector.get_roi()
        level_current = self.IMAGE.resolution_levels.index(self.resolution_channel_selector.get_resolution())

        if level_current == 0:
            return
        
        level_target = level_current - 1
        self.resolution_level = level_target

        spaceing_current = self.IMAGE.info[level_current]['spacing']
        spacing_target = self.IMAGE.info[level_target]['spacing']
        scale = [i/j for i,j in zip(spaceing_current,spacing_target)]

        self.resolution_channel_selector.set_resolution(self.IMAGE.resolution_levels[level_target])

        center_target = [int(center[i] * scale[i]) for i in range(3)]
        self.roi_selector.set_center(center_target)
        self.refresh()

    def on_double_click(self, layer:napari.layers.Layer, event):
        """Callback for double-click events on the image layer."""
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
        print('Put point at: ', max_point)
        if(event.button==1):
            self.roi_selector.set_center(max_point)
            self.refresh()
    
    def refresh(self):
        """Refresh the image layer with the current ROI and resolution."""
        if self.IMAGE is None:
            return
        # Get the ROI, resolution, and channel from the selectors
        center, size = self.roi_selector.get_roi()
        roi = [int(center[i] - size[i] // 2) for i in range(3)] + [size[i] for i in range(3)]

        # update the image layer with the new ROI, resolution, and channel
        self.goal_layer.data = center
        self.image_layer.data = self.IMAGE.from_roi(roi, self.resolution_level, self.channel)
        self.image_layer.translate = roi[:3]

        camera_state = self.viewer.camera.angles
        self.viewer.reset_view()
        self.viewer.camera.angles = camera_state
        self.viewer.layers.selection.active = self.image_layer
        self.image_layer.reset_contrast_limits()
        # self.image_layer.contrast_limits = [
        #     np.median(self.image_layer.data), 
        #     self.image_layer.data.mean()+self.image_layer.data.std()
        # ]
        
        info = "\n".join(f"{key}: {value}" for key, value in self.IMAGE.info[self.resolution_level].items())
        self.image_finder.update_info(info)
        

def main():
    viewer = napari.Viewer()
    simple_viewer_container = SimpleViewer(viewer)
    viewer.window.add_dock_widget(simple_viewer_container, name="Simple Viewer")
    napari.run()

if __name__ == "__main__":
    main()