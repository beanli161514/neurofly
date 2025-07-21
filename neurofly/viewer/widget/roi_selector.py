from magicgui import widgets
from typing import List, Tuple
from functools import partial

class ROISelector(widgets.Container):
    def __init__(
        self,
        max_size: int = 512,
        default_center: Tuple[int, int, int] = (64, 64, 64),
        default_size: Tuple[int, int, int] = (64, 64, 64),
        power_step: int = 32,
        **kwargs
    ):
        self.max_size = max_size
        self.power_step = power_step

        self.x_center = widgets.LineEdit(value=default_center[0], label="X Center")
        self.y_center = widgets.LineEdit(value=default_center[1], label="Y Center") 
        self.z_center = widgets.LineEdit(value=default_center[2], label="Z Center")
        self.x_size = widgets.Slider(value=default_size[0], min=1, max=max_size, step=1, label="Size", enabled=False)
        self.y_size = widgets.Slider(value=default_size[1], min=1, max=max_size, step=1, label="Size", enabled=False)
        self.z_size = widgets.Slider(value=default_size[2], min=1, max=max_size, step=1, label="Size", enabled=False)
        self.power_mode_checkbox = widgets.CheckBox(value=True, text="Clip Mode")
        self.sync_mode_checkbox = widgets.CheckBox(value=True, text="Sync Size")
        self.sync_size = widgets.Slider(value=default_size[0], min=1, max=max_size, step=1, enabled=True)
        
        self.control_row = widgets.Container(
            widgets=[self.power_mode_checkbox, self.sync_mode_checkbox, self.sync_size], 
            layout="horizontal",
            labels=False
        )
        for center_widget in [self.x_center, self.y_center, self.z_center]:
            center_widget.max_width = 50
        for size_widget in [self.x_size, self.y_size, self.z_size]:
            size_widget.max_width = 150
            size_widget.min_width = 80
        self.x_row = widgets.Container(widgets=[self.x_center, self.x_size], layout="horizontal")
        self.y_row = widgets.Container(widgets=[self.y_center, self.y_size], layout="horizontal")
        self.z_row = widgets.Container(widgets=[self.z_center, self.z_size], layout="horizontal")
        self.x_row.margins = (0, 0, 0, 0)
        self.y_row.margins = (0, 0, 0, 0) 
        self.z_row.margins = (0, 0, 0, 0)
        
        super().__init__(
            widgets=[self.control_row, self.x_row, self.y_row, self.z_row],
            layout="vertical",
            labels=False,
            **kwargs
        )
        self._add_callback()
    
    def _add_callback(self):
        """Set up callbacks for the widgets."""
        self._set_xyz_center_callback()
        if not self.sync_mode_checkbox.value:
            self._set_xyz_size_callback(True)
        self._set_control_callback()
    
    def _set_xyz_center_callback(self):
        """Set callbacks for the center widgets to validate and clip values."""
        for widget in [self.x_center, self.y_center, self.z_center]:
            widget.native.editingFinished.connect(partial(self._validate_value, widget))

    def _set_xyz_size_callback(self, enabled):
        """Set callbacks for the size widgets to validate and clip values."""
        if enabled:
            for widget in [self.x_size, self.y_size, self.z_size]:
                widget.enabled = True
                widget.changed.connect(partial(self._clip_size, widget))
        else:
            for widget in [self.x_size, self.y_size, self.z_size]:
                widget.enabled = False
                widget.changed.disconnect()
    
    def _set_control_callback(self):
        """Set callbacks for the control widgets."""
        self.power_mode_checkbox.changed.connect(self._on_power_mode_changed)
        self.sync_mode_checkbox.changed.connect(self._on_sync_mode_changed)
        self.sync_size.changed.connect(partial(self._on_sync_size_changed, self.sync_size))

    def _validate_value(self, widget):
        """Validate the value of a LineEdit widget to ensure it is a non-negative integer."""
        value = widget.value
        if value == "" or value is None:
            widget.value = 0
            return
        try:
            int_value = int(value)
            if int_value < 0:
                widget.value = 0
            else:
                widget.value = int_value
        except ValueError:
            widget.value = 0
    
    def _clip_size(self, widget, new_value=None):
        """Clip the size to the nearest power of two if in power of two mode."""
        if self.power_mode_checkbox.value:
            current_value = widget.value
            clipped_value = (current_value // self.power_step) * self.power_step
            if clipped_value < self.power_step:
                clipped_value = self.power_step
            clipped_value = max(widget.min, min(widget.max, clipped_value))
            
            if clipped_value != current_value:
                with widget.changed.blocked():
                    widget.value = clipped_value
            return clipped_value

    def _on_power_mode_changed(self, enabled):
        """Enable or disable power of two mode."""
        if enabled:
            if self.sync_mode_checkbox.value:
                self._clip_size(self.sync_size)
            else:
                self._clip_size(self.x_size)
                self._clip_size(self.y_size)
                self._clip_size(self.z_size)
    
    def _on_sync_mode_changed(self, enabled):
        """Enable or disable synchronization of size widgets."""
        if enabled:
            self.sync_size.enabled = True
            min_size = min(self.x_size.value, self.y_size.value, self.z_size.value)
            self.sync_size.value = min_size
            self._set_xyz_size_callback(False)
        else:
            self.sync_size.enabled = False
            self._set_xyz_size_callback(True)
    
    def _on_sync_size_changed(self, widget, new_value=None):
        """Synchronize the size of the ROI when sync size mode is enabled."""
        clipped_value = self._clip_size(widget, new_value)
        for size_widget in [self.x_size, self.y_size, self.z_size]:
            size_widget.value = clipped_value

    def get_roi(self) -> Tuple[List[int], List[int]]:
        # """Get the current ROI as (offset, size) tuple."""
        """Get the current ROI as (center, size) tuple."""
        center = [int(self.x_center.value), int(self.y_center.value), int(self.z_center.value)]
        size = [int(self.x_size.value), int(self.y_size.value), int(self.z_size.value)]
        # offset = [center[0]-size[0]//2, center[1]-size[1]//2, center[2]-size[2]//2]
        return center, size
    
    def set_center(self, center: List[int]):
        """Set the center of the ROI."""
        self.x_center.value = center[0]
        self.y_center.value = center[1]
        self.z_center.value = center[2]
    
    def set_size(self, size: List[int]):
        """Set the size of the ROI."""
        self.x_size.value = size[0]
        self.y_size.value = size[1]
        self.z_size.value = size[2]

    def set_roi(self, center: List[int], size: List[int]):
        """Set the ROI using center and size."""
        self.set_center(center)
        self.set_size(size)
        self.sync_size.value = min(size)
        self._on_power_mode_changed(self.power_mode_checkbox.value)