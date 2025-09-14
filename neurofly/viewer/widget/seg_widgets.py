from magicgui import widgets

class SegWidget(widgets.Container):
    def __init__(self):
        self.init_widgets()
        self.init_container_row()
        super().__init__(
            widgets=[
                self.control_row,
                self.run_row,
                self.save_row,
            ],
            layout='vertical',
            labels=True
        )
        self.control_row.margins = [0,0,0,0]
        self.run_row.margins = [0,0,0,0]
        self.save_row.margins = [0,0,0,0]
        
    def init_widgets(self):
        self.bg_thres_slider = widgets.Slider(
            label='BG_Threshold',
            min=0, max=255, step=1,
            value=150,
        )
        self.keepBranch_checkbox = widgets.CheckBox(
            label='KeepBranch',
            tooltip='Keep branch segments in the segmentation result',
            value=False
        )
        self.deconv_button = widgets.PushButton(
            text='Deconvolve',
            tooltip='Run deconvolution on the selected data',
            enabled=True
        )
        self.runSeg_button = widgets.PushButton(
            text='Segment',
            tooltip='Run segmentation on the current image',
            enabled=True
        )
        self.progress_bar = widgets.ProgressBar(
            label='Progress',
            min=0, max=100,
            value=0,
            visible=True
        )
        self.save_path_widget = widgets.FileEdit(
            label='Save Path',
            tooltip='Save segmentation result as sqlite db file, format: ${image_name}_${timestamp}.db',
            mode='d',
        )
        self.save_button = widgets.PushButton(
            text='Save',
            tooltip='Save segmentation result to the specified path',
            enabled=True
        )

    def init_container_row(self):
        self.control_row = widgets.Container(
            widgets=[
                self.bg_thres_slider,
                self.keepBranch_checkbox,
            ],
            layout='horizontal',
            label=False
        )
        self.run_row = widgets.Container(
            widgets=[
                self.deconv_button,
                self.runSeg_button,
                self.progress_bar,
            ],
            layout='vertical',
            label=False
        )
        self.save_row = widgets.Container(
            widgets=[
                self.save_path_widget,
                self.save_button,
            ],
            layout='vertical',
            label=False
        )
    
    def reset_deconv_button_callback(self, on_deconv_callback):
        self.deconv_button.clicked.disconnect()
        self.deconv_button.clicked.connect(on_deconv_callback)
    
    def reset_runSeg_button_callback(self, on_seg_callback):
        self.runSeg_button.clicked.disconnect()
        self.runSeg_button.clicked.connect(on_seg_callback)
    
    def reset_save_path_callback(self, on_change_callback):
        self.save_path_widget.changed.disconnect()
        self.save_path_widget.changed.connect(on_change_callback)
    
    def reset_save_button_callback(self, on_save_callback):
        self.save_button.clicked.disconnect()
        self.save_button.clicked.connect(on_save_callback)
    
    def get_bg_threshold(self):
        return self.bg_thres_slider.value
    
    def get_keepBranch_status(self):
        return self.keepBranch_checkbox.value

    def get_save_dir(self):
        save_dir = self.save_path_widget.value
        return save_dir
    
    def set_runSeg_button_enabled(self, enabled:bool):
        self.runSeg_button.enabled = enabled
    
    def set_deconv_button_enabled(self, enabled:bool):
        self.deconv_button.enabled = enabled

    def set_progress_bar_value(self, value:int, max_value:int):
        self.progress_bar.max = max_value
        self.progress_bar.value = value
    
    def set_bg_threshold(self, value:int, max_value:int=255):
        self.bg_thres_slider.max = max_value
        self.bg_thres_slider.value = value
    

