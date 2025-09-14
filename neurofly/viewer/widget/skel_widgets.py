from magicgui import widgets

class SkelWidget(widgets.Container):
    def __init__(self):
        self.init_widgets()
        self.init_container_row()
        super().__init__(
            widgets=[
                self.control_row,
            ],
            layout='vertical',
            labels=True
        )
        self.control_row.margins = [0,0,0,0]
        
    def init_widgets(self):
        self.freeze_button = widgets.PushButton(
            text='Freeze View (F)',
            tooltip='Freeze the current view (disable ROI selection)',
            enabled=True
        )
        self.trace_button = widgets.PushButton(
            text='Trace (T)',
            tooltip='Trace skeleton from start to end point',
            enabled=False
        )
        self.delete_button = widgets.PushButton(
            text='Delete Last Skeleton (D)',
            tooltip='Delete the last traced skeleton',
            enabled=False
        )

    def init_container_row(self):
        self.control_row = widgets.Container(
            widgets=[
                self.freeze_button,
                self.trace_button,
                self.delete_button,
            ],
            layout='vertical',
            labels=False
        )
    
    def reset_freeze_button_callback(self, on_freeze_callback):
        self.freeze_button.clicked.disconnect()
        self.freeze_button.clicked.connect(on_freeze_callback)
        
    def reset_trace_button_callback(self, on_trace_callback):
        self.trace_button.clicked.disconnect()
        self.trace_button.clicked.connect(on_trace_callback)

    def reset_delete_button_callback(self, on_delete_callback):
        self.delete_button.clicked.disconnect()
        self.delete_button.clicked.connect(on_delete_callback)
    
    