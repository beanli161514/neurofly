from magicgui import widgets

class RecWidgets(widgets.Container):
    def __init__(self, **kwargs):
        self.init_widgets()
        self.init_container_row()
        super().__init__(
            widgets=[
                self.finder_row,
                self.model_exec_row,
                self.finish_row,
                self.task_row
            ],
            layout='vertical',
            labels=True,
        )
        self.finder_row.margins = [0,0,0,0]
        self.model_exec_row.margins = [0,0,0,0]
        self.finish_row.margins = [0,0,0,0]
        self.task_row.margins = [0,0,0,0]
    
    def init_widgets(self):
        self.database_path_widget = widgets.FileEdit(
            label='Database Path',
            tooltip='Select the path to the database'
        )
        self.deconv_button = widgets.PushButton(
            text='Deconvolve',
            tooltip='Run deconvolution on the selected data'
        )
        self.tracer_button = widgets.PushButton(
            text='Trace',
            tooltip='Run tracing on the selected data'
        )
        self.revoke_button = widgets.PushButton(
            text='Revoke',
            tooltip='Revoke the last action'
        )
        self.check_button = widgets.PushButton(
            text='Check',
            tooltip='Check the current state of the reconstruction',
            enabled=True
        )
        self.submit_button = widgets.PushButton(
            text='Submit',
            tooltip='Submit the current reconstruction',
            enabled=False
        )
        self.next_task_button = widgets.PushButton(
            text='Next Task',
            tooltip='Proceed to the next task in the workflow'
        )
    
    def init_container_row(self):
        self.finder_row = widgets.Container(widgets=[self.database_path_widget], layout='horizontal', label=False)
        self.model_exec_row = widgets.Container(
            widgets=[
                self.deconv_button,
                self.tracer_button,
            ],
            layout='horizontal',
            label=False
        )
        self.finish_row = widgets.Container(
            widgets=[
                self.revoke_button,
                self.check_button,
                self.submit_button,
            ],
            layout='horizontal',
            label=False
        )
        self.task_row = widgets.Container(
            widgets=[self.next_task_button],
            layout='horizontal',
            label=False
        )

    def _add_callback(self):
        """Add callbacks to the buttons and widgets."""
        self.check_button.clicked.connect(self.on_check_button_clicked)

    def reset_database_path_widget_callback(self, on_render_callback):
        """Set the callback for the render button."""
        self.database_path_widget.changed.disconnect()
        self.database_path_widget.changed.connect(on_render_callback)
    
    def reset_deconv_button_callback(self, on_deconv_callback):
        """Set the callback for the deconvolution button."""
        self.deconv_button.clicked.disconnect()
        self.deconv_button.clicked.connect(on_deconv_callback)
    
    def reset_tracer_button_callback(self, on_tracer_callback):
        """Set the callback for the tracer button."""
        self.tracer_button.clicked.disconnect()
        self.tracer_button.clicked.connect(on_tracer_callback)
    
    def reset_revoke_button_callback(self, on_revoke_callback):
        """Set the callback for the revoke button."""
        self.revoke_button.clicked.disconnect()
        self.revoke_button.clicked.connect(on_revoke_callback)
    
    def reset_next_task_button_callback(self, on_next_task_callback):
        """Set the callback for the next task button."""
        self.next_task_button.clicked.disconnect()
        self.next_task_button.clicked.connect(on_next_task_callback)

    def get_database_path(self):
        """Get the path to the database."""
        path = str(self.database_path_widget.value) if self.database_path_widget.value else ''
        return path
    
    def on_check_button_clicked(self):
        if self.check_button.text == 'check':
            self.submit_button.enabled = True
            self.check_button.text = 'uncheck'
            return True
        elif self.check_button.text == 'uncheck':
            self.submit_button.enabled = False
            self.check_button.text = 'check'
            return False
        
    