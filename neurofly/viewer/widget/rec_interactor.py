from magicgui import widgets

class RecWidgets(widgets.Container):
    def __init__(self, **kwargs):
        self.init_widgets()
        self.init_container_row()
        self._add_callback()

        super().__init__(
            widgets=[
                self.finder_row,
                self.config_row,
                self.view_trans_row,
                self.finish_row,
                self.task_row,
            ],
            layout='vertical',
            labels=True,
        )
        self.finder_row.margins = [0,0,0,0]
        self.config_row.margins = [0,0,0,0]
        self.view_trans_row.margins = [0,0,0,0]
        self.finish_row.margins = [0,0,0,0]
        self.task_row.margins = [0,0,0,0]
    
    def init_widgets(self):
        self.database_path_widget = widgets.FileEdit(
            label='Database Path',
            tooltip='Select the path to the database',
            filter='*.db'
        )
        self.username_widget = widgets.LineEdit(
            label='User',
            tooltip='Enter username',
            value=''
        )
        self.node_type_comboBox = widgets.ComboBox(
            label='NodeType',
            tooltip='Select the type of node to create',
            choices=[
                "undefined",         # 0
                "soma",              # 1
                "axon",              # 2
                "(basal) dendrite",  # 3
                "apical dendrite",   # 4
                "fork point",        # 5
                "end point",         # 6
                "ambiguous"          # 7]
            ],
            value='undefined'  # Default value
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
        self.last_task_button = widgets.PushButton(
            text='Last Checked Task',
            tooltip='Back to the last task in the workflow'
        )
        self.next_task_button = widgets.PushButton(
            text='Next Unchecked Task',
            tooltip='Proceed to the next task in the workflow'
        )
        self.proofreading_checkbox = widgets.CheckBox(
            label='Proofreading',
            tooltip='Enable proofreading mode for the reconstruction'
        )
    
    def init_container_row(self):
        self.finder_row = widgets.Container(
            widgets=[self.database_path_widget], 
            layout='horizontal', 
            label=False
        )
        self.config_row = widgets.Container(
            widgets=[
                self.username_widget,
                self.node_type_comboBox
            ],
            layout='horizontal',
            label=False
        )
        self.view_trans_row = widgets.Container(
            widgets=[
                self.proofreading_checkbox,
                self.deconv_button,
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
            widgets=[
                self.last_task_button,
                self.next_task_button
            ],
            layout='horizontal',
            label=False
        )

    def _add_callback(self):
        """Add callbacks to the buttons and widgets."""
        self.check_button.clicked.connect(self.on_check_button_clicked)

    def reset_database_path_widget_callback(self, on_dbpath_callback):
        """Set the callback for the render button."""
        self.database_path_widget.changed.disconnect()
        self.database_path_widget.changed.connect(on_dbpath_callback)
    
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
    
    def reset_last_task_button_callback(self, on_last_task_callback):
        """Set the callback for the next task button."""
        self.last_task_button.clicked.disconnect()
        self.last_task_button.clicked.connect(on_last_task_callback)
    
    def reset_next_task_button_callback(self, on_next_task_callback):
        """Set the callback for the next task button."""
        self.next_task_button.clicked.disconnect()
        self.next_task_button.clicked.connect(on_next_task_callback)
    
    def reset_submit_button_callback(self, on_submit_callback):
        """Set the callback for the submit button."""
        self.submit_button.clicked.disconnect()
        self.submit_button.clicked.connect(on_submit_callback)

    def reset_proofreading_checkbox_callback(self, on_proofreading_callback):
        """Set the callback for the proofreading checkbox."""
        self.proofreading_checkbox.changed.disconnect()
        self.proofreading_checkbox.changed.connect(on_proofreading_callback)

    def get_database_path(self):
        """Get the path to the database."""
        path = str(self.database_path_widget.value) if self.database_path_widget.value else ''
        return path

    def get_username(self):
        """Get the username for the database."""
        username = str(self.username_widget.value) if self.username_widget.value else ''
        return username

    def set_node_type_idx(self, idx:int=None):
        """Set the index of the node type."""
        if idx is None or idx < 0 or idx >= len(self.node_type_comboBox.choices):
            idx = 0
        self.node_type_comboBox.value = self.node_type_comboBox.choices[idx]

    def get_node_type_idx(self):
        """Get the index of the selected node type."""
        node_type = self.node_type_comboBox.value
        idx = int(self.node_type_comboBox.choices.index(node_type))
        return idx
    
    def set_proofreading_mode(self, state:bool):
        """Set the state of the proofreading checkbox."""
        self.proofreading_checkbox.value = state

    def get_proofreading_mode(self):
        """Get the state of the proofreading checkbox."""
        return self.proofreading_checkbox.value
    
    def set_check_button_status(self, text:str):
        """Set the text of the check button."""
        self.check_button.text = text
        if text == 'Check':
            self.submit_button.enabled = False
        elif text == 'Uncheck':
            self.submit_button.enabled = True

    def on_check_button_clicked(self):
        if self.check_button.text == 'Check':
            self.set_check_button_status('Uncheck')
            return True
        elif self.check_button.text == 'Uncheck':
            self.set_check_button_status('Check')
            return False
        
        
    