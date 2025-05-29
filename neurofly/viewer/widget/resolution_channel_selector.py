from magicgui import widgets

class ResolutionChannelSelector(widgets.Container):
    def __init__(self, **kwargs):
        self.resolution_combobox = widgets.ComboBox(
            choices=[], 
            label='Resolution',
            tooltip='Select resolution level'
        )
        self.channel_combobox = widgets.ComboBox(
            choices=[], 
            label='Channel',
            tooltip='Select channel'
        )
        self.level_up_button = widgets.PushButton(text='Level Up')
        self.level_down_button = widgets.PushButton(text='Level Down')
        # self.full_view_button = widgets.PushButton(text='Full View')
        self.refresh_button = widgets.PushButton(text='Refresh')

        self.combobox_row = widgets.Container(
            widgets=[self.resolution_combobox, self.channel_combobox],
            layout='horizontal',
            labels=True
        )
        self.level_button_row = widgets.Container(
            widgets=[self.level_up_button, self.level_down_button],
            layout='horizontal',
            labels=False
        )
        self.update_button_row = widgets.Container(
            # widgets=[self.full_view_button, self.refresh_button],
            widgets=[self.refresh_button],
            layout='horizontal',
            labels=False
        )

        self.combobox_row.margins = (0, 0, 0, 0)
        self.level_button_row.margins = (0, 0, 0, 0)
        self.update_button_row.margins = (0, 0, 0, 0)

        super().__init__(
            widgets=[
                self.combobox_row,
                self.level_button_row,
                self.update_button_row,
            ],
            layout='vertical',
            labels=False,
            **kwargs
        )
    
    def reset_combobox_choices(self, resolution_choices, channel_choices):
        """Reset the choices for resolution and channel."""
        self.resolution_combobox.choices = resolution_choices
        self.channel_combobox.choices = channel_choices
        # if resolution_choices:
        #     self.resolution_combobox.value = resolution_choices[0]
        # if channel_choices:
        #     self.channel_combobox.value = channel_choices[0]
    
    def reset_resolution_combobox_callback(self, callback):
        """Reset the callback for resolution changes."""
        self.resolution_combobox.changed.disconnect()
        self.resolution_combobox.changed.connect(callback)
    
    def reset_channel_combobox_callback(self, callback):
        """Reset the callback for channel changes."""
        self.channel_combobox.changed.disconnect()
        self.channel_combobox.changed.connect(callback)

    def reset_combobox_callbacks(self, resolution_callback, channel_callback):
        """Reset the callbacks for resolution and channel changes."""
        self.reset_resolution_combobox_callback(resolution_callback)
        self.reset_channel_combobox_callback(channel_callback)

    def reset_button_callbacks(self, level_up_callback, level_down_callback, refresh_callback):
        """Reset the callbacks for level up, level down, and refresh buttons."""
        self.level_up_button.clicked.disconnect()
        self.level_up_button.clicked.connect(level_up_callback)
        
        self.level_down_button.clicked.disconnect()
        self.level_down_button.clicked.connect(level_down_callback)
        
        self.refresh_button.clicked.disconnect()
        self.refresh_button.clicked.connect(refresh_callback)
    
    def set_resolution(self, resolution):
        """Set the currently selected resolution."""
        if resolution in self.resolution_combobox.choices:
            self.resolution_combobox.value = resolution
        else:
            raise ValueError(f"Resolution '{resolution}' not in available choices.")
    
    def set_channel(self, channel):
        """Set the currently selected channel."""
        if channel in self.channel_combobox.choices:
            self.channel_combobox.value = channel
        else:
            raise ValueError(f"Channel '{channel}' not in available choices.")

    def get_resolution(self):
        """Get the currently selected resolution."""
        return self.resolution_combobox.value
    
    def get_channel(self):
        """Get the currently selected channel."""
        return self.channel_combobox.value
    