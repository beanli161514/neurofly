from magicgui import widgets

class DatabaseFinder(widgets.Container):
    def __init__(self, **kwargs):
        self.database_path_widget = widgets.FileEdit(
            label='Database Path',
            tooltip='Select the path to the database'
        )
        self.render_button = widgets.PushButton(
            text='Render',
            tooltip='Render the database'
        )
        super().__init__(
            widgets=[
                self.database_path_widget, 
                # self.render_button
            ],
            layout='vertical',
            labels=True,
            **kwargs
        )
    
    def reset_database_path_widget_callbacks(self, on_render_callback):
        """Set the callback for the render button."""
        self.database_path_widget.changed.disconnect()
        self.database_path_widget.changed.connect(on_render_callback)
    
    def reset_render_button_callback(self, callback):
        """Reset the callback for the render button."""
        self.render_button.clicked.disconnect()
        self.render_button.clicked.connect(callback)

    def get_database_path(self):
        """Get the path to the database."""
        path = str(self.database_path_widget.value) if self.database_path_widget.value else ''
        return path
    