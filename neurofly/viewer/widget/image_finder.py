from magicgui import widgets

class ImageFinder(widgets.Container):
    def __init__(self, **kwargs):
        self.image_path_widget = widgets.FileEdit(
            label='Image Path',
            tooltip='Load an image file to display in the viewer',
        )
        self.save_path_widget = widgets.FileEdit(
            label='Save Path',
            tooltip='Save the current ROI',
            mode='d',
        )
        self.save_button = widgets.PushButton(text='Save the ROI')
        self.image_info_textEdit = widgets.TextEdit(
            tooltip='Information about the image',
        )

        self.image_path_row = widgets.Container(widgets=[self.image_path_widget])
        self.save_path_row = widgets.Container(widgets=[self.save_path_widget])
        self.save_button_row = widgets.Container(widgets=[self.save_button])
        self.image_info_row = widgets.Container(widgets=[self.image_info_textEdit])
        self.image_path_row.margins = (0, 0, 0, 0)
        self.save_path_row.margins = (0, 0, 0, 0)
        self.save_button_row.margins = (0, 0, 0, 0)
        super().__init__(
            widgets=[
                self.image_path_row,
                self.save_path_row,
                self.save_button_row,
                self.image_info_row,
            ],
            layout='vertical',
            labels=False,
            **kwargs
        )

    def reset_image_path_widget_callback(self, on_change_callback):
        """Reset the callback for the image path widget."""
        self.image_path_widget.changed.disconnect()
        self.image_path_widget.changed.connect(on_change_callback)
    
    def update_info(self, info:str, append:bool=False):
        """Update the text edit with image information."""
        if append:
            current_text = self.image_info_textEdit.value
            info = current_text + '\n' + info
        else:
           self.image_info_textEdit.value = info

    def get_image_path(self):
        path = str(self.image_path_widget.value)
        return path