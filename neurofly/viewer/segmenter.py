import os
import napari
import napari.utils
import napari.utils.notifications
from napari.qt.threading import thread_worker
import numpy as np
import datetime

from neurofly.viewer.viewer import NeuronViewer
from neurofly.viewer.widget.seg_widgets import SegWidget
from neurofly.viewer.config.config import Config
from neurofly.model import Deconver, default_dec_weight_path
from neurofly.model import Seger, SegNet, default_seger_weight_path
from neurofly.backend.neuron_graph import NeuroGraph
from neurofly.neurodb.neurodb_sqlite import NeurodbSQLite


class NeuronSegmenter(NeuronViewer):
    def __init__(self, napari_viewer:napari.Viewer):
        super().__init__(napari_viewer)
        # config
        self.config = Config()

        # viewer
        self.viewer.__dict__['neurofly']['segmenter'] = self
        self.nodes_layer = self.viewer.add_points(ndim=3, size=1, shading='spherical', name='nodes')
        self.edges_layer = self.viewer.add_vectors(ndim=3, vector_style='line', edge_color='orange', edge_width=0.3, name='edges')
        self.viewer.layers.selection.active = self.image_layer
        
        self.init_attributes()
        self.init_model()
        self.add_callback()
    
    def init_attributes(self):
        """Initialize attributes for the NeuronSegmenter."""
        super().init_attributes()
        self.SegWidget = SegWidget()
        self.NeuroG = None
        self.SEGS = []
        self.SAVE_PATH = None
        self.extend([self.SegWidget])

    def _init_segn_net(self):
        seger_ckpt_path = self.config.segmenter_cfg.model_ckpt_path
        if seger_ckpt_path.strip() == '' or not os.path.exists(seger_ckpt_path):
            seger_ckpt_path = default_seger_weight_path
        bg_thres = self.SegWidget.get_bg_threshold()
        self.seg_net = SegNet(seger_ckpt_path, bg_thres=bg_thres)
        self.seger = Seger(self.seg_net)

    def init_model(self):
        self.Deconver = Deconver(default_dec_weight_path)
    
    def add_callback(self):
        super().add_callback()
        self.SegWidget.reset_deconv_button_callback(self.run_deconv)
        self.SegWidget.reset_runSeg_button_callback(self.run_segmentation)
        self.SegWidget.reset_save_path_callback(self.on_save_path_changed)
        self.SegWidget.reset_save_button_callback(self.on_save_button_clicked)
    
    def on_save_path_changed(self):
        flag = False
        image_path = self.ImageFinder.get_image_path()
        if image_path == '':
            napari.utils.notifications.show_info('Please load an image first!')
        else:
            save_dir = self.SegWidget.get_save_dir()
            if save_dir == '':
                napari.utils.notifications.show_info('Please set the save path for SWC result!')
            elif os.access(save_dir, os.W_OK) == False:
                napari.utils.notifications.show_info('The save path is not writable!')
            else:
                image_name = os.path.basename(image_path).split('.')[0]
                timestamp = datetime.datetime.now().strftime('%H%M%S')
                save_path = os.path.join(save_dir, f'{image_name}_{timestamp}.db')
                flag = True
        if flag:
            self.SAVE_PATH = save_path
        else:
            self.SAVE_PATH = None
    
    def get_seg_info(self):
        BG_THRESHOLD = self.SegWidget.get_bg_threshold()
        KEEP_BRANCH = self.SegWidget.get_keepBranch_status()
        N_NODES = len(self.NeuroG.nodes) if self.NeuroG else 0
        N_EDGES = len(self.NeuroG.edges) if self.NeuroG else 0
        N_SEGS = sum([len(segs) for segs in self.SEGS]) if self.SEGS else 0
        info = ''
        info += f'segs_num: {N_SEGS}\n'
        info += f'nodes_num: {N_NODES}\n'
        info += f'edges_num: {N_EDGES}\n'
        info += f'Background_Threshold: {BG_THRESHOLD}\n'
        info += f'KeepBranch: {KEEP_BRANCH}\n'
        return info

    def get_info(self):
        image_info = self.get_image_info()
        seg_info = self.get_seg_info()
        info = f"{image_info}\n--------\n{seg_info}"
        return info

    def render(self, init_layers:bool=False):
        if init_layers:
            self.nodes_layer.data = np.zeros((0, 3))
            self.edges_layer.data = np.empty((0, 2, 3))

        if not self.NeuroG or len(self.NeuroG.nodes) == 0:
            return
        
        nodes_coords, nodes_properties, edges_coords, edges_properties = self.NeuroG.get_render_data()
        self.nodes_layer.data = nodes_coords
        self.nodes_layer.properties = {
            'nids': nodes_properties['nids'], 
        }
        self.nodes_layer.size = nodes_properties['sizes']
        if len(nodes_properties['colors']) > 0:
            self.nodes_layer.face_color = nodes_properties['colors']
        self.edges_layer.data = edges_coords
        self.edges_layer.properties = edges_properties

    def refresh(self):
        super().refresh()
        super().update_contrast()
        self.reset_status_for_seg()
        self.SegWidget.set_bg_threshold(150, self.image_layer.data.max() if self.image_layer.data.size > 0 else 255)
        self.render(init_layers=True)

    def patchify_without_splices(self, roi, patch_size):
        rois = []
        xs = list(range(roi[0],roi[0]+roi[3],patch_size))
        xs.append(roi[0]+roi[3])
        ys = list(range(roi[1],roi[1]+roi[4],patch_size))
        ys.append(roi[1]+roi[4])
        zs = list(range(roi[2],roi[2]+roi[5],patch_size))
        zs.append(roi[2]+roi[5])
        for x1,x2 in zip(xs[:-1],xs[1:]):
            for y1,y2 in zip(ys[:-1],ys[1:]):
                for z1,z2 in zip(zs[:-1],zs[1:]):
                    rois.append([x1,y1,z1,x2-x1,y2-y1,z2-z1])
        return rois
    
    def set_run_button_enabled(self, enabled:bool):
        self.SegWidget.set_runSeg_button_enabled(enabled)
        self.SegWidget.set_deconv_button_enabled(enabled)

    def run_deconv(self):
        self.set_run_button_enabled(False)
        deconv_worker = self.process_deconv()
        deconv_worker.yielded.connect(self.update_deconv_progress)
        deconv_worker.finished.connect(self.on_deconv_finished)
        deconv_worker.returned.connect(lambda img: setattr(self.image_layer, 'data', img))
        deconv_worker.start()

    @thread_worker
    def process_deconv(self):
        patch_size = 128
        image = self.image_layer.data
        local_roi = [0, 0, 0] + list(image.shape)
        patch_roi_list = self.patchify_without_splices(local_roi, patch_size)
        for patch_idx, roi in enumerate(patch_roi_list):
            image_patch = image[roi[0]:roi[0]+roi[3], roi[1]:roi[1]+roi[4], roi[2]:roi[2]+roi[5]]
            sr_patch = self.Deconver.process_one(image_patch)
            image[roi[0]:roi[0]+roi[3], roi[1]:roi[1]+roi[4], roi[2]:roi[2]+roi[5]] = sr_patch
            yield patch_idx, len(patch_roi_list), image
        return image
    
    def update_deconv_progress(self, value):
        idx, whole, image = value
        self.SegWidget.set_progress_bar_value(idx+1, whole)
        self.image_layer.data = image

    def on_deconv_finished(self):
       self.set_run_button_enabled(True)

    def run_segmentation(self):
        self.reset_status_for_seg()
        self.set_run_button_enabled(False)
        seg_worker = self.process_segmentation()
        seg_worker.yielded.connect(self.update_seg_progress)
        seg_worker.finished.connect(self.on_seg_finished)
        seg_worker.start()

    def reset_status_for_seg(self):
        self._init_segn_net()
        self.NeuroG = NeuroGraph()
        self.SEGS = []
        self.SegWidget.set_progress_bar_value(0, 100)
        self.render(init_layers=True)
        return True

    @thread_worker
    def process_segmentation(self):
        patch_size = 300
        keep_branch = self.SegWidget.get_keepBranch_status()
        center, size = self.ROISelector.get_roi()
        global_offset = [int(c-s//2) for c,s in zip(center, size)]

        image = self.image_layer.data
        local_roi = [0, 0, 0] + list(image.shape)
        patch_roi_list = self.patchify_without_splices(local_roi, patch_size)
        nid = 0
        for patch_idx, roi in enumerate(patch_roi_list):
            image_patch = image[roi[0]:roi[0]+roi[3], roi[1]:roi[1]+roi[4], roi[2]:roi[2]+roi[5]]
            print(f'Processing patch {patch_idx+1}/{len(patch_roi_list)} with shape {image_patch.shape}')
            if (np.array(roi[3:]) <= np.array([128, 128, 128])).all():
                mask = self.seg_net.get_mask(image_patch)
            else:
                border_width = self.seger.border_width
                image_patch = np.pad(image_patch, (border_width), mode='reflect')
                mask = self.seger.get_large_mask(image_patch)
            patch_offset = [roi[i]+global_offset[i] for i in range(3)]
            SEGS = self.seger.mask_to_segs(mask, keep_branch, offset=patch_offset)
            NODES = {}
            EDGES = {}
            for seg in SEGS:
                for n_idx, node_coord in enumerate(seg['nodes']):
                    _nid = nid + n_idx
                    NODES[_nid] = {'nid':_nid, 'coord':node_coord}
                for e_idx, (src_n_idx, dst_n_idx) in enumerate(seg['edges']):
                    _src_nid = nid + src_n_idx
                    _dst_nid = nid + dst_n_idx
                    EDGES[(_src_nid, _dst_nid)] = {'src':_src_nid, 'dst':_dst_nid}
                nid += len(seg['nodes'])
            self.SEGS.append(SEGS)
            self.NeuroG.add_nodes(NODES)
            self.NeuroG.add_edges(EDGES)
            print(f'Patch {patch_idx+1}/{len(patch_roi_list)}: {len(NODES)} nodes, {len(EDGES)} edges')
            yield patch_idx, len(patch_roi_list), NODES
        return self.SEGS

    def update_seg_progress(self, value):
        idx, whole, NODES = value
        self.SegWidget.set_progress_bar_value(idx+1, whole)
        if len(NODES) > 0:
            existed_node = self.nodes_layer.data
            appended_node = np.array([node['coord'] for nid, node in NODES.items()])
            self.nodes_layer.data = np.vstack([existed_node, appended_node])

    def on_seg_finished(self):
        self.set_run_button_enabled(True)
        self.render()
        self.update_info(self.get_info())
    
    def on_save_button_clicked(self):
        if self.SAVE_PATH is None:
            napari.utils.notifications.show_info('Please set a valid save path first!')
            return
        if len(self.SEGS) == 0 or self.NeuroG is None or len(self.NeuroG.nodes) == 0:
            napari.utils.notifications.show_info('No segmentation result to save!')
            return
        neurodb = NeurodbSQLite(db_path=self.SAVE_PATH)
        for SEGS in self.SEGS:
            neurodb.segs2db(SEGS)
        napari.utils.notifications.show_info(f'Segmentation result saved to {self.SAVE_PATH}!')
        
def main():
    """Main function to run the NeuronSegmenter."""
    viewer = napari.Viewer()
    reconstructor = NeuronSegmenter(viewer)
    viewer.window.add_dock_widget(reconstructor, name='Neuron Segmenter')
    napari.run()

if __name__ == "__main__":
    main()
