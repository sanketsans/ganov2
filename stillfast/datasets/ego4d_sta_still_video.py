import torch
import json
import os.path
import numpy as np
from PIL import Image
import io
from typing import List

from .build import DATASET_REGISTRY
from stillfast.datasets.sta_hlmdb import Ego4DHLMDB, Ego4DHLMDB_detections
from stillfast.datasets.ego4d_sta_still import Ego4dShortTermAnticipationStill
from stillfast.datasets import StillFastImageTensor
from stillfast.utils import box_ops
import cv2 



class Ego4DHLMDB_STA_Still_Video(Ego4DHLMDB):
    def get(self, video_id: str, frame: int) -> np.ndarray:
        with self._get_parent(video_id) as env:
            with env.begin(write=False) as txn:
                data = txn.get(self.frame_template.format(video_id=video_id,frame_number=frame).encode())

                file_bytes = np.asarray(
                    bytearray(io.BytesIO(data).read()), dtype=np.uint8
                )
                return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    def get_batch(self, video_id: str, frames: List[int]) -> List[np.ndarray]:
        out = []
        with self._get_parent(video_id) as env:
            with env.begin() as txn:
                for frame in frames:
                    data = txn.get(self.frame_template.format(video_id=video_id,frame_number=frame).encode())
                    file_bytes = np.asarray(
                        bytearray(io.BytesIO(data).read()), dtype=np.uint8
                    )
                    out.append(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))
            return out

# TODO: refactor as reconfigurable
@DATASET_REGISTRY.register()
class Ego4dShortTermAnticipationStillVideo(Ego4dShortTermAnticipationStill):
    """
    Ego4d Short Term Anticipation StillVideo Dataset
    """

    def __init__(self, cfg, split):
        super(Ego4dShortTermAnticipationStillVideo, self).__init__(cfg, split)
        self._fast_hlmdb = Ego4DHLMDB_STA_Still_Video(self.cfg.EGO4D_STA.FAST_LMDB_PATH, readonly=True, lock=False)
        self._obj_lmdb = Ego4DHLMDB_detections(self.cfg.EGO4D_STA.OBJECT_DETECTIONS, readonly=True, lock=False)

    def _load_frames_lmdb(self, video_id, frames):
        """ Load images from lmdb. """
        imgs = self._fast_hlmdb.get_batch(video_id, frames)
        return imgs

    def _sample_frames(self, frame):
        """ Sample frames from a video. """
        frames = (
                frame
                - np.arange(
            self.cfg.DATA.FAST.NUM_FRAMES * self.cfg.DATA.FAST.SAMPLING_RATE,
            step=self.cfg.DATA.FAST.SAMPLING_RATE,
            )[::-1]
        )
        frames[frames < 0] = 0

        frames = frames.astype(int)

        return frames

    def _load_still_fast_frames(self, video_id, frame_number, frame_width, frame_height):
        """ Load frames from video_id and frame_number """
        frames_list = self._sample_frames(frame_number)

        fast_imgs = self._load_frames_lmdb(
                video_id, frames_list
            )

        # still_img = self._load_still_frame(video_id, frame_number)
        still_img = self._fast_hlmdb.get(video_id, frame_number)
        still_img = cv2.resize(still_img, (frame_width, frame_height))

        inter_obj_boxes, inter_obj_scores, inter_obj_nouns = self._load_detections(video_id, frames_list, frame_width, frame_height)
        orig_pred_boxes, orig_pred_nouns, orig_pred_scores = inter_obj_boxes[-1], inter_obj_nouns[-1], inter_obj_scores[-1]
        nn = np.array([frame_width, frame_height]*2).reshape(1,-1)
        inter_obj_boxes = [inter_obj_boxes[i] / nn for i in range(len(inter_obj_boxes)) if len(inter_obj_boxes[i]) > 0]
        for i in range(len(inter_obj_boxes)):
            inter_obj_boxes[i][:,[0,2]] *= fast_imgs[0].shape[1]    ## fast_img width
            inter_obj_boxes[i][:,[1,3]] *= fast_imgs[0].shape[0]    ## height

        out_boxes = self.collect_hand_obj_boxes(inter_obj_boxes, inter_obj_scores, inter_obj_nouns)
        orvit_boxes = self.prepare_boxes(out_boxes)
            
        # orvit_boxes = torch.cat((orvit_boxes, orvit_noun_labels), dim=-1)
        return still_img, fast_imgs, orvit_boxes, orig_pred_boxes, orig_pred_nouns, orig_pred_scores
    

    def prepare_boxes(self, boxes):
        ## Not using tensors; instead using numpy arrays 

        boxes = torch.from_numpy(boxes)
        # noun_labels = torch.from_numpy(noun_labels)
        # noun_scores = torch.from_numpy(noun_scores)
        boxes[boxes < 0] = 0
        ### boxes[boxes > 1] = 1
        boxes = boxes.permute(1,0,2) # T, O, 4
        # noun_labels = noun_labels.permute(1, 0, 2)
        # noun_scores = noun_scores.permute(1, 0, 2)
        # boxes = box_ops.box_xyxy_to_cxcywh(boxes) # T, O, 4
        boxes = box_ops.zero_empty_boxes(boxes, mode='xyxy', eps = 0.05)
        return boxes #, noun_labels, noun_scores

    def collect_hand_obj_boxes(self, all_frame_boxes, all_frame_scores, all_frame_labels, with_score=True):
        #  pred_object_labels, inter_obj_nouns,  

        assert with_score

        max_objects = self.cfg.MODEL.MAX_OBJ
        out_boxes = np.zeros([len(all_frame_boxes), max_objects, 4])
        # out_noun_labels = np.ones([len(all_frame_boxes), max_objects, 1]) * self.cfg.MODEL.NOUN_CLASSES + 1 # 100
        # out_noun_scores = np.zeros([len(all_frame_boxes), max_objects, 1])
        for fidx, (curr_box, curr_score, curr_label) in enumerate(zip(all_frame_boxes, all_frame_scores, all_frame_labels)):
            boxes = np.hstack((curr_box, np.expand_dims(curr_score, 1)))
            boxes = box_ops.remove_empty_boxes(boxes) 
            # boxes = osort.update(boxes) 
            if len(boxes) == 0:
                continue 
            cboxes, iboxes = boxes[:,:4], np.arange(len(boxes))[:max_objects] #np.argsort(boxes[:,-1])[::-1][:max_objects] # vgetidx(boxes[:,-1].astype(np.uint64))
            mask = iboxes #iboxes < out_boxes.shape[1]
            cboxes = cboxes[mask]
            out_boxes[fidx, iboxes,:] = cboxes
            # out_noun_labels[fidx, iboxes, :] = np.expand_dims(curr_label[mask], -1)
            # out_noun_scores[fidx, iboxes, :] = np.expand_dims(curr_score[mask], -1)

        return out_boxes.transpose([1, 0, 2]).astype(np.float32) #, out_noun_labels.transpose([1,0,2]).astype(np.float32), out_noun_scores.transpose([1,0,2]).astype(np.float32) ## max_obj, T, 4

    
    def _load_detections(self, video_uid, frame_list, frame_width, frame_height):
        detections = self._obj_lmdb.get_batch(video_uid, frame_list)

        subclips_obj = [] ## bbox, noun
        subclips_nouns = []
        subclips_scores = []

        for d in detections:
            if len(d) > 0:
                boxes = d[:, :4].astype(np.float32)
                scores = d[:, 4].astype(np.float32)
                nouns = d[:, -1].astype(np.float32)
            else:
                boxes = np.array([[0.0, 0.0, float(frame_width), float(frame_height)]], dtype=np.float32) 
                scores = np.zeros(1, dtype=np.float32)
                nouns = np.ones(1, dtype=np.float32) * self.cfg.MODEL.NOUN_CLASSES + 1

            subclips_obj.append(boxes)
            subclips_nouns.append(nouns)
            subclips_scores.append(scores)

        return subclips_obj, subclips_scores, subclips_nouns
    
    def __getitem__(self, idx):
        """ Get the idx-th sample. """
        uid, video_id, frame_width, frame_height, frame_number, gt_boxes, gt_noun_labels, gt_verb_labels, gt_ttc_targets = self._load_annotations(idx)

        still_img, fast_imgs, orvit_boxes, orig_pred_boxes, orig_pred_nouns, orig_pred_scores = self._load_still_fast_frames(video_id, frame_number, frame_width, frame_height)
        
        still_img = self.convert_tensor(still_img)
        fast_imgs = torch.stack([self.convert_tensor(img) for img in fast_imgs], dim=1)

        # FIXME: this is a hack to make the dataset compatible with the original Ego4d dataset
        # This could create problems when producing results on the test set and sending them to the
        # evaluation server.
        if 'v1' not in self.cfg.MODEL.STILLFAST.ROI_HEADS.VERSION:
            verb_offset = 1
        else:
            verb_offset = 0
            
        targets = {
            'boxes': torch.from_numpy(gt_boxes),
            'noun_labels': torch.Tensor(gt_noun_labels).long()+1,
            'verb_labels': torch.Tensor(gt_verb_labels).long()+verb_offset,
            'ttc_targets': torch.Tensor(gt_ttc_targets),
        } if gt_boxes is not None else None
        extra_data = {
            'orig_pred_boxes': orig_pred_boxes, 
            'orig_pred_nouns': orig_pred_nouns,
            'orig_pred_scores': orig_pred_scores
        }

        return {'still_img': still_img, 'fast_imgs': fast_imgs, 'targets': targets, 'uids': uid, 'inter_boxes':orvit_boxes, 'extra_data': extra_data}