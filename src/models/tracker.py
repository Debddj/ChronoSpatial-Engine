import collections
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Track:
    def __init__(self, track_id, bbox, confidence, class_id, max_history=5, distance_scale_factor=1.0):
        self.track_id = track_id
        self.bbox = bbox  # [xmin, ymin, xmax, ymax]
        self.confidence = confidence
        self.class_id = class_id
        self.max_history = max_history
        self.distance_scale_factor = distance_scale_factor
        
        self.time_since_update = 0
        self.age = 0
        
        # Deque of centers to compute velocity
        self.centers_history = collections.deque(maxlen=max_history)
        # Deque of velocity vectors [vx, vy]
        self.velocities_history = collections.deque(maxlen=max_history)
        # Deque of scalar distances
        self.distances_history = collections.deque(maxlen=max_history)

    def update(self, bbox, confidence, w, h):
        self.bbox = bbox
        self.confidence = confidence
        self.time_since_update = 0
        self.age += 1
        
        xmin, ymin, xmax, ymax = bbox
        # Calculate center in normalized coordinates [0.0, 1.0]
        cx = (xmin + xmax) / (2.0 * w) if w > 0 else 0.5
        cy = (ymin + ymax) / (2.0 * h) if h > 0 else 0.5
        self.centers_history.append((cx, cy))
        
        # Calculate velocity: change in center coordinates from previous step
        if len(self.centers_history) > 1:
            prev_cx, prev_cy = self.centers_history[-2]
            vx = cx - prev_cx
            vy = cy - prev_cy
            self.velocities_history.append([vx, vy])
        else:
            self.velocities_history.append([0.0, 0.0])
            
        # Calculate distance: inverse height relative to image height
        bbox_height_ratio = (ymax - ymin) / h if h > 0 else 0.1
        # Clamp distance between 1.0 and 50.0 (matching model expected range)
        distance = float(np.clip(self.distance_scale_factor / (bbox_height_ratio + 1e-6), 1.0, 50.0))
        self.distances_history.append(distance)

    def mark_missed(self):
        self.time_since_update += 1
        # If missed, we propagate its position assuming constant velocity
        if len(self.velocities_history) > 0 and len(self.centers_history) > 0:
            vx, vy = self.velocities_history[-1]
            cx, cy = self.centers_history[-1]
            new_cx = cx + vx
            new_cy = cy + vy
            self.centers_history.append((new_cx, new_cy))
            # Keep same velocity and distance
            self.velocities_history.append([vx, vy])
            if len(self.distances_history) > 0:
                self.distances_history.append(self.distances_history[-1])

    def get_temporal_history(self):
        """
        Returns:
            velocity_vectors: np.ndarray of shape (max_history, 2)
            asset_distances: np.ndarray of shape (max_history,)
        """
        v_list = list(self.velocities_history)
        d_list = list(self.distances_history)
        
        # Pad velocities with [0.0, 0.0] at the beginning if length < max_history
        while len(v_list) < self.max_history:
            v_list.insert(0, [0.0, 0.0])
            
        # Pad distances with the first distance (or 50.0 if empty)
        default_dist = d_list[0] if len(d_list) > 0 else 50.0
        while len(d_list) < self.max_history:
            d_list.insert(0, default_dist)
            
        return np.array(v_list, dtype=np.float32), np.array(d_list, dtype=np.float32)


class SimpleTracker:
    def __init__(self, config):
        model_cfg = config.get("model", {})
        spatial_cfg = model_cfg.get("spatial_hyperparameters", {})
        if not spatial_cfg:
            spatial_cfg = config.get("spatial_hyperparameters", {})
            
        self.grid_size = spatial_cfg.get("grid_size", 16)
        self.max_objects = spatial_cfg.get("max_objects", 50)
        self.iou_threshold = spatial_cfg.get("iou_threshold", 0.3)
        self.distance_scale_factor = spatial_cfg.get("distance_scale_factor", 1.0)
        
        ann_cfg = model_cfg.get("ann_regressor", {})
        self.max_history = ann_cfg.get("max_history", 5)
        
        self.next_track_id = 1
        self.tracks = {}  # track_id -> Track
        
    def update(self, detections, frame_shape):
        """
        Detections: list of dicts with keys "bbox", "confidence", "class_id"
        frame_shape: (H, W, C)
        """
        h, w = frame_shape[0], frame_shape[1]
        
        # Compute IoU matrix between active tracks and detections
        track_ids = list(self.tracks.keys())
        matched_detections = set()
        matched_tracks = set()
        
        if len(track_ids) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                track_box = self.tracks[tid].bbox
                for j, det in enumerate(detections):
                    det_box = det["bbox"]
                    iou_matrix[i, j] = self._compute_iou(track_box, det_box)
            
            # Greedy matching in descending order of IoU score
            flat_indices = np.argsort(iou_matrix, axis=None)[::-1]
            for idx in flat_indices:
                score = iou_matrix.flat[idx]
                if score < self.iou_threshold:
                    break
                # Convert flat index back to row/col index
                i, j = np.unravel_index(idx, iou_matrix.shape)
                tid = track_ids[i]
                if tid in matched_tracks or j in matched_detections:
                    continue
                
                # Match found!
                matched_tracks.add(tid)
                matched_detections.add(j)
                self.tracks[tid].update(detections[j]["bbox"], detections[j]["confidence"], w, h)
                
        # Mark unmatched tracks as missed
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].mark_missed()
                
        # Add new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_detections:
                tid = self.next_track_id
                self.next_track_id += 1
                new_track = Track(
                    track_id=tid,
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                    class_id=det["class_id"],
                    max_history=self.max_history,
                    distance_scale_factor=self.distance_scale_factor
                )
                new_track.update(det["bbox"], det["confidence"], w, h)
                self.tracks[tid] = new_track
                
        # Clean up stale tracks (e.g. not updated for more than 5 frames)
        stale_threshold = 5
        self.tracks = {
            tid: tr for tid, tr in self.tracks.items()
            if tr.time_since_update <= stale_threshold
        }
        
        # Grid-cell mapping and duplicate suppression:
        # If multiple active tracks fall into the exact same cell on the grid_size x grid_size grid,
        # we keep only the highest-confidence track to avoid redundant processing.
        grid_occupancy = {}
        final_tracks = {}
        
        for tid, tr in self.tracks.items():
            if len(tr.centers_history) == 0:
                continue
            cx, cy = tr.centers_history[-1]
            grid_x = int(np.clip(cx * self.grid_size, 0, self.grid_size - 1))
            grid_y = int(np.clip(cy * self.grid_size, 0, self.grid_size - 1))
            grid_cell = (grid_x, grid_y)
            
            if grid_cell not in grid_occupancy:
                grid_occupancy[grid_cell] = (tid, tr.confidence)
                final_tracks[tid] = tr
            else:
                prev_tid, prev_conf = grid_occupancy[grid_cell]
                if tr.confidence > prev_conf:
                    # Replace with higher confidence track
                    del final_tracks[prev_tid]
                    grid_occupancy[grid_cell] = (tid, tr.confidence)
                    final_tracks[tid] = tr
                    
        self.tracks = final_tracks
        
        # Limit to max_objects
        if len(self.tracks) > self.max_objects:
            sorted_tids = sorted(self.tracks.keys(), key=lambda tid: self.tracks[tid].confidence, reverse=True)
            self.tracks = {tid: self.tracks[tid] for tid in sorted_tids[:self.max_objects]}
            
        return self.tracks

    def _compute_iou(self, boxA, boxB):
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
            
        # Compute the area of both rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        # Compute the intersection over union
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
        return iou
