
import numpy as np
from filterpy.kalman import KalmanFilter

def iou(bb_test, bb_gt):
    """
    Computes IOU between two bounding boxes [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    o = inter / (area1 + area2 - inter + 1e-6)  # Added epsilon for stability
    return o

def convert_bbox_to_z(bbox):
    """
    Converts [x1,y1,x2,y2] to [x, y, s, r]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h + 1e-6)  # Avoid div by zero
    return np.array([x, y, s, r]).reshape((4,1))
def convert_bbox_to_z(bbox):
    """
    Convert bbox from [x1, y1, x2, y2] to [x_center, y_center, scale, ratio]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is area
    r = w / float(h + 1e-6)  # add epsilon to avoid division by zero
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Convert state vector [x, y, s, r] to bounding box [x1, y1, x2, y2],
    optionally append score
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-6)
    x1 = x[0] - w / 2.
    y1 = x[1] - h / 2.
    x2 = x[0] + w / 2.
    y2 = x[1] + h / 2.
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))


class KalmanBoxTracker:
    count = 0  # Class variable to assign unique IDs

    def __init__(self, bbox):
        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count  # Assign unique ID
        KalmanBoxTracker.count += 1       # Increment for next tracker
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0


    def update(self, bbox):
        self.time_since_update = 0
        self.history.clear()
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        bbox = convert_x_to_bbox(self.kf.x)
        self.history.append(bbox)
        return bbox

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0,5))):
        self.frame_count += 1

        trks = np.zeros((len(self.trackers), 5))
        to_del = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :4] = pos
            trks[t, 4] = 0  # Dummy score
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Update matched trackers
        for t, trk_idx in enumerate(matched[:,1]):
            det_idx = matched[t,0]
            self.trackers[trk_idx].update(dets[det_idx,:4])

        # Create new trackers for unmatched detections
        for idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[idx,:4]))

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1,-1))  # ID starts at 1
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0,5))

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d,t] = iou(det[:4], trk[:4])

    from scipy.optimize import linear_sum_assignment
    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(matched_indices).T

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:,0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:,1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if len(matches) == 0:
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    