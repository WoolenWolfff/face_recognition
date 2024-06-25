from supervision import Detections
import numpy as np

def custom_detections(network_output):
    xyxy = []
    confidence = []
    class_ids = []
    ages = []
    for item in network_output:
        x = item['region']['x']
        y = item['region']['y']
        w = item['region']['w']
        h = item['region']['h']
        face_confidence = item['face_confidence']
        dominant_gender = item['dominant_gender']
        age = item['age']
        
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        
        if dominant_gender == 'Man':
            class_id = 1
        else:
            class_id = 0
        
        xyxy.append([x1, y1, x2, y2])
        confidence.append(face_confidence)
        class_ids.append(class_id)
        ages.append(age)
        
    ages = np.array(ages)
    xyxy = np.array(xyxy)
    confidence = np.array(confidence)
    class_ids = np.array(class_ids)
    detections = Detections(xyxy=xyxy, confidence=confidence, class_id=class_ids)
    detections.data = {'ages': ages}
    return detections