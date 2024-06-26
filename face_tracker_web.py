import cv2
from deepface import DeepFace
import supervision as sv
import numpy as np
from utils import custom_detections

bbox_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
tracker = sv.ByteTrack()
tracker.reset()

def main(
    source=0
):
    video = cv2.VideoCapture(source)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    crop_dict = {}
    miniature_size = (75, 75)
    insert_x = width - miniature_size[0]
    insert_y = 0
    miniature = np.zeros((miniature_size[1], miniature_size[0], 3), dtype=np.uint8)
    result_text = ' '
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        results = DeepFace.analyze(frame, 
            actions = ['age', 'gender'],
            enforce_detection=False,
            detector_backend='yolov8'
            )
        detections = custom_detections(results)
        detections = detections[detections.confidence > 0.5]
        detections = tracker.update_with_detections(detections=detections)
        
        annotated_frame = frame.copy()
        annotated_frame = bbox_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        
        removed_tracks = [i.external_track_id for i in tracker.removed_tracks]
        if detections.tracker_id.size != 0:
            labels = []
            ages = detections.data['ages']           
            
            for i, id in enumerate(detections.tracker_id):
                if id not in crop_dict or detections.confidence[i] > crop_dict[id]['conf']:
                    x1, y1, x2, y2 = detections.xyxy[i]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    crop = frame[y1:y2, x1:x2]
                    _miniature = cv2.resize(crop, miniature_size)
                    
                    crop_dict[id] = {
                        'conf': detections.confidence[i],
                        'Gender': 'Man' if detections.class_id[i] == 1 else 'Woman',
                        'Age': ages[i],
                        'miniature': _miniature
                    }
                
                labels.append(f"Id #{id} {'Man' if detections.class_id[i] == 1 else 'Woman'} {ages[i]}")
                
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )
        
        if len(removed_tracks) != 0 and removed_tracks[0] in crop_dict:
            miniature = crop_dict[removed_tracks[0]]['miniature']
            result_text = crop_dict[removed_tracks[0]]['Gender'] + ' ' + str(crop_dict[removed_tracks[0]]['Age'])
        
        cv2.putText(annotated_frame, result_text, (width-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA, )
        
        annotated_frame[insert_y:insert_y+miniature_size[1], insert_x:insert_x+miniature_size[0]] = miniature
        cv2.imshow('video', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows() 
    video.release()

if __name__ == '__main__':
    main()