from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import base64
import cv2
import numpy as np
from ultralytics import YOLO

# Install libraries: pip install fastapi uvicorn opencv-python-headless

app = FastAPI()

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model (using YOLOv8 pose model as default for body analysis)
try:
    model = YOLO('yolov8n-pose.pt')
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    model = None

class DetectionRequest(BaseModel):
    body_image_base64: str
    tattoo_image_base64: str
    
def point_to_segment_dist(p, a, b):
    if np.all(a == b):
        return np.linalg.norm(p - a)
    d = b - a
    t = np.dot(p - a, d) / np.dot(d, d)
    t = max(0.0, min(1.0, t))
    projection = a + t * d
    return np.linalg.norm(p - projection)

@app.post("/detect")
async def detect_body_part(request: DetectionRequest):
    if not model:
        raise HTTPException(status_code=500, detail="YOLO model not loaded.")
        
    try:
        def decode_img(b64, flags):
            data = b64.split(',')[1] if ',' in b64 else b64
            nparr = np.frombuffer(base64.b64decode(data), np.uint8)
            return cv2.imdecode(nparr, flags)

        img_body = decode_img(request.body_image_base64, cv2.IMREAD_COLOR)
        img_tattoo = decode_img(request.tattoo_image_base64, cv2.IMREAD_UNCHANGED)
        
        if img_body is None or img_tattoo is None:
            raise ValueError("Failed to decode images")

        # Find tattoo center using normalized coordinates
        if img_tattoo.shape[2] == 4:
            alpha_channel = img_tattoo[:, :, 3]
            y_indices, x_indices = np.where(alpha_channel > 0)
            if len(y_indices) > 0:
                tattoo_center_x = float(np.mean(x_indices)) / img_tattoo.shape[1]
                tattoo_center_y = float(np.mean(y_indices)) / img_tattoo.shape[0]
            else:
                tattoo_center_x, tattoo_center_y = 0.5, 0.5
        else:
            tattoo_center_x, tattoo_center_y = 0.5, 0.5

        # Denormalize to body image size
        tattoo_pt = np.array([tattoo_center_x * img_body.shape[1], tattoo_center_y * img_body.shape[0]])

        # Run inference
        results = model(img_body)
        detected_part = "unknown"
        
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            # Find the person closest to the tattoo
            best_person_idx = 0
            min_person_dist = float('inf')
            xy_data = results[0].keypoints.xy.cpu().numpy()
            
            for i, kp in enumerate(xy_data):
                valid_kp = kp[(kp[:, 0] > 0) & (kp[:, 1] > 0)]
                if len(valid_kp) > 0:
                    person_center = valid_kp.mean(axis=0)
                    dist = np.linalg.norm(person_center - tattoo_pt)
                    if dist < min_person_dist:
                        min_person_dist = dist
                        best_person_idx = i

            kp = xy_data[best_person_idx]
            
            # Helper to check how far along a segment a point is (0 to 1, or >1 if past the end)
            def get_t(p, a, b):
                if np.all(a == b): return 0
                d = b - a
                return np.dot(p - a, d) / np.dot(d, d)
            
            distances = {}
            
            # Face (points 0,1,2,3,4)
            face_pts = [kp[i] for i in range(5) if kp[i][0] > 0]
            if face_pts:
                distances["face"] = min([np.linalg.norm(p - tattoo_pt) for p in face_pts])
                
            # Arm segments: (5,7) Upper Left, (7,9) Lower Left, (6,8) Upper Right, (8,10) Lower Right
            arm_segs = [(5,7), (7,9), (6,8), (8,10)]
            arm_dists = []
            for a, b in arm_segs:
                if kp[a][0] > 0 and kp[b][0] > 0:
                    arm_dists.append(point_to_segment_dist(tattoo_pt, kp[a], kp[b]))
                else:
                    arm_dists.append(float('inf'))
                    
            if min(arm_dists) < float('inf'):
                distances["arm"] = min(arm_dists)
                closest_arm_seg_idx = arm_dists.index(distances["arm"])
                
            # Leg segments: (11,13), (13,15) and (12,14), (14,16)
            leg_segs = [(11,13), (13,15), (12,14), (14,16)]
            leg_dists = []
            for a, b in leg_segs:
                if kp[a][0] > 0 and kp[b][0] > 0:
                    leg_dists.append(point_to_segment_dist(tattoo_pt, kp[a], kp[b]))
            if leg_dists:
                distances["leg"] = min(leg_dists)
                
            # Torso: segments between shoulders (5,6), hips (11,12), and the vertical center line
            torso_dists = []
            if kp[5][0] > 0 and kp[6][0] > 0:
                torso_dists.append(point_to_segment_dist(tattoo_pt, kp[5], kp[6]))
            if kp[11][0] > 0 and kp[12][0] > 0:
                torso_dists.append(point_to_segment_dist(tattoo_pt, kp[11], kp[12]))
                
            # Center vertical line of torso
            if kp[5][0] > 0 and kp[6][0] > 0 and kp[11][0] > 0 and kp[12][0] > 0:
                mid_shoulder = (kp[5] + kp[6]) / 2.0
                mid_hip = (kp[11] + kp[12]) / 2.0
                torso_dists.append(point_to_segment_dist(tattoo_pt, mid_shoulder, mid_hip))
                # Also diagonals
                torso_dists.append(point_to_segment_dist(tattoo_pt, kp[5], kp[12]))
                torso_dists.append(point_to_segment_dist(tattoo_pt, kp[6], kp[11]))
                
            if torso_dists:
                distances["torso"] = min(torso_dists)
                
            if distances:
                detected_part = min(distances, key=distances.get)
                print(f"Distances: {distances}")
                
            # Sub-classify parts
            if detected_part == "torso":
                # Compare X coordinates of Left Shoulder (5) and Right Shoulder (6)
                if kp[5][0] > 0 and kp[6][0] > 0:
                    if kp[5][0] < kp[6][0]:
                        detected_part = "back"
                    else:
                        detected_part = "chest"
                else:
                    detected_part = "chest" # Default fallback
                    
                # Sub-classify chest into belly if it's placed on the lower half of the torso
                if detected_part == "chest" and kp[5][0] > 0 and kp[6][0] > 0 and kp[11][0] > 0 and kp[12][0] > 0:
                    mid_shoulder = (kp[5] + kp[6]) / 2.0
                    mid_hip = (kp[11] + kp[12]) / 2.0
                    # get_t returns 0 at shoulder level and 1 at hip level
                    if get_t(tattoo_pt, mid_shoulder, mid_hip) > 0.5:
                        detected_part = "belly"
                        
            elif detected_part == "arm":
                # Check if it's near or past the wrist (hand)
                if closest_arm_seg_idx == 1: # Left Forearm (7->9)
                    if get_t(tattoo_pt, kp[7], kp[9]) > 0.85:
                        detected_part = "hand"
                elif closest_arm_seg_idx == 3: # Right Forearm (8->10)
                    if get_t(tattoo_pt, kp[8], kp[10]) > 0.85:
                        detected_part = "hand"

        print(f"Tattoo center: {tattoo_pt}, Detected part: {detected_part}")
        return {"status": "success", "detected_body_part": detected_part}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during detection: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
