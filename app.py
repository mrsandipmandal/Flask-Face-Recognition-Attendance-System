from flask import Flask, render_template, request, Response, send_file, jsonify, redirect, url_for
import sqlite3
from datetime import datetime
import cv2
import os
import numpy as np
import dlib
import logging
from datetime import date


# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Face detection setup
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
path_photos_from_camera = "data/faces/"
attendance_path = "data/attendance/"
current_face_dir = ""
captured_poses = {}
manual_capture = True
pose_order = ["front", "up", "down", "left", "right"]
current_pose_idx = 0

# Camera setup
def init_camera():
    global cap
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logging.warning("CAP_DSHOW failed, trying default backend")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open camera with any backend")
        return False
    return True

# Database initialization
def init_db(db_name='attendance.db'):
    # Check if database file exists
    db_exists = os.path.exists(db_name)
    
    # Connect to database (will create it if it doesn't exist)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Only create tables if database didn't exist before
    if not db_exists:
        # Create attendance_employee table
        cursor.execute('''CREATE TABLE attendance_employee (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL,
            emp_id VARCHAR(50) NOT NULL UNIQUE,
            front_image VARCHAR(100),
            up_image VARCHAR(100),
            down_image VARCHAR(100),
            left_image VARCHAR(100),
            right_image VARCHAR(100),
            face_encoding BLOB NULL
        )''')
        
        # Create attendance table
        cursor.execute('''CREATE TABLE attendance (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            emp_id VARCHAR(50) NOT NULL UNIQUE,
            name TEXT,
            time TEXT,
            date DATE NOT NULL,
            path VARCHAR(100)
        )''')
        
        conn.commit()
        print(f"Database '{db_name}' and tables created successfully")
    else:
        print(f"Database '{db_name}' already exists")
    
    conn.close()

if __name__ == "__main__":
    init_db()

def estimate_head_pose(landmarks):
    """
    Improved head pose estimation using facial landmarks
    """
    # Key landmarks
    nose_tip = landmarks.part(30)        # Nose tip
    chin = landmarks.part(8)             # Chin
    left_eye_left = landmarks.part(36)   # Left eye left corner
    left_eye_right = landmarks.part(39)  # Left eye right corner
    right_eye_left = landmarks.part(42)  # Right eye left corner
    right_eye_right = landmarks.part(45) # Right eye right corner
    left_mouth = landmarks.part(48)      # Left mouth corner
    right_mouth = landmarks.part(54)     # Right mouth corner
    
    # Calculate eye centers
    left_eye_center = ((left_eye_left.x + left_eye_right.x) // 2,
                       (left_eye_left.y + left_eye_right.y) // 2)
    right_eye_center = ((right_eye_left.x + right_eye_right.x) // 2,
                        (right_eye_left.y + right_eye_right.y) // 2)
    
    # Calculate eye center as reference point
    eye_center_x = (left_eye_center[0] + right_eye_center[0]) // 2
    eye_center_y = (left_eye_center[1] + right_eye_center[1]) // 2
    
    # Calculate eye width (distance between eyes)
    eye_width = np.sqrt((right_eye_center[0] - left_eye_center[0])**2 +
                        (right_eye_center[1] - left_eye_center[1])**2)
    
    # Calculate face height (distance from eyes to chin)
    face_height = np.sqrt((eye_center_x - chin.x)**2 + (eye_center_y - chin.y)**2)
    
    # Normalize measurements relative to face dimensions
    # Vertical movement (pitch: up/down)
    nose_to_eye_y_norm = (nose_tip.y - eye_center_y) / face_height
    
    # Horizontal movement (yaw: left/right)
    nose_to_eye_x_norm = (nose_tip.x - eye_center_x) / eye_width
    
    # Calculate mouth angle for additional roll information
    mouth_angle = np.arctan2(right_mouth.y - left_mouth.y,
                            right_mouth.x - left_mouth.x) * 180 / np.pi
    
    # Log values for debugging
    # logging.debug(f"Eye Center: ({eye_center_x}, {eye_center_y})")
    # logging.debug(f"Nose: ({nose_tip.x}, {nose_tip.y}), Chin: ({chin.x}, {chin.y})")
    # logging.debug(f"Vertical norm: {nose_to_eye_y_norm:.3f}")
    # logging.debug(f"Horizontal norm: {nose_to_eye_x_norm:.3f}")
    # logging.debug(f"Mouth angle: {mouth_angle:.2f} degrees")
    
    # Modified threshold for "up" detection to be more sensitive
    if nose_to_eye_y_norm < -0.005:  # Relaxed threshold for "up" detection
        return "up"
    elif nose_to_eye_y_norm > 0.25:  # Nose below eyes = looking down
        return "down"
    elif nose_to_eye_x_norm < -0.25:  # Nose left of center = looking left
        return "left"
    elif nose_to_eye_x_norm > 0.20:   # Nose right of center = looking right
        return "right"
    else:  # Neutral position
        return "front"

def is_image_clear(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > 50

def generate_3d_face_encoding(images):
    encodings = []
    for img_path in images.values():
        img = cv2.imread(img_path)
        faces = detector(img, 0)
        if len(faces) == 1:
            shape = predictor(img, faces[0])
            encoding = face_reco_model.compute_face_descriptor(img, shape)
            encodings.append(np.array(encoding))
    if encodings:
        return np.mean(encodings, axis=0).tobytes()
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attendance_list')
def attendance_list():
    today = date.today().strftime('%Y-%m-%d')
    return render_template('attendance_list.html', selected_date='', no_data=False, today=today)

# @app.route('/attendance', methods=['GET', 'POST'])
# def attendance():
#     if request.method == 'POST':
#         conn = sqlite3.connect('attendance.db')
#         cursor = conn.cursor()
#         if 'delete_attendance' in request.form:
#             selected_emp_ids = request.form.getlist('emp_ids')
#             if selected_emp_ids:
#                 # First get all the image paths for the selected employees
#                 cursor.execute("""
#                     SELECT emp_id, name, time, date, path 
#                     FROM attendance 
#                     WHERE emp_id IN ({})
#                 """.format(','.join('?' for _ in selected_emp_ids)), selected_emp_ids)
#                 attendance_to_delete = cursor.fetchall()
                
#                 deleted_count = 0
#                 for emp in attendance_to_delete:
#                     emp_id, name = emp[0], emp[1]
#                     image_paths = emp[4]  # image paths
#                     print(f"Image Path : {image_paths}")
#                     # Delete image files
#                     for path in image_paths:
#                         if path and os.path.exists(path):
#                             try:
#                                 os.remove(path)
#                                 logging.info(f"Deleted file: {path}")
#                             except Exception as e:
#                                 logging.error(f"Error deleting file {path}: {e}")
                    
#                     # Also delete from attendance table
#                     cursor.execute("DELETE FROM attendance WHERE emp_id = ?", (emp_id,))
#                     deleted_count += 1
                
#                 conn.commit()
#                 message = f"Successfully deleted {deleted_count} attendance and their data"
#             else:
#                 message = "No attendance selected for deletion"
        
#         selected_date = request.form.get('selected_date')
#         selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
#         formatted_date = selected_date_obj.strftime('%Y-%m-%d')
#         cursor.execute("SELECT name, time, path, emp_id FROM attendance WHERE date = ?", (formatted_date,))
#         attendance_data = cursor.fetchall()
#         conn.close()
#         if not attendance_data:
#             return render_template('attendance_list.html', selected_date=selected_date, no_data=True, today = selected_date)
#         return render_template('attendance_list.html', selected_date=selected_date, attendance_data=attendance_data, today = selected_date)
#     return render_template('attendance_list.html', selected_date='', no_data=False, today = selected_date)

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        # Initialize database connection and cursor at the beginning of the function
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        if 'delete_attendance' in request.form:
            selected_emp_ids = request.form.getlist('emp_ids')
            if selected_emp_ids:
                # First get all the image paths for the selected employees
                cursor.execute("""
                    SELECT emp_id, name, time, date, path 
                    FROM attendance 
                    WHERE emp_id IN ({})
                """.format(','.join('?' for _ in selected_emp_ids)), selected_emp_ids)
                attendance_to_delete = cursor.fetchall()
                
                deleted_count = 0
                for emp in attendance_to_delete:
                    emp_id, name = emp[0], emp[1]
                    image_path = emp[4]  # image path
                    
                    # Delete image file if it exists
                    if image_path and os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                            logging.info(f"Deleted file: {image_path}")
                        except Exception as e:
                            logging.error(f"Error deleting file {image_path}: {e}")
                    
                    # Also delete from attendance table
                    cursor.execute("DELETE FROM attendance WHERE emp_id = ?", (emp_id,))
                    deleted_count += 1
                
                conn.commit()
                message = f"Successfully deleted {deleted_count} attendance and their data"
            else:
                message = "No attendance selected for deletion"
        
        # Get the selected date from the form
        selected_date = request.form.get('selected_date')
        
        # Check if selected_date is None or empty
        if not selected_date:
            # Use today's date as default if no date is selected
            selected_date = date.today().strftime('%Y-%m-%d')
            formatted_date = selected_date
        else:
            # Parse the date if it's provided
            try:
                selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
                formatted_date = selected_date_obj.strftime('%Y-%m-%d')
            except ValueError:
                # Handle invalid date format
                formatted_date = date.today().strftime('%Y-%m-%d')
                selected_date = formatted_date
        
        # Use the existing connection if we're in a POST request
        cursor.execute("SELECT name, time, path, emp_id FROM attendance WHERE date = ?", (formatted_date,))
        attendance_data = cursor.fetchall()
        conn.close()
        
        if not attendance_data:
            return render_template('attendance_list.html', selected_date=selected_date, no_data=True, today=selected_date)
        return render_template('attendance_list.html', selected_date=selected_date, attendance_data=attendance_data, today=selected_date)
    
    # For GET requests
    today = date.today().strftime('%Y-%m-%d')
    return render_template('attendance_list.html', selected_date='', no_data=False, today=today)

# @app.route('/collect_faces', methods=['GET', 'POST'])
# def collect_faces():
#     global current_face_dir, captured_poses, manual_capture, current_pose_idx
#     if request.method == 'POST':
#         name = request.form.get('name')
#         emp_id = request.form.get('emp_id')  # This will be like "001"
#         capture_mode = request.form.get('capture_mode', 'auto')
#         if name is None or emp_id is None:
#             return render_template('collect_faces.html', message="Name and Employee ID are required.")
        
#         # First, insert the employee into the database
#         conn = sqlite3.connect('attendance.db')
#         cursor = conn.cursor()
#         cursor.execute("""
#             INSERT INTO attendance_employee (name, emp_id) 
#             VALUES (?, ?)
#         """, (name, emp_id))
#         conn.commit()
#         conn.close()
        
#         if not os.path.isdir(path_photos_from_camera):
#             os.makedirs(path_photos_from_camera)
        
#         current_face_dir = f"{path_photos_from_camera}{emp_id}_{name}"
#         os.makedirs(current_face_dir, exist_ok=True)
#         captured_poses = {}
#         manual_capture = (capture_mode == 'manual')
#         current_pose_idx = 0
#         logging.info(f"Starting collection for {name} (ID: {emp_id}), Mode: {capture_mode}")
#         return render_template('collect_faces.html', 
#                             message=f"Started collecting 5 poses for {name} (ID: {emp_id})", 
#                             video_feed=(capture_mode == 'auto'), 
#                             manual=manual_capture,
#                             emp_id=emp_id,
#                             instructions="Order: Front -> Up -> Down -> Left -> Right")
#     return render_template('collect_faces.html')

@app.route('/collect_faces', methods=['GET', 'POST'])
def collect_faces():
    global current_face_dir, captured_poses, current_pose_idx
    if request.method == 'POST':
        name = request.form.get('name')
        emp_id = request.form.get('emp_id')
        if name is None or emp_id is None:
            return render_template('collect_faces.html', message="Name and Employee ID are required.")
        
        # First, insert the employee into the database
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO attendance_employee (name, emp_id) 
            VALUES (?, ?)
        """, (name, emp_id))
        conn.commit()
        conn.close()
        
        if not os.path.isdir(path_photos_from_camera):
            os.makedirs(path_photos_from_camera)
        
        current_face_dir = f"{path_photos_from_camera}{emp_id}_{name}"
        os.makedirs(current_face_dir, exist_ok=True)
        captured_poses = {}
        # Removed manual_capture variable
        current_pose_idx = 0
        logging.info(f"Starting collection for {name} (ID: {emp_id})")
        return render_template('collect_faces.html', 
                            message=f"Started collecting 5 poses for {name} (ID: {emp_id})", 
                            video_feed=True,
                            emp_id=emp_id,
                            instructions="Order: Front -> Up -> Down -> Left -> Right")
    return render_template('collect_faces.html')

@app.route('/video_feed')
def video_feed():
    emp_id = request.args.get('emp_id', 'EMP')
    return Response(gen_frames(emp_id), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames(emp_id):
    global captured_poses, manual_capture, current_pose_idx
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        yield b'Camera error'
        return
    
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    while True:  # Changed to infinite loop to prevent stream breaking
        if len(captured_poses) >= 5:  # When all poses are captured, keep streaming but don't capture more
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            cv2.putText(frame, "All poses captured!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            break
            # continue
            
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to grab frame")
            break
        frame = cv2.resize(frame, (640, 480))
        faces = detector(frame, 0)
        pose = "None"
        logging.debug(f"Detected {len(faces)} faces")
        
        if len(faces) == 1:
            d = faces[0]
            hh, ww = int((d.bottom() - d.top()) / 2), int((d.right() - d.left()) / 2)
            if (d.right() + ww <= 640 and d.bottom() + hh <= 480 and d.left() - ww >= 0 and d.top() - hh >= 0):
                shape = predictor(frame, d)
                pose = estimate_head_pose(shape)
                logging.debug(f"Pose detected: {pose}")
                expected_pose = pose_order[current_pose_idx]
                if pose == expected_pose and pose not in captured_poses:
                    face_roi = frame[d.top() - hh:d.bottom() + hh, d.left() - ww:d.right() + ww]
                    if is_image_clear(face_roi):
                        # Use numeric emp_id format like "001_front"
                        image_path = f"{current_face_dir}/{emp_id}_{pose}.jpg"
                        cv2.imwrite(image_path, face_roi)
                        captured_poses[pose] = image_path
                        logging.info(f"Captured {pose} at {image_path}")
                        current_pose_idx += 1
                        
                        # Update database with image path
                        column_name = f"{pose}_image"
                        cursor.execute(f"UPDATE attendance_employee SET {column_name} = ? WHERE emp_id = ?",
                                     (image_path, emp_id))
                        conn.commit()
                    else:
                        logging.debug(f"Image for {pose} is blurry, skipping")
            
            cv2.rectangle(frame, (d.left() - ww, d.top() - hh), (d.right() + ww, d.bottom() + hh), (255, 255, 255), 2)
        
        # Only show "Next" text if we haven't captured all poses
        cv2.putText(frame, f"Pose: {pose}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if current_pose_idx < len(pose_order):
            cv2.putText(frame, f"Next: {pose_order[current_pose_idx]}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Captured: {len(captured_poses)}/5", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    # Generate and save face encoding when done
    if len(captured_poses) == 5:
        face_encoding = generate_3d_face_encoding(captured_poses)
        cursor.execute("UPDATE attendance_employee SET face_encoding = ? WHERE emp_id = ?",
                      (face_encoding, emp_id))
        conn.commit()
        logging.info(f"Saved 3D encoding for employee ID: {emp_id}")
    
    conn.close()
    cap.release()

@app.route('/check_capture_status')
def check_capture_status():
    complete = len(captured_poses) >= 5
    return jsonify({'complete': complete})


@app.route('/add_faces', methods=['GET', 'POST'])
def add_faces():
    emp_id = request.args.get('emp_id')
    name = request.args.get('name')
    return Response(gen_frames2(emp_id, name), mimetype='multipart/x-mixed-replace; boundary=frame')

# def gen_frames2(emp_id, name):
#     global captured_poses, current_pose_idx

#     # Reset pose tracking for each new employee
#     captured_poses = {}  
#     current_pose_idx = 0  

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         yield b'Camera error'
#         return
    
#     conn = sqlite3.connect('attendance.db')
#     cursor = conn.cursor()
    
#     while True:
#         if len(captured_poses) >= 5:  
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.resize(frame, (640, 480))
#             cv2.putText(frame, "All poses captured Thank You.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             ret, buffer = cv2.imencode('.jpg', frame)
#             yield (b'--frame\r\n'
#                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             break
            
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, (640, 480))
#         faces = detector(frame, 0)
#         pose = "None"
        
#         if len(faces) == 1:
#             d = faces[0]
#             hh, ww = int((d.bottom() - d.top()) / 2), int((d.right() - d.left()) / 2)
#             if (d.right() + ww <= 640 and d.bottom() + hh <= 480 and d.left() - ww >= 0 and d.top() - hh >= 0):
#                 shape = predictor(frame, d)
#                 pose = estimate_head_pose(shape)
#                 logging.debug(f"Pose detected: {pose}")
                
#                 if current_pose_idx < len(pose_order):
#                     expected_pose = pose_order[current_pose_idx]
#                     logging.debug(f"Expected pose: {expected_pose}, pose in captured_poses: {pose in captured_poses}")

#                     if pose == expected_pose and pose not in captured_poses:
#                         face_roi = frame[d.top() - hh:d.bottom() + hh, d.left() - ww:d.right() + ww]
#                         if is_image_clear(face_roi):
#                             if not os.path.isdir(path_photos_from_camera):
#                                 os.makedirs(path_photos_from_camera)
                                
#                             current_face_dir = f"{path_photos_from_camera}{emp_id}_{name}"
#                             image_path = f"{current_face_dir}/{emp_id}_{pose}.jpg"
                            
#                             if os.path.exists(image_path):
#                                 os.remove(image_path)
#                                 logging.info(f"Old image {image_path} removed")
                            
#                             cv2.imwrite(image_path, face_roi)
#                             captured_poses[pose] = image_path
#                             logging.info(f"Captured {pose} at {image_path}")
#                             current_pose_idx += 1
                            
#                             column_name = f"{pose}_image"
#                             cursor.execute(f"UPDATE attendance_employee SET {column_name} = ? WHERE emp_id = ?",
#                                          (image_path, emp_id))
#                             conn.commit()
#                         else:
#                             logging.debug(f"Image for {pose} is blurry, skipping")

#             cv2.rectangle(frame, (d.left() - ww, d.top() - hh), (d.right() + ww, d.bottom() + hh), (255, 255, 255), 2)
        
#         cv2.putText(frame, f"Pose: {pose}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         if current_pose_idx < len(pose_order):
#             cv2.putText(frame, f"Next: {pose_order[current_pose_idx]}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#         cv2.putText(frame, f"Captured: {len(captured_poses)}/5", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
#     if len(captured_poses) == 5:
#         face_encoding = generate_3d_face_encoding(captured_poses)
#         cursor.execute("UPDATE attendance_employee SET face_encoding = ? WHERE emp_id = ?",
#                       (face_encoding, emp_id))
#         conn.commit()
#         logging.info(f"Saved 3D encoding for employee ID: {emp_id}")
    
#     conn.close()
#     cap.release()

def gen_frames2(emp_id, name):
    global captured_poses, current_pose_idx

    captured_poses = {}  
    current_pose_idx = 0  

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        yield b'Camera error'
        return
    
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    while True:
        if len(captured_poses) >= 5:  
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            cv2.putText(frame, "All poses captured. Thank You!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Send a signal to frontend to stop streaming and redirect
            yield b"--frame\r\nContent-Type: text/plain\r\n\r\nREDIRECT\r\n"
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        faces = detector(frame, 0)
        pose = "None"
        
        if len(faces) == 1:
            d = faces[0]
            hh, ww = int((d.bottom() - d.top()) / 2), int((d.right() - d.left()) / 2)
            if (d.right() + ww <= 640 and d.bottom() + hh <= 480 and d.left() - ww >= 0 and d.top() - hh >= 0):
                shape = predictor(frame, d)
                pose = estimate_head_pose(shape)
                logging.debug(f"Pose detected: {pose}")
                
                if current_pose_idx < len(pose_order):
                    expected_pose = pose_order[current_pose_idx]

                    if pose == expected_pose and pose not in captured_poses:
                        face_roi = frame[d.top() - hh:d.bottom() + hh, d.left() - ww:d.right() + ww]
                        if is_image_clear(face_roi):
                            if not os.path.isdir(path_photos_from_camera):
                                os.makedirs(path_photos_from_camera)
                                
                            current_face_dir = f"{path_photos_from_camera}{emp_id}_{name}"
                            image_path = f"{current_face_dir}/{emp_id}_{pose}.jpg"
                            
                            if os.path.exists(image_path):
                                os.remove(image_path)
                            
                            cv2.imwrite(image_path, face_roi)
                            captured_poses[pose] = image_path
                            current_pose_idx += 1
                            
                            column_name = f"{pose}_image"
                            cursor.execute(f"UPDATE attendance_employee SET {column_name} = ? WHERE emp_id = ?",
                                         (image_path, emp_id))
                            conn.commit()

            cv2.rectangle(frame, (d.left() - ww, d.top() - hh), (d.right() + ww, d.bottom() + hh), (255, 255, 255), 2)
        
        cv2.putText(frame, f"Pose: {pose}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if current_pose_idx < len(pose_order):
            cv2.putText(frame, f"Next: {pose_order[current_pose_idx]}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Captured: {len(captured_poses)}/5", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    if len(captured_poses) == 5:
        face_encoding = generate_3d_face_encoding(captured_poses)
        cursor.execute("UPDATE attendance_employee SET face_encoding = ? WHERE emp_id = ?",
                      (face_encoding, emp_id))
        conn.commit()
        logging.info(f"Saved 3D encoding for employee ID: {emp_id}")
    
    conn.close()
    cap.release()

@app.route('/manual_capture', methods=['POST'])
def manual_capture():
    global captured_poses, manual_capture, current_pose_idx, cap
    emp_id = request.form.get('emp_id', 'EMP')
    if not manual_capture:
        return jsonify({'status': 'error', 'message': 'Not in manual mode'}), 400
    if len(captured_poses) >= 5:
        return jsonify({'status': 'error', 'message': 'All poses already captured'}), 400
    
    if cap is None or not cap.isOpened():
        if not init_camera():
            return jsonify({'status': 'error', 'message': 'Camera not available'}), 500
    
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to grab frame for manual capture, reinitializing")
        init_camera()
        return jsonify({'status': 'error', 'message': 'Failed to capture frame, retrying'}), 500
    
    frame = cv2.resize(frame, (640, 480))
    faces = detector(frame, 0)
    logging.debug(f"Manual capture: Detected {len(faces)} faces")
    
    if len(faces) != 1:
        return jsonify({'status': 'error', 'message': f'Detected {len(faces)} faces, expected 1'}), 400
    
    d = faces[0]
    hh, ww = int((d.bottom() - d.top()) / 2), int((d.right() - d.left()) / 2)
    if not (d.right() + ww <= 640 and d.bottom() + hh <= 480 and d.left() - ww >= 0 and d.top() - hh >= 0):
        return jsonify({'status': 'error', 'message': 'Face out of bounds'}), 400
    
    shape = predictor(frame, d)
    pose = estimate_head_pose(shape)
    expected_pose = pose_order[current_pose_idx]
    logging.debug(f"Manual capture: Detected pose {pose}, Expected pose {expected_pose}, Captured poses: {list(captured_poses.keys())}")
    
    if pose != expected_pose:
        return jsonify({'status': 'warning', 'message': f'Wrong pose, expected {expected_pose}, got {pose}'}), 400
    
    if pose in captured_poses:
        return jsonify({'status': 'warning', 'message': f'{pose} already captured'}), 400
    
    face_roi = frame[d.top() - hh:d.bottom() + hh, d.left() - ww:d.right() + ww]
    if not is_image_clear(face_roi, threshold=30):
        return jsonify({'status': 'warning', 'message': 'Image too blurry, try again'}), 400
    
    image_path = f"{current_face_dir}/{emp_id}_{pose}.jpg"
    cv2.imwrite(image_path, face_roi)
    captured_poses[pose] = image_path
    logging.info(f"Manual capture: Captured {pose} at {image_path}")
    
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute(f"UPDATE attendance_employee SET {pose}_image = ? WHERE emp_id = ?", (image_path, emp_id))
    conn.commit()
    
    if len(captured_poses) == 5:
        face_encoding = generate_3d_face_encoding(captured_poses)
        cursor.execute("UPDATE attendance_employee SET face_encoding = ? WHERE emp_id = ?", (face_encoding, emp_id))
        conn.commit()
        logging.info(f"Saved 3D encoding for employee ID: {emp_id}")
    
    current_pose_idx += 1
    conn.close()
    
    return jsonify({'status': 'success', 'message': f'Captured {pose}', 'pose': pose, 'captured_count': len(captured_poses)})

@app.route('/employee_list', methods=['GET', 'POST'])
def employee_list():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    message = None
    
    if request.method == 'POST':
        if 'update_encodings' in request.form:
            selected_emp_ids = request.form.getlist('emp_ids')
            if selected_emp_ids:
                cursor.execute("SELECT emp_id, front_image, up_image, down_image, left_image, right_image FROM attendance_employee WHERE emp_id IN ({})".format(
                    ','.join('?' for _ in selected_emp_ids)), selected_emp_ids)
                employee_images = cursor.fetchall()
                
                updated_count = 0
                for emp_id, front, up, down, left, right in employee_images:
                    images = {'front': front, 'up': up, 'down': down, 'left': left, 'right': right}
                    face_encoding = generate_3d_face_encoding(images)
                    if face_encoding:
                        cursor.execute("UPDATE attendance_employee SET face_encoding = ? WHERE emp_id = ?", (face_encoding, emp_id))
                        updated_count += 1
                conn.commit()
                message = f"Face encodings updated for {updated_count} employees"
            else:
                message = "No employees selected for encoding update"
        
        elif 'delete_employees' in request.form:
            selected_emp_ids = request.form.getlist('emp_ids')
            if selected_emp_ids:
                # First get all the image paths for the selected employees
                cursor.execute("""
                    SELECT emp_id, name, front_image, up_image, down_image, left_image, right_image 
                    FROM attendance_employee 
                    WHERE emp_id IN ({})
                """.format(','.join('?' for _ in selected_emp_ids)), selected_emp_ids)
                employees_to_delete = cursor.fetchall()
                
                deleted_count = 0
                for emp in employees_to_delete:
                    emp_id, name = emp[0], emp[1]
                    image_paths = emp[2:7]  # All image paths
                    
                    # Delete all image files
                    for path in image_paths:
                        if path and os.path.exists(path):
                            try:
                                os.remove(path)
                                logging.info(f"Deleted file: {path}")
                            except Exception as e:
                                logging.error(f"Error deleting file {path}: {e}")
                    
                    # Delete the employee directory
                    emp_dir = f"{path_photos_from_camera}{emp_id}_{name}"
                    if os.path.exists(emp_dir):
                        try:
                            import shutil
                            shutil.rmtree(emp_dir)
                            logging.info(f"Deleted directory: {emp_dir}")
                        except Exception as e:
                            logging.error(f"Error deleting directory {emp_dir}: {e}")
                    
                    # Delete from database
                    cursor.execute("DELETE FROM attendance_employee WHERE emp_id = ?", (emp_id,))
                    # Also delete from attendance table
                    cursor.execute("DELETE FROM attendance WHERE emp_id = ?", (emp_id,))
                    deleted_count += 1
                
                conn.commit()
                message = f"Successfully deleted {deleted_count} employees and their data"
            else:
                message = "No employees selected for deletion"
        
        elif 'edit_employee' in request.form:
            old_emp_id = request.form.get('emp_id')
            name = request.form.get('name')
            new_emp_id = request.form.get('new_emp_id')
            new_image = request.files.get('new_image')
            
            # First, get the current employee data including image paths
            cursor.execute("SELECT name, front_image, up_image, down_image, left_image, right_image FROM attendance_employee WHERE emp_id = ?", 
                          (old_emp_id,))
            employee_data = cursor.fetchone()
            
            if employee_data:
                old_name = employee_data[0]
                image_paths = {
                    'front_image': employee_data[1],
                    'up_image': employee_data[2],
                    'down_image': employee_data[3],
                    'left_image': employee_data[4],
                    'right_image': employee_data[5]
                }
                
                # Update image file paths if emp_id changed
                if old_emp_id != new_emp_id:
                    import shutil
                    
                    # Get the old directory path
                    old_dir = f"{path_photos_from_camera}{old_emp_id}_{old_name}"
                    
                    # Create new directory for the updated employee ID if it doesn't exist
                    new_dir = f"{path_photos_from_camera}{new_emp_id}_{name}"
                    os.makedirs(new_dir, exist_ok=True)
                    
                    # Update each image path and move the files
                    updated_paths = {}
                    for pose, old_path in image_paths.items():
                        if old_path:  # Only process if there's an actual path
                            # Extract pose from the filename (e.g., "001_front.jpg" -> "front")
                            pose_name = pose.split('_')[0]  # This gets "front" from "front_image"
                            
                            # Create new path
                            new_path = f"{new_dir}/{new_emp_id}_{pose_name}.jpg"
                            
                            # Move the file if it exists
                            if os.path.exists(old_path):
                                try:
                                    # Copy the file to new location
                                    shutil.copy2(old_path, new_path)
                                    logging.info(f"Copied image from {old_path} to {new_path}")
                                    updated_paths[pose] = new_path
                                except Exception as e:
                                    logging.error(f"Error copying file: {e}")
                                    updated_paths[pose] = old_path  # Keep old path if copy fails
                            else:
                                updated_paths[pose] = old_path  # Keep old path if file doesn't exist
                        else:
                            updated_paths[pose] = None
                    
                    # Update database with new paths
                    cursor.execute("""
                        UPDATE attendance_employee 
                        SET name = ?, emp_id = ?, front_image = ?, up_image = ?, down_image = ?, left_image = ?, right_image = ? 
                        WHERE emp_id = ?
                    """, (name, new_emp_id, updated_paths['front_image'], updated_paths['up_image'], 
                          updated_paths['down_image'], updated_paths['left_image'], updated_paths['right_image'], old_emp_id))
                    
                    # Delete old files after successful database update
                    for old_path in image_paths.values():
                        if old_path and os.path.exists(old_path):
                            try:
                                os.remove(old_path)
                                logging.info(f"Deleted old file: {old_path}")
                            except Exception as e:
                                logging.error(f"Error deleting file {old_path}: {e}")
                    
                    # Delete the old directory if it exists
                    if os.path.exists(old_dir):
                        try:
                            # Try to remove the directory and its contents
                            shutil.rmtree(old_dir)
                            logging.info(f"Deleted old directory and contents: {old_dir}")
                        except Exception as e:
                            logging.error(f"Error deleting directory {old_dir}: {e}")
                else:
                    # If emp_id didn't change, just update the name
                    cursor.execute("UPDATE attendance_employee SET name = ? WHERE emp_id = ?", (name, old_emp_id))
            
            # Handle new image upload if provided
            if new_image:
                new_image_path = f"{path_photos_from_camera}{new_emp_id}_edited_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                new_image.save(new_image_path)
                img = cv2.imread(new_image_path)
                faces = detector(img, 0)
                face_encoding_bytes = None
                if len(faces) == 1:
                    shape = predictor(img, faces[0])
                    face_encoding = face_reco_model.compute_face_descriptor(img, shape)
                    face_encoding_bytes = np.array(face_encoding).tobytes()
                cursor.execute("UPDATE attendance_employee SET face_encoding = ? WHERE emp_id = ?", 
                              (face_encoding_bytes, new_emp_id))
            
            # Also update the attendance table to maintain referential integrity
            cursor.execute("UPDATE attendance SET emp_id = ?, name = ? WHERE emp_id = ?", 
                          (new_emp_id, name, old_emp_id))
            
            conn.commit()
            message = f"Employee {old_emp_id} updated to {new_emp_id}"
    
    # employee data for list
    cursor.execute("SELECT id, name, emp_id, front_image, up_image, down_image, left_image, right_image FROM attendance_employee GROUP BY emp_id")
    employees = cursor.fetchall()
    conn.close()
    return render_template('employee_list.html', employees=employees, message=message)

@app.route('/images/<path:image_path>')
def serve_image(image_path):
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/take_attendance')
def take_attendance():
    return render_template('take_attendance.html')

@app.route('/start_attendance')
def start_attendance():
    return render_template('take_attendance.html', message="", video_feed=True)

# def gen_attendance_frames():
#     conn = sqlite3.connect('attendance.db')
#     cursor = conn.cursor()
#     cursor.execute("SELECT emp_id, name, face_encoding FROM attendance_employee WHERE face_encoding IS NOT NULL")
#     known_faces = cursor.fetchall()
#     known_emp_ids = [emp_id for emp_id, _, _ in known_faces]
#     known_names = [name for _, name, _ in known_faces]
#     known_encodings = [np.frombuffer(encoding, dtype=np.float64) for _, _, encoding in known_faces]
        
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         logging.error("Cannot open camera for attendance")
#         yield b'Camera error'
#         return
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logging.error("Failed to grab frame for attendance")
#             break
#         frame = cv2.resize(frame, (640, 480))
#         faces = detector(frame, 0)
#         for i, d in enumerate(faces):
#             shape = predictor(frame, d)
#             feature = face_reco_model.compute_face_descriptor(frame, shape)
#             e_distances = [np.linalg.norm(feature - known_encoding) for known_encoding in known_encodings]
#             if e_distances:
#                 min_distance_idx = np.argmin(e_distances)
#                 if e_distances[min_distance_idx] < 0.4:  # Adjust the threshold as needed
#                     emp_id = known_emp_ids[min_distance_idx]
#                     name = known_names[min_distance_idx]
#                     current_date = datetime.now().strftime('%Y-%m-%d')
                    
#                     cursor.execute("SELECT COUNT(*) FROM attendance WHERE emp_id = ? AND date = ?", (emp_id, current_date))
#                     if cursor.fetchone()[0] == 0:
#                         date_dir = f"{attendance_path}{current_date}/"
#                         os.makedirs(date_dir, exist_ok=True)
#                         image_path = f"{date_dir}{emp_id}_{datetime.now().strftime('%H%M%S')}.jpg"
#                         # cv2.imwrite(image_path, frame) # Save the image with full resolution

#                         # Only save the face detection box only
#                         # Extract face region with some margin
#                         top, right, bottom, left = d.top(), d.right(), d.bottom(), d.left()
#                         # Add some margin to the face region (20% of face size)
#                         height, width = bottom - top, right - left
#                         margin_h, margin_w = int(height * 0.2), int(width * 0.2)
                        
#                         # Ensure we don't go out of frame bounds
#                         top_with_margin = max(0, top - margin_h)
#                         bottom_with_margin = min(frame.shape[0], bottom + margin_h)
#                         left_with_margin = max(0, left - margin_w)
#                         right_with_margin = min(frame.shape[1], right + margin_w)
                        
#                         # Extract the face region with margin
#                         face_image = frame[top_with_margin:bottom_with_margin, 
#                                           left_with_margin:right_with_margin]
                        
#                         # Save only the face region instead of the full frame
#                         cv2.imwrite(image_path, face_image)
                                                
#                         current_time = datetime.now().strftime('%H:%M:%S')
#                         cursor.execute("INSERT INTO attendance (emp_id, name, time, date, path) VALUES (?, ?, ?, ?, ?)",
#                                        (emp_id, name, current_time, current_date, image_path))
#                         conn.commit()
                    
#                     # Draw emp_id
#                     cv2.putText(frame, f"{emp_id}", (d.left() + 5, d.bottom() - 20), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (5, 214, 118), 2)

#                     # Draw name
#                     cv2.putText(frame, f"{name}", (d.left() + 5, d.bottom() - 5), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (5, 214, 118), 2)
                    
#             cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (5, 214, 118), 2)
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
#     conn.close()
#     cap.release()

def gen_attendance_frames():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT emp_id, name, face_encoding FROM attendance_employee WHERE face_encoding IS NOT NULL")
    known_faces = cursor.fetchall()
    known_emp_ids = [emp_id for emp_id, _, _ in known_faces]
    known_names = [name for _, name, _ in known_faces]
    known_encodings = [np.frombuffer(encoding, dtype=np.float64) for _, _, encoding in known_faces]
    conn.close()
    
    # Track which employees have been marked present in this session
    marked_today = set()
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera for attendance")
        yield b'Camera error'
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to grab frame for attendance")
            break
            
        frame = cv2.resize(frame, (640, 480))
        faces = detector(frame, 0)
        
        for i, d in enumerate(faces):
            # Draw rectangle around every detected face
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (5, 214, 118), 2)
            
            # Get face features
            shape = predictor(frame, d)
            feature = face_reco_model.compute_face_descriptor(frame, shape)
            
            # Compare with known faces
            e_distances = [np.linalg.norm(feature - known_encoding) for known_encoding in known_encodings]
            if e_distances:
                min_distance_idx = np.argmin(e_distances)
                min_distance = e_distances[min_distance_idx]
                
                if min_distance < 0.4:  # Recognition threshold
                    emp_id = known_emp_ids[min_distance_idx]
                    name = known_names[min_distance_idx]
                    
                    # Display recognition info
                    cv2.putText(frame, f"{emp_id}", (d.left() + 5, d.bottom() - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (5, 214, 118), 2)
                    cv2.putText(frame, f"{name}", (d.left() + 5, d.bottom() - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (5, 214, 118), 2)
                    cv2.putText(frame, f"Match: {min_distance:.2f}", (d.left() + 5, d.top() - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (5, 214, 118), 2)
                    
                    # Mark attendance if not already marked today
                    if emp_id not in marked_today:
                        attendance_marked = mark_attendance(frame, d, emp_id, name)
                        if attendance_marked:
                            marked_today.add(emp_id)
                            # Show confirmation message on frame
                            cv2.putText(frame, "Attendance Marked!", (d.left(), d.top() - 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Unknown face
                    cv2.putText(frame, "Unknown", (d.left() + 5, d.bottom() - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Convert frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

def mark_attendance(frame, face_location, emp_id, name):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Check if attendance already marked today
    cursor.execute("SELECT COUNT(*) FROM attendance WHERE emp_id = ? AND date = ?", 
                  (emp_id, current_date))
    if cursor.fetchone()[0] > 0:
        conn.close()
        return False  # Already marked attendance today
    
    # Create directory for today's attendance if it doesn't exist
    date_dir = f"{attendance_path}{current_date}/"
    os.makedirs(date_dir, exist_ok=True)
    
    # Generate filename with timestamp
    image_path = f"{date_dir}{emp_id}_{datetime.now().strftime('%H%M%S')}.jpg"
    
    # Extract face region with margin
    d = face_location
    top, right, bottom, left = d.top(), d.right(), d.bottom(), d.left()
    
    # Add margin (20% of face size)
    height, width = bottom - top, right - left
    margin_h, margin_w = int(height * 0.2), int(width * 0.2)
    
    # Ensure we don't go out of frame bounds
    top_with_margin = max(0, top - margin_h)
    bottom_with_margin = min(frame.shape[0], bottom + margin_h)
    left_with_margin = max(0, left - margin_w)
    right_with_margin = min(frame.shape[1], right + margin_w)
    
    # Extract and save only the face region
    face_image = frame[top_with_margin:bottom_with_margin, 
                      left_with_margin:right_with_margin]
    cv2.imwrite(image_path, face_image)
    
    # Record attendance in database
    current_time = datetime.now().strftime('%H:%M:%S')
    cursor.execute("INSERT INTO attendance (emp_id, name, time, date, path) VALUES (?, ?, ?, ?, ?)",
                  (emp_id, name, current_time, current_date, image_path))
    conn.commit()
    conn.close()
    
    logging.info(f"Marked attendance for {name} (ID: {emp_id}) at {current_time}")
    return True

@app.route('/attendance_feed')
def attendance_feed():
    return Response(gen_attendance_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)