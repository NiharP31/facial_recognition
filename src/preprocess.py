import os
import cv2
import face_recognition
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def preprocess_image(image_path, output_path, target_size = (224, 224)):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        print(f"No face detected in {image_path}")
        return
    
    top, right, bottom, left = face_locations[0]
    face_landmarks = face_recognition.face_landmarks(image, face_locations=face_locations)[0]

    left_eye = np.mean(face_landmarks['left_eye'], axis=0)
    right_eye = np.mean(face_landmarks['right_eye'], axis=0)

    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    face_center = ((left + right) // 2, (top + bottom) // 2)

    M = cv2.getRotationMatrix2D(face_center, angle, 1.0)
    aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    face = aligned_face[top:bottom, left:right]

    face_image = Image.fromarray(face)
    face_image = face_image.resize(target_size, Image.LANCZOS)

    face_image.save(output_path)

def preprocess_dataset(input_folder, output_folder, target_size = (224, 224)):
    os.makedirs(output_folder, exist_ok=True)

    input_paths = []
    output_paths = []

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                input_paths.append(input_path)
                output_paths.append(output_path)   

    num_cores = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers = num_cores) as executor:
        executor.map(preprocess_image, input_paths, output_paths, [target_size]*len(input_paths))

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_folder = os.path.join(project_root, "data", "processed", "aligned_faces")
    output_folder = os.path.join(project_root, "data", "processed", "preprocessed_faces")

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    if not os.path.exists(input_folder):
        print(f"Input folder does not exist: {input_folder}")
    else:
        file_count = sum(len(files) for _, _, files in os.walk(input_folder))
        print(f"Number of files in input folder: {file_count}")

    preprocess_dataset(input_folder, output_folder)
    print("Preprocessing done!")