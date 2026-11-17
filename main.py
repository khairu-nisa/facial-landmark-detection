import argparse
import sys

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def draw_landmarks(frame, results):
    if not results.multi_face_landmarks:
        return frame
    for face in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=frame, landmark_list=face,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=frame, landmark_list=face,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=frame, landmark_list=face,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style())
    return frame


def run_on_image(path):
    image = cv2.imread(path)
    if image is None:
        print(f"Could not read image: {path}")
        sys.exit(1)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=5,
                               refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
    print(f"Faces detected: {faces}")

    output = draw_landmarks(image, results)
    cv2.imwrite("output.jpg", output)
    print("Saved result to output.jpg")
    cv2.imshow("Facial Landmarks - press any key to close", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2,
                               refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)

            frame = draw_landmarks(frame, results)
            count = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
            cv2.putText(frame, f"Faces: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Facial Landmark Detection - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Facial landmark detection")
    parser.add_argument("--image", help="Path to an image (uses webcam if omitted)")
    args = parser.parse_args()
    if args.image:
        run_on_image(args.image)
    else:
        run_on_webcam()


if __name__ == "__main__":
    main()
