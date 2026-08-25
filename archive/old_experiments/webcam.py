import time

import cv2


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    previous_time = time.perf_counter()

    print("DepthFX Webcam Pipeline")
    print("Press Q to quit.")

    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to read frame from webcam.")
            break

        current_time = time.perf_counter()
        elapsed = current_time - previous_time
        previous_time = current_time

        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("DepthFX - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()