# Imports :
import cv2


def save_webcam(outpath, fps, mirror=False):
    # Capturing video from webcam:
    cap = cv2.VideoCapture(0)
    current_frame, f_frame = 0, None

    # Get current width of frame
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    # Define the codec and create VideoWriter object
    vw_fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(outpath, vw_fourcc, fps, (int(width), int(height)))

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        black_white = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        black_white = cv2.GaussianBlur(black_white, (21, 21), 0)

        if ret:
            if mirror:
                # Mirror the output video frame
                frame = cv2.flip(frame, 1)

            # Saves for video
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('Original Frame', frame)
        else:
            break

        if f_frame is None:
            f_frame = black_white
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):  # if 'q' is pressed then quit
            break

        # To stop duplicate images
        current_frame += 1

        f_delta = cv2.absdiff(f_frame, black_white)
        thresh = cv2.threshold(f_delta, 20, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=3)
        iterations = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        iterations = iterations[1]

        for it in iterations:
            if cv2.contourArea(it) < 5000:
                continue
            (x, y, w, h) = cv2.boundingRect(it)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Movements Frame", frame)
        cv2.imshow("Contrasts Frame (black and white)", thresh)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            f_frame = black_white

        if key == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    save_webcam('output.avi', 30.0, mirror=True)


if __name__ == '__main__':
    main()
