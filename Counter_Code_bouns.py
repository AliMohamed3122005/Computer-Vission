import cv2
import numpy as np


class ShapeDetector:

    def __init__(self, video_path):

        self.cap = cv2.VideoCapture(video_path)

        # Counters
        self.red_count = 0
        self.blue_count = 0

        # Object tracks
        self.red_tracks = {}
        self.blue_tracks = {}

        # Next IDs
        self.next_red_id = 0
        self.next_blue_id = 0

        # Tracking parameters
        self.max_distance = 80
        self.max_missing = 5
        self.min_hits = 3


    def update_tracks(self, current_centers, tracks, next_id, count):

        matched_tracks = set()
        matched_centers = set()

        # =========================
        # Build all possible matches
        # =========================

        possible_matches = []

        for track_id, track_data in tracks.items():

            old_cx, old_cy = track_data["center"]

            for center_index, center in enumerate(current_centers):

                cx, cy = center

                distance = np.sqrt(
                    (cx - old_cx) ** 2 +
                    (cy - old_cy) ** 2
                )

                if distance < self.max_distance:

                    possible_matches.append(
                        (
                            distance,
                            track_id,
                            center_index
                        )
                    )

        # Closest matches first
        possible_matches.sort(
            key=lambda x: x[0]
        )

        # =========================
        # Match old tracks
        # =========================

        for distance, track_id, center_index in possible_matches:

            if track_id in matched_tracks:
                continue

            if center_index in matched_centers:
                continue

            # Match found
            tracks[track_id]["center"] = current_centers[center_index]

            tracks[track_id]["missing"] = 0

            tracks[track_id]["hits"] += 1

            matched_tracks.add(track_id)
            matched_centers.add(center_index)

        # =========================
        # Update unmatched tracks
        # =========================

        tracks_to_remove = []

        for track_id in tracks:

            if track_id not in matched_tracks:

                tracks[track_id]["missing"] += 1

                if tracks[track_id]["missing"] > self.max_missing:

                    tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:

            del tracks[track_id]

        # =========================
        # Create new tracks
        # =========================

        for center_index, center in enumerate(current_centers):

            if center_index in matched_centers:
                continue

            tracks[next_id] = {

                "center": center,

                "missing": 0,

                "hits": 1,

                "counted": False
            }

            next_id += 1

        # =========================
        # Count confirmed objects
        # =========================

        for track_id in tracks:

            track = tracks[track_id]

            if (
                track["hits"] >= self.min_hits
                and not track["counted"]
            ):

                count += 1

                track["counted"] = True

        return tracks, next_id, count


    def hsv_frame(self, frame):

        # =========================
        # Reduce video noise
        # =========================

        frame = cv2.medianBlur(
            frame,
            5
        )

        # =========================
        # BGR -> HSV
        # =========================

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        # =========================
        # Red mask
        # =========================

        lower_red1 = np.array(
            [0, 80, 80]
        )

        upper_red1 = np.array(
            [10, 255, 255]
        )

        lower_red2 = np.array(
            [170, 80, 80]
        )

        upper_red2 = np.array(
            [179, 255, 255]
        )

        red_mask1 = cv2.inRange(
            hsv,
            lower_red1,
            upper_red1
        )

        red_mask2 = cv2.inRange(
            hsv,
            lower_red2,
            upper_red2
        )

        red_mask = red_mask1 | red_mask2

        # =========================
        # Blue mask
        # =========================

        lower_blue = np.array(
            [90, 100, 100]
        )

        upper_blue = np.array(
            [130, 255, 255]
        )

        blue_mask = cv2.inRange(
            hsv,
            lower_blue,
            upper_blue
        )

        # =========================
        # Morphological operations
        # =========================

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        blue_mask = cv2.morphologyEx(
            blue_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        blue_mask = cv2.morphologyEx(
            blue_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # =========================
        # Find contours
        # =========================

        red_contours, _ = cv2.findContours(
            red_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        blue_contours, _ = cv2.findContours(
            blue_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # =========================
        # Filter red contours
        # =========================

        red_filtered = []

        for c in red_contours:

            if cv2.contourArea(c) > 100:

                red_filtered.append(c)

        red_contours = red_filtered

        # =========================
        # Filter blue contours
        # =========================

        blue_filtered = []

        for c in blue_contours:

            if cv2.contourArea(c) > 100:

                blue_filtered.append(c)

        blue_contours = blue_filtered

        # =========================
        # Current centers
        # =========================

        current_red_centers = []

        current_blue_centers = []

        # =========================
        # Red circles
        # =========================

        for c in red_contours:

            area = cv2.contourArea(c)

            peri = cv2.arcLength(
                c,
                True
            )

            if peri == 0:
                continue

            circularity = (
                4 * np.pi * area
                / (peri ** 2)
            )

            if circularity > 0.85:

                M = cv2.moments(c)

                if M["m00"] == 0:
                    continue

                cx = int(
                    M["m10"] / M["m00"]
                )

                cy = int(
                    M["m01"] / M["m00"]
                )

                current_red_centers.append(
                    (cx, cy)
                )

                # Draw contour
                cv2.drawContours(
                    frame,
                    [c],
                    -1,
                    (0, 255, 0),
                    2
                )

                x, y, w, h = cv2.boundingRect(c)

                cv2.putText(
                    frame,
                    "Red Circle",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # =========================
        # Blue squares
        # =========================

        for c in blue_contours:

            area = cv2.contourArea(c)

            peri = cv2.arcLength(
                c,
                True
            )

            if peri == 0:
                continue

            approx = cv2.approxPolyDP(
                c,
                0.04 * peri,
                True
            )

            if len(approx) == 4:

                x, y, w, h = cv2.boundingRect(c)

                ratio = (
                    max(w, h)
                    / min(w, h)
                )

                if ratio <= 1.2:

                    M = cv2.moments(c)

                    if M["m00"] == 0:
                        continue

                    cx = int(
                        M["m10"] / M["m00"]
                    )

                    cy = int(
                        M["m01"] / M["m00"]
                    )

                    current_blue_centers.append(
                        (cx, cy)
                    )

                    # Draw contour
                    cv2.drawContours(
                        frame,
                        [c],
                        -1,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Blue Square",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        # =========================
        # Update Red tracking
        # =========================

        (
            self.red_tracks,
            self.next_red_id,
            self.red_count
        ) = self.update_tracks(
            current_red_centers,
            self.red_tracks,
            self.next_red_id,
            self.red_count
        )

        # =========================
        # Update Blue tracking
        # =========================

        (
            self.blue_tracks,
            self.next_blue_id,
            self.blue_count
        ) = self.update_tracks(
            current_blue_centers,
            self.blue_tracks,
            self.next_blue_id,
            self.blue_count
        )

        # =========================
        # Display counters
        # =========================

        cv2.putText(
            frame,
            f"Red Circles: {self.red_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Blue Squares: {self.blue_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        return frame


    def run(self):

        # =========================
        # Get video information
        # =========================

        width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps == 0:
            fps = 30

        # =========================
        # Create output video
        # =========================

        fourcc = cv2.VideoWriter_fourcc(
            *"XVID"
        )

        out = cv2.VideoWriter(
            "output.avi",
            fourcc,
            fps,
            (width, height)
        )

        # =========================
        # Read video frame by frame
        # =========================

        while True:

            ret, frame = self.cap.read()

            if not ret:
                break

            frame = self.hsv_frame(frame)

            # Save every processed frame
            out.write(frame)

            # Show frame
            cv2.imshow(
                "Shape Detection",
                frame
            )

            if cv2.waitKey(30) == 27:
                break

        # =========================
        # Release
        # =========================

        self.cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    detector = ShapeDetector(
        "video.mp4"
    )

    detector.run()