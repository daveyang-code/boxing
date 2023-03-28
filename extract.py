import json
import cv2
import time
import os
import mediapipe as mp
import numpy as np

def minFrames(L):
    return min([y - x for x, y in zip(L, L[1:])])


def exportClips():

    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for i in range(len(startFrames)):

        vid_name = os.path.join(path, "clips", str(i) + ".mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, startFrames[i] - start_up)
        out = cv2.VideoWriter(vid_name, fourcc, FPS, frame_size)

        for j in range(0, minFrames(startFrames) + recovery):
            res, frame = cap.read()
            if not res:
                break
            out.write(frame)
        out.release()

    cap.release()


def viewClips():

    for i in range(len(startFrames)):

        cap.set(cv2.CAP_PROP_POS_FRAMES, startFrames[i] - start_up)

        for j in range(0, minFrames(startFrames) + recovery):

            cap.set(cv2.CAP_PROP_POS_FRAMES, startFrames[i] + j)
            res, frame = cap.read()

            cv2.putText(
                frame, punchType[i] + str(i), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255
            )
            cv2.imshow("", frame)
            # time.sleep(.8)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def extractKeypoints(results):
    return (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )


def exportKeypoints():

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:

        for i in range(len(startFrames)):

            cap.set(cv2.CAP_PROP_POS_FRAMES, startFrames[i] - start_up)

            for j in range(0, minFrames(startFrames) + recovery):

                success, image = cap.read()

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                directory = os.path.join(path, "keypoints", punchType[i], str(i))

                if not os.path.exists(directory):
                    os.makedirs(directory)

                np.save(
                    os.path.join(directory, str(j)),
                    extractKeypoints(results),
                )



f = open("segmentation_results.json")
punches = json.load(f)
FPS = 30
startFrames = []
punchType = []
start_up = 1
recovery = 10
path = "/home/david/Documents/Projects/boxing"

for i in punches:
    startFrames.append(int(punches[i]["startTime"] * FPS) - start_up)
    punchType.append(punches[i]["labelText"])

cap = cv2.VideoCapture("punches.mp4")

# viewClips()
# exportClips()
exportKeypoints()

f.close()
cap.release()
cv2.destroyAllWindows()
