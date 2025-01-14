from flask import Flask, render_template, request, Response, redirect
import numpy as np
import cv2
import threading
import pyttsx3
from ultralytics import YOLO
from PIL import Image
import time
import threading
import mediapipe as mp

app = Flask(__name__)


#MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Dictionary to track bounding box sizes over time
bbox_history = {}

MODEL_FILE="yolov8n\yolov8n.pt"
COCO_FILE="yolov8n\coco.txt"
video = cv2.VideoCapture('')

GUN_MODEL_FILE="static/models/gun.pt"
GUN_COCO_FILE="static/models/gun.txt"

MONEY_MODEL_FILE="static/models/best.pt"
MONEY_COCO_FILE="static/models/money.txt"


@app.route('/')
def index():
     return render_template('index.html')


VIDEO_EXTENSIONS = ['mp4']
PHOTO_EXTENSIONS = ['png', 'jpeg', 'jpg']


def fextension(filename):
    return filename.rsplit('.', 1)[1].lower()


@app.route('/upload', methods=['POST'])
def upload():
    global video
    if 'video' not in request.files:
        return 'No video file found'
    file = request.files['video']
    if file.filename == '':
        return 'No video selected'
    if file:
        exttype = fextension(file.filename)
        print(exttype)
        if exttype in VIDEO_EXTENSIONS:
            file.save('static/input/video/' + file.filename)
            print('video')
            video = cv2.VideoCapture('static/input/video/' + file.filename)
            return redirect('/video_feed_new')
        elif exttype in PHOTO_EXTENSIONS:
            file.save('static/input/photo/' + file.filename)
            print('photo')
            return render_template('preview_photo.html', file_name=file.filename, type='image/'+exttype)
    return 'Invalid video file'



@app.route('/upload_pose', methods=['POST'])
def upload_pose():
    global video
    if 'video' not in request.files:
        return 'No video file found'
    file = request.files['video']
    if file.filename == '':
        return 'No video selected'
    if file:
        exttype = fextension(file.filename)
        print(exttype)
        if exttype in VIDEO_EXTENSIONS:
            file.save('static/input/video/' + file.filename)
            print('video')
            video = cv2.VideoCapture('static/input/video/' + file.filename)
            return redirect('/video_feed_pose')
        elif exttype in PHOTO_EXTENSIONS:
            file.save('static/input/photo/' + file.filename)
            print('photo')
            return render_template('preview_photo.html', file_name=file.filename, type='image/'+exttype)
    return 'Invalid video file'

@app.route('/upload_money', methods=['POST'])
def upload_money():
    global video
    if 'video' not in request.files:
        return 'No video file found'
    file = request.files['video']
    if file.filename == '':
        return 'No video selected'
    if file:
        exttype = fextension(file.filename)
        print(exttype)
        if exttype in VIDEO_EXTENSIONS:
            file.save('static/input/video/' + file.filename)
            print('video')
            video = cv2.VideoCapture('static/input/video/' + file.filename)
            return redirect('/video_feed_money')
        elif exttype in PHOTO_EXTENSIONS:
            file.save('static/input/photo/' + file.filename)
            print('photo')
            return render_template('preview_photo.html', file_name=file.filename, type='image/'+exttype)
    return 'Invalid video file'

@app.route('/upload_stair', methods=['POST'])
def upload_stair():
    global video
    if 'video' not in request.files:
        return 'No video file found'
    file = request.files['video']
    if file.filename == '':
        return 'No video selected'
    if file:
        exttype = fextension(file.filename)
        print(exttype)
        if exttype in VIDEO_EXTENSIONS:
            file.save('static/input/video/' + file.filename)
            print('video')
            video = cv2.VideoCapture('static/input/video/' + file.filename)
            return redirect('/video_feed_stair')
        elif exttype in PHOTO_EXTENSIONS:
            file.save('static/input/photo/' + file.filename)
            print('photo')
            return render_template('preview_photo.html', file_name=file.filename, type='image/'+exttype)
    return 'Invalid video file'

class PrepareImage:
    """
    Class for preprocessing images with Gaussian blur, Canny edge detection,
    and segmentation to aid in lane detection.

    Attributes:
    - gauss_size (tuple): Kernel size for Gaussian blur (must be odd and > 1).
    - gauss_deviation (list-like): Standard deviations for Gaussian blur.
    - auto_canny (bool): If True, thresholds for Canny are calculated automatically.
    - canny_low (int): Lower threshold for Canny edge detection.
    - canny_high (int): Upper threshold for Canny edge detection.
    - segment_x (float): Fraction of image width for segmentation peak.
    - segment_y (float): Fraction of image height for segmentation peak.
    """

    def __init__(self,
                 gauss_size=(9, 9),
                 gauss_deviation=(3, 3),
                 auto_canny=False,
                 canny_low=50,
                 canny_high=175,
                 segment_x=0.5,
                 segment_y=0.5):
        # Validate Gaussian kernel size
        if len(gauss_size) != 2 or gauss_size[0] % 2 == 0 or gauss_size[1] % 2 == 0:
            raise ValueError("Gaussian kernel size must be a tuple of two odd numbers.")
        self.gauss_kernel = gauss_size

        # Validate Gaussian deviation
        if len(gauss_deviation) != 2:
            raise ValueError("Gaussian deviation must be a list-like object of size 2.")
        self.gauss_deviation = gauss_deviation

        # Validate Canny parameters
        if not isinstance(auto_canny, bool):
            raise TypeError("auto_canny must be a boolean.")
        self.auto_canny = auto_canny
        if not auto_canny:
            if not isinstance(canny_low, int) or not isinstance(canny_high, int):
                raise TypeError("Canny thresholds must be integers.")
        self.canny_low = canny_low
        self.canny_high = canny_high

        # Validate segmentation parameters
        if not (0 < segment_x < 1) or not (0 < segment_y < 1):
            raise ValueError("Segment fractions must be in the range (0, 1).")
        self.segment_x = segment_x
        self.segment_y = segment_y

    def do_canny(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, self.gauss_kernel, self.gauss_deviation[0])
        if self.auto_canny:
            v = np.median(blur)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            return cv2.Canny(blur, lower, upper)
        return cv2.Canny(blur, self.canny_low, self.canny_high)

    def segment_image(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        shift = int(0.08 * width)
        points = np.array([[
            (0, height), (width, height),
            (int(width * self.segment_x) + shift, int(height * self.segment_y)),
            (int(width * self.segment_x) - shift, int(height * self.segment_y))
        ]])
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, points, 255)
        return cv2.bitwise_and(frame, mask)

    def get_poly_maskpoints(self, frame: np.ndarray) -> tuple:
        height, width = frame.shape[:2]
        shift = int(0.08 * width)
        points = np.array([[
            (2 * shift, height), (width - 2 * shift, height),
            (int(width * self.segment_x) + 2 * shift, int(height * self.segment_y)),
            (int(width * self.segment_x) - 2 * shift, int(height * self.segment_y))
        ]])
        left = (points[0][0], points[0][3])
        right = (points[0][1], points[0][2])
        return left, right

    def get_binary_image(self, frame: np.ndarray) -> np.ndarray:
        canny = self.do_canny(frame)
        return self.segment_image(canny)



def speak(text, rate=1.2, voice_type="female", volume=1):
    def speak_voice():
        engine = pyttsx3.init()
        engine.setProperty('rate', rate*engine.getProperty('rate'))

        if (voice_type == "female"):
            engine.setProperty('voice', engine.getProperty('voices')[1].id)
        else:
            engine.setProperty('voice', engine.getProperty('voices')[0].id)

        engine.setProperty('volume', volume)
        engine.say(text)
        engine.runAndWait()

    try:
        start_voice = threading.Thread(target=speak_voice)
        start_voice.start()
    except:
        print("Warning")


class Curve():
    '''PARAMETERS: 
                     window_size -> float (0,1): how wide you want the 
                                    window to be 
      METHODS: // to-do
    '''

    def __init__(self, draw=False):
        self.non_zero = []
        self.prev_left = None
        self.prev_right = None
        self.draw = False

    def get_lane_points(self, frame, left, right, pxx=10, pxy=30):
        '''
        PARAMETERS: frame, left -> points of left boundary, right -> points of right boundary
                    pxx -> pixel size of x 
                    pxy -> pixel size in y
        RETURNS: points : a list of 2 tuples of proposed lane coords'''
        x_start = left[0][0]
        x_end = right[0][0]
        y_start = left[1][1]
        y_end = left[0][1]
        x = np.array([], dtype=np.uint32)
        y = np.array([], dtype=np.uint32)
        for i in range(y_start, y_end, pxy):
            for j in range(x_start, x_end, pxx):
                if ((pxx*pxy)/40 < np.count_nonzero(frame[i:i+pxx, j:j+pxy]) < (pxx*pxy)/15):
                    nz = np.nonzero(frame[i:i+pxx, j:j+pxy])
                    x = np.hstack((x, nz[0]+i))
                    y = np.hstack((y, nz[1]+j))

        return np.transpose((x, y))

    def detect_curve(self, img, x, y, left, right):
        ''' PARAMETRS: Frame, x-> X coordinates of white points , y-> Y coordinates of white points
                       left -> Points of left boundary
                       right -> Points of right boundary

            RETURNS: -> Image with the single curve traced
        '''
        img2 = np.zeros_like(img)

#         y = -y
        a, b, c = np.polyfit(x, y, 2)
        x_start = left[0][0]
        x_end = right[0][0]
        y_start = left[1][1]
        y_end = left[0][1]
        for i in range(min(x), max(x)):
            y_ = int(a*i*i+b*i+c)
            try:
                if (y_ < img2.shape[0] and y_ > 0):
                    img2[i, y_] = 255
            except:
                pass
        return img2

    def curveTrace(self, frame, left, right):
        '''
        PARAMETERS:  frame,left - coordinates of left boundary, right - coordinates of right boundary
        '''
        height, width = frame.shape
        self.non_zero = []
        # splitting the image to two parts
        left_img = frame[:, :width//2]
        right_img = frame[:, width//2+1:]

        # Working on the left curve
        try:
            curr_points = self.get_lane_points(left_img, left, right, 10, 30)
         # what if very less points?
            if (self.prev_left is None):
                self.prev_left = curr_points
                self.non_zero.append(curr_points)
                x, y = np.transpose(curr_points)
            else:
                if (len(curr_points) < int(0.6*len(self.prev_left)) or curr_points is None):
                    x, y = np.transpose(self.prev_left)
                    self.non_zero.append(self.prev_left)
                else:
                    x, y = np.transpose(curr_points)
                    self.prev_left = curr_points
                    self.non_zero.append(curr_points)

            left_curve = self.detect_curve(left_img, x, y, left, right)
        except:
            left_curve = left_img

        # Working on the right curve
        try:
            flipped_right_img = cv2.flip(right_img, 1)
            curr_points = self.get_lane_points(
                flipped_right_img, left, right, 10, 30)

        # what if very less points?
            if (self.prev_right is None):
                self.prev_right = curr_points
                x, y = np.transpose(curr_points)
                self.non_zero.append(curr_points)
            else:
                if (len(curr_points) < int(0.6*len(self.prev_right)) or curr_points is None):  # 30 %
                    x, y = np.transpose(self.prev_right)
                    self.non_zero.append(self.prev_right)
                else:
                    self.prev_right = curr_points
                    x, y = np.transpose(curr_points)
                    self.non_zero.append(curr_points)

            right_curve = self.detect_curve(
                flipped_right_img, x, y, left, right)
            flipped_right_curve = cv2.flip(right_curve, 1)
        except:
            flipped_right_curve = right_img

        img2 = np.hstack((left_curve, flipped_right_curve))
        return img2

    def drawCurve(self, image, curve, color=(255, 255, 0), thickness=3):
        '''
        PARAMETERS:  image: Original image colored
                     curve -> Curve to draw on the image
                     color -> color of the curve
                     thickness -> Thickness of the curve '''
        height, width, col = image.shape
        if (self.draw == True):
            start = curve.shape[0]//3
        else:
            start = curve.shape[0]
        for i in range(start, curve.shape[0]):
            for j in range(curve.shape[1]):
                if (curve[i, j] != 0):
                    for x in range(thickness):
                        try:
                            image[i, j+x] = color
                        except:
                            pass
        return image


class Predictions():
    '''Provides predictions for a given binary frame where 
       the noise in the image has been removed.
       PARAMETERS: basis: string -> "mean" or "median" 
                           how do you provide the output 
                           for the lane that you acquired
                   threshold: float(0,1) : how closely you 
                           want the lane to be detected relative 
                           to center of image '''

    def __init__(self, basis="mean",
                 threshold=0.1):

        if (basis not in ["mean", "median"]):
            raise ValueError("Basis should be either mean or median")
        self.basis = basis

        if (threshold <= 0 or threshold >= 1):
            raise ValueError("Invalid range for threshold")
        self.threshold = threshold

    def get_lane_middle(self, img, X):
        '''RETURNS: middle x co-ordinate based on the 
                    basis defined in class parameters '''
        if (self.basis == "mean"):
            try:
                mid = int(np.mean(X))
            except:
                mid = img.shape[1]//2
        else:
            try:
                mid = int(np.median(X))
            except:
                mid = img.shape[1]//2
        return mid

    def shifted_lane(self, frame, deviation):
        '''Generates outputs for where to shift 
        given the deviation of the lane center 
        with the image center orientation 

        RETURNS: frame with shift outputs '''
        height, width = frame.shape[0], frame.shape[1]
        shift_left = ["Lane present on left", "Shift left"]
        shift_right = ["Lane present on right", "Shift right"]
        if (deviation < 0):
            # means person on the right and lane on the left
            # need to shift left
            cv2.putText(frame, shift_left[0],
                        (40, 40), 5, 1.1, (100, 10, 255), 2)
            cv2.putText(frame, shift_left[1],
                        (40, 70), 5, 1.1, (100, 10, 255), 2)

            speak(shift_left)
        else:
            # person needs to shift right
            cv2.putText(frame, shift_right[0],
                        (40, 40), 5, 1.1, (100, 255, 10), 2)
            cv2.putText(frame, shift_right[1],
                        (40, 70), 5, 1.1, (100, 255, 10), 2)

            speak(shift_right)
        return frame

    def get_outputs(self, frame, points):
        '''Generates predictions for walking 
           on a lane 
           PARAMETERS: frame : original frame on which we draw
                             predicted outputs. This already has the 
                             lanes drawn on it 
                       points : list of 2-tuples : the list 
                              which contains the points of the lane 
                              which is drawn on the image 
           RETURNS : a frame with the relevant outputs 
           '''

        height, width = frame.shape[0], frame.shape[1]
        # get the center of frame
        center_x = width//2
        # get the distribution of points on
        # left and right of image center
        left_x, right_x = 0, 0
        X = []
        for i in points:
            for k in i:
                x = k
                if (x < center_x):
                    left_x += 1
                else:
                    right_x += 1
                X.append(k)
        # get the lane middle and draw
        try:
            lane_mid = self.get_lane_middle(frame, X)
        except:
            lane_mid = center_x
        cv2.line(frame, (lane_mid, height-1),
                 (lane_mid, height - width//10), (0, 0, 0), 2)
        # calculate shift
        shift_allowed = int(self.threshold*width)
        # calculate deviations and put on image
        deviation = lane_mid - center_x
        deviation_text = "Deviation: " + \
            str(np.round((deviation * 100/width), 3)) + "%"
        cv2.putText(frame, deviation_text, (int(lane_mid-60),
                    int(height-width//(9.5))), 1, 1.3, (250, 20, 250), 2)
        # speak(deviation_text)

        if (abs(deviation) >= shift_allowed):
            # large deviation : give shift outputs only
            frame = self.shifted_lane(frame, deviation)
            return frame
        else:
            # if deviation lesser then that means either correct path
            # or a turn is approaching : text put at the center of the
            # frame

            total_points = left_x + right_x
            correct = ["Good Lane Maintainance", " Continue straight"]
            left_turn = ["Left turn is approaching",
                         "Please start turning left"]
            right_turn = ["Right turn is approaching",
                          "Please start turning right"]
            # if relative change in percentage of points is < 10% then
            # going fine
            try:
                left_perc = left_x*100/(total_points)
                right_perc = right_x*100/(total_points)
            except:
                left_perc = 50
                right_perc = 50
            if (abs(left_perc - right_perc) < 25):
                cv2.putText(frame, correct[0],
                            (40, 40), 5, 1.1, (100, 255, 10), 2)
                cv2.putText(frame, correct[1],
                            (40, 70), 5, 1.1, (100, 255, 10), 2)

                speak(correct)
            else:
                if (left_perc > right_perc):  # more than 25% relative change
                    # means a approximately a right turn is approaching
                    cv2.putText(
                        frame, right_turn[0], (40, 40), 5, 1.1, (100, 10, 255), 2)
                    cv2.putText(
                        frame, right_turn[1], (40, 70), 5, 1.1, (100, 10, 255), 2)

                    speak(right_turn)
                else:
                    cv2.putText(
                        frame, left_turn[0], (40, 40), 5, 1.1, (100, 10, 255), 2)
                    cv2.putText(
                        frame, left_turn[1], (40, 70), 5, 1.1, (100, 10, 255), 2)

                    speak(left_turn)
            return frame


def gen_new(video):
    i = 0
    ImagePreprocessor = PrepareImage(
        (11, 11), (2, 0), False, 50, 170, 0.5, 0.37)
    CurveMaker = Curve(draw=True)
    Predict = Predictions(basis='median', threshold=0.3)

    #yolo model prediction code

    my_file = open(COCO_FILE, "r")
    # reading the file
    data = my_file.read()
    class_list = data.split("\n")
    my_file.close()

    model = YOLO(MODEL_FILE, "v8") 


    while True:
        ret, frame = video.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        
        if ret:

            image = ImagePreprocessor.get_binary_image(frame)
            if image is None or image.size == 0:
                print("Binary image is empty")
                continue


            left, right = ImagePreprocessor.get_poly_maskpoints(image)
            curve = CurveMaker.curveTrace(image, left, right)
            if curve is None or curve.size == 0:
                print("Curve image is empty")
                continue


            curve_with_text = CurveMaker.drawCurve(
                frame, curve)
            points = np.argwhere(curve == 255)
            final = Predict.get_outputs(curve_with_text, points)
            # display_image(final, "Final Output")
            # Write the final output to the output video
            # out.write(final)

            ret, jpeg = cv2.imencode('.jpg', final)
            final = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + final + b'\r\n\r\n')
            i += 1

        frame = cv2.resize(frame, (720, 480))


        detect_params = model.predict(source=[frame], conf=0.45, save=False)
        DP = detect_params[0].numpy()
        no_faces=0
        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                print(i)

                boxes = detect_params[0].boxes
                box = boxes[i]  # returns one box
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                cv2.rectangle(frame,(int(bb[0]), int(bb[1])),(int(bb[2]), int(bb[3])),(255,255,255),3,)
                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                cv2.putText(frame,class_list[int(clsID)],(int(bb[0]), int(bb[1]) - 10),font,0.5,(255, 255, 255),1,)
                if class_list[int(clsID)] == "person":
                    person_detect = ["A person is approaching",
                         "Please be cautious"]
                    speak(person_detect)

        
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')





def speak2(message):
    """Speak the given message."""
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()


def is_person_approaching(id, current_bbox):
    """
    Determine if a person is approaching based on bounding box size over time.
    :param id: Unique ID for the detected person.
    :param current_bbox: Current bounding box dimensions (x1, y1, x2, y2).
    :return: Boolean indicating if the person is approaching.
    """
    global bbox_history

    x1, y1, x2, y2 = current_bbox
    area = (x2 - x1) * (y2 - y1)

    if id not in bbox_history:
        bbox_history[id] = []
    bbox_history[id].append(area)

    if len(bbox_history[id]) > 5:
        bbox_history[id].pop(0)

    if len(bbox_history[id]) > 1 and all(
        bbox_history[id][i] < bbox_history[id][i + 1] for i in range(len(bbox_history[id]) - 1)
    ):
        return True

    return False

def analyze_pose(landmarks):
    """
    Analyze pose landmarks to detect potential threats or possible threats.
    :param landmarks: Normalized landmarks from MediaPipe Pose.
    :return: Tuple (possible_threat, confirmed_threat).
    """
    if not landmarks:
        return False, False

    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    possible_threat = (left_wrist.y < nose.y) or (right_wrist.y < nose.y)

    confirmed_threat = possible_threat and (
        abs(left_wrist.z) < 0.15 or abs(right_wrist.z) < 0.15
    )

    return possible_threat, confirmed_threat

def gen_pose(video):
    model = YOLO(MODEL_FILE)

    frame_count = 0

    fps = int(video.get(cv2.CAP_PROP_FPS))

    frame_interval = max(fps // 4, 1)  

    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0: 
            continue

        frame_resized = cv2.resize(frame, (640, 360))

        results = model(frame_resized)
        detections = results[0].boxes

        for i, box in enumerate(detections):
            if int(box.cls[0]) != 0: 
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if is_person_approaching(i, (x1, y1, x2, y2)):
                speak2("Person approaching")

            person_crop = frame_resized[y1:y2, x1:x2]

            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            result = pose.process(person_rgb)

            if result.pose_landmarks:
                possible_threat, confirmed_threat = analyze_pose(result.pose_landmarks.landmark)

                if confirmed_threat:
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    speak2("Threat detected! Please be cautious!")
                elif possible_threat:
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    speak2("Possible threat, be cautious")
                else:
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame_resized)
        frame_data = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")




def speak1(detections):
    """
    Speak out the detected objects using pyttsx3 with a slight delay.
    :param detections: List of detected object names.
    """
    engine = pyttsx3.init()

    if detections:
        message = f"Money detected: {', '.join(detections)}."
    else:
        message=""
    engine.say(message)
    engine.runAndWait()
    time.sleep(1)  


def gen_money(video):
    model = YOLO('static/models/best.pt')  

    frame_count = 0 
    speak_interval = 30  

    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model(frame)

        annotated_frame = results[0].plot()

        if frame_count % speak_interval == 0:
            detections = []
            if results[0].boxes:
                for detection in results[0].boxes.data.tolist():  
                    class_id = int(detection[5])  
                    class_name = model.names[class_id]  
                    detections.append(class_name)  

            print(f"Detections: {detections}")

            speak1(detections)

        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        frame_count += 1  


def speak3(message):
    """Speak the given message using pyttsx3."""
    if message:
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()

def gen_stair(video):
    model = YOLO('static/models/stair.pt')  
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            results = model(frame)
            annotated_frame = results[0].plot()

            # Check for "stair" in detections
            detections = [model.names[int(d[5])] for d in results[0].boxes.data.tolist()] if results[0].boxes else []
            print(f"Detections: {detections}")

        if detections:
            speak3(", ".join(detections) + " ahead")


        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        frame_count += 1

@app.route('/video_feed_stair')
def video_feed_stair():
    global video
    if not (video.isOpened()):
        return 'Could not process video'
    return Response(gen_stair(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_money')
def video_feed_money():
    global video
    if not (video.isOpened()):
        return 'Could not process video'
    return Response(gen_money(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_pose')
def video_feed_pose():
    global video
    if not (video.isOpened()):
        return 'Could not process video'
    return Response(gen_pose(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_new')
def video_feed_new():
    global video
    if not (video.isOpened()):
        return 'Could not process video'
    return Response(gen_new(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_feed_pose')
def camera_feed_pose():
    global video
    video = cv2.VideoCapture(0)
    if not (video.isOpened()):
        return 'Could not connect to camera'
    return Response(gen_pose(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_feed_money')
def camera_feed_money():
    global video
    video = cv2.VideoCapture(0)
    if not (video.isOpened()):
        return 'Could not connect to camera'
    return Response(gen_money(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera_feed')
def camera_feed():
    global video
    video = cv2.VideoCapture(0)
    if not (video.isOpened()):
        return 'Could not connect to camera'
    return Response(gen_new(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
