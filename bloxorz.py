import cv2
import mediapipe as mp
import math
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
import pyautogui

# Initialize Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define a function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.hypot(point2[0]-point1[0], point2[1]-point1[1])

def recognize_pointLeft(hand_landmarks):
    # check if the distance between the tip of index finger is closer to 0 than base
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP] 
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP] 
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # pinky base should be lower than index base
    pinky_base_lower = pinky_tip.y > index_tip.y

    # tip of middle, ring and pinky should be right of knuckle
    is_all_tips_lower = middle_tip.x > middle_pip.x and ring_tip.x > ring_pip.x and pinky_tip.x > pinky_pip.x

    # pointing left
    is_pointing_left = index_tip.x < index_mcp.x

    is_above_threshold = abs(index_tip.x - index_pip.x) - abs(index_mcp.x - index_pip.x) > -0.005

    # print(f"STRAINT INDEX { abs(index_tip.x - index_pip.x) - abs(index_mcp.x - index_pip.x)}")

    # knuckles vertical
    index_pinky_knuckle_x_dist = abs(middle_mcp.x - pinky_mcp.x)
    knuckles_vertical= index_pinky_knuckle_x_dist < 0.1

    if is_pointing_left and pinky_base_lower and is_all_tips_lower and knuckles_vertical and is_above_threshold:
        return "Point Left"
    else:
        return None
    
def recognize_pointRight(hand_landmarks):
    # get thumb coordinated
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]

    # get all other fingers PIP
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    # get all other fingers TIP
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # tips of all other fingers lower than pip of all other fingers
    is_all_tips_lower = index_tip.y > index_pip.y and middle_tip.y > middle_pip.y and ring_tip.y > ring_pip.y and pinky_tip.y > pinky_pip.y

    index_pinky_knuckle_y_dist = abs(thumb_tip.y - index_tip.y)
    knuckles_horizontal = index_pinky_knuckle_y_dist < 0.15

    # x of tip of thumb > x of thumb cmc and mcp and is_all_tips_lower = True and knuckles are horizontal
    if thumb_tip.x > thumb_cmc.x and is_all_tips_lower and knuckles_horizontal:
        return "Point Right"
    else:
        return None


# Define a simple gesture recognition function
def recognize_palm(hand_landmarks):
    # Example: Recognize if the hand is showing a fist or open palm
    # We'll check the distance between the tip of the thumb and the base
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    thumb_dist = calculate_distance(
        (thumb_tip.x, thumb_tip.y), 
        (thumb_mcp.x, thumb_mcp.y)
    )
    index_dist = calculate_distance(
        (index_tip.x, index_tip.y), 
        (index_mcp.x, index_mcp.y)
    )

    if thumb_dist > 0.1 and index_dist > 0.1:
        return "Open Palm"
    else:
        return None
    
def recognize_ok(hand_landmarks):
    # Extract necessary landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Calculate distance between thumb tip and index tip
    distance = calculate_distance(
        (thumb_tip.x, thumb_tip.y), 
        (index_tip.x, index_tip.y)
    )

    # Check if thumb and index are close and other fingers are open
    if distance < 0.05:
        if (middle_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            ring_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            pinky_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y):
            return "Okay Gesture"
    return "Unknown"

model_path = "gesture_recognizer.task" 

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)

gesture_recognizer = GestureRecognizer.create_from_options(options)

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        is_prev_rest = True
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally and convert the BGR image to RGB.
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the image to a Mediapipe Image object for the gesture recognizer
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Perform gesture recognition on the image
            result = gesture_recognizer.recognize(mp_image)


            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)

            # Draw the hand annotations on the image.
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            

            if result.gestures:
                recognized_gesture = result.gestures[0][0].category_name
                confidence = result.gestures[0][0].score
                
                # Example of pressing keys with pyautogui based on recognized gesture

                if is_prev_rest:
                    if recognized_gesture == "Thumb_Up":
                        pyautogui.press("w")
                        is_prev_rest = False
                        pass
                    elif recognized_gesture == "Thumb_Down":
                        pyautogui.press("s")
                        is_prev_rest = False
                        pass
                
                if recognized_gesture == "Open_Palm":
                    is_prev_rest = True

                # Display recognized gesture and confidence 
                cv2.putText(image, f"Gesture: {recognized_gesture} ({confidence:.2f})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Custom
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Recognize gesture
                    # re = recognize_palm(hand_landmarks)
                    
                    gesture = recognize_pointRight(hand_landmarks)

                    if is_prev_rest:
                        if gesture == "Point Right":
                            pyautogui.press("d")
                            is_prev_rest = False
                            pass
                        else:
                            gesture = recognize_pointLeft(hand_landmarks)
                            if gesture == "Point Left":
                                pyautogui.press("a")
                                is_prev_rest = False
                                pass

                    # Display gesture near hand location
                    cv2.putText(image, gesture, 
                                (int(hand_landmarks.landmark[0].x * image.shape[1]), 
                                 int(hand_landmarks.landmark[0].y * image.shape[0]) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Display the resulting image
            cv2.imshow('Gesture Recognition', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
