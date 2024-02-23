import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from bidi.algorithm import get_display
from arabic_reshaper import reshape
from PIL import Image, ImageDraw, ImageFont
import math
import pyttsx3
import threading
from gtts import gTTS
import os
import pygame

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize pygame mixer
pygame.mixer.init()

# Initialize the camera, hand detector, and classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(r"C:\Users\smohn\PycharmProjects\cap\Model13\keras_model.h5",
                        r"C:\Users\smohn\PycharmProjects\cap\Model13\labels.txt")

# Set parameters
offset = 20
imgsize = 500
labels = ["اهلا", "انا احبك", "نعم", "مساعده"]

# Load Arabic font using Pillow
font_path = "F:\IBMPlexSansArabic-Thin.ttf"
font_size = 60
font = ImageFont.truetype(font_path, font_size)

# Flag to check if the sound has been played for each label in the current frame
sound_played = {label: False for label in labels}

# Function to play sound in a separate thread
def play_sound(label):
    if label in labels:

        # Generate speech using gTTS
        if label == "اهلا":
            tts = gTTS(text="اهلا", lang="ar")
        elif label == "انا احبك":
            tts = gTTS(text="انا أحبك", lang="ar")
        elif label == "نعم":
            tts = gTTS(text="نعم", lang="ar")
        elif label == "مساعده":
            tts = gTTS(text="مساعدة", lang="ar")
        elif label == "بحبك":
            tts = gTTS(text="أنا أحبك", lang="ar")
        elif label == "كيف حالك":
            tts = gTTS(text="كيف حالك", lang="ar")
        elif label == "شكرا":
            tts = gTTS(text="شكرا", lang="ar")
        else:
            return

        temp_file_path = "temp.mp3"
        tts.save(temp_file_path)

        # Load the sound file using pygame.mixer.Sound
        sound_effect = pygame.mixer.Sound(temp_file_path)

        # Play the sound
        sound_effect.play()

        # Wait for the sound to finish playing
        pygame.time.delay(int(tts.get_duration() * 1000))  # Convert duration to milliseconds

        # Clean up temporary files
        os.remove(temp_file_path)

# ... (rest of your script)

while True:
    success, img = cap.read()

    if not success or img is None:
        continue  # Skip the current iteration if there is an issue with capturing the frame

    imgoutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
        imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Check if imgcrop is empty before attempting to resize
        if imgcrop.size == 0:
            continue

        imgcropShape = imgcrop.shape
        aspecratio = h / w

        if aspecratio > 1:
            k = imgsize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgcrop, (wCal, imgsize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgsize - imgResizeShape[1]) / 2)

            # Adjust the size of the destination array
            imgwhite[:, wGap:wGap + min(imgResizeShape[1], imgsize)] = imgResize[:, :min(imgResizeShape[1], imgsize)]

            prediction, index = classifier.getPrediction(imgwhite, draw=False)
        else:
            k = imgsize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgcrop, (imgsize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgsize - imgResizeShape[0]) / 2)

            # Adjust the size of the destination array
            imgwhite[hGap:hGap + min(imgResizeShape[0], imgsize), :] = imgResize[:min(imgResizeShape[0], imgsize), :]

            prediction, index = classifier.getPrediction(imgwhite, draw=False)
            cv2.rectangle(imgoutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED), (
            x - offset, y - offset - 50)

        if index < len(labels):
            if labels[index] == "none":
                # Display a warning message if no gesture is recognized
                warning_message = "تحذير: حركة غير معروفة!"
                img_pil = Image.fromarray(imgoutput)
                draw = ImageDraw.Draw(img_pil)
                text_size = draw.textbbox((x, y - offset - 50), warning_message, font=font)
                x_text = x - text_size[0] // 2

                y_text = y - offset - 50 - text_size[1] - 10
                draw.text((x_text, y_text), warning_message, font=font, fill=(255, 0, 0))
                imgoutput = np.array(img_pil)
                for label in labels:
                    sound_played[label] = False  # Reset the flag when no gesture is detected
            else:
                # Arabic text formatting
                arabic_text = labels[index]
                reshaped_text = reshape(arabic_text)
                bidi_text = get_display(reshaped_text)

                # Drawing text using Pillow library
                img_pil = Image.fromarray(imgoutput)
                draw = ImageDraw.Draw(img_pil)

                # Assuming point 8 represents the center of the middle finger
                text_size = draw.textbbox((x, y - offset - 50), bidi_text, font=font)
                x_text = x - text_size[0] // 2
                y_text = y - offset - 50 - text_size[1] + 130

                for char in bidi_text:
                    draw.text((x_text, y_text), char, font=font, fill=(255, 255, 255))
                    x_text += font.getlength(char)

                imgoutput = np.array(img_pil)

                # Check for the appearance of a word and play the sound
                if not sound_played[labels[index]]:
                    threading.Thread(target=play_sound, args=(labels[index],)).start()
                    sound_played[labels[index]] = True  # Set the flag to indicate that the sound has been played

                    # Reset the sound_played flag after a delay (adjust the duration as needed)
                    threading.Timer(2.0, lambda: sound_played.update({labels[index]: False})).start()

    cv2.imshow("IMAGE", imgoutput)
    cv2.waitKey(1)
