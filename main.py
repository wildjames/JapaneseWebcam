from PIL import Image
import pytesseract
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "num",
    help='How many images to do?',
    type=int
)
args = parser.parse_args()
N_test = args.num


# counter for later, hopefully quicker
X = 0

# The URL to this camera
# http://www.insecam.org/en/view/805350/
URL = "http://210.148.5.180:8084/?action=stream"

# Open a connection to the weird camera
cap = cv2.VideoCapture(URL)

# I need a temp file to hold my images
if not os.path.isdir("TMP"):
    os.mkdir("TMP")

# I need a permanent dir to hold training data for OCR
if not os.path.isdir("Training_sample"):
    os.mkdir("Training_sample")

margin = 180

def grabber():
    ret, image = cap.read()

    # Crop the image down
    image = image[margin:-margin, margin:-margin]

    global X
    X += 1
    temp_file = os.path.join("TMP", "IMAGE_{:05d}.png".format(X))
    # temp_file = "IMAGE.png"
    cv2.imwrite(temp_file, image)


def human_parser():
    print()
    # Grab the first file in the TMP dir
    files = os.listdir("TMP")
    files = [int(f[-9:-4]) for f in files]
    if len(files) == 0:
        print("No files in the TMP directory!")
        return
    to_proc = min(files)
    to_proc = os.path.join("TMP", "IMAGE_{:05d}.png".format(to_proc))
    print("File: {}".format(to_proc))

    # show the output images
    frame = cv2.imread(to_proc)

    cv2.imshow("Output", frame)
    cv2.waitKey(25)

    num = input("What is the number on screen: ")
    if num.lower() == 'q':
        exit()

    if len(num) != 2:
        return

    try:
        num = int(num)
    except:
        return

    # The data file that contains (path, value)
    num_file = os.path.join("Training_sample", "data.txt")

    # Check how many are already in the directory
    existing_training_images = os.listdir("Training_sample")
    files = [0]
    for file in existing_training_images:
        try:
            file = file[-9:-4]
            files.append(int(file))
        except:
            pass

    # The image filename will be the next number
    mynum = max(files) + 1
    oname = os.path.join("Training_sample", "IMAGE_{:05d}.png".format(mynum))

    with open(num_file, 'a+') as f:
        f.write("{} {}\n".format(oname, num))


    # B/W and smaller resolution makes it easier for the NN
    frame = cv2.resize(frame, (20,20))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    # Save the BW, cropped image
    cv2.imwrite(oname, frame)

    # Now we're done with that raw image. Delete it.
    os.remove(to_proc)


def tesseract_parser():
    # Grab the first file in the TMP dir
    files = os.listdir("TMP")
    files = [int(f[-9:-4]) for f in files]
    to_proc = min(files)
    to_proc = os.path.join("TMP", "IMAGE_{:05d}.png".format(to_proc))

    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text = pytesseract.image_to_string(
        Image.open(to_proc),
        # config='-c tessedit_char_whitelist=0123456789'
    )

    with open("PARSED_IMAGES.txt", 'a+') as f:
        f.write("{}\n".format(text))
    print("Text: {}".format(text))

    # show the output images
    frame = cv2.imread(to_proc)
    cv2.imshow("Output", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        exit()

    # Once processed, idgaf about this anymore
    os.remove(to_proc)
