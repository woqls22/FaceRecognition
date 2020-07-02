import cv2, dlib, sys
import numpy as np

scaler = 0.5

# initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load video
cap = cv2.VideoCapture('samples/speeches.mp4')
# load overlay image
overlay = cv2.imread('samples/ryan_transparent.png', cv2.IMREAD_UNCHANGED)

def draw_line(img, L):
  for i in range(len(L)-1):
    x = list(L[i])[0]+3
    y = list(L[i])[1]+3
    b,g,r = img[y,x]
    pallete = (int(b),int(g),int(r))
    #pallete=(255,255,255)
    try:
      cv2.line(img,(list(L[i])[0], list(L[i])[1]),(list(L[i+1])[0], list(L[i+1])[1]), color=pallete,thickness=4, lineType=cv2.LINE_AA)
    except:
      pass


# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
  bg_img = background_img.copy()
  # convert 3 channels to 4 channels
  if bg_img.shape[2] == 3:
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

  if overlay_size is not None:
    img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

  b, g, r, a = cv2.split(img_to_overlay_t)

  mask = cv2.medianBlur(a, 5)

  h, w, _ = img_to_overlay_t.shape
  roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

  img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
  img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

  bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

  # convert 4 channels to 4 channels
  bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

  return bg_img

face_roi = []
face_sizes = []

# loop
while True:
  # read frame buffer from video
  ret, img = cap.read()
  if not ret:
    break

  # resize frame
  img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
  ori = img.copy()

  # find faces
  if len(face_roi) == 0:
    faces = detector(img, 1)
  else:
    roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
    # cv2.imshow('roi', roi_img)
    faces = detector(roi_img)

  # no faces


  # find facial landmarks
  for face in faces:
    if len(face_roi) == 0:
      dlib_shape = predictor(img, face)
      shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
    else:
      dlib_shape = predictor(roi_img, face)
      shape_2d = np.array([[p.x + face_roi[2], p.y + face_roi[0]] for p in dlib_shape.parts()])
    count = 0
    mouse=[]
    left_eyes=[]
    right_eyes=[]
    nose = []
    face_line = []
    left_eyes_brow=[]
    right_eyes_brow=[]
    for s in shape_2d:
      if(count>=48): ## Mouse Part
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        mouse.append(s)
      elif(count>=36and count<=41): #right_eyes Part
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        right_eyes.append(s)
      elif (count >= 42 and count <= 47):  # left_eyes Part
        cv2.circle(img, center=tuple(s), radius=1, color=(30, 255, 54), thickness=2, lineType=cv2.LINE_AA)
        left_eyes.append(s)
      elif(count>=27 and count<=35): # nose Part
        cv2.circle(img, center=tuple(s), radius=1, color=(140, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        nose.append(s)
      elif(count>=0 and count<=16):#face_line Part
        cv2.circle(img, center=tuple(s), radius=1, color=(175, 255, 130), thickness=2, lineType=cv2.LINE_AA)
        face_line.append(s)
      elif (count >= 17 and count <= 21):  # left_eyes_brow Part
        cv2.circle(img, center=tuple(s), radius=1, color=(170, 170, 170), thickness=2, lineType=cv2.LINE_AA)
        left_eyes_brow.append(s)
      elif (count >= 22 and count <= 26):  # right_eyes_brow Part
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        right_eyes_brow.append(s)
      else:
        print("EXCEPT")
      count = count+1
    #draw_line(ori, list(mouse))
    # compute face center
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

    # compute face boundaries
    min_coords = np.min(shape_2d, axis=0)
    max_coords = np.max(shape_2d, axis=0)

    # draw min, max coords
    cv2.circle(img, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    # compute face size
    face_size = max(max_coords - min_coords)
    face_sizes.append(face_size)
    if len(face_sizes) > 10:
      del face_sizes[0]
    mean_face_size = int(np.mean(face_sizes) * 1.8)

    # compute face roi
    face_roi = np.array([int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2), int(min_coords[0] - face_size / 2), int(max_coords[0] + face_size / 2)])
    face_roi = np.clip(face_roi, 0, 10000)

  # visualize
  cv2.imshow('original', ori)
  cv2.imshow('facial landmarks', img)

  if cv2.waitKey(1) == ord('q'):
    sys.exit(1)