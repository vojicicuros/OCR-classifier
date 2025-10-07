from idlelib.debugger_r import wrap_frame
import cv2
import os


def extract_frame_and_save(filename):
    new_filename = filename[4:]+'.png'
    data = cv2.VideoCapture(filename)
    frame = data.read()
    frame = cv2.cvtColor(frame[1], cv2.COLOR_BGR2GRAY)
    # print(frame)
    # cv2.imshow('Grey image', frame)
    # cv2.waitKey()
    cv2.imwrite(new_filename, frame)
    print(f'Saved {new_filename} to {os.getcwd()}!')

for i in range(0,10):
    for j in range(1,113):
        filename = 'test'+str(i)+f'{j:03}'
        extract_frame_and_save(filename)
        print('Done')


