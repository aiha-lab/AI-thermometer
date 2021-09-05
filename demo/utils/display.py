
"""display.py
"""


import cv2


def open_window(window_name, title, width=None, height=None):
    """Open the display window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(window_name, title)
    if width and height:
        cv2.resizeWindow(window_name, width, height)


def show_help_text(img, help_text):
    """Draw help text on image."""
    cv2.putText(img, help_text, (11, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (32, 32, 32), 4, cv2.LINE_AA)
    cv2.putText(img, help_text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (240, 240, 240), 1, cv2.LINE_AA)
    return img


def show_fps(img, fps):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    return img

def show_temperature(img, temp):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    line = cv2.LINE_AA
    temp_text = 'Temperature: {:.2f}'.format(temp)
    cv2.putText(img, temp_text, (11, 10), font, 0.3, (32, 32, 32), 4, line)
    cv2.putText(img, temp_text, (10, 10), font, 0.3, (240, 240, 240), 1, line)
    return img

def show_waiting(img):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    line = cv2.LINE_AA
    wait_text = "Waiting..."
    press_text = "Please keep your face close"
    cv2.putText(img, wait_text, (41, 30), font, 0.5, (32, 32, 32), 4, line)
    cv2.putText(img, wait_text, (40, 30), font, 0.5, (240, 240, 240), 1, line)
    cv2.putText(img, press_text, (11, 80), font, 0.3, (32, 32, 32), 4, line)
    cv2.putText(img, press_text, (10, 80), font, 0.3, (240, 240, 240), 1, line)
    
    return img

def show_status(img, stat, color):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    line = cv2.LINE_AA
    cv2.putText(img, stat, (115, 10), font, 0.3, color, 4, line)
    cv2.putText(img, stat, (116, 10), font, 0.3, (240, 240, 240), 1, line)
    return img

def set_display(window_name, full_scrn):
    """Set disply window to either full screen or normal."""
    if full_scrn:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_NORMAL)
