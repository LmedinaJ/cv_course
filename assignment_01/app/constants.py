import cv2

class Constants:
    # Window settings
    CAMERA_WINDOW = 'Camera'
    
    # Font settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_LARGE = 0.8
    FONT_SCALE_MEDIUM = 0.7
    FONT_SCALE_SMALL = 0.6
    FONT_SCALE_TINY = 0.5
    FONT_SCALE_MICRO = 0.4
    FONT_THICKNESS_BOLD = 2
    FONT_THICKNESS_NORMAL = 1
    
    # UI Colors (BGR format) - Named by usage
    UI_MENU_TEXT = (0, 0, 255)        # Verde - Texto del menú principal
    UI_ERROR_TEXT = (0, 0, 255)       # Rojo - Mensajes de error
    UI_INFO_TEXT = (255, 255, 255)    # Blanco - Información general
    UI_HIGHLIGHT = (0, 255, 255)      # Amarillo - Elementos destacados
    UI_SECONDARY = (200, 200, 200)    # Gris - Texto secundario
    UI_BACKGROUND = (0, 0, 0)         # Negro - Fondo
    
    # AR/Detection Colors
    AR_MARKER_DETECTED = (0, 255, 0)  # Verde - Marcador AR detectado
    AR_MARKER_ERROR = (0, 0, 255)     # Rojo - Error en marcador AR
    CALIBRATION_SUCCESS = (0, 255, 0) # Verde - Calibración exitosa

    
    # Histogram settings
    HIST_HEIGHT = 200
    HIST_WIDTH = 512
    HIST_BINS = 256
    HIST_RANGE = [0, 256]
    HIST_POSITION_X_OFFSET = 20
    HIST_POSITION_Y = 80
    HIST_BACKGROUND_ALPHA = 0.3
    HIST_LINE_THICKNESS = 2
    
    # Default filter parameters
    DEFAULT_BLUR_KERNEL = 15
    DEFAULT_CANNY_LOW = 100
    DEFAULT_CANNY_HIGH = 200
    DEFAULT_BILATERAL_D = 9
    DEFAULT_BILATERAL_SIGMA_COLOR = 75
    DEFAULT_BILATERAL_SIGMA_SPACE = 75
    DEFAULT_HOUGH_THRESHOLD = 100
    DEFAULT_HOUGH_MIN_LINE_LENGTH = 50
    DEFAULT_HOUGH_MAX_LINE_GAP = 10
    DEFAULT_CONTRAST = 10
    DEFAULT_BRIGHTNESS = 0
    
    # Default transformation parameters
    DEFAULT_TRANSLATE_X = 0
    DEFAULT_TRANSLATE_Y = 0
    DEFAULT_ROTATION = 0
    DEFAULT_SCALE = 100
    
    # Trackbar limits
    MAX_BLUR_KERNEL = 50
    MAX_CANNY_THRESHOLD = 300
    MAX_BILATERAL_D = 25
    MAX_BILATERAL_SIGMA = 200
    MAX_HOUGH_THRESHOLD = 200
    MAX_HOUGH_LINE_LENGTH = 200
    MAX_HOUGH_LINE_GAP = 50
    MAX_CONTRAST = 30
    MAX_BRIGHTNESS = 200
    
    # Transformation limits
    MAX_TRANSLATE = 200  # -200 to +200 pixels
    MAX_ROTATION = 360   # 0 to 360 degrees
    MIN_SCALE = 10       # 10% minimum scale
    MAX_SCALE = 300      # 300% maximum scale
    
    # Camera calibration settings
    CHESSBOARD_SIZE = (9, 6)  # Number of internal corners (width, height)
    SQUARE_SIZE_MM = 25       # Real-world size of chessboard square in mm
    TARGET_CALIBRATION_IMAGES = 20  # Number of images needed for calibration
    CALIBRATION_CAPTURE_DELAY = 2   # Seconds between automatic captures
    CALIBRATION_FILE = 'sources/calibration.npz'  # Output file for calibration data
    
    # UI layout
    UI_TEXT_Y_OFFSET = 30
    UI_TEXT_Y_SPACING = 35
    UI_QUIT_TEXT_Y_OFFSET = 20