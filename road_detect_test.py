#####################################
# Last Name  : McIntyre Garcia      #
# First Name : Cristopher           #
# SN         : 300025114            #
# Email      : cmcin019@uottawa.ca  #
#####################################

# Lab 05
import cv2 as cv 
import numpy as np

class ObjectDetector:

    # Initialize image
    def __init__(self, img):

        # Original image to process
        self.img = img

        # Circle detection globals
        self.DP = 0.6 # 2.4 for circles_simple
        self.DIST = 76 # 26 for circles_simple

        # Line detection globals
        self.T_LOW = 570
        self.T_HIGH = 700
        self.H_THRESHOLD = 49
        self.A_SIZE = 3
        self.R_VAL = 17
        self.K_VAL = 1

        pass

    @staticmethod
    def make_cricle():

        # Globals used 
        global DP
        global DIST
        global img

        # Convort original image to grey scale 
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Blur image with 3x3 matrix
        img_gray = cv.blur(img_gray, (3, 3)) 

        # Create output image
        output = img.copy()

        # Detect circles
        circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, DP, DIST, param1 = 50, param2 = 30, minRadius = 30, maxRadius = 60)

        # Check for circles
        if circles is not None:

            # Convert coordinates and radius to integers
            circles = np.round(circles[0, :]).astype("int")

            # Coordinates for every circle 
            for (x, y, r) in circles:

                # Draw detected circle border 
                cv.circle(output, (x, y), r, (0, 255, 0), 2)

                # Draw center of circle 
                cv.circle(output, (x, y), 3, (0,0,255), -1)

            # Show output image
            cv.imshow('Window (Press any key to continue)', output)

        pass

    @staticmethod
    def intersection(line1, line2):

        # Coordinate values of first line 
        rho1, theta1 = line1[0]

        # Coordinate values of second line 
        rho2, theta2 = line2[0]

        # Create np array with theta values 
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2+0.000001), np.sin(theta2+0.000001)]
        ])

        # Create np array with rho values 
        b = np.array([[rho1], [rho2]])

        # Detect intersection between lines
        x0, y0 = np.linalg.solve(A, b)

        # Convert coordinates to integers 
        x0, y0 = int(np.round(x0)), int(np.round(y0))

        # Return coordinates 
        return [x0, y0]

    @staticmethod
    def make_line():

        # Globals used 
        global img
        global T_LOW
        global T_HIGH
        global H_THRESHOLD
        global A_SIZE
        global R_VAL
        global K_VAL

        # Creat copy image for output 
        output = img.copy()

        # Creat copy image to visualize lines
        # lines_img = img.copy()
        
        # Image information
        channels, width, height = img.shape[::-1]
        lines_img = np.full((height, width, channels), (0,0,0), dtype=np.uint8)

        # Convert the img to grayscale 
        gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY) 

        # Blur image 
        gray = cv.blur(gray, (K_VAL, K_VAL)) 
        
        # Apply edge detection method on the image 
        edges = cv.Canny(gray, T_LOW, T_HIGH, apertureSize = A_SIZE, L2gradient = True) 
	
	# Apply close to the image 
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        
        # Apply edge detection method on the image 
        edges = cv.Canny(edges, T_LOW, T_HIGH, apertureSize = A_SIZE, L2gradient = True)[height//2:height, 0:width] 
        
        # Detect lines
        lines = cv.HoughLinesP(edges, R_VAL, np.pi/180, H_THRESHOLD, minLineLength=100, maxLineGap=500) 


        # Loop over every line  #
        #                       #
        # # # # # # # # # # # # #

        # min_x = min(list(map(lambda x: min(x[0][0], x[0][2]), lines)))
        # max_x = max(list(map(lambda x: max(x[0][0], x[0][2]), lines)))
        min_x = width
        max_x = 0
        min_corresponding_y, max_corresponding_y = height, height
        alpha_x, alpha_y = 0, 0
        betha_x, betha_y = 0, 0
	
        # Loop over every line
        for xL in lines: 


            # Calculate the x, y values of the line 
            # With help from https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/
            for x1, y1, x2, y2 in xL:
                
                y1 += height//2
                y2 += height//2

                # Draw line to edges image 
                cv.line(edges,(x1,y1 - height//2), (x2,y2 - height//2), (0,0,0),8) 
                
                line_orientation = np.arctan2(y1 - y2, x1 - x2) * 180. / np.pi
                
                if ((line_orientation > -100 and line_orientation < 170) or (line_orientation > -160 and line_orientation < -120)):
                    # Draw line to output image 
                    # cv.line(output,(x1,y1), (x2,y2), (0,255,0),10) 
                    
                    # Draw line to lines image 
                    # cv.line(lines_img,(x1,y1), (x2,y2), (255,255,255),5) 
                    
                    min_x, min_corresponding_y, alpha_x, alpha_y = min([(x1, y1, x2, y2), (x2, y2, x1, y1), (min_x, min_corresponding_y, alpha_x, alpha_y)])
                    max_x, max_corresponding_y, betha_x, betha_y = max([(x1, y1, x2, y2), (x2, y2, x1, y1), (max_x, max_corresponding_y, betha_x, betha_y)])

                pass

            pass
            
        cv.line(output,(min_x, min_corresponding_y), (max_x, max_corresponding_y), (0,255,0),10)
        cv.line(output,(alpha_x, alpha_y), (betha_x, betha_y), (0,255,0),10)
        cv.line(output,(min_x, min_corresponding_y), (alpha_x, alpha_y), (0,255,0),10)
        cv.line(output,(max_x, max_corresponding_y), (betha_x, betha_y), (0,255,0),10)
        
        cv.line(lines_img,(min_x, min_corresponding_y), (max_x, max_corresponding_y), (255,255,255),5)
        cv.line(lines_img,(alpha_x, alpha_y), (betha_x, betha_y), (255,255,255),5)
        cv.line(lines_img,(min_x, min_corresponding_y), (alpha_x, alpha_y), (255,255,255),5)
        cv.line(lines_img,(max_x, max_corresponding_y), (betha_x, betha_y), (255,255,255),5)
        
        
        thresh = cv.threshold(lines_img, 0, 255, cv.THRESH_BINARY)[1]
        rect=cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        lines_img = cv.dilate(thresh, rect, iterations = 3)
        lines_img = cv.erode(thresh, rect, iterations = 1)
        
        # get the (largest) contour
        edged = cv.Canny(lines_img, 30, 200)
        contours = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv.contourArea)
        
        # draw white filled contour on black background
        cv.drawContours(lines_img, contours, -1, (255,255,255), cv.FILLED)
        cv.drawContours(output, contours, -1, (0,255,0), cv.FILLED)


        # Intersection list
        lst_intersect = []

        # Show images 
        cv.imshow('Canny', edges)
        # Show images 
        cv.imshow('Lines', lines_img)
        # cv.imshow('Canny',edges)
        
        text = "Straight Road"
        textsize = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        
        # get coords based on boundary
        textX = (width - textsize[0]) // 2
        textY = (height + textsize[1]) // 2 + height // 4
        
        #output = cv.putText(output, text, (textX, textY), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        cv.imshow('Window (Press any key to continue)', output)

        pass


    @staticmethod
    def on_change_dp(value):

        # Update dp value 
        global DP
        DP = (value+1)/10

        # Process image
        ObjectDetector.make_cricle()
        pass

    @staticmethod
    def on_change_dist(value):

        # Update distance threshold value 
        global DIST
        DIST = value + 1

        # Process image
        ObjectDetector.make_cricle()
        pass

    @staticmethod
    def on_change_t_high(value):

        # Update threshold upper value 
        global T_HIGH
        T_HIGH = value + 1

        # Process image
        ObjectDetector.make_line()
        pass

    @staticmethod
    def on_change_t_low(value):

        # Update threshold lower value 
        global T_LOW
        T_LOW = value + 1

        # Process image
        ObjectDetector.make_line()
        pass

    @staticmethod
    def on_change_h(value):

        # Update threshold value 
        global H_THRESHOLD
        H_THRESHOLD = value + 1

        # Process image
        ObjectDetector.make_line()

        pass

    @staticmethod
    def on_change_aperture(value):

        # Update aperture size value 
        global A_SIZE
        if value % 2 == 0:
            value += 1
            pass
        A_SIZE = value + 2

        # Process image
        ObjectDetector.make_line()
        pass

    @staticmethod
    def on_change_R_VAL(value):

        # Update r value 
        global R_VAL
        R_VAL = (value + 1)/10

        # Process image
        ObjectDetector.make_line()
        pass

    @staticmethod
    def on_change_K_VAL(value):

        # Update k value 
        global K_VAL
        K_VAL = (value + 1)

        # Process image
        ObjectDetector.make_line()
        pass

    @staticmethod
    def globalize(self):

        # Get globals 
        global DP
        global DIST
        global img
        global T_LOW
        global T_HIGH
        global H_THRESHOLD
        global A_SIZE
        global R_VAL
        global K_VAL

        # Assign values 
        DP = self.DP
        DIST = self.DIST
        img = self.img

        T_LOW = self.T_LOW
        T_HIGH = self.T_HIGH
        H_THRESHOLD = self.H_THRESHOLD
        A_SIZE = self.A_SIZE
        R_VAL = self.R_VAL
        K_VAL = self.K_VAL

        return img

    def _findCircles(self):

        # Globalize variables 
        img = ObjectDetector.globalize(self)

        # Process image 
        ObjectDetector.make_cricle()

        # Create track bars 
        windowName = 'Window (Press any key to continue)'
        cv.createTrackbar('Dp', windowName, 5, 50, ObjectDetector.on_change_dp)
        cv.createTrackbar('Dist', windowName, 75, 200, ObjectDetector.on_change_dist)

        # Wait for key and destroy all windows
        cv.waitKey(0)
        cv.destroyAllWindows()

        pass


    def _findLinesAndIntersections(self):

        # Globalize variables 
        img = ObjectDetector.globalize(self)

        # Process image 
        ObjectDetector.make_line()

        # Create track bars 
        windowName = 'Window (Press any key to continue)'
        cv.createTrackbar('High threshold', windowName, 700, 800, ObjectDetector.on_change_t_high)
        cv.createTrackbar('Low threshold', windowName, 570, 800, ObjectDetector.on_change_t_low)
        cv.createTrackbar('Hough threshold', windowName, 100, 230, ObjectDetector.on_change_h)
        cv.createTrackbar('Aperture size', windowName, 3, 4, ObjectDetector.on_change_aperture)
        cv.createTrackbar('R value', windowName, 17, 19, ObjectDetector.on_change_R_VAL)
        cv.createTrackbar('K value', windowName, 1, 6, ObjectDetector.on_change_K_VAL)

        # Wait for key and destroy all windows
        cv.waitKey(0)
        cv.destroyAllWindows()

        pass

    pass


def main():

    # Part B
    # Loading image
    img = cv.imread('./images/000000_10_mask.png', cv.IMREAD_COLOR) 
    # img = cv.imread('./images/lines_target.jpg', cv.IMREAD_COLOR) 
    cv.imshow('Original',img)
    cd = ObjectDetector(img)
    cd._findLinesAndIntersections()
    


    pass

main()

