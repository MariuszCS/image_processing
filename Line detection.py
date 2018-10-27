import cv2
import numpy as np
import sys

def check_group_for_line(line, group, min_distance):
    """Checks if the line may be merged to the group 
    
    Arguments:
        line {dict} -- line which is checked
        group {dict} -- group which is checked
        min_distance {int/float} -- minimal distance that was found so far (also a maximum distance allowed)
    
    Returns:
        float -- the distance between the group and the line (or the min_distance if it was smaller than the one calculated)
        bool -- True if the line can be merged to the group, False otherwise
    """
    # the distance between the y of the line and the y (for the same x) of the group is calculated
    # if the distance (error) is smaller than the min_distance (which is the smallest distance found so far among other groups)
    # the group is considered as the one to which the line will be merged
    # the distance is calculated in both ways (from the line to group and from group to the line)
    # and the mean value is taken as the final distance
    # n is the ratio from the following equation
    # x2 = x1 + n * vx
    # y2 = y1 + n * vy
    # where (x1, y1) is point that belongs to the group and (x2, y2) to the line we want to classify
    n1, n2, d1, d2, d_mean = 0, 0, 0, 0, sys.maxsize
    if line["vx"] != 0 and group["vx"] != 0:
        
        n1 = (line["point"][0] - group["point"][0]) / group["vx"]
        n2 = (group["point"][0] - line["point"][0]) / line["vx"]
        # from the above equations the distance is the absolute value from (y1 - y2)
        d1 = (group["point"][1] + n1 * group["vy"]) - line["point"][1]
        d2 = (line["point"][1] + n2 * line["vy"]) - group["point"][1]
        if d1 < 0:
            d1 = -d1
        if d2 < 0:
            d2 = -d2
        line_length = line["original"].shape[0]
        group_length = group["original"].shape[0]
        ratio = line_length / group_length
        # we calculate the mean distance
        if ratio < 1:
            if ratio < 0.2:
                ratio = 0.2
            d_mean = (d1  + d2 * ratio) / (1 + ratio)
        else:
            if ratio > 5:
                ratio = 5
            d_mean = (d1 * 1 / ratio + d2) / (1 + 1 / ratio)
    # rare cases - one or both lines are vertical
    elif line["vx"] == 0 and group["vx"] != 0:
        n1 = (line["point"][0] - group["point"][0]) / group["vx"]
        d1 = abs((group["point"][1] + n1 * group["vy"]) - line["point"][1])
        d_mean = (d1 + abs(line["point"][0] - group["point"][0])) / 2
    elif line["vx"] != 0 and group["vx"] == 0:
        n2 = (group["point"][0] - line["point"][0]) / line["vx"]
        d2 = abs((line["point"][1] + n2 * line["vy"]) - group["point"][1])
        d_mean = (d2 + abs(line["point"][0] - group["point"][0])) / 2
    else:
        d_mean = abs(line["point"][0] - group["point"][0])

    if d_mean <= min_distance: 
        return d_mean, True
        
    return min_distance, False

def fit_line(fit_to, save_to):
    """Fits the line to the set of points
    
    Arguments:
        fit_to {np.array} -- set of points
        save_to {dict} -- result of the fitting is saved to this line
    """
    vx, vy, x, y = cv2.fitLine(fit_to, cv2.DIST_L2, 0, 0.01, 0.01)
    save_to["point"] = (int(x), int(y))
    save_to["vx"] = float(vx)
    save_to["vy"] = float(vy)

def determine_lines_for_contour(contour, batch_size = -1):
    """Represents the contour as set of lines
    
    Arguments:
        contour {np.array} -- contour represented by set of points
    
    Keyword Arguments:
        batch_size {int} -- number of points of the contour that each line will represent, if default it will be 15% of all the points (default: {-1})
    
    Returns:
        list -- list of lines representing the contour
    """
    lines = []
    # fits line to every 15% of the points of the contour (if batch_size paramter not specified)
    # or every 2 points if the contour length is < 14
    contour_length = len(contour)
    
    if batch_size == -1:
        batch_size = 2
        if contour_length > 13:
            batch_size = int(0.15 * contour_length)
    for index in range(0, contour_length - 1, batch_size):
        # make sure that we won't end up with one point
        points = []
        if contour_length - (index + batch_size) != 2:
            points = contour[index : index + batch_size]
        else:
            points = contour[index : index + batch_size + 1]
            index += batch_size + 1
        lines.append(
            dict(
                {
                    "point": (0, 0),
                    "vx": 0.0,
                    "vy": 0.0,
                    "original": points,
                    "merged": False,
                    "remove": False,
                    "skip": False
                }
            )
        )
        
        # fit line to the points that have just been chosen
        fit_line(points, lines[-1])

    return lines

def group_lines(lines, divide_into_subsets = False, group_all_contours = False, max_error = 5, max_line_distance = 100):
    """Groups lines that are collinear 
    
    Arguments:
        lines {list} -- list of lines to be grouped
    
    Keyword Arguments:
        divide_into_subsets {bool} -- if True, lines that were not grouped will be devided into two new lines (in half) (default: {False})
        group_all_contours {bool} -- if True, checks the distance between the lines so contours that are too far away will not be grouped (default: {False})
        max_error {int} -- max error of considering the lines collinear (default: {5})
        max_line_distance {int} -- used when group_all_contours is True, max distance between the closest ends of the lines that is allowed for grouping(default: {100})
    
    Returns:
        list -- grouped lines
        bool -- True if there were added new lines (only possible if flag divide_into_subsets is True), False otherwise
    """
    # for every line we check if it can be merged to other line
    for line in lines:
        # check if the line which we want to marge is not already marged
        # other case is that this line was considered as best match of previously checked line but
        # is too far, so there is no need to calculate it again (if 1 is too far from 2, than 2 is too far from 1 as well)
        if not line["remove"] and not line["skip"]:
            best = None
            min_distance = sys.maxsize
            # iterating through all lines, to find the best candidate
            for group in lines:
                # check if the line we want to merge to was not previously merged
                # or it is not itself
                # or as previously, has been already considered and is too far
                if not group["remove"] and not group["skip"] and group is not line:
                    min_distance, labeled = check_group_for_line(line, group, min_distance)
                    # we check if the min_distance is <= to the max error that we allow
                    if labeled and min_distance <= max_error:
                        best = group
            # if any appropriate candidate was found the distance should be different than the init value
            if min_distance <= max_error:

                lines_too_far = False

                if group_all_contours:
                    # check if the two lines are not too far from each other
                    lines_too_far = check_lines_too_far(best, line, max_line_distance)

                # if the group_all_contours is False, than this condition will always be True
                if not lines_too_far:
                    # mark line as merged so it won't be considered in next iterations
                    line["remove"] = True
                    # merge lines
                    best["merged"] = True
                    best["original"] = np.append(best["original"], line["original"], axis=0)
                    # recalculate the point that represents the line
                    fit_line(best["original"], best)
                else:
                    # this lines are each other best candidates, however they are too far from each other,
                    # so they are marked as lines to be skipped in next iterations
                    best["skip"] = True
                    line["skip"] = True

    # remove lines that were merged to other lines
    #

    if divide_into_subsets:
        new_lines = []
        
        for line in lines:
            # if the line was not merged and if it contains more than 3 points
            if not line["merged"] and not line["remove"] and line["original"].shape[0] >= 4:
                # mark line as to be removed
                line["remove"] = True
                # create 2 new lines containing half of the points of the original line each (+/- 1 for odd number of points)
                new_lines += determine_lines_for_contour(line["original"], int(line["original"].shape[0] / 2))

        # remove lines marked as to be removed in the previous step
        lines = [line for line in lines if not line["remove"]]

        # add newly created lines to the whole list of lines, so they will be considered in 
        # the next iterations of this algorithm
        if len(new_lines) > 0:
            lines += new_lines
            return lines, True
    else:
        lines = [line for line in lines if not line["remove"]]

    return lines, False


def check_lines_too_far(first_line, second_line, max_line_distance):
    """Checks if the distance between closest ends of two lines is less than max_line_distance
    
    Arguments:
        first_line {dict} -- line
        second_line {dict} -- line
        max_line_distance {int/float} -- max allowed distance
    
    Returns:
        bool -- True if lines are too far (distance between them is > max_line_distance)
    """
    # find extreme points of the lines
    l1_min_x_point, l1_max_x_point, l1_min_y_point, l1_max_y_point = find_extreme_points(first_line["original"])
    l2_min_x_point, l2_max_x_point, l2_min_y_point, l2_max_y_point = find_extreme_points(second_line["original"])
    l1_min_point, l1_max_point = -1, -1
    l2_min_point, l2_max_point = -1, -1
    # in order to chose 2 extreme points (from the 4 available) we have to check
    # the relation between the distance in x direction and distance in y direction
    # of that line
    if l1_max_x_point[0] - l1_min_x_point[0] > l1_max_y_point[1] - l1_min_y_point[1]:
        l1_min_point = l1_min_x_point
        l1_max_point = l1_max_x_point
    else:
        l1_min_point = l1_min_y_point
        l1_max_point = l1_max_y_point
    if l2_max_x_point[0] - l2_min_x_point[0] > l2_max_y_point[1] - l2_min_y_point[1]:
        l2_min_point = l2_min_x_point
        l2_max_point = l2_max_x_point
    else:
        l2_min_point = l2_min_y_point
        l2_max_point = l2_max_y_point
    # the min distance between the opposite line ends is the distance between the lines
    # if it is bigger than max_line_distance, the lines are too far
    line_distance1 = ((l1_min_point[0] - l2_max_point[0]) ** 2 + (l1_min_point[1] - l2_max_point[1]) ** 2) ** 0.5
    line_distance2 = ((l1_max_point[0] - l2_min_point[0]) ** 2 + (l1_max_point[1] - l2_min_point[1]) ** 2) ** 0.5
    min_line_distance = 0
    if line_distance1 < line_distance2:
        min_line_distance = line_distance1
    else:
        min_line_distance = line_distance2
    if max_line_distance < min_line_distance:
        return True

    return False
    

def find_trajectory(lines, frame):
    """Considering that the camera is in the middle of the frame, this function detects two lines
    one closest to the middle from the left and one from the right and based on them creates the resultant line

    Arguments:
        lines {list} -- list of lines detected in the given frame
        frame {np.array} -- frame in which lines where detected (the resultant line will be drawn on this frame)
    """
    x_left = -sys.maxsize
    x_right = sys.maxsize 
    x_middle = frame_length
    y_middle = frame_height
    line_left = None
    line_right = None
    for line in lines:
        # x is somewhere on the line (one of its ends)
        # we chose the line that is the closest to the middle of the frame from its left and right side
        x = line["point"][0]
        if x < x_middle and x > x_left:
            x_left = x
            line_left = line
        elif x > x_middle and x < x_right:
            x_right = x
            line_right = line
    # if both left and right lines are present
    # we calculate the value of x in the middle of the frame height
    # then we take the middle point between x from right and left
    # we change the vy vector to be pointing the positive y direction
    # then we calculate the resultant vector which is the direction for the trajectory
    if line_left is not None and line_right  is not None:
        n = (y_middle - line_left["point"][1]) / line_left["vy"]
        x_left = line_left["point"][0] + n * line_left["vx"]
        n = (y_middle - line_right["point"][1]) / line_right["vy"]
        x_right = line_right["point"][0] + n * line_right["vx"]
        x_mean = (x_left + x_right) / 2
        if line_left["vy"] < 0:
            line_left["vy"] = -line_left["vy"]
            line_left["vx"] = -line_left["vx"]
        if line_right["vy"] < 0:
            line_right["vy"] = -line_right["vy"]
            line_right["vx"] = -line_right["vx"]
        vy = line_left["vy"] + line_right["vy"]
        vx = line_left["vx"] + line_right["vx"]
        n1 = -y_middle / vy
        n2 = (frame_height - y_middle) / vy
        x_top = x_mean + n1 * vx
        x_bottom = x_mean + n2 * vx
        cv2.circle(frame, (int(x_mean), int(y_middle)), 10, (255, 0, 0), -1)
        cv2.circle(frame, (int(x_left), int(y_middle)), 10, (0, 255, 0), -1)
        cv2.circle(frame, (int(x_right), int(y_middle)), 10, (0, 0, 255), -1)
        cv2.line(frame, (int(x_top), 0), (int(x_bottom), frame_height), (255, 0, 0), 2) 

def find_extreme_points(points):
    """Finds all 4 extreme points in the set of points
    
    Arguments:
        points {np.array} -- set of points
    
    Returns:
        touple -- 4 touples with extreme points (x_min, x_max, y_min, y_max with corresponding second coordinate)
    """
    max_x, max_x_y = -sys.maxsize, -sys.maxsize
    min_x, min_x_y = sys.maxsize, sys.maxsize
    max_y, max_y_x = -sys.maxsize, -sys.maxsize
    min_y, min_y_x = sys.maxsize, sys.maxsize
    # iterate through all points, and find all 4 extreme points, with corresponding second coordinate
    # since the points are approximated by a line, but are not perfectly collinear,
    # there could be 4 extreme points and not only 2
    for point in points:
        x = point[0][0]
        y = point[0][1]
        if x >= max_x:
            max_x = x
            max_x_y = y
        if x <= min_x:
            min_x = x
            min_x_y = y
        if y >= max_y:
            max_y = y
            max_y_x = x
        if y <= min_y:
            min_y = y
            min_y_x = x
    return (min_x, min_x_y), (max_x, max_x_y), (min_y_x, min_y), (max_y_x, max_y)

def draw_line(line, frame, color = (0, 0, 255), thickness = 2):
    """Draws the line
    
    Arguments:
        line {dict} -- line to draw (set of points)
        frame {np.array} -- frame on which the line is drawn
    
    Keyword Arguments:
        color {tuple} -- color of the line in BGR format (default: {(0, 0, 255)})
        thickness {int} -- thickness of the line (default: {2})
    """
    # draws the line between the extreme points of the given line
    min_x_point, max_x_point, min_y_point, max_y_point = find_extreme_points(line["original"])
    if max_x_point[0] - min_x_point[0] > max_y_point[1] - min_y_point[1]:
        cv2.line(frame, min_x_point, max_x_point, color, thickness)
    else:
        cv2.line(frame, min_y_point, max_y_point, color, thickness)


def filter_small_lines(lines, x_distance, y_distance):
    """Filters lines that are too shorter than x_distance and y_distance
    
    Arguments:
        lines {list} -- lines
        x_distance {int/float} -- min length of the line in x direction
        y_distance {int/float} -- min length of the line in y direction
    
    Returns:
        list -- list of lines where lines that are too short are filtered out
    """
    for line in lines:
        min_x_point, max_x_point, min_y_point, max_y_point = find_extreme_points(line["original"])
        # if line width is smaller than x_distance and line height is smaller than y_distance
        # than the line is considered as too small and is marked as to be removed
        if max_x_point[0] - min_x_point[0] < x_distance and max_y_point[1] - min_y_point[1] < y_distance:
            line["remove"] = True
    return [line for line in lines if not line["remove"]]

def filter_not_paired_lines(lines):
    """Filters lines that don't have "parallel" (with some error) pair that is not too far.
    Most likely used when lines are mostly straight ones.
    
    Arguments:
        lines {list} -- list of lines 
    
    Returns:
        list -- list of lines where lines without a pair are filtered out
    """
    # for every line in the group we try to find the corresponding pair (the other/parallel side of the rectangle)
    # since on this stage of the algorithm we expect to have only shapes that remaind rectangles
    for line in lines:
        paired = False
        for line2 in lines:
            if line2 is not line and not line2["remove"]:
                if line["vy"] == 0:
                    line["vy"] = 0.00001
                n = (line2["point"][1] - line["point"][1]) / line["vy"]
                x = line["point"][0] + n * line["vx"]
                d = abs(x - line2["point"][0])
                a1 = line["vy"] / line["vx"]
                a2 = line2["vy"] / line2["vx"]
                # we consider lines to be pairs when their slope and length are similar
                # as well as we choose the closest ones
                if abs(a1 - a2) < 0.1 and d < 50:
                    paired = True
                    break
        if not paired:
            line["remove"] = True

    return [line for line in lines if not line["remove"]]

def find_contours(frame, hue):
    """Find the contour on the frame
    
    Arguments:
        frame {np.array} -- frame
        hue {int} -- color of the contours 
    
    Returns:
        np.array -- contours found in the frame
    """
    # convert color space from BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define the color range we are interested in
    lower_H = np.array([[[hue - 5, 75, 75]]])
    upper_H = np.array([[[hue + 5, 255, 255]]])

    # change the frame into binary image, where regions with color that lies in the
    # previously specified range are white and the rest is black    
    extracted_color_frame = cv2.inRange(hsv_frame, lower_H, upper_H)

    # extract contours from the binary image
    _, contours, _ = cv2.findContours(extracted_color_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    return contours


def detect_lines(frame, frame_area, hue, max_line_distance):
    """Detects lines
    
    Arguments:
        frame {np.array} -- frame from the video or image
        frame_area {int} -- area of the frame/image
        hue {int} -- hue value - color of the lines
        max_line_distance {int/float} -- max allowed distance used in checking if lines are not too far from each other 
    
    Returns:
        list -- lines found in the give frame
    """

    contours = find_contours(frame, hue)
    
    lines = []

    for contour in contours:
        
        # initial filtration, do not consider contours that are too small (smaller than frame_area / 3000)
        if cv2.contourArea(contour) >= frame_area / 3000:
            
            # represent contour using lines
            contour_lines = determine_lines_for_contour(contour)
            
            new_lines = True
            
            # perform grouping until the ungrouped lines are divided by the algorithm
            while new_lines:
                contour_lines, new_lines = group_lines(contour_lines, True, False, 4, max_line_distance)

            lines += contour_lines

    number_of_groups = sys.maxsize

    # group lines between all the contours in the current frame
    while number_of_groups > len(lines):
        number_of_groups = len(lines)
        lines, _ = group_lines(lines, False, True, 7, max_line_distance)

    # remove lines based on number of points that they represent
    # small amount of points indicate that the line is short so there is no need
    # to calculate its length what is more computationally demanding
    lines = [line for line in lines if line["original"].shape[0] >= 6]
    # remove lines based on their length
    lines = filter_small_lines(lines, frame_height / 20, frame_length / 20)

    #lines = filter_not_paired_lines(lines)

    return lines


video = cv2.VideoCapture("Cut films/1.mp4")
video_read_correctly, frame = video.read()
cv2.namedWindow('binary frame')
H = 24 # ~yellow

next_frame = None
orb = cv2.ORB_create(nfeatures=100)
kp1, des1 = orb.detectAndCompute(frame, None)
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)
flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

frame_height = frame.shape[0]
frame_length = frame.shape[1]
frame_area = frame_height * frame_length
ref_point = np.zeros((3,1), dtype=np.float)

while video_read_correctly:
    video_read_correctly, next_frame = video.read()
    #SELECTING ROI-----to consider
    #frame = frame[int(frame_height / 2):, (int(frame_length / 2) - int(frame_length / 3)):(int(frame_length / 2) + int(frame_length / 3))]
    if video_read_correctly:
        kp2, des2 = orb.detectAndCompute(next_frame, None)
        matches = flann_matcher.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        first_set = []
        second_set = []
        for match in matches:
            first_set += [kp1[match.queryIdx].pt]
            second_set += [kp2[match.trainIdx].pt]
        first_set = np.array(first_set, dtype=np.float)
        second_set = np.array(second_set, dtype=np.float)

        ''' for p1,p2 in zip(first_set,second_set):
            cv2.circle(frame, (int(p1[0]), int(p1[1])), 5, (0, 0, 0), -1)
            cv2.circle(frame, (int(p2[0]), int(p2[1])), 5, (255, 255, 255), -1)
        cv2.imshow("binary frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break '''

        essential_matrix, mask = cv2.findEssentialMat(first_set, second_set, 1.0, (0.0, 0.0), cv2.RANSAC, 0.999, 1.0)

        points, R, t, mask = cv2.recoverPose(essential_matrix, first_set, second_set, mask=mask)

        '''first_set = np.array([x for i, x in enumerate(first_set) if mask[i] == 1])
        second_set = np.array([x for i, x in enumerate(second_set) if mask[i] == 1])'''

        ref_point = np.matmul(R, ref_point) + t

        kp1 = kp2.copy()
        des1 = des2.copy()

    max_line_distance = (frame_height + frame_length) / 20
    lines = detect_lines(frame, frame_area, H, max_line_distance)

    #find_trajectory(lines, frame)
    for line in lines:

        draw_line(line, frame)

    cv2.imshow("binary frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if video_read_correctly:
        frame = next_frame.copy()

video.release()
cv2.destroyAllWindows()
print(ref_point)