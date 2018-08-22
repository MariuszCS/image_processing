import cv2
import numpy as np
import sys
import math
import time


def check_group_for_line(line, group, min_distance, max_distance):
    # the distance between the y of the line and the y (for the same x) of the group is calculated
    # if the distance (error) is small enough (<5) than the line is considered to match the group
    # the distance is calculated in both ways (from the line to group and from group to the line)
    # and the mean value is taken as the final distance
    # n is the ratio from the following equation
    # x2 = x1 + n * vx
    # y2 = y1 + n * vy
    # where (x1, y1) is point that belongs to the group and (x2, y2) to the line we want to classify
    n1, n2, d1, d2, d_mean = 0, 0, 0, 0, sys.maxsize
    if line["vx"] != 0 and group["vx"] != 0:
        line_length = line["original"].shape[0]
        group_length = group["original"].shape[0]
        n1 = (line["points"][0][0] - group["points"][0][0]) / group["vx"]
        n2 = (group["points"][0][0] - line["points"][0][0]) / line["vx"]
        # from the above equations the distance is the absolute value from (y1 - y2)
        d1 = abs((group["points"][0][1] + n1 * group["vy"]) - line["points"][0][1])
        d2 = abs((line["points"][0][1] + n2 * line["vy"]) - group["points"][0][1])
        # we calculate the mean distance
        ratio = line_length / group_length
        if ratio < 1:
            d_mean = (d1  + d2 * ratio) / (1 + ratio)
        else:
            d_mean = (d1 * 1 / ratio + d2) / (1 + 1 / ratio)
    elif line["vx"] == 0 and group["vx"] != 0:
        n1 = (line["points"][0][0] - group["points"][0][0]) / group["vx"]
        d1 = abs((group["points"][0][1] + n1 * group["vy"]) - line["points"][0][1])
        d_mean = (d1 + abs(line["points"][0][0] - group["points"][0][0])) / 2
        print("Line d_mean = {}".format(d_mean))
    elif line["vx"] != 0 and group["vx"] == 0:
        n2 = (group["points"][0][0] - line["points"][0][0]) / line["vx"]
        d2 = abs((line["points"][0][1] + n2 * line["vy"]) - group["points"][0][1])
        d_mean = (d2 + abs(line["points"][0][0] - group["points"][0][0])) / 2
        print("Group d_mean = {}".format(d_mean))
    else:
        d_mean = abs(line["points"][0][0] - group["points"][0][0])
        print("Both d_mean = {}".format(d_mean))

    if d_mean <= min([max_distance, min_distance]): 
        return d_mean, True
        
    return min_distance, False

def fit_line(fit_to, save_to):
    vx, vy, x, y = cv2.fitLine(fit_to, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = float(vx), float(vy), int(x), int(y)
    save_to["points"][0] = (x, y)
    save_to["vx"] = vx
    save_to["vy"] = vy

def determine_lines_for_contour(contour, batch_size = -1):
    lines = []
    # fits line to every 10% of the points of the contour
    # or every 2 points if the contour length is < 20
    contour_length = len(contour)
    if batch_size == -1:
        batch_size = 2
        if contour_length > 13:
            batch_size = int(0.16 * contour_length)
    for index in range(0, contour_length - 1, batch_size):
        points = []
        if contour_length - (index + batch_size) != 2:
            points = contour[index : index + batch_size]
        else:
            points = contour[index : index + batch_size + 1]
            index += batch_size + 1
        lines.append(
            dict(
                {
                    "points": [(0, 0)],
                    "vx": 0,
                    "vy": 0,
                    "original": points
                }
            )
        )
        fit_line(points, lines[-1])

    return lines

def group_lines(lines, divide_into_subsets = False, group_all_contours = False, frame = None):
    to_be_removed = []
    to_skip = []
    # for every line (group) we check if it can be merged to other line (group)
    for line in lines:
        # check if the line with which we want to marge is not already marged
        if line not in to_be_removed and line not in to_skip:
            min_distance = sys.maxsize
            temp = None
            for index in range(0, len(lines)):
                # check if the line we want to merge to was not previously merged
                if lines[index] not in to_be_removed and lines[index] not in to_skip and lines[index] is not line: 
                    min_distance, labeled = check_group_for_line(line, lines[index], min_distance, 5)
                    if labeled:
                        temp = lines[index]
            if min_distance != sys.maxsize:

                lines_too_far = False

                if group_all_contours:

                    lines_too_far = check_lines_distance(temp, line)
                    
                if not lines_too_far:
                    to_be_removed.append(line)
                    temp["points"] += line["points"]
                    temp["original"] = np.append(temp["original"], line["original"], axis=0)
                    # if we have merged two lines/groups, we recalculate the mean point of the group
                    fit_line(temp["original"], temp)
                else:
                    to_skip += [temp, line]

    # remove lines that were merged to other lines
    for group in to_be_removed:
        lines.remove(group)

    if divide_into_subsets:
        new_lines = []
        to_be_removed = []

        for line in lines:
            if len(line["points"]) == 1 and line["original"].shape[0] >= 4:
                to_be_removed.append(line)
                new_lines += determine_lines_for_contour(line["original"], int(line["original"].shape[0] / 2))

        for line in to_be_removed:
            lines.remove(line)

        if len(new_lines) > 0:
            lines += new_lines
            return lines, True

    return lines, False


def check_lines_distance(first_line, second_line):
    l1_min_x_point, l1_max_x_point, l1_min_y_point, l1_max_y_point = find_extreme_points(first_line["original"])
    l2_min_x_point, l2_max_x_point, l2_min_y_point, l2_max_y_point = find_extreme_points(second_line["original"])
    l1_min_point, l1_max_point = -1, -1
    l2_min_point, l2_max_point = -1, -1
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
    if min([math.sqrt((l1_min_point[0] - l2_max_point[0]) ** 2 + (l1_min_point[1] - l2_max_point[1]) ** 2),\
        math.sqrt((l1_max_point[0] - l2_min_point[0]) ** 2 + (l1_max_point[1] - l2_min_point[1]) ** 2)]) > 100:
        """ draw_line(line, frame, (0,255,0))
        draw_line(temp, frame)
        cv2.imshow("binary frame", frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
        input() """
        return True

    return False



def find_trajectory(lines, frame):
    x_left = -sys.maxsize
    x_right = sys.maxsize 
    x_middle = frame.shape[1] / 2
    y_middle = frame.shape[0] / 2
    line_left = dict({})
    line_right = dict({})
    for line in lines:
        # x is somewhere on the line (one of its ends)
        # we chose the line that is the closest to the middle of the frame from its left and right side
        x = line["points"][0][0]
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
    if (x_left != -sys.maxsize and x_right != sys.maxsize):
        n = (y_middle - line_left["points"][0][1]) / line_left["vy"]
        x_left = line_left["points"][0][0] + n * line_left["vx"]
        n = (y_middle - line_right["points"][0][1]) / line_right["vy"]
        x_right = line_right["points"][0][0] + n * line_right["vx"]
        x_mean = (x_left + x_right) / 2
        if line_left["vy"] < 0:
            line_left["vy"] = abs(line_left["vy"])
            line_left["vx"] = -line_left["vx"]
        if line_right["vy"] < 0:
            line_right["vy"] = abs(line_right["vy"])
            line_right["vx"] = -line_right["vx"]
        vy = line_left["vy"] + line_right["vy"]
        vx = line_left["vx"] + line_right["vx"]
        n1 = -y_middle / vy
        n2 = (frame.shape[0] - y_middle) / vy
        x_top = x_mean + n1 * vx
        x_bottom = x_mean + n2 * vx
        cv2.circle(frame, (int(x_mean), int(y_middle)), 10, (255, 0, 0), -1)
        cv2.circle(frame, (int(x_left), int(y_middle)), 10, (0, 255, 0), -1)
        cv2.circle(frame, (int(x_right), int(y_middle)), 10, (0, 0, 255), -1)
        cv2.line(frame, (int(x_top), 0), (int(x_bottom), frame.shape[0]), (255, 0, 0), 2) 

def find_extreme_points(points):
    max_x, max_x_y = -sys.maxsize, -sys.maxsize
    min_x, min_x_y = sys.maxsize, sys.maxsize
    max_y, max_y_x = -sys.maxsize, -sys.maxsize
    min_y, min_y_x = sys.maxsize, sys.maxsize
    for point in points:
        if point[0][0] >= max_x:
            max_x = point[0][0]
            max_x_y = point[0][1]
        if point[0][0] <= min_x:
            min_x = point[0][0]
            min_x_y = point[0][1]
        if point[0][1] >= max_y:
            max_y = point[0][1]
            max_y_x = point[0][0]
        if point[0][1] <= min_y:
            min_y = point[0][1]
            min_y_x = point[0][0]
    return (min_x, min_x_y), (max_x, max_x_y), (min_y_x, min_y), (max_y_x, max_y)

def draw_line(line, frame, color = (0, 0, 255), thickness = 2):
    min_x_point, max_x_point, min_y_point, max_y_point = find_extreme_points(line["original"])
    if max_x_point[0] - min_x_point[0] > max_y_point[1] - min_y_point[1]:
        cv2.line(frame, min_x_point, max_x_point, color, thickness)
    else:
        cv2.line(frame, min_y_point, max_y_point, color, thickness)


def filter_small_lines(lines, x_distance, y_distance):
    to_be_removed = []
    for line in lines:
        min_x_point, max_x_point, min_y_point, max_y_point = find_extreme_points(line["original"])
        if max_x_point[0] - min_x_point[0] < x_distance and max_y_point[1] - min_y_point[1] < y_distance:
            to_be_removed.append(line)
    for line in to_be_removed:
        lines.remove(line)

def filter_not_paired_lines(lines):
    not_paired = []
    # for every line in the group we try to find the corresponding pair (the other/parallel side of the rectangle)
    # since on this stage of the algorithm we expect to have only shpaes that remaind rectangles
    for line in lines:
        paired = False
        for index in range(0, len(lines)):
            if lines[index] is not line and lines[index] not in not_paired:
                if line["vy"] == 0:
                    line["vy"] = 0.00001
                n = (lines[index]["points"][0][1] - line["points"][0][1]) / line["vy"]
                x = line["points"][0][0] + n * line["vx"]
                d = abs(x - lines[index]["points"][0][0])
                a1 = line["vy"] / line["vx"]
                a2 = lines[index]["vy"] / lines[index]["vx"]
                # we consider lines to be pairs when their slope and length are similar
                # as well as we choose the closest ones
                if abs(a1 - a2) < 0.5 and d < 20:
                    paired = True
                    break
        if not paired:
            not_paired.append(line)

    for line in not_paired:
        lines.remove(line)

    return lines

def compute_rotation_and_translation(first_set, second_set):
    # using Kabsch algorithm: https://en.wikipedia.org/wiki/Kabsch_algorithm
    first_centroid = np.mean(first_set, axis=0)
    second_centroid = np.mean(second_set, axis=0)
    first_set_centered = first_set - first_centroid
    second_set_centered = second_set - second_centroid
    cross_covariance_matrix = np.matmul(np.transpose(first_set_centered), second_set_centered)
    u, s, vh = np.linalg.svd(cross_covariance_matrix)
    d = np.linalg.det(np.matmul(np.transpose(vh), np.transpose(u)))
    rotation = np.matmul(np.matmul(np.transpose(vh), np.array([[1, 0], [0, d]])), np.transpose(u))
    translation = second_centroid - np.matmul(rotation, first_centroid)
    return rotation, translation


def detect_lines(frame, frame_area, hue):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_H = np.array([[[hue - 5, 75, 75]]])
    upper_H = np.array([[[hue + 5, 255, 255]]])
        
    extracted_color_frame = cv2.inRange(hsv_frame, lower_H, upper_H)

    _, contours, _ = cv2.findContours(extracted_color_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    
    lines = []
    #cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)
    
    for contour in contours:
        
        if cv2.contourArea(contour) >= frame_area / 3000:
            
            contour_lines = determine_lines_for_contour(contour)
            
            new_lines = True
            
            while new_lines:
                contour_lines, new_lines = group_lines(contour_lines, True)

            number_of_groups = sys.maxsize

            while number_of_groups > len(contour_lines):
                number_of_groups = len(contour_lines)
                contour_lines, _ = group_lines(contour_lines)

            lines += contour_lines

    number_of_groups = sys.maxsize

    while number_of_groups > len(lines):
        number_of_groups = len(lines)
        lines, _ = group_lines(lines, False, True, frame)

    lines = [line for line in lines if line["original"].shape[0] >= 6]
    filter_small_lines(lines, frame.shape[1] / 20, frame.shape[0] / 20)

    return lines


video = cv2.VideoCapture("Cut films/5.mp4")

video_read_correctly, frame = video.read()

cv2.namedWindow('binary frame')
cv2.createTrackbar('H', 'binary frame', 20, 174, lambda x: None)
cv2.setTrackbarMin('H', 'binary frame', 5)
H = cv2.getTrackbarPos('H', 'binary frame')

next_frame = None
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(frame, None)
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 12,
                   key_size = 20,     
                   multi_probe_level = 2)
search_params = dict(checks=50)
flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

displ_x = 0
displ_y = 0
frame_area = frame.shape[0] * frame.shape[1]

while video_read_correctly:
    #start = time.process_time()
    if next_frame is not None:
        kp2, des2 = orb.detectAndCompute(next_frame,None)
        matches = flann_matcher.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        first_set = []
        second_set = []
        for match in matches[:10]:
            first_set += [kp1[match.queryIdx].pt]
            second_set += [kp2[match.trainIdx].pt]
        first_set = np.array(first_set, dtype=np.int32)
        second_set = np.array(second_set, dtype=np.int32)
       
        matrices = compute_rotation_and_translation(first_set, second_set)
        displ_x += matrices[1][0]
        displ_y += matrices[1][1]

        # Draw first 10 matchesframe
        #img3 = cv2.drawMatches(frame, kp1, next_frame, kp2, matches[:10], outImg=np.array([]), flags=0)
        #plt.imshow(img3),plt.show()
        kp1 = kp2.copy()
        des1 = des2.copy()

    #end = time.process_time()
    #print(end - start)
    #input()
    lines = detect_lines(frame, frame_area, H)

    


    #for line in lines:

        #draw_line(line, frame)

    cv2.imshow("binary frame", frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
    #input()


        #cv2.drawContours(frame, [np.append(line["original"][0], line["original"][-1], axis=0)], -1, (0, 0, 255), 1, cv2.LINE_AA)
    #find_trajectory(lines)

    video_read_correctly, frame = video.read()
    #print(displ_x)
    #print(displ_y)
    if video_read_correctly:
        next_frame = frame.copy()

video.release()
cv2.destroyAllWindows()


# przedstawic jak algorytm sie zmienial wraz z obrazkami i przykladami
# poprawic zeby nie laczyl linii ktore sa daleko od siebie