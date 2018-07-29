import cv2
import numpy as np
import sys
import math

#video = cv2.VideoCapture(0) ----- for camera read
video = cv2.VideoCapture("Cut films/7.mp4")

cv2.namedWindow('binary frame')
cv2.createTrackbar('H', 'binary frame', 31, 174, lambda x: None)
cv2.setTrackbarMin('H', 'binary frame', 5)

video_read_correctly, frame = video.read()

def contour_to_lines(contour):
    lines = []
    # fits line to every 8 points of the contour
    for index in range(0, len(contour), 10):
        vx, vy, x, y = cv2.fitLine(contour[index : index + 10], cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x, y = float(vx), float(vy), float(x), float(y)
        lines.append(
            dict(
                {
                    "points": [(x, y)],
                    #"mean_point": (x, y),
                    "vx": vx,
                    "vy": vy
                }
            )
        )

    labeled_group = []
    # sets the first line group to be represented by a first line which vx != 0
    for line in lines:
        if line["vx"] != 0:
            lines.remove(line)
            labeled_group.append(line)
            break
    # if there is no lines with vx != 0 than different approach should be applied
    if len(labeled_group) == 0:
        return
    for line in lines:
        # different approach for lines with vx = 0
        if line["vx"] == 0:
            continue
        # indicates if the line matches to any existing group
        labeled = False
        # used to chose the best matching goup
        min_distance = sys.maxsize
        # stores the group to which the line matches best
        temp = dict({})
        # check every group 
        # the distance between the y of the line and the y (for the same x) of the group is calculated
        # if the distance (error) is small enough (<10) than the line is considered to match the group
        # the distance is calculated in both ways (from the line to group and from group to the line)
        # and the mean value is taken as the final distance
        for labeled_line in labeled_group:
            # n is the ratio from the following equation
            # x2 = x1 + n * vx
            # y2 = y1 + n * vy
            # where (x1, y1) is point that belongs to the group and (x2, y2) to the line we want to classify
            n1 = (line["points"][0][0] - labeled_line["points"][0][0]) / labeled_line["vx"]
            #n2 = (labeled_line["points"][0][0] - line["points"][0][0]) / line["vx"]
            # from the above equations the distance is the absolute value from (y1 - y2)
            d1 = abs((labeled_line["points"][0][1] + n1 * labeled_line["vy"]) - line["points"][0][1])
            #d2 = abs((line["points"][0][1] + n2 * line["vy"]) - labeled_line["points"][0][1])
            #d_mean = (d1 + d2) / 2
            # if d1 <= min([10, min_distance]) and d2 <= min([10, min_distance]):  ?
            # if d_mean <= min([5, min_distance]): ?
            if d1 <= min([5, min_distance]):
                temp = labeled_line
                #min_distance = min([d1, d2])
                min_distance = d1
                labeled = True
        # if the line did not match any group, it starts to represent a new group
        if not labeled:
            labeled_group.append(line)
        else:
            # the first element in "points" represents the mean point of the whole group
            if len(temp["points"]) == 1:
                temp["points"].append(temp["points"][0])
            temp["points"].append(line["points"][0])
            # if we have new line in the group, we recalculate the mean point of the group
            first_point = np.array([temp["points"][0][0], temp["points"][0][1]], dtype=float)
            second_point = np.array([line["points"][0][0], line["points"][0][1]], dtype=float)
            points = np.array([first_point, second_point])
            vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x, y = float(vx), float(vy), float(x), float(y)
            temp["points"][0] = (x, y)
            temp["vx"] = vx
            temp["vy"] = vy
    
    vertical_lines = []
    horizontal_lines = []

    # divide groups into horizontal and vertical ones
    for labeled_line in labeled_group:
        # consider only groups consisting of at least 3 elements
        # the first element is the mean point of the group
        if len(labeled_line["points"][1:]) > 2:
            # 0.6 is aprrox tan 45 degrees
            if abs(labeled_line["vy"] / labeled_line["vx"]) > 1:
                vertical_lines.append(labeled_line)
            else:
                horizontal_lines.append(labeled_line)

    """ for line_group in vertical_lines:
        for line_pair in line_group["points"][1:]:
            cv2.circle(frame, (int(line_pair[0]), int(line_pair[1])), 2, (255,255,255), -1)
            #cv2.circle(frame, (int(line_pair[1]), int(line_pair[1])), 2, (255,255,255), -1)
        cv2.imshow("binary frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        input() """
    return vertical_lines, horizontal_lines
    """ line_pairs = []
    start_index = 0
    indices_taken = []

    # for every line in the group we try to find the corresponding pair (the other/parallel side of the rectangle)
    # since on this stage of the algorithm we expect to have only shpaes that remaind rectangles
    for labeled_line in labeled_group:
        start_index += 1
        if labeled_group.index(labeled_line) not in indices_taken:
            paired = False
            d_min = sys.maxsize
            length = len(labeled_line["points"][1:])
            paired_index = -1
            for index in range(start_index, len(labeled_group)):
                if index not in indices_taken:
                    n = (labeled_group[index]["points"][0][1] - labeled_line["points"][0][1]) / labeled_line["vy"]
                    x = labeled_line["points"][0][0] + n * labeled_line["vx"]
                    d = abs(x - labeled_group[index]["points"][0][0])
                    a1 = labeled_line["vy"] / labeled_line["vx"]
                    a2 = labeled_group[index]["vy"] / labeled_group[index]["vx"]
                    length2 = len(labeled_group[index]["points"][1:])
                    # we consider lines to be pairs when their slope and length are similar
                    # as well as we choose the closest ones
                    if abs(a1 - a2) < 0.5 and abs(length - length2) < 5 and d < d_min:
                        d_min = d
                        paired = True
                        paired_index = index
            if paired:
                indices_taken.append(paired_index)
                line_pairs.append((labeled_line, labeled_group[paired_index]))

    for line_pair in line_pairs:
        cv2.circle(frame, (int(line_pair[0]["points"][0][0]), int(line_pair[0]["points"][0][1])), 2, (255,255,255), -1)
        cv2.circle(frame, (int(line_pair[1]["points"][0][0]), int(line_pair[1]["points"][0][1])), 2, (255,255,255), -1)
        cv2.imshow("binary frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        input("INPUT") 
    return line_pairs
    """


def find_trajectory(lines):
    x_left = -sys.maxsize
    x_right = sys.maxsize 
    x_middle = frame.shape[1] / 2
    y_middle = frame.shape[0] / 2
    line_left = dict({})
    line_right = dict({})
    for line in lines:
        # calculate the value of x for the y equal to the middle of the screen
        #n = (y_middle - line["points"][0][1]) / line["vy"]
        #x = line["points"][0][0] + n * line["vx"]
        x = line["points"][0][0]
        if x < x_middle and x > x_left:
            x_left = x
            line_left = line
        elif x > x_middle and x < x_right:
            x_right = x
            line_right = line
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
        #cv2.line(frame, line_left["points"][0], line_right["points"][0], (255, 0, 0), 2) 
        #input()
        """ points = np.array([line_left["points"] + line_right["points"]])
        vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x, y = float(vx), float(vy), float(x), float(y)
        n1 = -y / vy
        n2 = (frame.shape[0] - y) / vy
        x1 = x + n1 * vx
        x2 = x + n2 * vx"""
        cv2.circle(frame, (int(x_mean), int(y_middle)), 10, (255, 0, 0), -1)
        cv2.circle(frame, (int(x_left), int(y_middle)), 10, (0, 255, 0), -1)
        cv2.circle(frame, (int(x_right), int(y_middle)), 10, (0, 0, 255), -1)
        cv2.line(frame, (int(x_top), 0), (int(x_bottom), frame.shape[0]), (255, 0, 0), 2) 

#x = cv2.createHausdorffDistanceExtractor()

while(video_read_correctly):

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    H = cv2.getTrackbarPos('H', 'binary frame')
    lower_H = np.array([[[H - 5, 75, 75]]])
    upper_H = np.array([[[H + 5, 255, 255]]])
        
    extracted_color_frame = cv2.inRange(hsv_frame, lower_H, upper_H)

    _, contours, _ = cv2.findContours(extracted_color_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    """ filtered_contours = []
    rectangles = []
    rectangles_vertices = [] """
    varitcal_lines = []
    #cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)
    for contour in contours:

        if cv2.contourArea(contour) >= 200:

            varitcal_lines += contour_to_lines(contour)[0]

    find_trajectory(varitcal_lines)
    input()
    cv2.imshow("binary frame", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    video_read_correctly, frame = video.read()
"""             hull = cv2.convexHull(contour, _, True)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            

            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            height, width = rect[1]
            #if cv2.matchShapes(np.array([box]), contour, cv2.CONTOURS_MATCH_I3, 0) < 5:
            #if (x.computeDistance(cv2.convexHull(contour, _, True), np.array([box]))) < 20:

            if (height > width * 5 or width > height * 5):
            #and cv2.contourArea(np.array([box])) <= 2 * cv2.contourArea(hull):
                rectangles.append(rect)
                filtered_contours.append(contour)
                rectangles_vertices.append(np.array([box]))

    cv2.drawContours(frame, filtered_contours, -1, (0, 0, 255), 1)
    cv2.drawContours(frame, rectangles_vertices, -1, (0, 255, 0), 1)

    lines = []

    for rectangle in rectangles:

        center_x, center_y = rectangle[0]
        height, width = rectangle[1]
        angle = rectangle[2]

        if height > width:
            x1 = int(center_x - abs((height / 2) * math.cos(math.radians(angle))))
            y1 = int(center_y + abs((height / 2) * math.sin(math.radians(angle))))
            x2 = int(center_x + abs((height / 2) * math.cos(math.radians(angle))))
            y2 = int(center_y - abs((height / 2) * math.sin(math.radians(angle))))
        else:
            x1 = int(center_x + abs((width / 2) * math.sin(math.radians(angle))))
            y1 = int(center_y + abs((width / 2) * math.cos(math.radians(angle))))
            x2 = int(center_x - abs((width / 2) * math.sin(math.radians(angle))))
            y2 = int(center_y - abs((width / 2) * math.cos(math.radians(angle))))

        lines.append(((x1, y1), (x2, y2)))

        #cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
       
    middle_point_x = int(frame.shape[1] / 2)
    closest_left_line = tuple()
    closest_right_line = tuple()

    min_distance_left = -sys.maxsize
    min_distance_right = sys.maxsize

    for line in lines:
        if (line[0][0] <= middle_point_x and line[1][0] <= middle_point_x) \
        and (line[0][0] >= min_distance_left or line[1][0] >= min_distance_left):
            if line[0][0] > line[1][0]:
                min_distance_left = line[0][0]
            else:
                min_distance_left = line[1][0]
            closest_left_line = line
        elif (line[0][0] >= middle_point_x and line[1][0] >= middle_point_x) \
        and (line[0][0] <= min_distance_right or line[1][0] <= min_distance_right):
            if line[0][0] < line[1][0]:
                min_distance_right = line[0][0]
            else:
                min_distance_right = line[1][0]
            closest_right_line = line

    cv2.line(frame, (middle_point_x, frame.shape[0]), (middle_point_x, 0), (0, 0, 255), 2)

    if len(closest_left_line) > 0:
        cv2.line(frame, closest_left_line[0], closest_left_line[1], (255, 0, 0), 2)

    if len(closest_right_line) > 0:
        cv2.line(frame, closest_right_line[0], closest_right_line[1], (255, 0, 0), 2)

    if len(closest_left_line) > 0 and len(closest_right_line) > 0:
        cv2.line(frame, (int((closest_right_line[0][0] + closest_left_line[0][0] + closest_right_line[1][0] + closest_left_line[1][0]) / 4), 0)\
        , (int((closest_right_line[0][0] + closest_left_line[0][0] + closest_right_line[1][0] + closest_left_line[1][0]) / 4), frame.shape[0]),\
        (255, 0, 0), 2) 
    """

    
video.release()
cv2.destroyAllWindows()