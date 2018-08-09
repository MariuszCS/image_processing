import cv2
import numpy as np
import sys
import math

#video = cv2.VideoCapture(0) ----- for camera read
video = cv2.VideoCapture("Cut films/5.mp4")

cv2.namedWindow('binary frame')
cv2.createTrackbar('H', 'binary frame', 20, 174, lambda x: None)
cv2.setTrackbarMin('H', 'binary frame', 5)

video_read_correctly, frame = video.read()

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
    # fits line to every 10 points of the contour
    contour_length = len(contour)
    if batch_size == -1:
        batch_size = 2
        if 25 < contour_length:
            batch_size = int(0.08 * contour_length)
        #elif contour_length >= 150:
            #batch_size = 15
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

def group_dashed_lines(lines, divide_into_subsets):
    groups_to_be_removed = []
    # for every line (group) we check if it can be merged to other line (group)
    for line in lines:
        # check if the line with which we want to marge is not already marged
        if line not in groups_to_be_removed:
            min_distance = sys.maxsize
            temp = dict({})
            for index in range(0, len(lines)):
                # check if the line we want to merge to was not previously merged
                if lines[index] not in groups_to_be_removed and lines[index] is not line: 
                    min_distance, labeled = check_group_for_line(line, lines[index], min_distance, 5)
                    if labeled:
                        temp = lines[index]
            if min_distance != sys.maxsize:
                groups_to_be_removed.append(line)
                temp["points"] += line["points"]
                temp["original"] = np.append(temp["original"], line["original"], axis=0)
                 # if we have merged two lines/groups, we recalculate the mean point of the group
                first_point = np.array([temp["points"][0][0], temp["points"][0][1]], dtype=np.int32)
                second_point = np.array([line["points"][0][0], line["points"][0][1]], dtype=np.int32)
                points = np.array([first_point, second_point])
                #p1, p2 = find_extreme_points(temp)
                #points = np.array([p1, p2])
                fit_line(temp["original"], temp)
                #fit_line(np.array(points, dtype=float), temp)

    # remove lines that were merged to other lines
    for group in groups_to_be_removed:
        lines.remove(group)

    if divide_into_subsets:
        new_lines = []
        groups_to_be_removed = []

        for line in lines:
            if len(line["points"]) == 1 and line["original"].shape[0] >= 6:
                groups_to_be_removed.append(line)
                new_lines += determine_lines_for_contour(line["original"], int(line["original"].shape[0] / 2))
        # remove groups consisting of only one point
        #lines = [line for line in lines if len(line["points"]) > 1]
        for group in groups_to_be_removed:
            lines.remove(group)

        if len(new_lines) > 0:
            lines += new_lines
            return lines, True

    return lines, False

def find_trajectory(lines):
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
    # we calculate the value of x in the middle og=f the frame height
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

def find_extreme_points(line):
    max_y = -sys.maxsize
    min_y = sys.maxsize
    x1 = 0
    x2 = 0
    for point in line["original"]:
        if point[0][1] > max_y:
            max_y = point[0][1]
            x1 = point[0][0]
        if point[0][1] < min_y:
            min_y = point[0][1]
            x2 = point[0][0]
    return (x1, max_y), (x2, min_y)

def draw_line(line):
    for index in range(0, line["original"].shape[0]):
        cv2.circle(frame, (line["original"][index][0][0], line["original"][index][0][1]), 2, (255, 0, 0), -1)
        #(line["original"][index + 1][0][0], line["original"][index + 1][0][1])


while(video_read_correctly):

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    H = cv2.getTrackbarPos('H', 'binary frame')
    lower_H = np.array([[[H - 5, 75, 50]]])
    upper_H = np.array([[[H + 5, 255, 255]]])
        
    extracted_color_frame = cv2.inRange(hsv_frame, lower_H, upper_H)

    _, contours, _ = cv2.findContours(extracted_color_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    lines = []
    
    #cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)
    for contour in contours:

        if cv2.contourArea(contour) >= 300:

            temp = determine_lines_for_contour(contour)

            new_lines = True

            while new_lines:
                temp, new_lines = group_dashed_lines(temp, True)

            number_of_groups = sys.maxsize

            while number_of_groups > len(temp):
                number_of_groups = len(temp)
                temp, _ = group_dashed_lines(temp, False)

            lines += temp

    number_of_groups = sys.maxsize

    while number_of_groups > len(lines):
        number_of_groups = len(lines)
        lines, _ = group_dashed_lines(lines, False)
    #lines = group_dashed_lines(lines)
    lines = [line for line in lines if line["original"].shape[0] >= 6 or len(line["points"]) > 2]

    for line in lines:
        #p1, p2 = find_extreme_points(line)
        #fit_line(np.array([[p1, p2]], dtype=np.int32), line)
        
        #print(line["original"][0][0])
        #cv2.line(frame, p1, p2, (255, 0, 0), 2) 
        #find_extreme_points(line)
        #print(line)
        draw_line(line)
        #for point in line["points"]:
            #cv2.circle(frame, (int(point[0]), int(point[1])), 2, (255, 255, 255), -1)
        cv2.imshow("binary frame", frame)
        #input()
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
        input()
        #print(type(contours))
        #print((np.array([line["points"]], dtype=float)).shape)
        #cv2.drawContours(frame, [np.append(line["original"][0], line["original"][-1], axis=0)], -1, (0, 0, 255), 1, cv2.LINE_AA)
    #find_trajectory(lines)
    #lines = group_dashed_lines(lines)

    video_read_correctly, frame = video.read()

video.release()
cv2.destroyAllWindows()


#print groups
""" for line in lines:
        for points in line["points"][1:]:
            cv2.circle(frame, (int(points[0]), int(points[1])), 2, (255,255,255), -1)
            #cv2.circle(frame, (int(line_pair[1]), int(line_pair[1])), 2, (255,255,255), -1)
        cv2.imshow("binary frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #input() """

# jesli jakas grupa nie zostala zgrupowana z zadna inna to podzielic ja na DWIE GRUPY i sprobowac jeszcze raz

        #ORB