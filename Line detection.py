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
    n1 = (line["points"][0][0] - group["points"][0][0]) / group["vx"]
    n2 = (group["points"][0][0] - line["points"][0][0]) / line["vx"]
    # from the above equations the distance is the absolute value from (y1 - y2)
    d1 = abs((group["points"][0][1] + n1 * group["vy"]) - line["points"][0][1])
    d2 = abs((line["points"][0][1] + n2 * line["vy"]) - group["points"][0][1])
    d_mean = (d1 + d2) / 2
    # if d1 <= min([10, min_distance]) and d2 <= min([10, min_distance]):  ?
    if d_mean <= min([max_distance, min_distance]): 
    #if d1 <= min([max_distance, min_distance]):
        #min_distance = min([d1, d2])
        min_distance = d1
        return d1, True
        
    return min_distance, False


def determine_lines_for_contour(contour):
    lines = []
    # fits line to every 10 points of the contour
    for index in range(0, len(contour), 10):
        vx, vy, x, y = cv2.fitLine(contour[index : index + 10], cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x, y = float(vx), float(vy), float(x), float(y)
        lines.append(
            dict(
                {
                    "points": [(x, y)],
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
        min_distance = sys.maxsize
        # stores the group to which the line matches best
        temp = dict({}) 
        # check every group 
        for labeled_line in labeled_group:
            min_distance, labeled = check_group_for_line(line, labeled_line, min_distance, 5)
            if labeled:
                temp = labeled_line
        # if the line did not match any group, it starts to represent a new group
        if min_distance == sys.maxsize:
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
            # 1 is tan 45 degrees
            if abs(labeled_line["vy"] / labeled_line["vx"]) > 1:
                vertical_lines.append(labeled_line)
            else:
                horizontal_lines.append(labeled_line)

    return vertical_lines, horizontal_lines

def group_dashed_lines(lines):
    start_index = 0
    indices_taken = []
    groups_to_be_removed = []
    # for every line (group) we check if it can be merged to other line (group)
    for line in lines:
        start_index += 1
        if line["vx"] == 0:
            continue
        # check if the line with which we want to marge is not already marged
        if lines.index(line) not in indices_taken:
            min_distance = sys.maxsize
            temp = dict({})
            for index in range(0, len(lines)):
                # check if the line we want to merged to was not previously merged
                if index not in indices_taken and lines[index] is not line:
                    min_distance, labeled = check_group_for_line(line, lines[index], min_distance, 10)
                    if labeled:
                        temp = lines[index]
            if min_distance != sys.maxsize:
                indices_taken.append(lines.index(line))
                groups_to_be_removed.append(line)
                temp["points"] += line["points"][1:]
                 # if we have merged two lines/groups, we recalculate the mean point of the group
                first_point = np.array([temp["points"][0][0], temp["points"][0][1]], dtype=float)
                second_point = np.array([line["points"][0][0], line["points"][0][1]], dtype=float)
                points = np.array([first_point, second_point])
                vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy, x, y = float(vx), float(vy), float(x), float(y)
                temp["points"][0] = (x, y)
                temp["vx"] = vx
                temp["vy"] = vy

    # remove lines that were merged to other lines
    for group in groups_to_be_removed:
        lines.remove(group)

    return lines

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

while(video_read_correctly):

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    H = cv2.getTrackbarPos('H', 'binary frame')
    lower_H = np.array([[[H - 5, 75, 75]]])
    upper_H = np.array([[[H + 5, 255, 255]]])
        
    extracted_color_frame = cv2.inRange(hsv_frame, lower_H, upper_H)

    _, contours, _ = cv2.findContours(extracted_color_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    vertical_lines = []
    #cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)
    for contour in contours:

        if cv2.contourArea(contour) >= 200:

            vertical_lines += determine_lines_for_contour(contour)[0]

    group_dashed_lines(vertical_lines)

    find_trajectory(vertical_lines)
    cv2.imshow("binary frame", frame)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

    video_read_correctly, frame = video.read()
    
video.release()
cv2.destroyAllWindows()


#print groups
""" for line_group in vertical_lines:
        for line_pair in line_group["points"][1:]:
            cv2.circle(frame, (int(line_pair[0]), int(line_pair[1])), 2, (255,255,255), -1)
            #cv2.circle(frame, (int(line_pair[1]), int(line_pair[1])), 2, (255,255,255), -1)
        cv2.imshow("binary frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        input() """