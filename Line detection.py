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

def fit_lines_to_contour(contour):
    lines = []
    for index in range(0, len(contour), 10):
        vx, vy, x, y = cv2.fitLine(contour[index : index + 10], cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x, y = float(vx), float(vy), float(x), float(y)
        lines.append(dict(
            {
                "points": [(x, y)],
                "vx": vx,
                "vy": vy
            })
        )

    """ if len(lines) > 0:
        cv2.circle(frame, (int(lines[0]["points"][0][0]), int(lines[0]["points"][0][1])), 2, (0,0,255), -1)
        cv2.circle(frame, (int(lines[0]["points"][0][0]), int(lines[0]["a"] * lines[0]["points"][0][0] + lines[0]["b"])), 2, (0,0,255), -1)
        print(int(lines[0]["points"][0][1]))
        print(int(lines[0]["a"] * lines[0]["points"][0][0] + lines[0]["b"]))
        cv2.imshow("binary frame", frame)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            pass
        input("INPUT") """
    labeled_group = []
    for line in lines:
        if line["vx"] != 0:
            lines.remove(line)
            labeled_group.append(line)
            break
    if len(labeled_group) == 0:
        return
    for line in lines:
        if line["vx"] == 0:
            continue
        labeled = False
        min_distance = sys.maxsize
        temp = dict({})
        for labeled_line in labeled_group:
            n1 = (line["points"][0][0] - labeled_line["points"][0][0]) / labeled_line["vx"]
            n2 = (labeled_line["points"][0][0] - line["points"][0][0]) / line["vx"]
            d1 = abs((labeled_line["points"][0][1] + n1 * labeled_line["vy"]) - line["points"][0][1])
            d2 = abs((line["points"][0][1] + n2 * line["vy"]) - labeled_line["points"][0][1])
            d_mean = (d1 + d2) / 2
            #if d1 <= min([10, min_distance]) and d2 <= min([10, min_distance]):
            if d_mean <= min([10, min_distance]):
                temp = labeled_line
                #min_distance = min([d1, d2])
                min_distance = d_mean
                labeled = True
        if not labeled:
            labeled_group.append(line)
        else:
            if len(temp["points"]) == 1:
                temp["points"].append(temp["points"][0])
            temp["points"].append(line["points"][0])
            first_point = np.array([temp["points"][0][0], temp["points"][0][1]], dtype=float)
            second_point = np.array([line["points"][0][0], line["points"][0][1]], dtype=float)
            points = np.array([first_point, second_point])
            vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x, y = float(vx), float(vy), float(x), float(y)
            temp["points"][0] = (x, y)
            temp["vx"] = vx
            temp["vy"] = vy

    labeled_group = [labeled_line for labeled_line in labeled_group if len(labeled_line["points"]) > 1]
    
    for labeled_line in labeled_group:
        if len(labeled_line["points"]) > 1:
            for points in labeled_line["points"][1:]:
                cv2.circle(frame, (int(points[0]), int(points[1])), 2, (255,255,255), -1)
            cv2.imshow("binary frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
            input("INPUT")
    

#x = cv2.createHausdorffDistanceExtractor()

while(video_read_correctly):

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    H = cv2.getTrackbarPos('H', 'binary frame')
    lower_H = np.array([[[H - 5, 75, 75]]])
    upper_H = np.array([[[H + 5, 255, 255]]])
        
    extracted_color_frame = cv2.inRange(hsv_frame, lower_H, upper_H)

    _, contours, _ = cv2.findContours(extracted_color_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    filtered_contours = []
    rectangles = []
    rectangles_vertices = []
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)
    for contour in contours:

        contour_area = cv2.contourArea(contour)

        if contour_area >= 200:

            hull = cv2.convexHull(contour, _, True)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            fit_lines_to_contour(contour)

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

    cv2.imshow("binary frame", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    video_read_correctly, frame = video.read()
    
video.release()
cv2.destroyAllWindows()

#regresja liniowa
