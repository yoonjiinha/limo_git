#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class LimoCompetitionMaster:
    def __init__(self):
        rospy.init_node("limo_competition_master")
        
        # === ROS 통신 설정 ===
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.camera_cb)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb)
        self.bridge = CvBridge()

        # === 주행 파라미터 (미션 #1, #2 대응) ===
        self.base_speed = 0.3       # 기본 직선 속도
        self.k_angle = 0.010         # 라인트레이싱 조향 게인
        
        # === 상태 및 센서 변수 ===
        self.scan_ranges = []
        self.front_min = 999.0
        self.state = "LANE"
        self.state_start = rospy.Time.now().to_sec()

        rospy.loginfo("=== 자체리모대회 미션 통합 알고리즘 시작 ===")

    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw

        # [수정] 전방 90도 범위를 전부 감시하여 측면 충돌 방지 (±45도)
        front_zone = np.concatenate([raw[:45], raw[-45:]])
        
        # [수정] 0.05m ~ 1.2m 사이의 모든 유효 데이터를 필터링
        cleaned = [d for d in front_zone if 0.05 < d < 1.2 and not np.isnan(d) and not np.isinf(d)]
        
        if cleaned:
            # [핵심] Median 대신 Min을 사용하여 얇은 라바콘을 즉시 감지함
            self.front_min = np.min(cleaned)
        else:
            self.front_min = 999.0

    def camera_cb(self, msg):
        try:
            twist = Twist()
            now = rospy.Time.now().to_sec()
            
            # 1. 후진 및 탈출 로직 (미션 #3 장애물 회피 대응 [cite: 35, 36])
            if self.state == "ESCAPE":
                self.escape_control()
                return
            if self.state == "BACK":
                self.back_control()
                return

            if self.state == "LANE":
                # [긴급 제동] 0.35m 이내 장애물 감지 시 즉시 정지 및 후진
                if self.front_min < 0.35:
                    rospy.logwarn(f"!!! EMERGENCY !!! Distance: {self.front_min:.2f}m")
                    self.state = "BACK"
                    self.state_start = now
                    return

                # 이미지 전처리 (ROI: 하단 50%)
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                h, w = frame.shape[:2]
                roi = frame[int(h * 0.5):, :]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # ------------------------------------------------
                # 🔴 미션 #4 라바콘 주행 로직 (130pts [cite: 40, 42])
                # ------------------------------------------------
                lower_r1, upper_r1 = np.array([0, 70, 60]), np.array([10, 255, 255])
                lower_r2, upper_r2 = np.array([160, 70, 60]), np.array([180, 255, 255])
                mask_r = cv2.bitwise_or(cv2.inRange(hsv, lower_r1, upper_r1), 
                                        cv2.inRange(hsv, lower_r2, upper_r2))
                mask_r = cv2.dilate(mask_r, np.ones((5,5), np.uint8))
                red_contours, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                valid_cones = []
                for cnt in red_contours:
                    area = cv2.contourArea(cnt)
                    if area > 100:
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            valid_cones.append((int(M["m10"] / M["m00"]), area))

                if len(valid_cones) > 0:
                    valid_cones.sort(key=lambda x: x[0])
                    
                    # [전략] 라바콘 발견 시 초저속 주행 (충돌 감점 -20pts 방지 )
                    max_area = max([c[1] for c in valid_cones])
                    current_speed = 0.05 if max_area > 2000 or self.front_min < 0.6 else 0.10

                    if len(valid_cones) >= 2:
                        target_x = (valid_cones[0][0] + valid_cones[-1][0]) // 2
                        steer_gain = 0.012
                    else:
                        # 하나만 보일 땐 크게 우회
                        cone_x = valid_cones[0][0]
                        safe_margin = 280 
                        target_x = cone_x + safe_margin if cone_x < w // 2 else cone_x - safe_margin
                        steer_gain = 0.015

                    twist.linear.x = current_speed
                    twist.angular.z = max(min((w//2 - target_x) * steer_gain, 1.5), -1.5)
                    self.pub.publish(twist)
                    return

                # ------------------------------------------------
                # ⚫ 미션 #1, #2 차선 주행 로직 (120pts/90pts [cite: 31, 33])
                # ------------------------------------------------
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                col_sum = np.sum(binary > 0, axis=0) 
                
                if col_sum.size > 0 and np.max(col_sum) > 5:
                    track_center_x = np.argmax(col_sum)
                    error = (w / 2.0) - track_center_x
                    twist.linear.x = self.base_speed
                    twist.angular.z = max(min(-self.k_angle * error, 0.8), -0.8)
                else:
                    # 라인을 놓치면 제자리 회전하며 찾기
                    twist.linear.x = 0.0
                    twist.angular.z = 0.25
                
                self.pub.publish(twist)

        except Exception as e:
            rospy.logerr(f"Error: {e}")

    # --- 후진 및 탈출 제어 (미션 #3 장애물 회피 성공을 위한 필수 로직 [cite: 29]) ---
    def back_control(self):
        twist = Twist()
        if rospy.Time.now().to_sec() - self.state_start < 1.4:
            twist.linear.x = -0.12 # 천천히 후진
            self.pub.publish(twist)
        else:
            self.state = "ESCAPE"
            self.state_start = rospy.Time.now().to_sec()

    def escape_control(self):
        twist = Twist()
        if rospy.Time.now().to_sec() - self.state_start < 1.7:
            twist.linear.x = 0.12
            twist.angular.z = 1.0 # 장애물이 없는 방향으로 크게 회전
            self.pub.publish(twist)
        else:
            self.state = "LANE"

if __name__ == "__main__":
    master = LimoCompetitionMaster()
    rospy.spin()
