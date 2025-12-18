#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class LineTracerWithObstacleAvoidance:
    def __init__(self):
        rospy.init_node("line_tracer_with_obstacle_avoidance")
        
        # === ROS 통신 ===
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.camera_cb)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb)

        self.bridge = CvBridge()

        # === 주행 파라미터 ===
        self.speed = 0.2        # 기본 주행 속도
        self.search_speed = 0.25 # 라인 놓쳤을 때 회전 속도
        
        # === 검은선 트레이싱 튜닝 파라미터 ===
        self.k_angle = 0.008     # 조향 게인 (반응 민감도)
        self.dark_min_pixels = 5 # 이 값보다 픽셀이 적으면 라인 없음으로 간주

        # === 상태 변수 ===
        self.scan_ranges = []
        self.front = 999.0
        
        self.state = "LANE"
        self.state_start = rospy.Time.now().to_sec()
        self.escape_angle = 0.0

        # 회피 로직 변수
        self.left_escape_count = 0
        self.force_right_escape = 0

        rospy.loginfo("=== 라인트레이서(검은선) + 장애물 회피 + 라바콘 주행 시작 ===")

    # ============================================================
    # LIDAR 콜백 (장애물 감지 - 그대로 유지)
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw

        # 전방 20도 범위의 장애물 감지
        front_zone = np.concatenate([raw[:15], raw[-15:]])
        # 20cm 이상의 유효한 데이터만 필터링
        cleaned = [d for d in front_zone if d > 0.10 and not np.isnan(d) and not np.isinf(d)]
        
        if cleaned:
            self.front = np.median(cleaned)
        else:
            self.front = 999.0

    # ============================================================
    # CAMERA 콜백 (메인 로직)
    # ============================================================
    def camera_cb(self, msg):
        try:
            twist = Twist()
            now = rospy.Time.now().to_sec()
            
            # 1. ESCAPE 모드 (장애물 회피 중 - 그대로 유지)
            if self.state == "ESCAPE":
                self.escape_control()
                return

            # 2. BACK 모드 (장애물 감지 후 후진 - 그대로 유지)
            if self.state == "BACK":
                self.back_control()
                return

            # 3. LANE 모드 (라인/라바콘 주행)
            if self.state == "LANE":
                # 장애물 감지 시 BACK으로 전환
                limit_dist = 0.45
                if self.front < limit_dist:
                    rospy.logwarn(f"장애물 감지: {self.front:.2f}m -> 후진")
                    self.state = "BACK"
                    self.state_start = now
                    return

                # 이미지 처리
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                h, w = frame.shape[:2]
                
                # ROI: 바닥 쪽 50% 사용 (제공해주신 코드 기준)
                roi_y_start = int(h * 0.5)
                roi = frame[roi_y_start:, :]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # ------------------------------------------------
                # 🔴 [기존 유지] 라바콘(빨간색) 우선 검출 로직
                # ------------------------------------------------
                # 1. 색상 검출 (기존 유지하되 노이즈 제거 강화)
                lower_r1, upper_r1 = np.array([0, 120, 70]), np.array([10, 255, 255])
                lower_r2, upper_r2 = np.array([170, 120, 70]), np.array([180, 255, 255])

                mask_r = cv2.bitwise_or(cv2.inRange(hsv, lower_r1, upper_r1), 
                                        cv2.inRange(hsv, lower_r2, upper_r2))
                mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
                red_contours, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                valid_cones = []
                for cnt in red_contours:
                    area = cv2.contourArea(cnt)
                    if 150 < area < 10000: # 너무 작거나 큰 노이즈 제거
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            valid_cones.append((cx, area))

                if len(valid_cones) > 0:
                    valid_cones.sort(key=lambda x: x[0]) # X좌표 기준 정렬
                    
                    # LiDAR 연동: 라바콘 구역에서는 감속 (미션 실패 방지)
                    cone_speed = 0.12 
                    if self.front < 0.5: cone_speed = 0.08 # 가까우면 더 감속 [cite: 29]

                    if len(valid_cones) >= 2:
                        # 두 라바콘 사이의 중앙으로 조향
                        target_x = (valid_cones[0][0] + valid_cones[-1][0]) // 2
                        error = (w // 2) - target_x
                        steer = error * 0.007 # 게인값 미세 조정
                    else:
                        # 라바콘이 하나만 보일 때: 급회전 대신 '회피 여유' 확보
                        cone_x = valid_cones[0][0]
                        safe_margin = 150 # 라바콘으로부터 떨어질 거리 (픽셀)
                        
                        if cone_x < w // 2: # 왼쪽 라바콘 발견 -> 약간 오른쪽으로
                            target_x = cone_x + safe_margin
                        else: # 오른쪽 라바콘 발견 -> 약간 왼쪽으로
                            target_x = cone_x - safe_margin
                        
                        error = (w // 2) - target_x
                        steer = error * 0.005

                    twist.linear.x = cone_speed
                    twist.angular.z = max(min(steer, 0.8), -0.8)
                    self.pub.publish(twist)
                    return

                # ------------------------------------------------
                # ⚫ [수정됨] 검은색 라인 트레이싱 로직
                # (EdgeLaneNoBridge 코드 이식)
                # ------------------------------------------------
                
                # 1. 그레이스케일 + 블러
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                # 2. 검은 트랙 강조: THRESH_BINARY_INV + OTSU
                # (검은색 라인이 흰색(255)이 되고 배경이 검은색(0)이 됨)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # 3. 노이즈 제거
                kernel = np.ones((3, 3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                # 4. 열별 "검은 픽셀(=255)" 개수 합산
                # col_sum 배열의 각 값은 해당 열(세로줄)에 있는 흰색 점의 개수
                col_sum = np.sum(binary > 0, axis=0) 
                
                if col_sum.size > 0:
                    max_val = int(np.max(col_sum))
                else:
                    max_val = 0

                # 5. 라인이 잡혔는지 확인 (너무 어두운 픽셀이 적으면 라인 못 찾음)
                if max_val < self.dark_min_pixels:
                    # 라인 못 찾음 -> 제자리 회전하며 찾기
                    twist.linear.x = 0.0
                    twist.angular.z = self.search_speed
                    self.pub.publish(twist)
                    return

                # 6. 유효한 트랙 후보 열 추출 (max값의 일정 비율 이상인 곳만)
                dark_col_ratio = 0.3
                threshold_val = max(self.dark_min_pixels, int(max_val * dark_col_ratio))
                candidates = np.where(col_sum >= threshold_val)[0]

                if candidates.size == 0:
                    # 후보가 없으면 회전
                    twist.linear.x = 0.0
                    twist.angular.z = self.search_speed
                    self.pub.publish(twist)
                    return

                # 7. 무게 중심 계산 (Weighted Average)
                x_indices = np.arange(len(col_sum))
                # 검은색 덩어리들의 무게중심 X좌표
                track_center_x = float(np.sum(x_indices[candidates] * col_sum[candidates]) /
                                       np.sum(col_sum[candidates]))

                # 8. 조향 계산
                center = w / 2.0
                offset = track_center_x - center # +: 트랙이 오른쪽, -: 트랙이 왼쪽
                
                # 제공해주신 코드의 조향 로직: ang = -self.k_angle * offset
                # 트랙이 오른쪽에 있으면(offset > 0) -> ang는 음수(우회전) -> 맞음
                ang = -self.k_angle * offset
                
                # 조향값 제한 (-0.8 ~ 0.8)
                ang = max(min(ang, 0.8), -0.8)

                # 최종 명령 발행
                twist.linear.x = self.speed
                twist.angular.z = ang
                self.pub.publish(twist)

        except Exception as e:
            rospy.logerr(f"Camera Callback Error: {e}")

    # ============================================================
    # BACK MODE (후진 - 그대로 유지)
    # ============================================================
    def back_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.2:
            twist.linear.x = -0.15
            twist.angular.z = 0.0
            self.pub.publish(twist)
        else:
            angle = self.find_gap_max()
            angle = self.apply_escape_direction_logic(angle)

            self.escape_angle = angle
            self.state = "ESCAPE"
            self.state_start = now
            rospy.loginfo(f"ESCAPE 모드 진입: 각도 {self.escape_angle:.2f}")

    # ============================================================
    # ESCAPE MODE (탈출 - 그대로 유지)
    # ============================================================
    def escape_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.4:
            twist.linear.x = 0.15
            twist.angular.z = self.escape_angle * 1.5 
            self.pub.publish(twist)
        else:
            rospy.loginfo("LANE 모드 복귀 (라바콘/라인 탐색)")
            self.state = "LANE"

    # ============================================================
    # 알고리즘 헬퍼 함수들
    # ============================================================
    def apply_escape_direction_logic(self, angle):
        if self.force_right_escape > 0:
            self.force_right_escape -= 1
            return -0.7 

        if angle > 0: 
            self.left_escape_count += 1
            if self.left_escape_count >= 3:
                self.force_right_escape = 2
                self.left_escape_count = 0
                return -0.7
        else:
            self.left_escape_count = 0
        
        return angle

    def find_gap_max(self):
        if len(self.scan_ranges) == 0:
            return 0.0

        raw = np.array(self.scan_ranges)
        ranges = np.concatenate([raw[-60:], raw[:60]])
        ranges = np.where((ranges < 0.20) | np.isnan(ranges), 0.0, ranges)

        idx = np.argmax(ranges) 
        
        angle_deg = (idx - 60) 
        angle_rad = angle_deg * (np.pi / 180.0)

        return angle_rad

if __name__ == "__main__":
    try:
        node = LineTracerWithObstacleAvoidance()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
