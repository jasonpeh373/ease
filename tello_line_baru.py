#!/usr/bin/env python

################################################################################
## {Description}: Detecting an Apriltag3
## {Description}: Publish /isApriltag topic
## {Description}: If AprilTag3 detected; /isApriltag --> True
## {Description}: If AprilTag3 detected; /isApriltag --> False
################################################################################
## Author: Khairul Izwan Bin Kamsani
## Version: {1}.{0}.{0}
## Email: {wansnap@gmail.com}
################################################################################

"""
Image published (CompressedImage) from tello originally size of 960x720 pixels
We will try to resize it using imutils.resize (with aspect ratio) to width = 320
and then republish it as Image
"""

# import the necessary Python packages
from __future__ import print_function
import sys
import cv2
import time
import numpy as np
import imutils
import random
import os
import select


# import the necessary ROS packages
from std_msgs.msg import String, Bool, Float32,Empty
from std_msgs.msg import Int16
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
from cv_bridge import CvBridgeError


import rospy

if os.name == 'nt':
	import msvcrt
else:
	import tty, termios


class CameraAprilTag:
	def __init__(self):

		# OpenCV -- ROS
		self.bridge = CvBridge()
		
		self.twist = Twist()
	
		# state
		self.image_received = False

		
		rospy.logwarn("Line Detection Node [ONLINE]...")
		
		# rospy shutdown
		rospy.on_shutdown(self.cbShutdown)
		self.pub_error = rospy.Publisher('error', Int16, queue_size=10)
		self.pub_angle = rospy.Publisher('angle', Int16, queue_size=10)
		# Subscribe to CompressedImage msg
		self.telloImage_topic = "/tello/image_raw"
		self.telloImage_sub = rospy.Subscriber(
						self.telloImage_topic, 
						Image, 
						self.cbImage
						)
						
		# Publish to Twist msg
		self.telloTwist_topic = "/tello/cmd_vel"
		self.telloTwist_pub = rospy.Publisher(
					self.telloTwist_topic, 
					Twist, 
					queue_size=10
					)
		self.pubTakeoff = rospy.Publisher('/tello/takeoff', Empty, queue_size=10)
		self.pubLand = rospy.Publisher('/tello/land', Empty, queue_size=10)
		
		self.Kp = 0.112                 # Ku=0.14 T=6. PID: p=0.084,i=0.028,d=0.063. PD: p=0.112, d=0.084/1. P: p=0.07
		self.Ki = 0
		self.kd = 1
		self.integral = 0
		self.derivative = 0
		self.last_error = 0
		self.Kp_ang = 0.01             # Ku=0.04 T=2. PID: p=0.024,i=0.024,d=0.006. PD: p=0.032, d=0.008. P: p=0.02/0.01
		self.Ki_ang = 0
		self.kd_ang = 0
		self.integral_ang = 0
		self.derivative_ang = 0
		self.last_ang = 0
		self.was_line = 0
		self.line_side = 0
		self.ctrl_c = False
		self.line_back = 1
		self.landed = 0
		self.takeoffed = 0
		self.error = []
		self.angle = []
		self.fly_time = 0.0
		self.start = 0.0
		self.stop = 0.0
		self.velocity = 0.5 #asal 0.2
		self.ang_corr = 0
		
		

		# Allow up to one second to connection
		rospy.sleep(1)
	# Convert image to OpenCV format
	def cbImage(self, msg):

		try:
			# direct conversion to cv2
			self.cv_image = self.bridge.imgmsg_to_cv2(
								msg, 
								"bgr8"
						    		)
		except CvBridgeError as e:
			print(e)
		if self.cv_image is not None:
			self.image_received = True
		else:
			self.image_received = False
	
	def cbZoom(self, scale=20):
		self.image_zoom = self.cv_image.copy()
		self.height, self.width, _ = self.image_zoom.shape
		
		# prepare the crop
		self.centerX, self.centerY = int(self.height / 2), int(self.width / 2)
		self.radiusX, self.radiusY = int(scale * self.height / 100), int(scale * self.width / 100)

		self.minX, self.maxX = self.centerX - self.radiusX, self.centerX + self.radiusX
		self.minY, self.maxY = self.centerY - self.radiusY, self.centerY + self.radiusY

		self.image_zoom = self.image_zoom[self.minX:self.maxX, self.minY:self.maxY]
		self.image_zoom = cv2.resize(self.image_zoom, (self.width, self.height))
		self.image_zoom = cv2.add(self.image_zoom, np.array([-50.0]))
		
	def cbDetect(self):
		self.cbZoom()
		# Info parameters configuration
		fontFace = cv2.FONT_HERSHEY_PLAIN
		fontScale = 0.7
		color = (255, 255, 255)
		colorPose = (0, 0, 255)
		colorIMU = (255, 0, 255)
		thickness = 1
		lineType = cv2.LINE_AA
		bottomLeftOrigin = False # if True (text upside down)
		mask = cv2.inRange(self.image_zoom.copy(), (0,0,0), (175,50,255))
		kernel = np.ones((3, 3), np.uint8)
		mask = cv2.erode(mask, kernel, iterations=5)
		mask = cv2.dilate(mask, kernel, iterations=9)
		contours_blk, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours_blk.sort(key=cv2.minAreaRect)

		
		rospy.loginfo(len(contours_blk))

		if len(contours_blk) > 0 and cv2.contourArea(contours_blk[0]) > 5000:
		# if len(contours_blk) > 0:
			self.was_line = 1
			blackbox = cv2.minAreaRect(contours_blk[0])
			(x_min, y_min), (w_min, h_min), angle = blackbox

			if angle < -45:
				angle = 90 + angle
			if w_min < h_min and angle > 0:
				angle = (90 - angle) * -1
			if w_min > h_min and angle < 0:
				angle = 90 + angle

			setpoint = self.image_zoom.shape[1] / 2
			error = int(x_min - setpoint)
			self.error.append(error)
			self.angle.append(angle)
			normal_error = float(error) / setpoint

			if error > 0:
				self.line_side = 1  # line in right
			elif error <= 0:
				self.line_side = -1  # line in left

			self.integral = float(self.integral + normal_error)
			self.derivative = normal_error - self.last_error
			self.last_error = normal_error


			error_corr = -1 * (self.Kp * normal_error + self.Ki * self.integral + self.kd * self.derivative)  # PID controler
			# print("error_corr:  ", error_corr, "\nP", normal_error * self.Kp, "\nI", self.integral* self.Ki, "\nD", self.kd * self.derivative)

			angle = int(angle)
			normal_ang = float(angle) / 90

			self.integral_ang = float(self.integral_ang + angle)
			self.derivative_ang = angle - self.last_ang
			self.last_ang = angle

			self.ang_corr = -1 * (self.Kp_ang * angle + self.Ki_ang * self.integral_ang + self.kd_ang * self.derivative_ang)  # PID controler

			box = cv2.boxPoints(blackbox)
			box = np.int0(box)

			cv2.drawContours(self.image_zoom, [box], 0, (0, 0, 255), 3)

			cv2.putText(self.image_zoom, "Angle: " + str(angle), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)

			cv2.putText(self.image_zoom, "Error: " + str(error), (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)
			cv2.line(self.image_zoom, (int(x_min), 200), (int(x_min), 250), (255, 0, 0), 3)

			rospy.loginfo("velo:"+ str(self.velocity) + " angle: " + str(angle))
			if angle >= 30 or angle <= -30:
				self.velocity = 0.05
			elif angle <= 29 and angle >= -29:
				self.velocity = 0.3
		
			twist = Twist()
			twist.linear.x = self.velocity
			twist.linear.y = error_corr
			twist.linear.z = 0
			twist.angular.x = 0
			twist.angular.y = 0
			twist.angular.z = self.ang_corr
			self.telloTwist_pub.publish(twist)
			
			ang = Int16()
			ang.data = angle
			self.pub_angle.publish(ang)

			err = Int16()
			err.data = error
			self.pub_error.publish(err)
		
		if len(contours_blk) == 0 :
			twist = Twist()
			twist.linear.x = 0.075
			rospy.loginfo("masuk")
			rospy.loginfo("anglecor: " + str(self.ang_corr))
			self.telloTwist_pub.publish(twist)
			if self.line_side == 1:  # line at the right
				twist.linear.y = -0.2 #asal -0.05
				twist.angular.z = -0.075
				self.telloTwist_pub.publish(twist)
			if self.line_side == -1:  # line at the left
				twist.linear.y = 0.2 #asal 0.05
				twist.angular.z = 0.075
				self.telloTwist_pub.publish(twist)
			
	def distance_to_camera(self, perWidth):
		# compute and return the distance from the maker to the camera
		return (self.knownWidth * self.focalLength) / perWidth
	def detect_landing(self):
		self.image_land = self.cv_image.copy()
		gray_landing_page = cv2.cvtColor(self.image_land,cv2.COLOR_BGR2GRAY)
		orb = cv2.ORB_create(nfeatures=100)
		kpl,des1 = orb.detectAndCompute(gray_landing_page,None)
		bf = cv2.BFMatcher()

		template = cv2.imread('/home/ijud/catkin_ws/src/common_bebop_application/script/land.png')
		gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		kp2, des2 = orb.detectAndCompute(gray_template,None)
		matches = bf.match(des1,des2)
		distance_threshold = 50
		matches = [m for m in matches if m.distance < distance_threshold]
		rospy.loginfo("matches: " + str(len(matches)))
		if len(matches) > 10:
			rospy.loginfo("ada H")
			if(self.line_no > 1500):
				twist = Twist()
				twist.linear.x = 0
				twist.linear.y = 0
				twist.linear.z = 0
				twist.angular.x = 0
				twist.angular.y = 0
				twist.angular.z = self.ang_corr
				self.telloTwist_pub.publish(twist)
				land = Empty()
				self.pubLand.publish(land)

				


	# Show the output frame
	def cbShowImage(self):
		cv2.imshow("Line Detection", self.image_zoom)
#		cv2.imshow("Line Detection Mask", self.thresh)
		cv2.waitKey(1)
	def getKey(self):
		if os.name != 'nt':
			settings = termios.tcgetattr(sys.stdin)

		if os.name == 'nt':
			return msvcrt.getch()

		tty.setraw(sys.stdin.fileno())
		rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
		if rlist:
			key = sys.stdin.read(1)
		else:
			key = ''

		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
		return key
	# Preview image + info
	def cbPreview(self):
		# take off
		key = self.getKey()
		if key == 'v' :
			takeoff = Empty()
			self.pubTakeoff.publish(takeoff)
		
		# land
		elif key == 'b' :
			land = Empty()
			self.pubLand.publish(land)
		if self.image_received:
			self.cbDetect()
			self.cbShowImage()
			self.detect_landing()
		else:
			rospy.logerr("No images recieved")
			
	# rospy shutdown callback
	def cbShutdown(self):
		rospy.logerr("Line Detection Node [OFFLINE]...")
		cv2.destroyAllWindows()

if __name__ == '__main__':
	# Initialize
	rospy.init_node('camera_qr_detection', anonymous=False)
	camera = CameraAprilTag()
	
	r = rospy.Rate(30)
	
	# Camera preview
	while not rospy.is_shutdown():
		camera.cbPreview()
		r.sleep()
