# COSC428_Hand_Detection
Computer Vision Project for Univeristy of Canterbury

Code structure:

	removeBG(frame)					# apply background subtraction
	calculateFingers(res, drawing)	# Count the finger, draw it
	read_key()						# Wait input
	get_frame(camera)				# Read from webcam draw ROI
	frame_process(frame_BG)			# Apply filter and threshold
	main							# 

How to Run the Code:
	1. run main.py
	2. Press b to trigger the background subtraction
	3. Make sure no hand in the frame when press b
	4. Esc to quit the program
