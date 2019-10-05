## Fall-Alert ##

Prerequisites:
- python-3.6 (Preferably 3.6.5)
- pip package manager (To get additional packages) 
- imutils package (pip install imutils) 


Steps:
- Download the yolo-coco folder into the project folder from the following link:
	https://drive.google.com/open?id=1JEqFtpYEEYxJPGBY0cEpoLYZXJQ6CKPE
- Change into project directory.
- Create a Folder named output_video if not already existing.
- Inside the code yolo_video, change authentication settings: in s.login(<sender-email>,<sender-password>) present under 
  authentication. 
- Run the command: python yolo_video.py --input input-video-file.mp4 --yolo yolo-coco.
- Wait for all frames to be processed to find objects.
- Additionally we can also change email body content in the code.
