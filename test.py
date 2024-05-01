from ultralytics import YOLO
yaml_file_combined = "yaml_files/combined.yaml"
yaml_file_ball = "yaml_files/ball.yaml"
yaml_file_player = "yaml_files/player.yaml"

test_combined = "yaml_files/test_files/test_combined.yaml"
test_ball = "yaml_files/test_files/test_ball.yaml"
test_players = "yaml_files/test_files/test_players.yaml"

model_combined = YOLO('yolov8n.pt')
model_ball = YOLO('yolov8n.pt')
model_players = YOLO('yolov8n.pt')
model_combined = YOLO('/work/mbergst/TDT4265_Project/runs/detect/combined/train23/weights/best.pt')
test_combined = model_combined.val(data=test_combined, batch=14, imgsz=(1920, 1080), project = '/work/mbergst/TDT4265_Project/runs/detect/player/test')