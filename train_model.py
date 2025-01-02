from ultralytics import YOLO

model = YOLO("yolo11n.pt")
print("Starting training...")
model.train(data="C:\\Users\\KIIT0001\\OneDrive\\Documents\\Blind\\data.yaml", epochs=100, imgsz=640)

print("Training complete, saving model...")
model.save("trained_yolo11n.pt")
print("Model saved!")
