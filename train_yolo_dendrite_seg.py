from ultralytics import YOLO

# weights of yolo26
model = YOLO('yolo26n-seg.pt')

results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,  
    device=0,
    overlap_mask=True,
    mask_ratio = 4,
    optimize='auto',
    project='runs/segment',
    name='yolo26_dendrite_v1'
    )

