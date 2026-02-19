from ultralytics import YOLO

def main():
    model = YOLO('yolo26n-seg.pt')
    model.train(
        data='dendrite_dataset_tiled/data.yaml',
        epochs=300,
        batch=2,
        imgsz=896,
        device=0,
        overlap_mask=True,
        mask_ratio=4,
        name='yolo26_dendrite_tiled_v1',
        workers=0
    )

if __name__ == "__main__":
    main()
