from ultralytics import YOLO

def main():
    model = YOLO(r"C:\Users\moham\OneDrive\Desktop\jce\Image_processing\dendrite_segmentation\runs\segment\yolo26_dendrite_tiled_v1\weights\best.pt")

    model.predict(
        source=r"C:\Users\moham\OneDrive\Desktop\jce\Image_processing\dendrite_segmentation\dendrite_dataset\images\val",
        imgsz=896,
        device="cpu",
        conf=0.15,
        show_boxes=False,
        show_labels=False,
        show_conf=False,
        save=True
    )

if __name__ == "__main__":
    main()
