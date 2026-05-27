from ultralytics import YOLO

def main():
    #Config
    name= 'yolo11_vntraffic_merged_32classes_dataset'
    data= "../dataset/merge_Real-time_Viet_Nam_Traffic_Dtctn.v2i.yolov11/data.yaml"
    print("Train Process start! Model: YOLOv11s")
    model = YOLO("yolo11s.pt")
    model.train(data=data,
                name=name,
                epochs=200,
                imgsz=640,
                batch=16,
                amp=True,
                device=0,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                mixup=0.1,
                copy_paste=0.1,
                mosaic=1.0,
                scale=0.5,
                fliplr=0.0,
                close_mosaic=30,
                workers=4,
                patience=50,
                dropout=0.2,
                resume=False,
                weight_decay=0.0005
                )

if "__main__" == __name__:
    main()