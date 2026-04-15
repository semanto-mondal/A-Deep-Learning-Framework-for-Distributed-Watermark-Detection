class CFG:
    classes = ["original", "watermarked"] 
    dataset_path = "/ibiscostorage/rchandraghosh/video_dataset/video_dataset_new_paper"
    batch_size = 32
    n_frames = 10
    output_size = (224, 224)
    frame_step = 15
    test_size = 0.2
    random_state = 42
    epochs = 100
    model_select = "custom_model"  # options: pre_model/custom_model
    model_name = "resnet50"  # options: vgg16, vgg19, resnet50
    checkpoint_path = "resnet50_model.h5"
