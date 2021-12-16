# YogaPose

## Introduction
Yoga pose is [...]

### Data Preparation
Run the script [./get_dataset.sh](https://github.com/michele17284/YogaPose/blob/master/get_dataset.sh). After completion you should get this data configuration. Feel free to interchange the examples images as long as the directory tree remains the same.
There are also the splitted sets into "./data/split/". those splitted images are taken with a per-class splitting of 90-10 from the original dataset
```txt
${POSE_ROOT}/data/
|-- annotations
|   `-- annotations.json
|-- examples
|	|-- 0
|	|   |-- 00000003.jpg
|	|   |-- ... 
|	|-- ...
`-- images
	|-- 0
	    |-- 0000009.jpg
	    |-- ... 
 	|-- ...
```

### Links

Yoga pose By Saxena (Kaggle) resized to match 256x192: [GoogleDrive](https://drive.google.com/file/d/1K-pgnHm6cWfV7q9Enhdgu1yDNGEKG3cH/view?usp=sharing)

TransPose applied to a sample of Saxena's dataset (256x192): [GoogleDrive](https://drive.google.com/file/d/1UcwwjRlyqU9dMQ-R47hZAITcEq2umato/view?usp=sharing)

Models_logs: [GoogleDrive](https://drive.google.com/file/d/1IM2KJl265Tpm-IekxzuAccuX5duorIT2/view?usp=sharing)

### Acknowledgements

Great thanks for this paper and his open-source codes：[TransPose](https://github.com/yangsenius/TransPose)
