# Project Write-Up-People Counter Application

## Here all the codse section are refered udacity Intel AI  course material and inel site refrence with logic implemnt

  This people counter application will work on Edge device so it need small size application &  memory  and adobtable latency & accuracy. It is used for simple human counter and duration of human on that video or image idetification.
  
############ Special note:
Due to large size of downloaded model only link is mention and not added downloaded.

## Explaining Custom Layers
Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

When implementing a custom layer for your pre-trained model in the Intel® Distribution of OpenVINO™ toolkit, you will need to add extensions to both the Model Optimizer and the Inference Engine.


    Custom Layer Extractor
        Responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer. The layer parameters are stored per instance and used by the layer operation before finally appearing in the output IR. Typically the input layer parameters are unchanged, which is the case covered by this tutorial.
    Custom Layer Operation
        Responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters.
        The --mo-op command-line argument shown in the examples below generates a custom layer operation for the Model Optimizer.
###Models to evaluvate
i choose 3 model to evaluvate 
a.ssd_mobilenet_v2_oid_v4_2018_12_12
b.ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
c.models_VGGNet_coco_SSD_300x300


I will choose first prefernce as intel pretrained model and second as sd_mobilenet_v2_oid_v4_2018_12_12

# 



### Comparing Model Performance
Details given on every model section under


#### Assess Model Use Cases

Some of the potential use cases of the people counter app are...

1.big store security servilance as number of people in and out
    where 24*7 shopshow many people are in for shoping and what kind of product where how many times will help intrest of people 
2.Visitor counting on the event/Time series analysis
  Any public events/seminar/conferenmce how many people attend and pattern of the people for coming
3.Public transport in/out as per allowed number of people
  if ope ticket pulic allowed system it can guide how many more people allowed or not

4.Canteen management 
  if people count comes to know we can prepare food as per no of people to avoid wastage.
  



Each of these use cases would be useful because...

##### Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on application but it may be fixed it are one time installation
deployed edge model. The potential effects of each of these are as follows... Without lighting analysis will effect and model accuracy depend how much it was trained.
camera focal length/image size have different effects due to nonfocus /distorted images and thersold mismatching alerts.

###### Model Research
###cloud 				vs	 Edge

###cloud
1. some delay procee input and detect condition
2.simple one platform all connected device so update and maintain easy
3.remote condition handling hard where connectivity breaks
4.connected cost also involved
5 if anyissue in connctivity or hardwar complete network will collapse.

##edge

1.Time sensitive data and instant detect
2. can be connected to cloud though gateway for one time preproces 
3.effective on remote locations and emergncy detect
4.if any of the issue comes only one device will fail



##for Model  i am consider  this page for tensorflow model:
<details>
  <summary>Source</summary>
  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
</details>


##for caffe model i took this page:
<details>
  <summary>Source</summary>
  https://github.com/weiliu89/caffe/tree/ssd
</details>

download link:
<details>
  <summary>Source</summary>
  https://drive.google.com/uc?id=0BzKzrI_SkD1_dUY1Ml9GRTFpUWc&export=download
</details>


In investigating potential people counter models, I tried each of the following three models:

- Model 1: ssd_mobilenet_v2_oid_v4_2018_12_12
  -Model Source download link:
  <details>
  <summary>Source</summary>
  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz
</details>
  
  
  - to unzip following command excuted:
  '''
  tar -xvf ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz
  '''
  - I converted the model to an Intermediate Representation with the following arguments...  first come to the model directory
  '''
  cd /home/workspace/ssd_mobilenet_v2_oid_v4_2018_12_12
  '''
  
  
  - run next command for openvino inference as follow
  '''
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  '''
  - to run the code i choose below command
    Open a new terminal to run the code. 
    Setup the environment

    You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:

    source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
    
    next commad:
    '''
    python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mobilenet_v2_oid_v4_2018_12_12/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    '''
    
    to view results used udacity workspace cloud link and it can not be access outside here i am attached images for refrence. I am generated one folder test_out_imaqges and put result images.
    image name as follow modelname +count increment wise.
    
    
    
   
   
  - i used udacity workshop this project 
  
  - The model was insufficient for the app because...
  It  accuracy very low and inference time is high with comparision of intel open vino model-person-detection-retail-0013
  - I tried to improve the model for the app by...
   variating probe thersold value
   if i increase value it not detect person else it detects multple person  and total count accuracy and duration time are get worst worst.
   
   The difference between model accuracy pre- and post-conversion was...
   0.78 & 0.45

   The size of the model pre- and post-conversion was...131MB,186MB

   The inference time of the model pre- and post-conversion was...
   389ms & 69ms
   
  
- Model 2: ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
  - Model Source:
  <details>
  <summary>Source</summary>
  http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
</details>
 
  
  To unzip use following command:
  tar -xvf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
  
  
  
  - I converted the model to an Intermediate Representation with the following arguments...  first come to the model directory
  
  cd /home/workspace/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
  
  run next command for openvino inference as follow
 ''' 
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  '''
  
  
   
   
   - to run the code i choose below command
    
    Open a new terminal to run the code. 
    Setup the environment

    You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:

    source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
    
    next commad:
    python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    
      
  To view results used udacity workspace cloud link and it can not be access outside here i am attached images for refrence. I am generated one folder test_out_imaqges and put result images.
    image name as follow modelname +count increment wise.
    
    
   
    - The model was insufficient for the app because...
    It  accuracy very low and inference time is high with comparision of intel open vino model-person-detection-retail-0013
  - I tried to improve the model for the app by...
  variating probe thersold value
   if i increase value it not detect person else it detects multple person  and total count accuracy and duration time are get worst worst.very latency and slow fps process
   
   
  The difference between model accuracy pre- and post-conversion was... 
 75.8 & 45.2
 
    The inference time of the model pre- and post-conversion was...
 2800ms & 2612ms

    The size of the model pre- and post-conversion was...
258MB,456MB


- Model 3: Name:models_VGGNet_coco_SSD_300x300
- Model Source:
   To download file we need to run below commands
   export fileid=0BzKzrI_SkD1_dUY1Ml9GRTFpUWc
export filename=models_VGGNet_coco_SSD_300x300.tar.gz


wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)

after downloaded unzip as follow
  
 
    unzip command as follow 
  - tar -xvf models_VGGNet_coco_SSD_300x300.tar.gz 
   it will unzip in models folder
  
   it will unzip in models folder
   
   - I converted the model to an Intermediate Representation with the following arguments...  first come to the model directory
  
  cd /home/workspace/models/VGGNet/coco/SSD_300x300
  - run next command for openvino inference as follow
   python /opt/intel/openvino/deployment_tools/model_optimizer/mo_caffe.py  --input_model /home/workspace/models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel --input_proto deploy.prototxt
   - to run the code i choose below command
  
    Open a new terminal to run the code. 
    Setup the environment

    You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:

    source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
    
    next commad:
    python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    
   
   
   
  _ 
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
   It  accuracy very low and inference time is high with comparision of intel open vino model-person-detection-retail-0013
  - I tried to improve the model for the app by...
   variating probe thersold value
   if i increase value it not detect person else it detects multple person  and total count accuracy and duration time are get worst worst.very latency and slow fps process
  
  To view results used udacity workspace cloud link and it can not be access outside here i am attached images for refrence. I am generated one folder test_out_imaqges and put result images.
    image name as follow modelname +count increment wise.
   
   
   
   
The inference time of the model pre- and post-conversion was...
1108ms & 968ms

The size of the model pre- and post-conversion was...
132MB & 263MB
The difference between model accuracy pre- and post-conversion was...
75.7 & 78

in comparision of all above 3 model i am found that
Intel OpenVino pretrained model:Person-detection-retail-0013 have lower inference time 40ms and high fps with better accuracy.

To download pretrained model Person-detection-retail-0013 follow below command

cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader

./downloader.py --name person-detection-retail-0013 -o /home/workspace

to run using pretraind intel open vino downloaded as
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

    -I am added test_images.zip  for test result screen shot images.
    Written by
    K S RAJANBABU