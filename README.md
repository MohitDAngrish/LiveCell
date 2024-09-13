# LiveCell

In this exercise, i have used SkBr3 cell for training as in my analysis i have found that this data has the least missing annotations compared to the other cells.

After your response of the mail, i have even tried removing the images with large number of missing annotation (~1k from train set of LiveCell Data). The mask mAP improved by 3% even after a droping ~27% data points. I could have gotten better metric if i would have used sqaured image as input(explained below in training experiment).


## Data Distribution

| Sample |  Count  | 
|:-------|:--------|
| Train  |     443 |
| Val    |      85 |
| Test   |     176 |

I have used this open-source repo to convert coco-json into yolo format [json2yolo](https://github.com/ultralytics/JSON2YOLO)


## Training Experiment

For training i have used yolov8 and yolov9. Since it is a medical usecase we would prefer higher mAP(mask) model over a faster model with lower mAP(mask).

For benchmarking, the confidence threshold used was 0.4, and iou threshold for nms was 0.6

| Model Arch        | bbox (mAP) |  mask(mAP)|
|:------------------|:-----------|-----------|
| detectron2        |   *64.35*  |   *65.85* |
| yolov8x (520*704) |     66.0   |     53.5  |
| yolov8x (704*704) | **65.1**   | **58.2**  |
| yolov9e (704*704) |     62.1   |     57.9  |

It is mentioned in the yolo docs that the model have higher accuracy when trained with sqaure image when compared to rectangle images. This can also be seen in the above table, model with square image input (704 * 704) has higher mAp when compared to rectangular image input (520 * 704).

## Deployment

Deployment of endpoint is done using docker in local. I have used fast api to spwan the server which takes image as an input and returns an image with outline of each cell predicted by the model.

`docker compose up --build` to build the image and run the api server after going inside the workspace

<img width="1518" alt="Screenshot 2024-09-13 at 8 18 49 PM" src="https://github.com/user-attachments/assets/1da88bcd-8091-4977-b2af-732a87af2410">

Output of endpoint for a sample image from test set.
<img width="1065" alt="Screenshot 2024-09-13 at 8 25 11 PM" src="https://github.com/user-attachments/assets/625c8702-4e60-4fa0-9c34-8a5ad3f1fae0">



### Alternative Approach
Deploying model as a seperate endpoint using triton server, after converting the model into tensorrt format or any other platform support by triton server. Here model will be a different service of its own and can be consumed by anyone, wehreas in the first case the model is tightly coupled with the fast api service. In the interest of time i have not implemented this approach.

In this approach:

* The request will first go to fast api endpoint which would read the image and do the pre processing required for the image.
* After preprocessing, there will be an invokation to the triton server which is known for efficient inferecing of AI models.
* Triton server would return the model response and we will do the final post processing of image and finally return the image with outline of each cell as output.
  
        

