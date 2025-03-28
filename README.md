# YOLO-CBR
Optimizing Lightweight Networks for Small Object Detection in Classroom Behavior Recognition
# Abstract
Accurately recognizing students’ behaviors can be challenging due to the signif-
icant occlusions and inconsistent object scales in classroom behavior images. To
address these issues, we propose YOLO-CBR, a novel classroom behavior detec-
tion model based on YOLOV8. Specifically, we design a novel detection head,
Task-aligned Prediction Head (TAPH), in which the use of shared convolution
and task-aligned structure enhances the model’s ability to recognize small objects.
The network also incorporates the Ghost Dynamic Convolution module (GDC)
and Reparam Dilation-wise Residual (R-DWR) to reduce computational com-
plexity and model size. In GDC, the Ghost Module generates redundant feature
maps cost-effectively, and Conditional Parameterized Convolution (CondConv) is
used to improve inference efficiency. The R-DWR module also reduces the model’s
computational complexity through reparameterization techniques and extracts
multi-scale features using dilation convolutions with multiple dilation rates. In
addition, we construct a Student-Teacher Classroom Behavior dataset (STCB)
covering seven student classroom behaviors and three types of teacher classroom
behaviors, containing 4,242 images and 74,571 annotations. On this dataset, the
YOLO-CBR model achieves a mean average precision (mAP) of 79.2%, with
FLOPs and Params of only 6.9G and 1.8M, respectively. To further validate the
model’s ability to detect small objects in other scenarios, we conducted experi-
ments on the PASCAL VOC 2007 dataset. In conclusion, YOLO-CBR features
a concise yet practical network structure, making it highly capable of accurately
recognizing student classroom behaviors.
