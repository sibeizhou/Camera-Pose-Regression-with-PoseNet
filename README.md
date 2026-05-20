# Camera Pose Regression with PoseNet

本项目实现了一个基于 PoseNet 的 6-DoF camera pose regression 模型。模型以单张 RGB 图像为输入，直接回归相机在三维空间中的位置 `xyz` 和姿态四元数 `wpqr`。

实验使用 Cambridge Landmarks 数据集中的 `KingsCollege` 场景，并以 InceptionV1 / GoogLeNet 风格网络作为 PoseNet 主干。

## 项目亮点

- 使用 PyTorch 搭建 PoseNet 网络结构。
- 支持加载 `places-googlenet.pickle` 预训练权重初始化 backbone。
- 使用三个 pose regression heads 进行中间监督和最终位姿回归。
- 支持训练、checkpoint 保存、测试评估和 ONNX 导出。
- 包含 `KingsCollege` 数据集、训练结果曲线和 Netron 网络结构可视化结果。

## 项目结构

```text
Camera-Pose-Regression-with-PoseNet/
|-- README.md
|-- .gitignore
|-- Description.pdf
|-- training plot.png
|-- workspace/
|   |-- train.py
|   |-- test.py
|   |-- netron.py
|   |-- posenet.onnx
|   |-- posenet.png
|   |-- __init__.py
|   |-- models/
|   |   |-- PoseNet.py
|   |   |-- __init__.py
|   |-- data/
|   |   |-- DataSource.py
|   |   |-- __init__.py
|   |   |-- datasets/
|   |       |-- readme.txt
|   |       |-- KingsCollege/
|   |           |-- dataset_train.txt
|   |           |-- dataset_test.txt
|   |           |-- mean_image.npy
|   |           |-- reconstruction.nvm
|   |           |-- seq1/ ... seq8/
|   |           |-- videos/
|   |-- pretrained_models/
|   |   |-- places-googlenet.pickle
|   |-- checkpoints/
|   |   |-- epoch_*.pth
|   |-- wandb/
|       |-- run-*/
```

### 目录说明

| 路径 | 说明 |
| --- | --- |
| `workspace/train.py` | 训练入口，负责数据加载、模型训练、W&B 记录和 checkpoint 保存。 |
| `workspace/test.py` | 测试入口，加载指定 epoch 的 checkpoint 并计算位置误差和姿态误差。 |
| `workspace/netron.py` | 将 PoseNet 导出为 ONNX，方便使用 Netron 查看模型结构。 |
| `workspace/models/PoseNet.py` | PoseNet 主体实现，包含 Inception block、回归 heads 和损失函数。 |
| `workspace/data/DataSource.py` | KingsCollege 数据读取与图像预处理逻辑。 |
| `workspace/data/datasets/KingsCollege/` | Cambridge Landmarks `KingsCollege` 数据集。 |
| `workspace/pretrained_models/` | InceptionV1 / GoogLeNet 预训练权重。 |
| `workspace/checkpoints/` | 训练过程中保存的 `.pth` 模型权重。 |
| `workspace/wandb/` | Weights & Biases 本地实验日志。 |
| `training plot.png` | 训练损失曲线。 |
| `Description.pdf` | 项目说明或作业要求文档。 |

## 环境要求

建议使用 Python 3.9。核心依赖如下：

```bash
pip install torch==2.2.2 torchvision==0.17.2 numpy==1.23.5 pillow==11.1.0 wandb==0.19.5 onnx==1.17.0
```

如果只想在本地记录 W&B 日志，可以设置离线模式：

```powershell
$env:WANDB_MODE="offline"
```

Linux / macOS:

```bash
export WANDB_MODE=offline
```

## 数据集

项目默认读取以下路径的数据：

```text
workspace/data/datasets/KingsCollege/
```

其中关键文件包括：

- `dataset_train.txt`：训练集图片路径和位姿标签。
- `dataset_test.txt`：测试集图片路径和位姿标签。
- `mean_image.npy`：训练图像均值，用于预处理。
- `seq1/` 到 `seq8/`：图片序列。
- `videos/`：原始视频文件。

每条位姿标签格式为：

```text
filename tx ty tz qw qx qy qz
```

如果 `mean_image.npy` 不存在，`DataSource.py` 会根据训练图像重新计算并保存。

## 模型结构

`workspace/models/PoseNet.py` 中的主要模块：

- `InceptionBlock`：GoogLeNet 风格的多分支卷积模块。
- `PoseNet`：完整位姿回归网络，训练模式下输出三个回归 head。
- `LossHeader`：最终位姿回归 head。
- `LossHeader2`：中间辅助监督 head。
- `PoseLoss`：组合位置 MSE 和四元数 MSE 的加权损失。

训练阶段模型输出：

```text
loss1_xyz, loss1_wpqr,
loss2_xyz, loss2_wpqr,
loss3_xyz, loss3_wpqr
```

测试阶段模型输出：

```text
xyz, wpqr
```

## 训练

进入工作目录：

```bash
cd workspace
```

使用默认参数训练：

```bash
python train.py
```

默认训练参数：

| 参数 | 默认值 |
| --- | --- |
| `epochs` | `200` |
| `batch_size` | `75` |
| `learning_rate` | `0.0001` |
| `save_freq` | `20` |
| `data_dir` | `data/datasets/KingsCollege/` |

也可以显式指定参数：

```bash
python train.py --epochs 200 --batch_size 75 --learning_rate 0.0001 --save_freq 20 --data_dir data/datasets/KingsCollege/
```

训练得到的 checkpoint 会保存到：

```text
workspace/checkpoints/
```

## 测试

使用默认的第 180 个 epoch checkpoint 测试：

```bash
cd workspace
python test.py --epoch 180 --batch_size 75
```

测试脚本会输出每张测试图像的：

- ground truth pose
- predicted pose
- position error，单位为 meter
- orientation error，单位为 degree

当前 README 记录的参考结果：

| 指标 | 结果 |
| --- | ---: |
| Median position error | `3.5984 m` |
| Median orientation error | `2.2383 degrees` |

## ONNX 导出与可视化

重新导出 ONNX：

```bash
cd workspace
python netron.py
```

导出结果：

```text
workspace/posenet.onnx
```

Netron 可视化图片：

```text
workspace/posenet.png
```

## 结果展示

训练损失曲线：

![Training plot](training%20plot.png)

## 版本管理建议

仓库已经包含数据集、checkpoint、ONNX 和 W&B 日志等大文件。新增的 `.gitignore` 会忽略这些生成物和本地缓存，后续建议只把核心代码、README、配置文件和必要的小型结果图提交到 Git。

如果需要保留模型权重或数据集版本，建议使用以下方式之一：

- Git LFS
- 云盘或对象存储
- 单独的 release 附件
- 在 README 中提供下载链接和放置路径

## 注意事项

- `workspace/test.py` 虽然提供了 `--data_dir` 参数，但当前内部实际使用的是固定路径 `data/datasets/KingsCollege/`。
- `workspace/models/PoseNet.py` 默认从 `pretrained_models/places-googlenet.pickle` 加载权重，因此运行脚本时通常需要先进入 `workspace/` 目录。
- 如果 checkpoint 来自不可信来源，加载 `.pth` 文件前需要注意 PyTorch 反序列化安全风险。
- 数据集和模型权重文件较大，不建议直接提交到普通 Git 仓库。

## Reference

Alex Kendall, Matthew Grimes, Roberto Cipolla.

**PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization.**

ICCV 2015.
