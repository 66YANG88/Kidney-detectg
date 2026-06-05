# 基于肾CT影像的囊性病变检测 — nnU-Net全流程技术文档

> 本文档结合 nnU-Net v2 原论文原理与本项目实际代码，对数据准备、数据指纹提取、实验规划、数据预处理、模型架构、模型训练、模型推理及可视化系统进行系统性、深度性介绍，供毕业论文撰写参考。

---

## 目录

1. [项目背景与总体架构](#1-项目背景与总体架构)
2. [nnU-Net 核心设计哲学](#2-nnu-net-核心设计哲学)
3. [环境配置与目录结构](#3-环境配置与目录结构)
4. [数据集准备与格式转换](#4-数据集准备与格式转换)
5. [数据指纹提取（Dataset Fingerprint）](#5-数据指纹提取dataset-fingerprint)
6. [实验规划（Experiment Planning）](#6-实验规划experiment-planning)
7. [数据预处理（Preprocessing）](#7-数据预处理preprocessing)
8. [模型架构（Network Architecture）](#8-模型架构network-architecture)
9. [模型训练（Training）](#9-模型训练training)
10. [模型推理（Inference）](#10-模型推理inference)
11. [后处理与结果评估](#11-后处理与结果评估)
12. [医学影像可视化系统](#12-医学影像可视化系统)
13. [关键技术总结](#13-关键技术总结)

---

## 1. 项目背景与总体架构

### 1.1 研究背景

肾脏囊性病变（Kidney Cyst）是肾脏最常见的病变类型之一，其在成年人群中发病率可高达 50% 以上。CT 影像是临床诊断肾脏囊性病变的主要手段。然而，依赖放射科医生的人工阅片不仅耗时耗力，在大规模筛查场景下效率低下，且存在一定的主观误差。

基于深度学习的自动分割方法可从体素级别对肾脏及囊性病变进行精确定位与勾画，有望显著提升临床诊断效率。本项目以 **nnU-Net v2**（自适应医学图像分割框架）为核心，面向肾脏 CT 三维影像，实现了：
- **肾实质分割**（标签 1：Kidney）
- **肾脏囊性病变检测与分割**（标签 2：Kidney Cyst）

### 1.2 总体流程

```
原始数据集
    │
    ▼
数据格式转换 (convert_data.py / fix_and_convert_data.py)
    │  → nnUNet_raw/Dataset001_KidneyCyst/
    │     ├── imagesTr/  (79 例训练图像, *_0000.nii.gz)
    │     ├── labelsTr/  (79 例标签, *.nii.gz)
    │     ├── imagesTs/  (20 例测试图像)
    │     └── dataset.json
    │
    ▼
数据指纹提取 (nnUNetv2_extract_fingerprint)
    │  → nnUNet_preprocessed/Dataset001_KidneyCyst/dataset_fingerprint.json
    │
    ▼
实验规划 (nnUNetv2_plan_experiment)
    │  → nnUNet_preprocessed/Dataset001_KidneyCyst/nnUNetPlans.json
    │     (自动确定: 目标间距、patch size、batch size、网络拓扑)
    │
    ▼
数据预处理 (nnUNetv2_preprocess)
    │  → nnUNet_preprocessed/Dataset001_KidneyCyst/nnUNetPlans_3d_fullres/
    │     (重采样 + CT 窗口化归一化 + 裁剪 + 压缩存储)
    │
    ▼
模型训练 (nnUNetv2_train)
    │  → nnUNet_results/Dataset001_KidneyCyst/
    │     nnUNetTrainer__nnUNetPlans__3d_fullres/
    │     fold_0/ (checkpoint_best.pth, checkpoint_final.pth)
    │
    ▼
模型推理 (run_inference.py / medical_viewer)
    │  → 对测试集 case_3_0000.nii.gz 等执行滑动窗口推理
    │     输出三类分割图: 0=背景, 1=肾脏, 2=肾脏囊肿
    │
    ▼
医学影像可视化系统 (medical_viewer/app.py)
       → Flask Web 应用，多平面重建 + 三维渲染 + 覆盖显示
```

### 1.3 数据集规模

| 子集 | 数量 | 说明 |
|------|------|------|
| 训练集 imagesTr | 79 例 | 用于 5-fold 交叉验证训练 |
| 测试集 imagesTs | 20 例 | 用于最终推理评估 |
| 总计 | 99 例 | 3D 腹部 CT，单通道（CT 模态） |

标签体系：
- **0**：背景（Background）
- **1**：肾脏（Kidney）
- **2**：肾脏囊肿（Kidney Cyst）

---

## 2. nnU-Net 核心设计哲学

### 2.1 自动化分割管线（Self-configuring Pipeline）

nnU-Net（no-new-U-Net）由德国癌症研究中心（DKFZ）的 Fabian Isensee 等人于 2021 年发表于 *Nature Methods*，其核心创新在于**自适应的流水线配置机制**，彻底摆脱了手工调参的依赖。

nnU-Net 将分割管线的参数划分为三类：

| 参数类型 | 含义 | 示例 |
|--------|------|------|
| **固定参数（Fixed）** | 经充分验证不需调整的超参数 | 损失函数、学习率初始值、优化器类型 |
| **基于规则的参数（Rule-based）** | 由数据指纹通过启发式规则自动推导 | patch size、batch size、网络深度、池化层数 |
| **经验参数（Empirical）** | 通过对比实验选定 | 最优配置（2D/3D/级联）、后处理策略 |

### 2.2 三种 U-Net 配置

nnU-Net 可自动生成以下配置，并根据数据集特征决定实际使用哪些：

| 配置名 | 说明 |
|-------|------|
| `2d` | 将3D图像按切片训练2D U-Net |
| `3d_fullres` | 在原始分辨率下的3D U-Net（本项目使用此配置） |
| `3d_lowres` | 在降低分辨率下的3D U-Net（图像较大时使用） |
| `3d_cascade_fullres` | 级联3D U-Net（低分辨率预测+高分辨率细化） |

**本项目最终采用 `3d_fullres` 配置**，原因是肾脏 CT 数据集图像尺寸适中，全分辨率 3D U-Net 的 patch size 已能覆盖足够大的感受野，无需级联方案。

### 2.3 核心论文贡献总结

nnU-Net 的成功来自以下几点系统性设计：

1. **数据指纹驱动（Data Fingerprint）**：通过统计分析数据集几何属性（体素间距、图像尺寸）和强度属性（前景像素强度分布），作为所有下游决策的统一输入。

2. **自动网络拓扑设计**：根据 patch size 与体素间距比例，动态计算每个轴向的最优池化次数，使特征图在最深层不小于 $4 \times 4 \times 4$。

3. **混合损失函数**：固定使用 Dice Loss 与 Cross-Entropy Loss 的线性组合，对类别不平衡具有天然鲁棒性。

4. **深度监督（Deep Supervision）**：在解码器的每个分辨率层级引入辅助损失，以指数衰减权重加权，加速梯度传播、提升收敛稳定性。

5. **大规模数据增强**：包含空间变换（旋转、缩放、弹性形变）和强度变换（高斯噪声、模糊、亮度对比度调整、Gamma变换），极大地提升了模型泛化能力。

---

## 3. 环境配置与目录结构

### 3.1 软件环境

| 组件 | 版本 |
|------|------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1.2（本项目使用 CUDA 加速） |
| nnU-Net | v2.6.2 |
| SimpleITK | ≥ 2.2.1（医学图像 I/O） |
| Flask | 最新稳定版（Web 服务） |

### 3.2 路径环境变量

nnU-Net 通过三个环境变量管理数据路径（见 `setup_env.sh`）：

```bash
export nnUNet_raw="/data/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="/data/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="/data/nnUNet_data/nnUNet_results"
```

| 变量 | 用途 |
|------|------|
| `nnUNet_raw` | 原始格式化数据集存放目录 |
| `nnUNet_preprocessed` | 预处理后数据及规划文件存放目录 |
| `nnUNet_results` | 训练结果（模型权重、日志）存放目录 |

### 3.3 项目文件结构说明

```
/data/
├── nnUNet-master/               # nnU-Net v2 框架源码
│   ├── nnunetv2/
│   │   ├── experiment_planning/ # 指纹提取与实验规划
│   │   ├── preprocessing/       # 重采样与归一化
│   │   ├── training/            # 训练器、损失函数、数据增强
│   │   ├── inference/           # 推理预测器
│   │   └── ...
│   ├── convert_data.py          # 数据格式转换（基础版）
│   ├── fix_and_convert_data.py  # 数据格式转换（含标签几何对齐修复）
│   └── setup_env.sh             # 环境变量配置脚本
│
├── nnUNet_data/
│   ├── nnUNet_raw/Dataset001_KidneyCyst/   # 原始格式数据
│   ├── nnUNet_preprocessed/Dataset001_KidneyCyst/ # 预处理数据+规划
│   ├── nnUNet_results/Dataset001_KidneyCyst/      # 训练结果
│   └── window_inference_output/            # 推理输出（Web端）
│
├── run_inference.py             # 单案例命令行推理脚本
├── split_data.py                # 训练集/测试集划分脚本
└── medical_viewer/              # Flask 可视化 Web 系统
    ├── app.py                   # 后端服务（推理+渲染）
    ├── templates/index.html     # 前端页面
    └── static/                  # CSS/JS 静态资源
```

---

## 4. 数据集准备与格式转换

### 4.1 nnU-Net 数据格式规范

nnU-Net 要求数据以如下目录结构组织：

```
Dataset{ID:03d}_{Name}/
├── dataset.json          # 元数据文件
├── imagesTr/             # 训练图像
│   ├── case_1_0000.nii.gz   # 命名规则: {case_id}_{channel_idx:04d}.nii.gz
│   ├── case_2_0000.nii.gz
│   └── ...
├── labelsTr/             # 训练标签（多类别整数掩码）
│   ├── case_1.nii.gz
│   └── ...
└── imagesTs/             # 测试图像（无标签）
```

**命名约定**：
- `_0000` 后缀代表第 0 个输入通道（CT 为单通道）；若为多模态（如 MRI T1+T2），则同一病例有 `_0000.nii.gz`、`_0001.nii.gz` 等多个文件。
- 标签文件与图像文件同名（去掉 `_0000`），为整数类型 NIfTI。

### 4.2 dataset.json 结构解析

```python
dataset_json = {
    "channel_names": {
        "0": "CT"            # 单通道 CT 影像
    },
    "labels": {
        "background":    0,  # 背景
        "Kidney":        1,  # 肾脏实质
        "Kidney Cyst":   2   # 肾脏囊肿
    },
    "numTraining": 79,       # 训练案例数
    "file_ending": ".nii.gz",
    "name": "KidneyCyst",
    "description": "Kidney Cyst Segmentation"
}
```

`channel_names` 中的模态名至关重要：nnU-Net 依据此字段自动选择归一化策略。当模态为 `"CT"` 时，系统自动选用 **CT 归一化**（基于前景强度的百分位裁剪 + Z-score 标准化）。

### 4.3 数据转换脚本详解

**基础版（convert_data.py）**：
1. 扫描 `/root/dataset/label/` 目录中所有 `.nii.gz` 图像文件
2. 逐个匹配 `/root/dataset/mask_gt/` 中对应的标签文件（命名规则 `{id}_seg.nii.gz`）
3. 重命名为 nnU-Net 标准格式（`case_{id}_0000.nii.gz`）并 `shutil.copy2` 复制
4. 生成 `dataset.json`

**增强版（fix_and_convert_data.py）**：

基础版的问题在于当图像和标签的 **几何信息（spacing/origin/direction）不一致**时，nnU-Net 在加载时会因坐标系不匹配导致错误。增强版额外执行了**标签重采样对齐**操作：

```python
def process_case(case_info):
    img = sitk.ReadImage(src_img_path)
    seg = sitk.ReadImage(src_label_path)
    
    # 以图像为参考，对标签执行最近邻插值重采样
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # 最近邻：保证整数标签不失真
    resampler.SetDefaultPixelValue(0)
    resampled_seg = resampler.Execute(seg)
    
    shutil.copy2(src_img_path, dst_img_path)  # 图像直接复制
    sitk.WriteImage(resampled_seg, dst_label_path)  # 保存对齐后标签
```

**关键设计**：标签使用**最近邻插值（Nearest Neighbor）** 而非线性插值，确保标签值保持为整数（0/1/2），避免出现小数类别值。图像可安全使用三次样条或线性插值（用于后续重采样），但此处直接复制原始图像，间距信息由 nnU-Net 在预处理阶段统一处理。

### 4.4 数据集划分

`split_data.py` 将全部 99 例数据划分为：
- **训练集**：前 79 例（按文件名字典序排列）→ `imagesTr/` + `labelsTr/`
- **测试集**：后 20 例 → `imagesTs/`（标签移至 `labelsTs/` 保存以备评估）

```python
test_cases = cases[-20:]    # 后20例作为测试集
train_cases = cases[:-20]   # 前79例作为训练集
```

---

## 5. 数据指纹提取（Dataset Fingerprint）

### 5.1 概念与意义

**数据指纹（Dataset Fingerprint）** 是 nnU-Net 一切自动化决策的基础。它描述了数据集的核心统计特征，包括：
- 体素间距（Spacing）分布
- 裁剪后图像尺寸（Shape after crop）分布
- 前景像素强度统计（用于归一化参数）
- 裁剪后图像相对原图大小的比例中位数

执行命令：
```bash
nnUNetv2_extract_fingerprint -d 001 -np 8
```

### 5.2 核心实现：DatasetFingerprintExtractor

源码文件：`nnunetv2/experiment_planning/dataset_fingerprint/fingerprint_extractor.py`

**分析流程（针对每个训练案例）**：

#### Step 1：加载图像与标签

```python
images, properties_images = rw.read_images(image_files)   # shape: (C, X, Y, Z)
segmentation, properties_seg = rw.read_seg(segmentation_file)  # shape: (1, X, Y, Z)
```

#### Step 2：裁剪非零区域（crop_to_nonzero）

```python
data_cropped, seg_cropped, bbox = crop_to_nonzero(images, segmentation)
```

将图像中全为零的边缘区域裁去，只保留包含有意义体素的最小包围盒。这是因为 CT 图像通常在患者体外区域为零，裁剪后显著减少无用背景区域，提升统计精度和后续处理效率。

#### Step 3：前景强度采样

```python
@staticmethod
def collect_foreground_intensities(segmentation, images, seed=1234, num_samples=10000):
    foreground_mask = segmentation[0] > 0  # 前景掩码（非背景体素）
    for i in range(len(images)):
        foreground_pixels = images[i][foreground_mask]
        # 有放回采样，防止前景体素极少的案例被低估
        intensities_per_channel.append(
            rs.choice(foreground_pixels, num_samples, replace=True)
        )
```

对每个案例采集 `num_samples=10000` 个前景体素的强度值，整个数据集总采样量约 $10^7$，在精度与内存之间取得平衡。

#### Step 4：汇聚全局统计量

```python
fingerprint = {
    "spacings": spacings,           # 每个案例的体素间距列表 [(sz,sy,sx), ...]
    "shapes_after_crop": shapes_after_crop,   # 裁剪后各案例形状
    "foreground_intensity_properties_per_channel": {
        0: {
            "mean": ...,
            "median": ...,
            "std": ...,
            "min": ...,
            "max": ...,
            "percentile_99_5": ...,   # 用于CT裁剪上界
            "percentile_00_5": ...,   # 用于CT裁剪下界
        }
    },
    "median_relative_size_after_cropping": ...  # 裁剪后相对原图大小的中位数
}
```

### 5.3 指纹信息的作用映射

| 指纹字段 | 用途 |
|---------|------|
| `spacings` | 计算目标重采样间距（中位数间距） |
| `shapes_after_crop` | 计算重采样后的中位图像尺寸，作为 patch size 上界 |
| `foreground_intensity_properties` | CT 归一化的裁剪范围（percentile_00_5 ~ percentile_99_5）和 Z-score 参数（mean, std） |
| `median_relative_size_after_cropping` | 决定是否对前景区域使用掩码归一化（比例 < 3/4 则使用） |

---

## 6. 实验规划（Experiment Planning）

### 6.1 总体流程

实验规划阶段将数据指纹"翻译"为具体的实验配置，生成 `nnUNetPlans.json`。

执行命令：
```bash
nnUNetv2_plan_experiment -d 001 -pl ExperimentPlanner
```

核心类：`ExperimentPlanner`（`experiment_planning/experiment_planners/default_experiment_planner.py`）

### 6.2 目标间距确定（determine_fullres_target_spacing）

目标间距决定了图像被重采样到的分辨率，对计算量和分割精度有决定性影响。

```python
def determine_fullres_target_spacing(self) -> np.ndarray:
    spacings = np.vstack(self.dataset_fingerprint['spacings'])
    target = np.percentile(spacings, 50, 0)  # 各轴中位数间距
```

**各向异性处理**：对于 Z 轴分辨率极低（如腹部 CT 中层厚 5mm 而面内分辨率 0.7mm）的数据集，中位数会保留较大间距，可能导致插值伪影。nnU-Net 通过各向异性判断逻辑自动修正：

```python
has_aniso_spacing = target[worst_spacing_axis] > (ANISO_THRESHOLD * max(other_spacings))
has_aniso_voxels  = target_size[worst_spacing_axis] * ANISO_THRESHOLD < min(other_sizes)
if has_aniso_spacing and has_aniso_voxels:
    # 取第10百分位数（更细的间距），防止重采样插值失真过大
    target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
```

`ANISO_THRESHOLD = 3`（来自 `nnunetv2/configuration.py`），即当最差轴间距超过其他轴最大间距的 3 倍，且对应维度体素数也明显少于其他轴时，判定为各向异性数据集，使用更保守的间距估计。

### 6.3 轴向转置（determine_transpose）

nnU-Net 通过重排轴的顺序使间距最大的轴排在第一位，从而使后续规划（如各向异性数据增强的选择）更为统一：

```python
max_spacing_axis = np.argmax(target_spacing)
remaining_axes   = [i for i in list(range(3)) if i != max_spacing_axis]
transpose_forward = [max_spacing_axis] + remaining_axes
```

这确保了在各向异性数据集中，间距最大（分辨率最低）的轴始终排在第一位，便于后续的 2D 虚假数据增强逻辑统一处理。

### 6.4 网络拓扑与 Patch Size 规划（get_plans_for_configuration）

这是实验规划中最复杂也最关键的部分，实现了**自动化的网络深度和 patch size 联合优化**。

**初始 patch size**（基于间距的宽高比推导）：

```python
# 3D配置：以间距反比为宽高比，等比缩放使总体积接近 256^3
tmp = 1 / np.array(spacing)
initial_patch_size = [round(i) for i in tmp * (256**3 / np.prod(tmp))**(1/3)]
# 不超过中位图像尺寸
initial_patch_size = np.array([min(i, j) for i, j in zip(initial_patch_size, median_shape)])
```

**网络拓扑推导（get_pool_and_conv_props）**：

```python
def get_pool_and_conv_props(spacing, patch_size, min_feature_map_size=4, max_numpool=999999):
    while True:
        valid_axes = [i for i in range(dim) if current_size[i] >= 2 * min_feature_map_size]
        if len(valid_axes) < 1:
            break
        # 只对间距相差不超过2倍的轴同时下采样（避免过度插值各向异性轴）
        valid_axes = [i for i in valid_axes if current_spacing[i] / min_spacing < 2]
        
        for v in valid_axes:
            pool_kernel_sizes[v] = 2   # 2x池化
            current_spacing[v] *= 2
            current_size[v] = ceil(current_size[v] / 2)
        ...
```

算法保证：
1. 任意轴的特征图边长不小于 `min_feature_map_size = 4`
2. 间距相差超过 2 倍的轴暂不下采样（保持各向异性）
3. 卷积核在各轴间距差距在 2 倍以内时才从 1×1×1 切换为 3×3×3

**显存估算与迭代削减**：

```python
# 估算当前网络+patch size的显存消耗
estimate = self.static_estimate_VRAM_usage(patch_size, ...)

reference = UNet_reference_val_3d * (vram_target_GB / 8)  # 线性缩放到目标显存

while (estimate / ref_bs * 2) > reference:
    # 找到相对于中位形状比例最大的轴，优先削减
    axis_to_reduce = np.argsort([i/j for i,j in zip(patch_size, median_shape)])[-1]
    patch_size[axis_to_reduce] -= shape_must_be_divisible_by[axis_to_reduce]
    # 重新计算拓扑
    ...
```

**批大小确定**：

```python
batch_size = round((reference / estimate) * ref_bs)
# 防止过拟合：单次前向传播不超过 5% 数据集体素
bs_5pct = round(approximate_n_voxels_dataset * 0.05 / np.prod(patch_size))
batch_size = max(min(batch_size, bs_5pct), UNet_min_batch_size)  # 最小batch=2
```

### 6.5 低分辨率配置判断（3d_lowres 触发逻辑）

```python
# 若全分辨率patch覆盖体素数 < 中位图像体素数的 25%，则需要级联低分辨率配置
while num_voxels_in_patch / median_num_voxels < self.lowres_creation_threshold:  # 0.25
    lowres_spacing *= 1.03  # 逐步增大间距（降低分辨率）
    ...
```

由于本项目肾脏 CT 数据集图像尺寸适中，3d_fullres 的 patch size 已覆盖超过 25% 的体素，**故不生成 3d_lowres 配置**。

### 6.6 规划结果（nnUNetPlans.json 内容）

规划完成后生成包含如下关键字段的 JSON 文件（以本项目 3d_fullres 配置为参考说明）：

```json
{
  "dataset_name": "Dataset001_KidneyCyst",
  "plans_name": "nnUNetPlans",
  "transpose_forward": [0, 1, 2],
  "transpose_backward": [0, 1, 2],
  "configurations": {
    "3d_fullres": {
      "data_identifier": "nnUNetPlans_3d_fullres",
      "preprocessor_name": "DefaultPreprocessor",
      "batch_size": 2,
      "patch_size": [128, 128, 128],  // 示例值，依数据集实际间距推导
      "spacing": [...],               // 目标重采样间距
      "normalization_schemes": ["CTNormalization"],
      "architecture": {
        "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
        "arch_kwargs": {
          "n_stages": 6,
          "features_per_stage": [32, 64, 128, 256, 320, 320],
          "conv_op": "torch.nn.modules.conv.Conv3d",
          "kernel_sizes": [[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]],
          "strides": [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]],
          "n_conv_per_stage": [2, 2, 2, 2, 2, 2],
          "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
          "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
          "nonlin": "torch.nn.LeakyReLU"
        }
      }
    }
  }
}
```

**各向异性数据集特殊处理**：训练日志显示 `do_dummy_2d_data_aug: True`，表明 nnU-Net 检测到本数据集 Z 轴与其他轴间距差异超过阈值（约 3 倍），自动启用了 **虚假 2D 数据增强（Dummy 2D Data Augmentation）**。这意味着在数据增强中，所有 3D 的空间变换只在面内（Y-X 平面）进行，不对 Z 轴施加旋转/弹性形变，以避免各向异性插值产生的伪影。

---

## 7. 数据预处理（Preprocessing）

### 7.1 执行命令

```bash
nnUNetv2_preprocess -d 001 -c 3d_fullres -np 8
```

该命令依据 `nnUNetPlans.json` 中的 `3d_fullres` 配置，对全部训练案例执行预处理并保存。

### 7.2 预处理流程（DefaultPreprocessor）

预处理管线按如下顺序执行：

#### Step 1：读取图像

使用 `SimpleITK` 读取 NIfTI 文件，提取体素间距、方向矩阵、原点等元数据。

#### Step 2：转置操作

将图像轴的顺序按 `transpose_forward`（规划阶段确定）重排，使间距最大的轴排在最前面（对应滑动窗口推理中的特殊处理）。

#### Step 3：裁剪非零区域（crop_to_nonzero）

将图像中各维度全为零的边界层裁去，保留最小包围盒。裁剪坐标信息被记录在 `properties` 文件中，用于推理后将结果恢复到原始坐标系。

#### Step 4：重采样至目标间距

```python
# 图像重采样：三次样条插值（order=3），Z轴0阶（nearest，防止 aniso 轴插值失真）
resampling_data_kwargs = {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None}

# 标签重采样：线性插值（order=1）后四舍五入到最近整数
resampling_seg_kwargs  = {"is_seg": True, "order": 1, "order_z": 0, "force_separate_z": None}
```

**为什么标签用 order=1 而非 order=0（最近邻）？**
线性插值后进行取整操作，能够生成比最近邻更平滑的标签边界，有助于减少重采样引入的锯齿效应，同时保证标签值的整数性。

#### Step 5：CT 强度归一化（CTNormalization）

```python
class CTNormalization(ImageNormalization):
    def run(self, image, seg=None):
        lower_bound = self.intensityproperties['percentile_00_5']  # 前景0.5%分位
        upper_bound = self.intensityproperties['percentile_99_5']  # 前景99.5%分位
        mean_intensity = self.intensityproperties['mean']
        std_intensity  = self.intensityproperties['std']
        
        np.clip(image, lower_bound, upper_bound, out=image)  # 截断极端值
        image -= mean_intensity   # 减均值
        image /= max(std_intensity, 1e-8)  # 除标准差
        return image
```

CT 归一化的独特之处在于：
1. **使用全局参数**（来自数据指纹，基于所有训练案例前景体素统计），而非逐案例归一化；
2. **先截断后标准化**：先用 0.5% 和 99.5% 分位数截断 Hounsfield 单位（HU）极值，消除金属伪影等极端值的干扰，再做 Z-score 标准化。

与 MRI 的 Z-score 归一化（使用掩码区域内均值/标准差）不同，CT 归一化不区分背景与前景，因为 HU 值具有物理意义（水=0HU、空气=-1000HU），是绝对量。

#### Step 6：压缩存储

预处理后的数据以 `.npz`（numpy 压缩格式）保存，同时记录各案例的几何属性 `properties` 字典，包含：
- 原始间距
- 裁剪前后尺寸和包围盒坐标
- 重采样后尺寸
- 转置信息

---

## 8. 模型架构（Network Architecture）

### 8.1 PlainConvUNet：3D 全卷积 U-Net

nnU-Net v2 默认使用 `PlainConvUNet`（来自 `dynamic-network-architectures` 库），这是一个经典的编码器-解码器对称结构的 3D U-Net，其架构由规划阶段**自动配置**。

**架构参数（以本项目 3d_fullres 配置为例）**：

| 参数 | 值 | 说明 |
|------|-----|------|
| n_stages | 6（典型值） | 编码器深度（包含瓶颈层） |
| features_per_stage | [32, 64, 128, 256, 320, 320] | 各层特征通道数 |
| conv_op | Conv3d | 3D 卷积 |
| kernel_sizes | 3×3×3（各层） | 卷积核尺寸 |
| strides | [[1,1,1],[2,2,2],...] | 下采样步长 |
| n_conv_per_stage | [2, 2, 2, 2, 2, 2] | 每层卷积块数 |
| norm_op | InstanceNorm3d | 实例归一化 |
| nonlin | LeakyReLU | 激活函数 |
| deep_supervision | True（训练时） | 深度监督 |

### 8.2 编码器（Encoder）

编码器由 `n_stages` 个阶段组成，每个阶段包含：
- `n_conv_per_stage[i]` 个卷积模块，每个卷积模块 = `Conv3d → InstanceNorm3d → LeakyReLU`
- 当 `strides[i] != (1,1,1)` 时，在该阶段末尾通过步长为 2 的卷积（而非池化层）进行下采样

**特征通道数规律**：
$$\text{features}[i] = \min(\text{max\_features}_{3D}, 32 \times 2^i)$$

其中 $\text{max\_features}_{3D} = 320$（防止通道数无限增长）。因此通道序列为：32 → 64 → 128 → 256 → 320 → 320。

**为什么使用 InstanceNorm 而非 BatchNorm？**

在医学图像分割中，批大小通常很小（本项目 batch_size=2），BatchNorm 在小 batch 下统计量不准确。InstanceNorm 对每个样本独立归一化，不依赖 batch 统计，更适合小 batch 3D 分割任务。

### 8.3 解码器（Decoder）

解码器也由 `n_stages - 1` 个阶段组成，每个阶段：
1. 使用双线性/三线性上采样（或转置卷积）将特征图分辨率×2
2. 与编码器对应层的特征图进行 **skip connection**（通道维度拼接）
3. 经过 `n_conv_per_stage_decoder[i]` 个卷积模块

最终输出层：`Conv3d(features[0], n_classes, kernel_size=1)`，输出每个体素在各类别上的 logit 值（未经 softmax），维度为 `(B, n_classes, X, Y, Z)` = `(B, 3, X, Y, Z)`。

### 8.4 深度监督（Deep Supervision）

**原理**：在解码器的每个分辨率层级（从最高分辨率开始，共 `n_stages - 1` 个输出）都附加一个 `1×1×1` 卷积头，输出该分辨率下的分割预测。训练时所有输出头都参与损失计算，以指数衰减权重加权：

$$\mathcal{L}_{total} = \sum_{i=0}^{K-1} w_i \cdot \mathcal{L}(y_{pred}^{(i)}, y_{true}^{(i)})$$

其中 $w_i = 1/2^i$（再归一化使权重和为1），分辨率最高的输出头权重最大（$w_0$ 最大），最低分辨率输出头权重接近0（实际设为 $10^{-6}$ 或 0 以避免 DDP 未使用参数问题）。

```python
deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(
    np.vstack(self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
weights[-1] = 0
weights = weights / weights.sum()
loss = DeepSupervisionWrapper(loss, weights)
```

**优势**：
- 各分辨率层级的梯度能直接回传到浅层，缓解梯度消失
- 在训练初期各层快速收敛，加速整体训练过程
- 推理时**仅使用最高分辨率输出**，关闭深度监督（`enable_deep_supervision=False`）

### 8.5 网络整体结构示意

```
输入 (B, 1, Z, Y, X)
   │
   ├─► [Conv3d×2, IN, LReLU] → 特征图 E₀ (B, 32, Z, Y, X)
   │                        skip₀ ─────────────────────┐
   ├─► stride2 ↓ → [Conv3d×2, IN, LReLU] → E₁ (B, 64, Z/2, Y/2, X/2)
   │                        skip₁ ─────────────────┐   │
   ├─► stride2 ↓ → [Conv3d×2, IN, LReLU] → E₂ (B, 128, ...)
   │                        skip₂ ──────────────┐  │   │
   ├─► stride2 ↓ → [Conv3d×2, IN, LReLU] → E₃ (B, 256, ...)
   │                        skip₃ ───────────┐  │  │   │
   ├─► stride2 ↓ → [Conv3d×2, IN, LReLU] → E₄ (B, 320, ...)
   │                        skip₄ ────────┐  │  │  │   │
   └─► stride2 ↓ → [瓶颈层 Conv3d×2]     │  │  │  │   │
                           E₅ (B, 320, ...)  │  │  │   │
   Decoder (上采样+skip拼接):              │  │  │   │
   upsample ↑ + cat(skip₄) → [Conv3d×2] ──┘  │  │   │
   upsample ↑ + cat(skip₃) → [Conv3d×2] ─────┘  │   │
   upsample ↑ + cat(skip₂) → [Conv3d×2] ────────┘   │
   upsample ↑ + cat(skip₁) → [Conv3d×2] ────────────┘
   upsample ↑ + cat(skip₀) → [Conv3d×2]
                           │
   [Conv3d 1×1×1] → logits (B, 3, Z, Y, X)
                           │
                    argmax → 分割结果
```

---

## 9. 模型训练（Training）

### 9.1 训练命令

```bash
nnUNetv2_train 001 3d_fullres 0 --npz
# 参数: 数据集ID=001, 配置=3d_fullres, fold=0
```

训练日志（`train_100epochs.log`）显示的关键信息：
```
Using device: cuda:0
Using torch.compile...             # PyTorch 2.x JIT编译加速
do_dummy_2d_data_aug: True         # 各向异性数据集启用2D虚假增强
Using splits from existing split file ...
The split file contains 5 splits.
Desired fold for training: 0
This split has 63 training and 16 validation cases.  # 5折中fold 0的划分
```

### 9.2 五折交叉验证策略

nnU-Net 默认使用**5折交叉验证（5-fold Cross-Validation）**，按固定随机种子（12345）将训练案例均匀划分：

```python
splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
```

79 个训练案例分配：
- 每折训练集约 **63 例**，验证集约 **16 例**
- 共训练 5 个独立模型（fold 0~4）
- 推理时可集成全部 5 个折的模型（ensemble）提升鲁棒性

**本项目训练了 fold 0**，在推理时使用 `fold_0/checkpoint_best.pth`。

### 9.3 优化器与学习率调度

**优化器**：带 Nesterov 动量的 SGD（随机梯度下降）

```python
optimizer = torch.optim.SGD(
    self.network.parameters(),
    lr=1e-2,
    weight_decay=3e-5,
    momentum=0.99,
    nesterov=True
)
```

选择 SGD 而非 Adam 的原因：在医学图像分割任务中，大量实践表明带 Nesterov 动量的 SGD 配合多项式学习率调度，最终收敛效果不逊于 Adam，且泛化性能通常更好。

**学习率调度**：多项式衰减（Polynomial Decay）

```python
class PolyLRScheduler(_LRScheduler):
    def step(self, current_step):
        new_lr = initial_lr * (1 - current_step / max_steps) ** 0.9
```

$$lr_t = lr_0 \cdot \left(1 - \frac{t}{T}\right)^{0.9}$$

其中 $lr_0 = 0.01$（初始学习率），$T = 1000$（最大训练轮数，本项目训练了 100 epochs）。指数为 0.9 使学习率在训练早期缓慢下降，在训练后期快速收敛至接近0。

### 9.4 损失函数（DC+CE Loss）

nnU-Net 使用 **Dice Loss 与 Cross-Entropy Loss 的线性组合**（权重各为 1）：

$$\mathcal{L} = \mathcal{L}_{CE} + \mathcal{L}_{Dice}$$

**Dice Loss（MemoryEfficientSoftDiceLoss）**：

$$\mathcal{L}_{Dice} = 1 - \frac{2 \sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$$

其中 $p_i$ 为 softmax 输出的预测概率，$g_i$ 为 one-hot 编码的真实标签，$\epsilon = 10^{-5}$ 防止除零。

Dice Loss 对类别不平衡天然鲁棒：即使囊肿（标签2）体积远小于背景，其 Dice 值也能在损失中占有与其体积不成比例的权重。

**Cross-Entropy Loss（RobustCrossEntropyLoss）**：

$$\mathcal{L}_{CE} = -\sum_i \sum_c g_{i,c} \log p_{i,c}$$

CE Loss 提供了更强的逐体素监督信号，有助于在边界区域获得更精确的预测。

```python
loss = DC_and_CE_loss(
    {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False}, 
    {},
    weight_ce=1, 
    weight_dice=1,
    dice_class=MemoryEfficientSoftDiceLoss
)
```

`do_bg=False` 表示 Dice 计算时排除背景类（标签0），仅计算前景类（肾脏+囊肿）的 Dice，避免背景类 Dice 值因数量极大而主导损失。

### 9.5 数据加载与过采样策略

**前景过采样（Foreground Oversampling）**：

```python
self.oversample_foreground_percent = 0.33  # 33% 的 patch 从前景中心采样
```

在每个批次中，约 1/3 的 patch 强制以目标前景区域（肾脏或囊肿）为中心随机采样，防止模型因背景比例过大而忽视病变区域。剩余 2/3 的 patch 随机采样。

**patch 采样流程**：
1. 从预处理后的 `.npz` 文件中加载完整的 3D 体积
2. 随机选取一个体素位置作为 patch 中心（前景过采样时从前景体素中随机选）
3. 从中心向外裁取 `patch_size` 大小的子体积
4. 不足时进行边界填充（pad）

### 9.6 数据增强策略

训练时采用丰富的在线数据增强（`get_training_transforms`），所有增强均实时计算（On-the-fly），增强后的数据不落盘：

**空间变换**（SpatialTransform）：
- **随机旋转**：对等向数据 ±30°，对各向异性数据 ±180°（仅面内，即虚假2D模式）
- **随机缩放**：比例因子 0.7~1.4
- **弹性形变**：未启用（`p_elastic_deform=0`，在CT中通常不必要）

**强度变换**：
| 增强类型 | 概率 | 参数范围 | 说明 |
|---------|------|---------|------|
| 高斯噪声 | 10% | 方差 0~0.1 | 模拟低 SNR 图像 |
| 高斯模糊 | 20% | σ=0.5~1.0 | 模拟对焦不实 |
| 乘性亮度 | 15% | 乘数 0.75~1.25 | 模拟扫描仪差异 |
| 对比度调整 | 15% | 对比度 0.75~1.25 | 模拟灰度偏差 |
| 低分辨率模拟 | 25% | 缩放 0.5~1.0 | 模拟不同层厚CT |
| Gamma 反转 | 10% | γ=0.7~1.5，翻转 | 模拟伽马校正偏差 |
| Gamma 正向 | 30% | γ=0.7~1.5，不翻转 | |

**镜像增强**（MirrorTransform）：
- 以 50% 概率对三个轴独立翻转：`allowed_axes=(0, 1, 2)`
- 推理时同样使用测试时增强（Test-Time Augmentation, TTA），对8种镜像状态的预测结果取平均

**深度监督下采样**（DownsampleSegForDSTransform）：
- 将标签分割图下采样到各解码器输出分辨率，与对应深度监督头配对

### 9.7 训练流程

```python
# 单个 epoch 训练步骤
def train_step(self, batch):
    data, target = batch['data'], batch['target']
    data = data.to(device)
    target = [t.to(device) for t in target]  # 多尺度深度监督目标
    
    optimizer.zero_grad()
    with autocast('cuda'):          # 混合精度训练（AMP）
        output = network(data)      # 前向传播
        l = loss(output, target)    # 深度监督损失
    
    grad_scaler.scale(l).backward() # 梯度缩放反向传播
    grad_scaler.unscale_(optimizer)
    clip_grad_norm_(network.parameters(), 12)  # 梯度裁剪（防止梯度爆炸）
    grad_scaler.step(optimizer)
    grad_scaler.update()
```

**关键训练设置**：
| 参数 | 值 |
|------|-----|
| 每 epoch 迭代次数 | 250 次（训练） |
| 验证迭代次数 | 50 次/epoch |
| 总训练 epoch | 1000（本项目运行了约100） |
| 混合精度（AMP） | 开启（torch.cuda.amp） |
| 梯度裁剪 | max_norm=12 |
| torch.compile | 开启（JIT编译加速） |

**检查点保存策略**：
- 每 50 个 epoch 保存一次 `checkpoint_latest.pth`
- 根据验证集 Dice 保存最优模型 `checkpoint_best.pth`
- 训练结束保存 `checkpoint_final.pth`

### 9.8 验证与在线评估

每个 epoch 训练结束后进行验证（`validation_step`）：

```python
def validation_step(self, batch):
    with autocast('cuda'):
        output = network(data)
        l = loss(output, target)
    
    output = output[0]   # 取最高分辨率输出
    output_seg = output.argmax(1)[:, None]   # 逐体素取最大概率类别
    
    # 计算 TP, FP, FN（用于 Dice 估计）
    tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_seg_onehot, target, axes=...)
    return {'loss': l, 'tp_hard': tp, 'fp_hard': fp, 'fn_hard': fn}
```

训练器实时计算并记录**伪 Dice（Pseudo Dice）**，即基于当前批次的估计 Dice 值（非全集Dice），用于监控训练进度。真实的全集 Dice 评估在 `nnUNetv2_find_best_configuration` 阶段完成。

---

## 10. 模型推理（Inference）

### 10.1 推理入口

**命令行推理**（`run_inference.py`）：

```python
predictor = nnUNetPredictor(
    tile_step_size=0.5,            # 滑动步长为 patch_size 的 50%
    use_gaussian=True,             # 使用高斯重要性加权
    use_mirroring=True,            # 启用测试时镜像增强(TTA)
    perform_everything_on_device=True,  # 全部在GPU执行
    device=torch.device("cuda"),
    verbose=True,
    allow_tqdm=True,
)
predictor.initialize_from_trained_model_folder(
    model_training_output_dir=".../nnUNetTrainer__nnUNetPlans__3d_fullres",
    use_folds=(0,),                # 使用 fold 0 的模型
    checkpoint_name="checkpoint_best.pth",  # 最优检查点
)
predictor.predict_from_files(
    list_of_lists_or_source_folder=tmp_input,
    output_folder=tmp_output,
    save_probabilities=False,      # 只保存类别预测图（不保存概率图）
    num_processes_preprocessing=2,
    num_processes_segmentation_export=2,
)
```

### 10.2 推理预处理

在推理阶段，对测试图像执行与训练完全一致的预处理步骤：裁剪 → 重采样 → 归一化，确保数据分布与训练数据一致。

### 10.3 滑动窗口推理（Sliding Window Prediction）

由于 CT 图像体积（如 $512 \times 512 \times 400$）通常远大于训练时的 patch size，推理时使用**滑动窗口策略**：

**步长计算**：

```python
# tile_step_size=0.5 意味着相邻窗口重叠 50%
target_step_sizes = [patch_size[i] * tile_step_size for i in range(3)]
num_steps = [ceil((image_size[i] - patch_size[i]) / step_size[i]) + 1 for i in range(3)]
```

若图像某轴尺寸小于 patch size，则进行零填充（zero-padding）后推理，再裁回原始尺寸。

**高斯加权聚合（Gaussian Importance Weighting）**：

重叠区域会被多次预测，不同预测间的聚合方式至关重要。nnU-Net 使用**以 patch 中心为峰值的 3D 高斯权重**进行加权平均：

```python
def compute_gaussian(tile_size, sigma_scale=1/8):
    # 生成以中心为峰值的3D高斯权重图
    center_coords = [i // 2 for i in tile_size]
    tmp = np.zeros(tile_size)
    tmp[tuple(center_coords)] = 1
    gaussian_map = gaussian_filter(tmp, sigmas=[i * sigma_scale for i in tile_size])
    return gaussian_map / gaussian_map.max()
```

对于位置 $v$ 的预测，其最终概率为：

$$P(v) = \frac{\sum_{k} G_k(v) \cdot p_k(v)}{\sum_{k} G_k(v)}$$

其中 $G_k(v)$ 为第 $k$ 个滑动窗口中位置 $v$ 对应的高斯权重，$p_k(v)$ 为该窗口对 $v$ 的 softmax 输出概率。

**物理意义**：patch 中心区域的预测最可靠（边缘效应最小），因此中心权重最大；靠近边缘的预测受边界影响较大，权重较小。

### 10.4 测试时增强（Test-Time Augmentation, TTA）

`use_mirroring=True` 启用 TTA，对测试图像进行 $2^3 = 8$ 种镜像状态（三轴各自翻转的所有组合）的推理，将 8 个 softmax 概率图平均后再 argmax 得到最终分割：

```python
if use_mirroring:
    # 对所有允许镜像轴的组合依次推理
    for axes in all_mirror_combinations(allowed_mirroring_axes):
        mirrored_input = flip(data, axes)
        pred = network(mirrored_input)
        pred = flip(pred, axes)  # 翻转回来
        aggregated_pred += pred
    aggregated_pred /= num_mirror_combinations
```

TTA 相当于集成了 8 个数据增强视角的预测，在推理阶段无需额外训练，可稳定提升 Dice 约 0.5~1.5%。

### 10.5 后处理：输出恢复

推理完成后，预测结果需从预处理后的坐标空间恢复到原始图像坐标空间：

1. **反转置**（transpose backward）：将轴顺序还原
2. **反重采样**：将预测图从目标间距重采样回原始间距（使用最近邻插值，保证整数标签）
3. **填充裁剪区域**：在之前裁剪掉的非零区域外侧补零
4. **写入 NIfTI 文件**：恢复原始图像的几何元数据（spacing/origin/direction）后保存

最终输出与输入图像完全对齐的分割 mask，每个体素值为：
- **0**：背景
- **1**：肾脏
- **2**：肾脏囊肿

---

## 11. 后处理与结果评估

### 11.1 nnU-Net 内置后处理

nnU-Net 提供连通域分析（Connected Component Analysis）后处理，可选择性去除过小的预测区域：

```bash
nnUNetv2_determine_postprocessing -d 001 -c 3d_fullres
nnUNetv2_apply_postprocessing ...
```

`nnUNetv2_determine_postprocessing` 会在验证集上自动评估是否保留最大连通域、是否去除小区域等后处理策略，选择能提升 Dice 的策略。

### 11.2 评估指标

分割结果的评估主要使用以下指标：

**Dice 相似系数（DSC）**：

$$\text{DSC} = \frac{2|P \cap G|}{|P| + |G|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

对于每个类别（肾脏和囊肿）分别计算，范围 [0, 1]，越大越好。

**Hausdorff 距离（HD95）**：

$$\text{HD95} = \max\left(\text{perc}_{95}(d(P, G)), \text{perc}_{95}(d(G, P))\right)$$

衡量预测边界与真实边界的最大偏差（取第95百分位以避免离群值影响），单位 mm，越小越好。

**交并比（IoU/Jaccard）**：

$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|} = \frac{TP}{TP + FP + FN}$$

### 11.3 寻找最优配置

```bash
nnUNetv2_find_best_configuration -d 001 -c 3d_fullres
```

该命令在所有 fold 的验证集上评估模型，计算平均 Dice，并与 2D/3D-lowres/ensemble 等配置对比，最终推荐最优配置用于测试集推理。

---

## 12. 医学影像可视化系统

### 12.1 系统架构

本项目开发了一套基于 Flask 的医学影像可视化 Web 系统（`medical_viewer/app.py`），集成了：
- **多平面重建（MPR）**：轴位（Axial）、冠状位（Coronal）、矢状位（Sagittal）三视图
- **3D 体积渲染**（WebGL 纹理图集方案）
- **最大密度投影（MIP）**
- **nnU-Net 实时推理**（单案例 + 批量）
- **分割结果叠加显示**（Overlay 模式）

### 12.2 CT 窗宽窗位处理

```python
def apply_window(slice_data, ww, wl):
    lower = wl - ww / 2.0
    upper = wl + ww / 2.0
    windowed = np.clip(slice_data, lower, upper)
    windowed = (windowed - lower) / (upper - lower) * 255.0
    return windowed.astype(np.uint8)
```

窗宽（WW）和窗位（WL）参数可由用户调节，模拟放射科软件的窗技术，适应腹部CT（WW=400, WL=40）的显示需求。

### 12.3 坐标系处理

医学影像的坐标系转换是可视化中的关键挑战。系统将 SimpleITK 的 LPS 坐标系转换为 3D Slicer 标准的 RAS 坐标系：

```python
image_ras = sitk.DICOMOrientImageToCoordinateAxes(image, 'RAS')
# LPS → RAS: X轴方向反转（L→R），Y轴方向反转（P→A）
```

各切面的显示约定（参照放射学惯例）：
- **轴位**：前方（Anterior）在上，右侧（Right）在左（影像学左右与解剖学相反）
- **冠状位**：上方（Superior）在上，右侧在左
- **矢状位**：上方在上，前方在左

### 12.4 分割叠加渲染

```python
SEG_COLORS = {
    1: (0, 255, 0),    # 肾脏 - 绿色
    2: (255, 255, 0),  # 囊肿 - 黄色
}

def render_overlay(gray_img, seg_slice, opacity=0.4):
    for label, color in SEG_COLORS.items():
        mask = (seg_slice == label)
        for c in range(3):
            rgb[:, :, c][mask] = (1 - opacity) * rgb[:, :, c][mask] + opacity * color[c]
```

透明度（opacity=0.4）可在 Web 界面实时调节，支持"纯分割"和"叠加"两种显示模式。

### 12.5 高性能缓存（blosc2）

NIfTI 文件的读取和坐标系转换计算量较大（数百 MB 的 3D 体积），系统使用 **blosc2** 进行高效压缩缓存：

```python
def save_to_blosc2_cache(array, cache_path):
    blosc2.asarray(
        array, chunks=chunk_size, blocks=block_size,
        urlpath=cache_path, mode='w',
        cparams=blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=3)
    )
```

缓存策略：以文件路径+修改时间+文件大小的 MD5 哈希作为缓存键，重新访问同一文件时直接从 `.b2nd` 缓存读取，避免重复 IO 和坐标系转换计算。

### 12.6 Web 端推理流程

系统支持**异步推理**（后台线程执行，前端轮询进度）：

```
用户触发推理 (POST /api/inference/start)
       │
       ▼
生成 task_id, 启动后台线程
       │
       ▼
进度跟踪 (GET /api/inference/progress/{task_id})
   ├── status: 'loading'    →  5%  正在加载模型
   ├── status: 'preprocessing' → 15% 正在预处理
   ├── status: 'inferring'  → 30%~85% 滑动窗口推理中
   ├── status: 'postprocessing' → 85% 后处理
   ├── status: 'caching'    → 92% 缓存分割结果
   └── status: 'done'       → 100% 推理完成
       │
       ▼
分割结果加载到内存 (loaded_volumes[filepath + "_seg"])
       │
       ▼
前端请求切片时实时叠加分割 (/api/slice/{plane}/{idx}?overlay=overlay)
```

---

## 13. 关键技术总结

### 13.1 nnU-Net 成功的核心要素

| 方面 | 技术手段 | 本项目实践 |
|------|---------|-----------|
| 自适应配置 | 数据指纹 → 规则化参数推导 | 自动配置 3d_fullres，patch ~128³ |
| 网络架构 | 自动拓扑设计，最大特征320 | 6阶段 PlainConvUNet，深度监督 |
| 归一化 | CT: 百分位裁剪+Z-score | percentile(0.5%,99.5%) clip, then Z-score |
| 损失函数 | DC+CE 混合损失 | weight_dice=1, weight_ce=1 |
| 优化策略 | SGD+Nesterov+PolyLR | lr=0.01, momentum=0.99, decay^0.9 |
| 数据增强 | 空间+强度多维增强 | 8类变换，Dummy2D for aniso |
| 推理策略 | 滑动窗口+高斯权重+TTA | step=0.5, gaussian, mirroring |
| 训练效率 | AMP+torch.compile+DDP | cuda AMP, JIT编译加速 |

### 13.2 针对肾脏囊肿检测的特殊考量

1. **类别不平衡**：囊肿体积远小于肾脏和背景，Dice Loss 的类别不感知性（天然对小目标友好）配合前景过采样策略（33%的 patch 以前景为中心）有效缓解了此问题。

2. **各向异性数据**：腹部 CT 的 Z 轴（层间距）通常远大于面内分辨率，nnU-Net 自动检测并启用 Dummy 2D 数据增强，仅在面内进行旋转等变换，同时在各向异性轴使用 1×1 卷积核（而非 3×3），适应低分辨率轴的特性。

3. **多类别分割**：框架同时输出肾实质（粗粒度）和囊肿（细粒度）的分割，两个任务共享同一网络权重，前者提供了后者的上下文约束，有助于囊肿的精确定位。

4. **标签几何对齐**：原始数据集中标签与图像存在几何不一致问题，`fix_and_convert_data.py` 通过 SimpleITK 重采样对齐修复，确保训练数据质量。

### 13.3 本项目的技术贡献

1. **完整实现了基于 nnU-Net v2 的肾脏囊性病变检测流水线**，覆盖数据转换到模型推理的全流程。

2. **开发了集成推理能力的医学影像可视化系统**，支持多平面重建、3D 渲染、实时推理和分割叠加显示，具备工程实用价值。

3. **针对数据标签几何不一致问题**，设计了标签重采样对齐预处理方案，保证了多源数据的几何一致性。

---

*文档生成时间：基于 nnU-Net v2.6.2 源码及本项目实际代码分析整理*

*引用：Isensee, F., Jaeger, P.F., Kohl, S.A., Petersen, J., & Maier-Hein, K.H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.*
