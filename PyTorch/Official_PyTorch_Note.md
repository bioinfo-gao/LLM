(1) Colab保存到我的github， 分2-5步：

1.1) 原链接只有save a copy as Github Gist 选项
1.2) 在新打开的Gisthub Gist才有 save a copy in Github.
1.3) 确认仓库和分支仓库 (Repository): 确认选择了正确的仓库，例如 bioinfo-gao/LLM 和分支 (Branch): 确认选择了您要保存的分支（通常是 main 或 master）。
1.4) 手动输入完整路径在 “文件名” 输入框中，您需要输入 目录名/文件名 的完整结构。区域您的输入内容文件名 (File path)/PyTorch/quickstart_tutorial.ipynb
1.5) 在该gihub 页面中点击colab 打开，此时如果有修改或者调试，点击save，即直接save到github ，无需其他转存步骤，也无需设置目录结构，即自动保存为原来的目录树下

(2) 在Github网页界面创造新目录的步骤： (在Github网页界面上，无法直接创建空目录，只能通过变通方法，创建该新目录下的文件)

到仓库主页。点击 “Add file” 按钮，然后选择 “Create new file”; 如果您想创建名为 data 的目录，您输入的文件名应该是：data/.gitkeep （输入"/"符号后，Github会自动识别这是一个目录分隔符）。
有高度可能会自动重复data/data/  需要backspace 回退一个，并在其位置改成文件名

(3) check the nn is running on GPU or CPU

确认 Colab 实例本身是否被配置为使用 GPU 或 TPU。查看 Colab 菜单栏：点击 **运行时 (Runtime) **。选择 **更改运行时类型 (Change runtime type) **。

在弹出的对话框中，检查 “硬件加速器 (Hardware accelerator) ” 选项：

如果它显示为 None，则您的 Notebook 只能使用 CPU。如果它显示为 GPU 或 TPU，则您有权使用加速器。

注意： 设置为 GPU 并不意味着您的代码一定使用了 GPU，只是说它可以使用。

torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device(type='cuda', index=0)

除了运行上面的代码检查是CPU 还是cuda之外， 看网页右下方有T4选项，那就是runningtime 里的T4

(4) NVIDIA T4 GPU 是一款专为数据中心设计的加速卡，主要侧重于推理 (Inference) ** 和轻量级训练**，特别是在云环境（如 AWS 的 g4dn 系列）中非常常见。

T4 采用与消费级 RTX 20 系列相似的 Turing 架构，但具备专业特性，使其非常适合成本敏感且需要加速计算的应用。
🚀 T4 GPU 的核心性能与定位T4 GPU 的主要优势不在于绝对的 FP32 峰值速度，而在于其多精度支持、高能效比和低延迟。

备注架构Turing与 RTX 20 系列相同。

4.1) 性能参数概览    T4 GPU (单个)
显存 (VRAM)16 GB GDDR6拥有 ECC 纠错功能，保障数据中心可靠性。FP32 性能约 8.1 TFLOPS性能适中，适合轻量级训练。AI 性能 (INT8)约 130 TOPST4 的核心优势，非常适合低精度推理。

4.2) 体积和能耗参数 T4 GPU (单个)
PCIe半高半长（Half-Height, Half-Length）物理尺寸小，能耗低 (70W)，适合高密度部署。

云实例AWS g4dn 系列云服务商的主流入门级 GPU 实例。

(5) PyTorch offers domain-specific libraries such as TorchText, TorchVision, and TorchAudio, all of which include datasets. For this tutorial, we will be using a TorchVision dataset.

The torchvision.datasets module contains Dataset objects for many real-world vision data like CIFAR, COCO (full list here). In this tutorial, we use the FashionMNIST dataset. Every TorchVision Dataset includes two arguments: transform and target_transform to modify the samples and labels respectively.

(6）GPU 运行备注
CPU GPU需要分别下载数据
CPU running time : Around 5 min
GPU running time : Around 1 min
For this Quickstart  dataset, T4 GPU took 1/5 time of the CPU4

注意： switch running time 之后，所有变量都消失，可以理解成两台机器：必须从头 run <<<=====

(7)关闭网页时，Colab 没有“关闭网页时自动断开”的设置选项。 您必须养成一个手动习惯：在关闭 Notebook 之前，明确地告诉 Colab 断开与 Runtime 的连接。

7.1) 关闭单个 Session 的标准步骤（推荐习惯）在您完成工作后，请执行以下任一操作： ==> 7.1.3 Easiest! Please use
7.1.1) 断开并删除运行时 (Terminate Runtime):      
点击顶部菜单栏的 “运行时 (Runtime)”。选择 “管理 Session (Manage sessions)”。
在弹出的侧边栏中，找到您要关闭的 Notebook，点击右侧的 “终止” 或 “X” 图标。
7.1.2) 断开连接 (Disconnect):点击顶部菜单栏的 “运行时 (Runtime)”。
选择 “断开连接并删除运行时 (Disconnect and delete runtime)”。
7.1.3) 点击 Notebook 右上角 RAM 和 Disk 显示旁边的小箭头，然后点击 “Disconnect and delete runtime 断开连接”。

7.2) 批量关闭所有空闲 Session
如果堆积了许多空闲 Session，可以使用这个方法快速清理：
点击 “运行时 (Runtime)”。选择 “管理 Session (Manage sessions)”。在侧边栏中，您可以终止所有处于 “空闲 (Idle)” 或 “忙碌 (Busy)” 状态的 Session。

💡 建议养成在关闭 Colab 标签页前，执行 “运行时” -> “断开连接并删除运行时” 的习惯，这样能确保 GPU/TPU 资源立即被释放，避免占用配额。

(8) 在github 中colab 文件不能进行改名，只能在colab本身改名，在github 改名，colab就再也无法对应上原文件了
