# OSWorld Mobile 4B 与 DeepSeek 评测

这里是本地 OSWorld 长任务对照实验的代码。评测入口、环境依赖、模型版本和运行命令见[复现说明](Mobile-Agent-v3.5-mac/README.md)。[任务清单](task-plan.json)收录了 10 道题的 UUID、人工参考步数和批次参数。

- `Mobile-Agent-v3.5-mac/` 包含 GUI-Owl-1.5-4B-Instruct 的推理和评测入口。
- `UI-S1-mac/` 包含 DeepSeek 评测入口和两个入口共用的动作转换。目录名沿用实验代码结构。
- `OSWorld/` 包含本次使用的评测环境代码、10 道任务的定义和动作控制器测试。OSWorld 代码来自 [XLANG Lab](https://github.com/xlang-ai/OSWorld)，本地加入了动作失败处理和多次 Enter 的支持。

Mobile 的桌面提示来自 [Mobile-Agent-v3.5](https://github.com/X-PLUG/MobileAgent)。上游许可见各目录中的许可证文件。
