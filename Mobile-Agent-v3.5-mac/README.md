# Mobile 4B 与 DeepSeek 的 OSWorld 长任务实验

从项目根目录运行。目录结构保留 `Mobile-Agent-v3.5-mac/`、`UI-S1-mac/` 和 `OSWorld/`。评测进程读取本地 OSWorld 任务 JSON，并使用其中的评测器。两条命令都需要一个专用的 Ubuntu VMware 虚拟机，以及该虚拟机中已配置好 OSWorld 客户机服务的快照。运行会恢复该快照。

## 运行环境

- 主机使用包含 OSWorld 依赖的 Python 3.12 环境。下面的 `$OSWORLD_PYTHON` 指向其 Python 可执行文件，本机使用 `OSWorld/.venv-osworld/bin/python`。DeepSeek 运行环境还需要 `openai`，本机版本为 3.57.0，并通过环境变量 `DEEPSEEK_API_KEY` 获取密钥。
- 本地 GUI-Owl 推理用单独的 Python 3.12 环境，本机版本为 `torch==2.9.1`、`transformers==4.57.6`、`accelerate==1.15.0`、`Pillow==13.0.0` 和 `huggingface-hub==0.36.2`。下面的 `$MODEL_PYTHON` 指向该环境的 Python 可执行文件。
- 将完整模型文件下载到 `$WEIGHTS_DIR`。下载时指定 `hf download mPLUG/GUI-Owl-1.5-4B-Instruct --revision 3f061c2c562cc860c42bf32542a70e07a7ff4840 --local-dir "$WEIGHTS_DIR"`。推理直接读取本地模型文件。

## 单题运行参数

`$TASK_ID` 填任务 UUID，`$VMX` 填专用 VM 的 `.vmx` 绝对路径，`$SNAPSHOT` 填对应快照名。长任务批次使用每题 30 轮、每轮一张当前截图、此前所有动作的文字描述和每轮 256 个输出 token。以下命令保留这些输入配置。

```sh
"$OSWORLD_PYTHON" Mobile-Agent-v3.5-mac/run_osworld_mobile.py \
  --task "$TASK_ID" --steps 30 --image-history 0 --max-tokens 256 \
  --model-python "$MODEL_PYTHON" --weights "$WEIGHTS_DIR" \
  --vmx "$VMX" --snapshot "$SNAPSHOT"

"$OSWORLD_PYTHON" UI-S1-mac/run_osworld_openai.py \
  --task "$TASK_ID" --provider deepseek --model deepseek-flash --policy mobile-history \
  --steps 30 --reasoning-effort none --max-output-tokens 256 \
  --vmx "$VMX" --snapshot "$SNAPSHOT"
```

如果要把历史窗口改为最近 8 轮，在两条命令中都显式加入 `--history 8`。这是不同的输入配置。Mobile 默认的历史截图数量是 4，复现这批设置时应保留 `--image-history 0`。`DONE` 和 `FAIL` 不计入已执行操作数。动作执行状态无法确认时，任务会以执行错误结束，评分字段为 `null`。脚本会在终端打印运行记录目录。
