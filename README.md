任务1：真实 exe 画面 → 18x14 网格数据

roi_exe.json
真实 exe 截图时的 ROI（截图区域）配置（用于定位棋盘区域）。

task_1_2.py / task_1_3.py
任务1的功能脚本（例如：截屏、ROI裁剪、网格化、动作记录等的实现模块）。

run_task_1_1.py
任务1运行入口：把任务1流程串起来执行（采集并生成输出文件）。

real_frames.npy
真实 exe 采集得到的棋盘序列（通常每帧为 18x14 的矩阵数据）。

real_actions.csv
与 real_frames.npy 对齐的动作记录（每帧/每步对应一个动作）。

4.4 任务2：pygame 仿真环境（保存 18x14 网格）

task_2_1.py / task_2_1_new.py / task_2_2.py
任务2相关模块：搭建/改进 pygame 环境、输出网格帧、动作接口等。

run_task_2_3.py
任务2运行入口：启动仿真并输出数据文件。

task2_frames.npy
任务2仿真采集到的棋盘帧（18x14 网格序列）。

task2_actions.csv
与 task2_frames.npy 对齐的动作记录。

4.5 任务3：规则/启发式自动玩（基于任务2的环境）

task_3_1.py / task_3_1_new.py
任务3核心逻辑（启发式策略/规则评估等）。

run_task_3_2.py
任务3运行入口：在 pygame 环境中使用策略自动玩，并可输出日志/数据。

4.6 任务4/5：进一步的数据采集/检测数据集（为训练做准备）

task_4_1.py
任务4相关脚本（通常用于扩展数据处理/采集检查/转换等）。

task_5_1tetris_env.py
任务5使用的 Tetris 环境实现/封装（训练/采集时的统一接口）。

task_5_2_heuristic_agent.py
启发式智能体（用于自动玩、生成数据）。

task_5_3_dataset_logger.py
数据记录器：把 frames/actions/标签等保存到本地文件夹或 npz/npy/csv。

task_5_4_run_task5_collect.py
任务5采集入口：运行环境 + agent，批量采集数据集。

task_5_5_check_task5_detect.py
任务5数据集检查：检查 frames/actions 是否对齐、shape 是否正确、标签范围是否合理等。

4.7 任务6：构建训练数据集 + 模型 + 训练/推理

task_6_1_tetris_dataset.py
任务6数据集定义：如何从保存的数据读取并喂给模型训练。

task_6_2_tetris_model.py
任务6模型定义（神经网络结构）。

task_6_3_train_task6.py
任务6训练入口：加载数据集、训练模型、保存权重/日志。

task_6_4_build_task6_npz.py
把采集数据转换/打包成更适合训练的格式（例如 npz）。

task_6_5_dataset_simple.py / task_6_6_model_simple.py / task_6_7_train_task6_simple.py
简化版数据集/模型/训练脚本（用于快速验证流程是否跑通）。

task_6_8_dataset_xyrot.py / task_6_9_model_xyrot.py
xyrot 版本数据与模型（通常表示：加入位置(xy)与旋转(rot)等特征/标签的建模版本）。

task_6_10_train_xyrot.py
xyrot 模型训练入口。

task_6_11_infer_xyrot.py
xyrot 推理/测试入口（用训练好的模型预测动作或策略相关输出）。

task_6_12_plan_actions.py
动作规划脚本：把模型输出转成可执行动作序列（例如 LEFT/RIGHT/ROTATE/DROP）。

task_6_13_nn_agent.py
神经网络智能体：把“模型预测 + 动作规划”封装成一个能在环境中玩游戏的 agent。