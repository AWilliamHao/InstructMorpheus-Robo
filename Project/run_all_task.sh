#!/usr/bin/bash
cd ~/RoboTwin
# 按顺序执行三个任务
bash run_task.sh block_hammer_beat 0 && \
bash run_task.sh blocks_stack_easy 0 && \
bash run_task.sh block_handover 0

# 当所有任务完成后打印提示
echo "所有任务已完成"