#!/bin/bash

# 定义颜色输出（可选，用于更清晰的提示）
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# 打印开始提示
echo -e "${GREEN}开始执行验证脚本...${NC}"

# 执行核心命令
/root/miniconda3/bin/conda run -n myenv --no-capture-output python scripts/inference.py

# 检查命令执行状态
if [ $? -eq 0 ]; then
    echo -e "${GREEN}验证脚本执行完成！${NC}"
else
    echo -e "${RED}验证脚本执行失败！${NC}"
    exit 1
fi
