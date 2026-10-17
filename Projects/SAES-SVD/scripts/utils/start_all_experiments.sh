#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# ================================
# 快速启动脚本 - 并行执行所有实验
# ================================

echo "=========================================="
echo "SAES-SVD 优化实验 - 快速启动"
echo "=========================================="
echo ""
echo "将启动以下实验（并行模式）："
echo "  1. 基线PPL测试 (GPU 0, 前台, 15min)"
echo "  2. 参数对照实验 (GPU 0, 后台, 4.5h)"
echo "  3. LoRA微调 (GPU 1, 后台, 6-8h)"
echo ""
echo "Task 3 (NSAMPLES消融) 需要等Task 2完成后手动启动"
echo "=========================================="
echo ""

read -p "确认启动所有实验？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 已取消"
    exit 1
fi

# ========== Task 1: 基线测试（前台） ==========
echo ""
echo "▶ Task 1: 测试基线模型PPL..."
bash test_baseline_ppl.sh

# ========== Task 2: 对照实验（后台GPU 0） ==========
echo ""
echo "▶ Task 2: 启动参数对照实验（后台）..."
screen -dmS contrast bash -c "cd /data/zhangs/LLMs/Projects/SAES-SVD && bash scripts/ablation/run_contrast_experiments.sh"
echo "✅ 对照实验已启动 (screen: contrast)"
echo "   查看进度: screen -r contrast"

# ========== Task 4: LoRA微调（后台GPU 1） ==========
echo ""
echo "▶ Task 4: 启动LoRA微调（后台GPU 1）..."

# 创建修改版LoRA脚本（使用GPU 1）
sed 's/CUDA_DEVICE=0/CUDA_DEVICE=1/g' "${PROJECT_ROOT}/scripts/lora/run_lora_finetuning.sh" > /tmp/run_lora_finetuning_gpu1.sh
chmod +x run_lora_finetuning_gpu1.sh

screen -dmS lora bash -c "bash /tmp/run_lora_finetuning_gpu1.sh"
echo "✅ LoRA微调已启动 (screen: lora, GPU 1)"
echo "   查看进度: screen -r lora"

# ========== 显示状态 ==========
echo ""
echo "=========================================="
echo "✅ 所有实验已启动！"
echo "=========================================="
echo ""
echo "当前活跃的screen会话："
screen -ls
echo ""
echo "📊 监控命令："
echo "  查看对照实验: screen -r contrast"
echo "  查看LoRA微调: screen -r lora"
echo "  查看GPU状态: watch -n 1 nvidia-smi"
echo ""
echo "📋 下一步操作："
echo "  1. 等待Task 2完成（约4.5小时）"
echo "  2. 查看结果: cat v2.1.4_contrast_experiments/results.csv"
echo "  3. 根据最优rho修改run_nsamples_ablation.sh"
echo "  4. 启动Task 3: screen -dmS nsamples bash run_nsamples_ablation.sh"
echo ""
echo "=========================================="
