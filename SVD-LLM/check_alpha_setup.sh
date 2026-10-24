#!/bin/bash

echo "=========================================="
echo "  Alpha区间消融实验环境检查"
echo "=========================================="

# 1. 检查脚本文件
echo ""
echo "📁 检查脚本文件..."
if [ -f "run_alpha_ablation_gpu0.sh" ]; then
    echo "  ✅ GPU 0脚本: run_alpha_ablation_gpu0.sh"
else
    echo "  ❌ GPU 0脚本缺失!"
fi

if [ -f "run_alpha_ablation_gpu1.sh" ]; then
    echo "  ✅ GPU 1脚本: run_alpha_ablation_gpu1.sh"
else
    echo "  ❌ GPU 1脚本缺失!"
fi

# 2. 检查Python文件是否有alpha参数
echo ""
echo "🐍 检查Python代码修改..."
if grep -q "parser.add_argument('--alpha_min'" SAES-SVD_v2.1.3.py; then
    echo "  ✅ alpha_min参数已添加"
else
    echo "  ❌ alpha_min参数缺失!"
fi

if grep -q "parser.add_argument('--alpha_max'" SAES-SVD_v2.1.3.py; then
    echo "  ✅ alpha_max参数已添加"
else
    echo "  ❌ alpha_max参数缺失!"
fi

# 3. 检查GPU状态
echo ""
echo "🎮 检查GPU状态..."
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | while IFS=, read idx name util mem_used mem_total; do
    echo "  GPU $idx: $name"
    echo "    使用率: $util"
    echo "    显存: $mem_used / $mem_total"
done

# 4. 检查conda环境
echo ""
echo "🔧 检查conda环境..."
if conda env list | grep -q "compress"; then
    echo "  ✅ conda环境 'compress' 存在"
else
    echo "  ❌ conda环境 'compress' 不存在!"
fi

# 5. 检查模型路径
echo ""
echo "📦 检查模型路径..."
if [ -d "/data/zhangs/LLMs/Models/jeffwan/llama-7b-hf" ]; then
    echo "  ✅ LLaMA-7B模型存在"
else
    echo "  ❌ LLaMA-7B模型路径不存在!"
fi

# 6. 统计实验配置
echo ""
echo "📊 实验配置统计..."
echo "  GPU 0实验数: $(grep -c "^    \"Exp-" run_alpha_ablation_gpu0.sh)"
echo "  GPU 1实验数: $(grep -c "^    \"Exp-" run_alpha_ablation_gpu1.sh)"

echo ""
echo "=========================================="
echo "  环境检查完成!"
echo "=========================================="
echo ""
echo "如果所有检查都通过，可以运行："
echo "  bash run_alpha_ablation_gpu0.sh  # 在screen中运行"
echo "  bash run_alpha_ablation_gpu1.sh  # 在screen中运行"
