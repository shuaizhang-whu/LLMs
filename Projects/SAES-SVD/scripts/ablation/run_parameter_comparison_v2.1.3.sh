#!/bin/bash
# 参数对比实验脚本 v2.1.3
# 用途：对比 eps、rho 和 NSAMPLES 参数对 PPL 的影响

echo "============================================"
echo "�� SAES-SVD 参数对比实验 v2.1.3"
echo "============================================"
echo ""
echo "实验1: eps=1e-3 (测试正则化参数影响)"
echo "实验2: rho=0.95 (测试收缩因子影响)"
echo "实验3: NSAMPLES=256 (测试样本数量影响)"
echo "对照组: NSAMPLES=128, eps=1e-4, rho=1.0, alpha_min=0.25, alpha_max=0.75, beta_cap=0.5 (默认配置)"
echo ""

# 创建实验结果目录
RESULT_DIR="./parameter_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

echo "📁 结果保存至: $RESULT_DIR"
echo ""

# # 实验1: 测试 eps=1e-3
# echo "=========================================="
# echo "🧪 实验1: eps=1e-3, rho=1.0, NSAMPLES=128"
# echo "=========================================="

# # 修改参数
# sed -i 's/^EPS=.*/EPS="1e-3"/' saes_svd_v2.1.3.sh
# sed -i 's/^RHO=.*/RHO="1.0"/' saes_svd_v2.1.3.sh
# sed -i 's/^NSAMPLES=.*/NSAMPLES="128"/' saes_svd_v2.1.3.sh

# # 运行
# bash saes_svd_v2.1.3.sh 0.2 2>&1 | tee "$RESULT_DIR/exp1_eps1e-3_rho1.0_ns128.log"

# # 提取结果
# echo "📈 实验1结果:" >> "$RESULT_DIR/summary.txt"
# grep "PPL after pruning" "$RESULT_DIR/exp1_eps1e-3_rho1.0_ns128.log" | tail -1 >> "$RESULT_DIR/summary.txt"
# grep "Average β across all layers" "$RESULT_DIR/exp1_eps1e-3_rho1.0_ns128.log" | tail -1 >> "$RESULT_DIR/summary.txt"
# echo "" >> "$RESULT_DIR/summary.txt"

# echo ""
# echo "✅ 实验1完成，等待10秒后开始实验2..."
# sleep 10

# 实验2: 测试 rho=0.95
echo "=========================================="
echo "🧪 实验2: eps=1e-4, rho=0.95, NSAMPLES=128"
echo "=========================================="

# 修改参数
sed -i 's/^EPS=.*/EPS="1e-4"/' saes_svd_v2.1.3.sh
sed -i 's/^RHO=.*/RHO="0.9"/' saes_svd_v2.1.3.sh
sed -i 's/^NSAMPLES=.*/NSAMPLES="128"/' saes_svd_v2.1.3.sh

# 运行
bash saes_svd_v2.1.3.sh 0.2 2>&1 | tee "$RESULT_DIR/exp2_eps1e-4_rho0.9_ns128.log"

# 提取结果
echo "📈 实验2结果:" >> "$RESULT_DIR/summary.txt"
grep "PPL after pruning" "$RESULT_DIR/exp2_eps1e-4_rho0.9_ns128.log" | tail -1 >> "$RESULT_DIR/summary.txt"
grep "Average β across all layers" "$RESULT_DIR/exp2_eps1e-4_rho0.9_ns128.log" | tail -1 >> "$RESULT_DIR/summary.txt"
echo "" >> "$RESULT_DIR/summary.txt"

# echo ""
# echo "✅ 实验2完成，等待10秒后开始实验3..."
# sleep 10

# # 实验3: 测试 NSAMPLES=256
# echo "=========================================="
# echo "🧪 实验3: eps=1e-4, rho=1.0, NSAMPLES=256"
# echo "=========================================="

# # 修改参数
# sed -i 's/^EPS=.*/EPS="1e-4"/' saes_svd_v2.1.3.sh
# sed -i 's/^RHO=.*/RHO="1.0"/' saes_svd_v2.1.3.sh
# sed -i 's/^NSAMPLES=.*/NSAMPLES="256"/' saes_svd_v2.1.3.sh

# # 运行
# bash saes_svd_v2.1.3.sh 0.2 2>&1 | tee "$RESULT_DIR/exp3_eps1e-4_rho1.0_ns256.log"

# # 提取结果
# echo "📈 实验3结果:" >> "$RESULT_DIR/summary.txt"
# grep "PPL after pruning" "$RESULT_DIR/exp3_eps1e-4_rho1.0_ns256.log" | tail -1 >> "$RESULT_DIR/summary.txt"
# grep "Average β across all layers" "$RESULT_DIR/exp3_eps1e-4_rho1.0_ns256.log" | tail -1 >> "$RESULT_DIR/summary.txt"
# echo "" >> "$RESULT_DIR/summary.txt"

# # 恢复默认参数
# sed -i 's/^EPS=.*/EPS="1e-4"/' saes_svd_v2.1.3.sh
# sed -i 's/^RHO=.*/RHO="1.0"/' saes_svd_v2.1.3.sh
# sed -i 's/^NSAMPLES=.*/NSAMPLES="128"/' saes_svd_v2.1.3.sh

# echo ""
# echo "=========================================="
# echo "✅ 所有实验完成！"
# echo "=========================================="
# echo ""
# echo "📊 结果汇总保存在: $RESULT_DIR/summary.txt"
# cat "$RESULT_DIR/summary.txt"

# echo ""
# echo "💡 分析提示:"
# echo "- PPL 越低越好（表示模型性能越好）"
# echo "- β 值反映误差补偿强度"
# echo "- eps 影响数值稳定性"
# echo "- rho 控制 β 的收缩程度"
# echo "- NSAMPLES 影响统计量准确性和运行时间"
