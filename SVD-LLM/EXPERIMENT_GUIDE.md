# SAES-SVD 优化实验执行指南

## 📋 实验目标

基于alpha消融实验（Exp-B5 最优结果 PPL=8.055），进行以下4个任务：
1. 测试基线模型PPL（确认论文基线）
2. 修正参数对照实验（优化rho和beta_cap）
3. NSAMPLES消融实验（找到最优校准样本数）
4. LoRA微调（进一步提升精度）

---

## 🎯 Task 1: 测试基线模型PPL

### 目的
确认原始LLaMA-7B在wikitext2上的困惑度，对比论文报告的基线PPL=5.68

### 执行命令
```bash
cd /data/zhangs/LLMs/SVD-LLM
chmod +x test_baseline_ppl.sh
bash test_baseline_ppl.sh
```

### 预期结果
- 论文报告: PPL=5.68
- 实际测试: PPL=? (需要确认)
- **关键**: 如果实际PPL≠5.68，说明基线不同，需要调整期望值

### 估计时间
约10-15分钟

---

## 🔬 Task 2: 修正参数对照实验

### 目的
测试v2.1.4参数修正（rho=0.95, beta_cap=0.375）是否改善avg_beta分布

### 背景问题
Exp-B5结果显示：
- avg_beta=0.287 < 理论下界0.286 (alpha=0.4对应值)
- 说明rho=0.8过度收缩了最优β

### 执行命令
```bash
cd /data/zhangs/LLMs/SVD-LLM
chmod +x run_contrast_experiments.sh

# 方式1: 直接运行（前台，约4.5小时）
bash run_contrast_experiments.sh

# 方式2: 使用screen（后台，推荐）
screen -S contrast
bash run_contrast_experiments.sh
# Ctrl+A+D 分离会话
# screen -r contrast 恢复会话
```

### 实验配置
| ExpID | Version | rho | alpha | 说明 |
|-------|---------|-----|-------|------|
| Contrast-A | v2.1.3 | 0.8 | [0.4,0.6] | 基线重现 |
| Contrast-B | v2.1.4 | 0.95 | [0.4,0.6] | 修正版 |
| Contrast-C | v2.1.4 | 1.0 | [0.4,0.6] | 无收缩 |

### 预期结果
```
Contrast-A: avg_beta≈0.287, PPL≈8.05 (已知)
Contrast-B: avg_beta≈0.33-0.35, PPL≈7.9-8.0 (预期改善)
Contrast-C: avg_beta≈0.36-0.37, PPL≈7.95-8.05 (可能过高)
```

### 关键指标
- **avg_beta是否进入[0.286, 0.375]合理区间**
- **PPL是否改善**
- **选择最优rho进入Task 3**

### 估计时间
约4.5小时（3个实验 × 90分钟）

### 结果分析
```bash
# 查看结果
cat v2.1.4_contrast_experiments/results.csv

# 对比avg_beta
grep "Average β" v2.1.4_contrast_experiments/*.log
```

---

## 📊 Task 3: NSAMPLES消融实验

### 目的
测试不同校准样本数对性能的影响，找到性价比最优点

### ⚠️ 重要提示
**必须先完成Task 2，根据结果修改脚本中的最优配置！**

### 预处理步骤
```bash
# 1. 查看Task 2结果
cat v2.1.4_contrast_experiments/results.csv

# 2. 确定最优rho（假设是0.95）
BEST_RHO=0.95

# 3. 修改run_nsamples_ablation.sh
nano run_nsamples_ablation.sh
# 修改以下行：
#   BEST_VERSION="SAES-SVD_v2.1.4.py"
#   BEST_RHO=0.95  # <-- 根据Task 2结果修改
#   BEST_EPS="1e-4"
#   BEST_ALPHA_MIN=0.4
#   BEST_ALPHA_MAX=0.6
```

### 执行命令
```bash
cd /data/zhangs/LLMs/SVD-LLM
chmod +x run_nsamples_ablation.sh

# 使用screen（后台运行，推荐）
screen -S nsamples
bash run_nsamples_ablation.sh
# Ctrl+A+D 分离会话
```

### 实验配置
| ExpID | NSAMPLES | 估计时间 | 说明 |
|-------|----------|---------|------|
| NS-64 | 64 | ~70min | 快速但可能不准确 |
| NS-128 | 128 | ~90min | 当前基线 |
| NS-256 | 256 | ~120min | 论文可能配置 |
| NS-512 | 512 | ~180min | 高精度 |
| NS-1024 | 1024 | ~300min | 极高精度（可选） |

### 预期结果
```
趋势预期:
  NSAMPLES ↑ → PPL ↓ (收益递减)
  NSAMPLES ↑ → 时间 ↑ (线性增长)

可能结果:
  NS-64:   PPL≈8.2-8.3
  NS-128:  PPL≈8.0-8.1 (已知)
  NS-256:  PPL≈7.8-7.9
  NS-512:  PPL≈7.7-7.85
  NS-1024: PPL≈7.65-7.8
```

### 关键分析
1. **性价比曲线**: 绘制 PPL vs NSAMPLES 和 时间 vs NSAMPLES
2. **边际收益**: 计算每增加一倍样本的PPL改善
3. **论文推测**: 如果256能达到论文水平，说明论文可能用256
4. **最优选择**: 找到收益/时间最优平衡点

### 估计时间
总计约12-15小时（根据是否运行NS-1024）

### 结果分析
```bash
# 查看结果
cat nsamples_ablation/results.csv

# 生成趋势图（需要Python）
python -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('nsamples_ablation/results.csv')
plt.plot(df['NSAMPLES'], df['PPL'], marker='o')
plt.xlabel('NSAMPLES')
plt.ylabel('PPL')
plt.title('NSAMPLES vs PPL')
plt.grid(True)
plt.savefig('nsamples_trend.png')
print('图表已保存至 nsamples_trend.png')
"
```

---

## 🎓 Task 4: LoRA微调

### 目的
对Exp-B5压缩模型进行两阶段LoRA微调，进一步提升精度

### 背景
参考SVDLLM的标准流程：
1. 第一阶段: 微调U投影矩阵（q_u, k_u, v_u, o_u, gate_u, down_u, up_u）
2. 第二阶段: 微调V投影矩阵（q_v, k_v, v_v, o_v, gate_v, down_v, up_v）

### 执行命令
```bash
cd /data/zhangs/LLMs/SVD-LLM
chmod +x run_lora_finetuning.sh

# 使用screen（后台运行，推荐）
screen -S lora
bash run_lora_finetuning.sh
# Ctrl+A+D 分离会话
```

### 流程步骤
```
Step 1: 评估压缩模型基线 (PPL ≈ 8.05)
        ↓
Step 2: 第一阶段LoRA训练 (3 epochs, U projections)
        ↓
Step 3: 合并第一阶段权重
        ↓
Step 4: 评估第一阶段模型 (预期 PPL ≈ 7.5-7.8)
        ↓
Step 5: 第二阶段LoRA训练 (3 epochs, V projections)
        ↓
Step 6: 合并第二阶段权重
        ↓
Step 7: 评估最终模型 (预期 PPL ≈ 7.2-7.5)
```

### 预期结果
```
Baseline (仅压缩):     PPL ≈ 8.05
First LoRA (U更新):    PPL ≈ 7.5-7.8  (-0.25 ~ -0.55)
Second LoRA (V更新):   PPL ≈ 7.2-7.5  (-0.3 ~ -0.3)
----------------------------------------
总改善:                 PPL ≈ -0.55 ~ -0.85
```

### 重要参数
```python
lora_r = 8            # LoRA秩（低秩近似维度）
lora_alpha = 16       # LoRA缩放系数
lora_dropout = 0.05   # Dropout率
num_epochs = 3        # 训练轮数
learning_rate = 1e-4  # 学习率
batch_size = 64       # 批大小
```

### 估计时间
- 第一阶段训练: 约3-4小时
- 第二阶段训练: 约3-4小时
- **总计**: 约6-8小时

### 监控命令
```bash
# 查看训练进度
screen -r lora

# 查看日志
tail -f exp_b5_lora_tuning/first_stage_training.log
tail -f exp_b5_lora_tuning/second_stage_training.log

# 查看评估结果
cat exp_b5_lora_tuning/baseline_eval.log
cat exp_b5_lora_tuning/first_stage_eval.log
cat exp_b5_lora_tuning/final_eval.log
```

### 故障排查
```bash
# 如果遇到CUDA OOM错误
# 修改run_lora_finetuning.sh中的batch_size:
#   --batch_size 64 → --batch_size 32
#   --micro_batch_size 4 → --micro_batch_size 2

# 如果tokenizer报错
# 检查LoRA.py中的tokenizer修复代码（已包含）
```

---

## 📊 完整实验时间表

| Task | 描述 | 估计时间 | 依赖 | 设备 |
|------|------|---------|------|------|
| Task 1 | 基线PPL测试 | 15min | 无 | GPU 0 |
| Task 2 | 参数对照实验 | 4.5h | Task 1 | GPU 0 |
| Task 3 | NSAMPLES消融 | 12-15h | Task 2 | GPU 0 |
| Task 4 | LoRA微调 | 6-8h | 无（独立） | GPU 0 |

### 并行策略
- **Option 1** (串行): Task 1 → Task 2 → Task 3 → Task 4 (总计23-28小时)
- **Option 2** (部分并行): 
  - GPU 0: Task 1 → Task 2 → Task 3
  - GPU 1: Task 4（与Task 2同时开始）
  - 总计: max(4.5+15, 8) ≈ 19.5小时

---

## 🎯 下一阶段实验顺序（推荐）

### 🔴 Phase 1: 基础验证（立即执行）
```bash
# 1. 测试基线（15min）
bash test_baseline_ppl.sh

# 2. 启动对照实验（后台4.5h）
screen -S contrast
bash run_contrast_experiments.sh
# Ctrl+A+D

# 3. 同时启动LoRA微调（后台6-8h，GPU 1）
screen -S lora
# 修改run_lora_finetuning.sh中的CUDA_DEVICE=1
bash run_lora_finetuning.sh
# Ctrl+A+D
```

### 🟡 Phase 2: 参数优化（等待Task 2完成）
```bash
# 4. 分析Task 2结果，确定最优rho
cat v2.1.4_contrast_experiments/results.csv

# 5. 修改run_nsamples_ablation.sh中的BEST_RHO
nano run_nsamples_ablation.sh

# 6. 启动NSAMPLES消融（后台12-15h）
screen -S nsamples
bash run_nsamples_ablation.sh
# Ctrl+A+D
```

### 🟢 Phase 3: 结果分析（所有实验完成后）
```bash
# 7. 汇总所有结果
echo "=== Task 1: Baseline ==="
grep "wikitext2" test_baseline_ppl.log

echo "=== Task 2: Contrast ==="
cat v2.1.4_contrast_experiments/results.csv

echo "=== Task 3: NSAMPLES ==="
cat nsamples_ablation/results.csv

echo "=== Task 4: LoRA ==="
cat exp_b5_lora_tuning/baseline_eval.log
cat exp_b5_lora_tuning/final_eval.log

# 8. 绘制趋势图
python analysis_scripts/plot_results.py  # 需要单独创建

# 9. 撰写实验报告
# 对比:
#   - 原始基线 vs 压缩后 vs LoRA后
#   - NSAMPLES效应
#   - 参数修正效果
```

---

## ✅ 成功标准

### Task 1
- ✅ 成功运行基线评估
- ✅ 获得准确的原始模型PPL

### Task 2
- ✅ 三个对照实验全部成功
- ✅ avg_beta进入合理区间[0.286, 0.375]
- ✅ 找到最优rho配置

### Task 3
- ✅ 至少完成4个NSAMPLES实验（64, 128, 256, 512）
- ✅ 观察到NSAMPLES↑ → PPL↓趋势
- ✅ 找到性价比最优点

### Task 4
- ✅ 两阶段LoRA训练成功完成
- ✅ PPL相比压缩基线改善≥0.5
- ✅ 最终PPL < 7.6

---

## 📞 故障处理

### CUDA Out of Memory
```bash
# 修改batch_size和micro_batch_size
# 或使用单个GPU + 显存监控
watch -n 1 nvidia-smi
```

### Screen会话管理
```bash
screen -ls              # 列出所有会话
screen -r contrast      # 恢复contrast会话
screen -X -S contrast quit  # 终止contrast会话
```

### 实验中断恢复
```bash
# Task 2-3可以单独运行特定实验
# 修改脚本，注释掉已完成的实验

# Task 4可以从任意步骤重启
# 检查output目录中已完成的步骤
```

---

## 📈 预期最终结果

```
原始LLaMA-7B:           PPL ≈ 5.68 (论文基线)
├─ SAES-SVD压缩(80%):   PPL ≈ 8.05 (Exp-B5)
│  ├─ v2.1.4优化:       PPL ≈ 7.9-8.0 (Task 2)
│  └─ NSAMPLES优化:     PPL ≈ 7.7-7.85 (Task 3)
│
└─ + LoRA微调:          PPL ≈ 7.2-7.5 (Task 4)

目标: 缩小与论文差距 (当前8.05 → 目标7.2)
论文可能还使用了: 更多微调轮数、混合精度训练、后处理技术
```

---

## 🎓 关键洞察

1. **参数护栏的影响**: rho=0.8过度收缩了β，需要0.95-1.0
2. **校准样本的重要性**: NSAMPLES可能是论文差距的关键因素
3. **LoRA的必要性**: 压缩后必须微调才能恢复精度
4. **基线的确认**: 如果基线PPL≠5.68，需要重新设定期望值

---

**作者**: SAES-SVD优化团队  
**日期**: 2025-12-30  
**版本**: v1.0
