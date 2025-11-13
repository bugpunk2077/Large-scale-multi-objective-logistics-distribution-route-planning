# 神经网络（NN）使用指南

## 一、NN 的角色与作用

在本 GMOEA 框架中，神经网络（NeuralVRPSolver）可用于以下三个角色：

### 1. 候选解生成器（Generator）
- **作用**：NN 训练完后，可用策略网络贪心生成完整的 VRP 解，无需演化。
- **优势**：快速、一次性前向通过，适合部署在低延迟场景。
- **劣势**：质量通常低于长期演化，单一前向通过无法调整多目标权衡。
- **何时有用**：
  - 已有大量高质量训练数据（数百个实例+对应解）
  - 需要快速响应且可接受 80-90% 质量的场景（例如实时配送快速估价）

### 2. Warm-start 解注入器（Warm-Start Injector）
- **作用**：每代让 NN 生成 M 个候选解，替换种群中最差的 M 个解，加速演化收敛。
- **优势**：结合学习启发式与全局搜索，通常能加快收敛 20-40%。
- **劣势**：需要定期 NN 训练（计算开销），且若 NN 质量差会拖累演化。
- **何时有用**：
  - 有中等规模训练数据（50-200 实例）
  - 计算预算充足（可容纳 NN 训练 + 演化）
  - 目标是在给定时间内找到更好解

### 3. 启发式特征提取器（Feature Extractor）
- **作用**：利用 NN 的编码器部分提取节点/图特征，用于指导演化中的选择、变异或局部搜索。
- **优势**：不依赖 NN 直接生成解，只借用其学到的特征表示。
- **劣势**：实现复杂，需要额外修改演化框架。
- **何时有用**：高级场景，暂不推荐初期采用。

---

## 二、何时 NN 有用、何时无用或有害

### ✅ NN 有用的条件（三个同时满足最佳）
1. **有训练数据**：至少 50-100 个实例的解对（instance + 对应的高质量解或最优解）
2. **计算资源充足**：可容纳 NN 模型存储（~50MB）+ 每代训练时间（~秒级）+ GPU/CPU
3. **目标明确**：追求快速收敛或快速响应，而非单纯"最优解"

### ❌ NN 无用或有害的情况
1. **无训练数据** → NN 从零初始化，行为克隆预训练成本大但效果微弱；纯演化反而更稳定
2. **过拟合** → 用少量数据反复训练同一实例集，NN 只记住那些实例而非泛化启发式
3. **分布失配** → 训练数据来自小实例（20-30 客户），但测试大实例（100+ 客户），NN 生成的解无效性高
4. **计算预算限制** → NN 训练每代需 2-5 秒，只能运行 100 代的总时间内，纯演化可能跑 1000 代更优
5. **冷启动** → 无预训练数据，NN 从随机初始化开始，前期生成的解质量很差，拖累演化

---

## 三、当前推荐配置（基于 your 100+ 客户 VRP 目标）

### 推荐 1：仅用演化，禁用 NN（最稳妥）✅
```python
config = {
    'use_nn': False,  # 禁用 NN
    'use_behavior_cloning': False,
    'population_size': 200-300,
    'max_generations': 2000-5000,  # 加大代数弥补
    'local_search_depth': 'heavy',  # 强化本地搜索代替 NN
}
```
**适用场景**：
- 无预训练数据
- 单次运行时间允许数小时
- 追求最优解质量而不在乎响应速度

**预期性能**：
- 稳定收敛，不受数据质量影响
- 100+ 客户实例：预计 2-5 小时内达到接近最优

---

### 推荐 2：Warm-start 模式（中等风险，高收益）⭐
前提：你有 50+ 个实例的 .sol 最优/高质量解
```python
config = {
    'use_behavior_cloning': True,  # 预训练
    'bc_epochs': 20,               # 多轮预训练
    'use_nn': True,                # 启用 NN
    'population_size': 200,
    'max_generations': 1000,
    'nn_inject_per_gen': int(0.1 * population_size),  # 每代注入 10% 候选
}
```
**操作流程**：
1. 将若干实例的 .sol 文件放在 `data/` 下，文件名与实例名对应
2. 运行训练，脚本会自动检测并加载 .sol，执行行为克隆预训练
3. 演化过程中，NN 每代生成 warm-start 候选

**预期性能**：
- 比纯演化快 20-40%
- 如果预训练数据好，100+ 客户实例可能 1-2 小时完成

**风险**：
- 如果预训练数据质量差（有很多次优解），NN 学坏了会拖累演化
- 需要多次迭代调参

---

### 推荐 3：生产化模式（需充足数据）🚀
前提：你有 100+ 个不同规模/布局的实例数据，且允许离线训练
```python
# 离线预训练阶段（一次性）
config_pretrain = {
    'use_behavior_cloning': True,
    'bc_epochs': 50,  # 充分训练
    'supervised_dataset': 'data/all_instances/',  # 所有实例
}

# 运行后保存模型到 saved_models/gmoea_bc_pretrained.pth

# 在线推理阶段（快速响应）
config_deploy = {
    'use_nn': True,
    'load_pretrained_model': 'saved_models/gmoea_bc_pretrained.pth',
    'max_generations': 100,  # 少量演化微调，快速响应
}
```

---

## 四、如何检测 NN 是否有用（对你的实例）

### 快速诊断脚本（你可运行）
```bash
# 1. 禁用 NN，运行 balanced 预设 200 代
python main_train.py  # config_preset = 'balanced', use_nn=False

# 记录最终 Pareto front 和运行时间 (T_no_nn, obj_no_nn)

# 2. 启用 NN 预训练，运行相同配置
# 修改 config 中 use_behavior_cloning = True, use_nn = True
python main_train.py

# 记录 (T_with_nn, obj_with_nn)

# 3. 比较
if obj_with_nn[0] < obj_no_nn[0] * 0.98 and T_with_nn < T_no_nn * 1.2:
    print("NN 有帮助，建议启用")
else:
    print("NN 无帮助或有害，建议禁用")
```

---

## 五、结论与建议

### 对你目前的情况
根据你的情况（双目标、100+ 客户、无充分预训练数据）：

**现阶段建议**：
1. ✅ **先关闭 NN**（`use_nn=False, use_behavior_cloning=False`）
2. ✅ **加强演化**：增加 population 到 300-500、max_generations 到 3000-5000、启用 `local_search_depth='heavy'`
3. ⏳ **可选未来**：若收集到 100+ 实例的高质量解，再启用 warm-start 模式

### NN 使用决策树
```
是否有 50+ 实例的预训练数据？
  ├─ 是 → 有足够计算资源（GPU）？
  │       ├─ 是 → 推荐模式 2（Warm-start）
  │       └─ 否 → 禁用 NN，加强演化
  └─ 否 → 禁用 NN，仅用演化（推荐模式 1）
```

---

## 六、快速参考：配置开关表

| 情景 | use_nn | use_behavior_cloning | 预期效果 | 运行时间 |
|------|--------|-------------------|--------|--------|
| 无数据+快速收敛 | False | False | ⭐⭐⭐ 稳定 | 2-5h (100+ 客户) |
| 有 50+ 解+足够资源 | True | True | ⭐⭐⭐⭐ 快速+优质 | 1-3h (100+ 客户) |
| 数据少+时间紧 | False | False | ⭐⭐ 一般 | 30-60m (100 代) |
| 生产环境+离线训练 | True | True (仅预训练) | ⭐⭐⭐⭐⭐ 最优 | 10-30s (部署推理) |

---

## 七、代码中如何切换

在 `main_train.py` 的 config 中修改即可（无需改代码逻辑）：

```python
config = {
    'use_nn': False,                    # 关键开关：False=禁用, True=启用
    'use_behavior_cloning': False,      # False=无预训练, True=用 .sol 预训练
    'bc_epochs': 10,                    # 预训练轮数（仅在 use_behavior_cloning=True 时生效）
    'population_size': 300,             # 人口大小（禁用 NN 时建议更大）
    'max_generations': 3000,            # 代数（禁用 NN 时建议更多）
    'local_search_depth': 'heavy',      # 本地搜索强度
}
```

运行后在 `results/` 下查看 JSON 中的 `bc_loss` 和 `training_history`，判断 NN 效果。
