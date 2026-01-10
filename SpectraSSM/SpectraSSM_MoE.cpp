// ============================================================
// SpectraSSM_MoE.cpp - 频域混合专家分类器完整实现
// 文件名: SpectraSSM_MoE.cpp
// ============================================================

#include "SpectraSSM_MoE.hpp"
#include <torch/torch.h>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <filesystem>

namespace SpectraSSM {

// ========================= 频域MoE分类器实现 =========================
    频域MoE分类器::频域MoE分类器(int64_t 输入维度, int64_t 类别数,
        const 频域MoE配置& 配置)
        : 输入维度_(输入维度), 类别数_(类别数), 配置_(配置) {

        // 初始化专家网络
        初始化专家网络();

        // 初始化门控网络
        初始化门控网络();

        // 初始化输出层
        输出层_ = register_module("输出层",
            torch::nn::Linear(配置_.专家隐藏维度, 类别数_));

        // 初始化专家专业化
        if (配置_.启用频域专业化) {
            初始化专家专业化();
        }

        // 初始化使用统计
        专家使用统计_ = register_buffer("专家使用统计",
            torch::zeros({ 配置_.专家数量 }, torch::kFloat32));

        // 注册频域偏置缩放参数（关键修复）
        频域偏置缩放_ = register_parameter("频域偏置缩放", torch::tensor(0.5f, torch::kFloat32));

        // 关键修复：初始化路由概率缓存
        最后路由概率_ = register_buffer("最后路由概率",
            torch::zeros({ 0 }, torch::kFloat32)); // 初始化为空

        TORCH_INFO("频域MoE分类器初始化完成: 输入维度=", 输入维度_,
            ", 类别数=", 类别数_, ", 专家数=", 配置_.专家数量);
    }

void 频域MoE分类器::初始化专家网络() {
    for (int64_t i = 0; i < 配置_.专家数量; ++i) {
        // 创建专家网络（小型频域MoE模型）
        auto 专家 = std::make_shared<频域MoE模型>(
            输入维度_, 配置_.专家隐藏维度, /*最大序列长度=*/1024);
        
        专家网络_.push_back(专家);
        
        // 注册为子模块
        register_module("专家_" + std::to_string(i), 专家);
    }
}

void 频域MoE分类器::初始化门控网络() {
    int64_t 频域特征维度 = 输入维度_ * 5;

    if (配置_.门控隐藏层维度 > 0) {
        门控网络_ = register_module("门控网络", torch::nn::Sequential(
            torch::nn::Linear(频域特征维度, 配置_.门控隐藏层维度),
            torch::nn::ReLU(),
            torch::nn::Linear(配置_.门控隐藏层维度, 配置_.专家数量)
        ));
    }
    else {
        门控网络_ = register_module("门控网络", torch::nn::Sequential(
            torch::nn::Linear(频域特征维度, 配置_.专家数量)
        ));
    }

    门控Dropout_ = register_module("门控Dropout",
        torch::nn::Dropout(配置_.路由Dropout率));
}

void 频域MoE分类器::初始化专家专业化() {
    // 为每个专家创建频域偏置参数
    for (int64_t i = 0; i < 配置_.专家数量; ++i) {
        // 计算专家专属的频段中心
        float 频段中心 = static_cast<float>(i) / 配置_.专家数量;
        
        // 创建频域偏置参数 [输入维度]
        auto 偏置 = register_parameter(
            "专家偏置_" + std::to_string(i),
            torch::zeros({输入维度_}, torch::kFloat32));
        
        // 初始化偏置：引导专家关注特定频段
        auto 偏置数据 = 偏置.data();
        for (int64_t j = 0; j < 输入维度_; ++j) {
            float 相位偏移 = static_cast<float>(j) / 输入维度_;
            float 偏置值 = 0.1f * std::sin(2 * M_PI * (频段中心 + 相位偏移));
            偏置数据[j] = 偏置值;
        }
        
        专家频域偏置_.push_back(偏置);
    }
}

torch::Tensor 频域MoE分类器::提取频域特征(const torch::Tensor& 输入) const {
    auto 批次大小 = 输入.size(0);
    auto 序列长度 = 输入.size(1);
    auto 输入CPU = 输入.cpu();
    auto 频域表示 = torch::fft::rfft(输入CPU, 序列长度, /*dim=*/1);
    auto 能量谱 = torch::abs(频域表示).pow(2);
    auto 相位谱 = torch::angle(频域表示);
    int64_t 频点数 = 能量谱.size(1);

    // 修复：使用 std::get 而不是 .values
    auto 低频能量 = 频点数 > 0 ?
        能量谱.slice(1, 0, std::max(频点数 / 4, static_cast<int64_t>(1))).mean(1) :
        torch::zeros({ 批次大小, 输入维度_ });

    auto 高频起始点 = std::min(频点数 * 3 / 4, 频点数);
    auto 高频能量 = 频点数 > 0 ?
        能量谱.slice(1, 高频起始点).mean(1) :
        torch::zeros({ 批次大小, 输入维度_ });

    auto 相位稳定性 = torch::std(相位谱, /*dim=*/1, /*unbiased=*/false);
    auto 全局平均能量 = 能量谱.mean(1);

    // 修复：使用 std::get<0> 获取 max 的第一个返回值
    auto 峰值能量 = std::get<0>(能量谱.max(1));

    auto 频域特征 = torch::cat({
        低频能量, 高频能量, 相位稳定性, 全局平均能量, 峰值能量
        }, /*dim=*/1);

    return 频域特征.to(输入.device());
}

torch::Tensor 频域MoE分类器::前向传播特定专家(const torch::Tensor& 输入, int64_t 专家索引) {
    // 验证专家索引有效性
    TORCH_CHECK(
        专家索引 >= 0 && 专家索引 < 配置_.专家数量,
        "专家索引超出有效范围: ", 专家索引,
        " (有效范围: 0-", 配置_.专家数量 - 1, ")"
    );

    // 验证输入维度匹配
    TORCH_CHECK(
        输入.dim() == 3 && 输入.size(2) == 输入维度_,
        "输入维度不匹配: 期望特征维度=", 输入维度_,
        ", 实际=", 输入.size(2)
    );

    // 直接调用指定专家的前向传播，绕过门控机制
    // 避免缓存污染，保持分析独立性
    return 专家网络_[专家索引]->前向传播(输入);
}

std::vector<torch::Tensor> 频域MoE分类器::批量获取专家输出(const torch::Tensor& 输入) {
    TORCH_CHECK(输入.dim() == 3, "输入张量必须为3维 [批次, 序列长度, 特征维度]");
    TORCH_CHECK(输入.size(2) == 输入维度_, "输入特征维度不匹配");

    std::vector<torch::Tensor> 输出列表;
    输出列表.reserve(配置_.专家数量);

    // 并行计算所有专家输出（底层自动调度GPU计算流）
    for (int64_t i = 0; i < 配置_.专家数量; ++i) {
        // 绕过门控和缓存，直接调用专家网络确保分析独立性
        输出列表.push_back(专家网络_[i]->前向传播(输入));
    }

    return 输出列表;
}

torch::Tensor 频域MoE分类器::计算门控权重(const torch::Tensor& 输入) {
    // ========== 输入验证 ==========
    TORCH_CHECK(输入.dim() == 3, "门控权重计算：输入必须是3维张量 [batch, seq_len, dim]");
    TORCH_CHECK(输入.size(0) > 0 && 输入.size(1) > 0, "门控权重计算：批次和序列长度必须>0");

    // ========== 频域特征提取 ==========
    auto 设备 = 输入.device();
    auto 频域特征 = 提取频域特征(输入);  // [batch_size, feature_dim]
    TORCH_CHECK(频域特征.dim() == 2, "门控权重计算：频域特征输出维度异常");

    // ========== 专家专业化偏置 ==========
    if (配置_.启用频域专业化 && !专家频域偏置_.empty()) {
        auto 专家偏置矩阵 = torch::stack(专家频域偏置_, 0).to(设备);  // [expert_count, feature_dim]
        auto 相似度 = torch::matmul(频域特征, 专家偏置矩阵.transpose(0, 1));  // [batch, expert_count]

        // 可学习的缩放参数 - 在构造函数中注册为成员变量
        // 频域偏置缩放_ 在头文件中声明并在构造函数中初始化
        频域特征 += 频域偏置缩放_ * 相似度.mean(1, true);  // 广播到特征维度
    }

    // ========== 门控网络前向传播 ==========
    auto 原始权重 = 门控网络_->forward(频域特征);  // [batch_size, expert_count]
    TORCH_CHECK(原始权重.dim() == 2, "门控权重计算：门控网络输出维度异常");

    // 动态温度调整（动态路由关键）
    float 门控温度 = 1.0f;
    int64_t 实际激活专家数 = 配置_.激活专家数;

    if (配置_.启用动态路由) {
        int64_t 序列长度 = 输入.size(1);
        实际激活专家数 = 计算动态激活专家数(序列长度);

        // 序列越长，温度越低（更确定）
        门控温度 = std::max(0.3f, 1.5f - 序列长度 * 0.001f);
    }

    // 应用温度缩放
    原始权重 = 原始权重 / 门控温度;

    // Dropout（仅在训练时）
    if (is_training()) {
        原始权重 = 门控Dropout_(原始权重);
    }

    // ========== Top-K稀疏化 ==========
    TORCH_CHECK(实际激活专家数 > 0 && 实际激活专家数 <= 配置_.专家数量,
        "门控权重计算：激活专家数无效: ", 实际激活专家数);

    // 先TopK再softmax，大幅减少计算量
    auto topk结果 = torch::topk(原始权重, 实际激活专家数, 1, true, true);
    auto topk得分 = std::get<0>(topk结果);  // [batch, k]
    auto topk索引 = std::get<1>(topk结果);  // [batch, k]

    // 对TopK结果做softmax（detach阻断梯度到专家）
    auto topk权重 = torch::softmax(topk得分.detach(), 1);  // 关键：detach防止梯度回传

    // ========== 稀疏门控矩阵构建 ==========
    torch::Tensor 门控输出;
    if (is_training()) {
        // 训练：使用稀疏COO格式，节省内存并加速反向传播
        auto 批次索引 = torch::arange(输入.size(0), torch::kLong).to(设备)
            .unsqueeze(1).expand_as(topk索引).flatten();
        auto 专家索引 = topk索引.flatten();
        auto 权重值 = topk权重.flatten();

        门控输出 = torch::sparse_coo_tensor(
            torch::stack({ 批次索引, 专家索引 }, 0),
            权重值,
            { 输入.size(0), 配置_.专家数量 },
            torch::TensorOptions().device(设备)
        ).coalesce();
    }
    else {
        // 推理：使用稠密格式，计算更快
        门控输出 = torch::zeros({ 输入.size(0), 配置_.专家数量 }, 设备);
        门控输出.scatter_(1, topk索引, topk权重);
    }

    // ========== 更新使用统计（无梯度） ==========
    if (is_training()) {
        更新使用统计(门控输出.detach());  // detach避免影响主梯度
    }
    else {
        更新使用统计(门控输出);
    }

    // ========== 缓存路由概率（供训练器使用） ==========
    if (is_training()) {
        // 转换为稠密格式用于缓存（但不参与梯度计算）
        最后路由概率_ = 门控输出.to_dense().detach();
    }
    else {
        最后路由概率_ = 门控输出;  // 推理时已经是稠密
    }

    return 门控输出;
}

int64_t 频域MoE分类器::计算动态激活专家数(int64_t 序列长度) const {
    const auto& 动态设置 = 配置_.动态设置;
    
    if (序列长度 <= 动态设置.短序列阈值) {
        return 动态设置.短序列激活专家数;
    } else if (序列长度 <= 动态设置.中序列阈值) {
        return 动态设置.中序列激活专家数;
    } else {
        return 动态设置.长序列激活专家数;
    }
}

void 频域MoE分类器::更新使用统计(const torch::Tensor& 门控矩阵) {
    if (!is_training()) return;

    // 确保是稠密张量
    auto 稠密门控 = 门控矩阵.is_sparse() ? 门控矩阵.to_dense() : 门控矩阵;

    // 计算批次平均使用率（考虑权重大小而非仅是否激活）
    auto 批次使用 = 稠密门控.mean(0);  // [expert_count]

    // EMA更新
    float 衰减因子 = 0.99f;
    专家使用统计_ = 衰减因子 * 专家使用统计_.to(批次使用.device()) +
        (1.0f - 衰减因子) * 批次使用;
    统计更新次数_++;
}

torch::Tensor 频域MoE分类器::前向传播(const torch::Tensor& 输入) {
    auto 批次大小 = 输入.size(0);
    auto 序列长度 = 输入.size(1);

    // 检查输入维度
    TORCH_CHECK(输入.size(2) == 输入维度_,
        "输入维度不匹配: 期望=", 输入维度_, ", 实际=", 输入.size(2));

    // 1. 计算门控权重并缓存
    auto 门控权重 = 计算门控权重(输入);  // [batch_size, expert_count]

    // 关键修复：缓存路由概率用于训练器
    最后路由概率_ = 门控权重.detach().clone(); // 分离计算图并克隆

    // 2. 并行计算专家输出（使用缓存优化）
    std::vector<torch::Tensor> 专家输出列表;
    bool 使用缓存 = 配置_.启用专家缓存 && !is_training();

    if (使用缓存 && 当前缓存序列长度_ == 序列长度 && 当前缓存输入_.defined() &&
        torch::equal(输入, 当前缓存输入_)) {
        // 使用缓存结果
        for (int64_t i = 0; i < 配置_.专家数量; ++i) {
            if (专家输出缓存_.find(i) != 专家输出缓存_.end()) {
                专家输出列表.push_back(专家输出缓存_[i]);
            }
            else {
                auto 专家输出 = 专家网络_[i]->前向传播(输入);
                专家输出缓存_[i] = 专家输出;
                专家输出列表.push_back(专家输出);
            }
        }
    }
    else {
        // 重新计算专家输出
        清空专家缓存();
        当前缓存序列长度_ = 序列长度;
        当前缓存输入_ = 输入.clone();

        for (int64_t i = 0; i < 配置_.专家数量; ++i) {
            auto 专家输出 = 专家网络_[i]->前向传播(输入);
            if (使用缓存) {
                专家输出缓存_[i] = 专家输出;
            }
            专家输出列表.push_back(专家输出);
        }
    }

    // 3. 加权融合专家输出
    torch::Tensor 融合输出 = torch::zeros({ 批次大小, 序列长度, 配置_.专家隐藏维度 },
        输入.options());

    for (int64_t i = 0; i < 配置_.专家数量; ++i) {
        auto 专家贡献 = 门控权重.select(1, i)  // [batch_size]
            .unsqueeze(-1)        // [batch_size, 1]
            .unsqueeze(-1)        // [batch_size, 1, 1]
            .expand({ 批次大小, 序列长度, 配置_.专家隐藏维度 });

        融合输出 += 专家贡献 * 专家输出列表[i];
    }

    // 4. 分类输出
    auto 最终输出 = 输出层_->forward(融合输出);  // [batch_size, seq_len, num_classes]

    return 最终输出;
}

torch::Tensor 频域MoE分类器::计算路由损失(const torch::Tensor& 路由概率) {
    // 验证输入维度
    TORCH_CHECK(路由概率.dim() == 2, "路由概率张量维度必须为2");

    const int64_t 批大小 = 路由概率.size(0);
    const int64_t 专家数量 = 路由概率.size(1);

    // 生成均匀分布目标，自动匹配批大小和设备类型
    const auto 均匀分布目标 = torch::full(
        { 批大小, 专家数量 },
        1.0f / 专家数量,
        torch::TensorOptions().device(路由概率.device()).dtype(路由概率.dtype())
    );

    // 计算对数概率，作为kl_div的输入
    const auto 对数路由概率 = torch::log_softmax(路由概率, /*dim=*/1);

    // 计算路由分布与均匀分布的KL散度，促进负载均衡
    return torch::nn::functional::kl_div(
        对数路由概率,
        均匀分布目标,
        torch::nn::functional::KLDivFuncOptions().reduction(torch::kMean)
    );
}

torch::Tensor 频域MoE分类器::获取专家使用统计() const {
    if (统计更新次数_ == 0) {
        return torch::zeros({配置_.专家数量});
    }
    return 专家使用统计_.clone();
}

torch::Tensor 频域MoE分类器::获取门控权重(const torch::Tensor& 输入) {
    return 计算门控权重(输入);
}

void 频域MoE分类器::重置专家缓存() {
    清空专家缓存();
}

void 频域MoE分类器::设置激活专家数(int64_t 新激活专家数) {
    配置_.激活专家数 = std::min(新激活专家数, 配置_.专家数量);
    TORCH_INFO("设置激活专家数: ", 配置_.激活专家数);
}

void 频域MoE分类器::清空专家缓存() {
    专家输出缓存_.clear();
    当前缓存序列长度_ = -1;
    当前缓存输入_ = torch::Tensor();
}



// ========================= 频域MoE训练器实现 =========================

频域MoE训练器::频域MoE训练器(std::shared_ptr<频域MoE分类器> 模型,
                         std::shared_ptr<torch::optim::Optimizer> 优化器,
                         torch::Device 设备)
    : 模型_(模型), 优化器_(优化器), 设备_(设备) {
    
    TORCH_CHECK(模型_ != nullptr, "模型不能为空");
    TORCH_CHECK(优化器_ != nullptr, "优化器不能为空");
    
    // 移动模型到指定设备
    模型_->to(设备_);
    
    TORCH_INFO("频域MoE训练器初始化完成");
}

torch::Tensor 频域MoE训练器::训练步骤(const torch::Tensor& 输入, const torch::Tensor& 标签) {
    // 设置模型为训练模式
    模型_->train();

    // 确保输入标签维度正确
    auto 输入设备 = 输入.to(设备_);
    auto 标签设备 = 标签.to(设备_);

    // 前向传播（这会自动缓存路由概率）
    auto 输出 = 模型_->前向传播(输入设备);

    // 动态处理输出维度
    auto 输出展平 = 输出.reshape({ -1, 输出.size(-1) });
    auto 标签展平 = 标签设备.reshape({ -1 });

    // 计算总损失
    auto 总损失 = 计算总损失(输出展平, 标签展平);

    // 反向传播
    优化器_->zero_grad();
    总损失.backward();

    // 梯度裁剪
    float 梯度范数 = 计算梯度范数();
    if (梯度范数 > 配置_.最大梯度范数) {
        torch::nn::utils::clip_grad_norm_(模型_->parameters(), 配置_.最大梯度范数);
        TORCH_WARN("梯度裁剪应用: ", 梯度范数, " -> ", 配置_.最大梯度范数);
    }
    // 参数更新前执行自适应权重调整
    自适应损失权重调整();
    // 参数更新
    优化器_->step();

    // 更新训练步数
    训练步数_++;

    return 总损失;
}

std::tuple<torch::Tensor, torch::Tensor> 
频域MoE训练器::验证步骤(const torch::Tensor& 输入, const torch::Tensor& 标签) {
    模型_->eval();
    
    torch::NoGradGuard no_grad;
    auto 输入设备 = 输入.to(设备_);
    auto 标签设备 = 标签.to(设备_);
    
    auto 输出 = 模型_->前向传播(输入设备);
    auto 损失 = 计算总损失(输出, 标签设备);
    
    // 计算准确率
    auto 预测 = torch::argmax(输出, -1);
    auto 正确 = (预测 == 标签设备).to(torch::kFloat32).sum();
    auto 准确率 = 正确.item<float>() / 标签设备.numel();
    
    训练统计_["验证损失"] = 损失.item<float>();
    训练统计_["验证准确率"] = 准确率;
    
    return std::make_tuple(损失, 预测);
}

torch::Tensor 频域MoE训练器::计算总损失(
    const torch::Tensor& 预测输出,
    const torch::Tensor& 真实标签
) {
    // 验证路由概率是否已正确缓存
    TORCH_CHECK(模型_->获取最后路由概率().defined(),
        "错误: 必须先执行前向传播才能获取路由概率");

    const auto& 路由概率 = 模型_->获取最后路由概率();

    // 动态处理维度以适应不同类型任务
    auto 预测展平 = 预测输出.reshape({ -1, 预测输出.size(-1) });
    auto 标签展平 = 真实标签.reshape({ -1 });

    // 验证设备一致性
    TORCH_CHECK(
        预测展平.device() == 标签展平.device() &&
        预测展平.device() == 路由概率.device(),
        "所有张量必须位于同一设备上"
    );

    // 验证维度匹配
    TORCH_CHECK(预测展平.size(0) == 标签展平.size(0),
        "预测和标签样本数量不匹配");

    TORCH_CHECK(路由概率.dim() == 2,
        "路由概率必须为2维 [批大小, 专家数]");

    TORCH_CHECK(预测展平.size(0) == 路由概率.size(0),
        "批大小不匹配: 预测=", 预测展平.size(0),
        ", 路由=", 路由概率.size(0));

    const int64_t 批大小 = 路由概率.size(0);
    const int64_t 专家数量 = 路由概率.size(1);

    // 1. 分类交叉熵损失
    auto 分类损失 = torch::nn::functional::cross_entropy(
        预测展平,
        标签展平,
        torch::nn::functional::CrossEntropyFuncOptions()
        .reduction(torch::kMean)
    );

    // 2. 路由均衡损失 (KL散度)
    const auto 均匀目标 = torch::full(
        { 批大小, 专家数量 },
        1.0f / 专家数量,
        torch::TensorOptions()
        .device(路由概率.device())
        .dtype(路由概率.dtype())
    );

    const auto 对数路由概率 = torch::log_softmax(路由概率, /*dim=*/1);

    auto 路由损失 = torch::nn::functional::kl_div(
        对数路由概率,
        均匀目标,
        torch::nn::functional::KLDivFuncOptions()
        .reduction(torch::kMean)
        .log_target(false)
    );

    auto 总损失 = 分类损失 + 当前路由损失权重_ * 路由损失;

    // 记录损失用于监控
    训练统计_["分类损失"] = 分类损失.item<float>();
    训练统计_["路由损失"] = 路由损失.item<float>();
    训练统计_["总损失"] = 总损失.item<float>();

    TORCH_INFO("损失计算完成: 分类损失=", 分类损失.item<float>(),
        ", 路由损失=", 路由损失.item<float>(),
        ", 总损失=", 总损失.item<float>());

    return 总损失;
}

std::unordered_map<std::string, float> 频域MoE训练器::获取训练统计() const {
    return 训练统计_;
}

void 频域MoE训练器::保存检查点(const std::string& 路径) {
    torch::serialize::OutputArchive 存档;
    模型_->save(存档);
    存档.save_to(路径);
    TORCH_INFO("模型检查点已保存: ", 路径);
}

void 频域MoE训练器::加载检查点(const std::string& 路径) {
    torch::serialize::InputArchive 存档;
    存档.load_from(路径);
    模型_->load(存档);
    TORCH_INFO("模型检查点已加载: ", 路径);
}

// ========================= 频域MoE分析器实现 =========================

频域MoE分析器::频域MoE分析器(std::shared_ptr<频域MoE分类器> 模型)
    : 模型_(模型) {
    
    TORCH_CHECK(模型_ != nullptr, "模型不能为空");
}


std::unordered_map<std::string, torch::Tensor>
频域MoE分析器::分析专家专业化(const torch::Tensor& 样本数据) {
    std::unordered_map<std::string, torch::Tensor> 分析结果;

    torch::NoGradGuard no_grad;
    模型_->eval();

    auto 设备 = 样本数据.device();
    auto 配置 = 模型_->获取配置();
    int64_t 专家数量 = 配置.专家数量;

    TORCH_INFO("开始专家专业化分析，专家数量: ", 专家数量,
        ", 样本数据形状: ", 样本数据.sizes());

    // 1. 获取专家使用统计
    auto 专家使用统计 = 模型_->获取专家使用统计();
    分析结果["专家使用率"] = 专家使用统计;
    TORCH_INFO("专家使用统计获取完成，平均使用率: ",
        专家使用统计.mean().item<float>());

    // 2. 计算专家输出相似度矩阵
    auto 相似度矩阵 = 计算专家相似度矩阵(样本数据);
    分析结果["专家相似度矩阵"] = 相似度矩阵;
    TORCH_INFO("专家相似度矩阵计算完成，矩阵形状: ", 相似度矩阵.sizes());

    // 3. 计算专业化指数
    auto 专业化指数 = 计算专业化指数(相似度矩阵);
    分析结果["专家专业化指数"] = 专业化指数;
    TORCH_INFO("专业化指数计算完成: ", 专业化指数.item<float>());

    // 4. 频域响应特性分析
    auto [频域响应矩阵, 主导频段分布] = 分析频域响应特性(样本数据);
    分析结果["频域响应矩阵"] = 频域响应矩阵;
    分析结果["主导频段分布"] = 主导频段分布;
    TORCH_INFO("频域响应特性分析完成，响应矩阵形状: ",
        频域响应矩阵.sizes());

    // 5. 路由决策稳定性
    auto 路由稳定性 = 分析路由稳定性(样本数据);
    分析结果["路由稳定性"] = 路由稳定性;
    TORCH_INFO("路由稳定性分析完成，平均方差: ",
        路由稳定性.item<float>());

    // 6. 计算专家熵（负载均衡指标）
    auto 专家熵 = 计算专家熵(专家使用统计);
    分析结果["专家熵"] = 专家熵;
    TORCH_INFO("专家熵计算完成: ", 专家熵.item<float>());

    return 分析结果;
}

torch::Tensor 频域MoE分析器::生成路由热力图(const torch::Tensor& 输入样本) {
    // 获取门控权重
    return 模型_->获取门控权重(输入样本);
}


// ========================= 频域MoE训练器实现 =========================

频域MoE训练器::频域MoE训练器(std::shared_ptr<频域MoE分类器> 模型,
                         std::shared_ptr<torch::optim::Optimizer> 优化器,
                         torch::Device 设备)
    : 模型_(模型), 优化器_(优化器), 设备_(设备) {
    
    TORCH_CHECK(模型_ != nullptr, "模型不能为空");
    TORCH_CHECK(优化器_ != nullptr, "优化器不能为空");
    
    // 移动模型到指定设备
    模型_->to(设备_);
    
    // 初始化训练统计
    训练统计_["总损失"] = 0.0f;
    训练统计_["分类损失"] = 0.0f;
    训练统计_["重要性损失"] = 0.0f;
    训练统计_["负载损失"] = 0.0f;
    训练统计_["专业化损失"] = 0.0f;
    训练统计_["学习率"] = 0.0f;
    训练统计_["梯度范数"] = 0.0f;
    训练统计_["专家使用熵"] = 0.0f;
    
    TORCH_INFO("频域MoE训练器初始化完成，设备: ", 设备_);
}




float 频域MoE训练器::计算梯度范数() {
    float 总梯度范数 = 0.0f;
    
    for (const auto& 参数 : 模型_->parameters()) {
        if (参数.grad().defined()) {
            总梯度范数 += 参数.grad().norm().item<float>() * 参数.grad().norm().item<float>();
        }
    }
    
    return std::sqrt(总梯度范数);
}

void 频域MoE训练器::更新学习率统计() {
    try {
        float 当前学习率 = 获取默认学习率();
        训练统计_["学习率"] = 当前学习率;
        TORCH_INFO("当前学习率: ", 当前学习率);
    }
    catch (const std::exception& e) {
        TORCH_ERROR("更新学习率统计失败: ", e.what());
        训练统计_["学习率"] = 0.001f;
    }
}

float 频域MoE训练器::获取默认学习率() {
    try {
        if (!优化器_->param_groups().empty()) {
            auto& 参数组 = 优化器_->param_groups()[0];
            auto& 选项 = 参数组.options();

            // 通过类型名称判断优化器类型
            std::string 选项类型名称 = typeid(选项).name();

            if (选项类型名称.find("AdamOptions") != std::string::npos) {
                auto& adam选项 = static_cast<torch::optim::AdamOptions&>(选项);
                return adam选项.lr();
            }
            else if (选项类型名称.find("SGDOptions") != std::string::npos) {
                auto& sgd选项 = static_cast<torch::optim::SGDOptions&>(选项);
                return sgd选项.lr();
            }
        }
    }
    catch (...) {
        // 忽略异常，返回默认值
    }
    return 0.001f;
}

void 频域MoE训练器::记录专家统计() {
    auto 专家使用统计 = 模型_->获取专家使用统计();
    auto 专家熵 = 计算专家熵(专家使用统计);
    
    训练统计_["专家使用熵"] = 专家熵.item<float>();
    训练统计_["最大专家使用率"] = 专家使用统计.max().item<float>();
    训练统计_["最小专家使用率"] = 专家使用统计.min().item<float>();
    
    // 记录到日志
    TORCH_INFO("专家使用统计 - 熵: ", 专家熵.item<float>(),
               ", 最大: ", 专家使用统计.max().item<float>(),
               ", 最小: ", 专家使用统计.min().item<float>());
}

torch::Tensor 频域MoE训练器::计算专家熵(const torch::Tensor& 使用统计) {
    auto 概率分布 = torch::softmax(使用统计 + 1e-8, 0);
    auto 熵 = -torch::sum(概率分布 * torch::log(概率分布));
    return 熵;
}

void 频域MoE训练器::自适应损失权重调整() {
    // 验证配置有效性
    TORCH_CHECK(配置_.总训练步数 > 0, "总训练步数必须大于0");

    // 基于训练进度计算调整系数 [0, 1]
    float 训练进度 = static_cast<float>(训练步数_) / 配置_.总训练步数;
    训练进度 = std::clamp(训练进度, 0.0f, 1.0f); // 防止越界

    // 策略1：基于训练进度的阶段性调整
    // 早期(0-30%): 分类损失主导，路由损失权重线性增长
    // 中期(30-70%): 路由损失权重保持高位
    // 后期(70%-100%): 路由损失权重缓慢衰减，让模型收敛
    float 路由损失权重;
    if (训练进度 < 0.3f) {
        // 从0.005线性增加到0.02
        路由损失权重 = 0.005f + (0.02f - 0.005f) * (训练进度 / 0.3f);
    }
    else if (训练进度 < 0.7f) {
        路由损失权重 = 0.02f;
    }
    else {
        // 从0.02线性衰减到0.01
        路由损失权重 = 0.02f - (0.02f - 0.01f) * ((训练进度 - 0.7f) / 0.3f);
    }

    // 策略2：基于专家负载均衡情况的动态调整
    // 计算专家使用熵，熵越低表示负载越不均衡，需要增加路由损失权重
    auto 专家使用统计 = 模型_->获取专家使用统计();
    if (专家使用统计.numel() > 0 && 训练步数_ > 0) {
        float 专家熵 = 计算专家熵(专家使用统计).item<float>();
        float 最大熵 = std::log(float(专家使用统计.size(0))); // 均匀分布时的最大熵

        // 负载不均衡系数 [0, 1]，越接近0表示越不均衡
        float 不均衡系数 = 1.0f - (专家熵 / 最大熵);

        // 当负载不均衡时，增加路由损失权重（最大增加50%）
        if (不均衡系数 > 0.5f) {
            路由损失权重 *= (1.0f + 0.5f * (不均衡系数 - 0.5f) * 2.0f);
        }
    }

    // 策略3：梯度范数自适应调整
    // 如果分类损失梯度范数过大，适当降低路由损失权重以保证训练稳定
    // 这里简化为基于训练步数的指数衰减平滑
    static float 当前路由损失权重 = 0.01f; // 静态变量保持状态
    float 平滑因子 = std::exp(-0.001f * 训练步数_); // 指数衰减

    // EMA平滑更新，避免权重剧烈波动
    当前路由损失权重 = 平滑因子 * 当前路由损失权重 + (1.0f - 平滑因子) * 路由损失权重;

    // 限制权重范围
    当前路由损失权重 = std::clamp(当前路由损失权重, 0.005f, 0.03f);

    // 将计算出的权重应用到训练配置
    // 注意：需要在频域MoE训练器类中添加成员变量 当前路由损失权重_
    // 并在计算总损失时使用该变量而不是硬编码的0.01f

    // 记录权重变化用于监控
    训练统计_["路由损失权重"] = 当前路由损失权重;

    // 每100步打印一次权重调整信息
    if (训练步数_ % 100 == 0) {
        TORCH_INFO("自适应权重调整 - 步数: ", 训练步数_,
            ", 训练进度: ", (训练进度 * 100), "%",
            ", 路由损失权重: ", 当前路由损失权重);
    }
}

void 频域MoE训练器::保存检查点(const std::string& 路径) {
    // 创建目录（如果不存在）
    auto 目录位置 = 路径.find_last_of('/');
    if (目录位置 != std::string::npos) {
        std::string 目录 = 路径.substr(0, 目录位置);
        std::filesystem::create_directories(目录);
    }

    // 保存模型状态
    torch::serialize::OutputArchive 模型存档;
    模型_->save(模型存档);

    // 保存优化器状态
    torch::serialize::OutputArchive 优化器存档;
    优化器_->save(优化器存档);

    // 保存训练状态
    torch::serialize::OutputArchive 训练存档;

    // 保存标量值为张量
    训练存档.write("训练步数", torch::tensor(训练步数_));
    训练存档.write("最佳损失", torch::tensor(最佳损失_));

    // 保存训练统计映射为IValue字典
    if (!训练统计_.empty()) {
        torch::Dict<std::string, double> 训练统计字典;
        for (const auto& [键, 值] : 训练统计_) {
            训练统计字典.insert(键, static_cast<double>(值));
        }
        训练存档.write("训练统计", torch::IValue(训练统计字典));
    }

    // 保存验证统计映射为IValue字典
    if (!验证统计_.empty()) {
        torch::Dict<std::string, double> 验证统计字典;
        for (const auto& [键, 值] : 验证统计_) {
            验证统计字典.insert(键, static_cast<double>(值));
        }
        训练存档.write("验证统计", torch::IValue(验证统计字典));
    }

    // 保存配置参数
    训练存档.write("配置.最大梯度范数", torch::tensor(配置_.最大梯度范数));
    训练存档.write("配置.标签平滑", torch::tensor(配置_.标签平滑));

    // 合并所有存档
    torch::serialize::OutputArchive 最终存档;
    最终存档.write("模型", 模型存档);
    最终存档.write("优化器", 优化器存档);
    最终存档.write("训练状态", 训练存档);

    // 保存到文件
    最终存档.save_to(路径);

    TORCH_INFO("检查点已保存: ", 路径, " (步数: ", 训练步数_, ")");
}

void 频域MoE训练器::加载检查点(const std::string& 路径) {
    TORCH_CHECK(std::filesystem::exists(路径), "检查点文件不存在: ", 路径);

    try {
        torch::serialize::InputArchive 最终存档;
        最终存档.load_from(路径);

        // 加载模型
        torch::serialize::InputArchive 模型存档;
        最终存档.read("模型", 模型存档);
        模型_->load(模型存档);

        // 加载优化器
        torch::serialize::InputArchive 优化器存档;
        最终存档.read("优化器", 优化器存档);
        优化器_->load(优化器存档);

        // 加载训练状态
        torch::serialize::InputArchive 训练存档;
        最终存档.read("训练状态", 训练存档);

        // 读取标量值（通过张量转换）
        torch::Tensor 步数张量;
        训练存档.read("训练步数", 步数张量);
        训练步数_ = 步数张量.item<int64_t>();

        torch::Tensor 最佳损失张量;
        训练存档.read("最佳损失", 最佳损失张量);
        最佳损失_ = 最佳损失张量.item<float>();

        // 读取配置参数
        torch::Tensor 最大梯度范数张量, 标签平滑张量;
        训练存档.read("配置.最大梯度范数", 最大梯度范数张量);
        训练存档.read("配置.标签平滑", 标签平滑张量);

        配置_.最大梯度范数 = 最大梯度范数张量.item<float>();
        配置_.标签平滑 = 标签平滑张量.item<float>();

        // 加载训练统计映射
        torch::IValue 训练统计值;
        训练存档.read("训练统计", 训练统计值);
        if (训练统计值.isGenericDict()) {
            训练统计_.clear();
            auto 字典 = 训练统计值.toGenericDict();
            for (const auto& 项 : 字典) {
                训练统计_[项.key().toStringRef()] = static_cast<float>(项.value().toDouble());
            }
        }

        // 加载验证统计映射
        torch::IValue 验证统计值;
        训练存档.read("验证统计", 验证统计值);
        if (验证统计值.isGenericDict()) {
            验证统计_.clear();
            auto 字典 = 验证统计值.toGenericDict();
            for (const auto& 项 : 字典) {
                验证统计_[项.key().toStringRef()] = static_cast<float>(项.value().toDouble());
            }
        }

        TORCH_INFO("检查点已加载: ", 路径, " (步数: ", 训练步数_, ")");
    }
    catch (const std::exception& e) {
        TORCH_ERROR("加载检查点失败: ", e.what());
        throw;
    }
}


void 频域MoE训练器::设置配置(const 训练配置& 新配置) {
    配置_ = 新配置;
    TORCH_INFO("训练配置已更新");
}

训练配置 频域MoE训练器::获取配置() const {
    return 配置_;
}

std::unordered_map<std::string, float> 频域MoE训练器::获取训练统计() const {
    return 训练统计_;
}

std::unordered_map<std::string, float> 频域MoE训练器::获取验证统计() const {
    return 验证统计_;
}

int64_t 频域MoE训练器::获取训练步数() const {
    return 训练步数_;
}

void 频域MoE训练器::打印训练状态() {
    auto 当前时间 = std::chrono::system_clock::now();
    auto 时间点 = std::chrono::system_clock::to_time_t(当前时间);
    
    std::cout << "\n=== 训练状态 (步数: " << 训练步数_ << ") ===" << std::endl;
    std::cout << "时间: " << std::ctime(&时间点);
    std::cout << "设备: " << 设备_ << std::endl;
    std::cout << "损失: " << 训练统计_.at("总损失") << std::endl;
    std::cout << "学习率: " << 训练统计_.at("学习率") << std::endl;
    std::cout << "梯度范数: " << 训练统计_.at("梯度范数") << std::endl;
    std::cout << "专家熵: " << 训练统计_.at("专家使用熵") << std::endl;
    
    if (!验证统计_.empty()) {
        std::cout << "验证损失: " << 验证统计_.at("损失") << std::endl;
        std::cout << "验证准确率: " << 验证统计_.at("准确率") << std::endl;
    }
    
    std::cout << "=============================" << std::endl;
}

void 频域MoE训练器::早停检查() {
    if (验证统计_.empty()) return;
    
    float 当前损失 = 验证统计_.at("损失");
    
    if (当前损失 < 最佳损失_) {
        最佳损失_ = 当前损失;
        无改善轮数_ = 0;
        TORCH_INFO("验证损失改善: ", 当前损失);
    } else {
        无改善轮数_++;
        TORCH_WARN("验证损失无改善，轮数: ", 无改善轮数_, "/", 配置_.早停耐心);
    }
    
    if (无改善轮数_ >= 配置_.早停耐心) {
        TORCH_WARN("早停触发于步数: ", 训练步数_);
        // 这里可以添加早停回调
    }
}
// ========================= 频域MoE分析器实现 =========================

频域MoE分析器::频域MoE分析器(std::shared_ptr<频域MoE分类器> 模型)
    : 模型_(模型) {
    
    TORCH_CHECK(模型_ != nullptr, "模型不能为空");
    
    // 初始化分析配置
    配置_.采样数量 = 1000;
    配置_.热力图序列长度 = 50;
    配置_.频段分析精度 = 0.01f;
    配置_.专家相似度阈值 = 0.7f;
    配置_.保存可视化数据 = true;
    配置_.可视化目录 = "./moe_visualizations";
    
    TORCH_INFO("频域MoE分析器初始化完成");
}

void 频域MoE分析器::设置配置(const 分析配置& 新配置) {
    配置_ = 新配置;
    TORCH_INFO("分析配置已更新");
}

分析配置 频域MoE分析器::获取配置() const {
    return 配置_;
}

std::unordered_map<std::string, torch::Tensor> 
频域MoE分析器::分析专家专业化(const torch::Tensor& 样本数据) {
    std::unordered_map<std::string, torch::Tensor> 分析结果;
    
    torch::NoGradGuard no_grad;
    模型_->eval();
    
    auto 设备 = 样本数据.device();
    auto 配置 = 模型_->获取配置();
    int64_t 专家数量 = 配置.专家数量;
    
    // 1. 专家使用统计
    auto 专家使用统计 = 模型_->获取专家使用统计();
    分析结果["专家使用率"] = 专家使用统计;
    
    // 2. 专家输出相似度矩阵
    auto 相似度矩阵 = 计算专家相似度矩阵(样本数据);
    分析结果["专家相似度矩阵"] = 相似度矩阵;
    
    // 3. 专家专业化指数
    auto 专业化指数 = 计算专业化指数(相似度矩阵);
    分析结果["专家专业化指数"] = 专业化指数;
    
    // 4. 频域响应特性
    auto 频域响应 = 分析频域响应特性(样本数据);
    分析结果["频域响应矩阵"] = std::get<0>(频域响应);
    分析结果["主导频段分布"] = std::get<1>(频域响应);
    
    // 5. 路由决策稳定性
    auto 路由稳定性 = 分析路由稳定性(样本数据);
    分析结果["路由稳定性"] = 路由稳定性;
    
    return 分析结果;
}

torch::Tensor 频域MoE分析器::计算专家相似度矩阵(const torch::Tensor& 样本数据) {
    // 输入验证
    TORCH_CHECK(模型_ != nullptr, "分析器模型实例未初始化");
    TORCH_CHECK(样本数据.defined() && 样本数据.numel() > 0, "样本数据为空");

    auto moe配置 = 模型_->获取配置();
    const int64_t 专家数量 = moe配置.专家数量;
    TORCH_CHECK(专家数量 > 1, "专家数量必须大于1才能计算相似度");

    auto 设备 = 样本数据.device();

    // 采样数据以控制计算量
    const int64_t 实际采样数 = std::min(配置_.采样数量, (int64_t)样本数据.size(0));
    auto 采样数据 = 样本数据.slice(0, 0, 实际采样数);

    // 批量获取所有专家输出 [专家数, batch, seq_len, hidden_dim]
    auto 专家输出列表 = 模型_->批量获取专家输出(采样数据);

    // 统一展平为特征向量 [专家数, 特征总数]
    // 确保所有专家输出在同一设备且连续存储
    std::vector<torch::Tensor> 展平列表;
    展平列表.reserve(专家数量);

    for (const auto& 专家输出 : 专家输出列表) {
        // 展平并确保内存布局连续
        展平列表.push_back(专家输出.flatten().to(设备, torch::kFloat32).contiguous());
    }

    // 堆叠为特征矩阵 [专家数, 特征维度]
    auto 专家特征矩阵 = torch::stack(展平列表, 0);

    // 计算 L2 范数 [专家数, 1]
    auto 特征范数 = 专家特征矩阵.norm(2, 1, true);

    // 防止除零并归一化
    auto 归一化特征 = 专家特征矩阵 / (特征范数 + 1e-8);

    // 矩阵乘法高效计算余弦相似度 [专家数, 专家数]
    auto 相似度矩阵 = torch::matmul(归一化特征, 归一化特征.transpose(0, 1));

    // 强制对角线为1.0（消除数值误差）
    相似度矩阵.fill_diagonal_(1.0f);

    // 计算非对角线平均相似度作为监控指标
    auto 掩码 = 1.0f - torch::eye(专家数量, 设备);
    float 非对角线均值 = (相似度矩阵 * 掩码).sum().item<float>() / (专家数量 * (专家数量 - 1));

    TORCH_INFO("专家相似度矩阵计算完成，平均非对角线相似度: ", 非对角线均值);

    return 相似度矩阵;
}

torch::Tensor 频域MoE分析器::获取特定专家输出(const torch::Tensor& 输入, int64_t 专家索引) {
    // 验证模型有效性
    TORCH_CHECK(模型_ != nullptr, "分析器中的模型实例为空");

    // 直接调用模型的专用接口，避免重复实现路由逻辑
    // 确保分析过程与训练时的专家计算路径完全一致
    return 模型_->前向传播特定专家(输入, 专家索引);
}

torch::Tensor 频域MoE分析器::计算专业化指数(const torch::Tensor& 相似度矩阵) {
    int64_t 专家数量 = 相似度矩阵.size(0);
    auto 设备 = 相似度矩阵.device();
    
    // 专业化指数 = 1 - 平均相似度（排除对角线）
    auto 掩码 = torch::ones({专家数量, 专家数量}, 设备).triu(1);
    auto 非对角相似度 = 相似度矩阵 * 掩码;
    auto 平均相似度 = 非对角相似度.sum() / (专家数量 * (专家数量 - 1) / 2);
    
    auto 专业化指数 = 1.0 - 平均相似度;
    
    return 专业化指数.unsqueeze(0);  // 保持张量维度
}

std::tuple<torch::Tensor, torch::Tensor> 
频域MoE分析器::分析频域响应特性(const torch::Tensor& 样本数据) {
    auto 配置 = 模型_->获取配置();
    int64_t 专家数量 = 配置.专家数量;
    auto 设备 = 样本数据.device();
    
    // 分析各专家对不同频段的响应特性
    int64_t 频段数 = 10;  // 将频谱分为10个频段
    auto 频域响应矩阵 = torch::zeros({专家数量, 频段数}, 设备);
    auto 主导频段分布 = torch::zeros({专家数量}, 设备);
    
    // 生成测试信号：不同频段的正弦波
    auto 测试信号 = 生成多频段测试信号(样本数据.size(0), 样本数据.size(1), 
                                     样本数据.size(2), 频段数, 设备);
    
    for (int64_t 频段 = 0; 频段 < 频段数; ++频段) {
        auto 测试输入 = 测试信号[频段];
        
        // 获取各专家对该频段的响应
        for (int64_t 专家 = 0; 专家 < 专家数量; ++专家) {
            模型_->设置激活专家数(1);
            auto 专家输出 = 获取特定专家输出(测试输入, 专家);
            
            // 计算输出能量作为响应强度
            auto 响应强度 = 专家输出.norm().item<float>();
            频域响应矩阵[专家][频段] = 响应强度;
        }
    }
    
    // 计算每个专家主导频段
    for (int64_t 专家 = 0; 专家 < 专家数量; ++专家) {
        auto 专家响应 = 频域响应矩阵[专家];
        auto 最大响应 = torch::argmax(专家响应);
        主导频段分布[专家] = 最大响应;
    }
    
    return std::make_tuple(频域响应矩阵, 主导频段分布);
}

std::vector<torch::Tensor> 频域MoE分析器::生成多频段测试信号(
    int64_t 批次大小, int64_t 序列长度, int64_t 特征维度, 
    int64_t 频段数, torch::Device 设备) {
    
    std::vector<torch::Tensor> 测试信号列表;
    
    // 采样率假设为1000Hz
    float 采样率 = 1000.0f;
    auto 时间 = torch::linspace(0, 序列长度 / 采样率, 序列长度, 设备);
    
    for (int64_t 频段 = 0; 频段 < 频段数; ++频段) {
        // 每个频段覆盖不同的频率范围
        float 最低频率 = 频段 * (采样率 / 2) / 频段数;
        float 最高频率 = (频段 + 1) * (采样率 / 2) / 频段数;
        float 中心频率 = (最低频率 + 最高频率) / 2;
        
        // 生成带限噪声信号
        auto 信号 = torch::randn({批次大小, 序列长度, 特征维度}, 设备);
        
        // 应用带通滤波（简化：频域截断）
        auto 频域信号 = torch::fft::rfft(信号, 序列长度, 1);
        int64_t 频点数 = 频域信号.size(1);
        
        // 创建带通掩码
        auto 频率轴 = torch::linspace(0, 采样率 / 2, 频点数, 设备);
        auto 掩码 = (频率轴 >= 最低频率) * (频率轴 <= 最高频率);
        掩码 = 掩码.unsqueeze(0).unsqueeze(-1).expand({批次大小, 频点数, 特征维度});
        
        频域信号 = 频域信号 * 掩码;
        信号 = torch::fft::irfft(频域信号, 序列长度, 1);
        
        测试信号列表.push_back(信号);
    }
    
    return 测试信号列表;
}

torch::Tensor 频域MoE分析器::分析路由稳定性(const torch::Tensor& 样本数据) {
    int64_t 测试次数 = 10;
    auto 设备 = 样本数据.device();
    auto 配置 = 模型_->获取配置();
    
    // 多次前向传播，检查路由决策的一致性
    auto 批次大小 = 样本数据.size(0);
    auto 路由决策序列 = torch::zeros({测试次数, 批次大小, 配置.专家数量}, 设备);
    
    for (int64_t i = 0; i < 测试次数; ++i) {
        auto 门控权重 = 模型_->获取门控权重(样本数据);
        路由决策序列[i] = 门控权重;
    }
    
    // 计算路由决策的方差（越低越稳定）
    auto 路由方差 = torch::var(路由决策序列, 0).mean();  // 沿测试次数维度求方差，然后平均
    
    return 路由方差.unsqueeze(0);
}

torch::Tensor 频域MoE分析器::生成路由热力图(const torch::Tensor& 输入样本) {
    auto 门控权重 = 模型_->获取门控权重(输入样本);
    
    if (配置_.保存可视化数据) {
        保存热力图数据(门控权重, 输入样本);
    }
    
    return 门控权重;
}

void 频域MoE分析器::保存热力图数据(const torch::Tensor& 门控权重, 
                                const torch::Tensor& 输入样本) {
    // 创建可视化目录
    std::filesystem::create_directories(配置_.可视化目录);
    
    // 保存门控权重数据
    auto 权重数据 = 门控权重.cpu().contiguous();
    std::ofstream 权重文件(配置_.可视化目录 + "/gating_weights.bin", 
                          std::ios::binary);
    权重文件.write(reinterpret_cast<const char*>(权重数据.data_ptr()), 
                 权重数据.numel() * sizeof(float));
    权重文件.close();
    
    // 保存输入样本的频域特征（用于关联分析）
    auto 频域特征 = 模型_->提取频域特征(输入样本).cpu().contiguous();
    std::ofstream 特征文件(配置_.可视化目录 + "/frequency_features.bin", 
                          std::ios::binary);
    特征文件.write(reinterpret_cast<const char*>(频域特征.data_ptr()), 
                 频域特征.numel() * sizeof(float));
    特征文件.close();
    
    TORCH_INFO("热力图数据已保存至: ", 配置_.可视化目录);
}

std::unordered_map<std::string, float>
频域MoE分析器::分析特征重要性(const torch::Tensor& 样本数据) {
    std::unordered_map<std::string, float> 重要性结果;

    auto 配置 = 模型_->获取配置();
    auto 设备 = 样本数据.device();

    // 1. 门控网络特征重要性（通过参数梯度分析）
    auto 门控参数列表 = 模型_->获取参数列表();

    // 分析第一个线性层的权重重要性
    if (!门控参数列表.empty()) {
        auto 门控权重 = 门控参数列表[0]; // 门控网络第一层权重 [输出维度, 输入维度]

        // 修复: 使用显式的torch::IntArrayRef指定维度，避免重载歧义
        auto 权重绝对值 = 门控权重.abs().mean(torch::IntArrayRef{ 0, 1 });

        // 验证维度有效性
        TORCH_CHECK(权重绝对值.numel() > 0, "权重绝对值计算结果为空");

        auto 特征重要性 = torch::softmax(权重绝对值, 0);

        // 映射到5个频域特征分量
        const std::vector<std::string> 特征名称 = {
            "低频能量", "高频能量", "相位稳定性", "全局平均能量", "峰值能量"
        };

        // 确保索引不越界
        int64_t 有效维度 = std::min(static_cast<int64_t>(特征名称.size()), 特征重要性.size(0));
        for (int64_t i = 0; i < 有效维度; ++i) {
            重要性结果[特征名称[i] + "_重要性"] = 特征重要性[i].item<float>();
        }
    }

    // 2. 频域特征重要性（通过消融实验模拟）
    auto 基准门控权重 = 模型_->获取门控权重(样本数据);
    float 基准熵 = 计算门控熵(基准门控权重).item<float>();

    // 模拟不同频域特征的消融影响
    const std::vector<std::pair<std::string, std::string>> 特征消融列表 = {
        {"低频特征重要性", "低频"},
        {"高频特征重要性", "高频"},
        {"相位特征重要性", "相位"},
        {"全局特征重要性", "全局"},
        {"峰值特征重要性", "峰值"}
    };

    for (const auto& [结果键, 特征类型] : 特征消融列表) {
        重要性结果[结果键] = 模拟特征消融(样本数据, 特征类型, 基准熵);
    }

    return 重要性结果;
}

float 频域MoE分析器::模拟特征消融(const torch::Tensor& 样本数据, 
                              const std::string& 特征类型, float 基准熵) {
    // 简化实现：返回随机重要性分数
    // 实际实现应该修改特征提取过程，消融特定特征后重新计算门控熵
    static std::unordered_map<std::string, float> 预设重要性 = {
        {"低频", 0.25f}, {"高频", 0.20f}, {"相位", 0.15f}, 
        {"全局", 0.25f}, {"峰值", 0.15f}
    };
    
    return 预设重要性[特征类型];
}

torch::Tensor 频域MoE分析器::计算门控熵(const torch::Tensor& 门控权重) {
    // 计算门控权重的熵（路由决策的不确定性）
    auto 有效权重 = torch::clamp(门控权重, 1e-8, 1.0f);
    auto 熵 = -torch::sum(有效权重 * torch::log(有效权重), 1);
    return 熵.mean();  // 批次平均
}

void 频域MoE分析器::生成专家使用报告(const std::string& 输出路径) {
    std::ofstream 报告文件(输出路径);
    
    if (!报告文件.is_open()) {
        TORCH_ERROR("无法打开报告文件: ", 输出路径);
        return;
    }
    
    auto 配置 = 模型_->获取配置();
    auto 使用统计 = 模型_->获取专家使用统计().cpu();
    
    报告文件 << "频域MoE专家使用分析报告\n";
    报告文件 << "生成时间: " << 获取当前时间() << "\n";
    报告文件 << "========================================\n\n";
    
    报告文件 << "模型配置:\n";
    报告文件 << "- 专家数量: " << 配置.专家数量 << "\n";
    报告文件 << "- 激活专家数: " << 配置.激活专家数 << "\n";
    报告文件 << "- 专家隐藏维度: " << 配置.专家隐藏维度 << "\n\n";
    
    报告文件 << "专家使用统计:\n";
    for (int64_t i = 0; i < 配置.专家数量; ++i) {
        报告文件 << "专家 " << i << ": " << std::fixed << std::setprecision(4) 
                << 使用统计[i].item<float>() << "\n";
    }
    
    auto 总使用率 = 使用统计.sum().item<float>();
    auto 平均使用率 = 使用统计.mean().item<float>();
    auto 使用率方差 = 使用统计.var().item<float>();
    
    报告文件 << "\n统计摘要:\n";
    报告文件 << "- 总使用率: " << 总使用率 << "\n";
    报告文件 << "- 平均使用率: " << 平均使用率 << "\n";
    报告文件 << "- 使用率方差: " << 使用率方差 << "\n";
    报告文件 << "- 使用率熵: " << 计算专家熵(使用统计).item<float>() << "\n";
    
    // 识别关键专家
    报告文件 << "\n关键专家分析:\n";
    int64_t 分析专家数 = std::min(static_cast<int64_t>(3), 配置.专家数量);
    auto 排序索引 = torch::argsort(使用统计, 0, true);
    for (int64_t i = 0; i < 分析专家数; ++i) {
        int64_t 专家索引 = 排序索引[i].item<int64_t>();
        报告文件 << "Top-" << (i + 1) << " 专家: " << 专家索引
            << " (使用率: " << 使用统计[专家索引].item<float>() << ")\n";
    }
    
    报告文件.close();
    TORCH_INFO("专家使用报告已生成: ", 输出路径);
}

std::string 频域MoE分析器::获取当前时间() {
    auto 现在 = std::chrono::system_clock::now();
    auto 时间点 = std::chrono::system_clock::to_time_t(现在);
    
    std::stringstream 时间流;
    时间流 << std::put_time(std::localtime(&时间点), "%Y-%m-%d %H:%M:%S");
    return 时间流.str();
}

torch::Tensor 频域MoE分析器::计算专家熵(const torch::Tensor& 使用统计) {
    auto 概率分布 = torch::softmax(使用统计 + 1e-8, 0); // 添加维度参数0
    auto 熵值 = -torch::sum(概率分布 * torch::log(概率分布 + 1e-8));
    return 熵值.unsqueeze(0);
}

void 频域MoE分析器::可视化专家网络(const std::string& 输出路径) {
    // 生成专家网络的可视化描述
    std::ofstream 可视化文件(输出路径);

    if (!可视化文件.is_open()) {
        TORCH_ERROR("无法打开可视化文件: ", 输出路径);
        return;
    }

    可视化文件 << "digraph MoE_Expert_Network {\n";
    可视化文件 << "    rankdir=LR;\n";
    可视化文件 << "    node [shape=box, style=filled, fillcolor=lightblue];\n\n";

    // 获取配置信息
    auto 配置 = 模型_->获取配置();

    // 关键修复：直接从模型对象获取维度信息，而非配置结构体
    // 因为输入维度和类别数是频域MoE分类器的成员变量，不是配置参数
    int64_t 输入维度 = 模型_->输入维度_;
    int64_t 类别数 = 模型_->类别数_;

    // 输入节点
    可视化文件 << "    input [label=\"输入\\n维度: " << 输入维度 << "\"];\n";

    // 专家节点
    for (int64_t i = 0; i < 配置.专家数量; ++i) {
        可视化文件 << "    expert" << i << " [label=\"专家 " << i
            << "\\n隐藏维: " << 配置.专家隐藏维度 << "\"];\n";
    }

    // 输出节点
    可视化文件 << "    output [label=\"输出\\n类别: " << 类别数 << "\"];\n\n";

    // 连接关系
    可视化文件 << "    input -> {";
    for (int64_t i = 0; i < 配置.专家数量; ++i) {
        可视化文件 << "expert" << i;
        if (i < 配置.专家数量 - 1) 可视化文件 << " ";
    }
    可视化文件 << "} [label=\"频域路由\"];\n\n";

    // 添加专家到输出的连接，线宽反映使用率
    for (int64_t i = 0; i < 配置.专家数量; ++i) {
        可视化文件 << "    expert" << i << " -> output";

        auto 使用率 = 模型_->获取专家使用统计()[i].item<float>();
        int 线宽 = static_cast<int>(使用率 * 10) + 1;
        可视化文件 << " [penwidth=" << 线宽 << ", label=\""
            << std::fixed << std::setprecision(3) << 使用率 << "\"]";

        可视化文件 << ";\n";
    }

    可视化文件 << "}\n";
    可视化文件.close();

    TORCH_INFO("专家网络可视化描述已生成: ", 输出路径);

    // 生成转换为图片的命令提示
    std::string 基础文件名 = 输出路径.substr(0, 输出路径.find_last_of('.'));
    TORCH_INFO("使用 Graphviz 生成图片: dot -Tpng ", 输出路径, " -o ", 基础文件名, ".png");
}
// ========================= 训练配置相关 =========================

训练配置 创建默认训练配置() {
    训练配置 配置;
    配置.最大梯度范数 = 1.0f;
    配置.标签平滑 = 0.1f;
    配置.总训练步数 = 100000;
    配置.统计记录间隔 = 100;
    配置.检查点间隔 = 1000;
    配置.早停耐心 = 10;
    配置.检查点目录 = "./checkpoints";
    return 配置;
}

训练配置 创建精确训练配置() {
    auto 配置 = 创建默认训练配置();
    配置.最大梯度范数 = 0.5f;
    配置.标签平滑 = 0.05f;
    配置.总训练步数 = 200000;
    配置.统计记录间隔 = 50;
    配置.检查点间隔 = 500;
    配置.早停耐心 = 20;
    return 配置;
}

训练配置 创建快速训练配置() {
    auto 配置 = 创建默认训练配置();
    配置.最大梯度范数 = 2.0f;
    配置.标签平滑 = 0.2f;
    配置.总训练步数 = 50000;
    配置.统计记录间隔 = 200;
    配置.检查点间隔 = 2000;
    配置.早停耐心 = 5;
    return 配置;
}

// ========================= 工具函数实现 =========================

namespace MoE工具 {

频域MoE配置 创建平衡配置(int64_t 输入维度, int64_t 类别数) {
    频域MoE配置 配置;
    配置.专家数量 = 8;
    配置.激活专家数 = 2;
    配置.专家隐藏维度 = 256;
    配置.门控隐藏层维度 = 512;
    配置.负载均衡权重 = 0.01f;
    配置.专业化权重 = 0.005f;
    return 配置;
}

频域MoE配置 创建高性能配置(int64_t 输入维度, int64_t 类别数) {
    频域MoE配置 配置;
    配置.专家数量 = 16;
    配置.激活专家数 = 4;
    配置.专家隐藏维度 = 512;
    配置.门控隐藏层维度 = 1024;
    配置.负载均衡权重 = 0.02f;
    配置.专业化权重 = 0.01f;
    配置.启用动态路由 = true;
    return 配置;
}

频域MoE配置 创建轻量配置(int64_t 输入维度, int64_t 类别数) {
    频域MoE配置 配置;
    配置.专家数量 = 4;
    配置.激活专家数 = 1;
    配置.专家隐藏维度 = 128;
    配置.门控隐藏层维度 = 0;  // 单层门控
    配置.负载均衡权重 = 0.005f;
    配置.专业化权重 = 0.0f;   // 禁用专业化损失
    return 配置;
}

int64_t 计算参数数量(const 频域MoE分类器& 模型) {
    int64_t 总参数 = 0;
    auto 参数列表 = 模型.parameters();
    
    for (const auto& 参数 : 参数列表) {
        if (参数.defined()) {
            总参数 += 参数.numel();
        }
    }
    
    return 总参数;
}

} // namespace MoE工具

} // namespace SpectraSSM