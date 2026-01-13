
// SpectraSSM - 频域状态空间模型库
// 版权所有 © 2026 SpectraSSM 项目所有者
// 
// 本软件受 SpectraSSM 自定义社区许可协议 v1.0 约束，使用需满足：
// - 年营业额 ≤ 5,000,000 人民币（滚动12个月）
// - 详见完整协议: LICENSE.txt
// 
// 警告：此许可证非开源许可证，包含商业使用限制和审计条款。
// 使用本软件即表示您同意接受所有条款，包括零责任赔偿限制。
//
#include "训练器.hpp"
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace SpectraSSM {

    训练器::训练器(
        std::shared_ptr<频域MoE模型> 模型,
        const 训练配置& 配置)
        : 模型_(模型), 配置_(配置), 当前轮次_(0), 最佳损失_(std::numeric_limits<float>::max()) {

        // 参数验证
        TORCH_CHECK(模型_ != nullptr, "模型指针不能为空");
        TORCH_CHECK(配置_.批大小 > 0, "批大小必须大于0");
        TORCH_CHECK(配置_.学习率 > 0.0f, "学习率必须大于0");

        // 初始化优化器
        初始化优化器();

        // 初始化损失函数
        损失函数_ = torch::nn::MSELoss();

        TORCH_INFO("训练器初始化完成");
        TORCH_INFO("优化器类型: ", 优化器类型转字符串(配置_.优化器类型));
        TORCH_INFO("学习率: ", 配置_.学习率);
        TORCH_INFO("批大小: ", 配置_.批大小);
    }

    void 训练器::初始化优化器() {
        TORCH_INFO("初始化优化器...");

        // 获取模型参数
        auto 参数列表 = 模型_->获取参数列表();

        if (参数列表.empty()) {
            TORCH_WARN("模型参数列表为空");
            return;
        }

        TORCH_INFO("模型参数数量: ", 参数列表.size());

        // 配置检查
        if (配置_.学习率 <= 0.0f) {
            TORCH_WARN("学习率不合法: ", 配置_.学习率, "，使用默认值 0.001");
            配置_.学习率 = 0.001f;
        }

        // 根据配置类型初始化相应的优化器
        switch (配置_.优化器类型) {
        case 优化器类型::SGD:
            初始化SGD优化器(参数列表);
            break;
        case 优化器类型::Adam:
            初始化Adam优化器(参数列表);
            break;
        case 优化器类型::AdamW:
            初始化AdamW优化器(参数列表);
            break;
        case 优化器类型::RMSprop:
            初始化RMSprop优化器(参数列表);
            break;
        default:
            TORCH_ERROR("不支持的优化器类型: ", static_cast<int>(配置_.优化器类型));
            throw std::invalid_argument("优化器类型不支持");
        }

        // 同时初始化 Vulkan GPU 优化器（如果可用）
        初始化Vulkan优化器();

        TORCH_INFO("优化器初始化完成 - 类型: ", 优化器类型转字符串(配置_.优化器类型));
    }

    void 训练器::初始化SGD优化器(const std::vector<torch::Tensor>& 参数列表) {
        TORCH_INFO("初始化 SGD 优化器...");
        TORCH_INFO("  - 学习率: ", 配置_.学习率);
        TORCH_INFO("  - 动量: ", 配置_.动量);
        TORCH_INFO("  - 权重衰减: ", 配置_.权重衰减);

        try {
            // 创建 PyTorch SGD 优化器
            torch优化器_ = std::make_unique<torch::optim::SGD>(
                参数列表,
                torch::optim::SGDOptions(配置_.学习率)
                .momentum(配置_.动量)
                .weight_decay(配置_.权重衰减)
            );

            // 验证优化器创建成功
            TORCH_CHECK(torch优化器_ != nullptr, "SGD优化器创建失败");
            TORCH_CHECK(!torch优化器_->param_groups().empty(), "优化器参数组为空");

            TORCH_INFO("SGD 优化器创建成功");

        }
        catch (const std::exception& e) {
            TORCH_ERROR("SGD 优化器初始化失败: ", e.what());
            throw;
        }
    }

    void 训练器::初始化Adam优化器(const std::vector<torch::Tensor>& 参数列表) {
        TORCH_INFO("初始化 Adam 优化器...");
        TORCH_INFO("  - 学习率: ", 配置_.学习率);
        TORCH_INFO("  - Beta1: ", 配置_.贝塔1);
        TORCH_INFO("  - Beta2: ", 配置_.贝塔2);
        TORCH_INFO("  - 权重衰减: ", 配置_.权重衰减);

        try {
            // 创建 PyTorch Adam 优化器
            torch优化器_ = std::make_unique<torch::optim::Adam>(
                参数列表,
                torch::optim::AdamOptions(配置_.学习率)
                .betas(std::make_tuple(配置_.贝塔1, 配置_.贝塔2))
                .weight_decay(配置_.权重衰减)
                .epsilon(1e-8)
            );

            TORCH_CHECK(torch优化器_ != nullptr, "Adam优化器创建失败");
            TORCH_CHECK(!torch优化器_->param_groups().empty(), "优化器参数组为空");

            TORCH_INFO("Adam 优化器创建成功");

        }
        catch (const std::exception& e) {
            TORCH_ERROR("Adam 优化器初始化失败: ", e.what());
            throw;
        }
    }

    void 训练器::初始化AdamW优化器(const std::vector<torch::Tensor>& 参数列表) {
        TORCH_INFO("初始化 AdamW 优化器...");
        TORCH_INFO("  - 学习率: ", 配置_.学习率);
        TORCH_INFO("  - Beta1: ", 配置_.贝塔1);
        TORCH_INFO("  - Beta2: ", 配置_.贝塔2);
        TORCH_INFO("  - 权重衰减: ", 配置_.权重衰减);

        try {
            // 创建 PyTorch AdamW 优化器
            torch优化器_ = std::make_unique<torch::optim::AdamW>(
                参数列表,
                torch::optim::AdamWOptions(配置_.学习率)
                .betas(std::make_tuple(配置_.贝塔1, 配置_.贝塔2))
                .weight_decay(配置_.权重衰减)
                .epsilon(1e-8)
            );

            TORCH_CHECK(torch优化器_ != nullptr, "AdamW优化器创建失败");
            TORCH_CHECK(!torch优化器_->param_groups().empty(), "优化器参数组为空");

            TORCH_INFO("AdamW 优化器创建成功");

        }
        catch (const std::exception& e) {
            TORCH_ERROR("AdamW 优化器初始化失败: ", e.what());
            throw;
        }
    }

    void 训练器::初始化RMSprop优化器(const std::vector<torch::Tensor>& 参数列表) {
        TORCH_INFO("初始化 RMSprop 优化器...");
        TORCH_INFO("  - 学习率: ", 配置_.学习率);
        TORCH_INFO("  - 动量: ", 配置_.动量);
        TORCH_INFO("  - 权重衰减: ", 配置_.权重衰减);

        try {
            // 创建 PyTorch RMSprop 优化器
            torch优化器_ = std::make_unique<torch::optim::RMSprop>(
                参数列表,
                torch::optim::RMSpropOptions(配置_.学习率)
                .momentum(配置_.动量)
                .weight_decay(配置_.权重衰减)
                .eps(1e-8)
            );

            TORCH_CHECK(torch优化器_ != nullptr, "RMSprop优化器创建失败");
            TORCH_CHECK(!torch优化器_->param_groups().empty(), "优化器参数组为空");

            TORCH_INFO("RMSprop 优化器创建成功");

        }
        catch (const std::exception& e) {
            TORCH_ERROR("RMSprop 优化器初始化失败: ", e.what());
            throw;
        }
    }

    void 训练器::初始化Vulkan优化器() {
        TORCH_INFO("初始化 Vulkan GPU 优化器...");

        try {
            // 检查是否有可用的 GPU
            if (!torch::cuda::is_available()) {
                TORCH_WARN("CUDA 不可用，Vulkan 优化器将回退到 CPU");
            }

            // 创建 Vulkan Adam 优化器
            auto vulkanAdam = std::make_shared<VulkanAdam优化器>(
                配置_.学习率,
                配置_.贝塔1,
                配置_.贝塔2,
                配置_.权重衰减,
                1e-8f,
                0  // 使用默认 GPU
            );

            vulkan优化器_ = vulkanAdam;

            TORCH_INFO("Vulkan GPU 优化器创建成功");

        }
        catch (const std::exception& e) {
            TORCH_WARN("Vulkan 优化器初始化失败: ", e.what(),
                "，将仅使用 PyTorch CPU 优化器");
            vulkan优化器_ = nullptr;
        }
    }

    // 以下是优化器类型转换函数，用于日志输出
    std::string 训练器::优化器类型转字符串(优化器类型 类型) {
        switch (类型) {
        case 优化器类型::SGD: return "SGD";
        case 优化器类型::Adam: return "Adam";
        case 优化器类型::AdamW: return "AdamW";
        case 优化器类型::RMSprop: return "RMSprop";
        default: return "未知优化器";
        }
    }
    float 训练器::训练步骤(const torch::Tensor& 输入, const torch::Tensor& 目标) {
        // 设置模型为训练模式
        模型_->train();

        // 前向传播
        auto 输出 = 模型_->前向传播(输入);

        // 计算损失
        auto 损失 = 损失函数_(输出, 目标);

        // 反向传播
        优化器_->zero_grad();
        损失.backward();

        // 梯度裁剪（如果启用）
        if (配置_.梯度裁剪 > 0.0f) {
            torch::nn::utils::clip_grad_norm_(模型_->获取参数列表(), 配置_.梯度裁剪);
        }

        // 参数更新
        优化器_->step();

        return 损失.item<float>();
    }

    float 训练器::验证步骤(const torch::Tensor& 输入, const torch::Tensor& 目标) {
        // 设置模型为评估模式
        模型_->eval();

        // 禁用梯度计算
        torch::NoGradGuard 无梯度;

        auto 输出 = 模型_->前向传播(输入);
        auto 损失 = 损失函数_(输出, 目标);

        return 损失.item<float>();
    }

    void 训练器::记录训练指标(int64_t 轮次, float 训练损失, float 验证损失) {
        训练历史_.轮次列表.push_back(轮次);
        训练历史_.训练损失列表.push_back(训练损失);

        if (验证损失 > 0.0f) {
            训练历史_.验证损失列表.push_back(验证损失);
        }
    }

    void 训练器::更新学习率(int64_t 当前轮次) {
        if (配置_.学习率调度类型 == 学习率调度类型::指数衰减) {
            // 每10轮次衰减一次
            if (当前轮次 > 0 && 当前轮次 % 10 == 0) {
                auto 当前学习率 = 配置_.学习率 * std::pow(配置_.衰减率, 当前轮次 / 10);
                设置学习率(当前学习率);
            }
        }
    }

    void 训练器::设置学习率(float 新学习率) {
        // 验证学习率合理性
        if (新学习率 <= 0.0f || 新学习率 > 1.0f) {
            TORCH_WARN("警告: 学习率 ", 新学习率, " 超出合理范围 (0, 1]，使用当前值: ", 配置_.学习率);
            return;
        }

        // 记录旧值
        float 旧学习率 = 配置_.学习率;
        配置_.学习率 = 新学习率;

        TORCH_INFO("设置学习率: ", 旧学习率, " -> ", 新学习率);

        // 1. 更新 PyTorch 优化器
        if (torch优化器_ && !torch优化器_->param_groups().empty()) {
            try {
                for (auto& 参数组 : torch优化器_->param_groups()) {
                    auto& 选项 = 参数组.options();

                    if (auto* sgd选项 = dynamic_cast<torch::optim::SGDOptions*>(&选项)) {
                        sgd选项->lr(新学习率);
                    }
                    else if (auto* adam选项 = dynamic_cast<torch::optim::AdamOptions*>(&选项)) {
                        adam选项->lr(新学习率);
                    }
                    else if (auto* adamw选项 = dynamic_cast<torch::optim::AdamWOptions*>(&选项)) {
                        adamw选项->lr(新学习率);
                    }
                    else if (auto* rmsprop选项 = dynamic_cast<torch::optim::RMSpropOptions*>(&选项)) {
                        rmsprop选项->lr(新学习率);
                    }
                }
                TORCH_INFO("  - PyTorch 优化器学习率已同步");
            }
            catch (const std::exception& e) {
                TORCH_ERROR("  - 更新 PyTorch 优化器失败: ", e.what());
            }
        }

        // 2. 更新 Vulkan 优化器
        if (vulkan优化器_) {
            try {
                vulkan优化器_->设置学习率(新学习率);
                TORCH_INFO("  - Vulkan GPU 优化器学习率已同步");
            }
            catch (const std::exception& e) {
                TORCH_ERROR("  - 更新 Vulkan 优化器失败: ", e.what());
            }
        }

        // 3. 更新训练统计
        训练统计_["学习率"] = 新学习率;
    }

    void 训练器::保存检查点(int64_t 轮次, float 损失) {
        std::string 文件名 = 配置_.检查点目录 + "/checkpoint_epoch_" +
            std::to_string(轮次) + "_loss_" +
            std::to_string(损失) + ".pt";

        torch::save(模型_, 文件名);
        训练历史_.检查点文件列表.push_back(文件名);

        TORCH_INFO("检查点已保存: ", 文件名);
    }

    void 训练器::加载检查点(const std::string& 文件路径) {
        torch::load(模型_, 文件路径);
        TORCH_INFO("检查点已加载: ", 文件路径);
    }

    std::string 训练器::优化器类型转字符串(优化器类型 类型) {
        switch (类型) {
        case 优化器类型::SGD: return "SGD";
        case 优化器类型::Adam: return "Adam";
        case 优化器类型::AdamW: return "AdamW";
        case 优化器类型::RMSprop: return "RMSprop";
        default: return "未知";
        }
    }

    训练统计信息 训练器::获取统计信息() const {
        训练统计信息 统计;

        if (!训练历史_.训练损失列表.empty()) {
            统计.最终训练损失 = 训练历史_.训练损失列表.back();
            统计.最佳训练损失 = *std::min_element(
                训练历史_.训练损失列表.begin(), 训练历史_.训练损失列表.end());
        }

        if (!训练历史_.验证损失列表.empty()) {
            统计.最终验证损失 = 训练历史_.验证损失列表.back();
            统计.最佳验证损失 = *std::min_element(
                训练历史_.验证损失列表.begin(), 训练历史_.验证损失列表.end());
        }

        统计.训练轮次 = 训练历史_.轮次列表.size();
        统计.检查点数量 = 训练历史_.检查点文件列表.size();

        return 统计;
    }
