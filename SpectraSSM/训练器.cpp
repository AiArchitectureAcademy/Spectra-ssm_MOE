// SpectraSSM/训练器.cpp
#include "训练器.hpp"
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace SpectraSSM {

    训练器::训练器(
        std::shared_ptr<频域状态空间模型> 模型,
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
        auto 参数 = 模型_->获取参数列表();

        switch (配置_.优化器类型) {
        case 优化器类型::SGD:
            优化器_ = std::make_unique<torch::optim::SGD>(
                参数, torch::optim::SGDOptions(配置_.学习率)
                .momentum(配置_.动量)
                .weight_decay(配置_.权重衰减));
            break;

        case 优化器类型::Adam:
            优化器_ = std::make_unique<torch::optim::Adam>(
                参数, torch::optim::AdamOptions(配置_.学习率)
                .betas(std::make_tuple(配置_.贝塔1, 配置_.贝塔2))
                .weight_decay(配置_.权重衰减));
            break;

        case 优化器类型::AdamW:
            优化器_ = std::make_unique<torch::optim::AdamW>(
                参数, torch::optim::AdamWOptions(配置_.学习率)
                .betas(std::make_tuple(配置_.贝塔1, 配置_.贝塔2))
                .weight_decay(配置_.权重衰减));
            break;

        case 优化器类型::RMSprop:
            优化器_ = std::make_unique<torch::optim::RMSprop>(
                参数, torch::optim::RMSpropOptions(配置_.学习率)
                .momentum(配置_.动量)
                .weight_decay(配置_.权重衰减));
            break;

        default:
            TORCH_ERROR("不支持的优化器类型: ", static_cast<int>(配置_.优化器类型));
            throw std::invalid_argument("不支持的优化器类型");
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
        for (auto& 参数组 : 优化器_->param_groups()) {
            if (参数组.has_options()) {
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
        }
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
