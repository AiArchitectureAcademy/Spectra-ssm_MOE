// SpectraSSM/训练器.hpp
#pragma once

#include "SpectraSSM.hpp"
#include <torch/torch.h>
#include <torch/data.h>
#include <torch/data/dataloader.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/sgd.h>
#include <torch/optim/adam.h>
#include <torch/optim/adamw.h>
#include <torch/optim/rmsprop.h>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include "Vulkan优化器.hpp"

namespace SpectraSSM {

    // 优化器类型枚举
    enum class 优化器类型 {
        SGD,
        Adam,
        AdamW,
        RMSprop
    };

    // 学习率调度类型枚举
    enum class 学习率调度类型 {
        恒定,
        指数衰减,
        阶梯衰减
    };

    // 训练配置结构体
    struct 训练配置 {
        优化器类型 优化器类型 = 优化器类型::Adam;
        学习率调度类型 学习率调度类型 = 学习率调度类型::恒定;
        float 学习率 = 0.001f;
        float 动量 = 0.9;
        float 权重衰减 = 0.0001f;
        float 贝塔1 = 0.9f;
        float 贝塔2 = 0.999f;
        float 梯度裁剪 = 1.0f;
        float 衰减率 = 0.95f;
        int64_t 批大小 = 32;
        std::string 检查点目录 = "./checkpoints";
    };

    // 训练历史记录
    struct 训练历史 {
        std::vector<int64_t> 轮次列表;
        std::vector<float> 训练损失列表;
        std::vector<float> 验证损失列表;
        std::vector<std::string> 检查点文件列表;
    };

    // 训练统计信息
    struct 训练统计信息 {
        float 最终训练损失 = 0.0f;
        float 最佳训练损失 = 0.0f;
        float 最终验证损失 = 0.0f;
        float 最佳验证损失 = 0.0f;
        size_t 训练轮次 = 0;
        size_t 检查点数量 = 0;
    };

    // 简单的数据集结构
    struct 训练数据集 : torch::data::Dataset<训练数据集> {
        torch::Tensor 数据, 目标;

        训练数据集(torch::Tensor 输入数据, torch::Tensor 目标数据)
            : 数据(输入数据), 目标(目标数据) {
        }

        torch::data::Example<> get(size_t index) override {
            return { 数据[index], 目标[index] };
        }

        torch::optional<size_t> size() const override {
            return 数据.size(0);
        }
    };

    /**
     * @class 训练器
     * @brief 模型训练管理器
     */
    class 训练器 {
    public:
        /**
         * @brief 构造函数
         * @param 模型 要训练的频域MoE模型
         * @param 配置 训练配置参数
         */
        训练器(std::shared_ptr<频域MoE模型> 模型, const 训练配置& 配置);

        /**
         * @brief 执行单次训练步骤
         * @param 输入 输入张量 [批大小, 序列长度, 模型维度]
         * @param 目标 目标张量 [批大小, 序列长度, 状态维度]
         * @return 训练损失值
         */
        float 训练步骤(const torch::Tensor& 输入, const torch::Tensor& 目标);

        /**
         * @brief 执行验证步骤
         * @param 输入 输入张量
         * @param 目标 目标张量
         * @return 验证损失值
         */
        float 验证步骤(const torch::Tensor& 输入, const torch::Tensor& 目标);

        /**
         * @brief 执行完整训练循环
         * @param 训练数据加载器 训练数据迭代器
         * @param 验证数据加载器 验证数据迭代器（可选）
         * @param 轮次数 训练轮次数量
         */
        template<typename DataLoaderType>
        void 训练(DataLoaderType& 训练数据加载器,
            DataLoaderType* 验证数据加载器 = nullptr,
            int64_t 轮次数 = 100);

        /**
         * @brief 设置学习率
         * @param 新学习率 新的学习率值
         */
        void 设置学习率(float 新学习率);

        /**
         * @brief 保存模型检查点
         * @param 轮次 当前训练轮次
         * @param 损失 当前损失值
         */
        void 保存检查点(int64_t 轮次, float 损失);

        /**
         * @brief 加载模型检查点
         * @param 文件路径 检查点文件路径
         */
        void 加载检查点(const std::string& 文件路径);

        /**
         * @brief 获取训练统计信息
         * @return 训练统计信息结构体
         */
        训练统计信息 获取统计信息() const;

        /**
         * @brief 获取训练历史记录
         * @return 训练历史记录引用
         */
        const 训练历史& 获取训练历史() const { return 训练历史_; }
        //@brief 切换到 Vulkan GPU 优化器

        void 使用Vulkan优化器(std::shared_ptr<Vulkan优化器> vulkan优化器) {
            vulkan优化器_ = vulkan优化器;
            TORCH_INFO("已切换到 ", 优化器类型转字符串(配置_.优化器类型), " (Vulkan GPU 加速)");
        }
    private:
        /**
         * @brief 初始化优化器
         */
        void 初始化优化器();

        void 初始化SGD优化器(const std::vector<torch::Tensor>& 参数列表);

        void 初始化Adam优化器(const std::vector<torch::Tensor>& 参数列表);

        void 初始化AdamW优化器(const std::vector<torch::Tensor>& 参数列表);

        void 初始化RMSprop优化器(const std::vector<torch::Tensor>& 参数列表);

        void 初始化Vulkan优化器();

        /**
         * @brief 记录训练指标
         * @param 轮次 当前轮次
         * @param 训练损失 训练损失值
         * @param 验证损失 验证损失值
         */
        void 记录训练指标(int64_t 轮次, float 训练损失, float 验证损失);

        /**
         * @brief 更新学习率
         * @param 当前轮次 当前训练轮次
         */
        void 更新学习率(int64_t 当前轮次);

        /**
         * @brief 优化器类型转字符串
         * @param 类型 优化器类型枚举
         * @return 优化器类型字符串表示
         */
        std::string 优化器类型转字符串(优化器类型 类型);

    private:
        std::shared_ptr<频域MoE模型> 模型_;  ///< 要训练的模型
        训练配置 配置_;                           ///< 训练配置参数
        // 传统优化器
        std::unique_ptr<torch::optim::Optimizer> torch优化器_;

        // Vulkan GPU 优化器
        std::shared_ptr<Vulkan优化器> vulkan优化器_;
        torch::nn::MSELoss 损失函数_;             ///< 损失函数

        训练历史 训练历史_;                        ///< 训练历史记录
        int64_t 当前轮次_;                        ///< 当前训练轮次
        float 最佳损失_;                          ///< 最佳验证损失值
    };

    // 模板方法的实现需要放在头文件中
    template<typename DataLoaderType>
    void 训练器::训练(DataLoaderType& 训练数据加载器,
        DataLoaderType* 验证数据加载器,
        int64_t 轮次数) {

        TORCH_INFO("开始训练，总轮次: ", 轮次数);

        auto 开始时间 = std::chrono::steady_clock::now();

        for (当前轮次_ = 0; 当前轮次_ < 轮次数; ++当前轮次_) {
            auto 轮次开始时间 = std::chrono::steady_clock::now();

            // 训练阶段
            float 累计训练损失 = 0.0f;
            int64_t 批次数 = 0;

            模型_->train();
            for (auto& 批次 : 训练数据加载器) {
                auto 损失 = 训练步骤(批次.data, 批次.target);
                累计训练损失 += 损失;
                批次数++;
            }

            float 平均训练损失 = 累计训练损失 / 批次数;

            // 验证阶段（如果提供了验证数据）
            float 平均验证损失 = 0.0f;
            if (验证数据加载器 != nullptr) {
                int64_t 验证批次数 = 0;
                float 累计验证损失 = 0.0f;

                模型_->eval();
                torch::NoGradGuard 无梯度;

                for (auto& 批次 : *验证数据加载器) {
                    auto 损失 = 验证步骤(批次.data, 批次.target);
                    累计验证损失 += 损失;
                    验证批次数++;
                }

                平均验证损失 = 累计验证损失 / 验证批次数;
            }

            // 记录指标
            记录训练指标(当前轮次_, 平均训练损失, 平均验证损失);

            // 检查点保存
            if (平均验证损失 < 最佳损失_) {
                最佳损失_ = 平均验证损失;
                保存检查点(当前轮次_, 平均验证损失);
            }

            // 学习率调度
            更新学习率(当前轮次_);

            auto 轮次耗时 = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - 轮次开始时间);

            TORCH_INFO("轮次 ", 当前轮次_ + 1, "/", 轮次数,
                " - 训练损失: ", std::fixed << std::setprecision(6) << 平均训练损失,
                (验证数据加载器 != nullptr ? " - 验证损失: " + std::to_string(平均验证损失) : ""),
                " - 耗时: ", 轮次耗时.count(), "ms");
        }

        auto 总耗时 = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - 开始时间);

        TORCH_INFO("训练完成，总耗时: ", 总耗时.count(), "秒");
    }

} // namespace SpectraSSM