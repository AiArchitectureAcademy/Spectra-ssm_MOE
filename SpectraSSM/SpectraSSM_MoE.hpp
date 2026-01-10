#pragma once
// ============================================================
// SpectraSSM_MoE.hpp - 频域混合专家分类器完整声明
// 文件名: SpectraSSM_MoE.hpp
// 依赖: SpectraSSM.hpp, torch, C++20
// ============================================================

#include "SpectraSSM.hpp"
#include <torch/torch.h>
#include <torch/nn.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <cmath>

namespace SpectraSSM {

/**
 * @brief 频域MoE配置结构体
 */
struct 频域MoE配置 {
    int64_t 专家数量 = 8;                    // 总专家数量
    int64_t 激活专家数 = 2;                  // 每个样本激活的专家数量
    int64_t 专家隐藏维度 = 256;              // 每个专家的隐藏维度
    int64_t 门控隐藏层维度 = 512;            // 门控网络的隐藏层维度
    float 负载均衡权重 = 0.01f;              // 负载均衡损失权重
    float 专业化权重 = 0.005f;               // 专业化损失权重
    bool 启用频域专业化 = true;              // 是否启用频域专业化引导
    bool 启用动态路由 = true;                // 是否启用动态专家选择
    int64_t 专业化频段数 = 4;                // 频域专业化频段数量
    float 路由Dropout率 = 0.1f;              // 门控网络Dropout率
    bool 启用专家缓存 = true;                // 是否启用专家计算缓存
    
    // 序列长度相关的动态配置
    struct 动态配置 {
        int64_t 短序列阈值 = 128;            // 短序列长度阈值
        int64_t 中序列阈值 = 512;            // 中序列长度阈值
        int64_t 短序列激活专家数 = 4;        // 短序列激活专家数
        int64_t 中序列激活专家数 = 2;        // 中序列激活专家数
        int64_t 长序列激活专家数 = 1;        // 长序列激活专家数
    } 动态设置;
};


/**
 * @brief 训练配置参数
 */
struct 训练配置 {
    float 最大梯度范数 = 1.0f;           // 梯度裁剪阈值
    float 标签平滑 = 0.1f;               // 分类标签平滑
    int64_t 总训练步数 = 100000;         // 总训练步数
    int64_t 统计记录间隔 = 100;          // 统计记录间隔
    int64_t 检查点间隔 = 1000;           // 检查点保存间隔
    int64_t 早停耐心 = 10;               // 早停耐心轮数
    std::string 检查点目录 = "./checkpoints"; // 检查点保存目录
};

/**
 * @brief 分析配置参数
 */
struct 分析配置 {
    int64_t 采样数量 = 1000;              // 分析采样数量
    int64_t 热力图序列长度 = 50;          // 热力图序列长度
    float 频段分析精度 = 0.01f;           // 频段分析精度
    float 专家相似度阈值 = 0.7f;          // 专家相似度阈值
    bool 保存可视化数据 = true;           // 是否保存可视化数据
    std::string 可视化目录 = "./moe_visualizations"; // 可视化数据目录
};

/**
 * @class 频域MoE分类器
 * @brief 基于频域特征路由的混合专家分类器
 * 
 * 核心特性：
 * - 频域特征驱动的智能专家路由
 * - 动态专家激活机制，适应不同序列长度
 * - 负载均衡与专业化损失，确保专家差异化
 * - 频域状态空间模型专家网络
 */
class 频域MoE分类器 : public torch::nn::Module {
public:
    /**
     * @brief 构造函数
     * @param 输入维度 输入特征维度
     * @param 类别数 分类类别数量
     * @param 配置 MoE配置参数
     */
    频域MoE分类器(int64_t 输入维度, int64_t 类别数, 
                 const 频域MoE配置& 配置 = 频域MoE配置{});

    /**
     * @brief 前向传播
     * @param 输入 输入张量 [批大小, 序列长度, 输入维度]
     * @return 分类输出 [批大小, 序列长度, 类别数]
     */
    torch::Tensor 前向传播(const torch::Tensor& 输入);

    /**
     * @brief 计算路由相关损失
     * @return 元组包含(重要性损失, 负载损失, 专业化损失)
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 计算路由损失();

    /**
     * @brief 获取专家使用统计
     * @return 专家使用频率向量 [专家数量]
     */
    torch::Tensor 获取专家使用统计() const;

    /**
     * @brief 获取门控权重（用于可视化分析）
     * @param 输入 输入张量
     * @return 门控权重矩阵 [批大小, 专家数量]
     */
    torch::Tensor 获取门控权重(const torch::Tensor& 输入);

    /**
     * @brief 重置专家缓存
     */
    void 重置专家缓存();

    /**
     * @brief 获取配置信息
     */
    频域MoE配置 获取配置() const { return 配置_; }

    /**
     * @brief 设置激活专家数（动态调整）
     */
    void 设置激活专家数(int64_t 新激活专家数);

private:
    int64_t 输入维度_;
    int64_t 类别数_;
    频域MoE配置 配置_;

    // 专家网络池
    std::vector<std::shared_ptr<频域状态空间模型>> 专家网络_;
    
    // 门控网络
    torch::nn::Sequential 门控网络_{nullptr};
    torch::nn::Dropout 门控Dropout_{nullptr};
    
    // 输出层
    torch::nn::Linear 输出层_{nullptr};

    // 专家专业化引导参数
    std::vector<torch::Tensor> 专家频域偏置_;  // 每个专家的频域偏置

    // 运行时统计
    mutable torch::Tensor 专家使用统计_;      // 累计使用统计
    mutable int64_t 统计更新次数_ = 0;

    // 专家输出缓存（提升推理性能）
    mutable std::unordered_map<int64_t, torch::Tensor> 专家输出缓存_;
    mutable int64_t 当前缓存序列长度_ = -1;
    mutable torch::Tensor 当前缓存输入_;

    /**
     * @brief 初始化专家网络
     */
    void 初始化专家网络();

    /**
     * @brief 初始化门控网络
     */
    void 初始化门控网络();

    /**
     * @brief 提取频域特征用于路由决策
     */
    torch::Tensor 提取频域特征(const torch::Tensor& 输入) const;

    /**
     * @brief 计算门控权重（包含Top-K稀疏化）
     */
    torch::Tensor 计算门控权重(const torch::Tensor& 输入);

    /**
     * @brief 动态计算激活专家数量
     */
    int64_t 计算动态激活专家数(int64_t 序列长度) const;

    /**
     * @brief 专业化引导：为不同专家设置频域偏置
     */
    void 初始化专家专业化();

    /**
     * @brief 应用专家专业化偏置
     */
    void 应用专家专业化();

    /**
     * @brief 更新专家使用统计
     */
    void 更新使用统计(const torch::Tensor& 门控权重);

    /**
     * @brief 清空专家缓存
     */
    void 清空专家缓存();
};

/**
 * @class 频域MoE训练器
 * @brief MoE分类器的专用训练器，处理负载均衡等特性
 */
class 频域MoE训练器 {
public:
    /**
     * @brief 构造函数
     * @param 模型 MoE分类器实例
     * @param 优化器 基础优化器
     * @param 设备 训练设备
     */
    频域MoE训练器(std::shared_ptr<频域MoE分类器> 模型,
                 std::shared_ptr<torch::optim::Optimizer> 优化器,
                 torch::Device 设备 = torch::kCPU);

    /**
     * @brief 训练步骤
     * @param 输入 输入张量
     * @param 标签 目标标签
     * @return 总损失值
     */
    torch::Tensor 训练步骤(const torch::Tensor& 输入, const torch::Tensor& 标签);

    /**
     * @brief 验证步骤
     */
    std::tuple<torch::Tensor, torch::Tensor> 验证步骤(
        const torch::Tensor& 输入, const torch::Tensor& 标签);

    /**
     * @brief 获取训练统计信息
     */
    std::unordered_map<std::string, float> 获取训练统计() const;

    std::unordered_map<std::string, float> 获取验证统计() const;

    int64_t 获取训练步数() const;

    /**
     * @brief 保存检查点
     */
    void 保存检查点(const std::string& 路径);

    /**
     * @brief 加载检查点
     */
    void 加载检查点(const std::string& 路径);

    void 设置配置(const 训练配置& 新配置);

    训练配置 获取配置() const;

private:
    训练配置 配置_;
    std::unordered_map<std::string, float> 验证统计_;
    float 最佳损失_ = std::numeric_limits<float>::max();
    int64_t 无改善轮数_ = 0;
    
    // 私有方法
    float 计算梯度范数();
    void 更新学习率统计();
    void 记录专家统计();
    torch::Tensor 计算专家熵(const torch::Tensor& 使用统计);
    void 自适应损失权重调整();
    void 打印训练状态();
    void 早停检查();
    std::shared_ptr<频域MoE分类器> 模型_;
    std::shared_ptr<torch::optim::Optimizer> 优化器_;
    torch::Device 设备_;

    // 训练统计
    std::unordered_map<std::string, float> 训练统计_;
    int64_t 训练步数_ = 0;

    /**
     * @brief 计算总损失（分类损失 + MoE特定损失）
     */
    torch::Tensor 计算总损失(const torch::Tensor& 输出, const torch::Tensor& 标签);
};

/**
 * @class 频域MoE分析器
 * @brief MoE分类器的分析和可视化工具
 */
class 频域MoE分析器 {
public:
    /**
     * @brief 构造函数
     * @param 模型 MoE分类器实例
     */
    频域MoE分析器(std::shared_ptr<频域MoE分类器> 模型);

    /**
     * @brief 分析专家专业化程度
     * @return 专家专业化度量和可视化数据
     */
    std::unordered_map<std::string, torch::Tensor> 分析专家专业化();

    /**
     * @brief 生成路由热力图
     * @param 输入样本 输入样本张量
     * @return 路由热力图数据
     */
    torch::Tensor 生成路由热力图(const torch::Tensor& 输入样本);

    /**
     * @brief 分析频域特征重要性
     */
    std::unordered_map<std::string, float> 分析特征重要性();

    /**
     * @brief 生成专家使用报告
     */
    void 生成专家使用报告();

private:
    // 私有方法
    torch::Tensor 计算专家相似度矩阵(const torch::Tensor& 样本数据);
    torch::Tensor 获取特定专家输出(const torch::Tensor& 输入, int64_t 专家索引);
    torch::Tensor 计算专业化指数(const torch::Tensor& 相似度矩阵);
    std::tuple<torch::Tensor, torch::Tensor> 分析频域响应特性(const torch::Tensor& 样本数据);
    std::vector<torch::Tensor> 生成多频段测试信号(int64_t 批次大小, int64_t 序列长度, 
                                               int64_t 特征维度, int64_t 频段数, torch::Device 设备);
    torch::Tensor 分析路由稳定性(const torch::Tensor& 样本数据);
    void 保存热力图数据(const torch::Tensor& 门控权重, const torch::Tensor& 输入样本);
    float 模拟特征消融(const torch::Tensor& 样本数据, const std::string& 特征类型, float 基准熵);
    torch::Tensor 计算门控熵(const torch::Tensor& 门控权重);
    std::string 获取当前时间();
    torch::Tensor 计算专家熵(const torch::Tensor& 使用统计);
    void 可视化专家网络(const std::string& 输出路径);
    std::shared_ptr<频域MoE分类器> 模型_;

    /**
     * @brief 计算专家输出相似度矩阵
     */
    torch::Tensor 计算专家相似度矩阵();

    /**
     * @brief 分析频段响应特性
     */
    std::vector<float> 分析频段响应();
};

// 工具函数
namespace MoE工具 {
    /**
     * @brief 创建预设MoE配置
     */
    频域MoE配置 创建平衡配置(int64_t 输入维度, int64_t 类别数);
    频域MoE配置 创建高性能配置(int64_t 输入维度, int64_t 类别数);
    频域MoE配置 创建轻量配置(int64_t 输入维度, int64_t 类别数);

    /**
     * @brief 专家权重初始化策略
     */
    void 初始化专家权重(torch::nn::Module& 专家网络, int64_t 专家索引, int64_t 总专家数);

    /**
     * @brief 计算MoE模型参数数量
     */
    int64_t 计算参数数量(const 频域MoE分类器& 模型);

} // namespace MoE工具

} // namespace SpectraSSM