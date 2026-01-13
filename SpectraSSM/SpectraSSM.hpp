
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
#pragma once
// ============================================================
// SpectraSSM - 频域MoE模型完整声明
// 文件名: SpectraSSM.hpp
// 依赖: torch, C++20
// ============================================================
#include <torch/torch.h>
#include <torch/nn.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include <memory>
#include <iostream>
#include <iomanip>
// 递归展开：处理单个参数
template<typename T>
void 日志输出(std::ostringstream& oss, T&& 值) {
    oss << std::forward<T>(值);
}

// 递归展开：处理多个参数（折叠表达式方式，C++17及以上）
template<typename T, typename... Args>
void 日志输出(std::ostringstream& oss, T&& 值, Args&&... 参数) {
    oss << std::forward<T>(值);
    日志输出(oss, std::forward<Args>(参数)...);
}

// 包装函数：添加时间戳
template<typename... Args>
void TORCH_INFO(Args&&... 参数) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time_t);
    std::ostringstream oss;
    oss << "[INFO][" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] ";
    日志输出(oss, std::forward<Args>(参数)...);
}

template<typename... Args>
void TORCH_ERROR(Args&&... 参数) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time_t);
    std::ostringstream oss;
    oss << "[ERROR][" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] ";
    日志输出(oss, std::forward<Args>(参数)...);
}

template<typename... Args>
void TORCH_WARN(Args&&... 参数) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time_t);
    std::ostringstream oss;
    oss << "[WARN][" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] ";
    日志输出(oss, std::forward<Args>(参数)...);
}



    // 获取张量的设备，如果张量未定义则返回CPU
inline torch::Device 获取安全设备(const torch::Tensor & 张量) {
    if (张量.defined()) {
        return 张量.device();
    }
#ifdef TORCH_CUDA_AVAILABLE
    if (torch::cuda::is_available() && torch::cuda::device_count() > 0) {
        return torch::kCUDA;
    }
#endif
    return torch::kCPU;
}

// 确保张量在目标设备上
inline torch::Tensor 移动到设备(torch::Tensor 张量, torch::Device 目标设备) {
    if (张量.defined() && 张量.device() != 目标设备) {
        return 张量.to(目标设备);
    }
    return 张量;
}


namespace SpectraSSM {



    /**
     * @class 频域MoE模型
     * @brief 基于傅里叶变换的并行状态空间模型核心实现
     *
     * 核心创新：
     * - 将时域递归转换为频域独立计算，消除序列长度上的数据依赖
     * - 传递函数核预计算与缓存机制，避免重复矩阵求逆
     * - 支持超长序列处理（最大32768 tokens），复杂度O(n log n)
     * - 完全可微分，支持端到端梯度反向传播
     *
     * 数学原理：
     *   H(ω) = (jωI - A)⁻¹B
     *   Y(ω) = H(ω) · X(ω)
     *   y(t) = iFFT(Y(ω))
     */
    class 频域MoE模型 : public torch::nn::Module {
    public:
        /**
         * @param 模型维度 输入/输出特征维度（d_model）
         * @param 状态维度 状态空间隐藏维度（d_state）
         * @param 最大序列长度 支持的最大序列长度（默认32768）
         */
        频域MoE模型(int64_t 模型维度, int64_t 状态维度,
            int64_t 最大序列长度 = 32768);

        /**
         * @brief 前向传播
         * @param 输入 输入张量 [批大小, 序列长度, 模型维度]
         * @return 输出状态 [批大小, 序列长度, 状态维度]
         */
        torch::Tensor 前向传播(const torch::Tensor& 输入)const;
        // 路由概率访问接口
        const torch::Tensor& 获取最后路由概率() const {
            TORCH_CHECK(
                最后路由概率_.defined() && 最后路由概率_.numel() > 0,
                "错误: 必须先执行前向传播才能获取路由概率"
            );
            return 最后路由概率_;
        }
        /**
         * @brief 重置传递函数缓存
         * @note 在参数更新后调用，强制重新计算传递函数核
         */
        void 重置缓存();

        /**
         * @brief 标记核需要重算（参数已更新）
         * @note 比reset_cache更轻量，不清除缓存数据
         */
        void 标记核需要重算();

        /**
         * @brief 获取所有可训练参数
         * @return 参数向量，可用于梯度管理器
         */
        std::vector<torch::Tensor> 获取参数列表();

        /**
         * @brief 获取命名参数映射
         * @return 参数名称到参数的映射，用于动态分组
         */
        std::unordered_map<std::string, torch::Tensor*> 获取命名参数映射();

        std::unordered_map<std::string, const torch::Tensor*> 获取命名参数映射() const;

        /**
         * @brief 获取模型配置信息
         * @return 配置字典（维度、序列长度等）
         */
        std::unordered_map<std::string, int64_t> 获取配置() const;
        /**
         * @brief 计算频率向量 ω = 2πk/n
         * @param 序列长度 当前输入序列长度
         * @return 频率向量 [n_freq]
         */
        torch::Tensor 计算频率(int64_t 序列长度)const;
    private:
        torch::Tensor 最后路由概率_; // 内部缓存
        // 模型核心维度
        int64_t 模型维度_;
        int64_t 状态维度_;
        int64_t 最大序列长度_;

        // 频域参数矩阵（可训练）
        torch::Tensor A_频率;  // 状态转移矩阵 [状态维度, 状态维度]
        torch::Tensor B_频率;  // 输入映射矩阵 [模型维度, 状态维度]  
        torch::Tensor C_频率;  // 输出映射矩阵 [状态维度, 模型维度]

        // 运行时缓存（不可训练）
        mutable torch::Tensor 频率缓存_;         // [序列长度/2 + 1]
        mutable torch::Tensor 传递函数缓存_;     // [序列长度/2 + 1, 状态维度, 模型维度]
        mutable bool 核已计算_ = false;
        mutable int64_t 当前缓存序列长度_ = 0;
        // 新增多尺度参数
        int 多尺度数量_ = 3;  // 基础+高频+低频
        mutable std::vector<torch::Tensor> 多尺度频率缓存_;      // 不同尺度的频率向量
        mutable std::vector<torch::Tensor> 多尺度传递函数缓存_;  // 不同尺度的传递函数核

        // 新增多尺度权重参数
        torch::Tensor 尺度融合权重_;  // 可学习的尺度融合权重

        // 新增多尺度计算方法
        torch::Tensor 计算多尺度频率(int64_t 序列长度) const;
        torch::Tensor 计算多尺度传递函数核(int64_t 序列长度) const;
        torch::Tensor 多尺度频域融合(const std::vector<torch::Tensor>& 尺度输出) const;


        /**
         * @brief 计算传递函数核 H(ω) = (jωI - A)⁻¹B
         * @param 序列长度 当前输入序列长度
         * @return 传递函数核 [n_freq, 状态维度, 模型维度]
         */
        torch::Tensor 计算传递函数核(int64_t 序列长度)const;

        /**
         * @brief 批量矩阵求逆（使用LU分解）
         * @param 矩阵组 [批大小, N, N]
         * @return 逆矩阵 [批大小, N, N]
         */
        torch::Tensor 批量矩阵求逆(const torch::Tensor& 矩阵组)const;
    };

} // namespace SpectraSSM
