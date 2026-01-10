// ============================================================
// 梯度特征分析器实现

#include "SpectraSSM.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <queue>
#include <tuple>        // std::get
#include <deque>
#include <iomanip>      // std::put_time
#include <sstream>      // std::ostringstream
#include <chrono>
#include <numeric>      // std::accumulate
#include <unordered_set>
#include <time.h>

namespace SpectraSSM {

    /**
     * @brief 构造函数 - 初始化频域参数矩阵和缓存
     * @param 模型维度 输入/输出特征维度
     * @param 状态维度 状态空间隐藏维度
     * @param 最大序列长度 支持的最大序列长度
     */
    频域状态空间模型::频域状态空间模型(
        int64_t 模型维度,
        int64_t 状态维度,
        int64_t 最大序列长度)
        : 模型维度_(模型维度), 状态维度_(状态维度), 最大序列长度_(最大序列长度) {

        // 参数验证
        TORCH_CHECK(模型维度 > 0, "模型维度必须大于0");
        TORCH_CHECK(状态维度 > 0, "状态维度必须大于0");
        TORCH_CHECK(最大序列长度 > 0 && 最大序列长度 <= 32768,
            "最大序列长度必须在1-32768之间");

        // 初始化频域参数矩阵，使用小随机值初始化
        // A矩阵: 状态转移矩阵 [状态维度, 状态维度]
        A_频率 = register_parameter(
            "A_频率",
            torch::randn({ 状态维度, 状态维度 }, torch::kFloat32) * 0.02
        );

        // B矩阵: 输入映射矩阵 [模型维度, 状态维度]
        B_频率 = register_parameter(
            "B_频率",
            torch::randn({ 模型维度, 状态维度 }, torch::kFloat32) * 0.01
        );

        // C矩阵: 输出映射矩阵 [状态维度, 模型维度]
        C_频率 = register_parameter(
            "C_频率",
            torch::randn({ 状态维度, 模型维度 }, torch::kFloat32) * 0.01
        );

        // 初始化缓存为空张量
        频率缓存_ = register_buffer("频率缓存", torch::zeros({ 0 }, torch::kFloat32));
        传递函数缓存_ = register_buffer("传递函数缓存", torch::zeros({ 0 }, torch::kComplexFloat));

        // 初始化状态标志
        核已计算_ = false;
        当前缓存序列长度_ = 0;

        TORCH_INFO("频域状态空间模型初始化完成，模型维度: ", 模型维度,
            ", 状态维度: ", 状态维度,
            ", 最大序列长度: ", 最大序列长度);
        // 新增：初始化多尺度融合权重 [多尺度数量]
        尺度融合权重_ = register_parameter(
            "尺度融合权重",
            torch::ones({ 多尺度数量_ }, torch::kFloat32) / 多尺度数量_
        );

        TORCH_INFO("多尺度频域模型初始化完成，尺度数量: ", 多尺度数量_);

    }

    // 频域状态空间模型前向传播实现
    // 功能：执行频域并行状态空间计算
    // 输入: [批大小, 序列长度, 模型维度]
    // 输出: [批大小, 序列长度, 状态维度]
    torch::Tensor 频域状态空间模型::前向传播(const torch::Tensor& 输入) const {
        // ------------------------------------------------------------------
        // 第一阶段：输入验证和设备检查
        // ------------------------------------------------------------------
        TORCH_CHECK(输入.defined(), "输入张量未定义");
        TORCH_CHECK(输入.dim() == 3, "输入必须是3维张量 [批大小, 序列长度, 模型维度]");
        TORCH_CHECK(输入.size(2) == 模型维度_,
            "输入特征维度不匹配，期望 ", 模型维度_, " 但得到 ", 输入.size(2));

        const torch::Device 计算设备 = 输入.device();

        // 确保模型参数在正确设备上
        if (A_频率.device() != 计算设备) {
            const_cast<频域状态空间模型*>(this)->to(计算设备);
            TORCH_INFO("模型参数已迁移到输入设备: ", 计算设备);
        }

        const int64_t 批大小 = 输入.size(0);
        const int64_t 序列长度 = 输入.size(1);
        const int64_t 频率数量 = 序列长度 / 2 + 1;

        TORCH_CHECK(序列长度 <= 最大序列长度_,
            "序列长度 ", 序列长度, " 超过最大支持长度 ", 最大序列长度_);

        // ------------------------------------------------------------------
        // 第二阶段：多尺度传递函数核计算与缓存管理
        // ------------------------------------------------------------------
        bool 需要重算核 = false;

        // 检查重算条件：首次计算、序列长度变化、或缓存为空
        if (!核已计算_ || 序列长度 != 当前缓存序列长度_ || 多尺度传递函数缓存_.empty()) {
            TORCH_INFO("触发多尺度传递函数核重算，原因:",
                !核已计算_ ? "首次计算" :
                序列长度 != 当前缓存序列长度_ ? "序列长度变化" : "缓存为空");

            需要重算核 = true;
        }

        if (需要重算核) {
            try {
                TORCH_INFO("开始计算多尺度传递函数核，序列长度: ", 序列长度);

                // 清空现有缓存
                多尺度频率缓存_.clear();
                多尺度传递函数缓存_.clear();

                // 计算多尺度频率向量
                auto 多尺度频率 = 计算多尺度频率(序列长度);

                // 计算多尺度传递函数核
                auto 堆叠核 = 计算多尺度传递函数核(序列长度);

                // 验证计算结果
                TORCH_CHECK(多尺度传递函数缓存_.size() == 多尺度数量_,
                    "多尺度核数量不匹配，期望 ", 多尺度数量_, " 但得到 ", 多尺度传递函数缓存_.size());

                核已计算_ = true;
                当前缓存序列长度_ = 序列长度;

                TORCH_INFO("多尺度传递函数核计算成功，缓存尺度数量: ", 多尺度传递函数缓存_.size());

            }
            catch (const std::exception& 异常) {
                TORCH_ERROR("多尺度传递函数核计算失败: ", 异常.what());
                throw;
            }
        }
        else {
            TORCH_INFO("使用缓存的多尺度传递函数核，序列长度: ", 序列长度);
        }

        // 最终验证缓存状态
        TORCH_CHECK(多尺度传递函数缓存_.size() == 多尺度数量_,
            "多尺度核缓存异常，期望 ", 多尺度数量_, " 但得到 ", 多尺度传递函数缓存_.size());

        // ------------------------------------------------------------------
        // 第三阶段：输入数据频域转换
        // ------------------------------------------------------------------
        TORCH_INFO("开始输入数据频域转换...");

        // 调整维度: [批大小, 序列长度, 模型维度] -> [批大小, 模型维度, 序列长度]
        auto 输入转置 = 输入.transpose(1, 2).contiguous();

        // 执行实数FFT: 输出复数张量 [批大小, 模型维度, 频率数量]
        auto 输入频域 = torch::fft_rfft(输入转置, /*n=*/序列长度, /*dim=*/-1);

        // 调整维度以适配后续矩阵乘法: [批大小, 频率数量, 模型维度]
        auto X = 输入频域.transpose(1, 2);

        TORCH_CHECK(X.defined(), "频域转换失败");
        TORCH_CHECK(X.size(0) == 批大小, "批大小维度错误");
        TORCH_CHECK(X.size(1) == 频率数量, "频域维度错误");
        TORCH_CHECK(X.size(2) == 模型维度_, "模型维度错误");

        TORCH_INFO("频域转换完成，输入频域形状: ", X.sizes());

        // ------------------------------------------------------------------
        // 第四阶段：多尺度并行频域计算
        // ------------------------------------------------------------------
        TORCH_INFO("开始多尺度并行频域计算...");

        std::vector<torch::Tensor> 尺度输出列表;
        尺度输出列表.reserve(多尺度数量_);

        // 为每个尺度计算频域响应
        for (int 尺度索引 = 0; 尺度索引 < 多尺度数量_; ++尺度索引) {
            try {
                const auto& 传递函数核 = 多尺度传递函数缓存_[尺度索引];

                // 验证核的形状和有效性
                TORCH_CHECK(传递函数核.defined(), "尺度 ", 尺度索引, " 的传递函数核未定义");
                TORCH_CHECK(传递函数核.size(0) == 频率数量,
                    "尺度 ", 尺度索引, " 频率维度不匹配，期望 ", 频率数量, " 但得到 ", 传递函数核.size(0));
                TORCH_CHECK(传递函数核.size(1) == 状态维度_,
                    "状态维度不匹配，期望 ", 状态维度_, " 但得到 ", 传递函数核.size(1));
                TORCH_CHECK(传递函数核.size(2) == 模型维度_,
                    "模型维度不匹配，期望 ", 模型维度_, " 但得到 ", 传递函数核.size(2));

                TORCH_INFO("尺度 ", 尺度索引, " 频域乘法开始，核形状: ", 传递函数核.sizes());

                // 频域矩阵乘法: Y[批, 状态, 频率] = sum(模型维度) X[批, 频率, 模型维度] * H[频率, 状态, 模型维度]
                auto 尺度输出频域 = torch::einsum("bmf,fds->bsf", { X, 传递函数核 });

                TORCH_CHECK(尺度输出频域.defined(), "尺度 ", 尺度索引, " 频域乘法失败");
                TORCH_CHECK(尺度输出频域.size(0) == 批大小, "批大小不匹配");
                TORCH_CHECK(尺度输出频域.size(1) == 状态维度_, "状态维度不匹配");
                TORCH_CHECK(尺度输出频域.size(2) == 频率数量, "频率维度不匹配");

                TORCH_INFO("尺度 ", 尺度索引, " 频域乘法完成，开始逆变换...");

                // 逆FFT变换回时域: [批大小, 状态维度, 序列长度]
                auto 尺度输出时域 = torch::fft_irfft(尺度输出频域, /*n=*/序列长度, /*dim=*/-1);

                TORCH_CHECK(尺度输出时域.defined(), "尺度 ", 尺度索引, " 逆变换失败");
                TORCH_CHECK(尺度输出时域.size(2) == 序列长度,
                    "序列长度不匹配，期望 ", 序列长度, " 但得到 ", 尺度输出时域.size(2));

                // 数值稳定性检查
                if (torch::isnan(尺度输出时域).any().item<bool>()) {
                    TORCH_WARN("尺度 ", 尺度索引, " 输出包含NaN，进行清理");
                    尺度输出时域 = torch::nan_to_num(尺度输出时域, 0.0, 0.0, 0.0);
                }

                if (torch::isinf(尺度输出时域).any().item<bool>()) {
                    TORCH_WARN("尺度 ", 尺度索引, " 输出包含Inf，进行裁剪");
                    尺度输出时域 = torch::clamp(尺度输出时域, -1e6, 1e6);
                }

                尺度输出列表.push_back(尺度输出时域);

                TORCH_INFO("尺度 ", 尺度索引, " 计算完成，输出形状: ", 尺度输出时域.sizes(),
                    ", 数值范围: [", 尺度输出时域.min().item<float>(),
                    ", ", 尺度输出时域.max().item<float>(), "]");

            }
            catch (const std::exception& 尺度异常) {
                TORCH_ERROR("尺度 ", 尺度索引, " 计算失败: ", 尺度异常.what());

                // 创建安全的零输出作为降级
                auto 零输出 = torch::zeros({ 批大小, 状态维度_, 序列长度 },
                    torch::TensorOptions().device(计算设备).dtype(torch::kFloat32));
                尺度输出列表.push_back(零输出);

                TORCH_WARN("尺度 ", 尺度索引, " 使用零输出作为降级");
            }
        }

        // 确保所有尺度都有输出
        TORCH_CHECK(尺度输出列表.size() == 多尺度数量_,
            "尺度输出数量不匹配，期望 ", 多尺度数量_, " 但得到 ", 尺度endl输出列表.size());

        // ------------------------------------------------------------------
        // 第五阶段：多尺度自适应融合
        // ------------------------------------------------------------------
        TORCH_INFO("开始多尺度自适应融合...");

        torch::Tensor 融合输出时域;

        try {
            融合输出时域 = 多尺度频域融合(尺度输出列表);

            TORCH_CHECK(融合输出时域.defined(), "多尺度融合失败");
            TORCH_CHECK(融合输出时域.sizes() == 尺度输出列表[0].sizes(),
                "融合输出形状不匹配");

            // 最终数值验证
            if (torch::isnan(融合输出时域).any().item<bool>()) {
                TORCH_WARN("融合输出包含NaN，使用第一尺度作为降级");
                融合输出时域 = 尺度输出列表[0].clone();
            }

            TORCH_INFO("多尺度融合成功，输出形状: ", 融合输出时域.sizes());

        }
        catch (const std::exception& 融合异常) {
            TORCH_ERROR("多尺度融合失败: ", 融合异常.what());

            // 降级策略：使用第一尺度输出
            融合输出时域 = 尺度输出列表[0].clone();
            TORCH_WARN("使用第一尺度输出作为融合降级");
        }

        // ------------------------------------------------------------------
        // 第六阶段：输出整理和最终验证
        // ------------------------------------------------------------------
        TORCH_INFO("开始输出整理...");

        // 转置回标准形状: [批大小, 序列长度, 状态维度]
        auto 最终结果 = 融合输出时域.transpose(1, 2);

        TORCH_CHECK(最终结果.defined(), "最终输出转置失败");
        TORCH_CHECK(最终结果.size(0) == 批大小, "最终输出批大小不匹配");
        TORCH_CHECK(最终结果.size(1) == 序列长度, "最终输出序列长度不匹配");
        TORCH_CHECK(最终结果.size(2) == 状态维度_, "最终输出状态维度不匹配");

        // 最终数值检查
        auto 最终数值检查 = [&]() {
            if (torch::isnan(最终结果).any().item<bool>()) {
                TORCH_ERROR("最终输出包含NaN值，计算失败");
                return false;
            }

            if (torch::isinf(最终结果).any().item<bool>()) {
                TORCH_WARN("最终输出包含Inf值，可能发生数值溢出");
                // 不返回失败，而是进行裁剪
                最终结果 = torch::clamp(最终结果, -1e6, 1e6);
            }

            // 检查输出范围是否合理
            float 输出范数 = 最终结果.norm().item<float>();
            float 预期范数 = std::sqrt(批大小 * 序列长度 * 状态维度_) * 0.1f; // 经验阈值

            if (输出范数 > 预期范数 * 10.0f) {
                TORCH_WARN("输出范数异常: ", 输出范数, " (预期约 ", 预期范数, ")");
            }

            return true;
            };

        if (!最终数值检查()) {
            // 数值检查失败，创建安全输出
            最终结果 = torch::zeros({ 批大小, 序列长度, 状态维度_ },
                torch::TensorOptions().device(计算设备).dtype(torch::kFloat32));
            TORCH_WARN("使用零张量作为最终输出降级");
        }

        // ------------------------------------------------------------------
        // 第七阶段：性能统计和日志输出
        // ------------------------------------------------------------------
        auto 统计信息 = [&]() -> std::string {
            std::ostringstream oss;
            oss << "批大小: " << 批大小
                << ", 序列长度: " << 序列长度
                << ", 输出形状: " << 最终结果.sizes()
                << ", 输出均值: " << std::fixed << std::setprecision(6) << 最终结果.mean().item<float>()
                << ", 输出标准差: " << 最终结果.std().item<float>()
                << ", 输出范围: [" << 最终结果.min().item<float>()
                << ", " << 最终结果.max().item<float>() << "]";
            return oss.str();
            };

        TORCH_INFO("多尺度频域前向传播完成 - " + 统计信息());

        return 最终结果;
    }
    torch::Tensor 频域状态空间模型::计算多尺度频率(int64_t 序列长度) const {
        TORCH_CHECK(序列长度 > 0, "序列长度必须大于0");

        int64_t 频率数量 = 序列长度 / 2 + 1;

        // 基础频率（线性分布）
        auto 基础频率 = torch::linspace(
            0,
            M_PI * 2.0 * (频率数量 - 1) / 序列长度,
            频率数量,
            torch::TensorOptions().dtype(torch::kFloat32).device(A_频率.device())
        );

        // 高频偏向频率（频率平方加权）
        auto 高频频率 = 基础频率.pow(2.0);
        高频频率 = 高频频率 / 高频频率.max() * M_PI * 2.0;

        // 低频偏向频率（频率开方加权）  
        auto 低频频率 = 基础频率.sqrt();
        低频频率 = 低频频率 / 低频频率.max() * M_PI * 2.0;

        多尺度频率缓存_ = { 基础频率, 高频频率, 低频频率 };

        TORCH_INFO("多尺度频率计算完成，基础范围: [", 基础频率.min().item<float>(),
            ", ", 基础频率.max().item<float>(), "]");

        return torch::stack(多尺度频率缓存_);
    }
    torch::Tensor 频域状态空间模型::计算多尺度传递函数核(int64_t 序列长度) const {
        TORCH_CHECK(序列长度 > 0, "序列长度必须大于0");

        // 计算多尺度频率（如果尚未计算）
        if (多尺度频率缓存_.empty()) {
            计算多尺度频率(序列长度);
        }

        int64_t 频率数量 = 序列长度 / 2 + 1;
        std::vector<torch::Tensor> 尺度核列表;

        // 将实数矩阵转换为复数矩阵
        auto A_复数 = torch::complex(A_频率, torch::zeros_like(A_频率));
        auto B_复数 = torch::complex(B_频率, torch::zeros_like(B_频率));
        auto B_转置 = B_复数.transpose(0, 1);

        // 创建单位矩阵
        auto 单位矩阵 = torch::eye(状态维度_, torch::kComplexFloat);

        // 为每个尺度计算传递函数核
        for (int 尺度索引 = 0; 尺度索引 < 多尺度数量_; ++尺度索引) {
            const auto& 频率 = 多尺度频率缓存_[尺度索引];

            // 扩展频率维度: [频率数量, 1, 1]
            auto 频率_扩展 = 频率.view({ 频率数量, 1, 1 });

            // 复数单位j的张量
            auto j张量 = torch::tensor({ 0.0f, 1.0f }, torch::kComplexFloat).view({ 1, 1, 1 });

            // 计算 M = jωI - A
            auto jω = j张量 * 频率_扩展;
            auto jωI = jω * 单位矩阵.unsqueeze(0);
            auto M = jωI - A_复数.unsqueeze(0);

            // 批量矩阵求逆
            auto M_逆 = 批量矩阵求逆(M);

            // 计算传递函数核: H(ω) = (jωI - A)⁻¹B
            auto B_转置_扩展 = B_转置.unsqueeze(0);
            auto 核 = torch::matmul(M_逆, B_转置_扩展); // [频率数量, 状态维度, 模型维度]

            尺度核列表.push_back(核);

            TORCH_INFO("尺度 ", 尺度索引, " 传递函数核计算完成，形状: ", 核.sizes());
        }

        多尺度传递函数缓存_ = 尺度核列表;

        // 堆叠所有尺度的核 [多尺度数量, 频率数量, 状态维度, 模型维度]
        auto 堆叠核 = torch::stack(尺度核列表);

        TORCH_INFO("多尺度传递函数核计算完成，总形状: ", 堆叠核.sizes());
        return 堆叠核;
    }
    torch::Tensor 频域状态空间模型::多尺度频域融合(
        const std::vector<torch::Tensor>& 尺度输出) const {

        TORCH_CHECK(尺度输出.size() == 多尺度数量_, "尺度输出数量不匹配");

        auto 设备 = 尺度输出[0].device();
        auto 权重 = 尺度融合权重_.to(设备);

        // 静态权重融合（基础版本）
        权重 = torch::softmax(权重, 0);

        torch::Tensor 融合输出 = torch::zeros_like(尺度输出[0]);

        for (int i = 0; i < 多尺度数量_; ++i) {
            融合输出 += 权重[i] * 尺度输出[i];
        }

        // 动态权重融合（增强版本）
        try {
            auto 输出特征 = 融合输出.mean(1); // [批大小, 状态维度]

            // 简单的线性投影生成动态权重
            auto 投影权重 = torch::ones({ 状态维度_, 多尺度数量_ }, 设备);
            auto 动态权重 = torch::sigmoid(输出特征 @ 投影权重); // [批大小, 多尺度数量]

            // 应用动态权重
            融合输出 = torch::zeros_like(尺度输出[0]);
            for (int i = 0; i < 多尺度数量_; ++i) {
                auto 尺度权重 = 动态权重.index({ torch::indexing::Slice(), i })
                    .unsqueeze(1).unsqueeze(2);
                融合输出 += 尺度权重 * 尺度输出[i];
            }

            TORCH_INFO("动态权重融合完成");
        }
        catch (const std::exception& 动态异常) {
            TORCH_WARN("动态权重融合失败，使用静态权重: ", 动态异常.what());
            // 保持静态权重结果
        }

        return 融合输出;
    }

    /**
     * @brief 计算频率向量
     * @param 序列长度 当前输入序列长度
     * @return 频率向量 [n_freq]，其中 n_freq = 序列长度 / 2 + 1
     */
    torch::Tensor 频域状态空间模型::计算频率(int64_t 序列长度)const{
        TORCH_CHECK(序列长度 > 0, "序列长度必须大于0");

        // FFT频率 bins: ω_k = 2πk / n, k = 0, 1, ..., n/2
        // 使用linspace生成频率值
        int64_t 频率数量 = 序列长度 / 2 + 1;

        auto 频率 = torch::linspace(
            0,
            M_PI * 2.0 * (频率数量 - 1) / 序列长度,
            频率数量,
            torch::TensorOptions().dtype(torch::kFloat32).device(A_频率.device())  // 修正API
        );

        TORCH_INFO("计算频率向量完成，序列长度: ", 序列长度,
            ", 频率数量: ", 频率数量);

        return 频率;
    }

    /**
     * @brief 计算传递函数核 H(ω) = (jωI - A)⁻¹B
     * @param 序列长度 当前输入序列长度
     * @return 传递函数核 [n_freq, 状态维度, 模型维度]
     */
    torch::Tensor 频域状态空间模型::计算传递函数核(int64_t 序列长度)const{
        TORCH_CHECK(序列长度 > 0, "序列长度必须大于0");

        TORCH_INFO("开始计算传递函数核，序列长度: ", 序列长度);

        int64_t 频率数量 = 序列长度 / 2 + 1;

        // 计算频率向量
        torch::Tensor 频率 = 计算频率(序列长度);

        // 将实数矩阵转换为复数矩阵
        // A矩阵: [状态维度, 状态维度] -> 复数
        auto A_复数 = torch::complex(A_频率, torch::zeros_like(A_频率));

        // B矩阵: [模型维度, 状态维度] -> 复数，并转置
        auto B_复数 = torch::complex(B_频率, torch::zeros_like(B_频率));
        auto B_转置 = B_复数.transpose(0, 1);  // [状态维度, 模型维度]

        // 创建单位矩阵 I: [状态维度, 状态维度]
        auto 单位矩阵 = torch::eye(
            状态维度_,
            torch::TensorOptions().dtype(torch::kComplexFloat).device(A_频率.device())
        );

        // 扩展频率维度: [频率数量, 1, 1]
        auto 频率_扩展 = 频率.view({ 频率数量, 1, 1 });

        // 创建复数单位j的张量表示 [1, 1, 1]
        auto j张量 = torch::tensor({ 0.0f, 1.0f },
            torch::TensorOptions().dtype(torch::kComplexFloat).device(A_频率.device()))
            .view({ 1, 1, 1 });

        // 正确计算 M = jωI - A
        // j张量 * 频率_扩展 得到 jω [频率数量, 1, 1]
        // 相乘后扩展为复数张量，再与单位矩阵相乘
        auto jω = j张量 * 频率_扩展;
        auto jωI = jω * 单位矩阵.unsqueeze(0);  // [频率数量, 状态维度, 状态维度]
        auto M = jωI - A_复数.unsqueeze(0);

        TORCH_CHECK(M.size(0) == 频率数量, "频率维度不匹配");
        TORCH_CHECK(M.size(1) == 状态维度_, "状态维度不匹配");
        TORCH_CHECK(M.size(2) == 状态维度_, "状态方阵维度不匹配");

        TORCH_INFO("构建矩阵 M = jωI - A 完成，形状: ", M.sizes());

        // 批量矩阵求逆 (jωI - A)⁻¹
        auto M_逆 = 批量矩阵求逆(M);

        TORCH_CHECK(M_逆.defined(), "矩阵求逆失败");

        // 计算传递函数核: H(ω) = (jωI - A)⁻¹B
        // M_逆: [频率数量, 状态维度, 状态维度]
        // B_转置: [状态维度, 模型维度]
        // 结果: [频率数量, 状态维度, 模型维度]

        // 扩展B矩阵维度以支持批量矩阵乘法
        auto B_转置_扩展 = B_转置.unsqueeze(0);  // [1, 状态维度, 模型维度]

        // 批量矩阵乘法
        auto 核 = torch::matmul(M_逆, B_转置_扩展);

        TORCH_CHECK(核.size(0) == 频率数量, "核频率维度不匹配");
        TORCH_CHECK(核.size(1) == 状态维度_, "核状态维度不匹配");
        TORCH_CHECK(核.size(2) == 模型维度_, "核模型维度不匹配");

        // 存储到缓存
        频率缓存_ = 频率;
        传递函数缓存_ = 核;
        核已计算_ = true;
        当前缓存序列长度_ = 序列长度;

        TORCH_INFO("传递函数核计算完成，缓存形状: ", 传递函数缓存_.sizes());

        return 核;
    }

    /**
     * @brief 批量矩阵求逆（使用LU分解）
     * @param 矩阵组 输入矩阵 [批大小, N, N]
     * @return 逆矩阵 [批大小, N, N]
     * @note PyTorch的inverse使用LAPACK/Magma库，数值稳定性好
     */
    torch::Tensor 频域状态空间模型::批量矩阵求逆(const torch::Tensor& 矩阵组)const{
        TORCH_CHECK(矩阵组.defined() && 矩阵组.dim() == 3,
            "输入必须是3维张量 [批大小, N, N]");
        TORCH_CHECK(矩阵组.size(1) == 矩阵组.size(2), "矩阵必须是方阵");

        // 使用torch::inverse进行批量求逆
        // 内部实现基于LU分解，复杂度约为O(n^3)
        auto 逆矩阵 = 矩阵组.inverse();

        TORCH_CHECK(逆矩阵.defined(), "矩阵求逆失败");

        // 检查数值稳定性：避免产生NaN或Inf
        if (torch::isnan(逆矩阵).any().item<bool>()) {
            TORCH_WARN("矩阵求逆结果包含NaN，可能发生数值不稳定");
        }
        if (torch::isinf(逆矩阵).any().item<bool>()) {
            TORCH_WARN("矩阵求逆结果包含Inf，可能发生数值溢出");
        }

        return 逆矩阵;
    }

    /**
     * @brief 重置传递函数缓存
     */
    void 频域状态空间模型::重置缓存() {
        核已计算_ = false;
        频率缓存_ = torch::zeros({ 0 }, torch::TensorOptions().device(A_频率.device()));  // 修正API
        传递函数缓存_ = torch::zeros({ 0 }, torch::TensorOptions().device(A_频率.device()));  // 修正API
        当前缓存序列长度_ = 0;

        TORCH_INFO("传递函数缓存已重置");
    }

    /**
     * @brief 标记核需要重算（参数已更新）
     */
    void 频域状态空间模型::标记核需要重算() {
        核已计算_ = false;
        TORCH_INFO("传递函数核标记为需要重新计算");
    }

    /**
     * @brief 获取所有可训练参数
     * @return 参数向量，用于梯度管理器
     */
    std::vector<torch::Tensor> 频域状态空间模型::获取参数列表() {
        std::vector<torch::Tensor> 参数列表;

        // 收集所有注册的可训练参数
        for (const auto& 参数对 : named_parameters()) {  
            参数列表.push_back(参数对.value());
        }

        TORCH_INFO("获取参数列表，共 ", 参数列表.size(), " 个参数");

        return 参数列表;
    }

    /**
     * @brief 获取命名参数映射
     * @return 参数名称到参数的映射，用于动态分组
     */
    std::unordered_map<std::string, torch::Tensor*> 频域状态空间模型::获取命名参数映射() {
        std::unordered_map<std::string, torch::Tensor*> 参数映射;

        // 直接使用named_parameters()返回的真实名称
        for (const auto& 参数对 : named_parameters()) {
            const std::string& 真实名称 = 参数对.key();
            torch::Tensor* 参数指针 = const_cast<torch::Tensor*>(&参数对.value());

            参数映射[真实名称] = 参数指针;

            TORCH_INFO("参数映射: ", 真实名称, " -> 形状=", 参数指针->sizes());
        }

        TORCH_INFO("获取命名参数映射完成，数量: ", 参数映射.size());

        return 参数映射;
    }
    std::unordered_map<std::string, const torch::Tensor*> 频域状态空间模型::获取命名参数映射() const {
        std::unordered_map<std::string, const torch::Tensor*> 参数映射;

        for (const auto& 参数对 : named_parameters()) {
            const std::string& 真实名称 = 参数对.key();
            const torch::Tensor* 参数指针 = &参数对.value();

            参数映射[真实名称] = 参数指针;
        }

        return 参数映射;
    }
    /**
     * @brief 获取模型配置信息
     * @return 配置字典（维度、序列长度等）
     */
    std::unordered_map<std::string, int64_t> 频域状态空间模型::获取配置() const {
        return {
            {"模型维度", 模型维度_},
            {"状态维度", 状态维度_},
            {"最大序列长度", 最大序列长度_},
            {"当前缓存序列长度", 当前缓存序列长度_},
            {"参数数量", A_频率.numel() + B_频率.numel() + C_频率.numel()}
        };
    }

} // namespace SpectraSSM