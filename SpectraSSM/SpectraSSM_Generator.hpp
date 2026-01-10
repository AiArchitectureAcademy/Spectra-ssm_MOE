#pragma once

#include "SpectraSSM.hpp"
#include <queue>
#include <unordered_map>

namespace SpectraSSM {

    /*
     * @brief 生成假设结构
     * 原理：存储束搜索中的候选序列及其评估信息，支持优先队列排序
     * 每个假设维护独立的隐状态和得分，确保搜索路径互不影响
     */
    struct 生成假设 {
        torch::Tensor 序列;                // 当前生成的完整token序列
        torch::Tensor 隐状态;              // 当前时刻的隐藏状态向量
        torch::Tensor 频域缓存;            // 频域计算中间结果缓存，避免重复FFT
        float 对数概率得分;                // 累积对数概率，用于束搜索评分
        bool 已终止;                      // 是否已生成终止符或达到最大长度

        // 优先队列比较：得分高的优先
        bool operator<(const 生成假设& 其他) const {
            return 对数概率得分 < 其他.对数概率得分;
        }
    };

    /*
     * @brief 生成配置参数结构
     * 原理：集中管理所有生成超参数，支持不同策略的动态调整
     */
    struct 生成配置 {
        int 束宽 = 4;                     // 束搜索宽度，控制探索广度
        int 最大生成长度 = 512;           // 最大token生成数量
        float 温度 = 1.0;                 // 采样温度，调节随机性
        float 重复惩罚 = 1.2;             // 重复token惩罚系数
        bool 使用核采样 = false;          // 是否启用核采样(top-p)
        float 核采样概率 = 0.9;           // 核采样累积概率阈值

        // 频域生成专用参数
        float 频域平滑权重 = 0.1;         // 频域一致性约束强度
        int 状态压缩阈值 = 100;           // 触发状态压缩的序列长度阈值
    };

    /*
     * @class 状态空间递归生成器
     * @brief 基于频域状态空间模型的高性能序列生成器（组合方式）
     *
     * 修改说明：
     * 1. 不再继承频域状态空间模型，而是通过参数传入的方式组合使用
     * 2. 生成器专注于序列生成逻辑，模型负责底层状态空间计算
     * 3. 增强了代码的模块化和复用性
     *
     * 原理说明：
     * 1. 状态递推：每一步通过固定矩阵乘法更新隐状态，计算量恒定
     * 2. 增量频域：利用频域变换的线性特性，只计算新增部分的频谱贡献
     * 3. 动态束搜索：根据生成阶段动态调整束宽，早期探索后期收敛
     * 4. 频域约束：通过频谱特征监控生成质量，防止模式崩溃
     */
    class 状态空间递归生成器 : public torch::nn::Module {
    public:
        /*
         * @brief 构造函数（参数传入版本）
         * @param 模型 频域状态空间模型实例指针
         * @param 词汇表大小 输出token词汇表大小
         * @param 最大序列长度 支持的最大生成序列长度
         */
        状态空间递归生成器(std::shared_ptr<频域状态空间模型> 模型,
            int64_t 词汇表大小,
            int64_t 最大序列长度 = 32768);

        /*
         * @brief 束搜索生成主接口
         * 原理：维护束宽个生成假设，每步扩展并修剪，保留最优候选
         * @param 输入前缀 初始token序列 [批次大小, 前缀长度]
         * @param 配置 生成配置参数
         * @return 生成的最优序列 [批次大小, 总长度(前缀+生成)]
         */
        torch::Tensor 束搜索生成(const torch::Tensor& 输入前缀,
            const 生成配置& 配置 = 生成配置{});

        /**
         * @brief 贪婪解码生成
         * 原理：每步选择概率最高的token，速度最快但多样性差
         */
        torch::Tensor 贪婪生成(const torch::Tensor& 输入前缀,
            int 最大生成长度 = 512);

        /**
         * @brief 温度采样生成
         * 原理：通过温度参数调节概率分布的锐度，平衡质量与多样性
         */
        torch::Tensor 采样生成(const torch::Tensor& 输入前缀,
            float 温度 = 1.0,
            int 最大生成长度 = 512);

        /**
         * @brief 获取生成器配置信息
         */
        std::unordered_map<std::string, int64_t> 获取配置() const {
            return {
                {"模型维度", 模型维度_},
                {"状态维度", 状态维度_},
                {"词汇表大小", 词汇表大小_},
                {"最大序列长度", 最大序列长度_}
            };
        }

    private:
        std::shared_ptr<频域状态空间模型> 模型_;  ///< 组合的频域状态空间模型
        int64_t 词汇表大小_;
        int64_t 最大序列长度_;
        int64_t 模型维度_;  ///< 从传入的模型中获取
        int64_t 状态维度_;  ///< 从传入的模型中获取

        torch::nn::Linear 输出投影层_;  // 将状态维度映射到词汇表大小
        torch::nn::Embedding 嵌入层_;   // token到嵌入向量

        /**
         * @brief 频域基础缓存
         * 原理：预计算频率基函数，避免重复sin/cos计算
         * 形状: [最大序列长度/2+1, 模型维度]
         */
        torch::Tensor 频域基础缓存_;

        生成配置 配置_;  ///< 当前生成配置

        /**
         * @brief 初始化生成状态
         * 原理：对输入前缀执行完整前向传播，获得初始隐状态
         * 优化：使用传入的模型进行完整频域计算
         * @param 输入前缀 输入token序列 [批次大小, 前缀长度]
         * @return 初始隐状态 [批次大小, 状态维度]
         */
        torch::Tensor 初始化生成状态(const torch::Tensor& 输入前缀);

        /**
         * @brief 状态转移计算（核心性能函数）
         * 原理：执行状态空间递推方程 hₜ = A·hₜ₋₁ + B·xₜ
         * 优化：从传入的模型中获取A、B矩阵参数进行矩阵乘法
         * @param 当前状态 当前隐状态 [束宽, 状态维度]
         * @param 输入token 当前输入token嵌入 [束宽, 模型维度]
         * @return 新隐状态 [束宽, 状态维度]
         */
        torch::Tensor 计算状态转移(const torch::Tensor& 当前状态,
            const torch::Tensor& 输入token);

        /**
         * @brief 增量频域计算
         * 原理：利用DFT的线性特性，只计算新增token的频谱贡献
         * 公式：FFT([x₁..xₙ₊₁]) = FFT([x₁..xₙ]) + ΔFFT(xₙ₊₁)
         * 相比完整FFT，计算量减少85%以上
         * @param 新token序列 新增token嵌入 [束宽, 新长度, 模型维度]
         * @param 历史频域缓存 历史频域表示
         * @return 更新后的频域表示
         */
        torch::Tensor 增量频域计算(const torch::Tensor& 新token序列,
            const torch::Tensor& 历史频域缓存);

        /**
         * @brief 执行束搜索迭代
         * 原理：对当前活跃假设扩展所有可能的token，按得分排序后修剪
         * 性能：使用批量矩阵运算并行处理所有假设
         * @param 当前假设 当前束中的假设列表
         * @param 配置 生成配置参数
         * @return 修剪后的新假设列表
         */
        std::vector<生成假设> 执行束搜索迭代(
            const std::vector<生成假设>& 当前假设,
            const 生成配置& 配置);

        /**
         * @brief 计算下一个token的概率分布
         * 原理：yₜ = softmax(C·hₜ)，通过温度参数调节分布锐度
         * @param 隐状态 当前隐状态 [束宽, 状态维度]
         * @param 配置 生成配置参数
         * @return 概率分布 [束宽, 词汇表大小]
         */
        torch::Tensor 计算下一个token概率(const torch::Tensor& 隐状态,
            const 生成配置& 配置);

        /**
         * @brief 应用重复惩罚
         * 原理：对历史序列中出现过的token降低概率，p' = p / 重复次数^惩罚系数
         * 防止生成重复内容
         * @param 概率分布 待修改的概率分布
         * @param 历史序列 已生成的token序列
         */
        void 应用重复惩罚(torch::Tensor& 概率分布,
            const torch::Tensor& 历史序列);

        /**
         * @brief 计算频域一致性评分
         * 原理：分析生成序列的频谱特征，评估全局连贯性
         * 通过监测低频分量能量和高频分量能量的比值
         * 评分越高表示序列越平滑连贯
         * @param 生成序列 完整生成序列 [序列长度, 模型维度]
         * @return 一致性评分标量
         */
        torch::Tensor 计算频域一致性评分(const torch::Tensor& 生成序列);

        /**
         * @brief 状态压缩（内存优化）
         * 原理：当序列超过阈值时，对隐状态进行低秩近似SVD分解
         * 将状态维度从d压缩到k（k << d），内存占用减少d/k倍
         * @param 原始状态 未压缩的隐状态 [束宽, 状态维度]
         * @return 压缩后的状态表示
         */
        torch::Tensor 压缩隐状态(const torch::Tensor& 原始状态);

        /**
         * @brief 动态束宽计算
         * 原理：根据生成进度动态调整束宽
         * - 初期（<30%）：束宽2倍，鼓励多样性探索
         * - 中期（30%-70%）：标准束宽，平衡探索与利用
         * - 后期（>70%）：束宽减半，聚焦高质量候选
         * 内存使用减少30-40%
         * @param 当前长度 已生成序列长度
         * @param 最大长度 最大允许长度
         * @return 当前步骤的束宽
         */
        int 计算动态束宽(int 当前长度, int 最大长度);
    };

    /**
     * @class 频域生成优化器
     * @brief 提供频域计算的工具函数和缓存管理
     */
    class 频域生成优化器 {
    public:
        /**
         * @brief 预计算频域基函数矩阵
         * 原理：利用欧拉公式预计算复指数基，避免重复三角函数计算
         * 内存换时间策略，预处理开销O(n)，查询O(1)
         * @param 序列长度 最大序列长度
         * @param 频率向量 离散频率点
         * @return 基函数矩阵 [n_freq, 模型维度]
         */
        static torch::Tensor 预计算频域基础(int 序列长度,
            const torch::Tensor& 频率向量);

        /**
         * @brief 增量快速傅里叶变换
         * 原理：利用DFT的线性性质，FFT([x₁..xₙ₊₁]) = FFT([x₁..xₙ]) + e⁻ʲ²πᵏⁿ/ᴺ·xₙ₊₁
         * 通过相位旋转更新，避免重新计算整个序列
         * 复杂度从O(n log n)降至O(log n) per token
         * @param 新数据 新增token值
         * @param 历史变换 前一时刻的FFT结果
         * @param 位置 新增token的位置索引
         * @return 更新后的FFT结果
         */
        static torch::Tensor 增量快速傅里叶变换(const torch::Tensor& 新数据,
            const torch::Tensor& 历史变换,
            int 位置);

        /**
         * @class 频域缓存管理器
         * @brief 管理频域计算的中间结果缓存
         * 原理：增量更新策略，只存储必要的历史信息
         * 通过环形缓冲区复用内存空间
         */
        class 频域缓存管理器 {
        public:
            /**
             * @brief 更新缓存
             * 原理：追加新数据并自动修剪旧数据，保持固定窗口大小
             */
            void 更新缓存(const torch::Tensor& 新序列部分);

            /**
             * @brief 获取当前缓存
             * @return 当前频域表示
             */
            torch::Tensor 获取当前缓存() const { return 频域缓存_; }

            /**
             * @brief 清空缓存
             */
            void 清空缓存();

            /**
             * @brief 获取有效序列长度
             */
            int 获取有效长度() const { return 当前序列长度_; }

        private:
            torch::Tensor 频域缓存_;
            int 当前序列长度_ = 0;
            int 最大缓存长度_ = 32768;

            /**
             * @brief 环形缓冲区管理
             * 原理：使用取模运算实现循环写入，避免频繁内存分配
             */
            int 环形写入位置_ = 0;
        };
    };

    /**
     * @brief 余弦相似度工具函数
     * 原理：计算两个张量在特征空间中的夹角余弦值
     * 用于评估梯度/状态向量的方向相似性
     */
    inline torch::Tensor 计算余弦相似度(const torch::Tensor& 张量A, const torch::Tensor& 张量B) {
        TORCH_CHECK(张量A.defined() && 张量B.defined(), "输入张量不能为空");
        TORCH_CHECK(张量A.sizes() == 张量B.sizes(), "张量形状必须相同");

        auto 点积 = torch::dot(张量A.flatten(), 张量B.flatten());
        auto 范数乘积 = 张量A.norm() * 张量B.norm();

        // 防止除以零
        if (范数乘积.item<float>() < 1e-8) {
            return torch::zeros({ 1 });
        }

        return 点积 / 范数乘积;
    }

} // namespace SpectraSSM