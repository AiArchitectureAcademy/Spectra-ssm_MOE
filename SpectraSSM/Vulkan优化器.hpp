// SpectraSSM/Vulkan优化器.hpp
#pragma once

#include "SpectraSSM.hpp"
#include <vulkan/vulkan.hpp>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>

namespace SpectraSSM {

    /**
     * @brief Vulkan GPU 优化器接口
     * @details 在 GPU 上执行优化算法，支持 PyTorch 张量
     */
    class Vulkan优化器 {
    public:
        virtual ~Vulkan优化器() = default;

        /**
         * @brief GPU 参数更新
         * @param 参数名列表 参数名称列表
         * @param 参数列表 参数张量列表
         * @param 梯度列表 梯度张量列表
         */
        virtual void 优化参数(
            const std::vector<std::string>& 参数名列表,
            const std::vector<torch::Tensor>& 参数列表,
            const std::vector<torch::Tensor>& 梯度列表) = 0;

        /**
         * @brief 获取当前学习率
         */
        virtual float 获取学习率() const = 0;

        /**
         * @brief 设置学习率
         */
        virtual void 设置学习率(float 新学习率) = 0;

        /**
         * @brief 步数更新
         */
        virtual void 步数递增() = 0;
    };

    /**
     * @brief Vulkan Adam GPU 优化器
     * @details 在 GPU 上完整实现 Adam 算法
     */
    class VulkanAdam优化器 : public Vulkan优化器 {
    public:
        VulkanAdam优化器(float 学习率 = 0.001f,
            float 贝塔1 = 0.9f,
            float 贝塔2 = 0.999f,
            float 权重衰减 = 0.0f,
            float 埃普西隆 = 1e-8f,
            int 设备编号 = 0);

        ~VulkanAdam优化器();

        void 优化参数(
            const std::vector<std::string>& 参数名列表,
            const std::vector<torch::Tensor>& 参数列表,
            const std::vector<torch::Tensor>& 梯度列表) override;

        float 获取学习率() const override { return 学习率_; }
        void 设置学习率(float 新学习率) override { 学习率_ = 新学习率; }
        void 步数递增() override { 步数_++; }

    private:
        // Vulkan 对象
        vk::Instance 实例_;
        vk::Device 逻辑设备_;
        vk::PhysicalDevice 物理设备_;
        vk::Queue 计算队列_;
        vk::CommandPool 命令池_;
        vk::DescriptorPool 描述符池_;

        // 计算管线
        vk::Pipeline 计算管线_;
        vk::PipelineLayout 管线布局_;
        vk::DescriptorSetLayout 描述符布局_;

        // 优化器参数
        float 学习率_ = 0.001f;
        float 贝塔1_ = 0.9f;
        float 贝塔2_ = 0.999f;
        float 权重衰减_ = 0.0f;
        float 埃普西隆_ = 1e-8f;
        int64_t 步数_ = 0;

        // GPU 设备编号
        int 设备编号_ = 0;

        // GPU 内存缓冲区管理
        struct GPU参数缓冲 {
            vk::Buffer 参数缓冲;
            vk::Buffer 梯度缓冲;
            vk::Buffer 动量缓冲;
            vk::Buffer 二阶动量缓冲;
            vk::DeviceMemory 内存;
            size_t 元素数量;
        };

        std::unordered_map<std::string, GPU参数缓冲> 参数缓冲映射_;

        // Uniform 缓冲区
        struct AdamUniforms {
            float 学习率;
            float 贝塔1;
            float 贝塔2;
            float 贝塔1修正;
            float 贝塔2修正;
            float 埃普西隆;
            float 权重衰减;
            int32_t 步数;
            int32_t 元素数量;
        };

        vk::Buffer uniform缓冲区_;
        vk::DeviceMemory uniform内存_;

        // 私有方法
        void 初始化Vulkan();
        void 创建计算管线();
        void 创建参数缓冲(const std::string& 参数名, size_t 元素数量);
        void 上传张量数据(const torch::Tensor& 张量, vk::Buffer 目标缓冲);
        void 下载张量数据(torch::Tensor& 张量, vk::Buffer 源缓冲);
        void 执行GPU优化(const std::string& 参数名);
        void 更新Uniform缓冲区();
        std::vector<uint32_t> 编译GLSL到SPIRV(const std::string& 着色器代码);
        std::vector<uint32_t> 加载预编译SPIRV();
        std::vector<uint32_t> 从文件加载SPIRV(const std::string& 文件路径);
    };

} // namespace SpectraSSM