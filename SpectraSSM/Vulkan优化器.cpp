// SpectraSSM/Vulkan优化器.cpp
#include "Vulkan优化器.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <shaderc/shaderc.hpp>
#include <filesystem>

namespace SpectraSSM {

    // Adam 计算着色器 (GLSL)
    constexpr const char* ADAM_SHADER_SOURCE = R"(
#version 450
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// 参数缓冲区
layout(set = 0, binding = 0) buffer Params {
    float params[];
};

// 梯度缓冲区
layout(set = 0, binding = 1) buffer Gradients {
    float grads[];
};

// 动量缓冲区
layout(set = 0, binding = 2) buffer Momentum {
    float m[];
};

// 二阶动量缓冲区
layout(set = 0, binding = 3) buffer SecondMoment {
    float v[];
};

// Uniform 缓冲区
layout(set = 0, binding = 4) uniform AdamParams {
    float lr;
    float beta1;
    float beta2;
    float beta1_corr;
    float beta2_corr;
    float epsilon;
    float weight_decay;
    int step;
    int element_count;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= element_count) return;
    
    float grad = grads[idx];
    
    // 权重衰减
    if (weight_decay > 0.0) {
        grad += weight_decay * params[idx];
    }
    
    // 更新动量
    m[idx] = beta1 * m[idx] + (1.0 - beta1) * grad;
    v[idx] = beta2 * v[idx] + (1.0 - beta2) * grad * grad;
    
    // 偏差校正
    float m_hat = m[idx] * beta1_corr;
    float v_hat = v[idx] * beta2_corr;
    
    // 参数更新
    params[idx] = params[idx] - lr * m_hat / (sqrt(v_hat) + epsilon);
}
)";

    VulkanAdam优化器::VulkanAdam优化器(float 学习率, float 贝塔1, float 贝塔2,
        float 权重衰减, float 埃普西隆, int 设备编号)
        : 学习率_(学习率), 贝塔1_(贝塔1), 贝塔2_(贝塔2),
        权重衰减_(权重衰减), 埃普西隆_(埃普西隆), 设备编号_(设备编号) {

        TORCH_INFO("初始化 Vulkan GPU Adam 优化器...");
        TORCH_INFO("  - 学习率: ", 学习率_);
        TORCH_INFO("  - Beta1: ", 贝塔1_);
        TORCH_INFO("  - Beta2: ", 贝塔2_);
        TORCH_INFO("  - 权重衰减: ", 权重衰减_);
        TORCH_INFO("  - 设备编号: ", 设备编号_);

        初始化Vulkan();
        创建计算管线();

        TORCH_INFO("Vulkan GPU Adam 优化器初始化完成");
    }

    VulkanAdam优化器::~VulkanAdam优化器() {
        // 清理 GPU 资源
        for (auto& [名称, 缓冲] : 参数缓冲映射_) {
            逻辑设备_.destroyBuffer(缓冲.参数缓冲);
            逻辑设备_.destroyBuffer(缓冲.梯度缓冲);
            逻辑设备_.destroyBuffer(缓冲.动量缓冲);
            逻辑设备_.destroyBuffer(缓冲.二阶动量缓冲);
            逻辑设备_.freeMemory(缓冲.内存);
        }

        if (uniform缓冲区_) {
            逻辑设备_.destroyBuffer(uniform缓冲区_);
            逻辑设备_.freeMemory(uniform内存_);
        }

        if (计算管线_) 逻辑设备_.destroyPipeline(计算管线_);
        if (管线布局_) 逻辑设备_.destroyPipelineLayout(管线布局_);
        if (描述符布局_) 逻辑设备_.destroyDescriptorSetLayout(描述符布局_);
        if (描述符池_) 逻辑设备_.destroyDescriptorPool(描述符池_);
        if (命令池_) 逻辑设备_.destroyCommandPool(命令池_);
        if (逻辑设备_) 逻辑设备_.destroy();
        if (实例_) 实例_.destroy();

        TORCH_INFO("Vulkan 资源已清理");
    }

    void VulkanAdam优化器::初始化Vulkan() {
        TORCH_INFO("初始化 Vulkan...");

        // 1. 创建 Vulkan 实例
        vk::ApplicationInfo 应用信息("VulkanAdamOptimizer", VK_MAKE_VERSION(1, 0, 0),
            "SpectraSSM", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_2);

        // 获取可用的扩展
        auto 可用扩展 = vk::enumerateInstanceExtensionProperties();
        std::vector<const char*> 启用扩展;

        // 检查所需扩展是否可用并启用
        for (const auto& 扩展 : 可用扩展) {
            std::string_view 扩展名(扩展.extensionName);
            if (扩展名 == VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) {
                启用扩展.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
            }
        }

        // 必须启用 surface 相关扩展
        启用扩展.push_back(VK_KHR_SURFACE_EXTENSION_NAME);

#ifdef _WIN32
        启用扩展.push_back("VK_KHR_win32_surface");
#elif __linux__
        启用扩展.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#endif

        // 修正：直接传递扩展指针数组
        vk::InstanceCreateInfo 实例创建信息({}, &应用信息, 0, nullptr,
            static_cast<uint32_t>(启用扩展.size()), 启用扩展.data());

        实例_ = vk::createInstance(实例创建信息);

        // 2. 选择 GPU 设备
        auto 物理设备列表 = 实例_.enumeratePhysicalDevices();
        TORCH_CHECK(!物理设备列表.empty(), "未找到支持 Vulkan 的 GPU");

        if (设备编号_ >= 物理设备列表.size()) {
            TORCH_WARN("设备编号超出范围，使用默认 GPU");
            设备编号_ = 0;
        }

        物理设备_ = 物理设备列表[设备编号_];
        vk::PhysicalDeviceProperties 设备属性 = 物理设备_.getProperties();
        TORCH_INFO("使用 GPU: ", 设备属性.deviceName.data(),
            " (设备编号: ", 设备编号_, ")");

        // 3. 创建逻辑设备
        float 队列优先级 = 1.0f;
        auto 队列创建信息 = vk::DeviceQueueCreateInfo()
            .setQueueFamilyIndex(0).setQueueCount(1).setPQueuePriorities(&队列优先级);

        // 获取设备支持的扩展
        auto 设备可用扩展 = 物理设备_.enumerateDeviceExtensionProperties();
        std::vector<const char*> 启用设备扩展;

        // 检查所需设备扩展并启用
        for (const auto& 扩展 : 设备可用扩展) {
            std::string_view 扩展名(扩展.extensionName);
            if (扩展名 == VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME ||
                扩展名 == VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) {
                启用设备扩展.push_back(扩展.extensionName);
            }
        }

        // 修正：启用所需设备特性
        vk::PhysicalDeviceFeatures 设备特性;
        设备特性.shaderStorageImageWriteWithoutFormat = VK_TRUE;

        vk::DeviceCreateInfo 设备创建信息({}, 1, &队列创建信息, 0, nullptr,
            static_cast<uint32_t>(启用设备扩展.size()), 启用设备扩展.data(),
            &设备特性);
        逻辑设备_ = 物理设备_.createDevice(设备创建信息);
        计算队列_ = 逻辑设备_.getQueue(0, 0);

        // 4. 创建命令池
        vk::CommandPoolCreateInfo 命令池信息(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0);
        命令池_ = 逻辑设备_.createCommandPool(命令池信息);

        // 5. 创建 Uniform 缓冲区
        vk::BufferCreateInfo uniform信息(
            {},  // flags
            sizeof(AdamUniforms),  // size
            vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst  // usage
        );

        uniform缓冲区_ = 逻辑设备_.createBuffer(uniform信息);

        auto uniform内存需求 = 逻辑设备_.getBufferMemoryRequirements(uniform缓冲区_);
        vk::MemoryAllocateInfo uniform分配(uniform内存需求.size, 0);
        uniform内存_ = 逻辑设备_.allocateMemory(uniform分配);
        逻辑设备_.bindBufferMemory(uniform缓冲区_, uniform内存_, 0);

        TORCH_INFO("Vulkan 初始化成功");
    }

    void VulkanAdam优化器::创建计算管线() {
        TORCH_INFO("创建 Vulkan 计算管线...");

        // 1. 编译着色器
        auto SPIRV代码 = 编译GLSL到SPIRV(ADAM_SHADER_SOURCE);

        // 2. 创建着色器模块
        vk::ShaderModuleCreateInfo 着色器模块信息({}, SPIRV代码.size() * sizeof(uint32_t), SPIRV代码.data());
        vk::ShaderModule 着色器模块 = 逻辑设备_.createShaderModule(着色器模块信息);

        // 3. 创建描述符布局
        std::array<vk::DescriptorSetLayoutBinding, 5> 绑定 = {
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute)
        };

        vk::DescriptorSetLayoutCreateInfo 布局信息({}, 绑定.size(), 绑定.data());
        描述符布局_ = 逻辑设备_.createDescriptorSetLayout(布局信息);

        // 4. 创建管线布局
        vk::PipelineLayoutCreateInfo 管线布局信息({}, 1, &描述符布局_);
        管线布局_ = 逻辑设备_.createPipelineLayout(管线布局信息);

        // 5. 创建计算管线
        vk::PipelineShaderStageCreateInfo 着色器阶段({}, vk::ShaderStageFlagBits::eCompute, 着色器模块, "main");
        vk::ComputePipelineCreateInfo 管线信息({}, 着色器阶段, 管线布局_);
        计算管线_ = 逻辑设备_.createComputePipeline(nullptr, 管线信息).value;

        // 6. 创建描述符池
        std::array<vk::DescriptorPoolSize, 2> 池大小 = {
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1000),
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 100)
        };

        vk::DescriptorPoolCreateInfo 池信息({}, 1000, 池大小.size(), 池大小.data());
        描述符池_ = 逻辑设备_.createDescriptorPool(池信息);

        // 7. 清理
        逻辑设备_.destroyShaderModule(着色器模块);

        TORCH_INFO("Vulkan 计算管线创建完成");
    }

    void VulkanAdam优化器::创建参数缓冲(const std::string& 参数名, size_t 元素数量) {
        // 检查是否已创建
        if (参数缓冲映射_.count(参数名)) return;

        size_t 缓冲区大小 = 元素数量 * sizeof(float);

        // 创建四个缓冲区：参数、梯度、动量、二阶动量
        vk::BufferCreateInfo 缓冲信息(
            {},  // flags
            缓冲区大小,  // size
            vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc |
            vk::BufferUsageFlagBits::eTransferDst  // usage
        );

        GPU参数缓冲 缓冲;
        缓冲.参数缓冲 = 逻辑设备_.createBuffer(缓冲信息);
        缓冲.梯度缓冲 = 逻辑设备_.createBuffer(缓冲信息);
        缓冲.动量缓冲 = 逻辑设备_.createBuffer(缓冲信息);
        缓冲.二阶动量缓冲 = 逻辑设备_.createBuffer(缓冲信息);
        缓冲.元素数量 = 元素数量;

        // 分配并绑定内存
        auto 内存需求 = 逻辑设备_.getBufferMemoryRequirements(缓冲.参数缓冲);
        vk::MemoryAllocateInfo 分配信息(内存需求.size * 4, 0);
        缓冲.内存 = 逻辑设备_.allocateMemory(分配信息);

        // 绑定内存
        逻辑设备_.bindBufferMemory(缓冲.参数缓冲, 缓冲.内存, 0);
        逻辑设备_.bindBufferMemory(缓冲.梯度缓冲, 缓冲.内存, 内存需求.size);
        逻辑设备_.bindBufferMemory(缓冲.动量缓冲, 缓冲.内存, 内存需求.size * 2);
        逻辑设备_.bindBufferMemory(缓冲.二阶动量缓冲, 缓冲.内存, 内存需求.size * 3);

        // 初始化动量和二阶动量为零
        float* 映射指针 = static_cast<float*>(逻辑设备_.mapMemory(缓冲.内存,
            内存需求.size * 2,
            内存需求.size * 2));
        memset(映射指针, 0, 内存需求.size * 2);
        逻辑设备_.unmapMemory(缓冲.内存);

        参数缓冲映射_[参数名] = std::move(缓冲);
    }

    void VulkanAdam优化器::上传张量数据(const torch::Tensor& 张量, vk::Buffer 目标缓冲) {
        // 确保张量在CUDA上并连续
        auto gpu张量 = 张量.to(torch::kCUDA).contiguous();
        float* 数据指针 = gpu张量.data_ptr<float>();
        size_t 数据大小 = gpu张量.numel() * sizeof(float);

        // 创建临时上传缓冲区
        vk::BufferCreateInfo 临时信息(
            {},  // flags
            数据大小,  // size
            vk::BufferUsageFlagBits::eTransferSrc  // usage
        );
        vk::Buffer 临时缓冲 = 逻辑设备_.createBuffer(临时信息);

        auto 内存需求 = 逻辑设备_.getBufferMemoryRequirements(临时缓冲);
        vk::MemoryAllocateInfo 分配信息(数据大小, 0);
        vk::DeviceMemory 临时内存 = 逻辑设备_.allocateMemory(分配信息);
        逻辑设备_.bindBufferMemory(临时缓冲, 临时内存, 0);

        // 将数据映射到GPU
        void* 映射 = 逻辑设备_.mapMemory(临时内存, 0, 数据大小);
        memcpy(映射, 数据指针, 数据大小);
        逻辑设备_.unmapMemory(临时内存);

        // 执行复制命令
        vk::CommandBufferAllocateInfo 分配信息命令(命令池_, vk::CommandBufferLevel::ePrimary, 1);
        vk::CommandBuffer 命令缓冲 = 逻辑设备_.allocateCommandBuffers(分配信息命令)[0];

        vk::CommandBufferBeginInfo 开始信息(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        命令缓冲.begin(开始信息);

        vk::BufferCopy 复制区域(0, 0, 数据大小);
        命令缓冲.copyBuffer(临时缓冲, 目标缓冲, 1, &复制区域);

        命令缓冲.end();

        vk::SubmitInfo 提交(0, nullptr, nullptr, 1, &命令缓冲);
        计算队列_.submit(1, &提交, nullptr);
        计算队列_.waitIdle();

        // 清理临时资源
        逻辑设备_.destroyBuffer(临时缓冲);
        逻辑设备_.freeMemory(临时内存);
    }

    void VulkanAdam优化器::下载张量数据(torch::Tensor& 张量, vk::Buffer 源缓冲) {
        size_t 数据大小 = 张量.numel() * sizeof(float);

        // 创建临时下载缓冲区
        vk::BufferCreateInfo 临时信息(
            {},  // flags
            数据大小,  // size
            vk::BufferUsageFlagBits::eTransferDst  // usage
        );
        vk::Buffer 临时缓冲 = 逻辑设备_.createBuffer(临时信息);

        auto 内存需求 = 逻辑设备_.getBufferMemoryRequirements(临时缓冲);
        vk::MemoryAllocateInfo 分配信息(数据大小, 0);
        vk::DeviceMemory 临时内存 = 逻辑设备_.allocateMemory(分配信息);
        逻辑设备_.bindBufferMemory(临时缓冲, 临时内存, 0);

        // 执行复制命令
        vk::CommandBufferAllocateInfo 分配信息命令(命令池_, vk::CommandBufferLevel::ePrimary, 1);
        vk::CommandBuffer 命令缓冲 = 逻辑设备_.allocateCommandBuffers(分配信息命令)[0];

        vk::CommandBufferBeginInfo 开始信息(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        命令缓冲.begin(开始信息);

        vk::BufferCopy 复制区域(0, 0, 数据大小);
        命令缓冲.copyBuffer(源缓冲, 临时缓冲, 1, &复制区域);

        命令缓冲.end();

        vk::SubmitInfo 提交(0, nullptr, nullptr, 1, &命令缓冲);
        计算队列_.submit(1, &提交, nullptr);
        计算队列_.waitIdle();

        // 下载数据
        void* 映射 = 逻辑设备_.mapMemory(临时内存, 0, 数据大小);
        memcpy(张量.data_ptr<float>(), 映射, 数据大小);
        逻辑设备_.unmapMemory(临时内存);

        // 清理临时资源
        逻辑设备_.destroyBuffer(临时缓冲);
        逻辑设备_.freeMemory(临时内存);
    }

    void VulkanAdam优化器::更新Uniform缓冲区() {
        AdamUniforms uniform数据 = {
            学习率_,
            贝塔1_,
            贝塔2_,
            1.0f / (1.0f - std::pow(贝塔1_, 步数_)),
            1.0f / (1.0f - std::pow(贝塔2_, 步数_)),
            埃普西隆_,
            权重衰减_,
            static_cast<int32_t>(步数_),
            0  // 元素数量在每次调用时设置
        };

        // 上传 uniform 数据
        void* 映射 = 逻辑设备_.mapMemory(uniform内存_, 0, sizeof(AdamUniforms));
        memcpy(映射, &uniform数据, sizeof(AdamUniforms));
        逻辑设备_.unmapMemory(uniform内存_);
    }

    void VulkanAdam优化器::执行GPU优化(const std::string& 参数名) {
        auto& 缓冲 = 参数缓冲映射_[参数名];
        uint32_t 工作组数量 = (缓冲.元素数量 + 255) / 256;

        // 更新 uniform 中的元素数量
        更新Uniform缓冲区();
        AdamUniforms* uniform = static_cast<AdamUniforms*>(
            逻辑设备_.mapMemory(uniform内存_, 0, sizeof(AdamUniforms)));
        uniform->元素数量 = 缓冲.元素数量;
        逻辑设备_.unmapMemory(uniform内存_);

        // 分配描述符集
        vk::DescriptorSetAllocateInfo 描述集分配(描述符池_, 1, &描述符布局_);
        vk::DescriptorSet 描述符集 = 逻辑设备_.allocateDescriptorSets(描述集分配)[0];

        // 更新描述符集
        std::array<vk::DescriptorBufferInfo, 5> 缓冲信息 = {
            vk::DescriptorBufferInfo(缓冲.参数缓冲, 0, 缓冲.元素数量 * sizeof(float)),
            vk::DescriptorBufferInfo(缓冲.梯度缓冲, 0, 缓冲.元素数量 * sizeof(float)),
            vk::DescriptorBufferInfo(缓冲.动量缓冲, 0, 缓冲.元素数量 * sizeof(float)),
            vk::DescriptorBufferInfo(缓冲.二阶动量缓冲, 0, 缓冲.元素数量 * sizeof(float)),
            vk::DescriptorBufferInfo(uniform缓冲区_, 0, sizeof(AdamUniforms))
        };

        std::array<vk::WriteDescriptorSet, 5> 描述符写入;
        for (uint32_t i = 0; i < 5; ++i) {
            描述符写入[i] = vk::WriteDescriptorSet(描述符集, i, 0, 1,
                i < 4 ? vk::DescriptorType::eStorageBuffer : vk::DescriptorType::eUniformBuffer,
                nullptr, &缓冲信息[i], nullptr);
        }

        逻辑设备_.updateDescriptorSets(描述符写入.size(), 描述符写入.data(), 0, nullptr);

        // 记录命令
        vk::CommandBufferAllocateInfo 分配信息(命令池_, vk::CommandBufferLevel::ePrimary, 1);
        vk::CommandBuffer 命令缓冲 = 逻辑设备_.allocateCommandBuffers(分配信息)[0];

        vk::CommandBufferBeginInfo 开始信息(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        命令缓冲.begin(开始信息);

        // 绑定管线
        命令缓冲.bindPipeline(vk::PipelineBindPoint::eCompute, 计算管线_);
        命令缓冲.bindDescriptorSets(vk::PipelineBindPoint::eCompute, 管线布局_, 0, 1, &描述符集, 0, nullptr);

        // 调度计算
        命令缓冲.dispatch(工作组数量, 1, 1);

        命令缓冲.end();

        // 提交执行
        vk::SubmitInfo 提交(0, nullptr, nullptr, 1, &命令缓冲);
        计算队列_.submit(1, &提交, nullptr);
        计算队列_.waitIdle();

        // 清理
        逻辑设备_.freeCommandBuffers(命令池_, 1, &命令缓冲);
        逻辑设备_.freeDescriptorSets(描述符池_, 1, &描述符集);
    }


    // SpectraSSM/Vulkan优化器.cpp
    // 位于 VulkanAdam优化器::编译GLSL到SPIRV 函数

    std::vector<uint32_t> VulkanAdam优化器::编译GLSL到SPIRV(const std::string& 着色器代码) {
        TORCH_INFO("开始编译 GLSL 到 SPIR-V...");

        // 方法1: 运行时编译（需要 shaderc）
#ifdef ENABLE_SHADERC_RUNTIME_COMPILATION
        try {
            TORCH_INFO("  使用 shaderc 运行时编译器");
     
            shaderc::Compiler 编译器;
            shaderc::CompileOptions 选项;  // 中文变量名

            // 设置 Vulkan 目标环境
            选项.SetTargetEnvironment(  // 修正：使用正确变量名
                shaderc_target_env_vulkan,
                shaderc_env_version_vulkan_1_2
            );

            // 设置优化级别
            选项.SetOptimizationLevel(shaderc_optimization_level_performance);  

            // 添加预处理宏
            选项.AddMacroDefinition("LOCAL_SIZE_X", "256");  // 修正：使用正确变量名

            TORCH_INFO("  - 目标环境: Vulkan 1.2");
            TORCH_INFO("  - 优化级别: 性能优化");

            // 执行编译
            shaderc::SpvCompilationResult 结果 = 编译器.CompileGlslToSpv(
                着色器代码,
                shaderc_glsl_compute_shader,
                "adam_compute.glsl",  // 源文件名
                选项  // 修正：使用正确变量名
            );

            // 检查编译状态
            auto 编译状态 = 结果.GetCompilationStatus();
            if (编译状态 != shaderc_compilation_status_success) {
                auto 错误信息 = 结果.GetErrorMessage();
                TORCH_ERROR("GLSL 编译失败: ", 错误信息);
                TORCH_WARN("运行时编译失败，尝试加载预编译二进制");
                return 加载预编译SPIRV();
            }

            auto 指令数量 = std::distance(结果.begin(), 结果.end());
            TORCH_INFO("  ✓ 编译成功，输出指令数: ", 指令数量);

            return { 结果.begin(), 结果.end() };
        }
        catch (const std::exception& 异常) {
            TORCH_ERROR("  ✗ shaderc 异常: ", 异常.what());
            TORCH_WARN("运行时编译器异常，降级到预编译二进制");
            return 加载预编译SPIRV();
        }
#else
        TORCH_INFO("  运行时编译已禁用，使用预编译 SPIR-V");
        TORCH_INFO("  如需启用，请定义 ENABLE_SHADERC_RUNTIME_COMPILATION 并链接 shaderc 库");
        return 加载预编译SPIRV();
#endif
    }
    // 预编译 SPIR-V 加载函数
    /**
    * @brief 加载预编译的 SPIR-V 二进制
    * @note 这是生产环境推荐方式，避免运行时编译开销
    */
    std::vector<uint32_t> VulkanAdam优化器::加载预编译SPIRV() {
        // 预编译的 adam_compute.glsl SPIR-V 二进制数据
        // 生成命令: 
        //   Windows: %VULKAN_SDK%\Bin\glslc.exe -fshader-stage=compute adam_compute.glsl -o adam.spv
        //   Linux:  $VULKAN_SDK/bin/glslc -fshader-stage=compute adam_compute.glsl -o adam.spv

        // 方式A：从嵌入资源加载（推荐）
        // 将 .spv 文件作为二进制资源嵌入可执行文件
#ifdef PRECOMPILED_SPIRV_EMBEDDED
        extern const unsigned char g_adam_compute_spv[];
        extern const size_t g_adam_compute_spv_size;

        TORCH_INFO("  从嵌入资源加载预编译 SPIR-V");
        TORCH_INFO("  数据大小: ", g_adam_compute_spv_size, " 字节");

        return {
            reinterpret_cast<const uint32_t*>(g_adam_compute_spv),
            reinterpret_cast<const uint32_t*>(g_adam_compute_spv + g_adam_compute_spv_size)
        };
#endif

        // 方式B：从文件加载（开发环境）
#ifdef PRECOMPILED_SPIRV_PATH
        std::string 文件路径 = PRECOMPILED_SPIRV_PATH;
        TORCH_INFO("  从文件加载预编译 SPIR-V: ", 文件路径);

        return 从文件加载SPIRV(文件路径);
#endif

        // 方式C：硬编码（备用）
        TORCH_WARN("  未找到预编译 SPIR-V 配置，使用硬编码空实现");
        TORCH_WARN("  在 VulkanAdam优化器::执行GPU优化 中将触发错误");

        // 空实现，实际必须在构建时提供有效 SPIR-V
        return {};
    }
    /**
     * @brief 从文件系统加载 SPIR-V 二进制
     * @param 文件路径 SPIR-V 文件路径
     * @return SPIR-V 指令向量
     */
    std::vector<uint32_t> VulkanAdam优化器::从文件加载SPIRV(const std::string& 文件路径) {
        if (!std::filesystem::exists(文件路径)) {
            TORCH_ERROR("  ✗ SPIR-V 文件不存在: ", 文件路径);
            TORCH_ERROR("    请确保已使用 glslc 编译着色器");
            TORCH_ERROR("    Windows: %VULKAN_SDK%\\Bin\\glslc.exe -fshader-stage=compute your_shader.glsl -o ", 文件路径);
            TORCH_ERROR("    Linux:  $VULKAN_SDK/bin/glslc -fshader-stage=compute your_shader.glsl -o ", 文件路径);

            // 致命错误
            TORCH_ERROR("无法继续执行：没有有效的 SPIR-V 着色器");
            throw std::runtime_error("未找到 SPIR-V 着色器: " + 文件路径);
        }

        std::ifstream 文件(文件路径, std::ios::binary | std::ios::ate);

        if (!文件.is_open()) {
            TORCH_ERROR("  ✗ 无法打开 SPIR-V 文件: ", 文件路径);
            throw std::runtime_error("无法打开 SPIR-V 文件: " + 文件路径);
        }

        size_t 文件大小 = 文件.tellg();
        文件.seekg(0, std::ios::beg);

        TORCH_INFO("  文件大小: ", 文件大小, " 字节");

        // 验证文件大小是 4 的倍数
        if (文件大小 % sizeof(uint32_t) != 0) {
            TORCH_ERROR("  ✗ 文件大小不是 4 的倍数，可能不是有效的 SPIR-V");
            文件.close();
            throw std::runtime_error("无效的 SPIR-V 文件大小");
        }

        std::vector<uint32_t> SPIRV二进制(文件大小 / sizeof(uint32_t));
        文件.read(reinterpret_cast<char*>(SPIRV二进制.data()), 文件大小);
        文件.close();

        // 验证 SPIR-V 魔数
        if (!SPIRV二进制.empty() && SPIRV二进制[0] != 0x07230203) {
            TORCH_ERROR("  ✗ 无效的 SPIR-V 魔数，文件可能损坏");
            throw std::runtime_error("无效的 SPIR-V 文件");
        }

        TORCH_INFO("  ✓ 加载成功，指令数: ", SPIRV二进制.size());

        return SPIRV二进制;
    }
    void VulkanAdam优化器::设置学习率(float 新学习率) {
        if (新学习率 <= 0.0f) {
            TORCH_WARN("学习率必须大于0，使用当前值: ", 学习率_);
            return;
        }
        学习率_ = 新学习率;
        TORCH_INFO("Vulkan Adam 学习率已更新: ", 学习率_);
    }

    void VulkanAdam优化器::优化参数(
        const std::vector<std::string>& 参数名列表,
        const std::vector<torch::Tensor>& 参数列表,
        const std::vector<torch::Tensor>& 梯度列表) {

        TORCH_CHECK(参数名列表.size() == 参数列表.size(),
            "参数名列表与参数列表大小不匹配");
        TORCH_CHECK(参数列表.size() == 梯度列表.size(),
            "参数列表与梯度列表大小不匹配");

        if (参数名列表.empty()) {
            TORCH_WARN("参数列表为空，跳过优化");
            return;
        }

        TORCH_INFO("开始 Vulkan GPU 参数优化，参数数量: ", 参数名列表.size());

        for (size_t i = 0; i < 参数名列表.size(); ++i) {
            const auto& 参数名 = 参数名列表[i];
            auto 参数 = 参数列表[i];
            auto 梯度 = 梯度列表[i];

            if (!梯度.defined() || 梯度.numel() == 0) {
                TORCH_WARN("参数 '", 参数名, "' 梯度未定义，跳过");
                continue;
            }

            size_t 元素数量 = 参数.numel();

            // 创建参数缓冲区
            创建参数缓冲(参数名, 元素数量);

            // 上传数据
            TORCH_INFO("  - 上传参数: ", 参数名, " (", 元素数量, " 个元素)");
            上传张量数据(参数, 参数缓冲映射_[参数名].参数缓冲);
            上传张量数据(梯度, 参数缓冲映射_[参数名].梯度缓冲);

            // 执行 GPU 优化
            执行GPU优化(参数名);

            // 下载结果
            下载张量数据(参数, 参数缓冲映射_[参数名].参数缓冲);
        }

        步数_++;

        TORCH_INFO("Vulkan GPU 参数优化完成 (步数: ", 步数_, ")");
    }

} // namespace SpectraSSM