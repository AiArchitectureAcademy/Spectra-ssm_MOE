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
        // 防止重复创建
        if (参数缓冲映射_.find(参数名) != 参数缓冲映射_.end()) {
            TORCH_WARN("参数缓冲 '", 参数名, "' 已存在，跳过创建");
            return;
        }

        // 计算缓冲区大小（4个缓冲：参数、梯度、动量、二阶动量）
        const vk::DeviceSize 缓冲区大小 = 元素数量 * sizeof(float);
        const vk::DeviceSize 总内存需求 = 缓冲区大小 * 4ULL;

        // ⚠️ 核显内存预算检查（780M实际可用约400-450MB）
        if (当前内存使用量_ + 总内存需求 > 最大显存预算_) {
            TORCH_ERROR("核显内存不足！");
            TORCH_ERROR("  - 当前使用量: ", 当前内存使用量_ / 1024 / 1024, " MB");
            TORCH_ERROR("  - 需要分配: ", 总内存需求 / 1024 / 1024, " MB");
            TORCH_ERROR("  - 剩余预算: ", (最大显存预算_ - 当前内存使用量_) / 1024 / 1024, " MB");
            TORCH_ERROR("  - 参数名: ", 参数名);

            // 提供优化建议
            if (元素数量 > 10 * 1024 * 1024) {
                TORCH_WARN("建议：该参数规模过大（", 元素数量 / 1024 / 1024, "M元素），");
                TORCH_WARN("      请使用梯度检查点或减小批大小");
            }

            throw std::runtime_error("核显内存溢出：参数 '" + 参数名 + "' 分配失败");
        }

        // 创建四个缓冲区对象（共享同一块内存）
        vk::BufferCreateInfo 缓冲创建信息(
            vk::BufferCreateFlags(),  // flags
            缓冲区大小,              // size
            vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc |
            vk::BufferUsageFlagBits::eTransferDst  // usage
        );

        GPU参数缓冲 新缓冲;
        新缓冲.参数缓冲 = 逻辑设备_.createBuffer(缓冲创建信息);
        新缓冲.梯度缓冲 = 逻辑设备_.createBuffer(缓冲创建信息);
        新缓冲.动量缓冲 = 逻辑设备_.createBuffer(缓冲创建信息);
        新缓冲.二阶动量缓冲 = 逻辑设备_.createBuffer(缓冲创建信息);
        新缓冲.元素数量 = 元素数量;

        // 获取内存需求（4个缓冲大小相同，只需查询一次）
        auto 内存需求 = 逻辑设备_.getBufferMemoryRequirements(新缓冲.参数缓冲);

        // 智能选择内存类型（核显用零拷贝内存）
        uint32_t 内存类型索引 = 0;
        if (使用零拷贝模式_) {
            // 核显：查找HOST_VISIBLE | HOST_COHERENT内存（通常为内存类型0）
            auto 内存属性 = 物理设备_.getMemoryProperties();
            bool 找到合适类型 = false;

            for (uint32_t i = 0; i < 内存属性.memoryTypeCount; ++i) {
                // 检查该内存类型是否支持此缓冲
                if ((内存需求.memoryTypeBits & (1 << i)) == 0) continue;

                auto 标志 = 内存属性.memoryTypes[i].propertyFlags;
                bool 是主机可见 = (标志 & vk::MemoryPropertyFlagBits::eHostVisible) != vk::MemoryPropertyFlags();
                bool 是主机一致 = (标志 & vk::MemoryPropertyFlagBits::eHostCoherent) != vk::MemoryPropertyFlags();

                // 核显最优：HOST_VISIBLE | HOST_COHERENT
                if (是主机可见 && 是主机一致) {
                    内存类型索引 = i;
                    找到合适类型 = true;

                    TORCH_INFO("核显内存：选中类型[", i, "] HOST_VISIBLE|HOST_COHERENT");
                    break;
                }
            }

            if (!找到合适类型) {
                // 降级策略：只要HOST_VISIBLE
                for (uint32_t i = 0; i < 内存属性.memoryTypeCount; ++i) {
                    if ((内存需求.memoryTypeBits & (1 << i)) &&
                        (内存属性.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {
                        内存类型索引 = i;
                        TORCH_WARN("核显内存：仅找到HOST_VISIBLE类型[", i, "]，性能可能下降");
                        break;
                    }
                }
            }
        }
        else {
            // 独显：使用DEVICE_LOCAL内存（显存）
            auto 内存属性 = 物理设备_.getMemoryProperties();
            for (uint32_t i = 0; i < 内存属性.memoryTypeCount; ++i) {
                if ((内存需求.memoryTypeBits & (1 << i)) &&
                    (内存属性.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
                    内存类型索引 = i;
                    break;
                }
            }
        }

        if (内存类型索引 == UINT32_MAX) {
            throw std::runtime_error("未找到合适的Vulkan内存类型");
        }

        // ✅ 一次性分配所有缓冲所需的内存
        vk::MemoryAllocateInfo 内存分配信息(内存需求.size * 4, 内存类型索引);
        新缓冲.内存 = 逻辑设备_.allocateMemory(内存分配信息);

        // 将四个缓冲区绑定到同一块内存的不同偏移位置
        逻辑设备_.bindBufferMemory(新缓冲.参数缓冲, 新缓冲.内存, 0);
        逻辑设备_.bindBufferMemory(新缓冲.梯度缓冲, 新缓冲.内存, 内存需求.size);
        逻辑设备_.bindBufferMemory(新缓冲.动量缓冲, 新缓冲.内存, 内存需求.size * 2);
        逻辑设备_.bindBufferMemory(新缓冲.二阶动量缓冲, 新缓冲.内存, 内存需求.size * 3);

        // ✅ 核显特殊处理：直接CPU映射初始化动量缓冲为0
        if (使用零拷贝模式_) {
            // 映射动量和二阶动量区域（偏移2*size，大小2*size）
            void* 映射指针 = 逻辑设备_.mapMemory(
                新缓冲.内存,
                内存需求.size * 2,  // 偏移量
                内存需求.size * 2   // 映射大小（两个缓冲）
            );

            if (映射指针) {
                memset(映射指针, 0, 内存需求.size * 2);  // 动量和二阶动量初始化为0
                逻辑设备_.unmapMemory(新缓冲.内存);
            }
            else {
                TORCH_ERROR("核显内存映射失败，无法初始化动量缓冲");
            }
        }
        else {
            // 独显：通过命令缓冲区上传初始化数据
            // （保持原有实现）
        }

        // 存入映射表
        参数缓冲映射_[参数名] = std::move(新缓冲);
        当前内存使用量_ += 总内存需求;

        // 📊 详细日志输出
        TORCH_INFO("✓ 创建参数缓冲: ", 参数名);
        TORCH_INFO("  - 元素数量: ", 元素数量, " (", 元素数量 / 1024.0 / 1024.0, "M)");
        TORCH_INFO("  - 单个缓冲: ", 缓冲区大小 / 1024, " KB");
        TORCH_INFO("  - 总内存: ", 总内存需求 / 1024 / 1024, " MB");
        TORCH_INFO("  - 内存类型索引: ", 内存类型索引);
        TORCH_INFO("  - 零拷贝模式: ", 使用零拷贝模式_ ? "是" : "否");
        TORCH_INFO("  - 累计显存使用: ", 当前内存使用量_ / 1024 / 1024, " MB / ",
            最大显存预算_ / 1024 / 1024, " MB");
    }

    void VulkanAdam优化器::上传张量数据(const torch::Tensor& 张量, vk::Buffer 目标缓冲) {
        TORCH_CHECK(张量.defined(), "输入张量未定义");
        TORCH_CHECK(张量.is_contiguous(), "张量必须连续");
        TORCH_CHECK(张量.dtype() == torch::kFloat32, "仅支持float32类型");

        const size_t 元素数量 = 张量.numel();
        const vk::DeviceSize 数据大小 = 元素数量 * sizeof(float);

        if (数据大小 == 0) {
            TORCH_WARN("上传数据大小为0，跳过");
            return;
        }

        // 获取张量数据指针
        // 核显模式：张量通常在CPU内存，可直接访问
        // 独显模式：需要确保数据在CUDA上
        const float* 张量数据指针 = 张量.data_ptr<float>();

        // 获取目标缓冲关联的内存
        vk::DeviceMemory 目标内存 = 获取缓冲关联内存(目标缓冲);
        TORCH_CHECK(目标内存, "未找到目标缓冲关联的内存");

        // 获取缓冲内存需求（验证大小）
        auto 内存需求 = 逻辑设备_.getBufferMemoryRequirements(目标缓冲);
        if (数据大小 > 内存需求.size) {
            TORCH_ERROR("数据大小 ", 数据大小, " 超过缓冲大小 ", 内存需求.size);
            throw std::runtime_error("Vulkan上传：数据溢出");
        }

        // ⭐ 核显零拷贝路径：直接内存映射
        if (使用零拷贝模式_) {
            TORCH_INFO("核显零拷贝上传: ", 数据大小 / 1024.0, " KB");

            // 映射Vulkan缓冲内存到CPU地址空间
            void* 映射内存 = 逻辑设备_.mapMemory(目标内存, 0, 数据大小);
            if (!映射内存) {
                TORCH_ERROR("核显内存映射失败");
                throw std::runtime_error("Vulkan映射内存失败");
            }

            // 直接内存拷贝（PyTorch CPU -> Vulkan共享内存）
            std::memcpy(映射内存, 张量数据指针, 数据大小);

            // ✅ 关键：核显需要flush确保GPU可见性
            // 即使HOST_COHERENT也建议显式flush保证跨设备一致性
            vk::MappedMemoryRange 刷新范围(目标内存, 0, 数据大小);
            vk::Result 结果 = 逻辑设备_.flushMappedMemoryRanges(1, &刷新范围);

            if (结果 != vk::Result::eSuccess) {
                TORCH_WARN("flushMappedMemoryRanges失败: ", vk::to_string(结果));
            }

            逻辑设备_.unmapMemory(目标内存);

            // 性能统计
            static std::atomic<uint64_t> 总上传量{ 0 };
            总上传量 += 数据大小;
            TORCH_INFO("  - 累计上传: ", 总上传量.load() / 1024 / 1024, " MB");
        }
        // 独显传统路径：使用临时缓冲和DMA
        else {
            TORCH_INFO("独显DMA上传: ", 数据大小 / 1024.0, " KB");

            // 确保张量在CUDA设备上
            auto cuda张量 = 张量.to(torch::kCUDA).contiguous();
            const float* cuda数据指针 = cuda张量.data_ptr<float>();

            // 创建临时上传缓冲（Staging Buffer）
            vk::BufferCreateInfo 临时缓冲信息(
                vk::BufferCreateFlags(),
                数据大小,
                vk::BufferUsageFlagBits::eTransferSrc  // 仅用于传输源
            );
            vk::Buffer 临时缓冲 = 逻辑设备_.createBuffer(临时缓冲信息);

            auto 临时内存需求 = 逻辑设备_.getBufferMemoryRequirements(临时缓冲);

            // 临时缓冲使用HOST_VISIBLE内存（CPU可写）
            uint32_t 临时内存类型 = 查找零拷贝内存类型(临时内存需求);
            vk::MemoryAllocateInfo 临时分配信息(数据大小, 临时内存类型);
            vk::DeviceMemory 临时内存 = 逻辑设备_.allocateMemory(临时分配信息);
            逻辑设备_.bindBufferMemory(临时缓冲, 临时内存, 0);

            // 1. 将数据写入临时缓冲
            void* 映射内存 = 逻辑设备_.mapMemory(临时内存, 0, 数据大小);
            std::memcpy(映射内存, cuda数据指针, 数据大小);  // CUDA -> CPU -> Vulkan
            逻辑设备_.unmapMemory(临时内存);

            // 2. 执行DMA传输（CPU -> GPU）
            vk::CommandBuffer 命令缓冲 = 开始单次命令();

            vk::BufferCopy 拷贝区域(0, 0, 数据大小);
            命令缓冲.copyBuffer(临时缓冲, 目标缓冲, 1, &拷贝区域);

            结束并提交命令(命令缓冲);

            // 3. 清理临时资源
            逻辑设备_.destroyBuffer(临时缓冲);
            逻辑设备_.freeMemory(临时内存);
        }
    }
    // 辅助函数：单次命令缓冲工具
    vk::CommandBuffer VulkanAdam优化器::开始单次命令() {
        vk::CommandBufferAllocateInfo 分配信息(命令池_, vk::CommandBufferLevel::ePrimary, 1);
        vk::CommandBuffer 命令缓冲 = 逻辑设备_.allocateCommandBuffers(分配信息)[0];

        vk::CommandBufferBeginInfo 开始信息(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        命令缓冲.begin(开始信息);

        return 命令缓冲;
    }
    void VulkanAdam优化器::结束并提交命令(vk::CommandBuffer 命令缓冲) {
        命令缓冲.end();

        vk::SubmitInfo 提交信息(0, nullptr, nullptr, 1, &命令缓冲);
        计算队列_.submit(1, &提交信息, nullptr);
        计算队列_.waitIdle();  // 单次命令通常同步等待

        逻辑设备_.freeCommandBuffers(命令池_, 1, &命令缓冲);
    }
    // 辅助函数：获取缓冲对应的内存对象
    vk::DeviceMemory VulkanAdam优化器::获取缓冲关联内存(vk::Buffer 缓冲) {
        for (const auto& [名称, 缓冲信息] : 参数缓冲映射_) {
            if (缓冲信息.参数缓冲 == 缓冲 ||
                缓冲信息.梯度缓冲 == 缓冲 ||
                缓冲信息.动量缓冲 == 缓冲 ||
                缓冲信息.二阶动量缓冲 == 缓冲) {
                return 缓冲信息.内存;
            }
        }

        // 尝试Uniform缓冲
        if (缓冲 == uniform缓冲区_) {
            return uniform内存_;
        }

        TORCH_ERROR("未找到缓冲关联的内存");
        return vk::DeviceMemory();  // 返回空句柄
    }

    // 辅助函数：查找零拷贝内存类型
    uint32_t VulkanAdam优化器::查找零拷贝内存类型(const vk::MemoryRequirements& 需求) {
        auto 内存属性 = 物理设备_.getMemoryProperties();

        // 优先级1: HOST_VISIBLE | HOST_COHERENT（零拷贝首选）
        // 优先级2: HOST_VISIBLE（可能需要手动flush）
        for (uint32_t i = 0; i < 内存属性.memoryTypeCount; ++i) {
            if ((需求.memoryTypeBits & (1 << i)) &&
                (内存属性.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {

                // 记录选择的内存类型
                vk::MemoryPropertyFlags 标志 = 内存属性.memoryTypes[i].propertyFlags;
                TORCH_INFO("核显内存类型[", i, "]: ",
                    (标志 & vk::MemoryPropertyFlagBits::eHostVisible ? "HOST_VISIBLE " : ""),
                    (标志 & vk::MemoryPropertyFlagBits::eHostCoherent ? "HOST_COHERENT " : ""),
                    (标志 & vk::MemoryPropertyFlagBits::eDeviceLocal ? "DEVICE_LOCAL" : ""));

                return i;
            }
        }

        TORCH_ERROR("未找到合适的核显内存类型");
        throw std::runtime_error("核显内存类型检测失败");
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