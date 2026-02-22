/**
 * EvoVadSDK - Evo VAD Engine SDK
 *
 * 语音活动检测引擎适配层，提供统一的 C++ 接口。
 *
 * 使用示例 1 - 单帧检测（阻塞模式）:
 *
 *   auto engine = std::make_shared<Evo::VadEngine>();
 *   auto result = engine->Detect(audio_frame);
 *   if (result && result->IsSpeech()) {
 *       std::cout << "语音概率: " << result->GetProbability() << std::endl;
 *   }
 *
 * 使用示例 2 - 带配置的初始化:
 *
 *   auto config = Evo::VadConfig::Silero()
 *       .withTriggerThreshold(0.6f)
 *       .withStopThreshold(0.35f);
 *   auto engine = std::make_shared<Evo::VadEngine>(config);
 *
 * 使用示例 3 - 流式检测:
 *
 *   auto engine = std::make_shared<Evo::VadEngine>();
 *   engine->SetCallback(std::make_shared<MyCallback>());
 *   engine->Start();
 *   while (recording) {
 *       engine->SendAudioFrame(audio_data);
 *   }
 *   engine->Stop();
 */

#ifndef VAD_API_HPP
#define VAD_API_HPP

#include <cstdint>

#include <functional>
#include <memory>
#include <string>
#include <vector>

// Forward declaration of internal types
namespace vad {
    class IVadBackend;
    struct DetectionResult;
    struct ErrorInfo;
}

namespace Evo {

// =============================================================================
// VadState - VAD 状态
// =============================================================================

enum class VadState {
    SILENCE,        ///< 静音状态
    SPEECH_START,   ///< 语音开始（从静音转为语音）
    SPEECH,         ///< 语音进行中
    SPEECH_END,     ///< 语音结束（从语音转为静音）
};

// =============================================================================
// VadBackendType - 后端类型
// =============================================================================

enum class VadBackendType {
    SILERO,         ///< Silero VAD (ONNX, 深度学习，推荐)
    ENERGY,         ///< 基于能量的 VAD (轻量级)
    WEBRTC,         ///< WebRTC VAD (预留)
    FSMN,           ///< 阿里 FSMN VAD (预留)
    CUSTOM,         ///< 自定义后端
};

// =============================================================================
// VadConfig - VAD 配置
// =============================================================================

/**
 * @brief VAD 引擎配置
 */
struct VadConfig {
    // -------------------------------------------------------------------------
    // 后端选择
    // -------------------------------------------------------------------------

    VadBackendType backend = VadBackendType::SILERO;  ///< 后端类型

    // -------------------------------------------------------------------------
    // 模型配置
    // -------------------------------------------------------------------------

    std::string model_dir;              ///< 模型目录路径，空则使用默认路径

    // -------------------------------------------------------------------------
    // 音频参数
    // -------------------------------------------------------------------------

    int sample_rate = 16000;            ///< 采样率 (Hz)
    int window_size = 512;              ///< 窗口大小 (样本数)

    // -------------------------------------------------------------------------
    // 检测参数
    // -------------------------------------------------------------------------

    float trigger_threshold = 0.5f;     ///< 语音开始阈值 [0.0, 1.0]
    float stop_threshold = 0.35f;       ///< 语音结束阈值 [0.0, 1.0]
    int min_speech_duration_ms = 250;   ///< 最小语音持续时间 (ms)
    int min_silence_duration_ms = 100;  ///< 最小静音持续时间 (ms)

    // -------------------------------------------------------------------------
    // 平滑参数
    // -------------------------------------------------------------------------

    int smoothing_window = 10;          ///< 平滑窗口大小
    bool use_smoothing = true;          ///< 是否使用概率平滑

    // -------------------------------------------------------------------------
    // 性能配置
    // -------------------------------------------------------------------------

    int num_threads = 1;                ///< 推理线程数

    // -------------------------------------------------------------------------
    // 便捷构建方法
    // -------------------------------------------------------------------------

    /// @brief 创建默认配置（Silero）
    static VadConfig Default() {
        return VadConfig();
    }

    /// @brief 创建 Silero VAD 配置
    /// @param model_dir 模型目录路径
    static VadConfig Silero(const std::string& model_dir = "~/.cache/silero_vad") {
        VadConfig config;
        config.backend = VadBackendType::SILERO;
        config.model_dir = model_dir;
        return config;
    }

    /// @brief 创建 Energy VAD 配置（轻量级）
    static VadConfig Energy() {
        VadConfig config;
        config.backend = VadBackendType::ENERGY;
        return config;
    }

    // -------------------------------------------------------------------------
    // 链式配置
    // -------------------------------------------------------------------------

    /// @brief 设置语音开始阈值
    VadConfig withTriggerThreshold(float threshold) const {
        auto c = *this;
        c.trigger_threshold = threshold;
        return c;
    }

    /// @brief 设置语音结束阈值
    VadConfig withStopThreshold(float threshold) const {
        auto c = *this;
        c.stop_threshold = threshold;
        return c;
    }

    /// @brief 设置平滑窗口大小
    VadConfig withSmoothingWindow(int window) const {
        auto c = *this;
        c.smoothing_window = window;
        return c;
    }

    /// @brief 设置最小语音持续时间
    VadConfig withMinSpeechDuration(int ms) const {
        auto c = *this;
        c.min_speech_duration_ms = ms;
        return c;
    }

    /// @brief 设置最小静音持续时间
    VadConfig withMinSilenceDuration(int ms) const {
        auto c = *this;
        c.min_silence_duration_ms = ms;
        return c;
    }

    /// @brief 设置窗口大小
    VadConfig withWindowSize(int size) const {
        auto c = *this;
        c.window_size = size;
        return c;
    }

    /// @brief 设置采样率
    VadConfig withSampleRate(int rate) const {
        auto c = *this;
        c.sample_rate = rate;
        return c;
    }

    /// @brief 是否启用平滑
    VadConfig withSmoothing(bool enable) const {
        auto c = *this;
        c.use_smoothing = enable;
        return c;
    }

    /// @brief 设置推理线程数
    VadConfig withNumThreads(int threads) const {
        auto c = *this;
        c.num_threads = threads;
        return c;
    }
};

// =============================================================================
// VadResult - 检测结果
// =============================================================================

/**
 * @brief VAD 检测结果
 *
 * 封装了语音活动检测的结果，包括概率、状态和时间信息。
 */
class VadResult {
public:
    VadResult();
    ~VadResult();

    // 禁止拷贝，允许移动
    VadResult(const VadResult&) = delete;
    VadResult& operator=(const VadResult&) = delete;
    VadResult(VadResult&&) noexcept;
    VadResult& operator=(VadResult&&) noexcept;

    // -------------------------------------------------------------------------
    // 概率信息
    // -------------------------------------------------------------------------

    /// @brief 获取原始语音概率
    /// @return 概率值 [0.0, 1.0]
    float GetProbability() const;

    /// @brief 获取平滑后的语音概率
    /// @return 概率值 [0.0, 1.0]
    float GetSmoothedProbability() const;

    // -------------------------------------------------------------------------
    // 状态信息
    // -------------------------------------------------------------------------

    /// @brief 是否检测到语音
    /// @return true 表示当前为语音
    bool IsSpeech() const;

    /// @brief 获取当前 VAD 状态
    /// @return VAD 状态枚举
    VadState GetState() const;

    /// @brief 是否为语音开始事件
    /// @return true 表示刚检测到语音开始
    bool IsSpeechStart() const;

    /// @brief 是否为语音结束事件
    /// @return true 表示刚检测到语音结束
    bool IsSpeechEnd() const;

    // -------------------------------------------------------------------------
    // 时间信息
    // -------------------------------------------------------------------------

    /// @brief 获取帧时间戳
    /// @return 时间戳（毫秒）
    int64_t GetTimestampMs() const;

    /// @brief 获取语音段开始时间
    /// @return 时间戳（毫秒），-1 表示无语音段
    int64_t GetSpeechStartMs() const;

    /// @brief 获取语音段结束时间
    /// @return 时间戳（毫秒），-1 表示语音进行中
    int64_t GetSpeechEndMs() const;

    /// @brief 获取语音段持续时间
    /// @return 持续时间（毫秒）
    int GetSpeechDurationMs() const;

    // -------------------------------------------------------------------------
    // 性能信息
    // -------------------------------------------------------------------------

    /// @brief 获取处理时间
    /// @return 处理耗时（毫秒）
    int GetProcessingTimeMs() const;

    // -------------------------------------------------------------------------
    // 状态检查
    // -------------------------------------------------------------------------

    /// @brief 检测是否成功
    /// @return true 表示成功
    bool IsSuccess() const;

    /// @brief 获取错误码
    /// @return 错误码字符串
    std::string GetCode() const;

    /// @brief 获取错误信息
    /// @return 错误描述
    std::string GetMessage() const;

private:
    friend class VadEngine;
    friend class CallbackAdapter;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// VadEngineCallback - 回调接口
// =============================================================================

/**
 * @brief VAD 引擎回调接口（多态基类）
 *
 * 用户可继承此类并重写虚函数，以接收流式检测过程中的事件通知。
 *
 * ## 回调调用链（调用顺序）
 *
 * ```
 *   Start()
 *      │
 *      ▼
 *   OnOpen()              ← 检测会话开始
 *      │
 *      ▼
 *   [检测进行中] ────────► OnEvent()  ← 每帧触发
 *      │                     │
 *      │               OnSpeechStart()  ← 语音开始
 *      │                     │
 *      │               OnSpeechEnd()    ← 语音结束
 *      ▼
 *   Stop()
 *      │
 *      ▼
 *   OnComplete()          ← 检测正常完成
 *      │
 *      ▼
 *   OnClose()             ← 会话关闭
 * ```
 *
 * ## 使用示例
 *
 * ```cpp
 * class MyCallback : public VadEngineCallback {
 * public:
 *     void OnSpeechStart(int64_t timestamp_ms) override {
 *         std::cout << "语音开始: " << timestamp_ms << "ms" << std::endl;
 *     }
 *
 *     void OnSpeechEnd(int64_t timestamp_ms, int duration_ms) override {
 *         std::cout << "语音结束: " << timestamp_ms << "ms"
 *                   << " (持续 " << duration_ms << "ms)" << std::endl;
 *     }
 *
 *     void OnEvent(std::shared_ptr<VadResult> result) override {
 *         std::cout << "概率: " << result->GetProbability() << std::endl;
 *     }
 *
 *     void OnError(const std::string& message) override {
 *         std::cerr << "错误: " << message << std::endl;
 *     }
 * };
 * ```
 */
class VadEngineCallback {
public:
    virtual ~VadEngineCallback() = default;

    /// @brief 检测会话开始
    virtual void OnOpen() {}

    /// @brief 收到检测结果
    /// @param result 检测结果对象
    virtual void OnEvent(std::shared_ptr<VadResult> result) {}

    /// @brief 检测到语音开始
    /// @param timestamp_ms 语音开始的时间戳（毫秒）
    virtual void OnSpeechStart(int64_t timestamp_ms) {}

    /// @brief 检测到语音结束
    /// @param timestamp_ms 语音结束的时间戳（毫秒）
    /// @param duration_ms 语音段持续时间（毫秒）
    virtual void OnSpeechEnd(int64_t timestamp_ms, int duration_ms) {}

    /// @brief 检测任务正常完成
    virtual void OnComplete() {}

    /// @brief 发生错误
    /// @param message 错误描述
    virtual void OnError(const std::string& message) {}

    /// @brief 会话关闭
    /// @note 无论正常结束还是错误，最后都会调用此方法
    virtual void OnClose() {}
};

// =============================================================================
// VadEngine - VAD 引擎
// =============================================================================

/**
 * @brief VAD 引擎
 *
 * 语音活动检测引擎，支持单帧检测和流式检测两种模式。
 */
class VadEngine {
public:
    // =========================================================================
    // 构造函数
    // =========================================================================

    /// @brief 构造 VAD 引擎（使用默认配置）
    /// @param backend 后端类型
    /// @param model_dir 模型目录路径，空则使用默认路径
    explicit VadEngine(VadBackendType backend = VadBackendType::SILERO,
                        const std::string& model_dir = "");

    /// @brief 构造 VAD 引擎（使用配置结构体）
    /// @param config 配置对象
    explicit VadEngine(const VadConfig& config);

    /// @brief 析构函数
    virtual ~VadEngine();

    // 禁止拷贝
    VadEngine(const VadEngine&) = delete;
    VadEngine& operator=(const VadEngine&) = delete;

    // =========================================================================
    // 非流式调用（阻塞）
    // =========================================================================

    /// @brief 检测单帧音频（阻塞直到完成）
    /// @param audio 音频数据 (float, [-1.0, 1.0])
    /// @param sample_rate 采样率
    /// @return 检测结果，失败返回 nullptr
    std::shared_ptr<VadResult> Detect(const std::vector<float>& audio,
                                        int sample_rate = 16000);

    /// @brief 检测单帧音频（原始指针版本）
    /// @param data 音频数据指针
    /// @param num_samples 样本数
    /// @param sample_rate 采样率
    /// @return 检测结果，失败返回 nullptr
    std::shared_ptr<VadResult> Detect(const float* data, size_t num_samples,
                                        int sample_rate = 16000);

    // =========================================================================
    // 流式调用
    // =========================================================================

    /// @brief 设置回调
    /// @param callback 回调对象
    void SetCallback(std::shared_ptr<VadEngineCallback> callback);

    /// @brief 开始流式检测
    /// @return 是否成功
    bool Start();

    /// @brief 发送音频帧
    /// @param data 音频数据 (float, [-1.0, 1.0])
    void SendAudioFrame(const std::vector<float>& data);

    /// @brief 发送音频帧（原始指针版本）
    /// @param data 音频数据指针
    /// @param num_samples 样本数
    void SendAudioFrame(const float* data, size_t num_samples);

    /// @brief 停止流式检测
    void Stop();

    // =========================================================================
    // 状态管理
    // =========================================================================

    /// @brief 重置引擎状态
    void Reset();

    /// @brief 获取当前 VAD 状态
    /// @return VAD 状态枚举
    VadState GetCurrentState() const;

    /// @brief 是否处于语音状态
    /// @return true 表示当前正在语音中
    bool IsInSpeech() const;

    /// @brief 检查引擎是否已初始化
    /// @return true 表示已初始化
    bool IsInitialized() const;

    /// @brief 检查是否正在进行流式检测
    /// @return true 表示正在流式检测
    bool IsStreaming() const;

    // =========================================================================
    // 动态配置
    // =========================================================================

    /// @brief 设置语音开始阈值
    /// @param threshold 阈值 [0.0, 1.0]
    void SetTriggerThreshold(float threshold);

    /// @brief 设置语音结束阈值
    /// @param threshold 阈值 [0.0, 1.0]
    void SetStopThreshold(float threshold);

    /// @brief 获取当前配置
    /// @return 配置对象
    VadConfig GetConfig() const;

    // =========================================================================
    // 辅助方法
    // =========================================================================

    /// @brief 获取引擎名称
    /// @return 引擎名称
    std::string GetEngineName() const;

    /// @brief 获取后端类型
    /// @return 后端类型枚举
    VadBackendType GetBackendType() const;

    /// @brief 获取最后一次的语音概率
    /// @return 概率值 [0.0, 1.0]
    float GetLastProbability() const;

private:
    friend class CallbackAdapter;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace Evo

#endif  // VAD_API_HPP
