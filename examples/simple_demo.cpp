/*
 * VAD Simple Demo
 *
 * 演示如何使用 EvoVadSDK 进行语音活动检测。
 */

#include <cmath>
#include <cstdlib>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "vad_api.hpp"

// 生成测试音频（正弦波模拟语音，静音模拟无语音）
std::vector<float> generateTestAudio(int sample_rate, float duration_sec, bool is_speech) {
    size_t num_samples = static_cast<size_t>(sample_rate * duration_sec);
    std::vector<float> audio(num_samples);

    if (is_speech) {
        // 生成正弦波模拟语音
        float frequency = 440.0f;  // 440Hz
        for (size_t i = 0; i < num_samples; ++i) {
            audio[i] = 0.5f * std::sin(2.0f * M_PI * frequency * i / sample_rate);
        }
    } else {
        // 生成静音（很小的噪声）
        for (size_t i = 0; i < num_samples; ++i) {
            unsigned int seed = static_cast<unsigned int>(i);
            audio[i] = 0.001f * (static_cast<float>(rand_r(&seed)) / RAND_MAX - 0.5f);
        }
    }

    return audio;
}

int main() {
    std::cout << "=== VAD Simple Demo ===" << std::endl;

    // 创建 VAD 引擎（使用默认 Silero 配置）
    auto config = Evo::VadConfig::Silero()
        .withTriggerThreshold(0.5f)
        .withStopThreshold(0.35f);

    auto engine = std::make_shared<Evo::VadEngine>(config);

    if (!engine->IsInitialized()) {
        std::cerr << "Failed to initialize VAD engine" << std::endl;
        std::cerr << "Make sure Silero VAD model is available at ~/.cache/silero_vad/silero_vad.onnx" << std::endl;
        return 1;
    }

    std::cout << "VAD Engine initialized: " << engine->GetEngineName() << std::endl;

    // 测试参数
    int sample_rate = 16000;
    int window_size = 512;  // 32ms at 16kHz
    float window_duration = static_cast<float>(window_size) / sample_rate;

    // 生成测试序列：静音 -> 语音 -> 静音
    std::vector<std::pair<std::string, bool>> test_sequence = {
        {"Silence", false},
        {"Silence", false},
        {"Speech",  true},
        {"Speech",  true},
        {"Speech",  true},
        {"Silence", false},
        {"Silence", false},
    };

    std::cout << "\nProcessing test audio frames:" << std::endl;
    std::cout << "-----------------------------" << std::endl;

    for (size_t i = 0; i < test_sequence.size(); ++i) {
        auto& [label, is_speech] = test_sequence[i];

        // 生成音频帧
        auto audio = generateTestAudio(sample_rate, window_duration, is_speech);

        // 检测
        auto result = engine->Detect(audio, sample_rate);

        if (result && result->IsSuccess()) {
            std::cout << "Frame " << i << " (" << label << "): "
                << "prob=" << result->GetProbability()
                << ", state=" << static_cast<int>(result->GetState());

            if (result->IsSpeechStart()) {
                std::cout << " [SPEECH START]";
            } else if (result->IsSpeechEnd()) {
                std::cout << " [SPEECH END]";
            }

            std::cout << std::endl;
        } else {
            std::cerr << "Detection failed" << std::endl;
        }
    }

    std::cout << "\n=== Demo Complete ===" << std::endl;
    return 0;
}
