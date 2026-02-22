# VAD - 语音活动检测引擎

基于 Silero 深度学习模型的语音活动检测框架，支持流式检测和回调机制，提供 C++ 和 Python 接口。

## 特性

- Silero VAD 深度学习模型（ONNX 推理），高准确率
- 单帧检测与流式检测两种模式
- 回调驱动事件通知：语音开始、语音结束、每帧概率
- 概率平滑机制，减少误触发
- 可配置阈值、最小语音/静音持续时间
- Python 绑定（pybind11），支持 NumPy 数组和 PCM 字节输入

## 依赖

### 系统依赖

| 平台 | 安装命令 |
|------|---------|
| Ubuntu/Debian | ONNX Runtime 需手动安装（见 [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases)） |
| macOS | `brew install onnxruntime` |

可选：`pybind11`（Python 绑定）、`portaudio`（流式示例）。

### 模型文件

首次运行自动下载到 `~/.cache/`：

| 文件 | 说明 |
|------|------|
| `silero_vad.onnx` | Silero VAD ONNX 模型 |

## 编译

```bash
mkdir -p build && cd build
cmake .. && make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

CMake 选项：

| 选项 | 默认 | 说明 |
|------|------|------|
| `BUILD_VAD_EXAMPLES` | ON | 构建示例程序 |
| `BUILD_VAD_PYTHON` | ON | 构建 Python 绑定 |

编译产物：

```
build/lib/libvad.a           # VAD 框架库
build/lib/libsilero_vad.a    # Silero 后端库
build/bin/vad_simple_demo    # C++ 示例
```

## 快速开始

### C++

```cpp
#include "vad_api.hpp"

int main() {
    auto config = Evo::VadConfig::Silero()
        .withTriggerThreshold(0.5f)
        .withStopThreshold(0.35f);
    auto engine = std::make_shared<Evo::VadEngine>(config);

    std::vector<float> audio_frame(512, 0.0f);  // 32ms @ 16kHz
    auto result = engine->Detect(audio_frame, 16000);
    std::cout << "语音概率: " << result->GetProbability() << std::endl;
    return 0;
}
```

### Python

```bash
cd modules/vad/python && pip install -e .
```

```python
import evo_vad
import numpy as np

audio = np.zeros(512, dtype=np.float32)  # 32ms @ 16kHz
result = evo_vad.detect(audio)
print(f"语音概率: {result.probability:.3f}")
```

## 示例

```bash
# C++
./build/bin/vad_simple_demo

# Python
python python/examples/simple_vad.py
```

## API 概览

### 核心类（命名空间 `Evo::`）

| 类 | 说明 |
|----|------|
| `VadEngine` | VAD 引擎，支持单帧检测和流式检测 |
| `VadConfig` | 引擎配置，提供 `Silero()` 工厂方法和链式调用 |
| `VadResult` | 检测结果，含概率、状态、时间戳 |
| `VadEngineCallback` | 流式回调接口（OnSpeechStart/OnSpeechEnd/OnEvent） |

### 枚举

| 枚举 | 值 | 说明 |
|------|----|------|
| `VadState::SILENCE` | 0 | 静音状态 |
| `VadState::SPEECH_START` | 1 | 语音开始（状态转换） |
| `VadState::SPEECH` | 2 | 语音进行中 |
| `VadState::SPEECH_END` | 3 | 语音结束（状态转换） |

### 关键方法

| 方法 | 说明 |
|------|------|
| `Detect(audio, sample_rate)` | 单帧检测（阻塞） |
| `SetCallback()` / `Start()` / `SendAudioFrame()` / `Stop()` | 流式检测 |
| `Reset()` | 重置内部状态（LSTM 隐藏状态、概率历史） |

详细文档见 [API.md](API.md)

## 配置参数

### VadConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend` | `VadBackendType` | `SILERO` | 后端类型 |
| `model_dir` | `string` | `"~/.cache/silero_vad"` | 模型目录 |
| `sample_rate` | `int` | `16000` | 采样率 (Hz) |
| `window_size` | `int` | `512` | 窗口大小（32ms @ 16kHz） |
| `trigger_threshold` | `float` | `0.5` | 语音开始阈值 [0.0, 1.0] |
| `stop_threshold` | `float` | `0.35` | 语音结束阈值 [0.0, 1.0] |
| `min_speech_duration_ms` | `int` | `250` | 最小语音持续时间 (ms) |
| `min_silence_duration_ms` | `int` | `100` | 最小静音持续时间 (ms) |
| `smoothing_window` | `int` | `10` | 概率平滑窗口大小 |
| `use_smoothing` | `bool` | `true` | 启用概率平滑 |
| `num_threads` | `int` | `1` | ONNX 推理线程数 |

## CMake 集成

```cmake
add_subdirectory(modules/vad)
target_link_libraries(your_target PRIVATE vad)
target_include_directories(your_target PRIVATE ${VAD_SOURCE_DIR}/inc)
```

## 许可证

MIT License - 详见 [LICENSE](LICENSE)
