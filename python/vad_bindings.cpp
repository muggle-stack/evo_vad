/*
 * VAD Python Bindings
 *
 * Provides Python interface to the VAD engine using pybind11.
 * Supports numpy arrays for audio data and Python callbacks.
 */

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "vad_api.hpp"

// Python version compatibility for Py_IsFinalizing
// Python 3.13+ has public Py_IsFinalizing(), older versions have internal _Py_IsFinalizing()
#if PY_VERSION_HEX >= 0x030D0000  // Python 3.13+
    #define IS_PYTHON_FINALIZING() Py_IsFinalizing()
#elif defined(_Py_IsFinalizing)
    #define IS_PYTHON_FINALIZING() _Py_IsFinalizing()
#else
    // Fallback: assume not finalizing if API unavailable
    #define IS_PYTHON_FINALIZING() false
#endif

namespace py = pybind11;

// =============================================================================
// Global Callback Registry for Safe Cleanup
// =============================================================================

class SafeCallbackWrapper;
static std::set<SafeCallbackWrapper*> g_active_wrappers;
static std::mutex g_wrappers_mutex;
static bool g_atexit_registered = false;

// Forward declaration
static void cleanup_all_wrappers();

// =============================================================================
// Safe Callback Wrapper - Manages Python callbacks with manual refcounting
// =============================================================================

// This wrapper uses raw PyObject* and only increments/decrements refcount
// when interpreter is known to be running. During shutdown, we leak the
// references intentionally to avoid crashes.
class SafeCallbackWrapper : public Evo::VadEngineCallback {
public:
    // Store as raw PyObject* - we manage refcount manually
    PyObject* on_event_ = nullptr;
    PyObject* on_speech_start_ = nullptr;
    PyObject* on_speech_end_ = nullptr;
    PyObject* on_error_ = nullptr;
    PyObject* on_open_ = nullptr;
    PyObject* on_complete_ = nullptr;
    PyObject* on_close_ = nullptr;

    std::atomic<bool> destroyed_{false};

    SafeCallbackWrapper() {
        std::lock_guard<std::mutex> lock(g_wrappers_mutex);
        g_active_wrappers.insert(this);

        if (!g_atexit_registered) {
            py::module_::import("atexit").attr("register")(
                py::cpp_function(&cleanup_all_wrappers));
            g_atexit_registered = true;
        }
    }

    ~SafeCallbackWrapper() {
        destroyed_.store(true);

        {
            std::lock_guard<std::mutex> lock(g_wrappers_mutex);
            g_active_wrappers.erase(this);
        }

        // If interpreter is still fully running (not finalizing), decref
        if (Py_IsInitialized() && !IS_PYTHON_FINALIZING()) {
            py::gil_scoped_acquire acquire;
            Py_XDECREF(on_event_);
            Py_XDECREF(on_speech_start_);
            Py_XDECREF(on_speech_end_);
            Py_XDECREF(on_error_);
            Py_XDECREF(on_open_);
            Py_XDECREF(on_complete_);
            Py_XDECREF(on_close_);
        }
        // If finalizing, intentionally leak the references
    }

    void clearAll() {
        destroyed_.store(true);
        if (Py_IsInitialized() && !IS_PYTHON_FINALIZING()) {
            py::gil_scoped_acquire acquire;
            Py_XDECREF(on_event_); on_event_ = nullptr;
            Py_XDECREF(on_speech_start_); on_speech_start_ = nullptr;
            Py_XDECREF(on_speech_end_); on_speech_end_ = nullptr;
            Py_XDECREF(on_error_); on_error_ = nullptr;
            Py_XDECREF(on_open_); on_open_ = nullptr;
            Py_XDECREF(on_complete_); on_complete_ = nullptr;
            Py_XDECREF(on_close_); on_close_ = nullptr;
        }
    }

    void setCallback(PyObject** target, PyObject* callback) {
        // Always executed with GIL held (called from Python)
        Py_XDECREF(*target);
        *target = callback;
        Py_XINCREF(*target);
    }

    bool isSafe() const {
        return !destroyed_.load() && Py_IsInitialized() && !IS_PYTHON_FINALIZING();
    }

    void OnOpen() override {
        if (on_open_ && isSafe()) {
            py::gil_scoped_acquire acquire;
            try {
                py::object func = py::reinterpret_borrow<py::object>(on_open_);
                func();
            } catch (...) {}
        }
    }

    void OnEvent(std::shared_ptr<Evo::VadResult> result) override {
        if (on_event_ && isSafe()) {
            py::gil_scoped_acquire acquire;
            try {
                py::object func = py::reinterpret_borrow<py::object>(on_event_);
                func(result);
            } catch (...) {}
        }
    }

    void OnSpeechStart(int64_t timestamp_ms) override {
        if (on_speech_start_ && isSafe()) {
            py::gil_scoped_acquire acquire;
            try {
                py::object func = py::reinterpret_borrow<py::object>(on_speech_start_);
                func(timestamp_ms);
            } catch (...) {}
        }
    }

    void OnSpeechEnd(int64_t timestamp_ms, int duration_ms) override {
        if (on_speech_end_ && isSafe()) {
            py::gil_scoped_acquire acquire;
            try {
                py::object func = py::reinterpret_borrow<py::object>(on_speech_end_);
                func(timestamp_ms, duration_ms);
            } catch (...) {}
        }
    }

    void OnComplete() override {
        if (on_complete_ && isSafe()) {
            py::gil_scoped_acquire acquire;
            try {
                py::object func = py::reinterpret_borrow<py::object>(on_complete_);
                func();
            } catch (...) {}
        }
    }

    void OnError(const std::string& message) override {
        if (on_error_ && isSafe()) {
            py::gil_scoped_acquire acquire;
            try {
                py::object func = py::reinterpret_borrow<py::object>(on_error_);
                func(message);
            } catch (...) {}
        }
    }

    void OnClose() override {
        if (on_close_ && isSafe()) {
            py::gil_scoped_acquire acquire;
            try {
                py::object func = py::reinterpret_borrow<py::object>(on_close_);
                func();
            } catch (...) {}
        }
    }
};

// Cleanup function called by atexit - clears all callbacks before interpreter finalizes
static void cleanup_all_wrappers() {
    std::lock_guard<std::mutex> lock(g_wrappers_mutex);
    for (auto* wrapper : g_active_wrappers) {
        if (wrapper) {
            wrapper->clearAll();
        }
    }
}

// =============================================================================
// Python Callback Class (exposed to Python)
// =============================================================================

class PyVadCallback {
public:
    std::shared_ptr<SafeCallbackWrapper> wrapper_;

    PyVadCallback() : wrapper_(std::make_shared<SafeCallbackWrapper>()) {}

    ~PyVadCallback() {
        if (wrapper_) {
            wrapper_->clearAll();
        }
    }

    void setOnEvent(py::object cb) {
        wrapper_->setCallback(&wrapper_->on_event_, cb.ptr());
    }
    void setOnSpeechStart(py::object cb) {
        wrapper_->setCallback(&wrapper_->on_speech_start_, cb.ptr());
    }
    void setOnSpeechEnd(py::object cb) {
        wrapper_->setCallback(&wrapper_->on_speech_end_, cb.ptr());
    }
    void setOnError(py::object cb) {
        wrapper_->setCallback(&wrapper_->on_error_, cb.ptr());
    }
    void setOnOpen(py::object cb) {
        wrapper_->setCallback(&wrapper_->on_open_, cb.ptr());
    }
    void setOnComplete(py::object cb) {
        wrapper_->setCallback(&wrapper_->on_complete_, cb.ptr());
    }
    void setOnClose(py::object cb) {
        wrapper_->setCallback(&wrapper_->on_close_, cb.ptr());
    }

    std::shared_ptr<Evo::VadEngineCallback> getWrapper() {
        return wrapper_;
    }
};

// =============================================================================
// Python Module Definition
// =============================================================================

PYBIND11_MODULE(_evo_vad, m) {
    m.doc() = "VAD (Voice Activity Detection) Python bindings";

    // -------------------------------------------------------------------------
    // Enums
    // -------------------------------------------------------------------------

    py::enum_<Evo::VadState>(m, "VadState", "VAD state")
        .value("SILENCE", Evo::VadState::SILENCE, "Silence state")
        .value("SPEECH_START", Evo::VadState::SPEECH_START, "Speech started")
        .value("SPEECH", Evo::VadState::SPEECH, "In speech")
        .value("SPEECH_END", Evo::VadState::SPEECH_END, "Speech ended")
        .export_values();

    py::enum_<Evo::VadBackendType>(m, "VadBackendType", "VAD backend types")
        .value("SILERO", Evo::VadBackendType::SILERO, "Silero VAD (ONNX)")
        .value("ENERGY", Evo::VadBackendType::ENERGY, "Energy-based VAD")
        .value("WEBRTC", Evo::VadBackendType::WEBRTC, "WebRTC VAD")
        .value("FSMN", Evo::VadBackendType::FSMN, "FSMN VAD")
        .value("CUSTOM", Evo::VadBackendType::CUSTOM, "Custom backend")
        .export_values();

    // -------------------------------------------------------------------------
    // VadConfig
    // -------------------------------------------------------------------------

    py::class_<Evo::VadConfig>(m, "VadConfig", "VAD configuration")
        .def(py::init<>())
        // Properties
        .def_readwrite("backend", &Evo::VadConfig::backend)
        .def_readwrite("model_dir", &Evo::VadConfig::model_dir)
        .def_readwrite("sample_rate", &Evo::VadConfig::sample_rate)
        .def_readwrite("window_size", &Evo::VadConfig::window_size)
        .def_readwrite("trigger_threshold", &Evo::VadConfig::trigger_threshold)
        .def_readwrite("stop_threshold", &Evo::VadConfig::stop_threshold)
        .def_readwrite("min_speech_duration_ms", &Evo::VadConfig::min_speech_duration_ms)
        .def_readwrite("min_silence_duration_ms", &Evo::VadConfig::min_silence_duration_ms)
        .def_readwrite("smoothing_window", &Evo::VadConfig::smoothing_window)
        .def_readwrite("use_smoothing", &Evo::VadConfig::use_smoothing)
        .def_readwrite("num_threads", &Evo::VadConfig::num_threads)
        // Static factory methods
        .def_static("default_config", &Evo::VadConfig::Default,
            "Create default configuration")
        .def_static("silero", &Evo::VadConfig::Silero,
            py::arg("model_dir") = "~/.cache/silero_vad",
            "Create Silero VAD configuration")
        .def_static("energy", &Evo::VadConfig::Energy,
            "Create energy-based VAD configuration")
        // Builder methods
        .def("with_trigger_threshold", &Evo::VadConfig::withTriggerThreshold,
            py::arg("threshold"), "Set trigger threshold")
        .def("with_stop_threshold", &Evo::VadConfig::withStopThreshold,
            py::arg("threshold"), "Set stop threshold")
        .def("with_smoothing_window", &Evo::VadConfig::withSmoothingWindow,
            py::arg("window"), "Set smoothing window size")
        .def("with_min_speech_duration", &Evo::VadConfig::withMinSpeechDuration,
            py::arg("ms"), "Set minimum speech duration")
        .def("with_min_silence_duration", &Evo::VadConfig::withMinSilenceDuration,
            py::arg("ms"), "Set minimum silence duration")
        .def("with_window_size", &Evo::VadConfig::withWindowSize,
            py::arg("size"), "Set window size")
        .def("with_sample_rate", &Evo::VadConfig::withSampleRate,
            py::arg("rate"), "Set sample rate")
        .def("with_smoothing", &Evo::VadConfig::withSmoothing,
            py::arg("enable"), "Enable/disable smoothing")
        .def("with_num_threads", &Evo::VadConfig::withNumThreads,
            py::arg("threads"), "Set number of threads");

    // -------------------------------------------------------------------------
    // VadResult
    // -------------------------------------------------------------------------

    py::class_<Evo::VadResult, std::shared_ptr<Evo::VadResult>>(m, "VadResult",
            "VAD detection result")
        .def_property_readonly("probability", &Evo::VadResult::GetProbability,
            "Get raw speech probability")
        .def_property_readonly("smoothed_probability", &Evo::VadResult::GetSmoothedProbability,
            "Get smoothed speech probability")
        .def_property_readonly("is_speech", &Evo::VadResult::IsSpeech,
            "Check if speech detected")
        .def_property_readonly("state", &Evo::VadResult::GetState,
            "Get VAD state")
        .def_property_readonly("is_speech_start", &Evo::VadResult::IsSpeechStart,
            "Check if speech just started")
        .def_property_readonly("is_speech_end", &Evo::VadResult::IsSpeechEnd,
            "Check if speech just ended")
        .def_property_readonly("timestamp_ms", &Evo::VadResult::GetTimestampMs,
            "Get frame timestamp in milliseconds")
        .def_property_readonly("speech_start_ms", &Evo::VadResult::GetSpeechStartMs,
            "Get speech start timestamp")
        .def_property_readonly("speech_end_ms", &Evo::VadResult::GetSpeechEndMs,
            "Get speech end timestamp")
        .def_property_readonly("speech_duration_ms", &Evo::VadResult::GetSpeechDurationMs,
            "Get speech duration")
        .def_property_readonly("processing_time_ms", &Evo::VadResult::GetProcessingTimeMs,
            "Get processing time")
        .def_property_readonly("success", &Evo::VadResult::IsSuccess,
            "Check if detection succeeded")
        .def_property_readonly("code", &Evo::VadResult::GetCode,
            "Get error code")
        .def_property_readonly("message", &Evo::VadResult::GetMessage,
            "Get error message")
        .def("__bool__", &Evo::VadResult::IsSuccess)
        .def("__repr__", [](const Evo::VadResult& r) {
            return "VadResult(prob=" + std::to_string(r.GetProbability()) +
                ", speech=" + (r.IsSpeech() ? "true" : "false") + ")";
        });

    // -------------------------------------------------------------------------
    // VadEngine
    // -------------------------------------------------------------------------

    py::class_<Evo::VadEngine, std::shared_ptr<Evo::VadEngine>>(m, "VadEngine",
            "VAD engine for speech detection")
        // Constructors
        .def(py::init<Evo::VadBackendType, const std::string&>(),
            py::arg("backend") = Evo::VadBackendType::SILERO,
            py::arg("model_dir") = "",
            "Create VAD engine with backend type and model directory")
        .def(py::init<const Evo::VadConfig&>(),
            py::arg("config"),
            "Create VAD engine with configuration")

        // Detect with numpy array (float)
        .def("detect", [](Evo::VadEngine& self, py::array_t<float> audio, int sample_rate) {
            auto buf = audio.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Audio array must be 1-dimensional");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            size_t size = buf.size;
            // Release GIL during inference
            py::gil_scoped_release release;
            return self.Detect(ptr, size, sample_rate);
        }, py::arg("audio"), py::arg("sample_rate") = 16000,
            "Detect speech in audio array (float)")

        // Detect with numpy array (int16)
        .def("detect", [](Evo::VadEngine& self, py::array_t<int16_t> audio, int sample_rate) {
            auto buf = audio.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Audio array must be 1-dimensional");
            }
            // Convert int16 to float
            std::vector<float> float_audio(buf.size);
            int16_t* ptr = static_cast<int16_t*>(buf.ptr);
            for (size_t i = 0; i < buf.size; ++i) {
                float_audio[i] = ptr[i] / 32768.0f;
            }
            // Release GIL during inference
            py::gil_scoped_release release;
            return self.Detect(float_audio, sample_rate);
        }, py::arg("audio"), py::arg("sample_rate") = 16000,
            "Detect speech in audio array (int16)")

        // Streaming API
        .def("set_callback", [](Evo::VadEngine& self, PyVadCallback& cb) {
            // Pass the SafeCallbackWrapper to the engine
            self.SetCallback(cb.getWrapper());
        }, py::arg("callback"), "Set streaming callback")
        .def("start", &Evo::VadEngine::Start, "Start streaming detection")
        .def("send_audio_frame", [](Evo::VadEngine& self, py::array_t<float> audio) {
            auto buf = audio.request();
            self.SendAudioFrame(static_cast<float*>(buf.ptr), buf.size);
        }, py::arg("audio"), "Send audio frame (float array)")
        .def("send_audio_frame", [](Evo::VadEngine& self, py::bytes data) {
            // PCM S16LE bytes -> float array
            std::string str = data;
            const int16_t* ptr = reinterpret_cast<const int16_t*>(str.data());
            size_t samples = str.size() / 2;
            std::vector<float> float_audio(samples);
            for (size_t i = 0; i < samples; ++i) {
                float_audio[i] = ptr[i] / 32768.0f;
            }
            self.SendAudioFrame(float_audio);
        }, py::arg("data"), "Send audio frame (PCM S16LE bytes)")
        .def("stop", &Evo::VadEngine::Stop, "Stop streaming detection")

        // State management
        .def("reset", &Evo::VadEngine::Reset, "Reset engine state")
        .def_property_readonly("current_state", &Evo::VadEngine::GetCurrentState,
            "Get current VAD state")
        .def_property_readonly("is_in_speech", &Evo::VadEngine::IsInSpeech,
            "Check if currently in speech")
        .def_property_readonly("is_initialized", &Evo::VadEngine::IsInitialized,
            "Check if engine is initialized")
        .def_property_readonly("is_streaming", &Evo::VadEngine::IsStreaming,
            "Check if streaming is active")

        // Dynamic configuration
        .def("set_trigger_threshold", &Evo::VadEngine::SetTriggerThreshold,
            py::arg("threshold"), "Set trigger threshold")
        .def("set_stop_threshold", &Evo::VadEngine::SetStopThreshold,
            py::arg("threshold"), "Set stop threshold")
        .def_property_readonly("config", &Evo::VadEngine::GetConfig,
            "Get current configuration")

        // Info
        .def_property_readonly("engine_name", &Evo::VadEngine::GetEngineName,
            "Get engine name")
        .def_property_readonly("backend_type", &Evo::VadEngine::GetBackendType,
            "Get backend type")
        .def_property_readonly("last_probability", &Evo::VadEngine::GetLastProbability,
            "Get last speech probability");

    // -------------------------------------------------------------------------
    // Python Callback
    // -------------------------------------------------------------------------

    py::class_<PyVadCallback>(m, "VadCallback",
            "Callback for streaming VAD results")
        .def(py::init<>())
        .def("on_event", &PyVadCallback::setOnEvent,
            py::arg("callback"), "Set event callback")
        .def("on_speech_start", &PyVadCallback::setOnSpeechStart,
            py::arg("callback"), "Set speech start callback")
        .def("on_speech_end", &PyVadCallback::setOnSpeechEnd,
            py::arg("callback"), "Set speech end callback")
        .def("on_error", &PyVadCallback::setOnError,
            py::arg("callback"), "Set error callback")
        .def("on_open", &PyVadCallback::setOnOpen,
            py::arg("callback"), "Set open callback")
        .def("on_complete", &PyVadCallback::setOnComplete,
            py::arg("callback"), "Set complete callback")
        .def("on_close", &PyVadCallback::setOnClose,
            py::arg("callback"), "Set close callback");

    // -------------------------------------------------------------------------
    // Quick Functions
    // -------------------------------------------------------------------------

    m.def("detect", [](py::array_t<float> audio, int sample_rate,
                        float trigger_threshold, float stop_threshold) {
        auto buf = audio.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Audio array must be 1-dimensional");
        }

        auto config = Evo::VadConfig::Silero()
            .withTriggerThreshold(trigger_threshold)
            .withStopThreshold(stop_threshold);
        auto engine = std::make_shared<Evo::VadEngine>(config);

        if (!engine->IsInitialized()) {
            throw std::runtime_error("Failed to initialize VAD engine");
        }

        float* ptr = static_cast<float*>(buf.ptr);
        size_t size = buf.size;

        py::gil_scoped_release release;
        return engine->Detect(ptr, size, sample_rate);
    }, py::arg("audio"),
        py::arg("sample_rate") = 16000,
        py::arg("trigger_threshold") = 0.5f,
        py::arg("stop_threshold") = 0.35f,
        "Quick detection of speech in audio array");

    // -------------------------------------------------------------------------
    // Module Info
    // -------------------------------------------------------------------------

    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "muggle";

    // Add a cleanup capsule - its destructor is called when module is unloaded
    // This ensures callbacks are cleared before Python objects are destroyed
    // Use a dummy non-null pointer since PyCapsule_New requires non-null
    static int cleanup_dummy = 0;
    m.attr("_cleanup") = py::capsule(&cleanup_dummy, [](void*) {
        // Clear all active wrappers during module unload
        std::lock_guard<std::mutex> lock(g_wrappers_mutex);
        for (auto* wrapper : g_active_wrappers) {
            if (wrapper) {
                // Just mark as destroyed, don't try to delete anything
                wrapper->destroyed_.store(true);
            }
        }
        g_active_wrappers.clear();
    });
}
