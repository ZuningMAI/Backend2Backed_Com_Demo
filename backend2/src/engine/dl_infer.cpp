#include "engine/dl_infer.h"
#include "onnxruntime_c_api.h"

#include <QDebug>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <new>

namespace engine {

struct DLInference::Impl {
    const OrtApi *api = nullptr;
    OrtEnv *env = nullptr;
    OrtSession *session = nullptr;
    OrtMemoryInfo *memInfo = nullptr;
    bool loaded = false;
    int lookback = 100;
    int forecastLen = 50;
};

DLInference::DLInference() : d(std::make_unique<Impl>()) {}

DLInference::~DLInference()
{
    if (d->session) d->api->ReleaseSession(d->session);
    if (d->memInfo) d->api->ReleaseMemoryInfo(d->memInfo);
    if (d->env) d->api->ReleaseEnv(d->env);
}

bool DLInference::loadModel(const std::string &modelPath)
{
    const OrtApiBase *apiBase = OrtGetApiBase();
    if (!apiBase) {
        qWarning() << "DLInference: OrtGetApiBase returned null";
        return false;
    }

    d->api = apiBase->GetApi(ORT_API_VERSION);
    if (!d->api) {
        qWarning() << "DLInference: GetApi returned null";
        return false;
    }

    // Create environment
    OrtStatus *status = d->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "Backend2", &d->env);
    if (status) {
        qWarning() << "DLInference: CreateEnv failed";
        d->api->ReleaseStatus(status);
        return false;
    }

    // Create CPU memory info
    status = d->api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &d->memInfo);
    if (status) {
        qWarning() << "DLInference: CreateCpuMemoryInfo failed";
        d->api->ReleaseStatus(status);
        return false;
    }

    // Create session options
    OrtSessionOptions *sessionOpts = nullptr;
    status = d->api->CreateSessionOptions(&sessionOpts);
    if (status) {
        qWarning() << "DLInference: CreateSessionOptions failed";
        d->api->ReleaseStatus(status);
        return false;
    }

    // Set number of intra-op threads
    d->api->SetIntraOpNumThreads(sessionOpts, 1);
    d->api->SetSessionGraphOptimizationLevel(sessionOpts, ORT_ENABLE_ALL);

    // Load model
    status = d->api->CreateSession(d->env, modelPath.c_str(), sessionOpts, &d->session);
    d->api->ReleaseSessionOptions(sessionOpts);

    if (status) {
        qWarning() << "DLInference: Failed to load model from" << modelPath.c_str();
        d->api->ReleaseStatus(status);
        return false;
    }

    d->loaded = true;
    qInfo() << "DLInference: ONNX model loaded:" << modelPath.c_str();
    return true;
}

bool DLInference::isLoaded() const
{
    return d->loaded;
}

DLPredictResult DLInference::predict(const std::vector<double> &positions,
                                      const std::vector<double> &energies,
                                      int forecastLen)
{
    DLPredictResult result;
    result.success = false;

    if (!d->loaded) {
        result.error = "Model not loaded";
        return result;
    }

    int lookback = d->lookback;
    int actualLen = std::min(lookback, (int)std::min(positions.size(), energies.size()));

    if (actualLen < 5) {
        result.error = "Insufficient history";
        return result;
    }

    // Build input: (1, lookback, 2) with position and energy as 2 features
    std::vector<float> inputData(lookback * 2, 0.0f);
    int offset = lookback - actualLen;
    double posStart = positions[0];

    for (int i = 0; i < actualLen; ++i) {
        int idx = offset + i;
        inputData[idx * 2 + 0] = (float)(positions[i] - posStart);
        inputData[idx * 2 + 1] = (float)energies[i];
    }

    // Create input tensor
    int64_t inputShape[] = {1, lookback, 2};
    OrtValue *inputTensor = nullptr;
    OrtStatus *status = d->api->CreateTensorWithDataAsOrtValue(
        d->memInfo, inputData.data(), lookback * 2 * sizeof(float),
        inputShape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputTensor);

    if (status) {
        d->api->ReleaseStatus(status);
        result.error = "CreateTensor failed";
        return result;
    }

    // Run inference
    const char *inputNames[] = {"input"};
    const char *outputNames[] = {"output"};
    OrtValue *outputTensor = nullptr;

    status = d->api->Run(d->session, nullptr, inputNames,
                          (const OrtValue *const *)&inputTensor, 1,
                          outputNames, 1, &outputTensor);

    d->api->ReleaseValue(inputTensor);

    if (status) {
        d->api->ReleaseStatus(status);
        result.error = "Inference failed";
        return result;
    }

    // Get output data
    float *outputData = nullptr;
    status = d->api->GetTensorMutableData(outputTensor, (void **)&outputData);

    if (status) {
        d->api->ReleaseStatus(status);
        d->api->ReleaseValue(outputTensor);
        result.error = "GetTensorData failed";
        return result;
    }

    // Parse output: (1, forecast_len, 2) → (position, energy) pairs
    int actualForecast = std::min(forecastLen, d->forecastLen);
    double lastPos = positions.back();

    for (int i = 0; i < actualForecast; ++i) {
        float predPos = outputData[i * 2 + 0] + (float)posStart;
        float predEnergy = outputData[i * 2 + 1];
        result.positions.push_back((double)predPos);
        result.energies.push_back((double)predEnergy);
    }

    d->api->ReleaseValue(outputTensor);
    result.success = true;
    return result;
}

} // namespace engine
