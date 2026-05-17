#ifndef ENGINE_DL_INFER_H
#define ENGINE_DL_INFER_H

#include <string>
#include <vector>
#include <memory>

namespace engine {

struct DLPredictResult {
    std::vector<double> positions;
    std::vector<double> energies;
    bool success = false;
    std::string error;
};

class DLInference {
public:
    DLInference();
    ~DLInference();

    bool loadModel(const std::string &modelPath);
    bool isLoaded() const;

    /**
     * Run DL inference: given (position, energy) history, predict future curve.
     * @param positions  input position sequence (lookback_window length)
     * @param energies   input energy sequence (lookback_window length)
     * @param forecastLen  number of points to predict
     */
    DLPredictResult predict(const std::vector<double> &positions,
                            const std::vector<double> &energies,
                            int forecastLen);

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};

} // namespace engine

#endif // ENGINE_DL_INFER_H
