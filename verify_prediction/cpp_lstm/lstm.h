#ifndef LSTM_H
#define LSTM_H
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>

using Matrix = Eigen::MatrixXf;
using RowVector = Eigen::RowVectorXf;

// ── LSTM single cell ──
struct LSTMCell {
    Matrix Wi, Ui, bi;   // input gate
    Matrix Wf, Uf, bf;   // forget gate
    Matrix Wg, Ug, bg;   // cell gate
    Matrix Wo, Uo, bo;   // output gate
    int input_size, hidden_size;

    LSTMCell(int in, int hid);
    void forward(const RowVector &x, RowVector &h, RowVector &c);
    // State for backward pass
    RowVector x_cached, h_prev_cached, c_prev_cached;
    RowVector i_gate, f_gate, g_gate, o_gate, c_state, h_out;
};

// ── LSTM layer (encoder) ──
struct LSTMEncoder {
    LSTMCell cell;
    LSTMEncoder(int in, int hid);

    // Forward: process sequence, return final (h, c)
    void forward(const Matrix &X, RowVector &h, RowVector &c);
    // Backward through time
    void backward(const Matrix &X, const RowVector &h0, const RowVector &c0,
                  const RowVector &dh_final, const RowVector &dc_final,
                  Matrix &dX, float lr);

    // Cached states for BPTT
    std::vector<RowVector> h_list, c_list, x_list;
    std::vector<RowVector> i_list, f_list, g_list, o_list;
};

// ── LSTM decoder (autoregressive) ──
struct LSTMDecoder {
    LSTMCell cell;
    Matrix W_out; RowVector b_out;
    LSTMDecoder(int out_dim, int hid);

    // Forward: autoregressive generate `steps` outputs
    // start_input: (1, out_dim) first input
    Matrix forward(const RowVector &h0, const RowVector &c0,
                   const RowVector &start_input, int steps);

    // Forward with teacher forcing (for training)
    Matrix forward_with_tf(const RowVector &h0, const RowVector &c0,
                           const RowVector &start_input,
                           const Matrix &targets, float tf_prob,
                           std::mt19937 &rng);

    // Backward
    void backward(const RowVector &h0, const RowVector &c0,
                  const RowVector &start_input, const Matrix &targets,
                  float tf_prob, std::mt19937 &rng, float lr);
};

// ── Adam optimizer ──
struct Adam {
    std::vector<Matrix> m_W, v_W;
    std::vector<RowVector> m_b, v_b;
    float lr, beta1, beta2, eps;
    int step;

    Adam(float lr=0.001);
    void update(Matrix &W, RowVector &b, const Matrix &dW, const RowVector &db,
                int param_idx);
};

// ── SeqPredictor: encoder-decoder ──
struct SeqPredictor {
    LSTMEncoder encoder;
    LSTMDecoder decoder;
    Adam optimizer;

    SeqPredictor(int in_dim=10, int hid=128, int out_dim=2);

    // Forward pass (inference only)
    Matrix predict(const Matrix &X);  // X: (800, 10) → returns (200, 2)

    // Train one step (forward + backward + update)
    float train_step(const Matrix &X, const Matrix &targets);

    // Compute loss
    float compute_loss(const Matrix &pred, const Matrix &targets);
};

// ── Feature extraction (same as Python) ──
struct TelemetryPoint {
    float pos_km, spd_n, frc_n, mod_n, trf_n, ebf_n, bpw_n, soc_n, rte_n, time_n;
    float speed_raw, rte_raw, pos_raw;
};

// ── Curve computation from predictions ──
void compute_curves(const std::vector<TelemetryPoint> &history,
                    const Matrix &pred,  // (200, 2) [spd_n, rte_n]
                    std::vector<float> &actual_pos, std::vector<float> &actual_ene,
                    std::vector<float> &pred_pos, std::vector<float> &pred_ene,
                    std::vector<float> &future_pos, std::vector<float> &future_ene,
                    const std::vector<TelemetryPoint> &future);

#endif
