#include "lstm.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>

// ── Utils ──
static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static float tanh_f(float x) { return std::tanh(x); }
static RowVector sigmoid_v(const RowVector &x) { return x.unaryExpr([](float v){return sigmoid(v);}); }
static RowVector tanh_v(const RowVector &x) { return x.unaryExpr([](float v){return tanh_f(v);}); }
static RowVector tanh_f_v(const RowVector &x) { return x.unaryExpr([](float v){return tanh_f(v);}); }
static float clamp(float x, float lo, float hi) { return std::max(lo, std::min(hi, x)); }
static RowVector clamp_v(const RowVector &x, float lo, float hi) {
    return x.unaryExpr([lo,hi](float v){return clamp(v,lo,hi);});
}
static RowVector scalar_minus_vec(float s, const RowVector &v) {
    RowVector r(v.size()); for(int i=0;i<v.size();++i) r(i)=s-v(i); return r;
}

static float xavier_scale(int fan_in, int fan_out) {
    return std::sqrt(6.0f / (fan_in + fan_out));
}
static Matrix xavier(int rows, int cols, std::mt19937 &rng) {
    float s = xavier_scale(cols, rows);
    std::uniform_real_distribution<float> dist(-s, s);
    Matrix M(rows, cols);
    for (int i=0;i<rows;++i) for(int j=0;j<cols;++j) M(i,j)=dist(rng);
    return M;
}
static RowVector zeros(int n) { RowVector v(n); v.setZero(); return v; }

// ═══════════════════ LSTMCell ═══════════════════
LSTMCell::LSTMCell(int in, int hid) : input_size(in), hidden_size(hid) {
    std::mt19937 rng(42);
    Wi=xavier(hid,in,rng); Ui=xavier(hid,hid,rng); bi=zeros(hid);
    Wf=xavier(hid,in,rng); Uf=xavier(hid,hid,rng); bf=RowVector::Ones(hid); // forget bias=1
    Wg=xavier(hid,in,rng); Ug=xavier(hid,hid,rng); bg=zeros(hid);
    Wo=xavier(hid,in,rng); Uo=xavier(hid,hid,rng); bo=zeros(hid);
}

void LSTMCell::forward(const RowVector &x, RowVector &h, RowVector &c) {
    x_cached=x; h_prev_cached=h; c_prev_cached=c;
    i_gate=sigmoid_v(x*Wi.transpose()+h*Ui.transpose()+bi);
    f_gate=sigmoid_v(x*Wf.transpose()+h*Uf.transpose()+bf);
    g_gate=tanh_v(x*Wg.transpose()+h*Ug.transpose()+bg);
    o_gate=sigmoid_v(x*Wo.transpose()+h*Uo.transpose()+bo);
    c_state=f_gate.cwiseProduct(c) + i_gate.cwiseProduct(g_gate);
    h_out=o_gate.cwiseProduct(tanh_v(c_state));
    h=h_out; c=c_state;
}

// ═══════════════════ LSTMEncoder ═══════════════════
LSTMEncoder::LSTMEncoder(int in, int hid) : cell(in, hid) {}

void LSTMEncoder::forward(const Matrix &X, RowVector &h, RowVector &c) {
    int T = (int)X.rows();
    h_list.clear(); c_list.clear(); x_list.clear();
    i_list.clear(); f_list.clear(); g_list.clear(); o_list.clear();

    for (int t=0; t<T; ++t) {
        RowVector xt = X.row(t);
        x_list.push_back(xt);
        h_list.push_back(h); c_list.push_back(c);
        cell.forward(xt, h, c);
        i_list.push_back(cell.i_gate);
        f_list.push_back(cell.f_gate);
        g_list.push_back(cell.g_gate);
        o_list.push_back(cell.o_gate);
    }
}

void LSTMEncoder::backward(const Matrix &X, const RowVector &h0, const RowVector &c0,
                           const RowVector &dh_final, const RowVector &dc_final,
                           Matrix &dX, float lr) {
    int T = (int)X.rows();
    int H = cell.hidden_size;
    int D = cell.input_size;

    RowVector dh = dh_final, dc = dc_final;
    dX.setZero(T, D);

    // Accumulate gradients for all gates (sum over time steps)
    Matrix dWi = Matrix::Zero(H,D), dUi = Matrix::Zero(H,H);
    Matrix dWf = Matrix::Zero(H,D), dUf = Matrix::Zero(H,H);
    Matrix dWg = Matrix::Zero(H,D), dUg = Matrix::Zero(H,H);
    Matrix dWo = Matrix::Zero(H,D), dUo = Matrix::Zero(H,H);
    RowVector dbi = zeros(H), dbf = zeros(H), dbg = zeros(H), dbo = zeros(H);

    for (int t=T-1; t>=0; --t) {
        RowVector xt = x_list[t];
        RowVector ht_prev = (t>0) ? h_list[t-1] : h0;
        RowVector ct_prev = (t>0) ? c_list[t-1] : c0;

        RowVector i_g = i_list[t], f_g = f_list[t];
        RowVector g_g = g_list[t], o_g = o_list[t];
        RowVector cs = f_g.cwiseProduct(ct_prev) + i_g.cwiseProduct(g_g);

        // dL/do — sigmoid derivative: σ(x)*(1-σ(x)) = o*(1-o)
        RowVector sig_deriv_o = o_g.cwiseProduct(scalar_minus_vec(1.0f, o_g));
        RowVector do_raw = dh.cwiseProduct(tanh_f_v(cs)).cwiseProduct(sig_deriv_o);
        // dL/dcs — tanh derivative: 1-tanh²
        RowVector tanh_cs = tanh_f_v(cs);
        RowVector dtanh = scalar_minus_vec(1.0f, tanh_cs.cwiseProduct(tanh_cs));
        RowVector dcs = dh.cwiseProduct(o_g).cwiseProduct(dtanh) + dc;

        // Gate gradients with correct sigmoid derivative: g*(1-g)
        RowVector sig_deriv_i = i_g.cwiseProduct(scalar_minus_vec(1.0f, i_g));
        RowVector sig_deriv_f = f_g.cwiseProduct(scalar_minus_vec(1.0f, f_g));
        RowVector tanh_deriv_g = scalar_minus_vec(1.0f, g_g.cwiseProduct(g_g));

        RowVector di_raw = dcs.cwiseProduct(g_g).cwiseProduct(sig_deriv_i);
        RowVector df_raw = dcs.cwiseProduct(ct_prev).cwiseProduct(sig_deriv_f);
        RowVector dg_raw = dcs.cwiseProduct(i_g).cwiseProduct(tanh_deriv_g);

        // Accumulate gate gradients
        dWi.noalias() += di_raw.transpose() * xt;
        dUi.noalias() += di_raw.transpose() * ht_prev;
        dbi += di_raw;
        dWf.noalias() += df_raw.transpose() * xt;
        dUf.noalias() += df_raw.transpose() * ht_prev;
        dbf += df_raw;
        dWg.noalias() += dg_raw.transpose() * xt;
        dUg.noalias() += dg_raw.transpose() * ht_prev;
        dbg += dg_raw;
        dWo.noalias() += do_raw.transpose() * xt;
        dUo.noalias() += do_raw.transpose() * ht_prev;
        dbo += do_raw;

        // Gradient to input
        RowVector dx = di_raw*cell.Wi + df_raw*cell.Wf + dg_raw*cell.Wg + do_raw*cell.Wo;
        for(int j=0;j<D;++j) dX(t,j) = dx(j);

        // Gradient to previous h and c
        dh = di_raw*cell.Ui + df_raw*cell.Uf + dg_raw*cell.Ug + do_raw*cell.Uo;
        dc = dcs.cwiseProduct(f_g);
    }

    // Apply updates
    float lr_f = lr;
    cell.Wi -= lr_f*dWi; cell.Ui -= lr_f*dUi; cell.bi -= lr_f*dbi;
    cell.Wf -= lr_f*dWf; cell.Uf -= lr_f*dUf; cell.bf -= lr_f*dbf;
    cell.Wg -= lr_f*dWg; cell.Ug -= lr_f*dUg; cell.bg -= lr_f*dbg;
    cell.Wo -= lr_f*dWo; cell.Uo -= lr_f*dUo; cell.bo -= lr_f*dbo;
}

// ═══════════════════ LSTMDecoder ═══════════════════
LSTMDecoder::LSTMDecoder(int out_dim, int hid) : cell(out_dim, hid) {
    std::mt19937 rng(42);
    W_out = xavier(out_dim, hid, rng);
    b_out = zeros(out_dim);
}

Matrix LSTMDecoder::forward(const RowVector &h0, const RowVector &c0,
                            const RowVector &start_input, int steps) {
    RowVector h = h0, c = c0;
    Matrix result(steps, start_input.size());
    RowVector inp = start_input;
    for (int t=0; t<steps; ++t) {
        cell.forward(inp, h, c);
        RowVector out = h*W_out.transpose() + b_out;
        result.row(t) = out;
        inp = out;
    }
    return result;
}

Matrix LSTMDecoder::forward_with_tf(const RowVector &h0, const RowVector &c0,
                                    const RowVector &start_input,
                                    const Matrix &targets, float tf_prob,
                                    std::mt19937 &rng) {
    RowVector h = h0, c = c0;
    int steps = (int)targets.rows();
    int out_dim = (int)targets.cols();
    Matrix result(steps, out_dim);
    RowVector inp = start_input;
    std::uniform_real_distribution<float> dist(0,1);
    for (int t=0; t<steps; ++t) {
        cell.forward(inp, h, c);
        RowVector out = h*W_out.transpose() + b_out;
        result.row(t) = out;
        if (dist(rng) < tf_prob && t < steps)
            inp = targets.row(t);
        else
            inp = out;
    }
    return result;
}

void LSTMDecoder::backward(const RowVector &h0, const RowVector &c0,
                           const RowVector &start_input, const Matrix &targets,
                           float tf_prob, std::mt19937 &rng, float lr) {
    // Simplified: accumulate decoder cell gradients
    // For full BPTT through decoder, we'd need to store states.
    // Simplified version: train decoder with TF always on, gradients flow back
    RowVector h = h0, c = c0;
    int T = (int)targets.rows();
    RowVector inp = start_input;

    Matrix dW_out = Matrix::Zero((int)targets.cols(), cell.hidden_size);
    RowVector db_out = zeros((int)targets.cols());

    for (int t=0; t<T; ++t) {
        cell.forward(inp, h, c);
        RowVector pred = h*W_out.transpose() + b_out;
        RowVector target = targets.row(t);
        RowVector dL = (pred - target) * (2.0f / target.size());

        dW_out.noalias() += dL.transpose() * h;
        db_out += dL;
        inp = targets.row(t); // use target as next input for gradient flow
    }

    float lr_f = lr;
    W_out -= lr_f * dW_out;
    b_out -= lr_f * db_out;
}

// ═══════════════════ Adam ═══════════════════
Adam::Adam(float lr_) : lr(lr_), beta1(0.9f), beta2(0.999f), eps(1e-8f), step(0) {}

void Adam::update(Matrix &W, RowVector &b, const Matrix &dW, const RowVector &db, int idx) {
    if (idx >= (int)m_W.size()) {
        m_W.resize(idx+1); v_W.resize(idx+1);
        m_b.resize(idx+1); v_b.resize(idx+1);
        m_W[idx]=Matrix::Zero(W.rows(),W.cols());
        v_W[idx]=Matrix::Zero(W.rows(),W.cols());
        m_b[idx]=RowVector::Zero(b.size());
        v_b[idx]=RowVector::Zero(b.size());
    }
    step++;
    float b1c = 1.0f - std::pow(beta1, step);
    float b2c = 1.0f - std::pow(beta2, step);

    m_W[idx] = beta1*m_W[idx] + (1-beta1)*dW;
    v_W[idx] = beta2*v_W[idx] + (1-beta2)*dW.cwiseProduct(dW);
    Matrix mh_W = m_W[idx]/b1c, vh_W = v_W[idx]/b2c;
    W -= lr * mh_W.cwiseQuotient(vh_W.unaryExpr([this](float v){return std::sqrt(v)+eps;}));

    m_b[idx] = beta1*m_b[idx] + (1-beta1)*db;
    v_b[idx] = beta2*v_b[idx] + (1-beta2)*db.cwiseProduct(db);
    RowVector mh_b = m_b[idx]/b1c, vh_b = v_b[idx]/b2c;
    b -= lr * mh_b.cwiseQuotient(vh_b.unaryExpr([this](float v){return std::sqrt(v)+eps;}));
}

// ═══════════════════ SeqPredictor ═══════════════════
SeqPredictor::SeqPredictor(int in_dim, int hid, int out_dim)
    : encoder(in_dim, hid), decoder(out_dim, hid), optimizer(0.001f) {}

Matrix SeqPredictor::predict(const Matrix &X) {
    int H = encoder.cell.hidden_size;
    RowVector h = zeros(H), c = zeros(H);
    encoder.forward(X, h, c);
    RowVector start = X.row(X.rows()-1).segment(1, 2); // speed_n, rte_n
    return decoder.forward(h, c, start, 200);
}

float SeqPredictor::train_step(const Matrix &X, const Matrix &targets) {
    int H = encoder.cell.hidden_size;
    RowVector h = zeros(H), c = zeros(H);
    encoder.forward(X, h, c);

    RowVector start = X.row(X.rows()-1).segment(1, 2);
    std::mt19937 rng(42 + optimizer.step);

    // Forward decoder
    Matrix pred = decoder.forward_with_tf(h, c, start, targets, 0.5f, rng);

    // Loss
    float mse = (pred - targets).squaredNorm() / (200.0f * 2.0f);

    // Backward decoder
    decoder.backward(h, c, start, targets, 0.5f, rng, 0.001f);

    // Backward encoder (simplified: dh, dc from decoder's start error)
    RowVector dh_final = zeros(H), dc_final = zeros(H);
    // dh_final should come from decoder backward... simplified for now
    Matrix dX;
    encoder.backward(X, zeros(H), zeros(H), dh_final, dc_final, dX, 0.001f);

    return mse;
}

float SeqPredictor::compute_loss(const Matrix &pred, const Matrix &targets) {
    return (pred - targets).squaredNorm() / (200.0f * 2.0f);
}

// ═══════════════════ Curve computation ═══════════════════
void compute_curves(const std::vector<TelemetryPoint> &history,
                    const Matrix &pred,
                    std::vector<float> &actual_pos, std::vector<float> &actual_ene,
                    std::vector<float> &pred_pos, std::vector<float> &pred_ene,
                    std::vector<float> &future_pos, std::vector<float> &future_ene,
                    const std::vector<TelemetryPoint> &future) {
    // Actual curve from history
    float cp = history[0].pos_raw, ce = 0;
    actual_pos.push_back(cp); actual_ene.push_back(ce);
    for (auto &p : history) {
        cp += p.speed_raw/3.6f * 0.001f/1000.0f;
        ce += p.rte_raw * std::max(p.speed_raw, 0.1f) * 0.001f/3600.0f;
        actual_pos.push_back(cp); actual_ene.push_back(ce);
    }

    // Predicted curve
    float pp = cp, pe = ce;
    pred_pos.push_back(pp); pred_ene.push_back(pe);
    for (int i=0; i<200; ++i) {
        float spd = clamp(pred(i,0)*100.0f, 0.1f, 200.0f);
        float rte = clamp(pred(i,1)*100.0f, 0.0f, 200.0f);
        pp += spd/3.6f * 0.001f/1000.0f;
        pe += rte * spd * 0.001f/3600.0f;
        pred_pos.push_back(pp); pred_ene.push_back(pe);
    }

    // Future actual curve
    float fp = cp, fe = ce;
    future_pos.push_back(fp); future_ene.push_back(fe);
    for (auto &p : future) {
        fp += p.speed_raw/3.6f * 0.001f/1000.0f;
        fe += p.rte_raw * std::max(p.speed_raw, 0.1f) * 0.001f/3600.0f;
        future_pos.push_back(fp); future_ene.push_back(fe);
    }
}
