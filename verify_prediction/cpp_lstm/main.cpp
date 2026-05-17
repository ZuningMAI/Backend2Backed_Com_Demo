#include "lstm.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>

int main() {
    // ── Load data ──
    std::ifstream fin("telemetry.bin", std::ios::binary);
    if (!fin) { std::cerr << "Cannot open telemetry.bin\n"; return 1; }

    int N;
    fin.read((char*)&N, sizeof(int));
    std::cout << "Loading " << N << " points...\n";

    std::vector<TelemetryPoint> pts(N);
    for (int i=0; i<N; ++i) {
        float feats[10];
        fin.read((char*)feats, 10*sizeof(float));
        pts[i].pos_km=feats[0]; pts[i].spd_n=feats[1];
        pts[i].frc_n=feats[2]; pts[i].mod_n=feats[3];
        pts[i].trf_n=feats[4]; pts[i].ebf_n=feats[5];
        pts[i].bpw_n=feats[6]; pts[i].soc_n=feats[7];
        pts[i].rte_n=feats[8]; pts[i].time_n=feats[9];
    }
    // Read raw data (speed, rte, pos)
    for (int i=0; i<N; ++i) {
        float raw[3];
        fin.read((char*)raw, 3*sizeof(float));
        pts[i].speed_raw=raw[0]; pts[i].rte_raw=raw[1]; pts[i].pos_raw=raw[2];
    }
    fin.close();
    std::cout << "Loaded.\n";

    // ── Model ──
    SeqPredictor model(10, 128, 2);
    const int WIN=800, PRED=200, STRIDE=100;

    // ── Open output ──
    std::ofstream fout("prediction_results.json");
    fout << "{\"windows\":[\n";
    bool first_win = true;

    int win_count = 0;
    float total_loss = 0;
    int loss_count = 0;

    // ── Window times to save ──
    std::vector<int> save_times;
    for (int t=0; t<500; t+=STRIDE) save_times.push_back(t);  // 0, 100, 200, 300, 400
    for (int t=1000; t<=N-WIN; t+=2000) save_times.push_back(t);
    if (save_times.back() != N-WIN) save_times.push_back(N-WIN);

    auto t_start = std::chrono::steady_clock::now();

    // ── Online loop ──
    for (int t_cur=0; t_cur < N-WIN-PRED; t_cur += STRIDE) {
        // Build X matrix (WIN, 10)
        Matrix X(WIN, 10);
        for (int i=0; i<WIN; ++i) {
            auto &p = pts[t_cur + i];
            X(i,0)=p.pos_km; X(i,1)=p.spd_n; X(i,2)=p.frc_n;
            X(i,3)=p.mod_n; X(i,4)=p.trf_n; X(i,5)=p.ebf_n;
            X(i,6)=p.bpw_n; X(i,7)=p.soc_n; X(i,8)=p.rte_n; X(i,9)=p.time_n;
        }

        // Build targets (PRED, 2)
        Matrix T(PRED, 2);
        for (int i=0; i<PRED; ++i) {
            auto &p = pts[t_cur + WIN + i];
            T(i,0)=p.spd_n; T(i,1)=p.rte_n;
        }

        // Train step
        float loss = model.train_step(X, T);
        total_loss += loss; loss_count++;

        // Save prediction for snapshot times
        bool is_save = false;
        for (auto st : save_times) if (t_cur == st) { is_save=true; break; }
        if (!is_save) continue;

        // Inference
        Matrix pred = model.predict(X);

        // Collect history and future
        std::vector<TelemetryPoint> hist(pts.begin()+t_cur, pts.begin()+t_cur+WIN);
        std::vector<TelemetryPoint> fut(pts.begin()+t_cur+WIN, pts.begin()+t_cur+WIN+PRED);

        // Compute curves
        std::vector<float> ap, ae, pp, pe, fp, fe;
        compute_curves(hist, pred, ap, ae, pp, pe, fp, fe, fut);

        // Compute errors
        float pos_err_m = std::abs(pp.back() - fp.back()) * 1000;
        float ene_err = std::abs(pe.back() - fe.back());
        float ene_mae = 0;
        for (size_t i=0; i<std::min(pe.size(),fe.size()); ++i)
            ene_mae += std::abs(pe[i]-fe[i]);
        ene_mae /= std::min(pe.size(),fe.size());

        // Write JSON
        if (!first_win) fout << ",\n"; first_win=false;
        fout << "  {\"t_start\":" << t_cur << ",\"t_end\":" << t_cur+WIN << ",";
        fout << "\"pred_start\":" << t_cur+WIN << ",\"pred_end\":" << t_cur+WIN+PRED << ",";
        fout << "\"loss\":" << loss << ",";
        fout << "\"pos_err_m\":" << pos_err_m << ",\"ene_err\":" << ene_err << ",";
        fout << "\"ene_mae\":" << ene_mae << ",";

        auto write_arr = [&](const char* name, const std::vector<float> &v) {
            fout << "\"" << name << "\":[";
            for (size_t i=0; i<v.size(); ++i) {
                if (i) fout << ",";
                fout << v[i];
            }
            fout << "]";
        };
        write_arr("actual_pos", ap); fout << ",";
        write_arr("actual_ene", ae); fout << ",";
        write_arr("pred_pos", pp); fout << ",";
        write_arr("pred_ene", pe); fout << ",";
        write_arr("future_pos", fp); fout << ",";
        write_arr("future_ene", fe);
        fout << "}";
        win_count++;
    }

    fout << "\n],\n";
    auto t_end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();
    fout << "\"total_windows\":" << win_count << ",\n";
    fout << "\"total_time_ms\":" << ms << ",\n";
    fout << "\"avg_loss\":" << (loss_count>0 ? total_loss/loss_count : 0) << "\n";
    fout << "}\n";
    fout.close();

    std::cout << "Done: " << win_count << " windows, " << ms << "ms total, "
              << (win_count>0?ms/win_count:0) << "ms/win\n";
    std::cout << "Avg loss: " << (loss_count>0?total_loss/loss_count:0) << "\n";
    return 0;
}
