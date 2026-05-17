#include "Com/server.h"
#include "config.h"
#include "engine/physics.h"

#include <QCoreApplication>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QTcpServer>
#include <cstdint>
#include <cmath>
#include <vector>
#include <QDebug>

Server::Server(QObject *parent)
    : QObject(parent)
    , m_server(new QHttpServer(this))
    , m_sessionMgr(new engine::SessionManager(Config::SESSION_TIMEOUT_SECONDS, this))
{}

Server::~Server() {}

// ── Polynomial fitting (least squares) ──
// Fit degree-d polynomial to data points (t_i, y_i), return coefficients [a_d, ..., a_0]
static std::vector<double> polyfit(const std::vector<double> &t, const std::vector<double> &y, int degree) {
    int n = (int)t.size();
    int m = degree + 1;
    // Build Vandermonde matrix X (n × m)
    std::vector<std::vector<double>> X(n, std::vector<double>(m, 1.0));
    for (int i = 0; i < n; ++i) {
        double ti = t[i];
        for (int j = 1; j < m; ++j)
            X[i][j] = X[i][j-1] * ti;
    }
    // X^T X  (m × m)
    std::vector<std::vector<double>> XTX(m, std::vector<double>(m, 0.0));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < n; ++k)
                XTX[i][j] += X[k][i] * X[k][j];
    // X^T y  (m × 1)
    std::vector<double> XTy(m, 0.0);
    for (int i = 0; i < m; ++i)
        for (int k = 0; k < n; ++k)
            XTy[i] += X[k][i] * y[k];
    // Gaussian elimination (m × m)
    // Forward
    for (int i = 0; i < m; ++i) {
        double pivot = XTX[i][i];
        if (std::abs(pivot) < 1e-12) continue;
        for (int j = i; j < m; ++j) XTX[i][j] /= pivot;
        XTy[i] /= pivot;
        for (int k = i + 1; k < m; ++k) {
            double factor = XTX[k][i];
            for (int j = i; j < m; ++j) XTX[k][j] -= factor * XTX[i][j];
            XTy[k] -= factor * XTy[i];
        }
    }
    // Back substitution
    std::vector<double> coeff(m, 0.0);
    for (int i = m - 1; i >= 0; --i) {
        coeff[i] = XTy[i];
        for (int j = i + 1; j < m; ++j)
            coeff[i] -= XTX[i][j] * coeff[j];
    }
    return coeff;
}

static double polyval(const std::vector<double> &coeff, double x) {
    double y = 0.0, xn = 1.0;
    for (int i = 0; i < (int)coeff.size(); ++i) {
        y += coeff[i] * xn;
        xn *= x;
    }
    return y;
}

void Server::registerRoutes()
{
    // ── GET /health ──
    m_server->route("/health", [](const QHttpServerRequest &req) {
        Q_UNUSED(req)
        QJsonObject r; r["status"] = 0; r["message"] = "Backend2 healthy"; r["version"] = "0.5.0";
        return QHttpServerResponse(QJsonDocument(r).toJson(QJsonDocument::Compact),
                                   QHttpServerResponse::StatusCode::Ok);
    });

    // ── POST /internal/calc/energy ──
    m_server->route("/internal/calc/energy", [this](const QHttpServerRequest &req) {
        QJsonDocument body = QJsonDocument::fromJson(req.body());
        QJsonObject obj = body.object();
        QString sid = obj.value("session_id").toString();
        QJsonArray pts = obj.value("data_points").toArray();
        double dt = obj.value("sample_interval").toDouble(0.001);

        for (const auto &v : pts) {
            QJsonObject p = v.toObject();
            engine::DataPoint dp;
            dp.time = (int64_t)p.value("time").toDouble();
            dp.tractive_force = p.value("tractive_force").toDouble();
            dp.electric_brake_force = p.value("electric_brake_force").toDouble();
            dp.speed = p.value("speed").toDouble();
            dp.battery_power = p.value("battery_power").toDouble();
            dp.soc = p.value("soc").toDouble();
            m_sessionMgr->append(sid, dp);
        }

        auto buf = m_sessionMgr->getBuffer(sid);
        auto res = engine::computeEnergy(buf, dt, 0, INT64_MAX);

        QJsonObject data;
        data["real_time_energy"] = res.real_time_energy;
        data["total_traction_energy"] = res.total_traction_energy;
        data["regenerative_energy"] = res.regenerative_energy;
        data["net_energy"] = res.net_energy;
        data["battery_energy"] = res.battery_energy;

        QJsonObject r; r["status"] = 0; r["data"] = data;
        r["message"] = QString("session=%1, pts=%2").arg(sid).arg(buf.size());
        return QHttpServerResponse(QJsonDocument(r).toJson(QJsonDocument::Compact),
                                   QHttpServerResponse::StatusCode::Ok);
    });

    // ── POST /internal/predict/train  (polynomial fit, zero training) ──
    m_server->route("/internal/predict/train", [](const QHttpServerRequest &req) {
        QJsonDocument body = QJsonDocument::fromJson(req.body());
        QJsonObject obj = body.object();
        QJsonArray history = obj.value("history_data").toArray();
        double cumulative_energy = obj.value("cumulative_energy").toDouble(0.0);

        if (history.size() < 200) {
            QJsonObject r; r["status"] = 1; r["message"] = "need >= 200 history points";
            r["data"] = QJsonObject{{"predicted_curve", QJsonArray()}};
            return QHttpServerResponse(QJsonDocument(r).toJson(QJsonDocument::Compact),
                                       QHttpServerResponse::StatusCode::Ok);
        }

        int n = (int)history.size();

        // Extract last 200ms for polynomial fitting
        int fit_len = std::min(200, n);
        int fit_start = n - fit_len;
        std::vector<double> t_fit(fit_len), spd(fit_len), rte(fit_len);
        for (int i = 0; i < fit_len; ++i) {
            QJsonObject p = history[fit_start + i].toObject();
            t_fit[i] = (double)i;
            spd[i] = p.value("speed").toDouble();
            rte[i] = p.value("real_time_energy").toDouble();
        }

        // Fit 3rd-order polynomial to speed and RTE
        auto spd_coef = polyfit(t_fit, spd, 3);
        auto rte_coef = polyfit(t_fit, rte, 2);

        // Extrapolate 200ms forward
        QJsonArray predicted;

        // Get last known position and start from cumulative energy
        QJsonObject lastP = history.last().toObject();
        double cumPos = lastP.value("position").toDouble(0.0) / 1000.0; // m → km
        double cumEnergy = cumulative_energy;

        for (int i = 0; i < 200; ++i) {
            double t = (double)(fit_len + i);
            double s = polyval(spd_coef, t);
            double r = polyval(rte_coef, t);
            // Clamp to valid range
            s = std::max(0.1, std::min(s, 200.0));
            r = std::max(0.0, std::min(r, 200.0));
            cumPos += s / 3.6 * 0.001 / 1000.0;      // km/h * 1ms → km
            cumEnergy += r * s * 0.001 / 3600.0;       // kWh
            QJsonObject pt;
            pt["position"] = cumPos;
            pt["energy"] = cumEnergy;
            predicted.append(pt);
        }

        QJsonObject data; data["predicted_curve"] = predicted;
        QJsonObject r; r["status"] = 0; r["data"] = data;
        r["message"] = "polyfit prediction";
        return QHttpServerResponse(QJsonDocument(r).toJson(QJsonDocument::Compact),
                                   QHttpServerResponse::StatusCode::Ok);
    });

    // ── POST /internal/session/reset ──
    m_server->route("/internal/session/reset", [this](const QHttpServerRequest &req) {
        QString sid = QJsonDocument::fromJson(req.body()).object().value("session_id").toString();
        m_sessionMgr->reset(sid);
        QJsonObject r; r["status"] = 0; r["message"] = QString("session %1 reset").arg(sid);
        return QHttpServerResponse(QJsonDocument(r).toJson(QJsonDocument::Compact),
                                   QHttpServerResponse::StatusCode::Ok);
    });
}

bool Server::start(quint16 port)
{
    registerRoutes();
    auto tcp = new QTcpServer(this);
    if (!tcp->listen(QHostAddress::Any, port)) {
        qCritical() << "Bind failed:" << tcp->errorString(); return false;
    }
    if (!m_server->bind(tcp)) { qCritical() << "QHttpServer bind failed"; return false; }
    qInfo() << "Backend2 v0.5.0 on port" << port;
    return true;
}
