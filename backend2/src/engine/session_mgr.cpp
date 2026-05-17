#include "engine/session_mgr.h"
#include <QDateTime>
#include <QDebug>

namespace engine {

SessionManager::SessionManager(int timeoutSec, QObject *parent)
    : QObject(parent)
    , m_timeoutSec(timeoutSec)
    , m_cleanupTimer(new QTimer(this))
{
    connect(m_cleanupTimer, &QTimer::timeout, this, &SessionManager::cleanupExpired);
    // Run cleanup every 60 seconds
    m_cleanupTimer->start(60000);
}

SessionManager::~SessionManager()
{
    m_cleanupTimer->stop();
}

void SessionManager::append(const QString &sessionId, const DataPoint &dp)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    auto &session = m_sessions[sessionId];
    session.lastActivity = QDateTime::currentMSecsSinceEpoch();

    if (session.buffer.size() >= SessionData::MAX_BUFFER_SIZE) {
        session.buffer.pop_front();
    }
    session.buffer.push_back(dp);
}

std::deque<DataPoint> SessionManager::getBuffer(const QString &sessionId) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_sessions.contains(sessionId)) {
        return m_sessions[sessionId].buffer;
    }
    return {};
}

void SessionManager::reset(const QString &sessionId)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_sessions.remove(sessionId);
}

std::size_t SessionManager::bufferSize(const QString &sessionId) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_sessions.contains(sessionId)) {
        return m_sessions[sessionId].buffer.size();
    }
    return 0;
}

void SessionManager::cleanupExpired()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    qint64 now = QDateTime::currentMSecsSinceEpoch();
    qint64 timeoutMs = static_cast<qint64>(m_timeoutSec) * 1000;

    QList<QString> expired;
    for (auto it = m_sessions.begin(); it != m_sessions.end(); ++it) {
        if (now - it->lastActivity > timeoutMs) {
            expired.append(it.key());
        }
    }

    for (const auto &sid : expired) {
        m_sessions.remove(sid);
        qDebug() << "Session expired and cleaned:" << sid;
    }
}

} // namespace engine
