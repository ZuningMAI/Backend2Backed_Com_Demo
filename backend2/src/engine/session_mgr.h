#ifndef ENGINE_SESSION_MGR_H
#define ENGINE_SESSION_MGR_H

#include "engine/physics.h"

#include <QObject>
#include <QTimer>
#include <QHash>
#include <QString>
#include <QDateTime>
#include <deque>
#include <mutex>

namespace engine {

struct SessionData {
    std::deque<DataPoint> buffer;
    qint64 lastActivity;  // ms since epoch

    static constexpr std::size_t MAX_BUFFER_SIZE = 100000;
};

class SessionManager : public QObject
{
    Q_OBJECT

public:
    explicit SessionManager(int timeoutSec = 3600, QObject *parent = nullptr);
    ~SessionManager() override;

    /**
     * Append a new data point to the session buffer.
     * Creates session if it doesn't exist.
     */
    void append(const QString &sessionId, const DataPoint &dp);

    /**
     * Get the data buffer for a session (thread-safe copy).
     */
    std::deque<DataPoint> getBuffer(const QString &sessionId) const;

    /**
     * Reset (clear) a session's buffer.
     */
    void reset(const QString &sessionId);

    /**
     * Get the number of points in a session's buffer.
     */
    std::size_t bufferSize(const QString &sessionId) const;

private slots:
    void cleanupExpired();

private:
    mutable std::mutex m_mutex;
    QHash<QString, SessionData> m_sessions;
    QTimer *m_cleanupTimer;
    int m_timeoutSec;
};

} // namespace engine

#endif // ENGINE_SESSION_MGR_H
