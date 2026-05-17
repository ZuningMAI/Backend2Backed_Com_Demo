#ifndef COM_SERVER_H
#define COM_SERVER_H

#include <QObject>
#include <QHttpServer>
#include "engine/session_mgr.h"

class Server : public QObject
{
    Q_OBJECT

public:
    explicit Server(QObject *parent = nullptr);
    ~Server() override;

    bool start(quint16 port);

private:
    void registerRoutes();
    QHttpServer *m_server;
    engine::SessionManager *m_sessionMgr;
};

#endif
