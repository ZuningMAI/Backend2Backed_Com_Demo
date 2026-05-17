#include <QCoreApplication>
#include "config.h"
#include "Com/server.h"

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    app.setApplicationName("Backend2");
    app.setApplicationVersion("0.1.0");

    Server server;
    if (!server.start(Config::SERVER_PORT))
        return 1;

    return app.exec();
}
