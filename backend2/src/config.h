#ifndef CONFIG_H
#define CONFIG_H

#include <QString>

struct Config {
    static constexpr int SERVER_PORT = 9000;
    static constexpr int SESSION_TIMEOUT_SECONDS = 3600;
    static constexpr double DEFAULT_DATA_FREQUENCY_HZ = 1.0;

    static QString serverAddress() {
        return QString("http://0.0.0.0:%1").arg(SERVER_PORT);
    }
};

#endif // CONFIG_H
